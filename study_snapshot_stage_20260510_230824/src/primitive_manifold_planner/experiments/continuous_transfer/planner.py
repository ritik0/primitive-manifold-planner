"""Top-level orchestration for the staged continuous-transfer planner in Example 65."""

from __future__ import annotations

from typing import Any

import numpy as np

from .algorithms.staged_planner import StagedPlannerDelegates, StagedPlannerShell
from .benchmarks.scene_loader import build_continuous_transfer_scene, default_example_65_scene_description
from .config import (
    DEFAULT_EVIDENCE_SATURATION_CONTINUATION,
    DEFAULT_FAMILY_EVIDENCE_ROUNDS,
    DEFAULT_POST_ROUTE_EVIDENCE_ROUNDS,
    FAMILY_ROUTE_SELECTION_EXIT_HINT,
    LAMBDA_BIN_WIDTH,
    LEFT_STAGE_BASE_ROUNDS,
    LAMBDA_SOURCE_TOL,
)
from .core.stage_graph import StageGraph, StageNode, TransitionConstraint
from .family_stage import plan_within_family_stage, start_reachable_family_node_ids
from .graph_types import ContinuousRouteAlternative, ContinuousTransferRoute, ExitSeed, FamilyConnectivityGraph, RightStageClosureResult
from .left_stage import discover_entry_seeds_from_left
from .lambda_utils import (
    clamp_lambda,
    family_lambda_values,
    refine_lambda_region_if_promising,
    summarize_explored_lambda_regions,
    summarize_lambda_coverage_gaps,
)
from .projection_utils import sphere_radius_from_family
from .right_stage import build_outer_route_from_family_exit
from .graph_paths import k_shortest_simple_paths_over_graph, shortest_path_over_graph
from .route_geometry import build_route_geometry_views, orient_edge_path
from .parallel_leaf_evidence import ParallelLeafWarmstartResult, run_locked_lambda_parallel_evidence_warmstart
from .route_semantics import (
    GraphRouteCandidate,
    profile_family_route,
    rank_graph_route_candidates,
    rank_transition_supports_for_route_selection,
    score_route_with_family_preferences,
    summarize_family_route_semantics,
)
from .strict_validation import (
    format_strict_validation_failure,
    sample_strict_family_leaf_path,
    sample_strict_sphere_motion_path,
    validate_family_leaf_motion_edge,
    validate_left_motion_edge,
    validate_right_motion_edge,
    validate_selected_route_strictly,
    validate_transition_edge,
)
from .support import (
    concatenate_paths,
    deduplicate_points,
    explored_points_from_edges,
    merge_edges,
    sample_chart_centers,
)
from .seed_utils import keep_diverse_stage_seeds


def _canonical_leaf_lambda(lambda_value: float | None) -> float | None:
    if lambda_value is None:
        return None
    return float(round(float(lambda_value) / float(LAMBDA_BIN_WIDTH)) * float(LAMBDA_BIN_WIDTH))


def _leaf_id_for_plane_lambda(lambda_value: float | None) -> str:
    canonical = _canonical_leaf_lambda(lambda_value)
    return "plane_unknown" if canonical is None else f"plane_{float(canonical):.2f}"


def _parse_leaf_sequence_lambda(leaf_sequence: list[str]) -> float | None:
    for leaf_id in leaf_sequence:
        leaf_name = str(leaf_id)
        if leaf_name.startswith("plane_"):
            try:
                return float(leaf_name.split("_", 1)[1])
            except Exception:
                return None
    return None


def _build_leaf_meta_graph(
    graph: FamilyConnectivityGraph,
) -> tuple[dict[str, set[str]], list[str], str, str]:
    start_leaf = "left_sphere"
    goal_leaf = "right_sphere"
    adjacency: dict[str, set[str]] = {
        start_leaf: set(),
        goal_leaf: set(),
    }
    for edge in graph.edges:
        kind = str(edge.kind)
        if kind not in {"entry_transition", "exit_transition"}:
            continue
        edge_lambda = None
        if len(edge.path_lambdas) > 0:
            edge_lambda = float(np.asarray(edge.path_lambdas, dtype=float).reshape(-1)[0])
        elif edge.lambda_value is not None:
            edge_lambda = float(edge.lambda_value)
        plane_leaf = _leaf_id_for_plane_lambda(edge_lambda)
        adjacency.setdefault(plane_leaf, set())
        if kind == "entry_transition":
            adjacency[start_leaf].add(plane_leaf)
        else:
            adjacency[plane_leaf].add(goal_leaf)
    return adjacency, sorted(adjacency), start_leaf, goal_leaf


def _enumerate_simple_leaf_paths(
    adjacency: dict[str, set[str]],
    start_leaf: str,
    goal_leaf: str,
    max_length: int = 5,
) -> list[list[str]]:
    paths: list[list[str]] = []

    def dfs(current: str, path: list[str], visited: set[str]) -> None:
        if len(path) > int(max_length):
            return
        if current == goal_leaf:
            paths.append(list(path))
            return
        for neighbor in sorted(adjacency.get(current, set())):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            path.append(neighbor)
            dfs(neighbor, path, visited)
            path.pop()
            visited.remove(neighbor)

    dfs(str(start_leaf), [str(start_leaf)], {str(start_leaf)})
    return paths


def extract_continuous_route_for_leaf_sequence(
    *,
    graph: FamilyConnectivityGraph,
    start_node_id: int,
    goal_node_id: int,
    left_manifold,
    left_center: np.ndarray,
    left_radius: float,
    transfer_family,
    right_manifold,
    right_center: np.ndarray,
    right_radius: float,
    leaf_sequence: list[str],
    k_per_sequence: int = 6,
) -> ContinuousRouteAlternative | None:
    if len(leaf_sequence) < 3:
        return None
    if str(leaf_sequence[0]) != "left_sphere" or str(leaf_sequence[-1]) != "right_sphere":
        return None
    target_lambda = _parse_leaf_sequence_lambda(leaf_sequence)
    if target_lambda is None:
        return None
    target_lambda = float(target_lambda)

    def edge_allowed(edge) -> bool:
        kind = str(edge.kind)
        if kind == "family_transverse":
            return False
        if kind not in {"family_leaf_motion", "entry_transition", "exit_transition"}:
            return True
        edge_lambdas: list[float] = []
        if len(edge.path_lambdas) > 0:
            edge_lambdas = [float(v) for v in np.asarray(edge.path_lambdas, dtype=float).reshape(-1)]
        elif edge.lambda_value is not None:
            edge_lambdas = [float(edge.lambda_value)]
        if len(edge_lambdas) == 0:
            return False
        canonical_target = _canonical_leaf_lambda(target_lambda)
        return all(
            _canonical_leaf_lambda(float(value)) is not None
            and abs(float(_canonical_leaf_lambda(float(value))) - float(canonical_target)) <= max(LAMBDA_SOURCE_TOL, 0.5 * float(LAMBDA_BIN_WIDTH))
            for value in edge_lambdas
        )

    best: ContinuousRouteAlternative | None = None
    for candidate_cost, candidate_nodes, candidate_edges in k_shortest_simple_paths_over_graph(
        graph,
        int(start_node_id),
        int(goal_node_id),
        k=max(1, int(k_per_sequence)),
        edge_allowed=edge_allowed,
    ):
        if len(candidate_edges) == 0:
            continue
        candidate_is_strict, _candidate_message, _candidate_failures, _candidate_invalid_points = validate_selected_route_strictly(
            graph=graph,
            edge_path=list(candidate_edges),
            left_manifold=left_manifold,
            left_center=left_center,
            left_radius=left_radius,
            transfer_family=transfer_family,
            right_manifold=right_manifold,
            right_center=right_center,
            right_radius=right_radius,
        )
        if not candidate_is_strict:
            continue
        candidate_views = build_route_geometry_views(
            graph=graph,
            node_path=list(candidate_nodes),
            edge_path=list(candidate_edges),
            left_manifold=left_manifold,
            left_center=left_center,
            left_radius=left_radius,
            transfer_family=transfer_family,
            right_manifold=right_manifold,
            right_center=right_center,
            right_radius=right_radius,
        )
        alternative = ContinuousRouteAlternative(
            total_cost=float(candidate_cost),
            leaf_sequence=[str(leaf) for leaf in leaf_sequence],
            transition_count=sum(
                1 for edge_id in candidate_edges if str(graph.edges[int(edge_id)].kind) in {"entry_transition", "exit_transition"}
            ),
            graph_node_path=[int(node_id) for node_id in candidate_nodes],
            graph_edge_path=[int(edge_id) for edge_id in candidate_edges],
            raw_path=np.asarray(candidate_views.certified_path, dtype=float),
            display_path=np.asarray(candidate_views.display_path, dtype=float),
            strict_valid=True,
        )
        if best is None or float(alternative.total_cost) + 1e-12 < float(best.total_cost):
            best = alternative
    return best


def realize_selected_transition_route_continuous_transfer(
    *,
    start: np.ndarray,
    entry_transition: np.ndarray,
    exit_transition: np.ndarray,
    goal: np.ndarray,
    selected_lambda: float,
    left_manifold,
    left_center: np.ndarray,
    left_radius: float,
    transfer_family,
    right_manifold,
    right_center: np.ndarray,
    right_radius: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Rebuild execution/display geometry from selected transitions only.

    The evidence graph selects the transition pair and fixed family leaf. The
    final non-robot route is then realized as three fresh constrained local
    motions, mirroring the selected-transition realization used by Example 66.1.
    """
    start = np.asarray(start, dtype=float).reshape(3)
    entry = np.asarray(entry_transition, dtype=float).reshape(3)
    exit_q = np.asarray(exit_transition, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)
    lam = float(selected_lambda)
    diagnostics: dict[str, object] = {
        "local_replan_left_success": False,
        "local_replan_family_success": False,
        "local_replan_right_success": False,
        "local_replan_strict_validation_success": False,
        "local_replan_path_points": 0,
        "local_replan_message": "",
    }

    transition_failures: list[str] = []
    entry_left_ok, entry_left_failures = validate_transition_edge(
        path=np.asarray([entry], dtype=float),
        side_manifold=left_manifold,
        transfer_family=transfer_family,
        lam=lam,
        side_center=np.asarray(left_center, dtype=float),
        side_radius=float(left_radius),
        side_mode="left",
    )
    exit_right_ok, exit_right_failures = validate_transition_edge(
        path=np.asarray([exit_q], dtype=float),
        side_manifold=right_manifold,
        transfer_family=transfer_family,
        lam=lam,
        side_center=np.asarray(right_center, dtype=float),
        side_radius=float(right_radius),
        side_mode="right",
    )
    if not entry_left_ok:
        transition_failures.extend(format_strict_validation_failure(failure) for failure in entry_left_failures[:2])
    if not exit_right_ok:
        transition_failures.extend(format_strict_validation_failure(failure) for failure in exit_right_failures[:2])
    if len(transition_failures) > 0:
        diagnostics["local_replan_message"] = "transition validation failed: " + " | ".join(transition_failures)
        return np.zeros((0, 3), dtype=float), diagnostics

    left_path = sample_strict_sphere_motion_path(left_center, left_radius, start, entry)
    left_ok, left_failures = validate_left_motion_edge(left_path, left_manifold, left_center, left_radius)
    diagnostics["local_replan_left_success"] = bool(left_ok)
    if not left_ok:
        diagnostics["local_replan_message"] = (
            "left segment failed: "
            + " | ".join(format_strict_validation_failure(failure) for failure in left_failures[:3])
        )
        return np.zeros((0, 3), dtype=float), diagnostics

    family_path = sample_strict_family_leaf_path(entry, exit_q)
    family_ok, family_failures = validate_family_leaf_motion_edge(family_path, transfer_family, lam)
    diagnostics["local_replan_family_success"] = bool(family_ok)
    if not family_ok:
        diagnostics["local_replan_message"] = (
            "family segment failed: "
            + " | ".join(format_strict_validation_failure(failure) for failure in family_failures[:3])
        )
        return np.zeros((0, 3), dtype=float), diagnostics

    right_path = sample_strict_sphere_motion_path(right_center, right_radius, exit_q, goal)
    right_ok, right_failures = validate_right_motion_edge(right_path, right_manifold, right_center, right_radius)
    diagnostics["local_replan_right_success"] = bool(right_ok)
    if not right_ok:
        diagnostics["local_replan_message"] = (
            "right segment failed: "
            + " | ".join(format_strict_validation_failure(failure) for failure in right_failures[:3])
        )
        return np.zeros((0, 3), dtype=float), diagnostics

    realized_path = concatenate_paths(left_path, family_path, right_path)
    if len(realized_path) == 0:
        diagnostics["local_replan_message"] = "local realization produced an empty path"
        return np.zeros((0, 3), dtype=float), diagnostics
    diagnostics["local_replan_path_points"] = int(len(realized_path))
    diagnostics["local_replan_strict_validation_success"] = True
    diagnostics["local_replan_message"] = (
        "selected transition local replan passed strict segment validation: "
        f"lambda={lam:.6f}, points={len(realized_path)}"
    )
    return np.asarray(realized_path, dtype=float), diagnostics


def _build_example_65_stage_graph() -> StageGraph:
    """Create the fixed left -> family -> right stage graph for Example 65."""

    return StageGraph(
        nodes=[
            StageNode(stage_id="left", label="Left Outer Stage", stage_kind="outer_support"),
            StageNode(stage_id="family", label="Continuous Family Stage", stage_kind="transfer_family"),
            StageNode(stage_id="right", label="Right Outer Stage", stage_kind="outer_support"),
        ],
        transitions=[
            TransitionConstraint(source_stage_id="left", target_stage_id="family", transition_kind="entry"),
            TransitionConstraint(source_stage_id="family", target_stage_id="right", transition_kind="exit"),
        ],
    )


def _build_left_stage_failure_route(
    *,
    scene_profile: str,
    transfer_family,
    start_q: np.ndarray,
    start_node_id: int,
    left_round_budget: int,
    graph: FamilyConnectivityGraph,
    frontier_ids: dict[str, list[int]],
    explored_edges_by_mode: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    family_transverse_edges: list[tuple[np.ndarray, np.ndarray]],
    entry_transition_points: list[np.ndarray],
    round_sources: list[np.ndarray],
    round_targets: list[np.ndarray],
    leaf_store_counts: dict[str, int] | None = None,
) -> ContinuousTransferRoute:
    explored_edges = merge_edges(*explored_edges_by_mode.values())
    explored_points = explored_points_from_edges(explored_edges)
    return ContinuousTransferRoute(
        success=False,
        message="Left-stage exploration did not discover any robust certified entry seed into the transfer family.",
        selected_lambda=None,
        graph_node_path=[int(start_node_id)],
        graph_edge_path=[],
        certified_path=np.asarray([start_q], dtype=float),
        display_path=np.asarray([start_q], dtype=float),
        path=np.asarray([start_q], dtype=float),
        raw_path=np.asarray([start_q], dtype=float),
        scene_profile=scene_profile,
        family_obstacle_count=len(transfer_family.obstacles),
        family_obstacle_summary=transfer_family.obstacle_summaries(),
        candidate_entries=np.zeros((0, 3), dtype=float),
        candidate_lambdas=np.zeros((0,), dtype=float),
        exit_candidates=np.zeros((0, 3), dtype=float),
        explored_edges=explored_edges,
        explored_points=explored_points,
        round_sources=np.asarray(round_sources, dtype=float) if round_sources else np.zeros((0, 3), dtype=float),
        round_targets=np.asarray(round_targets, dtype=float) if round_targets else np.zeros((0, 3), dtype=float),
        chart_centers=sample_chart_centers(explored_points, max(1, len(explored_points))),
        chart_count=len(explored_points),
        evaluation_count=len(round_targets),
        round_count=int(left_round_budget),
        explored_edges_by_mode=explored_edges_by_mode,
        family_transverse_edges=family_transverse_edges,
        entry_transition_points=deduplicate_points(entry_transition_points, tol=1e-4),
        exit_transition_points=np.zeros((0, 3), dtype=float),
        explored_lambda_values=family_lambda_values(graph),
        entry_seed_count=0,
        family_frontier_count=len(frontier_ids.get("family", [])),
        lambda_coverage_gaps=summarize_lambda_coverage_gaps(transfer_family, family_lambda_values(graph)),
        graph_node_count=len(graph.nodes),
        graph_edge_count=len(graph.edges),
        right_frontier_count=len(frontier_ids.get("right", [])),
        augmented_family_state_count=sum(1 for node in graph.nodes if str(node.mode) == "family"),
        augmented_family_edge_count=sum(
            1 for edge in graph.edges if str(edge.kind) in {"family_leaf_motion", "family_transverse"}
        ),
        augmented_family_changing_lambda_edge_count=sum(
            1
            for edge in graph.edges
            if str(edge.kind) in {"family_leaf_motion", "family_transverse"}
            and len(np.asarray(edge.path_lambdas, dtype=float)) > 1
            and float(np.max(np.asarray(edge.path_lambdas, dtype=float)) - np.min(np.asarray(edge.path_lambdas, dtype=float))) > 1.5e-2
        ),
        augmented_family_constant_lambda_edge_count=0,
        family_space_sampler_usage={},
        route_lambda_span=0.0,
        route_lambda_variation_total=0.0,
        route_constant_lambda_edge_count=0,
        route_lambda_varying_edge_count=0,
        leaf_store_counts=dict(leaf_store_counts or {}),
    )


def _select_route_exit_seeds_for_closure(
    graph: FamilyConnectivityGraph,
    start_node_id: int,
    family_result,
    transfer_family,
    prefer_primary_leaf_only: bool = True,
):
    committed_edge_allowed = lambda edge: str(edge.kind) != "family_transverse"
    route_exit_seeds = [
        seed
        for seed in family_result.exit_seeds
        if len(
            shortest_path_over_graph(
                graph,
                int(start_node_id),
                int(seed.right_node_id),
                edge_allowed=committed_edge_allowed,
            )[1]
        ) > 0
    ]
    if len(route_exit_seeds) == 0 and family_result.primary_entry_lambda is not None:
        reconstructed_same_leaf_exits: list[ExitSeed] = []
        for edge in graph.edges:
            if str(edge.kind) != "exit_transition":
                continue
            edge_lambda = None
            if len(edge.path_lambdas) > 0:
                edge_lambda = float(np.asarray(edge.path_lambdas, dtype=float).reshape(-1)[0])
            elif edge.lambda_value is not None:
                edge_lambda = float(edge.lambda_value)
            if edge_lambda is None or abs(float(edge_lambda) - float(family_result.primary_entry_lambda)) > LAMBDA_SOURCE_TOL:
                continue
            node_u = graph.nodes[int(edge.node_u)]
            node_v = graph.nodes[int(edge.node_v)]
            if node_u.mode == "family" and node_v.mode == "right":
                family_node_id, right_node_id = int(edge.node_u), int(edge.node_v)
            elif node_v.mode == "family" and node_u.mode == "right":
                family_node_id, right_node_id = int(edge.node_v), int(edge.node_u)
            else:
                continue
            if len(
                shortest_path_over_graph(
                    graph,
                    int(start_node_id),
                    int(right_node_id),
                    edge_allowed=committed_edge_allowed,
                )[1]
            ) == 0:
                continue
            reconstructed_same_leaf_exits.append(
                ExitSeed(
                    family_node_id=int(family_node_id),
                    right_node_id=int(right_node_id),
                    q=(
                        np.asarray(edge.path[-1], dtype=float).copy()
                        if len(edge.path) > 0
                        else np.asarray(graph.nodes[int(right_node_id)].q, dtype=float).copy()
                    ),
                    lambda_value=float(edge_lambda),
                    discovered_round=max(
                        int(graph.nodes[int(family_node_id)].discovered_round),
                        int(graph.nodes[int(right_node_id)].discovered_round),
                    ),
                    cluster_id=None,
                )
            )
        reconstructed_same_leaf_exits.sort(
            key=lambda seed: (
                seed.discovered_round,
                float(np.linalg.norm(np.asarray(seed.q, dtype=float))),
            )
        )
        route_exit_seeds = reconstructed_same_leaf_exits
    if len(route_exit_seeds) == 0:
        return []
    if (
        prefer_primary_leaf_only
        and family_result.primary_entry_lambda is not None
        and len(transfer_family.obstacles) == 0
    ):
        same_leaf_route_exit_seeds = [
            seed
            for seed in route_exit_seeds
            if abs(float(seed.lambda_value) - float(family_result.primary_entry_lambda)) <= LAMBDA_SOURCE_TOL
        ]
        if len(same_leaf_route_exit_seeds) > 0:
            route_exit_seeds = same_leaf_route_exit_seeds
    return route_exit_seeds


def _build_final_continuous_route(
    *,
    scene_profile: str,
    graph: FamilyConnectivityGraph,
    start_node_id: int,
    goal_node_id: int,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    left_manifold,
    left_center: np.ndarray,
    left_radius: float,
    transfer_family,
    right_manifold,
    right_center: np.ndarray,
    right_radius: float,
    frontier_ids: dict[str, list[int]],
    explored_edges_by_mode: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    family_transverse_edges: list[tuple[np.ndarray, np.ndarray]],
    entry_transition_points: list[np.ndarray],
    exit_transition_points: list[np.ndarray],
    round_sources: list[np.ndarray],
    round_targets: list[np.ndarray],
    entry_seeds,
    family_result,
    bridge_rounds: int,
    route_exit_seeds,
    right_closure_result: RightStageClosureResult,
    top_k_assignments: int,
    top_k_paths: int,
    left_round_budget: int,
    parallel_round_budget: int,
    continue_after_first_solution: bool,
    max_extra_rounds_after_first_solution: int | None,
    leaf_store_counts: dict[str, int] | None = None,
) -> ContinuousTransferRoute:
    committed_edge_allowed = lambda edge: str(edge.kind) != "family_transverse"

    def _canonical_route_lambda(lambda_value: float | None) -> float | None:
        if lambda_value is None:
            return None
        return float(round(float(lambda_value) / float(LAMBDA_BIN_WIDTH)) * float(LAMBDA_BIN_WIDTH))

    def _edge_matches_locked_lambda(edge, lambda_value: float) -> bool:
        kind = str(edge.kind)
        if kind == "family_transverse":
            return False
        if kind not in {"family_leaf_motion", "entry_transition", "exit_transition"}:
            return True
        edge_lambdas: list[float] = []
        if len(edge.path_lambdas) > 0:
            edge_lambdas = [float(v) for v in np.asarray(edge.path_lambdas, dtype=float).reshape(-1)]
        elif edge.lambda_value is not None:
            edge_lambdas = [float(edge.lambda_value)]
        if len(edge_lambdas) == 0:
            return False
        target_lambda = _canonical_route_lambda(float(lambda_value))
        return all(
            _canonical_route_lambda(float(v)) is not None
            and abs(float(_canonical_route_lambda(float(v))) - float(target_lambda)) <= max(LAMBDA_SOURCE_TOL, 0.5 * float(LAMBDA_BIN_WIDTH))
            for v in edge_lambdas
        )

    def _deduplicate_lambda_values(values: list[float]) -> list[float]:
        ordered = sorted(
            float(value)
            for value in (
                _canonical_route_lambda(float(raw_value))
                for raw_value in values
            )
            if value is not None
        )
        unique: list[float] = []
        for value in ordered:
            if len(unique) == 0 or abs(float(value) - float(unique[-1])) > max(LAMBDA_SOURCE_TOL, 0.5 * float(LAMBDA_BIN_WIDTH)):
                unique.append(float(value))
        return unique

    direct_graph_cost, direct_node_path, direct_edge_path = shortest_path_over_graph(
        graph,
        start_node_id,
        goal_node_id,
        edge_allowed=committed_edge_allowed,
    )
    candidate_routes: list[GraphRouteCandidate] = []
    if len(direct_edge_path) > 0:
        candidate_routes.append(
            GraphRouteCandidate(
                total_cost=float(direct_graph_cost),
                node_path=list(direct_node_path),
                edge_path=list(direct_edge_path),
                terminal_support=None,
            )
        )
    candidate_lambda_values = _deduplicate_lambda_values(
        [
            float(seed.lambda_value)
            for seed in list(entry_seeds) + list(family_result.exit_seeds)
            if getattr(seed, "lambda_value", None) is not None
        ]
    )
    if len(candidate_lambda_values) == 0 and family_result.primary_entry_lambda is not None:
        candidate_lambda_values = [float(family_result.primary_entry_lambda)]
    for candidate_lambda in candidate_lambda_values:
        lambda_edge_allowed = lambda edge, lambda_value=candidate_lambda: _edge_matches_locked_lambda(edge, lambda_value)
        lambda_graph_cost, lambda_node_path, lambda_edge_path = shortest_path_over_graph(
            graph,
            start_node_id,
            goal_node_id,
            edge_allowed=lambda_edge_allowed,
        )
        if len(lambda_edge_path) > 0:
            candidate_routes.append(
                GraphRouteCandidate(
                    total_cost=float(lambda_graph_cost),
                    node_path=list(lambda_node_path),
                    edge_path=list(lambda_edge_path),
                    terminal_support=None,
                )
            )
        if int(top_k_paths) > 1:
            for enumerated_cost, enumerated_nodes, enumerated_edges in k_shortest_simple_paths_over_graph(
                graph,
                start_node_id,
                goal_node_id,
                k=max(2, int(top_k_paths)),
                edge_allowed=lambda_edge_allowed,
            ):
                if len(enumerated_edges) == 0:
                    continue
                candidate_routes.append(
                    GraphRouteCandidate(
                        total_cost=float(enumerated_cost),
                        node_path=list(enumerated_nodes),
                        edge_path=list(enumerated_edges),
                        terminal_support=None,
                    )
                )
    if int(top_k_paths) > 1:
        enumerated_route_count = max(
            int(top_k_paths) * 12,
            max(6, int(top_k_assignments) * 3),
        )
        for enumerated_cost, enumerated_nodes, enumerated_edges in k_shortest_simple_paths_over_graph(
            graph,
            start_node_id,
            goal_node_id,
            k=enumerated_route_count,
            edge_allowed=committed_edge_allowed,
        ):
            if len(enumerated_edges) == 0:
                continue
            candidate_routes.append(
                GraphRouteCandidate(
                    total_cost=float(enumerated_cost),
                    node_path=list(enumerated_nodes),
                    edge_path=list(enumerated_edges),
                    terminal_support=None,
                )
            )
    candidate_exit_seeds = rank_transition_supports_for_route_selection(
        route_exit_seeds,
        target_q=np.asarray(goal_q, dtype=float),
        top_k_assignments=top_k_assignments,
    )
    for exit_seed in candidate_exit_seeds:
        cost_to_exit, nodes_to_exit, edges_to_exit = shortest_path_over_graph(
            graph,
            start_node_id,
            int(exit_seed.right_node_id),
            edge_allowed=committed_edge_allowed,
        )
        cost_to_goal, nodes_to_goal, edges_to_goal = shortest_path_over_graph(
            graph,
            int(exit_seed.right_node_id),
            goal_node_id,
            edge_allowed=committed_edge_allowed,
        )
        if len(edges_to_exit) == 0 or len(edges_to_goal) == 0:
            continue
        candidate_routes.append(
            GraphRouteCandidate(
                total_cost=float(cost_to_exit + cost_to_goal),
                node_path=list(nodes_to_exit) + list(nodes_to_goal[1:]),
                edge_path=list(edges_to_exit) + list(edges_to_goal),
                terminal_support=exit_seed,
            )
        )
    unique_candidates: dict[tuple[int, ...], GraphRouteCandidate] = {}
    for candidate in candidate_routes:
        key = tuple(int(edge_id) for edge_id in candidate.edge_path)
        if key not in unique_candidates or float(candidate.total_cost) < float(unique_candidates[key].total_cost):
            unique_candidates[key] = candidate
    candidate_profiles = {
        tuple(int(edge_id) for edge_id in candidate.edge_path): profile_family_route(
            graph=graph,
            edge_path=candidate.edge_path,
            primary_entry_lambda=family_result.primary_entry_lambda,
        )
        for candidate in unique_candidates.values()
    }
    rejected_lambda_changing_route_count = sum(
        1
        for key, profile in candidate_profiles.items()
        if len(key) > 0 and not bool(profile.fixed_lambda_valid)
    )
    ranked_candidates = rank_graph_route_candidates(
        graph=graph,
        candidates=list(unique_candidates.values()),
        primary_entry_lambda=family_result.primary_entry_lambda,
    )
    fixed_lambda_route_found = len(ranked_candidates) > 0
    if len(ranked_candidates) > 0:
        best_candidate = ranked_candidates[0]
        best_node_path = list(best_candidate.node_path)
        best_edge_path = list(best_candidate.edge_path)
        chosen_exit_seed = best_candidate.terminal_support
    else:
        best_node_path, best_edge_path, chosen_exit_seed = [], [], None

    explored_edges = merge_edges(*explored_edges_by_mode.values())
    explored_points = explored_points_from_edges(explored_edges)
    route_views = (
        build_route_geometry_views(
            graph=graph,
            node_path=best_node_path,
            edge_path=best_edge_path,
            left_manifold=left_manifold,
            left_center=left_center,
            left_radius=left_radius,
            transfer_family=transfer_family,
            right_manifold=right_manifold,
            right_center=right_center,
            right_radius=right_radius,
        )
        if len(best_edge_path) > 0
        else None
    )
    certified_path = (
        np.asarray(route_views.certified_path, dtype=float)
        if route_views is not None and len(route_views.certified_path) > 0
        else np.asarray([start_q], dtype=float)
    )
    display_path = (
        np.asarray(route_views.display_path, dtype=float)
        if route_views is not None and len(route_views.display_path) > 0
        else certified_path.copy()
    )
    route_is_strict, strict_message, strict_failures, strict_invalid_points = validate_selected_route_strictly(
        graph=graph,
        edge_path=best_edge_path,
        left_manifold=left_manifold,
        left_center=left_center,
        left_radius=left_radius,
        transfer_family=transfer_family,
        right_manifold=right_manifold,
        right_center=right_center,
        right_radius=right_radius,
    )
    has_graph_route = len(best_node_path) > 0 and len(best_edge_path) > 0
    lambda_constraint_blocked = (
        not has_graph_route
        and len(unique_candidates) > 0
        and rejected_lambda_changing_route_count > 0
    )
    has_certified_geometry = len(certified_path) > 0
    chosen_transition_nodes = (
        deduplicate_points([graph.nodes[nid].q for nid in best_node_path if graph.nodes[nid].kind == "transition"], tol=1e-4)
        if len(best_node_path) > 0
        else np.zeros((0, 3), dtype=float)
    )
    family_path_lambdas = [
        float(graph.nodes[nid].lambda_value)
        for nid in best_node_path
        if graph.nodes[nid].mode == "family" and graph.nodes[nid].lambda_value is not None
    ]
    explored_lambda_values = family_lambda_values(graph)
    explored_lambda_regions = summarize_explored_lambda_regions(graph)
    lambda_coverage_span = 0.0 if len(explored_lambda_values) == 0 else float(np.max(explored_lambda_values) - np.min(explored_lambda_values))
    lambda_coverage_gaps = summarize_lambda_coverage_gaps(transfer_family, explored_lambda_values)
    family_exit_discovery_round = (
        None if family_result.exit_discovery_round is None else int(left_round_budget + family_result.exit_discovery_round)
    )
    if len(best_node_path) == 0 or not route_is_strict:
        first_solution_round = None
        best_solution_round = None
    else:
        first_solution_round = int(
            left_round_budget
            + family_result.expansion_rounds
            + max(1, int(right_closure_result.route_discovery_round or right_closure_result.rounds))
        )
        best_solution_round = max(
            first_solution_round,
            int(left_round_budget + family_result.expansion_rounds + max(1, right_closure_result.rounds)),
        )
    exploration_mode_usage = {"left_entry_discovery": int(left_round_budget)}
    exploration_mode_usage.update(family_result.mode_usage)
    exploration_mode_usage["right_stage_closure"] = int(right_closure_result.rounds)
    selected_exit_seed = chosen_exit_seed or next(
        (
            seed
            for seed in family_result.exit_seeds
            if int(seed.family_node_id) in set(best_node_path) or int(seed.right_node_id) in set(best_node_path)
        ),
        None,
    )
    alternate_exit_seed_usage_count = 0
    if selected_exit_seed is not None and len(family_result.exit_seeds) > 1:
        earliest_exit_round = min(int(seed.discovered_round) for seed in family_result.exit_seeds)
        if int(selected_exit_seed.discovered_round) > earliest_exit_round:
            alternate_exit_seed_usage_count = 1
    final_route_same_leaf_first, final_route_same_leaf_only = summarize_family_route_semantics(
        graph=graph,
        edge_path=best_edge_path,
        primary_entry_lambda=family_result.primary_entry_lambda,
    )
    route_profile = profile_family_route(
        graph=graph,
        edge_path=best_edge_path,
        primary_entry_lambda=family_result.primary_entry_lambda,
    ) if len(best_edge_path) > 0 else None
    route_lambda_values: list[float] = []
    for edge_id in best_edge_path:
        edge = graph.edges[int(edge_id)]
        if str(edge.kind) not in {"family_leaf_motion", "family_transverse", "entry_transition", "exit_transition"}:
            continue
        if len(edge.path_lambdas) > 0:
            route_lambda_values.extend(float(v) for v in np.asarray(edge.path_lambdas, dtype=float).reshape(-1))
        elif edge.lambda_value is not None:
            route_lambda_values.append(float(edge.lambda_value))
    selected_lambda_values = _deduplicate_lambda_values(route_lambda_values)
    selected_lambda_range = (
        "none"
        if len(selected_lambda_values) == 0
        else f"[{min(selected_lambda_values):.6f}, {max(selected_lambda_values):.6f}]"
    )
    route_lambda_variation_total = 0.0
    if len(route_lambda_values) >= 2:
        route_lambda_variation_total = float(np.sum(np.abs(np.diff(np.asarray(route_lambda_values, dtype=float)))))
    selected_entry_point: np.ndarray | None = None
    selected_exit_point: np.ndarray | None = None
    selected_lambda_for_realization: float | None = None
    local_replan_diagnostics: dict[str, object] = {
        "local_replan_left_success": False,
        "local_replan_family_success": False,
        "local_replan_right_success": False,
        "local_replan_strict_validation_success": False,
        "local_replan_path_points": 0,
        "local_replan_message": "local replan skipped: no strict graph route selected",
    }
    graph_route_used_for_execution = bool(has_graph_route)
    final_route_realization = (
        "graph_route_strictly_validated_display_geometry"
        if has_graph_route and has_certified_geometry and route_is_strict
        else "no_strictly_validated_graph_route"
    )
    if has_graph_route and has_certified_geometry and route_is_strict and len(route_lambda_values) > 0:
        entry_idx = next(
            (idx for idx, edge_id in enumerate(best_edge_path) if str(graph.edges[int(edge_id)].kind) == "entry_transition"),
            None,
        )
        exit_idx = next(
            (idx for idx, edge_id in enumerate(best_edge_path) if str(graph.edges[int(edge_id)].kind) == "exit_transition"),
            None,
        )
        if entry_idx is not None and exit_idx is not None and int(exit_idx) > int(entry_idx):
            entry_path, _entry_lambdas = orient_edge_path(
                graph,
                int(best_edge_path[int(entry_idx)]),
                int(best_node_path[int(entry_idx)]),
                int(best_node_path[int(entry_idx) + 1]),
            )
            exit_path, _exit_lambdas = orient_edge_path(
                graph,
                int(best_edge_path[int(exit_idx)]),
                int(best_node_path[int(exit_idx)]),
                int(best_node_path[int(exit_idx) + 1]),
            )
            if len(entry_path) > 0 and len(exit_path) > 0:
                selected_entry_point = np.asarray(entry_path[-1], dtype=float)
                selected_exit_point = np.asarray(exit_path[-1], dtype=float)
                selected_lambda_for_realization = float(np.mean(np.asarray(route_lambda_values, dtype=float)))
                realized_path, local_replan_diagnostics = realize_selected_transition_route_continuous_transfer(
                    start=np.asarray(start_q, dtype=float),
                    entry_transition=selected_entry_point,
                    exit_transition=selected_exit_point,
                    goal=np.asarray(goal_q, dtype=float),
                    selected_lambda=float(selected_lambda_for_realization),
                    left_manifold=left_manifold,
                    left_center=np.asarray(left_center, dtype=float),
                    left_radius=float(left_radius),
                    transfer_family=transfer_family,
                    right_manifold=right_manifold,
                    right_center=np.asarray(right_center, dtype=float),
                    right_radius=float(right_radius),
                )
                if bool(local_replan_diagnostics.get("local_replan_strict_validation_success", False)) and len(realized_path) > 0:
                    certified_path = np.asarray(realized_path, dtype=float)
                    display_path = np.asarray(realized_path, dtype=float)
                    graph_route_used_for_execution = False
                    final_route_realization = "selected_transition_local_replan"
                else:
                    final_route_realization = "graph_route_strictly_validated_display_geometry"
                    graph_route_used_for_execution = True
        else:
            local_replan_diagnostics["local_replan_message"] = "local replan skipped: selected route lacks ordered entry/exit transitions"
    route_constant_lambda_edge_count = sum(
        1
        for edge_id in best_edge_path
        if str(graph.edges[int(edge_id)].kind) in {"family_leaf_motion", "family_transverse"}
        and len(np.asarray(graph.edges[int(edge_id)].path_lambdas, dtype=float)) > 0
        and float(np.max(np.asarray(graph.edges[int(edge_id)].path_lambdas, dtype=float)) - np.min(np.asarray(graph.edges[int(edge_id)].path_lambdas, dtype=float))) <= 1.5e-2
    )
    route_lambda_varying_edge_count = sum(
        1
        for edge_id in best_edge_path
        if str(graph.edges[int(edge_id)].kind) in {"family_leaf_motion", "family_transverse"}
        and len(np.asarray(graph.edges[int(edge_id)].path_lambdas, dtype=float)) > 0
        and float(np.max(np.asarray(graph.edges[int(edge_id)].path_lambdas, dtype=float)) - np.min(np.asarray(graph.edges[int(edge_id)].path_lambdas, dtype=float))) > 1.5e-2
    )
    committed_family_region_count = max(
        0,
        len(
            {
                round(float(graph.nodes[node_id].lambda_value) / 0.08)
                for node_id in best_node_path
                if graph.nodes[node_id].mode == "family" and graph.nodes[node_id].lambda_value is not None
            }
        ),
    )
    route_alternatives: list[ContinuousRouteAlternative] = []
    leaf_meta_graph, _leaf_meta_nodes, start_leaf_id, goal_leaf_id = _build_leaf_meta_graph(graph)
    leaf_sequences = _enumerate_simple_leaf_paths(
        leaf_meta_graph,
        start_leaf_id,
        goal_leaf_id,
        max_length=5,
    )
    best_alternative_by_leaf_sequence: dict[tuple[str, ...], ContinuousRouteAlternative] = {}
    for leaf_sequence in leaf_sequences:
        alternative = extract_continuous_route_for_leaf_sequence(
            graph=graph,
            start_node_id=int(start_node_id),
            goal_node_id=int(goal_node_id),
            left_manifold=left_manifold,
            left_center=left_center,
            left_radius=float(left_radius),
            transfer_family=transfer_family,
            right_manifold=right_manifold,
            right_center=right_center,
            right_radius=float(right_radius),
            leaf_sequence=[str(leaf_id) for leaf_id in leaf_sequence],
            k_per_sequence=max(4, int(top_k_assignments) * 2),
        )
        if alternative is None:
            continue
        leaf_key = tuple(str(leaf_id) for leaf_id in alternative.leaf_sequence)
        incumbent = best_alternative_by_leaf_sequence.get(leaf_key)
        if incumbent is None or float(alternative.total_cost) + 1e-12 < float(incumbent.total_cost):
            best_alternative_by_leaf_sequence[leaf_key] = alternative
    if len(best_alternative_by_leaf_sequence) == 0:
        for candidate in ranked_candidates:
            candidate_profile = profile_family_route(
                graph=graph,
                edge_path=list(candidate.edge_path),
                primary_entry_lambda=family_result.primary_entry_lambda,
            )
            if candidate_profile.entry_lambda is None:
                continue
            fallback_sequence = [
                "left_sphere",
                _leaf_id_for_plane_lambda(candidate_profile.entry_lambda),
                "right_sphere",
            ]
            alternative = extract_continuous_route_for_leaf_sequence(
                graph=graph,
                start_node_id=int(start_node_id),
                goal_node_id=int(goal_node_id),
                left_manifold=left_manifold,
                left_center=left_center,
                left_radius=float(left_radius),
                transfer_family=transfer_family,
                right_manifold=right_manifold,
                right_center=right_center,
                right_radius=float(right_radius),
                leaf_sequence=fallback_sequence,
                k_per_sequence=2,
            )
            if alternative is None:
                continue
            best_alternative_by_leaf_sequence[tuple(alternative.leaf_sequence)] = alternative
    route_alternatives = sorted(
        best_alternative_by_leaf_sequence.values(),
        key=lambda alternative: (
            float(alternative.total_cost),
            int(alternative.transition_count),
            len(alternative.graph_edge_path),
            tuple(alternative.leaf_sequence),
        ),
    )[: max(1, int(top_k_paths))]
    left_evidence_nodes = sum(1 for node in graph.nodes if str(node.mode) == "left")
    family_evidence_nodes = sum(1 for node in graph.nodes if str(node.mode) == "family")
    right_evidence_nodes = sum(1 for node in graph.nodes if str(node.mode) == "right")
    left_evidence_edges = sum(1 for edge in graph.edges if str(edge.kind) == "left_motion")
    family_evidence_edges = sum(1 for edge in graph.edges if str(edge.kind) in {"family_leaf_motion", "family_transverse"})
    right_evidence_edges = sum(1 for edge in graph.edges if str(edge.kind) == "right_motion")
    post_solution_rounds_completed = (
        0
        if family_result.exit_discovery_round is None
        else max(0, int(family_result.expansion_rounds) - int(family_result.exit_discovery_round))
    )
    return ContinuousTransferRoute(
        success=has_graph_route and has_certified_geometry and route_is_strict,
        message=(
            "The planner accumulated left/family/right evidence in parallel, extracted a certified sequential route from that evidence graph, and only then produced a display route after strict manifold validation passed."
            if has_graph_route and has_certified_geometry and route_is_strict
            else "A graph route was found, but it required changing lambda during transfer, which violates the fixed-family-member transfer constraint."
            if lambda_constraint_blocked
            else strict_message
            if has_graph_route and has_certified_geometry
            else "The planner reached the right stage but did not close a certified start-to-goal route within the right-stage closure budget."
            if len(family_result.exit_seeds) > 0
            else "The left stage found entry seeds, but the family evidence explorer did not discover a certified right exit within budget despite maintaining alternate family evidence regions."
        ),
        selected_lambda=None if len(family_path_lambdas) == 0 else float(np.mean(family_path_lambdas)),
        graph_node_path=[int(node_id) for node_id in best_node_path],
        graph_edge_path=[int(edge_id) for edge_id in best_edge_path],
        certified_path=certified_path,
        display_path=display_path,
        path=display_path,
        raw_path=certified_path,
        scene_profile=scene_profile,
        family_obstacle_count=len(transfer_family.obstacles),
        family_obstacle_summary=transfer_family.obstacle_summaries(),
        entry_switch=None if len(chosen_transition_nodes) == 0 else np.asarray(chosen_transition_nodes[0], dtype=float),
        exit_switch=None if len(chosen_transition_nodes) == 0 else np.asarray(chosen_transition_nodes[-1], dtype=float),
        candidate_entries=deduplicate_points([seed.q for seed in entry_seeds], tol=1e-4),
        candidate_lambdas=np.asarray([seed.lambda_value for seed in entry_seeds], dtype=float),
        exit_candidates=deduplicate_points([seed.q for seed in family_result.exit_seeds], tol=1e-4),
        explored_edges=explored_edges,
        explored_points=explored_points,
        round_sources=np.asarray(round_sources, dtype=float) if round_sources else np.zeros((0, 3), dtype=float),
        round_targets=np.asarray(round_targets, dtype=float) if round_targets else np.zeros((0, 3), dtype=float),
        chart_centers=sample_chart_centers(explored_points, max(1, len(explored_points))),
        chart_count=len(explored_points),
        evaluation_count=len(round_targets),
        round_count=left_round_budget + family_result.expansion_rounds + right_closure_result.rounds,
        explored_edges_by_mode=explored_edges_by_mode,
        family_transverse_edges=family_transverse_edges,
        entry_transition_points=deduplicate_points(entry_transition_points, tol=1e-4),
        exit_transition_points=deduplicate_points(exit_transition_points, tol=1e-4),
        explored_lambda_values=explored_lambda_values,
        family_leaf_motion_edge_count=sum(1 for edge in graph.edges if edge.kind == "family_leaf_motion"),
        family_transverse_edge_count=sum(1 for edge in graph.edges if edge.kind == "family_transverse"),
        rejected_transverse_count=family_result.rejected_transverse_count,
        entry_transition_count=sum(1 for edge in graph.edges if edge.kind == "entry_transition"),
        exit_transition_count=sum(1 for edge in graph.edges if edge.kind == "exit_transition"),
        entry_seed_count=len(entry_seeds),
        exit_seed_count=len(family_result.exit_seeds),
        family_frontier_count=len(frontier_ids.get("family", [])),
        family_expansion_rounds=family_result.expansion_rounds + bridge_rounds,
        family_exit_discovery_round=family_exit_discovery_round,
        family_cluster_count=family_result.cluster_count,
        active_family_cluster_count=family_result.active_cluster_count,
        stalled_family_cluster_count=family_result.stalled_cluster_count,
        cluster_switch_count=family_result.cluster_switch_count,
        alternate_entry_seed_usage_count=family_result.alternate_entry_seed_usage_count,
        alternate_exit_seed_usage_count=alternate_exit_seed_usage_count,
        family_exit_candidates_kept=len(family_result.exit_seeds),
        family_regions_with_no_exit=family_result.family_regions_with_no_exit,
        family_regions_with_exit=family_result.family_regions_with_exit,
        best_right_residual_seen=family_result.best_right_residual_seen,
        lambda_coverage_span=lambda_coverage_span,
        lambda_coverage_gaps=lambda_coverage_gaps,
        family_cluster_summaries=family_result.family_cluster_summaries,
        family_cluster_centers=family_result.family_cluster_centers,
        explored_lambda_region_count=len(explored_lambda_regions),
        explored_lambda_regions=explored_lambda_regions,
        first_solution_round=first_solution_round,
        best_solution_round=best_solution_round,
        continued_after_first_solution=bool(
            first_solution_round is not None and best_solution_round is not None and best_solution_round > first_solution_round
        ),
        lambda_hypothesis_count=len(explored_lambda_values),
        family_candidates_before_solution=len(entry_seeds),
        exploration_mode_usage=exploration_mode_usage,
        graph_node_count=len(graph.nodes),
        graph_edge_count=len(graph.edges),
        right_closure_rounds=right_closure_result.rounds,
        right_frontier_count=right_closure_result.right_frontier_count,
        right_closure_attempt_count=right_closure_result.closure_attempt_count,
        successful_right_closure_seed_count=right_closure_result.successful_seed_count,
        failed_right_closure_seed_count=right_closure_result.failed_seed_count,
        best_goal_residual_seen_on_right=right_closure_result.best_goal_residual_seen,
        right_goal_connection_attempts=right_closure_result.goal_connection_attempts,
        right_goal_connection_successes=right_closure_result.goal_connection_successes,
        right_goal_connection_failures=right_closure_result.goal_connection_failures,
        best_goal_distance_seen_on_right=right_closure_result.best_goal_distance_seen,
        final_graph_route_found_before_validation=has_graph_route,
        primary_entry_lambda=family_result.primary_entry_lambda,
        primary_entry_seed_id=family_result.primary_entry_seed_id,
        same_leaf_attempt_count=family_result.same_leaf_attempt_count,
        same_leaf_progress_count=family_result.same_leaf_progress_count,
        same_leaf_failure_count=family_result.same_leaf_failure_count,
        same_leaf_successful_exit_found=family_result.same_leaf_successful_exit_found,
        same_leaf_stagnation_triggered=family_result.same_leaf_stagnation_triggered,
        transverse_switch_count_after_entry=family_result.transverse_switch_count_after_entry,
        first_transverse_round_after_entry=family_result.first_transverse_round_after_entry,
        first_transverse_switch_reason=family_result.first_transverse_switch_reason,
        transverse_switch_reason_counts=family_result.transverse_switch_reason_counts,
        final_route_same_leaf_first=final_route_same_leaf_first,
        final_route_same_leaf_only=final_route_same_leaf_only,
        shared_proposals_processed=family_result.shared_proposals_processed,
        proposals_used_by_multiple_family_regions=family_result.proposals_used_by_multiple_family_regions,
        family_evidence_region_count=family_result.family_evidence_region_count,
        family_evidence_only_region_count=family_result.family_evidence_only_region_count,
        committed_family_region_count=committed_family_region_count,
        family_lambda_coverage_before_first_committed_route=family_result.family_lambda_coverage_before_first_committed_route,
        family_lambda_coverage_after_first_committed_route=family_result.family_lambda_coverage_after_first_committed_route,
        transition_hypotheses_left_family=family_result.transition_hypotheses_left_family,
        transition_hypotheses_family_right=family_result.transition_hypotheses_family_right,
        family_region_updates_per_round=family_result.family_region_updates_per_round,
        committed_route_changes_after_first_solution=family_result.committed_route_changes_after_first_solution,
        average_useful_family_regions_per_proposal=family_result.average_useful_family_regions_per_proposal,
        augmented_family_state_count=family_result.augmented_family_state_count,
        augmented_family_edge_count=family_result.augmented_family_edge_count,
        augmented_family_changing_lambda_edge_count=family_result.augmented_family_changing_lambda_edge_count,
        augmented_family_constant_lambda_edge_count=family_result.augmented_family_constant_lambda_edge_count,
        family_space_sampler_usage=family_result.family_space_sampler_usage,
        route_lambda_span=0.0 if route_profile is None else float(route_profile.lambda_span),
        route_lambda_variation_total=float(route_lambda_variation_total),
        route_constant_lambda_edge_count=int(route_constant_lambda_edge_count),
        route_lambda_varying_edge_count=int(route_lambda_varying_edge_count),
        fixed_lambda_route_found=bool(fixed_lambda_route_found),
        rejected_lambda_changing_route_count=int(rejected_lambda_changing_route_count),
        strict_validation_message=strict_message,
        strict_validation_failures=strict_failures,
        strict_invalid_points=strict_invalid_points,
        top_k_routes=route_alternatives,
        leaf_store_counts=dict(leaf_store_counts or {}),
        ambient_probe_rounds=int(parallel_round_budget) + int(family_result.expansion_rounds),
        continue_after_first_solution=bool(continue_after_first_solution),
        max_extra_rounds_after_first_solution=max_extra_rounds_after_first_solution,
        post_solution_rounds_completed=int(post_solution_rounds_completed),
        left_evidence_nodes=int(left_evidence_nodes),
        left_evidence_edges=int(left_evidence_edges),
        family_leaf_store_count=int(dict(leaf_store_counts or {}).get("family", 0)),
        family_evidence_nodes=int(family_evidence_nodes),
        family_evidence_edges=int(family_evidence_edges),
        right_evidence_nodes=int(right_evidence_nodes),
        right_evidence_edges=int(right_evidence_edges),
        route_candidates_evaluated=int(len(unique_candidates)),
        final_route_realization=str(final_route_realization),
        graph_route_used_for_execution=bool(graph_route_used_for_execution),
        selected_lambda_values=[round(float(value), 6) for value in selected_lambda_values],
        selected_lambda_range=str(selected_lambda_range),
        strict_validation_success=bool(route_is_strict and has_graph_route and has_certified_geometry),
        selected_entry_point=None if selected_entry_point is None else np.asarray(selected_entry_point, dtype=float),
        selected_exit_point=None if selected_exit_point is None else np.asarray(selected_exit_point, dtype=float),
        selected_lambda_for_realization=selected_lambda_for_realization,
        local_replan_left_success=bool(local_replan_diagnostics.get("local_replan_left_success", False)),
        local_replan_family_success=bool(local_replan_diagnostics.get("local_replan_family_success", False)),
        local_replan_right_success=bool(local_replan_diagnostics.get("local_replan_right_success", False)),
        local_replan_strict_validation_success=bool(local_replan_diagnostics.get("local_replan_strict_validation_success", False)),
        local_replan_path_points=int(local_replan_diagnostics.get("local_replan_path_points", 0)),
        local_replan_message=str(local_replan_diagnostics.get("local_replan_message", "")),
    )


def _capture_stage_value(stage_fn, sink: dict[str, object]):
    """Capture one staged delegate result for later finalization without changing behavior."""

    result = stage_fn()
    sink["value"] = result
    return result


def plan_continuous_transfer_route(
    max_ambient_probes: int | None = DEFAULT_FAMILY_EVIDENCE_ROUNDS,
    continue_after_first_solution: bool = DEFAULT_EVIDENCE_SATURATION_CONTINUATION,
    max_extra_rounds_after_first_solution: int | None = DEFAULT_POST_ROUTE_EVIDENCE_ROUNDS,
    top_k_assignments: int = FAMILY_ROUTE_SELECTION_EXIT_HINT,
    top_k_paths: int = 1,
    seed: int | None = 41,
    obstacle_profile: str = "none",
    scene_description: dict[str, Any] | str | None = None,
    _disable_parallel_leaf_warmstart: bool = False,
) -> ContinuousTransferRoute:
    scene_profile = "none" if obstacle_profile is None else str(obstacle_profile)
    resolved_scene_description = (
        default_example_65_scene_description(obstacle_profile=scene_profile)
        if scene_description is None
        else scene_description
    )
    scene = build_continuous_transfer_scene(resolved_scene_description)
    scene_profile = str(scene.description.get("transfer_family", {}).get("obstacle_profile", scene_profile))
    transfer_family_spec = dict(scene.description.get("transfer_family", {}))
    left_family = scene.left_support
    transfer_family = scene.transfer_family
    right_family = scene.right_support
    start_q = np.asarray(scene.start_q, dtype=float)
    goal_q = np.asarray(scene.goal_q, dtype=float)
    left_manifold = left_family.manifold(float(left_family.sample_lambdas()[0]))
    right_manifold = right_family.manifold(float(right_family.sample_lambdas()[0]))
    left_center = np.asarray(left_family.center, dtype=float)
    left_radius = sphere_radius_from_family(left_family)
    right_center = np.asarray(right_family.center, dtype=float)
    right_radius = sphere_radius_from_family(right_family)
    bounds_min = np.asarray(scene.bounds_min, dtype=float)
    bounds_max = np.asarray(scene.bounds_max, dtype=float)
    rng = np.random.default_rng(seed)

    graph = FamilyConnectivityGraph()
    start_node_id = graph.register_node("left", start_q, 0, "start")
    goal_node_id = graph.register_node("right", goal_q, 0, "goal")
    frontier_ids: dict[str, list[int]] = {"left": [start_node_id], "family": [], "right": [goal_node_id]}
    explored_edges_by_mode = {"left": [], "family_leaf": [], "family_transverse": [], "right": []}
    family_transverse_edges: list[tuple[np.ndarray, np.ndarray]] = []
    entry_transition_points: list[np.ndarray] = []
    exit_transition_points: list[np.ndarray] = []
    round_sources: list[np.ndarray] = []
    round_targets: list[np.ndarray] = []
    adaptive_lambda_values: set[float] = {float(v) for v in transfer_family.sample_lambdas({"count": 5})}
    left_round_budget = max(LEFT_STAGE_BASE_ROUNDS, 1 + int(0 if max_ambient_probes is None else max_ambient_probes) // 3)
    family_round_budget = DEFAULT_FAMILY_EVIDENCE_ROUNDS if max_ambient_probes is None else int(max_ambient_probes)
    if _disable_parallel_leaf_warmstart:
        warmstart = ParallelLeafWarmstartResult(parallel_round_budget=0)
    else:
        # The parallel shared-proposal phase is now the primary cross-family
        # evidence accumulator. Keep its budget substantial enough to discover
        # multiple locked plane leaves before any legacy staged fallback helps.
        if int(top_k_paths) > 1:
            parallel_round_budget = max(5, min(12, max(1, int(family_round_budget) // 2)))
            parallel_proposal_count = 2
        else:
            parallel_round_budget = max(4, min(8, max(1, int(family_round_budget) // 3)))
            parallel_proposal_count = 1
        warmstart = run_locked_lambda_parallel_evidence_warmstart(
            graph=graph,
            start_node_id=start_node_id,
            goal_node_id=goal_node_id,
            left_family=left_family,
            left_manifold=left_manifold,
            left_center=left_center,
            left_radius=left_radius,
            transfer_family=transfer_family,
            right_family=right_family,
            right_manifold=right_manifold,
            right_center=right_center,
            right_radius=right_radius,
            start_q=np.asarray(start_q, dtype=float),
            goal_q=np.asarray(goal_q, dtype=float),
            frontier_ids=frontier_ids,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            adaptive_lambda_values=adaptive_lambda_values,
            explored_edges_by_mode=explored_edges_by_mode,
            entry_transition_points=entry_transition_points,
            exit_transition_points=exit_transition_points,
            round_sources=round_sources,
            round_targets=round_targets,
            round_budget=parallel_round_budget,
            proposal_count=parallel_proposal_count,
            plane_candidate_lambdas=[
                float(lam)
                for lam in transfer_family_spec.get(
                    "plane_offsets",
                    list(transfer_family.sample_lambdas({"count": 7})),
                )
            ],
        )
    left_round_budget_total = left_round_budget + int(warmstart.parallel_round_budget)
    stage_graph = _build_example_65_stage_graph()
    def run_left_stage():
        discovered = discover_entry_seeds_from_left(
            graph=graph,
            start_node_id=start_node_id,
            left_manifold=left_manifold,
            left_center=left_center,
            left_radius=left_radius,
            transfer_family=transfer_family,
            clamp_lambda_fn=clamp_lambda,
            frontier_ids=frontier_ids,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            rng=rng,
            left_round_budget=left_round_budget,
            explored_edges_by_mode=explored_edges_by_mode,
            adaptive_lambda_values=adaptive_lambda_values,
            refine_lambda_region_if_promising_fn=refine_lambda_region_if_promising,
            entry_transition_points=entry_transition_points,
            round_sources=round_sources,
            round_targets=round_targets,
            round_offset=int(warmstart.parallel_round_budget),
        )
        return keep_diverse_stage_seeds(
            list(warmstart.entry_seeds) + list(discovered),
            max_keep=max(3, len(warmstart.entry_seeds) + len(discovered)),
            lambda_tol=1.5 * LAMBDA_SOURCE_TOL,
            spatial_tol=8e-2,
        )

    def build_left_stage_failure():
        return _build_left_stage_failure_route(
            scene_profile=scene_profile,
            transfer_family=transfer_family,
            start_q=np.asarray(start_q, dtype=float),
            start_node_id=start_node_id,
            left_round_budget=left_round_budget_total,
            graph=graph,
            frontier_ids=frontier_ids,
            explored_edges_by_mode=explored_edges_by_mode,
            family_transverse_edges=family_transverse_edges,
            entry_transition_points=entry_transition_points,
            round_sources=round_sources,
            round_targets=round_targets,
            leaf_store_counts=warmstart.leaf_store_counts,
        )

    def run_middle_stage(entry_seed_result):
        return plan_within_family_stage(
            graph=graph,
            start_node_id=start_node_id,
            entry_seeds=entry_seed_result,
            initial_exit_seeds=list(warmstart.initial_exit_seeds),
            transfer_family=transfer_family,
            right_manifold=right_manifold,
            right_center=right_center,
            right_radius=right_radius,
            frontier_ids=frontier_ids,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            adaptive_lambda_values=adaptive_lambda_values,
            explored_edges_by_mode=explored_edges_by_mode,
            family_transverse_edges=family_transverse_edges,
            exit_transition_points=exit_transition_points,
            round_sources=round_sources,
            round_targets=round_targets,
            rng=rng,
            round_offset=left_round_budget_total,
            family_round_budget=family_round_budget,
            continue_after_first_solution=continue_after_first_solution,
            max_extra_rounds_after_first_solution=max_extra_rounds_after_first_solution,
        )

    def bridge_middle_support(family_result):
        # A committed transfer route must stay on one fixed family member.
        # Do not add transverse family bridges after the family stage.
        return 0

    def select_right_stage_inputs(family_result):
        return _select_route_exit_seeds_for_closure(
            graph=graph,
            start_node_id=start_node_id,
            family_result=family_result,
            transfer_family=transfer_family,
            prefer_primary_leaf_only=int(top_k_paths) <= 1,
        )

    def run_right_stage(family_result, route_exit_seeds):
        right_closure_result = RightStageClosureResult(right_frontier_count=len(frontier_ids.get("right", [])))
        if len(route_exit_seeds) > 0:
            right_closure_result = build_outer_route_from_family_exit(
                graph=graph,
                start_node_id=start_node_id,
                goal_node_id=goal_node_id,
                exit_seeds=route_exit_seeds,
                right_manifold=right_manifold,
                right_center=right_center,
                right_radius=right_radius,
                goal_q=goal_q,
                frontier_ids=frontier_ids,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                round_idx=left_round_budget_total + family_result.expansion_rounds + 1,
                explored_edges_by_mode=explored_edges_by_mode,
                round_sources=round_sources,
                round_targets=round_targets,
                rng=rng,
                target_successful_seed_count=max(1, int(top_k_paths)),
            )
        return right_closure_result

    def finalize_route(family_result, bridge_rounds, route_exit_seeds, right_closure_result):
        return _build_final_continuous_route(
            scene_profile=scene_profile,
            graph=graph,
            start_node_id=start_node_id,
            goal_node_id=goal_node_id,
            start_q=np.asarray(start_q, dtype=float),
            goal_q=np.asarray(goal_q, dtype=float),
            left_manifold=left_manifold,
            left_center=left_center,
            left_radius=left_radius,
            transfer_family=transfer_family,
            right_manifold=right_manifold,
            right_center=right_center,
            right_radius=right_radius,
            frontier_ids=frontier_ids,
            explored_edges_by_mode=explored_edges_by_mode,
            family_transverse_edges=family_transverse_edges,
            entry_transition_points=entry_transition_points,
            exit_transition_points=exit_transition_points,
            round_sources=round_sources,
            round_targets=round_targets,
            entry_seeds=list(run_left_result.get("value", [])),
            family_result=family_result,
            bridge_rounds=int(bridge_rounds),
            route_exit_seeds=route_exit_seeds,
            right_closure_result=right_closure_result,
            top_k_assignments=top_k_assignments,
            top_k_paths=top_k_paths,
            left_round_budget=left_round_budget_total,
            parallel_round_budget=int(warmstart.parallel_round_budget),
            continue_after_first_solution=bool(continue_after_first_solution),
            max_extra_rounds_after_first_solution=max_extra_rounds_after_first_solution,
            leaf_store_counts=warmstart.leaf_store_counts,
        )

    run_left_result: dict[str, object] = {}
    delegates = StagedPlannerDelegates(
        run_left_stage=lambda: _capture_stage_value(run_left_stage, run_left_result),
        build_left_stage_failure=build_left_stage_failure,
        run_middle_stage=run_middle_stage,
        bridge_middle_support=bridge_middle_support,
        select_right_stage_inputs=select_right_stage_inputs,
        run_right_stage=run_right_stage,
        finalize_route=finalize_route,
    )
    shell = StagedPlannerShell(stage_graph=stage_graph, delegates=delegates)
    result = shell.run()
    if len(result.leaf_store_counts) == 0 and len(warmstart.leaf_store_counts) > 0:
        result.leaf_store_counts = dict(warmstart.leaf_store_counts)
    if len(result.leaf_store_counts) > 0:
        result.family_leaf_store_count = int(result.leaf_store_counts.get("family", result.family_leaf_store_count))
    if result.success or _disable_parallel_leaf_warmstart:
        return result
    fallback_result = plan_continuous_transfer_route(
        max_ambient_probes=max_ambient_probes,
        continue_after_first_solution=continue_after_first_solution,
        max_extra_rounds_after_first_solution=max_extra_rounds_after_first_solution,
        top_k_assignments=top_k_assignments,
        top_k_paths=top_k_paths,
        seed=seed,
        obstacle_profile=obstacle_profile,
        scene_description=resolved_scene_description,
        _disable_parallel_leaf_warmstart=True,
    )
    if len(fallback_result.leaf_store_counts) == 0 and len(warmstart.leaf_store_counts) > 0:
        fallback_result.leaf_store_counts = dict(warmstart.leaf_store_counts)
    if len(fallback_result.leaf_store_counts) > 0:
        fallback_result.family_leaf_store_count = int(
            fallback_result.leaf_store_counts.get("family", fallback_result.family_leaf_store_count)
        )
    return fallback_result


def print_continuous_route_summary(result: ContinuousTransferRoute) -> None:
    max_cluster_summaries = 24
    max_strict_failures = 12
    fields = [
        ("success", result.success),
        ("message", result.message),
        ("proposal_projection_supports", "left_sphere, transfer_family_candidate_lambdas, right_sphere"),
        ("lambda_selection_policy", "proposal-inferred + scene plane_offsets/candidate lambdas + active adaptive lambda regions"),
        ("route_selection_stage", "after left/family/right evidence accumulation, via fixed-lambda graph route ranking and top-k leaf-sequence extraction"),
        ("stopping_rule", "family evidence stops on budget/saturation; --stop-after-first-solution disables post-solution continuation"),
        ("scene_profile", result.scene_profile),
        ("family_obstacle_count", result.family_obstacle_count),
        ("family_obstacle_summary", result.family_obstacle_summary),
        ("selected_lambda", None if result.selected_lambda is None else round(float(result.selected_lambda), 6)),
        ("selected_lambda_values", result.selected_lambda_values),
        ("selected_lambda_range", result.selected_lambda_range),
        ("ambient_probe_rounds", result.ambient_probe_rounds),
        ("continue_after_first_solution", result.continue_after_first_solution),
        ("max_extra_rounds_after_first_solution", result.max_extra_rounds_after_first_solution),
        ("post_solution_rounds_completed", result.post_solution_rounds_completed),
        ("total_rounds", result.round_count),
        ("evaluation_events", result.evaluation_count),
        ("left_evidence_nodes", result.left_evidence_nodes),
        ("left_evidence_edges", result.left_evidence_edges),
        ("family_leaf_store_count", result.family_leaf_store_count),
        ("family_evidence_nodes", result.family_evidence_nodes),
        ("family_evidence_edges", result.family_evidence_edges),
        ("right_evidence_nodes", result.right_evidence_nodes),
        ("right_evidence_edges", result.right_evidence_edges),
        ("entry_seed_count", result.entry_seed_count),
        ("exit_seed_count", result.exit_seed_count),
        ("shared_proposals_processed", result.shared_proposals_processed),
        ("proposals_used_by_multiple_family_regions", result.proposals_used_by_multiple_family_regions),
        ("family_evidence_region_count", result.family_evidence_region_count),
        ("family_evidence_only_region_count", result.family_evidence_only_region_count),
        ("committed_family_region_count", result.committed_family_region_count),
        ("family_lambda_coverage_before_first_committed_route", result.family_lambda_coverage_before_first_committed_route),
        ("family_lambda_coverage_after_first_committed_route", result.family_lambda_coverage_after_first_committed_route),
        ("transition_hypotheses_left_family", result.transition_hypotheses_left_family),
        ("transition_hypotheses_family_right", result.transition_hypotheses_family_right),
        ("entry_transitions_found", result.entry_transition_count),
        ("exit_transitions_found", result.exit_transition_count),
        ("family_region_updates_per_round", round(float(result.family_region_updates_per_round), 4)),
        ("committed_route_changes_after_first_solution", result.committed_route_changes_after_first_solution),
        ("average_useful_family_regions_per_proposal", round(float(result.average_useful_family_regions_per_proposal), 4)),
        ("augmented_family_state_count", result.augmented_family_state_count),
        ("augmented_family_edge_count", result.augmented_family_edge_count),
        ("augmented_family_changing_lambda_edge_count", result.augmented_family_changing_lambda_edge_count),
        ("augmented_family_constant_lambda_edge_count", result.augmented_family_constant_lambda_edge_count),
        ("route_lambda_span", round(float(result.route_lambda_span), 6)),
        ("route_lambda_variation_total", round(float(result.route_lambda_variation_total), 6)),
        ("route_constant_lambda_edge_count", result.route_constant_lambda_edge_count),
        ("route_lambda_varying_edge_count", result.route_lambda_varying_edge_count),
        ("fixed_lambda_route_found", result.fixed_lambda_route_found),
        ("rejected_lambda_changing_route_count", result.rejected_lambda_changing_route_count),
        ("primary_entry_lambda", None if result.primary_entry_lambda is None else round(float(result.primary_entry_lambda), 6)),
        ("primary_entry_seed_id", result.primary_entry_seed_id),
        ("same_leaf_attempt_count", result.same_leaf_attempt_count),
        ("same_leaf_progress_count", result.same_leaf_progress_count),
        ("same_leaf_failure_count", result.same_leaf_failure_count),
        ("same_leaf_successful_exit_found", result.same_leaf_successful_exit_found),
        ("same_leaf_stagnation_triggered", result.same_leaf_stagnation_triggered),
        ("transverse_switch_count_after_entry", result.transverse_switch_count_after_entry),
        ("first_transverse_round_after_entry", result.first_transverse_round_after_entry),
        ("first_transverse_switch_reason", result.first_transverse_switch_reason),
        ("transverse_switch_reason_counts", result.transverse_switch_reason_counts),
        ("final_route_same_leaf_first", result.final_route_same_leaf_first),
        ("final_route_same_leaf_only", result.final_route_same_leaf_only),
        ("family_frontier_count", result.family_frontier_count),
        ("family_expansion_rounds", result.family_expansion_rounds),
        ("family_exit_discovery_round", result.family_exit_discovery_round),
        ("family_cluster_count", result.family_cluster_count),
        ("active_family_cluster_count", result.active_family_cluster_count),
        ("stalled_family_cluster_count", result.stalled_family_cluster_count),
        ("cluster_switch_count", result.cluster_switch_count),
        ("alternate_entry_seed_usage_count", result.alternate_entry_seed_usage_count),
        ("alternate_exit_seed_usage_count", result.alternate_exit_seed_usage_count),
        ("family_exit_candidates_kept", result.family_exit_candidates_kept),
        ("family_regions_with_no_exit", result.family_regions_with_no_exit),
        ("family_regions_with_exit", result.family_regions_with_exit),
        ("best_right_residual_seen", None if not np.isfinite(result.best_right_residual_seen) else round(float(result.best_right_residual_seen), 6)),
        ("right_closure_rounds", result.right_closure_rounds),
        ("right_frontier_count", result.right_frontier_count),
        ("right_closure_attempt_count", result.right_closure_attempt_count),
        ("successful_right_closure_seed_count", result.successful_right_closure_seed_count),
        ("failed_right_closure_seed_count", result.failed_right_closure_seed_count),
        ("right_goal_connection_attempts", result.right_goal_connection_attempts),
        ("right_goal_connection_successes", result.right_goal_connection_successes),
        ("right_goal_connection_failures", result.right_goal_connection_failures),
        ("best_goal_residual_seen_on_right", None if not np.isfinite(result.best_goal_residual_seen_on_right) else round(float(result.best_goal_residual_seen_on_right), 6)),
        ("best_goal_distance_seen_on_right", None if not np.isfinite(result.best_goal_distance_seen_on_right) else round(float(result.best_goal_distance_seen_on_right), 6)),
        ("lambda_coverage_span", round(float(result.lambda_coverage_span), 6)),
        ("lambda_coverage_gaps", result.lambda_coverage_gaps),
        ("graph_nodes", result.graph_node_count),
        ("graph_edges", result.graph_edge_count),
        ("graph_route_nodes", len(result.graph_node_path)),
        ("graph_route_edges", len(result.graph_edge_path)),
        ("explored_family_lambdas", len(result.explored_lambda_values)),
        ("family_leaf_motion_edges", result.family_leaf_motion_edge_count),
        ("family_transverse_edges", result.family_transverse_edge_count),
        ("rejected_transverse_attempts", result.rejected_transverse_count),
        ("certified_entry_transitions", result.entry_transition_count),
        ("certified_exit_transitions", result.exit_transition_count),
        ("explored_lambda_region_count", result.explored_lambda_region_count),
        ("explored_lambda_regions", result.explored_lambda_regions),
        ("first_solution_round", result.first_solution_round),
        ("best_solution_round", result.best_solution_round),
        ("continued_after_first_solution", result.continued_after_first_solution),
        ("certified_path_points", len(result.certified_path)),
        ("display_path_points", len(result.display_path)),
        ("path_points", len(result.path)),
        ("raw_path_points", len(result.raw_path)),
        ("strict_validation_message", result.strict_validation_message),
        ("strict_validation_success", result.strict_validation_success),
        ("strict_validation_failures", len(result.strict_validation_failures)),
        ("route_candidates_evaluated", result.route_candidates_evaluated),
        ("final_route_realization", result.final_route_realization),
        ("graph_route_used_for_execution", result.graph_route_used_for_execution),
        ("selected_entry_point", None if result.selected_entry_point is None else np.round(np.asarray(result.selected_entry_point, dtype=float), 6).tolist()),
        ("selected_exit_point", None if result.selected_exit_point is None else np.round(np.asarray(result.selected_exit_point, dtype=float), 6).tolist()),
        ("selected_lambda_for_realization", None if result.selected_lambda_for_realization is None else round(float(result.selected_lambda_for_realization), 6)),
        ("local_replan_left_success", result.local_replan_left_success),
        ("local_replan_family_success", result.local_replan_family_success),
        ("local_replan_right_success", result.local_replan_right_success),
        ("local_replan_strict_validation_success", result.local_replan_strict_validation_success),
        ("local_replan_path_points", result.local_replan_path_points),
        ("local_replan_message", result.local_replan_message),
        ("top_k_route_count", len(result.top_k_routes)),
        ("top_k_route_available_count", sum(bool(route.strict_valid) for route in result.top_k_routes)),
        ("leaf_store_counts", result.leaf_store_counts),
    ]
    for key, value in fields:
        print(f"{key} = {value}")
    for mode_name in sorted(result.exploration_mode_usage):
        print(f"mode_{mode_name} = {result.exploration_mode_usage[mode_name]}")
    for sampler_name in sorted(result.family_space_sampler_usage):
        print(f"family_space_sampler_{sampler_name} = {result.family_space_sampler_usage[sampler_name]}")
    cluster_summaries = list(result.family_cluster_summaries)
    if len(cluster_summaries) > max_cluster_summaries:
        head_count = max_cluster_summaries // 2
        tail_count = max_cluster_summaries - head_count
        cluster_summaries = cluster_summaries[:head_count] + cluster_summaries[-tail_count:]
    for summary in cluster_summaries:
        print(f"cluster_summary = {summary}")
    omitted_cluster_count = max(0, len(result.family_cluster_summaries) - len(cluster_summaries))
    if omitted_cluster_count > 0:
        print(f"cluster_summary_omitted = {omitted_cluster_count}")
    strict_failures = list(result.strict_validation_failures)
    if len(strict_failures) > max_strict_failures:
        strict_failures = strict_failures[:max_strict_failures]
    for failure in strict_failures:
        print(f"strict_failure = {failure}")
    omitted_failure_count = max(0, len(result.strict_validation_failures) - len(strict_failures))
    if omitted_failure_count > 0:
        print(f"strict_failure_omitted = {omitted_failure_count}")
    for idx, route in enumerate(result.top_k_routes):
        print(f"top_route_{idx}_cost = {round(float(route.total_cost), 6)}")
        print(f"top_route_{idx}_leaf_sequence = {' -> '.join(route.leaf_sequence)}")
        print(f"top_route_{idx}_transitions = {int(route.transition_count)}")
        print(f"top_route_{idx}_strict_valid = {bool(route.strict_valid)}")


def obstacle_profile_comparison_row(result: ContinuousTransferRoute) -> dict[str, str]:
    """Build one compact comparison row for obstacle-profile falsification runs."""

    return {
        "profile": str(result.scene_profile),
        "success": str(bool(result.success)),
        "primary_entry_lambda": (
            "-"
            if result.primary_entry_lambda is None
            else f"{float(result.primary_entry_lambda):.3f}"
        ),
        "same_leaf_successful_exit_found": str(bool(result.same_leaf_successful_exit_found)),
        "same_leaf_stagnation_triggered": str(bool(result.same_leaf_stagnation_triggered)),
        "first_transverse_switch_reason": (
            "-" if result.first_transverse_switch_reason is None else str(result.first_transverse_switch_reason)
        ),
        "transverse_switch_reason_counts": (
            "-"
            if len(result.transverse_switch_reason_counts) == 0
            else ",".join(
                f"{key}:{result.transverse_switch_reason_counts[key]}"
                for key in sorted(result.transverse_switch_reason_counts)
            )
        ),
        "explored_lambda_regions": (
            "-"
            if len(result.explored_lambda_regions) == 0
            else ",".join(result.explored_lambda_regions)
        ),
        "final_route_same_leaf_only": str(bool(result.final_route_same_leaf_only)),
    }


def print_obstacle_profile_comparison(results: list[ContinuousTransferRoute]) -> None:
    """Print a compact comparison table for systematic obstacle-profile runs."""

    if len(results) == 0:
        return
    rows = [obstacle_profile_comparison_row(result) for result in results]
    columns = [
        "profile",
        "success",
        "primary_entry_lambda",
        "same_leaf_successful_exit_found",
        "same_leaf_stagnation_triggered",
        "first_transverse_switch_reason",
        "transverse_switch_reason_counts",
        "explored_lambda_regions",
        "final_route_same_leaf_only",
    ]
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    divider = "-+-".join("-" * widths[column] for column in columns)
    print(header)
    print(divider)
    for row in rows:
        print(" | ".join(str(row[column]).ljust(widths[column]) for column in columns))
