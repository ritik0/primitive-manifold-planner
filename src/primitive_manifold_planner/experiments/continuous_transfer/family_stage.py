"""Family-stage exploration, clustering, and certified transverse motion for Example 65."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from .augmented_family_space import AugmentedFamilyConstrainedSpace, FamilyAugmentedState
from .config import (
    CLUSTER_PROGRESS_EPS,
    CLUSTER_STAGNATION_LIMIT,
    EXIT_LAMBDA_DIVERSITY_TOL,
    EXIT_SPATIAL_DIVERSITY_TOL,
    FAMILY_EVIDENCE_MIN_ROUNDS_BEFORE_SATURATION,
    FAMILY_EVIDENCE_SATURATION_WINDOW,
    FAMILY_MAX_SHARED_PROPOSALS_PER_ROUND,
    FAMILY_MAX_REGIONS_PER_PROPOSAL,
    FAMILY_MIN_SHARED_PROPOSALS_PER_ROUND,
    FAMILY_POST_SOLUTION_MIN_ROUNDS,
    FAMILY_PROPOSAL_LAMBDA_NEIGHBORS,
    FAMILY_PROPOSAL_NOVELTY_RADIUS,
    FAMILY_PROPOSAL_REGION_LAMBDA_TOL,
    FAMILY_SHARED_PROPOSALS_PER_ROUND,
    FAMILY_SUPPORT_IMPROVEMENT_EPS,
    FAMILY_TRANSVERSE_DELTA,
    LAMBDA_BIN_WIDTH,
    LAMBDA_SOURCE_TOL,
    MAX_ENTRY_SEEDS,
    MAX_EXPANSIONS_PER_NODE,
    P_FAMILY_EXIT_PROBE,
    P_FAMILY_LOCAL_NOVELTY,
    P_FAMILY_TRANSVERSE_STEP,
    P_FAMILY_UNDEREXPLORED_REGION,
    SAME_LEAF_STAGNATION_LIMIT,
    TRANSITION_INTERSECTION_TOL,
    TRANSVERSE_GOAL_TOL,
    TRANSVERSE_LAMBDA_STEP,
    TRANSVERSE_PATCH_STEP,
)
from .family_definition import ContinuousMaskedPlaneFamily
from .graph_insertions import add_certified_family_edge, register_frontier_node
from .graph_types import (
    ExitSeed,
    FamilyCluster,
    FamilyClusterProgress,
    FamilyConnectivityGraph,
    FamilyEvidenceRegion,
    FamilyStageResult,
    StageSeed,
)
from .lambda_utils import (
    choose_underexplored_lambda_region,
    clamp_lambda,
    family_nodes,
    quantize_lambda,
    summarize_explored_lambda_regions,
    refine_lambda_region_if_promising,
)
from .projection_utils import project_valid_family_state
from .route_semantics import choose_primary_entry_seed
from .seed_utils import keep_diverse_exit_seeds
from .strict_validation import (
    report_strict_validation_failures,
    validate_family_leaf_motion_edge,
    validate_family_transverse_edge,
    validate_transition_edge,
)
from .support import (
    explore_on_manifold_from_frontier,
    merge_edges,
    ompl_native_exploration_target,
    refine_intersection_on_both_manifolds,
    solve_exact_segment_on_manifold,
)


@dataclass
class FamilyStageContext:
    """Shared family-stage resources for local constrained exploration on the transfer family."""

    graph: FamilyConnectivityGraph
    transfer_family: ContinuousMaskedPlaneFamily
    family_space: AugmentedFamilyConstrainedSpace
    right_manifold: object
    right_center: np.ndarray
    right_radius: float
    frontier_ids: dict[str, list[int]]
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    adaptive_lambda_values: set[float]
    explored_edges_by_mode: dict[str, list[tuple[np.ndarray, np.ndarray]]]
    family_transverse_edges: list[tuple[np.ndarray, np.ndarray]]
    exit_transition_points: list[np.ndarray]
    round_sources: list[np.ndarray]
    round_targets: list[np.ndarray]


def connected_node_ids(graph: FamilyConnectivityGraph, root_node_id: int) -> set[int]:
    visited: set[int] = set()
    stack = [int(root_node_id)]
    while len(stack) > 0:
        node_id = int(stack.pop())
        if node_id in visited:
            continue
        visited.add(node_id)
        for edge_id in graph.adjacency.get(node_id, []):
            edge = graph.edges[int(edge_id)]
            other = int(edge.node_v if int(edge.node_u) == node_id else edge.node_u)
            if other not in visited:
                stack.append(other)
    return visited


def start_reachable_family_node_ids(
    graph: FamilyConnectivityGraph,
    start_node_id: int,
) -> set[int]:
    connected = connected_node_ids(graph, int(start_node_id))
    return {
        int(node_id)
        for node_id in connected
        if graph.nodes[int(node_id)].mode == "family"
    }


def family_state_from_node(
    graph: FamilyConnectivityGraph,
    node_id: int,
    cluster_id: int | None = None,
) -> FamilyAugmentedState | None:
    node = graph.nodes[int(node_id)]
    if node.mode != "family" or node.lambda_value is None:
        return None
    return FamilyAugmentedState(
        q=np.asarray(node.q, dtype=float).copy(),
        lambda_value=float(node.lambda_value),
        discovered_round=int(node.discovered_round),
        origin_sample_id=node.origin_sample_id,
        expansion_count=int(node.expansion_count),
        region_id=None if cluster_id is None else int(cluster_id),
        node_id=int(node.node_id),
        kind=str(node.kind),
    )


def family_states_from_ids(
    graph: FamilyConnectivityGraph,
    node_ids: Iterable[int],
    cluster_id: int | None = None,
) -> list[FamilyAugmentedState]:
    states: list[FamilyAugmentedState] = []
    for node_id in node_ids:
        state = family_state_from_node(graph, int(node_id), cluster_id=cluster_id)
        if state is not None:
            states.append(state)
    return states


def increment_family_state_expansion(
    context: FamilyStageContext,
    state: FamilyAugmentedState,
) -> FamilyAugmentedState:
    """Keep local family-stage reasoning state-centric and touch the graph only at update points."""

    node = context.graph.nodes[int(state.node_id)]
    node.expansion_count += 1
    return replace(state, expansion_count=int(node.expansion_count))


def choose_exit_seed_set(exit_seed_by_right_id: dict[int, ExitSeed]) -> list[ExitSeed]:
    return keep_diverse_exit_seeds(
        list(exit_seed_by_right_id.values()),
        max_keep=MAX_ENTRY_SEEDS,
        lambda_tol=EXIT_LAMBDA_DIVERSITY_TOL,
        spatial_tol=EXIT_SPATIAL_DIVERSITY_TOL,
    )


def certify_transverse_family_connection(
    transfer_family: ContinuousMaskedPlaneFamily,
    q_start: np.ndarray,
    lambda_start: float,
    q_goal: np.ndarray,
    lambda_goal: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    lam0 = float(lambda_start)
    lam1 = float(lambda_goal)
    q0 = np.asarray(q_start, dtype=float)
    q1 = np.asarray(q_goal, dtype=float)
    u0, v0 = transfer_family.patch_coords(lam0, q0)
    u1, v1 = transfer_family.patch_coords(lam1, q1)
    step_count = max(
        2,
        int(np.ceil(abs(lam1 - lam0) / TRANSVERSE_LAMBDA_STEP)),
        int(np.ceil(max(abs(u1 - u0), abs(v1 - v0)) / TRANSVERSE_PATCH_STEP)),
        int(np.ceil(np.linalg.norm(q1 - q0) / TRANSVERSE_PATCH_STEP)),
    )
    path: list[np.ndarray] = []
    lambdas: list[float] = []
    for t in np.linspace(0.0, 1.0, step_count + 1):
        lam = (1.0 - float(t)) * lam0 + float(t) * lam1
        u_coord = (1.0 - float(t)) * u0 + float(t) * u1
        v_coord = (1.0 - float(t)) * v0 + float(t) * v1
        guess = (
            transfer_family.point_on_leaf(lam)
            + u_coord * np.asarray(transfer_family._basis_u, dtype=float)
            + v_coord * np.asarray(transfer_family._basis_v, dtype=float)
        )
        q = project_valid_family_state(transfer_family, lam, guess)
        if q is None:
            return np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float), False
        if len(path) > 0 and np.linalg.norm(q - path[-1]) > 2.5 * TRANSVERSE_PATCH_STEP:
            return np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float), False
        path.append(q)
        lambdas.append(float(lam))
    if np.linalg.norm(path[-1] - q1) > TRANSVERSE_GOAL_TOL:
        return np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float), False
    path_arr = np.asarray(path, dtype=float)
    lambda_arr = np.asarray(lambdas, dtype=float)
    ok, failures = validate_family_transverse_edge(path_arr, lambda_arr, transfer_family)
    if not ok:
        report_strict_validation_failures("Rejected transverse family continuation", failures)
        return np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float), False
    return path_arr, lambda_arr, True


def certify_family_leaf_connection(
    transfer_family: ContinuousMaskedPlaneFamily,
    source_q: np.ndarray,
    target_q: np.ndarray,
    lam: float,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> np.ndarray:
    manifold = transfer_family.manifold(float(lam))
    result = solve_exact_segment_on_manifold(
        manifold=manifold,
        x_start=np.asarray(source_q, dtype=float),
        x_goal=np.asarray(target_q, dtype=float),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )
    path = np.asarray(getattr(result, "path", np.zeros((0, 3), dtype=float)), dtype=float)
    if len(path) == 0:
        return np.zeros((0, 3), dtype=float)
    if np.linalg.norm(path[-1] - np.asarray(target_q, dtype=float)) > TRANSVERSE_GOAL_TOL:
        return np.zeros((0, 3), dtype=float)
    ok, failures = validate_family_leaf_motion_edge(path, transfer_family, float(lam))
    if not ok:
        report_strict_validation_failures("Rejected family leaf connection", failures)
        return np.zeros((0, 3), dtype=float)
    return path


def plan_augmented_family_motion(
    context: FamilyStageContext,
    source_state: FamilyAugmentedState,
    target_state: FamilyAugmentedState,
    round_idx: int,
    guide_point: np.ndarray,
    origin_sample_id: int | None = None,
) -> list[int]:
    local_plan = context.family_space.plan_local_motion(
        source_state=source_state,
        target_state=target_state,
        bounds_min=context.bounds_min,
        bounds_max=context.bounds_max,
    )
    if not local_plan.success or len(local_plan.path) == 0:
        return []
    edge_kind = context.family_space.edge_kind_for_lambdas(local_plan.lambdas)
    if edge_kind == "family_leaf_motion":
        context.explored_edges_by_mode["family_leaf"] = merge_edges(
            context.explored_edges_by_mode.get("family_leaf", []),
            [(local_plan.path[idx - 1], local_plan.path[idx]) for idx in range(1, len(local_plan.path))],
        )
    if edge_kind == "family_transverse":
        context.explored_edges_by_mode["family_transverse"] = merge_edges(
            context.explored_edges_by_mode.get("family_transverse", []),
            [(local_plan.path[idx - 1], local_plan.path[idx]) for idx in range(1, len(local_plan.path))],
        )
        context.family_transverse_edges.extend(
            [(local_plan.path[idx - 1].copy(), local_plan.path[idx].copy()) for idx in range(1, len(local_plan.path))]
        )
        for lam in np.asarray(local_plan.lambdas, dtype=float):
            context.adaptive_lambda_values.add(float(lam))
            refine_lambda_region_if_promising(float(lam), context.adaptive_lambda_values, context.transfer_family)
    return add_certified_family_edge(
        graph=context.graph,
        transfer_family=context.transfer_family,
        path=np.asarray(local_plan.path, dtype=float),
        lambdas=np.asarray(local_plan.lambdas, dtype=float),
        round_idx=round_idx,
        frontier_ids=context.frontier_ids,
        guide_point=np.asarray(guide_point, dtype=float),
        edge_kind=edge_kind,
        edge_label=edge_kind,
        origin_sample_id=origin_sample_id,
    )


def expand_on_family_leaf(
    context: FamilyStageContext,
    source_state: FamilyAugmentedState,
    target_q: np.ndarray,
    round_idx: int,
    guide_point: np.ndarray,
    origin_sample_id: int | None = None,
) -> list[int]:
    target_state = context.family_space.project_augmented(
        x_guess=context.family_space.join(np.asarray(target_q, dtype=float), float(source_state.lambda_value)),
        lambda_hint=float(source_state.lambda_value),
        discovered_round=int(round_idx),
        origin_sample_id=origin_sample_id,
        region_id=source_state.region_id,
    )
    if target_state is None:
        return []
    return plan_augmented_family_motion(
        context=context,
        source_state=source_state,
        target_state=target_state,
        round_idx=round_idx,
        guide_point=guide_point,
        origin_sample_id=origin_sample_id,
    )


def build_family_clusters(
    graph: FamilyConnectivityGraph,
    frontier_ids: dict[str, list[int]],
    right_manifold,
    entry_seeds: list[StageSeed],
    exit_seeds: list[ExitSeed],
    cluster_progress: dict[int, FamilyClusterProgress],
    node_cluster_labels: dict[int, int] | None = None,
) -> tuple[list[FamilyCluster], dict[int, int]]:
    family_ids = [int(node.node_id) for node in family_nodes(graph) if node.lambda_value is not None]
    if len(family_ids) == 0:
        return [], {}

    entry_family_ids = {int(seed.family_node_id) for seed in entry_seeds}
    exit_family_ids = {int(seed.family_node_id) for seed in exit_seeds}
    clusters: list[FamilyCluster] = []
    node_to_cluster: dict[int, int] = {}

    label_lookup = dict(node_cluster_labels or {})
    grouped: dict[int, list[int]] = {}
    for node_id in family_ids:
        cluster_id = int(label_lookup.get(int(node_id), int(node_id)))
        grouped.setdefault(cluster_id, []).append(int(node_id))

    for cluster_id, component_ids_unsorted in sorted(grouped.items()):
        component_ids = sorted(int(node_id) for node_id in component_ids_unsorted)
        component = set(component_ids)
        frontier_component = [node_id for node_id in frontier_ids.get("family", []) if int(node_id) in component]
        states = family_states_from_ids(graph, component_ids, cluster_id=int(cluster_id))
        lambdas = np.asarray([float(state.lambda_value) for state in states], dtype=float)
        points = np.asarray([np.asarray(state.q, dtype=float) for state in states], dtype=float)
        best_residual = float(
            np.min(
                [
                    float(np.linalg.norm(np.asarray(right_manifold.residual(state.q), dtype=float)))
                    for state in states
                ]
            )
        )
        progress = cluster_progress.get(cluster_id, FamilyClusterProgress())
        progress.best_right_residual_seen = min(float(progress.best_right_residual_seen), best_residual)
        cluster_progress[cluster_id] = progress
        cluster = FamilyCluster(
            cluster_id=cluster_id,
            node_ids=component_ids,
            frontier_ids=[int(node_id) for node_id in frontier_component],
            entry_seed_ids=sorted(node_id for node_id in component_ids if node_id in entry_family_ids),
            exit_seed_ids=sorted(node_id for node_id in component_ids if node_id in exit_family_ids),
            lambda_min=float(np.min(lambdas)),
            lambda_max=float(np.max(lambdas)),
            lambda_span=float(np.max(lambdas) - np.min(lambdas)),
            centroid=np.mean(points, axis=0),
            best_right_residual=best_residual,
            active=len(frontier_component) > 0,
        )
        cluster.stagnating = (
            progress.no_gain_rounds >= CLUSTER_STAGNATION_LIMIT
            or progress.transverse_failures >= CLUSTER_STAGNATION_LIMIT
            or progress.exit_failures >= CLUSTER_STAGNATION_LIMIT
        )
        clusters.append(cluster)
        for node_id in component_ids:
            node_to_cluster[int(node_id)] = cluster_id

    clusters.sort(key=lambda cluster: (cluster.lambda_min, cluster.best_right_residual, cluster.cluster_id))
    return clusters, node_to_cluster


def family_cluster_summary(cluster: FamilyCluster, progress: FamilyClusterProgress) -> str:
    status = "stalled" if cluster.stagnating else ("active" if cluster.active else "inactive")
    has_exit = "exit" if len(cluster.exit_seed_ids) > 0 else "no-exit"
    return (
        f"cluster {cluster.cluster_id}: {status}, {has_exit}, "
        f"nodes={len(cluster.node_ids)}, frontier={len(cluster.frontier_ids)}, "
        f"lambda=[{cluster.lambda_min:.2f}, {cluster.lambda_max:.2f}], "
        f"best_right_residual={cluster.best_right_residual:.3f}, "
        f"no_gain_rounds={progress.no_gain_rounds}"
    )


def build_family_evidence_regions(
    graph: FamilyConnectivityGraph,
    frontier_ids: dict[str, list[int]],
    entry_seeds: list[StageSeed],
    exit_seeds: list[ExitSeed],
    cluster_progress: dict[int, FamilyClusterProgress],
    node_cluster_labels: dict[int, int] | None = None,
    right_manifold=None,
) -> tuple[list[FamilyEvidenceRegion], dict[int, int]]:
    clusters, node_to_cluster = build_family_clusters(
        graph=graph,
        frontier_ids=frontier_ids,
        right_manifold=right_manifold,
        entry_seeds=entry_seeds,
        exit_seeds=exit_seeds,
        cluster_progress=cluster_progress,
        node_cluster_labels=node_cluster_labels,
    )
    regions: list[FamilyEvidenceRegion] = []
    for cluster in clusters:
        states = family_states_from_ids(graph, cluster.node_ids, cluster_id=int(cluster.cluster_id))
        points = (
            np.asarray([np.asarray(state.q, dtype=float) for state in states], dtype=float)
            if len(states) > 0
            else np.zeros((0, 3), dtype=float)
        )
        progress = cluster_progress.get(cluster.cluster_id, FamilyClusterProgress())
        regions.append(
            FamilyEvidenceRegion(
                region_id=int(cluster.cluster_id),
                node_ids=list(cluster.node_ids),
                frontier_ids=list(cluster.frontier_ids),
                entry_seed_ids=list(cluster.entry_seed_ids),
                exit_seed_ids=list(cluster.exit_seed_ids),
                lambda_min=float(cluster.lambda_min),
                lambda_max=float(cluster.lambda_max),
                lambda_center=float(0.5 * (cluster.lambda_min + cluster.lambda_max)),
                lambda_span=float(cluster.lambda_span),
                centroid=np.asarray(cluster.centroid, dtype=float),
                explored_points=points,
                update_count=int(progress.selection_count),
                entry_support_count=len(cluster.entry_seed_ids),
                exit_support_count=len(cluster.exit_seed_ids),
                best_right_residual=float(cluster.best_right_residual),
                active=bool(cluster.active),
                stagnating=bool(cluster.stagnating),
                committed=False,
            )
        )
    return regions, node_to_cluster


def family_region_summary(region: FamilyEvidenceRegion, progress: FamilyClusterProgress) -> str:
    status = "stalled" if region.stagnating else ("active" if region.active else "inactive")
    has_exit = "exit" if region.exit_support_count > 0 else "no-exit"
    return (
        f"region {region.region_id}: {status}, {has_exit}, "
        f"nodes={len(region.node_ids)}, frontier={len(region.frontier_ids)}, "
        f"lambda=[{region.lambda_min:.2f}, {region.lambda_max:.2f}], "
        f"best_right_residual={region.best_right_residual:.3f}, "
        f"updates={region.update_count}, no_gain_rounds={progress.no_gain_rounds}"
    )


def choose_region_source_state(
    graph: FamilyConnectivityGraph,
    region: FamilyEvidenceRegion,
    target_q: np.ndarray,
    family_space: AugmentedFamilyConstrainedSpace | None = None,
    target_lambda: float | None = None,
) -> FamilyAugmentedState | None:
    state_ids = list(region.frontier_ids) + [node_id for node_id in region.node_ids if node_id not in region.frontier_ids]
    states = [
        state
        for state in family_states_from_ids(graph, state_ids, cluster_id=int(region.region_id))
        if state.expansion_count < MAX_EXPANSIONS_PER_NODE
    ]
    if len(states) == 0:
        return None
    target = np.asarray(target_q, dtype=float)
    states.sort(
        key=lambda state: (
            abs(float(state.lambda_value) - float(target_lambda)) if target_lambda is not None else 0.0,
            state.expansion_count,
            family_space.local_distance(
                state,
                FamilyAugmentedState(
                    q=np.asarray(target, dtype=float),
                    lambda_value=float(state.lambda_value if target_lambda is None else target_lambda),
                    discovered_round=int(state.discovered_round),
                ),
            )
            if family_space is not None
            else float(np.linalg.norm(np.asarray(state.q, dtype=float) - target)),
            state.discovered_round,
        )
    )
    return states[0]


def generate_family_ambient_proposals(
    transfer_family: ContinuousMaskedPlaneFamily,
    entry_seeds: list[StageSeed],
    right_center: np.ndarray,
    adaptive_lambda_values: set[float],
    target_lambda: float,
    round_idx: int,
    rng: np.random.Generator,
    proposal_count: int,
) -> list[np.ndarray]:
    proposals: list[np.ndarray] = []
    if len(entry_seeds) > 0:
        entry_points = np.asarray([np.asarray(seed.q, dtype=float) for seed in entry_seeds], dtype=float)
        entry_anchor = np.asarray(entry_points[rng.integers(len(entry_points))], dtype=float)
    else:
        entry_anchor = transfer_family.point_on_leaf(float(target_lambda))
    right_anchor = np.asarray(right_center, dtype=float)
    family_anchor = transfer_family.point_on_leaf(float(target_lambda))
    midpoint = 0.5 * (np.asarray(entry_anchor, dtype=float) + np.asarray(right_anchor, dtype=float))
    weakest_lambda = min(
        [float(v) for v in adaptive_lambda_values] or [float(target_lambda)],
        key=lambda lam: abs(float(lam) - float(target_lambda)),
    )
    weakest_anchor = transfer_family.point_on_leaf(float(weakest_lambda))
    count = max(1, int(proposal_count))
    for idx in range(count):
        selector = (round_idx + idx) % 5
        if selector == 0:
            q = np.asarray(midpoint, dtype=float) + rng.normal(scale=np.array([0.95, 0.85, 0.45], dtype=float), size=3)
        elif selector == 1:
            q = np.asarray(family_anchor, dtype=float) + rng.normal(scale=np.array([0.75, 0.90, 0.42], dtype=float), size=3)
        elif selector == 2:
            q = np.asarray(entry_anchor, dtype=float) + rng.normal(scale=np.array([0.85, 0.70, 0.35], dtype=float), size=3)
        elif selector == 3:
            q = np.asarray(weakest_anchor, dtype=float) + rng.normal(scale=np.array([0.85, 0.95, 0.38], dtype=float), size=3)
        else:
            bridge_anchor = 0.5 * (np.asarray(family_anchor, dtype=float) + np.asarray(weakest_anchor, dtype=float))
            q = np.asarray(bridge_anchor, dtype=float) + rng.normal(scale=np.array([0.70, 0.85, 0.34], dtype=float), size=3)
        proposals.append(np.asarray(q, dtype=float))
    return proposals


def compute_family_shared_proposal_count(
    transfer_family: ContinuousMaskedPlaneFamily,
    regions: list[FamilyEvidenceRegion],
    current_exit_seeds: list[ExitSeed],
    node_gain_history: list[int],
    lambda_region_history: list[int],
    region_update_history: list[int],
    useful_region_history: list[int],
) -> int:
    proposal_count = max(FAMILY_MIN_SHARED_PROPOSALS_PER_ROUND, FAMILY_SHARED_PROPOSALS_PER_ROUND)
    underexplored_regions = [region for region in regions if region.exit_support_count == 0 or region.update_count <= 1]
    active_stalled_regions = [region for region in regions if region.active and region.stagnating]
    if len(current_exit_seeds) == 0:
        proposal_count += 1
    if len(regions) <= 2 or len(underexplored_regions) >= max(2, len(regions) // 2):
        proposal_count += 1
    if len(regions) > 0:
        coverage_span = float(max(region.lambda_max for region in regions) - min(region.lambda_min for region in regions))
        total_span = float(transfer_family.lambda_max - transfer_family.lambda_min)
        if total_span > 1e-9 and coverage_span < 0.72 * total_span:
            proposal_count += 1
    if len(active_stalled_regions) > 0:
        proposal_count += 1
    if len(node_gain_history) >= 3 and sum(node_gain_history[-3:]) <= 1:
        proposal_count += 1
    if len(lambda_region_history) >= 3 and max(lambda_region_history[-3:]) == min(lambda_region_history[-3:]):
        proposal_count += 1
    if len(region_update_history) >= 3 and float(np.mean(region_update_history[-3:])) <= 1.0:
        proposal_count += 1
    if len(useful_region_history) >= 3 and float(np.mean(useful_region_history[-3:])) <= 1.25:
        proposal_count += 1
    return int(np.clip(proposal_count, FAMILY_MIN_SHARED_PROPOSALS_PER_ROUND, FAMILY_MAX_SHARED_PROPOSALS_PER_ROUND))


def project_proposal_to_family_regions(
    proposal_q: np.ndarray,
    transfer_family: ContinuousMaskedPlaneFamily,
    family_space: AugmentedFamilyConstrainedSpace,
    regions: list[FamilyEvidenceRegion],
    adaptive_lambda_values: set[float],
    target_lambda: float,
    rng: np.random.Generator,
    sample_id: int = 0,
    discovered_round: int = 0,
) -> list[tuple[float, float, np.ndarray, FamilyEvidenceRegion | None]]:
    hypotheses = family_space.sample_augmented_hypotheses(
        proposal_q=np.asarray(proposal_q, dtype=float),
        sample_id=int(sample_id),
        discovered_round=int(discovered_round),
        active_region_lambdas=[float(region.lambda_center) for region in regions],
        adaptive_lambda_values=adaptive_lambda_values,
        target_lambda=float(target_lambda),
        nominal_lambda=float(transfer_family.nominal_lambda),
        rng=rng,
        max_keep=FAMILY_MAX_REGIONS_PER_PROPOSAL + 1,
    )
    ranked_regions = sorted(
        regions,
        key=lambda region: (
            0 if region.active else 1,
            0 if region.exit_support_count == 0 else 1,
            region.update_count,
            0 if region.stagnating else 1,
            abs(float(region.lambda_center) - float(hypotheses.states[0].lambda_value)) if len(hypotheses.states) > 0 else 0.0,
        ),
    )

    ranked: list[tuple[float, float, np.ndarray, FamilyEvidenceRegion | None]] = []
    known_points = (
        np.asarray(
            [point for region in regions for point in np.asarray(region.explored_points, dtype=float)],
            dtype=float,
        )
        if len(regions) > 0
        else np.zeros((0, 3), dtype=float)
    )
    for projected_state in hypotheses.states:
        projected = np.asarray(projected_state.q, dtype=float)
        lam = float(projected_state.lambda_value)
        compatible_regions = [
            region
            for region in regions
            if float(region.lambda_min) - FAMILY_PROPOSAL_REGION_LAMBDA_TOL
            <= float(lam)
            <= float(region.lambda_max) + FAMILY_PROPOSAL_REGION_LAMBDA_TOL
        ]
        compatible_regions.sort(
            key=lambda region: (
                0 if region.active else 1,
                0 if region.exit_support_count == 0 else 1,
                region.update_count,
                abs(float(region.lambda_center) - float(lam)),
                -float(region.lambda_span),
            ),
        )
        matched_region = compatible_regions[0] if len(compatible_regions) > 0 else None
        novelty = (
            FAMILY_PROPOSAL_NOVELTY_RADIUS
            if len(known_points) == 0
            else min(
                FAMILY_PROPOSAL_NOVELTY_RADIUS,
                min(float(np.linalg.norm(np.asarray(projected, dtype=float) - point)) for point in known_points),
            )
        )
        margin = max(0.0, float(transfer_family.patch_margin(float(lam), np.asarray(projected, dtype=float))))
        underexplored_bonus = 0.32 if matched_region is None else 1.0 / (1.0 + float(matched_region.update_count))
        transition_bonus = 0.12 if matched_region is None else 0.08 * float(
            min(1, matched_region.entry_support_count) + min(1, matched_region.exit_support_count)
        )
        lagging_region_bonus = 0.0
        if matched_region is not None:
            lagging_region_bonus += 0.10 if matched_region.exit_support_count == 0 else 0.0
            lagging_region_bonus += 0.06 if matched_region.entry_support_count == 0 else 0.0
            lagging_region_bonus += 0.05 if matched_region.stagnating else 0.0
        creation_bonus = 0.08 if matched_region is None else 0.0
        score = (
            0.42 * float(novelty)
            + 0.26 * float(margin)
            + 0.18 * float(underexplored_bonus)
            + float(transition_bonus)
            + float(lagging_region_bonus)
            + float(creation_bonus)
        )
        ranked.append((float(score), float(lam), np.asarray(projected, dtype=float), matched_region))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[float, float, np.ndarray, FamilyEvidenceRegion | None]] = []
    used_regions: set[int] = set()
    for item in ranked:
        region = item[3]
        region_key = -1 if region is None else int(region.region_id)
        if region_key in used_regions:
            continue
        selected.append(item)
        used_regions.add(region_key)
        if len(selected) >= FAMILY_MAX_REGIONS_PER_PROPOSAL:
            break
    return selected


def should_stop_family_evidence_growth(
    local_round: int,
    exit_discovery_round: int | None,
    node_gain_history: list[int],
    exit_gain_history: list[int],
    lambda_region_history: list[int],
    region_update_history: list[int],
    useful_region_history: list[int],
    best_exit_support_history: list[float],
    continue_after_first_solution: bool,
    max_extra_rounds_after_first_solution: int | None,
) -> bool:
    if local_round < FAMILY_EVIDENCE_MIN_ROUNDS_BEFORE_SATURATION:
        return False
    if exit_discovery_round is not None:
        post_solution_floor = 0 if not continue_after_first_solution else FAMILY_POST_SOLUTION_MIN_ROUNDS
        if max_extra_rounds_after_first_solution is not None:
            post_solution_floor = max(post_solution_floor, max(0, int(max_extra_rounds_after_first_solution)))
        if local_round - int(exit_discovery_round) < int(post_solution_floor):
            return False
    window = FAMILY_EVIDENCE_SATURATION_WINDOW
    if len(node_gain_history) < window:
        return False
    recent_node_gain = sum(node_gain_history[-window:])
    recent_exit_gain = sum(exit_gain_history[-window:])
    recent_region_updates = float(np.mean(region_update_history[-window:])) if len(region_update_history) >= window else float("inf")
    recent_useful_regions = float(np.mean(useful_region_history[-window:])) if len(useful_region_history) >= window else float("inf")
    coverage_grew_recently = max(lambda_region_history[-window:]) > min(lambda_region_history[-window:])
    recent_support = [float(v) for v in best_exit_support_history[-window:] if np.isfinite(float(v))]
    support_improved_recently = False
    if len(recent_support) >= 2:
        support_improved_recently = (recent_support[0] - min(recent_support)) > FAMILY_SUPPORT_IMPROVEMENT_EPS
    return bool(
        recent_node_gain == 0
        and recent_exit_gain == 0
        and not coverage_grew_recently
        and not support_improved_recently
        and recent_region_updates <= 1.0
        and recent_useful_regions <= 1.0
    )


def update_family_region_progress(
    regions: list[FamilyEvidenceRegion],
    updated_region_ids: set[int],
    cluster_progress: dict[int, FamilyClusterProgress],
) -> None:
    for region in regions:
        progress = cluster_progress.setdefault(int(region.region_id), FamilyClusterProgress())
        residual_improved = float(region.best_right_residual) + CLUSTER_PROGRESS_EPS < float(progress.best_right_residual_seen)
        if int(region.region_id) in updated_region_ids or residual_improved or region.exit_support_count > 0:
            progress.no_gain_rounds = 0
            progress.last_gain_round += 1
        elif region.active:
            progress.no_gain_rounds += 1
        progress.best_right_residual_seen = min(float(progress.best_right_residual_seen), float(region.best_right_residual))


def bridge_committed_family_support(
    graph: FamilyConnectivityGraph,
    start_node_id: int,
    exit_seeds: list[ExitSeed],
    transfer_family: ContinuousMaskedPlaneFamily,
    frontier_ids: dict[str, list[int]],
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    adaptive_lambda_values: set[float],
    explored_edges_by_mode: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    family_transverse_edges: list[tuple[np.ndarray, np.ndarray]],
    round_sources: list[np.ndarray],
    round_targets: list[np.ndarray],
    round_idx: int,
    max_bridge_attempts: int = 10,
) -> int:
    if len(exit_seeds) == 0:
        return 0
    context = FamilyStageContext(
        graph=graph,
        transfer_family=transfer_family,
        family_space=AugmentedFamilyConstrainedSpace(transfer_family),
        right_manifold=None,
        right_center=np.zeros(3, dtype=float),
        right_radius=0.0,
        frontier_ids=frontier_ids,
        bounds_min=np.asarray(bounds_min, dtype=float),
        bounds_max=np.asarray(bounds_max, dtype=float),
        adaptive_lambda_values=adaptive_lambda_values,
        explored_edges_by_mode=explored_edges_by_mode,
        family_transverse_edges=family_transverse_edges,
        exit_transition_points=[],
        round_sources=round_sources,
        round_targets=round_targets,
    )
    bridge_count = 0
    attempt_count = 0
    while attempt_count < max_bridge_attempts:
        reachable_family_ids = start_reachable_family_node_ids(graph, int(start_node_id))
        reachable_exit_seeds = [
            seed for seed in exit_seeds if int(seed.family_node_id) in reachable_family_ids
        ]
        if len(reachable_exit_seeds) > 0:
            break
        reachable_states = family_states_from_ids(graph, sorted(reachable_family_ids))
        target_states = [
            state
            for state in family_states_from_ids(graph, [int(seed.family_node_id) for seed in exit_seeds])
            if int(state.node_id) not in reachable_family_ids
        ]
        if len(reachable_states) == 0 or len(target_states) == 0:
            break
        candidate_pairs: list[tuple[float, FamilyAugmentedState, FamilyAugmentedState]] = []
        for source_state in reachable_states[: min(28, len(reachable_states))]:
            for target_state in target_states[: min(12, len(target_states))]:
                lambda_gap = abs(float(source_state.lambda_value) - float(target_state.lambda_value))
                spatial_gap = float(
                    np.linalg.norm(np.asarray(source_state.q, dtype=float) - np.asarray(target_state.q, dtype=float))
                )
                candidate_pairs.append((
                    spatial_gap + 1.25 * lambda_gap,
                    source_state,
                    target_state,
                ))
        candidate_pairs.sort(key=lambda item: item[0])
        bridged = False
        for _score, source_state, target_state in candidate_pairs[: min(8, len(candidate_pairs))]:
            attempt_count += 1
            round_sources.append(np.asarray(source_state.q, dtype=float).copy())
            round_targets.append(np.asarray(target_state.q, dtype=float).copy())
            path, lambdas, ok = certify_transverse_family_connection(
                transfer_family=transfer_family,
                q_start=np.asarray(source_state.q, dtype=float),
                lambda_start=float(source_state.lambda_value),
                q_goal=np.asarray(target_state.q, dtype=float),
                lambda_goal=float(target_state.lambda_value),
            )
            if not ok:
                continue
            explored_edges_by_mode["family_transverse"] = merge_edges(
                explored_edges_by_mode.get("family_transverse", []),
                [(path[idx - 1], path[idx]) for idx in range(1, len(path))],
            )
            family_transverse_edges.extend(
                [(path[idx - 1].copy(), path[idx].copy()) for idx in range(1, len(path))]
            )
            for lam in np.asarray(lambdas, dtype=float):
                adaptive_lambda_values.add(float(lam))
                refine_lambda_region_if_promising(float(lam), adaptive_lambda_values, transfer_family)
            node_ids = add_certified_family_edge(
                graph=graph,
                transfer_family=transfer_family,
                path=path,
                lambdas=lambdas,
                round_idx=int(round_idx + bridge_count),
                frontier_ids=frontier_ids,
                guide_point=np.asarray(target_state.q, dtype=float),
                edge_kind="family_transverse",
                edge_label="family_bridge_transverse",
                origin_sample_id=round_idx,
                node_kind="bridge",
            )
            if len(node_ids) > 0:
                bridge_count += 1
                bridged = True
                break
        if not bridged:
            break
    return bridge_count


def choose_underexplored_family_region(
    regions: list[FamilyEvidenceRegion],
    cluster_progress: dict[int, FamilyClusterProgress],
    target_lambda: float | None,
    preferred_region_id: int | None = None,
) -> FamilyEvidenceRegion | None:
    if len(regions) == 0:
        return None
    ranked = sorted(
        regions,
        key=lambda region: (
            1 if region.stagnating else 0,
            0 if region.active else 1,
            0 if region.exit_support_count == 0 else 1,
            0 if region.entry_support_count > 0 else 1,
            cluster_progress.get(region.region_id, FamilyClusterProgress()).selection_count,
            0 if preferred_region_id is not None and int(region.region_id) == int(preferred_region_id) else 1,
            float("inf")
            if target_lambda is None
            else abs(float(region.lambda_center) - float(target_lambda)),
            region.best_right_residual,
            -region.lambda_span,
        ),
    )
    return ranked[0]


def choose_family_stage_mode(
    rng: np.random.Generator,
    cluster_count: int = 0,
    stalled_cluster_count: int = 0,
    exit_seed_count: int = 0,
) -> str:
    labels = [
        "family_local_novelty",
        "family_transverse_step",
        "family_exit_probe",
        "family_underexplored_region",
    ]
    probs = np.asarray(
        [
            P_FAMILY_LOCAL_NOVELTY,
            P_FAMILY_TRANSVERSE_STEP,
            P_FAMILY_EXIT_PROBE,
            P_FAMILY_UNDEREXPLORED_REGION,
        ],
        dtype=float,
    )
    if stalled_cluster_count > 0:
        probs[1] += 0.08
        probs[3] += 0.07
        probs[0] -= 0.05
    if cluster_count <= 1:
        probs[0] += 0.06
        probs[3] -= 0.03
    if exit_seed_count > 0:
        probs[2] += 0.04
        probs[1] += 0.03
    probs = probs / np.sum(probs)
    return str(rng.choice(labels, p=probs))


def choose_family_frontier_state(
    graph: FamilyConnectivityGraph,
    regions: list[FamilyEvidenceRegion],
    cluster_progress: dict[int, FamilyClusterProgress],
    mode: str,
    transfer_family: ContinuousMaskedPlaneFamily,
    right_manifold,
    preferred_region_id: int | None,
    target_lambda: float | None = None,
) -> tuple[FamilyAugmentedState | None, int | None]:
    region = choose_underexplored_family_region(
        regions=regions,
        cluster_progress=cluster_progress,
        target_lambda=target_lambda,
        preferred_region_id=preferred_region_id,
    )
    if region is None:
        return None, None

    states = [
        state
        for state in family_states_from_ids(graph, region.frontier_ids, cluster_id=int(region.region_id))
        if state.expansion_count < MAX_EXPANSIONS_PER_NODE
    ]
    if len(states) == 0:
        states = family_states_from_ids(graph, region.node_ids, cluster_id=int(region.region_id))
    if len(states) == 0:
        return None, int(region.region_id)

    if mode == "family_exit_probe":
        states.sort(
            key=lambda state: (
                float(np.linalg.norm(np.asarray(right_manifold.residual(state.q), dtype=float))),
                state.expansion_count,
                state.discovered_round,
            )
        )
        return states[0], int(region.region_id)

    if mode in {"family_transverse_step", "family_underexplored_region"} and target_lambda is not None:
        states.sort(
            key=lambda state: (
                abs(float(state.lambda_value) - float(target_lambda)),
                state.expansion_count,
                -float(transfer_family.patch_margin(float(state.lambda_value), state.q)),
                state.discovered_round,
            )
        )
        return states[0], int(region.region_id)

    states.sort(
        key=lambda state: (
            state.expansion_count,
            0 if region.exit_support_count == 0 else 1,
            -float(transfer_family.patch_margin(float(state.lambda_value), state.q)),
            state.discovered_round,
        )
    )
    return states[0], int(region.region_id)


def sample_target_on_family_leaf(
    family_space: AugmentedFamilyConstrainedSpace,
    source_q: np.ndarray,
    lam: float,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    transfer_family = family_space.transfer_family
    manifold = transfer_family.manifold(float(lam))
    target = ompl_native_exploration_target(
        manifold=manifold,
        q_seed=np.asarray(source_q, dtype=float),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )
    if target is not None and transfer_family.within_patch(float(lam), np.asarray(target, dtype=float)):
        return np.asarray(target, dtype=float)

    u_coord, v_coord = transfer_family.patch_coords(float(lam), np.asarray(source_q, dtype=float))
    margin = max(0.08, transfer_family.patch_margin(float(lam), np.asarray(source_q, dtype=float)))
    du = float(np.clip(rng.normal(scale=0.45 * margin), -margin, margin))
    dv = float(np.clip(rng.normal(scale=0.75 * margin), -margin, margin))
    guess = (
        transfer_family.point_on_leaf(float(lam))
        + (u_coord + du) * np.asarray(transfer_family._basis_u, dtype=float)
        + (v_coord + dv) * np.asarray(transfer_family._basis_v, dtype=float)
    )
    projected_state = family_space.project_augmented(
        x_guess=family_space.join(guess, float(lam)),
        lambda_hint=float(lam),
    )
    return np.asarray(source_q if projected_state is None else projected_state.q, dtype=float)


def expand_family_stage_locally(
    context: FamilyStageContext,
    source_state: FamilyAugmentedState,
    round_idx: int,
    rng: np.random.Generator,
    target_lambda: float | None = None,
) -> list[int]:
    source_state = increment_family_state_expansion(context, source_state)
    local_target_lambda = float(source_state.lambda_value) if target_lambda is None else float(target_lambda)
    target_state = context.family_space.sample_local_state(
        source_state=source_state,
        rng=rng,
        bounds_min=context.bounds_min,
        bounds_max=context.bounds_max,
        target_lambda=local_target_lambda,
    )
    if target_state is None:
        return []
    context.round_sources.append(np.asarray(source_state.q, dtype=float).copy())
    context.round_targets.append(np.asarray(target_state.q, dtype=float).copy())
    return plan_augmented_family_motion(
        context=context,
        source_state=source_state,
        target_state=target_state,
        round_idx=round_idx,
        guide_point=np.asarray(target_state.q, dtype=float),
        origin_sample_id=round_idx,
    )


def attempt_family_transverse_step(
    context: FamilyStageContext,
    source_state: FamilyAugmentedState,
    round_idx: int,
    target_lambda: float | None,
    rng: np.random.Generator,
) -> tuple[list[int], bool]:
    source_state = increment_family_state_expansion(context, source_state)
    lam_src = float(source_state.lambda_value)
    if target_lambda is None or abs(float(target_lambda) - lam_src) < 1e-6:
        target_lambda = clamp_lambda(
            context.transfer_family,
            lam_src + float(rng.choice([-1.0, 1.0])) * FAMILY_TRANSVERSE_DELTA,
        )
    else:
        step = float(np.clip(float(target_lambda) - lam_src, -FAMILY_TRANSVERSE_DELTA, FAMILY_TRANSVERSE_DELTA))
        target_lambda = clamp_lambda(context.transfer_family, lam_src + step)
    if abs(float(target_lambda) - lam_src) < 0.5 * LAMBDA_BIN_WIDTH:
        return [], False

    target_state = context.family_space.sample_local_state(
        source_state=source_state,
        rng=rng,
        bounds_min=context.bounds_min,
        bounds_max=context.bounds_max,
        target_lambda=float(target_lambda),
    )
    if target_state is None or abs(float(target_state.lambda_value) - lam_src) < 0.5 * LAMBDA_BIN_WIDTH:
        return [], False
    target_q = np.asarray(target_state.q, dtype=float)

    context.round_sources.append(np.asarray(source_state.q, dtype=float).copy())
    context.round_targets.append(np.asarray(target_q, dtype=float).copy())
    node_ids = plan_augmented_family_motion(
        context=context,
        source_state=source_state,
        target_state=target_state,
        round_idx=round_idx,
        guide_point=np.asarray(target_q, dtype=float),
        origin_sample_id=round_idx,
    )
    return node_ids, len(node_ids) > 0


def discover_exit_seeds_to_right(
    context: FamilyStageContext,
    family_states: Iterable[FamilyAugmentedState],
    round_idx: int,
    node_cluster_labels: dict[int, int] | None = None,
    source_cluster_id: int | None = None,
) -> list[ExitSeed]:
    seeds: list[ExitSeed] = []
    for family_state in family_states:
        lam = float(family_state.lambda_value)
        transfer_manifold = context.transfer_family.manifold(lam)
        residual_norm = float(np.linalg.norm(np.asarray(context.right_manifold.residual(family_state.q), dtype=float)))
        if residual_norm > 0.35:
            continue
        refined, ok = refine_intersection_on_both_manifolds(
            manifold_a=context.right_manifold,
            manifold_b=transfer_manifold,
            x0=np.asarray(family_state.q, dtype=float),
            tol=1e-8,
            max_iters=32,
        )
        if not ok:
            continue
        refined_q = np.asarray(refined, dtype=float)
        if not context.transfer_family.within_patch(lam, refined_q):
            continue
        if float(np.linalg.norm(np.asarray(context.right_manifold.residual(refined_q), dtype=float))) > TRANSITION_INTERSECTION_TOL:
            continue
        ok, failures = validate_transition_edge(
            path=np.asarray([refined_q], dtype=float),
            side_manifold=context.right_manifold,
            transfer_family=context.transfer_family,
            lam=float(lam),
            side_center=np.asarray(context.right_center, dtype=float),
            side_radius=float(context.right_radius),
            side_mode="right",
        )
        if not ok:
            report_strict_validation_failures("Rejected family-to-right exit seed", failures)
            continue

        family_path = certify_family_leaf_connection(
            transfer_family=context.transfer_family,
            source_q=np.asarray(family_state.q, dtype=float),
            target_q=refined_q,
            lam=lam,
            bounds_min=context.bounds_min,
            bounds_max=context.bounds_max,
        )
        if len(family_path) == 0:
            continue
        family_ids = add_certified_family_edge(
            graph=context.graph,
            transfer_family=context.transfer_family,
            path=family_path,
            lambdas=np.full((len(family_path),), lam, dtype=float),
            round_idx=round_idx,
            frontier_ids=context.frontier_ids,
            guide_point=np.asarray(refined_q, dtype=float),
            edge_kind="family_leaf_motion",
            edge_label="family_leaf_motion",
            origin_sample_id=family_state.origin_sample_id,
            node_kind="transition",
        )
        if len(family_ids) == 0:
            continue
        if node_cluster_labels is not None and source_cluster_id is not None:
            for family_id in family_ids:
                node_cluster_labels[int(family_id)] = int(source_cluster_id)
        right_node_id = context.graph.register_node(
            mode="right",
            q=refined_q,
            discovered_round=round_idx,
            kind="transition",
            origin_sample_id=family_state.origin_sample_id,
        )
        register_frontier_node(context.frontier_ids, context.graph, "right", right_node_id, np.asarray(refined_q, dtype=float))
        _edge_id, is_new = context.graph.add_edge(
            node_u=int(family_ids[-1]),
            node_v=int(right_node_id),
            kind="exit_transition",
            cost=0.0,
            path=np.asarray([refined_q], dtype=float),
            path_lambdas=np.asarray([lam], dtype=float),
            label="family_right_exit",
            lambda_value=float(lam),
        )
        if is_new:
            context.exit_transition_points.append(refined_q.copy())
            seeds.append(
                ExitSeed(
                    family_node_id=int(family_ids[-1]),
                    right_node_id=int(right_node_id),
                    q=refined_q.copy(),
                    lambda_value=float(lam),
                    discovered_round=int(round_idx),
                    cluster_id=None if source_cluster_id is None else int(source_cluster_id),
                )
            )
    return seeds


def recover_primary_leaf_exit_seed(
    context: FamilyStageContext,
    primary_entry_seed: StageSeed | None,
    round_idx: int,
    node_cluster_labels: dict[int, int] | None = None,
) -> list[ExitSeed]:
    if primary_entry_seed is None:
        return []
    primary_state = family_state_from_node(
        context.graph,
        int(primary_entry_seed.family_node_id),
        cluster_id=None if primary_entry_seed.cluster_id is None else int(primary_entry_seed.cluster_id),
    )
    if primary_state is None:
        return []

    lam = float(primary_entry_seed.lambda_value)
    transfer_manifold = context.transfer_family.manifold(lam)
    projected_guesses: list[np.ndarray] = []
    for guess in (
        np.asarray(context.right_center, dtype=float),
        0.5 * (np.asarray(primary_state.q, dtype=float) + np.asarray(context.right_center, dtype=float)),
        np.asarray(primary_state.q, dtype=float) + 0.8 * (np.asarray(context.right_center, dtype=float) - np.asarray(primary_state.q, dtype=float)),
    ):
        projected = project_valid_family_state(context.transfer_family, lam, guess)
        if projected is not None:
            projected_guesses.append(np.asarray(projected, dtype=float))

    for guess in projected_guesses:
        refined, ok = refine_intersection_on_both_manifolds(
            manifold_a=context.right_manifold,
            manifold_b=transfer_manifold,
            x0=np.asarray(guess, dtype=float),
            tol=1e-8,
            max_iters=48,
        )
        if not ok:
            continue
        refined_q = np.asarray(refined, dtype=float)
        if not context.transfer_family.within_patch(lam, refined_q):
            continue
        if float(np.linalg.norm(np.asarray(context.right_manifold.residual(refined_q), dtype=float))) > TRANSITION_INTERSECTION_TOL:
            continue
        ok, failures = validate_transition_edge(
            path=np.asarray([refined_q], dtype=float),
            side_manifold=context.right_manifold,
            transfer_family=context.transfer_family,
            lam=float(lam),
            side_center=np.asarray(context.right_center, dtype=float),
            side_radius=float(context.right_radius),
            side_mode="right",
        )
        if not ok:
            report_strict_validation_failures("Rejected primary-leaf exit recovery", failures)
            continue

        family_path = certify_family_leaf_connection(
            transfer_family=context.transfer_family,
            source_q=np.asarray(primary_state.q, dtype=float),
            target_q=refined_q,
            lam=lam,
            bounds_min=context.bounds_min,
            bounds_max=context.bounds_max,
        )
        if len(family_path) == 0:
            continue
        family_ids = add_certified_family_edge(
            graph=context.graph,
            transfer_family=context.transfer_family,
            path=family_path,
            lambdas=np.full((len(family_path),), lam, dtype=float),
            round_idx=round_idx,
            frontier_ids=context.frontier_ids,
            guide_point=np.asarray(refined_q, dtype=float),
            edge_kind="family_leaf_motion",
            edge_label="family_primary_leaf_exit",
            origin_sample_id=primary_state.origin_sample_id,
            node_kind="transition",
        )
        if len(family_ids) == 0:
            continue
        if node_cluster_labels is not None and primary_entry_seed.cluster_id is not None:
            for family_id in family_ids:
                node_cluster_labels[int(family_id)] = int(primary_entry_seed.cluster_id)
        right_node_id = context.graph.register_node(
            mode="right",
            q=refined_q,
            discovered_round=round_idx,
            kind="transition",
            origin_sample_id=primary_state.origin_sample_id,
        )
        register_frontier_node(context.frontier_ids, context.graph, "right", right_node_id, np.asarray(refined_q, dtype=float))
        _edge_id, is_new = context.graph.add_edge(
            node_u=int(family_ids[-1]),
            node_v=int(right_node_id),
            kind="exit_transition",
            cost=0.0,
            path=np.asarray([refined_q], dtype=float),
            path_lambdas=np.asarray([lam], dtype=float),
            label="family_right_primary_leaf_exit",
            lambda_value=float(lam),
        )
        if not is_new:
            continue
        context.exit_transition_points.append(refined_q.copy())
        return [
            ExitSeed(
                family_node_id=int(family_ids[-1]),
                right_node_id=int(right_node_id),
                q=refined_q.copy(),
                lambda_value=float(lam),
                discovered_round=int(round_idx),
                cluster_id=None if primary_entry_seed.cluster_id is None else int(primary_entry_seed.cluster_id),
            )
        ]
    return []


def plan_within_family_stage(
    graph: FamilyConnectivityGraph,
    start_node_id: int,
    entry_seeds: list[StageSeed],
    initial_exit_seeds: list[ExitSeed],
    transfer_family: ContinuousMaskedPlaneFamily,
    right_manifold,
    right_center: np.ndarray,
    right_radius: float,
    frontier_ids: dict[str, list[int]],
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    adaptive_lambda_values: set[float],
    explored_edges_by_mode: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    family_transverse_edges: list[tuple[np.ndarray, np.ndarray]],
    exit_transition_points: list[np.ndarray],
    round_sources: list[np.ndarray],
    round_targets: list[np.ndarray],
    rng: np.random.Generator,
    round_offset: int,
    family_round_budget: int,
    continue_after_first_solution: bool = True,
    max_extra_rounds_after_first_solution: int | None = None,
) -> FamilyStageResult:
    """Delegate the preserved family-stage exploration loop to the evidence explorer."""

    from .algorithms.evidence_explorer import FamilyEvidenceExplorer

    explorer = FamilyEvidenceExplorer(
        graph=graph,
        start_node_id=start_node_id,
        entry_seeds=entry_seeds,
        initial_exit_seeds=initial_exit_seeds,
        transfer_family=transfer_family,
        right_manifold=right_manifold,
        right_center=np.asarray(right_center, dtype=float),
        right_radius=float(right_radius),
        frontier_ids=frontier_ids,
        bounds_min=np.asarray(bounds_min, dtype=float),
        bounds_max=np.asarray(bounds_max, dtype=float),
        adaptive_lambda_values=adaptive_lambda_values,
        explored_edges_by_mode=explored_edges_by_mode,
        family_transverse_edges=family_transverse_edges,
        exit_transition_points=exit_transition_points,
        round_sources=round_sources,
        round_targets=round_targets,
        rng=rng,
        round_offset=int(round_offset),
        family_round_budget=int(family_round_budget),
        continue_after_first_solution=bool(continue_after_first_solution),
        max_extra_rounds_after_first_solution=max_extra_rounds_after_first_solution,
    )
    return explorer.run()
