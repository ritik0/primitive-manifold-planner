"""Right-stage closure logic for the continuous-transfer planner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (
    MAX_EXPANSIONS_PER_NODE,
    MAX_RIGHT_GOAL_CONNECTION_ATTEMPTS,
    POST_EXIT_RIGHT_CLOSURE_ROUNDS,
    RIGHT_GOAL_WARMSTART_ROUNDS,
    RIGHT_STAGE_BASE_ROUNDS,
    TRANSVERSE_GOAL_TOL,
)
from .graph_types import ExitSeed, FamilyConnectivityGraph, RightStageClosureResult, StageState
from .graph_paths import shortest_path_over_graph
from .graph_insertions import add_path_nodes_to_graph
from .stage_state_utils import (
    coerce_stage_state,
    increment_stage_state_expansion,
    stage_state_from_node,
    stage_states_from_ids,
)
from .strict_validation import (
    report_strict_validation_failures,
    sample_strict_sphere_motion_path,
    validate_left_motion_edge,
    validate_right_motion_edge,
)
from .support import (
    explore_on_manifold_from_frontier,
    merge_edges,
    ompl_native_exploration_target,
    solve_exact_segment_on_manifold,
)


@dataclass
class RightStageContext:
    """Shared right-stage resources so local closure logic can work on stage states directly."""

    graph: FamilyConnectivityGraph
    right_manifold: object
    right_center: np.ndarray
    right_radius: float
    goal_q: np.ndarray
    frontier_ids: dict[str, list[int]]
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    explored_edges_by_mode: dict[str, list[tuple[np.ndarray, np.ndarray]]]
    round_sources: list[np.ndarray]
    round_targets: list[np.ndarray]
    exit_anchor_points: np.ndarray

def certify_side_connection_to_target(
    graph: FamilyConnectivityGraph,
    source_node_id: int | StageState,
    side_mode: str,
    side_manifold,
    sphere_center: np.ndarray,
    sphere_radius: float,
    target_q: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    round_idx: int,
    frontier_ids: dict[str, list[int]],
    guide_point: np.ndarray,
    explored_edges_by_mode: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    origin_sample_id: int | None = None,
) -> list[int]:
    source_state = coerce_stage_state(graph, source_node_id, expected_mode=str(side_mode))
    if source_state is None:
        return []

    source_q = np.asarray(source_state.q, dtype=float)
    target = np.asarray(target_q, dtype=float)
    center = np.asarray(sphere_center, dtype=float)
    result = solve_exact_segment_on_manifold(
        manifold=side_manifold,
        x_start=source_q,
        x_goal=target,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )
    path = np.asarray(getattr(result, "path", np.zeros((0, 3), dtype=float)), dtype=float)
    if len(path) == 0 or np.linalg.norm(path[-1] - target) > TRANSVERSE_GOAL_TOL:
        if str(side_mode) != "right":
            return []
        path = sample_strict_sphere_motion_path(
            center=center,
            radius=float(sphere_radius),
            q_start=source_q,
            q_goal=target,
        )

    if str(side_mode) == "left":
        ok, failures = validate_left_motion_edge(path, side_manifold, center, float(sphere_radius))
    else:
        ok, failures = validate_right_motion_edge(path, side_manifold, center, float(sphere_radius))
    if not ok:
        if str(side_mode) != "right":
            report_strict_validation_failures(f"Rejected {side_mode} connection to target", failures)
            return []
        fallback_path = sample_strict_sphere_motion_path(
            center=center,
            radius=float(sphere_radius),
            q_start=source_q,
            q_goal=target,
        )
        if str(side_mode) == "left":
            ok, failures = validate_left_motion_edge(fallback_path, side_manifold, center, float(sphere_radius))
        else:
            ok, failures = validate_right_motion_edge(fallback_path, side_manifold, center, float(sphere_radius))
        if not ok:
            report_strict_validation_failures(f"Rejected {side_mode} connection to target", failures)
            return []
        path = fallback_path

    explored_edges_by_mode[side_mode] = merge_edges(
        explored_edges_by_mode.get(side_mode, []),
        list(getattr(result, "explored_edges", [])),
    )
    return add_path_nodes_to_graph(
        graph=graph,
        mode=side_mode,
        path=path,
        round_idx=round_idx,
        frontier_ids=frontier_ids,
        guide_point=np.asarray(guide_point, dtype=float),
        origin_sample_id=source_state.origin_sample_id if origin_sample_id is None else origin_sample_id,
        edge_kind=f"{side_mode}_motion",
        edge_label=f"{side_mode}_motion",
        side_manifold=side_manifold,
        sphere_center=center,
        sphere_radius=float(sphere_radius),
    )


def choose_right_stage_source(context: RightStageContext) -> StageState | None:
    candidate_states = [
        state
        for state in stage_states_from_ids(
            context.graph,
            context.frontier_ids.get("right", []),
            expected_mode="right",
        )
        if state.expansion_count < MAX_EXPANSIONS_PER_NODE + 2
    ]
    if len(candidate_states) == 0:
        return None

    goal = np.asarray(context.goal_q, dtype=float)
    candidate_states.sort(
        key=lambda state: (
            state.expansion_count,
            0 if state.kind != "goal" else 1,
            float(np.linalg.norm(np.asarray(state.q, dtype=float) - goal)),
            state.discovered_round,
        )
    )
    return candidate_states[0]


def sample_right_stage_target(
    right_manifold,
    source_q: np.ndarray,
    goal_q: np.ndarray,
    anchor_points: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    from primitive_manifold_planner.projection import project_newton

    source = np.asarray(source_q, dtype=float)
    goal = np.asarray(goal_q, dtype=float)
    anchors = np.asarray(anchor_points, dtype=float)
    if len(anchors) > 0:
        anchor = anchors[int(np.argmin(np.linalg.norm(anchors - source, axis=1)))]
    else:
        anchor = goal
    target_bias = anchor if np.linalg.norm(source - goal) <= 1e-6 else goal
    if rng.random() < 0.65:
        guess = 0.55 * source + 0.45 * target_bias + rng.normal(scale=0.30, size=3)
        projection = project_newton(
            manifold=right_manifold,
            x0=np.asarray(guess, dtype=float),
            tol=1e-10,
            max_iters=80,
            damping=1.0,
        )
        if projection.success:
            projected = np.asarray(projection.x_projected, dtype=float)
            if np.linalg.norm(projected - source) > 1e-5:
                return projected
    target = ompl_native_exploration_target(
        manifold=right_manifold,
        q_seed=source,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )
    if target is not None and np.linalg.norm(np.asarray(target, dtype=float) - source) > 1e-5:
        return np.asarray(target, dtype=float)
    return goal.copy()


def expand_right_stage_locally(
    context: RightStageContext,
    source_state: StageState,
    round_idx: int,
    rng: np.random.Generator,
) -> list[int]:
    source_state = increment_stage_state_expansion(context.graph, source_state)
    source_q = np.asarray(source_state.q, dtype=float)
    target_q = sample_right_stage_target(
        right_manifold=context.right_manifold,
        source_q=source_q,
        goal_q=np.asarray(context.goal_q, dtype=float),
        anchor_points=np.asarray(context.exit_anchor_points, dtype=float),
        bounds_min=context.bounds_min,
        bounds_max=context.bounds_max,
        rng=rng,
    )
    context.round_sources.append(source_q.copy())
    context.round_targets.append(target_q.copy())
    result = explore_on_manifold_from_frontier(
        manifold=context.right_manifold,
        x_start=source_q,
        x_goal=target_q,
        bounds_min=context.bounds_min,
        bounds_max=context.bounds_max,
    )
    context.explored_edges_by_mode["right"] = merge_edges(
        context.explored_edges_by_mode.get("right", []),
        list(getattr(result, "explored_edges", [])),
    )
    path = np.asarray(getattr(result, "path", np.zeros((0, 3), dtype=float)), dtype=float)
    return add_path_nodes_to_graph(
        graph=context.graph,
        mode="right",
        path=path,
        round_idx=round_idx,
        frontier_ids=context.frontier_ids,
        guide_point=np.asarray(context.goal_q, dtype=float),
        origin_sample_id=round_idx,
        edge_kind="right_motion",
        edge_label="right_motion",
        side_manifold=context.right_manifold,
        sphere_center=np.asarray(context.right_center, dtype=float),
        sphere_radius=float(context.right_radius),
    )


def attempt_goal_connection_from_right(
    context: RightStageContext,
    source_state: StageState,
    round_idx: int,
) -> list[int]:
    source_state = increment_stage_state_expansion(context.graph, source_state)
    return certify_side_connection_to_target(
        graph=context.graph,
        source_node_id=source_state,
        side_mode="right",
        side_manifold=context.right_manifold,
        sphere_center=np.asarray(context.right_center, dtype=float),
        sphere_radius=float(context.right_radius),
        target_q=np.asarray(context.goal_q, dtype=float),
        bounds_min=context.bounds_min,
        bounds_max=context.bounds_max,
        round_idx=round_idx,
        frontier_ids=context.frontier_ids,
        guide_point=np.asarray(context.goal_q, dtype=float),
        explored_edges_by_mode=context.explored_edges_by_mode,
        origin_sample_id=source_state.origin_sample_id,
    )


def collect_right_goal_component(graph: FamilyConnectivityGraph, goal_node_id: int) -> set[int]:
    component: set[int] = set()
    stack = [int(goal_node_id)]
    while len(stack) > 0:
        node_id = int(stack.pop())
        if node_id in component:
            continue
        state = stage_state_from_node(graph, int(node_id), expected_mode="right")
        if state is None:
            continue
        component.add(int(state.node_id))
        for edge_id in graph.adjacency.get(int(state.node_id), []):
            edge = graph.edges[int(edge_id)]
            if str(edge.kind) != "right_motion":
                continue
            other = int(edge.node_v if edge.node_u == int(state.node_id) else edge.node_u)
            if other not in component:
                stack.append(other)
    return component


def attempt_right_stage_bridge(
    context: RightStageContext,
    source_state: StageState,
    target_state: StageState,
    round_idx: int,
) -> list[int]:
    if int(source_state.node_id) == int(target_state.node_id):
        return []
    source_state = increment_stage_state_expansion(context.graph, source_state)
    return certify_side_connection_to_target(
        graph=context.graph,
        source_node_id=source_state,
        side_mode="right",
        side_manifold=context.right_manifold,
        sphere_center=np.asarray(context.right_center, dtype=float),
        sphere_radius=float(context.right_radius),
        target_q=np.asarray(target_state.q, dtype=float),
        bounds_min=context.bounds_min,
        bounds_max=context.bounds_max,
        round_idx=round_idx,
        frontier_ids=context.frontier_ids,
        guide_point=np.asarray(target_state.q, dtype=float),
        explored_edges_by_mode=context.explored_edges_by_mode,
        origin_sample_id=source_state.origin_sample_id,
    )


def robust_right_stage_closure(
    graph: FamilyConnectivityGraph,
    start_node_id: int,
    goal_node_id: int,
    exit_seeds: list[ExitSeed],
    right_manifold,
    right_center: np.ndarray,
    right_radius: float,
    goal_q: np.ndarray,
    frontier_ids: dict[str, list[int]],
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    round_idx_offset: int,
    explored_edges_by_mode: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    round_sources: list[np.ndarray],
    round_targets: list[np.ndarray],
    rng: np.random.Generator,
    target_successful_seed_count: int = 1,
) -> RightStageClosureResult:
    if len(exit_seeds) == 0:
        return RightStageClosureResult(right_frontier_count=len(frontier_ids.get("right", [])))

    context = RightStageContext(
        graph=graph,
        right_manifold=right_manifold,
        right_center=np.asarray(right_center, dtype=float),
        right_radius=float(right_radius),
        goal_q=np.asarray(goal_q, dtype=float),
        frontier_ids=frontier_ids,
        bounds_min=np.asarray(bounds_min, dtype=float),
        bounds_max=np.asarray(bounds_max, dtype=float),
        explored_edges_by_mode=explored_edges_by_mode,
        round_sources=round_sources,
        round_targets=round_targets,
        exit_anchor_points=np.asarray([np.asarray(seed.q, dtype=float) for seed in exit_seeds], dtype=float),
    )
    goal_state = stage_state_from_node(graph, int(goal_node_id), expected_mode="right")
    if goal_state is None:
        return RightStageClosureResult(right_frontier_count=len(frontier_ids.get("right", [])))

    closure_attempt_count = 0
    successful_seed_count = 0
    goal_connection_attempts = 0
    goal_connection_successes = 0
    best_goal_residual_seen = float("inf")
    best_goal_distance_seen = float("inf")
    seed_attempted_ids: set[int] = set()
    route_found = False
    rounds_used = 0
    route_discovery_round: int | None = None
    effective_budget = max(RIGHT_STAGE_BASE_ROUNDS, len(exit_seeds) + POST_EXIT_RIGHT_CLOSURE_ROUNDS)

    for warm_idx in range(min(RIGHT_GOAL_WARMSTART_ROUNDS, effective_budget)):
        warm_round_idx = int(round_idx_offset + warm_idx + 1)
        closure_attempt_count += 1
        rounds_used += 1
        expand_right_stage_locally(
            context=context,
            source_state=goal_state,
            round_idx=warm_round_idx,
            rng=rng,
        )

    ranked_exit_seeds = sorted(
        exit_seeds,
        key=lambda seed: (
            seed.discovered_round,
            float(np.linalg.norm(np.asarray(seed.q, dtype=float) - np.asarray(goal_q, dtype=float))),
        ),
    )

    target_successful_seed_count = max(1, int(target_successful_seed_count))

    for seed in ranked_exit_seeds:
        seed_state = stage_state_from_node(graph, int(seed.right_node_id), expected_mode="right")
        if seed_state is None:
            continue

        closure_attempt_count += 1
        seed_attempted_ids.add(int(seed_state.node_id))
        rounds_used += 1
        goal_component = collect_right_goal_component(graph, int(goal_node_id))
        bridge_target_states = stage_states_from_ids(
            graph,
            [node_id for node_id in goal_component if int(node_id) != int(seed_state.node_id)],
            expected_mode="right",
        )
        if len(bridge_target_states) > 0:
            bridge_target_states.sort(
                key=lambda state: float(
                    np.linalg.norm(np.asarray(state.q, dtype=float) - np.asarray(seed_state.q, dtype=float))
                )
            )
            bridge_ids = attempt_right_stage_bridge(
                context=context,
                source_state=seed_state,
                target_state=bridge_target_states[0],
                round_idx=int(round_idx_offset + rounds_used),
            )
            if len(bridge_ids) > 0:
                _graph_cost, node_path, _edge_path = shortest_path_over_graph(graph, int(start_node_id), int(goal_node_id))
                if len(node_path) > 0:
                    successful_seed_count += 1
                    route_found = True
                    if route_discovery_round is None:
                        route_discovery_round = int(rounds_used)
                    if successful_seed_count >= target_successful_seed_count:
                        break

        goal_connection_attempts += 1
        goal_ids = attempt_goal_connection_from_right(
            context=context,
            source_state=seed_state,
            round_idx=int(round_idx_offset + rounds_used),
        )
        best_goal_distance_seen = min(
            best_goal_distance_seen,
            float(np.linalg.norm(np.asarray(seed_state.q, dtype=float) - np.asarray(goal_q, dtype=float))),
        )
        best_goal_residual_seen = min(
            best_goal_residual_seen,
            float(np.linalg.norm(np.asarray(right_manifold.residual(np.asarray(seed_state.q, dtype=float)), dtype=float))),
        )
        seed_success = len(goal_ids) > 0 and int(goal_node_id) in [int(node_id) for node_id in goal_ids]
        if seed_success:
            goal_connection_successes += 1
            successful_seed_count += 1
            route_found = True
            if route_discovery_round is None:
                route_discovery_round = int(rounds_used)
            if successful_seed_count >= target_successful_seed_count:
                break
        if goal_connection_attempts >= MAX_RIGHT_GOAL_CONNECTION_ATTEMPTS:
            break

    local_round = len(seed_attempted_ids) + min(RIGHT_GOAL_WARMSTART_ROUNDS, effective_budget)
    while local_round < effective_budget and goal_connection_attempts < MAX_RIGHT_GOAL_CONNECTION_ATTEMPTS:
        closure_attempt_count += 1
        source_state = choose_right_stage_source(context)
        if source_state is None:
            break

        source_q = np.asarray(source_state.q, dtype=float)
        goal_component = collect_right_goal_component(graph, int(goal_node_id))
        best_goal_distance_seen = min(
            best_goal_distance_seen,
            float(np.linalg.norm(source_q - np.asarray(goal_q, dtype=float))),
        )
        best_goal_residual_seen = min(
            best_goal_residual_seen,
            float(np.linalg.norm(np.asarray(right_manifold.residual(source_q), dtype=float))),
        )
        new_right_ids = expand_right_stage_locally(
            context=context,
            source_state=source_state,
            round_idx=int(round_idx_offset + local_round + 1),
            rng=rng,
        )
        candidate_states = [source_state] + stage_states_from_ids(graph, new_right_ids, expected_mode="right")
        unique_candidate_states = list({int(state.node_id): state for state in candidate_states}.values())

        if len(goal_component) > 0:
            bridge_sources = [
                state for state in unique_candidate_states if int(state.node_id) not in goal_component
            ]
            goal_component_targets = stage_states_from_ids(
                graph,
                [node_id for node_id in goal_component if int(node_id) != int(source_state.node_id)],
                expected_mode="right",
            )
            for bridge_source in bridge_sources[:2]:
                if len(goal_component_targets) == 0:
                    break
                goal_component_targets.sort(
                    key=lambda state: float(
                        np.linalg.norm(
                            np.asarray(state.q, dtype=float) - np.asarray(bridge_source.q, dtype=float)
                        )
                    )
                )
                bridge_ids = attempt_right_stage_bridge(
                    context=context,
                    source_state=bridge_source,
                    target_state=goal_component_targets[0],
                    round_idx=int(round_idx_offset + local_round + 1),
                )
                if len(bridge_ids) > 0:
                    goal_component = collect_right_goal_component(graph, int(goal_node_id))
                    break

        unique_candidate_states.sort(
            key=lambda state: (
                float(np.linalg.norm(np.asarray(state.q, dtype=float) - np.asarray(goal_q, dtype=float))),
                state.expansion_count,
            )
        )
        for candidate_state in unique_candidate_states[:2]:
            if goal_connection_attempts >= MAX_RIGHT_GOAL_CONNECTION_ATTEMPTS:
                break
            goal_connection_attempts += 1
            goal_ids = attempt_goal_connection_from_right(
                context=context,
                source_state=candidate_state,
                round_idx=int(round_idx_offset + local_round + 1),
            )
            if len(goal_ids) > 0 and int(goal_node_id) in [int(candidate) for candidate in goal_ids]:
                goal_connection_successes += 1
                successful_seed_count += 1
                route_found = True
                if route_discovery_round is None:
                    route_discovery_round = int(local_round + 1)
                if successful_seed_count >= target_successful_seed_count:
                    break
        if route_found and successful_seed_count >= target_successful_seed_count:
            break

        _graph_cost, node_path, _edge_path = shortest_path_over_graph(graph, int(start_node_id), int(goal_node_id))
        if len(node_path) > 0:
            route_found = True
            if route_discovery_round is None:
                route_discovery_round = int(local_round + 1)
            if successful_seed_count >= target_successful_seed_count:
                break
        local_round += 1

    successful_seed_ids = [
        int(seed.right_node_id)
        for seed in ranked_exit_seeds
        if len(shortest_path_over_graph(graph, int(seed.right_node_id), int(goal_node_id))[1]) > 0
    ]
    successful_seed_count = len(successful_seed_ids)
    failed_seed_count = max(0, len(seed_attempted_ids) - successful_seed_count)
    goal_connection_failures = goal_connection_attempts - goal_connection_successes
    _graph_cost, node_path, _edge_path = shortest_path_over_graph(graph, int(start_node_id), int(goal_node_id))
    route_found = route_found or len(node_path) > 0

    return RightStageClosureResult(
        rounds=max(local_round, len(seed_attempted_ids)),
        closure_attempt_count=closure_attempt_count,
        successful_seed_count=successful_seed_count,
        failed_seed_count=failed_seed_count,
        goal_connection_attempts=goal_connection_attempts,
        goal_connection_successes=goal_connection_successes,
        goal_connection_failures=goal_connection_failures,
        best_goal_residual_seen=best_goal_residual_seen,
        best_goal_distance_seen=best_goal_distance_seen,
        right_frontier_count=len(frontier_ids.get("right", [])),
        route_found=route_found,
        route_discovery_round=route_discovery_round,
    )


def build_outer_route_from_family_exit(
    graph: FamilyConnectivityGraph,
    start_node_id: int,
    goal_node_id: int,
    exit_seeds: list[ExitSeed],
    right_manifold,
    right_center: np.ndarray,
    right_radius: float,
    goal_q: np.ndarray,
    frontier_ids: dict[str, list[int]],
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    round_idx: int,
    explored_edges_by_mode: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    round_sources: list[np.ndarray],
    round_targets: list[np.ndarray],
    rng: np.random.Generator,
    target_successful_seed_count: int = 1,
) -> RightStageClosureResult:
    return robust_right_stage_closure(
        graph=graph,
        start_node_id=int(start_node_id),
        goal_node_id=int(goal_node_id),
        exit_seeds=exit_seeds,
        right_manifold=right_manifold,
        right_center=np.asarray(right_center, dtype=float),
        right_radius=float(right_radius),
        goal_q=np.asarray(goal_q, dtype=float),
        frontier_ids=frontier_ids,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        round_idx_offset=int(round_idx),
        explored_edges_by_mode=explored_edges_by_mode,
        round_sources=round_sources,
        round_targets=round_targets,
        rng=rng,
        target_successful_seed_count=int(target_successful_seed_count),
    )
