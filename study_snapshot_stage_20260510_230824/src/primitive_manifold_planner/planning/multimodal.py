from __future__ import annotations

from dataclasses import dataclass
import heapq
import numpy as np

from primitive_manifold_planner.planning.local import constrained_interpolate
from primitive_manifold_planner.planning.mode_graph import ModeGraph, MultimodalRoute, TransitionStep


@dataclass
class SegmentPlan:
    """
    One continuous segment of a multimodal plan.

    This segment lies entirely on one mode/manifold.
    """

    mode_name: str
    start_point: np.ndarray
    end_point: np.ndarray
    path: np.ndarray
    success: bool
    message: str


@dataclass
class MultimodalPlanResult:
    """
    Full multimodal plan result.

    Contains:
    - the chosen route
    - per-mode segment plans
    - a stitched full path
    """

    success: bool
    route: MultimodalRoute | None
    segment_plans: list[SegmentPlan]
    full_path: np.ndarray | None
    message: str


def _stitch_paths(paths: list[np.ndarray]) -> np.ndarray:
    """
    Stitch multiple paths into one path, removing duplicate boundary points
    between consecutive segments.
    """
    if len(paths) == 0:
        return np.zeros((0, 0), dtype=float)

    stitched = [paths[0]]

    for p in paths[1:]:
        if p.shape[0] == 0:
            continue
        stitched.append(p[1:])  # drop the first point to avoid duplication

    return np.vstack(stitched)


def _path_length(path: np.ndarray) -> float:
    """
    Compute Euclidean polyline length of a path of shape (N, d).
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[0] <= 1:
        return 0.0
    diffs = np.diff(path, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _transition_context_cost(
    candidate_point: np.ndarray,
    incoming_anchor: np.ndarray,
    outgoing_anchor: np.ndarray,
) -> float:
    """
    Simple geometric proxy score for choosing a transition candidate.

    Lower is better.
    """
    candidate_point = np.asarray(candidate_point, dtype=float).reshape(-1)
    incoming_anchor = np.asarray(incoming_anchor, dtype=float).reshape(-1)
    outgoing_anchor = np.asarray(outgoing_anchor, dtype=float).reshape(-1)

    return float(
        np.linalg.norm(incoming_anchor - candidate_point)
        + np.linalg.norm(candidate_point - outgoing_anchor)
    )


def _estimate_local_segment_cost(
    graph: ModeGraph,
    mode_name: str,
    x_start: np.ndarray,
    x_end: np.ndarray,
    step_size: float,
    goal_tol: float,
    max_iters: int,
    projection_tol: float,
    projection_max_iters: int,
    projection_damping: float,
) -> float:
    """
    Estimate the cost of moving on one manifold from x_start to x_end
    using the actual local constrained planner.

    If local planning fails, return +inf.
    """
    manifold = graph.nodes[mode_name].manifold

    result = constrained_interpolate(
        manifold=manifold,
        x_start=x_start,
        x_goal=x_end,
        step_size=step_size,
        goal_tol=goal_tol,
        max_iters=max_iters,
        projection_tol=projection_tol,
        projection_max_iters=projection_max_iters,
        projection_damping=projection_damping,
    )

    if not result.success:
        return np.inf

    return _path_length(result.path)


def _choose_transition_steps_for_route(
    graph: ModeGraph,
    mode_sequence: list[str],
    start_point: np.ndarray,
    goal_point: np.ndarray,
) -> list[TransitionStep]:
    """
    Choose one transition candidate for each consecutive mode pair
    using a simple context-aware geometric score.
    """
    if len(mode_sequence) <= 1:
        return []

    chosen_steps: list[TransitionStep] = []
    incoming_anchor = np.asarray(start_point, dtype=float).reshape(-1)

    for i in range(len(mode_sequence) - 1):
        a = mode_sequence[i]
        b = mode_sequence[i + 1]

        edge = graph.get_edge(a, b)
        if edge is None:
            raise RuntimeError(
                f"Inconsistent graph state: no edge found between consecutive "
                f"modes '{a}' and '{b}' in the mode sequence."
            )

        outgoing_anchor = np.asarray(goal_point, dtype=float).reshape(-1)

        best_candidate = min(
            edge.transition_candidates,
            key=lambda cand: (
                _transition_context_cost(cand.point, incoming_anchor, outgoing_anchor),
                cand.score,
            ),
        )

        chosen_steps.append(
            TransitionStep(
                from_mode=a,
                to_mode=b,
                transition_point=best_candidate.point.copy(),
                residual_norm=best_candidate.residual_norm,
            )
        )

        incoming_anchor = best_candidate.point.copy()

    return chosen_steps


def _estimate_edge_cost(
    graph: ModeGraph,
    current_mode: str,
    next_mode: str,
    incoming_anchor: np.ndarray,
    goal_point: np.ndarray,
    *,
    step_size: float,
    goal_tol: float,
    max_iters: int,
    projection_tol: float,
    projection_max_iters: int,
    projection_damping: float,
) -> tuple[float, np.ndarray]:
    """
    Estimate the cost of traversing one graph edge.

    For each transition candidate on the edge:
      1. estimate actual local constrained segment cost from incoming_anchor
         to the candidate on current_mode
      2. add a heuristic geometric pull from the candidate toward final goal
      3. pick the best candidate

    Returns
    -------
    cost, chosen_transition_point
    """
    edge = graph.get_edge(current_mode, next_mode)
    if edge is None:
        raise RuntimeError(f"No edge found between '{current_mode}' and '{next_mode}'.")

    best_total_cost = np.inf
    best_point = None

    for cand in edge.transition_candidates:
        local_cost = _estimate_local_segment_cost(
            graph=graph,
            mode_name=current_mode,
            x_start=incoming_anchor,
            x_end=cand.point,
            step_size=step_size,
            goal_tol=goal_tol,
            max_iters=max_iters,
            projection_tol=projection_tol,
            projection_max_iters=projection_max_iters,
            projection_damping=projection_damping,
        )

        if not np.isfinite(local_cost):
            continue

        heuristic_cost = float(np.linalg.norm(np.asarray(goal_point) - cand.point))
        total_cost = local_cost + heuristic_cost

        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_point = cand.point.copy()

    if best_point is None:
        return np.inf, np.asarray(incoming_anchor, dtype=float).reshape(-1).copy()

    return best_total_cost, best_point


def _find_context_aware_mode_sequence(
    graph: ModeGraph,
    start_mode: str,
    goal_mode: str,
    start_point: np.ndarray,
    goal_point: np.ndarray,
    *,
    step_size: float,
    goal_tol: float,
    max_iters: int,
    projection_tol: float,
    projection_max_iters: int,
    projection_damping: float,
) -> list[str] | None:
    """
    Find a mode sequence using a Dijkstra-like search where edge costs
    are estimated from actual local segment-planning cost plus a simple
    heuristic to the final goal.
    """
    if not graph.has_node(start_mode):
        raise ValueError(f"Unknown start mode '{start_mode}'.")
    if not graph.has_node(goal_mode):
        raise ValueError(f"Unknown goal mode '{goal_mode}'.")

    if start_mode == goal_mode:
        return [start_mode]

    start_point = np.asarray(start_point, dtype=float).reshape(-1)
    goal_point = np.asarray(goal_point, dtype=float).reshape(-1)

    counter = 0
    heap: list[tuple[float, int, str, np.ndarray, list[str]]] = []
    heapq.heappush(heap, (0.0, counter, start_mode, start_point.copy(), [start_mode]))

    best_state_cost: dict[tuple[str, tuple[float, ...]], float] = {}

    while heap:
        cost_so_far, _, current_mode, current_anchor, path = heapq.heappop(heap)

        anchor_key = tuple(np.round(current_anchor, 8))
        state_key = (current_mode, anchor_key)

        if cost_so_far > best_state_cost.get(state_key, np.inf) + 1e-12:
            continue

        if current_mode == goal_mode:
            return path

        for nbr in graph.neighbors(current_mode):
            if nbr in path:
                continue

            edge_cost, next_anchor = _estimate_edge_cost(
                graph=graph,
                current_mode=current_mode,
                next_mode=nbr,
                incoming_anchor=current_anchor,
                goal_point=goal_point,
                step_size=step_size,
                goal_tol=goal_tol,
                max_iters=max_iters,
                projection_tol=projection_tol,
                projection_max_iters=projection_max_iters,
                projection_damping=projection_damping,
            )

            if not np.isfinite(edge_cost):
                continue

            new_cost = cost_so_far + edge_cost
            next_anchor_key = tuple(np.round(next_anchor, 8))
            next_state_key = (nbr, next_anchor_key)

            if new_cost + 1e-12 < best_state_cost.get(next_state_key, np.inf):
                best_state_cost[next_state_key] = new_cost
                counter += 1
                heapq.heappush(
                    heap,
                    (new_cost, counter, nbr, next_anchor.copy(), path + [nbr]),
                )

    return None


def plan_multimodal_route(
    graph: ModeGraph,
    start_mode: str,
    goal_mode: str,
    start_point: np.ndarray,
    goal_point: np.ndarray,
    step_size: float = 0.1,
    goal_tol: float = 1e-3,
    max_iters: int = 500,
    projection_tol: float = 1e-10,
    projection_max_iters: int = 50,
    projection_damping: float = 1.0,
) -> MultimodalPlanResult:
    """
    Plan a full multimodal path by:

    1. finding a context-aware mode route in the graph
    2. selecting context-aware transition points along that route
    3. planning one continuous constrained segment on each mode
    """
    mode_sequence = _find_context_aware_mode_sequence(
        graph=graph,
        start_mode=start_mode,
        goal_mode=goal_mode,
        start_point=start_point,
        goal_point=goal_point,
        step_size=step_size,
        goal_tol=goal_tol,
        max_iters=max_iters,
        projection_tol=projection_tol,
        projection_max_iters=projection_max_iters,
        projection_damping=projection_damping,
    )

    if mode_sequence is None:
        return MultimodalPlanResult(
            success=False,
            route=None,
            segment_plans=[],
            full_path=None,
            message=f"No route found from mode '{start_mode}' to mode '{goal_mode}'.",
        )

    transition_steps = _choose_transition_steps_for_route(
        graph=graph,
        mode_sequence=mode_sequence,
        start_point=start_point,
        goal_point=goal_point,
    )

    route = MultimodalRoute(
        mode_sequence=mode_sequence,
        transition_steps=transition_steps,
    )

    segment_plans: list[SegmentPlan] = []
    segment_paths: list[np.ndarray] = []

    for i, mode_name in enumerate(route.mode_sequence):
        manifold = graph.nodes[mode_name].manifold

        if i == 0:
            seg_start = np.asarray(start_point, dtype=float).reshape(-1)
        else:
            seg_start = route.transition_steps[i - 1].transition_point.copy()

        if i == len(route.mode_sequence) - 1:
            seg_end = np.asarray(goal_point, dtype=float).reshape(-1)
        else:
            seg_end = route.transition_steps[i].transition_point.copy()

        result = constrained_interpolate(
            manifold=manifold,
            x_start=seg_start,
            x_goal=seg_end,
            step_size=step_size,
            goal_tol=goal_tol,
            max_iters=max_iters,
            projection_tol=projection_tol,
            projection_max_iters=projection_max_iters,
            projection_damping=projection_damping,
        )

        segment_plan = SegmentPlan(
            mode_name=mode_name,
            start_point=seg_start,
            end_point=seg_end,
            path=result.path,
            success=result.success,
            message=result.message,
        )
        segment_plans.append(segment_plan)

        if not result.success:
            return MultimodalPlanResult(
                success=False,
                route=route,
                segment_plans=segment_plans,
                full_path=None,
                message=f"Segment planning failed on mode '{mode_name}': {result.message}",
            )

        segment_paths.append(result.path)

    full_path = _stitch_paths(segment_paths)

    return MultimodalPlanResult(
        success=True,
        route=route,
        segment_plans=segment_plans,
        full_path=full_path,
        message="Multimodal plan constructed successfully.",
    )