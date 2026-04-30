from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np

from primitive_manifold_planner.families.leaf_state import LeafState
from primitive_manifold_planner.planning.local import run_local_planner
from primitive_manifold_planner.planners.component_leaf_graph import (
    ComponentEdge,
    ComponentGraph,
    ComponentKey,
    route_node_sequence,
)


@dataclass
class ComponentRouteStep:
    family_name: str
    lam: float
    component_id: str
    path: np.ndarray
    transition_point: Optional[np.ndarray] = None
    message: str = ""


@dataclass
class ComponentRouteResult:
    success: bool
    steps: List[ComponentRouteStep]
    final_state: Optional[LeafState]
    message: str = ""


def identity_wrap(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=float).copy()


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def select_reachable_edge_between_nodes(
    graph: ComponentGraph,
    src_node: ComponentKey,
    dst_node: ComponentKey,
    current_x: np.ndarray,
    families,
    step_size: float,
    local_planner_name: str,
    local_planner_kwargs: dict,
    wrap_state_fn: Callable[[np.ndarray], np.ndarray] = identity_wrap,
    state_distance_fn: Callable[[np.ndarray, np.ndarray], float] = euclidean_distance,
    downstream_hint: Optional[np.ndarray] = None,
):
    family_map = {f.name: f for f in families}
    current_family = family_map[src_node[0]]
    current_leaf = current_family.manifold(src_node[1])

    dst_family = family_map[dst_node[0]]
    dst_leaf = dst_family.manifold(dst_node[1])

    candidates = [e for e in graph.neighbors(src_node) if e.dst == dst_node]
    if len(candidates) == 0:
        return None, None

    current_x_wrapped = wrap_state_fn(current_x)
    downstream_hint_wrapped = None if downstream_hint is None else wrap_state_fn(downstream_hint)

    def rank_key(edge: ComponentEdge):
        q_edge = wrap_state_fn(edge.transition_point)
        score = state_distance_fn(q_edge, current_x_wrapped)
        if downstream_hint_wrapped is not None:
            score += 0.5 * state_distance_fn(q_edge, downstream_hint_wrapped)
        score += 0.5 * float(edge.score)
        return score

    candidates.sort(key=rank_key)

    for edge in candidates:
        q_edge = wrap_state_fn(edge.transition_point)

        seg = run_local_planner(
            manifold=current_leaf,
            x_start=current_x_wrapped,
            x_goal=q_edge,
            planner_name=local_planner_name,
            step_size=step_size,
            **local_planner_kwargs,
        )
        if not seg.success or len(seg.path) == 0:
            continue

        if downstream_hint_wrapped is not None:
            downstream_trial = run_local_planner(
                manifold=dst_leaf,
                x_start=q_edge,
                x_goal=downstream_hint_wrapped,
                planner_name=local_planner_name,
                step_size=step_size,
                **local_planner_kwargs,
            )
            if not downstream_trial.success or len(downstream_trial.path) == 0:
                continue

        return edge, seg

    return None, None


def realize_component_route(
    graph: ComponentGraph,
    start_state: LeafState,
    start_component: str,
    goal_q: np.ndarray,
    families,
    route_edges,
    step_size: float = 0.08,
    local_planner_name: str = "projection",
    local_planner_kwargs: Optional[dict] = None,
    wrap_state_fn: Callable[[np.ndarray], np.ndarray] = identity_wrap,
    state_distance_fn: Callable[[np.ndarray, np.ndarray], float] = euclidean_distance,
):
    local_planner_kwargs = dict(local_planner_kwargs or {})
    start_key = (start_state.family_name, float(start_state.lam), str(start_component))
    node_seq = route_node_sequence(start_key, route_edges)

    current = LeafState(
        start_state.family_name,
        float(start_state.lam),
        wrap_state_fn(start_state.x),
    )
    current_component = str(start_component)
    steps: List[ComponentRouteStep] = []

    nominal_edges = {(e.src, e.dst): e for e in route_edges}

    for i in range(len(node_seq) - 1):
        src_node = node_seq[i]
        dst_node = node_seq[i + 1]

        if i + 2 < len(node_seq):
            next_dst = node_seq[i + 2]
            nominal_next_edge = nominal_edges.get((dst_node, next_dst), None)
            downstream_hint = None if nominal_next_edge is None else wrap_state_fn(nominal_next_edge.transition_point)
        else:
            downstream_hint = wrap_state_fn(goal_q)

        chosen_edge, seg = select_reachable_edge_between_nodes(
            graph=graph,
            src_node=src_node,
            dst_node=dst_node,
            current_x=current.x,
            families=families,
            step_size=step_size,
            local_planner_name=local_planner_name,
            local_planner_kwargs=local_planner_kwargs,
            wrap_state_fn=wrap_state_fn,
            state_distance_fn=state_distance_fn,
            downstream_hint=downstream_hint,
        )

        if chosen_edge is None or seg is None:
            return ComponentRouteResult(
                success=False,
                steps=steps,
                final_state=current,
                message=(
                    f"Failed finding reachable exact transition from {src_node} to {dst_node} "
                    f"given current_x={np.round(current.x, 4)}."
                ),
            )

        q_edge = wrap_state_fn(chosen_edge.transition_point)

        steps.append(
            ComponentRouteStep(
                family_name=current.family_name,
                lam=float(current.lam),
                component_id=current_component,
                path=np.asarray(seg.path),
                transition_point=q_edge.copy(),
                message=(
                    f"Used reachable candidate_index={chosen_edge.candidate_index} "
                    f"for node pair {src_node} -> {dst_node}."
                ),
            )
        )

        dst_family, dst_lam, dst_comp = chosen_edge.dst
        current = LeafState(dst_family, float(dst_lam), q_edge.copy())
        current_component = str(dst_comp)

    family_map = {f.name: f for f in families}
    final_family = family_map[current.family_name]
    final_leaf = final_family.manifold(current.lam)

    final_seg = run_local_planner(
        manifold=final_leaf,
        x_start=wrap_state_fn(current.x),
        x_goal=wrap_state_fn(goal_q),
        planner_name=local_planner_name,
        step_size=step_size,
        **local_planner_kwargs,
    )
    if not final_seg.success:
        return ComponentRouteResult(
            success=False,
            steps=steps,
            final_state=current,
            message=f"Failed final motion on goal leaf: {final_seg.message}",
        )

    steps.append(
        ComponentRouteStep(
            family_name=current.family_name,
            lam=float(current.lam),
            component_id=current_component,
            path=np.asarray(final_seg.path),
            transition_point=None,
            message="Final goal segment.",
        )
    )

    return ComponentRouteResult(
        success=True,
        steps=steps,
        final_state=LeafState(current.family_name, current.lam, wrap_state_fn(final_seg.path[-1])),
        message="Successfully realized component-aware route.",
    )