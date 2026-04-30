from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np

from primitive_manifold_planner.families.leaf_state import LeafState
from primitive_manifold_planner.planning.local import run_local_planner
from primitive_manifold_planner.planners.component_leaf_graph import route_node_sequence
from primitive_manifold_planner.planners.transition_manager import (
    TransitionManager,
    identity_wrap,
    euclidean_distance,
)


@dataclass
class ComponentRouteStep:
    family_name: str
    lam: float
    component_id: str
    path: np.ndarray
    transition_point: Optional[np.ndarray] = None
    nominal_transition_point: Optional[np.ndarray] = None
    nominal_candidate_index: Optional[int] = None
    realized_candidate_index: Optional[int] = None
    message: str = ""


@dataclass
class ComponentRouteResult:
    success: bool
    steps: List[ComponentRouteStep]
    final_state: Optional[LeafState]
    message: str = ""


def realize_component_route_with_manager(
    transition_manager: TransitionManager,
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
    transition_penalty_fn: Optional[Callable[[np.ndarray], float]] = None,
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
        nominal_edge = nominal_edges.get((src_node, dst_node), None)

        if i + 2 < len(node_seq):
            next_dst = node_seq[i + 2]
            nominal_next_edge = nominal_edges.get((dst_node, next_dst), None)
            downstream_hint = None if nominal_next_edge is None else wrap_state_fn(nominal_next_edge.transition_point)
        else:
            downstream_hint = wrap_state_fn(goal_q)

        sel = transition_manager.select_reachable_candidate(
            src=src_node,
            dst=dst_node,
            current_x=current.x,
            families=families,
            step_size=step_size,
            local_planner_name=local_planner_name,
            local_planner_kwargs=local_planner_kwargs,
            wrap_state_fn=wrap_state_fn,
            state_distance_fn=state_distance_fn,
            downstream_hint=downstream_hint,
            transition_penalty_fn=transition_penalty_fn,
        )

        if not sel.success or sel.candidate is None or sel.local_result is None:
            return ComponentRouteResult(
                success=False,
                steps=steps,
                final_state=current,
                message=sel.message,
            )

        q_edge = wrap_state_fn(sel.candidate.transition_point)

        steps.append(
            ComponentRouteStep(
                family_name=current.family_name,
                lam=float(current.lam),
                component_id=current_component,
                path=np.asarray(sel.local_result.path),
                transition_point=q_edge.copy(),
                nominal_transition_point=(
                    None
                    if nominal_edge is None
                    else wrap_state_fn(nominal_edge.transition_point)
                ),
                nominal_candidate_index=(
                    None
                    if nominal_edge is None
                    else int(nominal_edge.candidate_index)
                ),
                realized_candidate_index=int(sel.candidate.candidate_index),
                message=sel.message,
            )
        )

        dst_family, dst_lam, dst_comp = sel.candidate.dst
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
        message="Successfully realized component-aware route with transition manager.",
    )
