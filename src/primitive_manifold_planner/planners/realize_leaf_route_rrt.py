from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from primitive_manifold_planner.families.leaf_state import LeafState
from primitive_manifold_planner.planners.leaf_rrt import plan_on_leaf_rrt


@dataclass
class RRTRealizedRouteStep:
    kind: str
    family_name: str
    lam: object
    path: np.ndarray
    transition_point: Optional[np.ndarray] = None
    message: str = ""


@dataclass
class RRTRealizedRouteResult:
    success: bool
    steps: List[RRTRealizedRouteStep] = field(default_factory=list)
    final_state: Optional[LeafState] = None
    message: str = ""


def realize_leaf_route_rrt(
    start_state: LeafState,
    goal_point: np.ndarray,
    goal_family,
    goal_lam,
    families,
    route_edges,
    project_newton,
    bounds: Tuple[np.ndarray, np.ndarray],
    step_size: float = 0.10,
    max_nodes_per_segment: int = 250,
    goal_bias: float = 0.20,
    local_planner_name: str = "projection",
    local_planner_kwargs: Optional[dict] = None,
):
    family_map = {f.name: f for f in families}
    current = start_state.copy()
    steps = []
    local_planner_kwargs = dict(local_planner_kwargs or {})

    for edge in route_edges:
        current_family = family_map[current.family_name]
        current_leaf = current_family.manifold(current.lam)

        seg = plan_on_leaf_rrt(
            manifold=current_leaf,
            x_start=current.x,
            x_goal=edge.transition_point,
            project_newton=project_newton,
            bounds=bounds,
            step_size=step_size,
            max_nodes=max_nodes_per_segment,
            goal_bias=goal_bias,
            local_planner_name=local_planner_name,
            local_planner_kwargs=local_planner_kwargs,
        )
        if not seg.success:
            return RRTRealizedRouteResult(
                success=False,
                steps=steps,
                final_state=current,
                message=f"Failed reaching transition on leaf {current.family_name}[{current.lam}]: {seg.message}",
            )

        steps.append(
            RRTRealizedRouteStep(
                kind="stay",
                family_name=current.family_name,
                lam=current.lam,
                path=seg.path,
                transition_point=edge.transition_point.copy(),
                message=seg.message,
            )
        )

        current = LeafState(edge.dst[0], edge.dst[1], edge.transition_point.copy())

    final_family = family_map[current.family_name]
    final_leaf = final_family.manifold(current.lam)

    final_seg = plan_on_leaf_rrt(
        manifold=final_leaf,
        x_start=current.x,
        x_goal=goal_point,
        project_newton=project_newton,
        bounds=bounds,
        step_size=step_size,
        max_nodes=max_nodes_per_segment,
        goal_bias=goal_bias,
        local_planner_name=local_planner_name,
        local_planner_kwargs=local_planner_kwargs,
    )

    if not final_seg.success:
        return RRTRealizedRouteResult(
            success=False,
            steps=steps,
            final_state=current,
            message=f"Failed final motion on goal leaf: {final_seg.message}",
        )

    steps.append(
        RRTRealizedRouteStep(
            kind="stay",
            family_name=current.family_name,
            lam=current.lam,
            path=final_seg.path,
            transition_point=None,
            message=final_seg.message,
        )
    )

    return RRTRealizedRouteResult(
        success=True,
        steps=steps,
        final_state=LeafState(current.family_name, current.lam, final_seg.path[-1].copy()),
        message="Successfully realized leaf route with constrained leaf-RRT segments.",
    )