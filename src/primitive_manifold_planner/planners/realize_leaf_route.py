from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from primitive_manifold_planner.families.leaf_state import LeafState
from primitive_manifold_planner.planning.local import run_local_planner


@dataclass
class RealizedRouteStep:
    kind: str
    family_name: str
    lam: object
    path: np.ndarray
    transition_point: Optional[np.ndarray] = None
    planner_name: str = "projection"
    chart_count: int = 0


@dataclass
class RealizedRouteResult:
    success: bool
    steps: List[RealizedRouteStep] = field(default_factory=list)
    final_state: Optional[LeafState] = None
    message: str = ""


def realize_leaf_route(
    start_state: LeafState,
    goal_point: np.ndarray,
    goal_family,
    goal_lam,
    families,
    route_edges,
    constrained_interpolate=None,   # kept only for backward compatibility
    step_size: float = 0.08,
    local_planner_name: str = "projection",
    local_planner_kwargs: Optional[dict] = None,
):
    family_map = {f.name: f for f in families}
    current = start_state.copy()
    steps = []
    local_planner_kwargs = dict(local_planner_kwargs or {})

    # Follow each edge: move on current leaf to transition, then switch
    for edge in route_edges:
        current_family = family_map[current.family_name]
        current_leaf = current_family.manifold(current.lam)

        to_transition = run_local_planner(
            manifold=current_leaf,
            x_start=current.x,
            x_goal=edge.transition_point,
            planner_name=local_planner_name,
            step_size=step_size,
            **local_planner_kwargs,
        )
        if not to_transition.success:
            return RealizedRouteResult(
                success=False,
                steps=steps,
                final_state=current,
                message=f"Failed reaching transition on leaf {current.family_name}, {current.lam}: {to_transition.message}",
            )

        steps.append(
            RealizedRouteStep(
                kind="stay",
                family_name=current.family_name,
                lam=current.lam,
                path=to_transition.path,
                transition_point=edge.transition_point,
                planner_name=to_transition.planner_name,
                chart_count=to_transition.chart_count,
            )
        )

        current = LeafState(edge.dst[0], edge.dst[1], edge.transition_point.copy())

    # Final motion on goal leaf to actual goal point
    final_family = family_map[current.family_name]
    final_leaf = final_family.manifold(current.lam)

    final_segment = run_local_planner(
        manifold=final_leaf,
        x_start=current.x,
        x_goal=goal_point,
        planner_name=local_planner_name,
        step_size=step_size,
        **local_planner_kwargs,
    )

    if not final_segment.success:
        return RealizedRouteResult(
            success=False,
            steps=steps,
            final_state=current,
            message=f"Failed final motion on goal leaf: {final_segment.message}",
        )

    steps.append(
        RealizedRouteStep(
            kind="stay",
            family_name=current.family_name,
            lam=current.lam,
            path=final_segment.path,
            transition_point=None,
            planner_name=final_segment.planner_name,
            chart_count=final_segment.chart_count,
        )
    )

    return RealizedRouteResult(
        success=True,
        steps=steps,
        final_state=LeafState(current.family_name, current.lam, final_segment.path[-1]),
        message="Successfully realized leaf route.",
    )