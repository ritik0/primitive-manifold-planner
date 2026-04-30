from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from primitive_manifold_planner.families.leaf_state import LeafState
from primitive_manifold_planner.planning.local import run_local_planner
from primitive_manifold_planner.transitions.attraction_sampler import (
    discover_transition_candidates_via_attraction,
)


@dataclass
class RefinedRouteStep:
    kind: str
    family_name: str
    lam: object
    path: np.ndarray
    transition_point: Optional[np.ndarray] = None

    # New diagnostics for Example 36 / future papers
    nominal_transition_point: Optional[np.ndarray] = None
    transition_was_refined: bool = False
    used_fallback: bool = False
    candidate_count: int = 0

    message: str = ""
    planner_name: str = "projection"
    chart_count: int = 0


@dataclass
class RefinedRouteResult:
    success: bool
    steps: List[RefinedRouteStep] = field(default_factory=list)
    final_state: Optional[LeafState] = None
    message: str = ""


def choose_best_transition_candidates_ranked(
    candidates,
    current_x: np.ndarray,
    downstream_hint: Optional[np.ndarray] = None,
):
    ranked = []

    for cand in candidates:
        score = float(np.linalg.norm(cand.x - current_x))

        if downstream_hint is not None:
            score += 0.5 * float(np.linalg.norm(cand.x - downstream_hint))

        score += 0.2 * float(cand.score)
        ranked.append((score, cand))

    ranked.sort(key=lambda t: t[0])
    return [cand for _, cand in ranked]


def _is_locally_reachable_on_leaf(
    manifold,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    step_size: float,
    min_transition_separation: float,
    local_planner_name: str,
    local_planner_kwargs: dict,
):
    if np.linalg.norm(x_goal - x_start) <= min_transition_separation:
        return False, None

    trial = run_local_planner(
        manifold=manifold,
        x_start=x_start,
        x_goal=x_goal,
        planner_name=local_planner_name,
        step_size=step_size,
        **local_planner_kwargs,
    )

    ok = bool(trial.success and len(trial.path) > 0)
    return ok, trial


def _supports_downstream_progress(
    dst_leaf,
    transition_point: np.ndarray,
    downstream_hint: np.ndarray,
    step_size: float,
    min_transition_separation: float,
    local_planner_name: str,
    local_planner_kwargs: dict,
):
    if np.linalg.norm(downstream_hint - transition_point) <= min_transition_separation:
        return True, None

    trial = run_local_planner(
        manifold=dst_leaf,
        x_start=transition_point,
        x_goal=downstream_hint,
        planner_name=local_planner_name,
        step_size=step_size,
        **local_planner_kwargs,
    )

    ok = bool(trial.success and len(trial.path) > 0)
    return ok, trial


def realize_leaf_route_with_state_attraction(
    start_state: LeafState,
    goal_point: np.ndarray,
    goal_family,
    goal_lam,
    families,
    route_edges,
    constrained_interpolate=None,  # backward compatibility only
    project_newton=None,
    step_size: float = 0.08,
    ambient_local_radius: float = 0.6,
    n_local_samples: int = 35,
    n_goal_bias_samples: int = 25,
    goal_band_width: float = 0.20,
    target_residual_threshold: float = 0.25,
    min_transition_separation: float = 1e-6,
    local_planner_name: str = "projection",
    local_planner_kwargs: Optional[dict] = None,
):
    family_map = {f.name: f for f in families}
    current = start_state.copy()
    steps: List[RefinedRouteStep] = []
    local_planner_kwargs = dict(local_planner_kwargs or {})

    for edge_idx, edge in enumerate(route_edges):
        current_family = family_map[current.family_name]
        current_leaf = current_family.manifold(current.lam)

        dst_family_name, dst_lam = edge.dst
        dst_family = family_map[dst_family_name]
        dst_leaf = dst_family.manifold(dst_lam)

        if edge_idx + 1 < len(route_edges):
            downstream_hint = route_edges[edge_idx + 1].transition_point
        else:
            downstream_hint = goal_point

        nominal_transition = edge.transition_point.copy()
        chosen_transition = None
        chosen_path_result = None
        refinement_msg = ""
        used_fallback = False
        candidate_count = 0

        attr_result = discover_transition_candidates_via_attraction(
            source_family=current_family,
            source_lam=current.lam,
            target_family=dst_family,
            target_lam=dst_lam,
            current_x=current.x,
            goal_x=downstream_hint,
            project_newton=project_newton,
            ambient_local_radius=ambient_local_radius,
            n_local_samples=n_local_samples,
            n_goal_bias_samples=n_goal_bias_samples,
            goal_band_width=goal_band_width,
            target_residual_threshold=target_residual_threshold,
        )

        if attr_result.success and len(attr_result.transition_result.candidates) > 0:
            ranked = choose_best_transition_candidates_ranked(
                attr_result.transition_result.candidates,
                current_x=current.x,
                downstream_hint=downstream_hint,
            )
            candidate_count = len(ranked)

            for cand in ranked:
                transition_point = cand.x.copy()

                ok_src, src_trial = _is_locally_reachable_on_leaf(
                    manifold=current_leaf,
                    x_start=current.x,
                    x_goal=transition_point,
                    step_size=step_size,
                    min_transition_separation=min_transition_separation,
                    local_planner_name=local_planner_name,
                    local_planner_kwargs=local_planner_kwargs,
                )
                if not ok_src:
                    continue

                ok_dst, _ = _supports_downstream_progress(
                    dst_leaf=dst_leaf,
                    transition_point=transition_point,
                    downstream_hint=downstream_hint,
                    step_size=step_size,
                    min_transition_separation=min_transition_separation,
                    local_planner_name=local_planner_name,
                    local_planner_kwargs=local_planner_kwargs,
                )
                if not ok_dst:
                    continue

                chosen_transition = transition_point
                chosen_path_result = src_trial
                refinement_msg = "Used state-dependent attraction refinement."
                break

        if chosen_transition is None:
            ok_fallback_src, fallback_trial = _is_locally_reachable_on_leaf(
                manifold=current_leaf,
                x_start=current.x,
                x_goal=nominal_transition,
                step_size=step_size,
                min_transition_separation=min_transition_separation,
                local_planner_name=local_planner_name,
                local_planner_kwargs=local_planner_kwargs,
            )

            if not ok_fallback_src:
                return RefinedRouteResult(
                    success=False,
                    steps=steps,
                    final_state=current,
                    message=(
                        f"Failed reaching both refined and fallback transition on "
                        f"{current.family_name}[{current.lam}]."
                    ),
                )

            chosen_transition = nominal_transition
            chosen_path_result = fallback_trial
            refinement_msg = "Fell back to graph transition point."
            used_fallback = True

        steps.append(
            RefinedRouteStep(
                kind="stay",
                family_name=current.family_name,
                lam=current.lam,
                path=chosen_path_result.path,
                transition_point=chosen_transition,
                nominal_transition_point=nominal_transition,
                transition_was_refined=bool(
                    np.linalg.norm(chosen_transition - nominal_transition) > 1e-9
                ),
                used_fallback=used_fallback,
                candidate_count=candidate_count,
                message=refinement_msg,
                planner_name=chosen_path_result.planner_name,
                chart_count=chosen_path_result.chart_count,
            )
        )

        current = LeafState(dst_family_name, dst_lam, chosen_transition.copy())

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
        return RefinedRouteResult(
            success=False,
            steps=steps,
            final_state=current,
            message=f"Failed final motion on goal leaf: {final_segment.message}",
        )

    steps.append(
        RefinedRouteStep(
            kind="stay",
            family_name=current.family_name,
            lam=current.lam,
            path=final_segment.path,
            transition_point=None,
            nominal_transition_point=None,
            transition_was_refined=False,
            used_fallback=False,
            candidate_count=0,
            message="Final goal segment.",
            planner_name=final_segment.planner_name,
            chart_count=final_segment.chart_count,
        )
    )

    return RefinedRouteResult(
        success=True,
        steps=steps,
        final_state=LeafState(current.family_name, current.lam, final_segment.path[-1]),
        message="Successfully realized route with state-dependent attraction refinement.",
    )