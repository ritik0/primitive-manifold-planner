from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict
import numpy as np

from primitive_manifold_planner.families.leaf_state import LeafState
from primitive_manifold_planner.planning.local import run_local_planner
from primitive_manifold_planner.planners.admissibility import family_transition_feasibility
from primitive_manifold_planner.planners.mode_semantics import (
    ModeSemantics,
    PlanningSemanticContext,
    PlanningSemanticModel,
)
from primitive_manifold_planner.transitions.leaf_transition import (
    find_leaf_transition,
    LeafTransitionCandidate,
)


@dataclass
class RouteStep:
    kind: str   # "stay" or "switch"
    family_name: str
    lam: object
    path: List[np.ndarray]
    transition: Optional[LeafTransitionCandidate] = None
    score: float = 0.0
    planner_name: str = "projection"
    chart_count: int = 0


@dataclass
class FoliatedPlanResult:
    success: bool
    steps: List[RouteStep] = field(default_factory=list)
    final_state: Optional[LeafState] = None
    message: str = ""


def estimate_path_cost(path: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    return float(sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1)))


def build_default_seeds(
    current_x: np.ndarray,
    goal_x: np.ndarray,
    n_random: int = 8,
    box_halfwidth: float = 0.5,
):
    seeds = [current_x, goal_x, 0.5 * (current_x + goal_x)]
    dim = current_x.shape[0]
    for _ in range(n_random):
        seeds.append(current_x + np.random.uniform(-box_halfwidth, box_halfwidth, size=dim))
    return seeds


def _family_transition_allowed(
    current_family_name: str,
    target_family_name: str,
    allowed_family_transitions: Optional[Dict[str, List[str]]],
) -> bool:
    if allowed_family_transitions is None:
        return True
    allowed_targets = allowed_family_transitions.get(current_family_name, [])
    return target_family_name in allowed_targets


def _leaf_transition_allowed(
    current_family_name: str,
    current_lam,
    target_family_name: str,
    target_lam,
    mode_semantics: Optional[ModeSemantics],
) -> bool:
    if mode_semantics is None:
        return True
    return bool(
        mode_semantics.transition_allowed(
            source_family_name=str(current_family_name),
            source_lam=float(current_lam),
            target_family_name=str(target_family_name),
            target_lam=float(target_lam),
        )
    )


def _semantic_transition_allowed(
    current_family_name: str,
    current_lam,
    target_family_name: str,
    target_lam,
    semantic_model: Optional[PlanningSemanticModel],
) -> bool:
    if semantic_model is None:
        return True
    return bool(
        semantic_model.transition_allowed(
            PlanningSemanticContext(
                source_family_name=str(current_family_name),
                source_lam=float(current_lam),
                target_family_name=str(target_family_name),
                target_lam=float(target_lam),
            )
        )
    )


def _project_goal_to_leaf(
    goal_point: np.ndarray,
    target_leaf,
    project_newton: Callable,
) -> Optional[np.ndarray]:
    proj = project_newton(
        manifold=target_leaf,
        x0=goal_point,
        tol=1e-10,
        max_iters=60,
        damping=1.0,
    )
    if not proj.success:
        return None
    return proj.x_projected.copy()


def _default_target_progress_point(
    fam,
    lam,
    target_leaf,
    goal_point: np.ndarray,
    project_newton: Callable,
) -> Optional[np.ndarray]:
    _ = fam, lam
    return _project_goal_to_leaf(
        goal_point=goal_point,
        target_leaf=target_leaf,
        project_newton=project_newton,
    )


def _goal_leaf_mismatch_penalty(
    fam_name: str,
    lam,
    goal_family_name: str,
    goal_lam,
    same_family_penalty: float = 5.0,
    wrong_family_penalty: float = 0.0,
) -> float:
    """
    Penalize landing on the wrong leaf inside the goal family.

    This helps avoid ending on goal_line[0.9] when the actual target is
    goal_line[1.35] or goal_line[1.8].
    """
    if fam_name != goal_family_name:
        return wrong_family_penalty
    if lam != goal_lam:
        return same_family_penalty
    return 0.0


def plan_foliated_route(
    start_state: LeafState,
    goal_point: np.ndarray,
    goal_family,
    goal_lam,
    families: List,
    constrained_interpolate: Callable = None,   # backward compatibility only
    project_newton: Callable = None,
    max_switches: int = 2,
    local_step_size: float = 0.05,
    progress_tol: float = 1e-3,
    local_planner_name: str = "projection",
    local_planner_kwargs: Optional[dict] = None,
    allowed_family_transitions: Optional[Dict[str, List[str]]] = None,
    mode_semantics: Optional[ModeSemantics] = None,
    semantic_model: Optional[PlanningSemanticModel] = None,
    goal_leaf_mismatch_penalty: float = 5.0,
    target_progress_point_fn: Optional[Callable] = None,
) -> FoliatedPlanResult:
    """
    Family-aware planner with optional explicit family-transition restrictions.

    Behavior:
      - If already on the exact goal leaf, try direct constrained motion to goal_point.
      - Otherwise evaluate allowed switches.
      - For a non-goal leaf, use projection(goal_point onto target leaf) as the local
        progress target after switching.
      - Penalize ending on the wrong leaf inside the goal family.
    """
    current = start_state.copy()
    steps: List[RouteStep] = []
    switches_used = 0
    local_planner_kwargs = dict(local_planner_kwargs or {})
    target_progress_point_fn = target_progress_point_fn or _default_target_progress_point

    while switches_used <= max_switches:
        current_family = next(f for f in families if f.name == current.family_name)
        current_leaf = current_family.manifold(current.lam)

        if current.family_name == goal_family.name and current.lam == goal_lam:
            if float(np.linalg.norm(current.x - goal_point)) <= progress_tol:
                return FoliatedPlanResult(
                    success=True,
                    steps=steps,
                    final_state=LeafState(current.family_name, current.lam, current.x.copy()),
                    message="Reached goal on current leaf.",
                )

        best_kind = None
        best_payload = None
        best_score = float("inf")

        # ------------------------------------------------------------
        # Option 1: direct stay only if already on the exact goal leaf
        # ------------------------------------------------------------
        if current.family_name == goal_family.name and current.lam == goal_lam:
            stay_result = run_local_planner(
                manifold=current_leaf,
                x_start=current.x,
                x_goal=goal_point,
                planner_name=local_planner_name,
                step_size=local_step_size,
                **local_planner_kwargs,
            )

            if stay_result.success and len(stay_result.path) > 0:
                stay_cost = estimate_path_cost(stay_result.path)
                steps.append(
                    RouteStep(
                        kind="stay",
                        family_name=current.family_name,
                        lam=current.lam,
                        path=list(stay_result.path),
                        score=stay_cost,
                        planner_name=stay_result.planner_name,
                        chart_count=stay_result.chart_count,
                    )
                )
                return FoliatedPlanResult(
                    success=True,
                    steps=steps,
                    final_state=LeafState(current.family_name, current.lam, stay_result.path[-1]),
                    message="Reached goal on current leaf.",
                )

        # ------------------------------------------------------------
        # Option 2: try switching
        # ------------------------------------------------------------
        if switches_used < max_switches:
            for fam in families:
                if not _family_transition_allowed(
                    current_family_name=current.family_name,
                    target_family_name=fam.name,
                    allowed_family_transitions=allowed_family_transitions,
                ):
                    continue

                for lam in fam.sample_lambdas(context={"goal": goal_point, "current_lambda": current.lam}):
                    if fam.name == current.family_name and lam == current.lam:
                        continue
                    if not _leaf_transition_allowed(
                        current_family_name=current.family_name,
                        current_lam=current.lam,
                        target_family_name=fam.name,
                        target_lam=lam,
                        mode_semantics=mode_semantics,
                    ):
                        continue
                    if not _semantic_transition_allowed(
                        current_family_name=current.family_name,
                        current_lam=current.lam,
                        target_family_name=fam.name,
                        target_lam=lam,
                        semantic_model=semantic_model,
                    ):
                        continue

                    seeds = build_default_seeds(current.x, goal_point)

                    trans_result = find_leaf_transition(
                        source_family=current_family,
                        source_lam=current.lam,
                        target_family=fam,
                        target_lam=lam,
                        seeds=seeds,
                        project_newton=project_newton,
                        goal=goal_point,
                    )

                    if not trans_result.success:
                        continue

                    target_leaf = fam.manifold(lam)

                    # Exact-goal target if target leaf is the goal leaf, otherwise projected progress target
                    if fam.name == goal_family.name and lam == goal_lam:
                        target_goal = goal_point.copy()
                    else:
                        target_goal = target_progress_point_fn(
                            fam=fam,
                            lam=lam,
                            target_leaf=target_leaf,
                            goal_point=goal_point,
                            project_newton=project_newton,
                        )

                    if target_goal is None:
                        continue

                    for cand in trans_result.candidates[:2]:
                        if not family_transition_feasibility(
                            family=current_family,
                            lam=float(current.lam),
                            point=cand.x,
                            goal_point=goal_point,
                            metadata={"planner": "plan_foliated_route", "role": "source"},
                        ):
                            continue
                        if not family_transition_feasibility(
                            family=fam,
                            lam=float(lam),
                            point=cand.x,
                            goal_point=goal_point,
                            metadata={"planner": "plan_foliated_route", "role": "target"},
                        ):
                            continue

                        # segment 1: current leaf -> transition
                        to_switch = run_local_planner(
                            manifold=current_leaf,
                            x_start=current.x,
                            x_goal=cand.x,
                            planner_name=local_planner_name,
                            step_size=local_step_size,
                            **local_planner_kwargs,
                        )
                        if not to_switch.success or len(to_switch.path) == 0:
                            continue

                        # segment 2: target leaf -> target goal/progress point
                        after_switch = run_local_planner(
                            manifold=target_leaf,
                            x_start=cand.x,
                            x_goal=target_goal,
                            planner_name=local_planner_name,
                            step_size=local_step_size,
                            **local_planner_kwargs,
                        )
                        if not after_switch.success or len(after_switch.path) == 0:
                            continue

                        # must actually move on target leaf
                        target_leaf_progress = float(np.linalg.norm(after_switch.path[-1] - cand.x))
                        if target_leaf_progress < progress_tol:
                            continue

                        remaining_dist = float(np.linalg.norm(after_switch.path[-1] - goal_point))
                        mismatch_penalty = _goal_leaf_mismatch_penalty(
                            fam_name=fam.name,
                            lam=lam,
                            goal_family_name=goal_family.name,
                            goal_lam=goal_lam,
                            same_family_penalty=goal_leaf_mismatch_penalty,
                            wrong_family_penalty=0.0,
                        )

                        score = (
                            estimate_path_cost(to_switch.path)
                            + 0.5  # switch penalty
                            + estimate_path_cost(after_switch.path)
                            + 0.2 * remaining_dist
                            + mismatch_penalty
                        )
                        if mode_semantics is not None:
                            score += float(
                                mode_semantics.transition_cost(
                                    source_family_name=str(current.family_name),
                                    source_lam=float(current.lam),
                                    target_family_name=str(fam.name),
                                    target_lam=float(lam),
                                )
                            )
                        if semantic_model is not None:
                            score += float(
                                semantic_model.transition_cost(
                                    PlanningSemanticContext(
                                        source_family_name=str(current.family_name),
                                        source_lam=float(current.lam),
                                        target_family_name=str(fam.name),
                                        target_lam=float(lam),
                                        point=np.asarray(cand.x, dtype=float),
                                        goal_point=np.asarray(goal_point, dtype=float),
                                        metadata={"planner": "plan_foliated_route"},
                                    )
                                )
                            )
                            if not semantic_model.transition_feasible(
                                PlanningSemanticContext(
                                    source_family_name=str(current.family_name),
                                    source_lam=float(current.lam),
                                    target_family_name=str(fam.name),
                                    target_lam=float(lam),
                                    point=np.asarray(cand.x, dtype=float),
                                    goal_point=np.asarray(goal_point, dtype=float),
                                    metadata={"planner": "plan_foliated_route"},
                                )
                            ):
                                continue
                            score += float(
                                semantic_model.transition_admissibility_cost(
                                    PlanningSemanticContext(
                                        source_family_name=str(current.family_name),
                                        source_lam=float(current.lam),
                                        target_family_name=str(fam.name),
                                        target_lam=float(lam),
                                        point=np.asarray(cand.x, dtype=float),
                                        goal_point=np.asarray(goal_point, dtype=float),
                                        metadata={"planner": "plan_foliated_route"},
                                    )
                                )
                            )

                        if score < best_score:
                            best_score = score
                            best_kind = "switch"
                            best_payload = (fam, lam, cand, to_switch, after_switch)

        # ------------------------------------------------------------
        # No valid option
        # ------------------------------------------------------------
        if best_kind is None:
            return FoliatedPlanResult(
                success=False,
                steps=steps,
                final_state=current,
                message="No feasible stay or switch option found.",
            )

        # ------------------------------------------------------------
        # Execute best switch
        # ------------------------------------------------------------
        fam, lam, cand, to_switch, after_switch = best_payload

        prev_target_start = cand.x.copy()
        new_target_end = after_switch.path[-1].copy()

        steps.append(
            RouteStep(
                kind="switch",
                family_name=current.family_name,
                lam=current.lam,
                path=list(to_switch.path),
                transition=cand,
                score=best_score,
                planner_name=to_switch.planner_name,
                chart_count=to_switch.chart_count,
            )
        )

        steps.append(
            RouteStep(
                kind="stay",
                family_name=fam.name,
                lam=lam,
                path=list(after_switch.path),
                score=best_score,
                planner_name=after_switch.planner_name,
                chart_count=after_switch.chart_count,
            )
        )

        switches_used += 1

        # sanity check before overwriting current
        if np.linalg.norm(new_target_end - prev_target_start) < progress_tol:
            return FoliatedPlanResult(
                success=False,
                steps=steps,
                final_state=current,
                message="Switch option made insufficient progress on target leaf.",
            )

        current = LeafState(fam.name, lam, new_target_end)

    return FoliatedPlanResult(
        success=False,
        steps=steps,
        final_state=current,
        message="Switch budget exhausted.",
    )
