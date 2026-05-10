from __future__ import annotations

"""Joint-space variant of Example 66's parallel evidence planner.

This module keeps the original left -> plane -> right multimodal evidence
architecture, but stores robot joint configurations in the stage graphs.
Each stage manifold is an implicit constraint on the robot's end-effector
through forward kinematics. Route extraction still happens late from the
evidence graph, and the displayed path remains the end-effector trajectory in
world coordinates.
"""

import numpy as np

from . import parallel_evidence_planner as ex66
from primitive_manifold_planner.examplesupport.collision_utilities import configuration_in_collision, default_example_66_obstacles
from primitive_manifold_planner.examplesupport.jointspace_planner_utils import (
    detect_transitions_jointspace,
    end_effector_point,
    explore_joint_manifold,
    generate_joint_proposals,
    inverse_kinematics_start,
    joint_path_to_task_path,
)
from primitive_manifold_planner.manifolds.robot import RobotPlaneManifold, RobotSphereManifold

try:
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover - scipy is optional for lightweight imports.
    least_squares = None

LOCAL_MAX_JOINT_STEP = 0.08


def _task_point(robot, q: np.ndarray) -> np.ndarray:
    return end_effector_point(robot, np.asarray(q, dtype=float))


def _record_local_planner_result(store: ex66.StageEvidenceStore, result) -> None:
    method = str(getattr(result, "planner_method", "unknown"))
    if result is not None and bool(getattr(result, "success", False)):
        if method == "continuation":
            store.local_continuation_success_count = int(getattr(store, "local_continuation_success_count", 0)) + 1
        elif method == "interpolation":
            store.local_interpolation_success_count = int(getattr(store, "local_interpolation_success_count", 0)) + 1
        else:
            store.local_planner_success_count = int(getattr(store, "local_planner_success_count", 0)) + 1
    else:
        store.local_planner_failure_count = int(getattr(store, "local_planner_failure_count", 0)) + 1
    if result is not None and not bool(getattr(result, "joint_continuity_success", True)):
        store.local_joint_jump_rejections = int(getattr(store, "local_joint_jump_rejections", 0)) + 1


def _merge_transition_diagnostics(mode_counts: dict[str, int | float], prefix: str, diagnostics: dict[str, int | float]) -> None:
    for key, value in diagnostics.items():
        full_key = f"{prefix}_{key}"
        if key.endswith("_min"):
            mode_counts[full_key] = min(float(mode_counts.get(full_key, float(value))), float(value))
        elif key.endswith("_max"):
            mode_counts[full_key] = max(float(mode_counts.get(full_key, float(value))), float(value))
        elif key.endswith("_sum"):
            mode_counts[full_key] = float(mode_counts.get(full_key, 0.0)) + float(value)
        else:
            mode_counts[full_key] = int(mode_counts.get(full_key, 0)) + int(value)


def task_space_distance(
    store: ex66.StageEvidenceStore,
    node_id1: int,
    node_id2: int,
    robot,
) -> float:
    p1 = _task_point(robot, store.graph.nodes[int(node_id1)].q)
    p2 = _task_point(robot, store.graph.nodes[int(node_id2)].q)
    return float(np.linalg.norm(p1 - p2))


def _task_points_for_store(store: ex66.StageEvidenceStore, robot) -> np.ndarray:
    points = [_task_point(robot, node.q) for node in store.graph.nodes.values()]
    if len(points) == 0:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(points, dtype=float)


def _task_frontier_points(store: ex66.StageEvidenceStore, robot) -> np.ndarray:
    points = [
        _task_point(robot, store.graph.nodes[node_id].q)
        for node_id in store.frontier_ids
        if node_id in store.graph.nodes
    ]
    if len(points) == 0:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(points, dtype=float)


def _task_edge_segments(store: ex66.StageEvidenceStore, robot) -> list[tuple[np.ndarray, np.ndarray]]:
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for edge_start, edge_end in store.explored_edges:
        segments.append((_task_point(robot, edge_start), _task_point(robot, edge_end)))
    return segments


def _stage_evidence_points_joint(store: ex66.StageEvidenceStore, robot) -> np.ndarray:
    groups: list[np.ndarray] = []
    explored = ex66.explored_points_from_edges(_task_edge_segments(store, robot))
    if len(explored) > 0:
        groups.append(explored)
    if len(store.chart_centers) > 0:
        groups.append(np.asarray(store.chart_centers, dtype=float))
    frontier = _task_frontier_points(store, robot)
    if len(frontier) > 0:
        groups.append(frontier)
    nodes = _task_points_for_store(store, robot)
    if len(nodes) > 0:
        groups.append(nodes)
    if len(groups) == 0:
        return np.zeros((0, 3), dtype=float)
    return ex66.deduplicate_points([point for group in groups for point in np.asarray(group, dtype=float)], tol=1e-4)


def _merge_chart_centers_joint(existing: np.ndarray, robot, joint_path: np.ndarray) -> np.ndarray:
    task_path = joint_path_to_task_path(robot, joint_path)
    chart_count = max(0, min(8, len(task_path) // 6))
    centers = ex66.sample_chart_centers(task_path, chart_count)
    if len(existing) == 0:
        return np.asarray(centers, dtype=float)
    if len(centers) == 0:
        return np.asarray(existing, dtype=float)
    return ex66.deduplicate_points(list(existing) + list(centers), tol=1e-4)


def _proposal_stage_utility_joint(
    stage: str,
    projected_q: np.ndarray,
    store: ex66.StageEvidenceStore,
    guide_point: np.ndarray,
    stores: dict[str, ex66.StageEvidenceStore],
    robot,
) -> float:
    q = np.asarray(projected_q, dtype=float)
    task_q = _task_point(robot, q)
    known_points = _stage_evidence_points_joint(store, robot)
    if len(known_points) > 0:
        novelty = min(float(np.linalg.norm(task_q - point)) for point in known_points)
    else:
        novelty = ex66.TARGET_NOVELTY_RADIUS
    novelty_bonus = min(novelty, ex66.TARGET_NOVELTY_RADIUS)
    guide_bonus = 1.0 / (1.0 + float(np.linalg.norm(task_q - np.asarray(guide_point, dtype=float))))
    underexplored_bonus = ex66.stage_underexploration_factor(stage, stores)
    stage_bias = 0.08 if stage == ex66.RIGHT_STAGE else (0.16 if stage == ex66.PLANE_STAGE else 0.0)
    return 0.50 * novelty_bonus + 0.26 * guide_bonus + 0.18 * underexplored_bonus + stage_bias


def _update_stage_frontier_joint(
    store: ex66.StageEvidenceStore,
    new_ids: list[int],
    guide_point: np.ndarray,
    robot,
    max_points: int = ex66.FRONTIER_LIMIT,
) -> None:
    merged: list[int] = []
    seen: set[int] = set()
    for node_id in list(store.frontier_ids) + list(new_ids):
        if node_id in seen:
            continue
        seen.add(node_id)
        merged.append(node_id)

    if len(merged) <= max_points:
        store.frontier_ids = merged
        return

    guide = np.asarray(guide_point, dtype=float)
    remaining = list(merged)
    selected: list[int] = []
    while len(remaining) > 0 and len(selected) < max_points:
        best_id = None
        best_score = -float("inf")
        for node_id in remaining:
            node = store.graph.nodes[node_id]
            node_task = _task_point(robot, node.q)
            guide_bonus = 1.0 / (1.0 + float(np.linalg.norm(node_task - guide)))
            underexplored_bonus = 1.0 / (1.0 + float(node.expansion_count))
            seeded_bonus = 0.18 if node.seeded_from_proposal else 0.0
            if len(selected) == 0:
                novelty_bonus = guide_bonus
            else:
                novelty_bonus = min(
                    task_space_distance(store, node_id, selected_id, robot)
                    for selected_id in selected
                )
            score = 0.42 * novelty_bonus + 0.24 * underexplored_bonus + 0.14 * guide_bonus + seeded_bonus
            if score > best_score:
                best_score = score
                best_id = node_id
        if best_id is None:
            break
        selected.append(best_id)
        remaining.remove(best_id)
    store.frontier_ids = selected


def _choose_stage_source_joint(store: ex66.StageEvidenceStore, guide_point: np.ndarray, robot) -> int:
    guide = np.asarray(guide_point, dtype=float)
    candidate_ids = list(store.frontier_ids) if len(store.frontier_ids) > 0 else list(store.graph.nodes.keys())
    if len(candidate_ids) == 0:
        raise ValueError("Cannot choose a stage source from an empty evidence store.")
    scored = []
    for node_id in candidate_ids:
        node = store.graph.nodes[node_id]
        node_task = _task_point(robot, node.q)
        guide_bonus = 1.0 / (1.0 + float(np.linalg.norm(node_task - guide)))
        underexplored_bonus = 1.0 / (1.0 + float(node.expansion_count))
        if len(candidate_ids) > 1:
            diversity_bonus = min(
                task_space_distance(store, node_id, other_id, robot)
                for other_id in candidate_ids
                if other_id != node_id
            )
        else:
            diversity_bonus = 0.0
        seeded_bonus = 0.12 if node.seeded_from_proposal else 0.0
        score = -(0.46 * underexplored_bonus + 0.26 * guide_bonus + 0.18 * min(diversity_bonus, 1.0) + seeded_bonus)
        scored.append((score, int(node_id)))
    scored.sort(key=lambda item: item[0])
    top_ids = [node_id for _, node_id in scored[: min(ex66.FRONTIER_SELECTION_LIMIT, len(scored))]]
    return int(np.random.choice(top_ids))


def _ranked_stage_sources_joint(
    store: ex66.StageEvidenceStore,
    guide_point: np.ndarray,
    limit: int,
    robot,
) -> list[int]:
    guide = np.asarray(guide_point, dtype=float)
    candidate_ids = list(store.frontier_ids) if len(store.frontier_ids) > 0 else list(store.graph.nodes.keys())
    if len(candidate_ids) == 0:
        return []
    scored = []
    for node_id in candidate_ids:
        node = store.graph.nodes[node_id]
        node_task = _task_point(robot, node.q)
        guide_bonus = 1.0 / (1.0 + float(np.linalg.norm(node_task - guide)))
        underexplored_bonus = 1.0 / (1.0 + float(node.expansion_count))
        if len(candidate_ids) > 1:
            diversity_bonus = min(
                task_space_distance(store, node_id, other_id, robot)
                for other_id in candidate_ids
                if other_id != node_id
            )
        else:
            diversity_bonus = 0.0
        seeded_bonus = 0.12 if node.seeded_from_proposal else 0.0
        score = -(0.42 * guide_bonus + 0.28 * underexplored_bonus + 0.18 * min(diversity_bonus, 1.0) + seeded_bonus)
        scored.append((score, int(node_id)))
    scored.sort(key=lambda item: item[0])
    return [node_id for _, node_id in scored[: min(limit, len(scored))]]


def _maybe_seed_stage_component_joint(
    store: ex66.StageEvidenceStore,
    target_q: np.ndarray,
    guide_point: np.ndarray,
    robot,
    collision_fn=None,
) -> tuple[int, bool]:
    if collision_fn is not None and bool(collision_fn(target_q)):
        return -1, False
    known = _stage_evidence_points_joint(store, robot)
    if len(store.graph.nodes) == 0:
        node_id = ex66.add_stage_node(store, target_q, seeded_from_proposal=True)
        _update_stage_frontier_joint(store, [node_id], guide_point, robot)
        return node_id, True
    if len(known) == 0:
        node_id = ex66.add_stage_node(store, target_q, seeded_from_proposal=True)
        _update_stage_frontier_joint(store, [node_id], guide_point, robot)
        return node_id, True
    target_task = _task_point(robot, target_q)
    nearest = min(float(np.linalg.norm(target_task - point)) for point in known)
    if nearest >= ex66.EVIDENCE_SEED_RADIUS:
        node_id = ex66.add_stage_node(store, target_q, seeded_from_proposal=True)
        _update_stage_frontier_joint(store, [node_id], guide_point, robot)
        return node_id, True
    return -1, False


def _update_stage_evidence_from_proposal_joint(
    store: ex66.StageEvidenceStore,
    target_q: np.ndarray,
    guide_point: np.ndarray,
    robot,
    collision_fn=None,
) -> tuple[int, int, object | None, int | None, list[int]]:
    seeded_node_id, seeded = _maybe_seed_stage_component_joint(store, target_q, guide_point, robot, collision_fn=collision_fn)
    if seeded:
        store.update_count += 1
        return 1, 0, None, seeded_node_id, [seeded_node_id]

    if len(store.graph.nodes) == 0:
        return 0, 0, None, None, []

    source_node_id = _choose_stage_source_joint(store, guide_point, robot)
    ex66.increment_stage_node_expansion(store, source_node_id)
    source_q = store.graph.nodes[source_node_id].q
    result = explore_joint_manifold(
        manifold=store.manifold,
        start=source_q,
        goal=np.asarray(target_q, dtype=float),
        max_step=LOCAL_MAX_JOINT_STEP,
        local_max_joint_step=LOCAL_MAX_JOINT_STEP,
        collision_fn=collision_fn,
    )
    _record_local_planner_result(store, result)
    store.explored_edges = ex66.merge_edges(store.explored_edges, list(getattr(result, "explored_edges", [])))
    store.chart_centers = _merge_chart_centers_joint(store.chart_centers, robot, np.asarray(result.path, dtype=float))
    if not result.success:
        if not bool(getattr(result, "joint_continuity_success", True)):
            store.stage_edges_rejected_joint_jump += 1
        return 0, 1, result, source_node_id, [source_node_id]
    end_node_id, new_path_nodes, _edge_ids = ex66.connect_path_to_stage_graph(
        store=store,
        source_node_id=source_node_id,
        path=np.asarray(result.path, dtype=float),
        kind=f"{store.stage}_evidence_motion",
        preserve_dense_path=True,
        max_joint_step_for_edge=LOCAL_MAX_JOINT_STEP,
    )
    _update_stage_frontier_joint(store, new_path_nodes + [end_node_id], guide_point, robot)
    store.update_count += 1
    return max(0, len(new_path_nodes) - 1), 1, result, source_node_id, new_path_nodes + [end_node_id]


def _nearest_node_id(store: ex66.StageEvidenceStore, candidate_ids: list[int], target_q: np.ndarray) -> int | None:
    if len(candidate_ids) == 0:
        return None
    target = np.asarray(target_q, dtype=float)
    return min(candidate_ids, key=lambda node_id: float(np.linalg.norm(store.graph.nodes[node_id].q - target)))


def _shared_transition_configuration(
    *,
    source_manifold,
    target_manifold,
    transition_task: np.ndarray,
    source_q_hint: np.ndarray,
    target_q_hint: np.ndarray,
    robot,
    collision_fn=None,
    task_tol: float = 7.5e-2,
) -> np.ndarray | None:
    transition_task = np.asarray(transition_task, dtype=float).reshape(3)
    lower = np.asarray(getattr(source_manifold, "joint_lower", -np.pi * np.ones(3, dtype=float)), dtype=float)
    upper = np.asarray(getattr(source_manifold, "joint_upper", np.pi * np.ones(3, dtype=float)), dtype=float)
    hints = [
        np.asarray(source_q_hint, dtype=float).reshape(3),
        np.asarray(target_q_hint, dtype=float).reshape(3),
        0.5 * (
            np.asarray(source_q_hint, dtype=float).reshape(3)
            + np.asarray(target_q_hint, dtype=float).reshape(3)
        ),
    ]
    best_q: np.ndarray | None = None
    best_task_error = float("inf")
    for hint in hints:
        q_shared = inverse_kinematics_start(
            robot,
            transition_task,
            warm_start=hint,
            joint_lower=lower,
            joint_upper=upper,
            tol=task_tol,
        )
        if q_shared is None:
            continue
        candidates = [np.asarray(q_shared, dtype=float)]
        source_projection = source_manifold.project(np.asarray(q_shared, dtype=float), tol=1e-7, max_iters=80)
        if source_projection.success:
            candidates.append(np.asarray(source_projection.x_projected, dtype=float))
        target_projection = target_manifold.project(np.asarray(q_shared, dtype=float), tol=1e-7, max_iters=80)
        if target_projection.success:
            candidates.append(np.asarray(target_projection.x_projected, dtype=float))
        for candidate in candidates:
            task_error = float(np.linalg.norm(_task_point(robot, candidate) - transition_task))
            if task_error > task_tol:
                continue
            if not bool(source_manifold.within_bounds(candidate)) or not bool(target_manifold.within_bounds(candidate)):
                continue
            if not bool(source_manifold.is_valid(candidate, tol=2e-3)) or not bool(target_manifold.is_valid(candidate, tol=2e-3)):
                continue
            if collision_fn is not None and bool(collision_fn(candidate)):
                continue
            if task_error + 1e-9 < best_task_error:
                best_task_error = task_error
                best_q = np.asarray(candidate, dtype=float)
    return best_q


def _wrapped_joint_delta(q: np.ndarray, center: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    center = np.asarray(center, dtype=float)
    return (q - center + np.pi) % (2.0 * np.pi) - np.pi


def _safe_residual_norm(manifold, q: np.ndarray) -> float:
    try:
        return float(np.linalg.norm(np.ravel(manifold.residual(np.asarray(q, dtype=float)))))
    except Exception:
        return float("inf")


def _handoff_base_diagnostics(
    *,
    source_manifold,
    target_manifold,
    source_hint: np.ndarray,
    target_hint: np.ndarray,
    robot,
) -> dict[str, float]:
    source_hint = np.asarray(source_hint, dtype=float).reshape(3)
    target_hint = np.asarray(target_hint, dtype=float).reshape(3)
    return {
        "source_residual_at_source_hint": _safe_residual_norm(source_manifold, source_hint),
        "target_residual_at_source_hint": _safe_residual_norm(target_manifold, source_hint),
        "source_residual_at_target_hint": _safe_residual_norm(source_manifold, target_hint),
        "target_residual_at_target_hint": _safe_residual_norm(target_manifold, target_hint),
        "task_distance_between_hints": float(np.linalg.norm(_task_point(robot, source_hint) - _task_point(robot, target_hint))),
    }


def solve_shared_transition_q(
    *,
    robot,
    source_manifold,
    target_manifold,
    transition_task: np.ndarray,
    source_hint: np.ndarray,
    target_hint: np.ndarray,
    collision_fn=None,
    residual_tol: float = 2.0e-3,
    task_tol: float = 1.2e-1,
) -> tuple[np.ndarray | None, dict[str, object]]:
    """Solve a stacked joint-space handoff on two active constraints.

    The evidence graph can discover a source-side and target-side transition
    near the same task-space point while still landing on different IK
    branches. Final execution needs one shared q that satisfies both adjacent
    manifolds, so this solver stacks the two residuals and a small FK anchor.
    """
    transition_task = np.asarray(transition_task, dtype=float).reshape(3)
    source_hint = np.asarray(source_hint, dtype=float).reshape(3)
    target_hint = np.asarray(target_hint, dtype=float).reshape(3)
    midpoint = source_hint + 0.5 * _wrapped_joint_delta(target_hint, source_hint)

    source_lower = np.asarray(getattr(source_manifold, "joint_lower", -np.pi * np.ones(3, dtype=float)), dtype=float)
    source_upper = np.asarray(getattr(source_manifold, "joint_upper", np.pi * np.ones(3, dtype=float)), dtype=float)
    target_lower = np.asarray(getattr(target_manifold, "joint_lower", source_lower), dtype=float)
    target_upper = np.asarray(getattr(target_manifold, "joint_upper", source_upper), dtype=float)
    lower = np.maximum(source_lower, target_lower)
    upper = np.minimum(source_upper, target_upper)

    diagnostics: dict[str, object] = _handoff_base_diagnostics(
        source_manifold=source_manifold,
        target_manifold=target_manifold,
        source_hint=source_hint,
        target_hint=target_hint,
        robot=robot,
    )
    diagnostics.update(
        {
            "stacked_solver_available": bool(least_squares is not None),
            "best_stacked_objective_norm": float("inf"),
            "best_stacked_source_residual": float("inf"),
            "best_stacked_target_residual": float("inf"),
            "best_stacked_task_distance": float("inf"),
            "best_stacked_joint_distance_to_midpoint": float("inf"),
            "stacked_attempts": 0,
            "stacked_rejected_collision": 0,
            "stacked_rejected_bounds": 0,
            "stacked_rejected_validity": 0,
            "stacked_rejected_task": 0,
        }
    )

    if least_squares is None:
        return None, diagnostics

    def clipped(q: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(q, dtype=float).reshape(3), lower, upper)

    guesses: list[np.ndarray] = [
        clipped(source_hint),
        clipped(target_hint),
        clipped(midpoint),
    ]
    for alpha in (0.2, 0.4, 0.6, 0.8):
        guesses.append(clipped(source_hint + alpha * _wrapped_joint_delta(target_hint, source_hint)))
    rng = np.random.default_rng(1701)
    for scale in (0.035, 0.075, 0.125):
        for _ in range(4):
            guesses.append(clipped(midpoint + rng.normal(scale=scale, size=3)))

    regularization = 2.0e-3
    task_weight = 0.08

    def objective(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(3)
        source_residual = np.ravel(source_manifold.residual(q))
        target_residual = np.ravel(target_manifold.residual(q))
        task_residual = task_weight * (_task_point(robot, q) - transition_task)
        branch_residual = regularization * _wrapped_joint_delta(q, midpoint)
        return np.concatenate([source_residual, target_residual, task_residual, branch_residual])

    best_q: np.ndarray | None = None
    best_score = float("inf")
    seen: list[np.ndarray] = []
    for guess in guesses:
        if any(float(np.linalg.norm(_wrapped_joint_delta(guess, old))) <= 1.0e-5 for old in seen):
            continue
        seen.append(np.asarray(guess, dtype=float))
        diagnostics["stacked_attempts"] = int(diagnostics["stacked_attempts"]) + 1
        try:
            result = least_squares(
                objective,
                guess,
                bounds=(lower, upper),
                max_nfev=420,
                xtol=1.0e-10,
                ftol=1.0e-10,
                gtol=1.0e-10,
            )
        except Exception:
            continue

        q = clipped(result.x)
        source_residual = _safe_residual_norm(source_manifold, q)
        target_residual = _safe_residual_norm(target_manifold, q)
        task_distance = float(np.linalg.norm(_task_point(robot, q) - transition_task))
        joint_distance = float(np.linalg.norm(_wrapped_joint_delta(q, midpoint)))
        objective_norm = float(np.linalg.norm(objective(q)))
        score = (
            max(source_residual, target_residual)
            + 0.25 * task_distance
            + 1.0e-3 * joint_distance
            + 1.0e-4 * objective_norm
        )
        if score < best_score:
            best_score = score
            best_q = np.asarray(q, dtype=float)
            diagnostics["best_stacked_objective_norm"] = objective_norm
            diagnostics["best_stacked_source_residual"] = source_residual
            diagnostics["best_stacked_target_residual"] = target_residual
            diagnostics["best_stacked_task_distance"] = task_distance
            diagnostics["best_stacked_joint_distance_to_midpoint"] = joint_distance

        if not bool(source_manifold.within_bounds(q, tol=float(residual_tol))) or not bool(target_manifold.within_bounds(q, tol=float(residual_tol))):
            diagnostics["stacked_rejected_bounds"] = int(diagnostics["stacked_rejected_bounds"]) + 1
            continue
        if collision_fn is not None and bool(collision_fn(q)):
            diagnostics["stacked_rejected_collision"] = int(diagnostics["stacked_rejected_collision"]) + 1
            continue
        if not bool(source_manifold.is_valid(q, tol=float(residual_tol))) or not bool(target_manifold.is_valid(q, tol=float(residual_tol))):
            diagnostics["stacked_rejected_validity"] = int(diagnostics["stacked_rejected_validity"]) + 1
            continue
        if task_distance > float(task_tol):
            diagnostics["stacked_rejected_task"] = int(diagnostics["stacked_rejected_task"]) + 1
            continue
        return np.asarray(q, dtype=float), diagnostics

    diagnostics["best_stacked_q"] = best_q
    return None, diagnostics


def _format_handoff_diagnostics(
    diagnostics: dict[str, object],
    *,
    source_name: str,
    target_name: str,
) -> str:
    def fmt(key: str) -> str:
        value = diagnostics.get(key, float("nan"))
        if isinstance(value, (float, int, np.floating, np.integer)):
            return f"{float(value):.4g}"
        return str(value)

    return (
        f"{source_name}_residual_at_{source_name}_hint={fmt('source_residual_at_source_hint')}, "
        f"{target_name}_residual_at_{source_name}_hint={fmt('target_residual_at_source_hint')}, "
        f"{source_name}_residual_at_{target_name}_hint={fmt('source_residual_at_target_hint')}, "
        f"{target_name}_residual_at_{target_name}_hint={fmt('target_residual_at_target_hint')}, "
        f"task_distance_between_hints={fmt('task_distance_between_hints')}, "
        f"best_stacked_source_residual={fmt('best_stacked_source_residual')}, "
        f"best_stacked_target_residual={fmt('best_stacked_target_residual')}, "
        f"best_stacked_task_distance={fmt('best_stacked_task_distance')}, "
        f"best_stacked_objective_norm={fmt('best_stacked_objective_norm')}, "
        f"stacked_attempts={fmt('stacked_attempts')}"
    )


def _plane_right_target_task(entry_task: np.ndarray, plane_geom, right_geom) -> np.ndarray:
    entry = np.asarray(entry_task, dtype=float)
    plane_point = np.asarray(plane_geom.point, dtype=float)
    plane_normal = np.asarray(plane_geom.normal, dtype=float)
    right_center = np.asarray(right_geom.center, dtype=float)
    signed = float(np.dot(right_center - plane_point, plane_normal))
    center_on_plane = right_center - signed * plane_normal
    radius_sq = max(float(right_geom.radius) ** 2 - signed * signed, 0.0)
    radius_on_plane = float(np.sqrt(radius_sq))
    if radius_on_plane <= 1e-6:
        return center_on_plane
    direction = entry - center_on_plane
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        tangent = np.cross(plane_normal, np.asarray([1.0, 0.0, 0.0], dtype=float))
        if float(np.linalg.norm(tangent)) <= 1e-9:
            tangent = np.cross(plane_normal, np.asarray([0.0, 1.0, 0.0], dtype=float))
        direction = tangent
        norm = float(np.linalg.norm(direction))
    return center_on_plane + radius_on_plane * direction / max(norm, 1e-9)


def _plane_targeted_joint_proposals(
    *,
    plane_store: ex66.StageEvidenceStore,
    left_plane_hypotheses: list[ex66.TransitionHypothesis],
    plane_geom,
    right_geom,
    robot,
    collision_fn=None,
    limit: int = 6,
) -> list[np.ndarray]:
    if len(left_plane_hypotheses) == 0:
        return []
    seed_ids: list[int] = []
    for hyp in left_plane_hypotheses:
        if hyp.plane_node_id is not None and int(hyp.plane_node_id) in plane_store.graph.nodes:
            seed_ids.append(int(hyp.plane_node_id))
    seed_ids.extend([int(node_id) for node_id in plane_store.frontier_ids if int(node_id) in plane_store.graph.nodes])
    deduped_seed_ids: list[int] = []
    for node_id in seed_ids:
        if node_id not in deduped_seed_ids:
            deduped_seed_ids.append(node_id)
    proposals: list[np.ndarray] = []
    lower = np.asarray(plane_store.manifold.joint_lower, dtype=float)
    upper = np.asarray(plane_store.manifold.joint_upper, dtype=float)
    for node_id in deduped_seed_ids[: max(1, int(limit))]:
        q_seed = np.asarray(plane_store.graph.nodes[node_id].q, dtype=float)
        entry_task = _task_point(robot, q_seed)
        exit_task = _plane_right_target_task(entry_task, plane_geom, right_geom)
        for alpha in (0.25, 0.45, 0.65, 0.85, 1.0):
            if len(proposals) >= int(limit):
                return proposals
            target_task = (1.0 - float(alpha)) * entry_task + float(alpha) * exit_task
            q_guess = inverse_kinematics_start(
                robot,
                target_task,
                warm_start=q_seed,
                joint_lower=lower,
                joint_upper=upper,
                tol=1.2e-1,
            )
            if q_guess is None:
                continue
            projection = plane_store.manifold.project(np.asarray(q_guess, dtype=float), tol=1e-6, max_iters=80)
            if not projection.success:
                continue
            q_proj = np.asarray(projection.x_projected, dtype=float)
            if collision_fn is not None and bool(collision_fn(q_proj)):
                continue
            if not bool(plane_store.manifold.within_bounds(q_proj)):
                continue
            proposals.append(q_proj)
    return proposals


def sample_plane_right_intersection_task_points(plane_geom, right_geom, n: int = 64) -> np.ndarray:
    plane_point = np.asarray(plane_geom.point, dtype=float)
    plane_normal = np.asarray(plane_geom.normal, dtype=float)
    right_center = np.asarray(right_geom.center, dtype=float)
    signed = float(np.dot(right_center - plane_point, plane_normal))
    radius_sq = float(right_geom.radius) ** 2 - signed * signed
    if radius_sq < -1e-9:
        return np.zeros((0, 3), dtype=float)
    radius = float(np.sqrt(max(radius_sq, 0.0)))
    center = right_center - signed * plane_normal
    basis_u = np.cross(plane_normal, np.asarray([0.0, 0.0, 1.0], dtype=float))
    if float(np.linalg.norm(basis_u)) <= 1e-9:
        basis_u = np.cross(plane_normal, np.asarray([0.0, 1.0, 0.0], dtype=float))
    basis_u = basis_u / max(float(np.linalg.norm(basis_u)), 1e-9)
    basis_v = np.cross(plane_normal, basis_u)
    basis_v = basis_v / max(float(np.linalg.norm(basis_v)), 1e-9)
    points: list[np.ndarray] = []
    for theta in np.linspace(0.0, 2.0 * np.pi, num=max(4, int(n)), endpoint=False):
        p = center + radius * (np.cos(theta) * basis_u + np.sin(theta) * basis_v)
        if bool(plane_geom.within_bounds(np.asarray(p, dtype=float))):
            points.append(np.asarray(p, dtype=float))
    return np.asarray(points, dtype=float) if len(points) > 0 else np.zeros((0, 3), dtype=float)


def _explicit_plane_right_transition_search(
    *,
    plane_store: ex66.StageEvidenceStore,
    right_store: ex66.StageEvidenceStore,
    plane_geom,
    right_geom,
    left_plane_hypotheses: list[ex66.TransitionHypothesis],
    plane_right_hypotheses: list[ex66.TransitionHypothesis],
    guide_point: np.ndarray,
    robot,
    collision_fn=None,
    max_checks: int = 24,
) -> tuple[int, int]:
    if len(left_plane_hypotheses) == 0:
        return 0, 0
    intersection_points = sample_plane_right_intersection_task_points(plane_geom, right_geom, n=96)
    if len(intersection_points) == 0:
        return 0, 0
    known = [np.asarray(hyp.q, dtype=float) for hyp in plane_right_hypotheses]
    plane_node_ids = [
        int(hyp.plane_node_id)
        for hyp in left_plane_hypotheses
        if hyp.plane_node_id is not None and int(hyp.plane_node_id) in plane_store.graph.nodes
    ]
    plane_node_ids.extend([int(node_id) for node_id in plane_store.frontier_ids if int(node_id) in plane_store.graph.nodes])
    plane_node_ids.extend(list(plane_store.graph.nodes.keys()))
    deduped_plane_ids: list[int] = []
    for node_id in plane_node_ids:
        if int(node_id) in plane_store.graph.nodes and int(node_id) not in deduped_plane_ids:
            deduped_plane_ids.append(int(node_id))
    if len(deduped_plane_ids) == 0:
        return 0, 0

    scored: list[tuple[float, np.ndarray, int]] = []
    for point in intersection_points:
        if any(float(np.linalg.norm(point - prev)) <= ex66.TRANSITION_DEDUP_TOL for prev in known):
            continue
        nearest_plane_id = min(
            deduped_plane_ids[: min(len(deduped_plane_ids), 80)],
            key=lambda node_id: float(np.linalg.norm(_task_point(robot, plane_store.graph.nodes[node_id].q) - point)),
        )
        score = float(np.linalg.norm(_task_point(robot, plane_store.graph.nodes[nearest_plane_id].q) - point))
        scored.append((score, point, int(nearest_plane_id)))
    scored.sort(key=lambda item: item[0])

    added = 0
    eval_count = 0
    for _score, transition_task, plane_hint_id in scored[: max(1, int(max_checks))]:
        eval_count += 1
        q_plane_hint = np.asarray(plane_store.graph.nodes[int(plane_hint_id)].q, dtype=float)
        right_hint_candidates = list(right_store.frontier_ids) + list(right_store.graph.nodes.keys())
        if len(right_hint_candidates) > 0:
            right_hint_id = min(
                [int(node_id) for node_id in right_hint_candidates if int(node_id) in right_store.graph.nodes],
                key=lambda node_id: float(np.linalg.norm(_task_point(robot, right_store.graph.nodes[node_id].q) - transition_task)),
            )
            q_right_hint = np.asarray(right_store.graph.nodes[int(right_hint_id)].q, dtype=float)
        else:
            right_hint_id = -1
            q_right_hint = q_plane_hint
        q_shared = _shared_transition_configuration(
            source_manifold=plane_store.manifold,
            target_manifold=right_store.manifold,
            transition_task=transition_task,
            source_q_hint=q_plane_hint,
            target_q_hint=q_right_hint,
            robot=robot,
            collision_fn=collision_fn,
            task_tol=1.2e-1,
        )
        if q_shared is None:
            continue
        if any(float(np.linalg.norm(_task_point(robot, q_shared) - hyp.q)) <= ex66.TRANSITION_DEDUP_TOL for hyp in plane_right_hypotheses):
            continue
        plane_node_id, _plane_path_nodes = _connect_source_to_transition_joint(
            store=plane_store,
            source_node_id=int(plane_hint_id),
            path_node_ids=list(plane_store.frontier_ids),
            target_q=q_shared,
            guide_point=guide_point,
            robot=robot,
            edge_kind=ex66.PLANE_MOTION,
            collision_fn=collision_fn,
        )
        if plane_node_id is None:
            continue
        if right_hint_id >= 0:
            right_node_id, _right_path_nodes = _connect_source_to_transition_joint(
                store=right_store,
                source_node_id=int(right_hint_id),
                path_node_ids=list(right_store.frontier_ids),
                target_q=q_shared,
                guide_point=guide_point,
                robot=robot,
                edge_kind=ex66.RIGHT_MOTION,
                collision_fn=collision_fn,
            )
        else:
            right_node_id = None
        if right_node_id is None:
            continue
        plane_right_hypotheses.append(
            ex66.TransitionHypothesis(
                plane_node_id=int(plane_node_id),
                right_node_id=int(right_node_id),
                q=np.asarray(_task_point(robot, q_shared), dtype=float),
                provenance="explicit_plane_right_intersection",
                score=float(np.linalg.norm(np.asarray(_task_point(robot, q_shared), dtype=float) - np.asarray(guide_point, dtype=float))),
            )
        )
        added += 1
        ex66.prune_transition_hypotheses(plane_right_hypotheses)
        if added >= 4:
            break
    return added, eval_count


def _connect_source_to_transition_joint(
    *,
    store: ex66.StageEvidenceStore,
    source_node_id: int,
    path_node_ids: list[int],
    target_q: np.ndarray,
    guide_point: np.ndarray,
    robot,
    edge_kind: str,
    collision_fn=None,
) -> tuple[int | None, list[int]]:
    candidate_source_ids = [int(source_node_id)] + [int(node_id) for node_id in path_node_ids]
    graph_source_id = _nearest_node_id(store, candidate_source_ids, target_q)
    if graph_source_id is None:
        return None, []
    source_q = np.asarray(store.graph.nodes[int(graph_source_id)].q, dtype=float)
    target_q_arr = np.asarray(target_q, dtype=float)
    if float(np.linalg.norm(source_q - target_q_arr)) <= ex66.GRAPH_NODE_TOL:
        _update_stage_frontier_joint(store, [int(graph_source_id)], guide_point, robot)
        return int(graph_source_id), []

    exact_result = explore_joint_manifold(
        manifold=store.manifold,
        start=source_q,
        goal=target_q_arr,
        max_step=LOCAL_MAX_JOINT_STEP,
        local_max_joint_step=LOCAL_MAX_JOINT_STEP,
        collision_fn=collision_fn,
    )
    _record_local_planner_result(store, exact_result)
    if not exact_result.success:
        if not bool(getattr(exact_result, "joint_continuity_success", True)):
            store.stage_edges_rejected_joint_jump += 1
        return None, []
    store.explored_edges = ex66.merge_edges(store.explored_edges, list(getattr(exact_result, "explored_edges", [])))
    store.chart_centers = _merge_chart_centers_joint(
        store.chart_centers,
        robot,
        np.asarray(exact_result.path, dtype=float),
    )
    target_node_id, path_nodes, _ = ex66.connect_path_to_stage_graph(
        store=store,
        source_node_id=int(graph_source_id),
        path=np.asarray(exact_result.path, dtype=float),
        kind=edge_kind,
        preserve_dense_path=True,
        max_joint_step_for_edge=LOCAL_MAX_JOINT_STEP,
    )
    _update_stage_frontier_joint(store, list(path_nodes) + [int(target_node_id)], guide_point, robot)
    return int(target_node_id), list(path_nodes)


def _add_left_plane_hypotheses_joint(
    source_stage: str,
    source_store: ex66.StageEvidenceStore,
    plane_store: ex66.StageEvidenceStore,
    left_store: ex66.StageEvidenceStore,
    left_geom,
    plane_geom,
    result,
    source_node_id: int | None,
    path_node_ids: list[int],
    guide_point: np.ndarray,
    hypotheses: list[ex66.TransitionHypothesis],
    robot,
    collision_fn=None,
    diagnostics: dict[str, int | float] | None = None,
) -> tuple[int, int]:
    if result is None or source_node_id is None:
        return 0, 0
    eval_count = max(1, int(len(np.asarray(result.path, dtype=float))) - 1)

    hits = detect_transitions_jointspace(
        robot=robot,
        current_manifold=source_store.manifold,
        target_manifold=plane_store.manifold if source_stage == ex66.LEFT_STAGE else left_store.manifold,
        path_configs=np.asarray(result.path, dtype=float),
        collision_fn=collision_fn,
        diagnostics=diagnostics,
    )
    if source_stage == ex66.PLANE_STAGE:
        hits = detect_transitions_jointspace(
            robot=robot,
            current_manifold=source_store.manifold,
            target_manifold=left_store.manifold,
            path_configs=np.asarray(result.path, dtype=float),
            collision_fn=collision_fn,
            diagnostics=diagnostics,
        )
    if len(hits) == 0:
        return 0, eval_count

    added = 0
    known_points = [hyp.q for hyp in hypotheses]
    ranked_hits = ex66.rank_transition_hits(
        np.asarray([hit[2] for hit in hits], dtype=float),
        guide_point,
        known_points,
    )
    if len(ranked_hits) == 0:
        return 0, 0

    for hit_task in ranked_hits[: min(12, len(ranked_hits))]:
        matching = None
        for source_q_hit, target_q_hit, ee_hit in hits:
            if float(np.linalg.norm(np.asarray(ee_hit, dtype=float) - np.asarray(hit_task, dtype=float))) <= ex66.TRANSITION_DEDUP_TOL:
                matching = (source_q_hit, target_q_hit, ee_hit)
                break
        if matching is None:
            continue
        source_q_hit, target_q_hit, ee_hit = matching
        if any(float(np.linalg.norm(np.asarray(ee_hit, dtype=float) - hyp.q)) <= ex66.TRANSITION_DEDUP_TOL for hyp in hypotheses):
            continue
        refined_task, refined_ok = ex66.refine_intersection_on_both_manifolds(
            left_geom,
            plane_geom,
            0.5 * (np.asarray(ee_hit, dtype=float) + _task_point(robot, target_q_hit)),
            tol=1e-8,
            max_iters=25,
        )
        transition_task = np.asarray(refined_task if refined_ok else ee_hit, dtype=float)
        if not bool(plane_geom.within_bounds(transition_task)):
            continue
        target_manifold = plane_store.manifold if source_stage == ex66.LEFT_STAGE else left_store.manifold
        q_shared, _shared_diagnostics = solve_shared_transition_q(
            robot=robot,
            source_manifold=source_store.manifold,
            target_manifold=target_manifold,
            transition_task=transition_task,
            source_hint=np.asarray(source_q_hit, dtype=float),
            target_hint=np.asarray(target_q_hit, dtype=float),
            collision_fn=collision_fn,
            residual_tol=2.0e-3,
            task_tol=1.2e-1,
        )
        if q_shared is None:
            continue
        if source_stage == ex66.LEFT_STAGE:
            left_node_id, _left_path_nodes = _connect_source_to_transition_joint(
                store=left_store,
                source_node_id=int(source_node_id),
                path_node_ids=path_node_ids,
                target_q=np.asarray(q_shared, dtype=float),
                guide_point=guide_point,
                robot=robot,
                edge_kind=ex66.LEFT_MOTION,
                collision_fn=collision_fn,
            )
            if left_node_id is None:
                continue
            plane_node_id = ex66.add_stage_node(plane_store, np.asarray(q_shared, dtype=float), seeded_from_proposal=True)
            _update_stage_frontier_joint(plane_store, [plane_node_id], guide_point, robot)
        else:
            plane_node_id, _plane_path_nodes = _connect_source_to_transition_joint(
                store=plane_store,
                source_node_id=int(source_node_id),
                path_node_ids=path_node_ids,
                target_q=np.asarray(q_shared, dtype=float),
                guide_point=guide_point,
                robot=robot,
                edge_kind=ex66.PLANE_MOTION,
                collision_fn=collision_fn,
            )
            if plane_node_id is None:
                continue
            left_node_id = ex66.add_stage_node(left_store, np.asarray(q_shared, dtype=float), seeded_from_proposal=True)
            _update_stage_frontier_joint(left_store, [left_node_id], guide_point, robot)
        hypotheses.append(
            ex66.TransitionHypothesis(
                left_node_id=int(left_node_id),
                plane_node_id=int(plane_node_id),
                q=np.asarray(transition_task, dtype=float),
                transition_theta=np.asarray(q_shared, dtype=float),
                provenance=f"{source_stage}_jointspace",
                score=float(np.linalg.norm(np.asarray(transition_task, dtype=float) - np.asarray(guide_point, dtype=float))),
            )
        )
        added += 1
        ex66.prune_transition_hypotheses(hypotheses)
    return added, eval_count


def _add_plane_right_hypotheses_joint(
    source_stage: str,
    source_store: ex66.StageEvidenceStore,
    plane_store: ex66.StageEvidenceStore,
    right_store: ex66.StageEvidenceStore,
    plane_geom,
    right_geom,
    result,
    source_node_id: int | None,
    path_node_ids: list[int],
    guide_point: np.ndarray,
    hypotheses: list[ex66.TransitionHypothesis],
    robot,
    collision_fn=None,
    diagnostics: dict[str, int | float] | None = None,
) -> tuple[int, int]:
    if result is None or source_node_id is None:
        return 0, 0
    eval_count = max(1, int(len(np.asarray(result.path, dtype=float))) - 1)

    hits = detect_transitions_jointspace(
        robot=robot,
        current_manifold=source_store.manifold,
        target_manifold=right_store.manifold if source_stage == ex66.PLANE_STAGE else plane_store.manifold,
        path_configs=np.asarray(result.path, dtype=float),
        collision_fn=collision_fn,
        diagnostics=diagnostics,
    )
    if source_stage == ex66.RIGHT_STAGE:
        hits = detect_transitions_jointspace(
            robot=robot,
            current_manifold=source_store.manifold,
            target_manifold=plane_store.manifold,
            path_configs=np.asarray(result.path, dtype=float),
            collision_fn=collision_fn,
            diagnostics=diagnostics,
        )
    if len(hits) == 0:
        return 0, eval_count

    added = 0
    known_points = [hyp.q for hyp in hypotheses]
    ranked_hits = ex66.rank_transition_hits(
        np.asarray([hit[2] for hit in hits], dtype=float),
        guide_point,
        known_points,
    )
    if len(ranked_hits) == 0:
        return 0, 0

    for hit_task in ranked_hits[: min(12, len(ranked_hits))]:
        matching = None
        for source_q_hit, target_q_hit, ee_hit in hits:
            if float(np.linalg.norm(np.asarray(ee_hit, dtype=float) - np.asarray(hit_task, dtype=float))) <= ex66.TRANSITION_DEDUP_TOL:
                matching = (source_q_hit, target_q_hit, ee_hit)
                break
        if matching is None:
            continue
        source_q_hit, target_q_hit, ee_hit = matching
        if any(float(np.linalg.norm(np.asarray(ee_hit, dtype=float) - hyp.q)) <= ex66.TRANSITION_DEDUP_TOL for hyp in hypotheses):
            continue
        refined_task, refined_ok = ex66.refine_intersection_on_both_manifolds(
            plane_geom,
            right_geom,
            0.5 * (np.asarray(ee_hit, dtype=float) + _task_point(robot, target_q_hit)),
            tol=1e-8,
            max_iters=25,
        )
        transition_task = np.asarray(refined_task if refined_ok else ee_hit, dtype=float)
        if source_stage == ex66.PLANE_STAGE and not bool(plane_geom.within_bounds(transition_task)):
            continue
        target_manifold = right_store.manifold if source_stage == ex66.PLANE_STAGE else plane_store.manifold
        q_shared, _shared_diagnostics = solve_shared_transition_q(
            robot=robot,
            source_manifold=source_store.manifold,
            target_manifold=target_manifold,
            transition_task=transition_task,
            source_hint=np.asarray(source_q_hit, dtype=float),
            target_hint=np.asarray(target_q_hit, dtype=float),
            collision_fn=collision_fn,
            residual_tol=2.0e-3,
            task_tol=1.2e-1,
        )
        if q_shared is None:
            continue
        if source_stage == ex66.PLANE_STAGE:
            plane_node_id, _plane_path_nodes = _connect_source_to_transition_joint(
                store=plane_store,
                source_node_id=int(source_node_id),
                path_node_ids=path_node_ids,
                target_q=np.asarray(q_shared, dtype=float),
                guide_point=guide_point,
                robot=robot,
                edge_kind=ex66.PLANE_MOTION,
                collision_fn=collision_fn,
            )
            if plane_node_id is None:
                continue
            right_node_id = ex66.add_stage_node(right_store, np.asarray(q_shared, dtype=float), seeded_from_proposal=True)
            _update_stage_frontier_joint(right_store, [right_node_id], guide_point, robot)
        else:
            right_node_id, _right_path_nodes = _connect_source_to_transition_joint(
                store=right_store,
                source_node_id=int(source_node_id),
                path_node_ids=path_node_ids,
                target_q=np.asarray(q_shared, dtype=float),
                guide_point=guide_point,
                robot=robot,
                edge_kind=ex66.RIGHT_MOTION,
                collision_fn=collision_fn,
            )
            if right_node_id is None:
                continue
            plane_node_id = ex66.add_stage_node(plane_store, np.asarray(q_shared, dtype=float), seeded_from_proposal=True)
            _update_stage_frontier_joint(plane_store, [plane_node_id], guide_point, robot)
        hypotheses.append(
            ex66.TransitionHypothesis(
                plane_node_id=int(plane_node_id),
                right_node_id=int(right_node_id),
                q=np.asarray(transition_task, dtype=float),
                transition_theta=np.asarray(q_shared, dtype=float),
                provenance=f"{source_stage}_jointspace",
                score=float(np.linalg.norm(np.asarray(transition_task, dtype=float) - np.asarray(guide_point, dtype=float))),
            )
        )
        added += 1
        ex66.prune_transition_hypotheses(hypotheses)
    return added, eval_count


def _bridge_left_hypotheses_to_start_joint(
    left_store: ex66.StageEvidenceStore,
    start_node_id: int,
    left_plane_hypotheses: list[ex66.TransitionHypothesis],
    guide_point: np.ndarray,
    robot,
    collision_fn=None,
) -> tuple[int, int]:
    left_dist, _prev_node, _prev_edge = ex66.shortest_paths_in_stage(left_store, int(start_node_id))
    reachable_ids = sorted(int(node_id) for node_id in left_dist)
    target_ids = sorted(
        {
            int(hyp.left_node_id)
            for hyp in left_plane_hypotheses
            if hyp.left_node_id is not None and int(hyp.left_node_id) not in left_dist
        }
    )
    if len(reachable_ids) == 0 or len(target_ids) == 0:
        return 0, 0

    candidate_pairs: list[tuple[float, int, int]] = []
    for target_id in target_ids:
        target_task = _task_point(robot, left_store.graph.nodes[target_id].q)
        nearest_sources = sorted(
            reachable_ids,
            key=lambda source_id: float(np.linalg.norm(_task_point(robot, left_store.graph.nodes[source_id].q) - target_task)),
        )
        for source_id in nearest_sources[: min(6, len(nearest_sources))]:
            candidate_pairs.append(
                (
                    float(
                        np.linalg.norm(
                            _task_point(robot, left_store.graph.nodes[source_id].q) - target_task
                        )
                    ),
                    source_id,
                    target_id,
                )
            )
    candidate_pairs.sort(key=lambda item: item[0])

    eval_count = 0
    node_gain = 0
    tried_targets: set[int] = set()
    bridge_try_limit = max(ex66.LEFT_BRIDGE_TRY_LIMIT, 12)
    for _score, source_id, target_id in candidate_pairs:
        if len(tried_targets) >= bridge_try_limit:
            break
        if int(target_id) in tried_targets:
            continue
        tried_targets.add(int(target_id))
        exact_result = explore_joint_manifold(
            manifold=left_store.manifold,
            start=left_store.graph.nodes[source_id].q,
            goal=left_store.graph.nodes[target_id].q,
            max_step=LOCAL_MAX_JOINT_STEP,
            local_max_joint_step=LOCAL_MAX_JOINT_STEP,
            collision_fn=collision_fn,
        )
        _record_local_planner_result(left_store, exact_result)
        eval_count += 1
        if not exact_result.success:
            if not bool(getattr(exact_result, "joint_continuity_success", True)):
                left_store.stage_edges_rejected_joint_jump += 1
            continue
        left_store.explored_edges = ex66.merge_edges(left_store.explored_edges, list(getattr(exact_result, "explored_edges", [])))
        left_store.chart_centers = _merge_chart_centers_joint(left_store.chart_centers, robot, np.asarray(exact_result.path, dtype=float))
        _, node_ids, _ = ex66.connect_path_to_stage_graph(
            store=left_store,
            source_node_id=int(source_id),
            path=np.asarray(exact_result.path, dtype=float),
            kind=ex66.LEFT_MOTION,
            terminal_node_id=int(target_id),
            preserve_dense_path=True,
            max_joint_step_for_edge=LOCAL_MAX_JOINT_STEP,
        )
        _update_stage_frontier_joint(left_store, node_ids + [int(target_id)], guide_point, robot)
        node_gain += max(0, len(node_ids) - 2)
        break
    return node_gain, eval_count


def _bridge_plane_hypothesis_components_joint(
    plane_store: ex66.StageEvidenceStore,
    left_plane_hypotheses: list[ex66.TransitionHypothesis],
    plane_right_hypotheses: list[ex66.TransitionHypothesis],
    robot,
    collision_fn=None,
) -> tuple[int, int]:
    left_ids = sorted({int(hyp.plane_node_id) for hyp in left_plane_hypotheses if hyp.plane_node_id is not None})
    right_ids = sorted({int(hyp.plane_node_id) for hyp in plane_right_hypotheses if hyp.plane_node_id is not None})
    if len(left_ids) == 0 or len(right_ids) == 0:
        return 0, 0

    candidate_pairs: list[tuple[float, int, int]] = []
    for left_id in left_ids:
        q_left = plane_store.graph.nodes[left_id].q
        ee_left = _task_point(robot, q_left)
        for right_id in right_ids:
            if left_id == right_id:
                continue
            ee_right = _task_point(robot, plane_store.graph.nodes[right_id].q)
            candidate_pairs.append((float(np.linalg.norm(ee_left - ee_right)), left_id, right_id))
    candidate_pairs.sort(key=lambda item: item[0])

    def guided_plane_path(q_start: np.ndarray, q_goal: np.ndarray) -> np.ndarray | None:
        p_start = _task_point(robot, q_start)
        p_goal = _task_point(robot, q_goal)
        q_path = [np.asarray(q_start, dtype=float)]
        current = np.asarray(q_start, dtype=float)
        lower = np.asarray(plane_store.manifold.joint_lower, dtype=float)
        upper = np.asarray(plane_store.manifold.joint_upper, dtype=float)
        for alpha in np.linspace(0.125, 1.0, 8):
            target = (1.0 - float(alpha)) * p_start + float(alpha) * p_goal
            q_guess = inverse_kinematics_start(
                robot,
                target,
                warm_start=current,
                joint_lower=lower,
                joint_upper=upper,
                tol=8e-2,
            )
            if q_guess is None:
                return None
            projection = plane_store.manifold.project(np.asarray(q_guess, dtype=float), tol=1e-6, max_iters=80)
            if not projection.success:
                return None
            q_next = np.asarray(projection.x_projected, dtype=float)
            if collision_fn is not None and bool(collision_fn(q_next)):
                return None
            if float(np.linalg.norm(_task_point(robot, q_next) - np.asarray(target, dtype=float))) > 1e-1:
                return None
            q_path.append(q_next)
            current = q_next
        return np.asarray(q_path, dtype=float)

    eval_count = 0
    node_gain = 0
    tried = 0
    bridge_try_limit = max(ex66.PLANE_BRIDGE_TRY_LIMIT, min(40, len(candidate_pairs)))
    for _, left_id, right_id in candidate_pairs:
        if tried >= bridge_try_limit:
            break
        tried += 1
        exact_result = explore_joint_manifold(
            manifold=plane_store.manifold,
            start=plane_store.graph.nodes[left_id].q,
            goal=plane_store.graph.nodes[right_id].q,
            max_step=LOCAL_MAX_JOINT_STEP,
            local_max_joint_step=LOCAL_MAX_JOINT_STEP,
            collision_fn=collision_fn,
        )
        _record_local_planner_result(plane_store, exact_result)
        eval_count += 1
        if not exact_result.success:
            if not bool(getattr(exact_result, "joint_continuity_success", True)):
                plane_store.stage_edges_rejected_joint_jump += 1
            fallback_path = guided_plane_path(
                np.asarray(plane_store.graph.nodes[left_id].q, dtype=float),
                np.asarray(plane_store.graph.nodes[right_id].q, dtype=float),
            )
            if fallback_path is None or len(fallback_path) < 2:
                continue
            _steps, fallback_max_step, _fallback_mean_step, _fallback_worst_idx = _joint_step_statistics(fallback_path)
            if fallback_max_step > LOCAL_MAX_JOINT_STEP + 1e-9:
                plane_store.stage_edges_rejected_joint_jump += 1
                continue
            class _FallbackResult:
                success = True
                def __init__(self, path):
                    self.path = np.asarray(path, dtype=float)
                    self.explored_edges = [
                        (np.asarray(path[idx], dtype=float), np.asarray(path[idx + 1], dtype=float))
                        for idx in range(len(path) - 1)
                    ]
            exact_result = _FallbackResult(fallback_path)
        plane_store.explored_edges = ex66.merge_edges(plane_store.explored_edges, list(getattr(exact_result, "explored_edges", [])))
        plane_store.chart_centers = _merge_chart_centers_joint(plane_store.chart_centers, robot, np.asarray(exact_result.path, dtype=float))
        _, node_ids, _ = ex66.connect_path_to_stage_graph(
            store=plane_store,
            source_node_id=left_id,
            path=np.asarray(exact_result.path, dtype=float),
            kind=ex66.PLANE_MOTION,
            terminal_node_id=right_id,
            preserve_dense_path=True,
            max_joint_step_for_edge=LOCAL_MAX_JOINT_STEP,
        )
        _update_stage_frontier_joint(plane_store, node_ids + [left_id, right_id], _task_point(robot, plane_store.graph.nodes[right_id].q), robot)
        node_gain += max(0, len(node_ids) - 2)
        break
    return node_gain, eval_count


def _connect_right_hypothesis_to_goal_joint(
    right_store: ex66.StageEvidenceStore,
    plane_right_hypotheses: list[ex66.TransitionHypothesis],
    goal_node_id: int,
    guide_point: np.ndarray,
    robot,
    collision_fn=None,
) -> tuple[int, int]:
    right_ids = sorted({int(hyp.right_node_id) for hyp in plane_right_hypotheses if hyp.right_node_id is not None})
    if len(right_ids) == 0:
        return 0, 0

    scored = []
    for node_id in right_ids:
        q = right_store.graph.nodes[node_id].q
        scored.append((float(np.linalg.norm(_task_point(robot, q) - np.asarray(guide_point, dtype=float))), node_id))
    scored.sort(key=lambda item: item[0])

    eval_count = 0
    node_gain = 0
    bridge_try_limit = max(ex66.RIGHT_BRIDGE_TRY_LIMIT, 8)
    for _, node_id in scored[: min(bridge_try_limit, len(scored))]:
        exact_result = explore_joint_manifold(
            manifold=right_store.manifold,
            start=right_store.graph.nodes[node_id].q,
            goal=right_store.graph.nodes[goal_node_id].q,
            max_step=LOCAL_MAX_JOINT_STEP,
            local_max_joint_step=LOCAL_MAX_JOINT_STEP,
            collision_fn=collision_fn,
        )
        _record_local_planner_result(right_store, exact_result)
        eval_count += 1
        if not exact_result.success:
            if not bool(getattr(exact_result, "joint_continuity_success", True)):
                right_store.stage_edges_rejected_joint_jump += 1
            continue
        right_store.explored_edges = ex66.merge_edges(right_store.explored_edges, list(getattr(exact_result, "explored_edges", [])))
        right_store.chart_centers = _merge_chart_centers_joint(right_store.chart_centers, robot, np.asarray(exact_result.path, dtype=float))
        _, node_ids, _ = ex66.connect_path_to_stage_graph(
            store=right_store,
            source_node_id=node_id,
            path=np.asarray(exact_result.path, dtype=float),
            kind=ex66.RIGHT_MOTION,
            terminal_node_id=goal_node_id,
            preserve_dense_path=True,
            max_joint_step_for_edge=LOCAL_MAX_JOINT_STEP,
        )
        _update_stage_frontier_joint(right_store, node_ids + [goal_node_id], guide_point, robot)
        node_gain += max(0, len(node_ids) - 2)
        break
    return node_gain, eval_count


def _build_stage_task_path(store: ex66.StageEvidenceStore, node_ids: list[int], edge_ids: list[int], robot) -> np.ndarray:
    return joint_path_to_task_path(robot, ex66.build_stage_raw_path(store, node_ids, edge_ids))


def _labeled_dense_joint_path(*stage_paths: tuple[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    path_parts: list[np.ndarray] = []
    label_parts: list[str] = []
    for stage, path in stage_paths:
        arr = np.asarray(path, dtype=float)
        if len(arr) == 0:
            continue
        if len(path_parts) > 0 and float(np.linalg.norm(path_parts[-1][-1] - arr[0])) <= 1e-9:
            arr = arr[1:]
        if len(arr) == 0:
            continue
        path_parts.append(arr)
        label_parts.extend([str(stage)] * len(arr))
    dense_path = np.vstack(path_parts) if len(path_parts) > 0 else np.zeros((0, 3), dtype=float)
    if len(label_parts) != len(dense_path):
        label_parts = label_parts[: len(dense_path)]
        if len(label_parts) < len(dense_path) and len(stage_paths) > 0:
            label_parts.extend([str(stage_paths[-1][0])] * (len(dense_path) - len(label_parts)))
    return np.asarray(dense_path, dtype=float), label_parts


def _transition_theta_index(path: np.ndarray, theta: np.ndarray) -> int:
    arr = np.asarray(path, dtype=float)
    q = np.asarray(theta, dtype=float)
    if len(arr) == 0 or q.size != 3:
        return -1
    distances = np.linalg.norm((arr - q.reshape(1, 3) + np.pi) % (2.0 * np.pi) - np.pi, axis=1)
    return int(np.argmin(distances)) if len(distances) > 0 else -1


def _stack_residual_norm(source_manifold, target_manifold, theta: np.ndarray) -> float:
    q = np.asarray(theta, dtype=float)
    if q.size != 3:
        return float("inf")
    source = float(np.linalg.norm(np.ravel(source_manifold.residual(q))))
    target = float(np.linalg.norm(np.ravel(target_manifold.residual(q))))
    return float(np.linalg.norm([source, target]))


def _joint_step_statistics(joint_path: np.ndarray) -> tuple[np.ndarray, float, float, int]:
    q_path = np.asarray(joint_path, dtype=float)
    if len(q_path) < 2:
        return np.zeros(0, dtype=float), 0.0, 0.0, -1
    wrapped_deltas = (np.diff(q_path, axis=0) + np.pi) % (2.0 * np.pi) - np.pi
    joint_steps = np.linalg.norm(wrapped_deltas, axis=1)
    worst_step_index = int(np.argmax(joint_steps)) if len(joint_steps) > 0 else -1
    max_joint_step = float(np.max(joint_steps)) if len(joint_steps) > 0 else 0.0
    mean_joint_step = float(np.mean(joint_steps)) if len(joint_steps) > 0 else 0.0
    return np.asarray(joint_steps, dtype=float), max_joint_step, mean_joint_step, worst_step_index


def audit_stage_graph_joint_continuity(
    store: ex66.StageEvidenceStore,
    stage_name: str,
    max_joint_step: float = LOCAL_MAX_JOINT_STEP,
) -> dict[str, int | float | str]:
    total_edges = len(store.graph.edges)
    bad_edges = 0
    max_bad_step = 0.0
    worst_edge_id = -1
    worst_local_index = -1
    worst_src = -1
    worst_dst = -1
    for edge_id, edge in store.graph.edges.items():
        steps, max_step, _mean_step, local_idx = _joint_step_statistics(np.asarray(edge.path, dtype=float))
        if len(steps) > 0 and max_step > float(max_joint_step) + 1e-9:
            bad_edges += 1
            if max_step > max_bad_step:
                max_bad_step = float(max_step)
                worst_edge_id = int(edge_id)
                worst_local_index = int(local_idx)
                worst_src = int(edge.src)
                worst_dst = int(edge.dst)
    return {
        f"{stage_name}_stage_total_edges": int(total_edges),
        f"{stage_name}_stage_bad_edges": int(bad_edges),
        f"{stage_name}_stage_max_bad_edge_step": float(max_bad_step),
        f"{stage_name}_stage_worst_bad_edge_id": int(worst_edge_id),
        f"{stage_name}_stage_worst_bad_edge_local_index": int(worst_local_index),
        f"{stage_name}_stage_worst_bad_edge_src": int(worst_src),
        f"{stage_name}_stage_worst_bad_edge_dst": int(worst_dst),
        f"{stage_name}_stage_edges_rejected_joint_jump": int(store.stage_edges_rejected_joint_jump),
    }


def _certify_dense_joint_execution(
    dense_joint_path: np.ndarray,
    stage_labels: list[str],
    stores: dict[str, ex66.StageEvidenceStore],
    collision_fn=None,
    robot=None,
    constraint_tol: float = 2.0e-3,
    max_joint_step: float = 0.08,
) -> dict[str, object]:
    q_path = np.asarray(dense_joint_path, dtype=float)
    if len(q_path) == 0:
        return {
            "constraint_certified": False,
            "joint_continuity_certified": False,
            "execution_certified": False,
            "collision_free": False,
            "residuals": np.zeros(0, dtype=float),
            "joint_steps": np.zeros(0, dtype=float),
            "max_constraint_residual": float("inf"),
            "mean_constraint_residual": float("inf"),
            "max_joint_step": float("inf"),
            "mean_joint_step": float("inf"),
            "worst_constraint_index": -1,
            "worst_constraint_stage": "none",
            "worst_joint_step_index": -1,
            "message": "dense joint path is empty",
        }
    residuals: list[float] = []
    worst_index = -1
    worst_stage = "none"
    collision_free = True
    for idx, q in enumerate(q_path):
        stage = str(stage_labels[idx]) if idx < len(stage_labels) else ""
        store = stores.get(stage)
        if store is None:
            residual = float("inf")
        else:
            residual = float(np.linalg.norm(store.manifold.residual(np.asarray(q, dtype=float))))
            if not bool(store.manifold.within_bounds(np.asarray(q, dtype=float), tol=constraint_tol)):
                residual = max(residual, 10.0 * constraint_tol)
        if collision_fn is not None and bool(collision_fn(np.asarray(q, dtype=float))):
            collision_free = False
            residual = max(residual, 100.0 * constraint_tol)
        residuals.append(float(residual))
        if worst_index < 0 or residual > residuals[worst_index]:
            worst_index = idx
            worst_stage = stage
    residual_arr = np.asarray(residuals, dtype=float)
    max_residual = float(np.max(residual_arr)) if len(residual_arr) > 0 else float("inf")
    mean_residual = float(np.mean(residual_arr)) if len(residual_arr) > 0 else float("inf")
    constraint_certified = bool(collision_free and max_residual <= constraint_tol)

    joint_steps, max_step_seen, mean_step_seen, worst_step_index = _joint_step_statistics(q_path)
    joint_continuity_certified = bool(max_step_seen <= max_joint_step + 1e-9)
    execution_certified = bool(constraint_certified and joint_continuity_certified and collision_free)

    jump_details = ""
    if worst_step_index >= 0 and len(q_path) > worst_step_index + 1:
        stage_before = str(stage_labels[worst_step_index]) if worst_step_index < len(stage_labels) else "unknown"
        stage_after = str(stage_labels[worst_step_index + 1]) if worst_step_index + 1 < len(stage_labels) else "unknown"
        q_before = np.asarray(q_path[worst_step_index], dtype=float)
        q_after = np.asarray(q_path[worst_step_index + 1], dtype=float)
        task_distance = 0.0
        ee_before = np.zeros(3, dtype=float)
        ee_after = np.zeros(3, dtype=float)
        if robot is not None:
            ee_before = _task_point(robot, q_before)
            ee_after = _task_point(robot, q_after)
            task_distance = float(np.linalg.norm(ee_after - ee_before))
        jump_details = (
            f", worst_joint_step_index={worst_step_index}, stage_before={stage_before}, "
            f"stage_after={stage_after}, joint_distance_at_jump={max_step_seen:.4g}, "
            f"task_distance_at_jump={task_distance:.4g}, "
            f"q_before={np.array2string(q_before, precision=3)}, "
            f"q_after={np.array2string(q_after, precision=3)}"
        )
        if robot is not None:
            jump_details += (
                f", ee_before={np.array2string(ee_before, precision=3)}, "
                f"ee_after={np.array2string(ee_after, precision=3)}"
            )

    if execution_certified:
        message = (
            f"dense joint path execution certified: points={len(q_path)}, "
            f"max_residual={max_residual:.4g}, mean_residual={mean_residual:.4g}, "
            f"max_joint_step={max_step_seen:.4g}, mean_joint_step={mean_step_seen:.4g}, collision_free=True"
        )
    else:
        message = (
            f"dense joint path failed execution certification: points={len(q_path)}, "
            f"max_residual={max_residual:.4g}, mean_residual={mean_residual:.4g}, "
            f"worst_stage={worst_stage}, worst_index={worst_index}, collision_free={collision_free}, "
            f"constraint_certified={constraint_certified}, joint_continuity_certified={joint_continuity_certified}, "
            f"max_joint_step={max_step_seen:.4g}, mean_joint_step={mean_step_seen:.4g}"
            f"{jump_details}"
        )
    return {
        "constraint_certified": bool(constraint_certified),
        "joint_continuity_certified": bool(joint_continuity_certified),
        "execution_certified": bool(execution_certified),
        "collision_free": bool(collision_free),
        "residuals": np.asarray(residual_arr, dtype=float),
        "joint_steps": np.asarray(joint_steps, dtype=float),
        "max_constraint_residual": float(max_residual),
        "mean_constraint_residual": float(mean_residual),
        "max_joint_step": float(max_step_seen),
        "mean_joint_step": float(mean_step_seen),
        "worst_constraint_index": int(worst_index),
        "worst_constraint_stage": str(worst_stage),
        "worst_joint_step_index": int(worst_step_index),
        "message": str(message),
    }


def _joint_route_candidate_rank_key(candidate: ex66.SequentialRouteCandidate) -> tuple[float, float]:
    max_residual = (
        float(np.max(candidate.dense_joint_path_constraint_residuals))
        if len(candidate.dense_joint_path_constraint_residuals) > 0
        else 0.0
    )
    if bool(candidate.dense_joint_path_execution_certified):
        task_length = ex66.path_cost(candidate.raw_path)
        combined_cost = (
            float(candidate.total_cost)
            + 0.5 * float(task_length)
            + 0.2 * float(candidate.dense_joint_path_max_joint_step)
            + 10.0 * max_residual
        )
        return 0.0, float(combined_cost)
    rejection_penalty = (
        1000.0
        + 10.0 * float(candidate.dense_joint_path_max_joint_step)
        + 100.0 * max_residual
    )
    return 1.0, float(candidate.total_cost) + rejection_penalty


def realize_selected_transition_route_jointspace(
    *,
    robot,
    start_q: np.ndarray,
    left_transition_q: np.ndarray,
    plane_entry_q: np.ndarray,
    plane_exit_q: np.ndarray,
    right_transition_q: np.ndarray,
    goal_q: np.ndarray,
    transition_entry_task: np.ndarray,
    transition_exit_task: np.ndarray,
    left_robot_manifold,
    plane_robot_manifold,
    right_robot_manifold,
    joint_max_step: float,
    collision_fn=None,
) -> tuple[np.ndarray, list[str], dict[str, object]]:
    """Realize the selected transition structure with fresh local planning.

    The evidence graph chooses the transition pair. Execution is then built
    from three certified local segments, instead of following random graph
    exploration nodes inside each stage.
    """
    stores = {
        ex66.LEFT_STAGE: ex66.StageEvidenceStore(stage=ex66.LEFT_STAGE, manifold=left_robot_manifold),
        ex66.PLANE_STAGE: ex66.StageEvidenceStore(stage=ex66.PLANE_STAGE, manifold=plane_robot_manifold),
        ex66.RIGHT_STAGE: ex66.StageEvidenceStore(stage=ex66.RIGHT_STAGE, manifold=right_robot_manifold),
    }
    def choose_handoff(
        source_manifold,
        target_manifold,
        transition_task,
        source_hint,
        target_hint,
        *,
        source_name: str,
        target_name: str,
    ) -> tuple[np.ndarray | None, dict[str, object]]:
        task = np.asarray(transition_task, dtype=float)
        source_hint = np.asarray(source_hint, dtype=float)
        target_hint = np.asarray(target_hint, dtype=float)
        diagnostics: dict[str, object] = _handoff_base_diagnostics(
            source_manifold=source_manifold,
            target_manifold=target_manifold,
            source_hint=source_hint,
            target_hint=target_hint,
            robot=robot,
        )
        candidates = [source_hint, target_hint, 0.5 * (source_hint + target_hint)]
        valid: list[tuple[float, np.ndarray]] = []
        for candidate in candidates:
            q = np.asarray(candidate, dtype=float)
            if not bool(source_manifold.within_bounds(q, tol=2e-3)) or not bool(target_manifold.within_bounds(q, tol=2e-3)):
                continue
            if not bool(source_manifold.is_valid(q, tol=2e-3)) or not bool(target_manifold.is_valid(q, tol=2e-3)):
                continue
            if collision_fn is not None and bool(collision_fn(q)):
                continue
            task_error = float(np.linalg.norm(_task_point(robot, q) - task))
            hint_distance = max(
                float(np.linalg.norm((q - source_hint + np.pi) % (2.0 * np.pi) - np.pi)),
                float(np.linalg.norm((q - target_hint + np.pi) % (2.0 * np.pi) - np.pi)),
            )
            valid.append((task_error + 0.1 * hint_distance, q))
        if len(valid) > 0:
            diagnostics["handoff_method"] = "existing_valid_hint"
            return np.asarray(min(valid, key=lambda item: item[0])[1], dtype=float), diagnostics

        stacked_q, stacked_diagnostics = solve_shared_transition_q(
            robot=robot,
            source_manifold=source_manifold,
            target_manifold=target_manifold,
            transition_task=task,
            source_hint=source_hint,
            target_hint=target_hint,
            collision_fn=collision_fn,
            residual_tol=2.0e-3,
            task_tol=1.2e-1,
        )
        diagnostics.update(stacked_diagnostics)
        if stacked_q is not None:
            diagnostics["handoff_method"] = "stacked_least_squares"
            return np.asarray(stacked_q, dtype=float), diagnostics

        shared_q = _shared_transition_configuration(
            source_manifold=source_manifold,
            target_manifold=target_manifold,
            transition_task=task,
            source_q_hint=source_hint,
            target_q_hint=target_hint,
            robot=robot,
            collision_fn=collision_fn,
            task_tol=7.5e-2,
        )
        if shared_q is not None:
            diagnostics["handoff_method"] = "legacy_ik_projection"
            return np.asarray(shared_q, dtype=float), diagnostics
        diagnostics["handoff_method"] = f"failed_{source_name}_{target_name}"
        return None, diagnostics

    entry_q, entry_diagnostics = choose_handoff(
        left_robot_manifold,
        plane_robot_manifold,
        transition_entry_task,
        left_transition_q,
        plane_entry_q,
        source_name="left",
        target_name="plane",
    )
    if entry_q is None:
        return np.zeros((0, 3), dtype=float), [], {
            "success": False,
            "message": (
                "failed to construct shared left-plane transition configuration; "
                + _format_handoff_diagnostics(entry_diagnostics, source_name="left", target_name="plane")
            ),
            "certification": {},
            "handoff_diagnostics": entry_diagnostics,
        }
    exit_q, exit_diagnostics = choose_handoff(
        plane_robot_manifold,
        right_robot_manifold,
        transition_exit_task,
        plane_exit_q,
        right_transition_q,
        source_name="plane",
        target_name="right",
    )
    if exit_q is None:
        return np.zeros((0, 3), dtype=float), [], {
            "success": False,
            "message": (
                "failed to construct shared plane-right transition configuration; "
                + _format_handoff_diagnostics(exit_diagnostics, source_name="plane", target_name="right")
            ),
            "certification": {},
            "handoff_diagnostics": exit_diagnostics,
        }

    segment_specs = [
        (ex66.LEFT_STAGE, left_robot_manifold, np.asarray(start_q, dtype=float), np.asarray(entry_q, dtype=float)),
        (ex66.PLANE_STAGE, plane_robot_manifold, np.asarray(entry_q, dtype=float), np.asarray(exit_q, dtype=float)),
        (ex66.RIGHT_STAGE, right_robot_manifold, np.asarray(exit_q, dtype=float), np.asarray(goal_q, dtype=float)),
    ]

    def task_guided_segment(stage: str, manifold, q0: np.ndarray, q1: np.ndarray) -> np.ndarray | None:
        p0 = _task_point(robot, q0)
        p1 = _task_point(robot, q1)
        task_step = 1.2e-2
        if stage in {ex66.LEFT_STAGE, ex66.RIGHT_STAGE} and hasattr(manifold, "center") and hasattr(manifold, "radius"):
            task_path = ex66.smooth_sphere_arc(
                np.asarray(manifold.center, dtype=float),
                float(manifold.radius),
                p0,
                p1,
                num=max(120, int(np.ceil(float(np.linalg.norm(p1 - p0)) / task_step))),
            )
        else:
            task_path = ex66.smooth_plane_segment(
                p0,
                p1,
                num=max(80, int(np.ceil(float(np.linalg.norm(p1 - p0)) / task_step))),
            )
        path: list[np.ndarray] = [np.asarray(q0, dtype=float)]
        current = np.asarray(q0, dtype=float)
        lower = np.asarray(getattr(manifold, "joint_lower", -np.pi * np.ones(3, dtype=float)), dtype=float)
        upper = np.asarray(getattr(manifold, "joint_upper", np.pi * np.ones(3, dtype=float)), dtype=float)
        for target in np.asarray(task_path[1:], dtype=float):
            q_guess = inverse_kinematics_start(
                robot,
                target,
                warm_start=current,
                joint_lower=lower,
                joint_upper=upper,
                tol=6.0e-2,
            )
            if q_guess is None:
                return None
            if hasattr(manifold, "project_local"):
                projection = manifold.project_local(q_guess, tol=1e-6, max_iters=100, regularization=2.0e-3)
            else:
                projection = manifold.project(q_guess, tol=1e-6, max_iters=100)
            if not projection.success:
                return None
            q_next = np.asarray(projection.x_projected, dtype=float)
            if collision_fn is not None and bool(collision_fn(q_next)):
                return None
            if float(np.linalg.norm((q_next - current + np.pi) % (2.0 * np.pi) - np.pi)) > float(joint_max_step) + 1e-9:
                return None
            path.append(q_next.copy())
            current = q_next
        if float(np.linalg.norm((path[-1] - q1 + np.pi) % (2.0 * np.pi) - np.pi)) > float(joint_max_step) + 1e-9:
            path.append(np.asarray(q1, dtype=float))
        cert = _certify_dense_joint_execution(
            dense_joint_path=np.asarray(path, dtype=float),
            stage_labels=[stage] * len(path),
            stores={stage: stores[stage]},
            collision_fn=collision_fn,
            robot=robot,
            max_joint_step=float(joint_max_step),
        )
        return np.asarray(path, dtype=float) if bool(cert["execution_certified"]) else None

    segment_paths: list[tuple[str, np.ndarray]] = []
    for stage, manifold, q0, q1 in segment_specs:
        result = explore_joint_manifold(
            manifold=manifold,
            start=q0,
            goal=q1,
            max_step=0.5 * float(joint_max_step),
            local_max_joint_step=float(joint_max_step),
            collision_fn=collision_fn,
        )
        path = np.asarray(result.path, dtype=float) if result.success else np.zeros((0, 3), dtype=float)
        if not result.success:
            guided_path = task_guided_segment(stage, manifold, q0, q1)
            if guided_path is not None:
                path = np.asarray(guided_path, dtype=float)
            else:
                return np.zeros((0, 3), dtype=float), [], {
                    "success": False,
                    "message": f"local realization failed on {stage}: {result.message}; task-guided continuation also failed",
                    "certification": {},
                }
        if len(path) < 2:
            return np.zeros((0, 3), dtype=float), [], {
                "success": False,
                "message": f"local realization produced an empty segment on {stage}",
                "certification": {},
            }
        segment_paths.append((stage, np.asarray(path, dtype=float)))

    dense_joint_path, labels = _labeled_dense_joint_path(*segment_paths)
    certification = _certify_dense_joint_execution(
        dense_joint_path=dense_joint_path,
        stage_labels=labels,
        stores=stores,
        collision_fn=collision_fn,
        robot=robot,
        max_joint_step=float(joint_max_step),
    )
    if not bool(certification["execution_certified"]):
        return dense_joint_path, labels, {
            "success": False,
            "message": f"realized local route failed certification: {certification['message']}",
            "certification": certification,
        }
    return dense_joint_path, labels, {
        "success": True,
        "message": (
            "selected transition local replan certified: "
            f"points={len(dense_joint_path)}, max_joint_step={float(certification['max_joint_step']):.4g}, "
            f"max_residual={float(certification['max_constraint_residual']):.4g}"
        ),
        "certification": certification,
    }


def _extract_committed_route_joint(
    left_store: ex66.StageEvidenceStore,
    plane_store: ex66.StageEvidenceStore,
    right_store: ex66.StageEvidenceStore,
    start_node_id: int,
    goal_node_id: int,
    left_plane_hypotheses: list[ex66.TransitionHypothesis],
    plane_right_hypotheses: list[ex66.TransitionHypothesis],
    robot,
    collision_fn=None,
) -> tuple[ex66.SequentialRouteCandidate | None, bool, bool, int, dict[str, int]]:
    left_dist, left_prev_node, left_prev_edge = ex66.shortest_paths_in_stage(left_store, start_node_id)
    right_dist, right_prev_node, right_prev_edge = ex66.shortest_paths_in_stage(right_store, goal_node_id)

    entry_candidates = [hyp for hyp in left_plane_hypotheses if hyp.left_node_id in left_dist]
    exit_candidates = [hyp for hyp in plane_right_hypotheses if hyp.right_node_id in right_dist]
    has_committed_entry = len(entry_candidates) > 0
    has_committed_exit = len(exit_candidates) > 0
    route_stats = {
        "route_candidates_evaluated": 0,
        "route_candidates_constraint_certified": 0,
        "route_candidates_execution_certified": 0,
        "route_candidates_rejected_joint_jump": 0,
        "route_candidates_realized_by_local_replan": 0,
    }
    if not has_committed_entry or not has_committed_exit:
        return None, has_committed_entry, has_committed_exit, 0, route_stats

    best: ex66.SequentialRouteCandidate | None = None
    best_key: tuple[float, float] | None = None
    built_candidates: list[tuple[tuple[float, float], ex66.SequentialRouteCandidate]] = []
    pairs_evaluated = 0
    for entry_hyp in entry_candidates:
        plane_dist, plane_prev_node, plane_prev_edge = ex66.shortest_paths_in_stage(plane_store, int(entry_hyp.plane_node_id))
        for exit_hyp in exit_candidates:
            pairs_evaluated += 1
            plane_exit_id = int(exit_hyp.plane_node_id)
            if plane_exit_id not in plane_dist:
                continue

            left_node_path, left_edge_path = ex66.reconstruct_stage_path(
                left_store,
                start_node_id,
                int(entry_hyp.left_node_id),
                left_prev_node,
                left_prev_edge,
            )
            plane_node_path, plane_edge_path = ex66.reconstruct_stage_path(
                plane_store,
                int(entry_hyp.plane_node_id),
                plane_exit_id,
                plane_prev_node,
                plane_prev_edge,
            )
            right_goal_to_entry_nodes, right_goal_to_entry_edges = ex66.reconstruct_stage_path(
                right_store,
                goal_node_id,
                int(exit_hyp.right_node_id),
                right_prev_node,
                right_prev_edge,
            )
            if len(left_node_path) == 0 or len(plane_node_path) == 0 or len(right_goal_to_entry_nodes) == 0:
                continue
            right_node_path = list(reversed(right_goal_to_entry_nodes))
            right_edge_path = list(reversed(right_goal_to_entry_edges))

            left_joint_path = ex66.build_stage_raw_path(left_store, left_node_path, left_edge_path)
            plane_joint_path = ex66.build_stage_raw_path(plane_store, plane_node_path, plane_edge_path)
            right_joint_path = ex66.build_stage_raw_path(right_store, right_node_path, right_edge_path)
            left_task_path = _build_stage_task_path(left_store, left_node_path, left_edge_path, robot)
            plane_task_path = _build_stage_task_path(plane_store, plane_node_path, plane_edge_path, robot)
            right_task_path = _build_stage_task_path(right_store, right_node_path, right_edge_path, robot)
            raw_path = ex66.concatenate_paths(
                left_task_path,
                np.asarray([entry_hyp.q], dtype=float),
                plane_task_path,
                np.asarray([exit_hyp.q], dtype=float),
                right_task_path,
            )
            import traceback

            try:
                left_center = left_store.manifold.center
                left_radius = left_store.manifold.radius
                p_left_start = end_effector_point(robot, left_store.graph.nodes[left_node_path[0]].q)
                p_left_end = np.asarray(entry_hyp.q, dtype=float)
                left_disp = ex66.smooth_sphere_arc(
                    left_center,
                    left_radius,
                    p_left_start,
                    p_left_end,
                    num=ex66.DISPLAY_SPHERE_SAMPLES,
                )

                p_right_start = np.asarray(exit_hyp.q, dtype=float)
                p_right_end = end_effector_point(robot, right_store.graph.nodes[right_node_path[-1]].q)
                right_center = right_store.manifold.center
                right_radius = right_store.manifold.radius
                right_disp = ex66.smooth_sphere_arc(
                    right_center,
                    right_radius,
                    p_right_start,
                    p_right_end,
                    num=ex66.DISPLAY_SPHERE_SAMPLES,
                )

                plane_entry_anchor = (
                    np.asarray(plane_task_path[0], dtype=float)
                    if len(plane_task_path) > 0
                    else np.asarray(entry_hyp.q, dtype=float)
                )
                plane_exit_anchor = (
                    np.asarray(plane_task_path[-1], dtype=float)
                    if len(plane_task_path) > 0
                    else np.asarray(exit_hyp.q, dtype=float)
                )
                plane_entry_connector = ex66.smooth_plane_segment(
                    np.asarray(entry_hyp.q, dtype=float),
                    plane_entry_anchor,
                    num=max(4, ex66.DISPLAY_PLANE_SAMPLES // 2),
                )
                plane_exit_connector = ex66.smooth_plane_segment(
                    plane_exit_anchor,
                    np.asarray(exit_hyp.q, dtype=float),
                    num=max(4, ex66.DISPLAY_PLANE_SAMPLES // 2),
                )

                display_path_smooth = ex66.concatenate_paths(
                    left_disp,
                    plane_entry_connector,
                    plane_task_path,
                    plane_exit_connector,
                    right_disp,
                )
                if len(display_path_smooth) == 0:
                    raise RuntimeError("smoothed display path is empty")
                display_path_final = np.asarray(display_path_smooth, dtype=float)
            except Exception:
                print("Warning: display smoothing failed, using raw linear path. Details:", traceback.format_exc())
                display_path_final = np.asarray(raw_path, dtype=float)

            joint_path = ex66.concatenate_paths(
                left_joint_path,
                plane_joint_path,
                right_joint_path,
            )
            dense_joint_path, dense_labels = _labeled_dense_joint_path(
                (ex66.LEFT_STAGE, left_joint_path),
                (ex66.PLANE_STAGE, plane_joint_path),
                (ex66.RIGHT_STAGE, right_joint_path),
            )
            certification = _certify_dense_joint_execution(
                dense_joint_path=dense_joint_path,
                stage_labels=dense_labels,
                stores={
                    ex66.LEFT_STAGE: left_store,
                    ex66.PLANE_STAGE: plane_store,
                    ex66.RIGHT_STAGE: right_store,
                },
                collision_fn=collision_fn,
                robot=robot,
                max_joint_step=LOCAL_MAX_JOINT_STEP,
            )
            dense_residuals = np.asarray(certification["residuals"], dtype=float)
            dense_joint_steps = np.asarray(certification["joint_steps"], dtype=float)
            entry_theta = np.asarray(getattr(entry_hyp, "transition_theta", np.zeros(0, dtype=float)), dtype=float)
            exit_theta = np.asarray(getattr(exit_hyp, "transition_theta", np.zeros(0, dtype=float)), dtype=float)
            left_plane_stack_residual = _stack_residual_norm(left_store.manifold, plane_store.manifold, entry_theta)
            plane_right_stack_residual = _stack_residual_norm(plane_store.manifold, right_store.manifold, exit_theta)
            max_transition_stack_residual = float(max(left_plane_stack_residual, plane_right_stack_residual))
            transition_stack_certified = bool(max_transition_stack_residual <= 1.0e-3)
            selected_left_plane_transition_index = _transition_theta_index(dense_joint_path, entry_theta)
            selected_plane_right_transition_index = _transition_theta_index(dense_joint_path, exit_theta)
            execution_certified = bool(certification["execution_certified"] and transition_stack_certified)
            route_stats["route_candidates_evaluated"] += 1
            if bool(certification["constraint_certified"]):
                route_stats["route_candidates_constraint_certified"] += 1
            if bool(execution_certified):
                route_stats["route_candidates_execution_certified"] += 1
            elif bool(certification["constraint_certified"]) and not bool(certification["joint_continuity_certified"]):
                route_stats["route_candidates_rejected_joint_jump"] += 1

            total_cost = ex66.path_cost(raw_path)
            dense_message = str(certification["message"])
            if not transition_stack_certified:
                dense_message += (
                    "; transition stack not certified: "
                    f"left_plane={left_plane_stack_residual:.4g}, "
                    f"plane_right={plane_right_stack_residual:.4g}"
                )
            candidate = ex66.SequentialRouteCandidate(
                total_cost=float(total_cost),
                left_node_path=left_node_path,
                left_edge_path=left_edge_path,
                plane_node_path=plane_node_path,
                plane_edge_path=plane_edge_path,
                right_node_path=right_node_path,
                right_edge_path=right_edge_path,
                committed_nodes={
                    ex66.LEFT_STAGE: set(left_node_path),
                    ex66.PLANE_STAGE: set(plane_node_path),
                    ex66.RIGHT_STAGE: set(right_node_path),
                },
                raw_path=np.asarray(raw_path, dtype=float),
                display_path=display_path_final,
                joint_path=np.asarray(joint_path, dtype=float),
                dense_joint_path=np.asarray(dense_joint_path, dtype=float),
                dense_joint_path_stage_labels=list(dense_labels),
                dense_joint_path_constraint_residuals=np.asarray(dense_residuals, dtype=float),
                dense_joint_path_is_certified=bool(certification["constraint_certified"]),
                dense_joint_path_joint_steps=np.asarray(dense_joint_steps, dtype=float),
                dense_joint_path_max_joint_step=float(certification["max_joint_step"]),
                dense_joint_path_mean_joint_step=float(certification["mean_joint_step"]),
                dense_joint_path_worst_joint_step_index=int(certification["worst_joint_step_index"]),
                dense_joint_path_execution_certified=bool(execution_certified),
                dense_joint_path_constraint_certified=bool(certification["constraint_certified"]),
                dense_joint_path_joint_continuity_certified=bool(certification["joint_continuity_certified"]),
                dense_joint_path_message=dense_message,
                selected_left_plane_transition_theta=np.asarray(entry_theta, dtype=float),
                selected_plane_right_transition_theta=np.asarray(exit_theta, dtype=float),
                selected_left_plane_transition_index=int(selected_left_plane_transition_index),
                selected_plane_right_transition_index=int(selected_plane_right_transition_index),
                selected_left_plane_stack_residual=float(left_plane_stack_residual),
                selected_plane_right_stack_residual=float(plane_right_stack_residual),
                max_transition_stack_residual=float(max_transition_stack_residual),
                transition_stack_certified=bool(transition_stack_certified),
            )
            candidate_key = _joint_route_candidate_rank_key(candidate)
            built_candidates.append((candidate_key, candidate))
            if best is None or best_key is None or candidate_key < best_key:
                best = candidate
                best_key = candidate_key
    if best is not None:
        best.alternative_candidates = [candidate for _key, candidate in sorted(built_candidates, key=lambda item: item[0])]
    return best, has_committed_entry, has_committed_exit, pairs_evaluated, route_stats


def _realize_best_candidate_after_selection(
    candidate: ex66.SequentialRouteCandidate,
    stores: dict[str, ex66.StageEvidenceStore],
    robot,
    collision_fn=None,
) -> tuple[ex66.SequentialRouteCandidate, dict[str, int], str]:
    """Return the route assembled from stored dense joint-space graph edges.

    The thesis-facing robot demo must execute the configuration-space planner's
    own dense theta path. We therefore do not rebuild execution from selected
    task-space transition points here. Any post-selection local replan would be
    a different planner product and must be reported separately, not silently
    substituted for the graph route.
    """
    stats = {
        "route_candidates_realized_by_local_replan": 0,
        "final_route_realization_selected_transition_local_replan": 0,
        "graph_route_used_for_execution": 1,
        "route_candidates_local_replan_attempted": 0,
    }
    message = (
        "final_route_realization=stored_dense_joint_edges; "
        "graph_route_used_for_execution=True; "
        "execution_path_source=stored_dense_joint_edges; "
        f"{candidate.dense_joint_path_message}"
    )
    candidate.dense_joint_path_message = message
    return candidate, stats, message


def _empty_route(message: str, path_point: np.ndarray) -> ex66.FixedPlaneRoute:
    point = np.asarray(path_point, dtype=float).reshape(1, 3)
    return ex66.FixedPlaneRoute(
        success=False,
        message=message,
        total_rounds=0,
        candidate_evaluations=0,
        left_evidence_nodes=0,
        plane_evidence_nodes=0,
        right_evidence_nodes=0,
        committed_nodes=0,
        evidence_only_nodes=0,
        shared_proposals_processed=0,
        proposals_used_by_multiple_stages=0,
        plane_evidence_before_first_committed_entry=0,
        right_evidence_before_first_committed_exit=0,
        transition_hypotheses_left_plane=0,
        transition_hypotheses_plane_right=0,
        first_solution_round=None,
        best_solution_round=None,
        continued_after_first_solution=False,
        path=point,
        raw_path=point,
        certified_path_points=1,
        display_path_points=1,
        route_cost_raw=0.0,
        route_cost_display=0.0,
        graph_route_edges=0,
        obstacles=[],
        joint_path=np.zeros((0, 3), dtype=float),
        dense_joint_path=np.zeros((0, 3), dtype=float),
        dense_joint_path_stage_labels=[],
        dense_joint_path_constraint_residuals=np.zeros(0, dtype=float),
        dense_joint_path_is_certified=False,
        dense_joint_path_joint_steps=np.zeros(0, dtype=float),
        dense_joint_path_max_joint_step=0.0,
        dense_joint_path_mean_joint_step=0.0,
        dense_joint_path_worst_joint_step_index=-1,
        dense_joint_path_execution_certified=False,
        dense_joint_path_constraint_certified=False,
        dense_joint_path_joint_continuity_certified=False,
        dense_joint_path_message="joint-space route has sparse nodes only; dense constrained local edges not stored yet",
    )


def plan_fixed_manifold_multimodal_route_jointspace(
    families,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    robot,
    serial_mode: bool = False,
    obstacles=None,
    joint_max_step: float | None = None,
) -> ex66.FixedPlaneRoute:
    global LOCAL_MAX_JOINT_STEP
    if joint_max_step is not None:
        LOCAL_MAX_JOINT_STEP = max(1e-3, float(joint_max_step))
    left_family, plane_family, right_family = families
    left_geom = left_family.manifold(float(left_family.sample_lambdas()[0]))
    plane_geom = plane_family.manifold(float(plane_family.sample_lambdas()[0]))
    right_geom = right_family.manifold(float(right_family.sample_lambdas()[0]))

    left_base = ex66.unwrap_manifold(left_geom)
    plane_base = ex66.unwrap_manifold(plane_geom)
    right_base = ex66.unwrap_manifold(right_geom)
    if not isinstance(left_base, ex66.SphereManifold) or not isinstance(right_base, ex66.SphereManifold):
        return _empty_route("Robot joint-space mode requires sphere support manifolds on the left and right.", start_q)
    if not isinstance(plane_base, ex66.PlaneManifold):
        return _empty_route("Robot joint-space mode requires a plane transfer manifold in the middle stage.", start_q)

    joint_lower = -np.pi * np.ones(3, dtype=float)
    joint_upper = np.pi * np.ones(3, dtype=float)
    left_manifold = RobotSphereManifold(robot=robot, center=left_base.center, radius=left_base.radius, name="robot_left_sphere", joint_lower=joint_lower, joint_upper=joint_upper)
    plane_task_validity = (
        (lambda ee, geom=plane_geom: bool(geom.within_bounds(np.asarray(ee, dtype=float))))
        if isinstance(plane_geom, ex66.MaskedManifold)
        else None
    )
    plane_manifold = RobotPlaneManifold(
        robot=robot,
        point=plane_base.point,
        normal=plane_base.normal,
        name="robot_transfer_plane",
        joint_lower=joint_lower,
        joint_upper=joint_upper,
        task_space_validity_fn=plane_task_validity,
    )
    right_manifold = RobotSphereManifold(robot=robot, center=right_base.center, radius=right_base.radius, name="robot_right_sphere", joint_lower=joint_lower, joint_upper=joint_upper)
    active_obstacles = default_example_66_obstacles() if obstacles is None else list(obstacles)
    collision_fn = lambda joint_angles: configuration_in_collision(robot, joint_angles, active_obstacles)

    start_theta_seed = inverse_kinematics_start(robot, np.asarray(start_q, dtype=float), joint_lower=joint_lower, joint_upper=joint_upper)
    if start_theta_seed is None:
        return _empty_route("Failed to find a start joint configuration whose end-effector lies on the left support sphere.", start_q)
    goal_theta_seed = inverse_kinematics_start(robot, np.asarray(goal_q, dtype=float), warm_start=start_theta_seed, joint_lower=joint_lower, joint_upper=joint_upper)
    if goal_theta_seed is None:
        return _empty_route("Failed to find a goal joint configuration whose end-effector lies on the right support sphere.", goal_q)

    start_projection = left_manifold.project(start_theta_seed)
    goal_projection = right_manifold.project(goal_theta_seed)
    if not start_projection.success:
        return _empty_route("Failed to project the start IK seed onto the left robot-sphere manifold.", start_q)
    if not goal_projection.success:
        return _empty_route("Failed to project the goal IK seed onto the right robot-sphere manifold.", goal_q)
    start_theta = np.asarray(start_projection.x_projected, dtype=float)
    goal_theta = np.asarray(goal_projection.x_projected, dtype=float)
    if collision_fn(start_theta):
        return _empty_route("Start joint configuration collides with an obstacle.", start_q)
    if collision_fn(goal_theta):
        return _empty_route("Goal joint configuration collides with an obstacle.", goal_q)

    stores = {
        ex66.LEFT_STAGE: ex66.StageEvidenceStore(stage=ex66.LEFT_STAGE, manifold=left_manifold),
        ex66.PLANE_STAGE: ex66.StageEvidenceStore(stage=ex66.PLANE_STAGE, manifold=plane_manifold),
        ex66.RIGHT_STAGE: ex66.StageEvidenceStore(stage=ex66.RIGHT_STAGE, manifold=right_manifold),
    }

    start_node_id = ex66.add_stage_node(stores[ex66.LEFT_STAGE], start_theta, seeded_from_proposal=False)
    goal_node_id = ex66.add_stage_node(stores[ex66.RIGHT_STAGE], goal_theta, seeded_from_proposal=False)
    stores[ex66.LEFT_STAGE].frontier_ids = [start_node_id]
    stores[ex66.RIGHT_STAGE].frontier_ids = [goal_node_id]

    left_plane_hypotheses: list[ex66.TransitionHypothesis] = []
    plane_right_hypotheses: list[ex66.TransitionHypothesis] = []

    candidate_evaluations = 0
    total_rounds = 0
    shared_proposals_processed = 0
    proposals_used_by_multiple_stages = 0
    useful_stage_total = 0
    multi_stage_update_total = 0
    proposal_rounds_with_plane_updates = 0
    proposal_rounds_with_multi_stage_updates = 0
    committed_route_changes_after_first_solution = 0
    alternative_hypothesis_pairs_evaluated = 0

    stage_node_gains = {stage: [] for stage in ex66.STAGES}
    stage_transition_gains = {stage: [] for stage in ex66.STAGES}
    stage_route_gains = {stage: [] for stage in ex66.STAGES}
    best_cost_history: list[float] = []

    best_candidate: ex66.SequentialRouteCandidate | None = None
    first_solution_round: int | None = None
    best_solution_round: int | None = None
    first_committed_entry_round: int | None = None
    first_committed_exit_round: int | None = None
    plane_evidence_before_first_committed_entry = 0
    right_evidence_before_first_committed_exit = 0
    plane_evidence_at_first_solution = 0
    right_evidence_at_first_solution = 0

    guides_task = {
        ex66.LEFT_STAGE: np.asarray(start_q, dtype=float),
        ex66.PLANE_STAGE: 0.5 * (np.asarray(start_q, dtype=float) + np.asarray(goal_q, dtype=float)),
        ex66.RIGHT_STAGE: np.asarray(goal_q, dtype=float),
    }

    mode_counts = {
        "shared_proposal_round": 0,
        "right_goal_bias_updates": 0,
        "route_candidates_evaluated": 0,
        "route_candidates_constraint_certified": 0,
        "route_candidates_execution_certified": 0,
        "route_candidates_rejected_joint_jump": 0,
    }
    if serial_mode:
        mode_counts["serial_round"] = 0

    for round_idx in range(1, ex66.SAFETY_MAX_TOTAL_ROUNDS + 1):
        current_stage_counts = ex66.stage_evidence_counts(stores)
        total_rounds = round_idx
        if ex66.should_stop_exploration(
            first_solution_round=first_solution_round,
            total_rounds=round_idx,
            stage_node_gains=stage_node_gains,
            stage_transition_gains=stage_transition_gains,
            stage_route_gains=stage_route_gains,
            current_stage_counts=current_stage_counts,
        ):
            break

        proposal_count = ex66.effective_proposals_per_round(round_idx, first_solution_round, stores)
        mode_counts["shared_proposal_round"] += 1

        round_node_gain = {stage: 0 for stage in ex66.STAGES}
        round_transition_gain = {stage: 0 for stage in ex66.STAGES}
        route_improved_this_round = 0
        plane_updated_this_round = False
        multi_stage_updated_this_round = False
        active_serial_stage = ex66.greedy_stage_for_serial_round(stores) if serial_mode else None
        if serial_mode:
            mode_counts["serial_round"] += 1
            mode_counts[f"serial_active_{active_serial_stage}"] = mode_counts.get(f"serial_active_{active_serial_stage}", 0) + 1

        guide_joint_pool: list[np.ndarray] = [start_theta.copy(), goal_theta.copy()]
        for stage in ex66.STAGES:
            for node_id in stores[stage].frontier_ids[: min(8, len(stores[stage].frontier_ids))]:
                guide_joint_pool.append(np.asarray(stores[stage].graph.nodes[node_id].q, dtype=float))
        proposals = generate_joint_proposals(
            round_idx=round_idx,
            start_q=start_theta,
            goal_q=goal_theta,
            guides=guide_joint_pool,
            proposal_count=proposal_count,
            joint_lower=joint_lower,
            joint_upper=joint_upper,
        )

        adaptive_budget = ex66.adaptive_stage_update_budget(stores, first_solution_round)

        for proposal in proposals:
            shared_proposals_processed += 1
            stage_candidates: list[tuple[float, str, np.ndarray]] = []
            for stage in ex66.STAGES:
                projection = stores[stage].manifold.project(np.asarray(proposal, dtype=float))
                if not projection.success:
                    if stage == ex66.PLANE_STAGE:
                        mode_counts["plane_projection_failure_count"] = mode_counts.get("plane_projection_failure_count", 0) + 1
                    continue
                if stage == ex66.PLANE_STAGE:
                    mode_counts["plane_projection_success_count"] = mode_counts.get("plane_projection_success_count", 0) + 1
                projected_q = np.asarray(projection.x_projected, dtype=float)
                score = _proposal_stage_utility_joint(
                    stage=stage,
                    projected_q=projected_q,
                    store=stores[stage],
                    guide_point=guides_task[stage],
                    stores=stores,
                    robot=robot,
                )
                if stage == ex66.PLANE_STAGE and len(left_plane_hypotheses) > 0 and len(plane_right_hypotheses) == 0:
                    left_count = len(stores[ex66.LEFT_STAGE].graph.nodes)
                    right_count = len(stores[ex66.RIGHT_STAGE].graph.nodes)
                    plane_count = len(stores[ex66.PLANE_STAGE].graph.nodes)
                    score += 8.0
                    if plane_count < 0.35 * max(1, min(left_count, right_count)):
                        score += 12.0
                    if len(stores[ex66.PLANE_STAGE].graph.edges) < 30:
                        score += 6.0
                if stage == ex66.RIGHT_STAGE and len(left_plane_hypotheses) > 0 and len(plane_right_hypotheses) == 0:
                    score -= 6.0
                stage_candidates.append((score, stage, projected_q))

            if len(left_plane_hypotheses) > 0 and len(plane_right_hypotheses) == 0:
                targeted_plane_qs = _plane_targeted_joint_proposals(
                    plane_store=stores[ex66.PLANE_STAGE],
                    left_plane_hypotheses=left_plane_hypotheses,
                    plane_geom=plane_geom,
                    right_geom=right_geom,
                    robot=robot,
                    collision_fn=collision_fn,
                    limit=4,
                )
                mode_counts["plane_targeted_proposals_generated"] = mode_counts.get("plane_targeted_proposals_generated", 0) + len(targeted_plane_qs)
                for q_target in targeted_plane_qs:
                    stage_candidates.append((1e3, ex66.PLANE_STAGE, np.asarray(q_target, dtype=float)))

            if len(stage_candidates) == 0:
                continue
            stage_candidates.sort(key=lambda item: item[0], reverse=True)
            useful_stages = [stage for _, stage, _ in stage_candidates]
            useful_stage_total += len(useful_stages)
            if len(useful_stages) > 1:
                proposals_used_by_multiple_stages += 1
                if not serial_mode:
                    multi_stage_update_total += len(useful_stages)
                    multi_stage_updated_this_round = True

            active_candidates = stage_candidates
            if serial_mode:
                active_candidates = [candidate for candidate in stage_candidates if candidate[1] == active_serial_stage]
            elif len(left_plane_hypotheses) > 0 and len(plane_right_hypotheses) == 0:
                plane_candidates = [candidate for candidate in stage_candidates if candidate[1] == ex66.PLANE_STAGE]
                non_plane_candidates = [candidate for candidate in stage_candidates if candidate[1] != ex66.PLANE_STAGE]
                right_cap = 0 if len(stores[ex66.RIGHT_STAGE].graph.nodes) > max(80, 2 * len(stores[ex66.PLANE_STAGE].graph.nodes)) else 1
                capped_non_plane: list[tuple[float, str, np.ndarray]] = []
                left_kept = 0
                right_kept = 0
                for candidate in non_plane_candidates:
                    if candidate[1] == ex66.RIGHT_STAGE:
                        if right_kept >= right_cap:
                            continue
                        right_kept += 1
                    elif candidate[1] == ex66.LEFT_STAGE:
                        if left_kept >= 1:
                            continue
                        left_kept += 1
                    capped_non_plane.append(candidate)
                active_candidates = plane_candidates[: max(2, adaptive_budget)] + capped_non_plane
            updates_used = 0
            for _score, stage, projected_q in active_candidates:
                effective_budget = adaptive_budget
                if not serial_mode and len(left_plane_hypotheses) > 0 and len(plane_right_hypotheses) == 0:
                    effective_budget = max(adaptive_budget, 3)
                if not serial_mode and updates_used >= effective_budget:
                    break
                mode_counts[f"{stage}_update_attempts"] = mode_counts.get(f"{stage}_update_attempts", 0) + 1
                node_gain, evals, result, source_node_id, path_node_ids = _update_stage_evidence_from_proposal_joint(
                    store=stores[stage],
                    target_q=projected_q,
                    guide_point=guides_task[stage],
                    robot=robot,
                    collision_fn=collision_fn,
                )
                candidate_evaluations += evals
                round_node_gain[stage] += node_gain
                updates_used += 1
                if node_gain > 0:
                    mode_counts[f"{stage}_update_successes"] = mode_counts.get(f"{stage}_update_successes", 0) + 1
                if stage == ex66.PLANE_STAGE:
                    if result is not None and bool(getattr(result, "success", False)):
                        mode_counts["plane_local_motion_success_count"] = mode_counts.get("plane_local_motion_success_count", 0) + 1
                    elif result is not None:
                        mode_counts["plane_local_motion_failure_count"] = mode_counts.get("plane_local_motion_failure_count", 0) + 1
                if stage == ex66.PLANE_STAGE and (node_gain > 0 or result is not None):
                    plane_updated_this_round = True

                if stage in [ex66.LEFT_STAGE, ex66.PLANE_STAGE]:
                    transition_diag: dict[str, int | float] = {}
                    transition_gain, evals = _add_left_plane_hypotheses_joint(
                        source_stage=stage,
                        source_store=stores[stage],
                        plane_store=stores[ex66.PLANE_STAGE],
                        left_store=stores[ex66.LEFT_STAGE],
                        left_geom=left_geom,
                        plane_geom=plane_geom,
                        result=result,
                        source_node_id=source_node_id,
                        path_node_ids=path_node_ids,
                        guide_point=guides_task[ex66.PLANE_STAGE],
                        hypotheses=left_plane_hypotheses,
                        robot=robot,
                        collision_fn=collision_fn,
                        diagnostics=transition_diag,
                    )
                    _merge_transition_diagnostics(mode_counts, "left_plane_transition", transition_diag)
                    candidate_evaluations += evals
                    mode_counts["transition_attempts_left_plane"] = mode_counts.get("transition_attempts_left_plane", 0) + int(evals)
                    mode_counts["transition_success_left_plane"] = mode_counts.get("transition_success_left_plane", 0) + int(transition_gain)
                    round_transition_gain[ex66.LEFT_STAGE] += transition_gain
                    round_transition_gain[ex66.PLANE_STAGE] += transition_gain
                    if stage == ex66.PLANE_STAGE and transition_gain > 0:
                        plane_updated_this_round = True

                if stage in [ex66.PLANE_STAGE, ex66.RIGHT_STAGE]:
                    transition_diag = {}
                    transition_gain, evals = _add_plane_right_hypotheses_joint(
                        source_stage=stage,
                        source_store=stores[stage],
                        plane_store=stores[ex66.PLANE_STAGE],
                        right_store=stores[ex66.RIGHT_STAGE],
                        plane_geom=plane_geom,
                        right_geom=right_geom,
                        result=result,
                        source_node_id=source_node_id,
                        path_node_ids=path_node_ids,
                        guide_point=guides_task[ex66.RIGHT_STAGE],
                        hypotheses=plane_right_hypotheses,
                        robot=robot,
                        collision_fn=collision_fn,
                        diagnostics=transition_diag,
                    )
                    _merge_transition_diagnostics(mode_counts, "plane_right_transition", transition_diag)
                    candidate_evaluations += evals
                    mode_counts["transition_attempts_plane_right"] = mode_counts.get("transition_attempts_plane_right", 0) + int(evals)
                    mode_counts["transition_success_plane_right"] = mode_counts.get("transition_success_plane_right", 0) + int(transition_gain)
                    round_transition_gain[ex66.PLANE_STAGE] += transition_gain
                    round_transition_gain[ex66.RIGHT_STAGE] += transition_gain
                    if stage == ex66.PLANE_STAGE and transition_gain > 0:
                        plane_updated_this_round = True

        if not serial_mode or active_serial_stage == ex66.LEFT_STAGE:
            left_bridge_gain, left_bridge_evals = _bridge_left_hypotheses_to_start_joint(
                left_store=stores[ex66.LEFT_STAGE],
                start_node_id=start_node_id,
                left_plane_hypotheses=left_plane_hypotheses,
                guide_point=guides_task[ex66.PLANE_STAGE],
                robot=robot,
                collision_fn=collision_fn,
            )
            candidate_evaluations += left_bridge_evals
            round_node_gain[ex66.LEFT_STAGE] += left_bridge_gain

        if not serial_mode or active_serial_stage == ex66.PLANE_STAGE:
            plane_bridge_gain, plane_bridge_evals = _bridge_plane_hypothesis_components_joint(
                plane_store=stores[ex66.PLANE_STAGE],
                left_plane_hypotheses=left_plane_hypotheses,
                plane_right_hypotheses=plane_right_hypotheses,
                robot=robot,
                collision_fn=collision_fn,
            )
            candidate_evaluations += plane_bridge_evals
            round_node_gain[ex66.PLANE_STAGE] += plane_bridge_gain
            if plane_bridge_gain > 0:
                plane_updated_this_round = True

        if len(left_plane_hypotheses) > 0 and len(plane_right_hypotheses) == 0 and (round_idx % 2 == 0 or plane_updated_this_round):
            explicit_gain, explicit_evals = _explicit_plane_right_transition_search(
                plane_store=stores[ex66.PLANE_STAGE],
                right_store=stores[ex66.RIGHT_STAGE],
                plane_geom=plane_geom,
                right_geom=right_geom,
                left_plane_hypotheses=left_plane_hypotheses,
                plane_right_hypotheses=plane_right_hypotheses,
                guide_point=guides_task[ex66.RIGHT_STAGE],
                robot=robot,
                collision_fn=collision_fn,
                max_checks=32,
            )
            candidate_evaluations += explicit_evals
            mode_counts["explicit_plane_right_transition_attempts"] = mode_counts.get("explicit_plane_right_transition_attempts", 0) + int(explicit_evals)
            mode_counts["explicit_plane_right_transition_successes"] = mode_counts.get("explicit_plane_right_transition_successes", 0) + int(explicit_gain)
            round_transition_gain[ex66.PLANE_STAGE] += explicit_gain
            round_transition_gain[ex66.RIGHT_STAGE] += explicit_gain

        if not serial_mode or active_serial_stage == ex66.RIGHT_STAGE:
            right_bridge_gain, right_bridge_evals = _connect_right_hypothesis_to_goal_joint(
                right_store=stores[ex66.RIGHT_STAGE],
                plane_right_hypotheses=plane_right_hypotheses,
                goal_node_id=goal_node_id,
                guide_point=guides_task[ex66.RIGHT_STAGE],
                robot=robot,
                collision_fn=collision_fn,
            )
            candidate_evaluations += right_bridge_evals
            round_node_gain[ex66.RIGHT_STAGE] += right_bridge_gain

        allow_right_goal_bias = bool(
            len(plane_right_hypotheses) > 0
            or len(left_plane_hypotheses) == 0
            or len(stores[ex66.RIGHT_STAGE].graph.nodes) < max(80, 2 * len(stores[ex66.PLANE_STAGE].graph.nodes))
        )
        if allow_right_goal_bias and len(stores[ex66.RIGHT_STAGE].frontier_ids) > 0 and (not serial_mode or active_serial_stage == ex66.RIGHT_STAGE):
            mode_counts["right_goal_bias_updates"] += 1
            for source_node_id in _ranked_stage_sources_joint(stores[ex66.RIGHT_STAGE], guides_task[ex66.RIGHT_STAGE], limit=3, robot=robot):
                ex66.increment_stage_node_expansion(stores[ex66.RIGHT_STAGE], source_node_id)
                source_q = stores[ex66.RIGHT_STAGE].graph.nodes[source_node_id].q
                exact_result = explore_joint_manifold(
                    manifold=stores[ex66.RIGHT_STAGE].manifold,
                    start=source_q,
                    goal=goal_theta,
                    max_step=LOCAL_MAX_JOINT_STEP,
                    local_max_joint_step=LOCAL_MAX_JOINT_STEP,
                    collision_fn=collision_fn,
                )
                _record_local_planner_result(stores[ex66.RIGHT_STAGE], exact_result)
                candidate_evaluations += 1
                if not exact_result.success:
                    if not bool(getattr(exact_result, "joint_continuity_success", True)):
                        stores[ex66.RIGHT_STAGE].stage_edges_rejected_joint_jump += 1
                    continue
                stores[ex66.RIGHT_STAGE].explored_edges = ex66.merge_edges(
                    stores[ex66.RIGHT_STAGE].explored_edges,
                    list(getattr(exact_result, "explored_edges", [])),
                )
                stores[ex66.RIGHT_STAGE].chart_centers = _merge_chart_centers_joint(
                    stores[ex66.RIGHT_STAGE].chart_centers,
                    robot,
                    np.asarray(exact_result.path, dtype=float),
                )
                _, path_nodes, _ = ex66.connect_path_to_stage_graph(
                    store=stores[ex66.RIGHT_STAGE],
                    source_node_id=source_node_id,
                    path=np.asarray(exact_result.path, dtype=float),
                    kind=ex66.RIGHT_MOTION,
                    terminal_node_id=goal_node_id,
                    preserve_dense_path=True,
                    max_joint_step_for_edge=LOCAL_MAX_JOINT_STEP,
                )
                _update_stage_frontier_joint(stores[ex66.RIGHT_STAGE], path_nodes + [goal_node_id], guides_task[ex66.RIGHT_STAGE], robot)
                round_node_gain[ex66.RIGHT_STAGE] += max(0, len(path_nodes) - 1)
                break

        candidate, has_committed_entry, has_committed_exit, pairs_evaluated, route_stats = _extract_committed_route_joint(
            left_store=stores[ex66.LEFT_STAGE],
            plane_store=stores[ex66.PLANE_STAGE],
            right_store=stores[ex66.RIGHT_STAGE],
            start_node_id=start_node_id,
            goal_node_id=goal_node_id,
            left_plane_hypotheses=left_plane_hypotheses,
            plane_right_hypotheses=plane_right_hypotheses,
            robot=robot,
            collision_fn=collision_fn,
        )
        alternative_hypothesis_pairs_evaluated += pairs_evaluated
        for key, value in route_stats.items():
            mode_counts[key] = mode_counts.get(key, 0) + int(value)

        if has_committed_entry and first_committed_entry_round is None:
            first_committed_entry_round = round_idx
            plane_evidence_before_first_committed_entry = len(stores[ex66.PLANE_STAGE].graph.nodes)
        if has_committed_exit and first_committed_exit_round is None:
            first_committed_exit_round = round_idx
            right_evidence_before_first_committed_exit = len(stores[ex66.RIGHT_STAGE].graph.nodes)

        if candidate is not None:
            if first_solution_round is None:
                first_solution_round = round_idx
                plane_evidence_at_first_solution = len(stores[ex66.PLANE_STAGE].graph.nodes)
                right_evidence_at_first_solution = len(stores[ex66.RIGHT_STAGE].graph.nodes)
            if best_candidate is None or _joint_route_candidate_rank_key(candidate) < _joint_route_candidate_rank_key(best_candidate):
                if first_solution_round is not None and best_candidate is not None and round_idx > first_solution_round:
                    committed_route_changes_after_first_solution += 1
                best_candidate = candidate
                best_solution_round = round_idx
                route_improved_this_round = 1

        if plane_updated_this_round:
            proposal_rounds_with_plane_updates += 1
        if multi_stage_updated_this_round:
            proposal_rounds_with_multi_stage_updates += 1

        best_cost_history.append(best_candidate.total_cost if best_candidate is not None else 1e12)
        for stage in ex66.STAGES:
            stage_node_gains[stage].append(int(round_node_gain[stage]))
            stage_transition_gains[stage].append(int(round_transition_gain[stage]))
            stage_route_gains[stage].append(int(route_improved_this_round))

    success = bool(best_candidate is not None and best_candidate.dense_joint_path_execution_certified)
    committed_nodes = {stage: set() for stage in ex66.STAGES}
    raw_path = np.asarray([np.asarray(start_q, dtype=float)], dtype=float)
    display_path = np.asarray([np.asarray(start_q, dtype=float)], dtype=float)
    joint_path = np.zeros((0, 3), dtype=float)
    dense_joint_path = np.zeros((0, 3), dtype=float)
    dense_joint_path_stage_labels: list[str] = []
    dense_joint_path_constraint_residuals = np.zeros(0, dtype=float)
    dense_joint_path_is_certified = False
    dense_joint_path_joint_steps = np.zeros(0, dtype=float)
    dense_joint_path_max_joint_step = 0.0
    dense_joint_path_mean_joint_step = 0.0
    dense_joint_path_worst_joint_step_index = -1
    dense_joint_path_execution_certified = False
    dense_joint_path_constraint_certified = False
    dense_joint_path_joint_continuity_certified = False
    dense_joint_path_message = "joint-space route has sparse nodes only; dense constrained local edges not stored yet"
    selected_left_plane_transition_theta = np.zeros(0, dtype=float)
    selected_plane_right_transition_theta = np.zeros(0, dtype=float)
    selected_left_plane_transition_index = -1
    selected_plane_right_transition_index = -1
    selected_left_plane_stack_residual = float("inf")
    selected_plane_right_stack_residual = float("inf")
    max_transition_stack_residual = float("inf")
    transition_stack_certified = False
    route_cost_raw = 0.0
    route_cost_display = 0.0
    graph_route_edges = 0
    if best_candidate is not None:
        best_candidate, final_replan_stats, final_replan_message = _realize_best_candidate_after_selection(
            best_candidate,
            stores,
            robot,
            collision_fn=collision_fn,
        )
        mode_counts.update(final_replan_stats)
        mode_counts["final_route_realization_message"] = final_replan_message
        committed_nodes = best_candidate.committed_nodes
        raw_path = np.asarray(best_candidate.raw_path, dtype=float)
        display_path = np.asarray(best_candidate.display_path, dtype=float)
        joint_path = np.asarray(best_candidate.joint_path, dtype=float)
        dense_joint_path = np.asarray(best_candidate.dense_joint_path, dtype=float)
        dense_joint_path_stage_labels = list(best_candidate.dense_joint_path_stage_labels)
        dense_joint_path_constraint_residuals = np.asarray(best_candidate.dense_joint_path_constraint_residuals, dtype=float)
        dense_joint_path_is_certified = bool(best_candidate.dense_joint_path_is_certified)
        dense_joint_path_joint_steps = np.asarray(best_candidate.dense_joint_path_joint_steps, dtype=float)
        dense_joint_path_max_joint_step = float(best_candidate.dense_joint_path_max_joint_step)
        dense_joint_path_mean_joint_step = float(best_candidate.dense_joint_path_mean_joint_step)
        dense_joint_path_worst_joint_step_index = int(best_candidate.dense_joint_path_worst_joint_step_index)
        dense_joint_path_execution_certified = bool(best_candidate.dense_joint_path_execution_certified)
        dense_joint_path_constraint_certified = bool(best_candidate.dense_joint_path_constraint_certified)
        dense_joint_path_joint_continuity_certified = bool(best_candidate.dense_joint_path_joint_continuity_certified)
        dense_joint_path_message = str(best_candidate.dense_joint_path_message)
        selected_left_plane_transition_theta = np.asarray(best_candidate.selected_left_plane_transition_theta, dtype=float)
        selected_plane_right_transition_theta = np.asarray(best_candidate.selected_plane_right_transition_theta, dtype=float)
        selected_left_plane_transition_index = int(best_candidate.selected_left_plane_transition_index)
        selected_plane_right_transition_index = int(best_candidate.selected_plane_right_transition_index)
        selected_left_plane_stack_residual = float(best_candidate.selected_left_plane_stack_residual)
        selected_plane_right_stack_residual = float(best_candidate.selected_plane_right_stack_residual)
        max_transition_stack_residual = float(best_candidate.max_transition_stack_residual)
        transition_stack_certified = bool(best_candidate.transition_stack_certified)
        route_cost_raw = ex66.path_cost(raw_path)
        route_cost_display = ex66.path_cost(display_path)
        graph_route_edges = (
            len(best_candidate.left_edge_path)
            + len(best_candidate.plane_edge_path)
            + len(best_candidate.right_edge_path)
        )

    total_evidence_nodes = sum(len(stores[stage].graph.nodes) for stage in ex66.STAGES)
    committed_node_count = sum(len(committed_nodes[stage]) for stage in ex66.STAGES)
    evidence_only_nodes = max(0, total_evidence_nodes - committed_node_count)
    continued_after_first_solution = bool(first_solution_round is not None and total_rounds > first_solution_round)
    plane_evidence_growth_after_first_solution = (
        max(0, len(stores[ex66.PLANE_STAGE].graph.nodes) - plane_evidence_at_first_solution)
        if first_solution_round is not None
        else 0
    )
    right_evidence_growth_after_first_solution = (
        max(0, len(stores[ex66.RIGHT_STAGE].graph.nodes) - right_evidence_at_first_solution)
        if first_solution_round is not None
        else 0
    )
    multi_stage_updates_per_round = multi_stage_update_total / max(total_rounds, 1)
    average_useful_stages_per_proposal = useful_stage_total / max(shared_proposals_processed, 1)
    stage_stagnation_flags = {
        stage: ex66.stage_stagnating(stage_node_gains[stage], stage_transition_gains[stage], stage_route_gains[stage])
        for stage in ex66.STAGES
    }
    saturated_before_solution = bool(not success and total_rounds > 0)
    stagnation_stage = None
    if saturated_before_solution:
        stagnant = [stage for stage in ex66.STAGES if stage_stagnation_flags.get(stage, False)]
        if len(stagnant) == 1:
            stagnation_stage = stagnant[0]
        elif len(stagnant) == len(ex66.STAGES):
            stagnation_stage = "all"
        elif len(stagnant) > 1:
            stagnation_stage = ",".join(stagnant)
        else:
            stagnation_stage = ex66.greedy_stage_for_serial_round(stores) if serial_mode else None

    for stage in ex66.STAGES:
        audit = audit_stage_graph_joint_continuity(stores[stage], stage, max_joint_step=LOCAL_MAX_JOINT_STEP)
        mode_counts.update({key: int(value) if isinstance(value, (int, np.integer)) else value for key, value in audit.items()})
        mode_counts["local_continuation_success_count"] = mode_counts.get("local_continuation_success_count", 0) + int(
            getattr(stores[stage], "local_continuation_success_count", 0)
        )
        mode_counts["local_interpolation_success_count"] = mode_counts.get("local_interpolation_success_count", 0) + int(
            getattr(stores[stage], "local_interpolation_success_count", 0)
        )
        mode_counts["local_planner_failure_count"] = mode_counts.get("local_planner_failure_count", 0) + int(
            getattr(stores[stage], "local_planner_failure_count", 0)
        )
        mode_counts["local_joint_jump_rejections"] = mode_counts.get("local_joint_jump_rejections", 0) + int(
            getattr(stores[stage], "local_joint_jump_rejections", 0)
        )
    mode_counts["plane_frontier_count"] = len(stores[ex66.PLANE_STAGE].frontier_ids)
    mode_counts["joint_local_max_step"] = int(round(float(LOCAL_MAX_JOINT_STEP) * 1_000_000))

    return ex66.FixedPlaneRoute(
        success=success,
        message=(
            (
                "Serial stage evidence accumulated enough certified sequential structure to extract a fixed left-plane-right robot joint-space route."
                if serial_mode
                else "Parallel stage evidence accumulated enough certified sequential structure to extract a fixed left-plane-right robot joint-space route."
            )
            if success
            else (
                "Robot joint-space serial exploration accumulated evidence across the fixed left / plane / right manifolds, but no certified sequential route was extracted before exploration saturated."
                if serial_mode
                else "Robot joint-space evidence accumulated across the fixed left / plane / right manifolds, but no certified sequential route was extracted before exploration saturated."
            )
        ),
        total_rounds=total_rounds,
        candidate_evaluations=candidate_evaluations,
        left_evidence_nodes=len(stores[ex66.LEFT_STAGE].graph.nodes),
        plane_evidence_nodes=len(stores[ex66.PLANE_STAGE].graph.nodes),
        right_evidence_nodes=len(stores[ex66.RIGHT_STAGE].graph.nodes),
        committed_nodes=committed_node_count,
        evidence_only_nodes=evidence_only_nodes,
        shared_proposals_processed=shared_proposals_processed,
        proposals_used_by_multiple_stages=proposals_used_by_multiple_stages,
        plane_evidence_before_first_committed_entry=plane_evidence_before_first_committed_entry,
        right_evidence_before_first_committed_exit=right_evidence_before_first_committed_exit,
        transition_hypotheses_left_plane=len(left_plane_hypotheses),
        transition_hypotheses_plane_right=len(plane_right_hypotheses),
        first_solution_round=first_solution_round,
        best_solution_round=best_solution_round,
        continued_after_first_solution=continued_after_first_solution,
        path=np.asarray(display_path, dtype=float),
        raw_path=np.asarray(raw_path, dtype=float),
        certified_path_points=int(len(raw_path)),
        display_path_points=int(len(display_path)),
        route_cost_raw=route_cost_raw,
        route_cost_display=route_cost_display,
        graph_route_edges=graph_route_edges,
        stage_evidence_points={stage: _stage_evidence_points_joint(stores[stage], robot) for stage in ex66.STAGES},
        stage_evidence_edges={stage: _task_edge_segments(stores[stage], robot) for stage in ex66.STAGES},
        stage_frontier_points={stage: _task_frontier_points(stores[stage], robot) for stage in ex66.STAGES},
        stage_chart_centers={stage: np.asarray(stores[stage].chart_centers, dtype=float) for stage in ex66.STAGES},
        stage_frontier_counts={stage: len(stores[stage].frontier_ids) for stage in ex66.STAGES},
        stage_stagnation_flags=stage_stagnation_flags,
        recent_graph_node_gain=sum(ex66.stage_recent_sum(stage_node_gains[stage]) for stage in ex66.STAGES),
        recent_transition_gain=sum(ex66.stage_recent_sum(stage_transition_gains[stage]) for stage in ex66.STAGES),
        recent_route_improvement_gain=ex66.recent_route_improvement(best_cost_history),
        plane_evidence_growth_after_first_solution=plane_evidence_growth_after_first_solution,
        right_evidence_growth_after_first_solution=right_evidence_growth_after_first_solution,
        multi_stage_updates_per_round=float(multi_stage_updates_per_round),
        average_useful_stages_per_proposal=float(average_useful_stages_per_proposal),
        proposal_rounds_with_plane_updates=proposal_rounds_with_plane_updates,
        proposal_rounds_with_multi_stage_updates=proposal_rounds_with_multi_stage_updates,
        committed_route_changes_after_first_solution=committed_route_changes_after_first_solution,
        alternative_hypothesis_pairs_evaluated=alternative_hypothesis_pairs_evaluated,
        left_plane_hypothesis_points=ex66.deduplicate_points([hyp.q for hyp in left_plane_hypotheses], tol=ex66.TRANSITION_DEDUP_TOL),
        plane_right_hypothesis_points=ex66.deduplicate_points([hyp.q for hyp in plane_right_hypotheses], tol=ex66.TRANSITION_DEDUP_TOL),
        committed_stage_nodes={
            ex66.LEFT_STAGE: np.asarray([_task_point(robot, stores[ex66.LEFT_STAGE].graph.nodes[node_id].q) for node_id in sorted(committed_nodes[ex66.LEFT_STAGE])], dtype=float)
            if len(committed_nodes[ex66.LEFT_STAGE]) > 0
            else np.zeros((0, 3), dtype=float),
            ex66.PLANE_STAGE: np.asarray([_task_point(robot, stores[ex66.PLANE_STAGE].graph.nodes[node_id].q) for node_id in sorted(committed_nodes[ex66.PLANE_STAGE])], dtype=float)
            if len(committed_nodes[ex66.PLANE_STAGE]) > 0
            else np.zeros((0, 3), dtype=float),
            ex66.RIGHT_STAGE: np.asarray([_task_point(robot, stores[ex66.RIGHT_STAGE].graph.nodes[node_id].q) for node_id in sorted(committed_nodes[ex66.RIGHT_STAGE])], dtype=float)
            if len(committed_nodes[ex66.RIGHT_STAGE]) > 0
            else np.zeros((0, 3), dtype=float),
        },
        mode_counts=mode_counts,
        serial_mode=serial_mode,
        saturated_before_solution=saturated_before_solution,
        stagnation_stage=stagnation_stage,
        obstacles=active_obstacles,
        joint_path=np.asarray(joint_path, dtype=float),
        dense_joint_path=np.asarray(dense_joint_path, dtype=float),
        dense_joint_path_stage_labels=list(dense_joint_path_stage_labels),
        dense_joint_path_constraint_residuals=np.asarray(dense_joint_path_constraint_residuals, dtype=float),
        dense_joint_path_is_certified=bool(dense_joint_path_is_certified),
        dense_joint_path_joint_steps=np.asarray(dense_joint_path_joint_steps, dtype=float),
        dense_joint_path_max_joint_step=float(dense_joint_path_max_joint_step),
        dense_joint_path_mean_joint_step=float(dense_joint_path_mean_joint_step),
        dense_joint_path_worst_joint_step_index=int(dense_joint_path_worst_joint_step_index),
        dense_joint_path_execution_certified=bool(dense_joint_path_execution_certified),
        dense_joint_path_constraint_certified=bool(dense_joint_path_constraint_certified),
        dense_joint_path_joint_continuity_certified=bool(dense_joint_path_joint_continuity_certified),
        dense_joint_path_message=str(dense_joint_path_message),
        selected_left_plane_transition_theta=np.asarray(selected_left_plane_transition_theta, dtype=float),
        selected_plane_right_transition_theta=np.asarray(selected_plane_right_transition_theta, dtype=float),
        selected_left_plane_transition_index=int(selected_left_plane_transition_index),
        selected_plane_right_transition_index=int(selected_plane_right_transition_index),
        selected_left_plane_stack_residual=float(selected_left_plane_stack_residual),
        selected_plane_right_stack_residual=float(selected_plane_right_stack_residual),
        max_transition_stack_residual=float(max_transition_stack_residual),
        transition_stack_certified=bool(transition_stack_certified),
    )
