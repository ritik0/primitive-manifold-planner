from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

try:
    from scipy.optimize import least_squares
except Exception:
    least_squares = None


def wrap_joint_angles(joint_angles: np.ndarray) -> np.ndarray:
    """Wrap theta values into the conventional [-pi, pi) joint range."""

    arr = np.asarray(joint_angles, dtype=float)
    return (arr + np.pi) % (2.0 * np.pi) - np.pi


def end_effector_point(robot: Any, q: np.ndarray) -> np.ndarray:
    """Return FK(theta) for the end-effector, i.e. robot points[-1]."""

    return np.asarray(robot.forward_kinematics_3d(np.asarray(q, dtype=float))[-1], dtype=float)


def joint_path_to_task_path(robot: Any, path: np.ndarray) -> np.ndarray:
    """Convert a dense theta path into its FK trace for visualization/debugging."""

    arr = np.asarray(path, dtype=float)
    if len(arr) == 0:
        return np.zeros((0, 3), dtype=float)
    return np.asarray([end_effector_point(robot, q) for q in arr], dtype=float)


def inverse_kinematics_start(
    robot: Any,
    task_point: np.ndarray,
    warm_start: np.ndarray | None = None,
    joint_lower: np.ndarray | None = None,
    joint_upper: np.ndarray | None = None,
    tol: float = 8e-2,
) -> np.ndarray | None:
    """Find a starting theta whose end-effector is near one task point."""

    target = np.asarray(task_point, dtype=float).reshape(3)
    lower = np.asarray(joint_lower, dtype=float).reshape(3) if joint_lower is not None else -np.pi * np.ones(3, dtype=float)
    upper = np.asarray(joint_upper, dtype=float).reshape(3) if joint_upper is not None else np.pi * np.ones(3, dtype=float)

    if warm_start is None:
        rel = target - np.asarray(robot.base_world, dtype=float)
        yaw_guess = math.atan2(float(rel[1]), float(rel[0]))
        radial_guess = float(np.linalg.norm(rel[:2]))
        pitch_guess = math.atan2(float(rel[2]), max(radial_guess, 1e-6))
        warm = np.asarray([yaw_guess, pitch_guess, -0.65], dtype=float)
    else:
        warm = np.asarray(warm_start, dtype=float).reshape(3)
    warm = np.clip(wrap_joint_angles(warm), lower, upper)

    if least_squares is not None:

        def residual(theta: np.ndarray) -> np.ndarray:
            ee = end_effector_point(robot, theta)
            reg = 0.015 * wrap_joint_angles(theta - warm)
            return np.concatenate([ee - target, reg], dtype=float)

        result = least_squares(
            residual,
            warm,
            bounds=(lower, upper),
            max_nfev=120,
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
        )
        q = wrap_joint_angles(np.asarray(result.x, dtype=float))
        if float(np.linalg.norm(end_effector_point(robot, q) - target)) <= tol:
            return q
        return None

    theta = warm.copy()
    for _ in range(120):
        current = end_effector_point(robot, theta)
        error = target - current
        if float(np.linalg.norm(error)) <= tol:
            return wrap_joint_angles(theta)
        jac = np.zeros((3, 3), dtype=float)
        eps = 1e-4
        for col in range(3):
            perturbed = theta.copy()
            perturbed[col] += eps
            jac[:, col] = (end_effector_point(robot, perturbed) - current) / eps
        damp = 5e-2
        delta = jac.T @ np.linalg.solve(jac @ jac.T + damp * np.eye(3, dtype=float), error)
        theta = np.clip(wrap_joint_angles(theta + 0.6 * delta), lower, upper)
    return None


@dataclass
class JointExploreResult:
    """Result of local joint-manifold exploration or interpolation."""

    success: bool
    path: np.ndarray
    explored_edges: list[tuple[np.ndarray, np.ndarray]]
    iterations: int
    message: str
    joint_steps: np.ndarray | None = None
    max_joint_step: float = 0.0
    mean_joint_step: float = 0.0
    worst_joint_step_index: int = -1
    joint_continuity_success: bool = True
    planner_method: str = "unknown"


def joint_step_statistics(joint_path: np.ndarray) -> tuple[np.ndarray, float, float, int]:
    """Measure wrapped theta-step continuity along a joint path."""

    q_path = np.asarray(joint_path, dtype=float)
    if len(q_path) < 2:
        return np.zeros(0, dtype=float), 0.0, 0.0, -1
    wrapped = wrap_joint_angles(np.diff(q_path, axis=0))
    steps = np.linalg.norm(wrapped, axis=1)
    worst_idx = int(np.argmax(steps)) if len(steps) > 0 else -1
    return (
        np.asarray(steps, dtype=float),
        float(np.max(steps)) if len(steps) > 0 else 0.0,
        float(np.mean(steps)) if len(steps) > 0 else 0.0,
        worst_idx,
    )


def numerical_robot_jacobian(robot: Any, q: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Finite-difference Jacobian of the end-effector FK trace."""

    theta = np.asarray(q, dtype=float).reshape(3)
    base = end_effector_point(robot, theta)
    jac = np.zeros((3, 3), dtype=float)
    for idx in range(3):
        perturbed = theta.copy()
        perturbed[idx] += float(eps)
        jac[:, idx] = (end_effector_point(robot, perturbed) - base) / float(eps)
    return jac


def generate_joint_proposals(
    round_idx: int,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    guides: list[np.ndarray],
    proposal_count: int,
    joint_lower: np.ndarray | None = None,
    joint_upper: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Generate theta proposals biased by start/goal and current guides."""

    lower = np.asarray(joint_lower, dtype=float).reshape(3) if joint_lower is not None else -np.pi * np.ones(3, dtype=float)
    upper = np.asarray(joint_upper, dtype=float).reshape(3) if joint_upper is not None else np.pi * np.ones(3, dtype=float)
    midpoint = 0.5 * (np.asarray(start_q, dtype=float) + np.asarray(goal_q, dtype=float))
    proposals: list[np.ndarray] = []
    for idx in range(int(proposal_count)):
        selector = (round_idx + idx) % 5
        if selector == 0:
            # Uniform proposals keep broad C-space exploration alive.
            q = np.random.uniform(lower, upper)
        elif selector == 1:
            # Midpoint proposals search the likely corridor between start and goal.
            q = midpoint + np.random.normal(scale=np.array([0.55, 0.45, 0.45], dtype=float), size=3)
        elif selector == 2:
            q = 0.35 * np.asarray(start_q, dtype=float) + 0.65 * np.asarray(goal_q, dtype=float)
            q = q + np.random.normal(scale=np.array([0.40, 0.35, 0.35], dtype=float), size=3)
        elif selector == 3 and len(guides) > 0:
            # Guide-biased proposals revisit frontier/transition neighborhoods.
            guide = np.asarray(guides[np.random.randint(len(guides))], dtype=float)
            q = guide + np.random.normal(scale=np.array([0.28, 0.25, 0.25], dtype=float), size=3)
        else:
            blend = float(np.random.uniform(0.2, 0.8))
            q = blend * np.asarray(start_q, dtype=float) + (1.0 - blend) * np.asarray(goal_q, dtype=float)
            q = q + np.random.normal(scale=np.array([0.35, 0.28, 0.28], dtype=float), size=3)
        proposals.append(np.asarray(np.clip(wrap_joint_angles(q), lower, upper), dtype=float))
    return proposals


def _joint_result(
    *,
    success: bool,
    path: list[np.ndarray] | np.ndarray,
    explored_edges: list[tuple[np.ndarray, np.ndarray]],
    iterations: int,
    message: str,
    planner_method: str,
    local_limit: float,
) -> JointExploreResult:
    path_arr = np.asarray(path, dtype=float)
    joint_steps, max_seen, mean_seen, worst_idx = joint_step_statistics(path_arr)
    joint_ok = bool(max_seen <= float(local_limit) + 1e-9)
    return JointExploreResult(
        success=bool(success and joint_ok),
        path=path_arr,
        explored_edges=explored_edges,
        iterations=int(iterations),
        message=str(message if success and joint_ok else f"{message}; max_step={max_seen:.4g}, limit={local_limit:.4g}"),
        joint_steps=joint_steps,
        max_joint_step=max_seen,
        mean_joint_step=mean_seen,
        worst_joint_step_index=worst_idx,
        joint_continuity_success=joint_ok,
        planner_method=str(planner_method),
    )


def _collision_free_edge(current: np.ndarray, next_q: np.ndarray, collision_fn) -> bool:
    if collision_fn is None:
        return True
    edge_delta = wrap_joint_angles(np.asarray(next_q, dtype=float) - np.asarray(current, dtype=float))
    for beta in np.linspace(0.0, 1.0, 5):
        q_check = wrap_joint_angles(np.asarray(current, dtype=float) + float(beta) * edge_delta)
        if bool(collision_fn(q_check)):
            return False
    return True


def _explore_joint_manifold_interpolation(
    manifold: Any,
    start: np.ndarray,
    goal: np.ndarray,
    max_step: float = 0.08,
    projection_tol: float = 1e-6,
    collision_fn=None,
    local_max_joint_step: float | None = None,
) -> JointExploreResult:
    """Follow a straight theta guess while projecting each waypoint to the manifold."""

    local_limit = float(local_max_joint_step if local_max_joint_step is not None else max_step)
    q_start = np.asarray(start, dtype=float).reshape(3)
    q_goal = np.asarray(goal, dtype=float).reshape(3)
    delta = wrap_joint_angles(q_goal - q_start)
    distance = float(np.linalg.norm(delta))
    steps = max(2, int(np.ceil(distance / max(max_step, 1e-6))))

    path: list[np.ndarray] = [q_start.copy()]
    explored_edges: list[tuple[np.ndarray, np.ndarray]] = []
    current = q_start.copy()
    if collision_fn is not None and bool(collision_fn(current)):
        return JointExploreResult(
            success=False,
            path=np.asarray(path, dtype=float),
            explored_edges=explored_edges,
            iterations=0,
            message="Start joint state is in collision.",
            joint_continuity_success=True,
            planner_method="interpolation",
        )
    for idx in range(1, steps + 1):
        alpha = float(idx / steps)
        guess = wrap_joint_angles(q_start + alpha * delta)
        # Projection makes the interpolated theta lie on the active manifold.
        projection = manifold.project(guess, tol=projection_tol, max_iters=60)
        if not projection.success:
            return JointExploreResult(
                success=False,
                path=np.asarray(path, dtype=float),
                explored_edges=explored_edges,
                iterations=idx,
                message="Projection failed while following the straight-line joint-space guess.",
                planner_method="interpolation",
            )
        next_q = np.asarray(projection.x_projected, dtype=float)
        if hasattr(manifold, "within_bounds") and not bool(manifold.within_bounds(next_q)):
            return JointExploreResult(
                success=False,
                path=np.asarray(path, dtype=float),
                explored_edges=explored_edges,
                iterations=idx,
                message="Projected joint state violated joint bounds.",
                planner_method="interpolation",
            )
        if not _collision_free_edge(current, next_q, collision_fn):
            return JointExploreResult(
                success=False,
                path=np.asarray(path, dtype=float),
                explored_edges=explored_edges,
                iterations=idx,
                message="Projected joint path collided with an obstacle.",
                planner_method="interpolation",
            )
        local_step = float(np.linalg.norm(wrap_joint_angles(next_q - current)))
        if local_step > local_limit + 1e-9:
            candidate_path = np.asarray([*path, next_q.copy()], dtype=float)
            steps_arr, max_seen, mean_seen, worst_idx = joint_step_statistics(candidate_path)
            return JointExploreResult(
                success=False,
                path=np.asarray(path, dtype=float),
                explored_edges=explored_edges,
                iterations=idx,
                message=(
                    "Rejected discontinuous constrained joint interpolation: "
                    f"local_step={local_step:.4g}, limit={local_limit:.4g}, "
                    f"worst_step_index={worst_idx}"
                ),
                joint_steps=steps_arr,
                max_joint_step=max_seen,
                mean_joint_step=mean_seen,
                worst_joint_step_index=worst_idx,
                joint_continuity_success=False,
                planner_method="interpolation",
            )
        explored_edges.append((current.copy(), next_q.copy()))
        path.append(next_q.copy())
        current = next_q

    path_arr = np.asarray(path, dtype=float)
    joint_steps, max_seen, mean_seen, worst_idx = joint_step_statistics(path_arr)
    joint_ok = bool(max_seen <= local_limit + 1e-9)
    return _joint_result(
        success=joint_ok,
        path=path_arr,
        explored_edges=explored_edges,
        iterations=steps,
        message=(
            "interpolation fallback succeeded"
            if joint_ok
            else f"Rejected discontinuous joint path: max_step={max_seen:.4g}, limit={local_limit:.4g}"
        ),
        planner_method="interpolation",
        local_limit=local_limit,
    )


def explore_joint_manifold_continuation(
    manifold: Any,
    start: np.ndarray,
    goal: np.ndarray,
    max_step: float = 0.08,
    projection_tol: float = 1e-6,
    collision_fn=None,
    local_max_joint_step: float | None = None,
    max_iters: int = 90,
    damping: float = 0.05,
) -> JointExploreResult:
    """Grow a local theta path by FK-guided projected continuation."""

    local_limit = float(local_max_joint_step if local_max_joint_step is not None else max_step)
    q_start = np.asarray(start, dtype=float).reshape(3)
    q_goal = np.asarray(goal, dtype=float).reshape(3)
    robot = getattr(manifold, "robot", None)
    if robot is None:
        return _joint_result(
            success=False,
            path=[q_start],
            explored_edges=[],
            iterations=0,
            message="continuation unavailable: manifold has no robot reference",
            planner_method="continuation",
            local_limit=local_limit,
        )
    if collision_fn is not None and bool(collision_fn(q_start)):
        return _joint_result(
            success=False,
            path=[q_start],
            explored_edges=[],
            iterations=0,
            message="Start joint state is in collision.",
            planner_method="continuation",
            local_limit=local_limit,
        )

    path: list[np.ndarray] = [q_start.copy()]
    explored_edges: list[tuple[np.ndarray, np.ndarray]] = []
    current = q_start.copy()
    goal_ee = end_effector_point(robot, q_goal)
    best_task_distance = float(np.linalg.norm(goal_ee - end_effector_point(robot, current)))
    stagnant_steps = 0

    for iteration in range(1, int(max_iters) + 1):
        current_ee = end_effector_point(robot, current)
        to_goal = goal_ee - current_ee
        task_distance = float(np.linalg.norm(to_goal))
        joint_delta_to_goal = wrap_joint_angles(q_goal - current)
        joint_distance = float(np.linalg.norm(joint_delta_to_goal))

        if joint_distance <= local_limit + 1e-9:
            projection = manifold.project(q_goal, tol=projection_tol, max_iters=80)
            if projection.success:
                next_q = np.asarray(projection.x_projected, dtype=float)
                local_step = float(np.linalg.norm(wrap_joint_angles(next_q - current)))
                if (
                    local_step <= local_limit + 1e-9
                    and bool(manifold.within_bounds(next_q))
                    and _collision_free_edge(current, next_q, collision_fn)
                ):
                    explored_edges.append((current.copy(), next_q.copy()))
                    path.append(next_q.copy())
                    return _joint_result(
                        success=True,
                        path=path,
                        explored_edges=explored_edges,
                        iterations=iteration,
                        message="continuation succeeded",
                        planner_method="continuation",
                        local_limit=local_limit,
                    )

        if task_distance <= 3e-2 and joint_distance <= 2.5 * local_limit:
            return _joint_result(
                success=True,
                path=path,
                explored_edges=explored_edges,
                iterations=iteration,
                message="continuation succeeded near task goal",
                planner_method="continuation",
                local_limit=local_limit,
            )

        direction = to_goal / max(task_distance, 1e-9)
        accepted = False
        step_scale = local_limit
        for _retry in range(8):
            desired_task_delta = direction * min(0.075, max(0.01, task_distance), step_scale)
            jac = numerical_robot_jacobian(robot, current)
            system = jac @ jac.T + float(damping) ** 2 * np.eye(3, dtype=float)
            # Pull a small task-space FK step back to theta, then reproject to the manifold.
            dq = jac.T @ np.linalg.solve(system, desired_task_delta)
            dq_norm = float(np.linalg.norm(dq))
            if dq_norm > step_scale:
                dq = dq * (step_scale / max(dq_norm, 1e-9))
            guess = wrap_joint_angles(current + dq)
            projection = manifold.project(guess, tol=projection_tol, max_iters=80)
            if not projection.success:
                step_scale *= 0.5
                continue
            next_q = np.asarray(projection.x_projected, dtype=float)
            if not bool(manifold.within_bounds(next_q)):
                step_scale *= 0.5
                continue
            local_step = float(np.linalg.norm(wrap_joint_angles(next_q - current)))
            if local_step > local_limit + 1e-9 or local_step <= 1e-7:
                step_scale *= 0.5
                continue
            if not _collision_free_edge(current, next_q, collision_fn):
                step_scale *= 0.5
                continue
            next_task_distance = float(np.linalg.norm(goal_ee - end_effector_point(robot, next_q)))
            progress = task_distance - next_task_distance
            if progress < -1e-3 and next_task_distance > best_task_distance + 1e-3:
                step_scale *= 0.5
                continue
            explored_edges.append((current.copy(), next_q.copy()))
            path.append(next_q.copy())
            current = next_q
            best_task_distance = min(best_task_distance, next_task_distance)
            stagnant_steps = 0 if progress > 1e-3 else stagnant_steps + 1
            accepted = True
            break
        if not accepted:
            return _joint_result(
                success=False,
                path=path,
                explored_edges=explored_edges,
                iterations=iteration,
                message="continuation failed: no projected progress step accepted",
                planner_method="continuation",
                local_limit=local_limit,
            )
        if stagnant_steps >= 10:
            return _joint_result(
                success=False,
                path=path,
                explored_edges=explored_edges,
                iterations=iteration,
                message="continuation failed: stalled without task progress",
                planner_method="continuation",
                local_limit=local_limit,
            )

    return _joint_result(
        success=False,
        path=path,
        explored_edges=explored_edges,
        iterations=max_iters,
        message="continuation failed: iteration budget exhausted",
        planner_method="continuation",
        local_limit=local_limit,
    )


def explore_joint_manifold(
    manifold: Any,
    start: np.ndarray,
    goal: np.ndarray,
    max_step: float = 0.08,
    projection_tol: float = 1e-6,
    collision_fn=None,
    local_max_joint_step: float | None = None,
) -> JointExploreResult:
    """Try FK-guided continuation, then a projected interpolation fallback."""

    continuation = explore_joint_manifold_continuation(
        manifold=manifold,
        start=start,
        goal=goal,
        max_step=max_step,
        projection_tol=projection_tol,
        collision_fn=collision_fn,
        local_max_joint_step=local_max_joint_step,
    )
    if continuation.success:
        return continuation
    fallback = _explore_joint_manifold_interpolation(
        manifold=manifold,
        start=start,
        goal=goal,
        max_step=max_step,
        projection_tol=projection_tol,
        collision_fn=collision_fn,
        local_max_joint_step=local_max_joint_step,
    )
    if fallback.success:
        fallback.message = "interpolation fallback succeeded after continuation failed: " + continuation.message
        return fallback
    fallback.message = "both local planners failed: " + continuation.message + " | " + fallback.message
    fallback.planner_method = "failed"
    return fallback


def detect_transitions_jointspace(
    robot: Any,
    current_manifold: Any,
    target_manifold: Any,
    path_configs: np.ndarray,
    projection_tol: float = 1e-6,
    task_tol: float = 1.2e-1,
    transition_joint_tol: float = 0.40,
    collision_fn=None,
    diagnostics: dict[str, float | int] | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Detect source-to-target transition candidates along a theta path.

    Each returned tuple stores the source theta, projected target theta, and
    FK transition point used for visualization/debugging.
    """

    path = np.asarray(path_configs, dtype=float)
    if len(path) == 0:
        return []

    sampled: list[np.ndarray] = [np.asarray(q, dtype=float) for q in path]
    for idx in range(len(path) - 1):
        q0 = np.asarray(path[idx], dtype=float)
        delta = wrap_joint_angles(np.asarray(path[idx + 1], dtype=float) - q0)
        for beta in np.linspace(0.1, 0.9, 5):
            sampled.append(wrap_joint_angles(q0 + float(beta) * delta))

    hits: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    seen_task: list[np.ndarray] = []

    def inc(key: str, amount: int = 1) -> None:
        if diagnostics is not None:
            diagnostics[key] = int(diagnostics.get(key, 0)) + int(amount)

    def stat(prefix: str, value: float) -> None:
        if diagnostics is None:
            return
        value = float(value)
        diagnostics[f"{prefix}_sum"] = float(diagnostics.get(f"{prefix}_sum", 0.0)) + value
        diagnostics[f"{prefix}_count"] = int(diagnostics.get(f"{prefix}_count", 0)) + 1
        diagnostics[f"{prefix}_min"] = min(float(diagnostics.get(f"{prefix}_min", value)), value)
        diagnostics[f"{prefix}_max"] = max(float(diagnostics.get(f"{prefix}_max", value)), value)

    for q in sampled:
        inc("samples_total")
        if not bool(current_manifold.is_valid(q, tol=1e-4)):
            inc("current_manifold_invalid")
            continue
        if collision_fn is not None and bool(collision_fn(q)):
            inc("current_collision")
            continue
        target_residual_before = float(np.linalg.norm(target_manifold.residual(q))) if hasattr(target_manifold, "residual") else float("inf")
        stat("target_residual_before_projection", target_residual_before)
        # Project onto the neighboring manifold to test for a shared transition.
        projection = target_manifold.project(q, tol=projection_tol, max_iters=60)
        if not projection.success:
            inc("target_projection_failed")
            continue
        target_q = np.asarray(projection.x_projected, dtype=float)
        target_residual_after = float(np.linalg.norm(target_manifold.residual(target_q))) if hasattr(target_manifold, "residual") else float("inf")
        stat("projected_target_residual_after_projection", target_residual_after)
        if collision_fn is not None and bool(collision_fn(target_q)):
            inc("target_collision")
            continue
        joint_distance = float(np.linalg.norm(wrap_joint_angles(target_q - q)))
        stat("joint_distance_source_to_target", joint_distance)
        if joint_distance > float(transition_joint_tol):
            inc("joint_distance_too_large")
            continue
        ee_current = end_effector_point(robot, q)
        ee_target = end_effector_point(robot, target_q)
        task_distance = float(np.linalg.norm(ee_current - ee_target))
        stat("target_projection_task_distance", task_distance)
        if task_distance > task_tol:
            inc("task_distance_too_large")
            continue
        if any(float(np.linalg.norm(ee_target - prev)) <= max(task_tol * 0.75, 2e-2) for prev in seen_task):
            # Deduplicate by FK transition point so nearby theta variants do not flood the list.
            inc("duplicate_rejected")
            continue
        seen_task.append(ee_target.copy())
        hits.append((np.asarray(q, dtype=float), target_q, ee_target.copy()))
        inc("accepted_hits")
    return hits
