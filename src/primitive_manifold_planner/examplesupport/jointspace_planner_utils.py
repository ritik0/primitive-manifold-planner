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
    arr = np.asarray(joint_angles, dtype=float)
    return (arr + np.pi) % (2.0 * np.pi) - np.pi


def end_effector_point(robot: Any, q: np.ndarray) -> np.ndarray:
    return np.asarray(robot.forward_kinematics_3d(np.asarray(q, dtype=float))[-1], dtype=float)


def joint_path_to_task_path(robot: Any, path: np.ndarray) -> np.ndarray:
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
    success: bool
    path: np.ndarray
    explored_edges: list[tuple[np.ndarray, np.ndarray]]
    iterations: int
    message: str


def generate_joint_proposals(
    round_idx: int,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    guides: list[np.ndarray],
    proposal_count: int,
    joint_lower: np.ndarray | None = None,
    joint_upper: np.ndarray | None = None,
) -> list[np.ndarray]:
    lower = np.asarray(joint_lower, dtype=float).reshape(3) if joint_lower is not None else -np.pi * np.ones(3, dtype=float)
    upper = np.asarray(joint_upper, dtype=float).reshape(3) if joint_upper is not None else np.pi * np.ones(3, dtype=float)
    midpoint = 0.5 * (np.asarray(start_q, dtype=float) + np.asarray(goal_q, dtype=float))
    proposals: list[np.ndarray] = []
    for idx in range(int(proposal_count)):
        selector = (round_idx + idx) % 5
        if selector == 0:
            q = np.random.uniform(lower, upper)
        elif selector == 1:
            q = midpoint + np.random.normal(scale=np.array([0.55, 0.45, 0.45], dtype=float), size=3)
        elif selector == 2:
            q = 0.35 * np.asarray(start_q, dtype=float) + 0.65 * np.asarray(goal_q, dtype=float)
            q = q + np.random.normal(scale=np.array([0.40, 0.35, 0.35], dtype=float), size=3)
        elif selector == 3 and len(guides) > 0:
            guide = np.asarray(guides[np.random.randint(len(guides))], dtype=float)
            q = guide + np.random.normal(scale=np.array([0.28, 0.25, 0.25], dtype=float), size=3)
        else:
            blend = float(np.random.uniform(0.2, 0.8))
            q = blend * np.asarray(start_q, dtype=float) + (1.0 - blend) * np.asarray(goal_q, dtype=float)
            q = q + np.random.normal(scale=np.array([0.35, 0.28, 0.28], dtype=float), size=3)
        proposals.append(np.asarray(np.clip(wrap_joint_angles(q), lower, upper), dtype=float))
    return proposals


def explore_joint_manifold(
    manifold: Any,
    start: np.ndarray,
    goal: np.ndarray,
    max_step: float = 0.18,
    projection_tol: float = 1e-6,
    collision_fn=None,
) -> JointExploreResult:
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
        )
    for idx in range(1, steps + 1):
        alpha = float(idx / steps)
        guess = wrap_joint_angles(q_start + alpha * delta)
        projection = manifold.project(guess, tol=projection_tol, max_iters=60)
        if not projection.success:
            return JointExploreResult(
                success=False,
                path=np.asarray(path, dtype=float),
                explored_edges=explored_edges,
                iterations=idx,
                message="Projection failed while following the straight-line joint-space guess.",
            )
        next_q = np.asarray(projection.x_projected, dtype=float)
        if hasattr(manifold, "within_bounds") and not bool(manifold.within_bounds(next_q)):
            return JointExploreResult(
                success=False,
                path=np.asarray(path, dtype=float),
                explored_edges=explored_edges,
                iterations=idx,
                message="Projected joint state violated joint bounds.",
            )
        if collision_fn is not None:
            edge_delta = wrap_joint_angles(next_q - current)
            for beta in np.linspace(0.0, 1.0, 5):
                q_check = wrap_joint_angles(current + float(beta) * edge_delta)
                if bool(collision_fn(q_check)):
                    return JointExploreResult(
                        success=False,
                        path=np.asarray(path, dtype=float),
                        explored_edges=explored_edges,
                        iterations=idx,
                        message="Projected joint path collided with an obstacle.",
                    )
        explored_edges.append((current.copy(), next_q.copy()))
        path.append(next_q.copy())
        current = next_q

    return JointExploreResult(
        success=True,
        path=np.asarray(path, dtype=float),
        explored_edges=explored_edges,
        iterations=steps,
        message="Joint-space constrained interpolation succeeded.",
    )


def detect_transitions_jointspace(
    robot: Any,
    current_manifold: Any,
    target_manifold: Any,
    path_configs: np.ndarray,
    projection_tol: float = 1e-6,
    task_tol: float = 5e-2,
    collision_fn=None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    path = np.asarray(path_configs, dtype=float)
    if len(path) == 0:
        return []

    sampled: list[np.ndarray] = [np.asarray(q, dtype=float) for q in path]
    for idx in range(len(path) - 1):
        q_mid = wrap_joint_angles(0.5 * (np.asarray(path[idx], dtype=float) + np.asarray(path[idx + 1], dtype=float)))
        sampled.append(q_mid)

    hits: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    seen_task: list[np.ndarray] = []
    for q in sampled:
        if not bool(current_manifold.is_valid(q, tol=1e-4)):
            continue
        if collision_fn is not None and bool(collision_fn(q)):
            continue
        projection = target_manifold.project(q, tol=projection_tol, max_iters=60)
        if not projection.success:
            continue
        target_q = np.asarray(projection.x_projected, dtype=float)
        if collision_fn is not None and bool(collision_fn(target_q)):
            continue
        ee_current = end_effector_point(robot, q)
        ee_target = end_effector_point(robot, target_q)
        if float(np.linalg.norm(ee_current - ee_target)) > task_tol:
            continue
        if any(float(np.linalg.norm(ee_target - prev)) <= max(task_tol * 0.75, 2e-2) for prev in seen_task):
            continue
        seen_task.append(ee_target.copy())
        hits.append((np.asarray(q, dtype=float), target_q, ee_target.copy()))
    return hits
