from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from ompl import util as ou

from primitive_manifold_planner.examplesupport.collision_utilities import configuration_in_collision, default_example_66_obstacles
from primitive_manifold_planner.examplesupport.example66_scene import build_example66_scene
from primitive_manifold_planner.examplesupport.jointspace_planner_utils import explore_joint_manifold, joint_step_statistics
from primitive_manifold_planner.examplesupport.spatial_robot import SpatialRobot3DOF
from primitive_manifold_planner.manifolds.robot import RobotPlaneManifold, RobotSphereManifold
from primitive_manifold_planner.thesis import parallel_evidence_planner as ex66
from primitive_manifold_planner.visualization import pyvista_available
from primitive_manifold_planner.visualization.cspace_robot import show_cspace_robot_planning
from primitive_manifold_planner.visualization.robot import show_pyvista_robot_demo

try:
    import pyvista as pv
except Exception:
    pv = None

try:
    from scipy.optimize import least_squares
except Exception:
    least_squares = None


@dataclass
class RobotExecutionResult:
    """Robot animation/execution data derived from a dense theta path."""

    target_task_points_3d: np.ndarray
    joint_path: np.ndarray
    end_effector_points_3d: np.ndarray
    ik_success_count: int
    ik_failure_count: int
    max_tracking_error: float
    mean_tracking_error: float
    max_joint_step: float
    execution_success: bool
    diagnostics: str
    animation_enabled: bool
    planner_path_resampled_for_robot: bool
    planner_joint_path_used_directly: bool
    max_constraint_residual: float
    mean_constraint_residual: float
    constraint_validation_success: bool
    worst_constraint_stage: str
    worst_constraint_index: int
    execution_source: str


@dataclass
class JointspaceExecutionValidation:
    valid: bool
    max_constraint_residual: float
    mean_constraint_residual: float
    worst_index: int
    worst_stage: str
    stage_labels: list[str]
    per_waypoint_residuals: np.ndarray
    message: str


@dataclass
class JointRouteSmoothingResult:
    enabled: bool
    certified: bool
    fallback_used: bool
    attempts: int
    accepts: int
    nodes_before: int
    nodes_after: int
    dense_points_before: int
    dense_points_after: int
    joint_length_before: float
    joint_length_after: float
    task_length_before: float
    task_length_after: float
    curvature_cost_before: float
    curvature_cost_after: float
    max_joint_step_after: float
    max_constraint_residual_after: float
    mean_constraint_residual_after: float
    collision_free_after: bool
    message: str


def wrap_angles(joint_angles: np.ndarray) -> np.ndarray:
    arr = np.asarray(joint_angles, dtype=float)
    return (arr + np.pi) % (2.0 * np.pi) - np.pi #keeps joint angles inside a consistent range


def choose_robot_for_route(route: np.ndarray) -> SpatialRobot3DOF:
    """Create the educational 3-DOF robot positioned near the task-space scene."""

    pts = np.asarray(route, dtype=float)
    center = np.mean(pts, axis=0)
    mins = np.min(pts, axis=0)

    base_world = np.asarray(
        [
            float(center[0]),
            float(center[1] - 1.35),
            float(mins[2] - 0.55),
        ],
        dtype=float,
    )
    distances = np.linalg.norm(pts - base_world[None, :], axis=1)
    total_reach = max(3.1, 1.25 * float(np.max(distances)))
    link_lengths = np.asarray([0.45, 0.35, 0.25], dtype=float) * total_reach
    return SpatialRobot3DOF(
        link_lengths=link_lengths,
        base_world=base_world,
        link_radius=0.06,
        joint_radius=0.10,
        ee_radius=0.08,
    )


def solve_spatial_ik(
    robot: SpatialRobot3DOF,
    target_world: np.ndarray,
    warm_start: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    target = np.asarray(target_world, dtype=float).reshape(3)
    warm = wrap_angles(np.asarray(warm_start, dtype=float).reshape(3))

    if least_squares is not None:

        def residual(theta: np.ndarray) -> np.ndarray:
            ee = robot.forward_kinematics_3d(theta)[-1]
            reg = 0.015 * wrap_angles(theta - warm)
            return np.concatenate([ee - target, reg], dtype=float)

        result = least_squares(
            residual,
            warm,
            bounds=(-np.pi * np.ones(3, dtype=float), np.pi * np.ones(3, dtype=float)),
            max_nfev=120,
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
        )
        candidate = wrap_angles(result.x)
        err = float(np.linalg.norm(robot.forward_kinematics_3d(candidate)[-1] - target))
        return candidate, err, bool(result.success and err <= 8e-2)

    theta = warm.copy()
    for _ in range(120):
        current = robot.forward_kinematics_3d(theta)[-1]
        error = target - current
        err_norm = float(np.linalg.norm(error))
        if err_norm <= 8e-2:
            return wrap_angles(theta), err_norm, True
        jac = np.zeros((3, 3), dtype=float)
        eps = 1e-4
        for col in range(3):
            perturbed = theta.copy()
            perturbed[col] += eps
            jac[:, col] = (robot.forward_kinematics_3d(perturbed)[-1] - current) / eps
        damp = 5e-2
        delta = jac.T @ np.linalg.solve(jac @ jac.T + damp * np.eye(3, dtype=float), error)
        theta = wrap_angles(theta + 0.6 * delta)
    final_err = float(np.linalg.norm(robot.forward_kinematics_3d(theta)[-1] - target))
    return wrap_angles(theta), final_err, bool(final_err <= 8e-2)


def resample_polyline(path: np.ndarray, num_points: int) -> np.ndarray:
    pts = np.asarray(path, dtype=float)
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=float)
    if len(pts) == 1 or num_points <= 1:
        return pts[[0]].copy()

    seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)], dtype=float)
    total = float(cumulative[-1])
    if total <= 1e-12:
        return np.repeat(pts[[0]], repeats=max(1, int(num_points)), axis=0)

    samples = np.linspace(0.0, total, int(num_points))
    out: list[np.ndarray] = []
    seg_idx = 0
    for s in samples:
        while seg_idx + 1 < len(cumulative) and cumulative[seg_idx + 1] < s:
            seg_idx += 1
        if seg_idx >= len(seg_lengths):
            out.append(pts[-1].copy())
            continue
        start = float(cumulative[seg_idx])
        end = float(cumulative[seg_idx + 1])
        alpha = 0.0 if end <= start + 1e-12 else float((s - start) / (end - start))
        out.append((1.0 - alpha) * pts[seg_idx] + alpha * pts[seg_idx + 1])
    return np.asarray(out, dtype=float)


def densify_polyline_by_step(path: np.ndarray, max_step: float) -> np.ndarray:
    """Densify the selected planner route without changing its geometry."""
    pts = np.asarray(path, dtype=float)
    if len(pts) < 2:
        return pts.copy()
    out: list[np.ndarray] = [pts[0].copy()]
    for start, goal in zip(pts[:-1], pts[1:]):
        delta = np.asarray(goal, dtype=float) - np.asarray(start, dtype=float)
        steps = max(1, int(np.ceil(float(np.linalg.norm(delta)) / max(float(max_step), 1e-9))))
        for idx in range(1, steps + 1):
            out.append(np.asarray(start, dtype=float) + (float(idx) / float(steps)) * delta)
    return np.asarray(out, dtype=float)


def interpolate_joint_path_by_step(joint_path: np.ndarray, max_joint_step: float) -> np.ndarray:
    q_path = np.asarray(joint_path, dtype=float)
    if len(q_path) < 2:
        return q_path.copy()
    out: list[np.ndarray] = [wrap_angles(q_path[0])]
    for q0, q1 in zip(q_path[:-1], q_path[1:]):
        start = wrap_angles(np.asarray(q0, dtype=float))
        delta = wrap_angles(np.asarray(q1, dtype=float) - start)
        steps = max(1, int(np.ceil(float(np.linalg.norm(delta)) / max(float(max_joint_step), 1e-9))))
        for idx in range(1, steps + 1):
            out.append(wrap_angles(start + (float(idx) / float(steps)) * delta))
    return np.asarray(out, dtype=float)


def _sphere_residual(point: np.ndarray, manifold) -> float:
    base = ex66.unwrap_manifold(manifold)
    return abs(float(np.linalg.norm(np.asarray(point, dtype=float) - np.asarray(base.center, dtype=float)) - float(base.radius)))


def _plane_residual(point: np.ndarray, manifold) -> float:
    base = ex66.unwrap_manifold(manifold)
    normal = np.asarray(base.normal, dtype=float)
    return abs(float(np.dot(np.asarray(point, dtype=float) - np.asarray(base.point, dtype=float), normal)))


def _route_stage_labels(reference_path: np.ndarray, left_manifold, plane_manifold, right_manifold) -> list[str]:
    labels: list[str] = []
    for point in np.asarray(reference_path, dtype=float):
        residuals = {
            ex66.LEFT_STAGE: _sphere_residual(point, left_manifold),
            ex66.PLANE_STAGE: _plane_residual(point, plane_manifold),
            ex66.RIGHT_STAGE: _sphere_residual(point, right_manifold),
        }
        labels.append(min(residuals, key=residuals.get))
    return labels


def _nearest_reference_stage(point: np.ndarray, reference_path: np.ndarray, reference_labels: list[str]) -> str:
    if len(reference_path) == 0 or len(reference_labels) == 0:
        return ex66.PLANE_STAGE
    distances = np.linalg.norm(np.asarray(reference_path, dtype=float) - np.asarray(point, dtype=float), axis=1)
    idx = int(np.argmin(distances))
    return str(reference_labels[min(idx, len(reference_labels) - 1)])


def _stage_residual(point: np.ndarray, stage: str, left_manifold, plane_manifold, right_manifold, tolerance: float) -> float:
    if stage == ex66.LEFT_STAGE:
        return _sphere_residual(point, left_manifold)
    if stage == ex66.RIGHT_STAGE:
        return _sphere_residual(point, right_manifold)
    residual = _plane_residual(point, plane_manifold)
    if not bool(plane_manifold.within_bounds(np.asarray(point, dtype=float), tol=tolerance)):
        residual = max(float(residual), 10.0 * float(tolerance))
    return float(residual)


def validate_jointspace_execution_against_stages(
    joint_path: np.ndarray,
    robot: SpatialRobot3DOF,
    result: ex66.FixedPlaneRoute,
    families,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    tolerance: float = 1e-2,
) -> JointspaceExecutionValidation:
    """Validate FK samples against the active stage constraints before animation."""
    _ = start_q, goal_q
    q_path = np.asarray(joint_path, dtype=float)
    if len(q_path) == 0:
        return JointspaceExecutionValidation(
            valid=False,
            max_constraint_residual=float("inf"),
            mean_constraint_residual=float("inf"),
            worst_index=-1,
            worst_stage="none",
            stage_labels=[],
            per_waypoint_residuals=np.zeros(0, dtype=float),
            message="empty joint execution path",
        )

    left_family, plane_family, right_family = families
    left_manifold = left_family.manifold(float(left_family.sample_lambdas()[0]))
    plane_manifold = plane_family.manifold(float(plane_family.sample_lambdas()[0]))
    right_manifold = right_family.manifold(float(right_family.sample_lambdas()[0]))
    ee_points = np.asarray([robot.forward_kinematics_3d(q)[-1] for q in q_path], dtype=float)
    reference_path = np.asarray(result.raw_path if len(result.raw_path) >= 2 else result.path, dtype=float)
    reference_labels = _route_stage_labels(reference_path, left_manifold, plane_manifold, right_manifold)

    labels: list[str] = []
    residuals: list[float] = []
    for point in ee_points:
        stage = _nearest_reference_stage(point, reference_path, reference_labels)
        residual = _stage_residual(point, stage, left_manifold, plane_manifold, right_manifold, tolerance)
        labels.append(stage)
        residuals.append(float(residual))

    residual_arr = np.asarray(residuals, dtype=float)
    worst_index = int(np.argmax(residual_arr)) if len(residual_arr) > 0 else -1
    worst_stage = labels[worst_index] if 0 <= worst_index < len(labels) else "none"
    max_residual = float(np.max(residual_arr)) if len(residual_arr) > 0 else float("inf")
    mean_residual = float(np.mean(residual_arr)) if len(residual_arr) > 0 else float("inf")
    valid = bool(max_residual <= float(tolerance))
    message = (
        f"constraint residual ok: max={max_residual:.4g}, mean={mean_residual:.4g}"
        if valid
        else (
            "joint path endpoints may be valid, but local interpolated edges are not certified; "
            f"max residual {max_residual:.4g} at waypoint {worst_index} on inferred {worst_stage} stage"
        )
    )
    return JointspaceExecutionValidation(
        valid=valid,
        max_constraint_residual=max_residual,
        mean_constraint_residual=mean_residual,
        worst_index=worst_index,
        worst_stage=worst_stage,
        stage_labels=labels,
        per_waypoint_residuals=residual_arr,
        message=message,
    )


def _initial_ik_guess(robot: SpatialRobot3DOF, target: np.ndarray, q_start: np.ndarray | None = None) -> np.ndarray:
    if q_start is not None:
        return wrap_angles(np.asarray(q_start, dtype=float).reshape(3))
    rel = np.asarray(target, dtype=float) - np.asarray(robot.base_world, dtype=float)
    yaw_guess = math.atan2(float(rel[1]), float(rel[0]))
    radial_guess = float(np.linalg.norm(rel[:2]))
    pitch_guess = math.atan2(float(rel[2]), max(radial_guess, 1e-6))
    return np.asarray([yaw_guess, pitch_guess, -0.65], dtype=float)


def build_continuous_robot_execution_path(
    task_path: np.ndarray,
    robot_model: SpatialRobot3DOF,
    q_start: np.ndarray | None = None,
    max_cartesian_step: float = 0.03,
    max_joint_step: float = 0.08,
    ik_tolerance: float = 1e-3,
    max_ik_iters: int = 100,
) -> RobotExecutionResult | None:
    """Track only the selected planner route; planner evidence is not robot motion."""
    route = np.asarray(task_path, dtype=float)
    if len(route) < 2:
        return None
    targets = densify_polyline_by_step(route, max_step=max_cartesian_step)
    if len(targets) < 2:
        return None

    original_solver_limit = 120
    _ = original_solver_limit, max_ik_iters
    warm = _initial_ik_guess(robot_model, targets[0], q_start=q_start)
    joint_path: list[np.ndarray] = []
    ee_trace: list[np.ndarray] = []
    tracking_errors: list[float] = []
    ik_failures = 0
    diagnostics: list[str] = []

    for idx, target in enumerate(targets):
        solved, err, ok = solve_spatial_ik(robot_model, target, warm)
        jump = float(np.linalg.norm(wrap_angles(solved - warm))) if len(joint_path) > 0 else 0.0
        if (not ok or err > ik_tolerance or jump > max_joint_step) and len(joint_path) > 0:
            # Retry this waypoint by adding one local midpoint before it. This keeps
            # IK on the same branch instead of jumping to a different solution.
            previous_target = ee_trace[-1]
            midpoint = 0.5 * (np.asarray(previous_target, dtype=float) + np.asarray(target, dtype=float))
            mid_q, mid_err, mid_ok = solve_spatial_ik(robot_model, midpoint, warm)
            mid_jump = float(np.linalg.norm(wrap_angles(mid_q - warm)))
            if mid_ok and mid_err <= max(ik_tolerance, 3e-3) and mid_jump <= max_joint_step:
                warm = mid_q.copy()
                joint_path.append(warm.copy())
                mid_ee = np.asarray(robot_model.forward_kinematics_3d(warm)[-1], dtype=float)
                ee_trace.append(mid_ee)
                tracking_errors.append(float(np.linalg.norm(mid_ee - midpoint)))
                solved, err, ok = solve_spatial_ik(robot_model, target, warm)
                jump = float(np.linalg.norm(wrap_angles(solved - warm)))

        if ok and err <= max(ik_tolerance, 3e-3) and jump <= max_joint_step:
            warm = solved.copy()
        else:
            ik_failures += 1
            diagnostics.append(f"waypoint {idx}: IK/joint-step check failed; reused previous branch")

        joint_path.append(warm.copy())
        ee_world = np.asarray(robot_model.forward_kinematics_3d(warm)[-1], dtype=float)
        ee_trace.append(ee_world)
        tracking_errors.append(float(np.linalg.norm(ee_world - np.asarray(target, dtype=float))))

    joint_arr = np.asarray(joint_path, dtype=float)
    ee_arr = np.asarray(ee_trace, dtype=float)
    joint_steps = (
        np.linalg.norm([wrap_angles(b - a) for a, b in zip(joint_arr[:-1], joint_arr[1:])], axis=1)
        if len(joint_arr) >= 2
        else np.zeros(0, dtype=float)
    )
    max_step_seen = float(np.max(joint_steps)) if len(joint_steps) > 0 else 0.0
    max_error = float(max(tracking_errors) if tracking_errors else 0.0)
    execution_success = bool(ik_failures == 0 and max_error <= max(ik_tolerance, 3e-3) and max_step_seen <= max_joint_step + 1e-9)

    return RobotExecutionResult(
        target_task_points_3d=np.asarray(targets, dtype=float),
        joint_path=joint_arr,
        end_effector_points_3d=ee_arr,
        ik_success_count=int(len(targets) - ik_failures),
        ik_failure_count=int(ik_failures),
        max_tracking_error=max_error,
        mean_tracking_error=float(np.mean(tracking_errors) if tracking_errors else 0.0),
        max_joint_step=max_step_seen,
        execution_success=execution_success,
        diagnostics="; ".join(diagnostics[:4]) if diagnostics else "ok",
        animation_enabled=bool(len(joint_arr) >= 2),
        planner_path_resampled_for_robot=True,
        planner_joint_path_used_directly=False,
        max_constraint_residual=0.0,
        mean_constraint_residual=0.0,
        constraint_validation_success=True,
        worst_constraint_stage="task_space_tracking",
        worst_constraint_index=-1,
        execution_source="taskspace_fallback" if q_start is not None else "taskspace_ik_tracking",
    )


def build_robot_execution(
    result: ex66.FixedPlaneRoute,
    robot: SpatialRobot3DOF,
    use_planner_joint_path: bool = False,
    families=None,
    start_q: np.ndarray | None = None,
    goal_q: np.ndarray | None = None,
    validate_joint_execution: bool = True,
    allow_uncertified_joint_animation: bool = False,
    allow_taskspace_fallback: bool = False,
    joint_max_step: float = 0.08,
) -> RobotExecutionResult | None:
    if use_planner_joint_path and allow_taskspace_fallback:
        raise RuntimeError(
            "Task-space IK fallback is forbidden in joint-space planning mode. "
            "The robot execution path must come from the planner's dense theta trajectory."
        )
    if use_planner_joint_path and allow_uncertified_joint_animation:
        raise RuntimeError(
            "Uncertified joint animation is disabled for the thesis-facing joint-space demo."
        )
    dense_joint_path = np.asarray(getattr(result, "dense_joint_path", np.zeros((0, 3), dtype=float)), dtype=float)
    dense_constraint_certified = bool(
        getattr(result, "dense_joint_path_constraint_certified", getattr(result, "dense_joint_path_is_certified", False))
    )
    dense_execution_certified = bool(getattr(result, "dense_joint_path_execution_certified", False))
    if use_planner_joint_path and dense_constraint_certified and len(dense_joint_path) >= 2:
        ee_points = np.asarray([robot.forward_kinematics_3d(theta)[-1] for theta in dense_joint_path], dtype=float)
        joint_steps = np.asarray(getattr(result, "dense_joint_path_joint_steps", np.zeros(0, dtype=float)), dtype=float)
        if len(joint_steps) == 0:
            joint_steps = np.linalg.norm(
                [wrap_angles(np.asarray(b, dtype=float) - np.asarray(a, dtype=float)) for a, b in zip(dense_joint_path[:-1], dense_joint_path[1:])],
                axis=1,
            )
        residuals = np.asarray(getattr(result, "dense_joint_path_constraint_residuals", np.zeros(0, dtype=float)), dtype=float)
        max_step_seen = float(getattr(result, "dense_joint_path_max_joint_step", float(np.max(joint_steps)) if len(joint_steps) > 0 else 0.0))
        step_ok = bool(dense_execution_certified and max_step_seen <= float(joint_max_step) + 1e-9)
        return RobotExecutionResult(
            target_task_points_3d=ee_points,
            joint_path=dense_joint_path,
            end_effector_points_3d=ee_points,
            ik_success_count=2,
            ik_failure_count=0,
            max_tracking_error=0.0,
            mean_tracking_error=0.0,
            max_joint_step=max_step_seen,
            execution_success=bool(step_ok),
            diagnostics=(
                "using certified dense joint path from planner"
                if step_ok
                else f"constraint-certified dense joint path is not execution-certified: max_step={max_step_seen:.4g}"
            ),
            animation_enabled=bool(step_ok),
            planner_path_resampled_for_robot=False,
            planner_joint_path_used_directly=True,
            max_constraint_residual=float(np.max(residuals)) if len(residuals) > 0 else 0.0,
            mean_constraint_residual=float(np.mean(residuals)) if len(residuals) > 0 else 0.0,
            constraint_validation_success=bool(dense_constraint_certified),
            worst_constraint_stage="certified_dense_joint_path",
            worst_constraint_index=-1,
            execution_source="certified_dense_joint_path" if step_ok else "disabled_joint_step_violation",
        )

    planned_joint_path = np.asarray(getattr(result, "joint_path", np.zeros((0, 3), dtype=float)), dtype=float)
    if use_planner_joint_path:
        message = str(
            getattr(
                result,
                "dense_joint_path_message",
                "joint-space planner did not return an execution-certified dense theta path",
            )
        )
        fallback_theta = dense_joint_path if len(dense_joint_path) >= 2 else planned_joint_path
        ee_points = (
            np.asarray([robot.forward_kinematics_3d(theta)[-1] for theta in fallback_theta], dtype=float)
            if len(fallback_theta) >= 2
            else np.zeros((0, 3), dtype=float)
        )
        return RobotExecutionResult(
            target_task_points_3d=np.asarray(ee_points, dtype=float),
            joint_path=np.asarray(fallback_theta, dtype=float),
            end_effector_points_3d=ee_points,
            ik_success_count=0,
            ik_failure_count=0,
            max_tracking_error=0.0,
            mean_tracking_error=0.0,
            max_joint_step=float(getattr(result, "dense_joint_path_max_joint_step", 0.0)),
            execution_success=False,
            diagnostics=message + "; no task-space fallback or sparse joint interpolation was used",
            animation_enabled=False,
            planner_path_resampled_for_robot=False,
            planner_joint_path_used_directly=True,
            max_constraint_residual=float(
                np.max(getattr(result, "dense_joint_path_constraint_residuals", np.zeros(0, dtype=float)))
                if len(getattr(result, "dense_joint_path_constraint_residuals", [])) > 0
                else 0.0
            ),
            mean_constraint_residual=float(
                np.mean(getattr(result, "dense_joint_path_constraint_residuals", np.zeros(0, dtype=float)))
                if len(getattr(result, "dense_joint_path_constraint_residuals", [])) > 0
                else 0.0
            ),
            constraint_validation_success=bool(dense_constraint_certified),
            worst_constraint_stage="dense_theta_path_unavailable",
            worst_constraint_index=-1,
            execution_source="disabled_no_certified_dense_theta_path",
        )

    route = np.asarray(result.path if len(result.path) >= 2 else result.raw_path, dtype=float)
    if len(route) < 2:
        if len(planned_joint_path) < 2:
            return None
        ee_points = np.asarray([robot.forward_kinematics_3d(theta)[-1] for theta in planned_joint_path], dtype=float)
        return RobotExecutionResult(
            target_task_points_3d=np.asarray(ee_points, dtype=float),
            joint_path=np.asarray(planned_joint_path, dtype=float),
            end_effector_points_3d=np.asarray(ee_points, dtype=float),
            ik_success_count=int(len(planned_joint_path)),
            ik_failure_count=0,
            max_tracking_error=0.0,
            mean_tracking_error=0.0,
            max_joint_step=0.0,
            execution_success=True,
            diagnostics="using fallback planner joint path",
            animation_enabled=bool(len(planned_joint_path) >= 2),
            planner_path_resampled_for_robot=False,
            planner_joint_path_used_directly=True,
            max_constraint_residual=0.0,
            mean_constraint_residual=0.0,
            constraint_validation_success=False,
            worst_constraint_stage="not_validated",
            worst_constraint_index=-1,
            execution_source="fallback_planner_joint_path",
        )
    return build_continuous_robot_execution_path(route, robot)


def configure_example_66_budgets(args) -> None:
    planner_core = getattr(ex66, "planner_core", ex66)
    requested_rounds = args.max_rounds if args.max_rounds is not None else args.max_iters
    if requested_rounds is not None:
        ex66.SAFETY_MAX_TOTAL_ROUNDS = max(4, int(requested_rounds))
        ex66.MIN_ROUNDS_BEFORE_SATURATION_CHECK = min(
            ex66.MIN_ROUNDS_BEFORE_SATURATION_CHECK,
            max(4, ex66.SAFETY_MAX_TOTAL_ROUNDS // 2),
        )
        ex66.MIN_POST_SOLUTION_ROUNDS = min(
            ex66.MIN_POST_SOLUTION_ROUNDS,
            max(2, ex66.SAFETY_MAX_TOTAL_ROUNDS // 4),
        )
        planner_core.SAFETY_MAX_TOTAL_ROUNDS = ex66.SAFETY_MAX_TOTAL_ROUNDS
        planner_core.MIN_ROUNDS_BEFORE_SATURATION_CHECK = ex66.MIN_ROUNDS_BEFORE_SATURATION_CHECK
        planner_core.MIN_POST_SOLUTION_ROUNDS = ex66.MIN_POST_SOLUTION_ROUNDS
    elif args.planning_mode == "jointspace_constrained_planning" and not args.fast:
        ex66.SAFETY_MAX_TOTAL_ROUNDS = max(ex66.SAFETY_MAX_TOTAL_ROUNDS, 80)
        ex66.MIN_ROUNDS_BEFORE_SATURATION_CHECK = min(ex66.MIN_ROUNDS_BEFORE_SATURATION_CHECK, 40)
        ex66.MIN_POST_SOLUTION_ROUNDS = max(ex66.MIN_POST_SOLUTION_ROUNDS, 12)
        ex66.PROPOSALS_PER_ROUND = max(ex66.PROPOSALS_PER_ROUND, 2)
        ex66.PLANE_BRIDGE_TRY_LIMIT = max(ex66.PLANE_BRIDGE_TRY_LIMIT, 8)
        ex66.RIGHT_BRIDGE_TRY_LIMIT = max(ex66.RIGHT_BRIDGE_TRY_LIMIT, 8)
        planner_core.SAFETY_MAX_TOTAL_ROUNDS = ex66.SAFETY_MAX_TOTAL_ROUNDS
        planner_core.MIN_ROUNDS_BEFORE_SATURATION_CHECK = ex66.MIN_ROUNDS_BEFORE_SATURATION_CHECK
        planner_core.MIN_POST_SOLUTION_ROUNDS = ex66.MIN_POST_SOLUTION_ROUNDS
        planner_core.PROPOSALS_PER_ROUND = ex66.PROPOSALS_PER_ROUND
        planner_core.PLANE_BRIDGE_TRY_LIMIT = ex66.PLANE_BRIDGE_TRY_LIMIT
        planner_core.RIGHT_BRIDGE_TRY_LIMIT = ex66.RIGHT_BRIDGE_TRY_LIMIT
    ex66.STOP_AFTER_FIRST_SOLUTION = bool(args.stop_after_first_solution)
    planner_core.STOP_AFTER_FIRST_SOLUTION = ex66.STOP_AFTER_FIRST_SOLUTION
    if args.continue_after_first_solution:
        ex66.STOP_AFTER_FIRST_SOLUTION = False
        planner_core.STOP_AFTER_FIRST_SOLUTION = False
    if args.planning_mode == "jointspace_constrained_planning" and not args.fast:
        ex66.PROPOSALS_PER_ROUND = max(ex66.PROPOSALS_PER_ROUND, 2)
        ex66.PLANE_BRIDGE_TRY_LIMIT = max(ex66.PLANE_BRIDGE_TRY_LIMIT, 8)
        ex66.RIGHT_BRIDGE_TRY_LIMIT = max(ex66.RIGHT_BRIDGE_TRY_LIMIT, 8)
        planner_core.PROPOSALS_PER_ROUND = ex66.PROPOSALS_PER_ROUND
        planner_core.PLANE_BRIDGE_TRY_LIMIT = ex66.PLANE_BRIDGE_TRY_LIMIT
        planner_core.RIGHT_BRIDGE_TRY_LIMIT = ex66.RIGHT_BRIDGE_TRY_LIMIT
    if args.fast:
        ex66.SAFETY_MAX_TOTAL_ROUNDS = min(ex66.SAFETY_MAX_TOTAL_ROUNDS, 10)
        ex66.MIN_ROUNDS_BEFORE_SATURATION_CHECK = min(ex66.MIN_ROUNDS_BEFORE_SATURATION_CHECK, 6)
        ex66.MIN_POST_SOLUTION_ROUNDS = min(ex66.MIN_POST_SOLUTION_ROUNDS, 3)
        planner_core.SAFETY_MAX_TOTAL_ROUNDS = ex66.SAFETY_MAX_TOTAL_ROUNDS
        planner_core.MIN_ROUNDS_BEFORE_SATURATION_CHECK = ex66.MIN_ROUNDS_BEFORE_SATURATION_CHECK
        planner_core.MIN_POST_SOLUTION_ROUNDS = ex66.MIN_POST_SOLUTION_ROUNDS


def has_certified_dense_joint_path(result) -> bool:
    """Return True only when the planner produced an executable dense theta path."""

    theta_path = np.asarray(getattr(result, "dense_joint_path", np.zeros((0, 3), dtype=float)), dtype=float)
    return bool(
        theta_path.ndim == 2
        and theta_path.shape[1] == 3
        and len(theta_path) >= 2
        and bool(getattr(result, "dense_joint_path_execution_certified", False))
    )


def apply_quick_cspace_demo_preset(args) -> None:
    """Apply runtime-friendly Example 66 settings without changing certification."""

    if not bool(getattr(args, "quick_cspace_demo", False)):
        return
    args.planning_mode = "jointspace_constrained_planning"
    args.with_obstacles = False
    args.seed = int(args.seed if args.seed is not None else 41)
    if args.max_rounds is None and args.max_iters is None:
        args.max_rounds = 12
    args.fast = True
    if float(args.joint_max_step) == 0.08:
        args.joint_max_step = 0.12
    args.smooth_final_route = False
    args.show_cspace = True
    args.cspace_route_only = False
    args.cspace_no_surfaces = False
    args.cspace_surface_mode = "exact"


def planner_parity_stats(result: ex66.FixedPlaneRoute, planner_mode: str, robot_execution: RobotExecutionResult | None) -> dict[str, object]:
    nodes_explored = int(result.left_evidence_nodes + result.plane_evidence_nodes + result.right_evidence_nodes)
    edges_explored = int(sum(len(edges) for edges in result.stage_evidence_edges.values()))
    return {
        "planner_mode": planner_mode,
        "nodes_explored": nodes_explored,
        "edges_explored": edges_explored,
        "left_nodes": int(result.left_evidence_nodes),
        "plane_nodes": int(result.plane_evidence_nodes),
        "right_nodes": int(result.right_evidence_nodes),
        "entry_transitions_found": int(result.transition_hypotheses_left_plane),
        "exit_transitions_found": int(result.transition_hypotheses_plane_right),
        "first_solution_iteration": result.first_solution_round,
        "continued_after_solution": bool(result.continued_after_first_solution),
        "final_solution_iteration": result.best_solution_round,
        "best_route_score": float(result.route_cost_raw),
        "final_path_waypoints": int(len(result.path if len(result.path) >= 2 else result.raw_path)),
        "robot_execution_waypoints": int(0 if robot_execution is None else len(robot_execution.joint_path)),
    }


def _main_route_rejection_reason(result: ex66.FixedPlaneRoute) -> str:
    counts = getattr(result, "mode_counts", {}) or {}
    dense_message = str(getattr(result, "dense_joint_path_message", "") or "")
    if dense_message:
        return dense_message
    if int(counts.get("route_candidates_evaluated", 0)) <= 0:
        if int(counts.get("route_candidates_missing_plane_graph_path", 0)) > 0:
            return "transition pairs exist, but the plane graph has no entry-to-exit path for the evaluated candidates"
        if int(counts.get("route_candidates_missing_left_graph_path", 0)) > 0:
            return "left graph path from start to left-plane transition is missing"
        if int(counts.get("route_candidates_missing_right_graph_path", 0)) > 0:
            return "right graph path from plane-right transition to goal is missing"
        if int(counts.get("route_candidates_dense_edge_path_missing", 0)) > 0:
            return "graph route exists only as sparse nodes; stored dense edge path is missing"
        if int(getattr(result, "transition_hypotheses_left_plane", 0)) <= 0:
            return "no left-plane transition hypotheses were found"
        if int(getattr(result, "transition_hypotheses_plane_right", 0)) <= 0:
            return "no plane-right transition hypotheses were found"
        return "transitions exist, but no route candidate was built from them"
    if int(counts.get("route_candidates_local_connector_failed_plane", 0)) > 0:
        return "route candidate rejected: plane segment local connector failed residual/joint continuity certification"
    if int(counts.get("route_candidates_local_connector_failed_left", 0)) > 0:
        return "route candidate rejected: left segment local connector failed residual/joint continuity certification"
    if int(counts.get("route_candidates_local_connector_failed_right", 0)) > 0:
        return "route candidate rejected: right segment local connector failed residual/joint continuity certification"
    if int(counts.get("route_candidates_rejected_joint_jump", 0)) > 0:
        return "route candidates were rejected by joint-step continuity certification"
    if int(counts.get("route_candidates_constraint_certified", 0)) <= 0:
        return "route candidates did not pass constraint certification"
    if int(counts.get("route_candidates_execution_certified", 0)) <= 0:
        return "route candidates did not pass dense execution certification"
    if bool(getattr(result, "saturated_before_solution", False)):
        return f"exploration saturated before a certified dense route was extracted; stagnation_stage={getattr(result, 'stagnation_stage', None)}"
    return "certified dense route was not available"


def print_example66_failure_report(result: ex66.FixedPlaneRoute, *, quick_mode: bool) -> None:
    dense_points = int(len(getattr(result, "dense_joint_path", [])))
    dense_certified = bool(getattr(result, "dense_joint_path_execution_certified", False))
    if bool(getattr(result, "success", False)) and dense_points > 0 and dense_certified:
        return
    counts = getattr(result, "mode_counts", {}) or {}
    print_block(
        "Example 66 quick failure report" if quick_mode else "Example 66 dense-route failure report",
        {
            "planner_success": bool(getattr(result, "success", False)),
            "planner_raw_success_before_dense_certification": bool(
                getattr(result, "raw_planner_success_before_dense_certification", getattr(result, "success", False))
            ),
            "dense_joint_path_points": dense_points,
            "dense_joint_path_execution_certified": dense_certified,
            "transitions_found_left_plane": int(getattr(result, "transition_hypotheses_left_plane", 0)),
            "transitions_found_plane_right": int(getattr(result, "transition_hypotheses_plane_right", 0)),
            "route_candidates_built": int(counts.get("route_candidates_evaluated", 0)),
            "route_candidates_constraint_certified": int(counts.get("route_candidates_constraint_certified", 0)),
            "route_candidates_execution_certified": int(counts.get("route_candidates_execution_certified", 0)),
            "route_candidates_rejected_joint_jump": int(counts.get("route_candidates_rejected_joint_jump", 0)),
            "missing_left_graph_path": int(counts.get("route_candidates_missing_left_graph_path", 0)),
            "missing_plane_graph_path": int(counts.get("route_candidates_missing_plane_graph_path", 0)),
            "missing_right_graph_path": int(counts.get("route_candidates_missing_right_graph_path", 0)),
            "dense_edge_path_missing": int(counts.get("route_candidates_dense_edge_path_missing", 0)),
            "sparse_only_graph_path": int(counts.get("route_candidates_sparse_only_graph_path", 0)),
            "local_connector_attempts": int(counts.get("route_candidates_local_replan_attempted", 0)),
            "local_connector_failed_left": int(counts.get("route_candidates_local_connector_failed_left", 0)),
            "local_connector_failed_plane": int(counts.get("route_candidates_local_connector_failed_plane", 0)),
            "local_connector_failed_right": int(counts.get("route_candidates_local_connector_failed_right", 0)),
            "transition_stack_rejections": int(counts.get("route_candidates_transition_stack_rejected", 0)),
            "constraint_residual_rejections": int(counts.get("route_candidates_constraint_rejected", 0)),
            "collision_rejections": int(counts.get("route_candidates_collision_rejected", 0)),
            "query_connector_routes_built": int(counts.get("route_candidates_query_connectors_used", 0)),
            "main_rejection_reason": _main_route_rejection_reason(result),
            "suggestion": "Increase --max-rounds or inspect route realization / dense local edges.",
        },
    )


def enforce_dense_certified_success_for_jointspace(result: ex66.FixedPlaneRoute) -> None:
    """For the demo contract, success means an execution-certified dense theta route."""

    raw_success = bool(getattr(result, "success", False))
    result.raw_planner_success_before_dense_certification = raw_success
    if raw_success and not has_certified_dense_joint_path(result):
        result.success = False
        message = str(getattr(result, "message", "") or "")
        dense_message = str(getattr(result, "dense_joint_path_message", "") or "dense joint path was not execution-certified")
        result.message = f"{message} Dense route certification failed: {dense_message}"


def jointspace_display_route_from_dense(result: ex66.FixedPlaneRoute, robot: SpatialRobot3DOF) -> np.ndarray:
    dense_joint_path = np.asarray(getattr(result, "dense_joint_path", np.zeros((0, 3), dtype=float)), dtype=float)
    if not bool(getattr(result, "dense_joint_path_execution_certified", False)) or len(dense_joint_path) < 2:
        return np.zeros((0, 3), dtype=float)
    return np.asarray([robot.forward_kinematics_3d(theta)[-1] for theta in dense_joint_path], dtype=float)


def primary_display_route_for_mode(
    result: ex66.FixedPlaneRoute,
    robot: SpatialRobot3DOF,
    planner_mode: str,
) -> tuple[np.ndarray, str]:
    if planner_mode == "joint_space":
        dense_route = jointspace_display_route_from_dense(result, robot)
        if len(dense_route) >= 2:
            return dense_route, "FK(result.dense_joint_path)"
        return np.zeros((0, 3), dtype=float), "none"
    if len(result.path) >= 2:
        return np.asarray(result.path, dtype=float), "result.path"
    if len(result.raw_path) >= 2:
        return np.asarray(result.raw_path, dtype=float), "result.raw_path"
    return np.zeros((0, 3), dtype=float), "none"


def path_source_audit(
    result: ex66.FixedPlaneRoute,
    robot_execution: RobotExecutionResult | None,
    robot: SpatialRobot3DOF,
    planner_mode: str,
) -> dict[str, object]:
    display_route, display_source = primary_display_route_for_mode(result, robot, planner_mode)
    trace = (
        np.asarray(robot_execution.end_effector_points_3d, dtype=float)
        if robot_execution is not None
        else np.zeros((0, 3), dtype=float)
    )
    if len(display_route) >= 2 and len(trace) >= 2:
        sample_count = max(len(display_route), len(trace))
        display_cmp = resample_polyline(display_route, sample_count)
        trace_cmp = resample_polyline(trace, sample_count)
        errors = np.linalg.norm(display_cmp - trace_cmp, axis=1)
        max_error = float(np.max(errors))
        mean_error = float(np.mean(errors))
    else:
        max_error = 0.0
        mean_error = 0.0

    source_mismatch = bool(planner_mode == "joint_space" and max_error > 1e-3)
    return {
        "display_route_source": display_source,
        "robot_execution_source": "none" if robot_execution is None else robot_execution.execution_source,
        "display_route_points": int(len(display_route)),
        "raw_path_points": int(len(result.raw_path)),
        "result_path_points": int(len(result.path)),
        "joint_path_points": int(len(getattr(result, "joint_path", []))),
        "dense_joint_path_points": int(len(getattr(result, "dense_joint_path", []))),
        "ee_trace_points": int(len(trace)),
        "display_vs_trace_max_error": max_error,
        "display_vs_trace_mean_error": mean_error,
        "source_mismatch_warning": (
            "Displayed route and execution trace differ; visualization source mismatch"
            if source_mismatch
            else "ok"
        ),
    }


def _joint_path_length(path: np.ndarray) -> float:
    arr = np.asarray(path, dtype=float)
    if len(arr) < 2:
        return 0.0
    deltas = wrap_angles(np.diff(arr, axis=0))
    return float(np.sum(np.linalg.norm(deltas, axis=1)))


def _task_path_length(robot: SpatialRobot3DOF, joint_path: np.ndarray) -> float:
    task = jointspace_display_route_from_dense(
        type("_Route", (), {"dense_joint_path": np.asarray(joint_path, dtype=float), "dense_joint_path_execution_certified": True})(),
        robot,
    )
    return float(ex66.path_cost(task))


def _curvature_cost(joint_path: np.ndarray) -> float:
    arr = np.asarray(joint_path, dtype=float)
    if len(arr) < 3:
        return 0.0
    cost = 0.0
    for prev_q, q, next_q in zip(arr[:-2], arr[1:-1], arr[2:]):
        d0 = wrap_angles(np.asarray(q, dtype=float) - np.asarray(prev_q, dtype=float))
        d1 = wrap_angles(np.asarray(next_q, dtype=float) - np.asarray(q, dtype=float))
        cost += float(np.linalg.norm(d1 - d0) ** 2)
    return float(cost)


def _route_quality_cost(
    joint_path: np.ndarray,
    robot: SpatialRobot3DOF,
    joint_weight: float,
    task_weight: float,
    curvature_weight: float,
) -> float:
    return (
        float(joint_weight) * _joint_path_length(joint_path)
        + float(task_weight) * _task_path_length(robot, joint_path)
        + float(curvature_weight) * _curvature_cost(joint_path)
    )


def _stage_manifolds_for_robot(families, robot: SpatialRobot3DOF):
    """Build FK-pulled-back robot sphere/plane manifolds for Example 66."""

    left_family, plane_family, right_family = families
    left_geom = left_family.manifold(float(left_family.sample_lambdas()[0]))
    plane_geom = plane_family.manifold(float(plane_family.sample_lambdas()[0]))
    right_geom = right_family.manifold(float(right_family.sample_lambdas()[0]))
    left_base = ex66.unwrap_manifold(left_geom)
    plane_base = ex66.unwrap_manifold(plane_geom)
    right_base = ex66.unwrap_manifold(right_geom)
    joint_lower = -np.pi * np.ones(3, dtype=float)
    joint_upper = np.pi * np.ones(3, dtype=float)
    plane_task_validity = (
        (lambda ee, geom=plane_geom: bool(geom.within_bounds(np.asarray(ee, dtype=float))))
        if isinstance(plane_geom, ex66.MaskedManifold)
        else None
    )
    return {
        # Workspace sphere residual is pulled back through FK(theta).
        ex66.LEFT_STAGE: RobotSphereManifold(
            robot=robot,
            center=left_base.center,
            radius=left_base.radius,
            name="smooth_robot_left_sphere",
            joint_lower=joint_lower,
            joint_upper=joint_upper,
        ),
        # Workspace plane residual is pulled back through FK(theta).
        ex66.PLANE_STAGE: RobotPlaneManifold(
            robot=robot,
            point=plane_base.point,
            normal=plane_base.normal,
            name="smooth_robot_transfer_plane",
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            task_space_validity_fn=plane_task_validity,
        ),
        # Right support is another active manifold in robot joint space.
        ex66.RIGHT_STAGE: RobotSphereManifold(
            robot=robot,
            center=right_base.center,
            radius=right_base.radius,
            name="smooth_robot_right_sphere",
            joint_lower=joint_lower,
            joint_upper=joint_upper,
        ),
    }


def _stage_order_is_monotone(labels: list[str]) -> bool:
    order = {ex66.LEFT_STAGE: 0, ex66.PLANE_STAGE: 1, ex66.RIGHT_STAGE: 2}
    sequence = [order.get(str(label), -1) for label in labels]
    if any(value < 0 for value in sequence):
        return False
    return all(b >= a for a, b in zip(sequence[:-1], sequence[1:]))


def _transition_stack_residuals(
    q_path: np.ndarray,
    labels: list[str],
    manifolds: dict[str, object],
    *,
    selected_left_plane_theta: np.ndarray | None = None,
    selected_plane_right_theta: np.ndarray | None = None,
) -> dict[str, object]:
    """Measure whether selected transition theta values satisfy both stages."""

    q_arr = np.asarray(q_path, dtype=float)

    def stack_at(q: np.ndarray, source: str, target: str) -> float:
        source_res = float(np.linalg.norm(manifolds[source].residual(q)))
        target_res = float(np.linalg.norm(manifolds[target].residual(q)))
        return float(np.linalg.norm([source_res, target_res]))

    def best_boundary_stack(source: str, target: str) -> tuple[float, int]:
        best = float("inf")
        best_idx = -1
        for idx in range(len(labels) - 1):
            if labels[idx] != source or labels[idx + 1] != target:
                continue
            window_start = max(0, idx - 3)
            window_stop = min(len(labels), idx + 5)
            for candidate_idx in range(window_start, window_stop):
                q = q_arr[candidate_idx]
                value = stack_at(q, source, target)
                if value < best:
                    best = value
                    best_idx = int(candidate_idx)
        return best, best_idx

    left_plane, left_plane_idx = best_boundary_stack(ex66.LEFT_STAGE, ex66.PLANE_STAGE)
    plane_right, plane_right_idx = best_boundary_stack(ex66.PLANE_STAGE, ex66.RIGHT_STAGE)
    if selected_left_plane_theta is not None and np.asarray(selected_left_plane_theta, dtype=float).size == 3:
        left_plane = stack_at(np.asarray(selected_left_plane_theta, dtype=float), ex66.LEFT_STAGE, ex66.PLANE_STAGE)
        left_plane_idx = -2
    if selected_plane_right_theta is not None and np.asarray(selected_plane_right_theta, dtype=float).size == 3:
        plane_right = stack_at(np.asarray(selected_plane_right_theta, dtype=float), ex66.PLANE_STAGE, ex66.RIGHT_STAGE)
        plane_right_idx = -2
    values = [value for value in (left_plane, plane_right) if np.isfinite(value)]
    return {
        "selected_left_plane_stack_residual": float(left_plane) if np.isfinite(left_plane) else float("inf"),
        "selected_left_plane_transition_index": int(left_plane_idx),
        "selected_plane_right_stack_residual": float(plane_right) if np.isfinite(plane_right) else float("inf"),
        "selected_plane_right_transition_index": int(plane_right_idx),
        "max_transition_stack_residual": float(max(values)) if values else float("inf"),
    }


def compute_cspace_trajectory_audit(
    result: ex66.FixedPlaneRoute,
    robot: SpatialRobot3DOF,
    families,
) -> dict[str, object]:
    # TODO(jointspace_method): this audit should be migrated to the reusable
    # RouteCertification/CspaceDebugArtifact helpers once Example 65 and 66
    # share a single thesis-facing audit schema.
    theta_path = np.asarray(getattr(result, "dense_joint_path", np.zeros((0, 3), dtype=float)), dtype=float)
    labels = list(getattr(result, "dense_joint_path_stage_labels", []))
    residuals = np.asarray(getattr(result, "dense_joint_path_constraint_residuals", np.zeros(0, dtype=float)), dtype=float)
    joint_steps = np.asarray(getattr(result, "dense_joint_path_joint_steps", np.zeros(0, dtype=float)), dtype=float)
    if len(joint_steps) == 0 and len(theta_path) >= 2:
        joint_steps, _max_step, _mean_step, _worst = joint_step_statistics(theta_path)
    fk_trace = (
        # Thesis-facing path is dense theta; this FK trace is derived for display.
        np.asarray([robot.forward_kinematics_3d(theta)[-1] for theta in theta_path], dtype=float)
        if len(theta_path) > 0
        else np.zeros((0, 3), dtype=float)
    )
    manifolds = _stage_manifolds_for_robot(families, robot)
    stage_residuals: dict[str, list[float]] = {ex66.LEFT_STAGE: [], ex66.PLANE_STAGE: [], ex66.RIGHT_STAGE: []}
    for idx, theta in enumerate(theta_path):
        stage = labels[idx] if idx < len(labels) else ""
        if stage in manifolds:
            stage_residuals[stage].append(float(np.linalg.norm(manifolds[stage].residual(theta))))
    theta_min = np.min(theta_path, axis=0) if len(theta_path) > 0 else np.zeros(3, dtype=float)
    theta_max = np.max(theta_path, axis=0) if len(theta_path) > 0 else np.zeros(3, dtype=float)
    theta_span = theta_max - theta_min
    label_counts = {
        ex66.LEFT_STAGE: int(sum(1 for label in labels if label == ex66.LEFT_STAGE)),
        ex66.PLANE_STAGE: int(sum(1 for label in labels if label == ex66.PLANE_STAGE)),
        ex66.RIGHT_STAGE: int(sum(1 for label in labels if label == ex66.RIGHT_STAGE)),
    }
    selected_left_plane_theta = np.asarray(getattr(result, "selected_left_plane_transition_theta", np.zeros(0, dtype=float)), dtype=float)
    selected_plane_right_theta = np.asarray(getattr(result, "selected_plane_right_transition_theta", np.zeros(0, dtype=float)), dtype=float)
    transition = _transition_stack_residuals(
        theta_path,
        labels,
        manifolds,
        selected_left_plane_theta=selected_left_plane_theta,
        selected_plane_right_theta=selected_plane_right_theta,
    ) if len(theta_path) > 0 else {
        "selected_left_plane_stack_residual": float("inf"),
        "selected_left_plane_transition_index": -1,
        "selected_plane_right_stack_residual": float("inf"),
        "selected_plane_right_transition_index": -1,
        "max_transition_stack_residual": float("inf"),
    }
    return {
        "theta_path": theta_path,
        "fk_trace": fk_trace,
        "stage_labels": labels,
        "constraint_residuals": residuals,
        "joint_steps": np.asarray(joint_steps, dtype=float),
        "theta_path_points": int(len(theta_path)),
        "theta0_min": float(theta_min[0]),
        "theta0_max": float(theta_max[0]),
        "theta0_span": float(theta_span[0]),
        "theta1_min": float(theta_min[1]),
        "theta1_max": float(theta_max[1]),
        "theta1_span": float(theta_span[1]),
        "theta2_min": float(theta_min[2]),
        "theta2_max": float(theta_max[2]),
        "theta2_span": float(theta_span[2]),
        "total_joint_path_length": float(np.sum(joint_steps)) if len(joint_steps) > 0 else 0.0,
        "total_task_fk_path_length": float(ex66.path_cost(fk_trace)) if len(fk_trace) >= 2 else 0.0,
        "max_joint_step": float(np.max(joint_steps)) if len(joint_steps) > 0 else 0.0,
        "mean_joint_step": float(np.mean(joint_steps)) if len(joint_steps) > 0 else 0.0,
        "max_constraint_residual": float(np.max(residuals)) if len(residuals) > 0 else 0.0,
        "mean_constraint_residual": float(np.mean(residuals)) if len(residuals) > 0 else 0.0,
        "left_count": label_counts[ex66.LEFT_STAGE],
        "plane_count": label_counts[ex66.PLANE_STAGE],
        "right_count": label_counts[ex66.RIGHT_STAGE],
        "left_stage_max_residual": float(max(stage_residuals[ex66.LEFT_STAGE])) if stage_residuals[ex66.LEFT_STAGE] else 0.0,
        "plane_stage_max_residual": float(max(stage_residuals[ex66.PLANE_STAGE])) if stage_residuals[ex66.PLANE_STAGE] else 0.0,
        "right_stage_max_residual": float(max(stage_residuals[ex66.RIGHT_STAGE])) if stage_residuals[ex66.RIGHT_STAGE] else 0.0,
        "stage_order_valid": bool(_stage_order_is_monotone(labels)),
        "cspace_path_source": "dense_theta_path",
        "taskspace_trace_source": "FK(dense_theta_path)",
        "final_route_stored_evidence_edges": int(result.graph_route_edges),
        "final_route_projected_jointspace_edges": 0,
        "final_route_taskspace_edges": 0,
        **transition,
        "selected_left_plane_transition_index": int(getattr(result, "selected_left_plane_transition_index", transition["selected_left_plane_transition_index"])),
        "selected_plane_right_transition_index": int(getattr(result, "selected_plane_right_transition_index", transition["selected_plane_right_transition_index"])),
        "transition_stack_certified": bool(getattr(result, "transition_stack_certified", transition["max_transition_stack_residual"] <= 1.0e-3)),
    }


def _certify_joint_path_with_labels(
    joint_path: np.ndarray,
    labels: list[str],
    manifolds: dict[str, object],
    robot: SpatialRobot3DOF,
    obstacles: list[object],
    max_joint_step: float,
    constraint_tol: float = 2e-3,
) -> dict[str, object]:
    q_path = np.asarray(joint_path, dtype=float)
    residuals: list[float] = []
    collision_free = True
    for idx, q in enumerate(q_path):
        stage = labels[idx] if idx < len(labels) else ""
        manifold = manifolds.get(stage)
        residual = float("inf") if manifold is None else float(np.linalg.norm(manifold.residual(q)))
        if manifold is None or not bool(manifold.within_bounds(q, tol=constraint_tol)):
            residual = max(float(residual), 10.0 * float(constraint_tol))
        if configuration_in_collision(robot, q, obstacles):
            collision_free = False
            residual = max(float(residual), 100.0 * float(constraint_tol))
        residuals.append(float(residual))
    residual_arr = np.asarray(residuals, dtype=float)
    joint_steps, max_step, mean_step, worst_step = joint_step_statistics(q_path)
    max_residual = float(np.max(residual_arr)) if len(residual_arr) > 0 else float("inf")
    mean_residual = float(np.mean(residual_arr)) if len(residual_arr) > 0 else float("inf")
    constraint_ok = bool(collision_free and max_residual <= float(constraint_tol))
    joint_ok = bool(max_step <= float(max_joint_step) + 1e-9)
    return {
        "certified": bool(constraint_ok and joint_ok),
        "constraint_certified": bool(constraint_ok),
        "joint_continuity_certified": bool(joint_ok),
        "collision_free": bool(collision_free),
        "residuals": residual_arr,
        "joint_steps": np.asarray(joint_steps, dtype=float),
        "max_constraint_residual": max_residual,
        "mean_constraint_residual": mean_residual,
        "max_joint_step": float(max_step),
        "mean_joint_step": float(mean_step),
        "worst_joint_step_index": int(worst_step),
    }


def _same_stage_window(labels: list[str], i: int, j: int) -> str | None:
    if i < 0 or j >= len(labels) or i >= j:
        return None
    stage = str(labels[i])
    if stage == "" or any(str(label) != stage for label in labels[i : j + 1]):
        return None
    return stage


def _try_projected_interpolation_connector(
    q0: np.ndarray,
    q1: np.ndarray,
    stage: str,
    manifolds: dict[str, object],
    obstacles: list[object],
    robot: SpatialRobot3DOF,
    max_joint_step: float,
) -> np.ndarray | None:
    manifold = manifolds.get(stage)
    if manifold is None:
        return None
    start = np.asarray(q0, dtype=float)
    goal = np.asarray(q1, dtype=float)
    delta = wrap_angles(goal - start)
    steps = max(2, int(np.ceil(float(np.linalg.norm(delta)) / max(float(max_joint_step), 1e-9))))
    path: list[np.ndarray] = [start.copy()]
    current = start.copy()
    for idx in range(1, steps + 1):
        guess = wrap_angles(start + (float(idx) / float(steps)) * delta)
        projection = manifold.project(guess, tol=1e-6, max_iters=40)
        if not projection.success:
            return None
        q_next = np.asarray(projection.x_projected, dtype=float)
        if not bool(manifold.within_bounds(q_next, tol=2e-3)):
            return None
        if configuration_in_collision(robot, q_next, obstacles):
            return None
        if float(np.linalg.norm(wrap_angles(q_next - current))) > float(max_joint_step) + 1e-9:
            return None
        path.append(q_next.copy())
        current = q_next
    if float(np.linalg.norm(wrap_angles(path[-1] - goal))) > max(1e-3, 0.25 * float(max_joint_step)):
        return None
    cert = _certify_joint_path_with_labels(
        np.asarray(path, dtype=float),
        [stage] * len(path),
        manifolds,
        robot,
        obstacles,
        max_joint_step=max_joint_step,
    )
    return np.asarray(path, dtype=float) if bool(cert["certified"]) else None


def _try_stage_connector(
    q0: np.ndarray,
    q1: np.ndarray,
    stage: str,
    manifolds: dict[str, object],
    obstacles: list[object],
    robot: SpatialRobot3DOF,
    max_joint_step: float,
    *,
    use_local_planner: bool,
) -> np.ndarray | None:
    cheap = _try_projected_interpolation_connector(q0, q1, stage, manifolds, obstacles, robot, max_joint_step)
    if cheap is not None:
        return cheap
    if not bool(use_local_planner):
        return None
    manifold = manifolds.get(stage)
    if manifold is None:
        return None
    collision_fn = lambda q: configuration_in_collision(robot, q, obstacles)
    result = explore_joint_manifold(
        manifold=manifold,
        start=np.asarray(q0, dtype=float),
        goal=np.asarray(q1, dtype=float),
        max_step=float(max_joint_step),
        local_max_joint_step=float(max_joint_step),
        collision_fn=collision_fn,
    )
    if not result.success:
        return None
    path = np.asarray(result.path, dtype=float)
    if len(path) < 2:
        return None
    if float(np.linalg.norm(wrap_angles(path[0] - q0))) > max(1e-3, 0.25 * float(max_joint_step)):
        return None
    if float(np.linalg.norm(wrap_angles(path[-1] - q1))) > max(1e-3, 0.25 * float(max_joint_step)):
        return None
    return path


def _replace_path_window(path: np.ndarray, labels: list[str], i: int, j: int, replacement: np.ndarray, stage: str) -> tuple[np.ndarray, list[str]]:
    repl = np.asarray(replacement, dtype=float)
    if len(repl) >= 2 and float(np.linalg.norm(wrap_angles(repl[0] - path[i]))) <= 1e-8:
        repl = repl[1:]
    if len(repl) >= 1 and float(np.linalg.norm(wrap_angles(repl[-1] - path[j]))) <= 1e-8:
        repl = repl[:-1]
    new_path = np.vstack([path[: i + 1], repl, path[j:]])
    new_labels = labels[: i + 1] + [stage] * len(repl) + labels[j:]
    return np.asarray(new_path, dtype=float), new_labels


def smooth_joint_path_constrained(
    joint_path: np.ndarray,
    stage_labels: list[str],
    manifolds: dict[str, object],
    robot: SpatialRobot3DOF,
    obstacles: list[object],
    *,
    max_iterations: int,
    smoothing_passes: int,
    max_joint_step: float,
    joint_weight: float,
    task_weight: float,
    curvature_weight: float,
    preserve_transitions: bool = True,
    smoothing_time_limit: float = 5.0,
    max_connector_calls: int = 25,
    use_local_planner: bool = False,
) -> tuple[np.ndarray, list[str], dict[str, object]]:
    _ = preserve_transitions
    path = np.asarray(joint_path, dtype=float)
    labels = list(stage_labels)
    attempts = 0
    accepts = 0
    connector_calls = 0
    start_time = time.perf_counter()
    time_limit = max(0.05, float(smoothing_time_limit))
    connector_limit = max(0, int(max_connector_calls))

    def elapsed() -> float:
        return float(time.perf_counter() - start_time)

    def timed_out() -> bool:
        return elapsed() >= time_limit

    def connector_budget_exhausted() -> bool:
        return connector_calls >= connector_limit

    def progress(force: bool = False) -> None:
        if force or attempts == 0 or attempts % 10 == 0:
            print(
                f"[smoothing] attempts={attempts}, accepts={accepts}, "
                f"current_nodes={len(path)}, connector_calls={connector_calls}, elapsed={elapsed():.1f}s",
                flush=True,
            )

    def cost(window: np.ndarray) -> float:
        return _route_quality_cost(window, robot, joint_weight, task_weight, curvature_weight)

    # Deterministic local simplification: remove a middle sample when the
    # same-stage constrained connector is cheaper and re-certified.
    changed = True
    passes = 0
    while changed and passes < max(1, int(smoothing_passes)) and not timed_out() and not connector_budget_exhausted():
        changed = False
        passes += 1
        idx = 1
        while idx < len(path) - 1 and not timed_out() and not connector_budget_exhausted():
            stage = _same_stage_window(labels, idx - 1, idx + 1)
            if stage is None:
                idx += 1
                continue
            attempts += 1
            connector_calls += 1
            progress()
            replacement = _try_stage_connector(
                path[idx - 1],
                path[idx + 1],
                stage,
                manifolds,
                obstacles,
                robot,
                max_joint_step,
                use_local_planner=use_local_planner,
            )
            if replacement is None:
                idx += 1
                continue
            old = path[idx - 1 : idx + 2]
            if cost(replacement) + 1e-9 < cost(old):
                path, labels = _replace_path_window(path, labels, idx - 1, idx + 1, replacement, stage)
                accepts += 1
                changed = True
                idx = max(1, idx - 1)
            else:
                idx += 1

    # Randomized shortcutting within stage-homogeneous windows.
    rng = np.random.default_rng(12345)
    for _iter in range(max(0, int(max_iterations))):
        if len(path) < 4 or timed_out() or connector_budget_exhausted():
            break
        i = int(rng.integers(0, len(path) - 2))
        j = int(rng.integers(i + 2, len(path)))
        stage = _same_stage_window(labels, i, j)
        if stage is None:
            continue
        attempts += 1
        connector_calls += 1
        progress()
        replacement = _try_stage_connector(
            path[i],
            path[j],
            stage,
            manifolds,
            obstacles,
            robot,
            max_joint_step,
            use_local_planner=use_local_planner,
        )
        if replacement is None:
            continue
        old = path[i : j + 1]
        if cost(replacement) + 1e-9 < cost(old):
            path, labels = _replace_path_window(path, labels, i, j, replacement, stage)
            accepts += 1

    # Light tangent-space fairing, still projected and certified per stage.
    for _pass in range(max(0, int(smoothing_passes))):
        if timed_out():
            break
        for idx in range(1, len(path) - 1):
            if timed_out():
                break
            stage = _same_stage_window(labels, idx - 1, idx + 1)
            if stage is None:
                continue
            manifold = manifolds.get(stage)
            if manifold is None:
                continue
            current_window = path[idx - 1 : idx + 2]
            for alpha in (0.15, 0.25, 0.35):
                prev_delta = wrap_angles(path[idx - 1] - path[idx])
                next_delta = wrap_angles(path[idx + 1] - path[idx])
                guess = wrap_angles(path[idx] + float(alpha) * (0.5 * (prev_delta + next_delta)))
                projection = manifold.project(guess, tol=1e-6, max_iters=80)
                if not projection.success:
                    continue
                q_new = np.asarray(projection.x_projected, dtype=float)
                candidate_window = np.vstack([path[idx - 1], q_new, path[idx + 1]])
                cert = _certify_joint_path_with_labels(
                    candidate_window,
                    [stage, stage, stage],
                    manifolds,
                    robot,
                    obstacles,
                    max_joint_step=max_joint_step,
                )
                attempts += 1
                if bool(cert["certified"]) and cost(candidate_window) + 1e-9 < cost(current_window):
                    path[idx] = q_new
                    accepts += 1
                    break

    progress(force=True)
    return np.asarray(path, dtype=float), labels, {
        "attempts": attempts,
        "accepts": accepts,
        "connector_calls": connector_calls,
        "elapsed": elapsed(),
        "timed_out": timed_out(),
        "connector_budget_exhausted": connector_budget_exhausted(),
    }


def apply_jointspace_route_smoothing(
    result: ex66.FixedPlaneRoute,
    families,
    robot: SpatialRobot3DOF,
    obstacles: list[object],
    *,
    enabled: bool,
    max_iterations: int,
    smoothing_passes: int,
    max_joint_step: float,
    joint_weight: float,
    task_weight: float,
    curvature_weight: float,
    preserve_transitions: bool,
    smoothing_time_limit: float,
    max_connector_calls: int,
    use_local_planner: bool,
) -> JointRouteSmoothingResult:
    smoothing_started = time.perf_counter()
    original_path = np.asarray(getattr(result, "dense_joint_path", np.zeros((0, 3), dtype=float)), dtype=float)
    original_labels = list(getattr(result, "dense_joint_path_stage_labels", []))
    dense_before = int(len(original_path))
    joint_before = _joint_path_length(original_path)
    task_before = _task_path_length(robot, original_path)
    curvature_before = _curvature_cost(original_path)
    if not enabled or not bool(getattr(result, "dense_joint_path_execution_certified", False)) or len(original_path) < 3:
        return JointRouteSmoothingResult(
            enabled=bool(enabled),
            certified=bool(getattr(result, "dense_joint_path_execution_certified", False)),
            fallback_used=False,
            attempts=0,
            accepts=0,
            nodes_before=len(original_path),
            nodes_after=len(original_path),
            dense_points_before=dense_before,
            dense_points_after=dense_before,
            joint_length_before=joint_before,
            joint_length_after=joint_before,
            task_length_before=task_before,
            task_length_after=task_before,
            curvature_cost_before=curvature_before,
            curvature_cost_after=curvature_before,
            max_joint_step_after=float(getattr(result, "dense_joint_path_max_joint_step", 0.0)),
            max_constraint_residual_after=float(np.max(getattr(result, "dense_joint_path_constraint_residuals", np.zeros(0, dtype=float))) if len(getattr(result, "dense_joint_path_constraint_residuals", [])) > 0 else 0.0),
            mean_constraint_residual_after=float(np.mean(getattr(result, "dense_joint_path_constraint_residuals", np.zeros(0, dtype=float))) if len(getattr(result, "dense_joint_path_constraint_residuals", [])) > 0 else 0.0),
            collision_free_after=True,
            message="smoothing disabled or no certified dense joint path available",
        )

    print(
        "[smoothing] starting certified route smoothing: "
        f"iters={int(max_iterations)}, passes={int(smoothing_passes)}, "
        f"time_limit={float(smoothing_time_limit):.1f}s, max_connector_calls={int(max_connector_calls)}, "
        f"use_local_planner={bool(use_local_planner)}",
        flush=True,
    )
    manifolds = _stage_manifolds_for_robot(families, robot)
    smoothed_path, smoothed_labels, stats = smooth_joint_path_constrained(
        original_path,
        original_labels,
        manifolds,
        robot,
        obstacles,
        max_iterations=max_iterations,
        smoothing_passes=smoothing_passes,
        max_joint_step=max_joint_step,
        joint_weight=joint_weight,
        task_weight=task_weight,
        curvature_weight=curvature_weight,
        preserve_transitions=preserve_transitions,
        smoothing_time_limit=smoothing_time_limit,
        max_connector_calls=max_connector_calls,
        use_local_planner=use_local_planner,
    )
    cert = _certify_joint_path_with_labels(
        smoothed_path,
        smoothed_labels,
        manifolds,
        robot,
        obstacles,
        max_joint_step=max_joint_step,
    )
    joint_after = _joint_path_length(smoothed_path)
    task_after = _task_path_length(robot, smoothed_path)
    curvature_after = _curvature_cost(smoothed_path)
    old_cost = _route_quality_cost(original_path, robot, joint_weight, task_weight, curvature_weight)
    new_cost = _route_quality_cost(smoothed_path, robot, joint_weight, task_weight, curvature_weight)
    accepted = bool(cert["certified"] and new_cost <= old_cost + 1e-9)
    if accepted:
        task_route = jointspace_display_route_from_dense(
            type("_Route", (), {"dense_joint_path": smoothed_path, "dense_joint_path_execution_certified": True})(),
            robot,
        )
        result.dense_joint_path = np.asarray(smoothed_path, dtype=float)
        result.dense_joint_path_stage_labels = list(smoothed_labels)
        result.dense_joint_path_constraint_residuals = np.asarray(cert["residuals"], dtype=float)
        result.dense_joint_path_is_certified = bool(cert["constraint_certified"])
        result.dense_joint_path_joint_steps = np.asarray(cert["joint_steps"], dtype=float)
        result.dense_joint_path_max_joint_step = float(cert["max_joint_step"])
        result.dense_joint_path_mean_joint_step = float(cert["mean_joint_step"])
        result.dense_joint_path_worst_joint_step_index = int(cert["worst_joint_step_index"])
        result.dense_joint_path_execution_certified = bool(cert["certified"])
        result.dense_joint_path_constraint_certified = bool(cert["constraint_certified"])
        result.dense_joint_path_joint_continuity_certified = bool(cert["joint_continuity_certified"])
        result.dense_joint_path_message = (
            "smoothed dense joint path execution certified: "
            f"points={len(smoothed_path)}, max_residual={float(cert['max_constraint_residual']):.4g}, "
            f"mean_residual={float(cert['mean_constraint_residual']):.4g}, "
            f"max_joint_step={float(cert['max_joint_step']):.4g}, collision_free={bool(cert['collision_free'])}"
        )
        result.path = np.asarray(task_route, dtype=float)
        result.raw_path = np.asarray(task_route, dtype=float)
        result.joint_path = np.asarray(smoothed_path, dtype=float)
        result.certified_path_points = int(len(task_route))
        result.display_path_points = int(len(task_route))
        result.route_cost_raw = float(ex66.path_cost(task_route))
        result.route_cost_display = float(ex66.path_cost(task_route))

    final_path = smoothed_path if accepted else original_path
    final_cert = cert if accepted else _certify_joint_path_with_labels(
        original_path,
        original_labels,
        manifolds,
        robot,
        obstacles,
        max_joint_step=max_joint_step,
    )
    stop_reason = "completed"
    if bool(stats.get("timed_out", False)):
        stop_reason = "time_limit_reached"
    elif bool(stats.get("connector_budget_exhausted", False)):
        stop_reason = "connector_budget_exhausted"
    summary = JointRouteSmoothingResult(
        enabled=True,
        certified=bool(accepted),
        fallback_used=not accepted,
        attempts=int(stats["attempts"]),
        accepts=int(stats["accepts"]) if accepted else 0,
        nodes_before=len(original_path),
        nodes_after=len(final_path),
        dense_points_before=dense_before,
        dense_points_after=len(final_path),
        joint_length_before=joint_before,
        joint_length_after=_joint_path_length(final_path),
        task_length_before=task_before,
        task_length_after=_task_path_length(robot, final_path),
        curvature_cost_before=curvature_before,
        curvature_cost_after=_curvature_cost(final_path),
        max_joint_step_after=float(final_cert["max_joint_step"]),
        max_constraint_residual_after=float(final_cert["max_constraint_residual"]),
        mean_constraint_residual_after=float(final_cert["mean_constraint_residual"]),
        collision_free_after=bool(final_cert["collision_free"]),
        message=(
            f"smoothed path accepted and certified ({stop_reason})"
            if accepted
            else f"smoothing failed certification or did not improve cost; original certified route retained ({stop_reason})"
        ),
    )
    print(
        "[smoothing] finished: "
        f"certified={summary.certified}, fallback={summary.fallback_used}, "
        f"attempts={summary.attempts}, accepts={summary.accepts}, "
        f"elapsed={time.perf_counter() - smoothing_started:.1f}s",
        flush=True,
    )
    return summary


def print_jointspace_smoothing_block(summary: JointRouteSmoothingResult | None) -> None:
    if summary is None:
        return
    print_block(
        "Joint-space route smoothing",
        {
            "smoothing_enabled": bool(summary.enabled),
            "smoothing_attempts": int(summary.attempts),
            "smoothing_accepts": int(summary.accepts),
            "nodes_before_smoothing": int(summary.nodes_before),
            "nodes_after_smoothing": int(summary.nodes_after),
            "dense_points_before_smoothing": int(summary.dense_points_before),
            "dense_points_after_smoothing": int(summary.dense_points_after),
            "joint_length_before": float(summary.joint_length_before),
            "joint_length_after": float(summary.joint_length_after),
            "task_length_before": float(summary.task_length_before),
            "task_length_after": float(summary.task_length_after),
            "curvature_cost_before": float(summary.curvature_cost_before),
            "curvature_cost_after": float(summary.curvature_cost_after),
            "max_joint_step_after": float(summary.max_joint_step_after),
            "max_constraint_residual_after": float(summary.max_constraint_residual_after),
            "mean_constraint_residual_after": float(summary.mean_constraint_residual_after),
            "collision_free_after": bool(summary.collision_free_after),
            "smoothing_certified": bool(summary.certified),
            "smoothing_fallback_used": bool(summary.fallback_used),
            "smoothing_message": summary.message,
        },
    )


def print_block(title: str, rows: dict[str, object]) -> None:
    print(f"\n=== {title} ===")
    for key, value in rows.items():
        if isinstance(value, float):
            print(f"{key:<27}: {value:.6f}")
        else:
            print(f"{key:<27}: {value}")


def mode_banner(mode: str, obstacles_enabled: bool, stop_after_first_solution: bool, joint_max_step: float) -> dict[str, object]:
    if mode == "jointspace_constrained_planning":
        planning_space = "joint_space"
        execution_method = "dense_joint_trajectory"
    elif mode == "compare_taskspace_vs_jointspace":
        planning_space = "task_space + joint_space"
        execution_method = "comparison"
    else:
        planning_space = "task_space"
        execution_method = "IK_tracking"
    return {
        "mode": mode,
        "planning_space": planning_space,
        "execution_method": execution_method,
        "obstacles_enabled": bool(obstacles_enabled),
        "continue_after_first_solution": not bool(stop_after_first_solution),
        "joint_max_step": float(joint_max_step),
    }


def print_taskspace_execution_block(robot_execution: RobotExecutionResult | None, route: np.ndarray) -> None:
    print_block(
        "Task-space IK execution",
        {
            "execution_success": False if robot_execution is None else robot_execution.execution_success,
            "target_route_points": int(len(route)),
            "execution_waypoints": 0 if robot_execution is None else len(robot_execution.joint_path),
            "max_tracking_error": 0.0 if robot_execution is None else robot_execution.max_tracking_error,
            "mean_tracking_error": 0.0 if robot_execution is None else robot_execution.mean_tracking_error,
            "max_joint_step": 0.0 if robot_execution is None else robot_execution.max_joint_step,
            "ik_failures": 0 if robot_execution is None else robot_execution.ik_failure_count,
            "execution_source": "disabled" if robot_execution is None else robot_execution.execution_source,
        },
    )


def print_jointspace_methodology_block(
    result: ex66.FixedPlaneRoute,
    robot_execution: RobotExecutionResult | None,
    cspace_audit: dict[str, object] | None = None,
) -> None:
    residuals = np.asarray(getattr(result, "dense_joint_path_constraint_residuals", np.zeros(0, dtype=float)), dtype=float)
    stack_values = [
        float(value)
        for key, value in result.mode_counts.items()
        if "best_stacked" in str(key) and "residual" in str(key) and np.isfinite(float(value))
    ]
    transition_stack_residual = (
        float(cspace_audit["max_transition_stack_residual"])
        if cspace_audit is not None and np.isfinite(float(cspace_audit["max_transition_stack_residual"]))
        else (max(stack_values) if stack_values else "not_recorded")
    )
    graph_route_used = bool(result.mode_counts.get("graph_route_used_for_execution", 1))
    dense_path = np.asarray(getattr(result, "dense_joint_path", np.zeros((0, 3), dtype=float)), dtype=float)
    print_block(
        "Joint-space methodology guardrails",
        {
            "planning_space": "joint_space",
            "state_variable": "theta=[yaw, shoulder, elbow]",
            "task_space_robot_mode_enabled": False,
            "task_space_planner_used": False,
            "ik_used_for_start_goal_only": True,
            "ik_waypoint_fallback_used": False,
            "task_space_route_reconstruction": False,
            "execution_path_source": "stored_dense_joint_edges" if graph_route_used else "projected_jointspace_local_edges",
            "visual_trace_source": "FK(dense_theta_path)",
            "graph_route_used_for_execution": graph_route_used,
            "dense_theta_points": int(len(dense_path)),
            "max_joint_step": 0.0 if robot_execution is None else float(robot_execution.max_joint_step),
            "max_active_constraint_residual": float(np.max(residuals)) if len(residuals) > 0 else 0.0,
            "max_transition_stack_residual": transition_stack_residual,
            "left_plane_transition_count": int(result.transition_hypotheses_left_plane),
            "plane_right_transition_count": int(result.transition_hypotheses_plane_right),
        },
    )


def print_cspace_trajectory_audit(audit: dict[str, object]) -> None:
    print_block(
        "C-space trajectory audit",
        {
            "cspace_path_source": audit["cspace_path_source"],
            "taskspace_trace_source": audit["taskspace_trace_source"],
            "theta_path_points": audit["theta_path_points"],
            "theta0_min": audit["theta0_min"],
            "theta0_max": audit["theta0_max"],
            "theta0_span": audit["theta0_span"],
            "theta1_min": audit["theta1_min"],
            "theta1_max": audit["theta1_max"],
            "theta1_span": audit["theta1_span"],
            "theta2_min": audit["theta2_min"],
            "theta2_max": audit["theta2_max"],
            "theta2_span": audit["theta2_span"],
            "total_joint_path_length": audit["total_joint_path_length"],
            "total_task_fk_path_length": audit["total_task_fk_path_length"],
            "max_joint_step": audit["max_joint_step"],
            "mean_joint_step": audit["mean_joint_step"],
            "max_constraint_residual": audit["max_constraint_residual"],
            "mean_constraint_residual": audit["mean_constraint_residual"],
            "left_count": audit["left_count"],
            "plane_count": audit["plane_count"],
            "right_count": audit["right_count"],
            "left_stage_max_residual": audit["left_stage_max_residual"],
            "plane_stage_max_residual": audit["plane_stage_max_residual"],
            "right_stage_max_residual": audit["right_stage_max_residual"],
            "stage_order_valid": audit["stage_order_valid"],
            "selected_left_plane_transition_index": audit["selected_left_plane_transition_index"],
            "selected_plane_right_transition_index": audit["selected_plane_right_transition_index"],
            "selected_left_plane_stack_residual": audit["selected_left_plane_stack_residual"],
            "selected_plane_right_stack_residual": audit["selected_plane_right_stack_residual"],
            "max_transition_stack_residual": audit["max_transition_stack_residual"],
            "transition_stack_certified": audit["transition_stack_certified"],
            "final_route_stored_evidence_edges": audit["final_route_stored_evidence_edges"],
            "final_route_projected_jointspace_edges": audit["final_route_projected_jointspace_edges"],
            "final_route_taskspace_edges": audit["final_route_taskspace_edges"],
        },
    )


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def save_cspace_debug_artifacts(
    audit: dict[str, object],
    *,
    result: ex66.FixedPlaneRoute,
    output_root: Path | None = None,
) -> Path:
    # TODO(jointspace_method): replace with the generic debug_artifacts helper
    # after preserving the current Example 66 file names and summary fields.
    base = output_root or Path("outputs") / "ex66_jointspace_debug" / "latest"
    base.mkdir(parents=True, exist_ok=True)
    theta_path = np.asarray(audit["theta_path"], dtype=float)
    fk_trace = np.asarray(audit["fk_trace"], dtype=float)
    residuals = np.asarray(audit["constraint_residuals"], dtype=float)
    joint_steps = np.asarray(audit["joint_steps"], dtype=float)
    labels = list(audit["stage_labels"])

    # Save theta first because it is the executable joint-space route.
    np.save(base / "dense_theta_path.npy", theta_path)
    # Save FK(theta) only as the visualization/debug trace.
    np.save(base / "dense_fk_trace.npy", fk_trace)
    np.save(base / "constraint_residuals.npy", residuals)
    np.save(base / "joint_steps.npy", joint_steps)
    (base / "dense_stage_labels.txt").write_text("\n".join(labels), encoding="utf-8")
    (base / "dense_stage_labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "planning_space": "joint_space",
        "state_variable": "theta=[yaw, shoulder, elbow]",
        "execution_path_source": "stored_dense_joint_edges",
        "visual_trace_source": "FK(dense_theta_path)",
        "task_space_planner_used": False,
        "ik_waypoint_fallback_used": False,
        "task_space_route_reconstruction": False,
        "dense_joint_path_execution_certified": bool(getattr(result, "dense_joint_path_execution_certified", False)),
        "max_constraint_residual": audit["max_constraint_residual"],
        "max_joint_step": audit["max_joint_step"],
        "total_joint_path_length": audit["total_joint_path_length"],
        "total_task_fk_path_length": audit["total_task_fk_path_length"],
        "left_stage_max_residual": audit["left_stage_max_residual"],
        "plane_stage_max_residual": audit["plane_stage_max_residual"],
        "right_stage_max_residual": audit["right_stage_max_residual"],
        "selected_left_plane_stack_residual": audit["selected_left_plane_stack_residual"],
        "selected_plane_right_stack_residual": audit["selected_plane_right_stack_residual"],
        "max_transition_stack_residual": audit["max_transition_stack_residual"],
        "transition_stack_certified": audit["transition_stack_certified"],
        "selected_left_plane_transition_index": audit["selected_left_plane_transition_index"],
        "selected_plane_right_transition_index": audit["selected_plane_right_transition_index"],
        "final_route_taskspace_edges": audit["final_route_taskspace_edges"],
    }
    (base / "cspace_summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if len(theta_path) > 0:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(theta_path[:, 0], label="theta0")
            ax.plot(theta_path[:, 1], label="theta1")
            ax.plot(theta_path[:, 2], label="theta2")
            ax.set_xlabel("waypoint index")
            ax.set_ylabel("theta [rad]")
            ax.set_title("Dense theta trajectory")
            ax.legend()
            fig.tight_layout()
            fig.savefig(base / "theta_vs_index.png", dpi=160)
            plt.close(fig)

            fig = plt.figure(figsize=(6, 5.5))
            ax3 = fig.add_subplot(111, projection="3d")
            ax3.plot(theta_path[:, 0], theta_path[:, 1], theta_path[:, 2], color="#d32f2f", linewidth=2.0)
            ax3.scatter(theta_path[0, 0], theta_path[0, 1], theta_path[0, 2], color="black", s=40, label="start")
            ax3.scatter(theta_path[-1, 0], theta_path[-1, 1], theta_path[-1, 2], color="gold", s=50, label="goal")
            for key, color, label in (
                ("selected_left_plane_transition_index", "#ff7043", "left-plane transition"),
                ("selected_plane_right_transition_index", "#26a69a", "plane-right transition"),
            ):
                idx = int(audit.get(key, -1))
                if 0 <= idx < len(theta_path):
                    ax3.scatter(theta_path[idx, 0], theta_path[idx, 1], theta_path[idx, 2], color=color, s=45, label=label)
            ax3.set_xlabel("theta0")
            ax3.set_ylabel("theta1")
            ax3.set_zlabel("theta2")
            ax3.set_title("C-space dense path")
            ax3.legend()
            fig.tight_layout()
            fig.savefig(base / "cspace_3d_path.png", dpi=160)
            plt.close(fig)

        if len(residuals) > 0:
            fig, ax = plt.subplots(figsize=(8, 4.2))
            ax.plot(residuals, color="#455a64")
            ax.set_xlabel("waypoint index")
            ax.set_ylabel("active constraint residual")
            ax.set_title("Constraint residual along dense theta path")
            fig.tight_layout()
            fig.savefig(base / "residual_vs_index.png", dpi=160)
            plt.close(fig)
    except Exception as exc:
        (base / "plot_error.txt").write_text(str(exc), encoding="utf-8")

    return base


def assert_jointspace_methodology(
    *,
    result: ex66.FixedPlaneRoute,
    robot_execution: RobotExecutionResult | None,
    route_source: str,
    path_audit: dict[str, object],
    cspace_audit: dict[str, object],
) -> None:
    if result.success and not bool(getattr(result, "dense_joint_path_execution_certified", False)):
        raise RuntimeError("Planner reported success without an execution-certified dense theta path.")
    if robot_execution is not None and robot_execution.execution_source in {"taskspace_fallback", "taskspace_ik_tracking"}:
        raise RuntimeError("Task-space IK fallback/execution is forbidden in joint-space thesis mode.")
    if result.success and (robot_execution is None or robot_execution.execution_source != "certified_dense_joint_path"):
        raise RuntimeError("Successful joint-space thesis mode must execute the certified dense theta path.")
    if result.success and robot_execution is not None and not bool(robot_execution.planner_joint_path_used_directly):
        raise RuntimeError("Successful joint-space thesis mode must animate the dense theta path directly.")
    if result.success and robot_execution is not None and bool(robot_execution.planner_path_resampled_for_robot):
        raise RuntimeError("Successful joint-space thesis mode may not resample a task-space path for robot execution.")
    if route_source != "FK(result.dense_joint_path)" and result.success:
        raise RuntimeError(f"Unexpected visual route source in joint-space mode: {route_source}")
    if str(path_audit.get("display_route_source")) != "FK(result.dense_joint_path)" and result.success:
        raise RuntimeError("Display route is not the FK trace of the dense theta path.")
    if str(path_audit.get("robot_execution_source")) != "certified_dense_joint_path" and result.success:
        raise RuntimeError("Robot execution is not sourced from the certified dense theta path.")
    if float(path_audit.get("display_vs_trace_max_error", 0.0)) > 1.0e-6:
        raise RuntimeError("Displayed route and FK execution trace diverge.")
    if not bool(cspace_audit.get("stage_order_valid", False)) and result.success:
        raise RuntimeError("Dense theta path stage labels are not monotone left->plane->right.")
    if int(cspace_audit.get("final_route_taskspace_edges", 0)) != 0:
        raise RuntimeError("Final route contains task-space edges in joint-space thesis mode.")
    if result.success and float(cspace_audit.get("max_constraint_residual", 0.0)) > 2.0e-3:
        raise RuntimeError("Dense theta path exceeds active constraint residual tolerance.")
    if result.success and float(cspace_audit.get("max_transition_stack_residual", float("inf"))) > 1.0e-3:
        raise RuntimeError("Selected transition theta stack residual exceeds thesis tolerance.")
    if result.success and not bool(cspace_audit.get("transition_stack_certified", False)):
        raise RuntimeError("Selected transition theta configurations are not stack-certified.")


def print_jointspace_exploration_block(result: ex66.FixedPlaneRoute) -> None:
    def diag_mean(prefix: str) -> float:
        count = int(result.mode_counts.get(f"{prefix}_count", 0))
        return float(result.mode_counts.get(f"{prefix}_sum", 0.0)) / max(count, 1)

    print_block(
        "Joint-space exploration",
        {
            "total_rounds": result.total_rounds,
            "candidate_evaluations": result.candidate_evaluations,
            "left_evidence_nodes": result.left_evidence_nodes,
            "plane_evidence_nodes": result.plane_evidence_nodes,
            "right_evidence_nodes": result.right_evidence_nodes,
            "left_evidence_edges": len(result.stage_evidence_edges.get(ex66.LEFT_STAGE, [])),
            "plane_evidence_edges": len(result.stage_evidence_edges.get(ex66.PLANE_STAGE, [])),
            "right_evidence_edges": len(result.stage_evidence_edges.get(ex66.RIGHT_STAGE, [])),
            "entry_transitions_found": result.transition_hypotheses_left_plane,
            "exit_transitions_found": result.transition_hypotheses_plane_right,
            "first_solution_round": result.first_solution_round,
            "best_solution_round": result.best_solution_round,
            "continued_after_solution": result.continued_after_first_solution,
            "route_candidates_evaluated": result.alternative_hypothesis_pairs_evaluated,
            "route_candidates_built": int(result.mode_counts.get("route_candidates_evaluated", 0)),
            "route_candidates_constraint_certified": int(result.mode_counts.get("route_candidates_constraint_certified", 0)),
            "route_candidates_execution_certified": int(result.mode_counts.get("route_candidates_execution_certified", 0)),
            "route_candidates_rejected_joint_jump": int(result.mode_counts.get("route_candidates_rejected_joint_jump", 0)),
            "route_candidates_realized_by_local_replan": int(result.mode_counts.get("route_candidates_realized_by_local_replan", 0)),
            "missing_left_graph_path": int(result.mode_counts.get("route_candidates_missing_left_graph_path", 0)),
            "missing_plane_graph_path": int(result.mode_counts.get("route_candidates_missing_plane_graph_path", 0)),
            "missing_right_graph_path": int(result.mode_counts.get("route_candidates_missing_right_graph_path", 0)),
            "dense_edge_path_missing": int(result.mode_counts.get("route_candidates_dense_edge_path_missing", 0)),
            "sparse_only_graph_path": int(result.mode_counts.get("route_candidates_sparse_only_graph_path", 0)),
            "local_connector_attempts": int(result.mode_counts.get("route_candidates_local_replan_attempted", 0)),
            "local_connector_failed_left": int(result.mode_counts.get("route_candidates_local_connector_failed_left", 0)),
            "local_connector_failed_plane": int(result.mode_counts.get("route_candidates_local_connector_failed_plane", 0)),
            "local_connector_failed_right": int(result.mode_counts.get("route_candidates_local_connector_failed_right", 0)),
            "query_connector_routes_built": int(result.mode_counts.get("route_candidates_query_connectors_used", 0)),
            "left_update_attempts": int(result.mode_counts.get("left_update_attempts", 0)),
            "plane_update_attempts": int(result.mode_counts.get("plane_update_attempts", 0)),
            "right_update_attempts": int(result.mode_counts.get("right_update_attempts", 0)),
            "left_update_successes": int(result.mode_counts.get("left_update_successes", 0)),
            "plane_update_successes": int(result.mode_counts.get("plane_update_successes", 0)),
            "right_update_successes": int(result.mode_counts.get("right_update_successes", 0)),
            "plane_projection_success_count": int(result.mode_counts.get("plane_projection_success_count", 0)),
            "plane_projection_failure_count": int(result.mode_counts.get("plane_projection_failure_count", 0)),
            "plane_local_motion_success_count": int(result.mode_counts.get("plane_local_motion_success_count", 0)),
            "plane_local_motion_failure_count": int(result.mode_counts.get("plane_local_motion_failure_count", 0)),
            "plane_targeted_proposals_generated": int(result.mode_counts.get("plane_targeted_proposals_generated", 0)),
            "plane_frontier_count": int(result.mode_counts.get("plane_frontier_count", 0)),
            "local_continuation_success_count": int(result.mode_counts.get("local_continuation_success_count", 0)),
            "local_interpolation_success_count": int(result.mode_counts.get("local_interpolation_success_count", 0)),
            "local_planner_failure_count": int(result.mode_counts.get("local_planner_failure_count", 0)),
            "local_joint_jump_rejections": int(result.mode_counts.get("local_joint_jump_rejections", 0)),
            "transition_attempts_left_plane": int(result.mode_counts.get("transition_attempts_left_plane", 0)),
            "transition_attempts_plane_right": int(result.mode_counts.get("transition_attempts_plane_right", 0)),
            "transition_success_left_plane": int(result.mode_counts.get("transition_success_left_plane", 0)),
            "transition_success_plane_right": int(result.mode_counts.get("transition_success_plane_right", 0)),
            "explicit_plane_right_transition_attempts": int(result.mode_counts.get("explicit_plane_right_transition_attempts", 0)),
            "explicit_plane_right_transition_successes": int(result.mode_counts.get("explicit_plane_right_transition_successes", 0)),
            "left_stage_bad_edges": int(result.mode_counts.get("left_stage_bad_edges", 0)),
            "plane_stage_bad_edges": int(result.mode_counts.get("plane_stage_bad_edges", 0)),
            "right_stage_bad_edges": int(result.mode_counts.get("right_stage_bad_edges", 0)),
            "stage_edges_rejected_joint_jump": int(
                result.mode_counts.get("left_stage_edges_rejected_joint_jump", 0)
                + result.mode_counts.get("plane_stage_edges_rejected_joint_jump", 0)
                + result.mode_counts.get("right_stage_edges_rejected_joint_jump", 0)
            ),
            "max_bad_stage_edge_step": max(
                float(result.mode_counts.get("left_stage_max_bad_edge_step", 0.0)),
                float(result.mode_counts.get("plane_stage_max_bad_edge_step", 0.0)),
                float(result.mode_counts.get("right_stage_max_bad_edge_step", 0.0)),
            ),
        },
    )
    print_block(
        "Plane-right transition diagnostics",
        {
            "samples_total": int(result.mode_counts.get("plane_right_transition_samples_total", 0)),
            "current_manifold_invalid": int(result.mode_counts.get("plane_right_transition_current_manifold_invalid", 0)),
            "target_projection_failed": int(result.mode_counts.get("plane_right_transition_target_projection_failed", 0)),
            "target_collision": int(result.mode_counts.get("plane_right_transition_target_collision", 0)),
            "joint_distance_too_large": int(result.mode_counts.get("plane_right_transition_joint_distance_too_large", 0)),
            "task_distance_too_large": int(result.mode_counts.get("plane_right_transition_task_distance_too_large", 0)),
            "duplicate_rejected": int(result.mode_counts.get("plane_right_transition_duplicate_rejected", 0)),
            "accepted_hits": int(result.mode_counts.get("plane_right_transition_accepted_hits", 0)),
            "min_task_distance_to_target": float(result.mode_counts.get("plane_right_transition_target_projection_task_distance_min", 0.0)),
            "mean_task_distance_to_target": diag_mean("plane_right_transition_target_projection_task_distance"),
            "max_task_distance_to_target": float(result.mode_counts.get("plane_right_transition_target_projection_task_distance_max", 0.0)),
            "min_right_residual_before_proj": float(result.mode_counts.get("plane_right_transition_target_residual_before_projection_min", 0.0)),
            "min_joint_distance_to_target": float(result.mode_counts.get("plane_right_transition_joint_distance_source_to_target_min", 0.0)),
            "mean_joint_distance_to_target": diag_mean("plane_right_transition_joint_distance_source_to_target"),
            "max_joint_distance_to_target": float(result.mode_counts.get("plane_right_transition_joint_distance_source_to_target_max", 0.0)),
        },
    )


def print_dense_joint_certification_block(result: ex66.FixedPlaneRoute, robot_execution: RobotExecutionResult | None) -> None:
    residuals = np.asarray(getattr(result, "dense_joint_path_constraint_residuals", np.zeros(0, dtype=float)), dtype=float)
    labels = list(getattr(result, "dense_joint_path_stage_labels", []))
    worst_idx = int(np.argmax(residuals)) if len(residuals) > 0 else -1
    worst_stage = labels[worst_idx] if 0 <= worst_idx < len(labels) else "none"
    message = str(getattr(result, "dense_joint_path_message", ""))
    local_replan = bool(result.mode_counts.get("final_route_realization_selected_transition_local_replan", 0))
    graph_route_used = bool(result.mode_counts.get("graph_route_used_for_execution", 1))
    final_realization = "selected_transition_local_replan" if local_replan else (
        "stored_dense_joint_edges" if graph_route_used else "none"
    )
    print_block(
        "Dense joint route certification",
        {
            "final_route_realization": final_realization,
            "graph_route_used_for_execution": graph_route_used,
            "execution_path_source": "stored_dense_joint_edges" if graph_route_used else "projected_jointspace_local_edges",
            "dense_joint_path_points": int(len(getattr(result, "dense_joint_path", []))),
            "dense_joint_path_is_certified": bool(getattr(result, "dense_joint_path_is_certified", False)),
            "dense_joint_path_constraint_certified": bool(
                getattr(result, "dense_joint_path_constraint_certified", getattr(result, "dense_joint_path_is_certified", False))
            ),
            "dense_joint_path_joint_continuity_certified": bool(getattr(result, "dense_joint_path_joint_continuity_certified", False)),
            "dense_joint_path_execution_certified": bool(getattr(result, "dense_joint_path_execution_certified", False)),
            "max_dense_constraint_residual": float(np.max(residuals)) if len(residuals) > 0 else 0.0,
            "mean_dense_constraint_residual": float(np.mean(residuals)) if len(residuals) > 0 else 0.0,
            "dense_joint_path_max_joint_step": float(getattr(result, "dense_joint_path_max_joint_step", 0.0)),
            "dense_joint_path_mean_joint_step": float(getattr(result, "dense_joint_path_mean_joint_step", 0.0)),
            "dense_joint_path_worst_joint_step_index": int(getattr(result, "dense_joint_path_worst_joint_step_index", -1)),
            "worst_dense_stage": worst_stage,
            "collision_free": "collision_free=True" in message,
            "dense_joint_path_message": message,
        },
    )


def run_planner_and_execution(
    *,
    mode: str,
    families,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    robot: SpatialRobot3DOF,
    serial: bool,
    obstacles: list[object],
    validate_joint_execution: bool,
    allow_uncertified_joint_animation: bool,
    allow_taskspace_fallback: bool,
    joint_max_step: float,
    smooth_final_route: bool,
    smoothing_iters: int,
    smoothing_passes: int,
    smoothing_preserve_transitions: bool,
    smoothing_cost_joint_weight: float,
    smoothing_cost_task_weight: float,
    smoothing_cost_curvature_weight: float,
    smoothing_time_limit: float,
    smoothing_max_connector_calls: int,
    smoothing_use_local_planner: bool,
) -> tuple[ex66.FixedPlaneRoute, RobotExecutionResult | None, JointRouteSmoothingResult | None]:
    if mode == "jointspace_constrained_planning":
        # Thesis branch: plan in C-space against FK-pulled-back manifolds.
        print("[planner] starting joint-space evidence planning...", flush=True)
        result = ex66.plan_fixed_manifold_multimodal_route(
            families=families,
            start_q=start_q,
            goal_q=goal_q,
            robot=robot,
            serial_mode=serial,
            obstacles=obstacles,
            joint_max_step=joint_max_step,
        )
        print(
            "[planner] finished: "
            f"success={result.success}, rounds={result.total_rounds}, "
            f"dense_points={len(getattr(result, 'dense_joint_path', []))}, "
            f"route_points={len(result.path if len(result.path) >= 2 else result.raw_path)}",
            flush=True,
        )
        smoothing_summary = (
            apply_jointspace_route_smoothing(
                result,
                families,
                robot,
                obstacles,
                enabled=bool(smooth_final_route),
                max_iterations=int(smoothing_iters),
                smoothing_passes=int(smoothing_passes),
                max_joint_step=float(joint_max_step),
                joint_weight=float(smoothing_cost_joint_weight),
                task_weight=float(smoothing_cost_task_weight),
                curvature_weight=float(smoothing_cost_curvature_weight),
                preserve_transitions=bool(smoothing_preserve_transitions),
                smoothing_time_limit=float(smoothing_time_limit),
                max_connector_calls=int(smoothing_max_connector_calls),
                use_local_planner=bool(smoothing_use_local_planner),
            )
            if result.success
            else None
        )
        execution = (
            # Robot execution uses result.dense_joint_path directly.
            build_robot_execution(
                result,
                robot,
                use_planner_joint_path=True,
                families=families,
                start_q=start_q,
                goal_q=goal_q,
                validate_joint_execution=validate_joint_execution,
                allow_uncertified_joint_animation=allow_uncertified_joint_animation,
                allow_taskspace_fallback=allow_taskspace_fallback,
                joint_max_step=joint_max_step,
            )
            if result.success
            else None
        )
        return result, execution, smoothing_summary

    # Legacy branch: task-space route followed by IK, retained only for debugging.
    print("[planner] starting task-space evidence planning...", flush=True)
    result = ex66.plan_fixed_manifold_multimodal_route(
        families=families,
        start_q=start_q,
        goal_q=goal_q,
        serial_mode=serial,
    )
    print(
        "[planner] finished: "
        f"success={result.success}, rounds={result.total_rounds}, "
        f"route_points={len(result.path if len(result.path) >= 2 else result.raw_path)}",
        flush=True,
    )
    execution = build_robot_execution(result, robot, use_planner_joint_path=False) if result.success else None
    return result, execution, None


def print_comparison(
    task_result: ex66.FixedPlaneRoute,
    task_execution: RobotExecutionResult | None,
    joint_result: ex66.FixedPlaneRoute,
    joint_execution: RobotExecutionResult | None,
) -> None:
    print("\n=== Candidate route ranking ===")
    print("candidate_id | task_cost | robot_executable | max_tracking_error | max_joint_step | max_constraint_residual | selected | rejection_reason")
    rows = [
        ("taskspace_ik", task_result, task_execution),
        ("jointspace_dense", joint_result, joint_execution),
    ]
    executable = [
        (name, result, execution)
        for name, result, execution in rows
        if result.success and execution is not None and execution.execution_success
    ]
    selected_name = executable[0][0] if len(executable) > 0 else "none"
    for name, result, execution in rows:
        robot_executable = bool(result.success and execution is not None and execution.execution_success)
        rejection = "ok" if robot_executable else ("planner_failed" if not result.success else ("no_execution" if execution is None else execution.diagnostics))
        print(
            f"{name} | "
            f"{float(result.route_cost_raw):.6f} | "
            f"{robot_executable} | "
            f"{0.0 if execution is None else execution.max_tracking_error:.6f} | "
            f"{0.0 if execution is None else execution.max_joint_step:.6f} | "
            f"{0.0 if execution is None else execution.max_constraint_residual:.6f} | "
            f"{name == selected_name} | "
            f"{rejection}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Example 66.1: joint-space FK-constrained planning for a simple 3DOF robot."
    )
    # Arguments below choose between thesis joint-space planning and legacy/debug modes.
    parser.add_argument(
        "--quick-cspace-demo",
        action="store_true",
        help="Fast Example 66 preset: joint-space, no obstacles, modest rounds, exact C-space surfaces if certified.",
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--max-iters", type=int, default=None, help="Alias for --max-rounds.")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--serial", action="store_true")
    planning_mode = parser.add_mutually_exclusive_group()
    planning_mode.add_argument(
        "--legacy-taskspace-ik-demo",
        "--taskspace-planning",
        dest="planning_mode",
        action="store_const",
        const="taskspace_ik_execution",
        help="LEGACY DEBUG ONLY: run task-space planning then IK tracking. Not a joint-space planner.",
    )
    planning_mode.add_argument(
        "--jointspace-planning",
        dest="planning_mode",
        action="store_const",
        const="jointspace_constrained_planning",
        help="Plan directly in robot joint space with FK-pulled-back sphere/plane constraints.",
    )
    planning_mode.add_argument(
        "--compare-taskspace-jointspace",
        dest="planning_mode",
        action="store_const",
        const="compare_taskspace_vs_jointspace",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(planning_mode="jointspace_constrained_planning")
    first_solution_mode = parser.add_mutually_exclusive_group()
    first_solution_mode.add_argument("--continue-after-first-solution", action="store_true", default=False)
    first_solution_mode.add_argument("--stop-after-first-solution", action="store_true")
    exploration_mode = parser.add_mutually_exclusive_group()
    exploration_mode.add_argument("--show-exploration", dest="show_exploration", action="store_true")
    exploration_mode.add_argument("--hide-exploration", dest="show_exploration", action="store_false")
    parser.set_defaults(show_exploration=True)
    obstacle_mode = parser.add_mutually_exclusive_group()
    obstacle_mode.add_argument("--with-obstacles", dest="with_obstacles", action="store_true")
    obstacle_mode.add_argument("--without-obstacles", dest="with_obstacles", action="store_false")
    parser.set_defaults(with_obstacles=False)
    parser.add_argument(
        "--save-animation",
        default=None,
        help="Reserved for future offline animation export; current PyVista view remains interactive.",
    )
    parser.add_argument("--save-html", default=None, help="Reserved for future HTML export.")
    parser.add_argument("--debug-planner-parity", action="store_true")
    joint_execution_group = parser.add_mutually_exclusive_group()
    joint_execution_group.add_argument("--validate-joint-execution", dest="validate_joint_execution", action="store_true")
    joint_execution_group.add_argument("--no-validate-joint-execution", dest="validate_joint_execution", action="store_false")
    parser.set_defaults(validate_joint_execution=True)
    parser.add_argument(
        "--allow-uncertified-joint-animation",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--allow-taskspace-fallback",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--show-rejected-joint-interpolation",
        action="store_true",
        help="Debug only: draw the rejected FK trace for uncertified joint interpolation.",
    )
    parser.add_argument(
        "--joint-max-step",
        type=float,
        default=0.08,
        help="Maximum wrapped joint-space step allowed for certified local edges and robot execution.",
    )
    smoothing_mode = parser.add_mutually_exclusive_group()
    smoothing_mode.add_argument("--smooth-final-route", dest="smooth_final_route", action="store_true")
    smoothing_mode.add_argument("--no-smooth-final-route", dest="smooth_final_route", action="store_false")
    parser.set_defaults(smooth_final_route=False)
    parser.add_argument("--smoothing-iters", type=int, default=30)
    parser.add_argument("--smoothing-passes", type=int, default=2)
    parser.add_argument("--smoothing-time-limit", type=float, default=5.0)
    parser.add_argument("--smoothing-max-connector-calls", type=int, default=25)
    parser.add_argument(
        "--smoothing-use-local-planner",
        action="store_true",
        help="Opt in to expensive constrained local-planner shortcut attempts during route smoothing.",
    )
    smoothing_transition_mode = parser.add_mutually_exclusive_group()
    smoothing_transition_mode.add_argument("--smoothing-preserve-transitions", dest="smoothing_preserve_transitions", action="store_true")
    smoothing_transition_mode.add_argument("--no-smoothing-preserve-transitions", dest="smoothing_preserve_transitions", action="store_false")
    parser.set_defaults(smoothing_preserve_transitions=True)
    parser.add_argument("--smoothing-cost-joint-weight", type=float, default=1.0)
    parser.add_argument("--smoothing-cost-task-weight", type=float, default=0.25)
    parser.add_argument("--smoothing-cost-curvature-weight", type=float, default=0.05)
    parser.add_argument("--route-selection-top-k-for-smoothing", type=int, default=3)
    parser.add_argument(
        "--save-cspace-debug",
        action="store_true",
        help="Save dense theta path, FK trace, residuals, joint steps, summary JSON, and C-space plots.",
    )
    parser.add_argument(
        "--show-cspace",
        action="store_true",
        help="Show an additional theta-space PyVista view of the FK-pulled-back constraint surfaces.",
    )
    parser.add_argument("--cspace-presentation-style", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--cspace-grid-res",
        type=int,
        default=65,
        help="Grid resolution per joint axis for C-space implicit constraint surfaces.",
    )
    parser.add_argument("--cspace-no-surfaces", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cspace-route-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cspace-marker-scale", type=float, default=0.35, help="Scale C-space start/goal/transition marker sizes.")
    parser.add_argument("--cspace-opacity", type=float, default=0.28, help="Opacity for full C-space isosurface meshes.")
    parser.add_argument("--cspace-left-opacity", type=float, default=None, help="Opacity for the left C-space surface.")
    parser.add_argument("--cspace-middle-opacity", type=float, default=None, help="Opacity for the middle plane C-space surface.")
    parser.add_argument("--cspace-right-opacity", type=float, default=None, help="Opacity for the right C-space surface.")
    parser.add_argument("--cspace-middle-color", type=str, default=None, help="Color for the middle plane C-space surface.")
    parser.add_argument("--cspace-middle-only", action="store_true", help="Draw only the exact middle C-space isosurface plus the dense route and transition markers.")
    parser.add_argument("--save-cspace-surfaces", action="store_true", help="Save exact C-space isosurface meshes as .vtp files.")
    parser.add_argument("--cspace-allow-visual-proxy", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cspace-force-middle-sheet", action="store_true", help=argparse.SUPPRESS)
    cspace_smoothing_group = parser.add_mutually_exclusive_group()
    cspace_smoothing_group.add_argument("--cspace-smooth-surfaces", dest="cspace_smooth_surfaces", action="store_true", default=True)
    cspace_smoothing_group.add_argument("--cspace-no-smooth-surfaces", dest="cspace_smooth_surfaces", action="store_false")
    cspace_render_group = parser.add_mutually_exclusive_group()
    cspace_render_group.add_argument(
        "--cspace-safe-render",
        dest="cspace_safe_render",
        action="store_true",
        default=True,
        help="Use conservative C-space rendering; this is the default.",
    )
    cspace_render_group.add_argument(
        "--cspace-fancy-render",
        dest="cspace_safe_render",
        action="store_false",
        help="Allow prettier C-space rendering options that may trigger GPU/VTK shader warnings.",
    )
    vtk_warning_group = parser.add_mutually_exclusive_group()
    vtk_warning_group.add_argument(
        "--suppress-vtk-warnings",
        dest="suppress_vtk_warnings",
        action="store_true",
        default=True,
        help="Suppress VTK/OpenGL warning spam from the C-space renderer; this is the default.",
    )
    vtk_warning_group.add_argument(
        "--show-vtk-warnings",
        dest="suppress_vtk_warnings",
        action="store_false",
        help="Let VTK/OpenGL warnings print to the terminal.",
    )
    parser.add_argument(
        "--vtk-output-log",
        type=str,
        default="outputs/vtk_warnings.log",
        help="File for redirected native VTK/OpenGL stdout/stderr; use 'null' to discard.",
    )
    parser.add_argument(
        "--cspace-surface-mode",
        choices=("exact", "none"),
        default="exact",
        help="C-space surface mode: exact residual(theta)=0 isosurfaces, or none for route-only diagnostics.",
    )
    parser.add_argument(
        "--cspace-surface-style",
        choices=("points-outline", "points", "wireframe", "contour", "mesh"),
        default="mesh",
        help="Render style for exact extracted C-space isosurfaces.",
    )
    cspace_view_group = parser.add_mutually_exclusive_group()
    cspace_view_group.add_argument("--cspace-clean-view", dest="cspace_clean_view", action="store_true", default=True)
    cspace_view_group.add_argument("--cspace-box-view", dest="cspace_clean_view", action="store_false")
    cspace_quality = parser.add_mutually_exclusive_group()
    cspace_quality.add_argument("--cspace-lightweight", dest="cspace_lightweight", action="store_true", default=True)
    cspace_quality.add_argument("--cspace-full-surfaces", dest="cspace_lightweight", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()
    apply_quick_cspace_demo_preset(args)
    if bool(args.cspace_no_surfaces) or bool(args.cspace_route_only):
        args.cspace_surface_mode = "none"
    if bool(args.cspace_presentation_style) or bool(args.cspace_force_middle_sheet) or bool(args.cspace_allow_visual_proxy):
        print(
            "Deprecated C-space presentation/proxy flags are ignored; visualization uses exact residual(theta)=0 surfaces only.",
            flush=True,
        )

    if args.planning_mode == "jointspace_constrained_planning":
        if bool(args.allow_taskspace_fallback):
            parser.error("--allow-taskspace-fallback is forbidden in joint-space thesis mode.")
        if bool(args.allow_uncertified_joint_animation):
            parser.error("--allow-uncertified-joint-animation is forbidden in joint-space thesis mode.")
    if args.planning_mode == "taskspace_ik_execution":
        print(
            "WARNING: LEGACY DEBUG ONLY: task-space path followed by IK; not a joint-space planner.",
            flush=True,
        )
    if args.planning_mode == "compare_taskspace_vs_jointspace":
        print(
            "WARNING: LEGACY DEBUG ONLY: comparison mode includes task-space IK tracking and is not the thesis robot planner.",
            flush=True,
        )
    if bool(args.quick_cspace_demo):
        print_block(
            "Quick C-space demo preset",
            {
                "planning_mode": args.planning_mode,
                "with_obstacles": bool(args.with_obstacles),
                "seed": int(args.seed),
                "max_rounds": args.max_rounds if args.max_rounds is not None else args.max_iters,
                "fast_effort_guard": bool(args.fast),
                "joint_max_step": float(args.joint_max_step),
                "smooth_final_route": bool(args.smooth_final_route),
                "show_cspace": bool(args.show_cspace),
                "cspace_route_only": bool(args.cspace_route_only),
            },
        )

    np.random.seed(args.seed)
    ou.RNG.setSeed(args.seed)
    ou.setLogLevel(ou.LOG_ERROR)

    configure_example_66_budgets(args)

    # Task-space scene defines the supports; robot manifolds pull them back to theta.
    families, start_q, goal_q, plane_half_u, plane_half_v = build_example66_scene()
    active_obstacles = default_example_66_obstacles() if args.with_obstacles else []
    # Simple 3-DOF robot convention is theta=[yaw, shoulder, elbow].
    robot = SpatialRobot3DOF(
        link_lengths=np.asarray([1.35, 1.05, 0.75], dtype=float),
        base_world=np.asarray([0.0, -1.25, 0.10], dtype=float),
    )

    print_block(
        "Planning mode",
        mode_banner(
            args.planning_mode,
            bool(args.with_obstacles),
            bool(args.stop_after_first_solution),
            float(args.joint_max_step),
        ),
    )

    if args.planning_mode == "compare_taskspace_vs_jointspace":
        # Comparison mode is diagnostic; the selected thesis result remains the joint-space branch.
        np.random.seed(args.seed)
        task_result, task_execution, _task_smoothing = run_planner_and_execution(
            mode="taskspace_ik_execution",
            families=families,
            start_q=start_q,
            goal_q=goal_q,
            robot=robot,
            serial=bool(args.serial),
            obstacles=[],
            validate_joint_execution=bool(args.validate_joint_execution),
            allow_uncertified_joint_animation=bool(args.allow_uncertified_joint_animation),
            allow_taskspace_fallback=bool(args.allow_taskspace_fallback),
            joint_max_step=float(args.joint_max_step),
            smooth_final_route=False,
            smoothing_iters=int(args.smoothing_iters),
            smoothing_passes=int(args.smoothing_passes),
            smoothing_preserve_transitions=bool(args.smoothing_preserve_transitions),
            smoothing_cost_joint_weight=float(args.smoothing_cost_joint_weight),
            smoothing_cost_task_weight=float(args.smoothing_cost_task_weight),
            smoothing_cost_curvature_weight=float(args.smoothing_cost_curvature_weight),
            smoothing_time_limit=float(args.smoothing_time_limit),
            smoothing_max_connector_calls=int(args.smoothing_max_connector_calls),
            smoothing_use_local_planner=bool(args.smoothing_use_local_planner),
        )
        np.random.seed(args.seed)
        joint_result, joint_execution, joint_smoothing = run_planner_and_execution(
            mode="jointspace_constrained_planning",
            families=families,
            start_q=start_q,
            goal_q=goal_q,
            robot=robot,
            serial=bool(args.serial),
            obstacles=active_obstacles,
            validate_joint_execution=bool(args.validate_joint_execution),
            allow_uncertified_joint_animation=bool(args.allow_uncertified_joint_animation),
            allow_taskspace_fallback=bool(args.allow_taskspace_fallback),
            joint_max_step=float(args.joint_max_step),
            smooth_final_route=bool(args.smooth_final_route),
            smoothing_iters=int(args.smoothing_iters),
            smoothing_passes=int(args.smoothing_passes),
            smoothing_preserve_transitions=bool(args.smoothing_preserve_transitions),
            smoothing_cost_joint_weight=float(args.smoothing_cost_joint_weight),
            smoothing_cost_task_weight=float(args.smoothing_cost_task_weight),
            smoothing_cost_curvature_weight=float(args.smoothing_cost_curvature_weight),
            smoothing_time_limit=float(args.smoothing_time_limit),
            smoothing_max_connector_calls=int(args.smoothing_max_connector_calls),
            smoothing_use_local_planner=bool(args.smoothing_use_local_planner),
        )
        print_comparison(task_result, task_execution, joint_result, joint_execution)
        print_jointspace_exploration_block(joint_result)
        print_dense_joint_certification_block(joint_result, joint_execution)
        print_jointspace_smoothing_block(joint_smoothing)
        if joint_result.success and joint_execution is not None and joint_execution.execution_success:
            result, robot_execution = joint_result, joint_execution
            smoothing_summary = joint_smoothing
            planner_mode_label = "joint_space"
        else:
            result, robot_execution = task_result, task_execution
            smoothing_summary = None
            planner_mode_label = "task_space"
    else:
        result, robot_execution, smoothing_summary = run_planner_and_execution(
            mode=args.planning_mode,
            families=families,
            start_q=start_q,
            goal_q=goal_q,
            robot=robot,
            serial=bool(args.serial),
            obstacles=active_obstacles,
            validate_joint_execution=bool(args.validate_joint_execution),
            allow_uncertified_joint_animation=bool(args.allow_uncertified_joint_animation),
            allow_taskspace_fallback=bool(args.allow_taskspace_fallback),
            joint_max_step=float(args.joint_max_step),
            smooth_final_route=bool(args.smooth_final_route and args.planning_mode == "jointspace_constrained_planning"),
            smoothing_iters=int(args.smoothing_iters),
            smoothing_passes=int(args.smoothing_passes),
            smoothing_preserve_transitions=bool(args.smoothing_preserve_transitions),
            smoothing_cost_joint_weight=float(args.smoothing_cost_joint_weight),
            smoothing_cost_task_weight=float(args.smoothing_cost_task_weight),
            smoothing_cost_curvature_weight=float(args.smoothing_cost_curvature_weight),
            smoothing_time_limit=float(args.smoothing_time_limit),
            smoothing_max_connector_calls=int(args.smoothing_max_connector_calls),
            smoothing_use_local_planner=bool(args.smoothing_use_local_planner),
        )
        planner_mode_label = "joint_space" if args.planning_mode == "jointspace_constrained_planning" else "task_space"

    if planner_mode_label == "joint_space":
        enforce_dense_certified_success_for_jointspace(result)

    route, route_source = primary_display_route_for_mode(result, robot, planner_mode_label)
    path_audit = path_source_audit(result, robot_execution, robot, planner_mode_label)
    cspace_audit: dict[str, object] | None = None
    if planner_mode_label == "joint_space":
        cspace_audit = compute_cspace_trajectory_audit(result, robot, families)
        assert_jointspace_methodology(
            result=result,
            robot_execution=robot_execution,
            route_source=route_source,
            path_audit=path_audit,
            cspace_audit=cspace_audit,
        )

    print("\nExample 66.1: robot executing certified dense joint-space trajectory")
    print(
        "planner_mode = "
        + (
            "robot_jointspace_planning"
            if planner_mode_label == "joint_space"
            else "task_space_path_tracking"
        )
    )
    print(f"planner_success = {result.success}")
    print(f"planner_message = {result.message}")
    print(f"total_rounds = {result.total_rounds}")
    print(f"candidate_evaluations = {result.candidate_evaluations}")
    print(f"left_evidence_nodes = {result.left_evidence_nodes}")
    print(f"plane_evidence_nodes = {result.plane_evidence_nodes}")
    print(f"right_evidence_nodes = {result.right_evidence_nodes}")
    print(f"transition_hypotheses_left_plane = {result.transition_hypotheses_left_plane}")
    print(f"transition_hypotheses_plane_right = {result.transition_hypotheses_plane_right}")
    print(f"first_solution_round = {result.first_solution_round}")
    print(f"best_solution_round = {result.best_solution_round}")
    print(f"route_cost_raw = {result.route_cost_raw:.4f}")
    print(f"route_cost_display = {result.route_cost_display:.4f}")
    print(f"route_points = {len(route)}")
    print(f"route_source = {route_source}")
    print(f"robot_waypoints = {0 if robot_execution is None else len(robot_execution.joint_path)}")
    print(f"ik_success_count = {0 if robot_execution is None else robot_execution.ik_success_count}")
    print(f"ik_failure_count = {0 if robot_execution is None else robot_execution.ik_failure_count}")
    print(f"max_tracking_error = {0.0 if robot_execution is None else robot_execution.max_tracking_error:.4f}")
    print(f"mean_tracking_error = {0.0 if robot_execution is None else robot_execution.mean_tracking_error:.4f}")
    print(
        "planner_path_resampled_for_robot = "
        + str(False if robot_execution is None else robot_execution.planner_path_resampled_for_robot)
    )
    print(
        "planner_joint_path_used_directly = "
        + str(False if robot_execution is None else robot_execution.planner_joint_path_used_directly)
    )
    print(f"obstacle_count = {len(result.obstacles)}")
    print("replay_key = r")
    print_example66_failure_report(result, quick_mode=bool(args.quick_cspace_demo))
    if (
        planner_mode_label == "joint_space"
        and result.success
        and robot_execution is not None
        and not bool(robot_execution.execution_success)
        and str(robot_execution.execution_source).startswith("disabled")
    ):
        print(
            "Joint-space planner produced a route, but no certified smooth robot execution path is available. "
            "Robot animation disabled. Use --show-rejected-joint-interpolation to debug the rejected FK trace."
        )
    print_block("Planner parity check", planner_parity_stats(result, planner_mode_label, robot_execution))
    if planner_mode_label == "task_space":
        print_taskspace_execution_block(robot_execution, route)
    print_block(
        "Robot tracking diagnostics",
        {
            "execution_success": False if robot_execution is None else robot_execution.execution_success,
            "num_execution_waypoints": 0 if robot_execution is None else len(robot_execution.joint_path),
            "max_tracking_error": 0.0 if robot_execution is None else robot_execution.max_tracking_error,
            "mean_tracking_error": 0.0 if robot_execution is None else robot_execution.mean_tracking_error,
            "max_joint_step": 0.0 if robot_execution is None else robot_execution.max_joint_step,
            "ik_failures": 0 if robot_execution is None else robot_execution.ik_failure_count,
            "used_joint_path_directly": False if robot_execution is None else robot_execution.planner_joint_path_used_directly,
            "diagnostics": "no execution path" if robot_execution is None else robot_execution.diagnostics,
        },
    )
    if planner_mode_label == "joint_space":
        print_jointspace_methodology_block(result, robot_execution, cspace_audit)
        print_jointspace_exploration_block(result)
        print_dense_joint_certification_block(result, robot_execution)
        print_jointspace_smoothing_block(smoothing_summary)
        print_block("Path source audit", path_audit)
        if cspace_audit is not None:
            print_cspace_trajectory_audit(cspace_audit)
            if args.save_cspace_debug and result.success:
                debug_dir = save_cspace_debug_artifacts(cspace_audit, result=result)
                print_block(
                    "C-space debug artifacts",
                    {
                        "cspace_debug_dir": str(debug_dir),
                        "dense_theta_path_npy": str(debug_dir / "dense_theta_path.npy"),
                        "dense_fk_trace_npy": str(debug_dir / "dense_fk_trace.npy"),
                        "cspace_summary_json": str(debug_dir / "cspace_summary.json"),
                    },
                )
        print_block(
            "Joint-space execution validation",
            {
                "joint_path_nodes": int(len(getattr(result, "joint_path", []))),
                "execution_theta_waypoints": 0 if robot_execution is None else len(robot_execution.joint_path),
                "max_joint_step": 0.0 if robot_execution is None else robot_execution.max_joint_step,
                "dense_joint_path_constraint_certified": bool(
                    getattr(result, "dense_joint_path_constraint_certified", getattr(result, "dense_joint_path_is_certified", False))
                ),
                "dense_joint_path_joint_continuity_certified": bool(getattr(result, "dense_joint_path_joint_continuity_certified", False)),
                "dense_joint_path_execution_certified": bool(getattr(result, "dense_joint_path_execution_certified", False)),
                "worst_joint_step_index": int(getattr(result, "dense_joint_path_worst_joint_step_index", -1)),
                "constraint_validation_success": False if robot_execution is None else robot_execution.constraint_validation_success,
                "max_constraint_residual": 0.0 if robot_execution is None else robot_execution.max_constraint_residual,
                "mean_constraint_residual": 0.0 if robot_execution is None else robot_execution.mean_constraint_residual,
                "worst_stage": "none" if robot_execution is None else robot_execution.worst_constraint_stage,
                "worst_index": -1 if robot_execution is None else robot_execution.worst_constraint_index,
                "animation_enabled": False if robot_execution is None else robot_execution.animation_enabled,
                "execution_source": "disabled" if robot_execution is None else robot_execution.execution_source,
                "dense_joint_path_message": str(getattr(result, "dense_joint_path_message", "")),
            },
        )
    if args.save_animation is not None:
        print("save_animation = requested but not implemented in this interactive PyVista demo")
    if args.save_html is not None:
        print("save_html = requested but not implemented in this interactive PyVista demo")
    print(
        "pyvista_robot_animation = "
        + (
            "enabled"
            if (not args.no_viz and robot_execution is not None and robot_execution.animation_enabled)
            else "disabled"
        )
    )
    cspace_debug_dir = Path("outputs") / "ex66_jointspace_debug" / "latest"
    certified_dense_cspace_route = bool(planner_mode_label == "joint_space" and has_certified_dense_joint_path(result))
    should_render_cspace = bool(args.show_cspace and certified_dense_cspace_route)
    should_save_cspace = bool(args.save_cspace_debug and certified_dense_cspace_route)
    print_block(
        "C-space visualization",
        {
            "cspace_visualization_requested": bool(args.show_cspace),
            "cspace_axes": "theta0, theta1, theta2",
            "cspace_grid_res": int(args.cspace_grid_res),
            "cspace_route_source": "result.dense_joint_path" if certified_dense_cspace_route else "none",
            "cspace_surface_source": "FK-pulled-back residual(theta)=0" if planner_mode_label == "joint_space" else "none",
            "cspace_interactive_view": bool(should_render_cspace and not args.no_viz),
            "cspace_screenshot": str(cspace_debug_dir / "cspace_environment.png") if should_save_cspace else "disabled",
        },
    )

    if should_render_cspace and (not args.no_viz or should_save_cspace):
        cspace_output_dir = cspace_debug_dir if should_save_cspace else None
        screenshot_path = show_cspace_robot_planning(
            result=result,
            manifolds=_stage_manifolds_for_robot(families, robot),
            cspace_audit=cspace_audit,
            grid_res=int(args.cspace_grid_res),
            output_dir=cspace_output_dir,
            show=not bool(args.no_viz),
            show_surfaces=not bool(args.cspace_no_surfaces),
            route_only=bool(args.cspace_route_only),
            lightweight=bool(args.cspace_lightweight),
            marker_scale=float(args.cspace_marker_scale),
            surface_opacity=float(args.cspace_opacity),
            left_surface_opacity=args.cspace_left_opacity,
            middle_surface_opacity=args.cspace_middle_opacity,
            right_surface_opacity=args.cspace_right_opacity,
            middle_surface_color=args.cspace_middle_color,
            force_middle_sheet=bool(args.cspace_force_middle_sheet),
            allow_visual_proxy=bool(args.cspace_allow_visual_proxy),
            presentation_style=bool(args.cspace_presentation_style),
            surface_style=str(args.cspace_surface_style),
            surface_mode=str(args.cspace_surface_mode),
            smooth_surfaces=bool(args.cspace_smooth_surfaces),
            clean_view=bool(args.cspace_clean_view),
            safe_render=bool(args.cspace_safe_render),
            suppress_vtk_warnings=bool(args.suppress_vtk_warnings),
            vtk_warning_log=args.vtk_output_log,
            example_name="fixed_transfer_plane",
            middle_only=bool(args.cspace_middle_only),
            save_surfaces=bool(args.save_cspace_surfaces),
        )
        if screenshot_path is not None:
            print(f"cspace_environment_screenshot = {screenshot_path}")
    elif args.show_cspace:
        print("No certified dense joint path found; C-space route cannot be shown.", flush=True)

    if args.no_viz:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    if not args.no_viz:
        if not pyvista_available() or pv is None:
            print("PyVista is not available; skipping Example 66.1 visualization.")
        else:
            show_pyvista_robot_demo(
                families=families,
                result=result,
                start_q=start_q,
                goal_q=goal_q,
                plane_half_u=plane_half_u,
                plane_half_v=plane_half_v,
                robot=robot,
                robot_execution=robot_execution,
                show_exploration=bool(args.show_exploration),
                show_rejected_joint_interpolation=bool(args.show_rejected_joint_interpolation),
            )


if __name__ == "__main__":
    main()
