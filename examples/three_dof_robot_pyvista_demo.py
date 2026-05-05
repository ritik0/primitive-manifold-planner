from __future__ import annotations

"""Example 66.1: Example 66 with a simple 3DOF robot in the same PyVista scene.

This demo uses a simple 3DOF spatial positioning arm to follow the exact
task-space path produced by Example 66. It is still an execution-layer IK
demonstration, not yet full robot configuration-space constrained planning.
"""

import argparse
from dataclasses import dataclass
import math

import numpy as np
from ompl import util as ou

from primitive_manifold_planner.examplesupport.collision_utilities import default_example_66_obstacles
from primitive_manifold_planner.examplesupport.example66_scene import build_example66_scene
from primitive_manifold_planner.thesis import parallel_evidence_planner as ex66
from primitive_manifold_planner.visualization import pyvista_available
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
class SpatialRobot3DOF:
    link_lengths: np.ndarray
    base_world: np.ndarray
    link_radius: float = 0.055
    joint_radius: float = 0.095
    ee_radius: float = 0.075

    @property
    def max_reach(self) -> float:
        return float(np.sum(self.link_lengths))

    def forward_kinematics_3d(self, joint_angles: np.ndarray) -> np.ndarray:
        theta0, theta1, theta2 = np.asarray(joint_angles, dtype=float).reshape(3)
        l1, l2, l3 = np.asarray(self.link_lengths, dtype=float).reshape(3)

        yaw_dir = np.asarray([math.cos(theta0), math.sin(theta0), 0.0], dtype=float)

        radial_1 = l1 * math.cos(theta1)
        z_1 = l1 * math.sin(theta1)

        angle_12 = theta1 + theta2
        radial_2 = radial_1 + l2 * math.cos(angle_12)
        z_2 = z_1 + l2 * math.sin(angle_12)

        radial_3 = radial_2 + l3 * math.cos(angle_12)
        z_3 = z_2 + l3 * math.sin(angle_12)

        p0 = np.asarray(self.base_world, dtype=float)
        p1 = p0 + radial_1 * yaw_dir + np.asarray([0.0, 0.0, z_1], dtype=float)
        p2 = p0 + radial_2 * yaw_dir + np.asarray([0.0, 0.0, z_2], dtype=float)
        p3 = p0 + radial_3 * yaw_dir + np.asarray([0.0, 0.0, z_3], dtype=float)
        return np.asarray([p0, p1, p2, p3], dtype=float)


@dataclass
class RobotExecutionResult:
    target_task_points_3d: np.ndarray
    joint_path: np.ndarray
    end_effector_points_3d: np.ndarray
    ik_success_count: int
    ik_failure_count: int
    max_tracking_error: float
    mean_tracking_error: float
    animation_enabled: bool
    planner_path_resampled_for_robot: bool
    planner_joint_path_used_directly: bool


def wrap_angles(joint_angles: np.ndarray) -> np.ndarray:
    arr = np.asarray(joint_angles, dtype=float)
    return (arr + np.pi) % (2.0 * np.pi) - np.pi


def choose_robot_for_route(route: np.ndarray) -> SpatialRobot3DOF:
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


def build_robot_execution(
    result: ex66.FixedPlaneRoute,
    robot: SpatialRobot3DOF,
    use_planner_joint_path: bool = False,
) -> RobotExecutionResult | None:
    planned_joint_path = np.asarray(getattr(result, "joint_path", np.zeros((0, 3), dtype=float)), dtype=float)
    if use_planner_joint_path and len(planned_joint_path) >= 2:
        ee_points = np.asarray([robot.forward_kinematics_3d(theta)[-1] for theta in planned_joint_path], dtype=float)
        return RobotExecutionResult(
            target_task_points_3d=np.asarray(ee_points, dtype=float),
            joint_path=np.asarray(planned_joint_path, dtype=float),
            end_effector_points_3d=np.asarray(ee_points, dtype=float),
            ik_success_count=int(len(planned_joint_path)),
            ik_failure_count=0,
            max_tracking_error=0.0,
            mean_tracking_error=0.0,
            animation_enabled=bool(len(planned_joint_path) >= 2),
            planner_path_resampled_for_robot=False,
            planner_joint_path_used_directly=True,
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
            animation_enabled=bool(len(planned_joint_path) >= 2),
            planner_path_resampled_for_robot=False,
            planner_joint_path_used_directly=True,
        )

    waypoint_count = int(np.clip(max(80, len(route) // 6), 80, 120))
    targets = resample_polyline(route, num_points=waypoint_count)

    joint_solutions: list[np.ndarray] = []
    tracking_errors: list[float] = []
    success_count = 0
    failure_count = 0

    first_target = np.asarray(targets[0], dtype=float)
    rel = first_target - np.asarray(robot.base_world, dtype=float)
    yaw_guess = math.atan2(float(rel[1]), float(rel[0]))
    radial_guess = float(np.linalg.norm(rel[:2]))
    pitch_guess = math.atan2(float(rel[2]), max(radial_guess, 1e-6))
    warm = np.asarray([yaw_guess, pitch_guess, -0.65], dtype=float)

    for target in targets:
        solved, err, ok = solve_spatial_ik(robot, target, warm)
        if ok:
            warm = solved.copy()
            success_count += 1
        else:
            failure_count += 1
        joint_solutions.append(warm.copy())
        ee_world = robot.forward_kinematics_3d(warm)[-1]
        tracking_errors.append(float(np.linalg.norm(ee_world - np.asarray(target, dtype=float))))

    dense_joint_path: list[np.ndarray] = []
    if len(joint_solutions) > 0:
        dense_joint_path.append(np.asarray(joint_solutions[0], dtype=float))
        for prev, nxt in zip(joint_solutions[:-1], joint_solutions[1:]):
            prev_arr = wrap_angles(np.asarray(prev, dtype=float))
            next_arr = wrap_angles(np.asarray(nxt, dtype=float))
            delta = wrap_angles(next_arr - prev_arr)
            for alpha in np.linspace(0.25, 1.0, 4):
                dense_joint_path.append(wrap_angles(prev_arr + float(alpha) * delta))
    dense_joint_path_arr = np.asarray(dense_joint_path, dtype=float) if len(dense_joint_path) > 0 else np.zeros((0, 3), dtype=float)
    dense_ee_points = (
        np.asarray([robot.forward_kinematics_3d(theta)[-1] for theta in dense_joint_path_arr], dtype=float)
        if len(dense_joint_path_arr) > 0
        else np.zeros((0, 3), dtype=float)
    )

    return RobotExecutionResult(
        target_task_points_3d=np.asarray(targets, dtype=float),
        joint_path=dense_joint_path_arr,
        end_effector_points_3d=dense_ee_points,
        ik_success_count=int(success_count),
        ik_failure_count=int(failure_count),
        max_tracking_error=float(max(tracking_errors) if tracking_errors else 0.0),
        mean_tracking_error=float(np.mean(tracking_errors) if tracking_errors else 0.0),
        animation_enabled=bool(len(joint_solutions) >= 2),
        planner_path_resampled_for_robot=True,
        planner_joint_path_used_directly=False,
    )


def configure_example_66_budgets(args) -> None:
    planner_core = getattr(ex66, "planner_core", ex66)
    if args.max_rounds is not None:
        ex66.SAFETY_MAX_TOTAL_ROUNDS = max(4, int(args.max_rounds))
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
    if args.fast:
        ex66.SAFETY_MAX_TOTAL_ROUNDS = min(ex66.SAFETY_MAX_TOTAL_ROUNDS, 10)
        ex66.MIN_ROUNDS_BEFORE_SATURATION_CHECK = min(ex66.MIN_ROUNDS_BEFORE_SATURATION_CHECK, 6)
        ex66.MIN_POST_SOLUTION_ROUNDS = min(ex66.MIN_POST_SOLUTION_ROUNDS, 3)
        planner_core.SAFETY_MAX_TOTAL_ROUNDS = ex66.SAFETY_MAX_TOTAL_ROUNDS
        planner_core.MIN_ROUNDS_BEFORE_SATURATION_CHECK = ex66.MIN_ROUNDS_BEFORE_SATURATION_CHECK
        planner_core.MIN_POST_SOLUTION_ROUNDS = ex66.MIN_POST_SOLUTION_ROUNDS


def main():
    parser = argparse.ArgumentParser(
        description="Example 66.1: Example 66 with a simple 3DOF robot in the same PyVista scene."
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--serial", action="store_true")
    parser.add_argument(
        "--jointspace-planning",
        action="store_true",
        help="Plan directly in robot joint space instead of tracking the task-space route after planning.",
    )
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    ou.RNG.setSeed(args.seed)
    ou.setLogLevel(ou.LOG_ERROR)

    configure_example_66_budgets(args)

    families, start_q, goal_q, plane_half_u, plane_half_v = build_example66_scene()
    obstacles = default_example_66_obstacles()
    robot = SpatialRobot3DOF(
        link_lengths=np.asarray([1.35, 1.05, 0.75], dtype=float),
        base_world=np.asarray([0.0, -1.25, 0.10], dtype=float),
    )
    if args.jointspace_planning:
        result = ex66.plan_fixed_manifold_multimodal_route(
            families=families,
            start_q=start_q,
            goal_q=goal_q,
            robot=robot,
            serial_mode=args.serial,
            obstacles=obstacles,
        )
    else:
        result = ex66.plan_fixed_manifold_multimodal_route(
            families=families,
            start_q=start_q,
            goal_q=goal_q,
            serial_mode=args.serial,
        )

    route = np.asarray(result.path if len(result.path) >= 2 else result.raw_path, dtype=float)
    robot_execution = (
        build_robot_execution(result, robot, use_planner_joint_path=bool(args.jointspace_planning))
        if result.success
        else None
    )

    print("\nExample 66.1: 3DOF PyVista robot tracing selected multimodal path")
    print(
        "planner_mode = "
        + ("robot_jointspace_planning" if args.jointspace_planning else "task_space_path_tracking")
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
    print(
        "pyvista_robot_animation = "
        + (
            "enabled"
            if (not args.no_viz and robot_execution is not None and robot_execution.animation_enabled)
            else "disabled"
        )
    )

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
            )


if __name__ == "__main__":
    main()
