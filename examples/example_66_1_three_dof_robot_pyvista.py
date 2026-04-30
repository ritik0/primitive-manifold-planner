from __future__ import annotations

"""Example 66.1: Example 66 with a simple 3DOF robot in the same PyVista scene.

This demo uses a simple 3DOF spatial positioning arm to follow the exact
task-space path produced by Example 66. It is still an execution-layer IK
demonstration, not yet full robot configuration-space constrained planning.
"""

import argparse
from dataclasses import dataclass
import math
import time

import numpy as np
from ompl import util as ou

import example_66_multimodal_graph_search as ex66
from collision_utilities import default_example_66_obstacles

from primitive_manifold_planner.visualization import add_manifold, add_points, pyvista_available

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
    exact_planner_path_used: bool


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
) -> RobotExecutionResult | None:
    route = np.asarray(result.path if len(result.path) >= 2 else result.raw_path, dtype=float)
    if len(route) < 2:
        planned_joint_path = np.asarray(getattr(result, "joint_path", np.zeros((0, 3), dtype=float)), dtype=float)
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
            exact_planner_path_used=True,
        )

    waypoint_count = int(np.clip(max(80, len(route) // 6), 80, 120))
    targets = resample_polyline(route, num_points=waypoint_count)

    joint_solutions: list[np.ndarray] = []
    ee_points: list[np.ndarray] = []
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
        ee_points.append(np.asarray(ee_world, dtype=float))
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
        exact_planner_path_used=True,
    )


def make_link_mesh(p0: np.ndarray, p1: np.ndarray, radius: float):
    if pv is None:
        return None
    return pv.Line(np.asarray(p0, dtype=float), np.asarray(p1, dtype=float), resolution=1).tube(radius=float(radius))


def update_actor_mesh(actor, mesh) -> None:
    if actor is None or mesh is None:
        return
    mapper = getattr(actor, "mapper", None)
    if mapper is not None and hasattr(mapper, "SetInputData"):
        mapper.SetInputData(mesh)


def add_robot_pedestal(plotter, robot: SpatialRobot3DOF):
    if pv is None:
        return None
    center = np.asarray(robot.base_world, dtype=float) + np.asarray([0.0, 0.0, -0.18], dtype=float)
    return plotter.add_mesh(
        pv.Cylinder(center=center, direction=(0.0, 0.0, 1.0), radius=0.18, height=0.32, resolution=32),
        color="#5d4037",
        opacity=1.0,
        smooth_shading=True,
        label="robot pedestal",
    )


def make_robot_actor_bundle(
    plotter,
    robot: SpatialRobot3DOF,
    joint_angles: np.ndarray,
    opacity: float = 1.0,
) -> dict[str, list[object] | object | None]:
    joints = robot.forward_kinematics_3d(joint_angles)
    bundle: dict[str, list[object] | object | None] = {"all": [], "base": None, "links": [], "joints": []}

    base_actor = plotter.add_mesh(
        pv.Sphere(radius=float(robot.joint_radius), center=np.asarray(robot.base_world, dtype=float)),
        color="#2f2f2f",
        opacity=float(opacity),
        smooth_shading=True,
        label="robot base",
    )
    bundle["base"] = base_actor
    bundle["all"].append(base_actor)

    for idx in range(3):
        link_actor = plotter.add_mesh(
            make_link_mesh(joints[idx], joints[idx + 1], radius=float(robot.link_radius)),
            color="#ef6c00",
            opacity=float(opacity),
            smooth_shading=True,
            label="robot links" if idx == 0 else None,
        )
        bundle["links"].append(link_actor)
        bundle["all"].append(link_actor)

    for idx, pos in enumerate(joints):
        actor = plotter.add_mesh(
            pv.Sphere(
                radius=float(robot.joint_radius if idx < len(joints) - 1 else robot.ee_radius),
                center=np.asarray(pos, dtype=float),
            ),
            color="#37474f" if idx < len(joints) - 1 else "#d81b60",
            opacity=float(opacity),
            smooth_shading=True,
            label="robot joints" if idx == 0 else ("robot end effector" if idx == len(joints) - 1 else None),
        )
        bundle["joints"].append(actor)
        bundle["all"].append(actor)

    return bundle


def update_robot_actor_bundle(
    plotter,
    robot: SpatialRobot3DOF,
    bundle: dict[str, list[object] | object | None],
    joint_angles: np.ndarray,
) -> None:
    joints = robot.forward_kinematics_3d(joint_angles)
    update_actor_mesh(
        bundle.get("base"),
        pv.Sphere(radius=float(robot.joint_radius), center=np.asarray(robot.base_world, dtype=float)),
    )
    for idx, actor in enumerate(bundle.get("links", [])):
        update_actor_mesh(actor, make_link_mesh(joints[idx], joints[idx + 1], radius=float(robot.link_radius)))
    for idx, actor in enumerate(bundle.get("joints", [])):
        update_actor_mesh(
            actor,
            pv.Sphere(
                radius=float(robot.joint_radius if idx < len(joints) - 1 else robot.ee_radius),
                center=np.asarray(joints[idx], dtype=float),
            ),
        )
    plotter.render()


def show_pyvista_robot_demo(
    families,
    result: ex66.FixedPlaneRoute,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    plane_half_u: float,
    plane_half_v: float,
    robot: SpatialRobot3DOF,
    robot_execution: RobotExecutionResult | None,
) -> bool:
    if pv is None or not pyvista_available():
        print("PyVista is not available; skipping Example 66.1 visualization.")
        return False

    left_family, plane_family, right_family = families
    left_manifold = left_family.manifold(float(left_family.sample_lambdas()[0]))
    plane_manifold = plane_family.manifold(float(plane_family.sample_lambdas()[0]))
    right_manifold = right_family.manifold(float(right_family.sample_lambdas()[0]))

    colors = {
        ex66.LEFT_STAGE: "#81c784",
        ex66.PLANE_STAGE: "#64b5f6",
        ex66.RIGHT_STAGE: "#a5d6a7",
    }
    manifold_colors = {
        "left_support_3d": "#c58b4c",
        "transfer_plane_3d": "#7fa7c6",
        "right_support_3d": "#c58b4c",
    }

    plotter = pv.Plotter(window_size=(1440, 920))
    if hasattr(plotter, "set_background"):
        plotter.set_background("#edf3f8", top="#fdfdfd")
    if hasattr(plotter, "enable_anti_aliasing"):
        try:
            plotter.enable_anti_aliasing()
        except Exception:
            pass
    if hasattr(plotter, "enable_lightkit"):
        try:
            plotter.enable_lightkit()
        except Exception:
            pass
    plotter.add_text(
        "Example 66.1: Example 66 with a 3DOF robot following the exact selected path",
        font_size=12,
    )

    actor_groups: dict[str, list[object]] = {
        "Manifolds": [],
        "Obstacles": [],
        "Evidence": [],
        "Frontiers": [],
        "Hypotheses": [],
        "Committed": [],
        "Robot": [],
        "EETrace": [],
        "StartGoal": [],
    }

    for family, manifold in [(left_family, left_manifold), (right_family, right_manifold)]:
        actor = add_manifold(
            plotter,
            manifold,
            color=manifold_colors.get(family.name, "#999999"),
            opacity=0.10,
        )
        if actor is not None:
            actor_groups["Manifolds"].append(actor)

    plane_corners = ex66.plane_patch_corners(plane_manifold, half_u=plane_half_u, half_v=plane_half_v)
    plane_faces = np.hstack([[4, 0, 1, 2, 3]])
    plane_patch = pv.PolyData(plane_corners, faces=plane_faces)
    plane_actor = plotter.add_mesh(
        plane_patch,
        color=manifold_colors.get(plane_family.name, "#999999"),
        opacity=0.18,
        show_edges=False,
        smooth_shading=False,
        name=plane_family.name,
    )
    if plane_actor is not None:
        actor_groups["Manifolds"].append(plane_actor)

    for obstacle in getattr(result, "obstacles", []):
        mesh = obstacle.to_pyvista_mesh() if hasattr(obstacle, "to_pyvista_mesh") else None
        if mesh is None:
            continue
        actor = plotter.add_mesh(
            mesh,
            color="#8d6e63" if "cylinder" in getattr(obstacle, "name", "") else "#6d4c41",
            opacity=0.30,
            smooth_shading=True,
            label=getattr(obstacle, "name", "obstacle"),
        )
        if actor is not None:
            actor_groups["Obstacles"].append(actor)

    for stage in ex66.STAGES:
        stage_edges = result.stage_evidence_edges.get(stage, [])
        poly = ex66.build_segment_polydata(stage_edges)
        if poly is not None:
            actor = plotter.add_mesh(
                poly,
                color=colors[stage],
                line_width=2.0 if stage == ex66.PLANE_STAGE else 1.8,
                opacity=0.35,
                label=f"{stage} evidence edges",
            )
            if actor is not None:
                actor_groups["Evidence"].append(actor)

        stage_points = result.stage_evidence_points.get(stage, np.zeros((0, 3), dtype=float))
        if len(stage_points) > 0:
            actor = add_points(
                plotter,
                stage_points,
                color=colors[stage],
                size=5.0,
                label=f"{stage} evidence points",
            )
            if actor is not None:
                actor_groups["Evidence"].append(actor)

        frontier = result.stage_frontier_points.get(stage, np.zeros((0, 3), dtype=float))
        if len(frontier) > 0:
            actor = add_points(
                plotter,
                frontier,
                color="#fb8c00" if stage == ex66.LEFT_STAGE else ("#8e24aa" if stage == ex66.PLANE_STAGE else "#00897b"),
                size=8.0,
                label=f"{stage} frontier",
            )
            if actor is not None:
                actor_groups["Frontiers"].append(actor)

    if len(result.left_plane_hypothesis_points) > 0:
        actor = add_points(plotter, result.left_plane_hypothesis_points, color="#ff7043", size=10.0, label="left-plane hypotheses")
        if actor is not None:
            actor_groups["Hypotheses"].append(actor)
    if len(result.plane_right_hypothesis_points) > 0:
        actor = add_points(plotter, result.plane_right_hypothesis_points, color="#26a69a", size=10.0, label="plane-right hypotheses")
        if actor is not None:
            actor_groups["Hypotheses"].append(actor)

    if len(result.raw_path) >= 2:
        raw_polyline = pv.lines_from_points(np.asarray(result.raw_path, dtype=float))
        actor = plotter.add_mesh(raw_polyline, color="#90a4ae", line_width=3.0, opacity=0.8, label="certified route")
        if actor is not None:
            actor_groups["Committed"].append(actor)
    if len(result.path) >= 2:
        display_polyline = pv.lines_from_points(np.asarray(result.path, dtype=float))
        actor = plotter.add_mesh(display_polyline, color="#1565c0", line_width=7.0, label="display route")
        if actor is not None:
            actor_groups["Committed"].append(actor)

    actor = add_points(plotter, start_q, color="black", size=16.0, label="start")
    if actor is not None:
        actor_groups["StartGoal"].append(actor)
    actor = add_points(plotter, goal_q, color="gold", size=18.0, label="goal")
    if actor is not None:
        actor_groups["StartGoal"].append(actor)

    robot_state: dict[str, object] = {
        "bundle": None,
        "visible": True,
        "trace_actor": None,
        "trace_visible": True,
    }

    initial_angles = (
        np.asarray(robot_execution.joint_path[0], dtype=float)
        if robot_execution is not None and len(robot_execution.joint_path) > 0
        else np.asarray([0.0, 0.55, -0.85], dtype=float)
    )
    robot_state["bundle"] = make_robot_actor_bundle(plotter, robot, initial_angles, opacity=1.0)
    pedestal_actor = add_robot_pedestal(plotter, robot)
    if pedestal_actor is not None:
        actor_groups["Robot"].append(pedestal_actor)
    actor_groups["Robot"].extend(robot_state["bundle"].get("all", []))

    def set_robot_visibility(visible: bool) -> None:
        robot_state["visible"] = bool(visible)
        bundle = robot_state.get("bundle")
        if isinstance(bundle, dict):
            for actor in bundle.get("all", []):
                if actor is not None:
                    actor.SetVisibility(bool(visible))
        for actor in actor_groups.get("Robot", []):
            if actor is not None:
                actor.SetVisibility(bool(visible))
        plotter.render()

    def replace_robot_pose(joint_angles: np.ndarray) -> None:
        bundle = robot_state.get("bundle")
        if isinstance(bundle, dict):
            update_robot_actor_bundle(plotter, robot, bundle, joint_angles)
            if not bool(robot_state.get("visible", True)):
                for actor in bundle.get("all", []):
                    if actor is not None:
                        actor.SetVisibility(False)

    def replace_trace(points: np.ndarray) -> None:
        existing = robot_state.get("trace_actor")
        if existing is not None:
            plotter.remove_actor(existing, render=False)
        pts = np.asarray(points, dtype=float)
        if len(pts) == 0:
            robot_state["trace_actor"] = None
            return
        if len(pts) == 1:
            actor = plotter.add_mesh(
                pv.Sphere(radius=float(0.75 * robot.ee_radius), center=np.asarray(pts[0], dtype=float)),
                color="#d81b60",
                opacity=1.0,
                label="ee trace",
            )
        else:
            actor = plotter.add_mesh(
                pv.lines_from_points(pts),
                color="#d81b60",
                line_width=5.0,
                opacity=1.0,
                label="ee trace",
            )
        if not bool(robot_state.get("trace_visible", True)):
            actor.SetVisibility(False)
        robot_state["trace_actor"] = actor

    def set_trace_visibility(visible: bool) -> None:
        robot_state["trace_visible"] = bool(visible)
        actor = robot_state.get("trace_actor")
        if actor is not None:
            actor.SetVisibility(bool(visible))
        plotter.render()

    def set_visibility(name: str, visible: bool):
        if name == "Robot":
            set_robot_visibility(visible)
            return
        if name == "EETrace":
            set_trace_visibility(visible)
            return
        for actor_inner in actor_groups.get(name, []):
            if actor_inner is not None:
                actor_inner.SetVisibility(bool(visible))
        plotter.render()

    animation_state = {
        "frame": 0,
        "playing": False,
        "trace_points": [],
        "running": False,
    }

    def reset_animation() -> None:
        animation_state["frame"] = 0
        animation_state["playing"] = bool(robot_execution is not None and robot_execution.animation_enabled)
        animation_state["trace_points"] = []
        if robot_execution is not None and len(robot_execution.joint_path) > 0:
            replace_robot_pose(robot_execution.joint_path[0])
            replace_trace(np.zeros((0, 3), dtype=float))
        plotter.render()

    def replay_button_callback(_state: bool) -> None:
        start_replay()

    def animate_step() -> None:
        if robot_execution is None or not robot_execution.animation_enabled:
            return
        if not bool(animation_state["playing"]):
            return
        idx = min(int(animation_state["frame"]), len(robot_execution.joint_path) - 1)
        replace_robot_pose(robot_execution.joint_path[idx])
        animation_state["trace_points"].append(np.asarray(robot_execution.end_effector_points_3d[idx], dtype=float))
        replace_trace(np.asarray(animation_state["trace_points"], dtype=float))
        if idx >= len(robot_execution.joint_path) - 1:
            animation_state["playing"] = False
        else:
            animation_state["frame"] += 1
        plotter.render()

    def play_animation() -> None:
        if robot_execution is None or not robot_execution.animation_enabled:
            return
        if bool(animation_state["running"]):
            return
        animation_state["running"] = True
        try:
            while animation_state["playing"]:
                animate_step()
                plotter.update()
                time.sleep(0.02)
        finally:
            animation_state["running"] = False

    def start_replay() -> None:
        reset_animation()
        play_animation()

    default_visibility = {
        "Manifolds": True,
        "Obstacles": True,
        "Evidence": True,
        "Frontiers": True,
        "Hypotheses": True,
        "Committed": True,
        "Robot": True,
        "EETrace": True,
        "StartGoal": True,
    }

    for idx, label in enumerate(["Manifolds", "Obstacles", "Evidence", "Frontiers", "Hypotheses", "Committed", "Robot", "EETrace", "StartGoal"]):
        y = 10 + idx * 42
        plotter.add_text(label, position=(55, y + 7), font_size=10, color="black")
        plotter.add_checkbox_button_widget(
            callback=lambda state, name=label: set_visibility(name, state),
            value=bool(default_visibility[label]),
            position=(10, y),
            size=28,
            color_on="lightgray",
            color_off="white",
            background_color="gray",
        )

    plotter.add_text("Press r to replay robot motion", position=(1030, 18), font_size=10, color="black")
    plotter.add_text("Replay", position=(1160, 52), font_size=10, color="black")
    plotter.add_checkbox_button_widget(
        callback=replay_button_callback,
        value=False,
        position=(1110, 44),
        size=28,
        color_on="#90caf9",
        color_off="white",
        background_color="gray",
    )
    if hasattr(plotter, "add_key_event"):
        plotter.add_key_event("r", start_replay)

    plotter.add_axes()
    plotter.show_grid()
    plotter.camera_position = [
        (0.15, -7.9, 3.8),
        (0.0, -0.1, 0.45),
        (0.0, 0.0, 1.0),
    ]

    for name, visible in default_visibility.items():
        set_visibility(name, visible)

    reset_animation()
    plotter.show(auto_close=False, interactive_update=True)

    if robot_execution is not None and robot_execution.animation_enabled:
        play_animation()

    try:
        plotter.app.exec()
    except Exception:
        try:
            plotter.show()
        except Exception:
            pass
    return True


def configure_example_66_budgets(args) -> None:
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
    if args.fast:
        ex66.SAFETY_MAX_TOTAL_ROUNDS = min(ex66.SAFETY_MAX_TOTAL_ROUNDS, 10)
        ex66.MIN_ROUNDS_BEFORE_SATURATION_CHECK = min(ex66.MIN_ROUNDS_BEFORE_SATURATION_CHECK, 6)
        ex66.MIN_POST_SOLUTION_ROUNDS = min(ex66.MIN_POST_SOLUTION_ROUNDS, 3)


def main():
    parser = argparse.ArgumentParser(
        description="Example 66.1: Example 66 with a simple 3DOF robot in the same PyVista scene."
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    ou.RNG.setSeed(args.seed)
    ou.setLogLevel(ou.LOG_ERROR)

    configure_example_66_budgets(args)

    families, start_q, goal_q, plane_half_u, plane_half_v = ex66.build_scene()
    obstacles = default_example_66_obstacles()
    robot = SpatialRobot3DOF(
        link_lengths=np.asarray([1.35, 1.05, 0.75], dtype=float),
        base_world=np.asarray([0.0, -1.25, 0.10], dtype=float),
    )
    result = ex66.plan_fixed_manifold_multimodal_route(
        families=families,
        start_q=start_q,
        goal_q=goal_q,
        robot=robot,
        obstacles=obstacles,
    )

    route = np.asarray(result.path if len(result.path) >= 2 else result.raw_path, dtype=float)
    robot_execution = build_robot_execution(result, robot) if result.success else None

    print("\nExample 66.1: 3DOF PyVista robot tracing selected multimodal path")
    print(f"planner_success = {result.success}")
    print(f"planner_message = {result.message}")
    print(f"route_points = {len(route)}")
    print(f"robot_waypoints = {0 if robot_execution is None else len(robot_execution.joint_path)}")
    print(f"ik_success_count = {0 if robot_execution is None else robot_execution.ik_success_count}")
    print(f"ik_failure_count = {0 if robot_execution is None else robot_execution.ik_failure_count}")
    print(f"max_tracking_error = {0.0 if robot_execution is None else robot_execution.max_tracking_error:.4f}")
    print(f"mean_tracking_error = {0.0 if robot_execution is None else robot_execution.mean_tracking_error:.4f}")
    print(f"exact_planner_path_used = {False if robot_execution is None else robot_execution.exact_planner_path_used}")
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
