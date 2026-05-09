from __future__ import annotations

import time
from typing import Any

import numpy as np

from primitive_manifold_planner.thesis import parallel_evidence_planner as ex66
from primitive_manifold_planner.examplesupport.intrinsic_multimodal_helpers import build_segment_polydata
from primitive_manifold_planner.visualization.display import plane_patch_corners
from primitive_manifold_planner.visualization import add_manifold, add_points, pyvista_available

try:
    import pyvista as pv
except Exception:
    pv = None


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


def add_robot_pedestal(plotter, robot) -> Any:
    if pv is None:
        return None
    base = np.asarray(robot.base_world, dtype=float)
    center = base + np.asarray([0.0, 0.0, -0.20], dtype=float)
    return plotter.add_mesh(
        pv.Cylinder(
            center=center,
            direction=(0.0, 0.0, 1.0),
            radius=0.20,
            height=0.36,
            resolution=40,
        ),
        color="#5d4037",
        opacity=1.0,
        smooth_shading=True,
        label="fixed robot base",
    )


def _display_joints(robot, joint_angles: np.ndarray) -> np.ndarray:
    """Return visual joints for a clean 3DOF positioning arm.

    The FK still returns p0, p1, p2, p3. For visualization, p2 is an internal
    point on the forearm chain. Drawing p0->p1 and p1->p3 makes the robot look
    like a normal yaw-base + shoulder/elbow positioning arm without inventing
    a fake wrist.
    """
    fk = np.asarray(robot.forward_kinematics_3d(joint_angles), dtype=float)
    p0 = fk[0]
    p1 = fk[1]
    p3 = fk[-1]
    return np.asarray([p0, p1, p3], dtype=float)


def make_robot_actor_bundle(plotter, robot, joint_angles: np.ndarray, opacity: float = 1.0):
    if pv is None:
        return {"all": [], "base": None, "links": [], "joints": []}

    joints = _display_joints(robot, joint_angles)

    bundle: dict[str, list[object] | object | None] = {
        "all": [],
        "base": None,
        "links": [],
        "joints": [],
    }

    # Fixed yaw/base joint.
    base_actor = plotter.add_mesh(
        pv.Sphere(radius=float(robot.joint_radius * 1.20), center=joints[0]),
        color="#263238",
        opacity=float(opacity),
        smooth_shading=True,
        label="base yaw joint",
    )
    bundle["base"] = base_actor
    bundle["all"].append(base_actor)

    # Link 1: base/shoulder to elbow.
    link1 = plotter.add_mesh(
        make_link_mesh(joints[0], joints[1], radius=float(robot.link_radius)),
        color="#ef6c00",
        opacity=float(opacity),
        smooth_shading=True,
        label="robot links",
    )
    bundle["links"].append(link1)
    bundle["all"].append(link1)

    # Link 2: elbow to end-effector.
    link2 = plotter.add_mesh(
        make_link_mesh(joints[1], joints[2], radius=float(robot.link_radius)),
        color="#ef6c00",
        opacity=float(opacity),
        smooth_shading=True,
        label=None,
    )
    bundle["links"].append(link2)
    bundle["all"].append(link2)

    # Joint markers: base, elbow, end-effector.
    for idx, pos in enumerate(joints):
        is_ee = idx == len(joints) - 1
        radius = float(robot.ee_radius if is_ee else robot.joint_radius)
        color = "#d81b60" if is_ee else "#37474f"
        label = "end effector" if is_ee else ("robot joints" if idx == 0 else None)
        actor = plotter.add_mesh(
            pv.Sphere(radius=radius, center=np.asarray(pos, dtype=float)),
            color=color,
            opacity=float(opacity),
            smooth_shading=True,
            label=label,
        )
        bundle["joints"].append(actor)
        bundle["all"].append(actor)

    return bundle


def update_robot_actor_bundle(plotter, robot, bundle, joint_angles: np.ndarray) -> None:
    if pv is None:
        return

    joints = _display_joints(robot, joint_angles)

    update_actor_mesh(
        bundle.get("base"),
        pv.Sphere(radius=float(robot.joint_radius * 1.20), center=joints[0]),
    )

    links = bundle.get("links", [])
    if len(links) >= 1:
        update_actor_mesh(
            links[0],
            make_link_mesh(joints[0], joints[1], radius=float(robot.link_radius)),
        )
    if len(links) >= 2:
        update_actor_mesh(
            links[1],
            make_link_mesh(joints[1], joints[2], radius=float(robot.link_radius)),
        )

    for idx, actor in enumerate(bundle.get("joints", [])):
        is_ee = idx == len(joints) - 1
        radius = float(robot.ee_radius if is_ee else robot.joint_radius)
        update_actor_mesh(
            actor,
            pv.Sphere(radius=radius, center=np.asarray(joints[idx], dtype=float)),
        )

    plotter.render()


def show_pyvista_robot_demo(
    families,
    result,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    plane_half_u: float,
    plane_half_v: float,
    robot,
    robot_execution,
    show_exploration: bool = True,
    show_rejected_joint_interpolation: bool = False,
) -> bool:
    _ = show_rejected_joint_interpolation

    if pv is None or not pyvista_available():
        print("PyVista is not available; skipping Example 66.1 visualization.")
        return False

    left_family, plane_family, right_family = families
    left_manifold = left_family.manifold(float(left_family.sample_lambdas()[0]))
    plane_manifold = plane_family.manifold(float(plane_family.sample_lambdas()[0]))
    right_manifold = right_family.manifold(float(right_family.sample_lambdas()[0]))

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

    plotter.add_text("Example 66.1: clean joint-space robot demo", font_size=12)
    plotter.add_text(
        "Robot: simple 3DOF positioning arm, q=[base yaw, shoulder pitch, elbow pitch].",
        position=(18, 54),
        font_size=10,
        color="black",
    )
    plotter.add_text(
        "Red route = FK(dense theta path). No task-space execution path is used.",
        position=(18, 82),
        font_size=10,
        color="#b71c1c",
    )

    # Manifolds.
    for family, manifold in [(left_family, left_manifold), (right_family, right_manifold)]:
        actor = add_manifold(plotter, manifold, color="#c58b4c", opacity=0.10)
        if actor is not None:
            actor_groups["Manifolds"].append(actor)

    plane_corners = plane_patch_corners(plane_manifold, half_u=plane_half_u, half_v=plane_half_v)
    plane_faces = np.hstack([[4, 0, 1, 2, 3]])
    plane_patch = pv.PolyData(plane_corners, faces=plane_faces)
    actor = plotter.add_mesh(
        plane_patch,
        color="#7fa7c6",
        opacity=0.18,
        show_edges=False,
        smooth_shading=False,
        name=plane_family.name,
    )
    if actor is not None:
        actor_groups["Manifolds"].append(actor)

    # Obstacles.
    for obstacle in getattr(result, "obstacles", []):
        mesh = obstacle.to_pyvista_mesh() if hasattr(obstacle, "to_pyvista_mesh") else None
        if mesh is None:
            continue
        actor = plotter.add_mesh(
            mesh,
            color="#8d6e63",
            opacity=0.30,
            smooth_shading=True,
            label=getattr(obstacle, "name", "obstacle"),
        )
        if actor is not None:
            actor_groups["Obstacles"].append(actor)

    # Evidence.
    colors = {
        ex66.LEFT_STAGE: "#81c784",
        ex66.PLANE_STAGE: "#64b5f6",
        ex66.RIGHT_STAGE: "#a5d6a7",
    }
    for stage in ex66.STAGES:
        stage_edges = result.stage_evidence_edges.get(stage, [])
        poly = build_segment_polydata(stage_edges)
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
            actor = add_points(plotter, stage_points, color=colors[stage], size=5.0, label=f"{stage} evidence points")
            if actor is not None:
                actor_groups["Evidence"].append(actor)

        frontier = result.stage_frontier_points.get(stage, np.zeros((0, 3), dtype=float))
        if len(frontier) > 0:
            actor = add_points(plotter, frontier, color="#00897b", size=8.0, label=f"{stage} frontier")
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

    # Final executed route from dense theta path.
    dense_joint_path = np.asarray(getattr(result, "dense_joint_path", np.zeros((0, 3), dtype=float)), dtype=float)
    dense_execution_certified = bool(getattr(result, "dense_joint_path_execution_certified", False))
    if dense_execution_certified and len(dense_joint_path) >= 2:
        fk_route = np.asarray([robot.forward_kinematics_3d(theta)[-1] for theta in dense_joint_path], dtype=float)
        actor = plotter.add_mesh(
            pv.lines_from_points(fk_route),
            color="#d32f2f",
            line_width=8.0,
            label="FINAL EXECUTED ROUTE = FK(dense theta path)",
        )
        if actor is not None:
            actor_groups["Committed"].append(actor)

    # Start / goal.
    actor = add_points(plotter, start_q, color="black", size=16.0, label="start")
    if actor is not None:
        actor_groups["StartGoal"].append(actor)
    actor = add_points(plotter, goal_q, color="gold", size=18.0, label="goal")
    if actor is not None:
        actor_groups["StartGoal"].append(actor)

    # Robot.
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
                for actor_inner in bundle.get("all", []):
                    if actor_inner is not None:
                        actor_inner.SetVisibility(False)

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
                label="end-effector trace",
            )
        else:
            actor = plotter.add_mesh(
                pv.lines_from_points(pts),
                color="#d81b60",
                line_width=5.0,
                opacity=1.0,
                label="end-effector trace",
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

    def replay_button_callback(_state: bool) -> None:
        start_replay()

    default_visibility = {
        "Manifolds": True,
        "Obstacles": True,
        "Evidence": bool(show_exploration),
        "Frontiers": bool(show_exploration),
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


__all__ = [
    "add_robot_pedestal",
    "make_link_mesh",
    "make_robot_actor_bundle",
    "show_pyvista_robot_demo",
    "update_actor_mesh",
    "update_robot_actor_bundle",
]