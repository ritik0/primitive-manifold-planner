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
    center = np.asarray(robot.base_world, dtype=float) + np.asarray([0.0, 0.0, -0.18], dtype=float)
    return plotter.add_mesh(
        pv.Cylinder(center=center, direction=(0.0, 0.0, 1.0), radius=0.18, height=0.32, resolution=32),
        color="#5d4037",
        opacity=1.0,
        smooth_shading=True,
        label="robot pedestal",
    )


def make_robot_actor_bundle(plotter, robot, joint_angles: np.ndarray, opacity: float = 1.0):
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


def update_robot_actor_bundle(plotter, robot, bundle, joint_angles: np.ndarray) -> None:
    joints = robot.forward_kinematics_3d(joint_angles)
    update_actor_mesh(bundle.get("base"), pv.Sphere(radius=float(robot.joint_radius), center=np.asarray(robot.base_world, dtype=float)))
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
        "Example 66.1: planner evidence plus 3DOF robot tracking only the selected final route",
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

    plane_corners = plane_patch_corners(plane_manifold, half_u=plane_half_u, half_v=plane_half_v)
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

    # Planner evidence is rendered for diagnosis. Robot animation below uses
    # only the selected final route / execution path, never these branches.
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

    dense_joint_path = np.asarray(getattr(result, "dense_joint_path", np.zeros((0, 3), dtype=float)), dtype=float)
    dense_execution_certified = bool(getattr(result, "dense_joint_path_execution_certified", False))
    jointspace_display_route = (
        np.asarray([robot.forward_kinematics_3d(theta)[-1] for theta in dense_joint_path], dtype=float)
        if dense_execution_certified and len(dense_joint_path) >= 2
        else np.zeros((0, 3), dtype=float)
    )

    if len(jointspace_display_route) >= 2:
        # In joint-space mode, this FK trace is the source of truth: the
        # planner-certified dense joint path and the robot execution share it.
        skeleton = np.asarray(result.path if len(result.path) >= 2 else result.raw_path, dtype=float)
        if len(skeleton) >= 2:
            actor = plotter.add_mesh(
                pv.lines_from_points(skeleton),
                color="#546e7a",
                line_width=2.0,
                opacity=0.28,
                label="SPARSE/ABSTRACT ROUTE",
            )
            if actor is not None:
                actor_groups["Committed"].append(actor)
        actor = plotter.add_mesh(
            pv.lines_from_points(jointspace_display_route),
            color="#d32f2f",
            line_width=8.0,
            label="FINAL EXECUTED ROUTE",
        )
        if actor is not None:
            actor_groups["Committed"].append(actor)
    else:
        if len(result.raw_path) >= 2:
            raw_polyline = pv.lines_from_points(np.asarray(result.raw_path, dtype=float))
            actor = plotter.add_mesh(raw_polyline, color="#90a4ae", line_width=3.0, opacity=0.8, label="certified route")
            if actor is not None:
                actor_groups["Committed"].append(actor)
        if len(result.path) >= 2:
            display_polyline = pv.lines_from_points(np.asarray(result.path, dtype=float))
            actor = plotter.add_mesh(display_polyline, color="#d32f2f", line_width=8.0, label="FINAL PLANNER ROUTE")
            if actor is not None:
                actor_groups["Committed"].append(actor)

    if (
        bool(show_rejected_joint_interpolation)
        and
        robot_execution is not None
        and not bool(getattr(robot_execution, "execution_success", False))
        and len(getattr(robot_execution, "end_effector_points_3d", [])) >= 2
        and str(getattr(robot_execution, "execution_source", "")) in {"disabled", "uncertified_direct_joint_path"}
    ):
        rejected_polyline = pv.lines_from_points(np.asarray(robot_execution.end_effector_points_3d, dtype=float))
        actor = plotter.add_mesh(
            rejected_polyline,
            color="#b71c1c",
            line_width=3.0,
            opacity=0.22,
            label="REJECTED UNCERTIFIED JOINT INTERPOLATION",
        )
        if actor is not None:
            actor_groups["EETrace"].append(actor)

    if robot_execution is not None and not bool(getattr(robot_execution, "execution_success", True)):
        warning_text = "Joint-space execution not certified: robot animation disabled"
        if str(getattr(robot_execution, "execution_source", "")) == "disabled_joint_step_violation":
            warning_text = "Joint route satisfies constraints but has discontinuous joint jump; animation disabled"
        plotter.add_text(
            warning_text,
            position=(760, 52),
            font_size=10,
            color="#b71c1c",
        )

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
                label="ROBOT END-EFFECTOR TRACE",
            )
        else:
            actor = plotter.add_mesh(
                pv.lines_from_points(pts),
                color="#d81b60",
                line_width=5.0,
                opacity=1.0,
                label="ROBOT END-EFFECTOR TRACE",
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
        "Evidence": bool(show_exploration),
        "Frontiers": bool(show_exploration),
        "Hypotheses": bool(show_exploration),
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
