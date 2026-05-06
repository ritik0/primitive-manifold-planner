from __future__ import annotations

"""Continuous-transfer planner with 3DOF robot task-space execution.

Phase 1 intentionally keeps planning in task space:
- the continuous-transfer evidence graph discovers entry/exit/lambda structure;
- the final route is the selected-transition local replan from that planner;
- the robot tracks only that final route with sequential IK.

Planner evidence is diagnostic context. Robot motion is not allowed to follow
raw exploration branches.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from ompl import util as ou

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXAMPLES = ROOT / "examples"
for path in (ROOT, SRC, EXAMPLES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from primitive_manifold_planner.examplesupport.intrinsic_multimodal_helpers import build_segment_polydata
from primitive_manifold_planner.experiments.continuous_transfer import (
    build_continuous_transfer_scene,
    default_example_65_scene_description,
    plan_continuous_transfer_route,
    print_continuous_route_summary,
)
from primitive_manifold_planner.experiments.continuous_transfer.config import DEFAULT_POST_ROUTE_EVIDENCE_ROUNDS
from primitive_manifold_planner.experiments.continuous_transfer.family_definition import plane_leaf_patch
from primitive_manifold_planner.visualization import add_manifold, add_points, pyvista_available
from primitive_manifold_planner.visualization.robot import (
    add_robot_pedestal,
    make_robot_actor_bundle,
    update_robot_actor_bundle,
)

from three_dof_robot_pyvista_demo import (
    RobotExecutionResult,
    build_continuous_robot_execution_path,
    choose_robot_for_route,
)

try:
    import pyvista as pv
except Exception:
    pv = None


def _polyline_error(reference: np.ndarray, trace: np.ndarray) -> tuple[float, float]:
    ref = np.asarray(reference, dtype=float)
    pts = np.asarray(trace, dtype=float)
    if len(ref) == 0 or len(pts) == 0:
        return float("inf"), float("inf")
    if len(ref) == len(pts):
        errors = np.linalg.norm(ref - pts, axis=1)
    else:
        errors = np.asarray([float(np.min(np.linalg.norm(ref - p, axis=1))) for p in pts], dtype=float)
    return float(np.max(errors)), float(np.mean(errors))


def _print_key_value_block(title: str, values: dict[str, object]) -> None:
    print(f"\n=== {title} ===")
    width = max((len(str(key)) for key in values), default=1)
    for key, value in values.items():
        print(f"{key.ljust(width)} : {value}")


def plan_continuous_transfer_jointspace_robot(*, scene_description, seed: int, joint_max_step: float, **planner_kwargs):
    """Scaffold for Phase 2 direct joint-space continuous-family planning.

    The task-space mode in this file is complete: the foliation planner selects
    entry/exit/lambda structure, locally replans the final task route, and the
    robot tracks only that route by IK. The future joint-space mode should not
    reuse that IK layer as planning. It should mirror the fixed-manifold robot
    planner, but with a leaf-managed transfer family:

    - state: q = [theta0, theta1, theta2]
    - task point: FK(q)
    - left leaf: RobotSphereManifold(FK(q) on the left support sphere)
    - right leaf: RobotSphereManifold(FK(q) on the right support sphere)
    - transfer leaf: RobotPlaneLeafManifold / RobotFamilyLeafManifold

    RobotPlaneLeafManifold design notes:
    - stores a locked lambda value and a reference to the continuous transfer
      family;
    - residual(q) evaluates the active plane-leaf constraint at FK(q);
    - within_bounds(q) checks the family patch/obstacle mask at FK(q);
    - project(q) remains permissive enough for exploration, like the existing
      RobotConstraintBase.project();
    - project_local(q) may preserve IK branch for final selected-transition
      realization;
    - infer_lambda(q) delegates to family.infer_lambda(FK(q)) and candidate
      lambdas are managed by a robot leaf-store manager.

    Final execution rule for the future implementation:
    the evidence graph discovers candidate lambdas and transitions only. After
    selecting entry/exit/lambda, the executable robot route must be locally
    replanned as start_q -> entry_q on the left robot sphere, entry_q -> exit_q
    on the selected robot family leaf, and exit_q -> goal_q on the right robot
    sphere. The displayed route must be FK(result.dense_joint_path), and robot
    animation must use that exact dense_joint_path.
    """

    _print_key_value_block(
        "Continuous-Transfer Joint-Space Scaffold",
        {
            "planning_mode": "jointspace_constrained_continuous_transfer",
            "implementation_status": "pending_phase_2b",
            "taskspace_mode_status": "complete_selected_transition_replan_plus_ik_tracking",
            "state_space": "robot_joint_angles_q=[theta0, theta1, theta2]",
            "task_map": "FK(q)",
            "left_constraint": "FK(q) on left support sphere",
            "family_constraint": "FK(q) on selected/inferred transfer family leaf lambda",
            "right_constraint": "FK(q) on right support sphere",
            "lambda_policy": "infer from FK(q), then manage locked robot family leaf stores",
            "execution_rule": "selected-transition local replan in joint space, never raw graph execution",
            "display_rule": "route_source = FK(result.dense_joint_path)",
            "seed": int(seed),
            "joint_max_step": float(joint_max_step),
            "planner_kwargs_received": sorted(str(key) for key in planner_kwargs.keys()),
            "next_step": "implement RobotPlaneLeafManifold and robot leaf evidence manager",
        },
    )
    raise NotImplementedError(
        "Direct joint-space continuous-transfer planning is scaffolded for Phase 2B. "
        "Use --taskspace-planning for the working Phase 1 robot execution demo."
    )


def show_continuous_transfer_robot_demo(
    *,
    scene,
    result,
    robot,
    robot_execution: RobotExecutionResult | None,
    show_exploration: bool = True,
) -> bool:
    if pv is None or not pyvista_available():
        print("PyVista is not available; skipping continuous-transfer robot visualization.")
        return False

    left_family = scene.left_support
    transfer_family = scene.transfer_family
    right_family = scene.right_support
    left_manifold = left_family.manifold(float(left_family.sample_lambdas()[0]))
    right_manifold = right_family.manifold(float(right_family.sample_lambdas()[0]))
    selected_lambda = (
        float(result.selected_lambda_for_realization)
        if result.selected_lambda_for_realization is not None
        else float(result.selected_lambda)
        if result.selected_lambda is not None
        else float(transfer_family.nominal_lambda)
    )
    family_leaf = transfer_family.manifold(float(selected_lambda))

    plotter = pv.Plotter(window_size=(1440, 920))
    if hasattr(plotter, "set_background"):
        plotter.set_background("#edf3f8", top="#ffffff")
    if hasattr(plotter, "enable_anti_aliasing"):
        try:
            plotter.enable_anti_aliasing()
        except Exception:
            pass
    plotter.add_text(
        "Continuous transfer: evidence graph plus 3DOF robot tracking only the selected final route",
        font_size=12,
    )

    actor_groups: dict[str, list[object]] = {
        "Manifolds": [],
        "Evidence": [],
        "Transitions": [],
        "FinalRoute": [],
        "Robot": [],
        "EETrace": [],
        "StartGoal": [],
    }

    for manifold, color in ((left_manifold, "#c58b4c"), (right_manifold, "#c58b4c")):
        actor = add_manifold(plotter, manifold, color=color, opacity=0.10)
        if actor is not None:
            actor_groups["Manifolds"].append(actor)

    leaf_corners = plane_leaf_patch(transfer_family, float(selected_lambda))
    leaf_patch = pv.PolyData(leaf_corners, faces=np.hstack([[4, 0, 1, 2, 3]]))
    actor = plotter.add_mesh(
        leaf_patch,
        color="#7fa7c6",
        opacity=0.22,
        show_edges=False,
        label=f"selected transfer leaf lambda={selected_lambda:.3f}",
    )
    if actor is not None:
        actor_groups["Manifolds"].append(actor)

    if show_exploration:
        colors = {
            "left": "#81c784",
            "family_leaf": "#64b5f6",
            "family_transverse": "#8e24aa",
            "right": "#a5d6a7",
        }
        for mode, edges in result.explored_edges_by_mode.items():
            poly = build_segment_polydata(edges)
            if poly is None:
                continue
            actor = plotter.add_mesh(
                poly,
                color=colors.get(str(mode), "#78909c"),
                opacity=0.28,
                line_width=1.8,
                label=f"{mode} evidence",
            )
            if actor is not None:
                actor_groups["Evidence"].append(actor)

    if len(result.entry_transition_points) > 0:
        actor = add_points(plotter, result.entry_transition_points, color="#ff7043", size=9.0, label="entry transitions")
        if actor is not None:
            actor_groups["Transitions"].append(actor)
    if len(result.exit_transition_points) > 0:
        actor = add_points(plotter, result.exit_transition_points, color="#26a69a", size=9.0, label="exit transitions")
        if actor is not None:
            actor_groups["Transitions"].append(actor)

    final_route = np.asarray(result.path, dtype=float)
    if len(final_route) >= 2:
        actor = plotter.add_mesh(
            pv.lines_from_points(final_route),
            color="#d32f2f",
            line_width=8.0,
            label="FINAL SELECTED-TRANSITION ROUTE",
        )
        if actor is not None:
            actor_groups["FinalRoute"].append(actor)

    actor = add_points(plotter, scene.start_q, color="black", size=16.0, label="start")
    if actor is not None:
        actor_groups["StartGoal"].append(actor)
    actor = add_points(plotter, scene.goal_q, color="gold", size=18.0, label="goal")
    if actor is not None:
        actor_groups["StartGoal"].append(actor)

    initial_angles = (
        np.asarray(robot_execution.joint_path[0], dtype=float)
        if robot_execution is not None and len(robot_execution.joint_path) > 0
        else np.asarray([0.0, 0.55, -0.85], dtype=float)
    )
    robot_bundle = make_robot_actor_bundle(plotter, robot, initial_angles, opacity=1.0)
    pedestal_actor = add_robot_pedestal(plotter, robot)
    if pedestal_actor is not None:
        actor_groups["Robot"].append(pedestal_actor)
    actor_groups["Robot"].extend(robot_bundle.get("all", []))

    trace_actor = {"actor": None}

    def replace_trace(points: np.ndarray) -> None:
        if trace_actor["actor"] is not None:
            plotter.remove_actor(trace_actor["actor"], render=False)
        pts = np.asarray(points, dtype=float)
        if len(pts) == 0:
            trace_actor["actor"] = None
            return
        if len(pts) == 1:
            actor_inner = plotter.add_mesh(
                pv.Sphere(radius=float(0.75 * robot.ee_radius), center=pts[0]),
                color="#d81b60",
                label="ROBOT END-EFFECTOR TRACE",
            )
        else:
            actor_inner = plotter.add_mesh(
                pv.lines_from_points(pts),
                color="#d81b60",
                line_width=5.0,
                label="ROBOT END-EFFECTOR TRACE",
            )
        trace_actor["actor"] = actor_inner

    animation_state = {"frame": 0, "playing": False, "trace": []}

    def reset_animation() -> None:
        animation_state["frame"] = 0
        animation_state["playing"] = bool(robot_execution is not None and robot_execution.animation_enabled)
        animation_state["trace"] = []
        if robot_execution is not None and len(robot_execution.joint_path) > 0:
            update_robot_actor_bundle(plotter, robot, robot_bundle, robot_execution.joint_path[0])
        replace_trace(np.zeros((0, 3), dtype=float))
        plotter.render()

    def animate_step() -> None:
        if robot_execution is None or not robot_execution.animation_enabled or not animation_state["playing"]:
            return
        idx = min(int(animation_state["frame"]), len(robot_execution.joint_path) - 1)
        update_robot_actor_bundle(plotter, robot, robot_bundle, robot_execution.joint_path[idx])
        animation_state["trace"].append(np.asarray(robot_execution.end_effector_points_3d[idx], dtype=float))
        replace_trace(np.asarray(animation_state["trace"], dtype=float))
        if idx >= len(robot_execution.joint_path) - 1:
            animation_state["playing"] = False
        else:
            animation_state["frame"] += 1
        plotter.render()

    def play_animation() -> None:
        import time

        while bool(animation_state["playing"]):
            animate_step()
            plotter.update()
            time.sleep(0.02)

    def start_replay() -> None:
        reset_animation()
        play_animation()

    if hasattr(plotter, "add_key_event"):
        plotter.add_key_event("r", start_replay)
    plotter.add_text("Press r to replay robot motion", position=(1030, 18), font_size=10, color="black")
    plotter.add_axes()
    plotter.show_grid()
    plotter.camera_position = [
        (0.25, -7.8, 3.8),
        (0.0, -0.05, 0.35),
        (0.0, 0.0, 1.0),
    ]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous-transfer planner with 3DOF robot execution.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--taskspace-planning", action="store_true", help="Run task-space continuous-transfer planning plus IK execution.")
    mode.add_argument("--jointspace-planning", action="store_true", help="Reserved for Phase 2 joint-space continuous-family planning.")
    parser.set_defaults(taskspace_planning=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-probes", type=int, default=None)
    parser.add_argument("--extra-rounds-after-first-solution", type=int, default=DEFAULT_POST_ROUTE_EVIDENCE_ROUNDS)
    parser.add_argument("--stop-after-first-solution", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--top-k-paths", type=int, default=1)
    parser.add_argument("--obstacle-profile", type=str, default="none")
    parser.add_argument("--max-cartesian-step", type=float, default=0.025)
    parser.add_argument("--joint-max-step", type=float, default=0.12)
    parser.add_argument("--ik-tolerance", type=float, default=3.0e-3)
    parser.add_argument("--show-exploration", dest="show_exploration", action="store_true", default=True)
    parser.add_argument("--hide-exploration", dest="show_exploration", action="store_false")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    if args.jointspace_planning:
        scene_description = default_example_65_scene_description(obstacle_profile=args.obstacle_profile)
        try:
            plan_continuous_transfer_jointspace_robot(
                scene_description=scene_description,
                seed=int(args.seed),
                joint_max_step=float(args.joint_max_step),
                max_ambient_probes=args.max_probes,
                continue_after_first_solution=not args.stop_after_first_solution,
                max_extra_rounds_after_first_solution=args.extra_rounds_after_first_solution,
                top_k_assignments=args.top_k,
                top_k_paths=args.top_k_paths,
                obstacle_profile=args.obstacle_profile,
            )
        except NotImplementedError as exc:
            raise SystemExit(str(exc)) from exc

    np.random.seed(int(args.seed))
    ou.RNG.setSeed(int(args.seed))
    ou.setLogLevel(ou.LOG_ERROR)

    scene_description = default_example_65_scene_description(obstacle_profile=args.obstacle_profile)
    scene = build_continuous_transfer_scene(scene_description)
    result = plan_continuous_transfer_route(
        max_ambient_probes=args.max_probes,
        continue_after_first_solution=not args.stop_after_first_solution,
        max_extra_rounds_after_first_solution=args.extra_rounds_after_first_solution,
        top_k_assignments=args.top_k,
        top_k_paths=args.top_k_paths,
        seed=args.seed,
        obstacle_profile=args.obstacle_profile,
        scene_description=scene_description,
    )

    route = np.asarray(result.path, dtype=float)
    robot = choose_robot_for_route(route)
    robot_execution = build_continuous_robot_execution_path(
        route,
        robot,
        max_cartesian_step=float(args.max_cartesian_step),
        max_joint_step=float(args.joint_max_step),
        ik_tolerance=float(args.ik_tolerance),
    )
    display_vs_trace_max_error, display_vs_trace_mean_error = (
        _polyline_error(
            np.asarray(robot_execution.target_task_points_3d, dtype=float),
            np.asarray(robot_execution.end_effector_points_3d, dtype=float),
        )
        if robot_execution is not None
        else (float("inf"), float("inf"))
    )

    print_continuous_route_summary(result)
    _print_key_value_block(
        "Continuous-Transfer Robot Execution",
        {
            "planning_mode": "taskspace_ik_execution",
            "planner_success": bool(result.success),
            "selected_lambda_for_realization": None
            if result.selected_lambda_for_realization is None
            else round(float(result.selected_lambda_for_realization), 6),
            "final_route_realization": result.final_route_realization,
            "graph_route_used_for_execution": bool(result.graph_route_used_for_execution),
            "strict_validation_success": bool(result.strict_validation_success),
            "robot_execution_success": bool(robot_execution.execution_success) if robot_execution is not None else False,
            "robot_animation_enabled": bool(robot_execution.animation_enabled) if robot_execution is not None else False,
            "robot_execution_waypoints": 0 if robot_execution is None else len(robot_execution.joint_path),
            "max_tracking_error": float("inf") if robot_execution is None else round(float(robot_execution.max_tracking_error), 6),
            "mean_tracking_error": float("inf") if robot_execution is None else round(float(robot_execution.mean_tracking_error), 6),
            "max_joint_step": float("inf") if robot_execution is None else round(float(robot_execution.max_joint_step), 6),
            "ik_failures": -1 if robot_execution is None else int(robot_execution.ik_failure_count),
            "display_vs_trace_max_error": round(float(display_vs_trace_max_error), 6),
            "display_vs_trace_mean_error": round(float(display_vs_trace_mean_error), 6),
            "execution_source": "none" if robot_execution is None else str(robot_execution.execution_source),
            "diagnostics": "robot execution unavailable" if robot_execution is None else str(robot_execution.diagnostics),
        },
    )

    if not args.no_viz:
        show_continuous_transfer_robot_demo(
            scene=scene,
            result=result,
            robot=robot,
            robot_execution=robot_execution,
            show_exploration=bool(args.show_exploration),
        )

    sys.stdout.flush()
    sys.stderr.flush()
    if not args.no_viz:
        os._exit(0)


if __name__ == "__main__":
    main()
