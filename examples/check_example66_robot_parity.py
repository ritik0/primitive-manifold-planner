from __future__ import annotations

"""Lightweight parity check for Example 66 and the 3DOF robot add-on.

The important invariant is architectural rather than visual: default robot
mode runs the same task-space planner as the non-robot example, then tracks
only the selected final route with IK. Planner evidence is not robot motion.
"""

import argparse

import numpy as np
from ompl import util as ou

from primitive_manifold_planner.examplesupport.example66_scene import build_example66_scene
from primitive_manifold_planner.thesis import parallel_evidence_planner as ex66

from three_dof_robot_pyvista_demo import (
    SpatialRobot3DOF,
    build_robot_execution,
    configure_example_66_budgets,
    planner_parity_stats,
)


def polyline_length(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that the 3DOF robot demo preserves Example 66 task-space planner evidence."
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-tracking-error", type=float, default=2.0e-2)
    parser.add_argument("--max-joint-step", type=float, default=9.0e-2)
    parser.add_argument("--full", action="store_true", help="Use the normal budget instead of the fast smoke budget.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    ou.RNG.setSeed(args.seed)
    ou.setLogLevel(ou.LOG_ERROR)

    configure_example_66_budgets(
        argparse.Namespace(
            max_rounds=None,
            max_iters=None,
            fast=not args.full,
            stop_after_first_solution=False,
            continue_after_first_solution=True,
        )
    )
    ex66.STOP_AFTER_FIRST_SOLUTION = False

    families, start_q, goal_q, _plane_half_u, _plane_half_v = build_example66_scene()
    result = ex66.plan_fixed_manifold_multimodal_route(
        families=families,
        start_q=start_q,
        goal_q=goal_q,
        serial_mode=False,
    )
    route = np.asarray(result.path if len(result.path) >= 2 else result.raw_path, dtype=float)
    robot = SpatialRobot3DOF(
        link_lengths=np.asarray([1.35, 1.05, 0.75], dtype=float),
        base_world=np.asarray([0.0, -1.25, 0.10], dtype=float),
    )
    execution = build_robot_execution(result, robot, use_planner_joint_path=False) if result.success else None
    stats = planner_parity_stats(result, "task_space", execution)

    checks: list[tuple[str, bool, object]] = [
        ("planner_success", bool(result.success), result.message),
        ("nontrivial_total_nodes", int(stats["nodes_explored"]) > 50, stats["nodes_explored"]),
        ("nontrivial_plane_nodes", int(stats["plane_nodes"]) > 20, stats["plane_nodes"]),
        ("entry_transitions_found", int(stats["entry_transitions_found"]) > 0, stats["entry_transitions_found"]),
        ("exit_transitions_found", int(stats["exit_transitions_found"]) > 0, stats["exit_transitions_found"]),
        ("continued_after_solution", bool(stats["continued_after_solution"]), stats["continued_after_solution"]),
        ("final_path_exists", len(route) >= 2, len(route)),
        ("robot_execution_exists", execution is not None, execution is None),
    ]
    if execution is not None:
        target_length = polyline_length(execution.target_task_points_3d)
        trace_length = polyline_length(execution.end_effector_points_3d)
        ratio = trace_length / max(target_length, 1.0e-9)
        checks.extend(
            [
                ("execution_success", bool(execution.execution_success), execution.diagnostics),
                ("tracking_error_below_threshold", execution.max_tracking_error <= args.max_tracking_error, execution.max_tracking_error),
                ("joint_step_below_threshold", execution.max_joint_step <= args.max_joint_step, execution.max_joint_step),
                ("ee_trace_length_matches_target", 0.92 <= ratio <= 1.08, ratio),
                ("robot_tracks_resampled_final_path", bool(execution.planner_path_resampled_for_robot), execution.planner_path_resampled_for_robot),
                ("not_using_jointspace_path_by_default", not execution.planner_joint_path_used_directly, execution.planner_joint_path_used_directly),
            ]
        )

    print("Example 66 robot parity check")
    for key, value in stats.items():
        print(f"{key} = {value}")
    if execution is not None:
        print(f"max_tracking_error = {execution.max_tracking_error:.6f}")
        print(f"mean_tracking_error = {execution.mean_tracking_error:.6f}")
        print(f"max_joint_step = {execution.max_joint_step:.6f}")
        print(f"execution_waypoints = {len(execution.joint_path)}")

    failed = [(name, detail) for name, ok, detail in checks if not ok]
    if failed:
        print("\nFAILED CHECKS")
        for name, detail in failed:
            print(f"{name}: {detail}")
        return 1

    print("\nAll parity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
