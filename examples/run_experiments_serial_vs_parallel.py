from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
from ompl import util as ou

import example_66_multimodal_graph_search as ex66
from collision_utilities import default_example_66_obstacles
from example_66_1_three_dof_robot_pyvista import SpatialRobot3DOF


def fixed_robot() -> SpatialRobot3DOF:
    return SpatialRobot3DOF(
        link_lengths=np.asarray([1.35, 1.05, 0.75], dtype=float),
        base_world=np.asarray([0.0, -1.25, 0.10], dtype=float),
        link_radius=0.06,
        joint_radius=0.10,
        ee_radius=0.08,
    )


def configure_seed(seed: int) -> None:
    np.random.seed(int(seed))
    random.seed(int(seed))
    ou.RNG.setSeed(int(seed))
    ou.setLogLevel(ou.LOG_ERROR)


def run_once(seed: int, serial_mode: bool) -> dict[str, object]:
    configure_seed(seed)
    families, start_q, goal_q, _plane_half_u, _plane_half_v = ex66.build_scene()
    robot = fixed_robot()
    obstacles = default_example_66_obstacles()
    t0 = time.perf_counter()
    result = ex66.plan_fixed_manifold_multimodal_route(
        families=families,
        start_q=start_q,
        goal_q=goal_q,
        robot=robot,
        serial_mode=serial_mode,
        obstacles=obstacles,
    )
    elapsed = time.perf_counter() - t0
    return {
        "seed": int(seed),
        "mode": "serial" if serial_mode else "parallel",
        "serial_mode": bool(serial_mode),
        "success": bool(result.success),
        "planning_time_s": float(elapsed),
        "total_rounds": int(result.total_rounds),
        "candidate_evaluations": int(result.candidate_evaluations),
        "route_cost": float(result.route_cost_display),
        "left_evidence_nodes": int(result.left_evidence_nodes),
        "plane_evidence_nodes": int(result.plane_evidence_nodes),
        "right_evidence_nodes": int(result.right_evidence_nodes),
        "saturated_before_solution": bool(result.saturated_before_solution),
        "stagnation_stage": "" if result.stagnation_stage is None else str(result.stagnation_stage),
        "message": str(result.message),
        "obstacle_count": int(len(obstacles)),
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def summarize_rows(rows: list[dict[str, object]], label: str) -> None:
    success_values = [1.0 if bool(row["success"]) else 0.0 for row in rows]
    time_values = [float(row["planning_time_s"]) for row in rows]
    round_values = [float(row["total_rounds"]) for row in rows]
    eval_values = [float(row["candidate_evaluations"]) for row in rows]
    cost_values = [float(row["route_cost"]) for row in rows if bool(row["success"])]
    left_values = [float(row["left_evidence_nodes"]) for row in rows]
    plane_values = [float(row["plane_evidence_nodes"]) for row in rows]
    right_values = [float(row["right_evidence_nodes"]) for row in rows]
    saturated_count = sum(1 for row in rows if bool(row["saturated_before_solution"]))

    success_mean, success_std = _mean_std(success_values)
    time_mean, time_std = _mean_std(time_values)
    rounds_mean, rounds_std = _mean_std(round_values)
    eval_mean, eval_std = _mean_std(eval_values)
    cost_mean, cost_std = _mean_std(cost_values)
    left_mean, left_std = _mean_std(left_values)
    plane_mean, plane_std = _mean_std(plane_values)
    right_mean, right_std = _mean_std(right_values)

    print(f"\n{label}")
    print(f"runs = {len(rows)}")
    print(f"success_rate_mean = {success_mean:.3f}")
    print(f"success_rate_std = {success_std:.3f}")
    print(f"planning_time_mean_s = {time_mean:.4f}")
    print(f"planning_time_std_s = {time_std:.4f}")
    print(f"total_rounds_mean = {rounds_mean:.3f}")
    print(f"total_rounds_std = {rounds_std:.3f}")
    print(f"candidate_evaluations_mean = {eval_mean:.3f}")
    print(f"candidate_evaluations_std = {eval_std:.3f}")
    print(f"route_cost_mean_success_only = {cost_mean:.4f}")
    print(f"route_cost_std_success_only = {cost_std:.4f}")
    print(f"left_nodes_mean = {left_mean:.3f} +/- {left_std:.3f}")
    print(f"plane_nodes_mean = {plane_mean:.3f} +/- {plane_std:.3f}")
    print(f"right_nodes_mean = {right_mean:.3f} +/- {right_std:.3f}")
    print(f"saturated_before_solution_count = {saturated_count}")


def save_rows_to_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "mode",
        "serial_mode",
        "success",
        "planning_time_s",
        "total_rounds",
        "candidate_evaluations",
        "route_cost",
        "left_evidence_nodes",
        "plane_evidence_nodes",
        "right_evidence_nodes",
        "saturated_before_solution",
        "stagnation_stage",
        "message",
        "obstacle_count",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def concluding_statement(parallel_rows: list[dict[str, object]], serial_rows: list[dict[str, object]]) -> str:
    parallel_success = np.mean([1.0 if bool(row["success"]) else 0.0 for row in parallel_rows])
    serial_success = np.mean([1.0 if bool(row["success"]) else 0.0 for row in serial_rows])
    parallel_time = np.mean([float(row["planning_time_s"]) for row in parallel_rows])
    serial_time = np.mean([float(row["planning_time_s"]) for row in serial_rows])

    success_delta = float(parallel_success - serial_success)
    if serial_time <= 1e-9:
        time_delta_pct = 0.0
    else:
        time_delta_pct = 100.0 * float((serial_time - parallel_time) / serial_time)

    if success_delta >= 0.10 and time_delta_pct >= 5.0:
        verdict = "Parallel exploration shows a clear empirical advantage in both success rate and planning time on this benchmark."
    elif success_delta >= 0.10:
        verdict = "Parallel exploration shows a clear empirical advantage in success rate on this benchmark, with smaller timing differences."
    elif time_delta_pct >= 10.0:
        verdict = "Parallel exploration shows a clear empirical timing advantage on this benchmark, while success rates remain closer."
    else:
        verdict = "Parallel exploration does not show a strong advantage on this benchmark under the current settings; the two modes perform comparably."

    return (
        verdict
        + f" Parallel success rate = {parallel_success:.3f}, serial success rate = {serial_success:.3f},"
        + f" parallel mean time = {parallel_time:.4f}s, serial mean time = {serial_time:.4f}s."
    )


def run_batch(runs_per_mode: int, start_seed: int, output_path: Path) -> None:
    rows: list[dict[str, object]] = []
    parallel_rows: list[dict[str, object]] = []
    serial_rows: list[dict[str, object]] = []

    for idx in range(int(runs_per_mode)):
        parallel_seed = int(start_seed + idx)
        serial_seed = int(start_seed + runs_per_mode + idx)
        parallel_row = run_once(seed=parallel_seed, serial_mode=False)
        serial_row = run_once(seed=serial_seed, serial_mode=True)
        parallel_rows.append(parallel_row)
        serial_rows.append(serial_row)
        rows.extend([parallel_row, serial_row])
        print(
            f"completed pair {idx + 1}/{runs_per_mode}: "
            f"parallel_success={parallel_row['success']}, serial_success={serial_row['success']}"
        )

    summarize_rows(parallel_rows, "Parallel Results")
    summarize_rows(serial_rows, "Serial Results")
    statement = concluding_statement(parallel_rows, serial_rows)
    print("\nConclusion")
    print(statement)
    save_rows_to_csv(rows, output_path)
    print(f"\ncsv_saved_to = {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Example 66 robot joint-space planning in serial vs parallel evidence modes.",
    )
    parser.add_argument("--serial", action="store_true", help="Run a single serial-mode planning trial.")
    parser.add_argument("--batch", action="store_true", help="Run the full 30-vs-30 serial/parallel comparison.")
    parser.add_argument("--seed", type=int, default=41, help="Seed for a single run or the start seed for batch mode.")
    parser.add_argument("--runs", type=int, default=30, help="Runs per mode in batch mode.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples") / "serial_vs_parallel_results.csv",
        help="CSV output path for batch mode.",
    )
    args = parser.parse_args()

    if args.batch:
        run_batch(runs_per_mode=int(args.runs), start_seed=int(args.seed), output_path=Path(args.output))
        return

    row = run_once(seed=int(args.seed), serial_mode=bool(args.serial))
    print("\nExample 66 Serial vs Parallel Single Run")
    print(f"mode = {row['mode']}")
    print(f"seed = {row['seed']}")
    print(f"success = {row['success']}")
    print(f"planning_time_s = {float(row['planning_time_s']):.4f}")
    print(f"total_rounds = {row['total_rounds']}")
    print(f"candidate_evaluations = {row['candidate_evaluations']}")
    print(f"route_cost = {float(row['route_cost']):.4f}")
    print(f"left_evidence_nodes = {row['left_evidence_nodes']}")
    print(f"plane_evidence_nodes = {row['plane_evidence_nodes']}")
    print(f"right_evidence_nodes = {row['right_evidence_nodes']}")
    print(f"saturated_before_solution = {row['saturated_before_solution']}")
    print(f"stagnation_stage = {row['stagnation_stage']}")
    print(f"message = {row['message']}")
    print(f"obstacle_count = {row['obstacle_count']}")


if __name__ == "__main__":
    main()
