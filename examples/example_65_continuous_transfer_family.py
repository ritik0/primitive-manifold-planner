from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from ompl import util as ou

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from primitive_manifold_planner.experiments.continuous_transfer import (  # noqa: E402
    default_example_65_scene_description,
    obstacle_profile_comparison_row,
    plan_continuous_transfer_route,
    print_continuous_route_summary,
    show_continuous_route,
)

DEFAULT_OBSTACLE_SWEEP = [
    "none",
    "block_nominal_corridor",
    "block_positive_corridor",
    "block_negative_corridor",
    "split_nominal_window",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Example 65 staged continuous-transfer planning.")
    parser.add_argument("--seed", type=int, default=41, help="Deterministic seed used for NumPy and OMPL.")
    parser.add_argument("--max-probes", type=int, default=None, help="Optional family-stage ambient probe budget override.")
    parser.add_argument("--top-k", type=int, default=3, help="How many exit-seed route candidates to keep for final route selection.")
    parser.add_argument("--top-k-paths", type=int, default=1, help="How many distinct final certified routes to retain and report.")
    parser.add_argument(
        "--obstacle-profile",
        type=str,
        default="none",
        help=(
            "Named family obstacle profile for falsification runs. "
            "Options: none, block_nominal_corridor, block_positive_corridor, "
            "block_negative_corridor, split_nominal_window."
        ),
    )
    parser.add_argument(
        "--batch-obstacles",
        action="store_true",
        help="Run the standard obstacle-profile sweep and print a compact comparison table.",
    )
    parser.add_argument(
        "--comparison-row-json",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--stop-after-first-solution",
        action="store_true",
        help="Disable extra family-stage exploration after the first discovered exit.",
    )
    parser.add_argument(
        "--extra-rounds-after-first-solution",
        type=int,
        default=None,
        help="Optional number of extra family-stage rounds after the first discovered exit.",
    )
    parser.add_argument("--no-viz", action="store_true", help="Skip the PyVista visualization window.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    ou.RNG.setSeed(args.seed)
    ou.setLogLevel(ou.LOG_ERROR)
    scene_description = default_example_65_scene_description(obstacle_profile=args.obstacle_profile)

    if args.comparison_row_json:
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
        print(json.dumps(obstacle_profile_comparison_row(result)))
        return

    if args.batch_obstacles:
        rows = []
        for obstacle_profile in DEFAULT_OBSTACLE_SWEEP:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--seed",
                str(args.seed),
                "--obstacle-profile",
                obstacle_profile,
                "--top-k",
                str(args.top_k),
                "--top-k-paths",
                str(args.top_k_paths),
                "--comparison-row-json",
                "--no-viz",
            ]
            if args.max_probes is not None:
                command.extend(["--max-probes", str(args.max_probes)])
            if args.stop_after_first_solution:
                command.append("--stop-after-first-solution")
            if args.extra_rounds_after_first_solution is not None:
                command.extend(
                    [
                        "--extra-rounds-after-first-solution",
                        str(args.extra_rounds_after_first_solution),
                    ]
                )
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            stdout = completed.stdout.strip()
            if len(stdout) == 0:
                raise RuntimeError(f"No comparison row was returned for obstacle profile '{obstacle_profile}'.")
            rows.append(json.loads(stdout))
        columns = [
            "profile",
            "success",
            "primary_entry_lambda",
            "same_leaf_successful_exit_found",
            "same_leaf_stagnation_triggered",
            "first_transverse_switch_reason",
            "transverse_switch_reason_counts",
            "explored_lambda_regions",
            "final_route_same_leaf_only",
        ]
        widths = {
            column: max(len(column), *(len(str(row[column])) for row in rows))
            for column in columns
        }
        header = " | ".join(column.ljust(widths[column]) for column in columns)
        divider = "-+-".join("-" * widths[column] for column in columns)
        print(header)
        print(divider)
        for row in rows:
            print(" | ".join(str(row[column]).ljust(widths[column]) for column in columns))
        return

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
    print_continuous_route_summary(result)
    if not args.no_viz:
        show_continuous_route(result)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
