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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.obstacle_sweep import print_obstacle_sweep_table, run_obstacle_sweep
from primitive_manifold_planner.experiments.continuous_transfer import (  # noqa: E402
    default_example_65_scene_description,
    obstacle_profile_comparison_row,
    plan_continuous_transfer_route,
    print_continuous_route_summary,
    show_continuous_route,
)

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
        rows = run_obstacle_sweep(
            seed=int(args.seed),
            top_k=int(args.top_k),
            top_k_paths=int(args.top_k_paths),
            max_probes=args.max_probes,
            stop_after_first_solution=bool(args.stop_after_first_solution),
            extra_rounds_after_first_solution=args.extra_rounds_after_first_solution,
        )
        print_obstacle_sweep_table(rows)
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
