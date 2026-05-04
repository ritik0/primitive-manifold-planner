from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_OBSTACLE_SWEEP = [
    "none",
    "block_nominal_corridor",
    "block_positive_corridor",
    "block_negative_corridor",
    "split_nominal_window",
]


def run_obstacle_sweep(
    *,
    seed: int,
    top_k: int,
    top_k_paths: int,
    max_probes: int | None = None,
    stop_after_first_solution: bool = False,
    extra_rounds_after_first_solution: int | None = None,
) -> list[dict[str, object]]:
    script_path = Path(__file__).resolve().parents[1] / "examples" / "example_65_continuous_transfer_family.py"
    rows: list[dict[str, object]] = []
    for obstacle_profile in DEFAULT_OBSTACLE_SWEEP:
        command = [
            sys.executable,
            str(script_path),
            "--seed",
            str(seed),
            "--obstacle-profile",
            obstacle_profile,
            "--top-k",
            str(top_k),
            "--top-k-paths",
            str(top_k_paths),
            "--comparison-row-json",
            "--no-viz",
        ]
        if max_probes is not None:
            command.extend(["--max-probes", str(max_probes)])
        if stop_after_first_solution:
            command.append("--stop-after-first-solution")
        if extra_rounds_after_first_solution is not None:
            command.extend(
                [
                    "--extra-rounds-after-first-solution",
                    str(extra_rounds_after_first_solution),
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
    return rows


def print_obstacle_sweep_table(rows: list[dict[str, object]]) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Example 65 obstacle-profile sweep.")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--top-k-paths", type=int, default=1)
    parser.add_argument("--max-probes", type=int, default=None)
    parser.add_argument("--stop-after-first-solution", action="store_true")
    parser.add_argument("--extra-rounds-after-first-solution", type=int, default=None)
    args = parser.parse_args()

    rows = run_obstacle_sweep(
        seed=int(args.seed),
        top_k=int(args.top_k),
        top_k_paths=int(args.top_k_paths),
        max_probes=args.max_probes,
        stop_after_first_solution=bool(args.stop_after_first_solution),
        extra_rounds_after_first_solution=args.extra_rounds_after_first_solution,
    )
    print_obstacle_sweep_table(rows)


if __name__ == "__main__":
    main()
