from __future__ import annotations

"""Example 67: stress-test the Example 66 fixed-manifold method on a harder scene.

The planning method is intentionally unchanged from Example 66:

    left sphere -> fixed transfer plane -> right sphere

What changes here is only the environment. The middle plane patch now contains
blocked regions that disturb the obvious central corridor, so the same
parallel-evidence / delayed-commitment method must discover and exploit a
better side corridor.
"""

import argparse
import numpy as np
from ompl import util as ou

import multimodal_graph_search as base
from primitive_manifold_planner.scenes.stress_scene import build_stress_scene, print_route_summary, show_stress_route


def main():
    parser = argparse.ArgumentParser(description="Example 67: fixed-manifold stress test with the same method as Example 66.")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--variant", choices=["mild", "strong"], default="strong")
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--full", action="store_true", help="Use the uncapped Example 66-style exploration budget for this harder scene.")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--viz", dest="visualize", action="store_true", help="Show visualization after planning finishes.")
    parser.add_argument("--no-viz", dest="visualize", action="store_false", help="Skip visualization.")
    parser.set_defaults(visualize=False)
    args = parser.parse_args()

    np.random.seed(args.seed)
    ou.RNG.setSeed(args.seed)
    ou.setLogLevel(ou.LOG_ERROR)

    if args.max_rounds is not None:
        effective_max_rounds = max(4, int(args.max_rounds))
    elif args.full:
        effective_max_rounds = base.SAFETY_MAX_TOTAL_ROUNDS
    elif args.fast:
        effective_max_rounds = min(base.SAFETY_MAX_TOTAL_ROUNDS, 10)
    else:
        effective_max_rounds = min(base.SAFETY_MAX_TOTAL_ROUNDS, 12)

    planner_core = getattr(base, "planner_core", base)
    base.SAFETY_MAX_TOTAL_ROUNDS = int(effective_max_rounds)
    base.MIN_ROUNDS_BEFORE_SATURATION_CHECK = min(
        base.MIN_ROUNDS_BEFORE_SATURATION_CHECK,
        max(4, base.SAFETY_MAX_TOTAL_ROUNDS // 2),
    )
    base.MIN_POST_SOLUTION_ROUNDS = min(
        base.MIN_POST_SOLUTION_ROUNDS,
        max(2, base.SAFETY_MAX_TOTAL_ROUNDS // 4),
    )
    if args.fast:
        base.MIN_ROUNDS_BEFORE_SATURATION_CHECK = min(base.MIN_ROUNDS_BEFORE_SATURATION_CHECK, 6)
        base.MIN_POST_SOLUTION_ROUNDS = min(base.MIN_POST_SOLUTION_ROUNDS, 3)
    planner_core.SAFETY_MAX_TOTAL_ROUNDS = base.SAFETY_MAX_TOTAL_ROUNDS
    planner_core.MIN_ROUNDS_BEFORE_SATURATION_CHECK = base.MIN_ROUNDS_BEFORE_SATURATION_CHECK
    planner_core.MIN_POST_SOLUTION_ROUNDS = base.MIN_POST_SOLUTION_ROUNDS

    scene = build_stress_scene(args.variant)
    result = base.plan_fixed_manifold_multimodal_route(
        families=scene.families,
        start_q=scene.start_q,
        goal_q=scene.goal_q,
    )
    print_route_summary(result, scene.variant)

    if args.visualize:
        show_stress_route(scene, result)


if __name__ == "__main__":
    main()
