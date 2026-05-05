from __future__ import annotations

"""Example 66: fixed-manifold multimodal planning with parallel evidence accumulation."""

import argparse
from pathlib import Path
import sys

import numpy as np
from ompl import util as ou

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from primitive_manifold_planner.families.standard import MaskedFamily, PlaneFamily, SphereFamily

from intrinsic_multimodal_scene import build_segment_polydata
from planner import parallel_evidence_planner as planner_core
from planner.parallel_evidence_planner import *  # noqa: F401,F403
from visualization.display import (
    draw_edge_segments,
    plane_patch_corners,
    plot_manifold,
    plot_plane_patch,
    show_pyvista_route,
    show_pyvista_unknown_sequence_route,
    show_route,
    show_unknown_sequence_route,
)


def build_scene():
    left_support = SphereFamily(
        name="left_support_3d",
        center=np.array([-2.15, -0.55, 0.48], dtype=float),
        radii={1.05: 1.05},
    )
    base_plane = PlaneFamily(
        name="transfer_plane_3d",
        base_point=np.array([0.0, 0.0, 0.0], dtype=float),
        normal=np.array([0.0, 0.0, 1.0], dtype=float),
        offsets=[0.0],
        anchor_span=1.15,
    )
    plane_half_u = 0.95
    plane_half_v = 2.15
    basis_u = np.asarray(base_plane._basis_u, dtype=float)
    basis_v = np.asarray(base_plane._basis_v, dtype=float)
    base_point = np.asarray(base_plane.base_point, dtype=float)

    def rectangle_mask(_lam: float, q: np.ndarray) -> bool:
        qq = np.asarray(q, dtype=float)
        rel = qq - base_point
        u_coord = float(np.dot(rel, basis_u))
        v_coord = float(np.dot(rel, basis_v))
        return abs(u_coord) <= plane_half_u and abs(v_coord) <= plane_half_v

    transfer_plane = MaskedFamily(
        base_family=base_plane,
        validity_mask_fn=rectangle_mask,
        name="transfer_plane_3d",
    )
    right_support = SphereFamily(
        name="right_support_3d",
        center=np.array([2.15, 0.55, 0.48], dtype=float),
        radii={1.05: 1.05},
    )

    start_q = sphere_point(left_support.center, 1.05, azimuth_deg=0.0, elevation_deg=-90.0)
    goal_q = sphere_point(right_support.center, 1.05, azimuth_deg=0.0, elevation_deg=90.0)
    return [left_support, transfer_plane, right_support], start_q, goal_q, plane_half_u, plane_half_v


def main():
    parser = argparse.ArgumentParser(description="Example 66: fixed-manifold planning with parallel manifold evidence.")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--unknown-sequence", action="store_true")
    parser.add_argument("--compare-sequences", action="store_true")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    ou.RNG.setSeed(args.seed)
    ou.setLogLevel(ou.LOG_ERROR)

    if args.max_rounds is not None:
        planner_core.SAFETY_MAX_TOTAL_ROUNDS = max(4, int(args.max_rounds))
        planner_core.MIN_ROUNDS_BEFORE_SATURATION_CHECK = min(
            planner_core.MIN_ROUNDS_BEFORE_SATURATION_CHECK,
            max(4, planner_core.SAFETY_MAX_TOTAL_ROUNDS // 2),
        )
        planner_core.MIN_POST_SOLUTION_ROUNDS = min(
            planner_core.MIN_POST_SOLUTION_ROUNDS,
            max(2, planner_core.SAFETY_MAX_TOTAL_ROUNDS // 4),
        )
    if args.fast:
        planner_core.SAFETY_MAX_TOTAL_ROUNDS = min(planner_core.SAFETY_MAX_TOTAL_ROUNDS, 10)
        planner_core.MIN_ROUNDS_BEFORE_SATURATION_CHECK = min(planner_core.MIN_ROUNDS_BEFORE_SATURATION_CHECK, 6)
        planner_core.MIN_POST_SOLUTION_ROUNDS = min(planner_core.MIN_POST_SOLUTION_ROUNDS, 3)

    if args.compare_sequences:
        comparison = compare_fixed_and_unknown_sequence_demo(serial_mode=args.serial)
        print("\nExample 66 Comparison")
        print("Fixed-sequence versus free-sequence multimodal planning")
        for key in [
            "fixed_success",
            "fixed_message",
            "fixed_route_cost",
            "fixed_total_rounds",
            "unknown_success",
            "unknown_message",
            "unknown_sequence",
            "unknown_route_cost",
            "unknown_total_rounds",
            "unknown_meta_graph_edges",
        ]:
            print(f"{key} = {comparison[key]}")
        return

    if args.unknown_sequence:
        stage_manifolds, start_q, goal_q, plane_half_u, plane_half_v = build_unknown_sequence_scene()
        result = plan_multimodal_unknown_sequence(
            stage_manifolds=stage_manifolds,
            start_q=start_q,
            goal_q=goal_q,
            serial_mode=args.serial,
        )
        print("\nExample 66")
        print("Free-sequence multimodal planning with parallel manifold evidence accumulation")
        print(f"success = {result.success}")
        print(f"message = {result.message}")
        print(f"discovered_sequence = {result.discovered_sequence}")
        print(f"start_stages = {result.start_stages}")
        print(f"goal_stages = {result.goal_stages}")
        print(f"total_rounds = {result.total_rounds}")
        print(f"candidate_evaluations = {result.candidate_evaluations}")
        print(f"shared_proposals_processed = {result.shared_proposals_processed}")
        print(f"global_transition_hypotheses = {result.global_transition_hypotheses}")
        print(f"meta_graph_edges = {result.meta_graph_edges}")
        print(f"route_cost_raw = {result.route_cost_raw:.4f}")
        print(f"route_cost_display = {result.route_cost_display:.4f}")
        print(f"display_path_points = {len(result.path)}")
        print(f"raw_path_points = {len(result.raw_path)}")
        print(f"saturated_before_solution = {result.saturated_before_solution}")
        print(f"stagnation_stage = {result.stagnation_stage}")
        for stage_id in sorted(result.stage_node_counts):
            print(f"{stage_id}_evidence_nodes = {result.stage_node_counts[stage_id]}")
            print(f"{stage_id}_frontier_count = {result.stage_frontier_counts.get(stage_id, 0)}")
        for mode_name in sorted(result.mode_counts):
            print(f"mode_{mode_name} = {result.mode_counts[mode_name]}")

        if not args.no_viz:
            show_unknown_sequence_route(
                stage_manifolds=stage_manifolds,
                result=result,
                start_q=start_q,
                goal_q=goal_q,
                plane_half_u=plane_half_u,
                plane_half_v=plane_half_v,
            )
        return

    families, start_q, goal_q, plane_half_u, plane_half_v = build_scene()
    result = plan_fixed_manifold_multimodal_route(
        families=families,
        start_q=start_q,
        goal_q=goal_q,
        serial_mode=args.serial,
    )

    print("\nExample 66")
    print("Fixed-manifold planning with parallel manifold evidence accumulation")
    print(f"success = {result.success}")
    print(f"message = {result.message}")
    print(f"total_rounds = {result.total_rounds}")
    print(f"candidate_evaluations = {result.candidate_evaluations}")
    print(f"left_evidence_nodes = {result.left_evidence_nodes}")
    print(f"plane_evidence_nodes = {result.plane_evidence_nodes}")
    print(f"right_evidence_nodes = {result.right_evidence_nodes}")
    print(f"committed_nodes = {result.committed_nodes}")
    print(f"evidence_only_nodes = {result.evidence_only_nodes}")
    print(f"shared_proposals_processed = {result.shared_proposals_processed}")
    print(f"proposals_used_by_multiple_stages = {result.proposals_used_by_multiple_stages}")
    print(f"multi_stage_updates_per_round = {result.multi_stage_updates_per_round:.3f}")
    print(f"average_useful_stages_per_proposal = {result.average_useful_stages_per_proposal:.3f}")
    print(f"proposal_rounds_with_plane_updates = {result.proposal_rounds_with_plane_updates}")
    print(f"proposal_rounds_with_multi_stage_updates = {result.proposal_rounds_with_multi_stage_updates}")
    print(f"plane_evidence_before_first_committed_entry = {result.plane_evidence_before_first_committed_entry}")
    print(f"right_evidence_before_first_committed_exit = {result.right_evidence_before_first_committed_exit}")
    print(f"plane_evidence_growth_after_first_solution = {result.plane_evidence_growth_after_first_solution}")
    print(f"right_evidence_growth_after_first_solution = {result.right_evidence_growth_after_first_solution}")
    print(f"transition_hypotheses_left_plane = {result.transition_hypotheses_left_plane}")
    print(f"transition_hypotheses_plane_right = {result.transition_hypotheses_plane_right}")
    print(f"alternative_hypothesis_pairs_evaluated = {result.alternative_hypothesis_pairs_evaluated}")
    print(f"first_solution_round = {result.first_solution_round}")
    print(f"best_solution_round = {result.best_solution_round}")
    print(f"continued_after_first_solution = {result.continued_after_first_solution}")
    print(f"committed_route_changes_after_first_solution = {result.committed_route_changes_after_first_solution}")
    print(f"certified_path_points = {result.certified_path_points}")
    print(f"display_path_points = {result.display_path_points}")
    print(f"route_cost_raw = {result.route_cost_raw:.4f}")
    print(f"route_cost_display = {result.route_cost_display:.4f}")
    print(f"graph_route_edges = {result.graph_route_edges}")
    print(f"left_frontier_count = {result.stage_frontier_counts.get(LEFT_STAGE, 0)}")
    print(f"plane_frontier_count = {result.stage_frontier_counts.get(PLANE_STAGE, 0)}")
    print(f"right_frontier_count = {result.stage_frontier_counts.get(RIGHT_STAGE, 0)}")
    print(f"recent_graph_node_gain = {result.recent_graph_node_gain}")
    print(f"recent_transition_gain = {result.recent_transition_gain}")
    print(f"recent_route_improvement_gain = {result.recent_route_improvement_gain}")
    print(f"left_stagnating = {result.stage_stagnation_flags.get(LEFT_STAGE, False)}")
    print(f"plane_stagnating = {result.stage_stagnation_flags.get(PLANE_STAGE, False)}")
    print(f"right_stagnating = {result.stage_stagnation_flags.get(RIGHT_STAGE, False)}")
    for mode_name in sorted(result.mode_counts):
        print(f"mode_{mode_name} = {result.mode_counts[mode_name]}")

    if not args.no_viz:
        show_route(
            families=families,
            result=result,
            start_q=start_q,
            goal_q=goal_q,
            plane_half_u=plane_half_u,
            plane_half_v=plane_half_v,
        )


if __name__ == "__main__":
    main()
