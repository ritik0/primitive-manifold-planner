from __future__ import annotations

"""Example 67: stress-test the Example 66 fixed-manifold method on a harder scene.

The planning method is intentionally unchanged from Example 66:

    left sphere -> fixed transfer plane -> right sphere

What changes here is only the environment. The middle plane patch now contains
blocked regions that disturb the obvious central corridor, so the same
parallel-evidence / delayed-commitment method must discover and exploit a
better side corridor.
"""

from dataclasses import dataclass
import argparse

import matplotlib.pyplot as plt
import numpy as np
from ompl import util as ou

import example_66_multimodal_graph_search as base

from primitive_manifold_planner.families.standard import MaskedFamily, PlaneFamily, SphereFamily
from primitive_manifold_planner.visualization import add_manifold, add_points, pyvista_available


@dataclass(frozen=True)
class BlockedRectangle:
    u_min: float
    u_max: float
    v_min: float
    v_max: float


@dataclass(frozen=True)
class StressScene:
    families: list[object]
    start_q: np.ndarray
    goal_q: np.ndarray
    plane_half_u: float
    plane_half_v: float
    blocked_rectangles: list[BlockedRectangle]
    variant: str


def rectangle_corners_on_plane(
    plane_like_manifold,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
) -> np.ndarray:
    plane = base.unwrap_manifold(plane_like_manifold)
    center = np.asarray(plane.point, dtype=float)
    basis_u, basis_v = base._plane_basis(np.asarray(plane.normal, dtype=float))
    return np.asarray(
        [
            center + float(u_min) * basis_u + float(v_min) * basis_v,
            center + float(u_max) * basis_u + float(v_min) * basis_v,
            center + float(u_max) * basis_u + float(v_max) * basis_v,
            center + float(u_min) * basis_u + float(v_max) * basis_v,
        ],
        dtype=float,
    )


def build_stress_scene(variant: str = "strong") -> StressScene:
    variant_name = str(variant).strip().lower()
    if variant_name not in {"mild", "strong"}:
        raise ValueError(f"Unknown Example 67 variant: {variant!r}")

    left_support = SphereFamily(
        name="left_support_3d",
        center=np.array([-2.15, -0.70, 0.48], dtype=float),
        radii={1.05: 1.05},
    )
    base_plane = PlaneFamily(
        name="transfer_plane_3d",
        base_point=np.array([0.08, 0.0, 0.0], dtype=float),
        normal=np.array([0.0, 0.0, 1.0], dtype=float),
        offsets=[0.0],
        anchor_span=1.18,
    )
    right_support = SphereFamily(
        name="right_support_3d",
        center=np.array([2.25, 0.82, 0.48], dtype=float),
        radii={1.05: 1.05},
    )

    plane_half_u = 1.02
    plane_half_v = 2.25
    basis_u = np.asarray(base_plane._basis_u, dtype=float)
    basis_v = np.asarray(base_plane._basis_v, dtype=float)
    base_point = np.asarray(base_plane.base_point, dtype=float)

    if variant_name == "mild":
        blocked_rectangles = [
            BlockedRectangle(u_min=-0.30, u_max=0.30, v_min=-0.92, v_max=0.92),
        ]
    else:
        blocked_rectangles = [
            BlockedRectangle(u_min=-0.36, u_max=0.36, v_min=-1.18, v_max=1.18),
            BlockedRectangle(u_min=-0.72, u_max=-0.20, v_min=-0.30, v_max=0.30),
        ]

    def stress_mask(_lam: float, q: np.ndarray) -> bool:
        qq = np.asarray(q, dtype=float)
        rel = qq - base_point
        u_coord = float(np.dot(rel, basis_u))
        v_coord = float(np.dot(rel, basis_v))
        inside_outer_patch = abs(u_coord) <= plane_half_u and abs(v_coord) <= plane_half_v
        if not inside_outer_patch:
            return False
        for rect in blocked_rectangles:
            if rect.u_min <= u_coord <= rect.u_max and rect.v_min <= v_coord <= rect.v_max:
                return False
        return True

    transfer_plane = MaskedFamily(
        base_family=base_plane,
        validity_mask_fn=stress_mask,
        name="transfer_plane_3d",
    )

    if variant_name == "mild":
        start_q = base.sphere_point(left_support.center, 1.05, azimuth_deg=12.0, elevation_deg=-78.0)
        goal_q = base.sphere_point(right_support.center, 1.05, azimuth_deg=-10.0, elevation_deg=74.0)
    else:
        start_q = base.sphere_point(left_support.center, 1.05, azimuth_deg=22.0, elevation_deg=-76.0)
        goal_q = base.sphere_point(right_support.center, 1.05, azimuth_deg=-18.0, elevation_deg=71.0)

    return StressScene(
        families=[left_support, transfer_plane, right_support],
        start_q=np.asarray(start_q, dtype=float),
        goal_q=np.asarray(goal_q, dtype=float),
        plane_half_u=float(plane_half_u),
        plane_half_v=float(plane_half_v),
        blocked_rectangles=list(blocked_rectangles),
        variant=variant_name,
    )


def show_stress_route(scene: StressScene, result: base.FixedPlaneRoute) -> None:
    colors = {
        base.LEFT_STAGE: "#81c784",
        base.PLANE_STAGE: "#64b5f6",
        base.RIGHT_STAGE: "#a5d6a7",
    }
    manifold_colors = {
        "left_support_3d": "#c58b4c",
        "transfer_plane_3d": "#7fa7c6",
        "right_support_3d": "#c58b4c",
    }

    families = scene.families
    left_family, plane_family, right_family = families
    left_manifold = left_family.manifold(float(left_family.sample_lambdas()[0]))
    plane_manifold = plane_family.manifold(float(plane_family.sample_lambdas()[0]))
    right_manifold = right_family.manifold(float(right_family.sample_lambdas()[0]))

    if pyvista_available() and base.pv is not None:
        plotter = base.pv.Plotter(window_size=(1320, 840))
        plotter.add_text(
            f"Example 67: fixed-manifold stress test ({scene.variant}) with parallel manifold evidence",
            font_size=12,
        )

        actor_groups = {
            "Manifolds": [],
            "Blocked": [],
            "Evidence": [],
            "Frontiers": [],
            "Hypotheses": [],
            "Committed": [],
            "Charts": [],
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

        plane_corners = base.plane_patch_corners(plane_manifold, half_u=scene.plane_half_u, half_v=scene.plane_half_v)
        plane_faces = np.hstack([[4, 0, 1, 2, 3]])
        plane_patch = base.pv.PolyData(plane_corners, faces=plane_faces)
        plane_actor = plotter.add_mesh(
            plane_patch,
            color=manifold_colors.get(plane_family.name, "#999999"),
            opacity=0.14,
            show_edges=False,
            smooth_shading=False,
            name=plane_family.name,
        )
        if plane_actor is not None:
            actor_groups["Manifolds"].append(plane_actor)

        for rect in scene.blocked_rectangles:
            blocked_corners = rectangle_corners_on_plane(
                plane_manifold,
                rect.u_min,
                rect.u_max,
                rect.v_min,
                rect.v_max,
            )
            blocked_patch = base.pv.PolyData(blocked_corners, faces=plane_faces)
            actor = plotter.add_mesh(
                blocked_patch,
                color="#ef5350",
                opacity=0.30,
                show_edges=True,
                edge_color="#c62828",
                line_width=1.0,
            )
            if actor is not None:
                actor_groups["Blocked"].append(actor)

        for stage in base.STAGES:
            stage_edges = result.stage_evidence_edges.get(stage, [])
            poly = base.build_segment_polydata(stage_edges)
            if poly is not None:
                actor = plotter.add_mesh(
                    poly,
                    color=colors[stage],
                    line_width=2.2 if stage == base.PLANE_STAGE else 1.9,
                    opacity=0.42,
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
                    color="#fb8c00" if stage == base.LEFT_STAGE else ("#8e24aa" if stage == base.PLANE_STAGE else "#00897b"),
                    size=8.0,
                    label=f"{stage} frontier",
                )
                if actor is not None:
                    actor_groups["Frontiers"].append(actor)

            charts = result.stage_chart_centers.get(stage, np.zeros((0, 3), dtype=float))
            if len(charts) > 0:
                actor = add_points(
                    plotter,
                    charts,
                    color="#ef6c00" if stage == base.LEFT_STAGE else ("#6a1b9a" if stage == base.PLANE_STAGE else "#00695c"),
                    size=9.0,
                    label=f"{stage} chart centers",
                )
                if actor is not None:
                    actor_groups["Charts"].append(actor)

        if len(result.left_plane_hypothesis_points) > 0:
            actor = add_points(plotter, result.left_plane_hypothesis_points, color="#ff7043", size=10.0, label="left-plane hypotheses")
            if actor is not None:
                actor_groups["Hypotheses"].append(actor)
        if len(result.plane_right_hypothesis_points) > 0:
            actor = add_points(plotter, result.plane_right_hypothesis_points, color="#26a69a", size=10.0, label="plane-right hypotheses")
            if actor is not None:
                actor_groups["Hypotheses"].append(actor)

        for stage, color in [
            (base.LEFT_STAGE, "#1b5e20"),
            (base.PLANE_STAGE, "#0d47a1"),
            (base.RIGHT_STAGE, "#1b5e20"),
        ]:
            points = result.committed_stage_nodes.get(stage, np.zeros((0, 3), dtype=float))
            if len(points) > 0:
                actor = add_points(plotter, points, color=color, size=10.0, label=f"{stage} committed nodes")
                if actor is not None:
                    actor_groups["Committed"].append(actor)

        if len(result.raw_path) >= 2:
            raw_polyline = base.pv.lines_from_points(np.asarray(result.raw_path, dtype=float))
            actor = plotter.add_mesh(raw_polyline, color="#90a4ae", line_width=3.0, opacity=0.8, label="certified route")
            if actor is not None:
                actor_groups["Committed"].append(actor)
        if len(result.path) >= 2:
            display_polyline = base.pv.lines_from_points(np.asarray(result.path, dtype=float))
            actor = plotter.add_mesh(display_polyline, color="#1565c0", line_width=7.0, label="display route")
            if actor is not None:
                actor_groups["Committed"].append(actor)

        actor = add_points(plotter, scene.start_q, color="black", size=16.0, label="start")
        if actor is not None:
            actor_groups["StartGoal"].append(actor)
        actor = add_points(plotter, scene.goal_q, color="gold", size=18.0, label="goal")
        if actor is not None:
            actor_groups["StartGoal"].append(actor)

        def set_visibility(name: str, visible: bool):
            for actor in actor_groups.get(name, []):
                actor.SetVisibility(bool(visible))
            plotter.render()

        for idx, label in enumerate(["Manifolds", "Blocked", "Evidence", "Frontiers", "Hypotheses", "Committed", "Charts", "StartGoal"]):
            y = 10 + idx * 42
            plotter.add_text(label, position=(55, y + 7), font_size=10, color="black")
            plotter.add_checkbox_button_widget(
                callback=lambda state, name=label: set_visibility(name, state),
                value=True,
                position=(10, y),
                size=28,
                color_on="lightgray",
                color_off="white",
                background_color="gray",
            )

        plotter.add_axes()
        plotter.show_grid()
        plotter.show()
        return

    fig = plt.figure(figsize=(11.5, 8.5))
    ax = fig.add_subplot(111, projection="3d")

    base.plot_manifold(ax, left_manifold, color=manifold_colors[left_family.name], alpha=0.10)
    base.plot_plane_patch(
        ax,
        plane_manifold,
        half_u=scene.plane_half_u,
        half_v=scene.plane_half_v,
        color=manifold_colors[plane_family.name],
        alpha=0.13,
    )
    base.plot_manifold(ax, right_manifold, color=manifold_colors[right_family.name], alpha=0.10)

    for rect in scene.blocked_rectangles:
        corners = rectangle_corners_on_plane(
            plane_manifold,
            rect.u_min,
            rect.u_max,
            rect.v_min,
            rect.v_max,
        )
        xs = np.asarray([[corners[0, 0], corners[1, 0]], [corners[3, 0], corners[2, 0]]], dtype=float)
        ys = np.asarray([[corners[0, 1], corners[1, 1]], [corners[3, 1], corners[2, 1]]], dtype=float)
        zs = np.asarray([[corners[0, 2], corners[1, 2]], [corners[3, 2], corners[2, 2]]], dtype=float)
        ax.plot_surface(xs, ys, zs, color="#ef5350", alpha=0.28, linewidth=0.0, shade=False)

    for stage in base.STAGES:
        base.draw_edge_segments(ax, result.stage_evidence_edges.get(stage, []), color=colors[stage], alpha=0.33, linewidth=1.8 if stage == base.PLANE_STAGE else 1.5)

    for stage, color in [
        (base.LEFT_STAGE, "#fb8c00"),
        (base.PLANE_STAGE, "#8e24aa"),
        (base.RIGHT_STAGE, "#00897b"),
    ]:
        frontier = result.stage_frontier_points.get(stage, np.zeros((0, 3), dtype=float))
        if len(frontier) > 0:
            ax.scatter(frontier[:, 0], frontier[:, 1], frontier[:, 2], s=20, color=color, alpha=0.8, label=f"{stage} frontier")

    if len(result.left_plane_hypothesis_points) > 0:
        ax.scatter(
            result.left_plane_hypothesis_points[:, 0],
            result.left_plane_hypothesis_points[:, 1],
            result.left_plane_hypothesis_points[:, 2],
            s=34,
            color="#ff7043",
            label="left-plane hypotheses",
        )
    if len(result.plane_right_hypothesis_points) > 0:
        ax.scatter(
            result.plane_right_hypothesis_points[:, 0],
            result.plane_right_hypothesis_points[:, 1],
            result.plane_right_hypothesis_points[:, 2],
            s=34,
            color="#26a69a",
            label="plane-right hypotheses",
        )

    if len(result.raw_path) >= 2:
        ax.plot(result.raw_path[:, 0], result.raw_path[:, 1], result.raw_path[:, 2], color="#90a4ae", linewidth=2.0, alpha=0.85, label="certified route")
    if len(result.path) >= 2:
        ax.plot(result.path[:, 0], result.path[:, 1], result.path[:, 2], color="#1565c0", linewidth=3.8, label="display route")

    ax.scatter(scene.start_q[0], scene.start_q[1], scene.start_q[2], s=95, marker="s", color="black", label="start")
    ax.scatter(scene.goal_q[0], scene.goal_q[1], scene.goal_q[2], s=130, marker="*", color="gold", edgecolor="black", label="goal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Example 67: fixed-manifold stress test ({scene.variant}) with parallel manifold evidence")
    ax.legend(loc="upper right")
    ax.view_init(elev=23, azim=-56)
    plt.tight_layout()
    plt.show()


def print_route_summary(result: base.FixedPlaneRoute, variant: str) -> None:
    print("\nExample 67")
    print(f"Fixed-manifold stress test ({variant}) with parallel manifold evidence accumulation")
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
    print(f"left_frontier_count = {result.stage_frontier_counts.get(base.LEFT_STAGE, 0)}")
    print(f"plane_frontier_count = {result.stage_frontier_counts.get(base.PLANE_STAGE, 0)}")
    print(f"right_frontier_count = {result.stage_frontier_counts.get(base.RIGHT_STAGE, 0)}")
    print(f"recent_graph_node_gain = {result.recent_graph_node_gain}")
    print(f"recent_transition_gain = {result.recent_transition_gain}")
    print(f"recent_route_improvement_gain = {result.recent_route_improvement_gain}")
    print(f"left_stagnating = {result.stage_stagnation_flags.get(base.LEFT_STAGE, False)}")
    print(f"plane_stagnating = {result.stage_stagnation_flags.get(base.PLANE_STAGE, False)}")
    print(f"right_stagnating = {result.stage_stagnation_flags.get(base.RIGHT_STAGE, False)}")
    for mode_name in sorted(result.mode_counts):
        print(f"mode_{mode_name} = {result.mode_counts[mode_name]}")


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
