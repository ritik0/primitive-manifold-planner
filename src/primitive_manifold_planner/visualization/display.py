from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from primitive_manifold_planner.manifolds.geometric import PlaneManifold, SphereManifold
from primitive_manifold_planner.thesis.parallel_evidence_planner import (
    LEFT_STAGE,
    PLANE_STAGE,
    RIGHT_STAGE,
    STAGES,
    FixedPlaneRoute,
    UnknownSequenceRoute,
    _coerce_stage_manifolds,
    _plane_basis,
    is_plane_like,
    unwrap_manifold,
)
from primitive_manifold_planner.visualization import add_manifold, add_points, pyvista_available

from primitive_manifold_planner.examplesupport.intrinsic_multimodal_helpers import build_segment_polydata

try:
    import pyvista as pv
except Exception:
    pv = None


def plot_manifold(ax, manifold, color="lightgray", alpha=0.12):
    base = unwrap_manifold(manifold)
    if isinstance(base, SphereManifold):
        u = np.linspace(0.0, 2.0 * np.pi, 36)
        v = np.linspace(0.0, np.pi, 20)
        x = base.center[0] + base.radius * np.outer(np.cos(u), np.sin(v))
        y = base.center[1] + base.radius * np.outer(np.sin(u), np.sin(v))
        z = base.center[2] + base.radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(x, y, z, color=color, alpha=alpha, rstride=2, cstride=2, linewidth=0.55)


def plane_patch_corners(plane_like_manifold, half_u: float, half_v: float) -> np.ndarray:
    plane = unwrap_manifold(plane_like_manifold)
    center = np.asarray(plane.point, dtype=float)
    u, v = _plane_basis(np.asarray(plane.normal, dtype=float))
    return np.asarray(
        [
            center - half_u * u - half_v * v,
            center + half_u * u - half_v * v,
            center + half_u * u + half_v * v,
            center - half_u * u + half_v * v,
        ],
        dtype=float,
    )


def plot_plane_patch(ax, plane_like_manifold, half_u: float, half_v: float, color="lightgray", alpha=0.13):
    corners = plane_patch_corners(plane_like_manifold, half_u, half_v)
    xs = np.asarray([[corners[0, 0], corners[1, 0]], [corners[3, 0], corners[2, 0]]], dtype=float)
    ys = np.asarray([[corners[0, 1], corners[1, 1]], [corners[3, 1], corners[2, 1]]], dtype=float)
    zs = np.asarray([[corners[0, 2], corners[1, 2]], [corners[3, 2], corners[2, 2]]], dtype=float)
    ax.plot_surface(xs, ys, zs, color=color, alpha=alpha, linewidth=0.0, shade=False)


def draw_edge_segments(ax, edges: list[tuple[np.ndarray, np.ndarray]], color: str, alpha: float, linewidth: float):
    for q_a, q_b in edges:
        pts = np.asarray([q_a, q_b], dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, alpha=alpha, linewidth=linewidth)


def show_pyvista_route(
    families,
    result: FixedPlaneRoute,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    plane_half_u: float,
    plane_half_v: float,
) -> bool:
    if pv is None:
        return False

    colors = {
        LEFT_STAGE: "#81c784",
        PLANE_STAGE: "#64b5f6",
        RIGHT_STAGE: "#a5d6a7",
    }
    manifold_colors = {
        "left_support_3d": "#c58b4c",
        "transfer_plane_3d": "#7fa7c6",
        "right_support_3d": "#c58b4c",
    }

    plotter = pv.Plotter(window_size=(1320, 840))
    plotter.add_text("Example 66: fixed-manifold planning with parallel manifold evidence", font_size=12)

    actor_groups = {
        "Manifolds": [],
        "Obstacles": [],
        "Evidence": [],
        "Frontiers": [],
        "Hypotheses": [],
        "Committed": [],
        "StartGoal": [],
    }

    left_family, plane_family, right_family = families
    left_manifold = left_family.manifold(float(left_family.sample_lambdas()[0]))
    plane_manifold = plane_family.manifold(float(plane_family.sample_lambdas()[0]))
    right_manifold = right_family.manifold(float(right_family.sample_lambdas()[0]))

    for family, manifold in [
        (left_family, left_manifold),
        (right_family, right_manifold),
    ]:
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
        opacity=0.14,
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
            opacity=0.32,
            smooth_shading=True,
            label=getattr(obstacle, "name", "obstacle"),
        )
        if actor is not None:
            actor_groups["Obstacles"].append(actor)

    for stage in STAGES:
        stage_edges = result.stage_evidence_edges.get(stage, [])
        poly = build_segment_polydata(stage_edges)
        if poly is not None:
            actor = plotter.add_mesh(
                poly,
                color=colors[stage],
                line_width=2.2 if stage == PLANE_STAGE else 1.9,
                opacity=0.42,
                label=f"{stage} evidence edges",
            )
            if actor is not None:
                actor_groups["Evidence"].append(actor)

        stage_points = result.stage_evidence_points.get(stage, np.zeros((0, 3), dtype=float))
        if len(stage_points) > 0:
            actor = add_points(
                plotter,
                stage_points,
                color=colors[stage],
                size=5.0,
                label=f"{stage} evidence points",
            )
            if actor is not None:
                actor_groups["Evidence"].append(actor)

        frontier = result.stage_frontier_points.get(stage, np.zeros((0, 3), dtype=float))
        if len(frontier) > 0:
            actor = add_points(
                plotter,
                frontier,
                color="#fb8c00" if stage == LEFT_STAGE else ("#8e24aa" if stage == PLANE_STAGE else "#00897b"),
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

    for stage, color in [
        (LEFT_STAGE, "#1b5e20"),
        (PLANE_STAGE, "#0d47a1"),
        (RIGHT_STAGE, "#1b5e20"),
    ]:
        points = result.committed_stage_nodes.get(stage, np.zeros((0, 3), dtype=float))
        if len(points) > 0:
            actor = add_points(plotter, points, color=color, size=10.0, label=f"{stage} committed nodes")
            if actor is not None:
                actor_groups["Committed"].append(actor)

    if len(result.raw_path) >= 2:
        raw_polyline = pv.lines_from_points(np.asarray(result.raw_path, dtype=float))
        actor = plotter.add_mesh(raw_polyline, color="#90a4ae", line_width=3.0, opacity=0.8, label="certified route")
        if actor is not None:
            actor_groups["Committed"].append(actor)
    if len(result.path) >= 2:
        display_polyline = pv.lines_from_points(np.asarray(result.path, dtype=float))
        actor = plotter.add_mesh(display_polyline, color="#1565c0", line_width=7.0, label="display route")
        if actor is not None:
            actor_groups["Committed"].append(actor)

    actor = add_points(plotter, start_q, color="black", size=16.0, label="start")
    if actor is not None:
        actor_groups["StartGoal"].append(actor)
    actor = add_points(plotter, goal_q, color="gold", size=18.0, label="goal")
    if actor is not None:
        actor_groups["StartGoal"].append(actor)

    def set_visibility(name: str, visible: bool):
        for actor in actor_groups.get(name, []):
            actor.SetVisibility(bool(visible))
        plotter.render()

    for idx, label in enumerate(["Manifolds", "Obstacles", "Evidence", "Frontiers", "Hypotheses", "Committed", "StartGoal"]):
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
    return True


def show_route(families, result: FixedPlaneRoute, start_q: np.ndarray, goal_q: np.ndarray, plane_half_u: float, plane_half_v: float):
    colors = {
        LEFT_STAGE: "#81c784",
        PLANE_STAGE: "#64b5f6",
        RIGHT_STAGE: "#a5d6a7",
    }
    manifold_colors = {
        "left_support_3d": "#c58b4c",
        "transfer_plane_3d": "#7fa7c6",
        "right_support_3d": "#c58b4c",
    }

    if pyvista_available():
        shown = show_pyvista_route(
            families=families,
            result=result,
            start_q=start_q,
            goal_q=goal_q,
            plane_half_u=plane_half_u,
            plane_half_v=plane_half_v,
        )
        if shown:
            return

    fig = plt.figure(figsize=(11.5, 8.5))
    ax = fig.add_subplot(111, projection="3d")

    left_family, plane_family, right_family = families
    plot_manifold(ax, left_family.manifold(float(left_family.sample_lambdas()[0])), color=manifold_colors[left_family.name], alpha=0.10)
    plot_plane_patch(
        ax,
        plane_family.manifold(float(plane_family.sample_lambdas()[0])),
        half_u=plane_half_u,
        half_v=plane_half_v,
        color=manifold_colors[plane_family.name],
        alpha=0.13,
    )
    plot_manifold(ax, right_family.manifold(float(right_family.sample_lambdas()[0])), color=manifold_colors[right_family.name], alpha=0.10)

    for stage in STAGES:
        draw_edge_segments(ax, result.stage_evidence_edges.get(stage, []), color=colors[stage], alpha=0.33, linewidth=1.8 if stage == PLANE_STAGE else 1.5)

    for stage, color in [
        (LEFT_STAGE, "#fb8c00"),
        (PLANE_STAGE, "#8e24aa"),
        (RIGHT_STAGE, "#00897b"),
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
        ax.plot(
            result.raw_path[:, 0],
            result.raw_path[:, 1],
            result.raw_path[:, 2],
            color="#90a4ae",
            linewidth=2.0,
            alpha=0.85,
            label="certified route",
        )
    if len(result.path) >= 2:
        ax.plot(
            result.path[:, 0],
            result.path[:, 1],
            result.path[:, 2],
            color="#1565c0",
            linewidth=3.8,
            label="display route",
        )

    ax.scatter(start_q[0], start_q[1], start_q[2], s=95, marker="s", color="black", label="start")
    ax.scatter(goal_q[0], goal_q[1], goal_q[2], s=130, marker="*", color="gold", edgecolor="black", label="goal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Example 66: fixed-manifold planning with parallel manifold evidence")
    ax.legend(loc="upper right")
    ax.view_init(elev=23, azim=-56)
    plt.tight_layout()
    plt.show()


def _generic_stage_colors(stage_ids: list[str]) -> dict[str, str]:
    palette = ["#81c784", "#64b5f6", "#a5d6a7", "#ffb74d", "#ba68c8", "#4db6ac"]
    return {str(stage_id): palette[idx % len(palette)] for idx, stage_id in enumerate(stage_ids)}


def _generic_manifold_colors(stage_ids: list[str]) -> dict[str, str]:
    palette = ["#c58b4c", "#7fa7c6", "#c58b4c", "#9fa8da", "#80cbc4", "#bcaaa4"]
    return {str(stage_id): palette[idx % len(palette)] for idx, stage_id in enumerate(stage_ids)}


def show_pyvista_unknown_sequence_route(
    stage_manifolds,
    result: UnknownSequenceRoute,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    plane_half_u: float,
    plane_half_v: float,
) -> bool:
    if pv is None:
        return False

    stage_ids, manifold_map = _coerce_stage_manifolds(stage_manifolds)
    colors = _generic_stage_colors(stage_ids)
    manifold_colors = _generic_manifold_colors(stage_ids)

    plotter = pv.Plotter(window_size=(1320, 840))
    sequence_text = " -> ".join(result.discovered_sequence) if len(result.discovered_sequence) > 0 else "no committed sequence"
    plotter.add_text(f"Example 66: free-sequence multimodal planning\n{sequence_text}", font_size=12)

    actor_groups = {
        "Manifolds": [],
        "Evidence": [],
        "Frontiers": [],
        "Hypotheses": [],
        "Committed": [],
        "StartGoal": [],
    }

    for stage_id in stage_ids:
        manifold = manifold_map[stage_id]
        if is_plane_like(manifold):
            plane_corners = plane_patch_corners(manifold, half_u=plane_half_u, half_v=plane_half_v)
            plane_faces = np.hstack([[4, 0, 1, 2, 3]])
            plane_patch = pv.PolyData(plane_corners, faces=plane_faces)
            actor = plotter.add_mesh(
                plane_patch,
                color=manifold_colors[stage_id],
                opacity=0.14,
                show_edges=False,
                smooth_shading=False,
                name=stage_id,
            )
        else:
            actor = add_manifold(plotter, manifold, color=manifold_colors[stage_id], opacity=0.10)
        if actor is not None:
            actor_groups["Manifolds"].append(actor)

    for stage_id in stage_ids:
        stage_edges = result.stage_evidence_edges.get(stage_id, [])
        poly = build_segment_polydata(stage_edges)
        if poly is not None:
            actor = plotter.add_mesh(
                poly,
                color=colors[stage_id],
                line_width=2.0,
                opacity=0.42,
                label=f"{stage_id} evidence edges",
            )
            if actor is not None:
                actor_groups["Evidence"].append(actor)

        stage_points = result.stage_evidence_points.get(stage_id, np.zeros((0, 3), dtype=float))
        if len(stage_points) > 0:
            actor = add_points(plotter, stage_points, color=colors[stage_id], size=5.0, label=f"{stage_id} evidence points")
            if actor is not None:
                actor_groups["Evidence"].append(actor)

        frontier = result.stage_frontier_points.get(stage_id, np.zeros((0, 3), dtype=float))
        if len(frontier) > 0:
            actor = add_points(plotter, frontier, color="#fb8c00", size=8.0, label=f"{stage_id} frontier")
            if actor is not None:
                actor_groups["Frontiers"].append(actor)

        committed = result.committed_stage_nodes.get(stage_id, np.zeros((0, 3), dtype=float))
        if len(committed) > 0:
            actor = add_points(plotter, committed, color="#0d47a1", size=10.0, label=f"{stage_id} committed nodes")
            if actor is not None:
                actor_groups["Committed"].append(actor)

    if len(result.transition_points) > 0:
        actor = add_points(plotter, result.transition_points, color="#ef5350", size=10.0, label="transition hypotheses")
        if actor is not None:
            actor_groups["Hypotheses"].append(actor)

    if len(result.raw_path) >= 2:
        raw_polyline = pv.lines_from_points(np.asarray(result.raw_path, dtype=float))
        actor = plotter.add_mesh(raw_polyline, color="#90a4ae", line_width=3.0, opacity=0.8, label="certified route")
        if actor is not None:
            actor_groups["Committed"].append(actor)
    if len(result.path) >= 2:
        display_polyline = pv.lines_from_points(np.asarray(result.path, dtype=float))
        actor = plotter.add_mesh(display_polyline, color="#1565c0", line_width=7.0, label="display route")
        if actor is not None:
            actor_groups["Committed"].append(actor)

    actor = add_points(plotter, start_q, color="black", size=16.0, label="start")
    if actor is not None:
        actor_groups["StartGoal"].append(actor)
    actor = add_points(plotter, goal_q, color="gold", size=18.0, label="goal")
    if actor is not None:
        actor_groups["StartGoal"].append(actor)

    def set_visibility(name: str, visible: bool):
        for actor in actor_groups.get(name, []):
            actor.SetVisibility(bool(visible))
        plotter.render()

    for idx, label in enumerate(["Manifolds", "Evidence", "Frontiers", "Hypotheses", "Committed", "StartGoal"]):
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
    return True


def show_unknown_sequence_route(
    stage_manifolds,
    result: UnknownSequenceRoute,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    plane_half_u: float,
    plane_half_v: float,
):
    stage_ids, manifold_map = _coerce_stage_manifolds(stage_manifolds)
    colors = _generic_stage_colors(stage_ids)
    manifold_colors = _generic_manifold_colors(stage_ids)

    if pyvista_available():
        shown = show_pyvista_unknown_sequence_route(
            stage_manifolds=stage_manifolds,
            result=result,
            start_q=start_q,
            goal_q=goal_q,
            plane_half_u=plane_half_u,
            plane_half_v=plane_half_v,
        )
        if shown:
            return

    fig = plt.figure(figsize=(11.5, 8.5))
    ax = fig.add_subplot(111, projection="3d")

    for stage_id in stage_ids:
        manifold = manifold_map[stage_id]
        if is_plane_like(manifold):
            plot_plane_patch(ax, manifold, half_u=plane_half_u, half_v=plane_half_v, color=manifold_colors[stage_id], alpha=0.13)
        else:
            plot_manifold(ax, manifold, color=manifold_colors[stage_id], alpha=0.10)

    for stage_id in stage_ids:
        draw_edge_segments(ax, result.stage_evidence_edges.get(stage_id, []), color=colors[stage_id], alpha=0.33, linewidth=1.6)
        frontier = result.stage_frontier_points.get(stage_id, np.zeros((0, 3), dtype=float))
        if len(frontier) > 0:
            ax.scatter(frontier[:, 0], frontier[:, 1], frontier[:, 2], s=20, color="#fb8c00", alpha=0.8, label=f"{stage_id} frontier")

    if len(result.transition_points) > 0:
        ax.scatter(
            result.transition_points[:, 0],
            result.transition_points[:, 1],
            result.transition_points[:, 2],
            s=34,
            color="#ef5350",
            label="transition hypotheses",
        )

    if len(result.raw_path) >= 2:
        ax.plot(result.raw_path[:, 0], result.raw_path[:, 1], result.raw_path[:, 2], color="#90a4ae", linewidth=2.0, alpha=0.85, label="certified route")
    if len(result.path) >= 2:
        ax.plot(result.path[:, 0], result.path[:, 1], result.path[:, 2], color="#1565c0", linewidth=3.8, label="display route")

    ax.scatter(start_q[0], start_q[1], start_q[2], s=95, marker="s", color="black", label="start")
    ax.scatter(goal_q[0], goal_q[1], goal_q[2], s=130, marker="*", color="gold", edgecolor="black", label="goal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Example 66: free-sequence planning with parallel manifold evidence")
    ax.legend(loc="upper right")
    ax.view_init(elev=23, azim=-56)
    plt.tight_layout()
    plt.show()


__all__ = [
    "draw_edge_segments",
    "plane_patch_corners",
    "plot_manifold",
    "plot_plane_patch",
    "show_pyvista_route",
    "show_pyvista_unknown_sequence_route",
    "show_route",
    "show_unknown_sequence_route",
]
