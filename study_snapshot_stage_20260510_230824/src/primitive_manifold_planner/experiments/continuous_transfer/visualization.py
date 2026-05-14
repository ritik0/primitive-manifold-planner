"""PyVista visualization helpers for the staged continuous-transfer experiment.

The route overlay intentionally emphasizes the parallel-evidence outcome: when
multiple certified fixed-lambda routes are available, the best route is shown
prominently and the next-best certified alternatives are layered underneath in
distinct colors for direct visual comparison.
"""

from __future__ import annotations

import numpy as np

from primitive_manifold_planner.visualization import add_manifold, add_points, pyvista_available

from .family_definition import build_continuous_scene, plane_leaf_patch
from .graph_types import ContinuousTransferRoute
from .support import build_segment_polydata

try:
    import pyvista as pv
except Exception:
    pv = None


def show_continuous_route(result: ContinuousTransferRoute) -> bool:
    """Render the continuous-transfer scene and any certified top-k routes."""

    if not pyvista_available() or pv is None:
        print("PyVista is not available in this environment.")
        return False

    left_family, transfer_family, right_family, start_q, goal_q = build_continuous_scene(
        obstacle_profile=result.scene_profile
    )
    left_manifold = left_family.manifold(float(left_family.sample_lambdas()[0]))
    right_manifold = right_family.manifold(float(right_family.sample_lambdas()[0]))
    plotter = pv.Plotter(window_size=(1380, 860))
    plotter.add_text("Example 65: parallel family evidence under fixed left -> family -> right sequence", font_size=12)
    available_top_route_count = sum(bool(route.strict_valid) for route in result.top_k_routes)
    plotter.add_text(
        f"Top-k requested: {len(result.top_k_routes)} | certified alternatives shown: {available_top_route_count}",
        position="upper_right",
        font_size=10,
    )
    actor_groups = {
        "Supports": [],
        "FamilyLeaves": [],
        "LeftStage": [],
        "FamilyLeaf": [],
        "RightStage": [],
        "Seeds": [],
        "Path": [],
        "TopKPaths": [],
        "StartGoal": [],
    }

    actor = add_manifold(plotter, left_manifold, color="#c58b4c", opacity=0.11, label="left support sphere")
    if actor is not None:
        actor_groups["Supports"].append(actor)
    actor = add_manifold(plotter, right_manifold, color="#c58b4c", opacity=0.11, label="right support sphere")
    if actor is not None:
        actor_groups["Supports"].append(actor)

    leaf_lambdas = np.asarray(result.explored_lambda_values, dtype=float)
    if len(leaf_lambdas) > 12:
        ids = np.linspace(0, len(leaf_lambdas) - 1, 12, dtype=int)
        leaf_lambdas = leaf_lambdas[ids]
    for idx, lam in enumerate(leaf_lambdas):
        corners = plane_leaf_patch(transfer_family, lam)
        faces = np.hstack([[4, 0, 1, 2, 3]])
        patch = pv.PolyData(corners, faces)
        actor = plotter.add_mesh(
            patch,
            color="#90caf9",
            opacity=0.08,
            show_edges=False,
            smooth_shading=False,
            label="explored family leaves" if idx == 0 else None,
        )
        if actor is not None:
            actor_groups["FamilyLeaves"].append(actor)
    if result.primary_entry_lambda is not None:
        primary_corners = plane_leaf_patch(transfer_family, float(result.primary_entry_lambda))
        primary_faces = np.hstack([[4, 0, 1, 2, 3]])
        primary_patch = pv.PolyData(primary_corners, primary_faces)
        actor = plotter.add_mesh(
            primary_patch,
            color="#29b6f6",
            opacity=0.16,
            show_edges=True,
            line_width=1.2,
            label="primary entry leaf",
        )
        if actor is not None:
            actor_groups["FamilyLeaves"].append(actor)
    colors_by_mode = {
        "left": "#81c784",
        "family_leaf": "#4fc3f7",
        "right": "#81c784",
    }
    for mode, edges in result.explored_edges_by_mode.items():
        graph = build_segment_polydata(edges)
        if graph is None:
            continue
        actor = plotter.add_mesh(graph, color=colors_by_mode.get(mode, "#90a4ae"), line_width=2.2, opacity=0.40)
        if actor is not None:
            if mode == "left":
                actor_groups["LeftStage"].append(actor)
            elif mode == "family_leaf":
                actor_groups["FamilyLeaf"].append(actor)
            elif mode == "right":
                actor_groups["RightStage"].append(actor)

    route_palette = ["#d32f2f", "#1976d2", "#388e3c", "#f57c00", "#7b1fa2"]
    route_widths = [7.0, 5.0, 5.0, 4.0, 4.0]
    route_opacities = [1.0, 0.82, 0.82, 0.72, 0.72]

    routes_to_draw = [route for route in result.top_k_routes[:5] if bool(route.strict_valid)]

    fallback_best_display = np.asarray(result.display_path, dtype=float)
    for route_idx in range(min(5, max(len(routes_to_draw), 1))):
        if len(routes_to_draw) > 0:
            route = routes_to_draw[route_idx]
            route_display = np.asarray(route.display_path, dtype=float)
        else:
            route_display = np.asarray(fallback_best_display, dtype=float)
        if len(route_display) < 2:
            continue
        actor = plotter.add_mesh(
            pv.lines_from_points(route_display),
            color=route_palette[min(route_idx, len(route_palette) - 1)],
            line_width=route_widths[min(route_idx, len(route_widths) - 1)],
            opacity=route_opacities[min(route_idx, len(route_opacities) - 1)],
            label="best certified route" if route_idx == 0 else f"certified alternative {route_idx + 1}",
        )
        if actor is not None:
            actor_groups["Path" if route_idx == 0 else "TopKPaths"].append(actor)

    legend_routes = routes_to_draw if len(routes_to_draw) > 0 else []
    if len(legend_routes) > 0:
        legend_x = 915
        legend_y = 86
        plotter.add_text("Certified leaf sequences", position=(legend_x, legend_y), font_size=11, color="black")
        for idx, route in enumerate(legend_routes[:5]):
            plotter.add_text(
                f"{idx + 1}. {' -> '.join(route.leaf_sequence)}",
                position=(legend_x, legend_y + 24 + idx * 22),
                font_size=10,
                color=route_palette[min(idx, len(route_palette) - 1)],
            )
    elif len(fallback_best_display) >= 2:
        plotter.add_text(
            "Certified leaf sequences\n1. best certified route",
            position="lower_right",
            font_size=10,
            color="#d32f2f",
        )
    if len(result.explored_points) > 0:
        actor = add_points(plotter, result.explored_points, color="#5c6bc0", size=5.5, label="explored vertices")
        if actor is not None:
            actor_groups["FamilyLeaf"].append(actor)
    if len(result.candidate_entries) > 0:
        actor = add_points(plotter, result.candidate_entries, color="#00acc1", size=10.0, label="entry seeds")
        if actor is not None:
            actor_groups["Seeds"].append(actor)
    if len(result.entry_transition_points) > 0:
        actor = add_points(plotter, result.entry_transition_points, color="#26a69a", size=11.0, label="certified entry seeds")
        if actor is not None:
            actor_groups["Seeds"].append(actor)
    if len(result.exit_transition_points) > 0:
        actor = add_points(plotter, result.exit_transition_points, color="#66bb6a", size=11.0, label="discovered exit seeds")
        if actor is not None:
            actor_groups["Seeds"].append(actor)
    actor = add_points(plotter, start_q, color="black", size=16.0, label="start")
    if actor is not None:
        actor_groups["StartGoal"].append(actor)
    actor = add_points(plotter, goal_q, color="gold", size=18.0, label="goal")
    if actor is not None:
        actor_groups["StartGoal"].append(actor)
    if result.entry_switch is not None:
        actor = add_points(plotter, result.entry_switch, color="#212121", size=16.0, label="selected entry switch")
        if actor is not None:
            actor_groups["Seeds"].append(actor)
    if result.exit_switch is not None:
        actor = add_points(plotter, result.exit_switch, color="#2e7d32", size=16.0, label="selected exit switch")
        if actor is not None:
            actor_groups["Seeds"].append(actor)

    def set_visibility(name: str, visible: bool):
        for actor_inner in actor_groups.get(name, []):
            actor_inner.SetVisibility(bool(visible))
        plotter.render()

    labels = [
        "Supports",
        "LeftStage",
        "FamilyLeaves",
        "FamilyLeaf",
        "RightStage",
        "Seeds",
        "Path",
        "TopKPaths",
        "StartGoal",
    ]
    for idx, label in enumerate(labels):
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
