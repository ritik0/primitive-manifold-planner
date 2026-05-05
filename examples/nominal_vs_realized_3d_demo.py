from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from ompl import base as ob
from ompl import geometric as og

from primitive_manifold_planner.families.leaf_state import LeafState
from primitive_manifold_planner.families.standard import PlaneFamily, SphereFamily
from primitive_manifold_planner.manifolds import PlaneManifold, SphereManifold
from primitive_manifold_planner.planners.component_discovery import ComponentModelRegistry
from primitive_manifold_planner.planners.mode_semantics import PlanningSemanticContext
from primitive_manifold_planner.planners.multimodal_component_planner import (
    MultimodalComponentPlanner,
    PlannerConfig,
)
from primitive_manifold_planner.planners.semantic_templates import (
    build_support_transfer_goal_semantic_model,
)
from primitive_manifold_planner.projection import project_newton


def sphere_point(center: np.ndarray, radius: float, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = np.deg2rad(float(azimuth_deg))
    el = np.deg2rad(float(elevation_deg))
    direction = np.array(
        [
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ],
        dtype=float,
    )
    return np.asarray(center, dtype=float) + float(radius) * direction


def build_scene():
    left_support = SphereFamily(
        name="left_support_3d",
        center=np.array([-1.75, 0.0, 0.42], dtype=float),
        radii={1.08: 1.08},
    )
    transfer_plane = PlaneFamily(
        name="transfer_plane_3d",
        base_point=np.array([0.0, 0.0, 0.0], dtype=float),
        normal=np.array([0.0, 0.0, 1.0], dtype=float),
        offsets=[-0.02],
        anchor_span=1.2,
        feasibility_fn=lambda lam, point, goal, meta: bool(
            -1.20 <= float(point[0]) <= 1.20 and -0.95 <= float(point[1]) <= 0.95
        ),
    )
    right_support = SphereFamily(
        name="right_support_3d",
        center=np.array([1.75, 0.0, 0.42], dtype=float),
        radii={1.08: 1.08},
    )

    start_q = sphere_point(left_support.center, 1.08, azimuth_deg=-138.0, elevation_deg=-30.0)
    goal_q = sphere_point(right_support.center, 1.08, azimuth_deg=138.0, elevation_deg=-24.0)
    return [left_support, transfer_plane, right_support], start_q, goal_q


def build_component_registry(families):
    registry = ComponentModelRegistry()
    for fam in families:
        for lam in fam.sample_lambdas():
            registry.register_static_components(fam.name, float(lam), component_ids=["0"])
    return registry


def build_semantic_model():
    def semantic_feasibility(context: PlanningSemanticContext) -> bool:
        if context.point is None:
            return True
        involved = {context.source_family_name, context.target_family_name}
        if "transfer_plane_3d" not in involved:
            return True
        q = np.asarray(context.point, dtype=float)
        return bool(-1.20 <= float(q[0]) <= 1.20 and -0.95 <= float(q[1]) <= 0.95)

    def semantic_admissibility(context: PlanningSemanticContext) -> float:
        if context.point is None:
            return 0.0
        involved = {context.source_family_name, context.target_family_name}
        if "transfer_plane_3d" not in involved:
            return 0.0
        q = np.asarray(context.point, dtype=float)
        goal = None if context.goal_point is None else np.asarray(context.goal_point, dtype=float)

        # Discrete search is encouraged to prefer the upper side of the transfer circle.
        # Realization may still reselect a lower-side exact candidate if it is much more
        # reachable from the current state.
        cost = 0.35 * max(0.0, -float(q[1]))
        if goal is not None:
            cost += 0.03 * float(np.linalg.norm(q - goal))
        return float(cost)

    return build_support_transfer_goal_semantic_model(
        support_families=["left_support_3d"],
        transfer_families=["transfer_plane_3d"],
        goal_support_families=["right_support_3d"],
        transition_feasibility_fn=semantic_feasibility,
        transition_admissibility_fn=semantic_admissibility,
    )


def _circle_seeds_for_sphere_plane(sphere_family: SphereFamily, sphere_lam: float, plane_z: float) -> list[np.ndarray]:
    center = np.asarray(sphere_family.center, dtype=float)
    radius = float(sphere_lam)
    dz = float(plane_z - center[2])
    radial_sq = radius * radius - dz * dz
    if radial_sq <= 0.0:
        return []
    circle_radius = float(np.sqrt(radial_sq))
    seeds = []
    for deg in np.linspace(0.0, 315.0, 8):
        ang = np.deg2rad(float(deg))
        seeds.append(
            np.array(
                [
                    center[0] + circle_radius * np.cos(ang),
                    center[1] + circle_radius * np.sin(ang),
                    plane_z,
                ],
                dtype=float,
            )
        )
    return seeds


def seed_points_fn(source_family, source_lam, target_family, target_lam):
    seeds = []

    if isinstance(source_family, SphereFamily) and isinstance(target_family, PlaneFamily):
        plane_z = float(target_family.manifold(target_lam).point[2])
        seeds.extend(_circle_seeds_for_sphere_plane(source_family, float(source_lam), plane_z))
    elif isinstance(source_family, PlaneFamily) and isinstance(target_family, SphereFamily):
        plane_z = float(source_family.manifold(source_lam).point[2])
        seeds.extend(_circle_seeds_for_sphere_plane(target_family, float(target_lam), plane_z))

    seeds.extend(np.asarray(p, dtype=float).copy() for p in source_family.transition_seed_anchors(source_lam))
    seeds.extend(np.asarray(p, dtype=float).copy() for p in target_family.transition_seed_anchors(target_lam))
    return seeds


def plot_manifold(ax, manifold, color="lightgray", alpha=0.14):
    if isinstance(manifold, SphereManifold):
        u = np.linspace(0.0, 2.0 * np.pi, 36)
        v = np.linspace(0.0, np.pi, 20)
        x = manifold.center[0] + manifold.radius * np.outer(np.cos(u), np.sin(v))
        y = manifold.center[1] + manifold.radius * np.outer(np.sin(u), np.sin(v))
        z = manifold.center[2] + manifold.radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(x, y, z, color=color, alpha=alpha, rstride=2, cstride=2, linewidth=0.6)
    elif isinstance(manifold, PlaneManifold):
        xx = np.linspace(-1.20, 1.20, 16)
        yy = np.linspace(-0.95, 0.95, 12)
        X, Y = np.meshgrid(xx, yy)
        Z = np.zeros_like(X) + manifold.point[2]
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0.0, shade=False)


def concatenate_realization(realization):
    pts = []
    for i, step in enumerate(realization.steps):
        path = np.asarray(step.path)
        if len(path) == 0:
            continue
        if i == 0:
            pts.extend(list(path))
        else:
            pts.extend(list(path[1:]))
    if not pts:
        return np.zeros((0, 3))
    return np.asarray(pts)


def transition_rows(result) -> list[dict]:
    rows = []
    if result.realization is None:
        return rows
    for step in result.realization.steps:
        if step.realized_candidate_index is None:
            continue
        rows.append(
            {
                "family": step.family_name,
                "lam": float(step.lam),
                "nominal_idx": None if step.nominal_candidate_index is None else int(step.nominal_candidate_index),
                "realized_idx": int(step.realized_candidate_index),
                "deviation": bool(step.nominal_candidate_index != step.realized_candidate_index),
                "nominal_point": None if step.nominal_transition_point is None else np.asarray(step.nominal_transition_point, dtype=float),
                "realized_point": None if step.transition_point is None else np.asarray(step.transition_point, dtype=float),
            }
        )
    return rows


def main():
    np.random.seed(31)

    families, start_q, goal_q = build_scene()
    registry = build_component_registry(families)
    semantic_model = build_semantic_model()

    planner = MultimodalComponentPlanner(
        families=families,
        project_newton=project_newton,
        seed_points_fn=seed_points_fn,
        component_model_registry=registry,
        config=PlannerConfig(
            semantic_model=semantic_model,
            local_planner_name="ompl_projection",
            local_planner_kwargs=dict(
                goal_tol=1e-3,
                max_iters=900,
                projection_tol=1e-10,
                projection_max_iters=60,
                solve_time=0.18,
                ompl_planner_name="RRTConnect",
            ),
            step_size=0.10,
            max_candidates_per_pair=10,
        ),
    )

    result = planner.plan_with_inferred_components(
        start_state=LeafState("left_support_3d", 1.08, start_q.copy()),
        goal_family_name="right_support_3d",
        goal_lam=1.08,
        goal_q=goal_q.copy(),
    )

    print("\nExample 62: 3D nominal vs realized transition reselection")
    print(f"OMPL backend = {og.RRTConnect.__name__} on ProjectedStateSpace (delta={ob.CONSTRAINED_STATE_SPACE_DELTA})")
    print(f"route_found = {result.route_found}")
    print(f"realization_success = {result.realization_success}")
    print(f"message = {result.message}")
    print(f"route = {result.diagnostics.selected_route_string}")
    print(f"nominal candidate indices = {result.diagnostics.selected_candidate_indices}")
    print(f"realized candidate indices = {result.diagnostics.realized_candidate_indices}")
    print(f"realized deviations = {result.diagnostics.realized_transition_deviations}")

    rows = transition_rows(result)
    for idx, row in enumerate(rows):
        print(
            "  "
            f"transition {idx}: "
            f"{row['family']}[{row['lam']}] "
            f"nominal={row['nominal_idx']} "
            f"realized={row['realized_idx']} "
            f"deviation={row['deviation']}"
        )

    fig = plt.figure(figsize=(10.5, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = {
        "left_support_3d": "#c58b4c",
        "transfer_plane_3d": "#88a8c6",
        "right_support_3d": "#c58b4c",
    }
    for fam in families:
        for lam in fam.sample_lambdas():
            plot_manifold(ax, fam.manifold(lam), color=colors.get(fam.name, "lightgray"))

    if result.realization is not None and result.realization.success:
        path = concatenate_realization(result.realization)
        if len(path) > 0:
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color="#1565c0", linewidth=3.0, label="realized path")

    for row in rows:
        nominal_point = row["nominal_point"]
        realized_point = row["realized_point"]
        if nominal_point is not None:
            ax.scatter(
                nominal_point[0],
                nominal_point[1],
                nominal_point[2],
                s=110,
                marker="o",
                facecolors="none",
                edgecolors="#d62728",
                linewidths=1.8,
                label="nominal transition" if row is rows[0] else None,
            )
        if realized_point is not None:
            ax.scatter(
                realized_point[0],
                realized_point[1],
                realized_point[2],
                s=75,
                marker="x",
                color="black",
                label="realized transition" if row is rows[0] else None,
            )

    ax.scatter(start_q[0], start_q[1], start_q[2], s=90, marker="s", color="black", label="start")
    ax.scatter(goal_q[0], goal_q[1], goal_q[2], s=130, marker="*", color="gold", edgecolor="black", label="goal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Example 62: 3D nominal vs realized transition reselection")
    ax.legend(loc="upper right")
    ax.view_init(elev=24, azim=-58)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
