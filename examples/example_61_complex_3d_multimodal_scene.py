from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from ompl import base as ob
from ompl import geometric as og

from primitive_manifold_planner.families.standard import PlaneFamily, SphereFamily
from primitive_manifold_planner.manifolds import PlaneManifold, SphereManifold
from primitive_manifold_planner.planning.local import (
    ompl_atlas_interpolate,
    ompl_projected_interpolate,
)
from primitive_manifold_planner.visualization import add_path, add_points, pyvista_available

try:
    import pyvista as pv
except Exception:
    pv = None


@dataclass
class DirectPlanResult:
    success: bool
    message: str
    path: np.ndarray
    switch_point: np.ndarray | None
    route_string: str
    sphere_result: object | None = None
    plane_result: object | None = None


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
    sphere = SphereFamily(
        name="sphere_support_3d",
        center=np.array([0.0, 0.0, 0.90], dtype=float),
        radii={1.0: 1.0},
    )
    plane = PlaneFamily(
        name="transfer_plane_3d",
        base_point=np.array([0.0, 0.0, 0.0], dtype=float),
        normal=np.array([0.0, 0.0, 1.0], dtype=float),
        offsets=[0.20],
        anchor_span=1.10,
    )
    start_q = sphere_point(sphere.center, 1.0, azimuth_deg=10.0, elevation_deg=55.0)
    goal_q = np.array([0.55, 0.20, 0.20], dtype=float)
    return [sphere, plane], start_q, goal_q


def plot_manifold(ax, manifold, color="lightgray", alpha=0.18):
    if isinstance(manifold, SphereManifold):
        u = np.linspace(0.0, 2.0 * np.pi, 36)
        v = np.linspace(0.0, np.pi, 20)
        x = manifold.center[0] + manifold.radius * np.outer(np.cos(u), np.sin(v))
        y = manifold.center[1] + manifold.radius * np.outer(np.sin(u), np.sin(v))
        z = manifold.center[2] + manifold.radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(x, y, z, color=color, alpha=alpha, rstride=2, cstride=2, linewidth=0.7)
    elif isinstance(manifold, PlaneManifold):
        xx = np.linspace(-1.2, 1.2, 14)
        yy = np.linspace(-1.2, 1.2, 14)
        X, Y = np.meshgrid(xx, yy)
        Z = np.zeros_like(X) + manifold.point[2]
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0.0, shade=False)


def sphere_plane_intersection_points(
    sphere_center: np.ndarray,
    sphere_radius: float,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    num_points: int = 120,
) -> np.ndarray:
    center = np.asarray(sphere_center, dtype=float)
    point = np.asarray(plane_point, dtype=float)
    normal = np.asarray(plane_normal, dtype=float)
    normal = normal / max(np.linalg.norm(normal), 1e-12)

    signed_distance = float(np.dot(center - point, normal))
    circle_radius_sq = float(sphere_radius) ** 2 - signed_distance**2
    if circle_radius_sq <= 0.0:
        return np.zeros((0, 3), dtype=float)

    circle_center = center - signed_distance * normal
    circle_radius = np.sqrt(circle_radius_sq)
    reference = np.array([1.0, 0.0, 0.0], dtype=float) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=float)
    tangent_u = np.cross(normal, reference)
    tangent_u = tangent_u / max(np.linalg.norm(tangent_u), 1e-12)
    tangent_v = np.cross(normal, tangent_u)
    tangent_v = tangent_v / max(np.linalg.norm(tangent_v), 1e-12)

    angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    return np.asarray(
        [
            circle_center + circle_radius * (np.cos(theta) * tangent_u + np.sin(theta) * tangent_v)
            for theta in angles
        ],
        dtype=float,
    )


def choose_switch_point(start_q: np.ndarray, goal_q: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    if len(candidates) == 0:
        raise ValueError("No sphere-plane intersection points were found.")

    costs = []
    for q in candidates:
        cost = float(np.linalg.norm(q - start_q) + 1.2 * np.linalg.norm(q - goal_q))
        costs.append(cost)
    return np.asarray(candidates[int(np.argmin(costs))], dtype=float)


def smooth_sphere_arc(center: np.ndarray, radius: float, q_start: np.ndarray, q_goal: np.ndarray, num: int = 120) -> np.ndarray:
    u0 = (np.asarray(q_start, dtype=float) - np.asarray(center, dtype=float)) / float(radius)
    u1 = (np.asarray(q_goal, dtype=float) - np.asarray(center, dtype=float)) / float(radius)
    u0 = u0 / max(np.linalg.norm(u0), 1e-12)
    u1 = u1 / max(np.linalg.norm(u1), 1e-12)
    dot = float(np.clip(np.dot(u0, u1), -1.0, 1.0))

    if dot > 1.0 - 1e-8:
        ts = np.linspace(0.0, 1.0, num)
        pts = [(1.0 - t) * q_start + t * q_goal for t in ts]
        pts = np.asarray(pts, dtype=float)
        dirs = pts - np.asarray(center, dtype=float)
        dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
        return np.asarray(center, dtype=float) + float(radius) * dirs

    omega = float(np.arccos(dot))
    sin_omega = max(np.sin(omega), 1e-12)
    ts = np.linspace(0.0, 1.0, num)
    dirs = [
        (np.sin((1.0 - t) * omega) / sin_omega) * u0 + (np.sin(t * omega) / sin_omega) * u1
        for t in ts
    ]
    return np.asarray(center, dtype=float) + float(radius) * np.asarray(dirs, dtype=float)


def smooth_plane_segment(q_start: np.ndarray, q_goal: np.ndarray, num: int = 60) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, num)
    return np.asarray([(1.0 - t) * np.asarray(q_start, dtype=float) + t * np.asarray(q_goal, dtype=float) for t in ts], dtype=float)


def concatenate_paths(*paths: np.ndarray) -> np.ndarray:
    kept: list[np.ndarray] = []
    for idx, path in enumerate(paths):
        arr = np.asarray(path, dtype=float)
        if len(arr) == 0:
            continue
        if idx == 0 or len(kept) == 0:
            kept.extend(list(arr))
        else:
            kept.extend(list(arr[1:]))
    return np.asarray(kept, dtype=float) if kept else np.zeros((0, 3), dtype=float)


def plan_direct_multimodal_route(families, start_q: np.ndarray, goal_q: np.ndarray) -> DirectPlanResult:
    sphere_family = next(fam for fam in families if isinstance(fam, SphereFamily))
    plane_family = next(fam for fam in families if isinstance(fam, PlaneFamily))
    sphere = sphere_family.manifold(1.0)
    plane = plane_family.manifold(0.20)

    circle_points = sphere_plane_intersection_points(
        sphere_center=sphere.center,
        sphere_radius=sphere.radius,
        plane_point=plane.point,
        plane_normal=plane.normal,
        num_points=160,
    )
    switch_q = choose_switch_point(start_q, goal_q, circle_points)

    sphere_result = ompl_atlas_interpolate(
        manifold=sphere,
        x_start=start_q,
        x_goal=switch_q,
        step_size=0.10,
        goal_tol=1e-3,
        max_iters=600,
        projection_tol=1e-10,
        projection_max_iters=60,
        solve_time=0.40,
        ompl_planner_name="RRTConnect",
        ompl_lambda=2.0,
        atlas_epsilon=0.03,
        atlas_rho=0.24,
        atlas_exploration=0.75,
        atlas_alpha=np.pi / 8.0,
        atlas_max_charts_per_extension=80,
        atlas_separated=True,
    )
    if not sphere_result.success:
        return DirectPlanResult(
            success=False,
            message=f"Sphere segment failed: {sphere_result.message}",
            path=np.asarray([start_q], dtype=float),
            switch_point=switch_q,
            route_string="sphere_support_3d[1.0]",
            sphere_result=sphere_result,
        )

    plane_result = ompl_projected_interpolate(
        manifold=plane,
        x_start=switch_q,
        x_goal=goal_q,
        step_size=0.10,
        goal_tol=1e-3,
        max_iters=600,
        projection_tol=1e-10,
        projection_max_iters=60,
        solve_time=0.35,
        ompl_planner_name="RRTConnect",
        bounds_min=np.array([-1.5, -1.5, -0.2], dtype=float),
        bounds_max=np.array([1.5, 1.5, 2.0], dtype=float),
    )
    if not plane_result.success:
        return DirectPlanResult(
            success=False,
            message=f"Plane segment failed: {plane_result.message}",
            path=np.asarray(sphere_result.path, dtype=float),
            switch_point=switch_q,
            route_string="sphere_support_3d[1.0] -> transfer_plane_3d[0.2]",
            sphere_result=sphere_result,
            plane_result=plane_result,
        )

    smooth_sphere_path = smooth_sphere_arc(
        center=sphere.center,
        radius=sphere.radius,
        q_start=start_q,
        q_goal=switch_q,
        num=140,
    )
    smooth_plane_path = smooth_plane_segment(
        q_start=switch_q,
        q_goal=goal_q,
        num=70,
    )

    return DirectPlanResult(
        success=True,
        message="Direct OMPL multimodal route succeeded.",
        path=concatenate_paths(smooth_sphere_path, smooth_plane_path),
        switch_point=switch_q,
        route_string="sphere_support_3d[1.0] -> transfer_plane_3d[0.2]",
        sphere_result=sphere_result,
        plane_result=plane_result,
    )


def show_pyvista_scene(families, colors, result: DirectPlanResult, start_q: np.ndarray, goal_q: np.ndarray) -> bool:
    if pv is None:
        return False

    sphere_family = next(fam for fam in families if isinstance(fam, SphereFamily))
    plane_family = next(fam for fam in families if isinstance(fam, PlaneFamily))
    sphere = sphere_family.manifold(1.0)
    plane = plane_family.manifold(0.20)

    plotter = pv.Plotter(window_size=(1200, 780))
    plotter.add_text("Example 61: explicit OMPL sphere-to-plane transfer", font_size=12)

    sphere_mesh = pv.Sphere(
        radius=float(sphere.radius),
        center=tuple(np.asarray(sphere.center, dtype=float)),
        theta_resolution=64,
        phi_resolution=48,
    )
    plane_mesh = pv.Plane(
        center=tuple(np.asarray(plane.point, dtype=float)),
        direction=tuple(np.asarray(plane.normal, dtype=float)),
        i_size=2.4,
        j_size=2.4,
        i_resolution=12,
        j_resolution=12,
    )
    intersection_pts = sphere_plane_intersection_points(
        sphere_center=sphere.center,
        sphere_radius=sphere.radius,
        plane_point=plane.point,
        plane_normal=plane.normal,
    )

    geometry_actors = []
    path_actors = []
    geometry_actors.append(
        plotter.add_mesh(
            sphere_mesh,
            color=colors["sphere_support_3d"],
            opacity=0.22,
            smooth_shading=True,
            show_edges=False,
        )
    )
    geometry_actors.append(
        plotter.add_mesh(
            plane_mesh,
            color=colors["transfer_plane_3d"],
            opacity=0.35,
            style="wireframe",
            line_width=1.5,
        )
    )
    actor = add_path(plotter, intersection_pts, color="#2f4858", width=4.0, label="intersection circle")
    if actor is not None:
        geometry_actors.append(actor)

    if len(result.path) >= 2:
        actor = add_path(plotter, result.path, color="#1565c0", width=7.0, label="OMPL path")
        if actor is not None:
            path_actors.append(actor)

    if result.switch_point is not None:
        actor = add_points(plotter, result.switch_point, color="#333333", size=12.0, label="switch point")
        if actor is not None:
            path_actors.append(actor)

    actor = add_points(plotter, start_q, color="black", size=16.0, label="start")
    if actor is not None:
        path_actors.append(actor)
    actor = add_points(plotter, goal_q, color="gold", size=18.0, label="goal")
    if actor is not None:
        path_actors.append(actor)

    def set_group_visibility(actors, visible: bool):
        for actor in actors:
            if actor is not None:
                actor.SetVisibility(bool(visible))
        plotter.render()

    toggles = [("Geometry", geometry_actors, True), ("Path", path_actors, True)]
    y = 10
    for label, actors, default in toggles:
        plotter.add_checkbox_button_widget(
            callback=lambda state, actors=actors: set_group_visibility(actors, bool(state)),
            value=default,
            position=(10, y),
            size=24,
            color_on="lightgray",
            color_off="white",
            background_color="#4b5563",
        )
        plotter.add_text(label, position=(42, y + 2), font_size=9)
        y += 32

    plotter.add_axes()
    plotter.show_grid()
    plotter.show()
    return True


def main():
    np.random.seed(61)
    families, start_q, goal_q = build_scene()
    result = plan_direct_multimodal_route(families, start_q, goal_q)

    print("\nRunning Example 61: explicit OMPL sphere-to-plane transfer")
    print("What we are doing:")
    print("1. Build the exact circle where the sphere and plane intersect.")
    print("2. Choose one switch point on that circle.")
    print("3. Use OMPL AtlasStateSpace on the sphere segment.")
    print("4. Use OMPL ProjectedStateSpace on the plane segment.")
    print(f"OMPL ambient defaults: delta={ob.CONSTRAINED_STATE_SPACE_DELTA}, lambda={ob.CONSTRAINED_STATE_SPACE_LAMBDA}")
    print(f"example planners = sphere:{og.RRTConnect.__name__} with AtlasStateSpace, plane:{og.RRTConnect.__name__} with ProjectedStateSpace")
    print(f"success = {result.success}")
    print(f"message = {result.message}")
    print(f"route = {result.route_string}")
    print(f"switch_point = {None if result.switch_point is None else np.round(result.switch_point, 4)}")
    if result.sphere_result is not None:
        print(f"sphere charts = {getattr(result.sphere_result, 'chart_count', 0)}")
    if result.plane_result is not None:
        print(f"plane path points = {len(np.asarray(result.plane_result.path, dtype=float))}")

    colors = {
        "sphere_support_3d": "#c58b4c",
        "transfer_plane_3d": "#7fa7c6",
    }

    if pyvista_available():
        shown = show_pyvista_scene(families, colors, result, start_q, goal_q)
        if shown:
            return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for fam in families:
        for lam in fam.sample_lambdas():
            plot_manifold(ax, fam.manifold(lam), color=colors.get(fam.name, "#999999"))

    if result.switch_point is not None:
        circle_points = sphere_plane_intersection_points(
            sphere_center=families[0].center,
            sphere_radius=1.0,
            plane_point=families[1].manifold(0.20).point,
            plane_normal=families[1].manifold(0.20).normal,
        )
        ax.plot(
            circle_points[:, 0],
            circle_points[:, 1],
            circle_points[:, 2],
            color="#2f4858",
            linewidth=2.0,
            alpha=0.8,
            label="intersection circle",
        )

    if len(result.path) >= 2:
        ax.plot(
            result.path[:, 0],
            result.path[:, 1],
            result.path[:, 2],
            color="#1565c0",
            linewidth=3.0,
            label="OMPL path",
        )

    if result.switch_point is not None:
        ax.scatter(
            result.switch_point[0],
            result.switch_point[1],
            result.switch_point[2],
            s=55,
            marker="x",
            color="#333333",
            label="switch point",
        )

    ax.scatter(start_q[0], start_q[1], start_q[2], s=90, marker="s", color="black", label="start")
    ax.scatter(goal_q[0], goal_q[1], goal_q[2], s=130, marker="*", color="gold", edgecolor="black", label="goal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Example 61: direct OMPL multimodal planning")
    ax.view_init(elev=24, azim=-58)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
