"""Internal geometric and OMPL support helpers for the continuous-transfer experiment.

This module intentionally owns the reusable low-level helpers that Example 65 depends on,
so the staged planner no longer reaches back into Example 64 for core motion primitives.
"""

from __future__ import annotations

import numpy as np

from primitive_manifold_planner.manifolds import MaskedManifold, PlaneManifold
from primitive_manifold_planner.planning.local import (
    ompl_atlas_interpolate,
    ompl_projected_interpolate,
    ompl_sample_state_on_manifold,
)

try:
    import pyvista as pv
except Exception:
    pv = None


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


def unwrap_manifold(manifold):
    return manifold.base_manifold if isinstance(manifold, MaskedManifold) else manifold


def is_plane_like(manifold) -> bool:
    return isinstance(unwrap_manifold(manifold), PlaneManifold)


def sample_chart_centers(path: np.ndarray, chart_count: int) -> np.ndarray:
    arr = np.asarray(path, dtype=float)
    if len(arr) == 0 or chart_count <= 0:
        return np.zeros((0, 3), dtype=float)
    count = min(int(chart_count), len(arr))
    ids = np.linspace(0, len(arr) - 1, count, dtype=int)
    return np.asarray(arr[ids], dtype=float)


def smooth_sphere_arc(
    center: np.ndarray,
    radius: float,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    num: int = 120,
) -> np.ndarray:
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
    return np.asarray(
        [(1.0 - t) * np.asarray(q_start, dtype=float) + t * np.asarray(q_goal, dtype=float) for t in ts],
        dtype=float,
    )


def concatenate_paths(*paths: np.ndarray) -> np.ndarray:
    merged: list[np.ndarray] = []
    for idx, path in enumerate(paths):
        arr = np.asarray(path, dtype=float)
        if len(arr) == 0:
            continue
        if idx == 0 or len(merged) == 0:
            merged.extend(list(arr))
        else:
            merged.extend(list(arr[1:]))
    return np.asarray(merged, dtype=float) if merged else np.zeros((0, 3), dtype=float)


def deduplicate_points(points: list[np.ndarray], tol: float = 1e-4) -> np.ndarray:
    unique: list[np.ndarray] = []
    for q in points:
        qq = np.asarray(q, dtype=float).reshape(-1)
        if any(np.linalg.norm(qq - p) <= tol for p in unique):
            continue
        unique.append(qq)
    return np.asarray(unique, dtype=float) if unique else np.zeros((0, 3), dtype=float)


def explored_points_from_edges(edges: list[tuple[np.ndarray, np.ndarray]], tol: float = 1e-4) -> np.ndarray:
    pts: list[np.ndarray] = []
    for q_a, q_b in edges:
        pts.append(np.asarray(q_a, dtype=float))
        pts.append(np.asarray(q_b, dtype=float))
    return deduplicate_points(pts, tol=tol)


def merge_edges(*edge_groups: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
    merged: list[tuple[np.ndarray, np.ndarray]] = []
    for group in edge_groups:
        for q_a, q_b in group:
            merged.append((np.asarray(q_a, dtype=float).copy(), np.asarray(q_b, dtype=float).copy()))
    return merged


def ompl_native_exploration_target(
    manifold,
    q_seed: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> np.ndarray | None:
    constrained_space_name = "projection" if is_plane_like(manifold) else "atlas"
    return ompl_sample_state_on_manifold(
        manifold=manifold,
        x_seed=np.asarray(q_seed, dtype=float),
        constrained_space_name=constrained_space_name,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        projection_tol=1e-10,
        projection_max_iters=60,
        ompl_lambda=2.0,
        atlas_exploration=0.75,
        atlas_epsilon=0.03,
        atlas_rho=0.24,
        atlas_alpha=np.pi / 8.0,
        atlas_max_charts_per_extension=80,
        atlas_separated=True,
    )


def solve_exact_segment_on_manifold(
    manifold,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
):
    if is_plane_like(manifold):
        return ompl_projected_interpolate(
            manifold=manifold,
            x_start=x_start,
            x_goal=x_goal,
            step_size=0.12,
            goal_tol=1e-3,
            max_iters=800,
            projection_tol=1e-10,
            projection_max_iters=60,
            solve_time=0.35,
            ompl_planner_name="RRTConnect",
            bounds_min=bounds_min,
            bounds_max=bounds_max,
        )

    return ompl_atlas_interpolate(
        manifold=manifold,
        x_start=x_start,
        x_goal=x_goal,
        step_size=0.10,
        goal_tol=1e-3,
        max_iters=800,
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
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )


def explore_on_manifold_from_frontier(
    manifold,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
):
    if is_plane_like(manifold):
        return ompl_projected_interpolate(
            manifold=manifold,
            x_start=x_start,
            x_goal=x_goal,
            step_size=0.20,
            goal_tol=8e-2,
            max_iters=700,
            projection_tol=1e-10,
            projection_max_iters=60,
            solve_time=0.12,
            ompl_planner_name="RRT",
            bounds_min=bounds_min,
            bounds_max=bounds_max,
        )

    return ompl_atlas_interpolate(
        manifold=manifold,
        x_start=x_start,
        x_goal=x_goal,
        step_size=0.18,
        goal_tol=7e-2,
        max_iters=900,
        projection_tol=1e-10,
        projection_max_iters=60,
        solve_time=0.22,
        ompl_planner_name="RRT",
        ompl_lambda=2.0,
        atlas_epsilon=0.035,
        atlas_rho=0.28,
        atlas_exploration=0.92,
        atlas_alpha=np.pi / 5.0,
        atlas_max_charts_per_extension=96,
        atlas_separated=True,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )


def refine_intersection_on_both_manifolds(
    manifold_a,
    manifold_b,
    x0: np.ndarray,
    tol: float = 1e-8,
    max_iters: int = 25,
) -> tuple[np.ndarray, bool]:
    x = np.asarray(x0, dtype=float).reshape(-1).copy()
    best_x = x.copy()
    best_norm = float("inf")

    for _ in range(max_iters):
        ra = np.asarray(manifold_a.residual(x), dtype=float).reshape(-1)
        rb = np.asarray(manifold_b.residual(x), dtype=float).reshape(-1)
        residual = np.concatenate([ra, rb], axis=0)
        residual_norm = float(np.linalg.norm(residual))

        if residual_norm < best_norm:
            best_norm = residual_norm
            best_x = x.copy()
        if residual_norm <= tol:
            return x, True

        ja = np.asarray(manifold_a.jacobian(x), dtype=float)
        jb = np.asarray(manifold_b.jacobian(x), dtype=float)
        jacobian = np.vstack([ja, jb])

        try:
            delta, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        except np.linalg.LinAlgError:
            break

        if not np.all(np.isfinite(delta)):
            break
        if float(np.linalg.norm(delta)) <= 1e-14:
            break
        x = x + delta

    return best_x, best_norm <= max(10.0 * tol, 1e-5)


def build_segment_polydata(edges: list[tuple[np.ndarray, np.ndarray]]):
    if pv is None or len(edges) == 0:
        return None
    points: list[np.ndarray] = []
    lines: list[int] = []
    idx = 0
    for q_a, q_b in edges:
        a = np.asarray(q_a, dtype=float)
        b = np.asarray(q_b, dtype=float)
        points.extend([a, b])
        lines.extend([2, idx, idx + 1])
        idx += 2
    return pv.PolyData(np.asarray(points, dtype=float), lines=np.asarray(lines, dtype=np.int64))
