from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from ompl import util as ou

from primitive_manifold_planner.families.standard import MaskedFamily, PlaneFamily, SphereFamily
from primitive_manifold_planner.manifolds import MaskedManifold, PlaneManifold, SphereManifold
from primitive_manifold_planner.planning.local import (
    LocalPathResult,
    ompl_sample_state_on_manifold,
    ompl_atlas_interpolate,
    ompl_projected_interpolate,
)
from primitive_manifold_planner.projection import project_newton
from primitive_manifold_planner.visualization import add_manifold, add_points, pyvista_available

try:
    import pyvista as pv
except Exception:
    pv = None


@dataclass
class ExplicitOmplRoute:
    """High-level route plus exploration artifacts for intrinsic examples."""

    success: bool
    message: str
    route_string: str
    path: np.ndarray
    switch_points: list[np.ndarray] = field(default_factory=list)
    raw_path: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    transition_points_by_interface: list[np.ndarray] = field(default_factory=list)
    chart_centers_by_segment: list[np.ndarray] = field(default_factory=list)
    explored_points_by_segment: list[np.ndarray] = field(default_factory=list)
    explored_edges_by_segment: list[list[tuple[np.ndarray, np.ndarray]]] = field(default_factory=list)
    segment_results: list[object | None] = field(default_factory=list)

    @property
    def entry_switch(self) -> np.ndarray | None:
        return None if len(self.switch_points) == 0 else np.asarray(self.switch_points[0], dtype=float)

    @property
    def exit_switch(self) -> np.ndarray | None:
        return None if len(self.switch_points) == 0 else np.asarray(self.switch_points[-1], dtype=float)

#used to create start and goal points using angles
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


def is_sphere_like(manifold) -> bool:
    return isinstance(unwrap_manifold(manifold), SphereManifold)


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

    families = [left_support, transfer_plane, right_support]
    start_q = sphere_point(left_support.center, 1.05, azimuth_deg=0.0, elevation_deg=-90.0)
    goal_q = sphere_point(right_support.center, 1.05, azimuth_deg=0.0, elevation_deg=90.0)
    return families, start_q, goal_q


def segment_endpoints(
    start_q: np.ndarray,
    goal_q: np.ndarray,
    switch_points: list[np.ndarray],
    segment_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    q_start = np.asarray(start_q if segment_index == 0 else switch_points[segment_index - 1], dtype=float)
    q_goal = np.asarray(goal_q if segment_index == len(switch_points) else switch_points[segment_index], dtype=float)
    return q_start, q_goal


def display_segment_path(manifold, q_start: np.ndarray, q_goal: np.ndarray, raw_result) -> np.ndarray:
    if is_sphere_like(manifold):
        return smooth_sphere_arc(
            center=unwrap_manifold(manifold).center,
            radius=unwrap_manifold(manifold).radius,
            q_start=q_start,
            q_goal=q_goal,
            num=140,
        )
    if is_plane_like(manifold):
        return smooth_plane_segment(q_start=q_start, q_goal=q_goal, num=90)
    path = np.asarray(getattr(raw_result, "path", np.zeros((0, 3), dtype=float)), dtype=float)
    if len(path) > 0:
        return path
    return smooth_plane_segment(q_start=q_start, q_goal=q_goal, num=90)


def build_route_result(
    families,
    lambdas: list[float],
    manifolds,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    message: str,
    success: bool,
    switch_points: list[np.ndarray],
    transition_points_by_interface: list[np.ndarray],
    segment_results: list[object | None],
) -> ExplicitOmplRoute:
    """Assemble display/raw route data from per-manifold segment solves."""

    raw_segments = [
        np.asarray(getattr(result, "path", np.zeros((0, 3), dtype=float)), dtype=float)
        for result in segment_results
        if result is not None
    ]
    raw_path = concatenate_paths(*raw_segments) if len(raw_segments) > 0 else np.asarray([start_q], dtype=float)

    display_segments: list[np.ndarray] = []
    usable_segment_count = min(len(manifolds), len(segment_results))
    for seg_idx in range(usable_segment_count):
        segment_result = segment_results[seg_idx]
        if segment_result is None:
            continue
        q_seg_start, q_seg_goal = segment_endpoints(start_q, goal_q, switch_points, seg_idx)
        display_segments.append(
            display_segment_path(
                manifold=manifolds[seg_idx],
                q_start=q_seg_start,
                q_goal=q_seg_goal,
                raw_result=segment_result,
            )
        )

    path = concatenate_paths(*display_segments) if len(display_segments) > 0 else np.asarray([start_q], dtype=float)
    route_string = " -> ".join(f"{family.name}[{lam:g}]" for family, lam in zip(families, lambdas))
    chart_centers_by_segment = [
        sample_chart_centers(
            np.asarray(getattr(result, "path", np.zeros((0, 3), dtype=float)), dtype=float),
            int(getattr(result, "chart_count", 0)),
        )
        if result is not None
        else np.zeros((0, 3), dtype=float)
        for result in segment_results
    ]
    explored_edges_by_segment = [
        list(getattr(result, "explored_edges", [])) if result is not None else []
        for result in segment_results
    ]
    explored_points_by_segment = [
        explored_points_from_edges(edges) if len(edges) > 0 else np.zeros((0, 3), dtype=float)
        for edges in explored_edges_by_segment
    ]
    return ExplicitOmplRoute(
        success=success,
        message=message,
        route_string=route_string,
        path=path,
        switch_points=[np.asarray(q, dtype=float) for q in switch_points],
        raw_path=raw_path,
        transition_points_by_interface=[
            np.asarray(points, dtype=float) if len(np.asarray(points, dtype=float)) > 0 else np.zeros((0, 3), dtype=float)
            for points in transition_points_by_interface
        ],
        chart_centers_by_segment=chart_centers_by_segment,
        explored_points_by_segment=explored_points_by_segment,
        explored_edges_by_segment=explored_edges_by_segment,
        segment_results=segment_results,
    )

# so when in left sphere use right sphere center 
def guide_point_for_step(manifolds, step_index: int, goal_q: np.ndarray) -> np.ndarray:
    if step_index + 2 < len(manifolds) and is_sphere_like(manifolds[step_index + 2]):
        return np.asarray(unwrap_manifold(manifolds[step_index + 2]).center, dtype=float)
    if step_index + 1 < len(manifolds) and is_sphere_like(manifolds[step_index + 1]):
        return np.asarray(unwrap_manifold(manifolds[step_index + 1]).center, dtype=float)
    if step_index + 1 < len(manifolds) and is_plane_like(manifolds[step_index + 1]):
        return project_point_to_plane(manifolds[step_index + 1], goal_q)
    return np.asarray(goal_q, dtype=float)


def sample_chart_centers(path: np.ndarray, chart_count: int) -> np.ndarray:
    arr = np.asarray(path, dtype=float)
    if len(arr) == 0 or chart_count <= 0:
        return np.zeros((0, 3), dtype=float)
    count = min(int(chart_count), len(arr))
    ids = np.linspace(0, len(arr) - 1, count, dtype=int)
    return np.asarray(arr[ids], dtype=float)


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


def target_residual_scalar(manifold, q: np.ndarray) -> float:
    residual = np.asarray(manifold.residual(np.asarray(q, dtype=float)), dtype=float).reshape(-1)
    if residual.shape[0] != 1:
        raise ValueError(
            "Online sign-change transition detection currently requires the destination manifold to have codimension 1."
        )
    return float(residual[0])


#asks OMPL for one valid sample on the current manifold --> target point
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

#list of potential promising explored manifold points for next exploration
def update_frontier_points(
    frontier: list[np.ndarray],
    new_points: list[np.ndarray],
    guide_point: np.ndarray,
    max_points: int = 24,
    dedup_tol: float = 1e-4,
) -> list[np.ndarray]:
    merged = deduplicate_points(list(frontier) + list(new_points), tol=dedup_tol)
    if len(merged) == 0:
        return []

    goal = np.asarray(guide_point, dtype=float)
    scores = np.asarray([float(np.linalg.norm(np.asarray(q, dtype=float) - goal)) for q in merged], dtype=float)
    order = np.argsort(scores)
    kept = [np.asarray(merged[i], dtype=float) for i in order[:max_points]]
    return kept

# after the run it returns endpoints of the explored edges and few points along the returned path.
def frontier_points_from_result(result) -> list[np.ndarray]:
    if result is None:
        return []

    pts: list[np.ndarray] = []
    path = np.asarray(getattr(result, "path", np.zeros((0, 3), dtype=float)), dtype=float)
    if len(path) > 0:
        ids = np.linspace(0, len(path) - 1, min(6, len(path)), dtype=int)
        pts.extend(np.asarray(path[i], dtype=float) for i in ids)

    for q_a, q_b in getattr(result, "explored_edges", []):
        pts.append(np.asarray(q_a, dtype=float))
        pts.append(np.asarray(q_b, dtype=float))
    return pts


def exploration_failure_result(
    q_start: np.ndarray,
    explored_edges: list[tuple[np.ndarray, np.ndarray]],
    planner_name: str,
    message: str,
    chart_count: int = 0,
) -> LocalPathResult:
    explored_pts = explored_points_from_edges(explored_edges)
    if len(explored_pts) > 0:
        ids = np.argsort(np.linalg.norm(explored_pts - np.asarray(q_start, dtype=float), axis=1))
        path = np.asarray(explored_pts[ids[: min(8, len(explored_pts))]], dtype=float)
    else:
        path = np.asarray([q_start], dtype=float)
    return LocalPathResult(
        success=False,
        path=path,
        iterations=len(explored_edges),
        reached_goal=False,
        message=message,
        planner_name=planner_name,
        chart_count=int(chart_count),
        explored_edges=explored_edges,
    )


def solve_exact_segment_on_manifold(
    manifold,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
):
    """Solve a local segment on one active manifold with the appropriate OMPL helper."""

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
    """Expand evidence from a frontier point toward a sampled manifold target."""

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


def sample_edge_on_manifold(current_manifold, q_a: np.ndarray, q_b: np.ndarray, num: int = 21) -> np.ndarray:
    qa = np.asarray(q_a, dtype=float)
    qb = np.asarray(q_b, dtype=float)

    if is_plane_like(current_manifold):
        return smooth_plane_segment(qa, qb, num=num)

    samples: list[np.ndarray] = [qa.copy()]
    for t in np.linspace(0.0, 1.0, num)[1:-1]:
        guess = (1.0 - float(t)) * qa + float(t) * qb
        projection = project_newton(current_manifold, guess, tol=1e-10, max_iters=80, damping=1.0)
        if projection.success:
            samples.append(np.asarray(projection.x_projected, dtype=float))
    samples.append(qb.copy())
    return np.asarray(samples, dtype=float)


def refine_transition_bisection(
    current_manifold,
    target_manifold,
    q_a: np.ndarray,
    q_b: np.ndarray,
    target_tol: float = 1e-4,
    max_iters: int = 32,
) -> tuple[np.ndarray, bool]:
    """Refine a sign-change edge to a transition point on the current manifold."""

    a = np.asarray(q_a, dtype=float)
    b = np.asarray(q_b, dtype=float)
    fa = target_residual_scalar(target_manifold, a)
    fb = target_residual_scalar(target_manifold, b)

    best_q = a if abs(fa) <= abs(fb) else b
    best_val = fa if abs(fa) <= abs(fb) else fb
    if abs(fa) <= target_tol:
        return a, True
    if abs(fb) <= target_tol:
        return b, True

    for _ in range(max_iters):
        mid_guess = 0.5 * (a + b)
        # Keep each bisection candidate on the current manifold before testing the target residual.
        projection = project_newton(current_manifold, mid_guess, tol=1e-10, max_iters=80, damping=1.0)
        if not projection.success:
            break

        mid = np.asarray(projection.x_projected, dtype=float)
        fm = target_residual_scalar(target_manifold, mid)
        if abs(fm) < abs(best_val):
            best_q = mid
            best_val = fm
        if abs(fm) <= target_tol:
            return mid, True

        if np.sign(fa) != np.sign(fm):
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm

    return np.asarray(best_q, dtype=float), abs(best_val) <= max(10.0 * target_tol, 1e-3)


def refine_intersection_on_both_manifolds(
    manifold_a,
    manifold_b,
    x0: np.ndarray,
    tol: float = 1e-8,
    max_iters: int = 25,
) -> tuple[np.ndarray, bool]:
    """Newton-refine a transition configuration on both adjacent manifolds."""

    x = np.asarray(x0, dtype=float).reshape(-1).copy()
    best_x = x.copy()
    best_norm = float("inf")

    for _ in range(max_iters):
        ra = np.asarray(manifold_a.residual(x), dtype=float).reshape(-1)
        rb = np.asarray(manifold_b.residual(x), dtype=float).reshape(-1)
        # Stacked residual means the candidate must satisfy both active manifolds.
        residual = np.concatenate([ra, rb], axis=0)
        residual_norm = float(np.linalg.norm(residual))

        if residual_norm < best_norm:
            best_norm = residual_norm
            best_x = x.copy()
        if residual_norm <= tol:
            return x, True

        ja = np.asarray(manifold_a.jacobian(x), dtype=float)
        jb = np.asarray(manifold_b.jacobian(x), dtype=float)
        # Stacked Jacobian gives the least-squares correction for the intersection.
        jacobian = np.vstack([ja, jb])

        try:
            delta, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        except np.linalg.LinAlgError:
            break

        if not np.all(np.isfinite(delta)):
            break
        step_norm = float(np.linalg.norm(delta))
        if step_norm <= 1e-14:
            break

        x = x + delta

    return best_x, best_norm <= max(10.0 * tol, 1e-5)


def scan_path_for_transition(current_manifold, target_manifold, path: np.ndarray, target_tol: float = 1e-4) -> np.ndarray:
    """Scan a manifold path for target-residual zeros and refine crossings."""

    pts = np.asarray(path, dtype=float)
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=float)

    residuals = np.asarray([target_residual_scalar(target_manifold, q) for q in pts], dtype=float)
    hits: list[np.ndarray] = []
    for q, residual in zip(pts, residuals):
        if abs(float(residual)) <= target_tol:
            hits.append(np.asarray(q, dtype=float))

    for idx in range(len(pts) - 1):
        fa = float(residuals[idx])
        fb = float(residuals[idx + 1])
        if np.sign(fa) == np.sign(fb):
            continue
        # A sign change suggests the path crossed the neighboring manifold.
        refined, ok = refine_transition_bisection(
            current_manifold=current_manifold,
            target_manifold=target_manifold,
            q_a=pts[idx],
            q_b=pts[idx + 1],
            target_tol=target_tol,
        )
        if ok:
            refined_both, ok_both = refine_intersection_on_both_manifolds(
                manifold_a=current_manifold,
                manifold_b=target_manifold,
                x0=refined,
                tol=1e-8,
                max_iters=25,
            )
            hits.append(np.asarray(refined_both if ok_both else refined, dtype=float))

    return deduplicate_points(hits, tol=1e-4)


def scan_tree_edges_for_transition(
    current_manifold,
    target_manifold,
    explored_edges: list[tuple[np.ndarray, np.ndarray]],
    target_tol: float = 1e-4,
) -> np.ndarray:
    """Scan explored tree edges for transition configurations."""

    hits: list[np.ndarray] = []
    for q_a, q_b in explored_edges:
        edge_path = sample_edge_on_manifold(current_manifold, q_a, q_b, num=29)
        if len(edge_path) < 2:
            continue
        edge_hits = scan_path_for_transition(
            current_manifold=current_manifold,
            target_manifold=target_manifold,
            path=edge_path,
            target_tol=target_tol,
        )
        if len(edge_hits) > 0:
            hits.extend(np.asarray(edge_hits, dtype=float))
    return deduplicate_points(hits, tol=1e-4)


def find_online_switch(
    current_family,
    current_lam: float,
    current_manifold,
    next_manifold,
    q_start: np.ndarray,
    guide_point: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    max_targets: int = 12,
) -> tuple[np.ndarray | None, np.ndarray, object | None]:
    """Explore from a frontier until a useful transition switch is found."""

    frontier: list[np.ndarray] = [np.asarray(q_start, dtype=float).copy()]
    discovered: list[np.ndarray] = []
    all_explored_edges: list[tuple[np.ndarray, np.ndarray]] = []
    best_switch = None
    best_score = float("inf")
    best_chart_count = 0
    last_improvement_round = -1
    guide = np.asarray(guide_point, dtype=float)

    for round_idx in range(max_targets):
        if len(frontier) == 0:
            break
        # kaunch a new exploratory search from frontier towards OMPL sampled target
        source = np.asarray(frontier[round_idx % len(frontier)], dtype=float)
        target = ompl_native_exploration_target(
            manifold=current_manifold,
            q_seed=source,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
        )
        if target is None:
            continue
        if np.linalg.norm(np.asarray(target, dtype=float) - source) <= 1e-6:
            continue
        # gets in return explored local path, edges
        exploration_result = explore_on_manifold_from_frontier(
            manifold=current_manifold,
            x_start=source,
            x_goal=np.asarray(target, dtype=float),
            bounds_min=bounds_min,
            bounds_max=bounds_max,
        )
        best_chart_count = max(best_chart_count, int(getattr(exploration_result, "chart_count", 0)))
        explored_edges = list(getattr(exploration_result, "explored_edges", []))
        all_explored_edges = merge_edges(all_explored_edges, explored_edges)
        hits = scan_tree_edges_for_transition(
            current_manifold=current_manifold,
            target_manifold=next_manifold,
            explored_edges=explored_edges,
            target_tol=1e-4,
        )

        if len(hits) == 0 and len(np.asarray(exploration_result.path, dtype=float)) >= 2:
            hits = scan_path_for_transition(
                current_manifold=current_manifold,
                target_manifold=next_manifold,
                path=np.asarray(exploration_result.path, dtype=float),
                target_tol=1e-4,
            )

        if len(hits) == 0:
            # No transition yet: expand the frontier toward newly explored points.
            frontier = update_frontier_points(
                frontier=frontier,
                new_points=frontier_points_from_result(exploration_result),
                guide_point=guide,
                max_points=48,
            )
            continue

        discovered.extend(np.asarray(hits, dtype=float))
        for hit in np.asarray(hits, dtype=float):
            # Prefer transitions that are reachable, guide-directed, and admissible for the family.
            score = float(np.linalg.norm(np.asarray(q_start, dtype=float) - np.asarray(hit, dtype=float)))
            score += float(np.linalg.norm(np.asarray(hit, dtype=float) - guide))
            score += 0.25 * float(
                current_family.transition_admissibility_cost(current_lam, hit, goal_point=guide)
            )
            if score < best_score:
                best_score = score
                best_switch = np.asarray(hit, dtype=float)
                last_improvement_round = round_idx
        #even after hit, frontier is updated allowing search if needed
        frontier = update_frontier_points(
            frontier=frontier,
            new_points=frontier_points_from_result(exploration_result),
            guide_point=guide,
            max_points=24,
        )
        if best_switch is not None and round_idx >= 4:
            enough_hits = len(discovered) >= 12
            no_recent_improvement = (round_idx - last_improvement_round) >= 3
            if enough_hits or no_recent_improvement:
                break

    if best_switch is None:
        failure_result = exploration_failure_result(
            q_start=np.asarray(q_start, dtype=float),
            explored_edges=all_explored_edges,
            planner_name="online_atlas_explore" if not is_plane_like(current_manifold) else "online_plane_explore",
            message="No transition switch was detected on the explored manifold edges.",
            chart_count=best_chart_count,
        )
        return None, deduplicate_points(discovered, tol=1e-4), failure_result

    exact_result = solve_exact_segment_on_manifold(
        manifold=current_manifold,
        x_start=q_start,
        x_goal=best_switch,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )
    if not exact_result.success:
        if is_plane_like(current_manifold):
            direct_path = smooth_plane_segment(
                np.asarray(q_start, dtype=float),
                np.asarray(best_switch, dtype=float),
                num=120,
            )
            exact_result = LocalPathResult(
                success=True,
                path=np.asarray(direct_path, dtype=float),
                iterations=max(0, len(direct_path) - 1),
                reached_goal=True,
            message="Online plane segment fell back to direct manifold-valid interpolation after OMPL exact solve failed.",
            planner_name="online_plane_segment",
                chart_count=0,
                explored_edges=all_explored_edges,
            )
        else:
            return None, deduplicate_points(discovered, tol=1e-4), exact_result

    exact_result.explored_edges = merge_edges(all_explored_edges, list(getattr(exact_result, "explored_edges", [])))
    exact_result.chart_count = max(int(getattr(exact_result, "chart_count", 0)), best_chart_count)
    return np.asarray(best_switch, dtype=float), deduplicate_points(discovered, tol=1e-4), exact_result


def plan_explicit_ompl_route(families, start_q: np.ndarray, goal_q: np.ndarray) -> ExplicitOmplRoute:
    """Plan an explicit fixed-lambda route through the provided manifold families."""

    lambdas = [float(family.sample_lambdas()[0]) for family in families]
    manifolds = [family.manifold(lam) for family, lam in zip(families, lambdas)]
    if len(families) <= 3:
        bounds_min = np.array([-3.2, -1.8, -0.6], dtype=float)
        bounds_max = np.array([3.2, 1.8, 1.8], dtype=float)
    else:
        bounds_min = np.array([-6.8, -2.8, -0.9], dtype=float)
        bounds_max = np.array([6.8, 2.8, 2.0], dtype=float)

    switch_points: list[np.ndarray] = []
    transition_points_by_interface: list[np.ndarray] = []
    segment_results: list[object | None] = []
    current_start = np.asarray(start_q, dtype=float)

    for idx in range(len(families) - 1):
        current_family = families[idx]
        current_manifold = manifolds[idx]
        next_manifold = manifolds[idx + 1]
        guide_point = guide_point_for_step(manifolds, idx, goal_q)

        switch_point, transition_points, segment_result = find_online_switch(
            current_family=current_family,
            current_lam=lambdas[idx],
            current_manifold=current_manifold,
            next_manifold=next_manifold,
            q_start=current_start,
            guide_point=guide_point,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
        )
        transition_points_by_interface.append(transition_points)
        segment_results.append(segment_result)

        if switch_point is None or segment_result is None:
            return build_route_result(
                families=families,
                lambdas=lambdas,
                manifolds=manifolds,
                start_q=start_q,
                goal_q=goal_q,
                message=f"Online transition detection failed to find a switch while moving on manifold {idx + 1}.",
                success=False,
                switch_points=switch_points,
                transition_points_by_interface=transition_points_by_interface,
                segment_results=segment_results,
            )

        switch_points.append(np.asarray(switch_point, dtype=float))
        current_start = np.asarray(switch_point, dtype=float)

    final_result = solve_exact_segment_on_manifold(
        manifold=manifolds[-1],
        x_start=current_start,
        x_goal=goal_q,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )
    segment_results.append(final_result)
    if not final_result.success:
        return build_route_result(
            families=families,
            lambdas=lambdas,
            manifolds=manifolds,
            start_q=start_q,
            goal_q=goal_q,
            message=f"Final manifold segment failed: {final_result.message}",
            success=False,
            switch_points=switch_points,
            transition_points_by_interface=transition_points_by_interface,
            segment_results=segment_results,
        )

    return build_route_result(
        families=families,
        lambdas=lambdas,
        manifolds=manifolds,
        start_q=start_q,
        goal_q=goal_q,
        message="Online transition detection succeeded on the predefined fixed multimodal sequence.",
        success=True,
        switch_points=switch_points,
        transition_points_by_interface=transition_points_by_interface,
        segment_results=segment_results,
    )


def plot_manifold(ax, manifold, color="lightgray", alpha=0.13):
    base = unwrap_manifold(manifold)
    if isinstance(base, SphereManifold):
        u = np.linspace(0.0, 2.0 * np.pi, 36)
        v = np.linspace(0.0, np.pi, 20)
        x = base.center[0] + base.radius * np.outer(np.cos(u), np.sin(v))
        y = base.center[1] + base.radius * np.outer(np.sin(u), np.sin(v))
        z = base.center[2] + base.radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(x, y, z, color=color, alpha=alpha, rstride=2, cstride=2, linewidth=0.55)


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=float)
    if abs(n[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    u = np.cross(n, ref)
    u = u / max(np.linalg.norm(u), 1e-15)
    v = np.cross(n, u)
    v = v / max(np.linalg.norm(v), 1e-15)
    return u, v


def plane_rectangle_patch(plane_like_manifold) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plane = unwrap_manifold(plane_like_manifold)
    normal = np.asarray(plane.normal, dtype=float)
    center = np.asarray(plane.point, dtype=float)
    u, v = _plane_basis(normal)

    def extent(direction: np.ndarray) -> float:
        samples = np.linspace(0.0, 3.5, 281)
        valid = 0.0
        for dist in samples:
            p_pos = center + float(dist) * direction
            p_neg = center - float(dist) * direction
            if plane_like_manifold.within_bounds(p_pos) and plane_like_manifold.within_bounds(p_neg):
                valid = float(dist)
            else:
                break
        return max(valid, 1e-3)

    half_u = extent(u)
    half_v = extent(v)
    corners = np.asarray(
        [
            center - half_u * u - half_v * v,
            center + half_u * u - half_v * v,
            center + half_u * u + half_v * v,
            center - half_u * u + half_v * v,
        ],
        dtype=float,
    )
    return corners, u, v


def plane_connector_patch(plane: PlaneManifold, q_a: np.ndarray | None, q_b: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = np.asarray(plane.normal, dtype=float)
    if q_a is None or q_b is None:
        u, v = _plane_basis(normal)
        center = np.asarray(plane.point, dtype=float)
        half_len = 1.6
        half_wid = 0.7
    else:
        qa = np.asarray(q_a, dtype=float)
        qb = np.asarray(q_b, dtype=float)
        center = 0.5 * (qa + qb)
        direction = qb - qa
        direction = direction - float(np.dot(direction, normal)) * normal
        if np.linalg.norm(direction) <= 1e-12:
            u, v = _plane_basis(normal)
            half_len = 1.6
            half_wid = 0.7
        else:
            u = direction / np.linalg.norm(direction)
            v = np.cross(normal, u)
            v = v / max(np.linalg.norm(v), 1e-15)
            span = float(np.linalg.norm(qb - qa))
            # Extend only slightly beyond the detected switches so the connector
            # reads as a local transfer patch rather than a full plane cutting
            # deeply through either sphere.
            half_len = 0.5 * span + 0.12
            half_wid = min(1.05, max(0.60, 0.18 * span))

    corners = np.asarray(
        [
            center - half_len * u - half_wid * v,
            center + half_len * u - half_wid * v,
            center + half_len * u + half_wid * v,
            center - half_len * u + half_wid * v,
        ],
        dtype=float,
    )
    return corners, u, v


def plot_plane_connector(ax, plane: PlaneManifold, q_a: np.ndarray | None, q_b: np.ndarray | None, color="lightgray", alpha=0.13):
    corners, _, _ = plane_connector_patch(plane, q_a, q_b)
    xs = np.asarray([[corners[0, 0], corners[1, 0]], [corners[3, 0], corners[2, 0]]], dtype=float)
    ys = np.asarray([[corners[0, 1], corners[1, 1]], [corners[3, 1], corners[2, 1]]], dtype=float)
    zs = np.asarray([[corners[0, 2], corners[1, 2]], [corners[3, 2], corners[2, 2]]], dtype=float)
    ax.plot_surface(xs, ys, zs, color=color, alpha=alpha, linewidth=0.0, shade=False)


def project_point_to_plane(plane: PlaneManifold, q: np.ndarray) -> np.ndarray:
    normal = np.asarray(plane.normal, dtype=float)
    point = np.asarray(plane.point, dtype=float)
    qq = np.asarray(q, dtype=float)
    signed = float(np.dot(normal, qq - point))
    return qq - signed * normal


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


SEGMENT_DEBUG_COLORS = ["#4fc3f7", "#ff8a65", "#81c784", "#ba68c8", "#ffd54f"]
SEGMENT_CHART_COLORS = ["#ef6c00", "#8e24aa", "#2e7d32", "#6d4c41", "#00838f"]
INTERFACE_COLORS = ["#ff7043", "#42a5f5", "#66bb6a", "#ab47bc", "#ffa726"]


def show_pyvista_scene(families, colors, result: ExplicitOmplRoute, start_q: np.ndarray, goal_q: np.ndarray):
    if pv is None:
        return False

    plotter = pv.Plotter(window_size=(1280, 800))
    plotter.add_text("Example 64: online transition detection on a fixed manifold sequence", font_size=12)

    actor_groups = {
        "Manifolds": [],
        "Transitions": [],
        "Charts": [],
        "Exploration": [],
        "Path": [],
        "Debug": [],
        "StartGoal": [],
    }

    for idx, fam in enumerate(families):
        for lam in fam.sample_lambdas():
            manifold = fam.manifold(lam)
            if is_plane_like(manifold):
                if isinstance(manifold, MaskedManifold):
                    corners, _, _ = plane_rectangle_patch(manifold)
                else:
                    if idx == 0:
                        patch_a = np.asarray(start_q, dtype=float)
                    elif idx - 1 < len(result.switch_points):
                        patch_a = np.asarray(result.switch_points[idx - 1], dtype=float)
                    else:
                        patch_a = project_point_to_plane(manifold, start_q)

                    if idx < len(result.switch_points):
                        patch_b = np.asarray(result.switch_points[idx], dtype=float)
                    elif idx == len(families) - 1:
                        patch_b = np.asarray(goal_q, dtype=float)
                    else:
                        patch_b = project_point_to_plane(manifold, goal_q)
                    corners, _, _ = plane_connector_patch(unwrap_manifold(manifold), patch_a, patch_b)
                faces = np.hstack([[4, 0, 1, 2, 3]])
                patch = pv.PolyData(corners, faces)
                actor = plotter.add_mesh(
                    patch,
                    color=colors.get(fam.name, "#999999"),
                    opacity=0.14,
                    show_edges=False,
                    smooth_shading=False,
                    name=fam.name,
                )
            else:
                actor = add_manifold(plotter, manifold, color=colors.get(fam.name, "#999999"), opacity=0.10)
            if actor is not None:
                actor_groups["Manifolds"].append(actor)

    for idx, points in enumerate(result.transition_points_by_interface):
        if len(points) == 0:
            continue
        actor = add_points(
            plotter,
            points,
            color=INTERFACE_COLORS[idx % len(INTERFACE_COLORS)],
            size=8.0,
            label=f"interface {idx + 1} transition points",
        )
        if actor is not None:
            actor_groups["Transitions"].append(actor)

    for idx, chart_points in enumerate(result.chart_centers_by_segment):
        if len(chart_points) == 0:
            continue
        actor = add_points(
            plotter,
            chart_points,
            color=SEGMENT_CHART_COLORS[idx % len(SEGMENT_CHART_COLORS)],
            size=10.0,
            label=f"segment {idx + 1} chart centers",
        )
        if actor is not None:
            actor_groups["Charts"].append(actor)

    for idx, edges in enumerate(result.explored_edges_by_segment):
        graph = build_segment_polydata(edges)
        if graph is not None:
            actor = plotter.add_mesh(
                graph,
                color=SEGMENT_DEBUG_COLORS[idx % len(SEGMENT_DEBUG_COLORS)],
                line_width=2.0,
                opacity=0.45,
                label=f"segment {idx + 1} explored edges",
            )
            if actor is not None:
                actor_groups["Exploration"].append(actor)

    for idx, points in enumerate(result.explored_points_by_segment):
        if len(points) == 0:
            continue
        actor = add_points(
            plotter,
            points,
            color=SEGMENT_DEBUG_COLORS[idx % len(SEGMENT_DEBUG_COLORS)],
            size=5.0,
            label=f"segment {idx + 1} explored vertices",
        )
        if actor is not None:
            actor_groups["Exploration"].append(actor)

    if len(result.path) >= 2:
        # The path is already densely sampled on each manifold segment, so draw it
        # as a polyline instead of re-splining it in PyVista. That avoids visible
        # overshoot near the manifold switches.
        polyline = pv.lines_from_points(np.asarray(result.path, dtype=float))
        actor = plotter.add_mesh(polyline, color="#1565c0", line_width=7.0, label="display path")
        if actor is not None:
            actor_groups["Path"].append(actor)

    if len(result.raw_path) >= 2:
        raw_polyline = pv.lines_from_points(np.asarray(result.raw_path, dtype=float))
        actor = plotter.add_mesh(raw_polyline, color="#90a4ae", line_width=3.0, opacity=0.7, label="raw OMPL solver path")
        if actor is not None:
            actor_groups["Debug"].append(actor)

    if len(result.switch_points) > 0:
        actor = add_points(plotter, np.asarray(result.switch_points, dtype=float), color="#333333", size=12.0, label="switch points")
        if actor is not None:
            actor_groups["Path"].append(actor)

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

    for idx, label in enumerate(["Manifolds", "Transitions", "Charts", "Exploration", "Path", "Debug", "StartGoal"]):
        y = 10 + idx * 42
        plotter.add_text(label, position=(55, y + 7), font_size=10, color="black")
        plotter.add_checkbox_button_widget(
            callback=lambda state, name=label: set_visibility(name, state),
            value=(label != "Debug"),
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


def main():
    np.random.seed(41)
    ou.RNG.setSeed(41)
    ou.setLogLevel(ou.LOG_ERROR)
    families, start_q, goal_q = build_scene()
    result = plan_explicit_ompl_route(families, start_q, goal_q)

    print("\nExample 64")
    print(f"success = {result.success}")
    print(f"message = {result.message}")
    print(f"route = {result.route_string}")
    print(f"switch_count = {len(result.switch_points)}")
    if len(result.switch_points) > 0:
        print(f"switch_points = {[np.round(q, 4).tolist() for q in result.switch_points]}")
    for idx, points in enumerate(result.transition_points_by_interface):
        print(f"interface_{idx + 1}_hits = {len(points)}")
    for idx, points in enumerate(result.explored_points_by_segment):
        edge_count = len(result.explored_edges_by_segment[idx]) if idx < len(result.explored_edges_by_segment) else 0
        chart_count = int(getattr(result.segment_results[idx], 'chart_count', 0)) if idx < len(result.segment_results) and result.segment_results[idx] is not None else 0
        print(f"segment_{idx + 1}_explored_vertices = {len(points)}")
        print(f"segment_{idx + 1}_explored_edges = {edge_count}")
        print(f"segment_{idx + 1}_charts = {chart_count}")

    colors = {
        "left_support_3d": "#c58b4c",
        "transfer_plane_3d": "#7fa7c6",
        "right_support_3d": "#c58b4c",
    }

    if pyvista_available():
        shown = show_pyvista_scene(families, colors, result, start_q, goal_q)
        if shown:
            return

    fig = plt.figure(figsize=(11.5, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    for idx, fam in enumerate(families):
        for lam in fam.sample_lambdas():
            manifold = fam.manifold(lam)
            if is_plane_like(manifold):
                if isinstance(manifold, MaskedManifold):
                    corners, _, _ = plane_rectangle_patch(manifold)
                    xs = np.asarray([[corners[0, 0], corners[1, 0]], [corners[3, 0], corners[2, 0]]], dtype=float)
                    ys = np.asarray([[corners[0, 1], corners[1, 1]], [corners[3, 1], corners[2, 1]]], dtype=float)
                    zs = np.asarray([[corners[0, 2], corners[1, 2]], [corners[3, 2], corners[2, 2]]], dtype=float)
                    ax.plot_surface(xs, ys, zs, color=colors.get(fam.name, "#999999"), alpha=0.11, linewidth=0.0, shade=False)
                else:
                    if idx == 0:
                        patch_a = np.asarray(start_q, dtype=float)
                    elif idx - 1 < len(result.switch_points):
                        patch_a = np.asarray(result.switch_points[idx - 1], dtype=float)
                    else:
                        patch_a = project_point_to_plane(manifold, start_q)
                    if idx < len(result.switch_points):
                        patch_b = np.asarray(result.switch_points[idx], dtype=float)
                    elif idx == len(families) - 1:
                        patch_b = np.asarray(goal_q, dtype=float)
                    else:
                        patch_b = project_point_to_plane(manifold, goal_q)
                    plot_plane_connector(
                        ax,
                        unwrap_manifold(manifold),
                        patch_a,
                        patch_b,
                        color=colors.get(fam.name, "#999999"),
                        alpha=0.11,
                    )
            else:
                plot_manifold(ax, manifold, color=colors.get(fam.name, "#999999"), alpha=0.11)

    for idx, points in enumerate(result.transition_points_by_interface):
        if len(points) > 0:
            ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                s=12,
                color=INTERFACE_COLORS[idx % len(INTERFACE_COLORS)],
                alpha=0.65,
                label=f"interface {idx + 1} transition points",
            )
    for idx, points in enumerate(result.chart_centers_by_segment):
        if len(points) > 0:
            ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                s=22,
                color=SEGMENT_CHART_COLORS[idx % len(SEGMENT_CHART_COLORS)],
                alpha=0.95,
                label=f"segment {idx + 1} chart centers",
            )
    for idx, edges in enumerate(result.explored_edges_by_segment):
        for q_a, q_b in edges:
            seg = np.asarray([q_a, q_b], dtype=float)
            ax.plot(
                seg[:, 0], seg[:, 1], seg[:, 2],
                color=SEGMENT_DEBUG_COLORS[idx % len(SEGMENT_DEBUG_COLORS)],
                linewidth=1.0,
                alpha=0.35,
            )
    for idx, points in enumerate(result.explored_points_by_segment):
        if len(points) > 0:
            ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                s=8,
                color=SEGMENT_DEBUG_COLORS[idx % len(SEGMENT_DEBUG_COLORS)],
                alpha=0.35,
                label=f"segment {idx + 1} explored vertices",
            )

    if len(result.path) >= 2:
        ax.plot(result.path[:, 0], result.path[:, 1], result.path[:, 2], color="#1565c0", linewidth=3.0, label="display path")
    if len(result.raw_path) >= 2:
        ax.plot(result.raw_path[:, 0], result.raw_path[:, 1], result.raw_path[:, 2], color="#90a4ae", linewidth=1.5, alpha=0.7, label="raw OMPL solver path")

    if len(result.switch_points) > 0:
        switch_arr = np.asarray(result.switch_points, dtype=float)
        ax.scatter(switch_arr[:, 0], switch_arr[:, 1], switch_arr[:, 2], s=60, marker="x", color="#333333", label="switch points")

    ax.scatter(start_q[0], start_q[1], start_q[2], s=90, marker="s", color="black", label="start")
    ax.scatter(goal_q[0], goal_q[1], goal_q[2], s=130, marker="*", color="gold", edgecolor="black", label="goal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Example 64: online transition detection on a fixed sphere -> plane -> sphere sequence")
    ax.view_init(elev=24, azim=-58)
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
