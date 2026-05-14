from __future__ import annotations

from typing import Iterable, Optional
import numpy as np

from primitive_manifold_planner.manifolds import MaskedManifold, PlaneManifold, SphereManifold

try:
    import pyvista as pv
except Exception:  # pragma: no cover - optional dependency
    pv = None


def pyvista_available() -> bool:
    return pv is not None


def add_manifold(plotter, manifold, color: str = "lightgray", opacity: float = 0.18, label: Optional[str] = None):
    if pv is None:  # pragma: no cover - optional dependency
        raise RuntimeError("PyVista is not available.")

    mesh = manifold_to_mesh(manifold)
    actor = plotter.add_mesh(
        mesh,
        color=color,
        opacity=opacity,
        show_edges=False,
        smooth_shading=False,
        name=label,
    )
    return actor


def add_path(plotter, path: np.ndarray, color: str = "#1565c0", width: float = 5.0, label: Optional[str] = None):
    if pv is None:  # pragma: no cover - optional dependency
        raise RuntimeError("PyVista is not available.")
    pts = np.asarray(path, dtype=float)
    if len(pts) < 2:
        return None
    spline = pv.Spline(pts, n_points=max(len(pts) * 4, 2))
    return plotter.add_mesh(spline, color=color, line_width=width, label=label)


def add_points(
    plotter,
    points: np.ndarray,
    color: str = "black",
    size: float = 12.0,
    render_points_as_spheres: bool = True,
    label: Optional[str] = None,
):
    if pv is None:  # pragma: no cover - optional dependency
        raise RuntimeError("PyVista is not available.")
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    pdata = pv.PolyData(pts)
    return plotter.add_mesh(
        pdata,
        color=color,
        point_size=size,
        render_points_as_spheres=render_points_as_spheres,
        label=label,
    )


def add_transition_markers(plotter, points: Iterable[np.ndarray], color: str = "black", size: float = 14.0):
    pts = [np.asarray(p, dtype=float).copy() for p in points]
    if len(pts) == 0:
        return None
    return add_points(plotter, np.asarray(pts), color=color, size=size, render_points_as_spheres=True)


def manifold_to_mesh(manifold):
    if pv is None:  # pragma: no cover - optional dependency
        raise RuntimeError("PyVista is not available.")

    if isinstance(manifold, SphereManifold):
        return pv.Sphere(
            radius=float(manifold.radius),
            center=tuple(np.asarray(manifold.center, dtype=float)),
            theta_resolution=48,
            phi_resolution=36,
        )

    if isinstance(manifold, PlaneManifold):
        center = np.asarray(manifold.point, dtype=float)
        normal = np.asarray(manifold.normal, dtype=float)
        return pv.Plane(
            center=tuple(center),
            direction=tuple(normal),
            i_size=4.0,
            j_size=2.2,
            i_resolution=1,
            j_resolution=1,
        )

    if isinstance(manifold, MaskedManifold):
        base = manifold.base_manifold
        if isinstance(base, SphereManifold):
            pts = []
            u = np.linspace(0.0, 2.0 * np.pi, 72)
            v = np.linspace(0.0, np.pi, 36)
            for uu in u:
                for vv in v:
                    p = np.asarray(base.center, dtype=float) + float(base.radius) * np.array(
                        [np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv)],
                        dtype=float,
                    )
                    if manifold.within_bounds(p):
                        pts.append(p)
            pdata = pv.PolyData(np.asarray(pts, dtype=float))
            return pdata.delaunay_3d(alpha=2.0).extract_surface().triangulate()

    raise TypeError(f"Unsupported manifold type for PyVista visualization: {type(manifold).__name__}")


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
