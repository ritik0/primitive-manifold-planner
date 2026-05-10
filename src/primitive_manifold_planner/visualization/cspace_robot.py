from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


STAGE_COLORS = {
    "left": "#f97316",
    "plane": "#2563eb",
    "right": "#16a34a",
}


def _pyvista():
    try:
        import pyvista as pv

        return pv
    except Exception:
        return None


def _finite_theta_bounds(theta_path: np.ndarray, padding: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta_path, dtype=float)
    if theta.ndim != 2 or theta.shape[1] != 3 or len(theta) == 0:
        return -np.pi * np.ones(3, dtype=float), np.pi * np.ones(3, dtype=float)
    lower = np.min(theta, axis=0) - float(padding)
    upper = np.max(theta, axis=0) + float(padding)
    lower = np.maximum(lower, -np.pi)
    upper = np.minimum(upper, np.pi)
    too_thin = (upper - lower) < 0.35
    lower[too_thin] = np.maximum(theta[0, too_thin] - 0.35, -np.pi)
    upper[too_thin] = np.minimum(theta[0, too_thin] + 0.35, np.pi)
    return lower, upper


def _residual_grid(manifold, axes: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    a0, a1, a2 = axes
    residuals = np.empty((len(a0), len(a1), len(a2)), dtype=float)
    for i, q0 in enumerate(a0):
        for j, q1 in enumerate(a1):
            for k, q2 in enumerate(a2):
                q = np.asarray([q0, q1, q2], dtype=float)
                residuals[i, j, k] = float(np.ravel(manifold.residual(q))[0])
    return residuals


def _add_implicit_surface(plotter, pv, manifold, axes, *, color: str, opacity: float, label: str) -> int:
    a0, a1, a2 = axes
    x, y, z = np.meshgrid(a0, a1, a2, indexing="ij")
    grid = pv.StructuredGrid(x, y, z)
    grid["residual"] = _residual_grid(manifold, axes).ravel(order="F")
    surface = grid.contour(isosurfaces=[0.0], scalars="residual")
    if surface.n_points == 0:
        return 0
    plotter.add_mesh(surface, color=color, opacity=opacity, smooth_shading=True, label=label)
    return int(surface.n_points)


def _polyline_from_points(pv, points: np.ndarray):
    arr = np.asarray(points, dtype=float)
    poly = pv.PolyData(arr)
    if len(arr) >= 2:
        lines = np.hstack([[len(arr)], np.arange(len(arr), dtype=np.int64)])
        poly.lines = lines
    return poly


def _stage_segments(theta_path: np.ndarray, labels: list[str]) -> list[tuple[str, np.ndarray]]:
    theta = np.asarray(theta_path, dtype=float)
    if len(theta) == 0 or len(labels) != len(theta):
        return []
    segments: list[tuple[str, np.ndarray]] = []
    start = 0
    current = str(labels[0])
    for idx, label in enumerate(labels[1:], start=1):
        if str(label) == current:
            continue
        segments.append((current, theta[start : idx + 1]))
        start = idx
        current = str(label)
    segments.append((current, theta[start:]))
    return segments


def _add_marker(plotter, pv, point: np.ndarray, *, color: str, radius: float, label: str) -> None:
    q = np.asarray(point, dtype=float).reshape(3)
    plotter.add_mesh(pv.Sphere(radius=radius, center=q), color=color, label=label)


def show_cspace_robot_planning(
    *,
    result,
    manifolds: Mapping[str, object],
    cspace_audit: Mapping[str, object] | None = None,
    grid_res: int = 45,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> Path | None:
    """Show the FK-pulled-back constraint surfaces in theta-space.

    Axes are theta0, theta1, theta2. The red/dark route is the dense theta path
    itself; FK is used only by the constraints that define the implicit
    surfaces, not as the plotted route coordinates.
    """
    pv = _pyvista()
    theta_path = np.asarray(getattr(result, "dense_joint_path", np.zeros((0, 3), dtype=float)), dtype=float)
    labels = list(getattr(result, "dense_joint_path_stage_labels", []))
    if pv is None:
        return _save_matplotlib_cspace_fallback(theta_path, labels, cspace_audit, output_dir)
    if len(theta_path) == 0:
        raise RuntimeError("Cannot show C-space view without result.dense_joint_path.")
    if len(labels) != len(theta_path):
        raise RuntimeError("C-space view requires one stage label per dense theta waypoint.")

    grid_res = int(max(12, min(int(grid_res), 85)))
    lower, upper = _finite_theta_bounds(theta_path, padding=0.45)
    axes = tuple(np.linspace(lower[i], upper[i], grid_res) for i in range(3))

    screenshot_path: Path | None = None
    if output_dir is not None:
        screenshot_path = Path(output_dir) / "cspace_environment.png"
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)

    plotter = pv.Plotter(off_screen=not bool(show))
    plotter.set_background("white")

    surface_counts = {
        "left": _add_implicit_surface(plotter, pv, manifolds["left"], axes, color="#f97316", opacity=0.18, label="M_left"),
        "plane": _add_implicit_surface(plotter, pv, manifolds["plane"], axes, color="#2563eb", opacity=0.16, label="M_plane"),
        "right": _add_implicit_surface(plotter, pv, manifolds["right"], axes, color="#16a34a", opacity=0.18, label="M_right"),
    }

    full_line = _polyline_from_points(pv, theta_path)
    plotter.add_mesh(full_line, color="#111827", line_width=4, label="dense theta path")
    for stage, segment in _stage_segments(theta_path, labels):
        if len(segment) < 2:
            continue
        plotter.add_mesh(
            _polyline_from_points(pv, segment),
            color=STAGE_COLORS.get(stage, "#dc2626"),
            line_width=8,
            label=f"{stage} segment",
        )

    _add_marker(plotter, pv, theta_path[0], color="black", radius=0.045, label="start theta")
    _add_marker(plotter, pv, theta_path[-1], color="#facc15", radius=0.055, label="goal theta")

    audit = cspace_audit or {}
    for key, color, label in (
        ("selected_left_plane_transition_index", "#ef4444", "left-plane transition theta"),
        ("selected_plane_right_transition_index", "#14b8a6", "plane-right transition theta"),
    ):
        idx = int(audit.get(key, getattr(result, key, -1)))
        if 0 <= idx < len(theta_path):
            _add_marker(plotter, pv, theta_path[idx], color=color, radius=0.065, label=label)

    plotter.add_axes(
        xlabel="theta0 [rad]",
        ylabel="theta1 [rad]",
        zlabel="theta2 [rad]",
        line_width=2,
        labels_off=False,
    )
    plotter.add_text(
        "C-space view: axes are theta0, theta1, theta2\n"
        "Surfaces are FK-pulled-back constraints; route is dense theta path.",
        position="upper_left",
        font_size=10,
        color="black",
    )
    plotter.add_legend(size=(0.22, 0.22), bcolor="white", border=True)
    plotter.camera_position = "iso"

    print("cspace_surface_points_left = " + str(surface_counts["left"]))
    print("cspace_surface_points_plane = " + str(surface_counts["plane"]))
    print("cspace_surface_points_right = " + str(surface_counts["right"]))

    if show:
        plotter.show(screenshot=str(screenshot_path) if screenshot_path is not None else None)
    elif screenshot_path is not None:
        plotter.show(screenshot=str(screenshot_path))
        plotter.close()
    else:
        plotter.close()
    return screenshot_path


def _save_matplotlib_cspace_fallback(
    theta_path: np.ndarray,
    labels: list[str],
    cspace_audit: Mapping[str, object] | None,
    output_dir: str | Path | None,
) -> Path | None:
    if output_dir is None:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    output = Path(output_dir) / "cspace_environment.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(theta_path[:, 0], theta_path[:, 1], theta_path[:, 2], color="black", linewidth=1.0)
    for stage, segment in _stage_segments(theta_path, labels):
        if len(segment) >= 2:
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=STAGE_COLORS.get(stage, "#dc2626"), linewidth=2.5)
    if len(theta_path) > 0:
        ax.scatter(theta_path[0, 0], theta_path[0, 1], theta_path[0, 2], color="black", s=35)
        ax.scatter(theta_path[-1, 0], theta_path[-1, 1], theta_path[-1, 2], color="gold", s=45)
    audit = cspace_audit or {}
    for key, color in (
        ("selected_left_plane_transition_index", "#ef4444"),
        ("selected_plane_right_transition_index", "#14b8a6"),
    ):
        idx = int(audit.get(key, -1))
        if 0 <= idx < len(theta_path):
            ax.scatter(theta_path[idx, 0], theta_path[idx, 1], theta_path[idx, 2], color=color, s=55)
    ax.set_xlabel("theta0 [rad]")
    ax.set_ylabel("theta1 [rad]")
    ax.set_zlabel("theta2 [rad]")
    ax.set_title("C-space dense theta path")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
