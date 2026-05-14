from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sys
from typing import Mapping

import numpy as np


STAGE_COLORS = {
    "left": "#f97316",
    "plane": "#2563eb",
    "family": "#2563eb",
    "right": "#16a34a",
}


def _pyvista():
    try:
        import pyvista as pv

        return pv
    except Exception:
        return None


def _configure_vtk_warning_display(suppress: bool, log_path: str | Path | None = None) -> tuple[str, str | None]:
    """Redirect and disable VTK's global warning display for noisy OpenGL drivers."""

    try:
        import vtk
    except Exception:
        return "unavailable", None
    try:
        vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF if bool(suppress) else vtk.vtkLogger.VERBOSITY_INFO)
    except Exception:
        pass
    if bool(suppress):
        use_devnull = log_path is None or str(log_path).strip().lower() in {"null", "none", "devnull", "nul"}
        output_path = None if use_devnull else Path(log_path)
        try:
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if not output_path.exists():
                    output_path.write_text("VTK warnings redirected from C-space visualization.\n", encoding="utf-8")
                output_window = vtk.vtkFileOutputWindow()
                output_window.SetFileName(str(output_path))
                vtk.vtkOutputWindow.SetInstance(output_window)
            vtk.vtkObject.GlobalWarningDisplayOff()
            return "disabled", None if output_path is None else str(output_path)
        except Exception:
            vtk.vtkObject.GlobalWarningDisplayOff()
            return "disabled", None
    vtk.vtkObject.GlobalWarningDisplayOn()
    return "enabled", None


@contextmanager
def suppress_native_output(enabled: bool, log_path: str | Path | None):
    """Temporarily redirect OS-level stdout/stderr from native OpenGL/VTK code."""

    if not bool(enabled):
        yield None
        return
    use_devnull = log_path is None or str(log_path).strip().lower() in {"null", "none", "devnull", "nul"}
    output_path = None if use_devnull else Path(log_path)
    saved_stdout_fd = None
    saved_stderr_fd = None
    target_file = None
    try:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        if output_path is None:
            target_file = open(os.devnull, "ab", buffering=0)
            yielded_path = os.devnull
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            target_file = output_path.open("ab", buffering=0)
            target_file.write(b"\n--- Native stdout/stderr redirected during PyVista render ---\n")
            yielded_path = str(output_path)
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        os.dup2(target_file.fileno(), 1)
        os.dup2(target_file.fileno(), 2)
        yield yielded_path
    except Exception:
        yield None
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        if saved_stdout_fd is not None:
            try:
                os.dup2(saved_stdout_fd, 1)
            finally:
                try:
                    os.close(saved_stdout_fd)
                except Exception:
                    pass
        if saved_stderr_fd is not None:
            try:
                os.dup2(saved_stderr_fd, 2)
            finally:
                try:
                    os.close(saved_stderr_fd)
                except Exception:
                    pass
        if target_file is not None:
            try:
                target_file.close()
            except Exception:
                pass


def _finite_theta_bounds(theta_path: np.ndarray, extra_points: list[np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Choose route-aware theta0/theta1/theta2 bounds for surface extraction."""

    groups: list[np.ndarray] = []
    theta = np.asarray(theta_path, dtype=float)
    if theta.ndim == 2 and theta.shape[1] == 3 and len(theta) > 0:
        groups.append(theta)
    for point in extra_points or []:
        arr = np.asarray(point, dtype=float)
        if arr.shape == (3,) and np.all(np.isfinite(arr)):
            groups.append(arr.reshape(1, 3))
        elif arr.ndim == 2 and arr.shape[1] == 3 and len(arr) > 0:
            finite = arr[np.all(np.isfinite(arr), axis=1)]
            if len(finite) > 0:
                groups.append(finite)
    if len(groups) == 0:
        return -np.pi * np.ones(3, dtype=float), np.pi * np.ones(3, dtype=float)
    points = np.vstack(groups)
    base_lower = np.min(points, axis=0)
    base_upper = np.max(points, axis=0)
    span = np.maximum(base_upper - base_lower, 1.0e-9)
    margin = np.maximum(0.25 * span, 0.20)
    lower = base_lower - margin
    upper = base_upper + margin
    lower = np.maximum(lower, -np.pi)
    upper = np.minimum(upper, np.pi)
    too_thin = (upper - lower) < 0.40
    center = 0.5 * (base_lower + base_upper)
    lower[too_thin] = np.maximum(center[too_thin] - 0.20, -np.pi)
    upper[too_thin] = np.minimum(center[too_thin] + 0.20, np.pi)
    return lower, upper


def _residual_grid(manifold, axes: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Sample one implicit C-space surface over theta0/theta1/theta2 axes."""

    a0, a1, a2 = axes
    residuals = np.empty((len(a0), len(a1), len(a2)), dtype=float)
    for i, q0 in enumerate(a0):
        for j, q1 in enumerate(a1):
            for k, q2 in enumerate(a2):
                q = np.asarray([q0, q1, q2], dtype=float)
                if hasattr(manifold, "within_bounds") and not bool(manifold.within_bounds(q, tol=2.0e-3)):
                    residuals[i, j, k] = np.nan
                else:
                    residuals[i, j, k] = float(np.ravel(manifold.residual(q))[0])
    return residuals


def _add_implicit_surface(
    plotter,
    pv,
    manifold,
    axes,
    *,
    color: str,
    opacity: float,
    label: str,
    lightweight: bool,
    surface_style: str,
    safe_render: bool,
    exact_isosurface: bool,
    surface_mode: str,
    surface_opacity: float,
    smooth_surfaces: bool,
) -> int:
    """Draw the zero residual sheet for one FK-pulled-back constraint."""

    a0, a1, a2 = axes
    x, y, z = np.meshgrid(a0, a1, a2, indexing="ij")
    grid = pv.StructuredGrid(x, y, z)
    grid["residual"] = _residual_grid(manifold, axes).ravel(order="F")
    surface = grid.contour(isosurfaces=[0.0], scalars="residual")
    if surface.n_points == 0:
        return 0
    style = str(surface_style).lower()
    if bool(exact_isosurface):
        if bool(smooth_surfaces):
            try:
                surface = surface.smooth(
                    n_iter=15,
                    relaxation_factor=0.025,
                    feature_smoothing=False,
                    boundary_smoothing=True,
                )
            except Exception:
                pass
        solid_safe = str(surface_mode).lower() == "solid-safe"
        mesh_kwargs = {
            "color": color,
            "opacity": 1.0 if solid_safe else float(surface_opacity),
            "smooth_shading": False,
            "show_edges": False,
            "render_lines_as_tubes": False,
            "label": label,
        }
        plotter.add_mesh(surface, **mesh_kwargs)
        return int(surface.n_points)
    if style in {"points", "points-outline"}:
        points = np.asarray(surface.points, dtype=float)
        max_points = 450 if bool(lightweight) else 1400
        stride = max(1, int(np.ceil(len(points) / max(max_points, 1))))
        cloud_points = points[::stride]
        cloud = pv.PolyData(cloud_points)
        plotter.add_mesh(
            cloud,
            color=color,
            opacity=1.0 if bool(safe_render) or style == "points-outline" else float(opacity),
            point_size=2.7 if bool(lightweight) else 3.6,
            render_points_as_spheres=False,
            label=label,
        )
        count = int(len(cloud.points))
        label_lower = label.lower()
        if style == "points-outline" and ("left" in label_lower or "right" in label_lower):
            count += _add_pca_envelope_loops(
                plotter,
                pv,
                cloud_points,
                color=color,
                lightweight=bool(lightweight),
            )
        elif style == "points-outline" and ("plane" in label_lower or "family" in label_lower):
            count += _add_point_cloud_box_outline(plotter, pv, cloud_points, color=color, lightweight=bool(lightweight))
        return count
    if style == "contour":
        count = 0
        labeled = False
        normals = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        center = np.asarray([(axis[0] + axis[-1]) * 0.5 for axis in axes], dtype=float)
        for axis_idx, normal in enumerate(normals):
            values = np.linspace(axes[axis_idx][1], axes[axis_idx][-2], 4 if bool(lightweight) else 6)
            for value in values:
                origin = center.copy()
                origin[axis_idx] = float(value)
                try:
                    section = grid.slice(normal=normal, origin=origin)
                    curve = section.contour(isosurfaces=[0.0], scalars="residual")
                except Exception:
                    continue
                if curve.n_points == 0:
                    continue
                plotter.add_mesh(
                    curve,
                    color=color,
                    opacity=1.0 if bool(safe_render) else float(opacity),
                    line_width=1.4 if bool(lightweight) else 1.8,
                    render_lines_as_tubes=False,
                    label=label if not labeled else None,
                )
                labeled = True
                count += int(curve.n_points)
        return count
    mesh_kwargs = {
        "color": color,
        "opacity": 1.0 if bool(safe_render) else float(opacity),
        "smooth_shading": False,
        "label": label,
        "render_lines_as_tubes": False,
    }
    if bool(lightweight):
        mesh_kwargs.update({"style": "wireframe", "line_width": 0.55})
    plotter.add_mesh(surface, **mesh_kwargs)
    return int(surface.n_points)


def _middle_stage_points(theta_path: np.ndarray, labels: list[str], middle_name: str) -> np.ndarray:
    """Return dense theta points that belong to the active middle stage."""

    theta = np.asarray(theta_path, dtype=float)
    if theta.ndim != 2 or theta.shape[1] != 3 or len(theta) == 0:
        return np.zeros((0, 3), dtype=float)
    if len(labels) == len(theta):
        mask = np.asarray([str(label) == str(middle_name) for label in labels], dtype=bool)
        points = theta[mask]
        if len(points) >= 2:
            return np.asarray(points, dtype=float)
    return theta


def _add_middle_visual_sheet(
    plotter,
    pv,
    points: np.ndarray,
    *,
    color: str,
    opacity: float,
    label: str,
    surface_mode: str,
) -> int:
    """Draw a route-local rectangular middle sheet for presentation clarity only."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        return 0
    finite = pts[np.all(np.isfinite(pts), axis=1)]
    if len(finite) < 2:
        return 0
    center = np.mean(finite, axis=0)
    centered = finite - center
    try:
        _values, vectors = np.linalg.eigh(np.cov(centered, rowvar=False))
        basis = np.asarray(vectors[:, ::-1], dtype=float)
    except Exception:
        basis = np.eye(3, dtype=float)
    projected = centered @ basis
    extents = np.percentile(np.abs(projected[:, :2]), 95.0, axis=0) if len(finite) > 2 else np.asarray([0.3, 0.18])
    extents = np.maximum(extents * 1.35, np.asarray([0.35, 0.22], dtype=float))
    corners_2d = np.asarray(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ],
        dtype=float,
    )
    corners = (
        center[None, :]
        + corners_2d[:, 0:1] * float(extents[0]) * basis[:, 0][None, :]
        + corners_2d[:, 1:2] * float(extents[1]) * basis[:, 1][None, :]
    )
    patch = pv.PolyData(corners, faces=np.hstack([[4, 0, 1, 2, 3]]))
    plotter.add_mesh(
        patch,
        color=color,
        opacity=1.0 if str(surface_mode).lower() == "solid-safe" else float(opacity),
        smooth_shading=False,
        show_edges=False,
        render_lines_as_tubes=False,
        label=label,
    )
    return int(patch.n_points)


def _polydata_from_polylines(pv, polylines: list[np.ndarray]):
    """Create one flat line actor from one or more C-space polylines."""

    valid = [np.asarray(line, dtype=float) for line in polylines if len(line) >= 2]
    if len(valid) == 0:
        return pv.PolyData(np.zeros((0, 3), dtype=float))
    points = np.vstack(valid)
    lines: list[int] = []
    offset = 0
    for line in valid:
        count = int(len(line))
        lines.extend([count, *range(offset, offset + count)])
        offset += count
    poly = pv.PolyData(points)
    poly.lines = np.asarray(lines, dtype=np.int64)
    return poly


def _pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return a simple PCA envelope frame for sampled C-space surface points."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 6:
        return None
    center = np.mean(pts, axis=0)
    centered = pts - center
    try:
        _values, vectors = np.linalg.eigh(np.cov(centered, rowvar=False))
    except Exception:
        return None
    basis = np.asarray(vectors[:, ::-1], dtype=float)
    projected = centered @ basis
    extents = np.percentile(np.abs(projected), 94.0, axis=0)
    extents = np.maximum(extents, 0.035)
    return center, basis, extents


def _add_pca_envelope_loops(plotter, pv, points: np.ndarray, *, color: str, lightweight: bool) -> int:
    """Draw three clean ellipsoid loops around a sphere-like C-space cloud."""

    frame = _pca_frame(points)
    if frame is None:
        return 0
    center, basis, extents = frame
    angles = np.linspace(0.0, 2.0 * np.pi, 80 if bool(lightweight) else 120, endpoint=True)
    loops: list[np.ndarray] = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        loop = (
            center[None, :]
            + np.cos(angles)[:, None] * float(extents[i]) * basis[:, i][None, :]
            + np.sin(angles)[:, None] * float(extents[j]) * basis[:, j][None, :]
        )
        loops.append(np.asarray(loop, dtype=float))
    poly = _polydata_from_polylines(pv, loops)
    if poly.n_points == 0:
        return 0
    plotter.add_mesh(
        poly,
        color=color,
        line_width=1.5 if bool(lightweight) else 1.9,
        render_lines_as_tubes=False,
    )
    return int(poly.n_points)


def _add_point_cloud_box_outline(plotter, pv, points: np.ndarray, *, color: str, lightweight: bool) -> int:
    """Draw a sparse rectangular/slab outline around the middle C-space cloud."""

    frame = _pca_frame(points)
    if frame is None:
        return 0
    center, basis, extents = frame
    extents = np.asarray(extents, dtype=float)
    thin_axis = int(np.argmin(extents))
    plane_axes = [axis for axis in range(3) if axis != thin_axis]
    half_thickness = max(0.02, 0.25 * float(extents[thin_axis]))
    loops: list[np.ndarray] = []
    corners_2d = np.asarray(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ],
        dtype=float,
    )
    for sign in (-1.0, 1.0):
        loop = np.repeat(center[None, :], len(corners_2d), axis=0)
        loop += sign * half_thickness * basis[:, thin_axis][None, :]
        for idx, axis in enumerate(plane_axes):
            loop += corners_2d[:, idx : idx + 1] * float(extents[axis]) * basis[:, axis][None, :]
        loops.append(loop)
    for sx, sy in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
        edge = np.repeat(center[None, :], 2, axis=0)
        for axis, sign in zip(plane_axes, (sx, sy)):
            edge += sign * float(extents[axis]) * basis[:, axis][None, :]
        edge += np.asarray([-half_thickness, half_thickness], dtype=float)[:, None] * basis[:, thin_axis][None, :]
        loops.append(edge)
    poly = _polydata_from_polylines(pv, loops)
    if poly.n_points == 0:
        return 0
    plotter.add_mesh(
        poly,
        color=color,
        line_width=1.25 if bool(lightweight) else 1.6,
        render_lines_as_tubes=False,
    )
    return int(poly.n_points)


def _polyline_from_points(pv, points: np.ndarray):
    arr = np.asarray(points, dtype=float)
    poly = pv.PolyData(arr)
    if len(arr) >= 2:
        lines = np.hstack([[len(arr)], np.arange(len(arr), dtype=np.int64)])
        poly.lines = lines
    return poly


def _stage_segments(theta_path: np.ndarray, labels: list[str]) -> list[tuple[str, np.ndarray]]:
    """Split a dense theta route into stage-colored C-space segments."""

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
    """Draw one C-space configuration marker, usually start/goal/transition."""

    q = np.asarray(point, dtype=float).reshape(3)
    marker = pv.PolyData(q.reshape(1, 3))
    plotter.add_mesh(
        marker,
        color=color,
        point_size=max(3.0, 320.0 * float(radius)),
        render_points_as_spheres=False,
        label=label,
    )


def _valid_theta(value) -> np.ndarray | None:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        return None
    return arr


def _value_from_sources(result, audit: Mapping[str, object], key: str):
    if key in audit:
        return audit[key]
    return getattr(result, key, None)


def _transition_theta_points(result, audit: Mapping[str, object]) -> list[np.ndarray]:
    """Collect exact selected transition theta values for route-aware bounds."""

    points: list[np.ndarray] = []
    for key in (
        "selected_left_plane_transition_theta",
        "selected_plane_right_transition_theta",
        "selected_left_family_transition_theta",
        "selected_family_right_transition_theta",
    ):
        theta = _valid_theta(_value_from_sources(result, audit, key))
        if theta is not None:
            points.append(theta)
    return points


def _nearest_path_distance(theta_path: np.ndarray, theta: np.ndarray) -> float:
    if len(theta_path) == 0:
        return float("inf")
    return float(np.min(np.linalg.norm(np.asarray(theta_path, dtype=float) - theta[None, :], axis=1)))


def _stack_residual(manifolds: Mapping[str, object], source: str, target: str, theta: np.ndarray) -> float:
    if source not in manifolds or target not in manifolds:
        return float("inf")
    source_res = float(np.linalg.norm(manifolds[source].residual(theta)))
    target_res = float(np.linalg.norm(manifolds[target].residual(theta)))
    return float(np.linalg.norm([source_res, target_res]))


def _print_marker_audit(lines: list[str], audit: Mapping[str, object], result) -> None:
    for key in (
        "max_transition_stack_residual",
        "transition_stack_certified",
    ):
        value = _value_from_sources(result, audit, key)
        if value is not None:
            lines.append(f"{key}: {value}")
    print("C-space marker audit:", flush=True)
    for line in lines:
        print(line, flush=True)


def _add_exact_transition_markers(
    plotter,
    pv,
    *,
    result,
    manifolds,
    theta_path: np.ndarray,
    audit: Mapping[str, object],
    marker_scale: float,
) -> None:
    specs = (
        (
            "selected_left_plane_transition_theta",
            "selected_left_plane_stack_residual",
            "left",
            "plane",
            "#ef4444",
            "left-plane transition theta",
        ),
        (
            "selected_plane_right_transition_theta",
            "selected_plane_right_stack_residual",
            "plane",
            "right",
            "#14b8a6",
            "plane-right transition theta",
        ),
        (
            "selected_left_family_transition_theta",
            "selected_left_family_stack_residual",
            "left",
            "plane",
            "#ef4444",
            "left-family transition theta",
        ),
        (
            "selected_family_right_transition_theta",
            "selected_family_right_stack_residual",
            "plane",
            "right",
            "#14b8a6",
            "family-right transition theta",
        ),
    )
    lines: list[str] = []
    pair_residuals: list[float] = []
    for theta_key, residual_key, source, target, color, label in specs:
        raw_theta = _value_from_sources(result, audit, theta_key)
        raw_residual = _value_from_sources(result, audit, residual_key)
        if raw_theta is None and raw_residual is None:
            continue
        theta = _valid_theta(raw_theta)
        short_label = label.replace(" transition theta", "")
        if theta is None:
            lines.append(f"{short_label} marker source: exact theta unavailable; transition marker skipped.")
            print("Exact transition theta unavailable; transition marker skipped.", flush=True)
            continue
        residual = raw_residual
        if residual is None:
            residual = _stack_residual(manifolds, source, target, theta)
        source_residual = (
            float(np.linalg.norm(manifolds[source].residual(theta)))
            if source in manifolds
            else float("inf")
        )
        target_residual = (
            float(np.linalg.norm(manifolds[target].residual(theta)))
            if target in manifolds
            else float("inf")
        )
        pair_residuals.extend([source_residual, target_residual, float(residual)])
        distance = _nearest_path_distance(theta_path, theta)
        source_name = "family" if "family" in short_label and source == "plane" else source
        target_name = "family" if "family" in short_label and target == "plane" else target
        lines.append(f"{short_label} marker source: {theta_key}")
        lines.append(f"{short_label} theta: {np.array2string(theta, precision=6)}")
        lines.append(f"{short_label} stack residual: {residual}")
        lines.append(f"{short_label} {source_name} residual: {source_residual:.6g}")
        lines.append(f"{short_label} {target_name} residual: {target_residual:.6g}")
        lines.append(f"{short_label} distance-to-path: {distance:.6g}")
        _add_marker(plotter, pv, theta, color=color, radius=0.026 * float(marker_scale), label=label)
    max_stack = _value_from_sources(result, audit, "max_transition_stack_residual")
    stack_certified = _value_from_sources(result, audit, "transition_stack_certified")
    residual_ok = bool(len(pair_residuals) > 0 and max(pair_residuals) <= 1.0e-3)
    stack_ok = bool(stack_certified) if stack_certified is not None else bool(max_stack is not None and float(max_stack) <= 1.0e-3)
    if len(pair_residuals) > 0:
        lines.append(f"all_transition_constraints_satisfied: {bool(residual_ok and stack_ok)}")
    _print_marker_audit(lines, audit, result)


def _manifold_key_for_stage(stage: str, manifolds: Mapping[str, object]) -> str:
    if stage in manifolds:
        return stage
    if stage == "family" and "plane" in manifolds:
        return "plane"
    return stage


def _segment_bounds_and_residual_audit(
    theta_path: np.ndarray,
    labels: list[str],
    manifolds: Mapping[str, object],
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    result,
    audit: Mapping[str, object],
    middle_name: str,
    selected_lambda: float | None,
) -> dict[str, dict[str, object]]:
    """Check that each dense route segment is inside bounds and on its active surface."""

    if len(labels) != len(theta_path):
        print("Warning: C-space audit skipped because certified dense stage labels are unavailable.", flush=True)
        return {}

    q_path = np.asarray(theta_path, dtype=float)
    segment_specs: list[tuple[str, str, np.ndarray]] = []
    if middle_name == "family":
        entry_idx = int(_value_from_sources(result, audit, "selected_left_family_transition_index") or -1)
        exit_idx = int(_value_from_sources(result, audit, "selected_family_right_transition_index") or -1)
        if 0 <= entry_idx <= exit_idx < len(q_path):
            # Transition configurations may belong to both adjacent manifolds;
            # include them only in the segment whose active label matches the
            # certification labels to avoid evaluating points on the wrong leaf.
            segment_specs = [
                ("left", "left", q_path[:entry_idx]),
                ("family", "plane", q_path[entry_idx:exit_idx]),
                ("right", "right", q_path[exit_idx:]),
            ]
    else:
        entry_idx = int(_value_from_sources(result, audit, "selected_left_plane_transition_index") or -1)
        exit_idx = int(_value_from_sources(result, audit, "selected_plane_right_transition_index") or -1)
        if 0 <= entry_idx <= exit_idx < len(q_path):
            segment_specs = [
                ("left", "left", q_path[:entry_idx]),
                ("plane", "plane", q_path[entry_idx:exit_idx]),
                ("right", "right", q_path[exit_idx:]),
            ]
    if len(segment_specs) == 0:
        for stage in ("left", middle_name, "right"):
            mask = np.asarray([str(label) == stage for label in labels], dtype=bool)
            key = _manifold_key_for_stage(stage, manifolds)
            segment_specs.append((stage, key, q_path[mask]))

    support: dict[str, dict[str, object]] = {}
    for stage, key, segment in segment_specs:
        manifold = manifolds.get(key)
        points = np.asarray(segment, dtype=float)
        inside = bool(
            len(points) > 0
            and np.all(points >= lower.reshape(1, 3) - 1.0e-9)
            and np.all(points <= upper.reshape(1, 3) + 1.0e-9)
        )
        residuals = (
            np.asarray([float(np.linalg.norm(manifold.residual(theta))) for theta in points], dtype=float)
            if manifold is not None and len(points) > 0
            else np.asarray([], dtype=float)
        )
        support[str(stage)] = {
            "points": int(len(points)),
            "inside_bounds": bool(inside),
            "max_residual": float(np.max(residuals)) if len(residuals) > 0 else float("inf"),
            "label": (
                f"M_family(lambda={float(selected_lambda):.6g})"
                if str(stage) == "family" and selected_lambda is not None and np.isfinite(float(selected_lambda))
                else ("M_plane" if str(stage) == "plane" else f"M_{stage}")
            ),
        }
    return support


def _print_bounds_and_support_audit(
    *,
    theta_path: np.ndarray,
    labels: list[str],
    manifolds: Mapping[str, object],
    lower: np.ndarray,
    upper: np.ndarray,
    middle_name: str,
    result,
    audit: Mapping[str, object],
    selected_lambda: float | None,
) -> None:
    support = _segment_bounds_and_residual_audit(
        theta_path,
        labels,
        manifolds,
        lower,
        upper,
        result=result,
        audit=audit,
        middle_name=middle_name,
        selected_lambda=selected_lambda,
    )
    empty = {"points": 0, "inside_bounds": False, "max_residual": float("inf"), "label": ""}
    left = support.get("left", empty)
    middle = support.get(middle_name, empty)
    right = support.get("right", empty)
    print("C-space bounds audit:", flush=True)
    print(f"theta0_min = {float(lower[0])}", flush=True)
    print(f"theta0_max = {float(upper[0])}", flush=True)
    print(f"theta1_min = {float(lower[1])}", flush=True)
    print(f"theta1_max = {float(upper[1])}", flush=True)
    print(f"theta2_min = {float(lower[2])}", flush=True)
    print(f"theta2_max = {float(upper[2])}", flush=True)
    print("bounds_source = dense_theta_path + transitions + margin", flush=True)
    print(f"left segment inside surface bounds = {bool(left['inside_bounds'])}", flush=True)
    print(f"middle segment inside surface bounds = {bool(middle['inside_bounds'])}", flush=True)
    print(f"right segment inside surface bounds = {bool(right['inside_bounds'])}", flush=True)
    for stage_name, item in (("left", left), ("middle", middle), ("right", right)):
        if not bool(item["inside_bounds"]):
            print(
                f"Warning: C-space surface extraction did not cover certified route region for stage {stage_name}.",
                flush=True,
            )
    valid = bool(
        float(left["max_residual"]) <= 2.0e-3
        and float(middle["max_residual"]) <= 2.0e-3
        and float(right["max_residual"]) <= 2.0e-3
    )
    print("C-space visual support audit:", flush=True)
    print(f"left segment points = {int(left['points'])}", flush=True)
    print(f"left segment max residual on M_left = {float(left['max_residual'])}", flush=True)
    print(f"{middle_name} segment points = {int(middle['points'])}", flush=True)
    print(f"{middle_name} segment max residual on {middle['label']} = {float(middle['max_residual'])}", flush=True)
    print(f"right segment points = {int(right['points'])}", flush=True)
    print(f"right segment max residual on M_right = {float(right['max_residual'])}", flush=True)
    print(f"route_segments_constraint_valid = {bool(valid)}", flush=True)
    print("residual_tolerance = 0.002", flush=True)
    if bool(getattr(result, "dense_joint_path_execution_certified", False)):
        print("Note: route is certified; rendered surface is an approximate finite-resolution visualization.", flush=True)


def show_cspace_robot_planning(
    *,
    result,
    manifolds: Mapping[str, object],
    cspace_audit: Mapping[str, object] | None = None,
    grid_res: int = 65,
    output_dir: str | Path | None = None,
    show: bool = True,
    show_surfaces: bool = True,
    route_only: bool = False,
    lightweight: bool = True,
    marker_scale: float = 0.35,
    surface_opacity: float = 0.28,
    left_surface_opacity: float | None = None,
    middle_surface_opacity: float | None = None,
    right_surface_opacity: float | None = None,
    middle_surface_color: str | None = None,
    force_middle_sheet: bool = False,
    surface_style: str = "points-outline",
    surface_mode: str = "full",
    smooth_surfaces: bool = True,
    clean_view: bool = True,
    safe_render: bool = True,
    suppress_vtk_warnings: bool = True,
    vtk_warning_log: str | Path | None = None,
    exact_surfaces: bool | None = None,
    example_name: str = "generic",
    selected_lambda: float | None = None,
) -> Path | None:
    """Show the FK-pulled-back constraint surfaces in theta-space.

    Axes are theta0, theta1, theta2. The red/dark route is the dense theta path
    itself; FK is used only by the constraints that define the implicit
    surfaces, not as the plotted route coordinates.
    """
    effective_vtk_log = Path("outputs") / "vtk_warnings.log" if vtk_warning_log is None else vtk_warning_log
    vtk_warning_state, vtk_warning_log_path = _configure_vtk_warning_display(
        bool(suppress_vtk_warnings),
        log_path=effective_vtk_log,
    )
    pv = _pyvista()
    theta_path = np.asarray(getattr(result, "dense_joint_path", np.zeros((0, 3), dtype=float)), dtype=float)
    labels = list(getattr(result, "dense_joint_path_stage_labels", []))
    marker_scale = float(np.clip(float(marker_scale), 0.12, 2.0))
    if theta_path.ndim != 2 or theta_path.shape[1] != 3 or len(theta_path) == 0:
        print("No certified dense theta path found; C-space route cannot be shown.", flush=True)
        return None
    if pv is None:
        return _save_matplotlib_cspace_fallback(
            theta_path,
            labels,
            cspace_audit,
            output_dir,
            result=result,
            marker_scale=float(marker_scale),
        )
    if len(labels) != len(theta_path):
        raise RuntimeError("C-space view requires one stage label per dense theta waypoint.")

    surface_mode = str(surface_mode).lower()
    if bool(route_only):
        surface_mode = "route-only"
    if surface_mode not in {"full", "solid-safe", "route-only", "fallback"}:
        surface_mode = "full"
    if exact_surfaces is None:
        exact_surfaces = bool(surface_mode in {"full", "solid-safe"})
    else:
        exact_surfaces = bool(exact_surfaces)
        surface_mode = "full" if exact_surfaces else "fallback"
    route_only = bool(surface_mode == "route-only")
    if surface_mode == "solid-safe":
        safe_render = True
    surface_opacity = float(np.clip(float(surface_opacity), 0.05, 1.0))
    left_opacity = float(np.clip(float(left_surface_opacity) if left_surface_opacity is not None else min(surface_opacity, 0.30), 0.05, 1.0))
    middle_opacity = float(np.clip(float(middle_surface_opacity) if middle_surface_opacity is not None else max(surface_opacity, 0.46), 0.05, 1.0))
    right_opacity = float(np.clip(float(right_surface_opacity) if right_surface_opacity is not None else min(surface_opacity, 0.30), 0.05, 1.0))
    surface_style = str(surface_style).lower()
    if surface_style not in {"points-outline", "points", "wireframe", "contour"}:
        surface_style = "points-outline"
    audit = cspace_audit or {}
    transition_points = _transition_theta_points(result, audit)
    lower, upper = _finite_theta_bounds(theta_path, extra_points=transition_points)
    middle_name = "family" if str(example_name) == "continuous_transfer_family" else "plane"
    _print_bounds_and_support_audit(
        theta_path=theta_path,
        labels=labels,
        manifolds=manifolds,
        lower=lower,
        upper=upper,
        middle_name=middle_name,
        result=result,
        audit=audit,
        selected_lambda=selected_lambda,
    )

    if bool(safe_render):
        clean_view = True
    if bool(lightweight):
        if bool(exact_surfaces):
            grid_res = int(max(36, min(int(grid_res), 75)))
        else:
            grid_res = int(max(8, min(int(grid_res), 16 if surface_style == "wireframe" else 22)))
    else:
        grid_res = int(max(12, min(int(grid_res), 85)))
    axes = tuple(np.linspace(lower[i], upper[i], grid_res) for i in range(3))

    screenshot_path: Path | None = None
    if output_dir is not None:
        screenshot_path = Path(output_dir) / "cspace_environment.png"
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)

    plotter = pv.Plotter(off_screen=not bool(show))
    plotter.set_background("white")
    try:
        plotter.disable_anti_aliasing()
    except Exception:
        pass
    try:
        plotter.ren_win.SetMultiSamples(0)
    except Exception:
        try:
            plotter.render_window.SetMultiSamples(0)
        except Exception:
            pass
    for attr in ("disable_depth_peeling", "disable_eye_dome_lighting", "disable_shadows"):
        try:
            getattr(plotter, attr)()
        except Exception:
            pass

    draw_surfaces = bool(show_surfaces) and not bool(route_only)
    surface_counts = {"left": 0, "plane": 0, "right": 0}
    middle_color = str(middle_surface_color or ("#22d3ee" if middle_name == "family" else "#7dd3fc"))
    middle_surface_label = (
        f"M_family(lambda={float(selected_lambda):.6g})"
        if middle_name == "family" and selected_lambda is not None and np.isfinite(float(selected_lambda))
        else "M_plane"
    )
    middle_proxy_points = 0
    middle_exact_points = 0
    middle_surface_source = "none"
    if draw_surfaces:
        surface_counts["left"] = _add_implicit_surface(
            plotter,
            pv,
            manifolds["left"],
            axes,
            color="#f97316",
            opacity=0.26 if surface_style in {"points", "points-outline"} else (0.48 if lightweight else 0.18),
            label="left manifold surface",
            lightweight=bool(lightweight),
            surface_style=surface_style,
            safe_render=bool(safe_render),
            exact_isosurface=bool(exact_surfaces),
            surface_mode=str(surface_mode),
            surface_opacity=left_opacity,
            smooth_surfaces=bool(smooth_surfaces),
        )
        surface_counts["right"] = _add_implicit_surface(
            plotter,
            pv,
            manifolds["right"],
            axes,
            color="#16a34a",
            opacity=0.26 if surface_style in {"points", "points-outline"} else (0.48 if lightweight else 0.18),
            label="right manifold surface",
            lightweight=bool(lightweight),
            surface_style=surface_style,
            safe_render=bool(safe_render),
            exact_isosurface=bool(exact_surfaces),
            surface_mode=str(surface_mode),
            surface_opacity=right_opacity,
            smooth_surfaces=bool(smooth_surfaces),
        )
        middle_exact_points = _add_implicit_surface(
            plotter,
            pv,
            manifolds["plane"],
            axes,
            color=middle_color,
            opacity=0.55,
            label=middle_surface_label,
            lightweight=bool(lightweight),
            surface_style=surface_style,
            safe_render=bool(safe_render),
            exact_isosurface=bool(exact_surfaces),
            surface_mode=str(surface_mode),
            surface_opacity=middle_opacity,
            smooth_surfaces=bool(smooth_surfaces),
        )
        surface_counts["plane"] = int(middle_exact_points)
        middle_surface_source = (
            "selected_lambda_exact_isosurface"
            if middle_name == "family" and int(middle_exact_points) > 0
            else ("exact_isosurface" if int(middle_exact_points) > 0 else "none")
        )
        should_draw_middle_proxy = bool(force_middle_sheet or int(middle_exact_points) < 300 or bool(safe_render))
        if should_draw_middle_proxy:
            middle_proxy_points = _add_middle_visual_sheet(
                plotter,
                pv,
                _middle_stage_points(theta_path, labels, middle_name),
                color=middle_color,
                opacity=middle_opacity,
                label=f"{middle_surface_label} visual sheet",
                surface_mode=str(surface_mode),
            )
            if int(middle_proxy_points) > 0:
                middle_surface_source = "route_local_visual_proxy" if int(middle_exact_points) <= 0 else "exact_plus_proxy"
        for stage_name, count in (
            ("left", surface_counts["left"]),
            (middle_name, surface_counts["plane"] + middle_proxy_points),
            ("right", surface_counts["right"]),
        ):
            if int(count) <= 0:
                print(
                    f"Warning: C-space surface extraction did not cover certified route region for stage {stage_name}.",
                    flush=True,
                )

    full_line = _polyline_from_points(pv, theta_path)
    # The plotted route is configuration points/path in C-space, not robot geometry.
    plotter.add_mesh(full_line, color="#111827", line_width=4, render_lines_as_tubes=False, label="dense theta path")
    for stage, segment in _stage_segments(theta_path, labels):
        if len(segment) < 2:
            continue
        plotter.add_mesh(
            _polyline_from_points(pv, segment),
            color=STAGE_COLORS.get(stage, "#dc2626"),
            line_width=7 if bool(clean_view) else 8,
            render_lines_as_tubes=False,
            label=f"{stage} segment",
        )

    _add_marker(plotter, pv, theta_path[0], color="black", radius=0.032 * marker_scale, label="start theta")
    _add_marker(plotter, pv, theta_path[-1], color="#facc15", radius=0.038 * marker_scale, label="goal theta")

    _add_exact_transition_markers(
        plotter,
        pv,
        result=result,
        manifolds=manifolds,
        theta_path=theta_path,
        audit=audit,
        marker_scale=marker_scale,
    )

    try:
        plotter.show_bounds(
            bounds=(lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]),
            xtitle="theta0 [rad]",
            ytitle="theta1 [rad]",
            ztitle="theta2 [rad]",
            grid=False if bool(clean_view) else "front",
            location="outer",
            all_edges=not bool(clean_view),
        )
    except Exception:
        plotter.add_axes(
            xlabel="theta0 [rad]",
            ylabel="theta1 [rad]",
            zlabel="theta2 [rad]",
            line_width=2,
            labels_off=False,
        )
    if not bool(safe_render):
        plotter.add_text(
            "C-space view: axes are theta0, theta1, theta2\n"
            + (
                "Route-only view: dense theta path and exact transition theta."
                if not draw_surfaces
                else (
                    "exact zero-level-set surfaces; route is dense theta path."
                    if bool(exact_surfaces)
                    else f"{surface_style} surfaces are visual guides; route is dense theta path."
                )
        ),
            position="upper_left",
            font_size=10,
            color="black",
        )
        plotter.add_legend(size=(0.22, 0.22), bcolor="white", border=True)
    plotter.camera_position = "iso"

    print("cspace_visualization_requested = True")
    print("cspace_surface_mode = " + str(surface_mode))
    print("cspace_example = " + str(example_name))
    if selected_lambda is not None and np.isfinite(float(selected_lambda)):
        print("selected_lambda = " + str(float(selected_lambda)))
    print("cspace_grid_res = " + str(int(grid_res)))
    print("cspace_smooth_surfaces = " + str(bool(smooth_surfaces and exact_surfaces and draw_surfaces)))
    print("cspace_opacity = " + str(float(surface_opacity)))
    print("cspace_left_opacity = " + str(float(left_opacity)))
    print("cspace_middle_opacity = " + str(float(middle_opacity)))
    print("cspace_right_opacity = " + str(float(right_opacity)))
    print("cspace_surface_points_left = " + str(surface_counts["left"]))
    print("cspace_surface_points_middle = " + str(surface_counts["plane"] + middle_proxy_points))
    print(f"cspace_surface_points_{middle_name} = " + str(surface_counts["plane"] + middle_proxy_points))
    print("cspace_surface_points_right = " + str(surface_counts["right"]))
    print("middle_surface_label = " + str(middle_surface_label))
    print("middle_surface_source = " + str(middle_surface_source))
    print("middle_surface_exact_points = " + str(middle_exact_points))
    print("middle_surface_proxy_points = " + str(middle_proxy_points))
    print("middle_surface_points = " + str(surface_counts["plane"] + middle_proxy_points))
    print("middle_surface_opacity = " + str(float(middle_opacity)))
    print("middle_surface_color = " + str(middle_color))
    print("middle_surface_drawn = " + str(bool(draw_surfaces and int(surface_counts["plane"] + middle_proxy_points) > 0)))
    print("cspace_force_middle_sheet = " + str(bool(force_middle_sheet)))
    print("cspace_lightweight = " + str(bool(lightweight)))
    print("cspace_route_only = " + str(bool(route_only)))
    print("cspace_surfaces_drawn = " + str(bool(draw_surfaces)))
    print("cspace_surface_style = " + str(surface_style))
    print(
        "cspace_boundary_style = "
        + (
            "exact_isosurface_mesh"
            if bool(draw_surfaces) and bool(exact_surfaces)
            else ("none" if not draw_surfaces or surface_style != "points-outline" else "pca_envelope_loops")
        )
    )
    print("cspace_marker_scale = " + str(float(marker_scale)))
    print("cspace_safe_render = " + str(bool(safe_render)))
    print("cspace_fancy_render = " + str(not bool(safe_render)))
    print("cspace_clean_view = " + str(bool(clean_view)))
    print("vtk_warning_display = " + str(vtk_warning_state))
    if vtk_warning_log_path is not None:
        print("vtk_warning_log = " + str(vtk_warning_log_path))
        print("VTK/OpenGL render messages redirected to: " + str(vtk_warning_log_path))
    print("pyvista_auto_close = True")

    try:
        with suppress_native_output(bool(suppress_vtk_warnings), vtk_warning_log_path):
            if show:
                plotter.show(
                    screenshot=str(screenshot_path) if screenshot_path is not None else None,
                    auto_close=True,
                )
            elif screenshot_path is not None:
                plotter.show(screenshot=str(screenshot_path), auto_close=True)
            else:
                plotter.close()
    finally:
        with suppress_native_output(bool(suppress_vtk_warnings), vtk_warning_log_path):
            try:
                plotter.close()
            except Exception:
                pass
            try:
                del plotter
            except Exception:
                pass
        print("PyVista window closed cleanly.", flush=True)
    return screenshot_path


def _save_matplotlib_cspace_fallback(
    theta_path: np.ndarray,
    labels: list[str],
    cspace_audit: Mapping[str, object] | None,
    output_dir: str | Path | None,
    *,
    result=None,
    marker_scale: float = 0.40,
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
            # Matplotlib fallback still plots the dense theta route in C-space.
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=STAGE_COLORS.get(stage, "#dc2626"), linewidth=3.0)
    if len(theta_path) > 0:
        ax.scatter(theta_path[0, 0], theta_path[0, 1], theta_path[0, 2], color="black", s=35 * float(marker_scale) ** 2)
        ax.scatter(theta_path[-1, 0], theta_path[-1, 1], theta_path[-1, 2], color="gold", s=45 * float(marker_scale) ** 2)
    audit = cspace_audit or {}
    for key, color in (
        ("selected_left_plane_transition_theta", "#ef4444"),
        ("selected_plane_right_transition_theta", "#14b8a6"),
        ("selected_left_family_transition_theta", "#ef4444"),
        ("selected_family_right_transition_theta", "#14b8a6"),
    ):
        theta = _valid_theta(_value_from_sources(result, audit, key))
        if theta is not None:
            ax.scatter(theta[0], theta[1], theta[2], color=color, s=55 * float(marker_scale) ** 2)
        elif key in audit or (result is not None and getattr(result, key, None) is not None):
            print("Exact transition theta unavailable; transition marker skipped.", flush=True)
    ax.set_xlabel("theta0 [rad]")
    ax.set_ylabel("theta1 [rad]")
    ax.set_zlabel("theta2 [rad]")
    ax.set_title("C-space dense theta path")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
