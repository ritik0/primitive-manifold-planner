from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ConstraintGeometry:
    left_center: np.ndarray
    left_radius: float
    right_center: np.ndarray
    right_radius: float
    line_point: np.ndarray
    line_normal: np.ndarray


@dataclass(frozen=True)
class RouteSelection:
    left_branch: np.ndarray
    line_branch: np.ndarray
    right_branch: np.ndarray
    left_segment: np.ndarray
    line_segment: np.ndarray
    right_segment: np.ndarray
    start_theta: np.ndarray
    left_line_transition: np.ndarray
    line_right_transition: np.ndarray
    goal_theta: np.ndarray


@dataclass(frozen=True)
class FigureLayoutAudit:
    taskspace_xlim: tuple[float, float]
    taskspace_ylim: tuple[float, float]
    cspace_manifold_xlim: tuple[float, float]
    cspace_manifold_ylim: tuple[float, float]
    cspace_route_xlim: tuple[float, float]
    cspace_route_ylim: tuple[float, float]


def default_geometry() -> ConstraintGeometry:
    # The line is almost horizontal and placed between two separated circles.
    # It intersects each circle only near the inward-facing side, giving a
    # clear left -> transfer -> right task-space story.
    normal = np.asarray([0.06, 1.0], dtype=float)
    normal = normal / np.linalg.norm(normal)
    return ConstraintGeometry(
        left_center=np.asarray([0.45, 0.70], dtype=float),
        left_radius=0.28,
        right_center=np.asarray([1.45, 0.30], dtype=float),
        right_radius=0.28,
        line_point=np.asarray([0.95, 0.52], dtype=float),
        line_normal=normal,
    )


def set_equal_with_margin(ax, points: np.ndarray, margin: float = 0.15) -> tuple[tuple[float, float], tuple[float, float]]:
    """Fit an equal-aspect 2D axis around the provided points."""

    pts = np.asarray(points, dtype=float)
    pts = pts.reshape(-1, 2)
    finite = pts[np.all(np.isfinite(pts), axis=1)]
    if len(finite) == 0:
        xlim = (-1.0, 1.0)
        ylim = (-1.0, 1.0)
    else:
        lower = np.min(finite, axis=0)
        upper = np.max(finite, axis=0)
        center = 0.5 * (lower + upper)
        span = np.maximum(upper - lower, 1.0e-6)
        half = 0.5 * max(float(span[0]), float(span[1])) + float(margin)
        xlim = (float(center[0] - half), float(center[0] + half))
        ylim = (float(center[1] - half), float(center[1] + half))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    return xlim, ylim


def set_taskspace_limits(ax, all_task_points: np.ndarray, margin: float = 0.20) -> tuple[tuple[float, float], tuple[float, float]]:
    return set_equal_with_margin(ax, all_task_points, margin=margin)


def set_cspace_limits(ax, all_cspace_points: np.ndarray, margin: float = 0.25) -> tuple[tuple[float, float], tuple[float, float]]:
    pts = np.asarray(all_cspace_points, dtype=float).reshape(-1, 2)
    finite = pts[np.all(np.isfinite(pts), axis=1)]
    if len(finite) == 0:
        xlim = (-np.pi, np.pi)
        ylim = (-np.pi, np.pi)
    else:
        lower = np.min(finite, axis=0)
        upper = np.max(finite, axis=0)
        span = np.maximum(upper - lower, 1.0e-6)
        pad = np.maximum(float(margin) * span, 0.18)
        xlim = (float(max(-np.pi, lower[0] - pad[0])), float(min(np.pi, upper[0] + pad[0])))
        ylim = (float(max(-np.pi, lower[1] - pad[1])), float(min(np.pi, upper[1] + pad[1])))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return xlim, ylim


def forward_kinematics(theta: np.ndarray, link1: float, link2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map theta=[theta1, theta2] to the planar robot base, elbow, and end-effector."""

    theta = np.asarray(theta, dtype=float)
    theta1 = theta[..., 0]
    theta2 = theta[..., 1]
    base = np.zeros(theta.shape[:-1] + (2,), dtype=float)
    elbow = np.stack([link1 * np.cos(theta1), link1 * np.sin(theta1)], axis=-1)
    ee = elbow + np.stack(
        [link2 * np.cos(theta1 + theta2), link2 * np.sin(theta1 + theta2)],
        axis=-1,
    )
    return base, elbow, ee


def circle_residual(xy: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    diff = np.asarray(xy, dtype=float) - center.reshape((1,) * (np.asarray(xy).ndim - 1) + (2,))
    return diff[..., 0] ** 2 + diff[..., 1] ** 2 - float(radius) ** 2


def line_residual(xy: np.ndarray, normal: np.ndarray, point: np.ndarray) -> np.ndarray:
    diff = np.asarray(xy, dtype=float) - point.reshape((1,) * (np.asarray(xy).ndim - 1) + (2,))
    return diff[..., 0] * normal[0] + diff[..., 1] * normal[1]


def residuals_from_theta(theta: np.ndarray, *, geometry: ConstraintGeometry, link1: float, link2: float) -> dict[str, np.ndarray]:
    """Evaluate F_i(theta)=h_i(FK(theta)) for each active task-space constraint."""

    _base, _elbow, ee = forward_kinematics(theta, link1, link2)
    return {
        "left": circle_residual(ee, geometry.left_center, geometry.left_radius),
        "line": line_residual(ee, geometry.line_normal, geometry.line_point),
        "right": circle_residual(ee, geometry.right_center, geometry.right_radius),
    }


def contour_branches(theta1: np.ndarray, theta2: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    """Extract residual(theta)=0 contour branches using matplotlib."""

    fig, ax = plt.subplots()
    try:
        contour = ax.contour(theta1, theta2, values, levels=[0.0])
        branches = [np.asarray(seg, dtype=float) for seg in contour.allsegs[0] if len(seg) >= 2]
    finally:
        plt.close(fig)
    return branches


def branch_length(branch: np.ndarray) -> float:
    if len(branch) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(branch, axis=0), axis=1)))


def nearest_branch_pair(first: np.ndarray, second: np.ndarray) -> tuple[int, int, float]:
    """Find the closest pair of sampled points between two contour branches."""

    diff = first[:, None, :] - second[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    flat_idx = int(np.argmin(distances))
    i, j = np.unravel_index(flat_idx, distances.shape)
    return int(i), int(j), float(distances[i, j])


def targeted_branch_pair(
    first: np.ndarray,
    second: np.ndarray,
    target_xy: np.ndarray,
    *,
    link1: float,
    link2: float,
) -> tuple[int, int, float, np.ndarray]:
    """Find a contour intersection near a desired task-space transfer region."""

    diff = first[:, None, :] - second[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    candidate_mask = distances <= max(0.06, float(np.min(distances)) + 0.025)
    candidate_indices = np.argwhere(candidate_mask)
    if len(candidate_indices) == 0:
        i, j, distance = nearest_branch_pair(first, second)
        theta = 0.5 * (first[i] + second[j])
        _base, _elbow, fk = forward_kinematics(theta, link1, link2)
        return i, j, distance, np.asarray(fk, dtype=float)

    best: tuple[float, int, int, float, np.ndarray] | None = None
    target = np.asarray(target_xy, dtype=float)
    for i, j in candidate_indices:
        theta = 0.5 * (first[int(i)] + second[int(j)])
        _base, _elbow, fk = forward_kinematics(theta, link1, link2)
        curve_distance = float(distances[int(i), int(j)])
        task_distance = float(np.linalg.norm(np.asarray(fk, dtype=float) - target))
        score = curve_distance + 0.25 * task_distance
        if best is None or score < best[0]:
            best = (score, int(i), int(j), curve_distance, np.asarray(fk, dtype=float))
    assert best is not None
    return best[1], best[2], best[3], best[4]


def select_route(
    left_branches: list[np.ndarray],
    line_branches: list[np.ndarray],
    right_branches: list[np.ndarray],
    *,
    geometry: ConstraintGeometry,
    link1: float,
    link2: float,
) -> RouteSelection:
    """Pick contour branches whose FK images tell a left -> line -> right story."""

    best: tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, int]] | None = None
    line_direction = np.asarray([-geometry.line_normal[1], geometry.line_normal[0]], dtype=float)
    if line_direction[0] < 0.0:
        line_direction = -line_direction
    left_target = geometry.left_center + 0.65 * geometry.left_radius * line_direction
    right_target = geometry.right_center - 0.65 * geometry.right_radius * line_direction
    for left in left_branches:
        for line in line_branches:
            ll_i, ll_j, ll_dist, ll_fk = targeted_branch_pair(left, line, left_target, link1=link1, link2=link2)
            for right in right_branches:
                lr_j, lr_k, lr_dist, lr_fk = targeted_branch_pair(line, right, right_target, link1=link1, link2=link2)
                if abs(lr_j - ll_j) < 12:
                    continue
                story_penalty = 0.0 if float(ll_fk[0]) < float(lr_fk[0]) else 5.0
                target_penalty = 0.35 * (
                    float(np.linalg.norm(ll_fk - left_target))
                    + float(np.linalg.norm(lr_fk - right_target))
                )
                score = ll_dist + lr_dist + story_penalty + target_penalty - 0.004 * branch_length(line)
                candidate = (left, line, right, ll_i, ll_j, lr_j, lr_k)
                if best is None or score < best[0]:
                    best = (score, candidate)
    if best is None:
        raise RuntimeError("Could not find a separated left-line and line-right transition pair.")

    left, line, right, ll_i, ll_j, lr_j, lr_k = best[1]
    if lr_j < ll_j:
        line = line[::-1].copy()
        ll_j = len(line) - 1 - ll_j
        lr_j = len(line) - 1 - lr_j
    if ll_i < 4:
        left = left[::-1].copy()
        ll_i = len(left) - 1 - ll_i
    if lr_k > len(right) - 5:
        right = right[::-1].copy()
        lr_k = len(right) - 1 - lr_k

    left_segment = np.asarray(left[max(0, ll_i - max(18, ll_i // 2)) : ll_i + 1], dtype=float)
    line_segment = np.asarray(line[ll_j : lr_j + 1], dtype=float)
    right_segment = np.asarray(right[lr_k : min(len(right), lr_k + max(18, (len(right) - lr_k) // 2))], dtype=float)
    if len(left_segment) < 2 or len(line_segment) < 2 or len(right_segment) < 2:
        raise RuntimeError("Selected contour branches were too short for a readable route.")

    return RouteSelection(
        left_branch=left,
        line_branch=line,
        right_branch=right,
        left_segment=left_segment,
        line_segment=line_segment,
        right_segment=right_segment,
        start_theta=left_segment[0],
        left_line_transition=0.5 * (left[ll_i] + line[ll_j]),
        line_right_transition=0.5 * (line[lr_j] + right[lr_k]),
        goal_theta=right_segment[-1],
    )


def draw_robot(ax, theta: np.ndarray, link1: float, link2: float, *, color: str, alpha: float) -> np.ndarray:
    base, elbow, ee = forward_kinematics(np.asarray(theta, dtype=float), link1, link2)
    points = np.vstack([base, elbow, ee])
    ax.plot(points[:, 0], points[:, 1], color=color, linewidth=1.4, alpha=alpha)
    ax.scatter(points[-1, 0], points[-1, 1], color=color, s=18, alpha=min(1.0, alpha + 0.25))
    return np.asarray(points[-1], dtype=float)


def task_constraint_points(geometry: ConstraintGeometry) -> dict[str, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, 360)
    left = geometry.left_center + geometry.left_radius * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    right = geometry.right_center + geometry.right_radius * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    tangent = np.asarray([-geometry.line_normal[1], geometry.line_normal[0]], dtype=float)
    line_values = np.linspace(-0.48, 0.62, 2)
    line = geometry.line_point.reshape(1, 2) + line_values.reshape(-1, 1) * tangent.reshape(1, 2)
    return {"left": left, "line": line, "right": right}


def draw_task_constraints(ax, geometry: ConstraintGeometry, reach: float) -> dict[str, np.ndarray]:
    del reach
    points = task_constraint_points(geometry)
    left = points["left"]
    line = points["line"]
    right = points["right"]
    ax.plot(left[:, 0], left[:, 1], color="#f97316", linewidth=2.2, label="left circle")
    ax.plot(line[:, 0], line[:, 1], color="#2563eb", linewidth=2.4, label="transfer line")
    ax.plot(right[:, 0], right[:, 1], color="#16a34a", linewidth=2.2, label="right circle")
    return points


def mark_transition_candidates(ax, first_branches: list[np.ndarray], second_branches: list[np.ndarray], *, color: str, label: str) -> int:
    candidates = []
    for first in first_branches:
        for second in second_branches:
            i, j, distance = nearest_branch_pair(first, second)
            if distance <= 0.045:
                candidates.append(0.5 * (first[i] + second[j]))
    if candidates:
        points = np.asarray(candidates, dtype=float)
        ax.scatter(points[:, 0], points[:, 1], color=color, s=22, edgecolors="black", linewidths=0.3, label=label, zorder=6)
    return len(candidates)


def plot_contours(ax, branches: list[np.ndarray], *, color: str, label: str, linewidth: float = 1.7, alpha: float = 1.0) -> None:
    labeled = False
    for branch in branches:
        ax.plot(branch[:, 0], branch[:, 1], color=color, linewidth=linewidth, alpha=alpha, label=label if not labeled else None)
        labeled = True


def build_figure(
    *,
    theta1: np.ndarray,
    theta2: np.ndarray,
    residuals: dict[str, np.ndarray],
    branches: dict[str, list[np.ndarray]],
    route: RouteSelection,
    geometry: ConstraintGeometry,
    link1: float,
    link2: float,
    presentation_layout: bool,
) -> tuple[plt.Figure, FigureLayoutAudit]:
    reach = link1 + link2
    fig_size = (15.0, 11.0) if bool(presentation_layout) else (13.5, 10.0)
    fig, axes = plt.subplots(2, 2, figsize=fig_size, constrained_layout=bool(presentation_layout))
    fig.suptitle("2D analogue of Example 66: task constraints pulled back through FK", fontsize=14)
    ax_task, ax_manifolds, ax_route, ax_fk = axes.ravel()
    constraint_points = task_constraint_points(geometry)
    route_points = np.vstack([route.left_segment, route.line_segment, route.right_segment])
    _base, _elbow, fk_route = forward_kinematics(route_points, link1, link2)

    # Panel 1: Each task-space constraint will be pulled back through FK.
    drawn_robot_points = []
    draw_task_constraints(ax_task, geometry, reach)
    for segment, color in (
        (route.left_segment, "#f97316"),
        (route.line_segment, "#2563eb"),
        (route.right_segment, "#16a34a"),
    ):
        sample_idx = np.linspace(0, len(segment) - 1, min(2, len(segment)), dtype=int)
        for idx in sample_idx:
            base, elbow, ee = forward_kinematics(segment[idx], link1, link2)
            drawn_robot_points.extend([base, elbow, ee])
            draw_robot(ax_task, segment[idx], link1, link2, color=color, alpha=0.24)
    task_points = np.vstack(
        [
            constraint_points["left"],
            constraint_points["line"],
            constraint_points["right"],
            fk_route,
            np.asarray(drawn_robot_points, dtype=float).reshape(-1, 2),
        ]
    )
    task_xlim, task_ylim = set_taskspace_limits(ax_task, task_points, margin=0.18)
    ax_task.set_xlabel("x")
    ax_task.set_ylabel("y")
    ax_task.set_title("Task space constraints")
    ax_task.legend(loc="upper right", fontsize=8, framealpha=0.85)

    # Panel 2: These are actual zero contours of F_i(theta)=h_i(FK(theta)).
    plot_contours(ax_manifolds, branches["left"], color="#f97316", label="M_left")
    plot_contours(ax_manifolds, branches["line"], color="#2563eb", label="M_line")
    plot_contours(ax_manifolds, branches["right"], color="#16a34a", label="M_right")
    ll_count = mark_transition_candidates(ax_manifolds, branches["left"], branches["line"], color="#ef4444", label="left-line transitions")
    lr_count = mark_transition_candidates(ax_manifolds, branches["line"], branches["right"], color="#14b8a6", label="line-right transitions")
    ax_manifolds.set_xlabel("theta1 [rad]")
    ax_manifolds.set_ylabel("theta2 [rad]")
    manifold_points = np.vstack([*(branches["left"] + branches["line"] + branches["right"])])
    manifold_xlim, manifold_ylim = set_cspace_limits(ax_manifolds, manifold_points, margin=0.06)
    ax_manifolds.set_title("FK-pulled-back C-space manifolds")
    ax_manifolds.legend(loc="upper right", fontsize=8, framealpha=0.85)

    # Panel 3: The route changes active manifold only at transition configurations.
    plot_contours(ax_route, branches["left"], color="#fed7aa", label="M_left", linewidth=1.0, alpha=0.25)
    plot_contours(ax_route, branches["line"], color="#bfdbfe", label="M_line", linewidth=1.0, alpha=0.25)
    plot_contours(ax_route, branches["right"], color="#bbf7d0", label="M_right", linewidth=1.0, alpha=0.25)
    ax_route.plot(route.left_segment[:, 0], route.left_segment[:, 1], color="#f97316", linewidth=3.0, label="left segment")
    ax_route.plot(route.line_segment[:, 0], route.line_segment[:, 1], color="#2563eb", linewidth=3.0, label="line segment")
    ax_route.plot(route.right_segment[:, 0], route.right_segment[:, 1], color="#16a34a", linewidth=3.0, label="right segment")
    ax_route.scatter(route.start_theta[0], route.start_theta[1], color="black", s=42, label="start theta", zorder=8)
    ax_route.scatter(route.left_line_transition[0], route.left_line_transition[1], color="#ef4444", s=48, edgecolors="black", linewidths=0.4, label="left-line transition", zorder=8)
    ax_route.scatter(route.line_right_transition[0], route.line_right_transition[1], color="#14b8a6", s=48, edgecolors="black", linewidths=0.4, label="line-right transition", zorder=8)
    ax_route.scatter(route.goal_theta[0], route.goal_theta[1], color="#facc15", edgecolors="black", s=54, label="goal theta", zorder=8)
    ax_route.set_xlabel("theta1 [rad]")
    ax_route.set_ylabel("theta2 [rad]")
    local_branch_points = np.vstack([route.left_branch, route.line_branch, route.right_branch, route_points])
    route_xlim, route_ylim = set_cspace_limits(ax_route, local_branch_points, margin=0.05)
    ax_route.set_title("Selected route in C-space")
    ax_route.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.85)

    # Panel 4: FK of the theta route lands back on the task-space constraints.
    draw_task_constraints(ax_fk, geometry, reach)
    for segment, color, label in (
        (route.left_segment, "#f97316", "FK left segment"),
        (route.line_segment, "#2563eb", "FK line segment"),
        (route.right_segment, "#16a34a", "FK right segment"),
    ):
        _base, _elbow, ee = forward_kinematics(segment, link1, link2)
        ax_fk.plot(ee[:, 0], ee[:, 1], color=color, linewidth=3.0, label=label)
    for theta, color, label in (
        (route.start_theta, "black", "start"),
        (route.left_line_transition, "#ef4444", "left-line"),
        (route.line_right_transition, "#14b8a6", "line-right"),
        (route.goal_theta, "#facc15", "goal"),
    ):
        _base, _elbow, ee = forward_kinematics(theta, link1, link2)
        ax_fk.scatter(ee[0], ee[1], color=color, edgecolors="black" if color == "#facc15" else None, s=45, label=label)
    fk_points = np.vstack([constraint_points["left"], constraint_points["line"], constraint_points["right"], fk_route])
    set_taskspace_limits(ax_fk, fk_points, margin=0.16)
    ax_fk.set_xlabel("x")
    ax_fk.set_ylabel("y")
    ax_fk.set_title("FK image of selected route")
    ax_fk.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, framealpha=0.85)

    # Return candidate counts through figure metadata for terminal diagnostics.
    fig._ex67b_transition_counts = (ll_count, lr_count)  # type: ignore[attr-defined]
    if not bool(presentation_layout):
        fig.tight_layout(pad=2.0)
    return fig, FigureLayoutAudit(
        taskspace_xlim=task_xlim,
        taskspace_ylim=task_ylim,
        cspace_manifold_xlim=manifold_xlim,
        cspace_manifold_ylim=manifold_ylim,
        cspace_route_xlim=route_xlim,
        cspace_route_ylim=route_ylim,
    )


def max_abs_residual(theta: np.ndarray, key: str, geometry: ConstraintGeometry, link1: float, link2: float) -> float:
    values = residuals_from_theta(theta.reshape(1, -1, 2), geometry=geometry, link1=link1, link2=link2)[key].reshape(-1)
    return float(np.max(np.abs(values))) if len(values) else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 67B: 2D multimodal FK pullback analogue of Example 66.",
    )
    parser.add_argument("--grid-res", type=int, default=420)
    parser.add_argument("--link1", type=float, default=1.0)
    parser.add_argument("--link2", type=float, default=0.8)
    parser.add_argument("--save-figure", type=str, default="outputs/example_67b_planar_multimodal_pullback.png")
    layout_group = parser.add_mutually_exclusive_group()
    layout_group.add_argument("--presentation-layout", dest="presentation_layout", action="store_true", default=True)
    layout_group.add_argument("--basic-layout", dest="presentation_layout", action="store_false")
    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument("--show", dest="show", action="store_true", default=True)
    show_group.add_argument("--no-show", dest="show", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid_res = int(max(120, args.grid_res))
    link1 = float(args.link1)
    link2 = float(args.link2)
    geometry = default_geometry()

    theta1 = np.linspace(-np.pi, np.pi, grid_res)
    theta2 = np.linspace(-np.pi, np.pi, grid_res)
    theta1_grid, theta2_grid = np.meshgrid(theta1, theta2, indexing="xy")
    theta_grid = np.stack([theta1_grid, theta2_grid], axis=-1)

    residuals = residuals_from_theta(theta_grid, geometry=geometry, link1=link1, link2=link2)
    branches = {
        "left": contour_branches(theta1, theta2, residuals["left"]),
        "line": contour_branches(theta1, theta2, residuals["line"]),
        "right": contour_branches(theta1, theta2, residuals["right"]),
    }
    if not branches["left"] or not branches["line"] or not branches["right"]:
        raise RuntimeError("One or more pulled-back manifolds had no zero contour.")
    route = select_route(branches["left"], branches["line"], branches["right"], geometry=geometry, link1=link1, link2=link2)
    fig, layout_audit = build_figure(
        theta1=theta1,
        theta2=theta2,
        residuals=residuals,
        branches=branches,
        route=route,
        geometry=geometry,
        link1=link1,
        link2=link2,
        presentation_layout=bool(args.presentation_layout),
    )
    ll_count, lr_count = getattr(fig, "_ex67b_transition_counts", (0, 0))
    selected_route = np.vstack([route.left_segment, route.line_segment, route.right_segment])

    print("=== Example 67B: 2D multimodal FK pullback demo ===")
    print("robot_dof                         : 2")
    print("task_constraints                  : left circle, transfer line, right circle")
    print(f"left_circle_center                : {np.array2string(geometry.left_center, precision=5)}")
    print(f"left_circle_radius                : {float(geometry.left_radius):.5f}")
    print(f"right_circle_center               : {np.array2string(geometry.right_center, precision=5)}")
    print(f"right_circle_radius               : {float(geometry.right_radius):.5f}")
    print(f"transfer_line_point               : {np.array2string(geometry.line_point, precision=5)}")
    line_direction = np.asarray([-geometry.line_normal[1], geometry.line_normal[0]], dtype=float)
    if line_direction[0] < 0.0:
        line_direction = -line_direction
    print(f"transfer_line_direction           : {np.array2string(line_direction, precision=5)}")
    print("cspace_residuals                  : F_left(theta), F_line(theta), F_right(theta)")
    print("cspace_dimension                  : 2")
    print("manifold_dimension_each           : 1")
    print(f"left_contour_branches             : {len(branches['left'])}")
    print(f"line_contour_branches             : {len(branches['line'])}")
    print(f"right_contour_branches            : {len(branches['right'])}")
    print(f"left_line_transition_candidates   : {ll_count}")
    print(f"line_right_transition_candidates  : {lr_count}")
    print(f"selected_start_theta              : {np.array2string(route.start_theta, precision=5)}")
    print(f"selected_left_line_transition     : {np.array2string(route.left_line_transition, precision=5)}")
    print(f"selected_line_right_transition    : {np.array2string(route.line_right_transition, precision=5)}")
    print(f"selected_goal_theta               : {np.array2string(route.goal_theta, precision=5)}")
    print(f"selected_route_points             : {len(selected_route)}")
    print(f"max_left_residual_on_route        : {max_abs_residual(route.left_segment, 'left', geometry, link1, link2):.6e}")
    print(f"max_line_residual_on_route        : {max_abs_residual(route.line_segment, 'line', geometry, link1, link2):.6e}")
    print(f"max_right_residual_on_route       : {max_abs_residual(route.right_segment, 'right', geometry, link1, link2):.6e}")
    print("explanation                       : each task-space constraint is pulled back through FK into a 1D C-space manifold")

    save_path = Path(args.save_figure)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    print("=== Figure layout audit ===")
    print(f"taskspace_xlim : {layout_audit.taskspace_xlim}")
    print(f"taskspace_ylim : {layout_audit.taskspace_ylim}")
    print(f"cspace_manifold_xlim : {layout_audit.cspace_manifold_xlim}")
    print(f"cspace_manifold_ylim : {layout_audit.cspace_manifold_ylim}")
    print(f"cspace_route_xlim : {layout_audit.cspace_route_xlim}")
    print(f"cspace_route_ylim : {layout_audit.cspace_route_ylim}")
    print(f"figure_saved_to : {save_path}")
    print(f"figure_saved                      : {save_path}")
    if bool(args.show):
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
