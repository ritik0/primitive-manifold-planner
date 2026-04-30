from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from primitive_manifold_planner.manifolds import CircleManifold, EllipseManifold


def plot_circle_manifold(
    ax: plt.Axes,
    circle: CircleManifold,
    num_points: int = 400,
    label: str | None = None,
) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, num_points)
    x = circle.center[0] + circle.radius * np.cos(theta)
    y = circle.center[1] + circle.radius * np.sin(theta)
    ax.plot(x, y, label=label or circle.name)


def plot_ellipse_manifold(
    ax: plt.Axes,
    ellipse: EllipseManifold,
    num_points: int = 400,
    label: str | None = None,
) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, num_points)
    x = ellipse.center[0] + ellipse.a * np.cos(theta)
    y = ellipse.center[1] + ellipse.b * np.sin(theta)
    ax.plot(x, y, label=label or ellipse.name)


def plot_path_2d(
    ax: plt.Axes,
    path: np.ndarray,
    show_points: bool = True,
    label: str = "path",
) -> None:
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError(f"path must have shape (N, 2), got {path.shape}")

    ax.plot(path[:, 0], path[:, 1], label=label)

    if show_points:
        ax.scatter(path[:, 0], path[:, 1], s=20)


def plot_start_goal(
    ax: plt.Axes,
    x_start: np.ndarray,
    x_goal: np.ndarray,
) -> None:
    x_start = np.asarray(x_start, dtype=float).reshape(2)
    x_goal = np.asarray(x_goal, dtype=float).reshape(2)

    ax.scatter([x_start[0]], [x_start[1]], s=100, marker="o", label="start")
    ax.scatter([x_goal[0]], [x_goal[1]], s=100, marker="*", label="goal")


def plot_transition_candidates_2d(
    ax: plt.Axes,
    candidates: list,
    selected_point: np.ndarray | None = None,
    annotate: bool = True,
    label_all: str = "transition candidates",
    label_selected: str = "selected transition",
) -> None:
    if len(candidates) == 0:
        return

    pts = np.asarray([cand.point for cand in candidates], dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Transition candidate points must have shape (N, 2), got {pts.shape}")

    ax.scatter(pts[:, 0], pts[:, 1], s=80, marker="x", label=label_all)

    if annotate:
        for i, cand in enumerate(candidates):
            p = np.asarray(cand.point, dtype=float).reshape(2)
            ax.annotate(
                f"{i}: s={cand.score:.3f}",
                (p[0], p[1]),
                textcoords="offset points",
                xytext=(6, 6),
            )

    if selected_point is not None:
        selected_point = np.asarray(selected_point, dtype=float).reshape(2)
        ax.scatter(
            [selected_point[0]],
            [selected_point[1]],
            s=140,
            marker="s",
            label=label_selected,
        )


def finalize_2d_axes(
    ax: plt.Axes,
    title: str = "",
    equal: bool = True,
    grid: bool = True,
) -> None:
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if equal:
        ax.set_aspect("equal", adjustable="box")
    if grid:
        ax.grid(True)

    ax.legend()