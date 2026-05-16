from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def forward_kinematics(theta: np.ndarray, link1: float, link2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map theta=[theta1, theta2] to planar robot points in task space.

    Returns the base, elbow, and end-effector points. The end-effector is the
    point that the task-space constraint h(x,y)=0 is evaluated on.
    """

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
    """Task-space circle constraint h_circle(x,y)=0."""

    diff = np.asarray(xy, dtype=float) - center.reshape(1, 1, 2)
    return diff[..., 0] ** 2 + diff[..., 1] ** 2 - float(radius) ** 2


def line_residual(xy: np.ndarray, normal: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Task-space line constraint h_line(x,y)=0."""

    diff = np.asarray(xy, dtype=float) - point.reshape(1, 1, 2)
    return diff[..., 0] * normal[0] + diff[..., 1] * normal[1]


def residual_from_theta(
    theta: np.ndarray,
    *,
    constraint: str,
    link1: float,
    link2: float,
) -> np.ndarray:
    """Pull the task-space residual back through FK as F(theta)=h(FK(theta))."""

    _base, _elbow, ee = forward_kinematics(theta, link1, link2)
    if constraint == "circle":
        return circle_residual(ee, np.asarray([0.80, 0.35], dtype=float), 0.45)
    normal = np.asarray([1.0, 0.35], dtype=float)
    normal = normal / np.linalg.norm(normal)
    return line_residual(ee, normal, np.asarray([0.55, 0.20], dtype=float))


def extract_zero_contour(theta1: np.ndarray, theta2: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Extract F(theta)=0 contours and return the longest branch."""

    fig, ax = plt.subplots()
    try:
        contour = ax.contour(theta1, theta2, residual, levels=[0.0])
        branches = [np.asarray(seg, dtype=float) for seg in contour.allsegs[0] if len(seg) >= 2]
    finally:
        plt.close(fig)
    if not branches:
        raise RuntimeError("No F(theta)=0 contour branch found. Try a denser grid or different constraint.")
    lengths = []
    for branch in branches:
        deltas = np.diff(branch, axis=0)
        lengths.append(float(np.sum(np.linalg.norm(deltas, axis=1))))
    return branches[int(np.argmax(lengths))], branches


def path_from_branch(branch: np.ndarray) -> np.ndarray:
    """Choose a simple start-to-goal theta path along one connected contour branch."""

    if len(branch) < 8:
        return branch
    start_idx = max(0, int(0.12 * (len(branch) - 1)))
    goal_idx = min(len(branch) - 1, int(0.88 * (len(branch) - 1)))
    if goal_idx <= start_idx:
        return branch
    return np.asarray(branch[start_idx : goal_idx + 1], dtype=float)


def draw_robot(ax, theta: np.ndarray, link1: float, link2: float, *, color: str, alpha: float = 0.55) -> np.ndarray:
    """Draw one robot pose and return its end-effector point."""

    base, elbow, ee = forward_kinematics(np.asarray(theta, dtype=float), link1, link2)
    points = np.vstack([base, elbow, ee])
    ax.plot(points[:, 0], points[:, 1], color=color, linewidth=1.6, alpha=alpha)
    ax.scatter(points[-1, 0], points[-1, 1], color=color, s=18, alpha=min(1.0, alpha + 0.25))
    return np.asarray(points[-1], dtype=float)


def draw_task_constraint(ax, constraint: str, reach: float) -> None:
    """Draw the task-space h(x,y)=0 set used by the pullback."""

    if constraint == "circle":
        center = np.asarray([0.80, 0.35], dtype=float)
        radius = 0.45
        angles = np.linspace(0.0, 2.0 * np.pi, 360)
        curve = center + radius * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        ax.plot(curve[:, 0], curve[:, 1], color="#2563eb", linewidth=2.5, label="h_circle(x,y)=0")
        return
    normal = np.asarray([1.0, 0.35], dtype=float)
    normal = normal / np.linalg.norm(normal)
    point = np.asarray([0.55, 0.20], dtype=float)
    tangent = np.asarray([-normal[1], normal[0]], dtype=float)
    values = np.linspace(-reach, reach, 2)
    line = point.reshape(1, 2) + values.reshape(-1, 1) * tangent.reshape(1, 2)
    ax.plot(line[:, 0], line[:, 1], color="#2563eb", linewidth=2.5, label="h_line(x,y)=0")


def build_figure(args, theta1: np.ndarray, theta2: np.ndarray, residual: np.ndarray, branch: np.ndarray, path: np.ndarray):
    """Create the three-panel educational figure."""

    link1 = float(args.link1)
    link2 = float(args.link2)
    reach = link1 + link2
    _base, _elbow, fk_path = forward_kinematics(path, link1, link2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.7), constrained_layout=True)
    ax_task, ax_residual, ax_path = axes

    # Panel 1: FK maps theta states to task-space robot poses.
    draw_task_constraint(ax_task, args.constraint, reach)
    workspace = plt.Circle((0.0, 0.0), reach, color="#94a3b8", fill=False, linestyle=":", linewidth=1.3)
    ax_task.add_patch(workspace)
    pose_indices = np.linspace(0, len(path) - 1, 8, dtype=int)
    for idx in pose_indices:
        draw_robot(ax_task, path[idx], link1, link2, color="#ea580c", alpha=0.38)
    ax_task.scatter(fk_path[pose_indices, 0], fk_path[pose_indices, 1], color="#111827", s=20, label="FK(theta) on constraint")
    ax_task.set_aspect("equal", adjustable="box")
    ax_task.set_xlim(-reach - 0.15, reach + 0.15)
    ax_task.set_ylim(-reach - 0.15, reach + 0.15)
    ax_task.set_xlabel("x")
    ax_task.set_ylabel("y")
    ax_task.set_title("Task space: constraint h(x,y)=0")
    ax_task.legend(loc="upper right", fontsize=8)

    # Panel 2: F(theta)=h(FK(theta)) is the task constraint pulled back into C-space.
    image = ax_residual.imshow(
        residual,
        origin="lower",
        extent=[theta1[0], theta1[-1], theta2[0], theta2[-1]],
        cmap="coolwarm",
        aspect="auto",
    )
    zero = ax_residual.contour(theta1, theta2, residual, levels=[0.0], colors="black", linewidths=2.2)
    ax_residual.clabel(zero, fmt={0.0: "C-space manifold F(theta)=0"}, inline=True, fontsize=8)
    ax_residual.set_xlabel("theta1 [rad]")
    ax_residual.set_ylabel("theta2 [rad]")
    ax_residual.set_title("C-space residual F(theta)=h(FK(theta))")
    fig.colorbar(image, ax=ax_residual, fraction=0.046, pad=0.04, label="F(theta)")

    # Panel 3: one scalar equality in 2D C-space gives a 1D curve, and the path follows that curve.
    ax_path.contour(theta1, theta2, residual, levels=[0.0], colors="#94a3b8", linewidths=1.2)
    ax_path.plot(branch[:, 0], branch[:, 1], color="#64748b", linewidth=1.1, alpha=0.75, label="chosen contour branch")
    ax_path.plot(path[:, 0], path[:, 1], color="#dc2626", linewidth=3.0, label="theta path on F(theta)=0")
    ax_path.scatter(path[0, 0], path[0, 1], color="black", s=42, label="start theta", zorder=5)
    ax_path.scatter(path[-1, 0], path[-1, 1], color="#facc15", edgecolors="black", s=52, label="goal theta", zorder=5)
    ax_path.set_xlabel("theta1 [rad]")
    ax_path.set_ylabel("theta2 [rad]")
    ax_path.set_title("Path on C-space manifold")
    ax_path.legend(loc="upper right", fontsize=8)

    inset = ax_path.inset_axes([0.56, 0.08, 0.40, 0.36])
    draw_task_constraint(inset, args.constraint, reach)
    inset.plot(fk_path[:, 0], fk_path[:, 1], color="#dc2626", linewidth=2.2, label="FK(theta path)")
    inset.scatter(fk_path[0, 0], fk_path[0, 1], color="black", s=22)
    inset.scatter(fk_path[-1, 0], fk_path[-1, 1], color="#facc15", edgecolors="black", s=28)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xlim(np.min(fk_path[:, 0]) - 0.35, np.max(fk_path[:, 0]) + 0.35)
    inset.set_ylim(np.min(fk_path[:, 1]) - 0.35, np.max(fk_path[:, 1]) + 0.35)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("FK image", fontsize=9)

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 67: 2D planar FK pullback from a task-space constraint to a C-space curve.",
    )
    parser.add_argument("--constraint", choices=("circle", "line"), default="circle")
    parser.add_argument("--grid-res", type=int, default=400)
    parser.add_argument("--link1", type=float, default=1.0)
    parser.add_argument("--link2", type=float, default=0.8)
    parser.add_argument("--save-figure", type=str, default="outputs/example_67_planar_pullback.png")
    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument("--show", dest="show", action="store_true", default=True)
    show_group.add_argument("--no-show", dest="show", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid_res = int(max(80, args.grid_res))
    theta1 = np.linspace(-np.pi, np.pi, grid_res)
    theta2 = np.linspace(-np.pi, np.pi, grid_res)
    theta1_grid, theta2_grid = np.meshgrid(theta1, theta2, indexing="xy")
    theta_grid = np.stack([theta1_grid, theta2_grid], axis=-1)

    # F(theta)=h(FK(theta)) is not manually drawn. It is evaluated on a grid,
    # then matplotlib extracts the zero contour of the actual residual.
    residual = residual_from_theta(
        theta_grid,
        constraint=str(args.constraint),
        link1=float(args.link1),
        link2=float(args.link2),
    )
    branch, branches = extract_zero_contour(theta1, theta2, residual)
    path = path_from_branch(branch)
    path_residual = residual_from_theta(
        path.reshape(1, -1, 2),
        constraint=str(args.constraint),
        link1=float(args.link1),
        link2=float(args.link2),
    ).reshape(-1)

    formula = (
        "h_circle(x,y)=(x-cx)^2+(y-cy)^2-r^2"
        if args.constraint == "circle"
        else "h_line(x,y)=n dot ([x,y]-p0)"
    )
    print("=== Example 67: 2D FK pullback demo ===")
    print("robot_dof                         : 2")
    print(f"task_constraint                   : {args.constraint}")
    print(f"task_constraint_formula            : {formula}")
    print("cspace_residual                    : F(theta)=h(FK(theta))")
    print("cspace_dimension_before_constraint : 2")
    print("constraint_codimension             : 1")
    print("expected_manifold_dimension        : 1")
    print(f"grid_resolution                    : {grid_res}")
    print(f"zero_contour_branches              : {len(branches)}")
    print(f"zero_contour_points                : {len(branch)}")
    print(f"selected_path_points               : {len(path)}")
    print(f"max_path_residual                  : {float(np.max(np.abs(path_residual))):.6e}")
    print(f"mean_path_residual                 : {float(np.mean(np.abs(path_residual))):.6e}")
    print("explanation                        : task-space constraint is pulled back through FK; no artificial C-space curve is drawn")

    fig = build_figure(args, theta1, theta2, residual, branch, path)
    save_path = Path(args.save_figure)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    print(f"figure_saved                       : {save_path}")
    if bool(args.show):
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
