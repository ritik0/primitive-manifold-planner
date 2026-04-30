from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from primitive_manifold_planner.manifolds import SphereManifold, PlaneManifold


def plot_sphere_manifold(
    ax,
    sphere: SphereManifold,
    num_u: int = 50,
    num_v: int = 25,
    alpha: float = 0.25,
    label: str | None = None,
) -> None:
    """
    Plot a sphere surface in 3D.
    """
    u = np.linspace(0.0, 2.0 * np.pi, num_u)
    v = np.linspace(0.0, np.pi, num_v)

    uu, vv = np.meshgrid(u, v)

    x = sphere.center[0] + sphere.radius * np.cos(uu) * np.sin(vv)
    y = sphere.center[1] + sphere.radius * np.sin(uu) * np.sin(vv)
    z = sphere.center[2] + sphere.radius * np.cos(vv)

    ax.plot_surface(x, y, z, alpha=alpha)

    if label is not None:
        # dummy handle for legend
        ax.plot([], [], [], label=label)


def plot_plane_patch(
    ax,
    plane: PlaneManifold,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    num: int = 20,
    alpha: float = 0.25,
    label: str | None = None,
) -> None:
    """
    Plot a finite patch of a plane in 3D.

    This helper is currently specialized to planes whose normal has a
    nonzero z-component, so that z can be solved from x and y.
    """
    nx, ny, nz = plane.normal
    if abs(nz) < 1e-12:
        raise ValueError(
            "plot_plane_patch currently requires a plane with nonzero z normal component."
        )

    xs = np.linspace(xlim[0], xlim[1], num)
    ys = np.linspace(ylim[0], ylim[1], num)
    xx, yy = np.meshgrid(xs, ys)

    # Plane equation: n^T (x - p) = 0
    # nx(x-px) + ny(y-py) + nz(z-pz) = 0
    # => z = pz - [nx(x-px) + ny(y-py)] / nz
    px, py, pz = plane.point
    zz = pz - (nx * (xx - px) + ny * (yy - py)) / nz

    ax.plot_surface(xx, yy, zz, alpha=alpha)

    if label is not None:
        ax.plot([], [], [], label=label)


def plot_point_3d(
    ax,
    x: np.ndarray,
    label: str = "point",
    s: float = 60.0,
) -> None:
    """
    Plot a single point in 3D.
    """
    x = np.asarray(x, dtype=float).reshape(3)
    ax.scatter([x[0]], [x[1]], [x[2]], s=s, label=label)


def finalize_3d_axes(
    ax,
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    zlabel: str = "z",
) -> None:
    """
    Final formatting for a 3D plot.
    """
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.legend()