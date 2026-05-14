from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from primitive_manifold_planner.manifolds import ImplicitManifold
from primitive_manifold_planner.projection import project_newton
from primitive_manifold_planner.planning.local import constrained_interpolate


@dataclass
class RRTNode:
    """
    One node in the constrained RRT tree.
    """

    point: np.ndarray
    parent_index: int | None


@dataclass
class ConstrainedRRTResult:
    """
    Result of constrained RRT planning on a single manifold.
    """

    success: bool
    path: np.ndarray | None
    tree_points: np.ndarray
    iterations: int
    message: str


def _nearest_node_index(nodes: list[RRTNode], x: np.ndarray) -> int:
    x = np.asarray(x, dtype=float).reshape(-1)
    dists = [np.linalg.norm(node.point - x) for node in nodes]
    return int(np.argmin(dists))


def _reconstruct_path(nodes: list[RRTNode], goal_index: int) -> np.ndarray:
    path = []
    current = goal_index
    while current is not None:
        path.append(nodes[current].point.copy())
        current = nodes[current].parent_index
    path.reverse()
    return np.asarray(path)


def plan_constrained_rrt(
    manifold: ImplicitManifold,
    start_point: np.ndarray,
    goal_point: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    max_iters: int = 500,
    step_size: float = 0.15,
    goal_tol: float = 5e-2,
    projection_tol: float = 1e-10,
    projection_max_iters: int = 50,
    projection_damping: float = 1.0,
    goal_sample_rate: float = 0.1,
    rng: np.random.Generator | None = None,
) -> ConstrainedRRTResult:
    """
    Plan on a single implicit manifold using a projection-based constrained RRT.

    Parameters
    ----------
    manifold:
        The implicit manifold to plan on.
    start_point, goal_point:
        Points expected to lie on the manifold.
    bounds_min, bounds_max:
        Ambient-space sampling bounds.
    max_iters:
        Maximum number of RRT iterations.
    step_size:
        Local planner step size.
    goal_tol:
        Goal-reaching tolerance.
    projection_tol, projection_max_iters, projection_damping:
        Parameters for Newton projection.
    goal_sample_rate:
        Probability of sampling the goal directly.
    rng:
        Optional random generator.
    """
    start_point = manifold._coerce_point(start_point)
    goal_point = manifold._coerce_point(goal_point)

    if not manifold.is_valid(start_point, tol=1e-6):
        return ConstrainedRRTResult(
            success=False,
            path=None,
            tree_points=np.asarray([start_point]),
            iterations=0,
            message="Start point is not on the manifold.",
        )

    if not manifold.is_valid(goal_point, tol=1e-6):
        return ConstrainedRRTResult(
            success=False,
            path=None,
            tree_points=np.asarray([start_point]),
            iterations=0,
            message="Goal point is not on the manifold.",
        )

    dim = manifold.ambient_dim
    bounds_min = np.asarray(bounds_min, dtype=float).reshape(-1)
    bounds_max = np.asarray(bounds_max, dtype=float).reshape(-1)

    if bounds_min.shape[0] != dim or bounds_max.shape[0] != dim:
        return ConstrainedRRTResult(
            success=False,
            path=None,
            tree_points=np.asarray([start_point]),
            iterations=0,
            message="Sampling bounds dimension mismatch.",
        )

    if np.any(bounds_max <= bounds_min):
        return ConstrainedRRTResult(
            success=False,
            path=None,
            tree_points=np.asarray([start_point]),
            iterations=0,
            message="Invalid sampling bounds.",
        )

    if not (0.0 <= goal_sample_rate <= 1.0):
        return ConstrainedRRTResult(
            success=False,
            path=None,
            tree_points=np.asarray([start_point]),
            iterations=0,
            message="goal_sample_rate must lie in [0, 1].",
        )

    if rng is None:
        rng = np.random.default_rng()

    nodes: list[RRTNode] = [RRTNode(point=start_point.copy(), parent_index=None)]

    # quick direct-connect attempt
    direct = constrained_interpolate(
        manifold=manifold,
        x_start=start_point,
        x_goal=goal_point,
        step_size=step_size,
        goal_tol=goal_tol,
        max_iters=max_iters,
        projection_tol=projection_tol,
        projection_max_iters=projection_max_iters,
        projection_damping=projection_damping,
    )
    if direct.success:
        return ConstrainedRRTResult(
            success=True,
            path=direct.path,
            tree_points=np.asarray([start_point, goal_point]),
            iterations=0,
            message="Direct constrained connection succeeded.",
        )

    for it in range(max_iters):
        if rng.uniform() < goal_sample_rate:
            x_rand = goal_point.copy()
        else:
            x_ambient = rng.uniform(bounds_min, bounds_max)
            proj = project_newton(
                manifold=manifold,
                x0=x_ambient,
                tol=projection_tol,
                max_iters=projection_max_iters,
                damping=projection_damping,
            )
            if not proj.success:
                continue
            x_rand = proj.x_projected

        nearest_idx = _nearest_node_index(nodes, x_rand)
        x_near = nodes[nearest_idx].point

        local = constrained_interpolate(
            manifold=manifold,
            x_start=x_near,
            x_goal=x_rand,
            step_size=step_size,
            goal_tol=step_size,
            max_iters=max(10, int(2 * max_iters / 10)),
            projection_tol=projection_tol,
            projection_max_iters=projection_max_iters,
            projection_damping=projection_damping,
        )

        if not local.success or local.path.shape[0] < 2:
            continue

        x_new = local.path[-1].copy()

        if np.linalg.norm(x_new - x_near) <= 1e-10:
            continue

        nodes.append(RRTNode(point=x_new, parent_index=nearest_idx))
        new_idx = len(nodes) - 1

        # try connecting new node to goal
        to_goal = constrained_interpolate(
            manifold=manifold,
            x_start=x_new,
            x_goal=goal_point,
            step_size=step_size,
            goal_tol=goal_tol,
            max_iters=max_iters,
            projection_tol=projection_tol,
            projection_max_iters=projection_max_iters,
            projection_damping=projection_damping,
        )

        if to_goal.success:
            prefix = _reconstruct_path(nodes, new_idx)
            if prefix.shape[0] > 0 and np.linalg.norm(prefix[-1] - to_goal.path[0]) <= 1e-10:
                full_path = np.vstack([prefix, to_goal.path[1:]])
            else:
                full_path = np.vstack([prefix, to_goal.path])

            return ConstrainedRRTResult(
                success=True,
                path=full_path,
                tree_points=np.asarray([node.point for node in nodes]),
                iterations=it + 1,
                message="Constrained RRT reached the goal.",
            )

    return ConstrainedRRTResult(
        success=False,
        path=None,
        tree_points=np.asarray([node.point for node in nodes]),
        iterations=max_iters,
        message="Constrained RRT reached max iterations without finding a path.",
    )