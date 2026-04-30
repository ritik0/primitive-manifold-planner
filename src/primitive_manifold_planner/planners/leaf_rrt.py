from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from primitive_manifold_planner.planning.local import run_local_planner


@dataclass
class LeafRRTNode:
    x: np.ndarray
    parent: Optional[int] = None


@dataclass
class LeafRRTResult:
    success: bool
    path: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    message: str = ""
    nodes: List[LeafRRTNode] = field(default_factory=list)


def _backtrack_path(nodes: List[LeafRRTNode], idx: int) -> np.ndarray:
    pts = []
    cur = idx
    while cur is not None:
        pts.append(nodes[cur].x.copy())
        cur = nodes[cur].parent
    pts.reverse()
    return np.asarray(pts)


def _concat_paths(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0:
        return np.asarray(b)
    if len(b) == 0:
        return np.asarray(a)
    return np.vstack([a, b[1:]])


def plan_on_leaf_rrt(
    manifold,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    project_newton,
    bounds: Tuple[np.ndarray, np.ndarray],
    step_size: float = 0.10,
    max_nodes: int = 250,
    goal_bias: float = 0.20,
    local_planner_name: str = "projection",
    local_planner_kwargs: Optional[dict] = None,
):
    local_planner_kwargs = dict(local_planner_kwargs or {})
    lower, upper = bounds
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    # Quick direct connection first
    direct = run_local_planner(
        manifold=manifold,
        x_start=np.asarray(x_start, dtype=float),
        x_goal=np.asarray(x_goal, dtype=float),
        planner_name=local_planner_name,
        step_size=step_size,
        **local_planner_kwargs,
    )
    if direct.success and len(direct.path) > 0:
        return LeafRRTResult(
            success=True,
            path=np.asarray(direct.path),
            message="Solved directly without growing a tree.",
            nodes=[LeafRRTNode(np.asarray(x_start, dtype=float), None)],
        )

    nodes = [LeafRRTNode(np.asarray(x_start, dtype=float), None)]

    for _ in range(max_nodes):
        if np.random.rand() < goal_bias:
            x_rand = np.asarray(x_goal, dtype=float).copy()
        else:
            x_rand = np.random.uniform(lower, upper)

        proj = project_newton(
            manifold=manifold,
            x0=np.asarray(x_rand, dtype=float),
            tol=1e-10,
            max_iters=50,
            damping=1.0,
        )
        if not proj.success:
            continue

        x_proj = np.asarray(proj.x_projected, dtype=float)

        dists = [float(np.linalg.norm(n.x - x_proj)) for n in nodes]
        i_near = int(np.argmin(dists))
        x_near = nodes[i_near].x

        extend = run_local_planner(
            manifold=manifold,
            x_start=x_near,
            x_goal=x_proj,
            planner_name=local_planner_name,
            step_size=step_size,
            **local_planner_kwargs,
        )
        if not extend.success or len(extend.path) == 0:
            continue

        x_new = np.asarray(extend.path[-1], dtype=float)
        nodes.append(LeafRRTNode(x=x_new, parent=i_near))
        i_new = len(nodes) - 1

        to_goal = run_local_planner(
            manifold=manifold,
            x_start=x_new,
            x_goal=np.asarray(x_goal, dtype=float),
            planner_name=local_planner_name,
            step_size=step_size,
            **local_planner_kwargs,
        )
        if to_goal.success and len(to_goal.path) > 0:
            tree_path = _backtrack_path(nodes, i_new)
            full_path = _concat_paths(tree_path, np.asarray(to_goal.path))
            return LeafRRTResult(
                success=True,
                path=full_path,
                message="Solved with leaf-RRT.",
                nodes=nodes,
            )

    return LeafRRTResult(
        success=False,
        path=np.zeros((0, len(np.asarray(x_start).reshape(-1)))),
        message="Leaf-RRT failed to connect start to goal on the chosen leaf.",
        nodes=nodes,
    )