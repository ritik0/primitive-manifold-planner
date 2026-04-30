from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np

from primitive_manifold_planner.planning.local import run_local_planner
from primitive_manifold_planner.planners.admissibility import family_transition_feasibility
from primitive_manifold_planner.planners.mode_semantics import PlanningSemanticContext, PlanningSemanticModel
from primitive_manifold_planner.projection import project_newton
from primitive_manifold_planner.transitions.leaf_transition import find_leaf_transition


@dataclass
class LazyTreeNode:
    family_name: str
    lam: float
    x: np.ndarray
    parent_index: Optional[int]
    edge_kind: str = "move"  # move or switch
    edge_path: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    planner_name: str = "projection"
    chart_count: int = 0


@dataclass
class LazyMultimodalPlanResult:
    success: bool
    message: str
    nodes: list[LazyTreeNode]
    goal_node_index: Optional[int] = None
    path: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    switch_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    chart_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    route_string: str = ""
    iterations: int = 0


def _family_map(families):
    return {fam.name: fam for fam in families}


def _coerce_path(path: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(path, dtype=float)
    if arr.size == 0:
        return np.zeros((0, dim))
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _append_unique_point(points: list[np.ndarray], q: np.ndarray, tol: float = 1e-8) -> None:
    q = np.asarray(q, dtype=float)
    if any(np.linalg.norm(q - p) <= tol for p in points):
        return
    points.append(q.copy())


def _node_signature(node: LazyTreeNode, x_round: int = 4) -> tuple:
    q = tuple(np.round(np.asarray(node.x, dtype=float), x_round))
    return (str(node.family_name), float(node.lam), q)


def _try_project_to_leaf(fam, lam: float, x0: np.ndarray) -> Optional[np.ndarray]:
    proj = project_newton(
        manifold=fam.manifold(lam),
        x0=np.asarray(x0, dtype=float),
        tol=1e-10,
        max_iters=60,
        damping=1.0,
    )
    if not proj.success:
        return None
    return np.asarray(proj.x_projected, dtype=float)


def _allowed_switch(
    semantic_model: Optional[PlanningSemanticModel],
    source_family_name: str,
    source_lam: float,
    target_family_name: str,
    target_lam: float,
    point: np.ndarray,
    goal_point: np.ndarray,
) -> bool:
    if semantic_model is None:
        return True
    context = PlanningSemanticContext(
        source_family_name=str(source_family_name),
        source_lam=float(source_lam),
        target_family_name=str(target_family_name),
        target_lam=float(target_lam),
        point=np.asarray(point, dtype=float),
        goal_point=np.asarray(goal_point, dtype=float),
        metadata={"planner": "lazy_multimodal_rrt"},
    )
    return bool(semantic_model.transition_allowed(context) and semantic_model.transition_feasible(context))


def _switch_cost(
    semantic_model: Optional[PlanningSemanticModel],
    source_family_name: str,
    source_lam: float,
    target_family_name: str,
    target_lam: float,
    point: np.ndarray,
    goal_point: np.ndarray,
) -> float:
    if semantic_model is None:
        return 0.0
    context = PlanningSemanticContext(
        source_family_name=str(source_family_name),
        source_lam=float(source_lam),
        target_family_name=str(target_family_name),
        target_lam=float(target_lam),
        point=np.asarray(point, dtype=float),
        goal_point=np.asarray(goal_point, dtype=float),
        metadata={"planner": "lazy_multimodal_rrt"},
    )
    return float(
        semantic_model.transition_cost(context)
        + semantic_model.transition_admissibility_cost(context)
    )


def _backtrack_node_indices(nodes: list[LazyTreeNode], idx: int) -> list[int]:
    out = []
    cur = idx
    while cur is not None:
        out.append(cur)
        cur = nodes[cur].parent_index
    out.reverse()
    return out


def reconstruct_path(nodes: list[LazyTreeNode], goal_idx: int, ambient_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    indices = _backtrack_node_indices(nodes, goal_idx)
    full_path: list[np.ndarray] = []
    switch_points: list[np.ndarray] = []
    chart_points: list[np.ndarray] = []
    route_parts: list[str] = []

    for k, idx in enumerate(indices):
        node = nodes[idx]
        route_label = f"{node.family_name}[{node.lam:g}]"
        if not route_parts or route_parts[-1] != route_label:
            route_parts.append(route_label)

        if node.edge_kind == "switch":
            _append_unique_point(switch_points, node.x)
            _append_unique_point(chart_points, node.x)

        edge_path = _coerce_path(node.edge_path, ambient_dim)
        if edge_path.shape[0] > 0:
            for q in edge_path:
                _append_unique_point(chart_points, q)
            if len(full_path) == 0:
                full_path.extend(list(edge_path))
            else:
                full_path.extend(list(edge_path[1:]))
        elif k == 0:
            full_path.append(np.asarray(node.x, dtype=float).copy())

    return (
        np.asarray(full_path, dtype=float) if full_path else np.zeros((0, ambient_dim)),
        np.asarray(switch_points, dtype=float) if switch_points else np.zeros((0, ambient_dim)),
        np.asarray(chart_points, dtype=float) if chart_points else np.zeros((0, ambient_dim)),
        " -> ".join(route_parts),
    )


def plan_lazy_multimodal_rrt(
    families,
    start_state,
    goal_family_name: str,
    goal_lam: float,
    goal_q: np.ndarray,
    *,
    semantic_model: Optional[PlanningSemanticModel] = None,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    max_iters: int = 350,
    goal_bias: float = 0.20,
    switch_goal_bias: float = 0.35,
    local_planner_name: str = "atlas_like",
    local_planner_kwargs: Optional[dict] = None,
    step_size: float = 0.10,
    switch_projection_tol: float = 0.08,
    min_new_state_dist: float = 0.06,
) -> LazyMultimodalPlanResult:
    family_map = _family_map(families)
    ambient_dim = int(np.asarray(start_state.x, dtype=float).reshape(-1).shape[0])
    local_planner_kwargs = dict(local_planner_kwargs or {})
    lower = np.asarray(bounds_min, dtype=float)
    upper = np.asarray(bounds_max, dtype=float)

    nodes = [
        LazyTreeNode(
            family_name=str(start_state.family_name),
            lam=float(start_state.lam),
            x=np.asarray(start_state.x, dtype=float).copy(),
            parent_index=None,
            edge_kind="move",
            edge_path=np.asarray([start_state.x], dtype=float),
            planner_name=local_planner_name,
            chart_count=1,
        )
    ]
    visited = {_node_signature(nodes[0])}
    explored_switch_points: list[np.ndarray] = []
    explored_chart_points: list[np.ndarray] = [np.asarray(start_state.x, dtype=float).copy()]

    goal_family = family_map[str(goal_family_name)]
    goal_leaf = goal_family.manifold(float(goal_lam))

    for it in range(max_iters):
        node_idx = int(np.random.randint(0, len(nodes)))
        node = nodes[node_idx]
        current_family = family_map[node.family_name]
        current_leaf = current_family.manifold(node.lam)

        if np.random.rand() < goal_bias:
            ambient_target = np.asarray(goal_q, dtype=float).copy()
        else:
            ambient_target = np.random.uniform(lower, upper)

        projected_target = _try_project_to_leaf(current_family, node.lam, ambient_target)
        if projected_target is None:
            continue

        move = run_local_planner(
            manifold=current_leaf,
            x_start=node.x,
            x_goal=projected_target,
            planner_name=local_planner_name,
            step_size=step_size,
            **local_planner_kwargs,
        )
        if not move.success or len(move.path) < 2:
            continue

        for q in np.asarray(move.path, dtype=float):
            _append_unique_point(explored_chart_points, q)

        x_new = np.asarray(move.path[-1], dtype=float)
        if np.linalg.norm(x_new - node.x) < min_new_state_dist:
            continue

        new_node = LazyTreeNode(
            family_name=node.family_name,
            lam=node.lam,
            x=x_new.copy(),
            parent_index=node_idx,
            edge_kind="move",
            edge_path=np.asarray(move.path, dtype=float),
            planner_name=move.planner_name,
            chart_count=int(getattr(move, "chart_count", 0)),
        )
        sig = _node_signature(new_node)
        if sig in visited:
            continue
        visited.add(sig)
        nodes.append(new_node)
        new_idx = len(nodes) - 1

        if new_node.family_name == str(goal_family_name) and float(new_node.lam) == float(goal_lam):
            final_move = run_local_planner(
                manifold=goal_leaf,
                x_start=x_new,
                x_goal=np.asarray(goal_q, dtype=float),
                planner_name=local_planner_name,
                step_size=step_size,
                **local_planner_kwargs,
            )
            if final_move.success and len(final_move.path) > 0:
                goal_node = LazyTreeNode(
                    family_name=str(goal_family_name),
                    lam=float(goal_lam),
                    x=np.asarray(goal_q, dtype=float).copy(),
                    parent_index=new_idx,
                    edge_kind="move",
                    edge_path=np.asarray(final_move.path, dtype=float),
                    planner_name=final_move.planner_name,
                    chart_count=int(getattr(final_move, "chart_count", 0)),
                )
                nodes.append(goal_node)
                goal_idx = len(nodes) - 1
                path, switch_points, chart_points, route_string = reconstruct_path(nodes, goal_idx, ambient_dim)
                return LazyMultimodalPlanResult(
                    success=True,
                    message="Reached goal with lazy multimodal tree expansion.",
                    nodes=nodes,
                    goal_node_index=goal_idx,
                    path=path,
                    switch_points=switch_points,
                    chart_points=chart_points,
                    route_string=route_string,
                    iterations=it + 1,
                )

        candidate_order = list(families)
        if np.random.rand() < switch_goal_bias:
            candidate_order = sorted(
                families,
                key=lambda fam: 0.0 if fam.name == str(goal_family_name) else 1.0,
            )

        move_path = np.asarray(move.path, dtype=float)
        path_probe_indices = np.linspace(
            0,
            len(move_path) - 1,
            min(len(move_path), 6),
            dtype=int,
        )
        switch_found = False
        for probe_idx in path_probe_indices:
            x_probe = np.asarray(move_path[int(probe_idx)], dtype=float)
            probe_parent_idx = new_idx if probe_idx == len(move_path) - 1 else node_idx

            for fam in candidate_order:
                if fam.name == new_node.family_name:
                    continue
                for lam_raw in fam.sample_lambdas():
                    lam = float(lam_raw)
                    projected_switch = _try_project_to_leaf(fam, lam, x_probe)
                    if projected_switch is None:
                        continue
                    if np.linalg.norm(projected_switch - x_probe) > switch_projection_tol:
                        continue
                    transition_search = find_leaf_transition(
                        source_family=current_family,
                        source_lam=float(new_node.lam),
                        target_family=fam,
                        target_lam=lam,
                        seeds=[x_probe.copy(), projected_switch.copy()],
                        project_newton=project_newton,
                        goal=np.asarray(goal_q, dtype=float),
                    )
                    if not transition_search.success or len(transition_search.candidates) == 0:
                        continue

                    exact_switch = np.asarray(transition_search.candidates[0].x, dtype=float)

                    if not family_transition_feasibility(
                        current_family,
                        float(new_node.lam),
                        exact_switch,
                        goal_point=np.asarray(goal_q, dtype=float),
                        metadata={"planner": "lazy_multimodal_rrt", "role": "source"},
                    ):
                        continue
                    if not family_transition_feasibility(
                        fam,
                        lam,
                        exact_switch,
                        goal_point=np.asarray(goal_q, dtype=float),
                        metadata={"planner": "lazy_multimodal_rrt", "role": "target"},
                    ):
                        continue
                    if not _allowed_switch(
                        semantic_model,
                        new_node.family_name,
                        float(new_node.lam),
                        fam.name,
                        lam,
                        exact_switch,
                        np.asarray(goal_q, dtype=float),
                    ):
                        continue

                    to_exact_switch = run_local_planner(
                        manifold=current_leaf,
                        x_start=node.x,
                        x_goal=exact_switch,
                        planner_name=local_planner_name,
                        step_size=step_size,
                        **local_planner_kwargs,
                    )
                    if not to_exact_switch.success or len(to_exact_switch.path) == 0:
                        continue

                    for q in np.asarray(to_exact_switch.path, dtype=float):
                        _append_unique_point(explored_chart_points, q)
                    _append_unique_point(explored_switch_points, exact_switch)

                    probe_node = LazyTreeNode(
                        family_name=new_node.family_name,
                        lam=float(new_node.lam),
                        x=exact_switch.copy(),
                        parent_index=node_idx,
                        edge_kind="move",
                        edge_path=np.asarray(to_exact_switch.path, dtype=float),
                        planner_name=to_exact_switch.planner_name,
                        chart_count=int(getattr(to_exact_switch, "chart_count", 0)),
                    )
                    sig_probe = _node_signature(probe_node)
                    if sig_probe in visited:
                        continue
                    visited.add(sig_probe)
                    nodes.append(probe_node)
                    probe_idx_node = len(nodes) - 1

                    switch_node = LazyTreeNode(
                        family_name=fam.name,
                        lam=lam,
                        x=exact_switch.copy(),
                        parent_index=probe_idx_node,
                        edge_kind="switch",
                        edge_path=np.vstack([exact_switch.copy(), exact_switch.copy()]),
                        planner_name="lazy_switch",
                        chart_count=1,
                    )
                    sig_switch = _node_signature(switch_node)
                    if sig_switch in visited:
                        continue
                    visited.add(sig_switch)
                    nodes.append(switch_node)
                    switch_idx = len(nodes) - 1
                    switch_found = True

                    if fam.name == str(goal_family_name) and lam == float(goal_lam):
                        final_move = run_local_planner(
                            manifold=goal_leaf,
                            x_start=exact_switch,
                            x_goal=np.asarray(goal_q, dtype=float),
                            planner_name=local_planner_name,
                            step_size=step_size,
                            **local_planner_kwargs,
                        )
                        if final_move.success and len(final_move.path) > 0:
                            for q in np.asarray(final_move.path, dtype=float):
                                _append_unique_point(explored_chart_points, q)
                            goal_node = LazyTreeNode(
                                family_name=str(goal_family_name),
                                lam=float(goal_lam),
                                x=np.asarray(goal_q, dtype=float).copy(),
                                parent_index=switch_idx,
                                edge_kind="move",
                                edge_path=np.asarray(final_move.path, dtype=float),
                                planner_name=final_move.planner_name,
                                chart_count=int(getattr(final_move, "chart_count", 0)),
                            )
                            nodes.append(goal_node)
                            goal_idx = len(nodes) - 1
                            path, switch_points, chart_points, route_string = reconstruct_path(nodes, goal_idx, ambient_dim)
                            return LazyMultimodalPlanResult(
                                success=True,
                                message="Reached goal after lazily discovered manifold switch.",
                                nodes=nodes,
                                goal_node_index=goal_idx,
                                path=path,
                                switch_points=(
                                    np.asarray(explored_switch_points, dtype=float)
                                    if explored_switch_points
                                    else switch_points
                                ),
                                chart_points=(
                                    np.asarray(explored_chart_points, dtype=float)
                                    if explored_chart_points
                                    else chart_points
                                ),
                                route_string=route_string,
                                iterations=it + 1,
                            )
            if switch_found:
                break

    return LazyMultimodalPlanResult(
        success=False,
        message="Lazy multimodal tree reached max iterations without finding a route.",
        nodes=nodes,
        switch_points=np.asarray(explored_switch_points, dtype=float) if explored_switch_points else np.zeros((0, ambient_dim)),
        chart_points=np.asarray(explored_chart_points, dtype=float) if explored_chart_points else np.zeros((0, ambient_dim)),
        iterations=max_iters,
    )
