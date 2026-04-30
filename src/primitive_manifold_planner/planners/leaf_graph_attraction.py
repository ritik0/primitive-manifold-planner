from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import heapq
import numpy as np

from primitive_manifold_planner.transitions.attraction_sampler import (
    discover_transition_candidates_via_attraction,
)


LeafKey = Tuple[str, object]


@dataclass
class LeafGraphEdge:
    src: LeafKey
    dst: LeafKey
    transition_point: np.ndarray
    score: float
    candidate_index: int = 0


@dataclass
class LeafGraph:
    adjacency: Dict[LeafKey, List[LeafGraphEdge]] = field(default_factory=dict)

    def add_node(self, node: LeafKey) -> None:
        self.adjacency.setdefault(node, [])

    def add_edge(self, edge: LeafGraphEdge) -> None:
        self.add_node(edge.src)
        self.add_node(edge.dst)
        self.adjacency[edge.src].append(edge)

    def neighbors(self, node: LeafKey) -> List[LeafGraphEdge]:
        return self.adjacency.get(node, [])


def enumerate_leaf_nodes(families) -> List[Tuple[object, object, LeafKey]]:
    nodes = []
    for fam in families:
        for lam in fam.sample_lambdas():
            nodes.append((fam, lam, (fam.name, lam)))
    return nodes


def build_leaf_graph_via_attraction(
    families,
    project_newton,
    current_point_map: Optional[dict] = None,
    goal_point: Optional[np.ndarray] = None,
    max_candidates_per_pair: int = 3,
    ambient_local_radius: float = 0.6,
    n_local_samples: int = 35,
    n_goal_bias_samples: int = 25,
    goal_band_width: float = 0.25,
    target_residual_threshold: float = 0.25,
    rng: Optional[np.random.Generator] = None,
):
    """
    Build a leaf graph using attraction-based transition discovery.

    current_point_map:
        Optional dict mapping leaf key -> representative point on that leaf.
        If not provided, we fall back to a crude default representative point.
    """
    if rng is None:
        rng = np.random.default_rng()

    graph = LeafGraph()
    nodes = enumerate_leaf_nodes(families)

    def representative_point(fam, lam):
        key = (fam.name, lam)
        if current_point_map is not None and key in current_point_map:
            return np.asarray(current_point_map[key], dtype=float)

        # crude generic representative points
        leaf = fam.manifold(lam)
        if hasattr(leaf, "point"):
            return np.asarray(leaf.point, dtype=float)
        if hasattr(leaf, "center") and hasattr(leaf, "radius"):
            return np.asarray(leaf.center, dtype=float) + np.array([leaf.radius, 0.0], dtype=float)
        return np.zeros(2, dtype=float)

    for i, (fam_a, lam_a, key_a) in enumerate(nodes):
        graph.add_node(key_a)

        for j in range(i + 1, len(nodes)):
            fam_b, lam_b, key_b = nodes[j]

            x_rep = representative_point(fam_a, lam_a)

            attr_result = discover_transition_candidates_via_attraction(
                source_family=fam_a,
                source_lam=lam_a,
                target_family=fam_b,
                target_lam=lam_b,
                current_x=x_rep,
                goal_x=goal_point,
                project_newton=project_newton,
                ambient_local_radius=ambient_local_radius,
                n_local_samples=n_local_samples,
                n_goal_bias_samples=n_goal_bias_samples,
                goal_band_width=goal_band_width,
                target_residual_threshold=target_residual_threshold,
                rng=rng,
            )

            if not attr_result.success:
                # Try reverse direction too, because attraction is directional.
                x_rep_rev = representative_point(fam_b, lam_b)
                attr_result = discover_transition_candidates_via_attraction(
                    source_family=fam_b,
                    source_lam=lam_b,
                    target_family=fam_a,
                    target_lam=lam_a,
                    current_x=x_rep_rev,
                    goal_x=goal_point,
                    project_newton=project_newton,
                    ambient_local_radius=ambient_local_radius,
                    n_local_samples=n_local_samples,
                    n_goal_bias_samples=n_goal_bias_samples,
                    goal_band_width=goal_band_width,
                    target_residual_threshold=target_residual_threshold,
                    rng=rng,
                )

            if not attr_result.success:
                continue

            kept = attr_result.transition_result.candidates[:max_candidates_per_pair]

            for k, cand in enumerate(kept):
                graph.add_edge(
                    LeafGraphEdge(
                        src=key_a,
                        dst=key_b,
                        transition_point=cand.x.copy(),
                        score=float(cand.score),
                        candidate_index=k,
                    )
                )
                graph.add_edge(
                    LeafGraphEdge(
                        src=key_b,
                        dst=key_a,
                        transition_point=cand.x.copy(),
                        score=float(cand.score),
                        candidate_index=k,
                    )
                )

    return graph


def default_edge_cost(edge: LeafGraphEdge) -> float:
    return 1.0 + edge.score


def shortest_leaf_route(
    graph: LeafGraph,
    start: LeafKey,
    goal: LeafKey,
    edge_cost_fn: Optional[Callable[[LeafGraphEdge], float]] = None,
):
    if edge_cost_fn is None:
        edge_cost_fn = default_edge_cost

    pq = []
    heapq.heappush(pq, (0.0, start))

    dist = {start: 0.0}
    parent = {}
    parent_edge = {}

    while pq:
        g_cost, node = heapq.heappop(pq)

        if node == goal:
            break

        if g_cost > dist.get(node, float("inf")):
            continue

        for edge in graph.neighbors(node):
            new_cost = g_cost + float(edge_cost_fn(edge))
            if new_cost < dist.get(edge.dst, float("inf")):
                dist[edge.dst] = new_cost
                parent[edge.dst] = node
                parent_edge[edge.dst] = edge
                heapq.heappush(pq, (new_cost, edge.dst))

    if start == goal:
        return []

    if goal not in parent_edge:
        return None

    route = []
    cur = goal
    while cur != start:
        edge = parent_edge[cur]
        route.append(edge)
        cur = parent[cur]
    route.reverse()
    return route