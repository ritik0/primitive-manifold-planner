from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import heapq
import numpy as np

from primitive_manifold_planner.transitions.leaf_transition import find_leaf_transition


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


def build_leaf_graph(
    families,
    project_newton,
    seed_points_fn=None,
    goal_point: Optional[np.ndarray] = None,
    max_candidates_per_pair: int = 3,
):
    """
    Build a graph over all sampled leaves.

    Important upgrade:
    if two leaves have multiple transition candidates, we keep several of them.
    Each candidate becomes its own directed edge.
    """
    graph = LeafGraph()
    nodes = enumerate_leaf_nodes(families)

    for i, (fam_a, lam_a, key_a) in enumerate(nodes):
        graph.add_node(key_a)

        for j in range(i + 1, len(nodes)):
            fam_b, lam_b, key_b = nodes[j]

            if seed_points_fn is None:
                seeds = [
                    np.array([0.0, 0.0], dtype=float),
                    np.array([1.0, 0.0], dtype=float),
                    np.array([-1.0, 0.0], dtype=float),
                    np.array([0.0, 1.0], dtype=float),
                    np.array([0.0, -1.0], dtype=float),
                    np.array([0.5, 0.5], dtype=float),
                    np.array([-0.5, -0.5], dtype=float),
                ]
            else:
                seeds = seed_points_fn(fam_a, lam_a, fam_b, lam_b)

            result = find_leaf_transition(
                source_family=fam_a,
                source_lam=lam_a,
                target_family=fam_b,
                target_lam=lam_b,
                seeds=seeds,
                project_newton=project_newton,
                goal=goal_point,
            )

            if not result.success:
                continue

            kept = result.candidates[:max_candidates_per_pair]

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
    """
    Simple default cost:
      - 1.0 per switch
      - plus transition score
    """
    return 1.0 + edge.score


def shortest_leaf_route(
    graph: LeafGraph,
    start: LeafKey,
    goal: LeafKey,
    edge_cost_fn: Optional[Callable[[LeafGraphEdge], float]] = None,
) -> Optional[List[LeafGraphEdge]]:
    """
    Dijkstra shortest path over the leaf graph.
    """
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
            step_cost = float(edge_cost_fn(edge))
            new_cost = g_cost + step_cost

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