"""Graph-path utilities for the continuous-transfer experiment.

This module owns graph-centric route queries such as shortest-path extraction.
It stays intentionally small so planner orchestration can depend on path queries
without also importing route scoring or geometry shaping code.
"""

from __future__ import annotations

import heapq

from .graph_types import FamilyConnectivityGraph


def _shortest_path_with_constraints(
    graph: FamilyConnectivityGraph,
    start_node_id: int,
    goal_node_id: int,
    *,
    edge_allowed=None,
    banned_nodes: set[int] | None = None,
    banned_edges: set[int] | None = None,
) -> tuple[float, list[int], list[int]]:
    banned_nodes = set() if banned_nodes is None else {int(node_id) for node_id in banned_nodes}
    banned_edges = set() if banned_edges is None else {int(edge_id) for edge_id in banned_edges}
    if int(start_node_id) in banned_nodes or int(goal_node_id) in banned_nodes:
        return float("inf"), [], []
    dist = {int(start_node_id): 0.0}
    parent_node: dict[int, int] = {}
    parent_edge: dict[int, int] = {}
    queue: list[tuple[float, int]] = [(0.0, int(start_node_id))]
    while queue:
        current_dist, node_id = heapq.heappop(queue)
        if current_dist > dist.get(node_id, float("inf")) + 1e-12:
            continue
        if node_id == int(goal_node_id):
            break
        for edge_id in graph.adjacency.get(node_id, []):
            edge_id = int(edge_id)
            if edge_id in banned_edges:
                continue
            edge = graph.edges[edge_id]
            if edge_allowed is not None and not bool(edge_allowed(edge)):
                continue
            other = edge.node_v if edge.node_u == node_id else edge.node_u
            other = int(other)
            if other in banned_nodes:
                continue
            new_dist = float(current_dist + edge.cost)
            if new_dist + 1e-12 >= dist.get(other, float("inf")):
                continue
            dist[other] = new_dist
            parent_node[other] = node_id
            parent_edge[other] = edge_id
            heapq.heappush(queue, (new_dist, other))
    if int(goal_node_id) not in dist:
        return float("inf"), [], []
    node_path = [int(goal_node_id)]
    edge_path: list[int] = []
    cursor = int(goal_node_id)
    while cursor != int(start_node_id):
        edge_id = parent_edge[cursor]
        prev = parent_node[cursor]
        edge_path.append(edge_id)
        node_path.append(prev)
        cursor = prev
    node_path.reverse()
    edge_path.reverse()
    return float(dist[int(goal_node_id)]), node_path, edge_path


def shortest_path_over_graph(
    graph: FamilyConnectivityGraph,
    start_node_id: int,
    goal_node_id: int,
    edge_allowed=None,
) -> tuple[float, list[int], list[int]]:
    return _shortest_path_with_constraints(
        graph,
        start_node_id,
        goal_node_id,
        edge_allowed=edge_allowed,
    )


def k_shortest_simple_paths_over_graph(
    graph: FamilyConnectivityGraph,
    start_node_id: int,
    goal_node_id: int,
    *,
    k: int,
    edge_allowed=None,
) -> list[tuple[float, list[int], list[int]]]:
    """Enumerate up to ``k`` simple start-to-goal routes using a Yen-style search."""

    max_k = max(0, int(k))
    if max_k == 0:
        return []
    first_cost, first_nodes, first_edges = shortest_path_over_graph(
        graph,
        start_node_id,
        goal_node_id,
        edge_allowed=edge_allowed,
    )
    if len(first_edges) == 0:
        return []
    accepted: list[tuple[float, list[int], list[int]]] = [
        (float(first_cost), list(first_nodes), list(first_edges))
    ]
    queued_candidates: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = []
    queued_keys: set[tuple[int, ...]] = {tuple(int(edge_id) for edge_id in first_edges)}

    while len(accepted) < max_k:
        previous_cost, previous_nodes, previous_edges = accepted[-1]
        del previous_cost
        generated_any = False
        for spur_index in range(len(previous_nodes) - 1):
            spur_node_id = int(previous_nodes[spur_index])
            root_node_path = [int(node_id) for node_id in previous_nodes[: spur_index + 1]]
            root_edge_path = [int(edge_id) for edge_id in previous_edges[:spur_index]]
            banned_nodes = {int(node_id) for node_id in root_node_path[:-1]}
            banned_edges: set[int] = set()
            for _, accepted_nodes, accepted_edges in accepted:
                if len(accepted_nodes) <= spur_index:
                    continue
                if [int(node_id) for node_id in accepted_nodes[: spur_index + 1]] != root_node_path:
                    continue
                if spur_index < len(accepted_edges):
                    banned_edges.add(int(accepted_edges[spur_index]))
            spur_cost, spur_nodes, spur_edges = _shortest_path_with_constraints(
                graph,
                spur_node_id,
                goal_node_id,
                edge_allowed=edge_allowed,
                banned_nodes=banned_nodes,
                banned_edges=banned_edges,
            )
            if len(spur_edges) == 0:
                continue
            root_cost = float(sum(graph.edges[int(edge_id)].cost for edge_id in root_edge_path))
            total_node_path = root_node_path[:-1] + [int(node_id) for node_id in spur_nodes]
            total_edge_path = root_edge_path + [int(edge_id) for edge_id in spur_edges]
            path_key = tuple(int(edge_id) for edge_id in total_edge_path)
            if len(path_key) == 0 or path_key in queued_keys:
                continue
            heapq.heappush(
                queued_candidates,
                (
                    float(root_cost + spur_cost),
                    tuple(total_node_path),
                    path_key,
                ),
            )
            queued_keys.add(path_key)
            generated_any = True
        if not queued_candidates and not generated_any:
            break
        while queued_candidates:
            candidate_cost, candidate_nodes, candidate_edges = heapq.heappop(queued_candidates)
            if any(tuple(existing_edges) == candidate_edges for _, _, existing_edges in accepted):
                continue
            accepted.append(
                (
                    float(candidate_cost),
                    [int(node_id) for node_id in candidate_nodes],
                    [int(edge_id) for edge_id in candidate_edges],
                )
            )
            break
        else:
            break
    return accepted
