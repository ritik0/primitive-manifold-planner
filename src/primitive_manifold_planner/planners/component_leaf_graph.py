from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import heapq
import numpy as np

from primitive_manifold_planner.planners.transition_manager import TransitionGenerator


ComponentKey = Tuple[str, float, str]


@dataclass
class ComponentEdge:
    src: ComponentKey
    dst: ComponentKey
    transition_point: np.ndarray
    score: float
    candidate_index: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class ComponentGraph:
    adjacency: Dict[ComponentKey, List[ComponentEdge]] = field(default_factory=dict)

    def add_node(self, node: ComponentKey) -> None:
        self.adjacency.setdefault(node, [])

    def add_edge(self, edge: ComponentEdge) -> None:
        self.add_node(edge.src)
        self.add_node(edge.dst)
        self.adjacency[edge.src].append(edge)

    def neighbors(self, node: ComponentKey) -> List[ComponentEdge]:
        return self.adjacency.get(node, [])


def build_component_leaf_graph(
    families,
    project_newton,
    seed_points_fn: Callable,
    goal_point: np.ndarray,
    component_ids_for_family_fn: Callable[[object, float], List[str]],
    compatible_components_fn: Callable[[object, float, np.ndarray], List[str]],
    allowed_family_pair_fn: Optional[Callable[[str, str], bool]] = None,
    max_candidates_per_pair: int = 4,
    transition_generator: Optional[TransitionGenerator] = None,
) -> ComponentGraph:
    """
    Generic component-aware graph builder.

    Parameters
    ----------
    families:
        Iterable of leaf families.
    project_newton:
        Projection routine used by find_leaf_transition.
    seed_points_fn:
        Function (fam_a, lam_a, fam_b, lam_b) -> list[np.ndarray]
    goal_point:
        Goal point used for scoring transitions.
    component_ids_for_family_fn:
        Function (family, lam) -> list of valid component ids for that family/leaf.
    compatible_components_fn:
        Function (family, lam, q_transition) -> list of component ids of that leaf
        that are compatible with the exact transition point q_transition.
    allowed_family_pair_fn:
        Optional predicate (fam_a_name, fam_b_name) -> bool.
    max_candidates_per_pair:
        Keep at most this many exact transition candidates per leaf pair.
    """
    graph = ComponentGraph()
    generator = transition_generator or TransitionGenerator(
        seed_points_fn=seed_points_fn,
        project_newton=project_newton,
    )

    family_nodes = []
    for fam in families:
        for lam in fam.sample_lambdas():
            family_nodes.append((fam, float(lam)))
            for comp in component_ids_for_family_fn(fam, float(lam)):
                graph.add_node((fam.name, float(lam), comp))

    for i, (fam_a, lam_a) in enumerate(family_nodes):
        for j in range(i + 1, len(family_nodes)):
            fam_b, lam_b = family_nodes[j]

            if allowed_family_pair_fn is not None:
                if not allowed_family_pair_fn(fam_a.name, fam_b.name):
                    continue

            transition_result = generator.generate_transitions(
                source_family=fam_a,
                source_lam=lam_a,
                target_family=fam_b,
                target_lam=lam_b,
                goal_point=goal_point,
                max_candidates=max_candidates_per_pair,
            )

            if not transition_result.success:
                continue

            for k, cand in enumerate(transition_result.candidates):
                q = np.asarray(cand.transition_point, dtype=float).copy()

                src_components = compatible_components_fn(fam_a, lam_a, q)
                dst_components = compatible_components_fn(fam_b, lam_b, q)

                for src_comp in src_components:
                    for dst_comp in dst_components:
                        src = (fam_a.name, float(lam_a), str(src_comp))
                        dst = (fam_b.name, float(lam_b), str(dst_comp))

                        graph.add_edge(
                            ComponentEdge(
                                src=src,
                                dst=dst,
                                transition_point=q.copy(),
                                score=float(cand.score),
                                candidate_index=k,
                                metadata={
                                    **dict(cand.metadata),
                                    "source_component": str(src_comp),
                                    "target_component": str(dst_comp),
                                },
                            )
                        )
                        graph.add_edge(
                            ComponentEdge(
                                src=dst,
                                dst=src,
                                transition_point=q.copy(),
                                score=float(cand.score),
                                candidate_index=k,
                                metadata={
                                    **dict(cand.metadata),
                                    "source_component": str(dst_comp),
                                    "target_component": str(src_comp),
                                },
                            )
                        )

    return graph


def default_component_edge_cost(edge: ComponentEdge) -> float:
    if "base_score" in edge.metadata or "admissibility_cost" in edge.metadata:
        return (
            1.0
            + float(edge.metadata.get("base_score", edge.score))
            + float(edge.metadata.get("admissibility_cost", 0.0))
        )
    return 1.0 + float(edge.score)


def shortest_component_route(
    graph: ComponentGraph,
    start: ComponentKey,
    goal: ComponentKey,
    edge_cost_fn: Callable[[ComponentEdge], float] = default_component_edge_cost,
):
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


def route_node_sequence(start_key: ComponentKey, route_edges: List[ComponentEdge]) -> List[ComponentKey]:
    seq = [start_key]
    for e in route_edges:
        seq.append(e.dst)
    return seq
