from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
import numpy as np

from primitive_manifold_planner.manifolds import ImplicitManifold
from primitive_manifold_planner.planning.transitions import (
    TransitionCandidate,
    TransitionSearchResult,
    find_transition_candidates,
)


@dataclass
class ModeNode:
    name: str
    manifold: ImplicitManifold


@dataclass
class ModeEdge:
    """
    Undirected edge representing that two manifolds admit
    one or more valid transition candidates.
    """

    node_a: str
    node_b: str
    transition_candidates: list[TransitionCandidate]

    @property
    def best_transition_point(self) -> np.ndarray:
        return self.transition_candidates[0].point

    @property
    def best_residual_norm(self) -> float:
        return self.transition_candidates[0].residual_norm


@dataclass
class TransitionStep:
    from_mode: str
    to_mode: str
    transition_point: np.ndarray
    residual_norm: float


@dataclass
class MultimodalRoute:
    mode_sequence: list[str]
    transition_steps: list[TransitionStep]

    def __post_init__(self) -> None:
        expected_steps = max(0, len(self.mode_sequence) - 1)
        if len(self.transition_steps) != expected_steps:
            raise ValueError(
                f"Route mismatch: mode sequence of length {len(self.mode_sequence)} "
                f"requires {expected_steps} transition steps, got {len(self.transition_steps)}."
            )

    def __repr__(self) -> str:
        return (
            f"MultimodalRoute(num_modes={len(self.mode_sequence)}, "
            f"num_transitions={len(self.transition_steps)})"
        )


@dataclass
class ModeGraph:
    nodes: dict[str, ModeNode] = field(default_factory=dict)
    edges: list[ModeEdge] = field(default_factory=list)

    def add_node(self, name: str, manifold: ImplicitManifold) -> None:
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists in the mode graph.")
        self.nodes[name] = ModeNode(name=name, manifold=manifold)

    def has_node(self, name: str) -> bool:
        return name in self.nodes

    def add_edge(
        self,
        node_a: str,
        node_b: str,
        transition_candidates: list[TransitionCandidate],
    ) -> None:
        if node_a not in self.nodes or node_b not in self.nodes:
            raise ValueError("Both nodes must exist before adding an edge.")
        if len(transition_candidates) == 0:
            raise ValueError("An edge must contain at least one transition candidate.")

        a, b = sorted([node_a, node_b])
        for edge in self.edges:
            if sorted([edge.node_a, edge.node_b]) == [a, b]:
                return

        transition_candidates = sorted(transition_candidates, key=lambda c: c.score)

        self.edges.append(
            ModeEdge(
                node_a=node_a,
                node_b=node_b,
                transition_candidates=transition_candidates,
            )
        )

    def neighbors(self, node_name: str) -> list[str]:
        if node_name not in self.nodes:
            raise ValueError(f"Unknown node '{node_name}'.")

        nbrs: list[str] = []
        for edge in self.edges:
            if edge.node_a == node_name:
                nbrs.append(edge.node_b)
            elif edge.node_b == node_name:
                nbrs.append(edge.node_a)
        return nbrs

    def get_edge(self, node_a: str, node_b: str) -> ModeEdge | None:
        a, b = sorted([node_a, node_b])
        for edge in self.edges:
            if sorted([edge.node_a, edge.node_b]) == [a, b]:
                return edge
        return None

    def find_mode_sequence(self, start_node: str, goal_node: str) -> list[str] | None:
        if start_node not in self.nodes:
            raise ValueError(f"Unknown start node '{start_node}'.")
        if goal_node not in self.nodes:
            raise ValueError(f"Unknown goal node '{goal_node}'.")

        if start_node == goal_node:
            return [start_node]

        queue = deque([start_node])
        parent: dict[str, str | None] = {start_node: None}

        while queue:
            current = queue.popleft()

            for nbr in self.neighbors(current):
                if nbr in parent:
                    continue

                parent[nbr] = current

                if nbr == goal_node:
                    path = [goal_node]
                    while parent[path[-1]] is not None:
                        path.append(parent[path[-1]])
                    path.reverse()
                    return path

                queue.append(nbr)

        return None

    def build_route(self, start_node: str, goal_node: str) -> MultimodalRoute | None:
        mode_sequence = self.find_mode_sequence(start_node, goal_node)
        if mode_sequence is None:
            return None

        transition_steps: list[TransitionStep] = []

        for i in range(len(mode_sequence) - 1):
            a = mode_sequence[i]
            b = mode_sequence[i + 1]
            edge = self.get_edge(a, b)

            if edge is None:
                raise RuntimeError(
                    f"Inconsistent graph state: no edge found between consecutive "
                    f"modes '{a}' and '{b}' in the discovered mode sequence."
                )

            best = edge.transition_candidates[0]
            transition_steps.append(
                TransitionStep(
                    from_mode=a,
                    to_mode=b,
                    transition_point=best.point.copy(),
                    residual_norm=best.residual_norm,
                )
            )

        return MultimodalRoute(
            mode_sequence=mode_sequence,
            transition_steps=transition_steps,
        )

    def __repr__(self) -> str:
        return f"ModeGraph(num_nodes={len(self.nodes)}, num_edges={len(self.edges)})"


def build_mode_graph(
    manifolds: dict[str, ImplicitManifold],
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    num_seeds: int = 50,
    tol: float = 1e-8,
    max_nfev: int = 200,
    rng: np.random.Generator | None = None,
) -> tuple[ModeGraph, dict[tuple[str, str], TransitionSearchResult]]:
    graph = ModeGraph()
    diagnostics: dict[tuple[str, str], TransitionSearchResult] = {}

    for name, manifold in manifolds.items():
        graph.add_node(name, manifold)

    if rng is None:
        rng = np.random.default_rng()

    for name_a, name_b in combinations(manifolds.keys(), 2):
        manifold_a = manifolds[name_a]
        manifold_b = manifolds[name_b]

        result = find_transition_candidates(
            manifold_a=manifold_a,
            manifold_b=manifold_b,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            num_seeds=num_seeds,
            tol=tol,
            max_nfev=max_nfev,
            rng=rng,
        )

        diagnostics[(name_a, name_b)] = result

        if result.success and result.best_candidate is not None:
            graph.add_edge(
                node_a=name_a,
                node_b=name_b,
                transition_candidates=result.candidates,
            )

    return graph, diagnostics