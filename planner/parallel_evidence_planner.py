from __future__ import annotations

"""Shared parallel-evidence multimodal planning core extracted from Example 66.

This module holds the planner data structures and evidence-accumulation logic
for fixed-sequence and free-sequence multimodal planning. Scene construction
and visualization remain in the example scripts.
"""

from dataclasses import dataclass, field
import heapq
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from manifolds.geometric import MaskedManifold, PlaneManifold, SphereManifold
from manifolds.robot import RobotPlaneManifold, RobotSphereManifold
from primitive_manifold_planner.families.standard import MaskedFamily, PlaneFamily, SphereFamily
from primitive_manifold_planner.projection import project_newton

from intrinsic_multimodal_scene import (
    concatenate_paths,
    deduplicate_points,
    explored_points_from_edges,
    explore_on_manifold_from_frontier,
    merge_edges,
    ompl_native_exploration_target,
    refine_intersection_on_both_manifolds,
    sample_chart_centers,
    scan_path_for_transition,
    scan_tree_edges_for_transition,
    smooth_plane_segment,
    smooth_sphere_arc,
    solve_exact_segment_on_manifold,
    sphere_point,
)
from jointspace_planner_utils import (
    detect_transitions_jointspace,
    end_effector_point,
    explore_joint_manifold,
    generate_joint_proposals,
    inverse_kinematics_start,
    joint_path_to_task_path,
)
LEFT_STAGE = "left"
PLANE_STAGE = "plane"
RIGHT_STAGE = "right"
STAGES = [LEFT_STAGE, PLANE_STAGE, RIGHT_STAGE]

LEFT_MOTION = "left_motion"
PLANE_MOTION = "plane_motion"
RIGHT_MOTION = "right_motion"

GRAPH_NODE_TOL = 1e-4
TRANSITION_DEDUP_TOL = 2e-3
PROPOSALS_PER_ROUND = 1
SAFETY_MAX_TOTAL_ROUNDS = 24
FRONTIER_LIMIT = 24
FRONTIER_SELECTION_LIMIT = 10
TARGET_SAMPLE_ATTEMPTS = 4
TARGET_NOVELTY_RADIUS = 0.42
EVIDENCE_SEED_RADIUS = 0.58
MAX_STAGE_OMPL_UPDATES_PER_PROPOSAL = 2
TRANSITION_SCAN_LIMIT = 8
TRANSITION_FAILURE_STREAK_LIMIT = 6
LEFT_BRIDGE_TRY_LIMIT = 6
PLANE_BRIDGE_TRY_LIMIT = 2
RIGHT_BRIDGE_TRY_LIMIT = 2
PROGRESS_WINDOW = 10
SATURATION_WINDOW = 14
MIN_ROUNDS_BEFORE_SATURATION_CHECK = 24
MIN_POST_SOLUTION_ROUNDS = 10
DISPLAY_SPHERE_SAMPLES = 54
DISPLAY_PLANE_SAMPLES = 24
MAX_GRAPH_PATH_INTERNAL_POINTS = 4
GRAPH_PATH_POINT_STRIDE = 6
SOFT_STAGE_NODE_TARGET = 120
SOFT_TOTAL_NODE_TARGET = 360
SOFT_HYPOTHESIS_LIMIT = 20

BOUNDS_MIN = np.array([-3.5, -2.7, -0.7], dtype=float)
BOUNDS_MAX = np.array([3.5, 2.7, 1.9], dtype=float)


@dataclass
class StageNode:
    node_id: int
    q: np.ndarray
    expansion_count: int = 0
    seeded_from_proposal: bool = False


@dataclass
class StageEdge:
    edge_id: int
    src: int
    dst: int
    kind: str
    path: np.ndarray
    cost: float


@dataclass
class StageGraph:
    nodes: dict[int, StageNode] = field(default_factory=dict)
    edges: dict[int, StageEdge] = field(default_factory=dict)
    adjacency: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    next_node_id: int = 0
    next_edge_id: int = 0


@dataclass
class StageEvidenceStore:
    stage: str
    manifold: object
    graph: StageGraph = field(default_factory=StageGraph)
    frontier_ids: list[int] = field(default_factory=list)
    explored_edges: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    chart_centers: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    update_count: int = 0


@dataclass
class TransitionHypothesis:
    left_node_id: int | None = None
    plane_node_id: int | None = None
    right_node_id: int | None = None
    q: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    provenance: str = ""
    score: float = 0.0


@dataclass
class SequentialRouteCandidate:
    total_cost: float
    left_node_path: list[int]
    left_edge_path: list[int]
    plane_node_path: list[int]
    plane_edge_path: list[int]
    right_node_path: list[int]
    right_edge_path: list[int]
    committed_nodes: dict[str, set[int]]
    raw_path: np.ndarray
    display_path: np.ndarray
    joint_path: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))


@dataclass
class FixedPlaneRoute:
    success: bool
    message: str
    total_rounds: int
    candidate_evaluations: int
    left_evidence_nodes: int
    plane_evidence_nodes: int
    right_evidence_nodes: int
    committed_nodes: int
    evidence_only_nodes: int
    shared_proposals_processed: int
    proposals_used_by_multiple_stages: int
    plane_evidence_before_first_committed_entry: int
    right_evidence_before_first_committed_exit: int
    transition_hypotheses_left_plane: int
    transition_hypotheses_plane_right: int
    first_solution_round: int | None
    best_solution_round: int | None
    continued_after_first_solution: bool
    path: np.ndarray
    raw_path: np.ndarray
    certified_path_points: int
    display_path_points: int
    route_cost_raw: float
    route_cost_display: float
    graph_route_edges: int
    stage_evidence_points: dict[str, np.ndarray] = field(default_factory=dict)
    stage_evidence_edges: dict[str, list[tuple[np.ndarray, np.ndarray]]] = field(default_factory=dict)
    stage_frontier_points: dict[str, np.ndarray] = field(default_factory=dict)
    stage_chart_centers: dict[str, np.ndarray] = field(default_factory=dict)
    stage_frontier_counts: dict[str, int] = field(default_factory=dict)
    stage_stagnation_flags: dict[str, bool] = field(default_factory=dict)
    recent_graph_node_gain: int = 0
    recent_transition_gain: int = 0
    recent_route_improvement_gain: int = 0
    plane_evidence_growth_after_first_solution: int = 0
    right_evidence_growth_after_first_solution: int = 0
    multi_stage_updates_per_round: float = 0.0
    average_useful_stages_per_proposal: float = 0.0
    proposal_rounds_with_plane_updates: int = 0
    proposal_rounds_with_multi_stage_updates: int = 0
    committed_route_changes_after_first_solution: int = 0
    alternative_hypothesis_pairs_evaluated: int = 0
    left_plane_hypothesis_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    plane_right_hypothesis_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    committed_stage_nodes: dict[str, np.ndarray] = field(default_factory=dict)
    mode_counts: dict[str, int] = field(default_factory=dict)
    serial_mode: bool = False
    saturated_before_solution: bool = False
    stagnation_stage: str | None = None
    obstacles: list[object] = field(default_factory=list)
    joint_path: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))


@dataclass
class GenericTransitionHypothesis:
    source_stage: str
    target_stage: str
    source_node_id: int | None = None
    target_node_id: int | None = None
    q: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    provenance: str = ""
    score: float = 0.0


@dataclass
class MetaGraph:
    nodes: set[str] = field(default_factory=set)
    adjacency: dict[str, set[str]] = field(default_factory=dict)

    def add_node(self, node_id: str) -> None:
        self.nodes.add(str(node_id))
        self.adjacency.setdefault(str(node_id), set())

    def add_edge(self, src: str, dst: str) -> None:
        src_id = str(src)
        dst_id = str(dst)
        self.add_node(src_id)
        self.add_node(dst_id)
        self.adjacency[src_id].add(dst_id)

    def edges(self) -> list[tuple[str, str]]:
        return sorted((src, dst) for src, targets in self.adjacency.items() for dst in sorted(targets))


@dataclass
class UnknownSequenceCandidate:
    total_cost: float
    stage_sequence: list[str]
    stage_node_paths: dict[str, list[int]]
    stage_edge_paths: dict[str, list[int]]
    committed_nodes: dict[str, set[int]]
    raw_path: np.ndarray
    display_path: np.ndarray


@dataclass
class UnknownSequenceRoute:
    success: bool
    message: str
    discovered_sequence: list[str]
    total_rounds: int
    candidate_evaluations: int
    shared_proposals_processed: int
    global_transition_hypotheses: int
    meta_graph_edges: list[tuple[str, str]]
    start_stages: list[str]
    goal_stages: list[str]
    path: np.ndarray
    raw_path: np.ndarray
    route_cost_raw: float
    route_cost_display: float
    stage_node_counts: dict[str, int] = field(default_factory=dict)
    stage_frontier_counts: dict[str, int] = field(default_factory=dict)
    stage_evidence_points: dict[str, np.ndarray] = field(default_factory=dict)
    stage_evidence_edges: dict[str, list[tuple[np.ndarray, np.ndarray]]] = field(default_factory=dict)
    stage_frontier_points: dict[str, np.ndarray] = field(default_factory=dict)
    stage_chart_centers: dict[str, np.ndarray] = field(default_factory=dict)
    committed_stage_nodes: dict[str, np.ndarray] = field(default_factory=dict)
    transition_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    saturated_before_solution: bool = False
    stagnation_stage: str | None = None
    mode_counts: dict[str, int] = field(default_factory=dict)


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=float)
    ref = np.array([1.0, 0.0, 0.0], dtype=float) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=float)
    u = np.cross(n, ref)
    u = u / max(np.linalg.norm(u), 1e-15)
    v = np.cross(n, u)
    v = v / max(np.linalg.norm(v), 1e-15)
    return u, v


def unwrap_manifold(manifold):
    return manifold.base_manifold if isinstance(manifold, MaskedManifold) else manifold


def is_plane_like(manifold) -> bool:
    return isinstance(unwrap_manifold(manifold), PlaneManifold)


def path_cost(path: np.ndarray) -> float:
    arr = np.asarray(path, dtype=float)
    if len(arr) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


def add_stage_node(store: StageEvidenceStore, q: np.ndarray, seeded_from_proposal: bool = False, tol: float = GRAPH_NODE_TOL) -> int:
    qq = np.asarray(q, dtype=float).reshape(-1)
    for node_id, node in store.graph.nodes.items():
        if float(np.linalg.norm(node.q - qq)) <= tol:
            return node_id
    node_id = store.graph.next_node_id
    store.graph.next_node_id += 1
    store.graph.nodes[node_id] = StageNode(node_id=node_id, q=qq.copy(), seeded_from_proposal=bool(seeded_from_proposal))
    store.graph.adjacency[node_id] = []
    return node_id


def add_stage_edge(store: StageEvidenceStore, src: int, dst: int, kind: str, path: np.ndarray) -> int:
    arr = np.asarray(path, dtype=float)
    if len(arr) == 0:
        arr = np.asarray([store.graph.nodes[src].q.copy(), store.graph.nodes[dst].q.copy()], dtype=float)
    edge_id = store.graph.next_edge_id
    store.graph.next_edge_id += 1
    store.graph.edges[edge_id] = StageEdge(
        edge_id=edge_id,
        src=src,
        dst=dst,
        kind=kind,
        path=arr.copy(),
        cost=path_cost(arr),
    )
    store.graph.adjacency[src].append((dst, edge_id))
    store.graph.adjacency[dst].append((src, edge_id))
    return edge_id


def sparsify_graph_path(path: np.ndarray) -> np.ndarray:
    arr = np.asarray(path, dtype=float)
    if len(arr) <= 2:
        return arr
    internal = arr[1:-1:GRAPH_PATH_POINT_STRIDE]
    if len(internal) > MAX_GRAPH_PATH_INTERNAL_POINTS:
        picks = np.linspace(0, len(internal) - 1, num=MAX_GRAPH_PATH_INTERNAL_POINTS, dtype=int)
        internal = internal[picks]
    return np.asarray([arr[0], *internal, arr[-1]], dtype=float)


def sparsify_graph_path_indices(path: np.ndarray) -> list[int]:
    arr = np.asarray(path, dtype=float)
    if len(arr) <= 2:
        return list(range(len(arr)))
    internal_indices = list(range(1, len(arr) - 1, GRAPH_PATH_POINT_STRIDE))
    if len(internal_indices) > MAX_GRAPH_PATH_INTERNAL_POINTS:
        picks = np.linspace(0, len(internal_indices) - 1, num=MAX_GRAPH_PATH_INTERNAL_POINTS, dtype=int)
        internal_indices = [internal_indices[int(idx)] for idx in picks]
    return [0, *internal_indices, len(arr) - 1]


def connect_path_to_stage_graph(
    store: StageEvidenceStore,
    source_node_id: int,
    path: np.ndarray,
    kind: str,
    terminal_node_id: int | None = None,
) -> tuple[int, list[int], list[int]]:
    full_path = np.asarray(path, dtype=float)
    if len(full_path) == 0:
        return source_node_id, [source_node_id], []
    sparse_indices = sparsify_graph_path_indices(full_path)
    if len(sparse_indices) == 0:
        return source_node_id, [source_node_id], []

    current_id = source_node_id
    node_ids = [source_node_id]
    edge_ids: list[int] = []

    for step_idx, path_idx in enumerate(sparse_indices[1:], start=1):
        q = np.asarray(full_path[int(path_idx)], dtype=float)
        if terminal_node_id is not None and step_idx == len(sparse_indices) - 1:
            next_id = int(terminal_node_id)
        else:
            next_id = add_stage_node(store, np.asarray(q, dtype=float))
        prev_idx = int(sparse_indices[step_idx - 1])
        segment = np.asarray(full_path[prev_idx : int(path_idx) + 1], dtype=float)
        if len(segment) == 0:
            segment = np.asarray([store.graph.nodes[current_id].q.copy(), np.asarray(q, dtype=float).copy()], dtype=float)
        if next_id != current_id or step_idx == len(sparse_indices) - 1:
            edge_ids.append(add_stage_edge(store, current_id, next_id, kind, segment))
        current_id = next_id
        node_ids.append(current_id)
    return current_id, node_ids, edge_ids


def increment_stage_node_expansion(store: StageEvidenceStore, node_id: int) -> None:
    store.graph.nodes[node_id].expansion_count += 1


def stage_frontier_points(store: StageEvidenceStore) -> np.ndarray:
    pts = [store.graph.nodes[node_id].q for node_id in store.frontier_ids if node_id in store.graph.nodes]
    return np.asarray(pts, dtype=float) if len(pts) > 0 else np.zeros((0, 3), dtype=float)


def stage_recent_sum(history: list[int], window: int = PROGRESS_WINDOW) -> int:
    if len(history) == 0:
        return 0
    return int(sum(history[-window:]))


def stage_stagnating(history_nodes: list[int], history_transitions: list[int], history_route: list[int]) -> bool:
    if len(history_nodes) < PROGRESS_WINDOW:
        return False
    return (
        stage_recent_sum(history_nodes) == 0
        and stage_recent_sum(history_transitions) == 0
        and stage_recent_sum(history_route) == 0
    )


def stage_evidence_points(store: StageEvidenceStore) -> np.ndarray:
    groups: list[np.ndarray] = []
    explored = explored_points_from_edges(store.explored_edges)
    if len(explored) > 0:
        groups.append(explored)
    if len(store.chart_centers) > 0:
        groups.append(np.asarray(store.chart_centers, dtype=float))
    frontier = stage_frontier_points(store)
    if len(frontier) > 0:
        groups.append(frontier)
    all_nodes = [node.q for node in store.graph.nodes.values()]
    if len(all_nodes) > 0:
        groups.append(np.asarray(all_nodes, dtype=float))
    if len(groups) == 0:
        return np.zeros((0, 3), dtype=float)
    return deduplicate_points([point for group in groups for point in np.asarray(group, dtype=float)], tol=1e-4)


def stage_evidence_counts(stores: dict[str, StageEvidenceStore]) -> dict[str, int]:
    return {stage: len(stores[stage].graph.nodes) for stage in stores}


def stage_underexploration_factor(stage: str, stores: dict[str, StageEvidenceStore]) -> float:
    stage_names = list(stores.keys())
    counts = np.asarray([max(1, len(stores[name].graph.nodes)) for name in stage_names], dtype=float)
    mean_count = float(np.mean(counts))
    stage_count = float(max(1, len(stores[stage].graph.nodes)))
    deficit = max(0.0, mean_count - stage_count)
    bonus = deficit / max(mean_count, 1.0)
    if stage == PLANE_STAGE and PLANE_STAGE in stores:
        bonus *= 1.35
    elif stage == RIGHT_STAGE and RIGHT_STAGE in stores:
        bonus *= 1.10
    return float(bonus)


def greedy_stage_for_serial_round(stores: dict[str, StageEvidenceStore]) -> str:
    return max(
        STAGES,
        key=lambda stage: (
            stage_underexploration_factor(stage, stores),
            1 if stage == PLANE_STAGE else 0,
            -len(stores[stage].graph.nodes),
        ),
    )


def soft_stage_node_target(stage: str) -> int:
    return SOFT_STAGE_NODE_TARGET + (40 if stage == PLANE_STAGE else 0)


def adaptive_stage_update_budget(
    stores: dict[str, StageEvidenceStore],
    first_solution_round: int | None,
) -> int:
    total_nodes = sum(len(stores[stage].graph.nodes) for stage in stores)
    budget = int(MAX_STAGE_OMPL_UPDATES_PER_PROPOSAL)
    if total_nodes > SOFT_TOTAL_NODE_TARGET:
        budget -= 1
    if total_nodes > SOFT_TOTAL_NODE_TARGET + 220:
        budget -= 1
    if first_solution_round is not None and total_nodes > SOFT_TOTAL_NODE_TARGET:
        budget -= 1
    if PLANE_STAGE in stores and stage_underexploration_factor(PLANE_STAGE, stores) > 0.15:
        budget = max(budget, 2)
    return max(1, min(MAX_STAGE_OMPL_UPDATES_PER_PROPOSAL, budget))


def prune_transition_hypotheses(hypotheses: list[TransitionHypothesis]) -> None:
    if len(hypotheses) <= SOFT_HYPOTHESIS_LIMIT:
        return
    hypotheses.sort(key=lambda hyp: float(hyp.score))
    kept: list[TransitionHypothesis] = []
    for hyp in hypotheses:
        if any(float(np.linalg.norm(hyp.q - other.q)) <= TRANSITION_DEDUP_TOL for other in kept):
            continue
        kept.append(hyp)
        if len(kept) >= SOFT_HYPOTHESIS_LIMIT:
            break
    hypotheses[:] = kept


def prune_generic_transition_hypotheses(hypotheses: list[GenericTransitionHypothesis], limit: int = SOFT_HYPOTHESIS_LIMIT * 4) -> None:
    if len(hypotheses) <= limit:
        return
    hypotheses.sort(key=lambda hyp: (str(hyp.source_stage), str(hyp.target_stage), float(hyp.score)))
    kept: list[GenericTransitionHypothesis] = []
    for hyp in hypotheses:
        if any(
            str(hyp.source_stage) == str(other.source_stage)
            and str(hyp.target_stage) == str(other.target_stage)
            and float(np.linalg.norm(np.asarray(hyp.q, dtype=float) - np.asarray(other.q, dtype=float))) <= TRANSITION_DEDUP_TOL
            for other in kept
        ):
            continue
        kept.append(hyp)
        if len(kept) >= limit:
            break
    hypotheses[:] = kept


def build_meta_graph(stage_ids: list[str], hypotheses: list[GenericTransitionHypothesis]) -> MetaGraph:
    meta = MetaGraph()
    for stage_id in stage_ids:
        meta.add_node(str(stage_id))
    for hyp in hypotheses:
        meta.add_edge(str(hyp.source_stage), str(hyp.target_stage))
    return meta


def bfs_meta_path(meta_graph: MetaGraph, start_stages: list[str], goal_stages: list[str]) -> list[str]:
    goals = {str(stage) for stage in goal_stages}
    if len(goals) == 0:
        return []
    queue: list[str] = []
    previous: dict[str, str | None] = {}
    for stage in start_stages:
        stage_id = str(stage)
        if stage_id not in previous:
            previous[stage_id] = None
            queue.append(stage_id)
    head = 0
    while head < len(queue):
        current = queue[head]
        head += 1
        if current in goals:
            path = [current]
            while previous[path[-1]] is not None:
                path.append(str(previous[path[-1]]))
            path.reverse()
            return path
        for neighbor in sorted(meta_graph.adjacency.get(current, set())):
            if neighbor in previous:
                continue
            previous[neighbor] = current
            queue.append(neighbor)
    return []


def frontier_novelty_signal(store: StageEvidenceStore) -> float:
    frontier_ids = list(store.frontier_ids)
    if len(frontier_ids) == 0:
        return 0.0
    points = [store.graph.nodes[node_id].q for node_id in frontier_ids]
    expansion_bonus = float(np.mean([1.0 / (1.0 + store.graph.nodes[node_id].expansion_count) for node_id in frontier_ids]))
    if len(points) == 1:
        return expansion_bonus
    nearest = []
    for idx, point in enumerate(points):
        others = [float(np.linalg.norm(point - points[j])) for j in range(len(points)) if j != idx]
        nearest.append(min(others))
    return float(np.mean(nearest)) + 0.35 * expansion_bonus


def update_stage_frontier(store: StageEvidenceStore, new_ids: list[int], guide_point: np.ndarray, max_points: int = FRONTIER_LIMIT) -> None:
    merged: list[int] = []
    seen: set[int] = set()
    for node_id in list(store.frontier_ids) + list(new_ids):
        if node_id in seen:
            continue
        seen.add(node_id)
        merged.append(node_id)

    if len(merged) <= max_points:
        store.frontier_ids = merged
        return

    guide = np.asarray(guide_point, dtype=float)
    remaining = list(merged)
    selected: list[int] = []
    while len(remaining) > 0 and len(selected) < max_points:
        best_id = None
        best_score = -float("inf")
        for node_id in remaining:
            node = store.graph.nodes[node_id]
            guide_bonus = 1.0 / (1.0 + float(np.linalg.norm(node.q - guide)))
            underexplored_bonus = 1.0 / (1.0 + float(node.expansion_count))
            seeded_bonus = 0.18 if node.seeded_from_proposal else 0.0
            if len(selected) == 0:
                novelty_bonus = guide_bonus
            else:
                novelty_bonus = min(
                    float(np.linalg.norm(node.q - store.graph.nodes[selected_id].q))
                    for selected_id in selected
                )
            score = 0.42 * novelty_bonus + 0.24 * underexplored_bonus + 0.14 * guide_bonus + seeded_bonus
            if score > best_score:
                best_score = score
                best_id = node_id
        if best_id is None:
            break
        selected.append(best_id)
        remaining.remove(best_id)
    store.frontier_ids = selected


def choose_stage_source(store: StageEvidenceStore, guide_point: np.ndarray) -> int:
    guide = np.asarray(guide_point, dtype=float)
    scored = []
    for node_id in store.frontier_ids:
        node = store.graph.nodes[node_id]
        guide_bonus = 1.0 / (1.0 + float(np.linalg.norm(node.q - guide)))
        underexplored_bonus = 1.0 / (1.0 + float(node.expansion_count))
        if len(store.frontier_ids) > 1:
            diversity_bonus = min(
                float(np.linalg.norm(node.q - store.graph.nodes[other_id].q))
                for other_id in store.frontier_ids
                if other_id != node_id
            )
        else:
            diversity_bonus = 0.0
        seeded_bonus = 0.12 if node.seeded_from_proposal else 0.0
        score = -(0.46 * underexplored_bonus + 0.26 * guide_bonus + 0.18 * min(diversity_bonus, 1.0) + seeded_bonus)
        scored.append((score, node_id))
    scored.sort(key=lambda item: item[0])
    top_ids = [int(node_id) for _, node_id in scored[: min(FRONTIER_SELECTION_LIMIT, len(scored))]]
    return int(np.random.choice(top_ids))


def ranked_stage_sources(store: StageEvidenceStore, guide_point: np.ndarray, limit: int) -> list[int]:
    guide = np.asarray(guide_point, dtype=float)
    scored = []
    for node_id in store.frontier_ids:
        node = store.graph.nodes[node_id]
        guide_bonus = 1.0 / (1.0 + float(np.linalg.norm(node.q - guide)))
        underexplored_bonus = 1.0 / (1.0 + float(node.expansion_count))
        if len(store.frontier_ids) > 1:
            diversity_bonus = min(
                float(np.linalg.norm(node.q - store.graph.nodes[other_id].q))
                for other_id in store.frontier_ids
                if other_id != node_id
            )
        else:
            diversity_bonus = 0.0
        seeded_bonus = 0.12 if node.seeded_from_proposal else 0.0
        score = -(0.42 * guide_bonus + 0.28 * underexplored_bonus + 0.18 * min(diversity_bonus, 1.0) + seeded_bonus)
        scored.append((score, int(node_id)))
    scored.sort(key=lambda item: item[0])
    return [node_id for _, node_id in scored[: min(limit, len(scored))]]


def proposal_projection(manifold, proposal: np.ndarray) -> np.ndarray | None:
    projection = project_newton(
        manifold=manifold,
        x0=np.asarray(proposal, dtype=float),
        tol=1e-10,
        max_iters=60,
        damping=1.0,
    )
    if not projection.success:
        return None
    q = np.asarray(projection.x_projected, dtype=float)
    if isinstance(manifold, MaskedManifold) and not bool(manifold.within_bounds(q)):
        return None
    return q


def project_point_to_plane(plane_like_manifold, q: np.ndarray) -> np.ndarray:
    plane = unwrap_manifold(plane_like_manifold)
    normal = np.asarray(plane.normal, dtype=float)
    point = np.asarray(plane.point, dtype=float)
    qq = np.asarray(q, dtype=float)
    signed = float(np.dot(normal, qq - point))
    return qq - signed * normal


def proposal_stage_utility(
    stage: str,
    projected_q: np.ndarray,
    known_points: np.ndarray,
    guide_point: np.ndarray,
    stores: dict[str, StageEvidenceStore],
) -> float:
    q = np.asarray(projected_q, dtype=float)
    guide = np.asarray(guide_point, dtype=float)
    if len(known_points) > 0:
        novelty = min(float(np.linalg.norm(q - point)) for point in np.asarray(known_points, dtype=float))
    else:
        novelty = TARGET_NOVELTY_RADIUS
    novelty_bonus = min(novelty, TARGET_NOVELTY_RADIUS)
    guide_bonus = 1.0 / (1.0 + float(np.linalg.norm(q - guide)))
    underexplored_bonus = stage_underexploration_factor(stage, stores)
    stage_bias = 0.08 if stage == RIGHT_STAGE else (0.16 if stage == PLANE_STAGE else 0.0)
    return 0.50 * novelty_bonus + 0.26 * guide_bonus + 0.18 * underexplored_bonus + stage_bias


def rank_transition_hits(hits: np.ndarray, guide_point: np.ndarray, known_points: list[np.ndarray]) -> np.ndarray:
    if len(hits) == 0:
        return np.zeros((0, 3), dtype=float)
    guide = np.asarray(guide_point, dtype=float)
    scored = []
    for q in np.asarray(hits, dtype=float):
        guide_bonus = 1.0 / (1.0 + float(np.linalg.norm(q - guide)))
        if len(known_points) > 0:
            novelty = min(float(np.linalg.norm(q - np.asarray(p, dtype=float))) for p in known_points)
        else:
            novelty = TARGET_NOVELTY_RADIUS
        score = 0.58 * min(novelty, TARGET_NOVELTY_RADIUS) + 0.42 * guide_bonus
        scored.append((score, q.copy()))
    scored.sort(key=lambda item: item[0], reverse=True)
    return np.asarray([q for _, q in scored], dtype=float)


def transition_points_from_result(current_manifold, target_manifold, result) -> np.ndarray:
    explored_edges = list(getattr(result, "explored_edges", []))
    hits = scan_tree_edges_for_transition(
        current_manifold=current_manifold,
        target_manifold=target_manifold,
        explored_edges=explored_edges,
        target_tol=1e-4,
    )
    if len(hits) == 0:
        path = np.asarray(getattr(result, "path", np.zeros((0, 3), dtype=float)), dtype=float)
        if len(path) >= 2:
            hits = scan_path_for_transition(
                current_manifold=current_manifold,
                target_manifold=target_manifold,
                path=path,
                target_tol=1e-4,
            )
    return np.asarray(hits, dtype=float) if len(hits) > 0 else np.zeros((0, 3), dtype=float)


def add_generic_transition_hypotheses(
    source_stage: str,
    target_stage: str,
    stores: dict[str, StageEvidenceStore],
    result,
    source_node_id: int | None,
    guide_point: np.ndarray,
    hypotheses: list[GenericTransitionHypothesis],
) -> tuple[int, int]:
    if result is None or source_node_id is None or str(source_stage) == str(target_stage):
        return 0, 0

    source_store = stores[str(source_stage)]
    target_store = stores[str(target_stage)]
    hits = transition_points_from_result(source_store.manifold, target_store.manifold, result)
    ranked_hits = rank_transition_hits(np.asarray(hits, dtype=float), guide_point, [hyp.q for hyp in hypotheses])
    if len(ranked_hits) == 0:
        return 0, 0

    source_q = np.asarray(source_store.graph.nodes[int(source_node_id)].q, dtype=float)
    adaptive_scan_limit = min(
        12,
        TRANSITION_SCAN_LIMIT + max(0, len(ranked_hits) // 3) + max(0, len(source_store.frontier_ids) // 6),
    )
    added = 0
    eval_count = 0
    failure_streak = 0
    for hit in ranked_hits:
        if added >= adaptive_scan_limit:
            break
        if any(
            str(hyp.source_stage) == str(source_stage)
            and str(hyp.target_stage) == str(target_stage)
            and float(np.linalg.norm(np.asarray(hit, dtype=float) - np.asarray(hyp.q, dtype=float))) <= TRANSITION_DEDUP_TOL
            for hyp in hypotheses
        ):
            continue
        refined, refined_ok = refine_intersection_on_both_manifolds(source_store.manifold, target_store.manifold, hit, tol=1e-8, max_iters=25)
        q_hit = np.asarray(refined if refined_ok else hit, dtype=float)
        exact_result = solve_exact_segment_on_manifold(
            manifold=source_store.manifold,
            x_start=source_q,
            x_goal=q_hit,
            bounds_min=BOUNDS_MIN,
            bounds_max=BOUNDS_MAX,
        )
        eval_count += 1
        if not exact_result.success:
            failure_streak += 1
            if failure_streak >= TRANSITION_FAILURE_STREAK_LIMIT:
                break
            continue
        failure_streak = 0
        source_store.explored_edges = merge_edges(source_store.explored_edges, list(getattr(exact_result, "explored_edges", [])))
        source_store.chart_centers = merge_chart_centers(source_store.chart_centers, exact_result)
        source_hit_id, source_path_nodes, _ = connect_path_to_stage_graph(
            source_store,
            int(source_node_id),
            np.asarray(exact_result.path, dtype=float),
            f"{source_stage}_motion",
        )
        target_hit_id = add_stage_node(target_store, q_hit, seeded_from_proposal=True)
        update_stage_frontier(source_store, source_path_nodes + [source_hit_id], guide_point)
        update_stage_frontier(target_store, [target_hit_id], guide_point)
        hypotheses.append(
            GenericTransitionHypothesis(
                source_stage=str(source_stage),
                target_stage=str(target_stage),
                source_node_id=int(source_hit_id),
                target_node_id=int(target_hit_id),
                q=np.asarray(q_hit, dtype=float),
                provenance=f"{source_stage}_to_{target_stage}",
                score=float(np.linalg.norm(np.asarray(q_hit, dtype=float) - np.asarray(guide_point, dtype=float))),
            )
        )
        added += 1
        prune_generic_transition_hypotheses(hypotheses)
    return added, eval_count


def bridge_stage_node_sets(
    store: StageEvidenceStore,
    source_node_ids: list[int],
    target_node_ids: list[int],
    guide_point: np.ndarray,
    motion_kind: str,
    max_tries: int = 3,
) -> tuple[int, int]:
    source_ids = sorted({int(node_id) for node_id in source_node_ids if node_id in store.graph.nodes})
    target_ids = sorted({int(node_id) for node_id in target_node_ids if node_id in store.graph.nodes})
    if len(source_ids) == 0 or len(target_ids) == 0:
        return 0, 0

    reachable_targets: set[int] = set()
    for source_id in source_ids:
        dist, _prev_node, _prev_edge = shortest_paths_in_stage(store, int(source_id))
        reachable_targets.update(int(node_id) for node_id in dist if int(node_id) in target_ids)
    target_ids = [node_id for node_id in target_ids if node_id not in reachable_targets]
    if len(target_ids) == 0:
        return 0, 0

    candidate_pairs: list[tuple[float, int, int]] = []
    for source_id in source_ids:
        q_source = np.asarray(store.graph.nodes[source_id].q, dtype=float)
        for target_id in target_ids:
            if source_id == target_id:
                continue
            q_target = np.asarray(store.graph.nodes[target_id].q, dtype=float)
            candidate_pairs.append((float(np.linalg.norm(q_source - q_target)), source_id, target_id))
    candidate_pairs.sort(key=lambda item: item[0])

    eval_count = 0
    node_gain = 0
    tried_targets: set[int] = set()
    for _, source_id, target_id in candidate_pairs:
        if len(tried_targets) >= max_tries:
            break
        if int(target_id) in tried_targets:
            continue
        tried_targets.add(int(target_id))
        exact_result = solve_exact_segment_on_manifold(
            manifold=store.manifold,
            x_start=np.asarray(store.graph.nodes[source_id].q, dtype=float),
            x_goal=np.asarray(store.graph.nodes[target_id].q, dtype=float),
            bounds_min=BOUNDS_MIN,
            bounds_max=BOUNDS_MAX,
        )
        eval_count += 1
        if not exact_result.success:
            continue
        store.explored_edges = merge_edges(store.explored_edges, list(getattr(exact_result, "explored_edges", [])))
        store.chart_centers = merge_chart_centers(store.chart_centers, exact_result)
        _, node_ids, _ = connect_path_to_stage_graph(
            store=store,
            source_node_id=int(source_id),
            path=np.asarray(exact_result.path, dtype=float),
            kind=motion_kind,
            terminal_node_id=int(target_id),
        )
        update_stage_frontier(store, node_ids + [int(target_id)], guide_point)
        node_gain += max(0, len(node_ids) - 2)
        break
    return node_gain, eval_count


def merge_chart_centers(existing: np.ndarray, result) -> np.ndarray:
    centers = sample_chart_centers(
        np.asarray(getattr(result, "path", np.zeros((0, 3), dtype=float)), dtype=float),
        int(getattr(result, "chart_count", 0)),
    )
    if len(existing) == 0:
        return np.asarray(centers, dtype=float)
    if len(centers) == 0:
        return np.asarray(existing, dtype=float)
    return deduplicate_points(list(existing) + list(centers), tol=1e-4)


def effective_proposals_per_round(
    round_idx: int,
    first_solution_round: int | None,
    stores: dict[str, StageEvidenceStore],
) -> int:
    total_nodes = sum(len(stores[stage].graph.nodes) for stage in stores)
    count = int(PROPOSALS_PER_ROUND)
    if round_idx % 4 == 0:
        count += 1
    if first_solution_round is not None and total_nodes < SOFT_TOTAL_NODE_TARGET:
        count += 1
    if (
        PLANE_STAGE in stores
        and stage_underexploration_factor(PLANE_STAGE, stores) > 0.18
        and len(stores[PLANE_STAGE].graph.nodes) < soft_stage_node_target(PLANE_STAGE)
    ):
        count += 1
    if total_nodes > SOFT_TOTAL_NODE_TARGET:
        count -= 1
    if total_nodes > SOFT_TOTAL_NODE_TARGET + 220:
        count -= 1
    return max(1, min(4, count))


def generate_ambient_proposals(
    round_idx: int,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    plane_point: np.ndarray,
    stores: dict[str, StageEvidenceStore],
    guides: dict[str, np.ndarray],
    proposal_count: int,
) -> list[np.ndarray]:
    midpoint = 0.5 * (np.asarray(start_q, dtype=float) + np.asarray(goal_q, dtype=float))
    proposals = []
    stage_names = list(stores.keys())
    weakest_stage = min(stage_names, key=lambda stage: (len(stores[stage].graph.nodes), stores[stage].update_count))
    weakest_frontier = stage_frontier_points(stores[weakest_stage])
    weakest_anchor = guides[weakest_stage] if len(weakest_frontier) == 0 else weakest_frontier[np.random.randint(len(weakest_frontier))]
    for idx in range(proposal_count):
        selector = (round_idx + idx) % 5
        if selector == 0:
            q = np.random.uniform(BOUNDS_MIN, BOUNDS_MAX)
        elif selector == 1:
            q = midpoint + np.random.normal(scale=np.array([1.1, 0.9, 0.45], dtype=float), size=3)
        elif selector == 2:
            q = 0.35 * np.asarray(start_q, dtype=float) + 0.65 * np.asarray(goal_q, dtype=float)
            q = q + np.random.normal(scale=np.array([0.85, 0.7, 0.35], dtype=float), size=3)
        elif selector == 3:
            q = np.asarray(plane_point, dtype=float) + np.random.normal(scale=np.array([1.2, 1.0, 0.55], dtype=float), size=3)
        else:
            q = np.asarray(weakest_anchor, dtype=float) + np.random.normal(scale=np.array([0.95, 0.8, 0.40], dtype=float), size=3)
        proposals.append(np.asarray(np.clip(q, BOUNDS_MIN, BOUNDS_MAX), dtype=float))
    return proposals


def maybe_seed_stage_component(store: StageEvidenceStore, target_q: np.ndarray, guide_point: np.ndarray) -> tuple[int, bool]:
    known = stage_evidence_points(store)
    if len(store.graph.nodes) == 0:
        node_id = add_stage_node(store, target_q, seeded_from_proposal=True)
        update_stage_frontier(store, [node_id], guide_point)
        return node_id, True
    if len(known) == 0:
        node_id = add_stage_node(store, target_q, seeded_from_proposal=True)
        update_stage_frontier(store, [node_id], guide_point)
        return node_id, True
    nearest = min(float(np.linalg.norm(np.asarray(target_q, dtype=float) - point)) for point in np.asarray(known, dtype=float))
    if nearest >= EVIDENCE_SEED_RADIUS:
        node_id = add_stage_node(store, target_q, seeded_from_proposal=True)
        update_stage_frontier(store, [node_id], guide_point)
        return node_id, True
    return -1, False


def seed_stage_evidence_only(store: StageEvidenceStore, target_q: np.ndarray, guide_point: np.ndarray) -> int:
    node_id, seeded = maybe_seed_stage_component(store, target_q, guide_point)
    return int(node_id) if seeded else -1


def update_stage_evidence_from_proposal(
    store: StageEvidenceStore,
    target_q: np.ndarray,
    guide_point: np.ndarray,
) -> tuple[int, int, object | None, int | None]:
    seeded_node_id, seeded = maybe_seed_stage_component(store, target_q, guide_point)
    if seeded:
        store.update_count += 1
        return 1, 0, None, seeded_node_id

    source_node_id = choose_stage_source(store, np.asarray(target_q, dtype=float))
    increment_stage_node_expansion(store, source_node_id)
    source_q = store.graph.nodes[source_node_id].q
    result = explore_on_manifold_from_frontier(
        manifold=store.manifold,
        x_start=source_q,
        x_goal=np.asarray(target_q, dtype=float),
        bounds_min=BOUNDS_MIN,
        bounds_max=BOUNDS_MAX,
    )
    store.explored_edges = merge_edges(store.explored_edges, list(getattr(result, "explored_edges", [])))
    store.chart_centers = merge_chart_centers(store.chart_centers, result)
    end_node_id, new_path_nodes, _ = connect_path_to_stage_graph(
        store=store,
        source_node_id=source_node_id,
        path=np.asarray(result.path, dtype=float),
        kind=f"{store.stage}_evidence_motion",
    )
    update_stage_frontier(store, new_path_nodes + [end_node_id], guide_point)
    store.update_count += 1
    return max(0, len(new_path_nodes) - 1), 1, result, source_node_id


def add_left_plane_hypotheses(
    source_stage: str,
    source_store: StageEvidenceStore,
    plane_store: StageEvidenceStore,
    left_store: StageEvidenceStore,
    result,
    source_node_id: int | None,
    guide_point: np.ndarray,
    hypotheses: list[TransitionHypothesis],
) -> tuple[int, int]:
    if result is None or source_node_id is None:
        return 0, 0
    hits = transition_points_from_result(source_store.manifold, plane_store.manifold if source_stage == LEFT_STAGE else left_store.manifold, result)
    if source_stage == PLANE_STAGE:
        hits = transition_points_from_result(source_store.manifold, left_store.manifold, result)
    ranked_hits = rank_transition_hits(np.asarray(hits, dtype=float), guide_point, [hyp.q for hyp in hypotheses])
    if len(ranked_hits) == 0:
        return 0, 0

    source_q = source_store.graph.nodes[source_node_id].q
    adaptive_scan_limit = min(
        12,
        TRANSITION_SCAN_LIMIT + max(0, len(ranked_hits) // 3) + max(0, len(source_store.frontier_ids) // 6),
        max(2, SOFT_HYPOTHESIS_LIMIT - len(hypotheses) + 2),
    )
    added = 0
    eval_count = 0
    failure_streak = 0
    for hit in ranked_hits:
        if added >= adaptive_scan_limit:
            break
        if any(float(np.linalg.norm(np.asarray(hit, dtype=float) - hyp.q)) <= TRANSITION_DEDUP_TOL for hyp in hypotheses):
            continue
        if source_stage == LEFT_STAGE:
            refined, refined_ok = refine_intersection_on_both_manifolds(left_store.manifold, plane_store.manifold, hit, tol=1e-8, max_iters=25)
            q_hit = np.asarray(refined if refined_ok else hit, dtype=float)
            exact_result = solve_exact_segment_on_manifold(left_store.manifold, source_q, q_hit, BOUNDS_MIN, BOUNDS_MAX)
            eval_count += 1
            if not exact_result.success:
                failure_streak += 1
                if failure_streak >= TRANSITION_FAILURE_STREAK_LIMIT:
                    break
                continue
            failure_streak = 0
            left_store.explored_edges = merge_edges(left_store.explored_edges, list(getattr(exact_result, "explored_edges", [])))
            left_store.chart_centers = merge_chart_centers(left_store.chart_centers, exact_result)
            left_node_id, left_path_nodes, _ = connect_path_to_stage_graph(left_store, source_node_id, np.asarray(exact_result.path, dtype=float), LEFT_MOTION)
            plane_node_id = add_stage_node(plane_store, q_hit, seeded_from_proposal=True)
            update_stage_frontier(left_store, left_path_nodes + [left_node_id], guide_point)
            update_stage_frontier(plane_store, [plane_node_id], guide_point)
        else:
            refined, refined_ok = refine_intersection_on_both_manifolds(plane_store.manifold, left_store.manifold, hit, tol=1e-8, max_iters=25)
            q_hit = np.asarray(refined if refined_ok else hit, dtype=float)
            exact_result = solve_exact_segment_on_manifold(plane_store.manifold, source_q, q_hit, BOUNDS_MIN, BOUNDS_MAX)
            eval_count += 1
            if not exact_result.success:
                failure_streak += 1
                if failure_streak >= TRANSITION_FAILURE_STREAK_LIMIT:
                    break
                continue
            failure_streak = 0
            plane_store.explored_edges = merge_edges(plane_store.explored_edges, list(getattr(exact_result, "explored_edges", [])))
            plane_store.chart_centers = merge_chart_centers(plane_store.chart_centers, exact_result)
            plane_node_id, plane_path_nodes, _ = connect_path_to_stage_graph(plane_store, source_node_id, np.asarray(exact_result.path, dtype=float), PLANE_MOTION)
            left_node_id = add_stage_node(left_store, q_hit, seeded_from_proposal=True)
            update_stage_frontier(plane_store, plane_path_nodes + [plane_node_id], guide_point)
            update_stage_frontier(left_store, [left_node_id], guide_point)

        hypotheses.append(
            TransitionHypothesis(
                left_node_id=int(left_node_id),
                plane_node_id=int(plane_node_id),
                q=np.asarray(q_hit, dtype=float),
                provenance=f"{source_stage}_evidence",
                score=float(np.linalg.norm(q_hit - np.asarray(guide_point, dtype=float))),
            )
        )
        added += 1
        prune_transition_hypotheses(hypotheses)
    return added, eval_count


def add_plane_right_hypotheses(
    source_stage: str,
    source_store: StageEvidenceStore,
    plane_store: StageEvidenceStore,
    right_store: StageEvidenceStore,
    result,
    source_node_id: int | None,
    guide_point: np.ndarray,
    hypotheses: list[TransitionHypothesis],
) -> tuple[int, int]:
    if result is None or source_node_id is None:
        return 0, 0
    hits = transition_points_from_result(source_store.manifold, right_store.manifold if source_stage == PLANE_STAGE else plane_store.manifold, result)
    if source_stage == RIGHT_STAGE:
        hits = transition_points_from_result(source_store.manifold, plane_store.manifold, result)
    ranked_hits = rank_transition_hits(np.asarray(hits, dtype=float), guide_point, [hyp.q for hyp in hypotheses])
    if len(ranked_hits) == 0:
        return 0, 0

    source_q = source_store.graph.nodes[source_node_id].q
    adaptive_scan_limit = min(
        12,
        TRANSITION_SCAN_LIMIT + max(0, len(ranked_hits) // 3) + max(0, len(source_store.frontier_ids) // 6),
        max(2, SOFT_HYPOTHESIS_LIMIT - len(hypotheses) + 2),
    )
    added = 0
    eval_count = 0
    failure_streak = 0
    for hit in ranked_hits:
        if added >= adaptive_scan_limit:
            break
        if any(float(np.linalg.norm(np.asarray(hit, dtype=float) - hyp.q)) <= TRANSITION_DEDUP_TOL for hyp in hypotheses):
            continue
        if source_stage == PLANE_STAGE:
            refined, refined_ok = refine_intersection_on_both_manifolds(plane_store.manifold, right_store.manifold, hit, tol=1e-8, max_iters=25)
            q_hit = np.asarray(refined if refined_ok else hit, dtype=float)
            exact_result = solve_exact_segment_on_manifold(plane_store.manifold, source_q, q_hit, BOUNDS_MIN, BOUNDS_MAX)
            eval_count += 1
            if not exact_result.success:
                failure_streak += 1
                if failure_streak >= TRANSITION_FAILURE_STREAK_LIMIT:
                    break
                continue
            failure_streak = 0
            plane_store.explored_edges = merge_edges(plane_store.explored_edges, list(getattr(exact_result, "explored_edges", [])))
            plane_store.chart_centers = merge_chart_centers(plane_store.chart_centers, exact_result)
            plane_node_id, plane_path_nodes, _ = connect_path_to_stage_graph(plane_store, source_node_id, np.asarray(exact_result.path, dtype=float), PLANE_MOTION)
            right_node_id = add_stage_node(right_store, q_hit, seeded_from_proposal=True)
            update_stage_frontier(plane_store, plane_path_nodes + [plane_node_id], guide_point)
            update_stage_frontier(right_store, [right_node_id], np.asarray(q_hit, dtype=float))
        else:
            refined, refined_ok = refine_intersection_on_both_manifolds(right_store.manifold, plane_store.manifold, hit, tol=1e-8, max_iters=25)
            q_hit = np.asarray(refined if refined_ok else hit, dtype=float)
            exact_result = solve_exact_segment_on_manifold(right_store.manifold, source_q, q_hit, BOUNDS_MIN, BOUNDS_MAX)
            eval_count += 1
            if not exact_result.success:
                failure_streak += 1
                if failure_streak >= TRANSITION_FAILURE_STREAK_LIMIT:
                    break
                continue
            failure_streak = 0
            right_store.explored_edges = merge_edges(right_store.explored_edges, list(getattr(exact_result, "explored_edges", [])))
            right_store.chart_centers = merge_chart_centers(right_store.chart_centers, exact_result)
            right_node_id, right_path_nodes, _ = connect_path_to_stage_graph(right_store, source_node_id, np.asarray(exact_result.path, dtype=float), RIGHT_MOTION)
            plane_node_id = add_stage_node(plane_store, q_hit, seeded_from_proposal=True)
            update_stage_frontier(right_store, right_path_nodes + [right_node_id], guide_point)
            update_stage_frontier(plane_store, [plane_node_id], guide_point)

        hypotheses.append(
            TransitionHypothesis(
                plane_node_id=int(plane_node_id),
                right_node_id=int(right_node_id),
                q=np.asarray(q_hit, dtype=float),
                provenance=f"{source_stage}_evidence",
                score=float(np.linalg.norm(q_hit - np.asarray(guide_point, dtype=float))),
            )
        )
        added += 1
        prune_transition_hypotheses(hypotheses)
    return added, eval_count


def bridge_left_hypotheses_to_start(
    left_store: StageEvidenceStore,
    start_node_id: int,
    left_plane_hypotheses: list[TransitionHypothesis],
    guide_point: np.ndarray,
) -> tuple[int, int]:
    """Make left-plane transition hypotheses usable by the committed sequential route.

    Parallel evidence may discover a valid left-plane intersection from a
    left-side evidence node that is not yet in the start-connected component.
    This bridge uses the same OMPL exact constrained planner to connect
    start-reachable left evidence to those hypothesis nodes.
    """

    if len(left_plane_hypotheses) == 0:
        return 0, 0

    left_dist, _prev_node, _prev_edge = shortest_paths_in_stage(left_store, int(start_node_id))
    reachable_ids = sorted(int(node_id) for node_id in left_dist)
    if len(reachable_ids) == 0:
        return 0, 0

    target_ids = sorted(
        {
            int(hyp.left_node_id)
            for hyp in left_plane_hypotheses
            if hyp.left_node_id is not None and int(hyp.left_node_id) not in left_dist
        }
    )
    if len(target_ids) == 0:
        return 0, 0

    candidate_pairs: list[tuple[float, int, int]] = []
    for target_id in target_ids:
        if target_id not in left_store.graph.nodes:
            continue
        target_q = left_store.graph.nodes[target_id].q
        nearest_sources = sorted(
            reachable_ids,
            key=lambda source_id: float(np.linalg.norm(left_store.graph.nodes[source_id].q - target_q)),
        )
        for source_id in nearest_sources[: min(6, len(nearest_sources))]:
            source_q = left_store.graph.nodes[source_id].q
            candidate_pairs.append((float(np.linalg.norm(source_q - target_q)), source_id, target_id))
    candidate_pairs.sort(key=lambda item: item[0])

    eval_count = 0
    node_gain = 0
    tried_targets: set[int] = set()
    for _score, source_id, target_id in candidate_pairs:
        if len(tried_targets) >= LEFT_BRIDGE_TRY_LIMIT:
            break
        if int(target_id) in tried_targets:
            continue
        tried_targets.add(int(target_id))
        exact_result = solve_exact_segment_on_manifold(
            manifold=left_store.manifold,
            x_start=left_store.graph.nodes[source_id].q,
            x_goal=left_store.graph.nodes[target_id].q,
            bounds_min=BOUNDS_MIN,
            bounds_max=BOUNDS_MAX,
        )
        eval_count += 1
        if not exact_result.success:
            continue
        left_store.explored_edges = merge_edges(
            left_store.explored_edges,
            list(getattr(exact_result, "explored_edges", [])),
        )
        left_store.chart_centers = merge_chart_centers(left_store.chart_centers, exact_result)
        _, node_ids, _ = connect_path_to_stage_graph(
            store=left_store,
            source_node_id=int(source_id),
            path=np.asarray(exact_result.path, dtype=float),
            kind=LEFT_MOTION,
            terminal_node_id=int(target_id),
        )
        update_stage_frontier(left_store, node_ids + [int(target_id)], guide_point)
        node_gain += max(0, len(node_ids) - 2)
        # Recompute reachability after the first success; additional entries can
        # be connected in later rounds if needed.
        break
    return node_gain, eval_count


def bridge_plane_hypothesis_components(
    plane_store: StageEvidenceStore,
    left_plane_hypotheses: list[TransitionHypothesis],
    plane_right_hypotheses: list[TransitionHypothesis],
) -> tuple[int, int]:
    left_ids = sorted({int(hyp.plane_node_id) for hyp in left_plane_hypotheses if hyp.plane_node_id is not None})
    right_ids = sorted({int(hyp.plane_node_id) for hyp in plane_right_hypotheses if hyp.plane_node_id is not None})
    if len(left_ids) == 0 or len(right_ids) == 0:
        return 0, 0

    candidate_pairs: list[tuple[float, int, int]] = []
    for left_id in left_ids:
        q_left = plane_store.graph.nodes[left_id].q
        for right_id in right_ids:
            if left_id == right_id:
                continue
            q_right = plane_store.graph.nodes[right_id].q
            candidate_pairs.append((float(np.linalg.norm(q_left - q_right)), left_id, right_id))
    candidate_pairs.sort(key=lambda item: item[0])

    eval_count = 0
    node_gain = 0
    tried = 0
    for _, left_id, right_id in candidate_pairs:
        if tried >= PLANE_BRIDGE_TRY_LIMIT:
            break
        tried += 1
        q_left = plane_store.graph.nodes[left_id].q
        q_right = plane_store.graph.nodes[right_id].q
        exact_result = solve_exact_segment_on_manifold(
            manifold=plane_store.manifold,
            x_start=q_left,
            x_goal=q_right,
            bounds_min=BOUNDS_MIN,
            bounds_max=BOUNDS_MAX,
        )
        eval_count += 1
        if not exact_result.success:
            continue
        plane_store.explored_edges = merge_edges(
            plane_store.explored_edges,
            list(getattr(exact_result, "explored_edges", [])),
        )
        plane_store.chart_centers = merge_chart_centers(plane_store.chart_centers, exact_result)
        _, node_ids, _ = connect_path_to_stage_graph(
            store=plane_store,
            source_node_id=left_id,
            path=np.asarray(exact_result.path, dtype=float),
            kind=PLANE_MOTION,
            terminal_node_id=right_id,
        )
        update_stage_frontier(plane_store, node_ids + [left_id, right_id], plane_store.graph.nodes[right_id].q)
        node_gain += max(0, len(node_ids) - 2)
        break
    return node_gain, eval_count


def connect_right_hypothesis_to_goal(
    right_store: StageEvidenceStore,
    plane_right_hypotheses: list[TransitionHypothesis],
    goal_node_id: int,
    guide_point: np.ndarray,
) -> tuple[int, int]:
    right_ids = sorted({int(hyp.right_node_id) for hyp in plane_right_hypotheses if hyp.right_node_id is not None})
    if len(right_ids) == 0:
        return 0, 0

    scored = []
    for node_id in right_ids:
        q = right_store.graph.nodes[node_id].q
        scored.append((float(np.linalg.norm(q - np.asarray(guide_point, dtype=float))), node_id))
    scored.sort(key=lambda item: item[0])

    eval_count = 0
    node_gain = 0
    for _, node_id in scored[: min(RIGHT_BRIDGE_TRY_LIMIT, len(scored))]:
        q = right_store.graph.nodes[node_id].q
        exact_result = solve_exact_segment_on_manifold(
            manifold=right_store.manifold,
            x_start=q,
            x_goal=right_store.graph.nodes[goal_node_id].q,
            bounds_min=BOUNDS_MIN,
            bounds_max=BOUNDS_MAX,
        )
        eval_count += 1
        if not exact_result.success:
            continue
        right_store.explored_edges = merge_edges(
            right_store.explored_edges,
            list(getattr(exact_result, "explored_edges", [])),
        )
        right_store.chart_centers = merge_chart_centers(right_store.chart_centers, exact_result)
        _, node_ids, _ = connect_path_to_stage_graph(
            store=right_store,
            source_node_id=node_id,
            path=np.asarray(exact_result.path, dtype=float),
            kind=RIGHT_MOTION,
            terminal_node_id=goal_node_id,
        )
        update_stage_frontier(right_store, node_ids + [goal_node_id], guide_point)
        node_gain += max(0, len(node_ids) - 2)
        break
    return node_gain, eval_count


def shortest_paths_in_stage(store: StageEvidenceStore, start_id: int) -> tuple[dict[int, float], dict[int, int], dict[int, int]]:
    queue: list[tuple[float, int]] = [(0.0, start_id)]
    distances = {start_id: 0.0}
    previous_node: dict[int, int] = {}
    previous_edge: dict[int, int] = {}
    visited: set[int] = set()

    while queue:
        current_cost, node_id = heapq.heappop(queue)
        if node_id in visited:
            continue
        visited.add(node_id)
        for neighbor_id, edge_id in store.graph.adjacency.get(node_id, []):
            if neighbor_id in visited:
                continue
            edge = store.graph.edges[edge_id]
            new_cost = current_cost + float(edge.cost)
            if new_cost < distances.get(neighbor_id, float("inf")):
                distances[neighbor_id] = new_cost
                previous_node[neighbor_id] = node_id
                previous_edge[neighbor_id] = edge_id
                heapq.heappush(queue, (new_cost, neighbor_id))
    return distances, previous_node, previous_edge


def reconstruct_stage_path(
    store: StageEvidenceStore,
    start_id: int,
    goal_id: int,
    previous_node: dict[int, int],
    previous_edge: dict[int, int],
) -> tuple[list[int], list[int]]:
    if start_id == goal_id:
        return [start_id], []
    if goal_id not in previous_node and goal_id != start_id:
        return [], []
    node_path = [goal_id]
    edge_path: list[int] = []
    current = goal_id
    while current != start_id:
        edge_id = previous_edge[current]
        current = previous_node[current]
        edge_path.append(edge_id)
        node_path.append(current)
    node_path.reverse()
    edge_path.reverse()
    return node_path, edge_path


def oriented_stage_edge_path(
    store: StageEvidenceStore,
    edge_id: int,
    src_node_id: int,
    dst_node_id: int,
) -> np.ndarray:
    edge = store.graph.edges[int(edge_id)]
    path = np.asarray(edge.path, dtype=float)
    if int(edge.src) == int(src_node_id) and int(edge.dst) == int(dst_node_id):
        return np.asarray(path, dtype=float)
    if int(edge.src) == int(dst_node_id) and int(edge.dst) == int(src_node_id):
        return np.asarray(path[::-1], dtype=float)
    return np.asarray(path, dtype=float)


def build_stage_raw_path(store: StageEvidenceStore, node_ids: list[int], edge_ids: list[int]) -> np.ndarray:
    if len(edge_ids) == 0:
        if len(node_ids) == 1:
            return np.asarray([store.graph.nodes[int(node_ids[0])].q], dtype=float)
        return np.zeros((0, 3), dtype=float)
    segments = [
        oriented_stage_edge_path(store, edge_id, int(node_ids[idx]), int(node_ids[idx + 1]))
        for idx, edge_id in enumerate(edge_ids)
    ]
    return concatenate_paths(*segments) if len(segments) > 0 else np.zeros((0, 3), dtype=float)


def build_stage_display_path(store: StageEvidenceStore, node_ids: list[int], edge_ids: list[int]) -> np.ndarray:
    if len(edge_ids) == 0:
        if len(node_ids) == 1:
            return np.asarray([store.graph.nodes[int(node_ids[0])].q], dtype=float)
        return np.zeros((0, 3), dtype=float)
    segments: list[np.ndarray] = []
    for idx, edge_id in enumerate(edge_ids):
        src_q = np.asarray(store.graph.nodes[int(node_ids[idx])].q, dtype=float)
        dst_q = np.asarray(store.graph.nodes[int(node_ids[idx + 1])].q, dtype=float)
        if store.stage in [LEFT_STAGE, RIGHT_STAGE]:
            sphere = unwrap_manifold(store.manifold)
            segments.append(smooth_sphere_arc(sphere.center, sphere.radius, src_q, dst_q, num=DISPLAY_SPHERE_SAMPLES))
        elif isinstance(store.manifold, MaskedManifold):
            segments.append(oriented_stage_edge_path(store, edge_id, int(node_ids[idx]), int(node_ids[idx + 1])))
        else:
            segments.append(smooth_plane_segment(src_q, dst_q, num=DISPLAY_PLANE_SAMPLES))
    return concatenate_paths(*segments) if len(segments) > 0 else np.zeros((0, 3), dtype=float)


def build_masked_stage_connector(
    store: StageEvidenceStore,
    q_start: np.ndarray,
    q_goal: np.ndarray,
) -> np.ndarray:
    start = np.asarray(q_start, dtype=float)
    goal = np.asarray(q_goal, dtype=float)
    if np.linalg.norm(goal - start) <= 1e-9:
        return np.asarray([start], dtype=float)
    if not isinstance(store.manifold, MaskedManifold):
        return np.asarray([start, goal], dtype=float)
    exact_result = solve_exact_segment_on_manifold(
        manifold=store.manifold,
        x_start=start,
        x_goal=goal,
        bounds_min=BOUNDS_MIN,
        bounds_max=BOUNDS_MAX,
    )
    path = np.asarray(getattr(exact_result, "path", np.zeros((0, 3), dtype=float)), dtype=float)
    if not getattr(exact_result, "success", False) or len(path) == 0:
        return np.zeros((0, 3), dtype=float)
    if np.linalg.norm(np.asarray(path[0], dtype=float) - start) > 5e-4:
        path = concatenate_paths(np.asarray([start], dtype=float), path)
    if np.linalg.norm(np.asarray(path[-1], dtype=float) - goal) > 5e-4:
        path = concatenate_paths(path, np.asarray([goal], dtype=float))
    return np.asarray(path, dtype=float)


def extract_committed_route_for_sequence(
    stores: dict[str, StageEvidenceStore],
    stage_sequence: list[str],
    start_stage_nodes: dict[str, int],
    goal_stage_nodes: dict[str, int],
    transition_hypotheses: list[GenericTransitionHypothesis],
) -> UnknownSequenceCandidate | None:
    if len(stage_sequence) == 0:
        return None
    start_stage = str(stage_sequence[0])
    goal_stage = str(stage_sequence[-1])
    if start_stage not in start_stage_nodes or goal_stage not in goal_stage_nodes:
        return None

    if len(stage_sequence) == 1:
        store = stores[start_stage]
        dist, prev_node, prev_edge = shortest_paths_in_stage(store, int(start_stage_nodes[start_stage]))
        goal_node_id = int(goal_stage_nodes[goal_stage])
        if goal_node_id not in dist:
            return None
        node_path, edge_path = reconstruct_stage_path(store, int(start_stage_nodes[start_stage]), goal_node_id, prev_node, prev_edge)
        raw_path = build_stage_raw_path(store, node_path, edge_path)
        display_path = build_stage_display_path(store, node_path, edge_path)
        return UnknownSequenceCandidate(
            total_cost=float(dist[goal_node_id]),
            stage_sequence=[start_stage],
            stage_node_paths={start_stage: list(node_path)},
            stage_edge_paths={start_stage: list(edge_path)},
            committed_nodes={stage: (set(node_path) if stage == start_stage else set()) for stage in stores},
            raw_path=np.asarray(raw_path, dtype=float),
            display_path=np.asarray(display_path, dtype=float),
        )

    pair_hypotheses: dict[tuple[str, str], list[GenericTransitionHypothesis]] = {}
    for hyp in transition_hypotheses:
        pair_hypotheses.setdefault((str(hyp.source_stage), str(hyp.target_stage)), []).append(hyp)

    distance_cache: dict[tuple[str, int], tuple[dict[int, float], dict[int, int], dict[int, int]]] = {}

    def shortest_from(stage_id: str, node_id: int):
        key = (str(stage_id), int(node_id))
        if key not in distance_cache:
            distance_cache[key] = shortest_paths_in_stage(stores[str(stage_id)], int(node_id))
        return distance_cache[key]

    best: UnknownSequenceCandidate | None = None

    def recurse(
        seq_index: int,
        current_node_id: int,
        accumulated_cost: float,
        raw_parts: list[np.ndarray],
        display_parts: list[np.ndarray],
        stage_node_paths: dict[str, list[int]],
        stage_edge_paths: dict[str, list[int]],
        committed_nodes: dict[str, set[int]],
    ) -> None:
        nonlocal best
        current_stage = str(stage_sequence[seq_index])
        if seq_index == len(stage_sequence) - 1:
            goal_node_id = int(goal_stage_nodes[current_stage])
            dist, prev_node, prev_edge = shortest_from(current_stage, int(current_node_id))
            if goal_node_id not in dist:
                return
            node_path, edge_path = reconstruct_stage_path(stores[current_stage], int(current_node_id), goal_node_id, prev_node, prev_edge)
            if len(node_path) == 0:
                return
            final_raw = build_stage_raw_path(stores[current_stage], node_path, edge_path)
            final_display = build_stage_display_path(stores[current_stage], node_path, edge_path)
            new_stage_node_paths = {key: list(value) for key, value in stage_node_paths.items()}
            new_stage_edge_paths = {key: list(value) for key, value in stage_edge_paths.items()}
            new_committed_nodes = {key: set(value) for key, value in committed_nodes.items()}
            new_stage_node_paths[current_stage] = list(node_path)
            new_stage_edge_paths[current_stage] = list(edge_path)
            new_committed_nodes[current_stage].update(node_path)
            raw_path = concatenate_paths(*(raw_parts + [final_raw])) if len(raw_parts) > 0 else np.asarray(final_raw, dtype=float)
            display_path = concatenate_paths(*(display_parts + [final_display])) if len(display_parts) > 0 else np.asarray(final_display, dtype=float)
            candidate = UnknownSequenceCandidate(
                total_cost=float(accumulated_cost + dist[goal_node_id]),
                stage_sequence=[str(stage) for stage in stage_sequence],
                stage_node_paths=new_stage_node_paths,
                stage_edge_paths=new_stage_edge_paths,
                committed_nodes=new_committed_nodes,
                raw_path=np.asarray(raw_path, dtype=float),
                display_path=np.asarray(display_path, dtype=float),
            )
            if best is None or candidate.total_cost + 1e-9 < best.total_cost:
                best = candidate
            return

        next_stage = str(stage_sequence[seq_index + 1])
        pair_key = (current_stage, next_stage)
        if pair_key not in pair_hypotheses:
            return
        dist, prev_node, prev_edge = shortest_from(current_stage, int(current_node_id))
        for hyp in pair_hypotheses[pair_key]:
            if hyp.source_node_id is None or hyp.target_node_id is None:
                continue
            source_transition_id = int(hyp.source_node_id)
            target_transition_id = int(hyp.target_node_id)
            if source_transition_id not in dist:
                continue
            node_path, edge_path = reconstruct_stage_path(
                stores[current_stage],
                int(current_node_id),
                source_transition_id,
                prev_node,
                prev_edge,
            )
            if len(node_path) == 0:
                continue
            stage_raw = build_stage_raw_path(stores[current_stage], node_path, edge_path)
            stage_display = build_stage_display_path(stores[current_stage], node_path, edge_path)
            transition_point = np.asarray([np.asarray(hyp.q, dtype=float)], dtype=float)
            new_raw_parts = list(raw_parts) + [stage_raw, transition_point]
            new_display_parts = list(display_parts) + [stage_display, transition_point]
            new_stage_node_paths = {key: list(value) for key, value in stage_node_paths.items()}
            new_stage_edge_paths = {key: list(value) for key, value in stage_edge_paths.items()}
            new_committed_nodes = {key: set(value) for key, value in committed_nodes.items()}
            new_stage_node_paths[current_stage] = list(node_path)
            new_stage_edge_paths[current_stage] = list(edge_path)
            new_committed_nodes[current_stage].update(node_path)
            recurse(
                seq_index + 1,
                target_transition_id,
                float(accumulated_cost + dist[source_transition_id]),
                new_raw_parts,
                new_display_parts,
                new_stage_node_paths,
                new_stage_edge_paths,
                new_committed_nodes,
            )

    recurse(
        0,
        int(start_stage_nodes[start_stage]),
        0.0,
        [],
        [],
        {str(stage): [] for stage in stores},
        {str(stage): [] for stage in stores},
        {str(stage): set() for stage in stores},
    )
    return best


def extract_committed_route(
    left_store: StageEvidenceStore,
    plane_store: StageEvidenceStore,
    right_store: StageEvidenceStore,
    start_node_id: int,
    goal_node_id: int,
    left_plane_hypotheses: list[TransitionHypothesis],
    plane_right_hypotheses: list[TransitionHypothesis],
) -> tuple[SequentialRouteCandidate | None, bool, bool, int]:
    left_dist, left_prev_node, left_prev_edge = shortest_paths_in_stage(left_store, start_node_id)
    right_dist, right_prev_node, right_prev_edge = shortest_paths_in_stage(right_store, goal_node_id)

    entry_candidates = [hyp for hyp in left_plane_hypotheses if hyp.left_node_id in left_dist]
    exit_candidates = [hyp for hyp in plane_right_hypotheses if hyp.right_node_id in right_dist]
    has_committed_entry = len(entry_candidates) > 0
    has_committed_exit = len(exit_candidates) > 0
    if not has_committed_entry or not has_committed_exit:
        return None, has_committed_entry, has_committed_exit, 0

    best: SequentialRouteCandidate | None = None
    pairs_evaluated = 0
    for entry_hyp in entry_candidates:
        plane_dist, plane_prev_node, plane_prev_edge = shortest_paths_in_stage(plane_store, int(entry_hyp.plane_node_id))
        for exit_hyp in exit_candidates:
            pairs_evaluated += 1
            plane_exit_id = int(exit_hyp.plane_node_id)
            if plane_exit_id not in plane_dist:
                continue
            total_cost = float(left_dist[int(entry_hyp.left_node_id)] + plane_dist[plane_exit_id] + right_dist[int(exit_hyp.right_node_id)])
            left_node_path, left_edge_path = reconstruct_stage_path(left_store, start_node_id, int(entry_hyp.left_node_id), left_prev_node, left_prev_edge)
            plane_node_path, plane_edge_path = reconstruct_stage_path(plane_store, int(entry_hyp.plane_node_id), plane_exit_id, plane_prev_node, plane_prev_edge)
            right_goal_to_entry_nodes, right_goal_to_entry_edges = reconstruct_stage_path(right_store, goal_node_id, int(exit_hyp.right_node_id), right_prev_node, right_prev_edge)
            if len(left_node_path) == 0 or len(plane_node_path) == 0 or len(right_goal_to_entry_nodes) == 0:
                continue
            right_node_path = list(reversed(right_goal_to_entry_nodes))
            right_edge_path = list(reversed(right_goal_to_entry_edges))

            left_raw_path = build_stage_raw_path(left_store, left_node_path, left_edge_path)
            plane_raw_path = build_stage_raw_path(plane_store, plane_node_path, plane_edge_path)
            right_raw_path = build_stage_raw_path(right_store, right_node_path, right_edge_path)
            left_display_path = build_stage_display_path(left_store, left_node_path, left_edge_path)
            plane_display_path = build_stage_display_path(plane_store, plane_node_path, plane_edge_path)
            right_display_path = build_stage_display_path(right_store, right_node_path, right_edge_path)

            plane_entry_anchor = (
                np.asarray(plane_store.graph.nodes[int(plane_node_path[0])].q, dtype=float)
                if len(plane_node_path) > 0
                else np.asarray(entry_hyp.q, dtype=float)
            )
            plane_exit_anchor = (
                np.asarray(plane_store.graph.nodes[int(plane_node_path[-1])].q, dtype=float)
                if len(plane_node_path) > 0
                else np.asarray(exit_hyp.q, dtype=float)
            )
            right_entry_anchor = (
                np.asarray(right_store.graph.nodes[int(right_node_path[0])].q, dtype=float)
                if len(right_node_path) > 0
                else np.asarray(exit_hyp.q, dtype=float)
            )

            plane_entry_connector = build_masked_stage_connector(
                plane_store,
                np.asarray(entry_hyp.q, dtype=float),
                plane_entry_anchor,
            )
            plane_exit_connector = build_masked_stage_connector(
                plane_store,
                plane_exit_anchor,
                np.asarray(exit_hyp.q, dtype=float),
            )
            if len(plane_entry_connector) == 0 or len(plane_exit_connector) == 0:
                continue

            raw_path = concatenate_paths(
                left_raw_path,
                plane_entry_connector,
                plane_raw_path,
                plane_exit_connector,
                right_raw_path,
            )
            display_path = concatenate_paths(
                left_display_path,
                plane_entry_connector,
                plane_display_path,
                plane_exit_connector,
                right_display_path if len(right_display_path) > 0 else np.asarray([right_entry_anchor], dtype=float),
            )
            candidate = SequentialRouteCandidate(
                total_cost=total_cost,
                left_node_path=left_node_path,
                left_edge_path=left_edge_path,
                plane_node_path=plane_node_path,
                plane_edge_path=plane_edge_path,
                right_node_path=right_node_path,
                right_edge_path=right_edge_path,
                committed_nodes={
                    LEFT_STAGE: set(left_node_path),
                    PLANE_STAGE: set(plane_node_path),
                    RIGHT_STAGE: set(right_node_path),
                },
                raw_path=np.asarray(raw_path, dtype=float),
                display_path=np.asarray(display_path, dtype=float),
            )
            if best is None or candidate.total_cost + 1e-9 < best.total_cost:
                best = candidate
    return best, has_committed_entry, has_committed_exit, pairs_evaluated


def recent_route_improvement(best_cost_history: list[float]) -> int:
    if len(best_cost_history) < 2:
        return 0
    recent = best_cost_history[-PROGRESS_WINDOW:]
    improvements = 0
    for idx in range(1, len(recent)):
        if recent[idx] + 1e-9 < recent[idx - 1]:
            improvements += 1
    return improvements


def should_stop_exploration(
    first_solution_round: int | None,
    total_rounds: int,
    stage_node_gains: dict[str, list[int]],
    stage_transition_gains: dict[str, list[int]],
    stage_route_gains: dict[str, list[int]],
    current_stage_counts: dict[str, int],
) -> bool:
    if total_rounds >= SAFETY_MAX_TOTAL_ROUNDS:
        return True
    if total_rounds < MIN_ROUNDS_BEFORE_SATURATION_CHECK:
        return False

    stage_names = list(stage_node_gains.keys())
    node_gain = 0
    transition_gain = 0
    route_gain = 0
    for stage in stage_names:
        node_gain += stage_recent_sum(stage_node_gains[stage], window=SATURATION_WINDOW)
        transition_gain += stage_recent_sum(stage_transition_gains[stage], window=SATURATION_WINDOW)
        route_gain += stage_recent_sum(stage_route_gains[stage], window=SATURATION_WINDOW)
    if all(stage in current_stage_counts for stage in [PLANE_STAGE, LEFT_STAGE, RIGHT_STAGE]):
        plane_lagging = current_stage_counts[PLANE_STAGE] < 0.70 * max(current_stage_counts[LEFT_STAGE], current_stage_counts[RIGHT_STAGE], 1)
    else:
        plane_lagging = False

    if first_solution_round is None:
        return node_gain == 0 and transition_gain == 0 and not plane_lagging
    if total_rounds - first_solution_round < MIN_POST_SOLUTION_ROUNDS:
        return False
    return node_gain == 0 and transition_gain == 0 and route_gain == 0 and not plane_lagging


def _coerce_stage_manifolds(stage_manifolds) -> tuple[list[str], dict[str, object]]:
    stage_ids: list[str] = []
    manifold_map: dict[str, object] = {}
    for idx, item in enumerate(stage_manifolds):
        if isinstance(item, tuple) and len(item) == 2:
            stage_id, manifold = item
        else:
            stage_id = f"stage_{idx}"
            manifold = item
        stage_name = str(stage_id)
        stage_ids.append(stage_name)
        manifold_map[stage_name] = manifold
    return stage_ids, manifold_map


def _stage_boundary_nodes(
    stores: dict[str, StageEvidenceStore],
    stage_ids: list[str],
    q: np.ndarray,
    seeded_from_proposal: bool = False,
) -> dict[str, int]:
    nodes: dict[str, int] = {}
    qq = np.asarray(q, dtype=float)
    for stage_id in stage_ids:
        manifold = stores[stage_id].manifold
        projected = proposal_projection(manifold, qq)
        if projected is None:
            continue
        if float(np.linalg.norm(np.asarray(projected, dtype=float) - qq)) > 1e-3:
            continue
        nodes[stage_id] = add_stage_node(stores[stage_id], qq, seeded_from_proposal=seeded_from_proposal)
    return nodes


def build_unknown_sequence_scene():
    families, start_q, goal_q, plane_half_u, plane_half_v = build_scene()
    left_family, plane_family, right_family = families
    bridge_family = SphereFamily(
        name="bridge_support_3d",
        center=np.array([0.0, 0.05, 0.48], dtype=float),
        radii={1.42: 1.42},
    )
    stage_manifolds = [
        ("left_support", left_family.manifold(float(left_family.sample_lambdas()[0]))),
        ("transfer_plane", plane_family.manifold(float(plane_family.sample_lambdas()[0]))),
        ("bridge_sphere", bridge_family.manifold(float(bridge_family.sample_lambdas()[0]))),
        ("right_support", right_family.manifold(float(right_family.sample_lambdas()[0]))),
    ]
    return stage_manifolds, start_q, goal_q, plane_half_u, plane_half_v


def compare_fixed_and_unknown_sequence_demo(
    serial_mode: bool = False,
) -> dict[str, object]:
    """Compare the original fixed-sequence planner to the free-sequence variant.

    The fixed-sequence planner remains constrained to left -> plane -> right.
    The unknown-sequence planner receives an additional bridge manifold and may
    discover that a different manifold ordering yields a feasible route.
    """

    families, start_q, goal_q, _plane_half_u, _plane_half_v = build_scene()
    fixed_result = plan_fixed_manifold_multimodal_route(
        families=families,
        start_q=np.asarray(start_q, dtype=float),
        goal_q=np.asarray(goal_q, dtype=float),
        serial_mode=serial_mode,
    )

    stage_manifolds, stage_start_q, stage_goal_q, _plane_half_u, _plane_half_v = build_unknown_sequence_scene()
    unknown_result = plan_multimodal_unknown_sequence(
        stage_manifolds=stage_manifolds,
        start_q=np.asarray(stage_start_q, dtype=float),
        goal_q=np.asarray(stage_goal_q, dtype=float),
        serial_mode=serial_mode,
    )

    return {
        "fixed_success": bool(fixed_result.success),
        "fixed_message": str(fixed_result.message),
        "fixed_route_cost": float(fixed_result.route_cost_display),
        "fixed_total_rounds": int(fixed_result.total_rounds),
        "unknown_success": bool(unknown_result.success),
        "unknown_message": str(unknown_result.message),
        "unknown_sequence": list(unknown_result.discovered_sequence),
        "unknown_route_cost": float(unknown_result.route_cost_display),
        "unknown_total_rounds": int(unknown_result.total_rounds),
        "unknown_meta_graph_edges": list(unknown_result.meta_graph_edges),
    }


def plan_multimodal_unknown_sequence(
    stage_manifolds,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    serial_mode: bool = False,
) -> UnknownSequenceRoute:
    stage_ids, manifold_map = _coerce_stage_manifolds(stage_manifolds)
    stores = {
        stage_id: StageEvidenceStore(stage=stage_id, manifold=manifold_map[stage_id])
        for stage_id in stage_ids
    }
    start_stage_nodes = _stage_boundary_nodes(stores, stage_ids, np.asarray(start_q, dtype=float), seeded_from_proposal=False)
    goal_stage_nodes = _stage_boundary_nodes(stores, stage_ids, np.asarray(goal_q, dtype=float), seeded_from_proposal=False)
    if len(start_stage_nodes) == 0 or len(goal_stage_nodes) == 0:
        anchor = np.asarray(start_q if len(start_stage_nodes) == 0 else goal_q, dtype=float).reshape(1, 3)
        return UnknownSequenceRoute(
            success=False,
            message="Could not place start or goal on any provided manifold stage.",
            discovered_sequence=[],
            total_rounds=0,
            candidate_evaluations=0,
            shared_proposals_processed=0,
            global_transition_hypotheses=0,
            meta_graph_edges=[],
            start_stages=sorted(start_stage_nodes),
            goal_stages=sorted(goal_stage_nodes),
            path=anchor,
            raw_path=anchor,
            route_cost_raw=0.0,
            route_cost_display=0.0,
        )

    for stage_id, node_id in start_stage_nodes.items():
        stores[stage_id].frontier_ids = [int(node_id)]
    for stage_id, node_id in goal_stage_nodes.items():
        if int(node_id) not in stores[stage_id].frontier_ids:
            stores[stage_id].frontier_ids.append(int(node_id))

    candidate_evaluations = 0
    total_rounds = 0
    shared_proposals_processed = 0
    useful_stage_total = 0
    multi_stage_update_total = 0
    proposal_rounds_with_multi_stage_updates = 0
    stage_node_gains = {stage_id: [] for stage_id in stage_ids}
    stage_transition_gains = {stage_id: [] for stage_id in stage_ids}
    stage_route_gains = {stage_id: [] for stage_id in stage_ids}
    best_cost_history: list[float] = []
    best_candidate: UnknownSequenceCandidate | None = None
    best_sequence: list[str] = []
    first_solution_round: int | None = None
    mode_counts = {"shared_proposal_round": 0}
    if serial_mode:
        mode_counts["serial_round"] = 0

    guide_center = 0.5 * (np.asarray(start_q, dtype=float) + np.asarray(goal_q, dtype=float))
    guides = {
        stage_id: (
            np.asarray(goal_q, dtype=float)
            if stage_id in goal_stage_nodes
            else (np.asarray(start_q, dtype=float) if stage_id in start_stage_nodes else guide_center.copy())
        )
        for stage_id in stage_ids
    }
    meta_graph = MetaGraph()
    for stage_id in stage_ids:
        meta_graph.add_node(stage_id)
    global_hypotheses: list[GenericTransitionHypothesis] = []

    for round_idx in range(1, SAFETY_MAX_TOTAL_ROUNDS + 1):
        current_stage_counts = {stage_id: len(stores[stage_id].graph.nodes) for stage_id in stage_ids}
        if should_stop_exploration(
            first_solution_round=first_solution_round,
            total_rounds=round_idx - 1,
            stage_node_gains={stage: stage_node_gains[stage] for stage in stage_ids},
            stage_transition_gains={stage: stage_transition_gains[stage] for stage in stage_ids},
            stage_route_gains={stage: stage_route_gains[stage] for stage in stage_ids},
            current_stage_counts=current_stage_counts,
        ):
            break

        total_rounds = round_idx
        mode_counts["shared_proposal_round"] += 1
        round_node_gain = {stage_id: 0 for stage_id in stage_ids}
        round_transition_gain = {stage_id: 0 for stage_id in stage_ids}
        route_improved_this_round = 0
        active_serial_stage = max(stage_ids, key=lambda stage_id: stage_underexploration_factor(stage_id, stores)) if serial_mode else None
        if serial_mode:
            mode_counts["serial_round"] += 1
            mode_counts[f"serial_active_{active_serial_stage}"] = mode_counts.get(f"serial_active_{active_serial_stage}", 0) + 1

        proposal_count = effective_proposals_per_round(
            round_idx,
            first_solution_round,
            {stage: stores[stage] for stage in stage_ids},
        )
        proposals = generate_ambient_proposals(
            round_idx=round_idx,
            start_q=np.asarray(start_q, dtype=float),
            goal_q=np.asarray(goal_q, dtype=float),
            plane_point=guide_center,
            stores={stage: stores[stage] for stage in stage_ids},
            guides={stage: guides[stage] for stage in stage_ids},
            proposal_count=proposal_count,
        )

        promising_meta_path = bfs_meta_path(meta_graph, sorted(start_stage_nodes), sorted(goal_stage_nodes))
        promising_stages = set(promising_meta_path)

        for proposal in proposals:
            shared_proposals_processed += 1
            useful: dict[str, np.ndarray] = {}
            useful_scores: dict[str, float] = {}
            for stage_id in stage_ids:
                store = stores[stage_id]
                projection = proposal_projection(store.manifold, np.asarray(proposal, dtype=float))
                if projection is None:
                    continue
                utility = proposal_stage_utility(
                    stage=stage_id,
                    projected_q=projection,
                    known_points=stage_evidence_points(store),
                    guide_point=guides[stage_id],
                    stores=stores,
                )
                if stage_id in promising_stages:
                    utility += 0.08
                if utility >= 0.14 or len(store.graph.nodes) == 0:
                    useful[stage_id] = np.asarray(projection, dtype=float)
                    useful_scores[stage_id] = float(utility)
            useful_stage_total += len(useful)
            ranked_stages = [
                stage
                for _, stage in sorted([(score, stage) for stage, score in useful_scores.items()], reverse=True)
            ]
            if len(ranked_stages) >= 2:
                multi_stage_update_total += 1
            active_update_stages = [active_serial_stage] if serial_mode and active_serial_stage in ranked_stages else list(ranked_stages[: adaptive_stage_update_budget(stores, first_solution_round)])
            active_update_stage_set = set(active_update_stages)
            if len(active_update_stage_set) >= 2:
                proposal_rounds_with_multi_stage_updates += 1
            passive_seed_stages = [] if serial_mode else [stage for stage in ranked_stages if stage not in active_update_stage_set]

            for stage_id in passive_seed_stages:
                node_id = seed_stage_evidence_only(stores[stage_id], useful[stage_id], guides[stage_id])
                if node_id >= 0:
                    round_node_gain[stage_id] += 1

            for stage_id in ranked_stages:
                if stage_id not in active_update_stage_set:
                    continue
                store = stores[stage_id]
                node_gain, eval_gain, result, source_node_id = update_stage_evidence_from_proposal(
                    store=store,
                    target_q=np.asarray(useful[stage_id], dtype=float),
                    guide_point=guides[stage_id],
                )
                candidate_evaluations += eval_gain
                round_node_gain[stage_id] += node_gain
                if result is None:
                    continue
                for target_stage in stage_ids:
                    if str(target_stage) == str(stage_id):
                        continue
                    transition_gain, evals = add_generic_transition_hypotheses(
                        source_stage=stage_id,
                        target_stage=target_stage,
                        stores=stores,
                        result=result,
                        source_node_id=source_node_id,
                        guide_point=guides[target_stage],
                        hypotheses=global_hypotheses,
                    )
                    candidate_evaluations += evals
                    round_transition_gain[stage_id] += transition_gain
                    round_transition_gain[target_stage] += transition_gain

        meta_graph = build_meta_graph(stage_ids, global_hypotheses)
        promising_meta_path = bfs_meta_path(meta_graph, sorted(start_stage_nodes), sorted(goal_stage_nodes))
        if len(promising_meta_path) > 0:
            pair_hypotheses: dict[tuple[str, str], list[GenericTransitionHypothesis]] = {}
            for hyp in global_hypotheses:
                pair_hypotheses.setdefault((str(hyp.source_stage), str(hyp.target_stage)), []).append(hyp)

            if len(promising_meta_path) >= 2:
                first_stage = str(promising_meta_path[0])
                second_stage = str(promising_meta_path[1])
                first_pair = pair_hypotheses.get((first_stage, second_stage), [])
                first_targets = [
                    int(hyp.source_node_id)
                    for hyp in first_pair
                    if hyp.source_node_id is not None
                ]
                bridge_gain, bridge_evals = bridge_stage_node_sets(
                    store=stores[first_stage],
                    source_node_ids=[int(start_stage_nodes[first_stage])],
                    target_node_ids=first_targets,
                    guide_point=guides[first_stage],
                    motion_kind=f"{first_stage}_motion",
                    max_tries=3,
                )
                candidate_evaluations += bridge_evals
                round_node_gain[first_stage] += bridge_gain

                for seq_index in range(1, len(promising_meta_path) - 1):
                    current_stage = str(promising_meta_path[seq_index])
                    previous_stage = str(promising_meta_path[seq_index - 1])
                    next_stage = str(promising_meta_path[seq_index + 1])
                    incoming = [
                        int(hyp.target_node_id)
                        for hyp in pair_hypotheses.get((previous_stage, current_stage), [])
                        if hyp.target_node_id is not None
                    ]
                    outgoing = [
                        int(hyp.source_node_id)
                        for hyp in pair_hypotheses.get((current_stage, next_stage), [])
                        if hyp.source_node_id is not None
                    ]
                    bridge_gain, bridge_evals = bridge_stage_node_sets(
                        store=stores[current_stage],
                        source_node_ids=incoming,
                        target_node_ids=outgoing,
                        guide_point=guides[current_stage],
                        motion_kind=f"{current_stage}_motion",
                        max_tries=4,
                    )
                    candidate_evaluations += bridge_evals
                    round_node_gain[current_stage] += bridge_gain

                final_stage = str(promising_meta_path[-1])
                previous_stage = str(promising_meta_path[-2])
                final_sources = [
                    int(hyp.target_node_id)
                    for hyp in pair_hypotheses.get((previous_stage, final_stage), [])
                    if hyp.target_node_id is not None
                ]
                bridge_gain, bridge_evals = bridge_stage_node_sets(
                    store=stores[final_stage],
                    source_node_ids=final_sources,
                    target_node_ids=[int(goal_stage_nodes[final_stage])],
                    guide_point=guides[final_stage],
                    motion_kind=f"{final_stage}_motion",
                    max_tries=3,
                )
                candidate_evaluations += bridge_evals
                round_node_gain[final_stage] += bridge_gain

            candidate = extract_committed_route_for_sequence(
                stores=stores,
                stage_sequence=promising_meta_path,
                start_stage_nodes=start_stage_nodes,
                goal_stage_nodes=goal_stage_nodes,
                transition_hypotheses=global_hypotheses,
            )
            if candidate is not None:
                if best_candidate is None or candidate.total_cost + 1e-9 < best_candidate.total_cost:
                    best_candidate = candidate
                    best_sequence = list(promising_meta_path)
                    route_improved_this_round = 1
                    if first_solution_round is None:
                        first_solution_round = round_idx

        best_cost_history.append(best_candidate.total_cost if best_candidate is not None else 1e12)
        for stage_id in stage_ids:
            stage_node_gains[stage_id].append(int(round_node_gain[stage_id]))
            stage_transition_gains[stage_id].append(int(round_transition_gain[stage_id]))
            stage_route_gains[stage_id].append(int(route_improved_this_round))

    success = best_candidate is not None
    raw_path = np.asarray([np.asarray(start_q, dtype=float)], dtype=float)
    display_path = np.asarray([np.asarray(start_q, dtype=float)], dtype=float)
    route_cost_raw = 0.0
    route_cost_display = 0.0
    committed_stage_nodes = {stage_id: np.zeros((0, 3), dtype=float) for stage_id in stage_ids}
    if best_candidate is not None:
        raw_path = np.asarray(best_candidate.raw_path, dtype=float)
        display_path = np.asarray(best_candidate.display_path, dtype=float)
        route_cost_raw = path_cost(raw_path)
        route_cost_display = path_cost(display_path)
        committed_stage_nodes = {
            stage_id: (
                np.asarray([stores[stage_id].graph.nodes[node_id].q for node_id in sorted(best_candidate.committed_nodes.get(stage_id, set()))], dtype=float)
                if len(best_candidate.committed_nodes.get(stage_id, set())) > 0
                else np.zeros((0, 3), dtype=float)
            )
            for stage_id in stage_ids
        }

    stage_stagnation_flags = {
        stage_id: stage_stagnating(stage_node_gains[stage_id], stage_transition_gains[stage_id], stage_route_gains[stage_id])
        for stage_id in stage_ids
    }
    saturated_before_solution = bool(not success and total_rounds > 0)
    stagnation_stage = None
    if saturated_before_solution:
        stagnant = [stage for stage in stage_ids if stage_stagnation_flags.get(stage, False)]
        if len(stagnant) == 1:
            stagnation_stage = stagnant[0]
        elif len(stagnant) == len(stage_ids):
            stagnation_stage = "all"
        elif len(stagnant) > 1:
            stagnation_stage = ",".join(stagnant)

    return UnknownSequenceRoute(
        success=success,
        message=(
            "A feasible manifold sequence was discovered from the meta-graph and converted into a committed multimodal route."
            if success
            else "Parallel evidence grew on the provided manifolds, but no feasible start-to-goal manifold sequence produced a committed route before saturation."
        ),
        discovered_sequence=list(best_sequence),
        total_rounds=total_rounds,
        candidate_evaluations=candidate_evaluations,
        shared_proposals_processed=shared_proposals_processed,
        global_transition_hypotheses=len(global_hypotheses),
        meta_graph_edges=meta_graph.edges(),
        start_stages=sorted(start_stage_nodes),
        goal_stages=sorted(goal_stage_nodes),
        path=np.asarray(display_path, dtype=float),
        raw_path=np.asarray(raw_path, dtype=float),
        route_cost_raw=float(route_cost_raw),
        route_cost_display=float(route_cost_display),
        stage_node_counts={stage_id: len(stores[stage_id].graph.nodes) for stage_id in stage_ids},
        stage_frontier_counts={stage_id: len(stores[stage_id].frontier_ids) for stage_id in stage_ids},
        stage_evidence_points={stage_id: stage_evidence_points(stores[stage_id]) for stage_id in stage_ids},
        stage_evidence_edges={stage_id: stores[stage_id].explored_edges for stage_id in stage_ids},
        stage_frontier_points={stage_id: stage_frontier_points(stores[stage_id]) for stage_id in stage_ids},
        stage_chart_centers={stage_id: np.asarray(stores[stage_id].chart_centers, dtype=float) for stage_id in stage_ids},
        committed_stage_nodes=committed_stage_nodes,
        transition_points=deduplicate_points([hyp.q for hyp in global_hypotheses], tol=TRANSITION_DEDUP_TOL),
        saturated_before_solution=saturated_before_solution,
        stagnation_stage=stagnation_stage,
        mode_counts=mode_counts,
    )


def plan_fixed_manifold_multimodal_route(
    families,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    robot=None,
    serial_mode: bool = False,
    obstacles=None,
) -> FixedPlaneRoute:
    if robot is not None:
        from .jointspace_helpers import plan_fixed_manifold_multimodal_route_jointspace

        return plan_fixed_manifold_multimodal_route_jointspace(
            families=families,
            start_q=np.asarray(start_q, dtype=float),
            goal_q=np.asarray(goal_q, dtype=float),
            robot=robot,
            serial_mode=serial_mode,
            obstacles=obstacles,
        )

    left_family, plane_family, right_family = families
    left_manifold = left_family.manifold(float(left_family.sample_lambdas()[0]))
    plane_manifold = plane_family.manifold(float(plane_family.sample_lambdas()[0]))
    right_manifold = right_family.manifold(float(right_family.sample_lambdas()[0]))

    stores = {
        LEFT_STAGE: StageEvidenceStore(stage=LEFT_STAGE, manifold=left_manifold),
        PLANE_STAGE: StageEvidenceStore(stage=PLANE_STAGE, manifold=plane_manifold),
        RIGHT_STAGE: StageEvidenceStore(stage=RIGHT_STAGE, manifold=right_manifold),
    }

    start_node_id = add_stage_node(stores[LEFT_STAGE], np.asarray(start_q, dtype=float), seeded_from_proposal=False)
    goal_node_id = add_stage_node(stores[RIGHT_STAGE], np.asarray(goal_q, dtype=float), seeded_from_proposal=False)
    stores[LEFT_STAGE].frontier_ids = [start_node_id]
    stores[RIGHT_STAGE].frontier_ids = [goal_node_id]

    left_plane_hypotheses: list[TransitionHypothesis] = []
    plane_right_hypotheses: list[TransitionHypothesis] = []

    candidate_evaluations = 0
    total_rounds = 0
    shared_proposals_processed = 0
    proposals_used_by_multiple_stages = 0
    useful_stage_total = 0
    multi_stage_update_total = 0
    proposal_rounds_with_plane_updates = 0
    proposal_rounds_with_multi_stage_updates = 0
    committed_route_changes_after_first_solution = 0
    alternative_hypothesis_pairs_evaluated = 0

    stage_node_gains = {stage: [] for stage in STAGES}
    stage_transition_gains = {stage: [] for stage in STAGES}
    stage_route_gains = {stage: [] for stage in STAGES}
    best_cost_history: list[float] = []

    best_candidate: SequentialRouteCandidate | None = None
    first_solution_round: int | None = None
    best_solution_round: int | None = None
    first_committed_entry_round: int | None = None
    first_committed_exit_round: int | None = None
    plane_evidence_before_first_committed_entry = 0
    right_evidence_before_first_committed_exit = 0
    plane_evidence_at_first_solution = 0
    right_evidence_at_first_solution = 0

    plane_point = np.asarray(unwrap_manifold(plane_manifold).point, dtype=float)
    guides = {
        LEFT_STAGE: project_point_to_plane(plane_manifold, np.asarray(start_q, dtype=float)),
        PLANE_STAGE: project_point_to_plane(plane_manifold, 0.5 * (np.asarray(start_q, dtype=float) + np.asarray(goal_q, dtype=float))),
        RIGHT_STAGE: np.asarray(goal_q, dtype=float),
    }

    mode_counts = {
        "shared_proposal_round": 0,
        "right_goal_bias_updates": 0,
    }
    if serial_mode:
        mode_counts["serial_round"] = 0

    for round_idx in range(1, SAFETY_MAX_TOTAL_ROUNDS + 1):
        current_stage_counts = stage_evidence_counts(stores)
        if should_stop_exploration(
            first_solution_round=first_solution_round,
            total_rounds=round_idx - 1,
            stage_node_gains=stage_node_gains,
            stage_transition_gains=stage_transition_gains,
            stage_route_gains=stage_route_gains,
            current_stage_counts=current_stage_counts,
        ):
            break

        total_rounds = round_idx
        mode_counts["shared_proposal_round"] += 1

        round_node_gain = {stage: 0 for stage in STAGES}
        round_transition_gain = {stage: 0 for stage in STAGES}
        route_improved_this_round = 0
        plane_updated_this_round = False
        multi_stage_updated_this_round = False
        active_serial_stage = greedy_stage_for_serial_round(stores) if serial_mode else None
        if serial_mode:
            mode_counts["serial_round"] += 1
            mode_counts[f"serial_active_{active_serial_stage}"] = mode_counts.get(f"serial_active_{active_serial_stage}", 0) + 1

        proposal_count = effective_proposals_per_round(round_idx, first_solution_round, stores)
        proposals = generate_ambient_proposals(
            round_idx=round_idx,
            start_q=start_q,
            goal_q=goal_q,
            plane_point=plane_point,
            stores=stores,
            guides=guides,
            proposal_count=proposal_count,
        )
        for proposal in proposals:
            shared_proposals_processed += 1
            useful: dict[str, np.ndarray] = {}
            useful_scores: dict[str, float] = {}
            for stage, store in stores.items():
                projection = proposal_projection(store.manifold, np.asarray(proposal, dtype=float))
                if projection is None:
                    continue
                utility = proposal_stage_utility(
                    stage=stage,
                    projected_q=projection,
                    known_points=stage_evidence_points(store),
                    guide_point=guides[stage],
                    stores=stores,
                )
                if utility >= 0.16 or len(store.graph.nodes) == 0:
                    useful[stage] = np.asarray(projection, dtype=float)
                    useful_scores[stage] = float(utility)
            useful_stage_total += len(useful)
            if len(useful) >= 2:
                proposals_used_by_multiple_stages += 1

            ranked_stages = [
                stage
                for _, stage in sorted(
                    [(score, stage) for stage, score in useful_scores.items()],
                    reverse=True,
                )
            ]
            if serial_mode:
                active_update_stages = [active_serial_stage] if active_serial_stage in ranked_stages else []
            else:
                stage_update_budget = adaptive_stage_update_budget(stores, first_solution_round)
                active_update_stages = list(ranked_stages[:stage_update_budget])
                if (
                    PLANE_STAGE in ranked_stages
                    and stage_underexploration_factor(PLANE_STAGE, stores) > 0.10
                    and PLANE_STAGE not in active_update_stages
                ):
                    if len(active_update_stages) >= stage_update_budget and len(active_update_stages) > 0:
                        active_update_stages[-1] = PLANE_STAGE
                    else:
                        active_update_stages.append(PLANE_STAGE)
            active_update_stage_set = set(active_update_stages)
            if len(active_update_stage_set) >= 2:
                multi_stage_update_total += 1
                multi_stage_updated_this_round = True
            passive_seed_stages = [] if serial_mode else [stage for stage in ranked_stages if stage not in active_update_stages]

            for stage in passive_seed_stages:
                node_id = seed_stage_evidence_only(stores[stage], useful[stage], guides[stage])
                if node_id >= 0:
                    round_node_gain[stage] += 1
                    if stage == PLANE_STAGE:
                        plane_updated_this_round = True

            for stage in ranked_stages:
                if stage not in active_update_stage_set:
                    continue
                store = stores[stage]
                node_gain, eval_gain, result, source_node_id = update_stage_evidence_from_proposal(
                    store=store,
                    target_q=np.asarray(useful[stage], dtype=float),
                    guide_point=guides[stage],
                )
                candidate_evaluations += eval_gain
                round_node_gain[stage] += node_gain
                if stage == PLANE_STAGE and (node_gain > 0 or result is not None):
                    plane_updated_this_round = True

                if stage in [LEFT_STAGE, PLANE_STAGE]:
                    transition_gain, evals = add_left_plane_hypotheses(
                        source_stage=stage,
                        source_store=store,
                        plane_store=stores[PLANE_STAGE],
                        left_store=stores[LEFT_STAGE],
                        result=result,
                        source_node_id=source_node_id,
                        guide_point=guides[PLANE_STAGE],
                        hypotheses=left_plane_hypotheses,
                    )
                    candidate_evaluations += evals
                    round_transition_gain[LEFT_STAGE] += transition_gain
                    round_transition_gain[PLANE_STAGE] += transition_gain
                    if stage == PLANE_STAGE and transition_gain > 0:
                        plane_updated_this_round = True

                if stage in [PLANE_STAGE, RIGHT_STAGE]:
                    transition_gain, evals = add_plane_right_hypotheses(
                        source_stage=stage,
                        source_store=store,
                        plane_store=stores[PLANE_STAGE],
                        right_store=stores[RIGHT_STAGE],
                        result=result,
                        source_node_id=source_node_id,
                        guide_point=guides[RIGHT_STAGE],
                        hypotheses=plane_right_hypotheses,
                    )
                    candidate_evaluations += evals
                    round_transition_gain[PLANE_STAGE] += transition_gain
                    round_transition_gain[RIGHT_STAGE] += transition_gain
                    if stage == PLANE_STAGE and transition_gain > 0:
                        plane_updated_this_round = True

        if not serial_mode or active_serial_stage == LEFT_STAGE:
            left_bridge_gain, left_bridge_evals = bridge_left_hypotheses_to_start(
                left_store=stores[LEFT_STAGE],
                start_node_id=start_node_id,
                left_plane_hypotheses=left_plane_hypotheses,
                guide_point=guides[PLANE_STAGE],
            )
            candidate_evaluations += left_bridge_evals
            round_node_gain[LEFT_STAGE] += left_bridge_gain

        if not serial_mode or active_serial_stage == PLANE_STAGE:
            plane_bridge_gain, plane_bridge_evals = bridge_plane_hypothesis_components(
                plane_store=stores[PLANE_STAGE],
                left_plane_hypotheses=left_plane_hypotheses,
                plane_right_hypotheses=plane_right_hypotheses,
            )
            candidate_evaluations += plane_bridge_evals
            round_node_gain[PLANE_STAGE] += plane_bridge_gain
            if plane_bridge_gain > 0:
                plane_updated_this_round = True

        if not serial_mode or active_serial_stage == RIGHT_STAGE:
            right_bridge_gain, right_bridge_evals = connect_right_hypothesis_to_goal(
                right_store=stores[RIGHT_STAGE],
                plane_right_hypotheses=plane_right_hypotheses,
                goal_node_id=goal_node_id,
                guide_point=guides[RIGHT_STAGE],
            )
            candidate_evaluations += right_bridge_evals
            round_node_gain[RIGHT_STAGE] += right_bridge_gain

        # Once right-side evidence exists, bias one extra exact-goal closure pass
        # so future-stage evidence is not purely passive.
        if len(stores[RIGHT_STAGE].frontier_ids) > 0 and (not serial_mode or active_serial_stage == RIGHT_STAGE):
            mode_counts["right_goal_bias_updates"] += 1
            for source_node_id in ranked_stage_sources(stores[RIGHT_STAGE], guides[RIGHT_STAGE], limit=3):
                increment_stage_node_expansion(stores[RIGHT_STAGE], source_node_id)
                source_q = stores[RIGHT_STAGE].graph.nodes[source_node_id].q
                exact_result = solve_exact_segment_on_manifold(
                    manifold=stores[RIGHT_STAGE].manifold,
                    x_start=source_q,
                    x_goal=np.asarray(goal_q, dtype=float),
                    bounds_min=BOUNDS_MIN,
                    bounds_max=BOUNDS_MAX,
                )
                candidate_evaluations += 1
                if not exact_result.success:
                    continue
                stores[RIGHT_STAGE].explored_edges = merge_edges(
                    stores[RIGHT_STAGE].explored_edges,
                    list(getattr(exact_result, "explored_edges", [])),
                )
                stores[RIGHT_STAGE].chart_centers = merge_chart_centers(stores[RIGHT_STAGE].chart_centers, exact_result)
                _, path_nodes, _ = connect_path_to_stage_graph(
                    store=stores[RIGHT_STAGE],
                    source_node_id=source_node_id,
                    path=np.asarray(exact_result.path, dtype=float),
                    kind=RIGHT_MOTION,
                    terminal_node_id=goal_node_id,
                )
                update_stage_frontier(stores[RIGHT_STAGE], path_nodes + [goal_node_id], guides[RIGHT_STAGE])
                round_node_gain[RIGHT_STAGE] += max(0, len(path_nodes) - 1)
                break

        candidate, has_committed_entry, has_committed_exit, pairs_evaluated = extract_committed_route(
            left_store=stores[LEFT_STAGE],
            plane_store=stores[PLANE_STAGE],
            right_store=stores[RIGHT_STAGE],
            start_node_id=start_node_id,
            goal_node_id=goal_node_id,
            left_plane_hypotheses=left_plane_hypotheses,
            plane_right_hypotheses=plane_right_hypotheses,
        )
        alternative_hypothesis_pairs_evaluated += pairs_evaluated

        if has_committed_entry and first_committed_entry_round is None:
            first_committed_entry_round = round_idx
            plane_evidence_before_first_committed_entry = len(stores[PLANE_STAGE].graph.nodes)
        if has_committed_exit and first_committed_exit_round is None:
            first_committed_exit_round = round_idx
            right_evidence_before_first_committed_exit = len(stores[RIGHT_STAGE].graph.nodes)

        if candidate is not None:
            if first_solution_round is None:
                first_solution_round = round_idx
                plane_evidence_at_first_solution = len(stores[PLANE_STAGE].graph.nodes)
                right_evidence_at_first_solution = len(stores[RIGHT_STAGE].graph.nodes)
            if best_candidate is None or candidate.total_cost + 1e-9 < best_candidate.total_cost:
                if first_solution_round is not None and best_candidate is not None and round_idx > first_solution_round:
                    committed_route_changes_after_first_solution += 1
                best_candidate = candidate
                best_solution_round = round_idx
                route_improved_this_round = 1

        if plane_updated_this_round:
            proposal_rounds_with_plane_updates += 1
        if multi_stage_updated_this_round:
            proposal_rounds_with_multi_stage_updates += 1

        best_cost_history.append(best_candidate.total_cost if best_candidate is not None else 1e12)
        for stage in STAGES:
            stage_node_gains[stage].append(int(round_node_gain[stage]))
            stage_transition_gains[stage].append(int(round_transition_gain[stage]))
            stage_route_gains[stage].append(int(route_improved_this_round))

    success = best_candidate is not None
    committed_nodes = {stage: set() for stage in STAGES}
    raw_path = np.asarray([start_q], dtype=float)
    display_path = np.asarray([start_q], dtype=float)
    route_cost_raw = 0.0
    route_cost_display = 0.0
    graph_route_edges = 0
    if best_candidate is not None:
        committed_nodes = best_candidate.committed_nodes
        raw_path = np.asarray(best_candidate.raw_path, dtype=float)
        display_path = np.asarray(best_candidate.display_path, dtype=float)
        route_cost_raw = path_cost(raw_path)
        route_cost_display = path_cost(display_path)
        graph_route_edges = (
            len(best_candidate.left_edge_path)
            + len(best_candidate.plane_edge_path)
            + len(best_candidate.right_edge_path)
        )

    total_evidence_nodes = sum(len(stores[stage].graph.nodes) for stage in STAGES)
    committed_node_count = sum(len(committed_nodes[stage]) for stage in STAGES)
    evidence_only_nodes = max(0, total_evidence_nodes - committed_node_count)
    continued_after_first_solution = bool(first_solution_round is not None and total_rounds > first_solution_round)
    plane_evidence_growth_after_first_solution = (
        max(0, len(stores[PLANE_STAGE].graph.nodes) - plane_evidence_at_first_solution)
        if first_solution_round is not None
        else 0
    )
    right_evidence_growth_after_first_solution = (
        max(0, len(stores[RIGHT_STAGE].graph.nodes) - right_evidence_at_first_solution)
        if first_solution_round is not None
        else 0
    )
    multi_stage_updates_per_round = multi_stage_update_total / max(total_rounds, 1)
    average_useful_stages_per_proposal = useful_stage_total / max(shared_proposals_processed, 1)
    stage_stagnation_flags = {
        stage: stage_stagnating(stage_node_gains[stage], stage_transition_gains[stage], stage_route_gains[stage])
        for stage in STAGES
    }
    saturated_before_solution = bool(not success and total_rounds > 0)
    stagnation_stage = None
    if saturated_before_solution:
        stagnant = [stage for stage in STAGES if stage_stagnation_flags.get(stage, False)]
        if len(stagnant) == 1:
            stagnation_stage = stagnant[0]
        elif len(stagnant) == len(STAGES):
            stagnation_stage = "all"
        elif len(stagnant) > 1:
            stagnation_stage = ",".join(stagnant)
        else:
            stagnation_stage = greedy_stage_for_serial_round(stores) if serial_mode else None

    return FixedPlaneRoute(
        success=success,
        message=(
            (
                "Serial stage evidence accumulated enough certified sequential structure to extract a fixed left-plane-right route."
                if serial_mode
                else "Parallel stage evidence accumulated enough certified sequential structure to extract a fixed left-plane-right route."
            )
            if success
            else (
                "Serial exploration accumulated evidence on the fixed left / plane / right manifolds, but no certified sequential route was extracted before exploration saturated."
                if serial_mode
                else "Evidence accumulated on the fixed left / plane / right manifolds, but no certified sequential route was extracted before exploration saturated."
            )
        ),
        total_rounds=total_rounds,
        candidate_evaluations=candidate_evaluations,
        left_evidence_nodes=len(stores[LEFT_STAGE].graph.nodes),
        plane_evidence_nodes=len(stores[PLANE_STAGE].graph.nodes),
        right_evidence_nodes=len(stores[RIGHT_STAGE].graph.nodes),
        committed_nodes=committed_node_count,
        evidence_only_nodes=evidence_only_nodes,
        shared_proposals_processed=shared_proposals_processed,
        proposals_used_by_multiple_stages=proposals_used_by_multiple_stages,
        plane_evidence_before_first_committed_entry=plane_evidence_before_first_committed_entry,
        right_evidence_before_first_committed_exit=right_evidence_before_first_committed_exit,
        transition_hypotheses_left_plane=len(left_plane_hypotheses),
        transition_hypotheses_plane_right=len(plane_right_hypotheses),
        first_solution_round=first_solution_round,
        best_solution_round=best_solution_round,
        continued_after_first_solution=continued_after_first_solution,
        path=np.asarray(display_path, dtype=float),
        raw_path=np.asarray(raw_path, dtype=float),
        certified_path_points=int(len(raw_path)),
        display_path_points=int(len(display_path)),
        route_cost_raw=route_cost_raw,
        route_cost_display=route_cost_display,
        graph_route_edges=graph_route_edges,
        stage_evidence_points={stage: stage_evidence_points(stores[stage]) for stage in STAGES},
        stage_evidence_edges={stage: stores[stage].explored_edges for stage in STAGES},
        stage_frontier_points={stage: stage_frontier_points(stores[stage]) for stage in STAGES},
        stage_chart_centers={stage: np.asarray(stores[stage].chart_centers, dtype=float) for stage in STAGES},
        stage_frontier_counts={stage: len(stores[stage].frontier_ids) for stage in STAGES},
        stage_stagnation_flags=stage_stagnation_flags,
        recent_graph_node_gain=sum(stage_recent_sum(stage_node_gains[stage]) for stage in STAGES),
        recent_transition_gain=sum(stage_recent_sum(stage_transition_gains[stage]) for stage in STAGES),
        recent_route_improvement_gain=recent_route_improvement(best_cost_history),
        plane_evidence_growth_after_first_solution=plane_evidence_growth_after_first_solution,
        right_evidence_growth_after_first_solution=right_evidence_growth_after_first_solution,
        multi_stage_updates_per_round=float(multi_stage_updates_per_round),
        average_useful_stages_per_proposal=float(average_useful_stages_per_proposal),
        proposal_rounds_with_plane_updates=proposal_rounds_with_plane_updates,
        proposal_rounds_with_multi_stage_updates=proposal_rounds_with_multi_stage_updates,
        committed_route_changes_after_first_solution=committed_route_changes_after_first_solution,
        alternative_hypothesis_pairs_evaluated=alternative_hypothesis_pairs_evaluated,
        left_plane_hypothesis_points=deduplicate_points([hyp.q for hyp in left_plane_hypotheses], tol=TRANSITION_DEDUP_TOL),
        plane_right_hypothesis_points=deduplicate_points([hyp.q for hyp in plane_right_hypotheses], tol=TRANSITION_DEDUP_TOL),
        committed_stage_nodes={
            LEFT_STAGE: np.asarray([stores[LEFT_STAGE].graph.nodes[node_id].q for node_id in sorted(committed_nodes[LEFT_STAGE])], dtype=float)
            if len(committed_nodes[LEFT_STAGE]) > 0
            else np.zeros((0, 3), dtype=float),
            PLANE_STAGE: np.asarray([stores[PLANE_STAGE].graph.nodes[node_id].q for node_id in sorted(committed_nodes[PLANE_STAGE])], dtype=float)
            if len(committed_nodes[PLANE_STAGE]) > 0
            else np.zeros((0, 3), dtype=float),
            RIGHT_STAGE: np.asarray([stores[RIGHT_STAGE].graph.nodes[node_id].q for node_id in sorted(committed_nodes[RIGHT_STAGE])], dtype=float)
            if len(committed_nodes[RIGHT_STAGE]) > 0
            else np.zeros((0, 3), dtype=float),
        },
        mode_counts=mode_counts,
        serial_mode=serial_mode,
        saturated_before_solution=saturated_before_solution,
        stagnation_stage=stagnation_stage,
        obstacles=[] if obstacles is None else list(obstacles),
    )
