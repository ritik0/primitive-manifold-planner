"""Graph containers and result dataclasses for the continuous-transfer planner."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ContinuousRouteAlternative:
    total_cost: float
    leaf_sequence: list[str]
    transition_count: int
    graph_node_path: list[int]
    graph_edge_path: list[int]
    raw_path: np.ndarray
    display_path: np.ndarray
    strict_valid: bool = True


@dataclass
class ContinuousTransferRoute:
    success: bool
    message: str
    selected_lambda: float | None
    graph_node_path: list[int]
    graph_edge_path: list[int]
    certified_path: np.ndarray
    display_path: np.ndarray
    path: np.ndarray
    raw_path: np.ndarray
    scene_profile: str = "none"
    family_obstacle_count: int = 0
    family_obstacle_summary: list[str] = field(default_factory=list)
    entry_switch: np.ndarray | None = None
    exit_switch: np.ndarray | None = None
    candidate_entries: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    candidate_lambdas: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    exit_candidates: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    explored_edges: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    explored_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    round_sources: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    round_targets: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    chart_centers: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    chart_count: int = 0
    evaluation_count: int = 0
    round_count: int = 0
    left_result: object | None = None
    transfer_result: object | None = None
    right_result: object | None = None
    explored_edges_by_mode: dict[str, list[tuple[np.ndarray, np.ndarray]]] = field(default_factory=dict)
    family_transverse_edges: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    entry_transition_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    exit_transition_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    explored_lambda_values: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    family_leaf_motion_edge_count: int = 0
    family_transverse_edge_count: int = 0
    rejected_transverse_count: int = 0
    entry_transition_count: int = 0
    exit_transition_count: int = 0
    entry_seed_count: int = 0
    exit_seed_count: int = 0
    family_frontier_count: int = 0
    family_expansion_rounds: int = 0
    family_exit_discovery_round: int | None = None
    family_cluster_count: int = 0
    active_family_cluster_count: int = 0
    stalled_family_cluster_count: int = 0
    cluster_switch_count: int = 0
    alternate_entry_seed_usage_count: int = 0
    alternate_exit_seed_usage_count: int = 0
    family_exit_candidates_kept: int = 0
    family_regions_with_no_exit: int = 0
    family_regions_with_exit: int = 0
    best_right_residual_seen: float = float("inf")
    lambda_coverage_span: float = 0.0
    lambda_coverage_gaps: list[str] = field(default_factory=list)
    family_cluster_summaries: list[str] = field(default_factory=list)
    family_cluster_centers: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    explored_lambda_region_count: int = 0
    explored_lambda_regions: list[str] = field(default_factory=list)
    first_solution_round: int | None = None
    best_solution_round: int | None = None
    continued_after_first_solution: bool = False
    lambda_hypothesis_count: int = 0
    family_candidates_before_solution: int = 0
    exploration_mode_usage: dict[str, int] = field(default_factory=dict)
    graph_node_count: int = 0
    graph_edge_count: int = 0
    right_closure_rounds: int = 0
    right_frontier_count: int = 0
    right_closure_attempt_count: int = 0
    successful_right_closure_seed_count: int = 0
    failed_right_closure_seed_count: int = 0
    best_goal_residual_seen_on_right: float = float("inf")
    right_goal_connection_attempts: int = 0
    right_goal_connection_successes: int = 0
    right_goal_connection_failures: int = 0
    best_goal_distance_seen_on_right: float = float("inf")
    final_graph_route_found_before_validation: bool = False
    primary_entry_lambda: float | None = None
    primary_entry_seed_id: int | None = None
    same_leaf_attempt_count: int = 0
    same_leaf_progress_count: int = 0
    same_leaf_failure_count: int = 0
    same_leaf_successful_exit_found: bool = False
    same_leaf_stagnation_triggered: bool = False
    transverse_switch_count_after_entry: int = 0
    first_transverse_round_after_entry: int | None = None
    first_transverse_switch_reason: str | None = None
    transverse_switch_reason_counts: dict[str, int] = field(default_factory=dict)
    final_route_same_leaf_first: bool = False
    final_route_same_leaf_only: bool = False
    shared_proposals_processed: int = 0
    proposals_used_by_multiple_family_regions: int = 0
    family_evidence_region_count: int = 0
    family_evidence_only_region_count: int = 0
    committed_family_region_count: int = 0
    family_lambda_coverage_before_first_committed_route: list[str] = field(default_factory=list)
    family_lambda_coverage_after_first_committed_route: list[str] = field(default_factory=list)
    transition_hypotheses_left_family: int = 0
    transition_hypotheses_family_right: int = 0
    family_region_updates_per_round: float = 0.0
    committed_route_changes_after_first_solution: int = 0
    average_useful_family_regions_per_proposal: float = 0.0
    augmented_family_state_count: int = 0
    augmented_family_edge_count: int = 0
    augmented_family_changing_lambda_edge_count: int = 0
    augmented_family_constant_lambda_edge_count: int = 0
    family_space_sampler_usage: dict[str, int] = field(default_factory=dict)
    route_lambda_span: float = 0.0
    route_lambda_variation_total: float = 0.0
    route_constant_lambda_edge_count: int = 0
    route_lambda_varying_edge_count: int = 0
    fixed_lambda_route_found: bool = False
    rejected_lambda_changing_route_count: int = 0
    strict_validation_message: str = ""
    strict_validation_failures: list[str] = field(default_factory=list)
    strict_invalid_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    top_k_routes: list[ContinuousRouteAlternative] = field(default_factory=list)
    leaf_store_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class StageSeed:
    family_node_id: int
    side_node_id: int
    q: np.ndarray
    lambda_value: float
    discovered_round: int
    cluster_id: int | None = None


@dataclass
class ExitSeed:
    family_node_id: int
    right_node_id: int
    q: np.ndarray
    lambda_value: float
    discovered_round: int
    cluster_id: int | None = None


@dataclass
class FamilyStageResult:
    entry_seeds: list[StageSeed] = field(default_factory=list)
    exit_seeds: list[ExitSeed] = field(default_factory=list)
    expansion_rounds: int = 0
    exit_discovery_round: int | None = None
    rejected_transverse_count: int = 0
    mode_usage: dict[str, int] = field(default_factory=dict)
    cluster_count: int = 0
    active_cluster_count: int = 0
    stalled_cluster_count: int = 0
    cluster_switch_count: int = 0
    family_regions_with_exit: int = 0
    family_regions_with_no_exit: int = 0
    best_right_residual_seen: float = float("inf")
    alternate_entry_seed_usage_count: int = 0
    family_cluster_summaries: list[str] = field(default_factory=list)
    family_cluster_centers: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    primary_entry_lambda: float | None = None
    primary_entry_seed_id: int | None = None
    same_leaf_attempt_count: int = 0
    same_leaf_progress_count: int = 0
    same_leaf_failure_count: int = 0
    same_leaf_successful_exit_found: bool = False
    same_leaf_stagnation_triggered: bool = False
    transverse_switch_count_after_entry: int = 0
    first_transverse_round_after_entry: int | None = None
    first_transverse_switch_reason: str | None = None
    transverse_switch_reason_counts: dict[str, int] = field(default_factory=dict)
    shared_proposals_processed: int = 0
    proposals_used_by_multiple_family_regions: int = 0
    family_evidence_region_count: int = 0
    family_evidence_only_region_count: int = 0
    committed_family_region_count: int = 0
    family_lambda_coverage_before_first_committed_route: list[str] = field(default_factory=list)
    family_lambda_coverage_after_first_committed_route: list[str] = field(default_factory=list)
    transition_hypotheses_left_family: int = 0
    transition_hypotheses_family_right: int = 0
    family_region_updates_per_round: float = 0.0
    committed_route_changes_after_first_solution: int = 0
    average_useful_family_regions_per_proposal: float = 0.0
    augmented_family_state_count: int = 0
    augmented_family_edge_count: int = 0
    augmented_family_changing_lambda_edge_count: int = 0
    augmented_family_constant_lambda_edge_count: int = 0
    family_space_sampler_usage: dict[str, int] = field(default_factory=dict)


@dataclass
class RightStageClosureResult:
    rounds: int = 0
    closure_attempt_count: int = 0
    successful_seed_count: int = 0
    failed_seed_count: int = 0
    goal_connection_attempts: int = 0
    goal_connection_successes: int = 0
    goal_connection_failures: int = 0
    best_goal_residual_seen: float = float("inf")
    best_goal_distance_seen: float = float("inf")
    right_frontier_count: int = 0
    route_found: bool = False
    route_discovery_round: int | None = None


@dataclass
class FamilyCluster:
    cluster_id: int
    node_ids: list[int]
    frontier_ids: list[int]
    entry_seed_ids: list[int]
    exit_seed_ids: list[int]
    lambda_min: float
    lambda_max: float
    lambda_span: float
    centroid: np.ndarray
    best_right_residual: float
    active: bool
    stagnating: bool = False


@dataclass
class FamilyEvidenceRegion:
    region_id: int
    node_ids: list[int]
    frontier_ids: list[int]
    entry_seed_ids: list[int]
    exit_seed_ids: list[int]
    lambda_min: float
    lambda_max: float
    lambda_center: float
    lambda_span: float
    centroid: np.ndarray
    explored_points: np.ndarray
    update_count: int
    entry_support_count: int
    exit_support_count: int
    best_right_residual: float
    active: bool
    stagnating: bool = False
    committed: bool = False


@dataclass
class FamilyClusterProgress:
    selection_count: int = 0
    no_gain_rounds: int = 0
    local_failures: int = 0
    transverse_failures: int = 0
    exit_failures: int = 0
    best_right_residual_seen: float = float("inf")
    last_gain_round: int = 0


@dataclass(frozen=True)
class FamilyState:
    """A locally usable family-stage state represented in the foliation by (q, lambda)."""

    node_id: int
    q: np.ndarray
    lambda_value: float
    discovered_round: int
    kind: str
    origin_sample_id: int | None = None
    expansion_count: int = 0
    cluster_id: int | None = None


@dataclass(frozen=True)
class StageState:
    """A reusable graph-backed state for left/right staged motion on a single support manifold."""

    node_id: int
    mode: str
    q: np.ndarray
    discovered_round: int
    kind: str
    origin_sample_id: int | None = None
    expansion_count: int = 0


@dataclass
class StrictValidationFailure:
    edge_kind: str
    message: str
    point_index: int
    q: np.ndarray
    residual: float | None = None
    lambda_value: float | None = None


@dataclass
class FamilyGraphNode:
    node_id: int
    mode: str
    q: np.ndarray
    discovered_round: int
    kind: str
    lambda_value: float | None = None
    origin_sample_id: int | None = None
    expansion_count: int = 0

    @property
    def ambient_point(self) -> np.ndarray:
        """Compatibility alias for future geometry-agnostic code."""

        return np.asarray(self.q, dtype=float)

    @property
    def stage_mode(self) -> str:
        """Compatibility alias for future planner code that talks in stage terms."""

        return str(self.mode)


@dataclass
class FamilyGraphEdge:
    edge_id: int
    node_u: int
    node_v: int
    kind: str
    cost: float
    path: np.ndarray
    path_lambdas: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    label: str = ""
    lambda_value: float | None = None
    origin_sample_id: int | None = None

    @property
    def path_samples(self) -> np.ndarray:
        """Compatibility alias for future geometry-agnostic path handling."""

        return np.asarray(self.path, dtype=float)

    @property
    def edge_mode(self) -> str:
        """Compatibility alias for future stage/transition terminology."""

        return str(self.kind)


@dataclass
class FamilyConnectivityGraph:
    nodes: list[FamilyGraphNode] = field(default_factory=list)
    edges: list[FamilyGraphEdge] = field(default_factory=list)
    adjacency: dict[int, list[int]] = field(default_factory=dict)
    nodes_by_mode: dict[str, list[int]] = field(default_factory=dict)
    edge_lookup: dict[tuple[int, int, str], int] = field(default_factory=dict)

    def register_node(
        self,
        mode: str,
        q: np.ndarray,
        discovered_round: int,
        kind: str,
        lambda_value: float | None = None,
        origin_sample_id: int | None = None,
        tol_q: float = 1e-4,
        tol_lambda: float = 3e-2,
    ) -> int:
        qq = np.asarray(q, dtype=float).reshape(-1)
        existing_ids = self.nodes_by_mode.setdefault(str(mode), [])
        for node_id in existing_ids:
            existing = self.nodes[node_id]
            if np.linalg.norm(existing.q - qq) > tol_q:
                continue
            if mode == "family":
                if existing.lambda_value is None or lambda_value is None:
                    continue
                if abs(float(existing.lambda_value) - float(lambda_value)) > tol_lambda:
                    continue
            if existing.kind not in {"start", "goal"} and kind in {"transition", "start", "goal"}:
                existing.kind = kind
            if existing.origin_sample_id is None and origin_sample_id is not None:
                existing.origin_sample_id = int(origin_sample_id)
            return int(node_id)

        node_id = len(self.nodes)
        self.nodes.append(
            FamilyGraphNode(
                node_id=node_id,
                mode=str(mode),
                q=qq.copy(),
                discovered_round=int(discovered_round),
                kind=str(kind),
                lambda_value=None if lambda_value is None else float(lambda_value),
                origin_sample_id=None if origin_sample_id is None else int(origin_sample_id),
            )
        )
        existing_ids.append(node_id)
        self.adjacency.setdefault(node_id, [])
        return node_id

    def add_edge(
        self,
        node_u: int,
        node_v: int,
        kind: str,
        cost: float,
        path: np.ndarray,
        path_lambdas: np.ndarray | None = None,
        label: str = "",
        lambda_value: float | None = None,
        origin_sample_id: int | None = None,
    ) -> tuple[int, bool]:
        u = int(node_u)
        v = int(node_v)
        if u == v:
            return -1, False
        key = (min(u, v), max(u, v), str(kind))
        candidate_path = np.asarray(path, dtype=float)
        candidate_lambdas = (
            np.zeros((0,), dtype=float)
            if path_lambdas is None
            else np.asarray(path_lambdas, dtype=float).reshape(-1)
        )
        if key in self.edge_lookup:
            edge_id = self.edge_lookup[key]
            existing = self.edges[edge_id]
            if float(cost) + 1e-12 < existing.cost:
                existing.cost = float(cost)
                existing.path = candidate_path
                existing.path_lambdas = candidate_lambdas
                existing.label = str(label)
                existing.lambda_value = None if lambda_value is None else float(lambda_value)
            return edge_id, False

        edge_id = len(self.edges)
        self.edges.append(
            FamilyGraphEdge(
                edge_id=edge_id,
                node_u=u,
                node_v=v,
                kind=str(kind),
                cost=float(cost),
                path=candidate_path,
                path_lambdas=candidate_lambdas,
                label=str(label),
                lambda_value=None if lambda_value is None else float(lambda_value),
                origin_sample_id=None if origin_sample_id is None else int(origin_sample_id),
            )
        )
        self.edge_lookup[key] = edge_id
        self.adjacency.setdefault(u, []).append(edge_id)
        self.adjacency.setdefault(v, []).append(edge_id)
        return edge_id, True


@dataclass
class FamilyCandidate:
    sample_id: int
    mode: str
    ambient_q: np.ndarray
    projected_q: np.ndarray
    score: float
    projection_distance: float
    novelty: float
    reachability: float
    transition_potential: float
    source_node_id: int | None
    lambda_value: float | None = None
    source_kind: str | None = None
