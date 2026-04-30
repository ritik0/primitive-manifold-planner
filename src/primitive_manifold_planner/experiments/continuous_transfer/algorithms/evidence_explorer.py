"""Evidence-explorer wrapper around the existing Example 65 family-stage behavior."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..augmented_family_space import AugmentedFamilyConstrainedSpace, FamilyAugmentedState, family_edge_lambda_mode
from ..config import FAMILY_EVIDENCE_MIN_ROUNDS_BEFORE_SATURATION, LAMBDA_BIN_WIDTH, LAMBDA_SOURCE_TOL, SAME_LEAF_STAGNATION_LIMIT
from ..graph_insertions import register_frontier_node
from ..graph_types import ExitSeed, FamilyClusterProgress, FamilyConnectivityGraph, FamilyEvidenceRegion, FamilyStageResult, StageSeed
from ..lambda_utils import choose_underexplored_lambda_region, family_nodes, quantize_lambda, summarize_explored_lambda_regions
from ..route_semantics import choose_primary_entry_seed
from ..family_stage import (
    FamilyStageContext,
    attempt_family_transverse_step,
    build_family_evidence_regions,
    choose_exit_seed_set,
    choose_family_frontier_state,
    choose_family_stage_mode,
    choose_region_source_state,
    compute_family_shared_proposal_count,
    discover_exit_seeds_to_right,
    expand_family_stage_locally,
    expand_on_family_leaf,
    family_region_summary,
    family_state_from_node,
    family_states_from_ids,
    project_proposal_to_family_regions,
    recover_primary_leaf_exit_seed,
    start_reachable_family_node_ids,
    should_stop_family_evidence_growth,
    update_family_region_progress,
)

EvidenceRegion = FamilyEvidenceRegion


@dataclass
class FamilyEvidenceExplorerStats:
    """Mutable bookkeeping for the preserved Example 65 family evidence explorer."""

    mode_usage: dict[str, int] = field(
        default_factory=lambda: {
            "family_local_novelty": 0,
            "family_transverse_step": 0,
            "family_exit_probe": 0,
            "family_underexplored_region": 0,
            "family_shared_proposal": 0,
        }
    )
    cluster_progress: dict[int, FamilyClusterProgress] = field(default_factory=dict)
    exit_seed_by_right_id: dict[int, ExitSeed] = field(default_factory=dict)
    exit_discovery_round: int | None = None
    rejected_transverse_count: int = 0
    cluster_switch_count: int = 0
    best_right_residual_seen: float = float("inf")
    last_cluster_id: int | None = None
    used_entry_cluster_ids: set[int] = field(default_factory=set)
    local_round: int = 1
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
    useful_family_regions_total: int = 0
    family_region_update_total: int = 0
    committed_route_changes_after_first_solution: int = 0
    family_lambda_coverage_before_first_committed_route: list[str] = field(default_factory=list)
    node_gain_history: list[int] = field(default_factory=list)
    exit_gain_history: list[int] = field(default_factory=list)
    lambda_region_history: list[int] = field(default_factory=list)
    region_update_history: list[int] = field(default_factory=list)
    useful_region_history: list[int] = field(default_factory=list)
    best_exit_support_history: list[float] = field(default_factory=list)
    best_exit_support_score: float = float("inf")
    family_space_sampler_usage: dict[str, int] = field(
        default_factory=lambda: {
            "proposal_augmented": 0,
            "local_augmented": 0,
            "changing_lambda_motion": 0,
            "constant_lambda_motion": 0,
        }
    )
    current_regions: list[EvidenceRegion] = field(default_factory=list)
    current_exit_seeds: list[ExitSeed] = field(default_factory=list)
    node_to_region: dict[int, int] = field(default_factory=dict)
    node_cluster_labels: dict[int, int] = field(default_factory=dict)


@dataclass
class FamilyEvidenceExplorer:
    """Encapsulates family-stage evidence growth for Example 65.

    Lambda is a transfer-mode parameter. Multiple lambdas may be explored as
    alternative transfer hypotheses, but family-family motion stays on one leaf
    at a time. Cross-lambda family motion is not used during active transfer
    exploration.
    """

    graph: FamilyConnectivityGraph
    start_node_id: int
    entry_seeds: list[StageSeed]
    transfer_family: object
    right_manifold: object
    right_center: np.ndarray
    right_radius: float
    frontier_ids: dict[str, list[int]]
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    adaptive_lambda_values: set[float]
    explored_edges_by_mode: dict[str, list[tuple[np.ndarray, np.ndarray]]]
    family_transverse_edges: list[tuple[np.ndarray, np.ndarray]]
    exit_transition_points: list[np.ndarray]
    round_sources: list[np.ndarray]
    round_targets: list[np.ndarray]
    rng: np.random.Generator
    round_offset: int
    family_round_budget: int
    continue_after_first_solution: bool = True
    max_extra_rounds_after_first_solution: int | None = None
    initial_exit_seeds: list[ExitSeed] = field(default_factory=list)
    stats: FamilyEvidenceExplorerStats = field(default_factory=FamilyEvidenceExplorerStats)

    def __post_init__(self) -> None:
        self.bounds_min = np.asarray(self.bounds_min, dtype=float)
        self.bounds_max = np.asarray(self.bounds_max, dtype=float)
        self.right_center = np.asarray(self.right_center, dtype=float)
        self.stats.node_cluster_labels = {
            int(seed.family_node_id): int(seed.family_node_id if seed.cluster_id is None else seed.cluster_id)
            for seed in self.entry_seeds
        }
        self.primary_entry_seed = choose_primary_entry_seed(self.entry_seeds, right_manifold=self.right_manifold)
        self.primary_entry_lambda = None if self.primary_entry_seed is None else float(self.primary_entry_seed.lambda_value)
        self.primary_entry_seed_id = None if self.primary_entry_seed is None else int(self.primary_entry_seed.family_node_id)
        self.effective_budget = max(int(self.family_round_budget), FAMILY_EVIDENCE_MIN_ROUNDS_BEFORE_SATURATION)
        self.stats.exit_seed_by_right_id = {
            int(seed.right_node_id): seed for seed in self.initial_exit_seeds
        }
        self.family_space = AugmentedFamilyConstrainedSpace(self.transfer_family)
        self.context = FamilyStageContext(
            graph=self.graph,
            transfer_family=self.transfer_family,
            family_space=self.family_space,
            right_manifold=self.right_manifold,
            right_center=self.right_center,
            right_radius=float(self.right_radius),
            frontier_ids=self.frontier_ids,
            bounds_min=self.bounds_min,
            bounds_max=self.bounds_max,
            adaptive_lambda_values=self.adaptive_lambda_values,
            explored_edges_by_mode=self.explored_edges_by_mode,
            family_transverse_edges=self.family_transverse_edges,
            exit_transition_points=self.exit_transition_points,
            round_sources=self.round_sources,
            round_targets=self.round_targets,
        )

    def run(self) -> FamilyStageResult:
        while self.stats.local_round <= self.effective_budget:
            if not self._prepare_round():
                break
            round_idx = int(self.round_offset + self.stats.local_round)
            target_lambda = self._choose_target_lambda()
            updated_region_ids: set[int] = set()
            round_family_node_gain = 0
            round_exit_gain = 0
            round_useful_region_count = 0

            proposal_node_gain, proposal_exit_gain, proposal_useful_region_count, proposal_updated_region_ids = (
                self._process_shared_proposals(round_idx=round_idx, target_lambda=target_lambda)
            )
            round_family_node_gain += proposal_node_gain
            round_exit_gain += proposal_exit_gain
            round_useful_region_count += proposal_useful_region_count
            updated_region_ids.update(proposal_updated_region_ids)

            mode_node_gain, mode_exit_gain, mode_updated_region_ids = self._process_mode_step(
                round_idx=round_idx,
                target_lambda=target_lambda,
            )
            round_family_node_gain += mode_node_gain
            round_exit_gain += mode_exit_gain
            updated_region_ids.update(mode_updated_region_ids)

            self._finalize_round(
                updated_region_ids=updated_region_ids,
                round_family_node_gain=round_family_node_gain,
                round_exit_gain=round_exit_gain,
                round_useful_region_count=round_useful_region_count,
            )
            self.stats.local_round += 1

        self._maybe_recover_primary_leaf_exit()
        return self._build_result()

    def _prepare_round(self) -> bool:
        self.stats.current_exit_seeds = choose_exit_seed_set(self.stats.exit_seed_by_right_id)
        self.stats.exit_seed_by_right_id = {
            int(seed.right_node_id): seed for seed in self.stats.current_exit_seeds
        }
        self.stats.current_regions, self.stats.node_to_region = build_family_evidence_regions(
            graph=self.graph,
            frontier_ids=self.frontier_ids,
            entry_seeds=self.entry_seeds,
            exit_seeds=self.stats.current_exit_seeds,
            cluster_progress=self.stats.cluster_progress,
            node_cluster_labels=self.stats.node_cluster_labels,
            right_manifold=self.right_manifold,
        )
        if len(self.stats.current_regions) == 0:
            return False
        if should_stop_family_evidence_growth(
            local_round=self.stats.local_round - 1,
            exit_discovery_round=self.stats.exit_discovery_round,
            node_gain_history=self.stats.node_gain_history,
            exit_gain_history=self.stats.exit_gain_history,
            lambda_region_history=self.stats.lambda_region_history,
            region_update_history=self.stats.region_update_history,
            useful_region_history=self.stats.useful_region_history,
            best_exit_support_history=self.stats.best_exit_support_history,
            continue_after_first_solution=self.continue_after_first_solution,
            max_extra_rounds_after_first_solution=self.max_extra_rounds_after_first_solution,
        ):
            return False
        return True

    def _choose_target_lambda(self) -> float:
        lambda_region_load = {
            quantize_lambda(float(region.lambda_center)): max(1, len(region.node_ids))
            for region in self.stats.current_regions
        }
        return float(
            choose_underexplored_lambda_region(
                lambda_probe_assignments=lambda_region_load,
                adaptive_lambda_values=self.adaptive_lambda_values,
                transfer_family=self.transfer_family,
            )
        )

    def _process_shared_proposals(
        self,
        round_idx: int,
        target_lambda: float,
    ) -> tuple[int, int, int, set[int]]:
        round_family_node_gain = 0
        round_exit_gain = 0
        round_useful_region_count = 0
        updated_region_ids: set[int] = set()

        proposal_count = compute_family_shared_proposal_count(
            transfer_family=self.transfer_family,
            regions=self.stats.current_regions,
            current_exit_seeds=self.stats.current_exit_seeds,
            node_gain_history=self.stats.node_gain_history,
            lambda_region_history=self.stats.lambda_region_history,
            region_update_history=self.stats.region_update_history,
            useful_region_history=self.stats.useful_region_history,
        )
        from ..family_stage import generate_family_ambient_proposals

        proposals = generate_family_ambient_proposals(
            transfer_family=self.transfer_family,
            entry_seeds=self.entry_seeds,
            right_center=self.right_center,
            adaptive_lambda_values=self.adaptive_lambda_values,
            target_lambda=float(target_lambda),
            round_idx=round_idx,
            rng=self.rng,
            proposal_count=proposal_count,
        )
        for proposal_q in proposals:
            self.stats.mode_usage["family_shared_proposal"] += 1
            self.stats.shared_proposals_processed += 1
            self.stats.family_space_sampler_usage["proposal_augmented"] += 1
            candidates = project_proposal_to_family_regions(
                proposal_q=np.asarray(proposal_q, dtype=float),
                transfer_family=self.transfer_family,
                family_space=self.family_space,
                regions=self.stats.current_regions,
                adaptive_lambda_values=self.adaptive_lambda_values,
                target_lambda=float(target_lambda),
                rng=self.rng,
                sample_id=int(round_idx),
                discovered_round=int(round_idx),
            )
            self.stats.useful_family_regions_total += len(candidates)
            round_useful_region_count += len(candidates)
            if len(candidates) >= 2:
                self.stats.proposals_used_by_multiple_family_regions += 1
            for _score, lam, projected_q, matched_region in candidates:
                region_id, touched_states, node_gain = self._apply_proposal_candidate(
                    round_idx=round_idx,
                    lam=float(lam),
                    projected_q=np.asarray(projected_q, dtype=float),
                    matched_region=matched_region,
                )
                round_family_node_gain += node_gain
                if region_id >= 0:
                    updated_region_ids.add(int(region_id))
                    if int(region_id) in self._entry_supported_region_ids():
                        self.stats.used_entry_cluster_ids.add(int(region_id))
                    if self._is_primary_region(region_id):
                        self.stats.same_leaf_attempt_count += 1
                if len(touched_states) > 0:
                    new_exit_seeds = discover_exit_seeds_to_right(
                        context=self.context,
                        family_states=touched_states,
                        round_idx=round_idx,
                        node_cluster_labels=self.stats.node_cluster_labels,
                        source_cluster_id=None if region_id < 0 else int(region_id),
                    )
                    round_exit_gain += len(new_exit_seeds)
                    self._record_new_exit_seeds(new_exit_seeds)
        return round_family_node_gain, round_exit_gain, round_useful_region_count, updated_region_ids

    def _apply_proposal_candidate(
        self,
        round_idx: int,
        lam: float,
        projected_q: np.ndarray,
        matched_region: EvidenceRegion | None,
    ) -> tuple[int, list[FamilyAugmentedState], int]:
        region_id = int(matched_region.region_id) if matched_region is not None else -1
        touched_states: list[FamilyAugmentedState] = []
        node_gain = 0
        if matched_region is None:
            before_nodes = len(self.graph.nodes)
            node_id = self.graph.register_node(
                mode="family",
                q=np.asarray(projected_q, dtype=float),
                discovered_round=round_idx,
                kind="evidence",
                lambda_value=float(lam),
                origin_sample_id=round_idx,
            )
            register_frontier_node(self.frontier_ids, self.graph, "family", node_id, np.asarray(projected_q, dtype=float))
            if len(self.graph.nodes) > before_nodes:
                node_gain += 1
            self.stats.node_cluster_labels[int(node_id)] = int(node_id)
            region_id = int(node_id)
            touched_state = family_state_from_node(self.graph, int(node_id), cluster_id=int(region_id))
            if touched_state is not None:
                touched_states = [touched_state]
            return region_id, touched_states, node_gain

        progress = self.stats.cluster_progress.setdefault(int(matched_region.region_id), FamilyClusterProgress())
        progress.selection_count += 1
        source_state = choose_region_source_state(
            graph=self.graph,
            region=matched_region,
            target_q=np.asarray(projected_q, dtype=float),
            family_space=self.family_space,
            target_lambda=float(lam),
        )
        if source_state is None:
            before_nodes = len(self.graph.nodes)
            node_id = self.graph.register_node(
                mode="family",
                q=np.asarray(projected_q, dtype=float),
                discovered_round=round_idx,
                kind="evidence",
                lambda_value=float(lam),
                origin_sample_id=round_idx,
            )
            register_frontier_node(self.frontier_ids, self.graph, "family", node_id, np.asarray(projected_q, dtype=float))
            if len(self.graph.nodes) > before_nodes:
                node_gain += 1
            self.stats.node_cluster_labels[int(node_id)] = int(matched_region.region_id)
            touched_state = family_state_from_node(self.graph, int(node_id), cluster_id=int(matched_region.region_id))
            if touched_state is not None:
                touched_states = [touched_state]
            return int(matched_region.region_id), touched_states, node_gain

        self.context.round_sources.append(np.asarray(source_state.q, dtype=float).copy())
        self.context.round_targets.append(np.asarray(projected_q, dtype=float).copy())
        if abs(float(source_state.lambda_value) - float(lam)) <= max(LAMBDA_SOURCE_TOL, 0.5 * LAMBDA_BIN_WIDTH):
            self.stats.family_space_sampler_usage["constant_lambda_motion"] += 1
            new_ids = expand_on_family_leaf(
                context=self.context,
                source_state=source_state,
                target_q=np.asarray(projected_q, dtype=float),
                round_idx=round_idx,
                guide_point=np.asarray(projected_q, dtype=float),
                origin_sample_id=round_idx,
            )
            for node_id in new_ids:
                self.stats.node_cluster_labels[int(node_id)] = int(matched_region.region_id)
            touched_states = family_states_from_ids(self.graph, new_ids, cluster_id=int(matched_region.region_id))
            node_gain += max(0, len(new_ids) - 1)
            return int(matched_region.region_id), touched_states, node_gain

        # Keep multi-lambda evidence alive, but do not connect leaves by changing
        # lambda during family-family motion. A proposal on another lambda seeds
        # new evidence instead of creating a transverse edge.
        before_nodes = len(self.graph.nodes)
        node_id = self.graph.register_node(
            mode="family",
            q=np.asarray(projected_q, dtype=float),
            discovered_round=round_idx,
            kind="evidence",
            lambda_value=float(lam),
            origin_sample_id=round_idx,
        )
        register_frontier_node(self.frontier_ids, self.graph, "family", node_id, np.asarray(projected_q, dtype=float))
        if len(self.graph.nodes) > before_nodes:
            node_gain += 1
        self.stats.node_cluster_labels[int(node_id)] = int(node_id)
        touched_state = family_state_from_node(self.graph, int(node_id), cluster_id=int(node_id))
        if touched_state is not None:
            touched_states = [touched_state]
        return int(node_id), touched_states, node_gain

    def _process_mode_step(
        self,
        round_idx: int,
        target_lambda: float,
    ) -> tuple[int, int, set[int]]:
        updated_region_ids: set[int] = set()
        round_family_node_gain = 0
        round_exit_gain = 0
        mode = choose_family_stage_mode(
            rng=self.rng,
            cluster_count=len(self.stats.current_regions),
            stalled_cluster_count=sum(1 for region in self.stats.current_regions if region.stagnating),
            exit_seed_count=len(self.stats.exit_seed_by_right_id),
        )
        if mode == "family_transverse_step":
            mode = "family_local_novelty"
        self.stats.mode_usage[mode] += 1
        source_state, cluster_id = choose_family_frontier_state(
            graph=self.graph,
            regions=self.stats.current_regions,
            cluster_progress=self.stats.cluster_progress,
            mode=mode,
            transfer_family=self.transfer_family,
            right_manifold=self.right_manifold,
            preferred_region_id=self._primary_region_id(),
            target_lambda=target_lambda,
        )
        if source_state is None or cluster_id is None:
            return round_family_node_gain, round_exit_gain, updated_region_ids

        if self.stats.last_cluster_id is not None and int(cluster_id) != int(self.stats.last_cluster_id):
            self.stats.cluster_switch_count += 1
        self.stats.last_cluster_id = int(cluster_id)
        before_nodes = len(family_nodes(self.graph))
        touched_states = [source_state]
        if mode == "family_local_novelty":
            self.stats.family_space_sampler_usage["local_augmented"] += 1
            new_ids = expand_family_stage_locally(
                context=self.context,
                source_state=source_state,
                round_idx=round_idx,
                rng=self.rng,
                target_lambda=float(source_state.lambda_value),
            )
            for node_id in new_ids:
                self.stats.node_cluster_labels[int(node_id)] = int(cluster_id)
            touched_states.extend(family_states_from_ids(self.graph, new_ids, cluster_id=int(cluster_id)))
        elif mode == "family_transverse_step":
            self.stats.family_space_sampler_usage["local_augmented"] += 1
            new_ids, ok = attempt_family_transverse_step(
                context=self.context,
                source_state=source_state,
                round_idx=round_idx,
                target_lambda=target_lambda,
                rng=self.rng,
            )
            self.stats.transverse_switch_reason_counts["region_recovery"] = (
                self.stats.transverse_switch_reason_counts.get("region_recovery", 0) + 1
            )
            if not ok:
                self.stats.rejected_transverse_count += 1
            else:
                self.stats.family_space_sampler_usage["changing_lambda_motion"] += 1
                self.stats.transverse_switch_count_after_entry += 1
                if self.stats.first_transverse_round_after_entry is None:
                    self.stats.first_transverse_round_after_entry = int(self.stats.local_round)
                    self.stats.first_transverse_switch_reason = "region_recovery"
                for node_id in new_ids:
                    self.stats.node_cluster_labels[int(node_id)] = int(cluster_id)
                touched_states.extend(family_states_from_ids(self.graph, new_ids, cluster_id=int(cluster_id)))
        elif mode == "family_underexplored_region":
            self.stats.family_space_sampler_usage["local_augmented"] += 1
            new_ids = expand_family_stage_locally(
                context=self.context,
                source_state=source_state,
                round_idx=round_idx,
                rng=self.rng,
                target_lambda=float(source_state.lambda_value),
            )
            for node_id in new_ids:
                self.stats.node_cluster_labels[int(node_id)] = int(cluster_id)
            touched_states.extend(family_states_from_ids(self.graph, new_ids, cluster_id=int(cluster_id)))

        updated_region_ids.add(int(cluster_id))
        if int(cluster_id) in self._entry_supported_region_ids():
            self.stats.used_entry_cluster_ids.add(int(cluster_id))
        if self._is_primary_region(cluster_id):
            self.stats.same_leaf_attempt_count += 1
        round_family_node_gain += max(0, len(family_nodes(self.graph)) - before_nodes)
        new_exit_seeds = discover_exit_seeds_to_right(
            context=self.context,
            family_states=touched_states,
            round_idx=round_idx,
            node_cluster_labels=self.stats.node_cluster_labels,
            source_cluster_id=int(cluster_id),
        )
        round_exit_gain += len(new_exit_seeds)
        self._record_new_exit_seeds(new_exit_seeds)
        return round_family_node_gain, round_exit_gain, updated_region_ids

    def _finalize_round(
        self,
        updated_region_ids: set[int],
        round_family_node_gain: int,
        round_exit_gain: int,
        round_useful_region_count: int,
    ) -> None:
        self.stats.current_exit_seeds = choose_exit_seed_set(self.stats.exit_seed_by_right_id)
        self.stats.exit_seed_by_right_id = {
            int(seed.right_node_id): seed for seed in self.stats.current_exit_seeds
        }
        region_update_count = len(updated_region_ids)
        self.stats.family_region_update_total += region_update_count
        self.stats.node_gain_history.append(int(round_family_node_gain))
        self.stats.exit_gain_history.append(int(round_exit_gain))
        self.stats.lambda_region_history.append(len(summarize_explored_lambda_regions(self.graph)))
        self.stats.region_update_history.append(int(region_update_count))
        self.stats.useful_region_history.append(int(round_useful_region_count))
        update_family_region_progress(
            regions=self.stats.current_regions,
            updated_region_ids=updated_region_ids,
            cluster_progress=self.stats.cluster_progress,
        )
        self._update_same_leaf_progress(updated_region_ids)
        self._update_exit_support_metrics()

    def _update_same_leaf_progress(self, updated_region_ids: set[int]) -> None:
        primary_region_id = self._primary_region_id()
        if primary_region_id is None:
            return
        primary_region_progress = self.stats.cluster_progress.get(int(primary_region_id), FamilyClusterProgress())
        if int(primary_region_id) in updated_region_ids:
            self.stats.same_leaf_progress_count += 1
        elif len(updated_region_ids) > 0:
            self.stats.same_leaf_failure_count += 1
        self.stats.same_leaf_stagnation_triggered = bool(
            primary_region_progress.no_gain_rounds >= SAME_LEAF_STAGNATION_LIMIT
        )

    def _update_exit_support_metrics(self) -> None:
        if len(self.stats.current_exit_seeds) > 0:
            support_score = min(
                float(np.linalg.norm(np.asarray(self.right_manifold.residual(seed.q), dtype=float)))
                for seed in self.stats.current_exit_seeds
            )
            self.stats.best_right_residual_seen = min(self.stats.best_right_residual_seen, support_score)
            if self.stats.exit_discovery_round is None:
                self.stats.exit_discovery_round = int(self.stats.local_round)
                self.stats.family_lambda_coverage_before_first_committed_route = summarize_explored_lambda_regions(self.graph)
                self.stats.best_exit_support_score = support_score
            elif support_score + 1e-9 < self.stats.best_exit_support_score:
                self.stats.best_exit_support_score = support_score
                self.stats.committed_route_changes_after_first_solution += 1
            self.stats.best_exit_support_history.append(float(self.stats.best_exit_support_score))
        else:
            self.stats.best_exit_support_history.append(float("inf"))

    def _maybe_recover_primary_leaf_exit(self) -> None:
        entry_supported_exit_exists = any(
            int(seed.family_node_id) in self.stats.node_to_region
            and int(self.stats.node_to_region[int(seed.family_node_id)]) in self._entry_supported_region_ids()
            for seed in self.stats.current_exit_seeds
        )
        unobstructed_family = len(getattr(self.context.transfer_family, "obstacles", [])) == 0
        should_try_primary_leaf_recovery = (
            self.primary_entry_seed is not None
            and (unobstructed_family or (not self.stats.same_leaf_successful_exit_found and not entry_supported_exit_exists))
        )
        if not should_try_primary_leaf_recovery:
            return
        recovered_exit_seeds = recover_primary_leaf_exit_seed(
            context=self.context,
            primary_entry_seed=self.primary_entry_seed,
            round_idx=int(self.round_offset + self.stats.local_round),
            node_cluster_labels=self.stats.node_cluster_labels,
        )
        if len(recovered_exit_seeds) == 0:
            return
        for seed in recovered_exit_seeds:
            self.stats.exit_seed_by_right_id[int(seed.right_node_id)] = seed
        self.stats.same_leaf_successful_exit_found = True
        if self.stats.exit_discovery_round is None:
            self.stats.exit_discovery_round = int(self.stats.local_round)

    def _record_new_exit_seeds(self, new_exit_seeds: list[ExitSeed]) -> None:
        for seed in new_exit_seeds:
            self.stats.exit_seed_by_right_id[int(seed.right_node_id)] = seed

    def _entry_supported_region_ids(self) -> set[int]:
        return {
            int(region.region_id)
            for region in self.stats.current_regions
            if region.entry_support_count > 0
        }

    def _primary_region_id(self) -> int | None:
        if self.primary_entry_seed_id is None or int(self.primary_entry_seed_id) not in self.stats.node_to_region:
            return None
        return int(self.stats.node_to_region[int(self.primary_entry_seed_id)])

    def _is_primary_region(self, region_id: int | None) -> bool:
        primary_region_id = self._primary_region_id()
        return primary_region_id is not None and region_id is not None and int(region_id) == int(primary_region_id)

    def _build_result(self) -> FamilyStageResult:
        final_exit_seeds = choose_exit_seed_set(self.stats.exit_seed_by_right_id)
        if self.primary_entry_lambda is not None:
            committed_reachable_family_ids = start_reachable_family_node_ids(self.graph, int(self.start_node_id))
            preferred_same_leaf_seeds = [
                seed
                for seed in self.stats.exit_seed_by_right_id.values()
                if abs(float(seed.lambda_value) - float(self.primary_entry_lambda)) <= LAMBDA_SOURCE_TOL
                and int(seed.family_node_id) in committed_reachable_family_ids
            ]
            preferred_same_leaf_seeds.sort(
                key=lambda seed: (
                    seed.discovered_round,
                    float(np.linalg.norm(np.asarray(seed.q, dtype=float) - np.asarray(self.right_center, dtype=float))),
                )
            )
            if len(preferred_same_leaf_seeds) > 0:
                preferred_seed = preferred_same_leaf_seeds[0]
                final_exit_seeds = [preferred_seed] + [
                    seed
                    for seed in final_exit_seeds
                    if int(seed.right_node_id) != int(preferred_seed.right_node_id)
                ]
        final_regions, _node_to_region = build_family_evidence_regions(
            graph=self.graph,
            frontier_ids=self.frontier_ids,
            entry_seeds=self.entry_seeds,
            exit_seeds=final_exit_seeds,
            cluster_progress=self.stats.cluster_progress,
            node_cluster_labels=self.stats.node_cluster_labels,
            right_manifold=self.right_manifold,
        )
        return FamilyStageResult(
            entry_seeds=list(self.entry_seeds),
            exit_seeds=final_exit_seeds,
            expansion_rounds=int(self.stats.local_round - 1),
            exit_discovery_round=self.stats.exit_discovery_round,
            rejected_transverse_count=self.stats.rejected_transverse_count,
            mode_usage=self.stats.mode_usage,
            cluster_count=len(final_regions),
            active_cluster_count=sum(1 for region in final_regions if region.active),
            stalled_cluster_count=sum(1 for region in final_regions if region.stagnating),
            cluster_switch_count=self.stats.cluster_switch_count,
            family_regions_with_exit=sum(1 for region in final_regions if region.exit_support_count > 0),
            family_regions_with_no_exit=sum(1 for region in final_regions if region.exit_support_count == 0),
            best_right_residual_seen=self.stats.best_right_residual_seen,
            alternate_entry_seed_usage_count=max(0, len(self.stats.used_entry_cluster_ids) - 1),
            family_cluster_summaries=[
                family_region_summary(region, self.stats.cluster_progress.get(region.region_id, FamilyClusterProgress()))
                for region in final_regions
            ],
            family_cluster_centers=np.asarray([region.centroid for region in final_regions], dtype=float)
            if len(final_regions) > 0
            else np.zeros((0, 3), dtype=float),
            primary_entry_lambda=self.primary_entry_lambda,
            primary_entry_seed_id=self.primary_entry_seed_id,
            same_leaf_attempt_count=self.stats.same_leaf_attempt_count,
            same_leaf_progress_count=self.stats.same_leaf_progress_count,
            same_leaf_failure_count=self.stats.same_leaf_failure_count,
            same_leaf_successful_exit_found=self.stats.same_leaf_successful_exit_found,
            same_leaf_stagnation_triggered=self.stats.same_leaf_stagnation_triggered,
            transverse_switch_count_after_entry=self.stats.transverse_switch_count_after_entry,
            first_transverse_round_after_entry=self.stats.first_transverse_round_after_entry,
            first_transverse_switch_reason=self.stats.first_transverse_switch_reason,
            transverse_switch_reason_counts=self.stats.transverse_switch_reason_counts,
            shared_proposals_processed=self.stats.shared_proposals_processed,
            proposals_used_by_multiple_family_regions=self.stats.proposals_used_by_multiple_family_regions,
            family_evidence_region_count=len(final_regions),
            family_evidence_only_region_count=sum(
                1 for region in final_regions if region.entry_support_count == 0 or region.exit_support_count == 0
            ),
            committed_family_region_count=sum(
                1 for region in final_regions if region.entry_support_count > 0 and region.exit_support_count > 0
            ),
            family_lambda_coverage_before_first_committed_route=self.stats.family_lambda_coverage_before_first_committed_route,
            family_lambda_coverage_after_first_committed_route=summarize_explored_lambda_regions(self.graph),
            transition_hypotheses_left_family=len(self.entry_seeds),
            transition_hypotheses_family_right=len(final_exit_seeds),
            family_region_updates_per_round=(float(self.stats.family_region_update_total) / max(1, int(self.stats.local_round - 1))),
            committed_route_changes_after_first_solution=self.stats.committed_route_changes_after_first_solution,
            average_useful_family_regions_per_proposal=(
                float(self.stats.useful_family_regions_total) / max(1, int(self.stats.shared_proposals_processed))
            ),
            augmented_family_state_count=sum(1 for node in self.graph.nodes if str(node.mode) == "family"),
            augmented_family_edge_count=sum(
                1 for edge in self.graph.edges if str(edge.kind) in {"family_leaf_motion", "family_transverse"}
            ),
            augmented_family_changing_lambda_edge_count=sum(
                1
                for edge in self.graph.edges
                if str(edge.kind) in {"family_leaf_motion", "family_transverse"}
                and family_edge_lambda_mode(np.asarray(edge.path_lambdas, dtype=float), tol=LAMBDA_SOURCE_TOL) == "changing_lambda"
            ),
            augmented_family_constant_lambda_edge_count=sum(
                1
                for edge in self.graph.edges
                if str(edge.kind) in {"family_leaf_motion", "family_transverse"}
                and family_edge_lambda_mode(np.asarray(edge.path_lambdas, dtype=float), tol=LAMBDA_SOURCE_TOL) == "constant_lambda"
            ),
            family_space_sampler_usage=dict(self.stats.family_space_sampler_usage),
        )
