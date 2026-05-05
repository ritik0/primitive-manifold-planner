"""Lambda-locked parallel evidence warm-start for Example 65.

This module introduces a small bridge toward Example 66 style shared-proposal
evidence accumulation without discarding the current staged solver. Ambient
proposals are projected onto all families in parallel, but each projection
locks onto one concrete family leaf on first contact. Local evidence growth
then remains within that fixed leaf manifold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from primitive_manifold_planner.families.standard import SphereFamily
from primitive_manifold_planner.projection import project_newton

from .config import (
    ENTRY_LAMBDA_DIVERSITY_TOL,
    ENTRY_SPATIAL_DIVERSITY_TOL,
    EXIT_LAMBDA_DIVERSITY_TOL,
    EXIT_SPATIAL_DIVERSITY_TOL,
    FAMILY_ROUTE_SELECTION_MIN_EXIT_CANDIDATES,
    LAMBDA_SOURCE_TOL,
    MAX_ENTRY_SEEDS,
)
from .evidence_managers import FamilyEvidenceManager, LeafStoreManager
from .family_definition import ContinuousMaskedPlaneFamily
from .graph_insertions import add_certified_family_edge, add_path_nodes_to_graph, register_frontier_node
from .graph_types import ExitSeed, FamilyConnectivityGraph, StageSeed
from .lambda_utils import clamp_lambda, refine_lambda_region_if_promising
from .right_stage import certify_side_connection_to_target
from .seed_utils import keep_diverse_exit_seeds, keep_diverse_stage_seeds
from .strict_validation import report_strict_validation_failures, validate_transition_edge
from .support import refine_intersection_on_both_manifolds

from primitive_manifold_planner.thesis import parallel_evidence_planner as ex66


LEFT_STAGE = "left"
FAMILY_STAGE = "family"
RIGHT_STAGE = "right"
PARALLEL_ENTRY_SEED_BUFFER = max(12, 3 * MAX_ENTRY_SEEDS)
PARALLEL_EXIT_SEED_BUFFER = max(18, 3 * FAMILY_ROUTE_SELECTION_MIN_EXIT_CANDIDATES)
PARALLEL_ALL_LEAF_SCAN_LIMIT = 4
PARALLEL_HITS_PER_SCAN = 2


@dataclass
class FamilyProjection:
    success: bool
    x_projected: np.ndarray
    lambda_value: float | None
    manifold: object | None


@dataclass
class ParallelLeafWarmstartResult:
    entry_seeds: list[StageSeed] = field(default_factory=list)
    initial_exit_seeds: list[ExitSeed] = field(default_factory=list)
    left_manager: "FamilyEvidenceManager | None" = None
    family_manager: "FamilyEvidenceManager | None" = None
    right_manager: "FamilyEvidenceManager | None" = None
    parallel_round_budget: int = 0
    leaf_store_counts: dict[str, int] = field(default_factory=dict)


def project_onto_family(
    family,
    proposal: np.ndarray,
) -> FamilyProjection:
    qq = np.asarray(proposal, dtype=float)
    if isinstance(family, ContinuousMaskedPlaneFamily):
        lam = clamp_lambda(family, family.infer_lambda(qq))
        manifold = family.manifold(float(lam))
        projection = project_newton(manifold=manifold, x0=qq, tol=1e-10, max_iters=60, damping=1.0)
        if not projection.success:
            return FamilyProjection(False, np.zeros((3,), dtype=float), None, None)
        q_proj = np.asarray(projection.x_projected, dtype=float)
        if not family.within_patch(float(lam), q_proj):
            return FamilyProjection(False, np.zeros((3,), dtype=float), None, None)
        return FamilyProjection(True, q_proj, float(lam), manifold)

    best_q: np.ndarray | None = None
    best_manifold = None
    best_lambda: float | None = None
    best_dist = float("inf")
    for lam in family.sample_lambdas():
        manifold = family.manifold(float(lam))
        projection = project_newton(manifold=manifold, x0=qq, tol=1e-10, max_iters=60, damping=1.0)
        if not projection.success:
            continue
        q_proj = np.asarray(projection.x_projected, dtype=float)
        dist = float(np.linalg.norm(q_proj - qq))
        if dist + 1e-12 < best_dist:
            best_dist = dist
            best_q = q_proj
            best_manifold = manifold
            best_lambda = float(lam)
    if best_q is None or best_manifold is None or best_lambda is None:
        return FamilyProjection(False, np.zeros((3,), dtype=float), None, None)
    return FamilyProjection(True, best_q, best_lambda, best_manifold)


def _stage_role(stage_name: str) -> str:
    if str(stage_name) == FAMILY_STAGE:
        return ex66.PLANE_STAGE
    if str(stage_name) == LEFT_STAGE:
        return ex66.LEFT_STAGE
    return ex66.RIGHT_STAGE


def _aggregate_stage_stores(
    left_manager: FamilyEvidenceManager,
    family_manager: FamilyEvidenceManager,
    right_manager: FamilyEvidenceManager,
) -> dict[str, ex66.StageEvidenceStore]:
    stores: dict[str, ex66.StageEvidenceStore] = {}
    if left_manager.store is not None:
        stores[ex66.LEFT_STAGE] = left_manager.store
    if family_manager.manager is not None:
        stores[ex66.PLANE_STAGE] = family_manager.manager.representative_store()
    if right_manager.store is not None:
        stores[ex66.RIGHT_STAGE] = right_manager.store
    return stores


def _leaf_utility(
    stage_name: str,
    store: ex66.StageEvidenceStore,
    q: np.ndarray,
    guide: np.ndarray,
    aggregate_stores: dict[str, ex66.StageEvidenceStore],
) -> float:
    known_points = ex66.stage_evidence_points(store)
    return float(
        ex66.proposal_stage_utility(
            _stage_role(stage_name),
            np.asarray(q, dtype=float),
            np.asarray(known_points, dtype=float),
            np.asarray(guide, dtype=float),
            aggregate_stores,
        )
    )


def _project_guide_to_leaf(family, lam: float, guide_point: np.ndarray) -> np.ndarray:
    manifold = family.manifold(float(lam))
    projected = ex66.proposal_projection(manifold, np.asarray(guide_point, dtype=float))
    if projected is not None:
        return np.asarray(projected, dtype=float)
    point_on_leaf = getattr(family, "point_on_leaf", None)
    if callable(point_on_leaf):
        return np.asarray(point_on_leaf(float(lam)), dtype=float)
    return np.asarray(guide_point, dtype=float)


def _trim_entry_seed_map(entry_seed_map: dict[tuple[int, int], StageSeed]) -> None:
    if len(entry_seed_map) <= PARALLEL_ENTRY_SEED_BUFFER:
        return
    kept = keep_diverse_stage_seeds(
        list(entry_seed_map.values()),
        max_keep=PARALLEL_ENTRY_SEED_BUFFER,
        lambda_tol=ENTRY_LAMBDA_DIVERSITY_TOL,
        spatial_tol=ENTRY_SPATIAL_DIVERSITY_TOL,
    )
    entry_seed_map.clear()
    for seed in kept:
        entry_seed_map[(int(seed.family_node_id), int(seed.side_node_id))] = seed


def _trim_exit_seed_map(exit_seed_map: dict[tuple[int, int], ExitSeed]) -> None:
    if len(exit_seed_map) <= PARALLEL_EXIT_SEED_BUFFER:
        return
    kept = keep_diverse_exit_seeds(
        list(exit_seed_map.values()),
        max_keep=PARALLEL_EXIT_SEED_BUFFER,
        lambda_tol=EXIT_LAMBDA_DIVERSITY_TOL,
        spatial_tol=EXIT_SPATIAL_DIVERSITY_TOL,
    )
    exit_seed_map.clear()
    for seed in kept:
        exit_seed_map[(int(seed.family_node_id), int(seed.right_node_id))] = seed


def _mirror_store_result_into_graph(
    *,
    graph: FamilyConnectivityGraph,
    store: ex66.StageEvidenceStore,
    stage_mode: str,
    lambda_value: float | None,
    result,
    seed_q: np.ndarray,
    round_idx: int,
    frontier_ids: dict[str, list[int]],
    guide_point: np.ndarray,
    side_manifold=None,
    sphere_center: np.ndarray | None = None,
    sphere_radius: float | None = None,
) -> list[int]:
    if result is None:
        node_id = graph.register_node(
            mode=str(stage_mode),
            q=np.asarray(seed_q, dtype=float),
            discovered_round=int(round_idx),
            kind="explored",
            lambda_value=lambda_value,
            origin_sample_id=int(round_idx),
        )
        register_frontier_node(frontier_ids, graph, str(stage_mode), int(node_id), np.asarray(guide_point, dtype=float))
        return [int(node_id)]

    path = np.asarray(getattr(result, "path", np.zeros((0, 3), dtype=float)), dtype=float)
    if len(path) == 0:
        return []
    if str(stage_mode) == FAMILY_STAGE:
        return add_certified_family_edge(
            graph=graph,
            transfer_family=store.manifold.base_manifold if hasattr(store.manifold, "base_manifold") else None,
            path=path,
            lambdas=np.full((len(path),), float(lambda_value), dtype=float),
            round_idx=int(round_idx),
            frontier_ids=frontier_ids,
            guide_point=np.asarray(guide_point, dtype=float),
            edge_kind="family_leaf_motion",
            edge_label="parallel_family_leaf_motion",
            origin_sample_id=int(round_idx),
        )
    return add_path_nodes_to_graph(
        graph=graph,
        mode=str(stage_mode),
        path=path,
        round_idx=int(round_idx),
        frontier_ids=frontier_ids,
        guide_point=np.asarray(guide_point, dtype=float),
        origin_sample_id=int(round_idx),
        edge_kind=f"{stage_mode}_motion",
        edge_label=f"parallel_{stage_mode}_motion",
        side_manifold=side_manifold,
        sphere_center=np.asarray(sphere_center, dtype=float) if sphere_center is not None else None,
        sphere_radius=None if sphere_radius is None else float(sphere_radius),
    )


def _mirror_family_leaf_result_into_graph(
    *,
    graph: FamilyConnectivityGraph,
    transfer_family: ContinuousMaskedPlaneFamily,
    path: np.ndarray,
    lambda_value: float,
    round_idx: int,
    frontier_ids: dict[str, list[int]],
    guide_point: np.ndarray,
) -> list[int]:
    return add_certified_family_edge(
        graph=graph,
        transfer_family=transfer_family,
        path=np.asarray(path, dtype=float),
        lambdas=np.full((len(np.asarray(path, dtype=float)),), float(lambda_value), dtype=float),
        round_idx=int(round_idx),
        frontier_ids=frontier_ids,
        guide_point=np.asarray(guide_point, dtype=float),
        edge_kind="family_leaf_motion",
        edge_label="parallel_family_leaf_motion",
        origin_sample_id=int(round_idx),
    )


def _maybe_add_entry_seed(
    *,
    graph: FamilyConnectivityGraph,
    left_manifold,
    left_center: np.ndarray,
    left_radius: float,
    transfer_family: ContinuousMaskedPlaneFamily,
    frontier_ids: dict[str, list[int]],
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    explored_edges_by_mode: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    adaptive_lambda_values: set[float],
    entry_transition_points: list[np.ndarray],
    round_idx: int,
    source_q: np.ndarray,
    plane_q: np.ndarray,
    lambda_value: float,
) -> StageSeed | None:
    lam = float(lambda_value)
    family_manifold = transfer_family.manifold(lam)
    refined, ok = refine_intersection_on_both_manifolds(left_manifold, family_manifold, 0.5 * (np.asarray(source_q, dtype=float) + np.asarray(plane_q, dtype=float)), tol=1e-8, max_iters=32)
    if not ok:
        return None
    refined_q = np.asarray(refined, dtype=float)
    if not transfer_family.within_patch(lam, refined_q):
        return None
    valid, failures = validate_transition_edge(
        path=np.asarray([refined_q], dtype=float),
        side_manifold=left_manifold,
        transfer_family=transfer_family,
        lam=lam,
        side_center=np.asarray(left_center, dtype=float),
        side_radius=float(left_radius),
        side_mode="left",
    )
    if not valid:
        report_strict_validation_failures("Rejected parallel left-to-family entry seed", failures)
        return None
    left_node_ids = [int(node_id) for node_id in graph.nodes_by_mode.get(LEFT_STAGE, [])]
    source_left_id = (
        None
        if len(left_node_ids) == 0
        else min(
            left_node_ids,
            key=lambda node_id: float(np.linalg.norm(np.asarray(graph.nodes[int(node_id)].q, dtype=float) - refined_q)),
        )
    )
    if source_left_id is None:
        return None
    left_ids = certify_side_connection_to_target(
        graph=graph,
        source_node_id=int(source_left_id),
        side_mode="left",
        side_manifold=left_manifold,
        sphere_center=np.asarray(left_center, dtype=float),
        sphere_radius=float(left_radius),
        target_q=refined_q,
        bounds_min=np.asarray(bounds_min, dtype=float),
        bounds_max=np.asarray(bounds_max, dtype=float),
        round_idx=int(round_idx),
        frontier_ids=frontier_ids,
        guide_point=np.asarray(transfer_family.base_point, dtype=float),
        explored_edges_by_mode=explored_edges_by_mode,
        origin_sample_id=int(round_idx),
    )
    if len(left_ids) == 0:
        return None
    family_node_id = graph.register_node(
        "family",
        refined_q,
        int(round_idx),
        "transition",
        lambda_value=lam,
        origin_sample_id=int(round_idx),
    )
    register_frontier_node(frontier_ids, graph, "family", family_node_id, np.asarray(transfer_family.base_point, dtype=float))
    _edge_id, is_new = graph.add_edge(
        node_u=int(left_ids[-1]),
        node_v=int(family_node_id),
        kind="entry_transition",
        cost=0.0,
        path=np.asarray([refined_q], dtype=float),
        path_lambdas=np.asarray([lam], dtype=float),
        label="parallel_left_family_entry",
        lambda_value=lam,
        origin_sample_id=int(round_idx),
    )
    adaptive_lambda_values.add(float(lam))
    refine_lambda_region_if_promising(float(lam), adaptive_lambda_values, transfer_family)
    if is_new:
        entry_transition_points.append(refined_q.copy())
    return StageSeed(
        family_node_id=int(family_node_id),
        side_node_id=int(left_ids[-1]),
        q=refined_q.copy(),
        lambda_value=lam,
        discovered_round=int(round_idx),
        cluster_id=int(family_node_id),
    )


def _maybe_add_exit_seed(
    *,
    graph: FamilyConnectivityGraph,
    right_manifold,
    right_center: np.ndarray,
    right_radius: float,
    transfer_family: ContinuousMaskedPlaneFamily,
    frontier_ids: dict[str, list[int]],
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    explored_edges_by_mode: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    exit_transition_points: list[np.ndarray],
    round_idx: int,
    plane_q: np.ndarray,
    right_q: np.ndarray,
    lambda_value: float,
) -> ExitSeed | None:
    lam = float(lambda_value)
    family_manifold = transfer_family.manifold(lam)
    refined, ok = refine_intersection_on_both_manifolds(family_manifold, right_manifold, 0.5 * (np.asarray(plane_q, dtype=float) + np.asarray(right_q, dtype=float)), tol=1e-8, max_iters=32)
    if not ok:
        return None
    refined_q = np.asarray(refined, dtype=float)
    if not transfer_family.within_patch(lam, refined_q):
        return None
    valid, failures = validate_transition_edge(
        path=np.asarray([refined_q], dtype=float),
        side_manifold=right_manifold,
        transfer_family=transfer_family,
        lam=lam,
        side_center=np.asarray(right_center, dtype=float),
        side_radius=float(right_radius),
        side_mode="right",
    )
    if not valid:
        report_strict_validation_failures("Rejected parallel family-to-right exit seed", failures)
        return None
    right_node_ids = [int(node_id) for node_id in graph.nodes_by_mode.get(RIGHT_STAGE, [])]
    source_right_id = (
        None
        if len(right_node_ids) == 0
        else min(
            right_node_ids,
            key=lambda node_id: float(np.linalg.norm(np.asarray(graph.nodes[int(node_id)].q, dtype=float) - refined_q)),
        )
    )
    if source_right_id is None:
        return None
    right_ids = certify_side_connection_to_target(
        graph=graph,
        source_node_id=int(source_right_id),
        side_mode="right",
        side_manifold=right_manifold,
        sphere_center=np.asarray(right_center, dtype=float),
        sphere_radius=float(right_radius),
        target_q=refined_q,
        bounds_min=np.asarray(bounds_min, dtype=float),
        bounds_max=np.asarray(bounds_max, dtype=float),
        round_idx=int(round_idx),
        frontier_ids=frontier_ids,
        guide_point=np.asarray(right_center, dtype=float),
        explored_edges_by_mode=explored_edges_by_mode,
        origin_sample_id=int(round_idx),
    )
    if len(right_ids) == 0:
        return None
    family_node_id = graph.register_node(
        "family",
        refined_q,
        int(round_idx),
        "transition",
        lambda_value=lam,
        origin_sample_id=int(round_idx),
    )
    register_frontier_node(frontier_ids, graph, "family", family_node_id, np.asarray(transfer_family.base_point, dtype=float))
    _edge_id, is_new = graph.add_edge(
        node_u=int(family_node_id),
        node_v=int(right_ids[-1]),
        kind="exit_transition",
        cost=0.0,
        path=np.asarray([refined_q], dtype=float),
        path_lambdas=np.asarray([lam], dtype=float),
        label="parallel_family_right_exit",
        lambda_value=lam,
        origin_sample_id=int(round_idx),
    )
    if is_new:
        exit_transition_points.append(refined_q.copy())
    return ExitSeed(
        family_node_id=int(family_node_id),
        right_node_id=int(right_ids[-1]),
        q=refined_q.copy(),
        lambda_value=lam,
        discovered_round=int(round_idx),
        cluster_id=int(family_node_id),
    )


def run_locked_lambda_parallel_evidence_warmstart(
    *,
    graph: FamilyConnectivityGraph,
    start_node_id: int,
    goal_node_id: int,
    left_family: SphereFamily,
    left_manifold,
    left_center: np.ndarray,
    left_radius: float,
    transfer_family: ContinuousMaskedPlaneFamily,
    right_family: SphereFamily,
    right_manifold,
    right_center: np.ndarray,
    right_radius: float,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    frontier_ids: dict[str, list[int]],
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    adaptive_lambda_values: set[float],
    explored_edges_by_mode: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    entry_transition_points: list[np.ndarray],
    exit_transition_points: list[np.ndarray],
    round_sources: list[np.ndarray],
    round_targets: list[np.ndarray],
    round_budget: int,
    proposal_count: int = 1,
    plane_candidate_lambdas: list[float] | None = None,
) -> ParallelLeafWarmstartResult:
    """Run a parallel shared-proposal evidence phase with lambda-locked plane leaves.

    Each ambient proposal is projected onto the fixed left/right support manifolds
    and onto every candidate plane-family leaf. Only a utility-ranked subset of
    stores is expanded per proposal, but transitions are scanned against all
    currently known opposite-family leaves so the resulting graph can support
    multiple certified route alternatives.
    """

    left_lambda = float(list(left_family.sample_lambdas())[0])
    right_lambda = float(list(right_family.sample_lambdas())[0])
    left_manager = FamilyEvidenceManager(left_family, LEFT_STAGE, is_foliation=False, radius=left_lambda)
    family_manager = FamilyEvidenceManager(transfer_family, FAMILY_STAGE, is_foliation=True)
    right_manager = FamilyEvidenceManager(right_family, RIGHT_STAGE, is_foliation=False, radius=right_lambda)

    midpoint = 0.5 * (np.asarray(start_q, dtype=float) + np.asarray(goal_q, dtype=float))
    nominal_plane_guide = _project_guide_to_leaf(transfer_family, float(transfer_family.nominal_lambda), midpoint)
    guides = {
        LEFT_STAGE: np.asarray(nominal_plane_guide, dtype=float),
        FAMILY_STAGE: np.asarray(goal_q, dtype=float),
        RIGHT_STAGE: np.asarray(goal_q, dtype=float),
    }
    proposal_guides = {
        ex66.LEFT_STAGE: np.asarray(guides[LEFT_STAGE], dtype=float),
        ex66.PLANE_STAGE: np.asarray(nominal_plane_guide, dtype=float),
        ex66.RIGHT_STAGE: np.asarray(guides[RIGHT_STAGE], dtype=float),
    }
    plane_lambdas = (
        [float(lam) for lam in plane_candidate_lambdas]
        if plane_candidate_lambdas is not None and len(plane_candidate_lambdas) > 0
        else [float(lam) for lam in transfer_family.sample_lambdas({"count": 7})]
    )
    plane_guides = {
        float(lam): _project_guide_to_leaf(transfer_family, float(lam), midpoint)
        for lam in plane_lambdas
    }

    if left_manager.store is not None:
        start_projected = ex66.proposal_projection(left_manager.manifold, np.asarray(start_q, dtype=float))
        start_seed = np.asarray(start_q if start_projected is None else start_projected, dtype=float)
        start_id = ex66.add_stage_node(left_manager.store, start_seed, seeded_from_proposal=True)
        ex66.update_stage_frontier(left_manager.store, [int(start_id)], np.asarray(guides[LEFT_STAGE], dtype=float))
    if right_manager.store is not None:
        goal_projected = ex66.proposal_projection(right_manager.manifold, np.asarray(goal_q, dtype=float))
        goal_seed = np.asarray(goal_q if goal_projected is None else goal_projected, dtype=float)
        goal_id = ex66.add_stage_node(right_manager.store, goal_seed, seeded_from_proposal=True)
        ex66.update_stage_frontier(right_manager.store, [int(goal_id)], np.asarray(guides[RIGHT_STAGE], dtype=float))

    entry_seed_map: dict[tuple[int, int], StageSeed] = {}
    exit_seed_map: dict[tuple[int, int], ExitSeed] = {}

    for local_round in range(1, int(round_budget) + 1):
        stage_stores = _aggregate_stage_stores(left_manager, family_manager, right_manager)
        proposals = ex66.generate_ambient_proposals(
            round_idx=int(local_round),
            start_q=np.asarray(start_q, dtype=float),
            goal_q=np.asarray(goal_q, dtype=float),
            plane_point=np.asarray(transfer_family.base_point, dtype=float),
            stores=stage_stores,
            guides=proposal_guides,
            proposal_count=max(1, int(proposal_count)),
        )
        round_idx = int(local_round)
        for proposal in proposals:
            round_sources.append(np.asarray(proposal, dtype=float).copy())
            round_targets.append(np.asarray(proposal, dtype=float).copy())
            left_proj = ex66.proposal_projection(left_manager.manifold, np.asarray(proposal, dtype=float)) if left_manager.manifold is not None else None
            right_proj = ex66.proposal_projection(right_manager.manifold, np.asarray(proposal, dtype=float)) if right_manager.manifold is not None else None

            candidates: list[tuple[float, str, float | None, np.ndarray, ex66.StageEvidenceStore, np.ndarray, bool]] = []
            if left_proj is not None and left_manager.store is not None:
                q_left = np.asarray(left_proj, dtype=float)
                candidates.append(
                    (
                        _leaf_utility(LEFT_STAGE, left_manager.store, q_left, guides[LEFT_STAGE], stage_stores),
                        LEFT_STAGE,
                        None,
                        q_left,
                        left_manager.store,
                        np.asarray(guides[LEFT_STAGE], dtype=float),
                        False,
                    )
                )
            if right_proj is not None and right_manager.store is not None:
                q_right = np.asarray(right_proj, dtype=float)
                candidates.append(
                    (
                        _leaf_utility(RIGHT_STAGE, right_manager.store, q_right, guides[RIGHT_STAGE], stage_stores),
                        RIGHT_STAGE,
                        None,
                        q_right,
                        right_manager.store,
                        np.asarray(guides[RIGHT_STAGE], dtype=float),
                        False,
                    )
                )

            projected_plane_candidates: list[tuple[float, np.ndarray, ex66.StageEvidenceStore, np.ndarray, bool]] = []
            for lam in plane_lambdas:
                manifold = transfer_family.manifold(float(lam))
                plane_proj = ex66.proposal_projection(manifold, np.asarray(proposal, dtype=float))
                if plane_proj is None or family_manager.manager is None:
                    continue
                q_plane = np.asarray(plane_proj, dtype=float)
                adaptive_lambda_values.add(float(lam))
                plane_store, is_new = family_manager.manager.get_or_create_store(float(lam), q_plane, plane_guides[float(lam)])
                projected_plane_candidates.append(
                    (float(lam), q_plane, plane_store, np.asarray(plane_guides[float(lam)], dtype=float), bool(is_new))
                )

            stage_stores = _aggregate_stage_stores(left_manager, family_manager, right_manager)
            for lam, q_plane, plane_store, plane_guide, is_new in projected_plane_candidates:
                candidates.append(
                    (
                        _leaf_utility(FAMILY_STAGE, plane_store, q_plane, plane_guide, stage_stores),
                        FAMILY_STAGE,
                        float(lam),
                        q_plane,
                        plane_store,
                        plane_guide,
                        bool(is_new),
                    )
                )

            candidates.sort(key=lambda item: float(item[0]), reverse=True)
            budget = ex66.adaptive_stage_update_budget(stage_stores, None)
            selected_candidates = list(candidates[: max(1, int(budget))])
            best_plane_candidate = next((candidate for candidate in candidates if candidate[1] == FAMILY_STAGE), None)
            if best_plane_candidate is not None and not any(candidate[1] == FAMILY_STAGE for candidate in selected_candidates):
                if len(selected_candidates) < max(1, int(budget)):
                    selected_candidates.append(best_plane_candidate)
                elif len(selected_candidates) > 0:
                    selected_candidates[-1] = best_plane_candidate
            updated = 0
            updated_plane_candidates: list[tuple[float, np.ndarray]] = []
            updated_left_result = None
            updated_right_result = None
            updated_plane_results: list[tuple[float, ex66.StageEvidenceStore, object]] = []
            for _utility, family_name, lam, q_value, store, guide_point, _is_new in selected_candidates:
                _node_gain, _evals, result, _source = ex66.update_stage_evidence_from_proposal(
                    store,
                    np.asarray(q_value, dtype=float),
                    np.asarray(guide_point, dtype=float),
                )
                if family_name == LEFT_STAGE:
                    updated_left_result = result
                    _mirror_store_result_into_graph(
                        graph=graph,
                        store=store,
                        stage_mode=LEFT_STAGE,
                        lambda_value=None,
                        result=result,
                        seed_q=np.asarray(q_value, dtype=float),
                        round_idx=round_idx,
                        frontier_ids=frontier_ids,
                        guide_point=np.asarray(guide_point, dtype=float),
                        side_manifold=left_manifold,
                        sphere_center=np.asarray(left_center, dtype=float),
                        sphere_radius=float(left_radius),
                    )
                elif family_name == RIGHT_STAGE:
                    updated_right_result = result
                    _mirror_store_result_into_graph(
                        graph=graph,
                        store=store,
                        stage_mode=RIGHT_STAGE,
                        lambda_value=None,
                        result=result,
                        seed_q=np.asarray(q_value, dtype=float),
                        round_idx=round_idx,
                        frontier_ids=frontier_ids,
                        guide_point=np.asarray(guide_point, dtype=float),
                        side_manifold=right_manifold,
                        sphere_center=np.asarray(right_center, dtype=float),
                        sphere_radius=float(right_radius),
                    )
                else:
                    if lam is None:
                        continue
                    updated_plane_results.append((float(lam), store, result))
                    _mirror_family_leaf_result_into_graph(
                        graph=graph,
                        transfer_family=transfer_family,
                        path=(
                            np.asarray(getattr(result, "path", np.asarray([q_value], dtype=float)), dtype=float)
                            if result is not None
                            else np.asarray([q_value], dtype=float)
                        ),
                        lambda_value=float(lam),
                        round_idx=round_idx,
                        frontier_ids=frontier_ids,
                        guide_point=np.asarray(guide_point, dtype=float),
                    )
                    updated_plane_candidates.append((float(lam), np.asarray(q_value, dtype=float)))
                updated += 1

            if left_proj is not None:
                for lam, plane_q in updated_plane_candidates:
                    entry_seed = _maybe_add_entry_seed(
                        graph=graph,
                        left_manifold=left_manifold,
                        left_center=np.asarray(left_center, dtype=float),
                        left_radius=float(left_radius),
                        transfer_family=transfer_family,
                        frontier_ids=frontier_ids,
                        bounds_min=np.asarray(bounds_min, dtype=float),
                        bounds_max=np.asarray(bounds_max, dtype=float),
                        explored_edges_by_mode=explored_edges_by_mode,
                        adaptive_lambda_values=adaptive_lambda_values,
                        entry_transition_points=entry_transition_points,
                        round_idx=round_idx,
                        source_q=np.asarray(left_proj, dtype=float),
                        plane_q=np.asarray(plane_q, dtype=float),
                        lambda_value=float(lam),
                    )
                    if entry_seed is not None:
                        entry_seed_map[(int(entry_seed.family_node_id), int(entry_seed.side_node_id))] = entry_seed
                        _trim_entry_seed_map(entry_seed_map)
                if updated_left_result is not None and family_manager.manager is not None and left_manager.manifold is not None:
                    scanned_store_count = 0
                    for lam, plane_store in sorted(family_manager.manager.stores.items()):
                        if scanned_store_count >= PARALLEL_ALL_LEAF_SCAN_LIMIT:
                            break
                        scanned_store_count += 1
                        hits = ex66.transition_points_from_result(left_manager.manifold, plane_store.manifold, updated_left_result)
                        for hit in np.asarray(hits, dtype=float)[:PARALLEL_HITS_PER_SCAN]:
                            entry_seed = _maybe_add_entry_seed(
                                graph=graph,
                                left_manifold=left_manifold,
                                left_center=np.asarray(left_center, dtype=float),
                                left_radius=float(left_radius),
                                transfer_family=transfer_family,
                                frontier_ids=frontier_ids,
                                bounds_min=np.asarray(bounds_min, dtype=float),
                                bounds_max=np.asarray(bounds_max, dtype=float),
                                explored_edges_by_mode=explored_edges_by_mode,
                                adaptive_lambda_values=adaptive_lambda_values,
                                entry_transition_points=entry_transition_points,
                                round_idx=round_idx,
                                source_q=np.asarray(hit, dtype=float),
                                plane_q=np.asarray(hit, dtype=float),
                                lambda_value=float(lam),
                            )
                            if entry_seed is not None:
                                entry_seed_map[(int(entry_seed.family_node_id), int(entry_seed.side_node_id))] = entry_seed
                                _trim_entry_seed_map(entry_seed_map)

            if right_proj is not None:
                for lam, plane_q in updated_plane_candidates:
                    exit_seed = _maybe_add_exit_seed(
                        graph=graph,
                        right_manifold=right_manifold,
                        right_center=np.asarray(right_center, dtype=float),
                        right_radius=float(right_radius),
                        transfer_family=transfer_family,
                        frontier_ids=frontier_ids,
                        bounds_min=np.asarray(bounds_min, dtype=float),
                        bounds_max=np.asarray(bounds_max, dtype=float),
                        explored_edges_by_mode=explored_edges_by_mode,
                        exit_transition_points=exit_transition_points,
                        round_idx=round_idx,
                        plane_q=np.asarray(plane_q, dtype=float),
                        right_q=np.asarray(right_proj, dtype=float),
                        lambda_value=float(lam),
                    )
                    if exit_seed is not None:
                        exit_seed_map[(int(exit_seed.family_node_id), int(exit_seed.right_node_id))] = exit_seed
                        _trim_exit_seed_map(exit_seed_map)
                if updated_right_result is not None and family_manager.manager is not None and right_manager.manifold is not None:
                    scanned_store_count = 0
                    for lam, plane_store in sorted(family_manager.manager.stores.items()):
                        if scanned_store_count >= PARALLEL_ALL_LEAF_SCAN_LIMIT:
                            break
                        scanned_store_count += 1
                        hits = ex66.transition_points_from_result(right_manager.manifold, plane_store.manifold, updated_right_result)
                        for hit in np.asarray(hits, dtype=float)[:PARALLEL_HITS_PER_SCAN]:
                            exit_seed = _maybe_add_exit_seed(
                                graph=graph,
                                right_manifold=right_manifold,
                                right_center=np.asarray(right_center, dtype=float),
                                right_radius=float(right_radius),
                                transfer_family=transfer_family,
                                frontier_ids=frontier_ids,
                                bounds_min=np.asarray(bounds_min, dtype=float),
                                bounds_max=np.asarray(bounds_max, dtype=float),
                                explored_edges_by_mode=explored_edges_by_mode,
                                exit_transition_points=exit_transition_points,
                                round_idx=round_idx,
                                plane_q=np.asarray(hit, dtype=float),
                                right_q=np.asarray(hit, dtype=float),
                                lambda_value=float(lam),
                            )
                            if exit_seed is not None:
                                exit_seed_map[(int(exit_seed.family_node_id), int(exit_seed.right_node_id))] = exit_seed
                                _trim_exit_seed_map(exit_seed_map)

            if right_manager.manifold is not None:
                for lam, _plane_store, plane_result in updated_plane_results:
                    if plane_result is None:
                        continue
                    hits = ex66.transition_points_from_result(transfer_family.manifold(float(lam)), right_manager.manifold, plane_result)
                    for hit in np.asarray(hits, dtype=float)[:PARALLEL_HITS_PER_SCAN]:
                        exit_seed = _maybe_add_exit_seed(
                            graph=graph,
                            right_manifold=right_manifold,
                            right_center=np.asarray(right_center, dtype=float),
                            right_radius=float(right_radius),
                            transfer_family=transfer_family,
                            frontier_ids=frontier_ids,
                            bounds_min=np.asarray(bounds_min, dtype=float),
                            bounds_max=np.asarray(bounds_max, dtype=float),
                            explored_edges_by_mode=explored_edges_by_mode,
                            exit_transition_points=exit_transition_points,
                            round_idx=round_idx,
                            plane_q=np.asarray(hit, dtype=float),
                            right_q=np.asarray(hit, dtype=float),
                            lambda_value=float(lam),
                        )
                        if exit_seed is not None:
                            exit_seed_map[(int(exit_seed.family_node_id), int(exit_seed.right_node_id))] = exit_seed
                            _trim_exit_seed_map(exit_seed_map)
            if left_manager.manifold is not None:
                for lam, _plane_store, plane_result in updated_plane_results:
                    if plane_result is None:
                        continue
                    hits = ex66.transition_points_from_result(transfer_family.manifold(float(lam)), left_manager.manifold, plane_result)
                    for hit in np.asarray(hits, dtype=float)[:PARALLEL_HITS_PER_SCAN]:
                        entry_seed = _maybe_add_entry_seed(
                            graph=graph,
                            left_manifold=left_manifold,
                            left_center=np.asarray(left_center, dtype=float),
                            left_radius=float(left_radius),
                            transfer_family=transfer_family,
                            frontier_ids=frontier_ids,
                            bounds_min=np.asarray(bounds_min, dtype=float),
                            bounds_max=np.asarray(bounds_max, dtype=float),
                            explored_edges_by_mode=explored_edges_by_mode,
                            adaptive_lambda_values=adaptive_lambda_values,
                            entry_transition_points=entry_transition_points,
                            round_idx=round_idx,
                            source_q=np.asarray(hit, dtype=float),
                            plane_q=np.asarray(hit, dtype=float),
                            lambda_value=float(lam),
                        )
                        if entry_seed is not None:
                            entry_seed_map[(int(entry_seed.family_node_id), int(entry_seed.side_node_id))] = entry_seed
                            _trim_entry_seed_map(entry_seed_map)

        progress_stride = max(1, int(round_budget) // 4)
        if local_round == 1 or local_round == int(round_budget) or local_round % progress_stride == 0:
            print(
                "parallel_evidence_round"
                f" = {int(local_round)}"
                f", plane_leaf_stores = {len(family_manager.get_all_stores())}"
                f", entry_transitions = {len(entry_seed_map)}"
                f", exit_transitions = {len(exit_seed_map)}"
            )

    entry_seeds = keep_diverse_stage_seeds(
        list(entry_seed_map.values()),
        max_keep=MAX_ENTRY_SEEDS,
        lambda_tol=ENTRY_LAMBDA_DIVERSITY_TOL,
        spatial_tol=ENTRY_SPATIAL_DIVERSITY_TOL,
    )
    exit_seeds = keep_diverse_exit_seeds(
        list(exit_seed_map.values()),
        max_keep=FAMILY_ROUTE_SELECTION_MIN_EXIT_CANDIDATES,
        lambda_tol=EXIT_LAMBDA_DIVERSITY_TOL,
        spatial_tol=EXIT_SPATIAL_DIVERSITY_TOL,
    )
    return ParallelLeafWarmstartResult(
        entry_seeds=entry_seeds,
        initial_exit_seeds=exit_seeds,
        left_manager=left_manager,
        family_manager=family_manager,
        right_manager=right_manager,
        parallel_round_budget=int(round_budget),
        leaf_store_counts={
            LEFT_STAGE: len(left_manager.get_all_stores()),
            FAMILY_STAGE: len(family_manager.get_all_stores()),
            RIGHT_STAGE: len(right_manager.get_all_stores()),
        },
    )
