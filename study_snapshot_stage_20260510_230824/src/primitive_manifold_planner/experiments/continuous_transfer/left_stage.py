"""Left-stage entry discovery for the continuous-transfer planner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from primitive_manifold_planner.projection import project_newton

from .config import (
    ENTRY_LAMBDA_DIVERSITY_TOL,
    ENTRY_SPATIAL_DIVERSITY_TOL,
    MAX_ENTRY_SEEDS,
    MAX_EXPANSIONS_PER_NODE,
    MIN_ENTRY_SEEDS,
    TRANSITION_INTERSECTION_TOL,
)
from .family_definition import ContinuousMaskedPlaneFamily
from .graph_insertions import add_path_nodes_to_graph, register_frontier_node
from .graph_types import FamilyConnectivityGraph, StageSeed, StageState
from .lambda_utils import clamp_lambda, refine_lambda_region_if_promising
from .seed_utils import keep_diverse_stage_seeds
from .stage_state_utils import increment_stage_state_expansion, stage_state_from_node, stage_states_from_ids
from .strict_validation import report_strict_validation_failures, validate_transition_edge
from .support import (
    explore_on_manifold_from_frontier,
    merge_edges,
    ompl_native_exploration_target,
    refine_intersection_on_both_manifolds,
)
from .right_stage import certify_side_connection_to_target


@dataclass
class LeftStageContext:
    """Shared left-stage resources for support-manifold entry discovery."""

    graph: FamilyConnectivityGraph
    left_manifold: object
    left_center: np.ndarray
    left_radius: float
    transfer_family: ContinuousMaskedPlaneFamily
    frontier_ids: dict[str, list[int]]
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    explored_edges_by_mode: dict[str, list[tuple[np.ndarray, np.ndarray]]]
    adaptive_lambda_values: set[float]
    entry_transition_points: list[np.ndarray]
    round_sources: list[np.ndarray]
    round_targets: list[np.ndarray]


def choose_left_stage_source(
    context: LeftStageContext,
    guide_point: np.ndarray,
) -> StageState | None:
    candidate_states = [
        state
        for state in stage_states_from_ids(
            context.graph,
            context.frontier_ids.get("left", []),
            expected_mode="left",
        )
        if state.expansion_count < MAX_EXPANSIONS_PER_NODE
    ]
    if len(candidate_states) == 0:
        return None
    guide = np.asarray(guide_point, dtype=float)
    candidate_states.sort(
        key=lambda state: (
            state.expansion_count,
            float(np.linalg.norm(np.asarray(state.q, dtype=float) - guide)),
            state.discovered_round,
        )
    )
    return candidate_states[0]


def sample_left_stage_target(
    left_manifold,
    transfer_family: ContinuousMaskedPlaneFamily,
    source_state: StageState,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    source_q = np.asarray(source_state.q, dtype=float)
    if rng.random() < 0.55:
        guess = (
            0.55 * source_q
            + 0.45 * np.asarray(transfer_family.base_point, dtype=float)
            + rng.normal(scale=0.40, size=3)
        )
        projection = project_newton(
            manifold=left_manifold,
            x0=np.asarray(guess, dtype=float),
            tol=1e-10,
            max_iters=80,
            damping=1.0,
        )
        if projection.success:
            return np.asarray(projection.x_projected, dtype=float)
    target = ompl_native_exploration_target(
        manifold=left_manifold,
        q_seed=np.asarray(source_q, dtype=float),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )
    if target is not None:
        return np.asarray(target, dtype=float)
    return source_q.copy()


def certify_entry_seed_from_left_state(
    context: LeftStageContext,
    left_state: StageState,
    clamp_lambda_fn,
    round_idx: int,
    refine_lambda_region_if_promising_fn,
) -> StageSeed | None:
    q_seed = np.asarray(left_state.q, dtype=float)
    lam = clamp_lambda_fn(context.transfer_family, context.transfer_family.infer_lambda(q_seed))
    family_manifold = context.transfer_family.manifold(lam)
    if float(np.linalg.norm(np.asarray(family_manifold.residual(q_seed), dtype=float))) > 0.45:
        return None

    refined, ok = refine_intersection_on_both_manifolds(
        manifold_a=context.left_manifold,
        manifold_b=family_manifold,
        x0=q_seed,
        tol=1e-8,
        max_iters=32,
    )
    if not ok:
        return None

    refined_q = np.asarray(refined, dtype=float)
    if not context.transfer_family.within_patch(lam, refined_q):
        return None
    if float(np.linalg.norm(np.asarray(context.left_manifold.residual(refined_q), dtype=float))) > TRANSITION_INTERSECTION_TOL:
        return None
    ok, failures = validate_transition_edge(
        path=np.asarray([refined_q], dtype=float),
        side_manifold=context.left_manifold,
        transfer_family=context.transfer_family,
        lam=float(lam),
        side_center=np.asarray(context.left_center, dtype=float),
        side_radius=float(context.left_radius),
        side_mode="left",
    )
    if not ok:
        report_strict_validation_failures("Rejected left-to-family entry seed", failures)
        return None

    left_ids = certify_side_connection_to_target(
        graph=context.graph,
        source_node_id=left_state,
        side_mode="left",
        side_manifold=context.left_manifold,
        sphere_center=np.asarray(context.left_center, dtype=float),
        sphere_radius=float(context.left_radius),
        target_q=refined_q,
        bounds_min=context.bounds_min,
        bounds_max=context.bounds_max,
        round_idx=round_idx,
        frontier_ids=context.frontier_ids,
        guide_point=np.asarray(context.transfer_family.base_point, dtype=float),
        explored_edges_by_mode=context.explored_edges_by_mode,
        origin_sample_id=left_state.origin_sample_id,
    )
    if len(left_ids) == 0:
        return None

    family_node_id = context.graph.register_node(
        mode="family",
        q=refined_q,
        discovered_round=round_idx,
        kind="transition",
        lambda_value=float(lam),
        origin_sample_id=left_state.origin_sample_id,
    )
    register_frontier_node(
        context.frontier_ids,
        context.graph,
        "family",
        family_node_id,
        0.5 * (np.asarray(context.transfer_family.base_point, dtype=float) + np.asarray(left_state.q, dtype=float)),
    )
    _edge_id, is_new = context.graph.add_edge(
        node_u=int(left_ids[-1]),
        node_v=family_node_id,
        kind="entry_transition",
        cost=0.0,
        path=np.asarray([refined_q], dtype=float),
        path_lambdas=np.asarray([lam], dtype=float),
        label="left_family_entry",
        lambda_value=float(lam),
    )
    context.adaptive_lambda_values.add(float(lam))
    refine_lambda_region_if_promising_fn(float(lam), context.adaptive_lambda_values, context.transfer_family)
    if is_new:
        context.entry_transition_points.append(refined_q.copy())
    return StageSeed(
        family_node_id=int(family_node_id),
        side_node_id=int(left_ids[-1]),
        q=refined_q.copy(),
        lambda_value=float(lam),
        discovered_round=int(round_idx),
        cluster_id=int(family_node_id),
    )


def expand_left_stage_locally(
    context: LeftStageContext,
    source_state: StageState,
    round_idx: int,
    rng: np.random.Generator,
) -> list[int]:
    source_state = increment_stage_state_expansion(context.graph, source_state)
    target_q = sample_left_stage_target(
        left_manifold=context.left_manifold,
        transfer_family=context.transfer_family,
        source_state=source_state,
        bounds_min=context.bounds_min,
        bounds_max=context.bounds_max,
        rng=rng,
    )
    context.round_sources.append(np.asarray(source_state.q, dtype=float).copy())
    context.round_targets.append(np.asarray(target_q, dtype=float).copy())
    result = explore_on_manifold_from_frontier(
        manifold=context.left_manifold,
        x_start=np.asarray(source_state.q, dtype=float),
        x_goal=np.asarray(target_q, dtype=float),
        bounds_min=context.bounds_min,
        bounds_max=context.bounds_max,
    )
    context.explored_edges_by_mode["left"] = merge_edges(
        context.explored_edges_by_mode.get("left", []),
        list(getattr(result, "explored_edges", [])),
    )
    return add_path_nodes_to_graph(
        graph=context.graph,
        mode="left",
        path=np.asarray(getattr(result, "path", np.zeros((0, 3), dtype=float)), dtype=float),
        round_idx=round_idx,
        frontier_ids=context.frontier_ids,
        guide_point=np.asarray(context.transfer_family.base_point, dtype=float),
        origin_sample_id=round_idx,
        edge_kind="left_motion",
        edge_label="left_motion",
        side_manifold=context.left_manifold,
        sphere_center=np.asarray(context.left_center, dtype=float),
        sphere_radius=float(context.left_radius),
    )


def discover_entry_seeds_from_left(
    graph: FamilyConnectivityGraph,
    start_node_id: int,
    left_manifold,
    left_center: np.ndarray,
    left_radius: float,
    transfer_family: ContinuousMaskedPlaneFamily,
    frontier_ids: dict[str, list[int]],
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    rng: np.random.Generator,
    left_round_budget: int,
    explored_edges_by_mode: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    adaptive_lambda_values: set[float],
    entry_transition_points: list[np.ndarray],
    round_sources: list[np.ndarray],
    round_targets: list[np.ndarray],
    round_offset: int = 0,
    clamp_lambda_fn=None,
    refine_lambda_region_if_promising_fn=None,
) -> list[StageSeed]:
    clamp_lambda_impl = clamp_lambda if clamp_lambda_fn is None else clamp_lambda_fn
    refine_lambda_region_if_promising_impl = (
        refine_lambda_region_if_promising
        if refine_lambda_region_if_promising_fn is None
        else refine_lambda_region_if_promising_fn
    )
    context = LeftStageContext(
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
        round_sources=round_sources,
        round_targets=round_targets,
    )
    entry_seeds: dict[int, StageSeed] = {}
    extra_budget = 2 if int(left_round_budget) >= MIN_ENTRY_SEEDS else 0
    total_budget = int(left_round_budget + extra_budget)
    start_state = stage_state_from_node(graph, int(start_node_id), expected_mode="left")
    for local_round_idx in range(1, total_budget + 1):
        round_idx = int(round_offset + local_round_idx)
        source_state = choose_left_stage_source(
            context=context,
            guide_point=np.asarray(transfer_family.base_point, dtype=float),
        )
        if source_state is None:
            source_state = start_state
        if source_state is None:
            break

        left_ids = expand_left_stage_locally(
            context=context,
            source_state=source_state,
            round_idx=round_idx,
            rng=rng,
        )
        probe_states = [source_state] + stage_states_from_ids(graph, left_ids, expected_mode="left")
        for left_state in probe_states:
            seed = certify_entry_seed_from_left_state(
                context=context,
                left_state=left_state,
                clamp_lambda_fn=clamp_lambda_impl,
                round_idx=round_idx,
                refine_lambda_region_if_promising_fn=refine_lambda_region_if_promising_impl,
            )
            if seed is not None:
                entry_seeds[int(seed.family_node_id)] = seed
                diverse_now = keep_diverse_stage_seeds(
                    list(entry_seeds.values()),
                    max_keep=MAX_ENTRY_SEEDS,
                    lambda_tol=ENTRY_LAMBDA_DIVERSITY_TOL,
                    spatial_tol=ENTRY_SPATIAL_DIVERSITY_TOL,
                )
                entry_seeds = {int(diverse_seed.family_node_id): diverse_seed for diverse_seed in diverse_now}
                if len(entry_seeds) >= MAX_ENTRY_SEEDS:
                    break
        if len(entry_seeds) >= MAX_ENTRY_SEEDS:
            break
    diverse_seeds = keep_diverse_stage_seeds(
        list(entry_seeds.values()),
        max_keep=MAX_ENTRY_SEEDS,
        lambda_tol=ENTRY_LAMBDA_DIVERSITY_TOL,
        spatial_tol=ENTRY_SPATIAL_DIVERSITY_TOL,
    )
    if len(diverse_seeds) < MIN_ENTRY_SEEDS:
        return diverse_seeds
    return diverse_seeds
