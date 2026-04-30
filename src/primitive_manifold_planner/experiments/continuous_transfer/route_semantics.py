"""Route scoring and semantic preferences for the continuous-transfer experiment.

This module owns route-ranking logic and family-stage continuity summaries, so
planner decisions about which certified route to prefer stay separate from
graph search and geometric route reconstruction.
"""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass

from .config import (
    FAMILY_ROUTE_SELECTION_MIN_EXIT_CANDIDATES,
    LAMBDA_BIN_WIDTH,
    LAMBDA_SOURCE_TOL,
    SAME_LEAF_MIN_ATTEMPTS_BEFORE_SWITCH,
)
from .graph_types import FamilyConnectivityGraph, StageSeed


def _canonical_route_lambda(lambda_value: float | None) -> float | None:
    if lambda_value is None:
        return None
    return float(round(float(lambda_value) / float(LAMBDA_BIN_WIDTH)) * float(LAMBDA_BIN_WIDTH))


@dataclass(frozen=True)
class FamilyRouteProfile:
    entry_lambda: float | None
    exit_lambda: float | None
    lambda_min: float | None
    lambda_max: float | None
    lambda_span: float
    transverse_count: int
    constant_lambda_edge_count: int
    lambda_varying_edge_count: int
    same_leaf_only: bool
    same_leaf_with_primary: bool
    fixed_lambda_valid: bool


@dataclass(frozen=True)
class GraphRouteCandidate:
    """One graph-level route candidate before geometric reconstruction."""

    total_cost: float
    node_path: list[int]
    edge_path: list[int]
    terminal_support: object | None = None


def choose_primary_entry_seed(entry_seeds: list[StageSeed], right_manifold=None) -> StageSeed | None:
    if len(entry_seeds) == 0:
        return None
    ranked = sorted(
        entry_seeds,
        key=lambda seed: (
            float(np.linalg.norm(np.asarray(right_manifold.residual(seed.q), dtype=float)))
            if right_manifold is not None
            else 0.0,
            seed.discovered_round,
            abs(float(seed.lambda_value)),
            float(np.linalg.norm(np.asarray(seed.q, dtype=float))),
        ),
    )
    return ranked[0]


def score_route_with_family_preferences(
    graph: FamilyConnectivityGraph,
    edge_path: list[int],
    primary_entry_lambda: float | None,
) -> float:
    if len(edge_path) == 0:
        return float("inf")
    base_cost = float(sum(graph.edges[int(edge_id)].cost for edge_id in edge_path))
    family_edge_modes = []
    lambda_values: list[float] = []
    for edge_id in edge_path:
        edge = graph.edges[int(edge_id)]
        if str(edge.kind) not in {"family_leaf_motion", "family_transverse", "entry_transition", "exit_transition"}:
            continue
        if len(edge.path_lambdas) > 0:
            lambda_values.extend(float(v) for v in np.asarray(edge.path_lambdas, dtype=float).reshape(-1))
        elif edge.lambda_value is not None:
            lambda_values.append(float(edge.lambda_value))
        if str(edge.kind) in {"family_leaf_motion", "family_transverse"}:
            if len(edge.path_lambdas) > 0 and float(np.max(edge.path_lambdas) - np.min(edge.path_lambdas)) > LAMBDA_SOURCE_TOL:
                family_edge_modes.append("changing_lambda")
            else:
                family_edge_modes.append("constant_lambda")
    lambda_variation_penalty = 0.0 if len(lambda_values) == 0 else 0.22 * float(max(lambda_values) - min(lambda_values))
    lambda_varying_edge_count = int(sum(mode == "changing_lambda" for mode in family_edge_modes))
    transverse_edge_count_penalty = 0.55 * float(lambda_varying_edge_count)
    early_transverse_penalty = 0.0
    if "changing_lambda" in family_edge_modes:
        first_transverse_idx = int(family_edge_modes.index("changing_lambda"))
        family_steps_before_transverse = int(first_transverse_idx)
        if family_steps_before_transverse == 0:
            early_transverse_penalty += 0.18
        elif family_steps_before_transverse < SAME_LEAF_MIN_ATTEMPTS_BEFORE_SWITCH:
            early_transverse_penalty += 0.06 * float(
                SAME_LEAF_MIN_ATTEMPTS_BEFORE_SWITCH - family_steps_before_transverse
            )
    primary_leaf_departure_penalty = 0.0
    if primary_entry_lambda is not None and len(lambda_values) > 0:
        early_lambdas = [float(v) for v in lambda_values[: min(len(lambda_values), 8)]]
        primary_leaf_departure_penalty += 0.02 * float(
            sum(abs(float(v) - float(primary_entry_lambda)) > LAMBDA_SOURCE_TOL for v in early_lambdas)
        )
    profile = profile_family_route(graph, edge_path, primary_entry_lambda)
    same_leaf_bonus = -0.24 if profile.same_leaf_only else 0.0
    same_leaf_primary_bonus = -0.12 if profile.same_leaf_with_primary else 0.0
    return (
        base_cost
        + transverse_edge_count_penalty
        + early_transverse_penalty
        + lambda_variation_penalty
        + primary_leaf_departure_penalty
        + same_leaf_bonus
        + same_leaf_primary_bonus
    )


def profile_family_route(
    graph: FamilyConnectivityGraph,
    edge_path: list[int],
    primary_entry_lambda: float | None,
) -> FamilyRouteProfile:
    entry_lambda: float | None = None
    exit_lambda: float | None = None
    transverse_count = 0
    constant_lambda_edge_count = 0
    lambda_varying_edge_count = 0
    lambda_values: list[float] = []

    for edge_id in edge_path:
        edge = graph.edges[int(edge_id)]
        kind = str(edge.kind)
        if kind not in {"family_leaf_motion", "family_transverse", "entry_transition", "exit_transition"}:
            continue
        edge_lambdas: list[float] = []
        if len(edge.path_lambdas) > 0:
            edge_lambdas = [float(v) for v in np.asarray(edge.path_lambdas, dtype=float).reshape(-1)]
        elif edge.lambda_value is not None:
            edge_lambdas = [float(edge.lambda_value)]
        if len(edge_lambdas) == 0:
            continue
        if kind in {"family_leaf_motion", "family_transverse"}:
            if float(np.max(np.asarray(edge_lambdas, dtype=float)) - np.min(np.asarray(edge_lambdas, dtype=float))) > LAMBDA_SOURCE_TOL:
                transverse_count += 1
                lambda_varying_edge_count += 1
            else:
                constant_lambda_edge_count += 1
        lambda_values.extend(edge_lambdas)
        if kind == "entry_transition":
            entry_lambda = float(edge_lambdas[-1])
        elif kind == "exit_transition":
            exit_lambda = float(edge_lambdas[0])

    canonical_lambda_values = [
        float(value)
        for value in (_canonical_route_lambda(value) for value in lambda_values)
        if value is not None
    ]
    entry_lambda = _canonical_route_lambda(entry_lambda)
    exit_lambda = _canonical_route_lambda(exit_lambda)
    lambda_min = None if len(canonical_lambda_values) == 0 else float(min(canonical_lambda_values))
    lambda_max = None if len(canonical_lambda_values) == 0 else float(max(canonical_lambda_values))
    lambda_span = 0.0 if lambda_min is None or lambda_max is None else float(lambda_max - lambda_min)
    fixed_lambda_valid = (
        transverse_count == 0
        and entry_lambda is not None
        and exit_lambda is not None
        and abs(float(entry_lambda) - float(exit_lambda)) <= LAMBDA_SOURCE_TOL
        and lambda_span <= LAMBDA_SOURCE_TOL
    )
    same_leaf_only = bool(fixed_lambda_valid)
    same_leaf_with_primary = bool(
        fixed_lambda_valid
        and primary_entry_lambda is not None
        and entry_lambda is not None
        and exit_lambda is not None
        and abs(float(entry_lambda) - float(primary_entry_lambda)) <= LAMBDA_SOURCE_TOL
        and abs(float(exit_lambda) - float(primary_entry_lambda)) <= LAMBDA_SOURCE_TOL
    )
    return FamilyRouteProfile(
        entry_lambda=entry_lambda,
        exit_lambda=exit_lambda,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        lambda_span=lambda_span,
        transverse_count=int(transverse_count),
        constant_lambda_edge_count=int(constant_lambda_edge_count),
        lambda_varying_edge_count=int(lambda_varying_edge_count),
        same_leaf_only=bool(same_leaf_only),
        same_leaf_with_primary=bool(same_leaf_with_primary),
        fixed_lambda_valid=bool(fixed_lambda_valid),
    )


def summarize_family_route_semantics(
    graph: FamilyConnectivityGraph,
    edge_path: list[int],
    primary_entry_lambda: float | None,
) -> tuple[bool, bool]:
    family_edges = [
        graph.edges[int(edge_id)]
        for edge_id in edge_path
        if str(graph.edges[int(edge_id)].kind) in {"family_leaf_motion", "family_transverse"}
    ]
    if len(family_edges) == 0:
        return False, False
    stayed_same_leaf_first = True
    same_leaf_only = True
    for idx, edge in enumerate(family_edges):
        is_lambda_varying = (
            len(edge.path_lambdas) > 0
            and float(np.max(np.asarray(edge.path_lambdas, dtype=float)) - np.min(np.asarray(edge.path_lambdas, dtype=float))) > LAMBDA_SOURCE_TOL
        )
        if is_lambda_varying:
            same_leaf_only = False
            if idx < SAME_LEAF_MIN_ATTEMPTS_BEFORE_SWITCH:
                stayed_same_leaf_first = False
        if primary_entry_lambda is not None and len(edge.path_lambdas) > 0:
            early_lambdas = np.asarray(edge.path_lambdas[: min(len(edge.path_lambdas), 4)], dtype=float)
            if idx < SAME_LEAF_MIN_ATTEMPTS_BEFORE_SWITCH and np.any(
                np.abs(early_lambdas - float(primary_entry_lambda)) > LAMBDA_SOURCE_TOL
            ):
                stayed_same_leaf_first = False
    return bool(stayed_same_leaf_first), bool(same_leaf_only)


def rank_transition_supports_for_route_selection(
    supports,
    target_q: np.ndarray,
    top_k_assignments: int,
    min_keep: int = FAMILY_ROUTE_SELECTION_MIN_EXIT_CANDIDATES,
):
    """Rank transition supports using the same ordering Example 65 already used for exits."""

    ranked = sorted(
        supports,
        key=lambda support: (
            support.discovered_round,
            float(np.linalg.norm(np.asarray(support.q, dtype=float) - np.asarray(target_q, dtype=float))),
            abs(float(getattr(support, "lambda_value", 0.0))),
        ),
    )
    keep_count = 0
    if len(ranked) > 0:
        keep_count = max(
            1,
            int(top_k_assignments),
            int(min_keep),
            int(np.ceil(0.5 * len(ranked))),
        )
    return ranked[:keep_count]


def rank_graph_route_candidates(
    graph: FamilyConnectivityGraph,
    candidates: list[GraphRouteCandidate],
    primary_entry_lambda: float | None,
) -> list[GraphRouteCandidate]:
    """Rank graph route candidates with the same family-preference scoring as before."""

    candidate_profiles = {
        tuple(int(edge_id) for edge_id in candidate.edge_path): profile_family_route(
            graph,
            candidate.edge_path,
            primary_entry_lambda,
        )
        for candidate in candidates
    }
    fixed_lambda_candidates = [
        candidate
        for candidate in candidates
        if candidate_profiles[tuple(int(edge_id) for edge_id in candidate.edge_path)].fixed_lambda_valid
    ]
    return sorted(
        fixed_lambda_candidates,
        key=lambda candidate: (
            score_route_with_family_preferences(graph, candidate.edge_path, primary_entry_lambda),
            candidate.total_cost,
            candidate_profiles[tuple(int(edge_id) for edge_id in candidate.edge_path)].transverse_count,
            candidate_profiles[tuple(int(edge_id) for edge_id in candidate.edge_path)].lambda_span,
            0 if candidate_profiles[tuple(int(edge_id) for edge_id in candidate.edge_path)].same_leaf_with_primary else 1,
            0 if candidate_profiles[tuple(int(edge_id) for edge_id in candidate.edge_path)].same_leaf_only else 1,
            len(candidate.edge_path),
        ),
    )
