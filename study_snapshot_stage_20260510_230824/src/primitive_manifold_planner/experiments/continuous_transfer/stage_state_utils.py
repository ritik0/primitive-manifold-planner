"""Shared graph-backed stage-state helpers for left/right support-stage planning.

This module keeps support-stage logic state-centric by providing the reusable
adapter layer between graph storage and `StageState` planning helpers.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np

from .graph_types import FamilyConnectivityGraph, StageState


def stage_state_from_node(
    graph: FamilyConnectivityGraph,
    node_id: int,
    expected_mode: str | None = None,
) -> StageState | None:
    node = graph.nodes[int(node_id)]
    if expected_mode is not None and str(node.mode) != str(expected_mode):
        return None
    return StageState(
        node_id=int(node.node_id),
        mode=str(node.mode),
        q=np.asarray(node.q, dtype=float).copy(),
        discovered_round=int(node.discovered_round),
        kind=str(node.kind),
        origin_sample_id=node.origin_sample_id,
        expansion_count=int(node.expansion_count),
    )


def stage_states_from_ids(
    graph: FamilyConnectivityGraph,
    node_ids: Iterable[int],
    expected_mode: str | None = None,
) -> list[StageState]:
    states: list[StageState] = []
    for node_id in node_ids:
        state = stage_state_from_node(graph, int(node_id), expected_mode=expected_mode)
        if state is not None:
            states.append(state)
    return states


def coerce_stage_state(
    graph: FamilyConnectivityGraph,
    node_or_state: int | StageState,
    expected_mode: str,
) -> StageState | None:
    if isinstance(node_or_state, StageState):
        return node_or_state if str(node_or_state.mode) == str(expected_mode) else None
    return stage_state_from_node(graph, int(node_or_state), expected_mode=expected_mode)


def increment_stage_state_expansion(
    graph: FamilyConnectivityGraph,
    state: StageState,
) -> StageState:
    """Touch graph bookkeeping only at explicit update points."""

    node = graph.nodes[int(state.node_id)]
    node.expansion_count += 1
    return replace(state, expansion_count=int(node.expansion_count))
