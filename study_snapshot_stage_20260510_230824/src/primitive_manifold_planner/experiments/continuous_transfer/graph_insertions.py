"""Graph insertion and frontier-management helpers for the continuous-transfer experiment."""

from __future__ import annotations

import numpy as np

from .config import LAMBDA_SOURCE_TOL, MAX_EXPANSIONS_PER_NODE, MAX_FRONTIER_NODES_PER_GROUP
from .family_definition import ContinuousMaskedPlaneFamily
from .graph_types import FamilyConnectivityGraph
from .projection_utils import project_valid_family_state
from .strict_validation import (
    polyline_length,
    report_strict_validation_failures,
    sample_strict_family_leaf_path,
    sample_strict_family_transverse_segment,
    sample_strict_sphere_motion_path,
    validate_family_leaf_motion_edge,
    validate_family_transverse_edge,
    validate_left_motion_edge,
    validate_right_motion_edge,
)


def register_frontier_node(
    frontier_ids: dict[str, list[int]],
    graph: FamilyConnectivityGraph,
    mode: str,
    node_id: int,
    guide_point: np.ndarray,
    max_nodes: int = MAX_FRONTIER_NODES_PER_GROUP,
) -> None:
    node_id = int(node_id)
    ids = frontier_ids.setdefault(str(mode), [])
    if node_id not in ids:
        ids.append(node_id)
    if len(ids) <= max_nodes:
        return
    goal = np.asarray(guide_point, dtype=float)
    ranked = sorted(
        ids,
        key=lambda nid: (
            graph.nodes[nid].expansion_count,
            float(np.linalg.norm(graph.nodes[nid].q - goal)),
            graph.nodes[nid].discovered_round,
        ),
    )
    frontier_ids[str(mode)] = ranked[:max_nodes]


def choose_source_node_for_mode(
    graph: FamilyConnectivityGraph,
    frontier_ids: dict[str, list[int]],
    mode: str,
    candidate_q: np.ndarray,
    lambda_value: float | None = None,
) -> int | None:
    ids = [
        nid
        for nid in frontier_ids.get(str(mode), [])
        if graph.nodes[nid].expansion_count < MAX_EXPANSIONS_PER_NODE
    ]
    if mode == "family" and lambda_value is not None:
        ids = [
            nid
            for nid in ids
            if graph.nodes[nid].lambda_value is not None
            and abs(float(graph.nodes[nid].lambda_value) - float(lambda_value)) <= LAMBDA_SOURCE_TOL
        ]
    if len(ids) == 0:
        return None
    ids.sort(
        key=lambda nid: (
            graph.nodes[nid].expansion_count,
            float(np.linalg.norm(graph.nodes[nid].q - np.asarray(candidate_q, dtype=float))),
            graph.nodes[nid].discovered_round,
        )
    )
    return int(ids[0])


def add_path_nodes_to_graph(
    graph: FamilyConnectivityGraph,
    mode: str,
    path: np.ndarray,
    round_idx: int,
    frontier_ids: dict[str, list[int]],
    guide_point: np.ndarray,
    lambda_value: float | None = None,
    origin_sample_id: int | None = None,
    edge_kind: str | None = None,
    edge_label: str | None = None,
    node_kind: str = "explored",
    side_manifold=None,
    sphere_center: np.ndarray | None = None,
    sphere_radius: float | None = None,
) -> list[int]:
    pts = np.asarray(path, dtype=float)
    if len(pts) == 0:
        return []
    kind = str(edge_kind if edge_kind is not None else f"{mode}_motion")
    label = str(edge_label if edge_label is not None else kind)
    segment_records: list[tuple[np.ndarray, np.ndarray]] = []
    for idx in range(1, len(pts)):
        q_prev = np.asarray(pts[idx - 1], dtype=float)
        q_curr = np.asarray(pts[idx], dtype=float)
        segment_path = np.asarray([q_prev, q_curr], dtype=float)
        if kind in {"left_motion", "right_motion"}:
            if side_manifold is None or sphere_center is None or sphere_radius is None:
                return []
            center = np.asarray(sphere_center, dtype=float)
            radius = float(sphere_radius)
            segment_path = sample_strict_sphere_motion_path(
                center=center,
                radius=radius,
                q_start=q_prev,
                q_goal=q_curr,
            )
            if kind == "left_motion":
                ok, failures = validate_left_motion_edge(
                    segment_path,
                    left_manifold=side_manifold,
                    left_center=center,
                    left_radius=radius,
                )
            else:
                ok, failures = validate_right_motion_edge(
                    segment_path,
                    right_manifold=side_manifold,
                    right_center=center,
                    right_radius=radius,
                )
            if not ok:
                report_strict_validation_failures(f"Rejected {kind} edge before insertion", failures)
                return []
        segment_records.append((q_curr, segment_path))
    node_ids: list[int] = []
    previous_id = graph.register_node(
        mode,
        pts[0],
        round_idx,
        node_kind,
        lambda_value=lambda_value,
        origin_sample_id=origin_sample_id,
    )
    register_frontier_node(frontier_ids, graph, mode, previous_id, guide_point)
    node_ids.append(previous_id)
    for current_q, segment_path in segment_records:
        current_id = graph.register_node(
            mode,
            current_q,
            round_idx,
            node_kind,
            lambda_value=lambda_value,
            origin_sample_id=origin_sample_id,
        )
        graph.add_edge(
            node_u=previous_id,
            node_v=current_id,
            kind=kind,
            cost=polyline_length(segment_path),
            path=segment_path,
            label=label,
            lambda_value=lambda_value,
            origin_sample_id=origin_sample_id,
        )
        register_frontier_node(frontier_ids, graph, mode, current_id, guide_point)
        node_ids.append(current_id)
        previous_id = current_id
    return node_ids


def add_certified_family_edge(
    graph: FamilyConnectivityGraph,
    transfer_family: ContinuousMaskedPlaneFamily,
    path: np.ndarray,
    lambdas: np.ndarray,
    round_idx: int,
    frontier_ids: dict[str, list[int]],
    guide_point: np.ndarray,
    edge_kind: str,
    edge_label: str,
    origin_sample_id: int | None = None,
    node_kind: str = "explored",
) -> list[int]:
    pts = np.asarray(path, dtype=float)
    lam_arr = np.asarray(lambdas, dtype=float).reshape(-1)
    if len(pts) == 0 or len(pts) != len(lam_arr):
        return []
    segment_records: list[tuple[np.ndarray, float, np.ndarray, np.ndarray]] = []
    for idx in range(1, len(pts)):
        q_prev = np.asarray(pts[idx - 1], dtype=float)
        q_curr = np.asarray(pts[idx], dtype=float)
        lam_prev = float(lam_arr[idx - 1])
        lam_curr = float(lam_arr[idx])
        if str(edge_kind) == "family_leaf_motion":
            segment_path = sample_strict_family_leaf_path(q_prev, q_curr)
            segment_lambdas = np.full((len(segment_path),), lam_prev, dtype=float)
            ok, failures = validate_family_leaf_motion_edge(segment_path, transfer_family, lam_prev)
        elif str(edge_kind) == "family_transverse":
            segment_path, segment_lambdas = sample_strict_family_transverse_segment(
                transfer_family,
                q_prev,
                lam_prev,
                q_curr,
                lam_curr,
                project_valid_family_state,
            )
            ok, failures = validate_family_transverse_edge(segment_path, segment_lambdas, transfer_family)
        else:
            segment_path = np.asarray([q_prev, q_curr], dtype=float)
            segment_lambdas = np.asarray([lam_prev, lam_curr], dtype=float)
            ok, failures = True, []
        if not ok:
            report_strict_validation_failures(f"Rejected {edge_kind} edge before insertion", failures)
            return []
        segment_records.append((q_curr, lam_curr, segment_path, segment_lambdas))

    node_ids: list[int] = []
    previous_id = graph.register_node(
        "family",
        pts[0],
        round_idx,
        node_kind,
        lambda_value=float(lam_arr[0]),
        origin_sample_id=origin_sample_id,
    )
    register_frontier_node(frontier_ids, graph, "family", previous_id, guide_point)
    node_ids.append(previous_id)
    for current_q, current_lam, segment_path, segment_lambdas in segment_records:
        current_id = graph.register_node(
            "family",
            current_q,
            round_idx,
            node_kind,
            lambda_value=float(current_lam),
            origin_sample_id=origin_sample_id,
        )
        graph.add_edge(
            node_u=previous_id,
            node_v=current_id,
            kind=str(edge_kind),
            cost=polyline_length(segment_path),
            path=segment_path,
            path_lambdas=segment_lambdas,
            label=str(edge_label),
            lambda_value=float(0.5 * (float(segment_lambdas[0]) + float(segment_lambdas[-1]))),
            origin_sample_id=origin_sample_id,
        )
        register_frontier_node(frontier_ids, graph, "family", current_id, guide_point)
        node_ids.append(current_id)
        previous_id = current_id
    return node_ids
