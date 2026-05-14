"""Certified route geometry views for the continuous-transfer experiment.

This module reconstructs geometric routes from stored certified graph edges and
derives the optional display view separately from graph search and route scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LAMBDA_SOURCE_TOL
from .family_definition import ContinuousMaskedPlaneFamily
from .graph_types import FamilyConnectivityGraph
from .projection_utils import project_valid_family_state
from .strict_validation import (
    sample_strict_family_leaf_path,
    sample_strict_family_transverse_segment,
    sample_strict_sphere_motion_path,
    validate_family_leaf_motion_edge,
    validate_family_transverse_edge,
    validate_left_motion_edge,
    validate_right_motion_edge,
)
from .support import concatenate_paths


@dataclass
class RouteGeometryViews:
    """Different geometric views of one selected graph route."""

    certified_path: np.ndarray
    display_path: np.ndarray


def orient_edge_path(
    graph: FamilyConnectivityGraph,
    edge_id: int,
    from_node_id: int | None = None,
    to_node_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    edge = graph.edges[int(edge_id)]
    path = np.asarray(edge.path, dtype=float)
    path_lambdas = np.asarray(edge.path_lambdas, dtype=float)
    if from_node_id is None or to_node_id is None:
        return path, path_lambdas
    if int(edge.node_u) == int(from_node_id) and int(edge.node_v) == int(to_node_id):
        return path, path_lambdas
    if int(edge.node_v) == int(from_node_id) and int(edge.node_u) == int(to_node_id):
        flipped_path = np.asarray(path[::-1], dtype=float)
        flipped_lambdas = (
            np.asarray(path_lambdas[::-1], dtype=float)
            if len(path_lambdas) > 0
            else np.zeros((0,), dtype=float)
        )
        return flipped_path, flipped_lambdas
    return path, path_lambdas


def concatenate_edge_paths(
    graph: FamilyConnectivityGraph,
    edge_ids: list[int],
    node_path: list[int] | None = None,
) -> np.ndarray:
    if len(edge_ids) == 0:
        return np.zeros((0, 3), dtype=float)
    if node_path is not None and len(node_path) == len(edge_ids) + 1:
        return concatenate_paths(
            *[
                orient_edge_path(graph, int(edge_id), int(node_path[idx]), int(node_path[idx + 1]))[0]
                for idx, edge_id in enumerate(edge_ids)
            ]
        )
    return concatenate_paths(*[np.asarray(graph.edges[int(edge_id)].path, dtype=float) for edge_id in edge_ids])


def smooth_family_route_block(
    graph: FamilyConnectivityGraph,
    edge_ids: list[int],
    transfer_family: ContinuousMaskedPlaneFamily,
    node_path: list[int] | None = None,
) -> np.ndarray:
    if len(edge_ids) == 0:
        return np.zeros((0, 3), dtype=float)

    waypoints: list[tuple[np.ndarray, float]] = []
    first_from = None if node_path is None or len(node_path) < 2 else int(node_path[0])
    first_to = None if node_path is None or len(node_path) < 2 else int(node_path[1])
    first_path, first_lambdas = orient_edge_path(graph, int(edge_ids[0]), first_from, first_to)
    first_edge = graph.edges[int(edge_ids[0])]
    first_lambda = (
        float(first_lambdas[0])
        if len(first_lambdas) > 0
        else None if first_edge.lambda_value is None else float(first_edge.lambda_value)
    )
    if first_lambda is None:
        return concatenate_edge_paths(graph, edge_ids, node_path=node_path)
    waypoints.append((np.asarray(first_path[0], dtype=float), float(first_lambda)))
    for idx, edge_id in enumerate(edge_ids):
        edge = graph.edges[int(edge_id)]
        edge_from = None if node_path is None or idx + 1 >= len(node_path) else int(node_path[idx])
        edge_to = None if node_path is None or idx + 1 >= len(node_path) else int(node_path[idx + 1])
        edge_path, edge_lambdas = orient_edge_path(graph, int(edge_id), edge_from, edge_to)
        lam = (
            float(edge_lambdas[-1])
            if len(edge_lambdas) > 0
            else None if edge.lambda_value is None else float(edge.lambda_value)
        )
        if lam is None:
            return concatenate_edge_paths(graph, edge_ids, node_path=node_path)
        waypoints.append((np.asarray(edge_path[-1], dtype=float), float(lam)))

    smoothed_segments: list[np.ndarray] = []
    cursor = 0
    while cursor < len(waypoints) - 1:
        q_start, lam_start = waypoints[cursor]
        chosen_segment: np.ndarray | None = None
        chosen_next = cursor + 1
        for next_idx in range(len(waypoints) - 1, cursor, -1):
            q_goal, lam_goal = waypoints[next_idx]
            if abs(float(lam_goal) - float(lam_start)) <= LAMBDA_SOURCE_TOL:
                candidate = sample_strict_family_leaf_path(q_start, q_goal)
                ok, _failures = validate_family_leaf_motion_edge(candidate, transfer_family, float(lam_start))
                if ok:
                    chosen_segment = candidate
                    chosen_next = next_idx
                    break
            else:
                candidate_path, candidate_lambdas = sample_strict_family_transverse_segment(
                    transfer_family=transfer_family,
                    q_start=q_start,
                    lambda_start=float(lam_start),
                    q_goal=q_goal,
                    lambda_goal=float(lam_goal),
                    project_valid_family_state_fn=project_valid_family_state,
                )
                if len(candidate_path) == 0:
                    continue
                ok, _failures = validate_family_transverse_edge(candidate_path, candidate_lambdas, transfer_family)
                if ok:
                    chosen_segment = candidate_path
                    chosen_next = next_idx
                    break
        if chosen_segment is None:
            q_goal, lam_goal = waypoints[cursor + 1]
            if abs(float(lam_goal) - float(lam_start)) <= LAMBDA_SOURCE_TOL:
                chosen_segment = sample_strict_family_leaf_path(q_start, q_goal)
            else:
                chosen_segment, _chosen_lambdas = sample_strict_family_transverse_segment(
                    transfer_family=transfer_family,
                    q_start=q_start,
                    lambda_start=float(lam_start),
                    q_goal=q_goal,
                    lambda_goal=float(lam_goal),
                    project_valid_family_state_fn=project_valid_family_state,
                )
            if len(chosen_segment) == 0:
                chosen_segment = np.asarray([q_start, q_goal], dtype=float)
            chosen_next = cursor + 1
        smoothed_segments.append(np.asarray(chosen_segment, dtype=float))
        cursor = chosen_next

    return concatenate_paths(*smoothed_segments)


def smooth_route_section_on_sphere(
    graph: FamilyConnectivityGraph,
    edge_ids: list[int],
    node_path: list[int],
    start_q: np.ndarray,
    goal_q: np.ndarray,
    manifold,
    center: np.ndarray,
    radius: float,
    side: str,
) -> np.ndarray:
    if len(edge_ids) == 0:
        return np.asarray([np.asarray(start_q, dtype=float)], dtype=float)
    smoothed = sample_strict_sphere_motion_path(center, radius, start_q, goal_q)
    if str(side) == "left":
        ok, _failures = validate_left_motion_edge(smoothed, manifold, center, radius)
    else:
        ok, _failures = validate_right_motion_edge(smoothed, manifold, center, radius)
    if ok:
        return np.asarray(smoothed, dtype=float)
    return concatenate_edge_paths(graph, edge_ids, node_path=node_path)


def build_sectioned_display_path(
    graph: FamilyConnectivityGraph,
    node_path: list[int],
    edge_path: list[int],
    left_manifold,
    left_center: np.ndarray,
    left_radius: float,
    transfer_family: ContinuousMaskedPlaneFamily,
    right_manifold,
    right_center: np.ndarray,
    right_radius: float,
) -> np.ndarray:
    if len(edge_path) == 0 or len(node_path) == 0:
        return np.zeros((0, 3), dtype=float)

    certified_path = concatenate_edge_paths(graph, edge_path, node_path=node_path)
    if len(certified_path) == 0:
        return np.zeros((0, 3), dtype=float)

    entry_idx = next((idx for idx, edge_id in enumerate(edge_path) if str(graph.edges[int(edge_id)].kind) == "entry_transition"), None)
    exit_idx = next((idx for idx, edge_id in enumerate(edge_path) if str(graph.edges[int(edge_id)].kind) == "exit_transition"), None)
    if entry_idx is None or exit_idx is None or int(exit_idx) <= int(entry_idx):
        return np.asarray(certified_path, dtype=float)

    entry_path, _entry_lambdas = orient_edge_path(
        graph,
        int(edge_path[entry_idx]),
        int(node_path[entry_idx]),
        int(node_path[entry_idx + 1]),
    )
    exit_path, _exit_lambdas = orient_edge_path(
        graph,
        int(edge_path[exit_idx]),
        int(node_path[exit_idx]),
        int(node_path[exit_idx + 1]),
    )
    entry_switch = np.asarray(entry_path[-1], dtype=float)
    exit_switch = np.asarray(exit_path[-1], dtype=float)

    left_edge_ids = [int(edge_id) for edge_id in edge_path[:entry_idx] if str(graph.edges[int(edge_id)].kind) == "left_motion"]
    left_node_path = [int(node_id) for node_id in node_path[: entry_idx + 1]]
    family_edge_ids = [
        int(edge_id)
        for edge_id in edge_path[entry_idx + 1 : exit_idx]
        if str(graph.edges[int(edge_id)].kind) in {"family_leaf_motion", "family_transverse"}
    ]
    family_node_path = [int(node_id) for node_id in node_path[entry_idx + 1 : exit_idx + 1]]
    right_edge_ids = [int(edge_id) for edge_id in edge_path[exit_idx + 1 :] if str(graph.edges[int(edge_id)].kind) == "right_motion"]
    right_node_path = [int(node_id) for node_id in node_path[exit_idx + 1 :]]

    left_section = smooth_route_section_on_sphere(
        graph=graph,
        edge_ids=left_edge_ids,
        node_path=left_node_path,
        start_q=np.asarray(certified_path[0], dtype=float),
        goal_q=entry_switch,
        manifold=left_manifold,
        center=left_center,
        radius=left_radius,
        side="left",
    )
    if len(family_edge_ids) > 0:
        family_section = smooth_family_route_block(
            graph=graph,
            edge_ids=family_edge_ids,
            transfer_family=transfer_family,
            node_path=family_node_path,
        )
    else:
        family_section = np.asarray([entry_switch, exit_switch], dtype=float)
    if len(family_section) > 0:
        family_section[0] = np.asarray(entry_switch, dtype=float)
        family_section[-1] = np.asarray(exit_switch, dtype=float)
    right_section = smooth_route_section_on_sphere(
        graph=graph,
        edge_ids=right_edge_ids,
        node_path=right_node_path,
        start_q=exit_switch,
        goal_q=np.asarray(certified_path[-1], dtype=float),
        manifold=right_manifold,
        center=right_center,
        radius=right_radius,
        side="right",
    )

    display_path = concatenate_paths(left_section, family_section, right_section)
    if len(display_path) == 0:
        return np.asarray(certified_path, dtype=float)
    display_path[0] = np.asarray(certified_path[0], dtype=float)
    display_path[-1] = np.asarray(certified_path[-1], dtype=float)
    return np.asarray(display_path, dtype=float)


def build_route_geometry_views(
    graph: FamilyConnectivityGraph,
    node_path: list[int],
    edge_path: list[int],
    left_manifold,
    left_center: np.ndarray,
    left_radius: float,
    transfer_family: ContinuousMaskedPlaneFamily,
    right_manifold,
    right_center: np.ndarray,
    right_radius: float,
) -> RouteGeometryViews:
    certified_path = concatenate_edge_paths(graph, edge_path, node_path=node_path)
    if len(edge_path) == 0:
        return RouteGeometryViews(
            certified_path=np.zeros((0, 3), dtype=float),
            display_path=np.zeros((0, 3), dtype=float),
        )
    display_path = build_sectioned_display_path(
        graph=graph,
        node_path=node_path,
        edge_path=edge_path,
        left_manifold=left_manifold,
        left_center=left_center,
        left_radius=left_radius,
        transfer_family=transfer_family,
        right_manifold=right_manifold,
        right_center=right_center,
        right_radius=right_radius,
    )
    return RouteGeometryViews(
        certified_path=np.asarray(certified_path, dtype=float),
        display_path=np.asarray(display_path, dtype=float),
    )
