"""Lambda- and family-coverage helpers for the continuous-transfer experiment."""

from __future__ import annotations

import numpy as np

from .config import LAMBDA_BIN_WIDTH
from .family_definition import ContinuousMaskedPlaneFamily
from .graph_types import FamilyConnectivityGraph, FamilyGraphNode


def clamp_lambda(family: ContinuousMaskedPlaneFamily, lam: float) -> float:
    return float(np.clip(float(lam), family.lambda_min, family.lambda_max))


def quantize_lambda(lam: float) -> float:
    return float(np.round(float(lam) / LAMBDA_BIN_WIDTH) * LAMBDA_BIN_WIDTH)


def family_nodes(graph: FamilyConnectivityGraph) -> list[FamilyGraphNode]:
    return [graph.nodes[node_id] for node_id in graph.nodes_by_mode.get("family", [])]


def family_lambda_values(graph: FamilyConnectivityGraph) -> np.ndarray:
    vals = [
        float(node.lambda_value)
        for node in family_nodes(graph)
        if node.lambda_value is not None
    ]
    return np.asarray(sorted(set(round(v, 6) for v in vals)), dtype=float) if len(vals) > 0 else np.zeros((0,), dtype=float)


def family_lambda_bins(graph: FamilyConnectivityGraph) -> dict[float, list[int]]:
    bins: dict[float, list[int]] = {}
    for node in family_nodes(graph):
        if node.lambda_value is None:
            continue
        bins.setdefault(quantize_lambda(node.lambda_value), []).append(int(node.node_id))
    return bins


def summarize_explored_lambda_regions(graph: FamilyConnectivityGraph) -> list[str]:
    bins = sorted(family_lambda_bins(graph))
    if len(bins) == 0:
        return []
    regions: list[str] = []
    start = bins[0]
    prev = bins[0]
    for lam in bins[1:]:
        if float(lam - prev) <= 1.5 * LAMBDA_BIN_WIDTH:
            prev = lam
            continue
        regions.append(f"[{start:.2f}, {prev:.2f}]")
        start = lam
        prev = lam
    regions.append(f"[{start:.2f}, {prev:.2f}]")
    return regions


def summarize_lambda_coverage_gaps(
    transfer_family: ContinuousMaskedPlaneFamily,
    lambda_values: np.ndarray,
) -> list[str]:
    vals = np.asarray(sorted(set(float(v) for v in np.asarray(lambda_values, dtype=float).reshape(-1))), dtype=float)
    if len(vals) == 0:
        return [f"[{transfer_family.lambda_min:.2f}, {transfer_family.lambda_max:.2f}]"]
    gaps: list[str] = []
    boundaries = [float(transfer_family.lambda_min)] + [float(v) for v in vals] + [float(transfer_family.lambda_max)]
    for lam0, lam1 in zip(boundaries[:-1], boundaries[1:]):
        if float(lam1 - lam0) <= 1.5 * LAMBDA_BIN_WIDTH:
            continue
        gaps.append(f"[{lam0:.2f}, {lam1:.2f}]")
    return gaps


def choose_underexplored_lambda_region(
    lambda_probe_assignments: dict[float, int],
    adaptive_lambda_values: set[float],
    transfer_family: ContinuousMaskedPlaneFamily,
) -> float:
    candidates = sorted({quantize_lambda(v) for v in adaptive_lambda_values if transfer_family.lambda_in_range(v)})
    if len(candidates) == 0:
        candidates = [quantize_lambda(v) for v in transfer_family.sample_lambdas({"count": 7})]
    return min(
        candidates,
        key=lambda lam: (
            lambda_probe_assignments.get(float(lam), 0),
            abs(float(lam) - transfer_family.nominal_lambda),
        ),
    )


def refine_lambda_region_if_promising(
    lam: float,
    adaptive_lambda_values: set[float],
    transfer_family: ContinuousMaskedPlaneFamily,
) -> None:
    lam_center = float(quantize_lambda(lam))
    for candidate in (lam_center, lam_center):
        if transfer_family.lambda_in_range(candidate):
            adaptive_lambda_values.add(float(candidate))
    for offset in (-1.0, 1.0):
        candidate = float(lam_center + offset * LAMBDA_BIN_WIDTH)
        if transfer_family.lambda_in_range(candidate):
            adaptive_lambda_values.add(float(candidate))
