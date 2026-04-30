from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np


@dataclass
class TransitionRegionSummary:
    region_id: int
    centroid: np.ndarray
    max_radius: float
    points: np.ndarray
    candidate_indices: list[int]


def cluster_transition_regions(
    points: Sequence[np.ndarray],
    candidate_indices: Iterable[int] | None = None,
    radius: float = 0.22,
) -> list[TransitionRegionSummary]:
    pts = [np.asarray(p, dtype=float).copy() for p in points]
    if not pts:
        return []

    if candidate_indices is None:
        candidate_indices = list(range(len(pts)))
    candidate_indices = [int(v) for v in candidate_indices]

    parent = list(range(len(pts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if np.linalg.norm(pts[i] - pts[j]) <= radius:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(pts)):
        groups.setdefault(find(i), []).append(i)

    regions: list[TransitionRegionSummary] = []
    for region_id, idxs in enumerate(groups.values()):
        region_points = np.asarray([pts[i] for i in idxs], dtype=float)
        centroid = np.mean(region_points, axis=0)
        max_radius = 0.0
        if len(region_points) > 0:
            max_radius = float(np.max(np.linalg.norm(region_points - centroid, axis=1)))
        regions.append(
            TransitionRegionSummary(
                region_id=int(region_id),
                centroid=np.asarray(centroid, dtype=float),
                max_radius=float(max_radius),
                points=region_points,
                candidate_indices=[candidate_indices[i] for i in idxs],
            )
        )

    return regions
