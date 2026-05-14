from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .continuation import ProjectedContinuationSegment, concatenate_segments


@dataclass
class RouteCandidate:
    """A realized or rejected joint-space route candidate.

    The candidate stores projected joint-space continuation segments and
    exposes ``dense_theta_path``, ``stage_labels``, and ``lambda_labels`` for
    certification/debugging.
    """

    segments: list[ProjectedContinuationSegment] = field(default_factory=list)
    transition_thetas: dict[str, np.ndarray] = field(default_factory=dict)
    selected_lambda: float | None = None
    cost: float = float("inf")
    certified: bool = False
    rejection_reason: str = ""
    realization_source: str = "certified_projected_jointspace_continuation_segments"

    @property
    def dense_theta_path(self) -> np.ndarray:
        path, _labels, _lambda_labels = concatenate_segments(self.segments)
        return path

    @property
    def stage_labels(self) -> list[str]:
        _path, labels, _lambda_labels = concatenate_segments(self.segments)
        return labels

    @property
    def lambda_labels(self) -> np.ndarray:
        _path, _labels, lambda_labels = concatenate_segments(self.segments)
        if self.selected_lambda is not None and len(lambda_labels) > 0:
            # Keep explicitly supplied segment labels, but fill unlabeled family
            # callers can do this before certification when needed.
            return lambda_labels
        return lambda_labels

    @property
    def final_route_taskspace_edges(self) -> int:
        # This route object is joint-space only; task-space traces are derived by FK.
        return 0
