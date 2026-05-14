from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ProjectedContinuationSegment:
    """Dense theta segment produced by a joint-space projected connector."""

    mode_name: str
    stage: str
    start_theta: np.ndarray
    goal_theta: np.ndarray
    dense_theta_path: np.ndarray
    residuals: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    joint_steps: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    certified: bool = False
    source: str = "projected_jointspace_continuation"
    failure_reason: str = ""
    lambda_value: float | None = None

    @property
    def point_count(self) -> int:
        return int(len(np.asarray(self.dense_theta_path, dtype=float)))


def concatenate_segments(
    segments: list[ProjectedContinuationSegment],
    *,
    drop_duplicate_boundaries: bool = True,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Concatenate certified segment paths into one dense theta route.

    Returns ``(dense_theta_path, stage_labels, lambda_labels)``. Lambda labels
    are ``nan`` on non-family stages unless the segment carries lambda_value.
    """

    paths: list[np.ndarray] = []
    labels: list[str] = []
    lambda_labels: list[float] = []
    for segment in segments:
        path = np.asarray(segment.dense_theta_path, dtype=float)
        if path.ndim != 2 or path.shape[1] != 3 or len(path) == 0:
            continue
        if drop_duplicate_boundaries and paths:
            previous = paths[-1][-1]
            if float(np.linalg.norm(path[0] - previous)) <= 1e-10:
                path = path[1:]
        if len(path) == 0:
            continue
        paths.append(path)
        labels.extend([str(segment.stage)] * len(path))
        lam = float(segment.lambda_value) if segment.lambda_value is not None else float("nan")
        lambda_labels.extend([lam] * len(path))
    if not paths:
        return np.zeros((0, 3), dtype=float), [], np.zeros(0, dtype=float)
    return np.vstack(paths), labels, np.asarray(lambda_labels, dtype=float)
