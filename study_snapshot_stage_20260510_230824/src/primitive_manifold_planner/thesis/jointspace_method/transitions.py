from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .modes import ConstraintMode


@dataclass(frozen=True)
class TransitionCertification:
    """Certification result for a shared theta transition."""

    success: bool
    residual_vector: np.ndarray
    residual_norm: float
    tolerance: float
    message: str = ""


@dataclass(frozen=True)
class TransitionConstraint:
    """Stacked FK-pulled-back transition constraint.

    A transition theta is valid only when it satisfies both adjacent mode
    residuals, e.g. sphere+plane or family-leaf+sphere.
    """

    source_mode: ConstraintMode
    target_mode: ConstraintMode
    name: str
    lambda_value: float | None = None

    def stacked_residual(self, theta: np.ndarray) -> np.ndarray:
        source = np.ravel(self.source_mode.residual(theta)).astype(float)
        target = np.ravel(self.target_mode.residual(theta)).astype(float)
        return np.concatenate([source, target])

    def residual_norm(self, theta: np.ndarray) -> float:
        return float(np.linalg.norm(self.stacked_residual(theta)))

    def certify(self, theta: np.ndarray, tol: float = 1e-3) -> TransitionCertification:
        residual = self.stacked_residual(theta)
        norm = float(np.linalg.norm(residual))
        success = bool(norm <= float(tol))
        return TransitionCertification(
            success=success,
            residual_vector=residual,
            residual_norm=norm,
            tolerance=float(tol),
            message="transition stack certified" if success else "transition stack residual exceeds tolerance",
        )
