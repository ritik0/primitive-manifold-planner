from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .modes import ConstraintMode


@dataclass(frozen=True)
class TransitionCertification: #stores results
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
    lambda_value: float | None = None #for multiple possible lambdas in continuous space

    def stacked_residual(self, theta: np.ndarray) -> np.ndarray: #input theta output residual vecotr for both modes
        """Return source and target residuals together for one theta.

        A valid transition configuration must lie on both adjacent active
        manifolds, so certification checks this stacked vector near zero.
        """

        source = np.ravel(self.source_mode.residual(theta)).astype(float) #ravel --> flattens it so [[0.01]] becomes [0.01]
        target = np.ravel(self.target_mode.residual(theta)).astype(float)
        return np.concatenate([source, target]) #stacks them

    def residual_norm(self, theta: np.ndarray) -> float:
        return float(np.linalg.norm(self.stacked_residual(theta))) #norm of stacked residual sqrt(a^2 + b^2)

    def certify(self, theta: np.ndarray, tol: float = 1e-3) -> TransitionCertification: #perfomrs the transition check
        # Transition certification is stricter than per-stage membership:
        # the same theta must satisfy both adjacent residual equations.
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
