from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class ConstraintMode:
    """One FK-pulled-back robot constraint mode.

    The wrapped manifold remains the source of truth. This class gives the
    thesis examples a shared language for a stage such as "left sphere",
    "plane", or "family leaf at lambda".
    """

    name: str
    stage: str
    manifold: Any
    lambda_value: float | None = None
    color: str | None = None

    def residual(self, theta: np.ndarray) -> np.ndarray:
        # Delegate to the manifold so it remains the source of truth.
        return np.asarray(self.manifold.residual(np.asarray(theta, dtype=float)), dtype=float)

    def project(self, theta: np.ndarray, **kwargs: Any):
        # Projection semantics are owned by the wrapped active manifold.
        return self.manifold.project(np.asarray(theta, dtype=float), **kwargs)

    def within_bounds(self, theta: np.ndarray, tol: float = 1e-6) -> bool:
        return bool(self.manifold.within_bounds(np.asarray(theta, dtype=float), tol=tol))

    def end_effector(self, robot, theta: np.ndarray) -> np.ndarray:
        return np.asarray(robot.forward_kinematics_3d(np.asarray(theta, dtype=float))[-1], dtype=float)


@dataclass(frozen=True)
class FamilyLeafMode(ConstraintMode):
    """A constraint mode representing one fixed family leaf."""

    lambda_value: float


@dataclass(frozen=True)
class FamilyMode:
    """Factory for fixed-lambda family modes.

    ``leaf_factory`` should construct or return the robot manifold for the
    requested lambda value, for example ``RobotPlaneLeafManifold(lambda)``.
    """

    name: str
    stage: str
    leaf_factory: Callable[[float], Any]
    color: str | None = None

    def make_leaf(self, lambda_value: float) -> FamilyLeafMode:
        # Freeze one continuous family value into a fixed lambda leaf mode.
        lam = float(lambda_value)
        return FamilyLeafMode(
            name=f"{self.name}[lambda={lam:.6g}]",
            stage=self.stage,
            manifold=self.leaf_factory(lam),
            lambda_value=lam,
            color=self.color,
        )
