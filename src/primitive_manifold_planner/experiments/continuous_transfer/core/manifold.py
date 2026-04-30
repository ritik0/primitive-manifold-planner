"""Minimal constrained-manifold interface used by Example 65 adapters."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class ConstrainedManifold(Protocol):
    """Minimal geometry interface for future planner generalization."""

    def residual(self, x: np.ndarray) -> np.ndarray:
        ...

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        ...

    def ambient_dim(self) -> int:
        ...

    def intrinsic_dim(self) -> int:
        ...

    def project(self, x: np.ndarray) -> np.ndarray | None:
        ...
