"""Thin adapters that wrap the current Example 65 geometry in stable interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from primitive_manifold_planner.families.standard import SphereFamily
from primitive_manifold_planner.manifolds import ImplicitManifold
from primitive_manifold_planner.projection import project_newton

from .core.manifold import ConstrainedManifold
from .core.validity import ValidityRegion
from .family_definition import ContinuousMaskedPlaneFamily


@dataclass(frozen=True)
class ImplicitManifoldAdapter(ConstrainedManifold):
    """Adapter that exposes an existing implicit manifold through the core interface."""

    manifold: ImplicitManifold
    projection_tol: float = 1e-10
    projection_iters: int = 50

    def residual(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.manifold.residual(x), dtype=float)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.manifold.jacobian(x), dtype=float)

    def ambient_dim(self) -> int:
        return int(self.manifold.ambient_dim)

    def intrinsic_dim(self) -> int:
        return int(self.manifold.ambient_dim - self.manifold.codim)

    def project(self, x: np.ndarray) -> np.ndarray | None:
        projection = project_newton(
            manifold=self.manifold,
            x0=np.asarray(x, dtype=float),
            tol=float(self.projection_tol),
            max_iters=int(self.projection_iters),
        )
        if not projection.success:
            return None
        return np.asarray(projection.x_projected, dtype=float)


@dataclass(frozen=True)
class SphereManifoldAdapter(ImplicitManifoldAdapter):
    """Adapter for current left/right sphere members."""

    center: np.ndarray | None = None
    radius: float | None = None

    @classmethod
    def from_family(cls, family: SphereFamily, radius: float | None = None) -> "SphereManifoldAdapter":
        chosen_radius = float(radius if radius is not None else next(iter(family.radii.values())))
        manifold = family.manifold(float(chosen_radius))
        return cls(
            manifold=manifold,
            center=np.asarray(family.center, dtype=float),
            radius=float(chosen_radius),
        )


@dataclass(frozen=True)
class FamilyLeafManifoldAdapter(ImplicitManifoldAdapter):
    """Adapter for one lambda leaf of the current continuous masked plane family."""

    transfer_family: ContinuousMaskedPlaneFamily | None = None
    lambda_value: float | None = None

    @classmethod
    def from_family(cls, transfer_family: ContinuousMaskedPlaneFamily, lam: float) -> "FamilyLeafManifoldAdapter":
        lambda_value = float(lam)
        manifold = transfer_family.manifold(lambda_value)
        return cls(
            manifold=manifold,
            transfer_family=transfer_family,
            lambda_value=lambda_value,
        )


@dataclass(frozen=True)
class FunctionalValidityRegion(ValidityRegion):
    """Simple function-backed validity region wrapper."""

    valid_fn: Callable[[np.ndarray], bool]
    margin_fn: Callable[[np.ndarray], float]

    def is_valid(self, x: np.ndarray) -> bool:
        return bool(self.valid_fn(np.asarray(x, dtype=float)))

    def margin(self, x: np.ndarray) -> float:
        return float(self.margin_fn(np.asarray(x, dtype=float)))


@dataclass(frozen=True)
class FamilyLeafValidityRegion(ValidityRegion):
    """Validity-region adapter for one lambda leaf of the current family patch."""

    transfer_family: ContinuousMaskedPlaneFamily
    lambda_value: float

    def is_valid(self, x: np.ndarray) -> bool:
        return bool(self.transfer_family.within_patch(float(self.lambda_value), np.asarray(x, dtype=float)))

    def margin(self, x: np.ndarray) -> float:
        return float(self.transfer_family.patch_margin(float(self.lambda_value), np.asarray(x, dtype=float)))
