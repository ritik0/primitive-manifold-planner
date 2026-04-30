from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from .base import ImplicitManifold


class MaskedManifold(ImplicitManifold):
    """
    Wrapper that keeps the equality-constraint geometry of a base manifold but
    restricts validity to a user-defined subset through `within_bounds(...)`.

    This is useful when the manifold geometry is simple but the valid region on
    that manifold is disconnected, bounded, or task-dependent.
    """

    def __init__(
        self,
        base_manifold: ImplicitManifold,
        validity_fn: Callable[[np.ndarray], bool],
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            ambient_dim=int(base_manifold.ambient_dim),
            codim=int(base_manifold.codim),
            name=name or f"masked({getattr(base_manifold, 'name', 'manifold')})",
        )
        self.base_manifold = base_manifold
        self.validity_fn = validity_fn

    def residual(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.base_manifold.residual(x), dtype=float)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.base_manifold.jacobian(x), dtype=float)

    def _coerce_point(self, x: np.ndarray) -> np.ndarray:
        return self.base_manifold._coerce_point(x)

    def _coerce_vector(self, v: np.ndarray) -> np.ndarray:
        return self.base_manifold._coerce_vector(v)

    def is_valid(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        x = self._coerce_point(x)
        return bool(self.base_manifold.is_valid(x, tol=tol) and self.within_bounds(x, tol=tol))

    def within_bounds(self, x: np.ndarray, tol: float = 1e-9) -> bool:
        _ = tol
        x = self._coerce_point(x)
        return bool(self.validity_fn(np.asarray(x, dtype=float)))

    def __getattr__(self, item):
        return getattr(self.base_manifold, item)

    def __repr__(self) -> str:
        return f"MaskedManifold(name='{self.name}', base={repr(self.base_manifold)})"
