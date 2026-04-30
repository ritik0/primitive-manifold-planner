from __future__ import annotations

import numpy as np

from .base import ImplicitManifold


class CircleManifold(ImplicitManifold):


    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        name: str = "circle",
    ) -> None:
        center = np.asarray(center, dtype=float).reshape(-1)
        if center.shape[0] != 2:
            raise ValueError(f"Circle center must have dimension 2, got {center.shape[0]}")
        if radius <= 0.0:
            raise ValueError(f"Circle radius must be positive, got {radius}")

        super().__init__(ambient_dim=2, codim=1, name=name)
        self.center = center
        self.radius = float(radius)

    def residual(self, x: np.ndarray) -> np.ndarray:

        x = self._coerce_point(x)
        dx = x - self.center
        value = np.dot(dx, dx) - self.radius**2
        return np.array([value], dtype=float)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        x = self._coerce_point(x)
        dx = x - self.center
        return 2.0 * dx.reshape(1, 2)

    def closest_point_analytic(self, x: np.ndarray) -> np.ndarray:
        """
        Analytic projection of a point onto the circle.

        If x is the center itself, projection is undefined because every
        direction is equally valid. In that case, choose one canonical point
        on the circle: center + [radius, 0].
        """
        x = self._coerce_point(x)
        dx = x - self.center
        norm_dx = np.linalg.norm(dx)

        if norm_dx < 1e-12:
            return self.center + np.array([self.radius, 0.0], dtype=float)

        return self.center + (self.radius / norm_dx) * dx

    def is_valid(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        """
        Check whether x lies on the circle up to tolerance.
        """
        x = self._coerce_point(x)
        return abs(self.residual(x)[0]) <= tol

    def __repr__(self) -> str:
        return (
            f"CircleManifold(name='{self.name}', center={self.center.tolist()}, "
            f"radius={self.radius})"
        )