from __future__ import annotations

import numpy as np

from .base import ImplicitManifold


class SphereManifold(ImplicitManifold):
    """
    Sphere in R^3 represented implicitly as:

        M = { x in R^3 : ||x - c||^2 - r^2 = 0 }

    where:
        - c is the center
        - r is the radius

    This is a 2D manifold embedded in 3D.
    Ambient dimension = 3
    Codimension = 1
    """

    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        name: str = "sphere",
    ) -> None:
        center = np.asarray(center, dtype=float).reshape(-1)
        if center.shape[0] != 3:
            raise ValueError(f"Sphere center must have dimension 3, got {center.shape[0]}")
        if radius <= 0.0:
            raise ValueError(f"Sphere radius must be positive, got {radius}")

        super().__init__(ambient_dim=3, codim=1, name=name)
        self.center = center
        self.radius = float(radius)

    def residual(self, x: np.ndarray) -> np.ndarray:
        """
        Residual h(x) = ||x - c||^2 - r^2.

        Returns shape (1,).
        """
        x = self._coerce_point(x)
        dx = x - self.center
        value = np.dot(dx, dx) - self.radius**2
        return np.array([value], dtype=float)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of h(x) = ||x - c||^2 - r^2 is:

            dh/dx = 2 (x - c)^T

        Returns shape (1, 3).
        """
        x = self._coerce_point(x)
        dx = x - self.center
        return 2.0 * dx.reshape(1, 3)

    def closest_point_analytic(self, x: np.ndarray) -> np.ndarray:
        """
        Analytic projection of a point onto the sphere.

        If x is the center itself, projection is not unique; choose a
        canonical point on the sphere along +x direction.
        """
        x = self._coerce_point(x)
        dx = x - self.center
        norm_dx = np.linalg.norm(dx)

        if norm_dx < 1e-12:
            return self.center + np.array([self.radius, 0.0, 0.0], dtype=float)

        return self.center + (self.radius / norm_dx) * dx

    def is_valid(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        """
        Check whether x lies on the sphere up to tolerance.
        """
        x = self._coerce_point(x)
        return abs(self.residual(x)[0]) <= tol

    def __repr__(self) -> str:
        return (
            f"SphereManifold(name='{self.name}', center={self.center.tolist()}, "
            f"radius={self.radius})"
        )