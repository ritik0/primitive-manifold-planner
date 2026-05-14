from __future__ import annotations

import numpy as np

from .base import ImplicitManifold


class PlaneManifold(ImplicitManifold):
    """
    Plane in R^3 represented implicitly as:

        M = { x in R^3 : n^T (x - p) = 0 }

    where:
        - n is the plane normal (normalized internally)
        - p is a point on the plane

    This is a 2D manifold embedded in 3D.
    Ambient dimension = 3
    Codimension = 1
    """

    def __init__(
        self,
        point: np.ndarray,
        normal: np.ndarray,
        name: str = "plane",
    ) -> None:
        point = np.asarray(point, dtype=float).reshape(-1)
        normal = np.asarray(normal, dtype=float).reshape(-1)

        if point.shape[0] != 3:
            raise ValueError(f"Plane point must have dimension 3, got {point.shape[0]}")
        if normal.shape[0] != 3:
            raise ValueError(f"Plane normal must have dimension 3, got {normal.shape[0]}")

        normal_norm = np.linalg.norm(normal)
        if normal_norm <= 1e-12:
            raise ValueError("Plane normal must be nonzero.")

        super().__init__(ambient_dim=3, codim=1, name=name)
        self.point = point
        self.normal = normal / normal_norm

    def residual(self, x: np.ndarray) -> np.ndarray:
        """
        Residual h(x) = n^T (x - p).

        Returns shape (1,).
        """
        x = self._coerce_point(x)
        value = float(np.dot(self.normal, x - self.point))
        return np.array([value], dtype=float)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of h(x) = n^T (x - p) is simply:

            dh/dx = n^T

        Returns shape (1, 3).
        """
        x = self._coerce_point(x)
        return self.normal.reshape(1, 3)

    def closest_point_analytic(self, x: np.ndarray) -> np.ndarray:
        """
        Analytic orthogonal projection of a point onto the plane.

        For h(x) = n^T (x - p), the orthogonal projection is:

            x_proj = x - h(x) n
        """
        x = self._coerce_point(x)
        signed_distance = self.residual(x)[0]
        return x - signed_distance * self.normal

    def is_valid(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        """
        Check whether x lies on the plane up to tolerance.
        """
        x = self._coerce_point(x)
        return abs(self.residual(x)[0]) <= tol

    def __repr__(self) -> str:
        return (
            f"PlaneManifold(name='{self.name}', point={self.point.tolist()}, "
            f"normal={self.normal.tolist()})"
        )