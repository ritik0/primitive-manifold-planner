from __future__ import annotations

import numpy as np

from .base import ImplicitManifold


class DoubleSphereManifold(ImplicitManifold):
    """
    Disconnected union of two spheres in R^3 represented by a single implicit
    residual:

        h(x) = h1(x) * h2(x) = 0

    where h1 and h2 are the two sphere residuals.

    Away from simultaneous intersections this behaves as a valid codim-1 leaf
    with two disconnected components, which is useful for component-discovery
    validation in 3D.
    """

    def __init__(
        self,
        center_a: np.ndarray,
        center_b: np.ndarray,
        radius: float,
        name: str = "double_sphere",
    ) -> None:
        center_a = np.asarray(center_a, dtype=float).reshape(-1)
        center_b = np.asarray(center_b, dtype=float).reshape(-1)
        if center_a.shape[0] != 3 or center_b.shape[0] != 3:
            raise ValueError("DoubleSphereManifold centers must be 3D.")
        if radius <= 0.0:
            raise ValueError(f"radius must be positive, got {radius}")

        super().__init__(ambient_dim=3, codim=1, name=name)
        self.center_a = center_a
        self.center_b = center_b
        self.radius = float(radius)

    def _sphere_value(self, x: np.ndarray, center: np.ndarray) -> float:
        dx = x - center
        return float(np.dot(dx, dx) - self.radius**2)

    def residual(self, x: np.ndarray) -> np.ndarray:
        x = self._coerce_point(x)
        h1 = self._sphere_value(x, self.center_a)
        h2 = self._sphere_value(x, self.center_b)
        return np.array([h1 * h2], dtype=float)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        x = self._coerce_point(x)
        dx1 = x - self.center_a
        dx2 = x - self.center_b
        h1 = float(np.dot(dx1, dx1) - self.radius**2)
        h2 = float(np.dot(dx2, dx2) - self.radius**2)
        grad = h2 * (2.0 * dx1) + h1 * (2.0 * dx2)
        return grad.reshape(1, 3)

    def is_valid(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        x = self._coerce_point(x)
        h1 = abs(self._sphere_value(x, self.center_a))
        h2 = abs(self._sphere_value(x, self.center_b))
        return min(h1, h2) <= tol

    def __repr__(self) -> str:
        return (
            f"DoubleSphereManifold(name='{self.name}', center_a={self.center_a.tolist()}, "
            f"center_b={self.center_b.tolist()}, radius={self.radius})"
        )
