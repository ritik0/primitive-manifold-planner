from __future__ import annotations

import numpy as np

from .base import ImplicitManifold


class EllipseManifold(ImplicitManifold):
    """
    Axis-aligned ellipse in R^2 represented implicitly as:

        M = { x in R^2 :
              ((x0 - cx)^2 / a^2) + ((x1 - cy)^2 / b^2) - 1 = 0 }

    where:
        - center = [cx, cy]
        - a > 0 is the semi-axis length in x
        - b > 0 is the semi-axis length in y

    This is a 1D manifold embedded in 2D.
    Ambient dimension = 2
    Codimension = 1
    """

    def __init__(
        self,
        center: np.ndarray,
        a: float,
        b: float,
        name: str = "ellipse",
    ) -> None:
        center = np.asarray(center, dtype=float).reshape(-1)
        if center.shape[0] != 2:
            raise ValueError(f"Ellipse center must have dimension 2, got {center.shape[0]}")
        if a <= 0.0:
            raise ValueError(f"Ellipse semi-axis a must be positive, got {a}")
        if b <= 0.0:
            raise ValueError(f"Ellipse semi-axis b must be positive, got {b}")

        super().__init__(ambient_dim=2, codim=1, name=name)
        self.center = center
        self.a = float(a)
        self.b = float(b)

    def residual(self, x: np.ndarray) -> np.ndarray:
        """
        Residual:
            h(x) = ((x-cx)^2 / a^2) + ((y-cy)^2 / b^2) - 1

        Returns shape (1,).
        """
        x = self._coerce_point(x)
        dx = x[0] - self.center[0]
        dy = x[1] - self.center[1]
        value = (dx * dx) / (self.a * self.a) + (dy * dy) / (self.b * self.b) - 1.0
        return np.array([value], dtype=float)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of h(x):

            dh/dx = [ 2(x-cx)/a^2 , 2(y-cy)/b^2 ]

        Returns shape (1, 2).
        """
        x = self._coerce_point(x)
        dx = x[0] - self.center[0]
        dy = x[1] - self.center[1]
        return np.array(
            [[2.0 * dx / (self.a * self.a), 2.0 * dy / (self.b * self.b)]],
            dtype=float,
        )

    def point_from_angle(self, theta: float) -> np.ndarray:
        """
        Parametric point on the ellipse for visualization / testing:

            x = cx + a cos(theta)
            y = cy + b sin(theta)
        """
        return np.array(
            [
                self.center[0] + self.a * np.cos(theta),
                self.center[1] + self.b * np.sin(theta),
            ],
            dtype=float,
        )

    def is_valid(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        x = self._coerce_point(x)
        return abs(self.residual(x)[0]) <= tol

    def __repr__(self) -> str:
        return (
            f"EllipseManifold(name='{self.name}', center={self.center.tolist()}, "
            f"a={self.a}, b={self.b})"
        )