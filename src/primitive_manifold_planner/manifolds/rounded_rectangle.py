from __future__ import annotations

import numpy as np

from .base import ImplicitManifold


class RoundedRectangleManifold(ImplicitManifold):
    """
    Smooth bounded rectangle-like closed curve using a superellipse:

        ((x-cx)/a)^p + ((y-cy)/b)^p = 1

    where:
        - center = [cx, cy]
        - a > 0 and b > 0 are half-width / half-height scales
        - p is an even exponent > 2 controlling corner sharpness

    This remains smooth enough for projection-based constrained planning while
    behaving like a bounded connector rather than an infinite line.
    """

    def __init__(
        self,
        center: np.ndarray,
        a: float,
        b: float,
        power: float = 4.0,
        name: str = "rounded_rectangle",
    ) -> None:
        center = np.asarray(center, dtype=float).reshape(-1)
        if center.shape[0] != 2:
            raise ValueError("RoundedRectangleManifold center must be 2D.")
        if a <= 0.0 or b <= 0.0:
            raise ValueError("RoundedRectangleManifold semi-sizes must be positive.")
        if power <= 2.0 or int(power) % 2 != 0:
            raise ValueError("RoundedRectangleManifold power should be an even value > 2.")

        super().__init__(ambient_dim=2, codim=1, name=name)
        self.center = center
        self.a = float(a)
        self.b = float(b)
        self.power = float(power)

    def residual(self, x: np.ndarray) -> np.ndarray:
        x = self._coerce_point(x)
        dx = (x[0] - self.center[0]) / self.a
        dy = (x[1] - self.center[1]) / self.b
        value = np.power(dx, self.power) + np.power(dy, self.power) - 1.0
        return np.array([value], dtype=float)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        x = self._coerce_point(x)
        dx = x[0] - self.center[0]
        dy = x[1] - self.center[1]
        p = self.power
        jx = p * np.power(dx, p - 1.0) / np.power(self.a, p)
        jy = p * np.power(dy, p - 1.0) / np.power(self.b, p)
        return np.array([[jx, jy]], dtype=float)

    def point_from_angle(self, theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        expo = 2.0 / self.power
        return np.array(
            [
                self.center[0] + self.a * np.sign(c) * np.power(abs(c), expo),
                self.center[1] + self.b * np.sign(s) * np.power(abs(s), expo),
            ],
            dtype=float,
        )

    def within_bounds(self, x: np.ndarray, tol: float = 1e-9) -> bool:
        x = self._coerce_point(x)
        return (
            abs(float(x[0] - self.center[0])) <= self.a + tol
            and abs(float(x[1] - self.center[1])) <= self.b + tol
        )

    def __repr__(self) -> str:
        return (
            f"RoundedRectangleManifold(name='{self.name}', center={self.center.tolist()}, "
            f"a={self.a}, b={self.b}, power={self.power})"
        )
