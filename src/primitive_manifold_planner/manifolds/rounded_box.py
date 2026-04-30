from __future__ import annotations

import numpy as np

from .base import ImplicitManifold


class RoundedBoxManifold(ImplicitManifold):
    """
    Smooth bounded cube-like closed surface using a 3D superquadric:

        |(x-cx)/a|^p + |(y-cy)/b|^p + |(z-cz)/c|^p = 1

    with even p > 2.
    """

    def __init__(
        self,
        center: np.ndarray,
        a: float,
        b: float,
        c: float,
        power: float = 4.0,
        name: str = "rounded_box",
    ) -> None:
        center = np.asarray(center, dtype=float).reshape(-1)
        if center.shape[0] != 3:
            raise ValueError("RoundedBoxManifold center must be 3D.")
        if a <= 0.0 or b <= 0.0 or c <= 0.0:
            raise ValueError("RoundedBoxManifold semi-sizes must be positive.")
        if power <= 2.0 or int(power) % 2 != 0:
            raise ValueError("RoundedBoxManifold power should be an even value > 2.")

        super().__init__(ambient_dim=3, codim=1, name=name)
        self.center = center
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.power = float(power)

    def residual(self, x: np.ndarray) -> np.ndarray:
        x = self._coerce_point(x)
        dx = abs((x[0] - self.center[0]) / self.a)
        dy = abs((x[1] - self.center[1]) / self.b)
        dz = abs((x[2] - self.center[2]) / self.c)
        value = np.power(dx, self.power) + np.power(dy, self.power) + np.power(dz, self.power) - 1.0
        return np.array([value], dtype=float)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        x = self._coerce_point(x)
        p = self.power

        def grad_component(delta: float, scale: float) -> float:
            if abs(delta) < 1e-15:
                return 0.0
            return p * np.sign(delta) * np.power(abs(delta), p - 1.0) / np.power(scale, p)

        dx = x[0] - self.center[0]
        dy = x[1] - self.center[1]
        dz = x[2] - self.center[2]
        return np.array(
            [[
                grad_component(dx, self.a),
                grad_component(dy, self.b),
                grad_component(dz, self.c),
            ]],
            dtype=float,
        )

    def within_bounds(self, x: np.ndarray, tol: float = 1e-9) -> bool:
        x = self._coerce_point(x)
        return (
            abs(float(x[0] - self.center[0])) <= self.a + tol
            and abs(float(x[1] - self.center[1])) <= self.b + tol
            and abs(float(x[2] - self.center[2])) <= self.c + tol
        )

    def __repr__(self) -> str:
        return (
            f"RoundedBoxManifold(name='{self.name}', center={self.center.tolist()}, "
            f"a={self.a}, b={self.b}, c={self.c}, power={self.power})"
        )
