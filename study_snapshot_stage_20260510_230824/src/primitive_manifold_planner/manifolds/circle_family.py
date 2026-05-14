from __future__ import annotations

import numpy as np

from .circle import CircleManifold


class ConcentricCircleFamily:
    """
    Family of concentric circles in R^2 parameterized by lambda:

        M_lambda = { (x, y) : x^2 + y^2 - lambda^2 = 0 }

    with center fixed at the origin and lambda > 0 interpreted as radius.
    """

    def __init__(self, center: np.ndarray | None = None, name: str = "concentric_circle_family") -> None:
        if center is None:
            center = np.array([0.0, 0.0], dtype=float)
        center = np.asarray(center, dtype=float).reshape(-1)
        if center.shape[0] != 2:
            raise ValueError(f"Expected 2D center, got shape {center.shape}")

        self.center = center
        self.name = name

    def leaf(self, lam: float) -> CircleManifold:
        """
        Return the circle manifold corresponding to radius lambda.
        """
        lam = float(lam)
        if lam <= 0.0:
            raise ValueError(f"lambda/radius must be positive, got {lam}")

        return CircleManifold(
            center=self.center.copy(),
            radius=lam,
            name=f"{self.name}_lambda_{lam:g}",
        )

    def residual(self, x: np.ndarray, lam: float) -> np.ndarray:
        """
        Residual of the family at parameter lambda:
            h(x, lambda) = ||x - c||^2 - lambda^2
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != 2:
            raise ValueError(f"Expected 2D point, got shape {x.shape}")

        lam = float(lam)
        dx = x - self.center
        return np.array([np.dot(dx, dx) - lam**2], dtype=float)

    def is_on_leaf(self, x: np.ndarray, lam: float, tol: float = 1e-6) -> bool:
        """
        Check whether x lies on the leaf M_lambda.
        """
        return abs(self.residual(x, lam)[0]) <= tol

    def sample_lambdas(self, lam_min: float, lam_max: float, num: int) -> np.ndarray:
        """
        Sample a set of lambda values uniformly.
        """
        lam_min = float(lam_min)
        lam_max = float(lam_max)
        if lam_min <= 0.0:
            raise ValueError(f"lam_min must be positive, got {lam_min}")
        if lam_max <= lam_min:
            raise ValueError(f"lam_max must be greater than lam_min, got {lam_max} <= {lam_min}")
        if num <= 0:
            raise ValueError(f"num must be positive, got {num}")

        return np.linspace(lam_min, lam_max, num)