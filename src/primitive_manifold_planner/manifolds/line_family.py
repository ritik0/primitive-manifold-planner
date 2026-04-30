from __future__ import annotations

import numpy as np

from .line import LineManifold


class ParallelLineFamily:
    """
    Family of parallel horizontal lines in R^2 parameterized by lambda:

        M_lambda = { (x, y) : y - lambda = 0 }

    Each leaf is a LineManifold with normal [0, 1] and point [0, lambda].
    """

    def __init__(self, name: str = "parallel_line_family") -> None:
        self.name = name

    def leaf(self, lam: float) -> LineManifold:
        """
        Return the line manifold corresponding to parameter lambda.
        """
        return LineManifold(
            point=np.array([0.0, float(lam)]),
            normal=np.array([0.0, 1.0]),
            name=f"{self.name}_lambda_{lam:g}",
        )

    def residual(self, x: np.ndarray, lam: float) -> np.ndarray:
        """
        Residual of the family at parameter lambda:
            h(x, lambda) = y - lambda
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != 2:
            raise ValueError(f"Expected 2D point, got shape {x.shape}")
        return np.array([x[1] - float(lam)], dtype=float)

    def is_on_leaf(self, x: np.ndarray, lam: float, tol: float = 1e-6) -> bool:
        """
        Check whether x lies on the leaf M_lambda.
        """
        return abs(self.residual(x, lam)[0]) <= tol

    def sample_lambdas(self, lam_min: float, lam_max: float, num: int) -> np.ndarray:
        """
        Sample a set of lambda values uniformly.
        """
        if num <= 0:
            raise ValueError(f"num must be positive, got {num}")
        return np.linspace(lam_min, lam_max, num)