"""Projection and support-geometry helpers for the continuous-transfer experiment."""

from __future__ import annotations

import numpy as np

from primitive_manifold_planner.families.standard import SphereFamily
from primitive_manifold_planner.projection import project_newton

from .config import STRICT_FAMILY_RESIDUAL_TOL
from .family_definition import ContinuousMaskedPlaneFamily


def project_valid_family_state(
    transfer_family: ContinuousMaskedPlaneFamily,
    lam: float,
    guess: np.ndarray,
    tol: float = STRICT_FAMILY_RESIDUAL_TOL,
) -> np.ndarray | None:
    manifold = transfer_family.manifold(float(lam))
    projection = project_newton(
        manifold=manifold,
        x0=np.asarray(guess, dtype=float),
        tol=1e-10,
        max_iters=80,
        damping=1.0,
    )
    if not projection.success:
        return None
    q = np.asarray(projection.x_projected, dtype=float)
    if not transfer_family.within_patch(float(lam), q):
        return None
    residual_norm = float(np.linalg.norm(np.asarray(manifold.residual(q), dtype=float)))
    if residual_norm > tol:
        return None
    return q


def sphere_radius_from_family(support_family: SphereFamily) -> float:
    return float(next(iter(support_family.radii.values())))
