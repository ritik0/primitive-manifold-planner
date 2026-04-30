from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from primitive_manifold_planner.manifolds import ImplicitManifold


@dataclass
class ProjectionResult:
    success: bool
    x_projected: np.ndarray
    residual_norm: float
    iterations: int
    converged: bool
    message: str


def project_newton(
    manifold: ImplicitManifold,
    x0: np.ndarray,
    tol: float = 1e-10,
    max_iters: int = 50,
    damping: float = 1.0,
) -> ProjectionResult:
    if not (0.0 < damping <= 1.0):
        raise ValueError(f"damping must lie in (0, 1], got {damping}")

    x = np.asarray(x0, dtype=float).reshape(-1)
    x = manifold._coerce_point(x).copy()

    for k in range(max_iters):
        r = manifold.residual(x)
        r_norm = float(np.linalg.norm(r))

        if r_norm <= tol:
            return ProjectionResult(
                success=True,
                x_projected=x,
                residual_norm=r_norm,
                iterations=k,
                converged=True,
                message="Projection converged successfully.",
            )

        j = manifold.jacobian(x)

        try:
            delta, *_ = np.linalg.lstsq(j, -r, rcond=None)
        except np.linalg.LinAlgError:
            return ProjectionResult(
                success=False,
                x_projected=x,
                residual_norm=r_norm,
                iterations=k,
                converged=False,
                message="Projection failed: least-squares solve failed.",
            )

        if not np.all(np.isfinite(delta)):
            return ProjectionResult(
                success=False,
                x_projected=x,
                residual_norm=r_norm,
                iterations=k,
                converged=False,
                message="Projection failed: non-finite Newton step encountered.",
            )

        step_norm = float(np.linalg.norm(delta))
        if step_norm <= 1e-15:
            return ProjectionResult(
                success=False,
                x_projected=x,
                residual_norm=r_norm,
                iterations=k,
                converged=False,
                message="Projection stalled: step became too small before convergence.",
            )

        x = x + damping * delta

    final_r = manifold.residual(x)
    final_r_norm = float(np.linalg.norm(final_r))
    success = final_r_norm <= tol

    return ProjectionResult(
        success=success,
        x_projected=x,
        residual_norm=final_r_norm,
        iterations=max_iters,
        converged=success,
        message=(
            "Projection reached max iterations and converged."
            if success
            else "Projection reached max iterations without convergence."
        ),
    )