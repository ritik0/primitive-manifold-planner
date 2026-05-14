from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares

from primitive_manifold_planner.manifolds import ImplicitManifold


@dataclass
class TransitionResult:
    """
    Result of searching for an intersection / transition point
    between two implicit manifolds.
    """

    success: bool
    x_transition: np.ndarray | None
    residual_norm: float
    message: str
    seed_used: np.ndarray | None


@dataclass
class TransitionCandidate:
    """
    One candidate transition point between two manifolds.
    """

    point: np.ndarray
    residual_norm: float
    score: float
    seed_used: np.ndarray | None


@dataclass
class TransitionSearchResult:
    """
    Result of searching for multiple transition candidates.
    """

    success: bool
    best_candidate: TransitionCandidate | None
    candidates: list[TransitionCandidate]
    message: str


def combined_residual(
    manifold_a: ImplicitManifold,
    manifold_b: ImplicitManifold,
    x: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    ra = manifold_a.residual(x)
    rb = manifold_b.residual(x)
    return np.concatenate([ra, rb], axis=0)


def find_transition_point(
    manifold_a: ImplicitManifold,
    manifold_b: ImplicitManifold,
    seed: np.ndarray,
    tol: float = 1e-8,
    max_nfev: int = 200,
) -> TransitionResult:
    if manifold_a.ambient_dim != manifold_b.ambient_dim:
        raise ValueError(
            f"Ambient dimension mismatch: "
            f"{manifold_a.ambient_dim} vs {manifold_b.ambient_dim}"
        )

    seed = np.asarray(seed, dtype=float).reshape(-1)
    seed = manifold_a._coerce_point(seed)

    def residual_fn(x: np.ndarray) -> np.ndarray:
        return combined_residual(manifold_a, manifold_b, x)

    try:
        result = least_squares(
            residual_fn,
            x0=seed,
            max_nfev=max_nfev,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
    except Exception as exc:
        return TransitionResult(
            success=False,
            x_transition=None,
            residual_norm=np.inf,
            message=f"Optimization failed with exception: {exc}",
            seed_used=seed,
        )

    x_star = result.x
    r = residual_fn(x_star)
    r_norm = float(np.linalg.norm(r))

    if r_norm <= tol:
        return TransitionResult(
            success=True,
            x_transition=x_star,
            residual_norm=r_norm,
            message="Transition point found successfully.",
            seed_used=seed,
        )

    return TransitionResult(
        success=False,
        x_transition=x_star,
        residual_norm=r_norm,
        message="Optimization finished, but no valid transition was found.",
        seed_used=seed,
    )


def _candidate_score(point: np.ndarray, residual_norm: float) -> float:
    """
    Simple default score for ranking transition candidates.

    Lower is better.
    """
    point = np.asarray(point, dtype=float).reshape(-1)
    return float(residual_norm + 1e-6 * np.linalg.norm(point))


def _is_duplicate_candidate(
    point: np.ndarray,
    existing: list[TransitionCandidate],
    merge_tol: float,
) -> bool:
    for cand in existing:
        if np.linalg.norm(point - cand.point) <= merge_tol:
            return True
    return False


def find_transition_candidates(
    manifold_a: ImplicitManifold,
    manifold_b: ImplicitManifold,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    num_seeds: int = 50,
    tol: float = 1e-8,
    max_nfev: int = 200,
    merge_tol: float = 1e-3,
    max_candidates: int = 10,
    rng: np.random.Generator | None = None,
) -> TransitionSearchResult:
    """
    Search for multiple distinct transition candidates between two manifolds
    using repeated random seeding + local nonlinear solve.
    """
    if manifold_a.ambient_dim != manifold_b.ambient_dim:
        raise ValueError(
            f"Ambient dimension mismatch: "
            f"{manifold_a.ambient_dim} vs {manifold_b.ambient_dim}"
        )

    dim = manifold_a.ambient_dim
    bounds_min = np.asarray(bounds_min, dtype=float).reshape(-1)
    bounds_max = np.asarray(bounds_max, dtype=float).reshape(-1)

    if bounds_min.shape[0] != dim or bounds_max.shape[0] != dim:
        raise ValueError(
            f"Bounds dimension mismatch: expected {dim}, got "
            f"{bounds_min.shape[0]} and {bounds_max.shape[0]}"
        )

    if np.any(bounds_max <= bounds_min):
        raise ValueError("Each entry of bounds_max must be strictly greater than bounds_min.")

    if num_seeds <= 0:
        raise ValueError(f"num_seeds must be positive, got {num_seeds}")

    if max_candidates <= 0:
        raise ValueError(f"max_candidates must be positive, got {max_candidates}")

    if rng is None:
        rng = np.random.default_rng()

    candidates: list[TransitionCandidate] = []
    best_failed_residual = np.inf

    for _ in range(num_seeds):
        seed = rng.uniform(bounds_min, bounds_max)
        result = find_transition_point(
            manifold_a=manifold_a,
            manifold_b=manifold_b,
            seed=seed,
            tol=tol,
            max_nfev=max_nfev,
        )

        if result.success and result.x_transition is not None:
            point = np.asarray(result.x_transition, dtype=float).reshape(-1)

            if not _is_duplicate_candidate(point, candidates, merge_tol=merge_tol):
                candidates.append(
                    TransitionCandidate(
                        point=point,
                        residual_norm=result.residual_norm,
                        score=_candidate_score(point, result.residual_norm),
                        seed_used=result.seed_used,
                    )
                )

                candidates.sort(key=lambda c: c.score)
                if len(candidates) >= max_candidates:
                    candidates = candidates[:max_candidates]
        else:
            best_failed_residual = min(best_failed_residual, result.residual_norm)

    if len(candidates) == 0:
        return TransitionSearchResult(
            success=False,
            best_candidate=None,
            candidates=[],
            message=(
                "No valid transition candidates found."
                if not np.isfinite(best_failed_residual)
                else f"No valid transition candidates found. Best failed residual={best_failed_residual:.3e}"
            ),
        )

    candidates.sort(key=lambda c: c.score)
    return TransitionSearchResult(
        success=True,
        best_candidate=candidates[0],
        candidates=candidates,
        message=f"Found {len(candidates)} distinct transition candidate(s).",
    )


def random_transition_search(
    manifold_a: ImplicitManifold,
    manifold_b: ImplicitManifold,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    num_seeds: int = 50,
    tol: float = 1e-8,
    max_nfev: int = 200,
    rng: np.random.Generator | None = None,
) -> TransitionResult:
    """
    Backward-compatible wrapper that returns only the best transition point.
    """
    result = find_transition_candidates(
        manifold_a=manifold_a,
        manifold_b=manifold_b,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        num_seeds=num_seeds,
        tol=tol,
        max_nfev=max_nfev,
        rng=rng,
    )

    if not result.success or result.best_candidate is None:
        return TransitionResult(
            success=False,
            x_transition=None,
            residual_norm=np.inf,
            message=result.message,
            seed_used=None,
        )

    best = result.best_candidate
    return TransitionResult(
        success=True,
        x_transition=best.point.copy(),
        residual_norm=best.residual_norm,
        message=result.message,
        seed_used=best.seed_used,
    )