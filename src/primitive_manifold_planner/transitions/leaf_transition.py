from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class LeafTransitionCandidate:
    x: np.ndarray
    residual_norm: float
    score: float
    source_family: str
    source_lam: object
    target_family: str
    target_lam: object


@dataclass
class LeafTransitionSearchResult:
    success: bool
    candidates: List[LeafTransitionCandidate]
    message: str = ""


class StackedManifold:
    """
    Represents the intersection of two manifolds by stacking their residuals:
        h(x) = [h_a(x), h_b(x)]
    """

    def __init__(self, manifold_a, manifold_b):
        self.manifold_a = manifold_a
        self.manifold_b = manifold_b
        self.name = f"stacked({getattr(manifold_a, 'name', 'A')},{getattr(manifold_b, 'name', 'B')})"

    def _coerce_point(self, x: np.ndarray) -> np.ndarray:
        return self.manifold_a._coerce_point(x)

    def residual(self, x: np.ndarray) -> np.ndarray:
        ra = np.atleast_1d(self.manifold_a.residual(x))
        rb = np.atleast_1d(self.manifold_b.residual(x))
        return np.concatenate([ra, rb], axis=0)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        ja = np.atleast_2d(self.manifold_a.jacobian(x))
        jb = np.atleast_2d(self.manifold_b.jacobian(x))
        return np.vstack([ja, jb])

    def is_valid(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        r = self.residual(x)
        return float(np.linalg.norm(r)) <= tol


def score_transition_candidate(
    x: np.ndarray,
    residual_norm: float,
    goal: Optional[np.ndarray],
    lam_change_penalty: float = 0.0,
) -> float:
    """
    Lower score is better.
    """
    score = residual_norm
    if goal is not None:
        score += 0.25 * float(np.linalg.norm(x - goal))
    score += lam_change_penalty
    return float(score)


def find_leaf_transition(
    source_family,
    source_lam,
    target_family,
    target_lam,
    seeds: List[np.ndarray],
    project_newton,
    goal: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    merge_tol: float = 1e-3,
) -> LeafTransitionSearchResult:
    """
    Find transition candidates between:
        M_source(source_lam) ∩ M_target(target_lam)

    Uses stacked residual projection.
    """
    source_leaf = source_family.manifold(source_lam)
    target_leaf = target_family.manifold(target_lam)
    stacked = StackedManifold(source_leaf, target_leaf)

    found: List[LeafTransitionCandidate] = []

    for seed in seeds:
        proj = project_newton(
            manifold=stacked,
            x0=np.asarray(seed, dtype=float),
            tol=1e-10,
            max_iters=50,
            damping=1.0,
        )

        if not proj.success:
            continue

        x_proj = proj.x_projected
        rn = float(proj.residual_norm)

        if rn > tol:
            continue

        duplicate = False
        for c in found:
            if np.linalg.norm(c.x - x_proj) < merge_tol:
                duplicate = True
                break
        if duplicate:
            continue

        lam_penalty = 0.0
        try:
            lam_penalty = 0.1 * source_family.lambda_distance(source_lam, target_lam)
        except Exception:
            lam_penalty = 0.0

        score = score_transition_candidate(
            x=x_proj,
            residual_norm=rn,
            goal=goal,
            lam_change_penalty=lam_penalty,
        )

        found.append(
            LeafTransitionCandidate(
                x=x_proj,
                residual_norm=rn,
                score=score,
                source_family=source_family.name,
                source_lam=source_lam,
                target_family=target_family.name,
                target_lam=target_lam,
            )
        )

    found.sort(key=lambda c: c.score)

    return LeafTransitionSearchResult(
        success=len(found) > 0,
        candidates=found,
        message=f"Found {len(found)} transition candidate(s).",
    )