from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from primitive_manifold_planner.transitions.leaf_transition import (
    find_leaf_transition,
    LeafTransitionSearchResult,
)


@dataclass
class AttractionSample:
    x_ambient: np.ndarray
    x_on_source: np.ndarray
    target_residual_norm: float
    accepted_for_switch_attempt: bool


@dataclass
class AttractionSamplingResult:
    success: bool
    samples: List[AttractionSample]
    transition_result: LeafTransitionSearchResult
    message: str


def _unique_points(points: List[np.ndarray], merge_tol: float) -> List[np.ndarray]:
    unique = []
    for p in points:
        if not any(np.linalg.norm(p - q) < merge_tol for q in unique):
            unique.append(p)
    return unique


def sample_ambient_points_for_attraction(
    current_x: np.ndarray,
    goal_x: Optional[np.ndarray],
    n_local: int = 30,
    local_radius: float = 0.5,
    n_goal_bias: int = 20,
    goal_band_width: float = 0.25,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """
    Generate ambient samples:
    - local cloud around current_x
    - biased samples along current-to-goal direction if goal exists
    """
    if rng is None:
        rng = np.random.default_rng()

    current_x = np.asarray(current_x, dtype=float)
    dim = current_x.shape[0]

    samples: List[np.ndarray] = [current_x.copy()]

    # Local random cloud
    for _ in range(n_local):
        delta = rng.uniform(-local_radius, local_radius, size=dim)
        samples.append(current_x + delta)

    # Goal-biased corridor samples
    if goal_x is not None:
        goal_x = np.asarray(goal_x, dtype=float)
        direction = goal_x - current_x
        dist = float(np.linalg.norm(direction))
        if dist > 1e-12:
            direction = direction / dist

            for _ in range(n_goal_bias):
                alpha = rng.uniform(0.0, 1.0)
                center = current_x + alpha * (goal_x - current_x)

                # small lateral perturbation
                noise = rng.normal(0.0, goal_band_width, size=dim)

                # reduce perturbation along direction so the band is corridor-like
                noise = noise - np.dot(noise, direction) * direction
                samples.append(center + noise)

    return samples


def discover_transition_candidates_via_attraction(
    source_family,
    source_lam,
    target_family,
    target_lam,
    current_x: np.ndarray,
    project_newton,
    goal_x: Optional[np.ndarray] = None,
    ambient_local_radius: float = 0.5,
    n_local_samples: int = 30,
    n_goal_bias_samples: int = 20,
    goal_band_width: float = 0.25,
    target_residual_threshold: float = 0.25,
    projection_tol: float = 1e-10,
    projection_max_iters: int = 50,
    projection_damping: float = 1.0,
    merge_tol: float = 1e-3,
    rng: Optional[np.random.Generator] = None,
) -> AttractionSamplingResult:
    """
    Attraction-based transition discovery.

    Steps:
      1. Sample ambient points near current state / toward goal.
      2. Project them onto the source leaf.
      3. Compute target leaf residual norm.
      4. If the point is near the target leaf, try stacked projection
         using those projected points as seeds.
    """
    if rng is None:
        rng = np.random.default_rng()

    source_leaf = source_family.manifold(source_lam)
    target_leaf = target_family.manifold(target_lam)

    ambient_samples = sample_ambient_points_for_attraction(
        current_x=np.asarray(current_x, dtype=float),
        goal_x=None if goal_x is None else np.asarray(goal_x, dtype=float),
        n_local=n_local_samples,
        local_radius=ambient_local_radius,
        n_goal_bias=n_goal_bias_samples,
        goal_band_width=goal_band_width,
        rng=rng,
    )

    attraction_samples: List[AttractionSample] = []
    switch_seeds: List[np.ndarray] = []

    for x_amb in ambient_samples:
        proj = project_newton(
            manifold=source_leaf,
            x0=x_amb,
            tol=projection_tol,
            max_iters=projection_max_iters,
            damping=projection_damping,
        )

        if not proj.success:
            continue

        x_src = proj.x_projected
        target_r = target_leaf.residual(x_src)
        target_rn = float(np.linalg.norm(target_r))

        accepted = target_rn <= target_residual_threshold
        attraction_samples.append(
            AttractionSample(
                x_ambient=x_amb,
                x_on_source=x_src,
                target_residual_norm=target_rn,
                accepted_for_switch_attempt=accepted,
            )
        )

        if accepted:
            switch_seeds.append(x_src)

    switch_seeds = _unique_points(switch_seeds, merge_tol=merge_tol)

    if len(switch_seeds) == 0:
        empty_result = LeafTransitionSearchResult(
            success=False,
            candidates=[],
            message="No attraction samples were close enough to the target leaf.",
        )
        return AttractionSamplingResult(
            success=False,
            samples=attraction_samples,
            transition_result=empty_result,
            message="Attraction phase found no near-transition seeds.",
        )

    transition_result = find_leaf_transition(
        source_family=source_family,
        source_lam=source_lam,
        target_family=target_family,
        target_lam=target_lam,
        seeds=switch_seeds,
        project_newton=project_newton,
        goal=goal_x,
        tol=1e-6,
        merge_tol=merge_tol,
    )

    return AttractionSamplingResult(
        success=transition_result.success,
        samples=attraction_samples,
        transition_result=transition_result,
        message=transition_result.message,
    )