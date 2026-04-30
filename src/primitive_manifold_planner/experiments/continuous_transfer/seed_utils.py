"""Seed-diversity helpers for staged continuous-transfer exploration."""

from __future__ import annotations

import numpy as np

from .graph_types import ExitSeed, StageSeed


def keep_diverse_stage_seeds(
    seeds: list[StageSeed],
    max_keep: int,
    lambda_tol: float,
    spatial_tol: float,
) -> list[StageSeed]:
    if len(seeds) <= max_keep:
        return list(seeds)
    ranked = sorted(
        seeds,
        key=lambda seed: (
            abs(float(seed.lambda_value)),
            seed.discovered_round,
        ),
    )
    kept: list[StageSeed] = []
    for seed in ranked:
        if any(
            abs(float(seed.lambda_value) - float(other.lambda_value)) <= float(lambda_tol)
            and float(np.linalg.norm(np.asarray(seed.q, dtype=float) - np.asarray(other.q, dtype=float))) <= float(spatial_tol)
            for other in kept
        ):
            continue
        kept.append(seed)
        if len(kept) >= int(max_keep):
            return kept
    for seed in ranked:
        if seed in kept:
            continue
        kept.append(seed)
        if len(kept) >= int(max_keep):
            break
    return kept


def keep_diverse_exit_seeds(
    seeds: list[ExitSeed],
    max_keep: int,
    lambda_tol: float,
    spatial_tol: float,
) -> list[ExitSeed]:
    if len(seeds) <= max_keep:
        return list(seeds)
    ranked = sorted(
        seeds,
        key=lambda seed: (
            seed.discovered_round,
            abs(float(seed.lambda_value)),
        ),
    )
    kept: list[ExitSeed] = []
    for seed in ranked:
        if any(
            abs(float(seed.lambda_value) - float(other.lambda_value)) <= float(lambda_tol)
            and float(np.linalg.norm(np.asarray(seed.q, dtype=float) - np.asarray(other.q, dtype=float))) <= float(spatial_tol)
            for other in kept
        ):
            continue
        kept.append(seed)
        if len(kept) >= int(max_keep):
            return kept
    for seed in ranked:
        if seed in kept:
            continue
        kept.append(seed)
        if len(kept) >= int(max_keep):
            break
    return kept
