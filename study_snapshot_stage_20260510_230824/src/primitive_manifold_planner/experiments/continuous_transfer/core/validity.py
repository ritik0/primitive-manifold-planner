"""Minimal validity-region interface used by Example 65 adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ValidityRegion(Protocol):
    """Thin validity interface for bounded or masked subsets of a manifold."""

    def is_valid(self, x: np.ndarray) -> bool:
        ...

    def margin(self, x: np.ndarray) -> float:
        ...


@dataclass(frozen=True)
class BoxValidityRegion:
    """Axis-aligned ambient-space validity box."""

    lower: np.ndarray
    upper: np.ndarray

    def is_valid(self, x: np.ndarray) -> bool:
        point = np.asarray(x, dtype=float).reshape(-1)
        return bool(np.all(point >= np.asarray(self.lower, dtype=float)) and np.all(point <= np.asarray(self.upper, dtype=float)))

    def margin(self, x: np.ndarray) -> float:
        point = np.asarray(x, dtype=float).reshape(-1)
        lower_gap = point - np.asarray(self.lower, dtype=float)
        upper_gap = np.asarray(self.upper, dtype=float) - point
        return float(np.min(np.concatenate([lower_gap, upper_gap])))
