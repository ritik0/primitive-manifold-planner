"""Thin core interfaces for the continuous-transfer experiment.

These interfaces intentionally stay small. They give Example 65 a stable place
to grow future geometry, validity, and certification abstractions without
changing the current planner behavior.
"""

from .certification import CertificationResult
from .manifold import ConstrainedManifold
from .stage_graph import StageGraph, StageNode, TransitionConstraint
from .validity import BoxValidityRegion, ValidityRegion

__all__ = [
    "BoxValidityRegion",
    "CertificationResult",
    "ConstrainedManifold",
    "StageGraph",
    "StageNode",
    "TransitionConstraint",
    "ValidityRegion",
]
