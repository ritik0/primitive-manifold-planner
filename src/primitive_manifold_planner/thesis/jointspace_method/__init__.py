"""Reusable joint-space constrained planning method components.

This package collects thesis-facing vocabulary shared by the fixed-plane
Example 66 robot demo and the continuous-transfer Example 65 robot demo.
The examples still own their planner policies; these components describe and
audit the common method: theta-space constraints pulled back through FK,
projected joint-space continuation segments, route candidates, certification,
and C-space debug artifacts.
"""

from .certification import RouteCertification, certify_dense_joint_route
from .continuation import ProjectedContinuationSegment, concatenate_segments
from .debug_artifacts import CspaceDebugArtifact, save_cspace_debug_artifacts
from .modes import ConstraintMode, FamilyLeafMode, FamilyMode
from .route import RouteCandidate
from .transitions import TransitionCertification, TransitionConstraint

__all__ = [
    "ConstraintMode",
    "FamilyMode",
    "FamilyLeafMode",
    "TransitionConstraint",
    "TransitionCertification",
    "ProjectedContinuationSegment",
    "concatenate_segments",
    "RouteCandidate",
    "RouteCertification",
    "certify_dense_joint_route",
    "CspaceDebugArtifact",
    "save_cspace_debug_artifacts",
]
