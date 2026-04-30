"""Reusable staged continuous-transfer planning modules extracted from Example 65."""

from .benchmarks.scene_loader import (
    ContinuousTransferScene,
    build_continuous_transfer_scene,
    default_example_65_scene_description,
    parse_scene_description,
)
from .evidence_managers import FamilyEvidenceManager, LeafStoreManager
from .planner import (
    obstacle_profile_comparison_row,
    plan_continuous_transfer_route,
    print_continuous_route_summary,
    print_obstacle_profile_comparison,
)


def show_continuous_route(*args, **kwargs):
    """Lazily import PyVista visualization helpers only when visualization is requested."""

    from .visualization import show_continuous_route as _show_continuous_route

    return _show_continuous_route(*args, **kwargs)

__all__ = [
    "ContinuousTransferScene",
    "FamilyEvidenceManager",
    "LeafStoreManager",
    "build_continuous_transfer_scene",
    "default_example_65_scene_description",
    "obstacle_profile_comparison_row",
    "parse_scene_description",
    "plan_continuous_transfer_route",
    "print_continuous_route_summary",
    "print_obstacle_profile_comparison",
    "show_continuous_route",
]
