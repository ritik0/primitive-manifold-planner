"""Compatibility route-selection exports for the continuous-transfer experiment.

The original `route_selection.py` module has been split into smaller units:
- `graph_paths.py` for graph shortest-path utilities
- `route_semantics.py` for scoring and semantic preferences
- `route_geometry.py` for certified/display geometry views

This file remains as a thin compatibility wrapper during the refactor.
"""

from __future__ import annotations

from .graph_paths import shortest_path_over_graph
from .route_geometry import RouteGeometryViews, build_route_geometry_views
from .route_semantics import (
    choose_primary_entry_seed,
    score_route_with_family_preferences,
    summarize_family_route_semantics,
)

__all__ = [
    "RouteGeometryViews",
    "build_route_geometry_views",
    "choose_primary_entry_seed",
    "score_route_with_family_preferences",
    "shortest_path_over_graph",
    "summarize_family_route_semantics",
]
