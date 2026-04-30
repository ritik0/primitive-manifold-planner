"""Compatibility re-exports for helpers split out of the old `common.py`.

New code should import from the focused helper modules directly:
- `lambda_utils.py`
- `seed_utils.py`
- `graph_insertions.py`
- `projection_utils.py`
"""

from __future__ import annotations

from .graph_insertions import (
    add_certified_family_edge,
    add_path_nodes_to_graph,
    choose_source_node_for_mode,
    register_frontier_node,
)
from .lambda_utils import (
    choose_underexplored_lambda_region,
    clamp_lambda,
    family_lambda_bins,
    family_lambda_values,
    family_nodes,
    quantize_lambda,
    refine_lambda_region_if_promising,
    summarize_explored_lambda_regions,
    summarize_lambda_coverage_gaps,
)
from .projection_utils import project_valid_family_state, sphere_radius_from_family
from .seed_utils import keep_diverse_exit_seeds, keep_diverse_stage_seeds

__all__ = [
    "add_certified_family_edge",
    "add_path_nodes_to_graph",
    "choose_source_node_for_mode",
    "choose_underexplored_lambda_region",
    "clamp_lambda",
    "family_lambda_bins",
    "family_lambda_values",
    "family_nodes",
    "keep_diverse_exit_seeds",
    "keep_diverse_stage_seeds",
    "project_valid_family_state",
    "quantize_lambda",
    "refine_lambda_region_if_promising",
    "register_frontier_node",
    "sphere_radius_from_family",
    "summarize_explored_lambda_regions",
    "summarize_lambda_coverage_gaps",
]
