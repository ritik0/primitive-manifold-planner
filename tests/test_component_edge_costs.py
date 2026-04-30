import numpy as np

from primitive_manifold_planner.planners.component_leaf_graph import (
    ComponentEdge,
    default_component_edge_cost,
)


def test_default_component_edge_cost_uses_base_and_admissibility_metadata_when_available():
    edge = ComponentEdge(
        src=("a", 0.0, "0"),
        dst=("b", 1.0, "1"),
        transition_point=np.array([0.0, 0.0]),
        score=99.0,
        candidate_index=0,
        metadata={
            "base_score": 2.5,
            "admissibility_cost": 1.25,
        },
    )

    assert default_component_edge_cost(edge) == 4.75


def test_default_component_edge_cost_falls_back_to_score_without_metadata():
    edge = ComponentEdge(
        src=("a", 0.0, "0"),
        dst=("b", 1.0, "1"),
        transition_point=np.array([0.0, 0.0]),
        score=2.0,
        candidate_index=0,
    )

    assert default_component_edge_cost(edge) == 3.0
