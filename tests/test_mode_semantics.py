import numpy as np

from primitive_manifold_planner.planners.component_leaf_graph import (
    ComponentEdge,
    default_component_edge_cost,
)
from primitive_manifold_planner.planners.mode_semantics import (
    FamilyNameModeSemantics,
    build_allowed_leaf_pair_fn,
)


def test_family_name_mode_semantics_enforces_role_transition_rules():
    semantics = FamilyNameModeSemantics(
        family_to_role={
            "left_support": "support",
            "transfer_zone": "transfer",
            "right_support": "support",
        },
        allowed_role_pairs={
            ("support", "transfer"),
            ("transfer", "support"),
        },
    )

    allowed = build_allowed_leaf_pair_fn(semantics)

    assert allowed("left_support", 1.0, "transfer_zone", 0.0)
    assert allowed("transfer_zone", 0.0, "right_support", 1.0)
    assert not allowed("left_support", 1.0, "right_support", 1.0)


def test_family_name_mode_semantics_provides_role_transition_costs():
    semantics = FamilyNameModeSemantics(
        family_to_role={
            "support_a": "support",
            "connector": "transfer",
        },
        role_transition_costs={
            ("support", "transfer"): 2.5,
        },
    )

    assert semantics.transition_cost("support_a", 0.0, "connector", 1.0) == 2.5
    assert semantics.transition_cost("connector", 1.0, "support_a", 0.0) == 0.0


def test_default_component_edge_cost_uses_semantic_metadata_without_breaking():
    edge = ComponentEdge(
        src=("support_a", 0.0, "0"),
        dst=("connector", 1.0, "0"),
        transition_point=np.array([0.0, 0.0]),
        score=10.0,
        candidate_index=0,
        metadata={
            "base_score": 1.25,
            "admissibility_cost": 0.75,
        },
    )

    assert np.isclose(default_component_edge_cost(edge), 3.0)
