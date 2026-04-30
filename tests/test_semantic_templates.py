import numpy as np

from primitive_manifold_planner.planners.mode_semantics import PlanningSemanticContext
from primitive_manifold_planner.planners.semantic_templates import (
    build_support_bridge_transfer_goal_semantic_model,
    build_support_transfer_goal_semantic_model,
)


def test_support_transfer_goal_template_allows_expected_role_transitions():
    model = build_support_transfer_goal_semantic_model(
        support_families=["left_table"],
        transfer_families=["transfer_zone"],
        goal_support_families=["right_table"],
    )

    assert model.transition_allowed(
        PlanningSemanticContext("left_table", 1.0, "transfer_zone", 0.0)
    )
    assert model.transition_allowed(
        PlanningSemanticContext("transfer_zone", 0.0, "right_table", 1.0)
    )
    assert not model.transition_allowed(
        PlanningSemanticContext("left_table", 1.0, "right_table", 1.0)
    )


def test_support_bridge_transfer_goal_template_preserves_bridge_switch_path():
    model = build_support_bridge_transfer_goal_semantic_model(
        support_families=["left_x_family"],
        bridge_families=["bridge_y_family"],
        transfer_families=["switch_q2_family"],
        goal_support_families=["right_x_family"],
    )

    assert model.transition_allowed(
        PlanningSemanticContext("left_x_family", -1.0, "bridge_y_family", -1.0)
    )
    assert model.transition_allowed(
        PlanningSemanticContext("bridge_y_family", -1.0, "switch_q2_family", 0.0)
    )
    assert model.transition_allowed(
        PlanningSemanticContext("switch_q2_family", 0.0, "right_x_family", 1.0)
    )
    assert not model.transition_allowed(
        PlanningSemanticContext("left_x_family", -1.0, "right_x_family", 1.0)
    )


def test_support_transfer_goal_template_passes_through_custom_semantic_rules():
    model = build_support_transfer_goal_semantic_model(
        support_families=["left_table"],
        transfer_families=["transfer_zone"],
        goal_support_families=["right_table"],
        transition_feasibility_fn=lambda context: float(context.point[0]) <= 1.0 if context.point is not None else True,
        transition_admissibility_fn=lambda context: abs(float(context.point[1])) if context.point is not None else 0.0,
    )

    good = PlanningSemanticContext(
        "left_table",
        1.0,
        "transfer_zone",
        0.0,
        point=np.array([0.5, -0.25]),
    )
    bad = PlanningSemanticContext(
        "left_table",
        1.0,
        "transfer_zone",
        0.0,
        point=np.array([1.5, -0.25]),
    )

    assert model.transition_feasible(good)
    assert not model.transition_feasible(bad)
    assert np.isclose(model.transition_admissibility_cost(good), 0.25)
