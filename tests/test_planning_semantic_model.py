import numpy as np

from primitive_manifold_planner.planners.mode_semantics import (
    build_semantic_model_allowed_leaf_pair_fn,
    CompositePlanningSemanticModel,
    FamilyNameModeSemantics,
    PlanningSemanticContext,
)


def test_composite_planning_semantic_model_combines_roles_feasibility_and_admissibility():
    mode_semantics = FamilyNameModeSemantics(
        family_to_role={
            "support": "support",
            "transfer": "transfer",
        },
        allowed_role_pairs={
            ("support", "transfer"),
            ("transfer", "support"),
        },
        role_transition_costs={
            ("support", "transfer"): 2.0,
        },
    )

    model = CompositePlanningSemanticModel(
        mode_semantics=mode_semantics,
        transition_feasibility_fn=lambda context: float(context.point[0]) <= 1.0,
        transition_admissibility_fn=lambda context: 0.5 * abs(float(context.point[1])),
    )

    good = PlanningSemanticContext(
        source_family_name="support",
        source_lam=0.0,
        target_family_name="transfer",
        target_lam=1.0,
        point=np.array([0.75, -0.4]),
    )
    bad = PlanningSemanticContext(
        source_family_name="support",
        source_lam=0.0,
        target_family_name="transfer",
        target_lam=1.0,
        point=np.array([1.5, -0.4]),
    )

    assert model.transition_allowed(good)
    assert np.isclose(model.transition_cost(good), 2.0)
    assert model.transition_feasible(good)
    assert not model.transition_feasible(bad)
    assert np.isclose(model.transition_admissibility_cost(good), 0.2)


def test_semantic_model_allowed_leaf_pair_fn_uses_composite_role_logic():
    mode_semantics = FamilyNameModeSemantics(
        family_to_role={
            "left_support": "support",
            "transfer_zone": "transfer",
            "right_support": "goal_support",
        },
        allowed_role_pairs={
            ("support", "transfer"),
            ("transfer", "goal_support"),
        },
    )
    model = CompositePlanningSemanticModel(mode_semantics=mode_semantics)
    allowed = build_semantic_model_allowed_leaf_pair_fn(model)

    assert allowed("left_support", 1.0, "transfer_zone", 0.0)
    assert allowed("transfer_zone", 0.0, "right_support", 1.0)
    assert not allowed("left_support", 1.0, "right_support", 1.0)
