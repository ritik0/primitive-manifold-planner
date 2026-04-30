import numpy as np

from primitive_manifold_planner.planners.admissibility import (
    build_transition_admissibility_fn,
    CoordinateAbsoluteValueAdmissibility,
    FamilyPairGateModel,
    GoalDistanceAdmissibility,
    SumAdmissibilityModel,
    TransitionContext,
)


class DummyFamily:
    def __init__(self, name: str):
        self.name = name


def test_sum_admissibility_model_accumulates_terms():
    model = SumAdmissibilityModel(
        [
            GoalDistanceAdmissibility(weight=0.5),
            CoordinateAbsoluteValueAdmissibility(axis=0, weight=2.0),
        ]
    )
    context = TransitionContext(
        source_family_name="a",
        source_lam=0.0,
        target_family_name="b",
        target_lam=0.0,
        point=np.array([1.0, 0.0]),
        goal_point=np.array([3.0, 0.0]),
    )

    assert np.isclose(model(context), 3.0)


def test_family_pair_gate_model_only_applies_to_matching_family_names():
    inner = CoordinateAbsoluteValueAdmissibility(axis=0, weight=1.0)
    model = FamilyPairGateModel(inner=inner, family_names=["switch"])

    matching = TransitionContext(
        source_family_name="switch",
        source_lam=0.0,
        target_family_name="goal",
        target_lam=0.0,
        point=np.array([2.0, 0.0]),
    )
    non_matching = TransitionContext(
        source_family_name="start",
        source_lam=0.0,
        target_family_name="goal",
        target_lam=0.0,
        point=np.array([2.0, 0.0]),
    )

    assert np.isclose(model(matching), 2.0)
    assert np.isclose(model(non_matching), 0.0)


def test_build_transition_admissibility_fn_adapts_model_to_planner_callback():
    model = FamilyPairGateModel(
        inner=CoordinateAbsoluteValueAdmissibility(axis=1, weight=3.0),
        family_pairs=[("left", "switch")],
    )
    callback = build_transition_admissibility_fn(model)

    value = callback(
        DummyFamily("left"),
        0.0,
        DummyFamily("switch"),
        1.0,
        np.array([0.5, -2.0]),
        None,
        {"hello": "world"},
    )
    ignored = callback(
        DummyFamily("left"),
        0.0,
        DummyFamily("right"),
        1.0,
        np.array([0.5, -2.0]),
        None,
        {"hello": "world"},
    )

    assert np.isclose(value, 6.0)
    assert np.isclose(ignored, 0.0)
