import numpy as np

from primitive_manifold_planner.manifolds import CircleManifold
from primitive_manifold_planner.planners.admissibility import (
    AllFeasibilityModel,
    build_transition_feasibility_fn,
    CoordinateRangeFeasibility,
    FamilyPairGateFeasibilityModel,
)
from primitive_manifold_planner.planners.transition_manager import TransitionGenerator
from primitive_manifold_planner.projection import project_newton


class SingleLeafFamily:
    def __init__(self, name: str, manifold, feasibility_fn=None):
        self.name = name
        self._manifold = manifold
        self._feasibility_fn = feasibility_fn

    def sample_lambdas(self):
        return [0.0]

    def manifold(self, lam: float):
        _ = lam
        return self._manifold

    def lambda_distance(self, lam_a, lam_b):
        return abs(float(lam_a) - float(lam_b))

    def transition_feasibility(self, lam, point, goal_point=None, metadata=None):
        if self._feasibility_fn is None:
            return True
        return bool(
            self._feasibility_fn(
                float(lam),
                np.asarray(point, dtype=float),
                None if goal_point is None else np.asarray(goal_point, dtype=float),
                metadata,
            )
        )


def test_transition_generator_filters_candidates_with_semantic_feasibility():
    circle_a = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0, name="circle_a")
    circle_b = CircleManifold(center=np.array([2.0, 0.0]), radius=2.0, name="circle_b")

    fam_a = SingleLeafFamily("circle_a_family", circle_a)
    fam_b = SingleLeafFamily(
        "circle_b_family",
        circle_b,
        feasibility_fn=lambda lam, point, goal, metadata: float(point[1]) <= 0.0,
    )

    generator = TransitionGenerator(
        seed_points_fn=lambda *_: [np.array([1.0, 1.8]), np.array([1.0, -1.8])],
        project_newton=project_newton,
    )

    result = generator.generate_transitions(
        source_family=fam_a,
        source_lam=0.0,
        target_family=fam_b,
        target_lam=0.0,
        goal_point=np.array([1.0, 1.8]),
    )

    assert result.success
    assert all(float(c.transition_point[1]) <= 0.0 for c in result.candidates)


def test_build_transition_feasibility_fn_adapts_composed_model():
    model = FamilyPairGateFeasibilityModel(
        inner=AllFeasibilityModel(
            [
                CoordinateRangeFeasibility(axis=0, min_value=-1.0, max_value=1.0),
                CoordinateRangeFeasibility(axis=1, min_value=-0.5, max_value=0.5),
            ]
        ),
        family_names=["switch"],
    )
    callback = build_transition_feasibility_fn(model)

    class DummyFamily:
        def __init__(self, name: str):
            self.name = name

    assert callback(
        DummyFamily("left"),
        0.0,
        DummyFamily("switch"),
        0.0,
        np.array([0.25, 0.25]),
        None,
        None,
    )
    assert not callback(
        DummyFamily("left"),
        0.0,
        DummyFamily("switch"),
        0.0,
        np.array([1.5, 0.25]),
        None,
        None,
    )
