import numpy as np

from primitive_manifold_planner.manifolds import CircleManifold
from primitive_manifold_planner.planners.transition_manager import TransitionGenerator
from primitive_manifold_planner.projection import project_newton


class SingleLeafFamily:
    def __init__(self, name: str, manifold, admissibility_cost_fn=None):
        self.name = name
        self._manifold = manifold
        self._admissibility_cost_fn = admissibility_cost_fn

    def sample_lambdas(self):
        return [0.0]

    def manifold(self, lam: float):
        _ = lam
        return self._manifold

    def lambda_distance(self, lam_a, lam_b):
        return abs(float(lam_a) - float(lam_b))

    def transition_admissibility_cost(self, lam, point, goal_point=None, metadata=None):
        if self._admissibility_cost_fn is None:
            return 0.0
        return float(
            self._admissibility_cost_fn(
                float(lam),
                np.asarray(point, dtype=float),
                None if goal_point is None else np.asarray(goal_point, dtype=float),
                metadata,
            )
        )


def test_transition_generator_includes_admissibility_cost_metadata():
    circle_a = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0, name="circle_a")
    circle_b = CircleManifold(center=np.array([2.0, 0.0]), radius=2.0, name="circle_b")

    fam_a = SingleLeafFamily("circle_a_family", circle_a, admissibility_cost_fn=lambda lam, point, goal, meta: 0.25)
    fam_b = SingleLeafFamily("circle_b_family", circle_b, admissibility_cost_fn=lambda lam, point, goal, meta: 0.75)

    generator = TransitionGenerator(
        seed_points_fn=lambda *_: [np.array([1.0, 1.8]), np.array([1.0, -1.8])],
        project_newton=project_newton,
        admissibility_cost_fn=lambda src_f, src_l, dst_f, dst_l, x, goal, meta: 1.5,
    )

    result = generator.generate_transitions(
        source_family=fam_a,
        source_lam=0.0,
        target_family=fam_b,
        target_lam=0.0,
        goal_point=np.array([1.0, 1.8]),
    )

    assert result.success
    candidate = result.candidates[0]
    assert candidate.metadata["source_admissibility_cost"] == 0.25
    assert candidate.metadata["target_admissibility_cost"] == 0.75
    assert candidate.metadata["pair_admissibility_cost"] == 1.5
    assert candidate.metadata["admissibility_cost"] == 2.5
    assert np.isclose(candidate.score, candidate.metadata["base_score"] + 2.5)


def test_transition_admissibility_can_reorder_geometry_preferred_candidates():
    circle_a = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0, name="circle_a")
    circle_b = CircleManifold(center=np.array([2.0, 0.0]), radius=2.0, name="circle_b")

    fam_a = SingleLeafFamily("circle_a_family", circle_a)
    fam_b = SingleLeafFamily("circle_b_family", circle_b)

    generator = TransitionGenerator(
        seed_points_fn=lambda *_: [np.array([1.0, 1.8]), np.array([1.0, -1.8])],
        project_newton=project_newton,
        admissibility_cost_fn=(
            lambda src_f, src_l, dst_f, dst_l, x, goal, meta: 10.0 if float(x[1]) > 0.0 else 0.0
        ),
    )

    result = generator.generate_transitions(
        source_family=fam_a,
        source_lam=0.0,
        target_family=fam_b,
        target_lam=0.0,
        goal_point=np.array([1.0, 1.9]),
    )

    assert result.success
    assert len(result.candidates) >= 2
    assert result.candidates[0].transition_point[1] < 0.0
    assert result.candidates[0].metadata["pair_admissibility_cost"] == 0.0
    assert result.candidates[-1].transition_point[1] > 0.0
    assert result.candidates[-1].metadata["pair_admissibility_cost"] == 10.0
