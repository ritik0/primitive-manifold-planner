import numpy as np

from primitive_manifold_planner.manifolds import CircleManifold
from primitive_manifold_planner.planners.transition_manager import TransitionGenerator
from primitive_manifold_planner.projection import project_newton


class SingleLeafFamily:
    def __init__(self, name: str, manifold):
        self.name = name
        self._manifold = manifold

    def sample_lambdas(self):
        return [0.0]

    def manifold(self, lam: float):
        _ = lam
        return self._manifold

    def lambda_distance(self, lam_a, lam_b):
        return abs(float(lam_a) - float(lam_b))


def test_transition_generator_caches_raw_exact_candidates():
    calls = {"count": 0}

    circle_a = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0, name="circle_a")
    circle_b = CircleManifold(center=np.array([2.0, 0.0]), radius=2.0, name="circle_b")

    fam_a = SingleLeafFamily("circle_a_family", circle_a)
    fam_b = SingleLeafFamily("circle_b_family", circle_b)

    def seed_points_fn(_fam_a, _lam_a, _fam_b, _lam_b):
        calls["count"] += 1
        return [
            np.array([1.0, 1.8]),
            np.array([1.0, -1.8]),
            np.array([1.2, 1.6]),
        ]

    generator = TransitionGenerator(
        seed_points_fn=seed_points_fn,
        project_newton=project_newton,
    )

    upper_goal = np.array([1.0, 2.0])
    lower_goal = np.array([1.0, -2.0])

    upper = generator.generate_transitions(
        source_family=fam_a,
        source_lam=0.0,
        target_family=fam_b,
        target_lam=0.0,
        goal_point=upper_goal,
    )
    lower = generator.generate_transitions(
        source_family=fam_a,
        source_lam=0.0,
        target_family=fam_b,
        target_lam=0.0,
        goal_point=lower_goal,
    )

    assert upper.success
    assert lower.success
    assert calls["count"] == 1
    assert generator.cache_misses == 1
    assert generator.cache_hits == 1
    assert upper.from_cache is False
    assert lower.from_cache is True
    assert upper.seed_count == 3
    assert lower.seed_count == 3


def test_transition_generator_reranks_cached_candidates_by_goal():
    circle_a = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0, name="circle_a")
    circle_b = CircleManifold(center=np.array([2.0, 0.0]), radius=2.0, name="circle_b")

    fam_a = SingleLeafFamily("circle_a_family", circle_a)
    fam_b = SingleLeafFamily("circle_b_family", circle_b)

    def seed_points_fn(_fam_a, _lam_a, _fam_b, _lam_b):
        return [
            np.array([1.0, 1.8]),
            np.array([1.0, -1.8]),
        ]

    generator = TransitionGenerator(
        seed_points_fn=seed_points_fn,
        project_newton=project_newton,
    )

    upper = generator.generate_transitions(
        source_family=fam_a,
        source_lam=0.0,
        target_family=fam_b,
        target_lam=0.0,
        goal_point=np.array([1.0, 2.0]),
    )
    lower = generator.generate_transitions(
        source_family=fam_a,
        source_lam=0.0,
        target_family=fam_b,
        target_lam=0.0,
        goal_point=np.array([1.0, -2.0]),
    )

    assert len(upper.candidates) >= 2
    assert len(lower.candidates) >= 2
    assert upper.candidates[0].transition_point[1] > 0.0
    assert lower.candidates[0].transition_point[1] < 0.0
    assert "seed_count" in upper.candidates[0].metadata
    assert "raw_candidate_index" in upper.candidates[0].metadata
