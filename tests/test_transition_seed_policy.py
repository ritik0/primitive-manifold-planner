import numpy as np

from primitive_manifold_planner.manifolds import CircleManifold
from primitive_manifold_planner.planners.transition_manager import (
    AdaptiveTransitionSeedPolicy,
    TransitionGenerator,
)
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


def test_adaptive_transition_seed_policy_generates_goal_projection_bridge_seeds():
    circle_a = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0, name="circle_a")
    circle_b = CircleManifold(center=np.array([2.0, 0.0]), radius=2.0, name="circle_b")

    fam_a = SingleLeafFamily("circle_a_family", circle_a)
    fam_b = SingleLeafFamily("circle_b_family", circle_b)

    policy = AdaptiveTransitionSeedPolicy(
        project_newton=project_newton,
        base_seed_points_fn=None,
        rng=np.random.default_rng(1),
    )

    seeds = policy.generate(
        source_family=fam_a,
        source_lam=0.0,
        target_family=fam_b,
        target_lam=0.0,
        goal_point=np.array([1.0, 1.5]),
    )

    assert len(seeds) > 5
    assert any(np.linalg.norm(seed - np.array([1.0, 1.5])) < 1e-8 for seed in seeds)


def test_transition_generator_can_work_without_handcrafted_base_seed_function():
    circle_a = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0, name="circle_a")
    circle_b = CircleManifold(center=np.array([2.0, 0.0]), radius=2.0, name="circle_b")

    fam_a = SingleLeafFamily("circle_a_family", circle_a)
    fam_b = SingleLeafFamily("circle_b_family", circle_b)

    generator = TransitionGenerator(
        seed_points_fn=None,
        project_newton=project_newton,
        seed_policy=AdaptiveTransitionSeedPolicy(
            project_newton=project_newton,
            base_seed_points_fn=None,
            rng=np.random.default_rng(2),
        ),
    )

    result = generator.generate_transitions(
        source_family=fam_a,
        source_lam=0.0,
        target_family=fam_b,
        target_lam=0.0,
        goal_point=np.array([1.0, 1.5]),
    )

    assert result.success
    assert len(result.candidates) >= 1
    assert "seed_count" in result.candidates[0].metadata
