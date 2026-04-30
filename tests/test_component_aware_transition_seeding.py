import numpy as np

from primitive_manifold_planner.manifolds import CircleManifold
from primitive_manifold_planner.planners.component_discovery import (
    ComponentModelRegistry,
    DiscoveredComponent,
    LeafComponentDiscoveryResult,
)
from primitive_manifold_planner.planners.transition_manager import AdaptiveTransitionSeedPolicy
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


def test_component_registry_exposes_discovered_transition_seed_anchors():
    registry = ComponentModelRegistry()
    discovery = LeafComponentDiscoveryResult(
        success=True,
        components=[
            DiscoveredComponent(
                component_id=0,
                samples=np.asarray([[1.0, 0.0], [0.8, 0.2]]),
                representative=np.asarray([1.0, 0.0]),
            ),
            DiscoveredComponent(
                component_id=1,
                samples=np.asarray([[-1.0, 0.0], [-0.8, -0.2]]),
                representative=np.asarray([-1.0, 0.0]),
            ),
        ],
        sample_points=np.asarray([[1.0, 0.0], [0.8, 0.2], [-1.0, 0.0], [-0.8, -0.2]]),
        labels=np.asarray([0, 0, 1, 1]),
    )
    registry.register_discovered_components("leaf_family", 0.0, discovery)

    anchors = registry.transition_seed_anchors(
        family_name="leaf_family",
        lam=0.0,
        goal_point=np.asarray([0.9, 0.1]),
    )

    assert len(anchors) >= 4
    assert any(np.linalg.norm(anchor - np.array([1.0, 0.0])) < 1e-8 for anchor in anchors)


def test_adaptive_seed_policy_uses_component_anchor_fn():
    circle_a = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0, name="circle_a")
    circle_b = CircleManifold(center=np.array([2.0, 0.0]), radius=2.0, name="circle_b")

    fam_a = SingleLeafFamily("circle_a_family", circle_a)
    fam_b = SingleLeafFamily("circle_b_family", circle_b)

    def component_anchor_fn(family_name, lam, goal_point=None):
        _ = lam
        if family_name == "circle_a_family":
            return [np.array([1.0, 1.0])]
        if goal_point is not None:
            return [np.asarray(goal_point, dtype=float)]
        return []

    policy = AdaptiveTransitionSeedPolicy(
        project_newton=project_newton,
        base_seed_points_fn=None,
        component_anchor_fn=component_anchor_fn,
        rng=np.random.default_rng(3),
    )

    seeds = policy.generate(
        source_family=fam_a,
        source_lam=0.0,
        target_family=fam_b,
        target_lam=0.0,
        goal_point=np.array([1.0, 1.5]),
    )

    assert any(np.linalg.norm(seed - np.array([1.0, 1.0])) < 1e-8 for seed in seeds)
