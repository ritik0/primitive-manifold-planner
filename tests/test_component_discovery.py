import numpy as np

from primitive_manifold_planner.manifolds import LineManifold
from primitive_manifold_planner.planners.component_discovery import (
    ComponentModelRegistry,
    DiscoveredComponent,
    LeafComponentDiscoveryResult,
    StaticComponentModel,
    assign_point_to_discovered_component,
    build_component_model_registry,
)
from primitive_manifold_planner.planners.multimodal_component_planner import (
    MultimodalComponentPlanner,
)


class DummyFamily:
    def __init__(self, name: str, lambdas: list[float], manifold_builder):
        self.name = name
        self._lambdas = [float(v) for v in lambdas]
        self._manifold_builder = manifold_builder

    def sample_lambdas(self):
        return list(self._lambdas)

    def manifold(self, lam: float):
        return self._manifold_builder(float(lam))


def test_assign_point_to_discovered_component_uses_samples_not_only_representatives():
    discovery = LeafComponentDiscoveryResult(
        success=True,
        components=[
            DiscoveredComponent(
                component_id=0,
                samples=np.asarray([[0.0], [0.1]]),
                representative=np.asarray([100.0]),
            ),
            DiscoveredComponent(
                component_id=1,
                samples=np.asarray([[10.0]]),
                representative=np.asarray([10.0]),
            ),
        ],
        sample_points=np.asarray([[0.0], [0.1], [10.0]]),
        labels=np.asarray([0, 0, 1]),
    )

    cid = assign_point_to_discovered_component(
        q=np.asarray([0.05]),
        discovery=discovery,
    )

    assert cid == 0


def test_build_component_model_registry_mixes_discovered_and_static_models():
    moving_family = DummyFamily(
        name="line_family",
        lambdas=[0.0],
        manifold_builder=lambda lam: LineManifold(
            point=np.array([0.0, lam]),
            normal=np.array([0.0, 1.0]),
            name=f"line_{lam}",
        ),
    )
    static_family = DummyFamily(
        name="connector_family",
        lambdas=[0.0],
        manifold_builder=lambda lam: LineManifold(
            point=np.array([lam, 0.0]),
            normal=np.array([1.0, 0.0]),
            name=f"connector_{lam}",
        ),
    )

    def seed_samples_for_leaf(fam, lam):
        if fam.name == "line_family":
            return np.asarray([[-1.0, lam], [1.0, lam]])
        return np.asarray([[lam, 0.0]])

    def should_discover(fam, lam):
        _ = lam
        return fam.name == "line_family"

    def static_model_for_leaf(fam, lam):
        _ = lam
        if fam.name == "connector_family":
            return StaticComponentModel(ids=["connector"])
        return None

    registry, discoveries = build_component_model_registry(
        families=[moving_family, static_family],
        seed_samples_for_leaf_fn=seed_samples_for_leaf,
        should_discover_fn=should_discover,
        static_model_fn=static_model_for_leaf,
        local_planner_name="projection",
        local_planner_kwargs=dict(goal_tol=1e-6, max_iters=200),
        step_size=0.2,
        neighbor_radius=2.5,
    )

    assert ("line_family", 0.0) in discoveries
    assert registry.component_ids_for_family(moving_family, 0.0) == ["0"]
    assert registry.compatible_components_for_leaf(
        moving_family,
        0.0,
        np.array([0.2, 0.0]),
    ) == ["0"]
    assert registry.infer_component("connector_family", 0.0, np.array([0.0, 0.0])) == "connector"


def test_planner_can_infer_components_from_registry():
    registry = ComponentModelRegistry()
    registry.register_static_components("family_a", 0.0, ["left", "right"], lambda q: ["left"])

    planner = MultimodalComponentPlanner(
        families=[],
        project_newton=lambda *args, **kwargs: None,
        seed_points_fn=lambda *args, **kwargs: [],
        component_model_registry=registry,
    )

    assert planner.infer_component("family_a", 0.0, np.array([0.0, 0.0])) == "left"
