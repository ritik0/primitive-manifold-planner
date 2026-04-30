import numpy as np

from primitive_manifold_planner.manifolds import SphereManifold, PlaneManifold, CircleManifold
from primitive_manifold_planner.planning import build_mode_graph


def test_build_mode_graph_connects_intersecting_manifolds():
    manifolds = {
        "sphere": SphereManifold(center=np.array([0.0, 0.0, 0.0]), radius=2.0),
        "plane_z1": PlaneManifold(
            point=np.array([0.0, 0.0, 1.0]),
            normal=np.array([0.0, 0.0, 1.0]),
        ),
    }

    rng = np.random.default_rng(42)
    graph, diagnostics = build_mode_graph(
        manifolds=manifolds,
        bounds_min=np.array([-3.0, -3.0, -3.0]),
        bounds_max=np.array([3.0, 3.0, 3.0]),
        num_seeds=30,
        tol=1e-8,
        rng=rng,
    )

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    edge = graph.get_edge("sphere", "plane_z1")
    assert edge is not None
    assert len(edge.transition_candidates) >= 1
    assert "plane_z1" in graph.neighbors("sphere")
    assert "sphere" in graph.neighbors("plane_z1")
    assert ("sphere", "plane_z1") in diagnostics


def test_build_mode_graph_two_circles_stores_multiple_candidates():
    manifolds = {
        "circle_a": CircleManifold(center=np.array([0.0, 0.0]), radius=2.0),
        "circle_b": CircleManifold(center=np.array([2.0, 0.0]), radius=2.0),
    }

    rng = np.random.default_rng(42)
    graph, diagnostics = build_mode_graph(
        manifolds=manifolds,
        bounds_min=np.array([-3.0, -3.0]),
        bounds_max=np.array([5.0, 3.0]),
        num_seeds=100,
        tol=1e-8,
        rng=rng,
    )

    edge = graph.get_edge("circle_a", "circle_b")
    assert edge is not None
    assert len(edge.transition_candidates) >= 2
    assert diagnostics[("circle_a", "circle_b")].success


def test_build_mode_graph_leaves_disjoint_manifolds_unconnected():
    manifolds = {
        "sphere": SphereManifold(center=np.array([0.0, 0.0, 0.0]), radius=1.0),
        "plane_z3": PlaneManifold(
            point=np.array([0.0, 0.0, 3.0]),
            normal=np.array([0.0, 0.0, 1.0]),
        ),
    }

    rng = np.random.default_rng(7)
    graph, diagnostics = build_mode_graph(
        manifolds=manifolds,
        bounds_min=np.array([-2.0, -2.0, -2.0]),
        bounds_max=np.array([2.0, 2.0, 4.0]),
        num_seeds=20,
        tol=1e-8,
        rng=rng,
    )

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 0
    assert graph.get_edge("sphere", "plane_z3") is None
    assert ("sphere", "plane_z3") in diagnostics


def test_find_mode_sequence_direct_connection():
    manifolds = {
        "sphere": SphereManifold(center=np.array([0.0, 0.0, 0.0]), radius=2.0),
        "plane_z1": PlaneManifold(
            point=np.array([0.0, 0.0, 1.0]),
            normal=np.array([0.0, 0.0, 1.0]),
        ),
    }

    rng = np.random.default_rng(42)
    graph, _ = build_mode_graph(
        manifolds=manifolds,
        bounds_min=np.array([-3.0, -3.0, -3.0]),
        bounds_max=np.array([3.0, 3.0, 3.0]),
        num_seeds=30,
        tol=1e-8,
        rng=rng,
    )

    seq = graph.find_mode_sequence("sphere", "plane_z1")
    assert seq == ["sphere", "plane_z1"]


def test_build_route_direct_connection():
    manifolds = {
        "sphere": SphereManifold(center=np.array([0.0, 0.0, 0.0]), radius=2.0),
        "plane_z1": PlaneManifold(
            point=np.array([0.0, 0.0, 1.0]),
            normal=np.array([0.0, 0.0, 1.0]),
        ),
    }

    rng = np.random.default_rng(42)
    graph, _ = build_mode_graph(
        manifolds=manifolds,
        bounds_min=np.array([-3.0, -3.0, -3.0]),
        bounds_max=np.array([3.0, 3.0, 3.0]),
        num_seeds=30,
        tol=1e-8,
        rng=rng,
    )

    route = graph.build_route("sphere", "plane_z1")
    assert route is not None
    assert route.mode_sequence == ["sphere", "plane_z1"]
    assert len(route.transition_steps) == 1