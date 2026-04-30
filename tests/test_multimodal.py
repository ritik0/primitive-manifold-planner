import numpy as np

from primitive_manifold_planner.manifolds import CircleManifold
from primitive_manifold_planner.planning import build_mode_graph, plan_multimodal_route


def test_plan_multimodal_route_same_mode_circle():
    manifolds = {
        "circle": CircleManifold(center=np.array([0.0, 0.0]), radius=2.0),
    }

    rng = np.random.default_rng(0)
    graph, _ = build_mode_graph(
        manifolds=manifolds,
        bounds_min=np.array([-3.0, -3.0]),
        bounds_max=np.array([3.0, 3.0]),
        num_seeds=10,
        tol=1e-8,
        rng=rng,
    )

    start = np.array([2.0, 0.0])
    goal = np.array([0.0, 2.0])

    result = plan_multimodal_route(
        graph=graph,
        start_mode="circle",
        goal_mode="circle",
        start_point=start,
        goal_point=goal,
        step_size=0.2,
        goal_tol=5e-2,
        max_iters=500,
    )

    assert result.success
    assert result.route is not None
    assert result.route.mode_sequence == ["circle"]
    assert len(result.segment_plans) == 1
    assert result.full_path is not None
    assert result.full_path.shape[1] == 2


def test_plan_multimodal_route_fails_when_no_mode_route():
    manifolds = {
        "circle_a": CircleManifold(center=np.array([0.0, 0.0]), radius=1.0),
        "circle_b": CircleManifold(center=np.array([5.0, 5.0]), radius=1.0),
    }

    rng = np.random.default_rng(1)
    graph, _ = build_mode_graph(
        manifolds=manifolds,
        bounds_min=np.array([-2.0, -2.0]),
        bounds_max=np.array([7.0, 7.0]),
        num_seeds=20,
        tol=1e-8,
        rng=rng,
    )

    result = plan_multimodal_route(
        graph=graph,
        start_mode="circle_a",
        goal_mode="circle_b",
        start_point=np.array([1.0, 0.0]),
        goal_point=np.array([6.0, 5.0]),
        step_size=0.2,
        goal_tol=5e-2,
        max_iters=500,
    )

    assert not result.success
    assert result.route is None
    assert result.full_path is None


def test_plan_multimodal_route_two_intersecting_circles():
    circle_a = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0, name="circle_a")
    circle_b = CircleManifold(center=np.array([2.0, 0.0]), radius=2.0, name="circle_b")

    manifolds = {
        "circle_a": circle_a,
        "circle_b": circle_b,
    }

    rng = np.random.default_rng(42)
    graph, _ = build_mode_graph(
        manifolds=manifolds,
        bounds_min=np.array([-3.0, -3.0]),
        bounds_max=np.array([5.0, 3.0]),
        num_seeds=100,
        tol=1e-8,
        rng=rng,
    )

    start = np.array([-2.0, 0.0])  # on circle_a
    goal = np.array([4.0, 0.0])    # on circle_b

    result = plan_multimodal_route(
        graph=graph,
        start_mode="circle_a",
        goal_mode="circle_b",
        start_point=start,
        goal_point=goal,
        step_size=0.15,
        goal_tol=5e-2,
        max_iters=800,
    )

    assert result.success
    assert result.route is not None
    assert result.route.mode_sequence == ["circle_a", "circle_b"]
    assert len(result.route.transition_steps) == 1
    assert len(result.segment_plans) == 2
    assert result.full_path is not None
    assert result.full_path.shape[1] == 2

    transition = result.route.transition_steps[0].transition_point
    assert circle_a.is_valid(transition, tol=1e-6)
    assert circle_b.is_valid(transition, tol=1e-6)

    for x in result.segment_plans[0].path:
        assert circle_a.is_valid(x, tol=1e-5)

    for x in result.segment_plans[1].path:
        assert circle_b.is_valid(x, tol=1e-5)


def test_context_aware_transition_selection_changes_with_goal():
    circle_a = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0, name="circle_a")
    circle_b = CircleManifold(center=np.array([2.0, 0.0]), radius=2.0, name="circle_b")

    manifolds = {
        "circle_a": circle_a,
        "circle_b": circle_b,
    }

    rng = np.random.default_rng(42)
    graph, _ = build_mode_graph(
        manifolds=manifolds,
        bounds_min=np.array([-3.0, -3.0]),
        bounds_max=np.array([5.0, 3.0]),
        num_seeds=100,
        tol=1e-8,
        rng=rng,
    )

    start = np.array([-2.0, 0.0])

    result_upper = plan_multimodal_route(
        graph=graph,
        start_mode="circle_a",
        goal_mode="circle_b",
        start_point=start,
        goal_point=np.array([1.0, 1.73205081]),
        step_size=0.15,
        goal_tol=5e-2,
        max_iters=800,
    )

    result_lower = plan_multimodal_route(
        graph=graph,
        start_mode="circle_a",
        goal_mode="circle_b",
        start_point=start,
        goal_point=np.array([1.0, -1.73205081]),
        step_size=0.15,
        goal_tol=5e-2,
        max_iters=800,
    )

    assert result_upper.success
    assert result_lower.success

    upper_tp = result_upper.route.transition_steps[0].transition_point
    lower_tp = result_lower.route.transition_steps[0].transition_point

    assert upper_tp[1] > 0.0
    assert lower_tp[1] < 0.0

def test_plan_multimodal_route_parallel_lines_with_vertical_connector():
    from primitive_manifold_planner.manifolds import LineManifold

    # Horizontal leaves y = -1, 0, 1
    leaf_m1 = LineManifold(point=np.array([0.0, -1.0]), normal=np.array([0.0, 1.0]), name="leaf_-1")
    leaf_0 = LineManifold(point=np.array([0.0, 0.0]), normal=np.array([0.0, 1.0]), name="leaf_0")
    leaf_p1 = LineManifold(point=np.array([0.0, 1.0]), normal=np.array([0.0, 1.0]), name="leaf_1")

    # Vertical connector x = 0
    connector = LineManifold(point=np.array([0.0, 0.0]), normal=np.array([1.0, 0.0]), name="connector")

    manifolds = {
        "leaf_-1": leaf_m1,
        "leaf_0": leaf_0,
        "leaf_1": leaf_p1,
        "connector": connector,
    }

    rng = np.random.default_rng(123)
    graph, _ = build_mode_graph(
        manifolds=manifolds,
        bounds_min=np.array([-3.0, -3.0]),
        bounds_max=np.array([3.0, 3.0]),
        num_seeds=80,
        tol=1e-8,
        rng=rng,
    )

    start = np.array([-2.0, -1.0])  # on leaf_-1
    goal = np.array([2.0, 1.0])     # on leaf_1

    result = plan_multimodal_route(
        graph=graph,
        start_mode="leaf_-1",
        goal_mode="leaf_1",
        start_point=start,
        goal_point=goal,
        step_size=0.15,
        goal_tol=5e-2,
        max_iters=800,
    )

    assert result.success
    assert result.route is not None
    assert result.route.mode_sequence == ["leaf_-1", "connector", "leaf_1"]
    assert len(result.route.transition_steps) == 2
    assert len(result.segment_plans) == 3
    assert result.full_path is not None
    assert result.full_path.shape[1] == 2

    # Check segment validity
    for x in result.segment_plans[0].path:
        assert leaf_m1.is_valid(x, tol=1e-5)

    for x in result.segment_plans[1].path:
        assert connector.is_valid(x, tol=1e-5)

    for x in result.segment_plans[2].path:
        assert leaf_p1.is_valid(x, tol=1e-5)

def test_plan_multimodal_route_concentric_circles_with_radial_connector():
    from primitive_manifold_planner.manifolds import CircleManifold, LineManifold

    circle_1 = CircleManifold(center=np.array([0.0, 0.0]), radius=1.0, name="circle_1")
    circle_2 = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0, name="circle_2")
    circle_3 = CircleManifold(center=np.array([0.0, 0.0]), radius=3.0, name="circle_3")

    # Radial connector: y = 0
    connector = LineManifold(
        point=np.array([0.0, 0.0]),
        normal=np.array([0.0, 1.0]),
        name="radial_connector",
    )

    manifolds = {
        "circle_1": circle_1,
        "circle_2": circle_2,
        "circle_3": circle_3,
        "radial_connector": connector,
    }

    rng = np.random.default_rng(321)
    graph, _ = build_mode_graph(
        manifolds=manifolds,
        bounds_min=np.array([-4.0, -4.0]),
        bounds_max=np.array([4.0, 4.0]),
        num_seeds=120,
        tol=1e-8,
        rng=rng,
    )

    start = np.array([1.0, 0.0])  # on circle_1
    goal = np.array([3.0, 0.0])   # on circle_3

    result = plan_multimodal_route(
        graph=graph,
        start_mode="circle_1",
        goal_mode="circle_3",
        start_point=start,
        goal_point=goal,
        step_size=0.1,
        goal_tol=5e-2,
        max_iters=800,
    )

    assert result.success
    assert result.route is not None
    assert result.route.mode_sequence == ["circle_1", "radial_connector", "circle_3"]
    assert len(result.route.transition_steps) == 2
    assert len(result.segment_plans) == 3
    assert result.full_path is not None
    assert result.full_path.shape[1] == 2

    for x in result.segment_plans[0].path:
        assert circle_1.is_valid(x, tol=1e-5)

    for x in result.segment_plans[1].path:
        assert connector.is_valid(x, tol=1e-5)

    for x in result.segment_plans[2].path:
        assert circle_3.is_valid(x, tol=1e-5)

def test_plan_multimodal_route_parallel_lines_with_two_connectors_left_preferred():
    from primitive_manifold_planner.manifolds import LineManifold

    leaf_m1 = LineManifold(point=np.array([0.0, -1.0]), normal=np.array([0.0, 1.0]), name="leaf_-1")
    leaf_0 = LineManifold(point=np.array([0.0, 0.0]), normal=np.array([0.0, 1.0]), name="leaf_0")
    leaf_p1 = LineManifold(point=np.array([0.0, 1.0]), normal=np.array([0.0, 1.0]), name="leaf_1")

    connector_left = LineManifold(
        point=np.array([-1.0, 0.0]),
        normal=np.array([1.0, 0.0]),
        name="connector_left",
    )
    connector_right = LineManifold(
        point=np.array([1.0, 0.0]),
        normal=np.array([1.0, 0.0]),
        name="connector_right",
    )

    manifolds = {
        "leaf_-1": leaf_m1,
        "leaf_0": leaf_0,
        "leaf_1": leaf_p1,
        "connector_left": connector_left,
        "connector_right": connector_right,
    }

    rng = np.random.default_rng(123)
    graph, _ = build_mode_graph(
        manifolds=manifolds,
        bounds_min=np.array([-3.0, -3.0]),
        bounds_max=np.array([3.0, 3.0]),
        num_seeds=100,
        tol=1e-8,
        rng=rng,
    )

    start = np.array([-2.5, -1.0])
    goal = np.array([-2.0, 1.0])

    result = plan_multimodal_route(
        graph=graph,
        start_mode="leaf_-1",
        goal_mode="leaf_1",
        start_point=start,
        goal_point=goal,
        step_size=0.15,
        goal_tol=5e-2,
        max_iters=800,
    )

    assert result.success
    assert result.route is not None
    assert result.route.mode_sequence == ["leaf_-1", "connector_left", "leaf_1"]


def test_plan_multimodal_route_parallel_lines_with_two_connectors_right_preferred():
    from primitive_manifold_planner.manifolds import LineManifold

    leaf_m1 = LineManifold(point=np.array([0.0, -1.0]), normal=np.array([0.0, 1.0]), name="leaf_-1")
    leaf_0 = LineManifold(point=np.array([0.0, 0.0]), normal=np.array([0.0, 1.0]), name="leaf_0")
    leaf_p1 = LineManifold(point=np.array([0.0, 1.0]), normal=np.array([0.0, 1.0]), name="leaf_1")

    connector_left = LineManifold(
        point=np.array([-1.0, 0.0]),
        normal=np.array([1.0, 0.0]),
        name="connector_left",
    )
    connector_right = LineManifold(
        point=np.array([1.0, 0.0]),
        normal=np.array([1.0, 0.0]),
        name="connector_right",
    )

    manifolds = {
        "leaf_-1": leaf_m1,
        "leaf_0": leaf_0,
        "leaf_1": leaf_p1,
        "connector_left": connector_left,
        "connector_right": connector_right,
    }

    rng = np.random.default_rng(123)
    graph, _ = build_mode_graph(
        manifolds=manifolds,
        bounds_min=np.array([-3.0, -3.0]),
        bounds_max=np.array([3.0, 3.0]),
        num_seeds=100,
        tol=1e-8,
        rng=rng,
    )

    start = np.array([2.5, -1.0])
    goal = np.array([2.0, 1.0])

    result = plan_multimodal_route(
        graph=graph,
        start_mode="leaf_-1",
        goal_mode="leaf_1",
        start_point=start,
        goal_point=goal,
        step_size=0.15,
        goal_tol=5e-2,
        max_iters=800,
    )

    assert result.success
    assert result.route is not None
    assert result.route.mode_sequence == ["leaf_-1", "connector_right", "leaf_1"]