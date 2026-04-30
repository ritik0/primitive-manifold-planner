import numpy as np

from primitive_manifold_planner.manifolds import CircleManifold, SphereManifold
from primitive_manifold_planner.planning import plan_constrained_rrt


def test_constrained_rrt_circle_finds_path():
    circle = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0)

    start = np.array([2.0, 0.0])
    goal = np.array([0.0, 2.0])

    rng = np.random.default_rng(42)
    result = plan_constrained_rrt(
        manifold=circle,
        start_point=start,
        goal_point=goal,
        bounds_min=np.array([-3.0, -3.0]),
        bounds_max=np.array([3.0, 3.0]),
        max_iters=300,
        step_size=0.2,
        goal_tol=5e-2,
        goal_sample_rate=0.2,
        rng=rng,
    )

    assert result.success
    assert result.path is not None
    assert result.tree_points.shape[1] == 2

    for x in result.path:
        assert circle.is_valid(x, tol=1e-5)


def test_constrained_rrt_rejects_invalid_start():
    circle = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0)

    result = plan_constrained_rrt(
        manifold=circle,
        start_point=np.array([0.0, 0.0]),  # invalid
        goal_point=np.array([0.0, 2.0]),
        bounds_min=np.array([-3.0, -3.0]),
        bounds_max=np.array([3.0, 3.0]),
    )

    assert not result.success
    assert result.path is None


def test_constrained_rrt_sphere_finds_path():
    sphere = SphereManifold(center=np.array([0.0, 0.0, 0.0]), radius=2.0)

    start = np.array([2.0, 0.0, 0.0])
    goal = np.array([0.0, 2.0, 0.0])

    rng = np.random.default_rng(7)
    result = plan_constrained_rrt(
        manifold=sphere,
        start_point=start,
        goal_point=goal,
        bounds_min=np.array([-3.0, -3.0, -3.0]),
        bounds_max=np.array([3.0, 3.0, 3.0]),
        max_iters=500,
        step_size=0.25,
        goal_tol=1e-1,
        goal_sample_rate=0.2,
        rng=rng,
    )

    assert result.success
    assert result.path is not None
    assert result.tree_points.shape[1] == 3

    for x in result.path:
        assert sphere.is_valid(x, tol=1e-5)


def test_constrained_rrt_sphere_rejects_invalid_goal():
    sphere = SphereManifold(center=np.array([0.0, 0.0, 0.0]), radius=2.0)

    result = plan_constrained_rrt(
        manifold=sphere,
        start_point=np.array([2.0, 0.0, 0.0]),
        goal_point=np.array([0.0, 0.0, 0.0]),  # invalid
        bounds_min=np.array([-3.0, -3.0, -3.0]),
        bounds_max=np.array([3.0, 3.0, 3.0]),
    )

    assert not result.success
    assert result.path is None