import numpy as np

from primitive_manifold_planner.manifolds import CircleManifold
from primitive_manifold_planner.planning import constrained_interpolate


def test_constrained_interpolate_on_circle_reaches_goal():
    circle = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0)

    x_start = np.array([2.0, 0.0])
    x_goal = np.array([0.0, 2.0])

    result = constrained_interpolate(
        manifold=circle,
        x_start=x_start,
        x_goal=x_goal,
        step_size=0.2,
        goal_tol=5e-2,
        max_iters=500,
    )

    assert result.success
    assert result.reached_goal
    assert result.path.shape[1] == 2

    for x in result.path:
        assert circle.is_valid(x, tol=1e-6)

    assert np.linalg.norm(result.path[-1] - x_goal) <= 5e-2


def test_constrained_interpolate_rejects_invalid_start():
    circle = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0)

    x_start = np.array([0.0, 0.0])  # not on circle
    x_goal = np.array([0.0, 2.0])

    result = constrained_interpolate(circle, x_start, x_goal)

    assert not result.success
    assert not result.reached_goal
    assert "Start point is not on the manifold" in result.message


def test_constrained_interpolate_rejects_invalid_goal():
    circle = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0)

    x_start = np.array([2.0, 0.0])
    x_goal = np.array([0.0, 0.0])  # not on circle

    result = constrained_interpolate(circle, x_start, x_goal)

    assert not result.success
    assert not result.reached_goal
    assert "Goal point is not on the manifold" in result.message


def test_constrained_interpolate_path_stays_on_circle():
    circle = CircleManifold(center=np.array([1.0, -1.0]), radius=3.0)

    x_start = np.array([4.0, -1.0])
    x_goal = np.array([1.0, 2.0])

    result = constrained_interpolate(
        manifold=circle,
        x_start=x_start,
        x_goal=x_goal,
        step_size=0.15,
        goal_tol=5e-2,
        max_iters=500,
    )

    assert result.success

    residuals = [abs(circle.residual(x)[0]) for x in result.path]
    assert max(residuals) <= 1e-6