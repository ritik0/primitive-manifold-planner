import numpy as np

from primitive_manifold_planner.manifolds import CircleManifold, SphereManifold, PlaneManifold
from primitive_manifold_planner.projection import project_newton


def test_newton_projection_converges_on_circle_from_outside():
    circle = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0)
    x0 = np.array([3.0, 1.0])

    result = project_newton(circle, x0)

    assert result.success
    assert result.converged
    assert circle.is_valid(result.x_projected, tol=1e-8)
    assert result.residual_norm <= 1e-8


def test_newton_projection_converges_on_circle_from_inside():
    circle = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0)
    x0 = np.array([0.5, 0.5])

    result = project_newton(circle, x0)

    assert result.success
    assert result.converged
    assert circle.is_valid(result.x_projected, tol=1e-8)
    assert result.residual_norm <= 1e-8


def test_newton_projection_matches_analytic_circle_projection():
    circle = CircleManifold(center=np.array([1.0, -2.0]), radius=3.0)
    x0 = np.array([5.0, 4.0])

    result = project_newton(circle, x0)
    x_analytic = circle.closest_point_analytic(x0)

    assert result.success
    assert np.allclose(result.x_projected, x_analytic, atol=1e-6)


def test_newton_projection_converges_on_sphere():
    sphere = SphereManifold(center=np.array([0.0, 0.0, 0.0]), radius=2.0)
    x0 = np.array([3.0, 1.0, -2.0])

    result = project_newton(sphere, x0)

    assert result.success
    assert result.converged
    assert sphere.is_valid(result.x_projected, tol=1e-8)
    assert result.residual_norm <= 1e-8


def test_newton_projection_matches_analytic_sphere_projection():
    sphere = SphereManifold(center=np.array([1.0, -1.0, 0.5]), radius=4.0)
    x0 = np.array([7.0, 2.0, 5.0])

    result = project_newton(sphere, x0)
    x_analytic = sphere.closest_point_analytic(x0)

    assert result.success
    assert np.allclose(result.x_projected, x_analytic, atol=1e-6)


def test_newton_projection_converges_on_plane():
    plane = PlaneManifold(
        point=np.array([0.0, 0.0, 1.0]),
        normal=np.array([0.0, 0.0, 1.0]),
    )
    x0 = np.array([2.0, -3.0, 5.0])

    result = project_newton(plane, x0)

    assert result.success
    assert result.converged
    assert plane.is_valid(result.x_projected, tol=1e-8)
    assert result.residual_norm <= 1e-8


def test_newton_projection_matches_analytic_plane_projection():
    plane = PlaneManifold(
        point=np.array([1.0, 2.0, 3.0]),
        normal=np.array([0.0, 1.0, 0.0]),
    )
    x0 = np.array([10.0, -4.0, 8.0])

    result = project_newton(plane, x0)
    x_analytic = plane.closest_point_analytic(x0)

    assert result.success
    assert np.allclose(result.x_projected, x_analytic, atol=1e-8)


def test_newton_projection_rejects_invalid_damping():
    circle = CircleManifold(center=np.array([0.0, 0.0]), radius=1.0)

    try:
        project_newton(circle, np.array([2.0, 0.0]), damping=0.0)
        assert False, "Expected ValueError for invalid damping"
    except ValueError:
        pass