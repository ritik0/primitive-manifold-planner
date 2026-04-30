import numpy as np

from primitive_manifold_planner.manifolds import (
    CircleManifold,
    LineManifold,
    SphereManifold,
    PlaneManifold,
)


def test_circle_residual_on_manifold():
    circle = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0)
    x = np.array([2.0, 0.0])
    assert abs(circle.residual(x)[0]) < 1e-12


def test_circle_residual_inside_and_outside():
    circle = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0)
    inside = np.array([1.0, 0.0])
    outside = np.array([3.0, 0.0])

    assert circle.residual(inside)[0] < 0.0
    assert circle.residual(outside)[0] > 0.0


def test_circle_jacobian_shape():
    circle = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0)
    j = circle.jacobian(np.array([2.0, 0.0]))
    assert j.shape == (1, 2)


def test_circle_analytic_projection():
    circle = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0)
    x = np.array([3.0, 4.0])

    x_proj = circle.closest_point_analytic(x)

    assert np.isclose(np.linalg.norm(x_proj), 2.0, atol=1e-8)
    assert circle.is_valid(x_proj, tol=1e-8)


def test_circle_tangent_projection_is_orthogonal_to_normal():
    circle = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0)
    x = np.array([2.0, 0.0])
    v = np.array([1.0, 1.0])

    v_tan = circle.project_tangent(x, v)
    normal = circle.jacobian(x).reshape(-1)

    assert np.isclose(np.dot(normal, v_tan), 0.0, atol=1e-8)


def test_line_residual_on_manifold():
    line = LineManifold(
        point=np.array([0.0, 1.0]),
        normal=np.array([0.0, 1.0]),
    )
    x = np.array([2.0, 1.0])
    assert abs(line.residual(x)[0]) < 1e-12


def test_line_residual_above_and_below():
    line = LineManifold(
        point=np.array([0.0, 1.0]),
        normal=np.array([0.0, 1.0]),
    )

    above = np.array([0.0, 3.0])
    below = np.array([0.0, -2.0])

    assert line.residual(above)[0] > 0.0
    assert line.residual(below)[0] < 0.0


def test_line_jacobian_shape():
    line = LineManifold(
        point=np.array([1.0, 2.0]),
        normal=np.array([1.0, 0.0]),
    )
    j = line.jacobian(np.array([1.0, 5.0]))
    assert j.shape == (1, 2)


def test_line_analytic_projection():
    line = LineManifold(
        point=np.array([0.0, 1.0]),
        normal=np.array([0.0, 1.0]),
    )
    x = np.array([4.0, 5.0])

    x_proj = line.closest_point_analytic(x)

    assert np.allclose(x_proj, np.array([4.0, 1.0]), atol=1e-8)
    assert line.is_valid(x_proj, tol=1e-8)


def test_line_tangent_projection_is_orthogonal_to_normal():
    line = LineManifold(
        point=np.array([0.0, 0.0]),
        normal=np.array([0.0, 1.0]),
    )
    x = np.array([2.0, 0.0])
    v = np.array([3.0, -1.0])

    v_tan = line.project_tangent(x, v)
    normal = line.jacobian(x).reshape(-1)

    assert np.isclose(np.dot(normal, v_tan), 0.0, atol=1e-8)


def test_sphere_residual_on_manifold():
    sphere = SphereManifold(center=np.array([0.0, 0.0, 0.0]), radius=3.0)
    x = np.array([3.0, 0.0, 0.0])
    assert abs(sphere.residual(x)[0]) < 1e-12


def test_sphere_residual_inside_and_outside():
    sphere = SphereManifold(center=np.array([0.0, 0.0, 0.0]), radius=3.0)
    inside = np.array([1.0, 0.0, 0.0])
    outside = np.array([4.0, 0.0, 0.0])

    assert sphere.residual(inside)[0] < 0.0
    assert sphere.residual(outside)[0] > 0.0


def test_sphere_jacobian_shape():
    sphere = SphereManifold(center=np.array([0.0, 0.0, 0.0]), radius=3.0)
    j = sphere.jacobian(np.array([3.0, 0.0, 0.0]))
    assert j.shape == (1, 3)


def test_sphere_analytic_projection():
    sphere = SphereManifold(center=np.array([1.0, -2.0, 0.5]), radius=2.5)
    x = np.array([7.0, 1.0, 4.0])

    x_proj = sphere.closest_point_analytic(x)

    assert np.isclose(np.linalg.norm(x_proj - sphere.center), sphere.radius, atol=1e-8)
    assert sphere.is_valid(x_proj, tol=1e-8)


def test_sphere_tangent_projection_is_orthogonal_to_normal():
    sphere = SphereManifold(center=np.array([0.0, 0.0, 0.0]), radius=3.0)
    x = np.array([3.0, 0.0, 0.0])
    v = np.array([1.0, 2.0, -1.0])

    v_tan = sphere.project_tangent(x, v)
    normal = sphere.jacobian(x).reshape(-1)

    assert np.isclose(np.dot(normal, v_tan), 0.0, atol=1e-8)


def test_plane_residual_on_manifold():
    plane = PlaneManifold(
        point=np.array([0.0, 0.0, 1.0]),
        normal=np.array([0.0, 0.0, 1.0]),
    )
    x = np.array([2.0, -3.0, 1.0])
    assert abs(plane.residual(x)[0]) < 1e-12


def test_plane_residual_above_and_below():
    plane = PlaneManifold(
        point=np.array([0.0, 0.0, 1.0]),
        normal=np.array([0.0, 0.0, 1.0]),
    )

    above = np.array([0.0, 0.0, 3.0])
    below = np.array([0.0, 0.0, -2.0])

    assert plane.residual(above)[0] > 0.0
    assert plane.residual(below)[0] < 0.0