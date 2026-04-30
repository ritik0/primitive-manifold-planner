import numpy as np

from primitive_manifold_planner.manifolds import EllipseManifold


def test_ellipse_residual_on_manifold():
    ellipse = EllipseManifold(center=np.array([0.0, 0.0]), a=3.0, b=2.0)
    x = np.array([3.0, 0.0])
    assert abs(ellipse.residual(x)[0]) < 1e-12


def test_ellipse_residual_inside_and_outside():
    ellipse = EllipseManifold(center=np.array([0.0, 0.0]), a=3.0, b=2.0)

    inside = np.array([1.0, 0.0])
    outside = np.array([4.0, 0.0])

    assert ellipse.residual(inside)[0] < 0.0
    assert ellipse.residual(outside)[0] > 0.0


def test_ellipse_jacobian_shape():
    ellipse = EllipseManifold(center=np.array([0.0, 0.0]), a=3.0, b=2.0)
    j = ellipse.jacobian(np.array([3.0, 0.0]))
    assert j.shape == (1, 2)


def test_ellipse_point_from_angle_is_valid():
    ellipse = EllipseManifold(center=np.array([1.0, -1.0]), a=4.0, b=2.0)

    for theta in [0.0, np.pi / 4.0, np.pi / 2.0, np.pi]:
        x = ellipse.point_from_angle(theta)
        assert ellipse.is_valid(x, tol=1e-8)


def test_ellipse_tangent_projection_is_orthogonal_to_normal():
    ellipse = EllipseManifold(center=np.array([0.0, 0.0]), a=3.0, b=2.0)
    x = np.array([3.0, 0.0])
    v = np.array([1.0, 1.0])

    v_tan = ellipse.project_tangent(x, v)
    normal = ellipse.jacobian(x).reshape(-1)

    assert np.isclose(np.dot(normal, v_tan), 0.0, atol=1e-8)