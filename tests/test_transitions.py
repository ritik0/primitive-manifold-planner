import numpy as np

from primitive_manifold_planner.manifolds import (
    CircleManifold,
    EllipseManifold,
    LineManifold,
    PlaneManifold,
    SphereManifold,
)
from primitive_manifold_planner.planning import (
    combined_residual,
    find_transition_candidates,
    find_transition_point,
    random_transition_search,
)


def test_combined_residual_shape_for_sphere_plane():
    sphere = SphereManifold(center=np.array([0.0, 0.0, 0.0]), radius=2.0)
    plane = PlaneManifold(
        point=np.array([0.0, 0.0, 1.0]),
        normal=np.array([0.0, 0.0, 1.0]),
    )

    x = np.array([1.0, 1.0, 1.0])
    r = combined_residual(sphere, plane, x)
    assert r.shape == (2,)


def test_find_transition_point_sphere_plane():
    sphere = SphereManifold(center=np.array([0.0, 0.0, 0.0]), radius=2.0)
    plane = PlaneManifold(
        point=np.array([0.0, 0.0, 1.0]),
        normal=np.array([0.0, 0.0, 1.0]),
    )

    seed = np.array([1.5, 0.2, 1.2])
    result = find_transition_point(sphere, plane, seed=seed, tol=1e-8)

    assert result.success
    assert result.x_transition is not None
    assert sphere.is_valid(result.x_transition, tol=1e-6)
    assert plane.is_valid(result.x_transition, tol=1e-6)


def test_find_transition_candidates_circle_circle_finds_multiple():
    circle_a = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0)
    circle_b = CircleManifold(center=np.array([2.0, 0.0]), radius=2.0)

    rng = np.random.default_rng(42)
    result = find_transition_candidates(
        manifold_a=circle_a,
        manifold_b=circle_b,
        bounds_min=np.array([-3.0, -3.0]),
        bounds_max=np.array([5.0, 3.0]),
        num_seeds=100,
        tol=1e-8,
        max_candidates=10,
        rng=rng,
    )

    assert result.success
    assert result.best_candidate is not None
    assert len(result.candidates) >= 2

    for cand in result.candidates:
        assert circle_a.is_valid(cand.point, tol=1e-6)
        assert circle_b.is_valid(cand.point, tol=1e-6)


def test_find_transition_candidates_circle_line_finds_two_points():
    circle = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0)
    line = LineManifold(
        point=np.array([0.0, 1.0]),
        normal=np.array([0.0, 1.0]),
    )

    rng = np.random.default_rng(123)
    result = find_transition_candidates(
        manifold_a=circle,
        manifold_b=line,
        bounds_min=np.array([-3.0, -3.0]),
        bounds_max=np.array([3.0, 3.0]),
        num_seeds=100,
        tol=1e-8,
        max_candidates=10,
        rng=rng,
    )

    assert result.success
    assert len(result.candidates) >= 2

    for cand in result.candidates:
        assert circle.is_valid(cand.point, tol=1e-6)
        assert line.is_valid(cand.point, tol=1e-6)


def test_find_transition_candidates_ellipse_line_finds_two_points():
    ellipse = EllipseManifold(center=np.array([0.0, 0.0]), a=3.0, b=2.0)
    line = LineManifold(
        point=np.array([0.0, 1.0]),
        normal=np.array([0.0, 1.0]),  # y = 1
    )

    rng = np.random.default_rng(21)
    result = find_transition_candidates(
        manifold_a=ellipse,
        manifold_b=line,
        bounds_min=np.array([-4.0, -3.0]),
        bounds_max=np.array([4.0, 3.0]),
        num_seeds=120,
        tol=1e-8,
        max_candidates=10,
        rng=rng,
    )

    assert result.success
    assert len(result.candidates) >= 2

    for cand in result.candidates:
        assert ellipse.is_valid(cand.point, tol=1e-6)
        assert line.is_valid(cand.point, tol=1e-6)


def test_find_transition_candidates_ellipse_circle_can_find_intersection():
    ellipse = EllipseManifold(center=np.array([0.0, 0.0]), a=3.0, b=2.0)
    circle = CircleManifold(center=np.array([1.0, 0.0]), radius=2.0)

    rng = np.random.default_rng(9)
    result = find_transition_candidates(
        manifold_a=ellipse,
        manifold_b=circle,
        bounds_min=np.array([-4.0, -3.0]),
        bounds_max=np.array([4.0, 3.0]),
        num_seeds=150,
        tol=1e-8,
        max_candidates=10,
        rng=rng,
    )

    assert result.success
    assert len(result.candidates) >= 1

    for cand in result.candidates:
        assert ellipse.is_valid(cand.point, tol=1e-5)
        assert circle.is_valid(cand.point, tol=1e-5)


def test_random_transition_search_sphere_plane():
    sphere = SphereManifold(center=np.array([0.0, 0.0, 0.0]), radius=2.0)
    plane = PlaneManifold(
        point=np.array([0.0, 0.0, 1.0]),
        normal=np.array([0.0, 0.0, 1.0]),
    )

    rng = np.random.default_rng(42)
    result = random_transition_search(
        manifold_a=sphere,
        manifold_b=plane,
        bounds_min=np.array([-3.0, -3.0, -3.0]),
        bounds_max=np.array([3.0, 3.0, 3.0]),
        num_seeds=30,
        tol=1e-8,
        rng=rng,
    )

    assert result.success
    assert result.x_transition is not None
    assert sphere.is_valid(result.x_transition, tol=1e-6)
    assert plane.is_valid(result.x_transition, tol=1e-6)