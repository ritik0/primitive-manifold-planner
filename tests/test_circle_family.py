import numpy as np

from primitive_manifold_planner.manifolds import ConcentricCircleFamily, CircleManifold


def test_concentric_circle_family_leaf_returns_circle_manifold():
    family = ConcentricCircleFamily()
    leaf = family.leaf(2.0)

    assert isinstance(leaf, CircleManifold)
    assert leaf.is_valid(np.array([2.0, 0.0]), tol=1e-8)
    assert not leaf.is_valid(np.array([1.0, 0.0]), tol=1e-8)


def test_concentric_circle_family_residual():
    family = ConcentricCircleFamily()

    x_on = np.array([2.0, 0.0])
    x_off = np.array([3.0, 0.0])

    assert abs(family.residual(x_on, 2.0)[0]) < 1e-12
    assert abs(family.residual(x_off, 2.0)[0] - 5.0) < 1e-12


def test_concentric_circle_family_is_on_leaf():
    family = ConcentricCircleFamily()

    assert family.is_on_leaf(np.array([0.0, 3.0]), 3.0)
    assert not family.is_on_leaf(np.array([0.0, 2.5]), 3.0)


def test_concentric_circle_family_sample_lambdas():
    family = ConcentricCircleFamily()
    lambdas = family.sample_lambdas(1.0, 3.0, 5)

    assert len(lambdas) == 5
    assert np.allclose(lambdas, np.array([1.0, 1.5, 2.0, 2.5, 3.0]))