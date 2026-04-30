import numpy as np

from primitive_manifold_planner.manifolds import ParallelLineFamily, LineManifold


def test_parallel_line_family_leaf_returns_line_manifold():
    family = ParallelLineFamily()
    leaf = family.leaf(1.5)

    assert isinstance(leaf, LineManifold)
    assert leaf.is_valid(np.array([3.0, 1.5]), tol=1e-8)
    assert not leaf.is_valid(np.array([3.0, 1.2]), tol=1e-8)


def test_parallel_line_family_residual():
    family = ParallelLineFamily()

    x_on = np.array([2.0, 1.0])
    x_off = np.array([2.0, 1.5])

    assert abs(family.residual(x_on, 1.0)[0]) < 1e-12
    assert abs(family.residual(x_off, 1.0)[0] - 0.5) < 1e-12


def test_parallel_line_family_is_on_leaf():
    family = ParallelLineFamily()

    assert family.is_on_leaf(np.array([0.0, -2.0]), -2.0)
    assert not family.is_on_leaf(np.array([0.0, -1.5]), -2.0)


def test_parallel_line_family_sample_lambdas():
    family = ParallelLineFamily()
    lambdas = family.sample_lambdas(-2.0, 2.0, 5)

    assert len(lambdas) == 5
    assert np.allclose(lambdas, np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))