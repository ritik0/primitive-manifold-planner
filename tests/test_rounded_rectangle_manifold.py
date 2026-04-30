import numpy as np

from primitive_manifold_planner.manifolds import RoundedRectangleManifold


def test_rounded_rectangle_point_from_angle_is_on_manifold():
    manifold = RoundedRectangleManifold(
        center=np.array([0.0, 0.0]),
        a=1.2,
        b=0.5,
        power=4.0,
    )

    point = manifold.point_from_angle(np.pi / 3.0)

    assert manifold.is_valid(point, tol=1e-6)
    assert manifold.within_bounds(point)


def test_rounded_rectangle_within_bounds_rejects_far_point():
    manifold = RoundedRectangleManifold(
        center=np.array([0.0, 0.0]),
        a=1.0,
        b=0.4,
        power=4.0,
    )

    assert not manifold.within_bounds(np.array([1.5, 0.0]))
