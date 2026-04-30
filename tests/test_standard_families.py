import numpy as np

from primitive_manifold_planner.families.standard import (
    DoubleSphereFamily,
    EllipseFamily,
    PlaneFamily,
    RoundedRectangleConnectorFamily,
    SphereFamily,
)
from primitive_manifold_planner.manifolds import (
    DoubleSphereManifold,
    EllipseManifold,
    PlaneManifold,
    RoundedRectangleManifold,
    SphereManifold,
)


def test_ellipse_family_builds_manifold_and_anchors():
    family = EllipseFamily(
        name="ellipse_family",
        center=np.array([0.0, 0.0]),
        a_scales={1.0: 2.0},
        b_scales={1.0: 1.0},
    )

    manifold = family.manifold(1.0)
    anchors = list(family.transition_seed_anchors(1.0, goal_point=np.array([1.0, 0.2])))

    assert isinstance(manifold, EllipseManifold)
    assert len(anchors) >= 4


def test_rounded_rectangle_connector_family_builds_bounded_connector():
    family = RoundedRectangleConnectorFamily(
        name="connector_family",
        center=np.array([0.0, 0.0]),
        a_scales={1.0: 1.2},
        b_scales={1.0: 0.4},
        power=4.0,
    )

    manifold = family.manifold(1.0)
    anchors = list(family.transition_seed_anchors(1.0, goal_point=np.array([1.0, 0.0])))

    assert isinstance(manifold, RoundedRectangleManifold)
    assert manifold.within_bounds(anchors[0])
    assert len(anchors) >= 4


def test_sphere_family_builds_3d_manifold_and_anchors():
    family = SphereFamily(
        name="sphere_family",
        center=np.array([0.0, 0.0, 0.0]),
        radii={1.0: 2.0},
    )

    manifold = family.manifold(1.0)
    anchors = list(family.transition_seed_anchors(1.0, goal_point=np.array([1.0, 0.5, 0.25])))

    assert isinstance(manifold, SphereManifold)
    assert len(anchors) >= 6
    assert all(np.asarray(anchor).shape == (3,) for anchor in anchors)


def test_plane_family_builds_3d_manifold_and_anchors():
    family = PlaneFamily(
        name="plane_family",
        base_point=np.array([0.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        offsets=[0.0],
        anchor_span=1.25,
    )

    manifold = family.manifold(0.0)
    anchors = list(family.transition_seed_anchors(0.0, goal_point=np.array([0.5, -0.5, 3.0])))

    assert isinstance(manifold, PlaneManifold)
    assert len(anchors) >= 5
    assert all(np.asarray(anchor).shape == (3,) for anchor in anchors)


def test_double_sphere_family_builds_disconnected_3d_leaf():
    family = DoubleSphereFamily(
        name="double_sphere_family",
        center_a=np.array([-1.0, 0.0, 0.0]),
        center_b=np.array([1.0, 0.0, 0.0]),
        radii={1.0: 0.75},
    )

    manifold = family.manifold(1.0)
    anchors = list(family.transition_seed_anchors(1.0, goal_point=np.array([2.0, 0.0, 0.0])))

    assert isinstance(manifold, DoubleSphereManifold)
    assert len(anchors) >= 6
    assert all(np.asarray(anchor).shape == (3,) for anchor in anchors)
