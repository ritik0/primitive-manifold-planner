import numpy as np

from primitive_manifold_planner.families.standard import MaskedFamily, SphereFamily
from primitive_manifold_planner.manifolds import MaskedManifold, SphereManifold


def test_masked_manifold_respects_valid_subset():
    base = SphereManifold(center=np.array([0.0, 0.0, 0.0]), radius=1.0, name="sphere")
    masked = MaskedManifold(
        base_manifold=base,
        validity_fn=lambda q: float(q[0]) >= 0.0,
        name="masked_sphere",
    )

    assert masked.within_bounds(np.array([1.0, 0.0, 0.0]))
    assert not masked.within_bounds(np.array([-1.0, 0.0, 0.0]))
    assert masked.is_valid(np.array([1.0, 0.0, 0.0]))
    assert not masked.is_valid(np.array([-1.0, 0.0, 0.0]))


def test_masked_family_wraps_base_family_with_subset_logic():
    base_family = SphereFamily(
        name="sphere_family",
        center=np.array([0.0, 0.0, 0.0]),
        radii={1.0: 1.0},
    )
    masked_family = MaskedFamily(
        base_family=base_family,
        validity_mask_fn=lambda lam, q: float(q[1]) >= 0.0,
        name="masked_family",
    )

    manifold = masked_family.manifold(1.0)
    anchors = list(masked_family.transition_seed_anchors(1.0, goal_point=np.array([1.0, 0.5, 0.0])))

    assert isinstance(manifold, MaskedManifold)
    assert all(float(np.asarray(anchor)[1]) >= 0.0 for anchor in anchors)
