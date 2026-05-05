from __future__ import annotations

import numpy as np

from primitive_manifold_planner.families.standard import MaskedFamily, PlaneFamily, SphereFamily
from primitive_manifold_planner.examplesupport.intrinsic_multimodal_helpers import sphere_point


def build_example66_scene():
    """Build the fixed left-sphere / plane / right-sphere scene used by Example 66."""

    left_support = SphereFamily(
        name="left_support_3d",
        center=np.array([-2.15, -0.55, 0.48], dtype=float),
        radii={1.05: 1.05},
    )
    base_plane = PlaneFamily(
        name="transfer_plane_3d",
        base_point=np.array([0.0, 0.0, 0.0], dtype=float),
        normal=np.array([0.0, 0.0, 1.0], dtype=float),
        offsets=[0.0],
        anchor_span=1.15,
    )
    plane_half_u = 0.95
    plane_half_v = 2.15
    basis_u = np.asarray(base_plane._basis_u, dtype=float)
    basis_v = np.asarray(base_plane._basis_v, dtype=float)
    base_point = np.asarray(base_plane.base_point, dtype=float)

    def rectangle_mask(_lam: float, q: np.ndarray) -> bool:
        qq = np.asarray(q, dtype=float)
        rel = qq - base_point
        u_coord = float(np.dot(rel, basis_u))
        v_coord = float(np.dot(rel, basis_v))
        return abs(u_coord) <= plane_half_u and abs(v_coord) <= plane_half_v

    transfer_plane = MaskedFamily(
        base_family=base_plane,
        validity_mask_fn=rectangle_mask,
        name="transfer_plane_3d",
    )
    right_support = SphereFamily(
        name="right_support_3d",
        center=np.array([2.15, 0.55, 0.48], dtype=float),
        radii={1.05: 1.05},
    )

    start_q = sphere_point(left_support.center, 1.05, azimuth_deg=0.0, elevation_deg=-90.0)
    goal_q = sphere_point(right_support.center, 1.05, azimuth_deg=0.0, elevation_deg=90.0)
    return [left_support, transfer_plane, right_support], start_q, goal_q, plane_half_u, plane_half_v
