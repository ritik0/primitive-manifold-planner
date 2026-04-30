from __future__ import annotations

import numpy as np

from primitive_manifold_planner.examplesupport.planar3link import (
    Planar3LinkKinematics,
    torus_distance,
    wrap_q,
)
from primitive_manifold_planner.examplesupport.planar3link_families import (
    EndEffectorXFamily3Link,
    EndEffectorYFamily3Link,
)


def test_planar3link_fk_and_joint_positions_shape():
    robot = Planar3LinkKinematics(l1=1.0, l2=0.8, l3=0.6)
    q = np.array([0.2, -0.5, 0.7], dtype=float)

    ee = robot.fk(q)
    joints = robot.joint_positions(q)

    assert ee.shape == (2,)
    assert joints.shape == (4, 2)
    assert np.allclose(joints[-1], ee)


def test_planar3link_jacobian_matches_finite_difference():
    robot = Planar3LinkKinematics(l1=1.0, l2=0.8, l3=0.6)
    q = np.array([0.4, -0.7, 0.5], dtype=float)
    jac = robot.jacobian_xy(q)

    eps = 1e-6
    approx = np.zeros((2, 3), dtype=float)
    for i in range(3):
        dq = np.zeros((3,), dtype=float)
        dq[i] = eps
        approx[:, i] = (robot.fk(q + dq) - robot.fk(q - dq)) / (2.0 * eps)

    assert np.allclose(jac, approx, atol=1e-5)


def test_planar3link_leaf_families_define_valid_projected_points():
    robot = Planar3LinkKinematics(l1=1.0, l2=0.8, l3=0.6)
    q = np.array([0.6, -1.0, 0.4], dtype=float)
    ee = robot.fk(q)

    x_family = EndEffectorXFamily3Link("x_family", robot=robot, lambdas=[float(ee[0])])
    y_family = EndEffectorYFamily3Link("y_family", robot=robot, lambdas=[float(ee[1])])

    assert x_family.manifold(float(ee[0])).is_valid(q)
    assert y_family.manifold(float(ee[1])).is_valid(q)


def test_wrap_q_and_torus_distance_are_periodic():
    q = np.array([3.5, -3.6, 3.4], dtype=float)
    wrapped = wrap_q(q)

    assert np.all(np.abs(wrapped) <= np.pi + 1e-12)
    assert torus_distance(q, wrapped) < 1e-9
