from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class SpatialRobot3DOF:
    """Simple 3DOF serial positioning arm.

    Joint convention:
        q[0] = base yaw about world z
        q[1] = shoulder pitch in the yaw-defined vertical plane
        q[2] = elbow pitch relative to the shoulder

    This is a position-only educational manipulator for the thesis demo.
    It is not a wrist-orientation model and it is not a Franka/Panda model.

    FK returns four 3D points:
        p0 = fixed base / shoulder joint
        p1 = end of link 1
        p2 = elbow / end of link 2
        p3 = end-effector / end of link 3

    The planner uses only the end-effector position p3 for FK-pulled-back
    constraints:
        sphere: ||FK(q)[-1] - c||^2 - r^2 = 0
        plane : n · (FK(q)[-1] - p) = 0
    """

    link_lengths: np.ndarray
    base_world: np.ndarray
    link_radius: float = 0.045
    joint_radius: float = 0.080
    ee_radius: float = 0.070

    @property
    def max_reach(self) -> float:
        return float(np.sum(np.asarray(self.link_lengths, dtype=float)))

    def forward_kinematics_3d(self, joint_angles: np.ndarray) -> np.ndarray:
        q0, q1, q2 = np.asarray(joint_angles, dtype=float).reshape(3)
        l1, l2, l3 = np.asarray(self.link_lengths, dtype=float).reshape(3)

        # Base yaw defines the vertical plane in which the arm bends.
        yaw_dir = np.asarray([math.cos(q0), math.sin(q0), 0.0], dtype=float)
        z_dir = np.asarray([0.0, 0.0, 1.0], dtype=float)

        # First link is controlled by shoulder pitch.
        d1 = math.cos(q1) * yaw_dir + math.sin(q1) * z_dir

        # Second and third physical segments follow the elbow angle.
        # With only 3DOF total, there is no independent wrist orientation.
        d2 = math.cos(q1 + q2) * yaw_dir + math.sin(q1 + q2) * z_dir

        p0 = np.asarray(self.base_world, dtype=float).reshape(3)
        p1 = p0 + float(l1) * d1
        p2 = p1 + float(l2) * d2
        p3 = p2 + float(l3) * d2

        return np.asarray([p0, p1, p2, p3], dtype=float)