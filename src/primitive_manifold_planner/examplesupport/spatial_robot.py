from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class SpatialRobot3DOF:
    link_lengths: np.ndarray
    base_world: np.ndarray
    link_radius: float = 0.055
    joint_radius: float = 0.095
    ee_radius: float = 0.075

    @property
    def max_reach(self) -> float:
        return float(np.sum(self.link_lengths))

    def forward_kinematics_3d(self, joint_angles: np.ndarray) -> np.ndarray:
        theta0, theta1, theta2 = np.asarray(joint_angles, dtype=float).reshape(3)
        l1, l2, l3 = np.asarray(self.link_lengths, dtype=float).reshape(3)

        yaw_dir = np.asarray([math.cos(theta0), math.sin(theta0), 0.0], dtype=float)

        radial_1 = l1 * math.cos(theta1)
        z_1 = l1 * math.sin(theta1)

        angle_12 = theta1 + theta2
        radial_2 = radial_1 + l2 * math.cos(angle_12)
        z_2 = z_1 + l2 * math.sin(angle_12)

        radial_3 = radial_2 + l3 * math.cos(angle_12)
        z_3 = z_2 + l3 * math.sin(angle_12)

        p0 = np.asarray(self.base_world, dtype=float)
        p1 = p0 + radial_1 * yaw_dir + np.asarray([0.0, 0.0, z_1], dtype=float)
        p2 = p0 + radial_2 * yaw_dir + np.asarray([0.0, 0.0, z_2], dtype=float)
        p3 = p0 + radial_3 * yaw_dir + np.asarray([0.0, 0.0, z_3], dtype=float)
        return np.asarray([p0, p1, p2, p3], dtype=float)
