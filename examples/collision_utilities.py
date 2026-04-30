from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import pyvista as pv
except Exception:
    pv = None


def _segment_samples(p0: np.ndarray, p1: np.ndarray, samples: int) -> np.ndarray:
    a = np.asarray(p0, dtype=float)
    b = np.asarray(p1, dtype=float)
    count = max(2, int(samples))
    alphas = np.linspace(0.0, 1.0, count)
    return np.asarray([(1.0 - alpha) * a + alpha * b for alpha in alphas], dtype=float)


def _adaptive_segment_sample_count(p0: np.ndarray, p1: np.ndarray, resolution: float = 0.08) -> int:
    length = float(np.linalg.norm(np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)))
    return max(8, int(np.ceil(length / max(resolution, 1e-6))) + 1)


@dataclass
class Obstacle:
    name: str

    def check_collision(self, robot, joint_angles: np.ndarray) -> bool:
        joints = np.asarray(robot.forward_kinematics_3d(np.asarray(joint_angles, dtype=float)), dtype=float)
        for idx in range(len(joints) - 1):
            if self.segment_collision(joints[idx], joints[idx + 1], radius=float(getattr(robot, "link_radius", 0.0))):
                return True
        return self.point_collision(joints[-1], radius=float(getattr(robot, "ee_radius", 0.0)))

    def point_collision(self, point: np.ndarray, radius: float = 0.0) -> bool:
        raise NotImplementedError

    def segment_collision(self, p0: np.ndarray, p1: np.ndarray, radius: float = 0.0) -> bool:
        samples = _segment_samples(p0, p1, _adaptive_segment_sample_count(p0, p1))
        return any(self.point_collision(sample, radius=radius) for sample in samples)

    def to_pyvista_mesh(self):
        return None


@dataclass
class CylinderObstacle(Obstacle):
    center_xy: np.ndarray
    radius: float
    z_min: float
    z_max: float

    def point_collision(self, point: np.ndarray, radius: float = 0.0) -> bool:
        q = np.asarray(point, dtype=float)
        if q[2] < self.z_min - radius or q[2] > self.z_max + radius:
            return False
        delta_xy = q[:2] - np.asarray(self.center_xy, dtype=float).reshape(2)
        return float(np.linalg.norm(delta_xy)) <= float(self.radius + radius)

    def to_pyvista_mesh(self):
        if pv is None:
            return None
        height = float(self.z_max - self.z_min)
        center = np.asarray(
            [
                float(np.asarray(self.center_xy, dtype=float)[0]),
                float(np.asarray(self.center_xy, dtype=float)[1]),
                0.5 * float(self.z_min + self.z_max),
            ],
            dtype=float,
        )
        return pv.Cylinder(center=center, direction=(0.0, 0.0, 1.0), radius=float(self.radius), height=height, resolution=40)


@dataclass
class BoxObstacle(Obstacle):
    center: np.ndarray
    half_extents: np.ndarray

    def point_collision(self, point: np.ndarray, radius: float = 0.0) -> bool:
        q = np.asarray(point, dtype=float)
        center = np.asarray(self.center, dtype=float).reshape(3)
        extents = np.asarray(self.half_extents, dtype=float).reshape(3) + float(radius)
        return bool(np.all(np.abs(q - center) <= extents))

    def segment_collision(self, p0: np.ndarray, p1: np.ndarray, radius: float = 0.0) -> bool:
        center = np.asarray(self.center, dtype=float).reshape(3)
        extents = np.asarray(self.half_extents, dtype=float).reshape(3) + float(radius)
        start = np.asarray(p0, dtype=float) - center
        end = np.asarray(p1, dtype=float) - center
        direction = end - start
        t_min = 0.0
        t_max = 1.0
        for axis in range(3):
            if abs(direction[axis]) <= 1e-12:
                if abs(start[axis]) > extents[axis]:
                    return False
                continue
            inv = 1.0 / direction[axis]
            t1 = (-extents[axis] - start[axis]) * inv
            t2 = (extents[axis] - start[axis]) * inv
            lo = min(t1, t2)
            hi = max(t1, t2)
            t_min = max(t_min, lo)
            t_max = min(t_max, hi)
            if t_min > t_max:
                return False
        return True

    def to_pyvista_mesh(self):
        if pv is None:
            return None
        return pv.Box(bounds=(
            float(self.center[0] - self.half_extents[0]),
            float(self.center[0] + self.half_extents[0]),
            float(self.center[1] - self.half_extents[1]),
            float(self.center[1] + self.half_extents[1]),
            float(self.center[2] - self.half_extents[2]),
            float(self.center[2] + self.half_extents[2]),
        ))


def configuration_in_collision(robot, joint_angles: np.ndarray, obstacles: list[Obstacle] | None) -> bool:
    if obstacles is None:
        return False
    return any(obstacle.check_collision(robot, joint_angles) for obstacle in obstacles)


def joint_path_collision_free(
    robot,
    joint_path: np.ndarray,
    obstacles: list[Obstacle] | None,
    interpolation_substeps: int = 5,
) -> bool:
    if obstacles is None or len(obstacles) == 0:
        return True
    path = np.asarray(joint_path, dtype=float)
    if len(path) == 0:
        return True
    for q in path:
        if configuration_in_collision(robot, q, obstacles):
            return False
    for q0, q1 in zip(path[:-1], path[1:]):
        start = np.asarray(q0, dtype=float)
        delta = np.asarray(q1, dtype=float) - start
        for alpha in np.linspace(0.0, 1.0, max(2, int(interpolation_substeps))):
            q = start + float(alpha) * delta
            if configuration_in_collision(robot, q, obstacles):
                return False
    return True


def default_example_66_obstacles() -> list[Obstacle]:
    return [
        CylinderObstacle(
            name="left_transition_cylinder",
            center_xy=np.asarray([-0.95, -0.08], dtype=float),
            radius=0.16,
            z_min=0.00,
            z_max=1.05,
        ),
        BoxObstacle(
            name="goal_side_box",
            center=np.asarray([1.92, 0.58, 0.48], dtype=float),
            half_extents=np.asarray([0.16, 0.14, 0.22], dtype=float),
        ),
    ]
