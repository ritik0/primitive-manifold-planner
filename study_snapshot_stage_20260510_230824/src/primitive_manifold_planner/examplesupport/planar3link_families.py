from __future__ import annotations

from typing import Iterable, List
import numpy as np

from primitive_manifold_planner.examplesupport.planar3link import (
    Planar3LinkKinematics,
    ik_position_solutions,
)


class EndEffectorXManifold3Link:
    def __init__(self, robot: Planar3LinkKinematics, target_x: float, name: str):
        self.robot = robot
        self.target_x = float(target_x)
        self.name = name

    def _coerce_point(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        return q[:3].copy()

    def residual(self, q: np.ndarray) -> np.ndarray:
        q = self._coerce_point(q)
        return np.array([self.robot.fk(q)[0] - self.target_x], dtype=float)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        q = self._coerce_point(q)
        return self.robot.jacobian_xy(q)[0:1, :]

    def is_valid(self, q: np.ndarray, tol: float = 1e-6) -> bool:
        return float(np.linalg.norm(self.residual(q))) <= tol


class EndEffectorYManifold3Link:
    def __init__(self, robot: Planar3LinkKinematics, target_y: float, name: str):
        self.robot = robot
        self.target_y = float(target_y)
        self.name = name

    def _coerce_point(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        return q[:3].copy()

    def residual(self, q: np.ndarray) -> np.ndarray:
        q = self._coerce_point(q)
        return np.array([self.robot.fk(q)[1] - self.target_y], dtype=float)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        q = self._coerce_point(q)
        return self.robot.jacobian_xy(q)[1:2, :]

    def is_valid(self, q: np.ndarray, tol: float = 1e-6) -> bool:
        return float(np.linalg.norm(self.residual(q))) <= tol


class Joint3Manifold:
    def __init__(self, target_q3: float, name: str):
        self.target_q3 = float(target_q3)
        self.name = name

    def _coerce_point(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        return q[:3].copy()

    def residual(self, q: np.ndarray) -> np.ndarray:
        q = self._coerce_point(q)
        return np.array([q[2] - self.target_q3], dtype=float)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        _ = self._coerce_point(q)
        return np.array([[0.0, 0.0, 1.0]], dtype=float)

    def is_valid(self, q: np.ndarray, tol: float = 1e-6) -> bool:
        return float(np.linalg.norm(self.residual(q))) <= tol


class EndEffectorCircleManifold3Link:
    def __init__(self, robot: Planar3LinkKinematics, center_xy: np.ndarray, radius: float, name: str):
        self.robot = robot
        self.center_xy = np.asarray(center_xy, dtype=float).reshape(2)
        self.radius = float(radius)
        self.name = name

    def _coerce_point(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        return q[:3].copy()

    def residual(self, q: np.ndarray) -> np.ndarray:
        q = self._coerce_point(q)
        p = self.robot.fk(q)
        d = p - self.center_xy
        return np.array([np.dot(d, d) - self.radius**2], dtype=float)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        q = self._coerce_point(q)
        p = self.robot.fk(q)
        d = (p - self.center_xy).reshape(1, 2)
        return 2.0 * d @ self.robot.jacobian_xy(q)

    def is_valid(self, q: np.ndarray, tol: float = 1e-6) -> bool:
        return float(np.linalg.norm(self.residual(q))) <= tol


class EndEffectorEllipseManifold3Link:
    def __init__(self, robot: Planar3LinkKinematics, center_xy: np.ndarray, a: float, b: float, name: str):
        self.robot = robot
        self.center_xy = np.asarray(center_xy, dtype=float).reshape(2)
        self.a = float(a)
        self.b = float(b)
        self.name = name

    def _coerce_point(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        return q[:3].copy()

    def residual(self, q: np.ndarray) -> np.ndarray:
        q = self._coerce_point(q)
        p = self.robot.fk(q)
        dx = (p[0] - self.center_xy[0]) / self.a
        dy = (p[1] - self.center_xy[1]) / self.b
        return np.array([dx * dx + dy * dy - 1.0], dtype=float)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        q = self._coerce_point(q)
        p = self.robot.fk(q)
        grad_xy = np.array(
            [[
                2.0 * (p[0] - self.center_xy[0]) / (self.a * self.a),
                2.0 * (p[1] - self.center_xy[1]) / (self.b * self.b),
            ]],
            dtype=float,
        )
        return grad_xy @ self.robot.jacobian_xy(q)

    def is_valid(self, q: np.ndarray, tol: float = 1e-6) -> bool:
        return float(np.linalg.norm(self.residual(q))) <= tol


class BenchmarkLeafFamily3Link:
    def __init__(self, name: str, lambdas: Iterable[float]):
        self.name = name
        self._lambdas = [float(v) for v in lambdas]

    def manifold(self, lam: float):
        raise NotImplementedError

    def sample_lambdas(self, context=None):
        return list(self._lambdas)

    def lambda_distance(self, lam_a, lam_b) -> float:
        return abs(float(lam_a) - float(lam_b))

    def transition_seed_anchors(self, lam: float, goal_point: np.ndarray | None = None):
        _ = lam, goal_point
        return [
            np.array([-2.2, 1.3, 0.9], dtype=float),
            np.array([-1.3, 0.7, -0.8], dtype=float),
            np.array([0.0, -1.0, 1.0], dtype=float),
            np.array([1.2, 0.8, -0.6], dtype=float),
            np.array([2.2, -1.3, 0.5], dtype=float),
        ]


class EndEffectorXFamily3Link(BenchmarkLeafFamily3Link):
    def __init__(self, name: str, robot: Planar3LinkKinematics, lambdas: Iterable[float]):
        super().__init__(name=name, lambdas=lambdas)
        self.robot = robot

    def manifold(self, lam: float):
        return EndEffectorXManifold3Link(
            robot=self.robot,
            target_x=float(lam),
            name=f"{self.name}_lambda_{lam:g}",
        )


class EndEffectorYFamily3Link(BenchmarkLeafFamily3Link):
    def __init__(self, name: str, robot: Planar3LinkKinematics, lambdas: Iterable[float]):
        super().__init__(name=name, lambdas=lambdas)
        self.robot = robot

    def manifold(self, lam: float):
        return EndEffectorYManifold3Link(
            robot=self.robot,
            target_y=float(lam),
            name=f"{self.name}_lambda_{lam:g}",
        )


class Joint3Family(BenchmarkLeafFamily3Link):
    def manifold(self, lam: float):
        return Joint3Manifold(
            target_q3=float(lam),
            name=f"{self.name}_lambda_{lam:g}",
        )


class EndEffectorCircleFamily3Link(BenchmarkLeafFamily3Link):
    def __init__(self, name: str, robot: Planar3LinkKinematics, center_xy: np.ndarray, radii: Iterable[float]):
        super().__init__(name=name, lambdas=radii)
        self.robot = robot
        self.center_xy = np.asarray(center_xy, dtype=float).reshape(2)

    def manifold(self, lam: float):
        return EndEffectorCircleManifold3Link(
            robot=self.robot,
            center_xy=self.center_xy,
            radius=float(lam),
            name=f"{self.name}_lambda_{lam:g}",
        )

    def transition_seed_anchors(self, lam: float, goal_point: np.ndarray | None = None):
        radius = float(lam)
        angles = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
        workspace_points = [
            self.center_xy + radius * np.array([np.cos(theta), np.sin(theta)], dtype=float)
            for theta in angles
        ]
        if goal_point is not None:
            q_goal = np.asarray(goal_point, dtype=float).reshape(-1)
            goal_xy = self.robot.fk(q_goal[:3])
            direction = goal_xy - self.center_xy
            norm = float(np.linalg.norm(direction))
            if norm > 1e-12:
                workspace_points.append(self.center_xy + radius * direction / norm)

        anchors = []
        for xy in workspace_points:
            anchors.extend(ik_position_solutions(self.robot, xy))
            if len(anchors) >= 8:
                break
        if not anchors:
            return super().transition_seed_anchors(lam, goal_point=goal_point)
        return anchors


class EndEffectorEllipseFamily3Link(BenchmarkLeafFamily3Link):
    def __init__(self, name: str, robot: Planar3LinkKinematics, center_xy: np.ndarray, a_scales: dict[float, float], b_scales: dict[float, float]):
        super().__init__(name=name, lambdas=list(a_scales.keys()))
        self.robot = robot
        self.center_xy = np.asarray(center_xy, dtype=float).reshape(2)
        self.a_scales = {float(k): float(v) for k, v in a_scales.items()}
        self.b_scales = {float(k): float(v) for k, v in b_scales.items()}

    def manifold(self, lam: float):
        lam = float(lam)
        return EndEffectorEllipseManifold3Link(
            robot=self.robot,
            center_xy=self.center_xy,
            a=self.a_scales[lam],
            b=self.b_scales[lam],
            name=f"{self.name}_lambda_{lam:g}",
        )

    def transition_seed_anchors(self, lam: float, goal_point: np.ndarray | None = None):
        lam = float(lam)
        a = self.a_scales[lam]
        b = self.b_scales[lam]
        angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
        workspace_points = [
            self.center_xy + np.array([a * np.cos(theta), b * np.sin(theta)], dtype=float)
            for theta in angles
        ]
        if goal_point is not None:
            q_goal = np.asarray(goal_point, dtype=float).reshape(-1)
            goal_xy = self.robot.fk(q_goal[:3])
            theta = np.arctan2(goal_xy[1] - self.center_xy[1], goal_xy[0] - self.center_xy[0])
            workspace_points.append(self.center_xy + np.array([a * np.cos(theta), b * np.sin(theta)], dtype=float))

        anchors = []
        for xy in workspace_points:
            anchors.extend(ik_position_solutions(self.robot, xy))
            if len(anchors) >= 10:
                break
        if not anchors:
            return super().transition_seed_anchors(lam, goal_point=goal_point)
        return anchors
