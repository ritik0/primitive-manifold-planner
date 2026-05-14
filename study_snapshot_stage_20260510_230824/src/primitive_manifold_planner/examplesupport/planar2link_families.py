from __future__ import annotations

from typing import List
import numpy as np

from primitive_manifold_planner.examplesupport.planar2link import Planar2LinkKinematics


class EndEffectorXManifold:
    def __init__(self, robot: Planar2LinkKinematics, target_x: float, name: str):
        self.robot = robot
        self.target_x = float(target_x)
        self.name = name

    def _coerce_point(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        return q[:2].copy()

    def residual(self, q: np.ndarray) -> np.ndarray:
        q = self._coerce_point(q)
        return np.array([self.robot.fk(q)[0] - self.target_x], dtype=float)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        q = self._coerce_point(q)
        return self.robot.jacobian_xy(q)[0:1, :]

    def is_valid(self, q: np.ndarray, tol: float = 1e-6) -> bool:
        return float(np.linalg.norm(self.residual(q))) <= tol


class EndEffectorYManifold:
    def __init__(self, robot: Planar2LinkKinematics, target_y: float, name: str):
        self.robot = robot
        self.target_y = float(target_y)
        self.name = name

    def _coerce_point(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        return q[:2].copy()

    def residual(self, q: np.ndarray) -> np.ndarray:
        q = self._coerce_point(q)
        return np.array([self.robot.fk(q)[1] - self.target_y], dtype=float)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        q = self._coerce_point(q)
        return self.robot.jacobian_xy(q)[1:2, :]

    def is_valid(self, q: np.ndarray, tol: float = 1e-6) -> bool:
        return float(np.linalg.norm(self.residual(q))) <= tol


class Joint2Manifold:
    def __init__(self, target_q2: float, name: str):
        self.target_q2 = float(target_q2)
        self.name = name

    def _coerce_point(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        return q[:2].copy()

    def residual(self, q: np.ndarray) -> np.ndarray:
        q = self._coerce_point(q)
        return np.array([q[1] - self.target_q2], dtype=float)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        _ = self._coerce_point(q)
        return np.array([[0.0, 1.0]], dtype=float)

    def is_valid(self, q: np.ndarray, tol: float = 1e-6) -> bool:
        return float(np.linalg.norm(self.residual(q))) <= tol


class BenchmarkLeafFamily:
    def __init__(self, name: str, lambdas: List[float]):
        self.name = name
        self._lambdas = [float(v) for v in lambdas]

    def manifold(self, lam: float):
        raise NotImplementedError

    def sample_lambdas(self, context=None):
        return list(self._lambdas)

    def lambda_distance(self, lam_a, lam_b) -> float:
        return abs(float(lam_a) - float(lam_b))


class EndEffectorXFamily(BenchmarkLeafFamily):
    def __init__(self, name: str, robot: Planar2LinkKinematics, lambdas: List[float]):
        super().__init__(name=name, lambdas=lambdas)
        self.robot = robot

    def manifold(self, lam: float):
        return EndEffectorXManifold(
            robot=self.robot,
            target_x=float(lam),
            name=f"{self.name}_lambda_{lam:g}",
        )


class EndEffectorYFamily(BenchmarkLeafFamily):
    def __init__(self, name: str, robot: Planar2LinkKinematics, lambdas: List[float]):
        super().__init__(name=name, lambdas=lambdas)
        self.robot = robot

    def manifold(self, lam: float):
        return EndEffectorYManifold(
            robot=self.robot,
            target_y=float(lam),
            name=f"{self.name}_lambda_{lam:g}",
        )


class Joint2Family(BenchmarkLeafFamily):
    def manifold(self, lam: float):
        return Joint2Manifold(
            target_q2=float(lam),
            name=f"{self.name}_lambda_{lam:g}",
        )