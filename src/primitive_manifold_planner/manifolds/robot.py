from __future__ import annotations

from typing import Any, Callable

import numpy as np

from primitive_manifold_planner.manifolds import ImplicitManifold
from primitive_manifold_planner.projection import ProjectionResult, project_newton

try:
    from scipy.optimize import least_squares
except Exception:
    least_squares = None


def robot_fk_jacobian(robot: Any, theta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    q = np.asarray(theta, dtype=float).reshape(3)
    base = np.asarray(robot.forward_kinematics_3d(q)[-1], dtype=float)
    jac = np.zeros((3, 3), dtype=float)
    for idx in range(3):
        perturbed = q.copy()
        perturbed[idx] += float(eps)
        value = np.asarray(robot.forward_kinematics_3d(perturbed)[-1], dtype=float)
        jac[:, idx] = (value - base) / float(eps)
    return jac


class _RobotConstraintBase(ImplicitManifold):
    def __init__(
        self,
        robot: Any,
        name: str,
        joint_lower: np.ndarray | None = None,
        joint_upper: np.ndarray | None = None,
        task_space_validity_fn: Callable[[np.ndarray], bool] | None = None,
    ) -> None:
        super().__init__(ambient_dim=3, codim=1, name=name)
        self.robot = robot
        self.joint_lower = (
            np.asarray(joint_lower, dtype=float).reshape(3)
            if joint_lower is not None
            else -np.pi * np.ones(3, dtype=float)
        )
        self.joint_upper = (
            np.asarray(joint_upper, dtype=float).reshape(3)
            if joint_upper is not None
            else np.pi * np.ones(3, dtype=float)
        )
        self.task_space_validity_fn = task_space_validity_fn

    def within_bounds(self, theta: np.ndarray, tol: float = 1e-9) -> bool:
        q = self._coerce_point(theta)
        joint_ok = bool(np.all(q >= self.joint_lower - tol) and np.all(q <= self.joint_upper + tol))
        if not joint_ok:
            return False
        if self.task_space_validity_fn is None:
            return True
        ee = np.asarray(self.robot.forward_kinematics_3d(q)[-1], dtype=float)
        return bool(self.task_space_validity_fn(ee))

    def get_implicit_function(self) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
        def fn(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            q = self._coerce_point(theta)
            return self.residual(q), self.jacobian(q)

        return fn

    def project(
        self,
        theta_guess: np.ndarray,
        tol: float = 1e-8,
        max_iters: int = 80,
    ) -> ProjectionResult:
        q0 = np.asarray(theta_guess, dtype=float).reshape(3)
        q0 = np.clip(q0, self.joint_lower, self.joint_upper)

        if least_squares is not None:
            result = least_squares(
                lambda q: self.residual(q),
                q0,
                jac=lambda q: self.jacobian(q),
                bounds=(self.joint_lower, self.joint_upper),
                max_nfev=max_iters,
                xtol=tol,
                ftol=tol,
                gtol=tol,
            )
            q_proj = np.asarray(result.x, dtype=float)
            residual_norm = float(np.linalg.norm(self.residual(q_proj)))
            success = bool(result.success and residual_norm <= max(10.0 * tol, 1e-6) and self.within_bounds(q_proj))
            return ProjectionResult(
                success=success,
                x_projected=q_proj,
                residual_norm=residual_norm,
                iterations=int(getattr(result, "nfev", max_iters)),
                converged=success,
                message=str(getattr(result, "message", "least_squares finished")),
            )

        projected = project_newton(self, q0, tol=tol, max_iters=max_iters, damping=1.0)
        projected.success = bool(projected.success and self.within_bounds(projected.x_projected))
        projected.converged = bool(projected.converged and projected.success)
        return projected


class RobotSphereManifold(_RobotConstraintBase):
    def __init__(
        self,
        robot: Any,
        center: np.ndarray,
        radius: float,
        name: str = "robot_sphere",
        joint_lower: np.ndarray | None = None,
        joint_upper: np.ndarray | None = None,
        task_space_validity_fn: Callable[[np.ndarray], bool] | None = None,
    ) -> None:
        super().__init__(
            robot=robot,
            name=name,
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            task_space_validity_fn=task_space_validity_fn,
        )
        self.center = np.asarray(center, dtype=float).reshape(3)
        self.radius = float(radius)

    def residual(self, theta: np.ndarray) -> np.ndarray:
        q = self._coerce_point(theta)
        ee = np.asarray(self.robot.forward_kinematics_3d(q)[-1], dtype=float)
        dx = ee - self.center
        return np.asarray([float(np.dot(dx, dx) - self.radius**2)], dtype=float)

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        q = self._coerce_point(theta)
        ee = np.asarray(self.robot.forward_kinematics_3d(q)[-1], dtype=float)
        ee_jac = robot_fk_jacobian(self.robot, q)
        grad = 2.0 * (ee - self.center).reshape(1, 3) @ ee_jac
        return np.asarray(grad, dtype=float)


class RobotPlaneManifold(_RobotConstraintBase):
    def __init__(
        self,
        robot: Any,
        point: np.ndarray,
        normal: np.ndarray,
        name: str = "robot_plane",
        joint_lower: np.ndarray | None = None,
        joint_upper: np.ndarray | None = None,
        task_space_validity_fn: Callable[[np.ndarray], bool] | None = None,
    ) -> None:
        super().__init__(
            robot=robot,
            name=name,
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            task_space_validity_fn=task_space_validity_fn,
        )
        self.point = np.asarray(point, dtype=float).reshape(3)
        normal_arr = np.asarray(normal, dtype=float).reshape(3)
        norm = float(np.linalg.norm(normal_arr))
        if norm <= 1e-12:
            raise ValueError("Plane normal must be nonzero.")
        self.normal = normal_arr / norm

    def residual(self, theta: np.ndarray) -> np.ndarray:
        q = self._coerce_point(theta)
        ee = np.asarray(self.robot.forward_kinematics_3d(q)[-1], dtype=float)
        value = float(np.dot(ee - self.point, self.normal))
        return np.asarray([value], dtype=float)

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        q = self._coerce_point(theta)
        ee_jac = robot_fk_jacobian(self.robot, q)
        grad = self.normal.reshape(1, 3) @ ee_jac
        return np.asarray(grad, dtype=float)


__all__ = [
    "RobotPlaneManifold",
    "RobotSphereManifold",
    "robot_fk_jacobian",
]
