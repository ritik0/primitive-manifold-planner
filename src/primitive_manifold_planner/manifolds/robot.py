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
    """Finite-difference Jacobian d FK(theta) / d theta for the end-effector.

    ``theta`` is the 3-DOF joint vector. The returned 3x3 matrix maps a small
    joint-space change to the corresponding end-effector FK displacement.
    """

    q = np.asarray(theta, dtype=float).reshape(3) #makes sure that it has only 3 values 
    base = np.asarray(robot.forward_kinematics_3d(q)[-1], dtype=float) #computes current ee and returns p3(-1)
    jac = np.zeros((3, 3), dtype=float) #empty 3x3
    for idx in range(3):  #loop over yaw, shoulder elbow joints
        perturbed = q.copy() #copy making
        perturbed[idx] += float(eps) # add a tiny amount to one joint 
        value = np.asarray(robot.forward_kinematics_3d(perturbed)[-1], dtype=float) #compute the ee pos after tiny addition
        jac[:, idx] = (value - base) / float(eps) #finite diff derivative
    return jac #provides dFK/dtheta


class _RobotConstraintBase(ImplicitManifold):
    """Base class for FK-pulled-back robot constraints in theta space.

    Subclasses define a workspace residual at FK(theta). This base handles
    joint bounds, optional task-space validity masks, and projection helpers.
    """

    def __init__(
        self,
        robot: Any,
        name: str,
        joint_lower: np.ndarray | None = None,
        joint_upper: np.ndarray | None = None,
        task_space_validity_fn: Callable[[np.ndarray], bool] | None = None,
    ) -> None:
        super().__init__(ambient_dim=3, codim=1, name=name) # 3 values of theta 1 scalar equation (residual = 0) so 2d surface in theta space
        self.robot = robot
        self.joint_lower = (
            np.asarray(joint_lower, dtype=float).reshape(3)
            if joint_lower is not None
            else -np.pi * np.ones(3, dtype=float)
        )
        self.joint_upper = (
            np.asarray(joint_upper, dtype=float).reshape(3)
            if joint_upper is not None
            else np.pi * np.ones(3, dtype=float) #values provided use those or else (-pi, pi)
        )
        self.task_space_validity_fn = task_space_validity_fn

    def within_bounds(self, theta: np.ndarray, tol: float = 1e-9) -> bool: #checks joint limits and task space validity
        q = self._coerce_point(theta) #converts into expected format
        joint_ok = bool(np.all(q >= self.joint_lower - tol) and np.all(q <= self.joint_upper + tol))
        if not joint_ok:
            return False #theta is invalid 
        if self.task_space_validity_fn is None:
            return True
        ee = np.asarray(self.robot.forward_kinematics_3d(q)[-1], dtype=float) #ee xyz
        return bool(self.task_space_validity_fn(ee))

    def validate(self, theta: np.ndarray, tol: float = 1e-6) -> bool: #checks 
        """Return True when theta satisfies the FK-pulled-back constraint."""
        q = self._coerce_point(theta)
        residual_ok = bool(np.linalg.norm(self.residual(q)) <= float(tol)) #residual near zero
        return bool(residual_ok and self.within_bounds(q, tol=tol)) #constraint eq satisfied and bounds too

    def get_implicit_function(self) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
        def fn(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            q = self._coerce_point(theta)
            return self.residual(q), self.jacobian(q)

        return fn #returns residual and jacobian

    def project(
        self,
        theta_guess: np.ndarray,
        tol: float = 1e-8,
        max_iters: int = 80,
    ) -> ProjectionResult:
        """Project a theta guess onto the active manifold with bounds.

        This version may move to any feasible branch and is useful for broad
        joint-space exploration.
        """

        q0 = np.asarray(theta_guess, dtype=float).reshape(3)
        q0 = np.clip(q0, self.joint_lower, self.joint_upper) #clamps to -pi to pi

        if least_squares is not None:
            result = least_squares(
                lambda q: self.residual(q), #make this zero
                q0, #inital guess 
                jac=lambda q: self.jacobian(q), #analytical jacobian 
                bounds=(self.joint_lower, self.joint_upper), #bounds adv over newton
                max_nfev=max_iters,
                xtol=tol,
                ftol=tol,
                gtol=tol,
            )
            q_proj = np.asarray(result.x, dtype=float) #optimized theta
            residual_norm = float(np.linalg.norm(self.residual(q_proj))) #final error
            success = bool(result.success and residual_norm <= max(10.0 * tol, 1e-6) and self.within_bounds(q_proj)) #residual and theta passes 
            return ProjectionResult(
                success=success,
                x_projected=q_proj,
                residual_norm=residual_norm,
                iterations=int(getattr(result, "nfev", max_iters)),
                converged=success,
                message=str(getattr(result, "message", "least_squares finished")),
            )

        projected = project_newton(self, q0, tol=tol, max_iters=max_iters, damping=1.0) #if scipy not available 
        projected.success = bool(projected.success and self.within_bounds(projected.x_projected))
        projected.converged = bool(projected.converged and projected.success)
        return projected

    def project_local( # stays close to current theta guess 
        self,
        theta_guess: np.ndarray,
        tol: float = 1e-8,
        max_iters: int = 80,
        regularization: float = 1.0e-2,
    ) -> ProjectionResult:
        """Branch-preserving projection for final execution, not exploration.

        The regularization term keeps the result near ``theta_guess`` while
        still driving the active manifold residual toward zero.
        """

        q0 = np.asarray(theta_guess, dtype=float).reshape(3)
        q0 = np.clip(q0, self.joint_lower, self.joint_upper)

        if least_squares is not None:
            reg = float(regularization) #preserves branch that's being explored

            def objective(q: np.ndarray) -> np.ndarray:
                return np.concatenate([self.residual(q), reg * (np.asarray(q, dtype=float) - q0)])#constraint error + closeness error

            def objective_jacobian(q: np.ndarray) -> np.ndarray:
                return np.vstack([self.jacobian(q), reg * np.eye(3, dtype=float)]) #jacobian + closeness penalty jacobian 

            result = least_squares(
                objective,
                q0,
                jac=objective_jacobian,
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
                message=str(getattr(result, "message", "regularized least_squares finished")),
            )

        return self.project(q0, tol=tol, max_iters=max_iters)


class RobotSphereManifold(_RobotConstraintBase):
    """Joint-space manifold for the workspace sphere constraint.

    The residual is ||FK(theta)-center||^2-r^2, so valid theta values put the
    robot end-effector on the sphere.
    """

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

    def residual(self, theta: np.ndarray) -> np.ndarray: #SPHERE CONSTRAINT 
        q = self._coerce_point(theta)
        ee = np.asarray(self.robot.forward_kinematics_3d(q)[-1], dtype=float)
        dx = ee - self.center
        return np.asarray([float(np.dot(dx, dx) - self.radius**2)], dtype=float) #||ee - center||^2 - radius^2

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        q = self._coerce_point(theta)
        ee = np.asarray(self.robot.forward_kinematics_3d(q)[-1], dtype=float)
        ee_jac = robot_fk_jacobian(self.robot, q) #3x3 computes dee/dtheta
        # Chain rule: d sphere_residual / d theta = d residual / d FK * d FK / d theta.
        grad = 2.0 * (ee - self.center).reshape(1, 3) @ ee_jac #jacobian shows the change in row vector
        return np.asarray(grad, dtype=float) 


class RobotPlaneManifold(_RobotConstraintBase):
    """Joint-space manifold for the workspace plane constraint.

    The residual is n dot (FK(theta)-point), so valid theta values put the
    end-effector on the selected plane.
    """

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
        self.point = np.asarray(point, dtype=float).reshape(3) # a point on the plane 
        normal_arr = np.asarray(normal, dtype=float).reshape(3) # plane normal vector
        norm = float(np.linalg.norm(normal_arr)) #length of normal
        if norm <= 1e-12: #non zero checking 
            raise ValueError("Plane normal must be nonzero.")
        self.normal = normal_arr / norm #length has to be 1 

    def residual(self, theta: np.ndarray) -> np.ndarray:
        q = self._coerce_point(theta)
        ee = np.asarray(self.robot.forward_kinematics_3d(q)[-1], dtype=float)
        value = float(np.dot(ee - self.point, self.normal)) #computes (normal - fk(theta) - point) has to be zero for on it.
        return np.asarray([value], dtype=float)

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        q = self._coerce_point(theta) 
        ee_jac = robot_fk_jacobian(self.robot, q) 
        # Chain rule pulls the workspace plane normal back through FK(theta).
        grad = self.normal.reshape(1, 3) @ ee_jac 
        return np.asarray(grad, dtype=float)


__all__ = [*
    "RobotPlaneManifold",
    "RobotSphereManifold",
    "robot_fk_jacobian",
]
