from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

from .modes import ConstraintMode
from .transitions import TransitionConstraint


def _wrap_joint_delta(delta: np.ndarray) -> np.ndarray: #measures how big of a step next step is 
    return (np.asarray(delta, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _joint_steps(theta_path: np.ndarray) -> np.ndarray: #joint space distance between two cons theta points
    path = np.asarray(theta_path, dtype=float)
    if len(path) < 2:
        return np.zeros(0, dtype=float)
    return np.linalg.norm(_wrap_joint_delta(np.diff(path, axis=0)), axis=1) #to check that final path isn't jumping around


def _stage_order_valid(labels: list[str], order: tuple[str, ...] | None) -> bool: #if seq is given then it should be followed
    if order is None:
        return bool(labels)
    index = {stage: idx for idx, stage in enumerate(order)} #dic mapping 
    values = [index.get(str(label), -1) for label in labels] #into indexes 
    return bool(values and all(value >= 0 for value in values) and all(b >= a for a, b in zip(values[:-1], values[1:]))) #only works if order is not backwards


def _lambda_fixed(lambda_labels: np.ndarray) -> tuple[bool, float]: # checks for fixed lambda in continous mode 
    values = np.asarray(lambda_labels, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return True, 0.0
    variation = float(np.max(finite) - np.min(finite))
    return bool(variation <= 1.0e-6), variation #allows for lambda to vary by the tol (so small making sure that its same plane)


@dataclass(frozen=True)
class RouteCertification:
    """Summary of checks for a certified dense joint route.

    The booleans separate active-manifold residuals, joint continuity,
    transition configurations, stage order, fixed lambda leaf use, and the
    thesis rule that the final route must not contain task-space edges.
    """

    success: bool
    constraint_certified: bool
    joint_continuity_certified: bool
    transition_stack_certified: bool
    stage_order_valid: bool
    lambda_fixed_during_transfer: bool
    collision_free: bool
    max_constraint_residual: float
    mean_constraint_residual: float
    max_joint_step: float
    mean_joint_step: float
    max_transition_stack_residual: float
    final_route_taskspace_edges: int
    message: str
    residuals: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    joint_steps: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    lambda_variation_transfer: float = 0.0


def certify_dense_joint_route(
    *,
    dense_theta_path: np.ndarray,
    stage_labels: list[str],
    modes_by_stage: Mapping[str, ConstraintMode],
    transition_constraints: Mapping[str, TransitionConstraint] | None = None,
    transition_thetas: Mapping[str, np.ndarray] | None = None,
    lambda_labels: np.ndarray | None = None,
    selected_lambda: float | None = None,
    max_joint_step: float = 0.12,
    collision_fn: Callable[[np.ndarray], bool] | None = None,
    tolerance: float = 2.0e-3,
    transition_tolerance: float = 1.0e-3,
    stage_order: tuple[str, ...] | None = ("left", "plane", "family", "right"),
    final_route_taskspace_edges: int = 0,
) -> RouteCertification:
    """Certify a dense theta route against active stage constraints.

    The function is intentionally planner-agnostic. It works for Example 66
    (left/plane/right) and Example 65 (left/family/right) by changing
    ``modes_by_stage`` and optional transition constraints.
    """

    theta_path = np.asarray(dense_theta_path, dtype=float) #invalid path checks, correct shape
    if theta_path.ndim != 2 or theta_path.shape[1] != 3 or len(theta_path) == 0:
        return RouteCertification(
            success=False,
            constraint_certified=False,
            joint_continuity_certified=False,
            transition_stack_certified=False,
            stage_order_valid=False,
            lambda_fixed_during_transfer=False,
            collision_free=False,
            max_constraint_residual=float("inf"),
            mean_constraint_residual=float("inf"),
            max_joint_step=float("inf"),
            mean_joint_step=float("inf"),
            max_transition_stack_residual=float("inf"),
            final_route_taskspace_edges=int(final_route_taskspace_edges),
            message="empty or invalid dense theta path",
        )
    labels = [str(label) for label in stage_labels]
    if len(labels) != len(theta_path):
        return RouteCertification(
            success=False,
            constraint_certified=False,
            joint_continuity_certified=False,
            transition_stack_certified=False,
            stage_order_valid=False,
            lambda_fixed_during_transfer=False,
            collision_free=False,
            max_constraint_residual=float("inf"),
            mean_constraint_residual=float("inf"),
            max_joint_step=float("inf"),
            mean_joint_step=float("inf"),
            max_transition_stack_residual=float("inf"),
            final_route_taskspace_edges=int(final_route_taskspace_edges),
            message="stage label count does not match dense theta path",
        )

    residuals: list[float] = []
    collision_free = True
    # Per-waypoint residuals are evaluated against that waypoint's active manifold.
    for theta, stage in zip(theta_path, labels):
        mode = modes_by_stage.get(stage)
        residual = float("inf") if mode is None else float(np.linalg.norm(mode.residual(theta)))
        if mode is None or not mode.within_bounds(theta, tol=tolerance):
            residual = max(residual, 10.0 * float(tolerance))
        if collision_fn is not None and bool(collision_fn(np.asarray(theta, dtype=float))):
            collision_free = False
            residual = max(residual, 100.0 * float(tolerance))
        residuals.append(residual)
    residual_arr = np.asarray(residuals, dtype=float)
    # Joint continuity checks wrapped theta steps, not task-space distance.
    joint_steps = _joint_steps(theta_path)
    max_residual = float(np.max(residual_arr)) if len(residual_arr) else float("inf")
    mean_residual = float(np.mean(residual_arr)) if len(residual_arr) else float("inf")
    max_step = float(np.max(joint_steps)) if len(joint_steps) else 0.0
    mean_step = float(np.mean(joint_steps)) if len(joint_steps) else 0.0

    stack_values: list[float] = []
    if transition_constraints and transition_thetas:
        for name, constraint in transition_constraints.items():
            theta = transition_thetas.get(name)
            if theta is None:
                stack_values.append(float("inf"))
            else:
                # A transition theta is valid only if the stacked source/target residual is small.
                stack_values.append(float(constraint.residual_norm(theta)))
    max_transition = float(max(stack_values)) if stack_values else 0.0

    # Family-transfer routes must stay on one fixed lambda leaf during transfer.
    lambda_arr = (
        np.asarray(lambda_labels, dtype=float)
        if lambda_labels is not None
        else np.asarray([float(selected_lambda) if selected_lambda is not None and label == "family" else np.nan for label in labels])
    )
    lambda_fixed, lambda_variation = _lambda_fixed(lambda_arr)

    constraint_ok = bool(collision_free and max_residual <= float(tolerance))
    joint_ok = bool(max_step <= float(max_joint_step) + 1e-9)
    transition_ok = bool(max_transition <= float(transition_tolerance))
    order_ok = _stage_order_valid(labels, stage_order)
    # Joint-space thesis routes should not be reconstructed from task-space edges.
    taskspace_edge_ok = int(final_route_taskspace_edges) == 0
    success = bool(constraint_ok and joint_ok and transition_ok and order_ok and lambda_fixed and taskspace_edge_ok)
    message = "dense joint route certified" if success else (
        "dense joint route failed certification: "
        f"constraint={constraint_ok}, joint={joint_ok}, transition={transition_ok}, "
        f"stage_order={order_ok}, lambda_fixed={lambda_fixed}, taskspace_edges={taskspace_edge_ok}"
    )
    return RouteCertification(
        success=success,
        constraint_certified=constraint_ok,
        joint_continuity_certified=joint_ok,
        transition_stack_certified=transition_ok,
        stage_order_valid=order_ok,
        lambda_fixed_during_transfer=lambda_fixed,
        collision_free=bool(collision_free),
        max_constraint_residual=max_residual,
        mean_constraint_residual=mean_residual,
        max_joint_step=max_step,
        mean_joint_step=mean_step,
        max_transition_stack_residual=max_transition,
        final_route_taskspace_edges=int(final_route_taskspace_edges),
        message=message,
        residuals=residual_arr,
        joint_steps=joint_steps,
        lambda_variation_transfer=float(lambda_variation),
    )
