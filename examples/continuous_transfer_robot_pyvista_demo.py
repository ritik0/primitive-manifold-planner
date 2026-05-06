from __future__ import annotations

"""Continuous-transfer planner with 3DOF robot task-space execution.

Phase 1 intentionally keeps planning in task space:
- the continuous-transfer evidence graph discovers entry/exit/lambda structure;
- the final route is the selected-transition local replan from that planner;
- the robot tracks only that final route with sequential IK.

Planner evidence is diagnostic context. Robot motion is not allowed to follow
raw exploration branches.
"""

import argparse
from dataclasses import dataclass
import os
import sys
from pathlib import Path

import numpy as np
from ompl import util as ou

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXAMPLES = ROOT / "examples"
for path in (ROOT, SRC, EXAMPLES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from primitive_manifold_planner.examplesupport.intrinsic_multimodal_helpers import build_segment_polydata
from primitive_manifold_planner.examplesupport.jointspace_planner_utils import (
    end_effector_point,
    explore_joint_manifold,
    generate_joint_proposals,
    inverse_kinematics_start,
    joint_path_to_task_path,
    joint_step_statistics,
    wrap_joint_angles,
)
from primitive_manifold_planner.experiments.continuous_transfer import (
    build_continuous_transfer_scene,
    default_example_65_scene_description,
    plan_continuous_transfer_route,
    print_continuous_route_summary,
)
from primitive_manifold_planner.experiments.continuous_transfer.config import DEFAULT_POST_ROUTE_EVIDENCE_ROUNDS
from primitive_manifold_planner.experiments.continuous_transfer.family_definition import plane_leaf_patch
from primitive_manifold_planner.manifolds.robot import RobotPlaneManifold, RobotSphereManifold
from primitive_manifold_planner.visualization import add_manifold, add_points, pyvista_available
from primitive_manifold_planner.visualization.robot import (
    add_robot_pedestal,
    make_robot_actor_bundle,
    update_robot_actor_bundle,
)

from three_dof_robot_pyvista_demo import (
    RobotExecutionResult,
    build_continuous_robot_execution_path,
    choose_robot_for_route,
)

try:
    import pyvista as pv
except Exception:
    pv = None

try:
    from scipy.optimize import least_squares
except Exception:
    least_squares = None


LEFT_STAGE = "left"
FAMILY_STAGE = "family"
RIGHT_STAGE = "right"


def _polyline_error(reference: np.ndarray, trace: np.ndarray) -> tuple[float, float]:
    ref = np.asarray(reference, dtype=float)
    pts = np.asarray(trace, dtype=float)
    if len(ref) == 0 or len(pts) == 0:
        return float("inf"), float("inf")
    if len(ref) == len(pts):
        errors = np.linalg.norm(ref - pts, axis=1)
    else:
        errors = np.asarray([float(np.min(np.linalg.norm(ref - p, axis=1))) for p in pts], dtype=float)
    return float(np.max(errors)), float(np.mean(errors))


def _print_key_value_block(title: str, values: dict[str, object]) -> None:
    print(f"\n=== {title} ===")
    width = max((len(str(key)) for key in values), default=1)
    for key, value in values.items():
        print(f"{key.ljust(width)} : {value}")


class RobotPlaneLeafManifold(RobotPlaneManifold):
    """Robot configuration-space constraint for one locked continuous-transfer leaf.

    The planner state is q, but the leaf parameter is interpreted in task
    space: FK(q) must lie on the selected plane leaf and inside that leaf's
    validity patch. This is the first reusable brick for the later full
    robot-side leaf-store manager.
    """

    def __init__(
        self,
        *,
        robot,
        transfer_family,
        lambda_value: float,
        name: str = "robot_transfer_family_leaf",
        joint_lower: np.ndarray | None = None,
        joint_upper: np.ndarray | None = None,
    ) -> None:
        self.transfer_family = transfer_family
        self.lambda_value = float(lambda_value)
        super().__init__(
            robot=robot,
            point=transfer_family.point_on_leaf(float(lambda_value)),
            normal=transfer_family.normal,
            name=f"{name}_lambda_{float(lambda_value):.6f}",
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            task_space_validity_fn=lambda ee, family=transfer_family, lam=float(lambda_value): bool(
                family.within_patch(lam, np.asarray(ee, dtype=float), tol=2.0e-3)
            ),
        )

    def infer_lambda(self, theta: np.ndarray) -> float:
        return float(self.transfer_family.infer_lambda(end_effector_point(self.robot, theta)))


@dataclass
class JointspaceRouteRealization:
    success: bool
    dense_joint_path: np.ndarray
    stage_labels: list[str]
    task_path: np.ndarray
    residuals: np.ndarray
    joint_steps: np.ndarray
    max_constraint_residual: float
    mean_constraint_residual: float
    max_joint_step: float
    mean_joint_step: float
    collision_free: bool
    message: str
    segment_messages: dict[str, str]


@dataclass
class RobotFamilyLeafStore:
    lambda_value: float
    manifold: RobotPlaneLeafManifold
    nodes: list[np.ndarray]
    edges: list[tuple[int, int]]
    frontier_ids: list[int]

    def add_node(self, q: np.ndarray, dedup_tol: float = 1.0e-4) -> int:
        q_arr = np.asarray(q, dtype=float).reshape(3)
        for idx, existing in enumerate(self.nodes):
            if float(np.linalg.norm(wrap_joint_angles(q_arr - existing))) <= float(dedup_tol):
                return int(idx)
        self.nodes.append(q_arr)
        node_id = len(self.nodes) - 1
        self.frontier_ids.append(node_id)
        self.frontier_ids = self.frontier_ids[-72:]
        return int(node_id)


@dataclass
class RobotStageEvidenceStore:
    name: str
    manifold: object
    nodes: list[np.ndarray]
    edges: list[tuple[int, int]]
    frontier_ids: list[int]

    def add_node(self, q: np.ndarray, dedup_tol: float = 1.0e-4) -> int:
        q_arr = np.asarray(q, dtype=float).reshape(3)
        for idx, existing in enumerate(self.nodes):
            if float(np.linalg.norm(wrap_joint_angles(q_arr - existing))) <= float(dedup_tol):
                return int(idx)
        self.nodes.append(q_arr)
        node_id = len(self.nodes) - 1
        self.frontier_ids.append(node_id)
        self.frontier_ids = self.frontier_ids[-72:]
        return int(node_id)


@dataclass
class RobotJointspaceContinuousEvidence:
    left_store: RobotStageEvidenceStore
    right_store: RobotStageEvidenceStore
    family_leaf_stores: dict[float, RobotFamilyLeafStore]
    counters: dict[str, int]

    def get_or_create_family_store(self, robot, transfer_family, lam: float) -> RobotFamilyLeafStore:
        key = _lambda_key(float(lam))
        existing = self.family_leaf_stores.get(key)
        if existing is not None:
            return existing
        manifold = RobotPlaneLeafManifold(
            robot=robot,
            transfer_family=transfer_family,
            lambda_value=float(key),
            name="full_jointspace_family_leaf",
        )
        store = RobotFamilyLeafStore(
            lambda_value=float(key),
            manifold=manifold,
            nodes=[],
            edges=[],
            frontier_ids=[],
        )
        self.family_leaf_stores[key] = store
        return store


def _lambda_key(lam: float, digits: int = 6) -> float:
    return round(float(lam), int(digits))


def _scene_candidate_lambdas(scene_description: dict[str, object], transfer_family) -> list[float]:
    transfer_spec = dict(scene_description.get("transfer_family", {})) if isinstance(scene_description, dict) else {}
    values = transfer_spec.get("plane_offsets", None)
    if values is None:
        values = list(transfer_family.sample_lambdas({"count": 10}))
    result: list[float] = []
    for value in values:
        lam = float(value)
        if transfer_family.lambda_in_range(lam, tol=1.0e-9):
            result.append(_lambda_key(lam))
    return sorted(set(result))


def candidate_lambdas_for_joint_proposal(
    q: np.ndarray,
    robot,
    transfer_family,
    active_lambdas: list[float],
    scene_lambdas: list[float],
) -> list[float]:
    ee = end_effector_point(robot, q)
    candidates: list[float] = []
    inferred = float(transfer_family.infer_lambda(ee))
    if transfer_family.lambda_in_range(inferred, tol=5.0e-2):
        candidates.append(float(np.clip(inferred, transfer_family.lambda_min, transfer_family.lambda_max)))
    combined = sorted(set(float(v) for v in [*scene_lambdas, *active_lambdas]))
    if combined:
        nearest = sorted(combined, key=lambda lam: abs(lam - inferred))[:3]
        candidates.extend(nearest)
    candidates.extend(active_lambdas[-4:])
    deduped: list[float] = []
    for lam in candidates:
        clamped = _lambda_key(float(np.clip(lam, transfer_family.lambda_min, transfer_family.lambda_max)))
        if transfer_family.lambda_in_range(clamped, tol=1.0e-9) and all(abs(clamped - prev) > 1.0e-4 for prev in deduped):
            deduped.append(clamped)
    return deduped


def _project_joint_to_store(store, q: np.ndarray, counter_prefix: str, counters: dict[str, int]) -> bool:
    projection = store.manifold.project(q, tol=1.0e-6, max_iters=100)
    if projection.success and bool(store.manifold.within_bounds(projection.x_projected, tol=2.0e-3)):
        store.add_node(np.asarray(projection.x_projected, dtype=float))
        counters[f"projected_to_{counter_prefix}_count"] = int(counters.get(f"projected_to_{counter_prefix}_count", 0)) + 1
        return True
    counters[f"{counter_prefix}_projection_failure_count"] = int(counters.get(f"{counter_prefix}_projection_failure_count", 0)) + 1
    return False


def project_joint_proposal_to_supports(
    q_proposal: np.ndarray,
    evidence: RobotJointspaceContinuousEvidence,
    robot,
    families,
    scene_lambdas: list[float],
) -> None:
    _left_family, transfer_family, _right_family = families
    counters = evidence.counters
    q = np.asarray(q_proposal, dtype=float).reshape(3)
    _project_joint_to_store(evidence.left_store, q, "left", counters)
    _project_joint_to_store(evidence.right_store, q, "right", counters)
    active_lambdas = sorted(evidence.family_leaf_stores.keys())
    candidate_lambdas = candidate_lambdas_for_joint_proposal(q, robot, transfer_family, active_lambdas, scene_lambdas)
    counters["candidate_lambdas_evaluated"] = int(counters.get("candidate_lambdas_evaluated", 0)) + len(candidate_lambdas)
    if not candidate_lambdas:
        counters["family_projection_failure_count"] = int(counters.get("family_projection_failure_count", 0)) + 1
    for lam in candidate_lambdas:
        store = evidence.get_or_create_family_store(robot, transfer_family, float(lam))
        _project_joint_to_store(store, q, "family", counters)


def build_robot_manifolds_for_selected_lambda(robot, families, selected_lambda: float) -> dict[str, object]:
    left_family, transfer_family, right_family = families
    left_geom = left_family.manifold(float(left_family.sample_lambdas()[0]))
    right_geom = right_family.manifold(float(right_family.sample_lambdas()[0]))
    joint_lower = -np.pi * np.ones(3, dtype=float)
    joint_upper = np.pi * np.ones(3, dtype=float)
    return {
        LEFT_STAGE: RobotSphereManifold(
            robot=robot,
            center=np.asarray(left_geom.center, dtype=float),
            radius=float(left_geom.radius),
            name="continuous_robot_left_sphere",
            joint_lower=joint_lower,
            joint_upper=joint_upper,
        ),
        FAMILY_STAGE: RobotPlaneLeafManifold(
            robot=robot,
            transfer_family=transfer_family,
            lambda_value=float(selected_lambda),
            name="continuous_robot_family_leaf",
            joint_lower=joint_lower,
            joint_upper=joint_upper,
        ),
        RIGHT_STAGE: RobotSphereManifold(
            robot=robot,
            center=np.asarray(right_geom.center, dtype=float),
            radius=float(right_geom.radius),
            name="continuous_robot_right_sphere",
            joint_lower=joint_lower,
            joint_upper=joint_upper,
        ),
    }


def _project_to_manifold(manifold, q: np.ndarray, *, local: bool = True):
    if local and hasattr(manifold, "project_local"):
        projection = manifold.project_local(q, tol=1.0e-7, max_iters=120, regularization=2.0e-3)
        if projection.success:
            return np.asarray(projection.x_projected, dtype=float)
    projection = manifold.project(q, tol=1.0e-7, max_iters=120)
    if not projection.success:
        return None
    return np.asarray(projection.x_projected, dtype=float)


def _solve_shared_task_handoff(
    *,
    robot,
    target_task: np.ndarray,
    manifolds: list[object],
    warm_starts: list[np.ndarray | None],
    collision_fn=None,
    task_tol: float = 8.0e-2,
    residual_tol: float = 2.0e-3,
) -> tuple[np.ndarray | None, str]:
    target = np.asarray(target_task, dtype=float).reshape(3)
    joint_lower = -np.pi * np.ones(3, dtype=float)
    joint_upper = np.pi * np.ones(3, dtype=float)
    guesses: list[np.ndarray] = []
    for warm in warm_starts:
        q = inverse_kinematics_start(
            robot,
            target,
            warm_start=warm,
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            tol=max(float(task_tol), 8.0e-2),
        )
        if q is not None:
            guesses.append(np.asarray(q, dtype=float))
    if len(guesses) >= 2:
        guesses.append(wrap_joint_angles(0.5 * (guesses[0] + guesses[-1])))
    guesses.extend(
        [
            np.asarray([0.0, 0.55, -0.85], dtype=float),
            np.asarray([0.0, 0.85, -0.65], dtype=float),
            np.asarray([0.0, 0.20, 0.35], dtype=float),
        ]
    )

    best_q: np.ndarray | None = None
    best_score = float("inf")
    best_detail = "no candidate evaluated"

    def candidate_score(q: np.ndarray) -> tuple[float, float, float, bool]:
        ee = end_effector_point(robot, q)
        task_error = float(np.linalg.norm(ee - target))
        residual = max(float(np.linalg.norm(manifold.residual(q))) for manifold in manifolds)
        bounds_ok = all(bool(manifold.within_bounds(q, tol=residual_tol)) for manifold in manifolds)
        collision_ok = True if collision_fn is None else not bool(collision_fn(q))
        score = task_error + 10.0 * residual + (0.0 if bounds_ok and collision_ok else 100.0)
        return score, task_error, residual, bool(bounds_ok and collision_ok)

    for guess in guesses:
        q0 = np.clip(wrap_joint_angles(np.asarray(guess, dtype=float).reshape(3)), joint_lower, joint_upper)
        candidates = [q0]
        if least_squares is not None:

            def objective(q: np.ndarray) -> np.ndarray:
                q_arr = np.asarray(q, dtype=float)
                residual_parts = [end_effector_point(robot, q_arr) - target]
                residual_parts.extend(np.asarray(manifold.residual(q_arr), dtype=float).reshape(-1) for manifold in manifolds)
                residual_parts.append(2.0e-3 * wrap_joint_angles(q_arr - q0))
                return np.concatenate(residual_parts)

            try:
                solve = least_squares(
                    objective,
                    q0,
                    bounds=(joint_lower, joint_upper),
                    max_nfev=160,
                    xtol=1.0e-9,
                    ftol=1.0e-9,
                    gtol=1.0e-9,
                )
                candidates.insert(0, wrap_joint_angles(np.asarray(solve.x, dtype=float)))
            except Exception:
                pass
        for q_candidate in candidates:
            q = np.asarray(q_candidate, dtype=float)
            # Polish each candidate on all active manifolds. This keeps the
            # handoff shared while using branch-preserving projection locally.
            for manifold in manifolds:
                polished = _project_to_manifold(manifold, q, local=True)
                if polished is not None:
                    q = polished
            score, task_error, residual, ok = candidate_score(q)
            if score < best_score:
                best_score = float(score)
                best_q = np.asarray(q, dtype=float)
                best_detail = f"task_error={task_error:.4g}, max_residual={residual:.4g}, bounds_collision_ok={ok}"
            if ok and task_error <= float(task_tol) and residual <= float(residual_tol):
                return np.asarray(q, dtype=float), f"shared handoff solved ({best_detail})"

    return None, f"shared handoff failed ({best_detail})"


def _connect_joint_segment(stage: str, manifold, q0: np.ndarray, q1: np.ndarray, joint_max_step: float, collision_fn=None):
    result = explore_joint_manifold(
        manifold,
        np.asarray(q0, dtype=float),
        np.asarray(q1, dtype=float),
        max_step=float(joint_max_step),
        projection_tol=1.0e-6,
        collision_fn=collision_fn,
        local_max_joint_step=float(joint_max_step),
    )
    if not result.success:
        return None, f"{stage} segment failed: {result.message}"
    return np.asarray(result.path, dtype=float), f"{stage} segment succeeded: {result.message}, points={len(result.path)}"


def _sphere_task_arc(center: np.ndarray, radius: float, start: np.ndarray, goal: np.ndarray, step: float = 0.035) -> np.ndarray:
    c = np.asarray(center, dtype=float)
    r = float(radius)
    a = (np.asarray(start, dtype=float) - c) / max(r, 1.0e-12)
    b = (np.asarray(goal, dtype=float) - c) / max(r, 1.0e-12)
    a = a / max(float(np.linalg.norm(a)), 1.0e-12)
    b = b / max(float(np.linalg.norm(b)), 1.0e-12)
    omega = float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))
    count = max(2, int(np.ceil((r * omega) / max(float(step), 1.0e-6))) + 1)
    if omega <= 1.0e-9:
        return np.asarray([np.asarray(start, dtype=float), np.asarray(goal, dtype=float)], dtype=float)
    points = []
    for alpha in np.linspace(0.0, 1.0, count):
        u = (
            np.sin((1.0 - alpha) * omega) / np.sin(omega) * a
            + np.sin(alpha * omega) / np.sin(omega) * b
        )
        points.append(c + r * u)
    return np.asarray(points, dtype=float)


def _linear_task_path(start: np.ndarray, goal: np.ndarray, step: float = 0.035) -> np.ndarray:
    a = np.asarray(start, dtype=float)
    b = np.asarray(goal, dtype=float)
    distance = float(np.linalg.norm(b - a))
    count = max(2, int(np.ceil(distance / max(float(step), 1.0e-6))) + 1)
    return np.asarray([(1.0 - alpha) * a + alpha * b for alpha in np.linspace(0.0, 1.0, count)], dtype=float)


def _stage_task_waypoints(stage: str, manifold, task0: np.ndarray, task1: np.ndarray) -> np.ndarray:
    if stage in {LEFT_STAGE, RIGHT_STAGE} and hasattr(manifold, "center") and hasattr(manifold, "radius"):
        return _sphere_task_arc(manifold.center, manifold.radius, task0, task1)
    return _linear_task_path(task0, task1)


def _connect_joint_segment_by_task_waypoints(
    stage: str,
    manifold,
    q0: np.ndarray,
    q1: np.ndarray,
    task0: np.ndarray,
    task1: np.ndarray,
    joint_max_step: float,
    collision_fn=None,
) -> tuple[np.ndarray | None, str]:
    waypoints = _stage_task_waypoints(stage, manifold, task0, task1)
    path: list[np.ndarray] = [np.asarray(q0, dtype=float)]
    current = np.asarray(q0, dtype=float)
    for idx, target in enumerate(waypoints[1:], start=1):
        q_ik = inverse_kinematics_start(
            manifold.robot,
            np.asarray(target, dtype=float),
            warm_start=current,
            joint_lower=getattr(manifold, "joint_lower", None),
            joint_upper=getattr(manifold, "joint_upper", None),
            tol=6.0e-2,
        )
        if q_ik is None:
            return None, f"{stage} task-waypoint connector failed: IK failed at waypoint {idx}/{len(waypoints)-1}"
        q_next = _project_to_manifold(manifold, q_ik, local=True)
        if q_next is None:
            return None, f"{stage} task-waypoint connector failed: projection failed at waypoint {idx}/{len(waypoints)-1}"
        if not bool(manifold.within_bounds(q_next, tol=2.0e-3)):
            return None, f"{stage} task-waypoint connector failed: bounds/patch violation at waypoint {idx}/{len(waypoints)-1}"
        if collision_fn is not None and bool(collision_fn(q_next)):
            return None, f"{stage} task-waypoint connector failed: collision at waypoint {idx}/{len(waypoints)-1}"
        local_step = float(np.linalg.norm(wrap_joint_angles(q_next - current)))
        task_error = float(np.linalg.norm(end_effector_point(manifold.robot, q_next) - np.asarray(target, dtype=float)))
        residual = float(np.linalg.norm(manifold.residual(q_next)))
        if local_step > float(joint_max_step) + 1.0e-9:
            return None, (
                f"{stage} task-waypoint connector failed: joint step {local_step:.4g} "
                f"exceeds limit {float(joint_max_step):.4g} at waypoint {idx}/{len(waypoints)-1}"
            )
        if task_error > 8.0e-2 or residual > 2.0e-3:
            return None, (
                f"{stage} task-waypoint connector failed: task_error={task_error:.4g}, "
                f"residual={residual:.4g} at waypoint {idx}/{len(waypoints)-1}"
            )
        path.append(np.asarray(q_next, dtype=float))
        current = np.asarray(q_next, dtype=float)

    if float(np.linalg.norm(wrap_joint_angles(path[-1] - np.asarray(q1, dtype=float)))) <= float(joint_max_step) + 1.0e-9:
        path[-1] = np.asarray(q1, dtype=float)
    else:
        return None, f"{stage} task-waypoint connector failed: final q is not continuous with solved endpoint"
    return np.asarray(path, dtype=float), f"{stage} task-waypoint connector succeeded, points={len(path)}"


def _concatenate_labeled_segments(segments: list[tuple[str, np.ndarray]]) -> tuple[np.ndarray, list[str]]:
    path_parts: list[np.ndarray] = []
    labels: list[str] = []
    for segment_index, (stage, segment) in enumerate(segments):
        arr = np.asarray(segment, dtype=float)
        if len(arr) == 0:
            continue
        if segment_index > 0:
            arr = arr[1:]
        path_parts.append(arr)
        labels.extend([stage] * len(arr))
    if not path_parts:
        return np.zeros((0, 3), dtype=float), []
    return np.vstack(path_parts), labels


def _certify_labeled_joint_path(
    *,
    robot,
    joint_path: np.ndarray,
    stage_labels: list[str],
    manifolds: dict[str, object],
    joint_max_step: float,
    collision_fn=None,
    constraint_tol: float = 2.0e-3,
) -> dict[str, object]:
    q_path = np.asarray(joint_path, dtype=float)
    residuals: list[float] = []
    collision_free = True
    for idx, q in enumerate(q_path):
        stage = stage_labels[idx] if idx < len(stage_labels) else ""
        manifold = manifolds.get(stage)
        residual = float("inf") if manifold is None else float(np.linalg.norm(manifold.residual(q)))
        if manifold is None or not bool(manifold.within_bounds(q, tol=constraint_tol)):
            residual = max(residual, 10.0 * float(constraint_tol))
        if collision_fn is not None and bool(collision_fn(q)):
            collision_free = False
            residual = max(residual, 100.0 * float(constraint_tol))
        residuals.append(float(residual))
    residual_arr = np.asarray(residuals, dtype=float)
    joint_steps, max_step, mean_step, worst_step = joint_step_statistics(q_path)
    max_residual = float(np.max(residual_arr)) if len(residual_arr) > 0 else float("inf")
    mean_residual = float(np.mean(residual_arr)) if len(residual_arr) > 0 else float("inf")
    constraint_ok = bool(collision_free and max_residual <= float(constraint_tol))
    joint_ok = bool(max_step <= float(joint_max_step) + 1.0e-9)
    return {
        "certified": bool(constraint_ok and joint_ok),
        "constraint_certified": bool(constraint_ok),
        "joint_continuity_certified": bool(joint_ok),
        "collision_free": bool(collision_free),
        "residuals": residual_arr,
        "joint_steps": np.asarray(joint_steps, dtype=float),
        "max_constraint_residual": max_residual,
        "mean_constraint_residual": mean_residual,
        "max_joint_step": float(max_step),
        "mean_joint_step": float(mean_step),
        "worst_joint_step_index": int(worst_step),
    }


def realize_selected_continuous_transfer_route_jointspace(
    *,
    robot,
    families,
    start_task: np.ndarray,
    selected_entry_point: np.ndarray,
    selected_exit_point: np.ndarray,
    goal_task: np.ndarray,
    selected_lambda: float,
    joint_max_step: float,
    obstacles=None,
) -> tuple[JointspaceRouteRealization, dict[str, object]]:
    manifolds = build_robot_manifolds_for_selected_lambda(robot, families, float(selected_lambda))
    collision_fn = None
    if obstacles:
        # Continuous-transfer Phase 2B currently has no robot obstacle model.
        # Keep the hook explicit so obstacle-aware certification can be added
        # without changing route semantics.
        collision_fn = None

    segment_messages: dict[str, str] = {}
    q_start, msg = _solve_shared_task_handoff(
        robot=robot,
        target_task=start_task,
        manifolds=[manifolds[LEFT_STAGE]],
        warm_starts=[None],
        collision_fn=collision_fn,
    )
    segment_messages["start_ik"] = msg
    q_entry, msg = _solve_shared_task_handoff(
        robot=robot,
        target_task=selected_entry_point,
        manifolds=[manifolds[LEFT_STAGE], manifolds[FAMILY_STAGE]],
        warm_starts=[q_start, None],
        collision_fn=collision_fn,
    )
    segment_messages["entry_handoff"] = msg
    q_goal_seed, msg = _solve_shared_task_handoff(
        robot=robot,
        target_task=goal_task,
        manifolds=[manifolds[RIGHT_STAGE]],
        warm_starts=[q_entry, None],
        collision_fn=collision_fn,
    )
    segment_messages["goal_ik_seed"] = msg
    q_exit, msg = _solve_shared_task_handoff(
        robot=robot,
        target_task=selected_exit_point,
        manifolds=[manifolds[FAMILY_STAGE], manifolds[RIGHT_STAGE]],
        warm_starts=[q_entry, q_goal_seed, q_start, None],
        collision_fn=collision_fn,
    )
    segment_messages["exit_handoff"] = msg
    q_goal, msg = _solve_shared_task_handoff(
        robot=robot,
        target_task=goal_task,
        manifolds=[manifolds[RIGHT_STAGE]],
        warm_starts=[q_exit, q_goal_seed, None],
        collision_fn=collision_fn,
    )
    segment_messages["goal_ik"] = msg

    missing = [
        name
        for name, q in (
            ("start_q", q_start),
            ("entry_q", q_entry),
            ("exit_q", q_exit),
            ("goal_q", q_goal),
        )
        if q is None
    ]
    if missing:
        message = "joint-space realization failed before local planning: missing " + ", ".join(missing)
        return (
            JointspaceRouteRealization(
                success=False,
                dense_joint_path=np.zeros((0, 3), dtype=float),
                stage_labels=[],
                task_path=np.zeros((0, 3), dtype=float),
                residuals=np.zeros(0, dtype=float),
                joint_steps=np.zeros(0, dtype=float),
                max_constraint_residual=float("inf"),
                mean_constraint_residual=float("inf"),
                max_joint_step=float("inf"),
                mean_joint_step=float("inf"),
                collision_free=False,
                message=message,
                segment_messages=segment_messages,
            ),
            manifolds,
        )

    left_path, segment_messages["left_segment"] = _connect_joint_segment(
        LEFT_STAGE, manifolds[LEFT_STAGE], q_start, q_entry, float(joint_max_step), collision_fn=collision_fn
    )
    if left_path is None:
        left_path, segment_messages["left_segment_task_waypoint_fallback"] = _connect_joint_segment_by_task_waypoints(
            LEFT_STAGE,
            manifolds[LEFT_STAGE],
            q_start,
            q_entry,
            start_task,
            selected_entry_point,
            float(joint_max_step),
            collision_fn=collision_fn,
        )
    family_path, segment_messages["family_segment"] = _connect_joint_segment(
        FAMILY_STAGE, manifolds[FAMILY_STAGE], q_entry, q_exit, float(joint_max_step), collision_fn=collision_fn
    )
    if family_path is None:
        family_path, segment_messages["family_segment_task_waypoint_fallback"] = _connect_joint_segment_by_task_waypoints(
            FAMILY_STAGE,
            manifolds[FAMILY_STAGE],
            q_entry,
            q_exit,
            selected_entry_point,
            selected_exit_point,
            float(joint_max_step),
            collision_fn=collision_fn,
        )
    right_path, segment_messages["right_segment"] = _connect_joint_segment(
        RIGHT_STAGE, manifolds[RIGHT_STAGE], q_exit, q_goal, float(joint_max_step), collision_fn=collision_fn
    )
    if right_path is None:
        right_path, segment_messages["right_segment_task_waypoint_fallback"] = _connect_joint_segment_by_task_waypoints(
            RIGHT_STAGE,
            manifolds[RIGHT_STAGE],
            q_exit,
            q_goal,
            selected_exit_point,
            goal_task,
            float(joint_max_step),
            collision_fn=collision_fn,
        )
    if left_path is None or family_path is None or right_path is None:
        message = "joint-space realization failed during local segment planning"
        return (
            JointspaceRouteRealization(
                success=False,
                dense_joint_path=np.zeros((0, 3), dtype=float),
                stage_labels=[],
                task_path=np.zeros((0, 3), dtype=float),
                residuals=np.zeros(0, dtype=float),
                joint_steps=np.zeros(0, dtype=float),
                max_constraint_residual=float("inf"),
                mean_constraint_residual=float("inf"),
                max_joint_step=float("inf"),
                mean_joint_step=float("inf"),
                collision_free=False,
                message=message,
                segment_messages=segment_messages,
            ),
            manifolds,
        )

    dense_joint_path, labels = _concatenate_labeled_segments(
        [(LEFT_STAGE, left_path), (FAMILY_STAGE, family_path), (RIGHT_STAGE, right_path)]
    )
    cert = _certify_labeled_joint_path(
        robot=robot,
        joint_path=dense_joint_path,
        stage_labels=labels,
        manifolds=manifolds,
        joint_max_step=float(joint_max_step),
        collision_fn=collision_fn,
    )
    task_path = joint_path_to_task_path(robot, dense_joint_path)
    message = (
        "selected-lambda joint-space local replan certified"
        if bool(cert["certified"])
        else "selected-lambda joint-space local replan failed certification"
    )
    return (
        JointspaceRouteRealization(
            success=bool(cert["certified"]),
            dense_joint_path=np.asarray(dense_joint_path, dtype=float),
            stage_labels=list(labels),
            task_path=np.asarray(task_path, dtype=float),
            residuals=np.asarray(cert["residuals"], dtype=float),
            joint_steps=np.asarray(cert["joint_steps"], dtype=float),
            max_constraint_residual=float(cert["max_constraint_residual"]),
            mean_constraint_residual=float(cert["mean_constraint_residual"]),
            max_joint_step=float(cert["max_joint_step"]),
            mean_joint_step=float(cert["mean_joint_step"]),
            collision_free=bool(cert["collision_free"]),
            message=message,
            segment_messages=segment_messages,
        ),
        manifolds,
    )


def plan_continuous_transfer_jointspace_robot(*, scene_description, seed: int, joint_max_step: float, **planner_kwargs):
    """Phase 2B selected-lambda robot realization bridge.

    The task-space mode in this file is complete: the foliation planner selects
    entry/exit/lambda structure. This bridge mode then realizes that selected
    structure in robot joint space. It is intentionally not full joint-space
    exploration over continuous lambda yet.

    - state: q = [theta0, theta1, theta2]
    - task point: FK(q)
    - left leaf: RobotSphereManifold(FK(q) on the left support sphere)
    - right leaf: RobotSphereManifold(FK(q) on the right support sphere)
    - transfer leaf: RobotPlaneLeafManifold / RobotFamilyLeafManifold

    RobotPlaneLeafManifold design notes:
    - stores a locked lambda value and a reference to the continuous transfer
      family;
    - residual(q) evaluates the active plane-leaf constraint at FK(q);
    - within_bounds(q) checks the family patch/obstacle mask at FK(q);
    - project(q) remains permissive enough for exploration, like the existing
      RobotConstraintBase.project();
    - project_local(q) may preserve IK branch for final selected-transition
      realization;
    - infer_lambda(q) delegates to family.infer_lambda(FK(q)) and candidate
      lambdas are managed by a robot leaf-store manager.

    Final execution rule for the future implementation:
    the evidence graph discovers candidate lambdas and transitions only. After
    selecting entry/exit/lambda, the executable robot route must be locally
    replanned as start_q -> entry_q on the left robot sphere, entry_q -> exit_q
    on the selected robot family leaf, and exit_q -> goal_q on the right robot
    sphere. The displayed route must be FK(result.dense_joint_path), and robot
    animation must use that exact dense_joint_path.
    """

    np.random.seed(int(seed))
    ou.RNG.setSeed(int(seed))
    ou.setLogLevel(ou.LOG_ERROR)
    scene = build_continuous_transfer_scene(scene_description)
    result = plan_continuous_transfer_route(
        max_ambient_probes=planner_kwargs.get("max_ambient_probes"),
        continue_after_first_solution=bool(planner_kwargs.get("continue_after_first_solution", True)),
        max_extra_rounds_after_first_solution=int(
            planner_kwargs.get("max_extra_rounds_after_first_solution", DEFAULT_POST_ROUTE_EVIDENCE_ROUNDS)
        ),
        top_k_assignments=int(planner_kwargs.get("top_k_assignments", 3)),
        top_k_paths=int(planner_kwargs.get("top_k_paths", 1)),
        seed=int(seed),
        obstacle_profile=str(planner_kwargs.get("obstacle_profile", "none")),
        scene_description=scene_description,
    )

    selected_lambda = (
        float(result.selected_lambda_for_realization)
        if result.selected_lambda_for_realization is not None
        else float(result.selected_lambda)
        if result.selected_lambda is not None
        else float(scene.transfer_family.nominal_lambda)
    )
    entry = np.asarray(result.selected_entry_point, dtype=float)
    exit_point = np.asarray(result.selected_exit_point, dtype=float)
    if entry.size != 3 or exit_point.size != 3 or not bool(result.success):
        _print_key_value_block(
            "Continuous-Transfer Joint-Space Realization",
            {
                "planning_mode": "selected_lambda_jointspace_realization",
                "full_jointspace_exploration": False,
                "planner_success": bool(result.success),
                "jointspace_realization_success": False,
                "message": "task-space planner did not provide selected entry/exit/lambda for realization",
            },
        )
        return scene, result, None, None

    robot = choose_robot_for_route(np.asarray(result.path, dtype=float))
    realization, _manifolds = realize_selected_continuous_transfer_route_jointspace(
        robot=robot,
        families=(scene.left_support, scene.transfer_family, scene.right_support),
        start_task=scene.start_q,
        selected_entry_point=entry,
        selected_exit_point=exit_point,
        goal_task=scene.goal_q,
        selected_lambda=selected_lambda,
        joint_max_step=float(joint_max_step),
        obstacles=None,
    )

    if realization.success:
        result.dense_joint_path = np.asarray(realization.dense_joint_path, dtype=float)
        result.dense_joint_path_stage_labels = list(realization.stage_labels)
        result.dense_joint_path_constraint_residuals = np.asarray(realization.residuals, dtype=float)
        result.dense_joint_path_is_certified = True
        result.dense_joint_path_execution_certified = True
        result.dense_joint_path_constraint_certified = True
        result.dense_joint_path_joint_continuity_certified = True
        result.dense_joint_path_joint_steps = np.asarray(realization.joint_steps, dtype=float)
        result.dense_joint_path_max_joint_step = float(realization.max_joint_step)
        result.dense_joint_path_mean_joint_step = float(realization.mean_joint_step)
        result.dense_joint_path_worst_joint_step_index = int(np.argmax(realization.joint_steps)) if len(realization.joint_steps) > 0 else -1
        result.dense_joint_path_message = realization.message
        result.path = np.asarray(realization.task_path, dtype=float)
        result.raw_path = np.asarray(realization.task_path, dtype=float)
        result.display_path = np.asarray(realization.task_path, dtype=float)
        result.final_route_realization = "selected_lambda_jointspace_local_replan"
        result.graph_route_used_for_execution = False

    robot_execution = None
    if len(realization.dense_joint_path) > 0:
        task_path = joint_path_to_task_path(robot, realization.dense_joint_path)
        robot_execution = RobotExecutionResult(
            target_task_points_3d=np.asarray(task_path, dtype=float),
            joint_path=np.asarray(realization.dense_joint_path, dtype=float),
            end_effector_points_3d=np.asarray(task_path, dtype=float),
            ik_success_count=int(len(realization.dense_joint_path)),
            ik_failure_count=0,
            max_tracking_error=0.0,
            mean_tracking_error=0.0,
            max_joint_step=float(realization.max_joint_step),
            execution_success=bool(realization.success),
            diagnostics=str(realization.message),
            animation_enabled=bool(realization.success),
            planner_path_resampled_for_robot=False,
            planner_joint_path_used_directly=True,
            max_constraint_residual=float(realization.max_constraint_residual),
            mean_constraint_residual=float(realization.mean_constraint_residual),
            constraint_validation_success=bool(realization.success),
            worst_constraint_stage="selected_lambda_jointspace",
            worst_constraint_index=-1,
            execution_source="certified_dense_joint_path" if realization.success else "disabled_uncertified_jointspace_realization",
        )

    display_vs_trace_max_error, display_vs_trace_mean_error = (
        _polyline_error(np.asarray(result.path, dtype=float), np.asarray(robot_execution.end_effector_points_3d, dtype=float))
        if robot_execution is not None
        else (float("inf"), float("inf"))
    )

    print_continuous_route_summary(result)
    _print_key_value_block(
        "Continuous-Transfer Joint-Space Realization",
        {
            "planning_mode": "selected_lambda_jointspace_realization",
            "full_jointspace_exploration": False,
            "planner_success": bool(result.success),
            "selected_lambda_for_realization": round(float(selected_lambda), 6),
            "selected_entry_point": np.round(entry, 6).tolist(),
            "selected_exit_point": np.round(exit_point, 6).tolist(),
            "taskspace_final_route_realization": result.final_route_realization,
            "graph_route_used_for_execution": bool(result.graph_route_used_for_execution),
            "strict_validation_success": bool(result.strict_validation_success),
            "jointspace_realization_success": bool(realization.success),
            "dense_joint_path_execution_certified": bool(realization.success),
            "dense_joint_path_points": int(len(realization.dense_joint_path)),
            "max_dense_constraint_residual": round(float(realization.max_constraint_residual), 8)
            if np.isfinite(realization.max_constraint_residual)
            else float("inf"),
            "mean_dense_constraint_residual": round(float(realization.mean_constraint_residual), 8)
            if np.isfinite(realization.mean_constraint_residual)
            else float("inf"),
            "dense_joint_path_max_joint_step": round(float(realization.max_joint_step), 6)
            if np.isfinite(realization.max_joint_step)
            else float("inf"),
            "collision_free": bool(realization.collision_free),
            "execution_source": "none" if robot_execution is None else str(robot_execution.execution_source),
            "route_source": "FK(result.dense_joint_path)" if realization.success else "unavailable",
            "display_vs_trace_max_error": round(float(display_vs_trace_max_error), 6),
            "display_vs_trace_mean_error": round(float(display_vs_trace_mean_error), 6),
            "segment_start": realization.segment_messages.get("start_ik", ""),
            "segment_entry": realization.segment_messages.get("entry_handoff", ""),
            "segment_exit": realization.segment_messages.get("exit_handoff", ""),
            "segment_goal": realization.segment_messages.get("goal_ik", ""),
            "segment_left": realization.segment_messages.get("left_segment", ""),
            "segment_left_fallback": realization.segment_messages.get("left_segment_task_waypoint_fallback", ""),
            "segment_family": realization.segment_messages.get("family_segment", ""),
            "segment_family_fallback": realization.segment_messages.get("family_segment_task_waypoint_fallback", ""),
            "segment_right": realization.segment_messages.get("right_segment", ""),
            "segment_right_fallback": realization.segment_messages.get("right_segment_task_waypoint_fallback", ""),
            "message": realization.message,
        },
    )
    return scene, result, robot, robot_execution


def plan_full_jointspace_continuous_transfer_evidence_scaffold(
    *,
    scene_description,
    seed: int,
    joint_max_step: float,
    max_ambient_probes: int | None,
    **planner_kwargs,
):
    """Phase 3A: project ambient joint proposals onto all robot supports.

    This is deliberately evidence-only. It does not infer transitions, extract
    routes, or animate anything. The purpose is to prove that q-space proposal
    projection can populate left/right robot sphere stores and multiple locked
    robot family leaf stores before Phase 3B transition discovery.
    """

    np.random.seed(int(seed))
    ou.RNG.setSeed(int(seed))
    ou.setLogLevel(ou.LOG_ERROR)
    scene = build_continuous_transfer_scene(scene_description)
    rough_route = np.asarray(
        [
            scene.start_q,
            0.5 * (scene.start_q + scene.goal_q),
            scene.goal_q,
        ],
        dtype=float,
    )
    robot = choose_robot_for_route(rough_route)
    families = (scene.left_support, scene.transfer_family, scene.right_support)
    nominal_lambda = float(scene.transfer_family.nominal_lambda)
    manifolds = build_robot_manifolds_for_selected_lambda(robot, families, nominal_lambda)
    evidence = RobotJointspaceContinuousEvidence(
        left_store=RobotStageEvidenceStore(name="left_robot_sphere", manifold=manifolds[LEFT_STAGE], nodes=[], edges=[], frontier_ids=[]),
        right_store=RobotStageEvidenceStore(name="right_robot_sphere", manifold=manifolds[RIGHT_STAGE], nodes=[], edges=[], frontier_ids=[]),
        family_leaf_stores={},
        counters={},
    )
    scene_lambdas = _scene_candidate_lambdas(scene_description, scene.transfer_family)

    q_start = inverse_kinematics_start(robot, scene.start_q, tol=8.0e-2)
    q_goal = inverse_kinematics_start(robot, scene.goal_q, warm_start=q_start, tol=8.0e-2)
    if q_start is not None:
        projected = _project_to_manifold(evidence.left_store.manifold, q_start, local=False)
        if projected is not None:
            evidence.left_store.add_node(projected)
            q_start = projected
    if q_goal is not None:
        projected = _project_to_manifold(evidence.right_store.manifold, q_goal, local=False)
        if projected is not None:
            evidence.right_store.add_node(projected)
            q_goal = projected

    proposal_rounds = int(max_ambient_probes if max_ambient_probes is not None else 80)
    proposal_rounds = max(1, proposal_rounds)
    proposal_count = 8
    joint_lower = -np.pi * np.ones(3, dtype=float)
    joint_upper = np.pi * np.ones(3, dtype=float)
    start_seed = np.asarray(q_start if q_start is not None else np.zeros(3, dtype=float), dtype=float)
    goal_seed = np.asarray(q_goal if q_goal is not None else np.asarray([0.0, 0.85, -0.65], dtype=float), dtype=float)

    for round_idx in range(1, proposal_rounds + 1):
        active_guides = []
        for store in evidence.family_leaf_stores.values():
            active_guides.extend(store.nodes[-2:])
        proposals = generate_joint_proposals(
            round_idx,
            start_seed,
            goal_seed,
            active_guides,
            proposal_count=proposal_count,
            joint_lower=joint_lower,
            joint_upper=joint_upper,
        )
        for proposal in proposals:
            project_joint_proposal_to_supports(
                proposal,
                evidence,
                robot,
                families,
                scene_lambdas,
            )

    family_nodes = sum(len(store.nodes) for store in evidence.family_leaf_stores.values())
    active_lambdas = sorted(float(key) for key in evidence.family_leaf_stores.keys())
    lambda_range = (
        [round(float(min(active_lambdas)), 6), round(float(max(active_lambdas)), 6)]
        if active_lambdas
        else []
    )
    evidence_ready = bool(len(evidence.left_store.nodes) > 0 and len(evidence.right_store.nodes) > 0 and family_nodes > 0)
    _print_key_value_block(
        "Full Joint-Space Continuous-Transfer Exploration Scaffold",
        {
            "full_jointspace_exploration": True,
            "implementation_phase": "phase_3a_evidence_projection",
            "planner_success": "evidence_ready" if evidence_ready else False,
            "left_evidence_nodes": int(len(evidence.left_store.nodes)),
            "left_evidence_edges": int(len(evidence.left_store.edges)),
            "right_evidence_nodes": int(len(evidence.right_store.nodes)),
            "right_evidence_edges": int(len(evidence.right_store.edges)),
            "family_leaf_store_count": int(len(evidence.family_leaf_stores)),
            "family_evidence_nodes": int(family_nodes),
            "family_evidence_edges": int(sum(len(store.edges) for store in evidence.family_leaf_stores.values())),
            "projected_to_left_count": int(evidence.counters.get("projected_to_left_count", 0)),
            "projected_to_family_count": int(evidence.counters.get("projected_to_family_count", 0)),
            "projected_to_right_count": int(evidence.counters.get("projected_to_right_count", 0)),
            "left_projection_failure_count": int(evidence.counters.get("left_projection_failure_count", 0)),
            "family_projection_failure_count": int(evidence.counters.get("family_projection_failure_count", 0)),
            "right_projection_failure_count": int(evidence.counters.get("right_projection_failure_count", 0)),
            "candidate_lambdas_evaluated": int(evidence.counters.get("candidate_lambdas_evaluated", 0)),
            "active_family_lambdas": [round(float(v), 6) for v in active_lambdas[:20]],
            "active_family_lambdas_omitted": max(0, len(active_lambdas) - 20),
            "lambda_coverage_range": lambda_range,
            "entry_transitions_found": 0,
            "exit_transitions_found": 0,
            "final_route_realization": "pending_phase_3c",
            "graph_route_used_for_execution": False,
            "execution_source": "none",
            "route_source": "none",
            "display_vs_trace_max_error": "not_applicable",
            "joint_max_step": float(joint_max_step),
            "proposal_rounds": int(proposal_rounds),
            "proposals_per_round": int(proposal_count),
            "next_step": "phase_3b_transition_discovery",
        },
    )
    return scene, None, robot, None


def show_continuous_transfer_robot_demo(
    *,
    scene,
    result,
    robot,
    robot_execution: RobotExecutionResult | None,
    show_exploration: bool = True,
) -> bool:
    if pv is None or not pyvista_available():
        print("PyVista is not available; skipping continuous-transfer robot visualization.")
        return False

    left_family = scene.left_support
    transfer_family = scene.transfer_family
    right_family = scene.right_support
    left_manifold = left_family.manifold(float(left_family.sample_lambdas()[0]))
    right_manifold = right_family.manifold(float(right_family.sample_lambdas()[0]))
    selected_lambda = (
        float(result.selected_lambda_for_realization)
        if result.selected_lambda_for_realization is not None
        else float(result.selected_lambda)
        if result.selected_lambda is not None
        else float(transfer_family.nominal_lambda)
    )
    family_leaf = transfer_family.manifold(float(selected_lambda))

    plotter = pv.Plotter(window_size=(1440, 920))
    if hasattr(plotter, "set_background"):
        plotter.set_background("#edf3f8", top="#ffffff")
    if hasattr(plotter, "enable_anti_aliasing"):
        try:
            plotter.enable_anti_aliasing()
        except Exception:
            pass
    plotter.add_text(
        "Continuous transfer: evidence graph plus 3DOF robot tracking only the selected final route",
        font_size=12,
    )

    actor_groups: dict[str, list[object]] = {
        "Manifolds": [],
        "Evidence": [],
        "Transitions": [],
        "FinalRoute": [],
        "Robot": [],
        "EETrace": [],
        "StartGoal": [],
    }

    for manifold, color in ((left_manifold, "#c58b4c"), (right_manifold, "#c58b4c")):
        actor = add_manifold(plotter, manifold, color=color, opacity=0.10)
        if actor is not None:
            actor_groups["Manifolds"].append(actor)

    leaf_corners = plane_leaf_patch(transfer_family, float(selected_lambda))
    leaf_patch = pv.PolyData(leaf_corners, faces=np.hstack([[4, 0, 1, 2, 3]]))
    actor = plotter.add_mesh(
        leaf_patch,
        color="#7fa7c6",
        opacity=0.22,
        show_edges=False,
        label=f"selected transfer leaf lambda={selected_lambda:.3f}",
    )
    if actor is not None:
        actor_groups["Manifolds"].append(actor)

    if show_exploration:
        colors = {
            "left": "#81c784",
            "family_leaf": "#64b5f6",
            "family_transverse": "#8e24aa",
            "right": "#a5d6a7",
        }
        for mode, edges in result.explored_edges_by_mode.items():
            poly = build_segment_polydata(edges)
            if poly is None:
                continue
            actor = plotter.add_mesh(
                poly,
                color=colors.get(str(mode), "#78909c"),
                opacity=0.28,
                line_width=1.8,
                label=f"{mode} evidence",
            )
            if actor is not None:
                actor_groups["Evidence"].append(actor)

    if len(result.entry_transition_points) > 0:
        actor = add_points(plotter, result.entry_transition_points, color="#ff7043", size=9.0, label="entry transitions")
        if actor is not None:
            actor_groups["Transitions"].append(actor)
    if len(result.exit_transition_points) > 0:
        actor = add_points(plotter, result.exit_transition_points, color="#26a69a", size=9.0, label="exit transitions")
        if actor is not None:
            actor_groups["Transitions"].append(actor)

    final_route = np.asarray(result.path, dtype=float)
    if len(final_route) >= 2:
        actor = plotter.add_mesh(
            pv.lines_from_points(final_route),
            color="#d32f2f",
            line_width=8.0,
            label="FINAL SELECTED-TRANSITION ROUTE",
        )
        if actor is not None:
            actor_groups["FinalRoute"].append(actor)

    actor = add_points(plotter, scene.start_q, color="black", size=16.0, label="start")
    if actor is not None:
        actor_groups["StartGoal"].append(actor)
    actor = add_points(plotter, scene.goal_q, color="gold", size=18.0, label="goal")
    if actor is not None:
        actor_groups["StartGoal"].append(actor)

    initial_angles = (
        np.asarray(robot_execution.joint_path[0], dtype=float)
        if robot_execution is not None and len(robot_execution.joint_path) > 0
        else np.asarray([0.0, 0.55, -0.85], dtype=float)
    )
    robot_bundle = make_robot_actor_bundle(plotter, robot, initial_angles, opacity=1.0)
    pedestal_actor = add_robot_pedestal(plotter, robot)
    if pedestal_actor is not None:
        actor_groups["Robot"].append(pedestal_actor)
    actor_groups["Robot"].extend(robot_bundle.get("all", []))

    trace_actor = {"actor": None}

    def replace_trace(points: np.ndarray) -> None:
        if trace_actor["actor"] is not None:
            plotter.remove_actor(trace_actor["actor"], render=False)
        pts = np.asarray(points, dtype=float)
        if len(pts) == 0:
            trace_actor["actor"] = None
            return
        if len(pts) == 1:
            actor_inner = plotter.add_mesh(
                pv.Sphere(radius=float(0.75 * robot.ee_radius), center=pts[0]),
                color="#d81b60",
                label="ROBOT END-EFFECTOR TRACE",
            )
        else:
            actor_inner = plotter.add_mesh(
                pv.lines_from_points(pts),
                color="#d81b60",
                line_width=5.0,
                label="ROBOT END-EFFECTOR TRACE",
            )
        trace_actor["actor"] = actor_inner

    animation_state = {"frame": 0, "playing": False, "trace": []}

    def reset_animation() -> None:
        animation_state["frame"] = 0
        animation_state["playing"] = bool(robot_execution is not None and robot_execution.animation_enabled)
        animation_state["trace"] = []
        if robot_execution is not None and len(robot_execution.joint_path) > 0:
            update_robot_actor_bundle(plotter, robot, robot_bundle, robot_execution.joint_path[0])
        replace_trace(np.zeros((0, 3), dtype=float))
        plotter.render()

    def animate_step() -> None:
        if robot_execution is None or not robot_execution.animation_enabled or not animation_state["playing"]:
            return
        idx = min(int(animation_state["frame"]), len(robot_execution.joint_path) - 1)
        update_robot_actor_bundle(plotter, robot, robot_bundle, robot_execution.joint_path[idx])
        animation_state["trace"].append(np.asarray(robot_execution.end_effector_points_3d[idx], dtype=float))
        replace_trace(np.asarray(animation_state["trace"], dtype=float))
        if idx >= len(robot_execution.joint_path) - 1:
            animation_state["playing"] = False
        else:
            animation_state["frame"] += 1
        plotter.render()

    def play_animation() -> None:
        import time

        while bool(animation_state["playing"]):
            animate_step()
            plotter.update()
            time.sleep(0.02)

    def start_replay() -> None:
        reset_animation()
        play_animation()

    if hasattr(plotter, "add_key_event"):
        plotter.add_key_event("r", start_replay)
    plotter.add_text("Press r to replay robot motion", position=(1030, 18), font_size=10, color="black")
    plotter.add_axes()
    plotter.show_grid()
    plotter.camera_position = [
        (0.25, -7.8, 3.8),
        (0.0, -0.05, 0.35),
        (0.0, 0.0, 1.0),
    ]
    reset_animation()
    plotter.show(auto_close=False, interactive_update=True)
    if robot_execution is not None and robot_execution.animation_enabled:
        play_animation()
    try:
        plotter.app.exec()
    except Exception:
        try:
            plotter.show()
        except Exception:
            pass
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous-transfer planner with 3DOF robot execution.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--taskspace-planning", action="store_true", help="Run task-space continuous-transfer planning plus IK execution.")
    mode.add_argument("--jointspace-planning", action="store_true", help="Reserved for Phase 2 joint-space continuous-family planning.")
    parser.set_defaults(taskspace_planning=True)
    parser.add_argument(
        "--full-jointspace-exploration",
        action="store_true",
        help="Opt into Phase 3 robot q-space continuous-family evidence projection scaffold.",
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-probes", type=int, default=None)
    parser.add_argument("--extra-rounds-after-first-solution", type=int, default=DEFAULT_POST_ROUTE_EVIDENCE_ROUNDS)
    parser.add_argument("--stop-after-first-solution", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--top-k-paths", type=int, default=1)
    parser.add_argument("--obstacle-profile", type=str, default="none")
    parser.add_argument("--max-cartesian-step", type=float, default=0.025)
    parser.add_argument("--joint-max-step", type=float, default=0.12)
    parser.add_argument("--ik-tolerance", type=float, default=3.0e-3)
    parser.add_argument("--show-exploration", dest="show_exploration", action="store_true", default=True)
    parser.add_argument("--hide-exploration", dest="show_exploration", action="store_false")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    if args.jointspace_planning:
        scene_description = default_example_65_scene_description(obstacle_profile=args.obstacle_profile)
        if args.full_jointspace_exploration:
            scene, result, robot, robot_execution = plan_full_jointspace_continuous_transfer_evidence_scaffold(
                scene_description=scene_description,
                seed=int(args.seed),
                joint_max_step=float(args.joint_max_step),
                max_ambient_probes=args.max_probes,
                continue_after_first_solution=not args.stop_after_first_solution,
                max_extra_rounds_after_first_solution=args.extra_rounds_after_first_solution,
                top_k_assignments=args.top_k,
                top_k_paths=args.top_k_paths,
                obstacle_profile=args.obstacle_profile,
            )
        else:
            scene, result, robot, robot_execution = plan_continuous_transfer_jointspace_robot(
                scene_description=scene_description,
                seed=int(args.seed),
                joint_max_step=float(args.joint_max_step),
                max_ambient_probes=args.max_probes,
                continue_after_first_solution=not args.stop_after_first_solution,
                max_extra_rounds_after_first_solution=args.extra_rounds_after_first_solution,
                top_k_assignments=args.top_k,
                top_k_paths=args.top_k_paths,
                obstacle_profile=args.obstacle_profile,
            )
        if not args.no_viz and scene is not None and result is not None and robot is not None:
            show_continuous_transfer_robot_demo(
                scene=scene,
                result=result,
                robot=robot,
                robot_execution=robot_execution,
                show_exploration=bool(args.show_exploration),
            )
        sys.stdout.flush()
        sys.stderr.flush()
        if not args.no_viz:
            os._exit(0)
        return

    np.random.seed(int(args.seed))
    ou.RNG.setSeed(int(args.seed))
    ou.setLogLevel(ou.LOG_ERROR)

    scene_description = default_example_65_scene_description(obstacle_profile=args.obstacle_profile)
    scene = build_continuous_transfer_scene(scene_description)
    result = plan_continuous_transfer_route(
        max_ambient_probes=args.max_probes,
        continue_after_first_solution=not args.stop_after_first_solution,
        max_extra_rounds_after_first_solution=args.extra_rounds_after_first_solution,
        top_k_assignments=args.top_k,
        top_k_paths=args.top_k_paths,
        seed=args.seed,
        obstacle_profile=args.obstacle_profile,
        scene_description=scene_description,
    )

    route = np.asarray(result.path, dtype=float)
    robot = choose_robot_for_route(route)
    robot_execution = build_continuous_robot_execution_path(
        route,
        robot,
        max_cartesian_step=float(args.max_cartesian_step),
        max_joint_step=float(args.joint_max_step),
        ik_tolerance=float(args.ik_tolerance),
    )
    display_vs_trace_max_error, display_vs_trace_mean_error = (
        _polyline_error(
            np.asarray(robot_execution.target_task_points_3d, dtype=float),
            np.asarray(robot_execution.end_effector_points_3d, dtype=float),
        )
        if robot_execution is not None
        else (float("inf"), float("inf"))
    )

    print_continuous_route_summary(result)
    _print_key_value_block(
        "Continuous-Transfer Robot Execution",
        {
            "planning_mode": "taskspace_ik_execution",
            "planner_success": bool(result.success),
            "selected_lambda_for_realization": None
            if result.selected_lambda_for_realization is None
            else round(float(result.selected_lambda_for_realization), 6),
            "final_route_realization": result.final_route_realization,
            "graph_route_used_for_execution": bool(result.graph_route_used_for_execution),
            "strict_validation_success": bool(result.strict_validation_success),
            "robot_execution_success": bool(robot_execution.execution_success) if robot_execution is not None else False,
            "robot_animation_enabled": bool(robot_execution.animation_enabled) if robot_execution is not None else False,
            "robot_execution_waypoints": 0 if robot_execution is None else len(robot_execution.joint_path),
            "max_tracking_error": float("inf") if robot_execution is None else round(float(robot_execution.max_tracking_error), 6),
            "mean_tracking_error": float("inf") if robot_execution is None else round(float(robot_execution.mean_tracking_error), 6),
            "max_joint_step": float("inf") if robot_execution is None else round(float(robot_execution.max_joint_step), 6),
            "ik_failures": -1 if robot_execution is None else int(robot_execution.ik_failure_count),
            "display_vs_trace_max_error": round(float(display_vs_trace_max_error), 6),
            "display_vs_trace_mean_error": round(float(display_vs_trace_mean_error), 6),
            "execution_source": "none" if robot_execution is None else str(robot_execution.execution_source),
            "diagnostics": "robot execution unavailable" if robot_execution is None else str(robot_execution.diagnostics),
        },
    )

    if not args.no_viz:
        show_continuous_transfer_robot_demo(
            scene=scene,
            result=result,
            robot=robot,
            robot_execution=robot_execution,
            show_exploration=bool(args.show_exploration),
        )

    sys.stdout.flush()
    sys.stderr.flush()
    if not args.no_viz:
        os._exit(0)


if __name__ == "__main__":
    main()
