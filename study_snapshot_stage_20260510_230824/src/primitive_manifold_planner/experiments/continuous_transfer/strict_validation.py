"""Strict manifold validation helpers for the staged continuous-transfer planner."""

from __future__ import annotations

import numpy as np

from .augmented_family_space import FamilyAugmentedState
from .core.certification import CertificationResult
from .config import (
    LAMBDA_SOURCE_TOL,
    STRICT_FAMILY_RESIDUAL_TOL,
    STRICT_FAMILY_SAMPLE_STEP,
    STRICT_PATCH_TOL,
    STRICT_SPHERE_RESIDUAL_TOL,
    STRICT_SPHERE_SAMPLE_STEP,
    STRICT_TRANSITION_MAX_LENGTH,
    STRICT_TRANSITION_MAX_PATH_POINTS,
    STRICT_TRANSVERSE_SAMPLE_STEP,
    STRICT_VALIDATION_VERBOSE,
    TRANSVERSE_LAMBDA_STEP,
)
from .family_definition import ContinuousMaskedPlaneFamily, sphere_arc_length
from .graph_types import FamilyConnectivityGraph, StrictValidationFailure
from .support import deduplicate_points, smooth_plane_segment, smooth_sphere_arc


def polyline_length(path: np.ndarray) -> float:
    pts = np.asarray(path, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def format_strict_validation_failure(failure: StrictValidationFailure) -> str:
    parts = [
        f"{failure.edge_kind}",
        f"point_index={int(failure.point_index)}",
        f"q={np.round(np.asarray(failure.q, dtype=float), 6).tolist()}",
        failure.message,
    ]
    if failure.residual is not None:
        parts.append(f"residual={float(failure.residual):.3e}")
    if failure.lambda_value is not None:
        parts.append(f"lambda={float(failure.lambda_value):.6f}")
    return ", ".join(parts)


def collect_failure_points(failures: list[StrictValidationFailure]) -> np.ndarray:
    if len(failures) == 0:
        return np.zeros((0, 3), dtype=float)
    return deduplicate_points([np.asarray(failure.q, dtype=float) for failure in failures], tol=1e-6)


def report_strict_validation_failures(context: str, failures: list[StrictValidationFailure], limit: int = 3) -> None:
    if not STRICT_VALIDATION_VERBOSE or len(failures) == 0:
        return
    print(f"[strict-validation] {context}")
    for failure in failures[: max(1, int(limit))]:
        print(f"  - {format_strict_validation_failure(failure)}")


def sample_strict_sphere_motion_path(
    center: np.ndarray,
    radius: float,
    q_start: np.ndarray,
    q_goal: np.ndarray,
) -> np.ndarray:
    if np.linalg.norm(np.asarray(q_goal, dtype=float) - np.asarray(q_start, dtype=float)) <= 1e-12:
        return np.asarray([np.asarray(q_start, dtype=float)], dtype=float)
    arc_len = sphere_arc_length(center, radius, q_start, q_goal)
    num = max(2, int(np.ceil(float(arc_len) / STRICT_SPHERE_SAMPLE_STEP)) + 1)
    return np.asarray(smooth_sphere_arc(center, radius, q_start, q_goal, num=num), dtype=float)


def sample_strict_family_leaf_path(q_start: np.ndarray, q_goal: np.ndarray) -> np.ndarray:
    if np.linalg.norm(np.asarray(q_goal, dtype=float) - np.asarray(q_start, dtype=float)) <= 1e-12:
        return np.asarray([np.asarray(q_start, dtype=float)], dtype=float)
    length = float(np.linalg.norm(np.asarray(q_goal, dtype=float) - np.asarray(q_start, dtype=float)))
    num = max(2, int(np.ceil(length / STRICT_FAMILY_SAMPLE_STEP)) + 1)
    return np.asarray(smooth_plane_segment(q_start, q_goal, num=num), dtype=float)


def sample_strict_family_transverse_segment(
    transfer_family: ContinuousMaskedPlaneFamily,
    q_start: np.ndarray,
    lambda_start: float,
    q_goal: np.ndarray,
    lambda_goal: float,
    project_valid_family_state_fn,
) -> tuple[np.ndarray, np.ndarray]:
    lam0 = float(lambda_start)
    lam1 = float(lambda_goal)
    q0 = np.asarray(q_start, dtype=float)
    q1 = np.asarray(q_goal, dtype=float)
    if abs(lam1 - lam0) <= 1e-12 and np.linalg.norm(q1 - q0) <= 1e-12:
        return np.asarray([q0], dtype=float), np.asarray([lam0], dtype=float)

    u0, v0 = transfer_family.patch_coords(lam0, q0)
    u1, v1 = transfer_family.patch_coords(lam1, q1)
    step_count = max(
        2,
        int(np.ceil(abs(lam1 - lam0) / TRANSVERSE_LAMBDA_STEP)),
        int(np.ceil(max(abs(u1 - u0), abs(v1 - v0)) / STRICT_TRANSVERSE_SAMPLE_STEP)),
        int(np.ceil(np.linalg.norm(q1 - q0) / STRICT_TRANSVERSE_SAMPLE_STEP)),
    )
    path: list[np.ndarray] = []
    lambdas: list[float] = []
    for t in np.linspace(0.0, 1.0, step_count + 1):
        lam = (1.0 - float(t)) * lam0 + float(t) * lam1
        u_coord = (1.0 - float(t)) * u0 + float(t) * u1
        v_coord = (1.0 - float(t)) * v0 + float(t) * v1
        guess = (
            transfer_family.point_on_leaf(lam)
            + u_coord * np.asarray(transfer_family._basis_u, dtype=float)
            + v_coord * np.asarray(transfer_family._basis_v, dtype=float)
        )
        q = project_valid_family_state_fn(transfer_family, lam, guess, tol=STRICT_FAMILY_RESIDUAL_TOL)
        if q is None:
            return np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float)
        path.append(q)
        lambdas.append(float(lam))
    return np.asarray(path, dtype=float), np.asarray(lambdas, dtype=float)


def validate_path_on_sphere(
    path: np.ndarray,
    side_manifold,
    center: np.ndarray,
    radius: float,
    edge_kind: str,
) -> list[StrictValidationFailure]:
    pts = np.asarray(path, dtype=float)
    failures: list[StrictValidationFailure] = []
    if len(pts) == 0:
        failures.append(
            StrictValidationFailure(
                edge_kind=edge_kind,
                message="empty stored path",
                point_index=0,
                q=np.zeros((3,), dtype=float),
            )
        )
        return failures
    center_arr = np.asarray(center, dtype=float)
    for idx, q in enumerate(pts):
        residual = float(np.linalg.norm(np.asarray(side_manifold.residual(q), dtype=float)))
        if residual > STRICT_SPHERE_RESIDUAL_TOL:
            failures.append(
                StrictValidationFailure(
                    edge_kind=edge_kind,
                    message="sphere residual exceeded strict tolerance",
                    point_index=idx,
                    q=np.asarray(q, dtype=float),
                    residual=residual,
                )
            )
        radius_error = abs(float(np.linalg.norm(np.asarray(q, dtype=float) - center_arr)) - float(radius))
        if radius_error > STRICT_SPHERE_RESIDUAL_TOL:
            failures.append(
                StrictValidationFailure(
                    edge_kind=edge_kind,
                    message="radius consistency check failed; sphere motion would leave the surface",
                    point_index=idx,
                    q=np.asarray(q, dtype=float),
                    residual=radius_error,
                )
            )
    return failures


def validate_path_on_family_leaf(
    path: np.ndarray,
    transfer_family: ContinuousMaskedPlaneFamily,
    lam: float,
    edge_kind: str,
) -> list[StrictValidationFailure]:
    pts = np.asarray(path, dtype=float)
    failures: list[StrictValidationFailure] = []
    if len(pts) == 0:
        failures.append(
            StrictValidationFailure(
                edge_kind=edge_kind,
                message="empty stored path",
                point_index=0,
                q=np.zeros((3,), dtype=float),
                lambda_value=float(lam),
            )
        )
        return failures
    manifold = transfer_family.manifold(float(lam))
    for idx, q in enumerate(pts):
        residual = float(np.linalg.norm(np.asarray(manifold.residual(q), dtype=float)))
        if residual > STRICT_FAMILY_RESIDUAL_TOL:
            failures.append(
                StrictValidationFailure(
                    edge_kind=edge_kind,
                    message="family-leaf residual exceeded strict tolerance",
                    point_index=idx,
                    q=np.asarray(q, dtype=float),
                    residual=residual,
                    lambda_value=float(lam),
                )
            )
        patch_margin = float(transfer_family.patch_margin(float(lam), q))
        if patch_margin < -STRICT_PATCH_TOL:
            failures.append(
                StrictValidationFailure(
                    edge_kind=edge_kind,
                    message="family patch validity failed",
                    point_index=idx,
                    q=np.asarray(q, dtype=float),
                    residual=-patch_margin,
                    lambda_value=float(lam),
                )
            )
    return failures


def validate_augmented_family_state(
    state: FamilyAugmentedState,
    transfer_family: ContinuousMaskedPlaneFamily,
    edge_kind: str = "family_augmented_state",
) -> list[StrictValidationFailure]:
    failures: list[StrictValidationFailure] = []
    q = np.asarray(state.q, dtype=float)
    lam = float(state.lambda_value)
    manifold = transfer_family.manifold(lam)
    residual = float(np.linalg.norm(np.asarray(manifold.residual(q), dtype=float)))
    if residual > STRICT_FAMILY_RESIDUAL_TOL:
        failures.append(
            StrictValidationFailure(
                edge_kind=edge_kind,
                message="augmented family state residual exceeded strict tolerance",
                point_index=0,
                q=q,
                residual=residual,
                lambda_value=lam,
            )
        )
    if not transfer_family.lambda_in_range(lam, tol=STRICT_PATCH_TOL):
        failures.append(
            StrictValidationFailure(
                edge_kind=edge_kind,
                message="augmented family state lambda left configured family bounds",
                point_index=0,
                q=q,
                lambda_value=lam,
            )
        )
    patch_margin = float(transfer_family.patch_margin(lam, q))
    if patch_margin < -STRICT_PATCH_TOL:
        failures.append(
            StrictValidationFailure(
                edge_kind=edge_kind,
                message="augmented family state violated patch validity",
                point_index=0,
                q=q,
                residual=-patch_margin,
                lambda_value=lam,
            )
        )
    return failures


def validate_path_on_family_transverse(
    path: np.ndarray,
    lambdas: np.ndarray,
    transfer_family: ContinuousMaskedPlaneFamily,
    edge_kind: str,
) -> list[StrictValidationFailure]:
    pts = np.asarray(path, dtype=float)
    lam_arr = np.asarray(lambdas, dtype=float).reshape(-1)
    failures: list[StrictValidationFailure] = []
    if len(pts) == 0:
        failures.append(
            StrictValidationFailure(
                edge_kind=edge_kind,
                message="empty stored path",
                point_index=0,
                q=np.zeros((3,), dtype=float),
            )
        )
        return failures
    if len(pts) != len(lam_arr):
        failures.append(
            StrictValidationFailure(
                edge_kind=edge_kind,
                message="path and lambda samples have inconsistent lengths",
                point_index=0,
                q=np.asarray(pts[0], dtype=float),
            )
        )
        return failures
    for idx, (q, lam) in enumerate(zip(pts, lam_arr)):
        manifold = transfer_family.manifold(float(lam))
        residual = float(np.linalg.norm(np.asarray(manifold.residual(q), dtype=float)))
        if residual > STRICT_FAMILY_RESIDUAL_TOL:
            failures.append(
                StrictValidationFailure(
                    edge_kind=edge_kind,
                    message="transverse family sample drifted off its intermediate leaf",
                    point_index=idx,
                    q=np.asarray(q, dtype=float),
                    residual=residual,
                    lambda_value=float(lam),
                )
            )
        patch_margin = float(transfer_family.patch_margin(float(lam), q))
        if patch_margin < -STRICT_PATCH_TOL:
            failures.append(
                StrictValidationFailure(
                    edge_kind=edge_kind,
                    message="transverse family sample violated patch bounds",
                    point_index=idx,
                    q=np.asarray(q, dtype=float),
                    residual=-patch_margin,
                    lambda_value=float(lam),
                )
            )
    return failures


def validate_augmented_family_edge(
    path: np.ndarray,
    lambdas: np.ndarray,
    transfer_family: ContinuousMaskedPlaneFamily,
    edge_kind: str = "family_augmented_edge",
) -> list[StrictValidationFailure]:
    pts = np.asarray(path, dtype=float)
    lam_arr = np.asarray(lambdas, dtype=float).reshape(-1)
    if len(pts) == 0 or len(lam_arr) == 0:
        return [
            StrictValidationFailure(
                edge_kind=edge_kind,
                message="empty augmented family edge",
                point_index=0,
                q=np.zeros((3,), dtype=float),
            )
        ]
    failures: list[StrictValidationFailure] = []
    for idx, (q, lam) in enumerate(zip(pts, lam_arr)):
        state_failures = validate_augmented_family_state(
            FamilyAugmentedState(
                q=np.asarray(q, dtype=float),
                lambda_value=float(lam),
                discovered_round=0,
            ),
            transfer_family=transfer_family,
            edge_kind=edge_kind,
        )
        for failure in state_failures:
            failure.point_index = idx
        failures.extend(state_failures)
    return failures


def validate_transition_point(
    path: np.ndarray,
    side_manifold,
    transfer_family: ContinuousMaskedPlaneFamily,
    lam: float,
    side_center: np.ndarray,
    side_radius: float,
    edge_kind: str,
) -> list[StrictValidationFailure]:
    pts = np.asarray(path, dtype=float)
    failures: list[StrictValidationFailure] = []
    if len(pts) == 0:
        failures.append(
            StrictValidationFailure(
                edge_kind=edge_kind,
                message="empty transition path",
                point_index=0,
                q=np.zeros((3,), dtype=float),
                lambda_value=float(lam),
            )
        )
        return failures
    if len(pts) > STRICT_TRANSITION_MAX_PATH_POINTS or polyline_length(pts) > STRICT_TRANSITION_MAX_LENGTH:
        failures.append(
            StrictValidationFailure(
                edge_kind=edge_kind,
                message="transition path is not localized enough",
                point_index=0,
                q=np.asarray(pts[0], dtype=float),
                residual=polyline_length(pts),
                lambda_value=float(lam),
            )
        )
    failures.extend(validate_path_on_sphere(pts, side_manifold, side_center, side_radius, edge_kind=edge_kind))
    failures.extend(validate_path_on_family_leaf(pts, transfer_family, lam, edge_kind=edge_kind))
    return failures


def validate_left_motion_edge(path: np.ndarray, left_manifold, left_center: np.ndarray, left_radius: float):
    failures = validate_path_on_sphere(path, left_manifold, left_center, left_radius, edge_kind="left_motion")
    return len(failures) == 0, failures


def validate_right_motion_edge(path: np.ndarray, right_manifold, right_center: np.ndarray, right_radius: float):
    failures = validate_path_on_sphere(path, right_manifold, right_center, right_radius, edge_kind="right_motion")
    return len(failures) == 0, failures


def validate_family_leaf_motion_edge(path: np.ndarray, transfer_family: ContinuousMaskedPlaneFamily, lam: float):
    pts = np.asarray(path, dtype=float)
    failures = validate_path_on_family_leaf(pts, transfer_family, lam, edge_kind="family_leaf_motion")
    failures.extend(
        validate_augmented_family_edge(
            pts,
            np.full((len(pts),), float(lam), dtype=float),
            transfer_family,
            edge_kind="family_leaf_motion",
        )
    )
    return len(failures) == 0, failures


def validate_family_transverse_edge(path: np.ndarray, lambdas: np.ndarray, transfer_family: ContinuousMaskedPlaneFamily):
    pts = np.asarray(path, dtype=float)
    lam_arr = np.asarray(lambdas, dtype=float)
    failures = validate_path_on_family_transverse(pts, lam_arr, transfer_family, edge_kind="family_transverse")
    failures.extend(validate_augmented_family_edge(pts, lam_arr, transfer_family, edge_kind="family_transverse"))
    return len(failures) == 0, failures


def validate_transition_edge(
    path: np.ndarray,
    side_manifold,
    transfer_family: ContinuousMaskedPlaneFamily,
    lam: float,
    side_center: np.ndarray,
    side_radius: float,
    side_mode: str,
):
    edge_kind = "entry_transition" if str(side_mode) == "left" else "exit_transition"
    failures = validate_transition_point(
        path=path,
        side_manifold=side_manifold,
        transfer_family=transfer_family,
        lam=float(lam),
        side_center=np.asarray(side_center, dtype=float),
        side_radius=float(side_radius),
        edge_kind=edge_kind,
    )
    return len(failures) == 0, failures


def certify_left_motion_edge(path: np.ndarray, left_manifold, left_center: np.ndarray, left_radius: float) -> CertificationResult:
    _valid, failures = validate_left_motion_edge(path, left_manifold, left_center, left_radius)
    return CertificationResult.from_failures(failures)


def certify_right_motion_edge(path: np.ndarray, right_manifold, right_center: np.ndarray, right_radius: float) -> CertificationResult:
    _valid, failures = validate_right_motion_edge(path, right_manifold, right_center, right_radius)
    return CertificationResult.from_failures(failures)


def certify_family_leaf_motion_edge(path: np.ndarray, transfer_family: ContinuousMaskedPlaneFamily, lam: float) -> CertificationResult:
    _valid, failures = validate_family_leaf_motion_edge(path, transfer_family, lam)
    return CertificationResult.from_failures(failures)


def certify_family_transverse_edge(path: np.ndarray, lambdas: np.ndarray, transfer_family: ContinuousMaskedPlaneFamily) -> CertificationResult:
    _valid, failures = validate_family_transverse_edge(path, lambdas, transfer_family)
    return CertificationResult.from_failures(failures)


def certify_transition_edge(
    path: np.ndarray,
    side_manifold,
    transfer_family: ContinuousMaskedPlaneFamily,
    lam: float,
    side_center: np.ndarray,
    side_radius: float,
    side_mode: str,
) -> CertificationResult:
    _valid, failures = validate_transition_edge(
        path=path,
        side_manifold=side_manifold,
        transfer_family=transfer_family,
        lam=lam,
        side_center=side_center,
        side_radius=side_radius,
        side_mode=side_mode,
    )
    return CertificationResult.from_failures(failures)


def validate_selected_route_strictly(
    graph: FamilyConnectivityGraph,
    edge_path: list[int],
    left_manifold,
    left_center: np.ndarray,
    left_radius: float,
    transfer_family: ContinuousMaskedPlaneFamily,
    right_manifold,
    right_center: np.ndarray,
    right_radius: float,
) -> tuple[bool, str, list[str], np.ndarray]:
    if len(edge_path) == 0:
        return False, "No selected route was available for strict validation.", [], np.zeros((0, 3), dtype=float)

    failures: list[StrictValidationFailure] = []
    stage = "left"
    entry_transition_count = 0
    exit_transition_count = 0
    family_route_lambdas: list[float] = []
    family_transverse_count = 0

    for route_idx, edge_id in enumerate(edge_path):
        edge = graph.edges[int(edge_id)]
        kind = str(edge.kind)
        if stage == "left":
            allowed = {"left_motion", "entry_transition"}
        elif stage == "family":
            allowed = {"family_leaf_motion", "family_transverse", "exit_transition"}
        else:
            allowed = {"right_motion"}
        if kind not in allowed:
            failures.append(
                StrictValidationFailure(
                    edge_kind=kind,
                    message=f"stage leakage detected; {kind} is not allowed while stage={stage}",
                    point_index=0,
                    q=np.asarray(edge.path[0] if len(edge.path) > 0 else graph.nodes[edge.node_u].q, dtype=float),
                    lambda_value=edge.lambda_value,
                )
            )
            continue

        if kind == "left_motion":
            _ok, edge_failures = validate_left_motion_edge(edge.path, left_manifold, left_center, left_radius)
        elif kind == "right_motion":
            _ok, edge_failures = validate_right_motion_edge(edge.path, right_manifold, right_center, right_radius)
        elif kind == "family_leaf_motion":
            if len(edge.path_lambdas) == 0 and edge.lambda_value is None:
                edge_failures = [
                    StrictValidationFailure(
                        edge_kind=kind,
                        message="family-leaf edge is missing lambda metadata",
                        point_index=0,
                        q=np.asarray(edge.path[0] if len(edge.path) > 0 else graph.nodes[edge.node_u].q, dtype=float),
                    )
                ]
            else:
                edge_lam = float(edge.path_lambdas[0]) if len(edge.path_lambdas) > 0 else float(edge.lambda_value)
                family_route_lambdas.append(float(edge_lam))
                _ok, edge_failures = validate_family_leaf_motion_edge(edge.path, transfer_family, edge_lam)
        elif kind == "family_transverse":
            family_transverse_count += 1
            edge_failures = [
                StrictValidationFailure(
                    edge_kind=kind,
                    message="selected route uses a lambda-changing family edge, which violates the fixed-family-member transfer constraint",
                    point_index=0,
                    q=np.asarray(edge.path[0] if len(edge.path) > 0 else graph.nodes[edge.node_u].q, dtype=float),
                    lambda_value=edge.lambda_value,
                )
            ]
        elif kind == "entry_transition":
            if len(edge.path_lambdas) == 0 and edge.lambda_value is None:
                edge_failures = [
                    StrictValidationFailure(
                        edge_kind=kind,
                        message="entry transition is missing lambda metadata",
                        point_index=0,
                        q=np.asarray(edge.path[0] if len(edge.path) > 0 else graph.nodes[edge.node_u].q, dtype=float),
                    )
                ]
            else:
                edge_lam = float(edge.path_lambdas[0]) if len(edge.path_lambdas) > 0 else float(edge.lambda_value)
                family_route_lambdas.append(float(edge_lam))
                _ok, edge_failures = validate_transition_edge(
                    edge.path,
                    left_manifold,
                    transfer_family,
                    edge_lam,
                    left_center,
                    left_radius,
                    side_mode="left",
                )
            stage = "family"
            entry_transition_count += 1
        elif kind == "exit_transition":
            if len(edge.path_lambdas) == 0 and edge.lambda_value is None:
                edge_failures = [
                    StrictValidationFailure(
                        edge_kind=kind,
                        message="exit transition is missing lambda metadata",
                        point_index=0,
                        q=np.asarray(edge.path[0] if len(edge.path) > 0 else graph.nodes[edge.node_u].q, dtype=float),
                    )
                ]
            else:
                edge_lam = float(edge.path_lambdas[0]) if len(edge.path_lambdas) > 0 else float(edge.lambda_value)
                family_route_lambdas.append(float(edge_lam))
                _ok, edge_failures = validate_transition_edge(
                    edge.path,
                    right_manifold,
                    transfer_family,
                    edge_lam,
                    right_center,
                    right_radius,
                    side_mode="right",
                )
            stage = "right"
            exit_transition_count += 1
        else:
            edge_failures = [
                StrictValidationFailure(
                    edge_kind=kind,
                    message="unsupported edge kind in strict route validation",
                    point_index=0,
                    q=np.asarray(edge.path[0] if len(edge.path) > 0 else graph.nodes[edge.node_u].q, dtype=float),
                )
            ]
        for failure in edge_failures:
            failure.message = f"selected_route_edge={route_idx}: {failure.message}"
        failures.extend(edge_failures)

    if stage != "right":
        last_edge = graph.edges[int(edge_path[-1])]
        last_q = np.asarray(last_edge.path[-1] if len(last_edge.path) > 0 else graph.nodes[last_edge.node_v].q, dtype=float)
        failures.append(
            StrictValidationFailure(
                edge_kind="route",
                message=f"selected route ended before reaching the right-manifold stage; final stage={stage}",
                point_index=0,
                q=last_q,
            )
        )
    if entry_transition_count != 1 or exit_transition_count != 1:
        last_edge = graph.edges[int(edge_path[-1])]
        last_q = np.asarray(last_edge.path[-1] if len(last_edge.path) > 0 else graph.nodes[last_edge.node_v].q, dtype=float)
        failures.append(
            StrictValidationFailure(
                edge_kind="route",
                message=(
                    f"selected route must include exactly one entry transition and one exit transition; "
                    f"got entry={entry_transition_count}, exit={exit_transition_count}"
                ),
                point_index=0,
                q=last_q,
            )
        )
    if family_transverse_count > 0:
        last_edge = graph.edges[int(edge_path[-1])]
        last_q = np.asarray(last_edge.path[-1] if len(last_edge.path) > 0 else graph.nodes[last_edge.node_v].q, dtype=float)
        failures.append(
            StrictValidationFailure(
                edge_kind="route",
                message="selected route includes family_transverse motion, which is forbidden for fixed-lambda transfer",
                point_index=0,
                q=last_q,
            )
        )
    if len(family_route_lambdas) > 0:
        lambda_arr = np.asarray(family_route_lambdas, dtype=float)
        lambda_variation_total = float(np.sum(np.abs(np.diff(lambda_arr)))) if len(lambda_arr) >= 2 else 0.0
        if float(np.max(lambda_arr) - np.min(lambda_arr)) > LAMBDA_SOURCE_TOL or lambda_variation_total > LAMBDA_SOURCE_TOL:
            last_edge = graph.edges[int(edge_path[-1])]
            last_q = np.asarray(last_edge.path[-1] if len(last_edge.path) > 0 else graph.nodes[last_edge.node_v].q, dtype=float)
            failures.append(
                StrictValidationFailure(
                    edge_kind="route",
                    message="selected route changes lambda during transfer; a valid transfer must remain on one fixed family member",
                    point_index=0,
                    q=last_q,
                    lambda_value=float(lambda_arr[0]),
                )
            )

    failure_strings = [format_strict_validation_failure(failure) for failure in failures]
    invalid_points = collect_failure_points(failures)
    if len(failure_strings) == 0:
        return True, "Selected route passed strict manifold validation on every active stage.", [], invalid_points
    summary = f"Strict manifold validation failed on the selected route: {failure_strings[0]}"
    return False, summary, failure_strings, invalid_points


def certify_selected_route_strictly(
    graph: FamilyConnectivityGraph,
    edge_path: list[int],
    left_manifold,
    left_center: np.ndarray,
    left_radius: float,
    transfer_family: ContinuousMaskedPlaneFamily,
    right_manifold,
    right_center: np.ndarray,
    right_radius: float,
) -> CertificationResult:
    valid, _message, failure_strings, _invalid_points = validate_selected_route_strictly(
        graph=graph,
        edge_path=edge_path,
        left_manifold=left_manifold,
        left_center=left_center,
        left_radius=left_radius,
        transfer_family=transfer_family,
        right_manifold=right_manifold,
        right_center=right_center,
        right_radius=right_radius,
    )
    if valid:
        return CertificationResult(valid=True, failures=[])
    return CertificationResult(
        valid=False,
        failures=[
            StrictValidationFailure(
                edge_kind="route",
                message=str(message),
                point_index=idx,
                q=np.zeros((3,), dtype=float),
            )
            for idx, message in enumerate(failure_strings)
        ],
    )
