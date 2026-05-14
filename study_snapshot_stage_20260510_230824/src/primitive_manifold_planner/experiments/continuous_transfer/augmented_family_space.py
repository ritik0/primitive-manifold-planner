"""Augmented family-space primitives for Example 65.

Phase 1 keeps the current Example 65 planner runnable while shifting the middle
transfer family toward one coherent constrained state space over ``x=(q,lambda)``.
The family stage can still use bounded candidate lambdas and existing graph
machinery, but its projections, validity checks, and route semantics can now be
expressed through a single augmented constrained-space adapter instead of only
through separate leaf-local and transverse special cases.

Phase 2 can replace the bounded candidate-lambda generation below with a more
intrinsic augmented constrained-space sampler without changing the planner-facing
interface introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import FAMILY_TRANSVERSE_DELTA, LAMBDA_BIN_WIDTH, LAMBDA_SOURCE_TOL, TRANSVERSE_GOAL_TOL, TRANSVERSE_LAMBDA_STEP, TRANSVERSE_PATCH_STEP
from .family_definition import ContinuousMaskedPlaneFamily
from .projection_utils import project_valid_family_state
from .support import ompl_native_exploration_target, solve_exact_segment_on_manifold


@dataclass(frozen=True)
class FamilyAugmentedState:
    """Primary family-stage planning state in the augmented ``(q, lambda)`` space."""

    q: np.ndarray
    lambda_value: float
    discovered_round: int
    origin_sample_id: int | None = None
    expansion_count: int = 0
    region_id: int | None = None
    node_id: int | None = None
    kind: str = "explored"

    @property
    def x(self) -> np.ndarray:
        q = np.asarray(self.q, dtype=float).reshape(-1)
        return np.concatenate([q, np.asarray([float(self.lambda_value)], dtype=float)])


@dataclass(frozen=True)
class AugmentedProjectionResult:
    state: FamilyAugmentedState
    residual: np.ndarray


@dataclass(frozen=True)
class AugmentedFamilyPath:
    """Certified local motion in the augmented family space."""

    success: bool
    path: np.ndarray
    lambdas: np.ndarray
    motion_kind: str
    message: str = ""


@dataclass(frozen=True)
class AugmentedSamplerBatch:
    """One proposal batch that can support multiple family hypotheses in parallel."""

    sample_id: int
    states: list[FamilyAugmentedState] = field(default_factory=list)


def family_edge_lambda_mode(lambdas: np.ndarray, tol: float = 2e-3) -> str:
    lam_arr = np.asarray(lambdas, dtype=float).reshape(-1)
    if len(lam_arr) <= 1:
        return "constant_lambda"
    return "constant_lambda" if float(np.max(lam_arr) - np.min(lam_arr)) <= float(tol) else "changing_lambda"


class AugmentedFamilyConstrainedSpace:
    """Thin augmented constrained-space adapter for the continuous masked plane family."""

    def __init__(self, transfer_family: ContinuousMaskedPlaneFamily):
        self.transfer_family = transfer_family
        self.ambient_q_dim = 3
        self.augmented_dim = 4
        self.codim = 1
        self.default_lambda_scale = max(float(LAMBDA_BIN_WIDTH), 0.08)

    def split(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.shape[0] != self.augmented_dim:
            raise ValueError(f"Expected augmented state of length {self.augmented_dim}, got {arr.shape[0]}.")
        return np.asarray(arr[: self.ambient_q_dim], dtype=float), float(arr[-1])

    def join(self, q: np.ndarray, lambda_value: float) -> np.ndarray:
        return np.concatenate(
            [np.asarray(q, dtype=float).reshape(-1), np.asarray([float(lambda_value)], dtype=float)],
            dtype=float,
        )

    def residual_augmented(self, x: np.ndarray) -> np.ndarray:
        q, lam = self.split(x)
        residual = float(self.transfer_family.infer_lambda(q) - float(lam))
        return np.asarray([residual], dtype=float)

    def jacobian_augmented(self, _x: np.ndarray) -> np.ndarray:
        normal = np.asarray(self.transfer_family.normal, dtype=float).reshape(1, -1)
        return np.concatenate([normal, np.asarray([[-1.0]], dtype=float)], axis=1)

    def clamp_lambda(self, lambda_value: float) -> float:
        return float(
            np.clip(
                float(lambda_value),
                float(self.transfer_family.lambda_min),
                float(self.transfer_family.lambda_max),
            )
        )

    def is_valid_augmented(self, x: np.ndarray, tol: float = 1e-9) -> bool:
        q, lam = self.split(x)
        residual_ok = float(np.linalg.norm(self.residual_augmented(x))) <= max(1e-6, float(tol))
        return bool(
            residual_ok
            and self.transfer_family.lambda_in_range(float(lam), tol=float(tol))
            and self.transfer_family.within_patch(float(lam), np.asarray(q, dtype=float), tol=float(tol))
        )

    def project_augmented(
        self,
        x_guess: np.ndarray,
        lambda_hint: float | None = None,
        discovered_round: int = 0,
        origin_sample_id: int | None = None,
        region_id: int | None = None,
        node_id: int | None = None,
        kind: str = "explored",
    ) -> FamilyAugmentedState | None:
        arr = np.asarray(x_guess, dtype=float).reshape(-1)
        if arr.shape[0] < self.ambient_q_dim:
            return None
        q_guess = np.asarray(arr[: self.ambient_q_dim], dtype=float)
        raw_lambda = (
            float(lambda_hint)
            if lambda_hint is not None
            else float(arr[-1]) if arr.shape[0] >= self.augmented_dim else float(self.transfer_family.infer_lambda(q_guess))
        )
        lam = self.clamp_lambda(raw_lambda)
        q_projected = project_valid_family_state(self.transfer_family, float(lam), q_guess)
        if q_projected is None:
            return None
        state = FamilyAugmentedState(
            q=np.asarray(q_projected, dtype=float),
            lambda_value=float(lam),
            discovered_round=int(discovered_round),
            origin_sample_id=origin_sample_id,
            expansion_count=0,
            region_id=None if region_id is None else int(region_id),
            node_id=None if node_id is None else int(node_id),
            kind=str(kind),
        )
        if not self.is_valid_augmented(state.x):
            return None
        return state

    def proposal_candidate_lambdas(
        self,
        proposal_q: np.ndarray,
        active_region_lambdas: list[float] | None = None,
        adaptive_lambda_values: set[float] | None = None,
        target_lambda: float | None = None,
        nominal_lambda: float | None = None,
        neighbor_bins: int = 2,
    ) -> list[float]:
        inferred = self.clamp_lambda(self.transfer_family.infer_lambda(np.asarray(proposal_q, dtype=float)))
        candidates: list[float] = [float(inferred)]
        if target_lambda is not None:
            candidates.append(self.clamp_lambda(float(target_lambda)))
        if nominal_lambda is not None:
            candidates.append(self.clamp_lambda(float(nominal_lambda)))
        if active_region_lambdas:
            candidates.extend(self.clamp_lambda(float(value)) for value in active_region_lambdas)
        if adaptive_lambda_values:
            candidates.extend(self.clamp_lambda(float(value)) for value in adaptive_lambda_values)

        augmented: list[float] = []
        step = float(LAMBDA_BIN_WIDTH)
        for value in candidates:
            base_value = self.clamp_lambda(value)
            augmented.append(base_value)
            for offset in range(1, max(0, int(neighbor_bins)) + 1):
                augmented.append(self.clamp_lambda(base_value + offset * step))
                augmented.append(self.clamp_lambda(base_value - offset * step))

        deduped: list[float] = []
        for value in augmented:
            snapped = round(float(value) / step) * step if step > 0.0 else float(value)
            snapped = self.clamp_lambda(snapped)
            if any(abs(float(snapped) - float(existing)) <= 0.5 * step for existing in deduped):
                continue
            deduped.append(float(snapped))
        return deduped

    def local_distance(
        self,
        state_a: FamilyAugmentedState,
        state_b: FamilyAugmentedState,
        lambda_weight: float = 1.6,
    ) -> float:
        q_dist = float(np.linalg.norm(np.asarray(state_a.q, dtype=float) - np.asarray(state_b.q, dtype=float)))
        lam_dist = abs(float(state_a.lambda_value) - float(state_b.lambda_value))
        return float(q_dist + float(lambda_weight) * lam_dist)

    def lambda_varying(self, lambdas: np.ndarray, tol: float = LAMBDA_SOURCE_TOL) -> bool:
        return family_edge_lambda_mode(np.asarray(lambdas, dtype=float), tol=float(tol)) == "changing_lambda"

    def edge_kind_for_lambdas(self, lambdas: np.ndarray, tol: float = LAMBDA_SOURCE_TOL) -> str:
        return "family_transverse" if self.lambda_varying(lambdas, tol=float(tol)) else "family_leaf_motion"

    def sample_augmented_hypotheses(
        self,
        proposal_q: np.ndarray,
        *,
        sample_id: int,
        discovered_round: int,
        active_region_lambdas: list[float] | None,
        adaptive_lambda_values: set[float] | None,
        target_lambda: float | None,
        nominal_lambda: float | None,
        rng: np.random.Generator,
        max_keep: int = 4,
    ) -> AugmentedSamplerBatch:
        q_anchor = np.asarray(proposal_q, dtype=float)
        candidate_lambdas = self.proposal_candidate_lambdas(
            proposal_q=q_anchor,
            active_region_lambdas=active_region_lambdas,
            adaptive_lambda_values=adaptive_lambda_values,
            target_lambda=target_lambda,
            nominal_lambda=nominal_lambda,
            neighbor_bins=2,
        )
        guessed_lambdas = list(candidate_lambdas)
        guessed_lambdas.append(self.clamp_lambda(rng.uniform(self.transfer_family.lambda_min, self.transfer_family.lambda_max)))
        states: list[FamilyAugmentedState] = []
        for lam in guessed_lambdas:
            jitter = rng.normal(scale=np.array([0.18, 0.22, 0.08], dtype=float), size=3)
            state = self.project_augmented(
                x_guess=self.join(q_anchor + jitter, float(lam)),
                lambda_hint=float(lam),
                discovered_round=int(discovered_round),
                origin_sample_id=int(sample_id),
                kind="proposal",
            )
            if state is None:
                continue
            if any(
                np.linalg.norm(np.asarray(state.q, dtype=float) - np.asarray(existing.q, dtype=float)) <= 1e-4
                and abs(float(state.lambda_value) - float(existing.lambda_value)) <= 0.5 * float(LAMBDA_BIN_WIDTH)
                for existing in states
            ):
                continue
            states.append(state)
            if len(states) >= int(max_keep):
                break
        return AugmentedSamplerBatch(sample_id=int(sample_id), states=states)

    def sample_local_state(
        self,
        source_state: FamilyAugmentedState,
        *,
        rng: np.random.Generator,
        bounds_min: np.ndarray,
        bounds_max: np.ndarray,
        target_lambda: float | None = None,
    ) -> FamilyAugmentedState | None:
        lam_src = float(source_state.lambda_value)
        if target_lambda is None:
            lambda_guess = self.clamp_lambda(
                lam_src + float(rng.normal(scale=0.55 * self.default_lambda_scale))
            )
        else:
            step = float(np.clip(float(target_lambda) - lam_src, -FAMILY_TRANSVERSE_DELTA, FAMILY_TRANSVERSE_DELTA))
            lambda_guess = self.clamp_lambda(lam_src + step)

        manifold = self.transfer_family.manifold(lam_src)
        local_target = ompl_native_exploration_target(
            manifold=manifold,
            q_seed=np.asarray(source_state.q, dtype=float),
            bounds_min=np.asarray(bounds_min, dtype=float),
            bounds_max=np.asarray(bounds_max, dtype=float),
        )
        if local_target is None:
            u_coord, v_coord = self.transfer_family.patch_coords(lam_src, np.asarray(source_state.q, dtype=float))
            du = float(rng.normal(scale=0.28))
            dv = float(rng.normal(scale=0.42))
            local_target = (
                self.transfer_family.point_on_leaf(lam_src)
                + (u_coord + du) * np.asarray(self.transfer_family._basis_u, dtype=float)
                + (v_coord + dv) * np.asarray(self.transfer_family._basis_v, dtype=float)
            )
        return self.project_augmented(
            x_guess=self.join(np.asarray(local_target, dtype=float), float(lambda_guess)),
            lambda_hint=float(lambda_guess),
            discovered_round=int(source_state.discovered_round),
            origin_sample_id=source_state.origin_sample_id,
            region_id=source_state.region_id,
            node_id=source_state.node_id,
            kind="explored",
        )

    def _certify_lambda_varying_path(
        self,
        source_state: FamilyAugmentedState,
        target_state: FamilyAugmentedState,
    ) -> AugmentedFamilyPath:
        lam0 = float(source_state.lambda_value)
        lam1 = float(target_state.lambda_value)
        q0 = np.asarray(source_state.q, dtype=float)
        q1 = np.asarray(target_state.q, dtype=float)
        u0, v0 = self.transfer_family.patch_coords(lam0, q0)
        u1, v1 = self.transfer_family.patch_coords(lam1, q1)
        step_count = max(
            2,
            int(np.ceil(abs(lam1 - lam0) / TRANSVERSE_LAMBDA_STEP)),
            int(np.ceil(max(abs(u1 - u0), abs(v1 - v0)) / TRANSVERSE_PATCH_STEP)),
            int(np.ceil(np.linalg.norm(q1 - q0) / TRANSVERSE_PATCH_STEP)),
        )
        path: list[np.ndarray] = []
        lambdas: list[float] = []
        for t in np.linspace(0.0, 1.0, step_count + 1):
            lam = (1.0 - float(t)) * lam0 + float(t) * lam1
            u_coord = (1.0 - float(t)) * u0 + float(t) * u1
            v_coord = (1.0 - float(t)) * v0 + float(t) * v1
            guess = (
                self.transfer_family.point_on_leaf(lam)
                + u_coord * np.asarray(self.transfer_family._basis_u, dtype=float)
                + v_coord * np.asarray(self.transfer_family._basis_v, dtype=float)
            )
            projected = self.project_augmented(self.join(guess, lam), lambda_hint=float(lam))
            if projected is None:
                return AugmentedFamilyPath(False, np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float), "changing_lambda", "projection failed")
            path.append(np.asarray(projected.q, dtype=float))
            lambdas.append(float(projected.lambda_value))
        path_arr = np.asarray(path, dtype=float)
        lam_arr = np.asarray(lambdas, dtype=float)
        if np.linalg.norm(path_arr[-1] - q1) > TRANSVERSE_GOAL_TOL:
            return AugmentedFamilyPath(False, np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float), "changing_lambda", "goal mismatch")
        return AugmentedFamilyPath(True, path_arr, lam_arr, "changing_lambda")

    def plan_local_motion(
        self,
        source_state: FamilyAugmentedState,
        target_state: FamilyAugmentedState,
        *,
        bounds_min: np.ndarray,
        bounds_max: np.ndarray,
    ) -> AugmentedFamilyPath:
        lambda_gap = abs(float(source_state.lambda_value) - float(target_state.lambda_value))
        if lambda_gap <= float(LAMBDA_SOURCE_TOL):
            result = solve_exact_segment_on_manifold(
                manifold=self.transfer_family.manifold(float(source_state.lambda_value)),
                x_start=np.asarray(source_state.q, dtype=float),
                x_goal=np.asarray(target_state.q, dtype=float),
                bounds_min=np.asarray(bounds_min, dtype=float),
                bounds_max=np.asarray(bounds_max, dtype=float),
            )
            path = np.asarray(getattr(result, "path", np.zeros((0, 3), dtype=float)), dtype=float)
            if len(path) == 0:
                return AugmentedFamilyPath(False, np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float), "constant_lambda", "exact local planner failed")
            lambdas = np.full((len(path),), float(source_state.lambda_value), dtype=float)
            return AugmentedFamilyPath(True, path, lambdas, "constant_lambda")
        return self._certify_lambda_varying_path(source_state, target_state)

    def state_from_graph_node(self, node, region_id: int | None = None) -> FamilyAugmentedState | None:
        lambda_value = getattr(node, "lambda_value", None)
        if getattr(node, "mode", None) != "family" or lambda_value is None:
            return None
        return FamilyAugmentedState(
            q=np.asarray(node.q, dtype=float).copy(),
            lambda_value=float(lambda_value),
            discovered_round=int(node.discovered_round),
            origin_sample_id=getattr(node, "origin_sample_id", None),
            expansion_count=int(getattr(node, "expansion_count", 0)),
            region_id=None if region_id is None else int(region_id),
            node_id=int(node.node_id),
            kind=str(getattr(node, "kind", "explored")),
        )
