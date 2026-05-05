from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from primitive_manifold_planner.planning.local import run_local_planner
from primitive_manifold_planner.planners.admissibility import (
    family_transition_admissibility_cost,
    family_transition_feasibility,
)
from primitive_manifold_planner.planners.mode_semantics import (
    PlanningSemanticContext,
    PlanningSemanticModel,
)
from primitive_manifold_planner.transitions.leaf_transition import (
    find_leaf_transition,
    score_transition_candidate,
)


TransitionNodeKey = Tuple[str, float, str]
TransitionPairKey = Tuple[TransitionNodeKey, TransitionNodeKey]
LeafPairKey = Tuple[str, float, str, float]


@dataclass
class TransitionCandidate:
    src: TransitionNodeKey
    dst: TransitionNodeKey
    transition_point: np.ndarray
    score: float
    candidate_index: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class TransitionSelectionResult:
    success: bool
    candidate: Optional[TransitionCandidate] = None
    local_result: Optional[object] = None
    message: str = ""


@dataclass
class CandidateAttemptStats:
    successes: int = 0
    failures: int = 0

    @property
    def attempts(self) -> int:
        return int(self.successes + self.failures)

    @property
    def success_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return float(self.successes) / float(self.attempts)


@dataclass
class GeneratedTransitionCandidate:
    transition_point: np.ndarray
    residual_norm: float
    score: float
    source_family: str
    source_lam: float
    target_family: str
    target_lam: float
    raw_candidate_index: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class TransitionGenerationResult:
    success: bool
    candidates: List[GeneratedTransitionCandidate]
    message: str = ""
    from_cache: bool = False
    seed_count: int = 0


@dataclass
class TransitionSeedPolicyConfig:
    num_bridge_interpolants: int = 5
    num_anchor_jitter: int = 3
    anchor_jitter_scale: float = 0.08
    num_global_random: int = 12
    global_box_padding: float = 0.75


class AdaptiveTransitionSeedPolicy:
    """
    Generate transition seeds from multiple reusable sources:

    - optional example-specific seeds
    - projections of the goal onto source and target leaves
    - interpolation between those projected anchor points
    - small random jitter around anchor points
    - coarse ambient random samples in a bounding box around the anchors

    This is a modest but meaningful step toward less hand-crafted transition
    discovery and closer to the projection-biased exploration style used in
    constrained planning literature.
    """

    def __init__(
        self,
        project_newton,
        base_seed_points_fn: Optional[Callable] = None,
        component_anchor_fn: Optional[Callable] = None,
        config: Optional[TransitionSeedPolicyConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.project_newton = project_newton
        self.base_seed_points_fn = base_seed_points_fn
        self.component_anchor_fn = component_anchor_fn
        self.config = config if config is not None else TransitionSeedPolicyConfig()
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.last_base_seed_count: int = 0

    def _project_to_leaf(self, leaf, x0: np.ndarray) -> Optional[np.ndarray]:
        proj = self.project_newton(
            manifold=leaf,
            x0=np.asarray(x0, dtype=float),
            tol=1e-10,
            max_iters=60,
            damping=1.0,
        )
        if not proj.success:
            return None
        return np.asarray(proj.x_projected, dtype=float).copy()

    @staticmethod
    def _dedupe(points: List[np.ndarray], tol: float = 1e-8) -> List[np.ndarray]:
        unique: List[np.ndarray] = []
        for point in points:
            q = np.asarray(point, dtype=float).copy()
            if not any(np.linalg.norm(q - u) < tol for u in unique):
                unique.append(q)
        return unique

    def generate(
        self,
        source_family,
        source_lam,
        target_family,
        target_lam,
        goal_point: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        source_leaf = source_family.manifold(source_lam)
        target_leaf = target_family.manifold(target_lam)
        seeds: List[np.ndarray] = []

        if self.base_seed_points_fn is not None:
            base_seeds = [
                np.asarray(s, dtype=float).copy()
                for s in self.base_seed_points_fn(source_family, source_lam, target_family, target_lam)
            ]
            self.last_base_seed_count = len(base_seeds)
            seeds.extend(base_seeds)
        else:
            self.last_base_seed_count = 0

        anchors: List[np.ndarray] = []

        source_family_anchors = getattr(source_family, "transition_seed_anchors", None)
        if callable(source_family_anchors):
            anchors.extend(
                np.asarray(anchor, dtype=float).copy()
                for anchor in source_family_anchors(source_lam, goal_point=goal_point)
            )

        target_family_anchors = getattr(target_family, "transition_seed_anchors", None)
        if callable(target_family_anchors):
            anchors.extend(
                np.asarray(anchor, dtype=float).copy()
                for anchor in target_family_anchors(target_lam, goal_point=goal_point)
            )

        if self.component_anchor_fn is not None:
            anchors.extend(
                np.asarray(anchor, dtype=float).copy()
                for anchor in self.component_anchor_fn(
                    family_name=source_family.name,
                    lam=float(source_lam),
                    goal_point=goal_point,
                )
            )
            anchors.extend(
                np.asarray(anchor, dtype=float).copy()
                for anchor in self.component_anchor_fn(
                    family_name=target_family.name,
                    lam=float(target_lam),
                    goal_point=goal_point,
                )
            )

        if goal_point is not None:
            goal = np.asarray(goal_point, dtype=float)
            anchors.append(goal.copy())

            src_proj = self._project_to_leaf(source_leaf, goal)
            if src_proj is not None:
                anchors.append(src_proj)

            dst_proj = self._project_to_leaf(target_leaf, goal)
            if dst_proj is not None:
                anchors.append(dst_proj)

            if src_proj is not None and dst_proj is not None:
                midpoint = 0.5 * (src_proj + dst_proj)
                anchors.append(midpoint)
                for alpha in np.linspace(0.0, 1.0, self.config.num_bridge_interpolants):
                    anchors.append((1.0 - alpha) * src_proj + alpha * dst_proj)

        seeds.extend(anchors)

        for anchor in anchors:
            for _ in range(self.config.num_anchor_jitter):
                jitter = self.rng.normal(
                    loc=0.0,
                    scale=self.config.anchor_jitter_scale,
                    size=anchor.shape[0],
                )
                seeds.append(anchor + jitter)

        if len(anchors) > 0:
            stacked = np.asarray(anchors, dtype=float)
            lo = np.min(stacked, axis=0) - self.config.global_box_padding
            hi = np.max(stacked, axis=0) + self.config.global_box_padding
            for _ in range(self.config.num_global_random):
                seeds.append(self.rng.uniform(lo, hi))

        return self._dedupe(seeds)


def identity_wrap(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=float).copy()


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def point_within_manifold_bounds(manifold, x: np.ndarray, tol: float = 1e-9) -> bool:
    within_fn = getattr(manifold, "within_bounds", None)
    if callable(within_fn):
        return bool(within_fn(np.asarray(x, dtype=float), tol=tol))
    return True


class TransitionGenerator:
    """
    Reusable exact-transition generation service.

    The key design choice is to cache raw exact intersections per leaf pair, then
    rerank them per planning query using the current goal. This keeps transition
    generation reusable while preserving context-sensitive route quality.
    """

    def __init__(
        self,
        seed_points_fn: Optional[Callable],
        project_newton,
        seed_policy: Optional[AdaptiveTransitionSeedPolicy] = None,
        admissibility_cost_fn: Optional[Callable] = None,
        feasibility_fn: Optional[Callable] = None,
        semantic_model: Optional[PlanningSemanticModel] = None,
    ):
        self.seed_points_fn = seed_points_fn
        self.project_newton = project_newton
        self.seed_policy = seed_policy or AdaptiveTransitionSeedPolicy(
            project_newton=project_newton,
            base_seed_points_fn=seed_points_fn,
        )
        self.admissibility_cost_fn = admissibility_cost_fn
        self.feasibility_fn = feasibility_fn
        self.semantic_model = semantic_model
        self._pair_cache: Dict[LeafPairKey, dict] = {}
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    @staticmethod
    def _pair_key(source_family, source_lam, target_family, target_lam) -> LeafPairKey:
        return (
            str(source_family.name),
            float(source_lam),
            str(target_family.name),
            float(target_lam),
        )

    def _lam_change_penalty(self, source_family, source_lam, target_family, target_lam) -> float:
        try:
            return 0.1 * float(source_family.lambda_distance(source_lam, target_lam))
        except Exception:
            return 0.0

    def _get_or_create_raw_result(
        self,
        source_family,
        source_lam,
        target_family,
        target_lam,
        goal_point: Optional[np.ndarray] = None,
    ) -> Tuple[dict, bool]:
        key = self._pair_key(source_family, source_lam, target_family, target_lam)
        if key in self._pair_cache:
            self.cache_hits += 1
            return self._pair_cache[key], True

        seeds = list(
            self.seed_policy.generate(
                source_family=source_family,
                source_lam=source_lam,
                target_family=target_family,
                target_lam=target_lam,
                goal_point=goal_point,
            )
        )
        base_seed_count = int(getattr(self.seed_policy, "last_base_seed_count", len(seeds)))
        raw_result = find_leaf_transition(
            source_family=source_family,
            source_lam=source_lam,
            target_family=target_family,
            target_lam=target_lam,
            seeds=seeds,
            project_newton=self.project_newton,
            goal=None,
        )

        source_leaf = source_family.manifold(source_lam)
        target_leaf = target_family.manifold(target_lam)
        filtered_candidates = [
            cand
            for cand in raw_result.candidates
            if point_within_manifold_bounds(source_leaf, cand.x)
            and point_within_manifold_bounds(target_leaf, cand.x)
            and family_transition_feasibility(
                family=source_family,
                lam=source_lam,
                point=cand.x,
                goal_point=goal_point,
                metadata={"generator": "exact_cached_transition_generator"},
            )
            and family_transition_feasibility(
                family=target_family,
                lam=target_lam,
                point=cand.x,
                goal_point=goal_point,
                metadata={"generator": "exact_cached_transition_generator"},
            )
            and (
                self.semantic_model is None
                or bool(
                    self.semantic_model.transition_feasible(
                        PlanningSemanticContext(
                            source_family_name=str(source_family.name),
                            source_lam=float(source_lam),
                            target_family_name=str(target_family.name),
                            target_lam=float(target_lam),
                            point=np.asarray(cand.x, dtype=float),
                            goal_point=None if goal_point is None else np.asarray(goal_point, dtype=float),
                            metadata={"generator": "exact_cached_transition_generator"},
                        )
                    )
                )
            )
            and (
                self.feasibility_fn is None
                or bool(
                    self.feasibility_fn(
                        source_family,
                        float(source_lam),
                        target_family,
                        float(target_lam),
                        np.asarray(cand.x, dtype=float),
                        None if goal_point is None else np.asarray(goal_point, dtype=float),
                        {"generator": "exact_cached_transition_generator"},
                    )
                )
            )
        ]
        raw_result.candidates = filtered_candidates
        raw_result.success = len(filtered_candidates) > 0
        raw_result.message = (
            f"Found {len(filtered_candidates)} bounded transition candidate(s)."
        )

        entry = {
            "raw_result": raw_result,
            "seed_count": base_seed_count,
            "expanded_seed_count": len(seeds),
        }
        self._pair_cache[key] = entry
        self.cache_misses += 1
        return entry, False

    def generate_transitions(
        self,
        source_family,
        source_lam,
        target_family,
        target_lam,
        goal_point: Optional[np.ndarray] = None,
        max_candidates: Optional[int] = None,
    ) -> TransitionGenerationResult:
        entry, from_cache = self._get_or_create_raw_result(
            source_family=source_family,
            source_lam=source_lam,
            target_family=target_family,
            target_lam=target_lam,
            goal_point=goal_point,
        )
        raw_result = entry["raw_result"]
        lam_penalty = self._lam_change_penalty(
            source_family=source_family,
            source_lam=source_lam,
            target_family=target_family,
            target_lam=target_lam,
        )

        generated: List[GeneratedTransitionCandidate] = []
        for raw_idx, cand in enumerate(raw_result.candidates):
            metadata = {
                "generator": "exact_cached_transition_generator",
                "from_cache": from_cache,
                "seed_count": int(entry["seed_count"]),
                "expanded_seed_count": int(entry.get("expanded_seed_count", entry["seed_count"])),
                "raw_candidate_index": int(raw_idx),
            }
            base_score = score_transition_candidate(
                x=np.asarray(cand.x, dtype=float),
                residual_norm=float(cand.residual_norm),
                goal=None if goal_point is None else np.asarray(goal_point, dtype=float),
                lam_change_penalty=lam_penalty,
            )
            src_cost = family_transition_admissibility_cost(
                family=source_family,
                lam=source_lam,
                point=cand.x,
                goal_point=goal_point,
                metadata=metadata,
            )
            dst_cost = family_transition_admissibility_cost(
                family=target_family,
                lam=target_lam,
                point=cand.x,
                goal_point=goal_point,
                metadata=metadata,
            )
            pair_cost = 0.0
            if self.admissibility_cost_fn is not None:
                pair_cost = float(
                    self.admissibility_cost_fn(
                        source_family,
                        float(source_lam),
                        target_family,
                        float(target_lam),
                        np.asarray(cand.x, dtype=float),
                        None if goal_point is None else np.asarray(goal_point, dtype=float),
                        metadata,
                    )
                )
            semantic_cost = 0.0
            if self.semantic_model is not None:
                semantic_cost = float(
                    self.semantic_model.transition_admissibility_cost(
                        PlanningSemanticContext(
                            source_family_name=str(source_family.name),
                            source_lam=float(source_lam),
                            target_family_name=str(target_family.name),
                            target_lam=float(target_lam),
                            point=np.asarray(cand.x, dtype=float),
                            goal_point=None if goal_point is None else np.asarray(goal_point, dtype=float),
                            metadata=metadata,
                        )
                    )
                )
            admissibility_cost = float(src_cost + dst_cost + pair_cost + semantic_cost)
            score = float(base_score + admissibility_cost)
            metadata.update(
                {
                    "base_score": float(base_score),
                    "source_admissibility_cost": float(src_cost),
                    "target_admissibility_cost": float(dst_cost),
                    "pair_admissibility_cost": float(pair_cost),
                    "semantic_admissibility_cost": float(semantic_cost),
                    "admissibility_cost": float(admissibility_cost),
                }
            )
            generated.append(
                GeneratedTransitionCandidate(
                    transition_point=np.asarray(cand.x, dtype=float).copy(),
                    residual_norm=float(cand.residual_norm),
                    score=float(score),
                    source_family=str(source_family.name),
                    source_lam=float(source_lam),
                    target_family=str(target_family.name),
                    target_lam=float(target_lam),
                    raw_candidate_index=raw_idx,
                    metadata=metadata,
                )
            )

        generated.sort(key=lambda c: c.score)
        if max_candidates is not None:
            generated = generated[:max_candidates]

        return TransitionGenerationResult(
            success=len(generated) > 0,
            candidates=generated,
            message=raw_result.message,
            from_cache=from_cache,
            seed_count=int(entry["seed_count"]),
        )


class TransitionManager:
    """
    Stores exact transition candidates between node pairs and provides
    reusable ranking + reachable-selection logic.
    """

    def __init__(self):
        self._pair_to_candidates: Dict[TransitionPairKey, List[TransitionCandidate]] = {}
        self._candidate_attempt_stats: Dict[Tuple[TransitionPairKey, int], CandidateAttemptStats] = {}

    def add_candidate(self, candidate: TransitionCandidate) -> None:
        key = (candidate.src, candidate.dst)
        self._pair_to_candidates.setdefault(key, []).append(candidate)

    def add_candidates(self, candidates: List[TransitionCandidate]) -> None:
        for c in candidates:
            self.add_candidate(c)

    def get_candidates(
        self,
        src: TransitionNodeKey,
        dst: TransitionNodeKey,
    ) -> List[TransitionCandidate]:
        return list(self._pair_to_candidates.get((src, dst), []))

    def has_pair(self, src: TransitionNodeKey, dst: TransitionNodeKey) -> bool:
        return (src, dst) in self._pair_to_candidates

    def get_candidate_attempt_stats(
        self,
        src: TransitionNodeKey,
        dst: TransitionNodeKey,
        candidate_index: int,
    ) -> CandidateAttemptStats:
        key = ((src, dst), int(candidate_index))
        return self._candidate_attempt_stats.get(key, CandidateAttemptStats())

    def record_candidate_attempt(
        self,
        src: TransitionNodeKey,
        dst: TransitionNodeKey,
        candidate_index: int,
        success: bool,
    ) -> None:
        key = ((src, dst), int(candidate_index))
        stats = self._candidate_attempt_stats.setdefault(key, CandidateAttemptStats())
        if success:
            stats.successes += 1
        else:
            stats.failures += 1

    def rank_candidates(
        self,
        src: TransitionNodeKey,
        dst: TransitionNodeKey,
        current_x: np.ndarray,
        wrap_state_fn: Callable[[np.ndarray], np.ndarray] = identity_wrap,
        state_distance_fn: Callable[[np.ndarray, np.ndarray], float] = euclidean_distance,
        downstream_hint: Optional[np.ndarray] = None,
        transition_penalty_fn: Optional[Callable[[np.ndarray], float]] = None,
        candidate_score_weight: float = 0.5,
        downstream_weight: float = 0.5,
        historical_success_bonus: float = 0.35,
        historical_failure_penalty: float = 0.15,
    ) -> List[TransitionCandidate]:
        candidates = self.get_candidates(src, dst)
        if len(candidates) == 0:
            return []

        current_x_wrapped = wrap_state_fn(current_x)
        downstream_hint_wrapped = None if downstream_hint is None else wrap_state_fn(downstream_hint)

        def rank_key(candidate: TransitionCandidate):
            q = wrap_state_fn(candidate.transition_point)
            score = state_distance_fn(q, current_x_wrapped)
            if downstream_hint_wrapped is not None:
                score += downstream_weight * state_distance_fn(q, downstream_hint_wrapped)
            score += candidate_score_weight * float(candidate.score)
            if transition_penalty_fn is not None:
                score += float(transition_penalty_fn(q))
            stats = self.get_candidate_attempt_stats(src, dst, candidate.candidate_index)
            if stats.attempts > 0:
                score -= historical_success_bonus * float(stats.success_rate)
                score += historical_failure_penalty * float(stats.failures)
            return score

        return sorted(candidates, key=rank_key)

    def select_reachable_candidate(
        self,
        src: TransitionNodeKey,
        dst: TransitionNodeKey,
        current_x: np.ndarray,
        families,
        step_size: float,
        local_planner_name: str,
        local_planner_kwargs: dict,
        wrap_state_fn: Callable[[np.ndarray], np.ndarray] = identity_wrap,
        state_distance_fn: Callable[[np.ndarray, np.ndarray], float] = euclidean_distance,
        downstream_hint: Optional[np.ndarray] = None,
        transition_penalty_fn: Optional[Callable[[np.ndarray], float]] = None,
        candidate_score_weight: float = 0.5,
        downstream_weight: float = 0.5,
        require_downstream_reachable: bool = True,
        historical_success_bonus: float = 0.35,
        historical_failure_penalty: float = 0.15,
    ) -> TransitionSelectionResult:
        family_map = {f.name: f for f in families}

        if src[0] not in family_map or dst[0] not in family_map:
            return TransitionSelectionResult(
                success=False,
                message=f"Unknown family in src={src} or dst={dst}.",
            )

        src_leaf = family_map[src[0]].manifold(src[1])
        dst_leaf = family_map[dst[0]].manifold(dst[1])

        ranked = self.rank_candidates(
            src=src,
            dst=dst,
            current_x=current_x,
            wrap_state_fn=wrap_state_fn,
            state_distance_fn=state_distance_fn,
            downstream_hint=downstream_hint,
            transition_penalty_fn=transition_penalty_fn,
            candidate_score_weight=candidate_score_weight,
            downstream_weight=downstream_weight,
            historical_success_bonus=historical_success_bonus,
            historical_failure_penalty=historical_failure_penalty,
        )

        if len(ranked) == 0:
            return TransitionSelectionResult(
                success=False,
                message=f"No exact transition candidates stored for {src} -> {dst}.",
            )

        current_x_wrapped = wrap_state_fn(current_x)
        downstream_hint_wrapped = None if downstream_hint is None else wrap_state_fn(downstream_hint)

        for candidate in ranked:
            q_edge = wrap_state_fn(candidate.transition_point)

            local_result = run_local_planner(
                manifold=src_leaf,
                x_start=current_x_wrapped,
                x_goal=q_edge,
                planner_name=local_planner_name,
                step_size=step_size,
                **local_planner_kwargs,
            )
            if not local_result.success or len(local_result.path) == 0:
                self.record_candidate_attempt(src, dst, candidate.candidate_index, success=False)
                continue

            if require_downstream_reachable and downstream_hint_wrapped is not None:
                downstream_trial = run_local_planner(
                    manifold=dst_leaf,
                    x_start=q_edge,
                    x_goal=downstream_hint_wrapped,
                    planner_name=local_planner_name,
                    step_size=step_size,
                    **local_planner_kwargs,
                )
                if not downstream_trial.success or len(downstream_trial.path) == 0:
                    self.record_candidate_attempt(src, dst, candidate.candidate_index, success=False)
                    continue

            self.record_candidate_attempt(src, dst, candidate.candidate_index, success=True)

            return TransitionSelectionResult(
                success=True,
                candidate=candidate,
                local_result=local_result,
                message=(
                    f"Selected reachable candidate_index={candidate.candidate_index} "
                    f"for {src} -> {dst}."
                ),
            )

        return TransitionSelectionResult(
            success=False,
            message=(
                f"No reachable exact transition candidate found for {src} -> {dst} "
                f"from current_x={np.round(current_x_wrapped, 4)}."
            ),
        )
