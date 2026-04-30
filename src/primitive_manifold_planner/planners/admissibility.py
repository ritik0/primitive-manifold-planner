from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence
import numpy as np


@dataclass(frozen=True)
class TransitionContext:
    source_family_name: str
    source_lam: float
    target_family_name: str
    target_lam: float
    point: np.ndarray
    goal_point: np.ndarray | None = None
    metadata: dict | None = None


class TransitionAdmissibilityModel:
    """
    Reusable semantic scoring interface for transition candidates.

    Lower cost means more admissible / preferable. The intent is to separate
    task semantics from geometric transition generation so examples and future
    3D problems can reuse the same planner API.
    """

    def __call__(self, context: TransitionContext) -> float:
        raise NotImplementedError


def family_transition_admissibility_cost(
    family,
    lam: float,
    point: np.ndarray,
    goal_point: Optional[np.ndarray] = None,
    metadata: Optional[dict] = None,
) -> float:
    cost_fn = getattr(family, "transition_admissibility_cost", None)
    if callable(cost_fn):
        return float(
            cost_fn(
                lam=float(lam),
                point=np.asarray(point, dtype=float),
                goal_point=None if goal_point is None else np.asarray(goal_point, dtype=float),
                metadata=metadata,
            )
        )
    return 0.0


def family_transition_feasibility(
    family,
    lam: float,
    point: np.ndarray,
    goal_point: Optional[np.ndarray] = None,
    metadata: Optional[dict] = None,
) -> bool:
    feasibility_fn = getattr(family, "transition_feasibility", None)
    if callable(feasibility_fn):
        return bool(
            feasibility_fn(
                lam=float(lam),
                point=np.asarray(point, dtype=float),
                goal_point=None if goal_point is None else np.asarray(goal_point, dtype=float),
                metadata=metadata,
            )
        )
    return True


class SumAdmissibilityModel(TransitionAdmissibilityModel):
    def __init__(self, terms: Sequence[TransitionAdmissibilityModel]):
        self.terms = list(terms)

    def __call__(self, context: TransitionContext) -> float:
        return float(sum(float(term(context)) for term in self.terms))


class FamilyPairGateModel(TransitionAdmissibilityModel):
    """
    Apply a semantic term only when a transition touches one of the requested
    family names or family pairs.
    """

    def __init__(
        self,
        inner: TransitionAdmissibilityModel,
        family_names: Optional[Iterable[str]] = None,
        family_pairs: Optional[Iterable[tuple[str, str]]] = None,
    ):
        self.inner = inner
        self.family_names = None if family_names is None else {str(name) for name in family_names}
        self.family_pairs = None
        if family_pairs is not None:
            self.family_pairs = {
                tuple(sorted((str(a), str(b))))
                for a, b in family_pairs
            }

    def __call__(self, context: TransitionContext) -> float:
        if self.family_names is not None:
            involved = {context.source_family_name, context.target_family_name}
            if involved.isdisjoint(self.family_names):
                return 0.0
        if self.family_pairs is not None:
            pair = tuple(sorted((context.source_family_name, context.target_family_name)))
            if pair not in self.family_pairs:
                return 0.0
        return float(self.inner(context))


class GoalDistanceAdmissibility(TransitionAdmissibilityModel):
    def __init__(
        self,
        weight: float = 1.0,
        distance_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ):
        self.weight = float(weight)
        self.distance_fn = distance_fn or _euclidean_distance

    def __call__(self, context: TransitionContext) -> float:
        if context.goal_point is None:
            return 0.0
        return self.weight * float(self.distance_fn(context.point, context.goal_point))


class CoordinateAbsoluteValueAdmissibility(TransitionAdmissibilityModel):
    def __init__(self, axis: int, weight: float = 1.0):
        self.axis = int(axis)
        self.weight = float(weight)

    def __call__(self, context: TransitionContext) -> float:
        return self.weight * abs(float(np.asarray(context.point, dtype=float)[self.axis]))


class LambdaPreferenceAdmissibility(TransitionAdmissibilityModel):
    """
    Prefer lambda values near a target value for selected family names.
    """

    def __init__(self, family_name: str, preferred_lam: float, weight: float = 1.0):
        self.family_name = str(family_name)
        self.preferred_lam = float(preferred_lam)
        self.weight = float(weight)

    def __call__(self, context: TransitionContext) -> float:
        penalties = []
        if context.source_family_name == self.family_name:
            penalties.append(abs(context.source_lam - self.preferred_lam))
        if context.target_family_name == self.family_name:
            penalties.append(abs(context.target_lam - self.preferred_lam))
        if not penalties:
            return 0.0
        return self.weight * float(min(penalties))


@dataclass(frozen=True)
class TransitionFeasibilityContext:
    source_family_name: str
    source_lam: float
    target_family_name: str
    target_lam: float
    point: np.ndarray
    goal_point: np.ndarray | None = None
    metadata: dict | None = None


class TransitionFeasibilityModel:
    def __call__(self, context: TransitionFeasibilityContext) -> bool:
        raise NotImplementedError


class AllFeasibilityModel(TransitionFeasibilityModel):
    def __init__(self, terms: Sequence[TransitionFeasibilityModel]):
        self.terms = list(terms)

    def __call__(self, context: TransitionFeasibilityContext) -> bool:
        return all(bool(term(context)) for term in self.terms)


class FamilyPairGateFeasibilityModel(TransitionFeasibilityModel):
    def __init__(
        self,
        inner: TransitionFeasibilityModel,
        family_names: Optional[Iterable[str]] = None,
        family_pairs: Optional[Iterable[tuple[str, str]]] = None,
    ):
        self.inner = inner
        self.family_names = None if family_names is None else {str(name) for name in family_names}
        self.family_pairs = None
        if family_pairs is not None:
            self.family_pairs = {
                tuple(sorted((str(a), str(b))))
                for a, b in family_pairs
            }

    def __call__(self, context: TransitionFeasibilityContext) -> bool:
        if self.family_names is not None:
            involved = {context.source_family_name, context.target_family_name}
            if involved.isdisjoint(self.family_names):
                return True
        if self.family_pairs is not None:
            pair = tuple(sorted((context.source_family_name, context.target_family_name)))
            if pair not in self.family_pairs:
                return True
        return bool(self.inner(context))


class CoordinateRangeFeasibility(TransitionFeasibilityModel):
    def __init__(self, axis: int, min_value: float = -np.inf, max_value: float = np.inf):
        self.axis = int(axis)
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def __call__(self, context: TransitionFeasibilityContext) -> bool:
        value = float(np.asarray(context.point, dtype=float)[self.axis])
        return self.min_value <= value <= self.max_value


class LambdaRangeFeasibility(TransitionFeasibilityModel):
    def __init__(self, family_name: str, min_value: float = -np.inf, max_value: float = np.inf):
        self.family_name = str(family_name)
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def __call__(self, context: TransitionFeasibilityContext) -> bool:
        values = []
        if context.source_family_name == self.family_name:
            values.append(float(context.source_lam))
        if context.target_family_name == self.family_name:
            values.append(float(context.target_lam))
        if not values:
            return True
        return all(self.min_value <= value <= self.max_value for value in values)


def build_transition_admissibility_fn(
    model: TransitionAdmissibilityModel,
) -> Callable[..., float]:
    def _fn(
        source_family,
        source_lam: float,
        target_family,
        target_lam: float,
        x: np.ndarray,
        goal_point: np.ndarray | None,
        metadata: dict | None,
    ) -> float:
        context = TransitionContext(
            source_family_name=str(source_family.name),
            source_lam=float(source_lam),
            target_family_name=str(target_family.name),
            target_lam=float(target_lam),
            point=np.asarray(x, dtype=float),
            goal_point=None if goal_point is None else np.asarray(goal_point, dtype=float),
            metadata=metadata,
        )
        return float(model(context))

    return _fn


def build_transition_feasibility_fn(
    model: TransitionFeasibilityModel,
) -> Callable[..., bool]:
    def _fn(
        source_family,
        source_lam: float,
        target_family,
        target_lam: float,
        x: np.ndarray,
        goal_point: np.ndarray | None,
        metadata: dict | None,
    ) -> bool:
        context = TransitionFeasibilityContext(
            source_family_name=str(source_family.name),
            source_lam=float(source_lam),
            target_family_name=str(target_family.name),
            target_lam=float(target_lam),
            point=np.asarray(x, dtype=float),
            goal_point=None if goal_point is None else np.asarray(goal_point, dtype=float),
            metadata=metadata,
        )
        return bool(model(context))

    return _fn


@dataclass(frozen=True)
class ProgressTargetSelectionConfig:
    goal_distance_weight: float = 1.0
    admissibility_weight: float = 1.0
    anchor_projection_tol: float = 1e-8


def choose_semantic_progress_target(
    fam,
    lam,
    target_leaf,
    goal_point: np.ndarray,
    project_newton: Callable,
    config: Optional[ProgressTargetSelectionConfig] = None,
) -> np.ndarray | None:
    """
    Choose a progress target on a leaf using both geometric closeness to the goal
    and leaf-level admissibility semantics.

    This helps intermediate motion on connectors/support leaves become semantically
    meaningful instead of always following the raw goal projection.
    """
    config = config if config is not None else ProgressTargetSelectionConfig()
    goal = np.asarray(goal_point, dtype=float)

    candidates: list[np.ndarray] = []

    goal_proj = _project_to_leaf(target_leaf=target_leaf, point=goal, project_newton=project_newton)
    if goal_proj is not None:
        candidates.append(goal_proj)

    anchor_fn = getattr(fam, "transition_seed_anchors", None)
    if callable(anchor_fn):
        for anchor in anchor_fn(lam, goal_point=goal):
            proj = _project_to_leaf(
                target_leaf=target_leaf,
                point=np.asarray(anchor, dtype=float),
                project_newton=project_newton,
            )
            if proj is not None:
                candidates.append(proj)

    unique_candidates = _dedupe_points(candidates, tol=config.anchor_projection_tol)
    unique_candidates = [
        point
        for point in unique_candidates
        if family_transition_feasibility(
            family=fam,
            lam=float(lam),
            point=point,
            goal_point=goal,
            metadata={"selector": "semantic_progress_target"},
        )
    ]
    if not unique_candidates:
        return None

    def _score(point: np.ndarray) -> float:
        goal_distance = float(np.linalg.norm(point - goal))
        admissibility_cost = family_transition_admissibility_cost(
            family=fam,
            lam=float(lam),
            point=point,
            goal_point=goal,
            metadata={"selector": "semantic_progress_target"},
        )
        return (
            config.goal_distance_weight * goal_distance
            + config.admissibility_weight * admissibility_cost
        )

    best = min(unique_candidates, key=_score)
    return np.asarray(best, dtype=float).copy()


def build_semantic_target_progress_point_fn(
    config: Optional[ProgressTargetSelectionConfig] = None,
) -> Callable:
    def _fn(
        fam,
        lam,
        target_leaf,
        goal_point: np.ndarray,
        project_newton: Callable,
    ) -> np.ndarray | None:
        return choose_semantic_progress_target(
            fam=fam,
            lam=lam,
            target_leaf=target_leaf,
            goal_point=goal_point,
            project_newton=project_newton,
            config=config,
        )

    return _fn


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _project_to_leaf(target_leaf, point: np.ndarray, project_newton: Callable) -> np.ndarray | None:
    proj = project_newton(
        manifold=target_leaf,
        x0=np.asarray(point, dtype=float),
        tol=1e-10,
        max_iters=60,
        damping=1.0,
    )
    if not proj.success:
        return None
    return np.asarray(proj.x_projected, dtype=float).copy()


def _dedupe_points(points: Sequence[np.ndarray], tol: float = 1e-8) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for point in points:
        q = np.asarray(point, dtype=float).copy()
        if not any(np.linalg.norm(q - u) < tol for u in unique):
            unique.append(q)
    return unique
