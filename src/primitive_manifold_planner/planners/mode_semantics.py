from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Set, Tuple
import numpy as np


ModeRole = str
ModeRolePair = Tuple[ModeRole, ModeRole]


@dataclass(frozen=True)
class LeafMode:
    family_name: str
    lam: float
    role: ModeRole
    tags: frozenset[str] = field(default_factory=frozenset)


class ModeSemantics:
    """
    Reusable layer for planner-facing mode semantics.

    This separates geometric family names from task roles such as:
    - support
    - transfer
    - connector
    - goal_support
    - grasp
    - handover

    The planner can then reason over allowed transitions and semantic edge
    costs using roles instead of example-specific family-name logic.
    """

    def role_for_leaf(self, family_name: str, lam: float) -> ModeRole:
        raise NotImplementedError

    def tags_for_leaf(self, family_name: str, lam: float) -> Set[str]:
        _ = family_name, lam
        return set()

    def describe_leaf(self, family_name: str, lam: float) -> LeafMode:
        return LeafMode(
            family_name=str(family_name),
            lam=float(lam),
            role=str(self.role_for_leaf(family_name, lam)),
            tags=frozenset(self.tags_for_leaf(family_name, lam)),
        )

    def transition_allowed(
        self,
        source_family_name: str,
        source_lam: float,
        target_family_name: str,
        target_lam: float,
    ) -> bool:
        _ = source_family_name, source_lam, target_family_name, target_lam
        return True

    def transition_cost(
        self,
        source_family_name: str,
        source_lam: float,
        target_family_name: str,
        target_lam: float,
    ) -> float:
        _ = source_family_name, source_lam, target_family_name, target_lam
        return 0.0


class FamilyNameModeSemantics(ModeSemantics):
    """
    Lightweight reusable mode semantics driven by family-name assignment.
    """

    def __init__(
        self,
        family_to_role: Dict[str, str],
        allowed_role_pairs: Optional[Set[ModeRolePair]] = None,
        role_transition_costs: Optional[Dict[ModeRolePair, float]] = None,
        family_tags: Optional[Dict[str, Set[str]]] = None,
        default_role: str = "generic",
    ):
        self.family_to_role = {str(k): str(v) for k, v in family_to_role.items()}
        self.allowed_role_pairs = None
        if allowed_role_pairs is not None:
            self.allowed_role_pairs = {
                (str(a), str(b))
                for a, b in allowed_role_pairs
            }
        self.role_transition_costs = {
            (str(a), str(b)): float(cost)
            for (a, b), cost in (role_transition_costs or {}).items()
        }
        self.family_tags = {
            str(k): {str(tag) for tag in v}
            for k, v in (family_tags or {}).items()
        }
        self.default_role = str(default_role)

    def role_for_leaf(self, family_name: str, lam: float) -> ModeRole:
        _ = lam
        return self.family_to_role.get(str(family_name), self.default_role)

    def tags_for_leaf(self, family_name: str, lam: float) -> Set[str]:
        _ = lam
        return set(self.family_tags.get(str(family_name), set()))

    def transition_allowed(
        self,
        source_family_name: str,
        source_lam: float,
        target_family_name: str,
        target_lam: float,
    ) -> bool:
        if self.allowed_role_pairs is None:
            return True
        src_role = self.role_for_leaf(source_family_name, source_lam)
        dst_role = self.role_for_leaf(target_family_name, target_lam)
        return (src_role, dst_role) in self.allowed_role_pairs

    def transition_cost(
        self,
        source_family_name: str,
        source_lam: float,
        target_family_name: str,
        target_lam: float,
    ) -> float:
        src_role = self.role_for_leaf(source_family_name, source_lam)
        dst_role = self.role_for_leaf(target_family_name, target_lam)
        return float(self.role_transition_costs.get((src_role, dst_role), 0.0))


@dataclass(frozen=True)
class PlanningSemanticContext:
    source_family_name: str
    source_lam: float
    target_family_name: str
    target_lam: float
    point: np.ndarray | None = None
    goal_point: np.ndarray | None = None
    metadata: dict | None = None


class PlanningSemanticModel:
    """
    Unified semantic interface for multimodal planning.

    This combines:
    - mode role semantics
    - transition permission
    - semantic transition cost
    - semantic feasibility
    - semantic admissibility cost

    The goal is to let planners depend on one semantic layer instead of stitching
    together several ad hoc callbacks.
    """

    def describe_leaf(self, family_name: str, lam: float) -> LeafMode:
        raise NotImplementedError

    def transition_allowed(self, context: PlanningSemanticContext) -> bool:
        _ = context
        return True

    def transition_cost(self, context: PlanningSemanticContext) -> float:
        _ = context
        return 0.0

    def transition_feasible(self, context: PlanningSemanticContext) -> bool:
        _ = context
        return True

    def transition_admissibility_cost(self, context: PlanningSemanticContext) -> float:
        _ = context
        return 0.0


class ModeSemanticsAdapter(PlanningSemanticModel):
    def __init__(self, mode_semantics: ModeSemantics):
        self.mode_semantics = mode_semantics

    def describe_leaf(self, family_name: str, lam: float) -> LeafMode:
        return self.mode_semantics.describe_leaf(family_name=family_name, lam=lam)

    def transition_allowed(self, context: PlanningSemanticContext) -> bool:
        return bool(
            self.mode_semantics.transition_allowed(
                source_family_name=context.source_family_name,
                source_lam=context.source_lam,
                target_family_name=context.target_family_name,
                target_lam=context.target_lam,
            )
        )

    def transition_cost(self, context: PlanningSemanticContext) -> float:
        return float(
            self.mode_semantics.transition_cost(
                source_family_name=context.source_family_name,
                source_lam=context.source_lam,
                target_family_name=context.target_family_name,
                target_lam=context.target_lam,
            )
        )

    def transition_feasible(self, context: PlanningSemanticContext) -> bool:
        _ = context
        return True

    def transition_admissibility_cost(self, context: PlanningSemanticContext) -> float:
        _ = context
        return 0.0


class CompositePlanningSemanticModel(PlanningSemanticModel):
    """
    Compose a mode layer with optional feasibility/admissibility callbacks.
    """

    def __init__(
        self,
        mode_semantics: ModeSemantics,
        transition_feasibility_fn: Optional[Callable[[PlanningSemanticContext], bool]] = None,
        transition_admissibility_fn: Optional[Callable[[PlanningSemanticContext], float]] = None,
    ):
        self.mode_semantics = mode_semantics
        self.transition_feasibility_fn = transition_feasibility_fn
        self.transition_admissibility_fn = transition_admissibility_fn

    def describe_leaf(self, family_name: str, lam: float) -> LeafMode:
        return self.mode_semantics.describe_leaf(family_name=family_name, lam=lam)

    def transition_allowed(self, context: PlanningSemanticContext) -> bool:
        return bool(
            self.mode_semantics.transition_allowed(
                source_family_name=context.source_family_name,
                source_lam=context.source_lam,
                target_family_name=context.target_family_name,
                target_lam=context.target_lam,
            )
        )

    def transition_cost(self, context: PlanningSemanticContext) -> float:
        return float(
            self.mode_semantics.transition_cost(
                source_family_name=context.source_family_name,
                source_lam=context.source_lam,
                target_family_name=context.target_family_name,
                target_lam=context.target_lam,
            )
        )

    def transition_feasible(self, context: PlanningSemanticContext) -> bool:
        if self.transition_feasibility_fn is None:
            return True
        return bool(self.transition_feasibility_fn(context))

    def transition_admissibility_cost(self, context: PlanningSemanticContext) -> float:
        if self.transition_admissibility_fn is None:
            return 0.0
        return float(self.transition_admissibility_fn(context))


def build_allowed_family_pair_fn(mode_semantics: ModeSemantics) -> Callable[[str, str], bool]:
    """
    Compatibility helper for older planner paths that only expose family names.

    This uses lambda=0.0 as a placeholder; the full lambda-aware API should be
    preferred in new code paths.
    """

    def _fn(source_family_name: str, target_family_name: str) -> bool:
        return bool(
            mode_semantics.transition_allowed(
                source_family_name=str(source_family_name),
                source_lam=0.0,
                target_family_name=str(target_family_name),
                target_lam=0.0,
            )
        )

    return _fn


def build_allowed_leaf_pair_fn(mode_semantics: ModeSemantics) -> Callable[[str, float, str, float], bool]:
    def _fn(source_family_name: str, source_lam: float, target_family_name: str, target_lam: float) -> bool:
        return bool(
            mode_semantics.transition_allowed(
                source_family_name=str(source_family_name),
                source_lam=float(source_lam),
                target_family_name=str(target_family_name),
                target_lam=float(target_lam),
            )
        )

    return _fn


def build_semantic_model_allowed_leaf_pair_fn(
    semantic_model: PlanningSemanticModel,
) -> Callable[[str, float, str, float], bool]:
    def _fn(source_family_name: str, source_lam: float, target_family_name: str, target_lam: float) -> bool:
        return bool(
            semantic_model.transition_allowed(
                PlanningSemanticContext(
                    source_family_name=str(source_family_name),
                    source_lam=float(source_lam),
                    target_family_name=str(target_family_name),
                    target_lam=float(target_lam),
                )
            )
        )

    return _fn
