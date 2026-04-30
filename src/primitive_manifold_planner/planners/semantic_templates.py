from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Sequence, Set

from primitive_manifold_planner.planners.mode_semantics import (
    CompositePlanningSemanticModel,
    FamilyNameModeSemantics,
    PlanningSemanticModel,
)


def build_family_role_semantic_model(
    role_to_families: Dict[str, Sequence[str]],
    allowed_role_pairs: Set[tuple[str, str]],
    role_transition_costs: Optional[Dict[tuple[str, str], float]] = None,
    family_tags: Optional[Dict[str, Set[str]]] = None,
    default_role: str = "generic",
    transition_feasibility_fn: Optional[Callable] = None,
    transition_admissibility_fn: Optional[Callable] = None,
) -> PlanningSemanticModel:
    family_to_role: Dict[str, str] = {}
    for role, family_names in role_to_families.items():
        for family_name in family_names:
            family_to_role[str(family_name)] = str(role)

    mode_semantics = FamilyNameModeSemantics(
        family_to_role=family_to_role,
        allowed_role_pairs={(str(a), str(b)) for a, b in allowed_role_pairs},
        role_transition_costs={
            (str(a), str(b)): float(cost)
            for (a, b), cost in (role_transition_costs or {}).items()
        },
        family_tags=family_tags,
        default_role=default_role,
    )
    return CompositePlanningSemanticModel(
        mode_semantics=mode_semantics,
        transition_feasibility_fn=transition_feasibility_fn,
        transition_admissibility_fn=transition_admissibility_fn,
    )


def build_support_transfer_goal_semantic_model(
    support_families: Iterable[str],
    transfer_families: Iterable[str],
    goal_support_families: Iterable[str],
    support_transition_cost: float = 0.10,
    goal_transition_cost: float = 0.05,
    transition_feasibility_fn: Optional[Callable] = None,
    transition_admissibility_fn: Optional[Callable] = None,
) -> PlanningSemanticModel:
    return build_family_role_semantic_model(
        role_to_families={
            "support": list(support_families),
            "transfer": list(transfer_families),
            "goal_support": list(goal_support_families),
        },
        allowed_role_pairs={
            ("support", "transfer"),
            ("transfer", "support"),
            ("transfer", "goal_support"),
            ("goal_support", "transfer"),
        },
        role_transition_costs={
            ("support", "transfer"): float(support_transition_cost),
            ("transfer", "support"): float(support_transition_cost),
            ("transfer", "goal_support"): float(goal_transition_cost),
            ("goal_support", "transfer"): float(goal_transition_cost),
        },
        transition_feasibility_fn=transition_feasibility_fn,
        transition_admissibility_fn=transition_admissibility_fn,
    )


def build_support_bridge_transfer_goal_semantic_model(
    support_families: Iterable[str],
    bridge_families: Iterable[str],
    transfer_families: Iterable[str],
    goal_support_families: Iterable[str],
    support_transfer_cost: float = 0.10,
    bridge_transfer_cost: float = 0.20,
    bridge_goal_cost: float = 0.10,
    transfer_goal_cost: float = 0.10,
    transition_feasibility_fn: Optional[Callable] = None,
    transition_admissibility_fn: Optional[Callable] = None,
) -> PlanningSemanticModel:
    return build_family_role_semantic_model(
        role_to_families={
            "support": list(support_families),
            "bridge": list(bridge_families),
            "transfer": list(transfer_families),
            "goal_support": list(goal_support_families),
        },
        allowed_role_pairs={
            ("support", "bridge"),
            ("bridge", "support"),
            ("support", "transfer"),
            ("transfer", "support"),
            ("bridge", "transfer"),
            ("transfer", "bridge"),
            ("bridge", "goal_support"),
            ("goal_support", "bridge"),
            ("transfer", "goal_support"),
            ("goal_support", "transfer"),
        },
        role_transition_costs={
            ("support", "transfer"): float(support_transfer_cost),
            ("transfer", "support"): float(support_transfer_cost),
            ("bridge", "transfer"): float(bridge_transfer_cost),
            ("transfer", "bridge"): float(bridge_transfer_cost),
            ("bridge", "goal_support"): float(bridge_goal_cost),
            ("goal_support", "bridge"): float(bridge_goal_cost),
            ("transfer", "goal_support"): float(transfer_goal_cost),
            ("goal_support", "transfer"): float(transfer_goal_cost),
        },
        transition_feasibility_fn=transition_feasibility_fn,
        transition_admissibility_fn=transition_admissibility_fn,
    )
