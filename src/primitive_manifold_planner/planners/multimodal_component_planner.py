from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Any
import numpy as np

from primitive_manifold_planner.families.leaf_state import LeafState
from primitive_manifold_planner.planners.component_leaf_graph import (
    shortest_component_route,
)
from primitive_manifold_planner.planners.component_graph_with_transitions import (
    build_component_graph_with_transition_manager,
)
from primitive_manifold_planner.planners.component_discovery import ComponentModelRegistry
from primitive_manifold_planner.planners.realize_component_route_with_manager import (
    realize_component_route_with_manager,
)
from primitive_manifold_planner.planners.mode_semantics import (
    CompositePlanningSemanticModel,
    ModeSemantics,
    ModeSemanticsAdapter,
    PlanningSemanticContext,
    PlanningSemanticModel,
    build_allowed_leaf_pair_fn,
    build_semantic_model_allowed_leaf_pair_fn,
)
from primitive_manifold_planner.planners.transition_manager import (
    AdaptiveTransitionSeedPolicy,
    TransitionGenerator,
    identity_wrap,
    euclidean_distance,
)


@dataclass
class PlannerConfig:
    max_candidates_per_pair: int = 6
    edge_cost_fn: Optional[Callable[[Any], float]] = None
    mode_semantics: Optional[ModeSemantics] = None
    semantic_model: Optional[PlanningSemanticModel] = None
    wrap_state_fn: Callable[[np.ndarray], np.ndarray] = identity_wrap
    state_distance_fn: Callable[[np.ndarray, np.ndarray], float] = euclidean_distance
    transition_penalty_fn: Optional[Callable[[np.ndarray], float]] = None
    transition_admissibility_fn: Optional[Callable[..., float]] = None
    transition_feasibility_fn: Optional[Callable[..., bool]] = None
    local_planner_name: str = "projection"
    local_planner_kwargs: dict = field(default_factory=dict)
    step_size: float = 0.08


@dataclass
class PlannerDiagnostics:
    num_graph_nodes: int = 0
    num_graph_edges: int = 0
    num_transition_pairs: int = 0
    num_transition_candidates: int = 0
    transition_cache_hits: int = 0
    transition_cache_misses: int = 0
    transition_attempt_stats: dict = field(default_factory=dict)

    start_key: tuple | None = None
    goal_key: tuple | None = None

    selected_route_string: str = ""
    selected_candidate_indices: list[int] = field(default_factory=list)
    selected_route_base_score: float = 0.0
    selected_route_admissibility_cost: float = 0.0
    selected_route_total_edge_cost: float = 0.0
    realized_candidate_indices: list[int] = field(default_factory=list)
    realized_transition_deviations: list[bool] = field(default_factory=list)

    realization_num_steps: int = 0
    realization_path_length: float = 0.0
    failure_stage: str = ""
    failure_reason: str = ""


@dataclass
class MultimodalComponentPlanResult:
    success: bool
    message: str

    graph: Any | None = None
    transition_manager: Any | None = None
    route_edges: list | None = None
    realization: Any | None = None

    route_found: bool = False
    realization_success: bool = False
    path_length: float = 0.0

    diagnostics: PlannerDiagnostics = field(default_factory=PlannerDiagnostics)


def _default_edge_cost(edge) -> float:
    return 1.0 + float(edge.score)


def _mode_semantic_edge_cost(mode_semantics: ModeSemantics, edge) -> float:
    return (
        1.0
        + float(edge.score)
        + float(
            mode_semantics.transition_cost(
                source_family_name=str(edge.src[0]),
                source_lam=float(edge.src[1]),
                target_family_name=str(edge.dst[0]),
                target_lam=float(edge.dst[1]),
            )
        )
    )


def _planning_semantic_edge_cost(semantic_model: PlanningSemanticModel, edge) -> float:
    return (
        1.0
        + float(edge.score)
        + float(
            semantic_model.transition_cost(
                PlanningSemanticContext(
                    source_family_name=str(edge.src[0]),
                    source_lam=float(edge.src[1]),
                    target_family_name=str(edge.dst[0]),
                    target_lam=float(edge.dst[1]),
                    point=np.asarray(edge.transition_point, dtype=float),
                    metadata=getattr(edge, "metadata", {}),
                )
            )
        )
    )


def _default_path_length(realization_steps) -> float:
    total = 0.0
    for step in realization_steps:
        path = np.asarray(step.path)
        if len(path) >= 2:
            total += float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    return total


def _summarize_route(route_edges, start_key) -> str:
    if route_edges is None:
        return ""
    seq = [f"{start_key[0]}[{start_key[1]}|{start_key[2]}]"]
    for e in route_edges:
        fam, lam, comp = e.dst
        seq.append(f"{fam}[{lam}|{comp}]")
    return " -> ".join(seq)


class _FamilyStub:
    def __init__(self, name: str):
        self.name = str(name)


def _family_stub(name: str) -> _FamilyStub:
    return _FamilyStub(name)


class MultimodalComponentPlanner:
    """
    Unified planner pipeline:

    - build component-aware graph
    - store exact transition candidates in a transition manager
    - run discrete route search
    - realize the chosen route with reachable/downstream-aware transition selection
    """

    def __init__(
        self,
        families,
        project_newton,
        seed_points_fn: Callable,
        component_ids_for_family_fn: Optional[Callable[[object, float], list[str]]] = None,
        compatible_components_fn: Optional[Callable[[object, float, np.ndarray], list[str]]] = None,
        component_model_registry: Optional[ComponentModelRegistry] = None,
        transition_generator: Optional[TransitionGenerator] = None,
        allowed_family_pair_fn: Optional[Callable[[str, str], bool]] = None,
        config: Optional[PlannerConfig] = None,
    ):
        self.families = families
        self.project_newton = project_newton
        self.seed_points_fn = seed_points_fn
        self.config = config if config is not None else PlannerConfig()
        self.semantic_model = self.config.semantic_model
        if self.semantic_model is None and self.config.mode_semantics is not None:
            if (
                self.config.transition_admissibility_fn is not None
                or self.config.transition_feasibility_fn is not None
            ):
                self.semantic_model = CompositePlanningSemanticModel(
                    mode_semantics=self.config.mode_semantics,
                    transition_feasibility_fn=lambda context: bool(
                        True
                        if self.config.transition_feasibility_fn is None
                        else self.config.transition_feasibility_fn(
                            _family_stub(context.source_family_name),
                            context.source_lam,
                            _family_stub(context.target_family_name),
                            context.target_lam,
                            None if context.point is None else np.asarray(context.point, dtype=float),
                            None if context.goal_point is None else np.asarray(context.goal_point, dtype=float),
                            context.metadata,
                        )
                    ),
                    transition_admissibility_fn=lambda context: float(
                        0.0
                        if self.config.transition_admissibility_fn is None
                        else self.config.transition_admissibility_fn(
                            _family_stub(context.source_family_name),
                            context.source_lam,
                            _family_stub(context.target_family_name),
                            context.target_lam,
                            None if context.point is None else np.asarray(context.point, dtype=float),
                            None if context.goal_point is None else np.asarray(context.goal_point, dtype=float),
                            context.metadata,
                        )
                    ),
                )
            else:
                self.semantic_model = ModeSemanticsAdapter(self.config.mode_semantics)
        self.component_model_registry = component_model_registry
        if component_model_registry is not None:
            self.component_ids_for_family_fn = component_model_registry.component_ids_for_family
            self.compatible_components_fn = component_model_registry.compatible_components_for_leaf
        else:
            self.component_ids_for_family_fn = component_ids_for_family_fn
            self.compatible_components_fn = compatible_components_fn
        if self.component_ids_for_family_fn is None or self.compatible_components_fn is None:
            raise ValueError(
                "Provide either component_model_registry or both component_ids_for_family_fn "
                "and compatible_components_fn."
            )
        if transition_generator is not None:
            self.transition_generator = transition_generator
        else:
            component_anchor_fn = None
            if self.component_model_registry is not None:
                component_anchor_fn = self.component_model_registry.transition_seed_anchors
            self.transition_generator = TransitionGenerator(
                seed_points_fn=self.seed_points_fn,
                project_newton=self.project_newton,
                seed_policy=AdaptiveTransitionSeedPolicy(
                    project_newton=self.project_newton,
                    base_seed_points_fn=self.seed_points_fn,
                    component_anchor_fn=component_anchor_fn,
                ),
                admissibility_cost_fn=(
                    None
                    if self.semantic_model is not None
                    else self.config.transition_admissibility_fn
                ),
                feasibility_fn=(
                    None
                    if self.semantic_model is not None
                    else self.config.transition_feasibility_fn
                ),
                semantic_model=self.semantic_model,
            )
        self.allowed_family_pair_fn = allowed_family_pair_fn
        self.allowed_leaf_pair_fn = None
        if self.semantic_model is not None:
            self.allowed_leaf_pair_fn = build_semantic_model_allowed_leaf_pair_fn(self.semantic_model)
            if self.allowed_family_pair_fn is None:
                self.allowed_family_pair_fn = (
                    lambda source_family_name, target_family_name: True
                )
            if self.config.edge_cost_fn is None:
                self.config.edge_cost_fn = (
                    lambda edge: _planning_semantic_edge_cost(self.semantic_model, edge)
                )
        elif self.config.mode_semantics is not None:
            self.allowed_leaf_pair_fn = build_allowed_leaf_pair_fn(self.config.mode_semantics)
            if self.allowed_family_pair_fn is None:
                self.allowed_family_pair_fn = (
                    lambda source_family_name, target_family_name: True
                )
            if self.config.edge_cost_fn is None:
                self.config.edge_cost_fn = (
                    lambda edge: _mode_semantic_edge_cost(self.config.mode_semantics, edge)
                )

    def build_graph(self, goal_point: np.ndarray):
        graph, transition_manager = build_component_graph_with_transition_manager(
            families=self.families,
            project_newton=self.project_newton,
            seed_points_fn=self.seed_points_fn,
            goal_point=np.asarray(goal_point, dtype=float).copy(),
            component_ids_for_family_fn=self.component_ids_for_family_fn,
            compatible_components_fn=self.compatible_components_fn,
            allowed_family_pair_fn=self.allowed_family_pair_fn,
            allowed_leaf_pair_fn=self.allowed_leaf_pair_fn,
            max_candidates_per_pair=self.config.max_candidates_per_pair,
            transition_generator=self.transition_generator,
        )
        return graph, transition_manager

    def search_route(
        self,
        graph,
        start_key,
        goal_key,
    ):
        edge_cost_fn = self.config.edge_cost_fn or _default_edge_cost
        return shortest_component_route(
            graph=graph,
            start=start_key,
            goal=goal_key,
            edge_cost_fn=edge_cost_fn,
        )

    def infer_component(self, family_name: str, lam: float, q: np.ndarray) -> str:
        if self.component_model_registry is None:
            raise ValueError(
                "Component inference requires component_model_registry to be provided."
            )
        return self.component_model_registry.infer_component(
            family_name=family_name,
            lam=float(lam),
            q=np.asarray(q, dtype=float),
        )

    def realize_route(
        self,
        transition_manager,
        start_state: LeafState,
        start_component: str,
        goal_q: np.ndarray,
        route_edges,
    ):
        return realize_component_route_with_manager(
            transition_manager=transition_manager,
            start_state=start_state,
            start_component=start_component,
            goal_q=np.asarray(goal_q, dtype=float).copy(),
            families=self.families,
            route_edges=route_edges,
            step_size=self.config.step_size,
            local_planner_name=self.config.local_planner_name,
            local_planner_kwargs=dict(self.config.local_planner_kwargs),
            wrap_state_fn=self.config.wrap_state_fn,
            state_distance_fn=self.config.state_distance_fn,
            transition_penalty_fn=self.config.transition_penalty_fn,
        )

    def _collect_graph_diagnostics(self, graph, transition_manager) -> PlannerDiagnostics:
        diag = PlannerDiagnostics()
        diag.num_graph_nodes = len(graph.adjacency)
        diag.num_graph_edges = sum(len(v) for v in graph.adjacency.values())

        pair_to_candidates = getattr(transition_manager, "_pair_to_candidates", {})
        diag.num_transition_pairs = len(pair_to_candidates)
        diag.num_transition_candidates = sum(len(v) for v in pair_to_candidates.values())
        diag.transition_cache_hits = int(getattr(self.transition_generator, "cache_hits", 0))
        diag.transition_cache_misses = int(getattr(self.transition_generator, "cache_misses", 0))
        attempt_stats = getattr(transition_manager, "_candidate_attempt_stats", {})
        diag.transition_attempt_stats = {
            f"{pair_key[0]}->{pair_key[1]}#cand{candidate_index}": {
                "successes": int(stats.successes),
                "failures": int(stats.failures),
            }
            for (pair_key, candidate_index), stats in attempt_stats.items()
        }
        return diag

    def plan(
        self,
        start_state: LeafState,
        start_component: str,
        goal_family_name: str,
        goal_lam: float,
        goal_component: str,
        goal_q: np.ndarray,
    ) -> MultimodalComponentPlanResult:
        graph, transition_manager = self.build_graph(goal_point=goal_q)

        start_key = (
            start_state.family_name,
            float(start_state.lam),
            str(start_component),
        )
        goal_key = (
            str(goal_family_name),
            float(goal_lam),
            str(goal_component),
        )

        diagnostics = self._collect_graph_diagnostics(graph, transition_manager)
        diagnostics.start_key = start_key
        diagnostics.goal_key = goal_key

        route_edges = self.search_route(
            graph=graph,
            start_key=start_key,
            goal_key=goal_key,
        )

        if route_edges is None:
            diagnostics.failure_stage = "route_search"
            diagnostics.failure_reason = f"No component-aware route found from {start_key} to {goal_key}."
            return MultimodalComponentPlanResult(
                success=False,
                message=diagnostics.failure_reason,
                graph=graph,
                transition_manager=transition_manager,
                route_edges=None,
                realization=None,
                route_found=False,
                realization_success=False,
                path_length=0.0,
                diagnostics=diagnostics,
            )

        diagnostics.selected_route_string = _summarize_route(route_edges, start_key)
        diagnostics.selected_candidate_indices = [int(e.candidate_index) for e in route_edges]
        diagnostics.selected_route_base_score = float(
            sum(float(e.metadata.get("base_score", e.score)) for e in route_edges)
        )
        diagnostics.selected_route_admissibility_cost = float(
            sum(float(e.metadata.get("admissibility_cost", 0.0)) for e in route_edges)
        )
        edge_cost_fn = self.config.edge_cost_fn or _default_edge_cost
        diagnostics.selected_route_total_edge_cost = float(
            sum(float(edge_cost_fn(e)) for e in route_edges)
        )

        realization = self.realize_route(
            transition_manager=transition_manager,
            start_state=start_state,
            start_component=start_component,
            goal_q=goal_q,
            route_edges=route_edges,
        )

        diagnostics.realization_num_steps = len(realization.steps)
        path_length = _default_path_length(realization.steps) if realization.success else 0.0
        diagnostics.realization_path_length = path_length
        diagnostics.realized_candidate_indices = [
            int(step.realized_candidate_index)
            for step in realization.steps
            if step.realized_candidate_index is not None
        ]
        diagnostics.realized_transition_deviations = [
            bool(step.nominal_candidate_index != step.realized_candidate_index)
            for step in realization.steps
            if step.realized_candidate_index is not None
        ]

        if not realization.success:
            diagnostics.failure_stage = "realization"
            diagnostics.failure_reason = realization.message

        return MultimodalComponentPlanResult(
            success=bool(realization.success),
            message=realization.message,
            graph=graph,
            transition_manager=transition_manager,
            route_edges=route_edges,
            realization=realization,
            route_found=True,
            realization_success=bool(realization.success),
            path_length=path_length,
            diagnostics=diagnostics,
        )

    def plan_with_inferred_components(
        self,
        start_state: LeafState,
        goal_family_name: str,
        goal_lam: float,
        goal_q: np.ndarray,
    ) -> MultimodalComponentPlanResult:
        start_component = self.infer_component(
            family_name=start_state.family_name,
            lam=float(start_state.lam),
            q=start_state.x,
        )
        goal_component = self.infer_component(
            family_name=goal_family_name,
            lam=float(goal_lam),
            q=goal_q,
        )
        return self.plan(
            start_state=start_state,
            start_component=start_component,
            goal_family_name=goal_family_name,
            goal_lam=float(goal_lam),
            goal_component=goal_component,
            goal_q=np.asarray(goal_q, dtype=float).copy(),
        )
