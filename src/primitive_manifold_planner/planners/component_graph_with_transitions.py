from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from primitive_manifold_planner.planners.component_leaf_graph import (
    ComponentEdge,
    ComponentGraph,
)
from primitive_manifold_planner.planners.transition_manager import (
    TransitionCandidate,
    TransitionGenerator,
    TransitionManager,
)


def build_component_graph_with_transition_manager(
    families,
    project_newton,
    seed_points_fn: Callable,
    goal_point: np.ndarray,
    component_ids_for_family_fn: Callable[[object, float], list[str]],
    compatible_components_fn: Callable[[object, float, np.ndarray], list[str]],
    allowed_family_pair_fn: Optional[Callable[[str, str], bool]] = None,
    allowed_leaf_pair_fn: Optional[Callable[[str, float, str, float], bool]] = None,
    max_candidates_per_pair: int = 4,
    transition_generator: Optional[TransitionGenerator] = None,
):
    graph = ComponentGraph()
    manager = TransitionManager()
    generator = transition_generator or TransitionGenerator(
        seed_points_fn=seed_points_fn,
        project_newton=project_newton,
    )

    family_nodes = []
    for fam in families:
        for lam in fam.sample_lambdas():
            family_nodes.append((fam, float(lam)))
            for comp in component_ids_for_family_fn(fam, float(lam)):
                graph.add_node((fam.name, float(lam), str(comp)))

    for i, (fam_a, lam_a) in enumerate(family_nodes):
        for j in range(i + 1, len(family_nodes)):
            fam_b, lam_b = family_nodes[j]

            if allowed_leaf_pair_fn is not None and not allowed_leaf_pair_fn(fam_a.name, lam_a, fam_b.name, lam_b):
                continue
            if allowed_family_pair_fn is not None and not allowed_family_pair_fn(fam_a.name, fam_b.name):
                continue

            result = generator.generate_transitions(
                source_family=fam_a,
                source_lam=lam_a,
                target_family=fam_b,
                target_lam=lam_b,
                goal_point=goal_point,
                max_candidates=max_candidates_per_pair,
            )
            if not result.success:
                continue

            for k, cand in enumerate(result.candidates):
                q = np.asarray(cand.transition_point, dtype=float).copy()

                src_components = compatible_components_fn(fam_a, lam_a, q)
                dst_components = compatible_components_fn(fam_b, lam_b, q)

                for src_comp in src_components:
                    for dst_comp in dst_components:
                        src = (fam_a.name, float(lam_a), str(src_comp))
                        dst = (fam_b.name, float(lam_b), str(dst_comp))

                        edge_ab = ComponentEdge(
                            src=src,
                            dst=dst,
                            transition_point=q.copy(),
                            score=float(cand.score),
                            candidate_index=k,
                            metadata={
                                **dict(cand.metadata),
                                "source_component": str(src_comp),
                                "target_component": str(dst_comp),
                            },
                        )
                        edge_ba = ComponentEdge(
                            src=dst,
                            dst=src,
                            transition_point=q.copy(),
                            score=float(cand.score),
                            candidate_index=k,
                            metadata={
                                **dict(cand.metadata),
                                "source_component": str(dst_comp),
                                "target_component": str(src_comp),
                            },
                        )

                        graph.add_edge(edge_ab)
                        graph.add_edge(edge_ba)

                        manager.add_candidate(
                            TransitionCandidate(
                                src=src,
                                dst=dst,
                                transition_point=q.copy(),
                                score=float(cand.score),
                                candidate_index=k,
                                metadata={
                                    **dict(cand.metadata),
                                    "source_component": str(src_comp),
                                    "target_component": str(dst_comp),
                                },
                            )
                        )
                        manager.add_candidate(
                            TransitionCandidate(
                                src=dst,
                                dst=src,
                                transition_point=q.copy(),
                                score=float(cand.score),
                                candidate_index=k,
                                metadata={
                                    **dict(cand.metadata),
                                    "source_component": str(dst_comp),
                                    "target_component": str(src_comp),
                                },
                            )
                        )

    return graph, manager
