from __future__ import annotations

"""Joint-space variant of Example 66's parallel evidence planner.

This module keeps the original left -> plane -> right multimodal evidence
architecture, but stores robot joint configurations in the stage graphs.
Each stage manifold is an implicit constraint on the robot's end-effector
through forward kinematics. Route extraction still happens late from the
evidence graph, and the displayed path remains the end-effector trajectory in
world coordinates.
"""

import numpy as np

import example_66_multimodal_graph_search as ex66
from collision_utilities import configuration_in_collision, default_example_66_obstacles
from jointspace_planner_utils import (
    detect_transitions_jointspace,
    end_effector_point,
    explore_joint_manifold,
    generate_joint_proposals,
    inverse_kinematics_start,
    joint_path_to_task_path,
)
from robot_constraint_manifolds import RobotPlaneManifold, RobotSphereManifold


def _task_point(robot, q: np.ndarray) -> np.ndarray:
    return end_effector_point(robot, np.asarray(q, dtype=float))


def task_space_distance(
    store: ex66.StageEvidenceStore,
    node_id1: int,
    node_id2: int,
    robot,
) -> float:
    p1 = _task_point(robot, store.graph.nodes[int(node_id1)].q)
    p2 = _task_point(robot, store.graph.nodes[int(node_id2)].q)
    return float(np.linalg.norm(p1 - p2))


def _task_points_for_store(store: ex66.StageEvidenceStore, robot) -> np.ndarray:
    points = [_task_point(robot, node.q) for node in store.graph.nodes.values()]
    if len(points) == 0:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(points, dtype=float)


def _task_frontier_points(store: ex66.StageEvidenceStore, robot) -> np.ndarray:
    points = [
        _task_point(robot, store.graph.nodes[node_id].q)
        for node_id in store.frontier_ids
        if node_id in store.graph.nodes
    ]
    if len(points) == 0:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(points, dtype=float)


def _task_edge_segments(store: ex66.StageEvidenceStore, robot) -> list[tuple[np.ndarray, np.ndarray]]:
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for edge_start, edge_end in store.explored_edges:
        segments.append((_task_point(robot, edge_start), _task_point(robot, edge_end)))
    return segments


def _stage_evidence_points_joint(store: ex66.StageEvidenceStore, robot) -> np.ndarray:
    groups: list[np.ndarray] = []
    explored = ex66.explored_points_from_edges(_task_edge_segments(store, robot))
    if len(explored) > 0:
        groups.append(explored)
    if len(store.chart_centers) > 0:
        groups.append(np.asarray(store.chart_centers, dtype=float))
    frontier = _task_frontier_points(store, robot)
    if len(frontier) > 0:
        groups.append(frontier)
    nodes = _task_points_for_store(store, robot)
    if len(nodes) > 0:
        groups.append(nodes)
    if len(groups) == 0:
        return np.zeros((0, 3), dtype=float)
    return ex66.deduplicate_points([point for group in groups for point in np.asarray(group, dtype=float)], tol=1e-4)


def _merge_chart_centers_joint(existing: np.ndarray, robot, joint_path: np.ndarray) -> np.ndarray:
    task_path = joint_path_to_task_path(robot, joint_path)
    chart_count = max(0, min(8, len(task_path) // 6))
    centers = ex66.sample_chart_centers(task_path, chart_count)
    if len(existing) == 0:
        return np.asarray(centers, dtype=float)
    if len(centers) == 0:
        return np.asarray(existing, dtype=float)
    return ex66.deduplicate_points(list(existing) + list(centers), tol=1e-4)


def _proposal_stage_utility_joint(
    stage: str,
    projected_q: np.ndarray,
    store: ex66.StageEvidenceStore,
    guide_point: np.ndarray,
    stores: dict[str, ex66.StageEvidenceStore],
    robot,
) -> float:
    q = np.asarray(projected_q, dtype=float)
    task_q = _task_point(robot, q)
    known_points = _stage_evidence_points_joint(store, robot)
    if len(known_points) > 0:
        novelty = min(float(np.linalg.norm(task_q - point)) for point in known_points)
    else:
        novelty = ex66.TARGET_NOVELTY_RADIUS
    novelty_bonus = min(novelty, ex66.TARGET_NOVELTY_RADIUS)
    guide_bonus = 1.0 / (1.0 + float(np.linalg.norm(task_q - np.asarray(guide_point, dtype=float))))
    underexplored_bonus = ex66.stage_underexploration_factor(stage, stores)
    stage_bias = 0.08 if stage == ex66.RIGHT_STAGE else (0.16 if stage == ex66.PLANE_STAGE else 0.0)
    return 0.50 * novelty_bonus + 0.26 * guide_bonus + 0.18 * underexplored_bonus + stage_bias


def _update_stage_frontier_joint(
    store: ex66.StageEvidenceStore,
    new_ids: list[int],
    guide_point: np.ndarray,
    robot,
    max_points: int = ex66.FRONTIER_LIMIT,
) -> None:
    merged: list[int] = []
    seen: set[int] = set()
    for node_id in list(store.frontier_ids) + list(new_ids):
        if node_id in seen:
            continue
        seen.add(node_id)
        merged.append(node_id)

    if len(merged) <= max_points:
        store.frontier_ids = merged
        return

    guide = np.asarray(guide_point, dtype=float)
    remaining = list(merged)
    selected: list[int] = []
    while len(remaining) > 0 and len(selected) < max_points:
        best_id = None
        best_score = -float("inf")
        for node_id in remaining:
            node = store.graph.nodes[node_id]
            node_task = _task_point(robot, node.q)
            guide_bonus = 1.0 / (1.0 + float(np.linalg.norm(node_task - guide)))
            underexplored_bonus = 1.0 / (1.0 + float(node.expansion_count))
            seeded_bonus = 0.18 if node.seeded_from_proposal else 0.0
            if len(selected) == 0:
                novelty_bonus = guide_bonus
            else:
                novelty_bonus = min(
                    task_space_distance(store, node_id, selected_id, robot)
                    for selected_id in selected
                )
            score = 0.42 * novelty_bonus + 0.24 * underexplored_bonus + 0.14 * guide_bonus + seeded_bonus
            if score > best_score:
                best_score = score
                best_id = node_id
        if best_id is None:
            break
        selected.append(best_id)
        remaining.remove(best_id)
    store.frontier_ids = selected


def _choose_stage_source_joint(store: ex66.StageEvidenceStore, guide_point: np.ndarray, robot) -> int:
    guide = np.asarray(guide_point, dtype=float)
    candidate_ids = list(store.frontier_ids) if len(store.frontier_ids) > 0 else list(store.graph.nodes.keys())
    if len(candidate_ids) == 0:
        raise ValueError("Cannot choose a stage source from an empty evidence store.")
    scored = []
    for node_id in candidate_ids:
        node = store.graph.nodes[node_id]
        node_task = _task_point(robot, node.q)
        guide_bonus = 1.0 / (1.0 + float(np.linalg.norm(node_task - guide)))
        underexplored_bonus = 1.0 / (1.0 + float(node.expansion_count))
        if len(candidate_ids) > 1:
            diversity_bonus = min(
                task_space_distance(store, node_id, other_id, robot)
                for other_id in candidate_ids
                if other_id != node_id
            )
        else:
            diversity_bonus = 0.0
        seeded_bonus = 0.12 if node.seeded_from_proposal else 0.0
        score = -(0.46 * underexplored_bonus + 0.26 * guide_bonus + 0.18 * min(diversity_bonus, 1.0) + seeded_bonus)
        scored.append((score, int(node_id)))
    scored.sort(key=lambda item: item[0])
    top_ids = [node_id for _, node_id in scored[: min(ex66.FRONTIER_SELECTION_LIMIT, len(scored))]]
    return int(np.random.choice(top_ids))


def _ranked_stage_sources_joint(
    store: ex66.StageEvidenceStore,
    guide_point: np.ndarray,
    limit: int,
    robot,
) -> list[int]:
    guide = np.asarray(guide_point, dtype=float)
    candidate_ids = list(store.frontier_ids) if len(store.frontier_ids) > 0 else list(store.graph.nodes.keys())
    if len(candidate_ids) == 0:
        return []
    scored = []
    for node_id in candidate_ids:
        node = store.graph.nodes[node_id]
        node_task = _task_point(robot, node.q)
        guide_bonus = 1.0 / (1.0 + float(np.linalg.norm(node_task - guide)))
        underexplored_bonus = 1.0 / (1.0 + float(node.expansion_count))
        if len(candidate_ids) > 1:
            diversity_bonus = min(
                task_space_distance(store, node_id, other_id, robot)
                for other_id in candidate_ids
                if other_id != node_id
            )
        else:
            diversity_bonus = 0.0
        seeded_bonus = 0.12 if node.seeded_from_proposal else 0.0
        score = -(0.42 * guide_bonus + 0.28 * underexplored_bonus + 0.18 * min(diversity_bonus, 1.0) + seeded_bonus)
        scored.append((score, int(node_id)))
    scored.sort(key=lambda item: item[0])
    return [node_id for _, node_id in scored[: min(limit, len(scored))]]


def _maybe_seed_stage_component_joint(
    store: ex66.StageEvidenceStore,
    target_q: np.ndarray,
    guide_point: np.ndarray,
    robot,
    collision_fn=None,
) -> tuple[int, bool]:
    if collision_fn is not None and bool(collision_fn(target_q)):
        return -1, False
    known = _stage_evidence_points_joint(store, robot)
    if len(store.graph.nodes) == 0:
        node_id = ex66.add_stage_node(store, target_q, seeded_from_proposal=True)
        _update_stage_frontier_joint(store, [node_id], guide_point, robot)
        return node_id, True
    if len(known) == 0:
        node_id = ex66.add_stage_node(store, target_q, seeded_from_proposal=True)
        _update_stage_frontier_joint(store, [node_id], guide_point, robot)
        return node_id, True
    target_task = _task_point(robot, target_q)
    nearest = min(float(np.linalg.norm(target_task - point)) for point in known)
    if nearest >= ex66.EVIDENCE_SEED_RADIUS:
        node_id = ex66.add_stage_node(store, target_q, seeded_from_proposal=True)
        _update_stage_frontier_joint(store, [node_id], guide_point, robot)
        return node_id, True
    return -1, False


def _update_stage_evidence_from_proposal_joint(
    store: ex66.StageEvidenceStore,
    target_q: np.ndarray,
    guide_point: np.ndarray,
    robot,
    collision_fn=None,
) -> tuple[int, int, object | None, int | None, list[int]]:
    seeded_node_id, seeded = _maybe_seed_stage_component_joint(store, target_q, guide_point, robot, collision_fn=collision_fn)
    if seeded:
        store.update_count += 1
        return 1, 0, None, seeded_node_id, [seeded_node_id]

    if len(store.graph.nodes) == 0:
        return 0, 0, None, None, []

    source_node_id = _choose_stage_source_joint(store, guide_point, robot)
    ex66.increment_stage_node_expansion(store, source_node_id)
    source_q = store.graph.nodes[source_node_id].q
    result = explore_joint_manifold(
        manifold=store.manifold,
        start=source_q,
        goal=np.asarray(target_q, dtype=float),
        collision_fn=collision_fn,
    )
    store.explored_edges = ex66.merge_edges(store.explored_edges, list(getattr(result, "explored_edges", [])))
    store.chart_centers = _merge_chart_centers_joint(store.chart_centers, robot, np.asarray(result.path, dtype=float))
    if not result.success:
        return 0, 1, result, source_node_id, [source_node_id]
    end_node_id, new_path_nodes, _edge_ids = ex66.connect_path_to_stage_graph(
        store=store,
        source_node_id=source_node_id,
        path=np.asarray(result.path, dtype=float),
        kind=f"{store.stage}_evidence_motion",
    )
    _update_stage_frontier_joint(store, new_path_nodes + [end_node_id], guide_point, robot)
    store.update_count += 1
    return max(0, len(new_path_nodes) - 1), 1, result, source_node_id, new_path_nodes + [end_node_id]


def _nearest_node_id(store: ex66.StageEvidenceStore, candidate_ids: list[int], target_q: np.ndarray) -> int | None:
    if len(candidate_ids) == 0:
        return None
    target = np.asarray(target_q, dtype=float)
    return min(candidate_ids, key=lambda node_id: float(np.linalg.norm(store.graph.nodes[node_id].q - target)))


def _shared_transition_configuration(
    *,
    source_manifold,
    target_manifold,
    transition_task: np.ndarray,
    source_q_hint: np.ndarray,
    target_q_hint: np.ndarray,
    robot,
    collision_fn=None,
    task_tol: float = 7.5e-2,
) -> np.ndarray | None:
    transition_task = np.asarray(transition_task, dtype=float).reshape(3)
    lower = np.asarray(getattr(source_manifold, "joint_lower", -np.pi * np.ones(3, dtype=float)), dtype=float)
    upper = np.asarray(getattr(source_manifold, "joint_upper", np.pi * np.ones(3, dtype=float)), dtype=float)
    hints = [
        np.asarray(source_q_hint, dtype=float).reshape(3),
        np.asarray(target_q_hint, dtype=float).reshape(3),
        0.5 * (
            np.asarray(source_q_hint, dtype=float).reshape(3)
            + np.asarray(target_q_hint, dtype=float).reshape(3)
        ),
    ]
    best_q: np.ndarray | None = None
    best_task_error = float("inf")
    for hint in hints:
        q_shared = inverse_kinematics_start(
            robot,
            transition_task,
            warm_start=hint,
            joint_lower=lower,
            joint_upper=upper,
            tol=task_tol,
        )
        if q_shared is None:
            continue
        candidates = [np.asarray(q_shared, dtype=float)]
        source_projection = source_manifold.project(np.asarray(q_shared, dtype=float), tol=1e-7, max_iters=80)
        if source_projection.success:
            candidates.append(np.asarray(source_projection.x_projected, dtype=float))
        target_projection = target_manifold.project(np.asarray(q_shared, dtype=float), tol=1e-7, max_iters=80)
        if target_projection.success:
            candidates.append(np.asarray(target_projection.x_projected, dtype=float))
        for candidate in candidates:
            task_error = float(np.linalg.norm(_task_point(robot, candidate) - transition_task))
            if task_error > task_tol:
                continue
            if not bool(source_manifold.within_bounds(candidate)) or not bool(target_manifold.within_bounds(candidate)):
                continue
            if not bool(source_manifold.is_valid(candidate, tol=2e-3)) or not bool(target_manifold.is_valid(candidate, tol=2e-3)):
                continue
            if collision_fn is not None and bool(collision_fn(candidate)):
                continue
            if task_error + 1e-9 < best_task_error:
                best_task_error = task_error
                best_q = np.asarray(candidate, dtype=float)
    return best_q


def _connect_source_to_transition_joint(
    *,
    store: ex66.StageEvidenceStore,
    source_node_id: int,
    path_node_ids: list[int],
    target_q: np.ndarray,
    guide_point: np.ndarray,
    robot,
    edge_kind: str,
    collision_fn=None,
) -> tuple[int | None, list[int]]:
    candidate_source_ids = [int(source_node_id)] + [int(node_id) for node_id in path_node_ids]
    graph_source_id = _nearest_node_id(store, candidate_source_ids, target_q)
    if graph_source_id is None:
        return None, []
    source_q = np.asarray(store.graph.nodes[int(graph_source_id)].q, dtype=float)
    target_q_arr = np.asarray(target_q, dtype=float)
    if float(np.linalg.norm(source_q - target_q_arr)) <= ex66.GRAPH_NODE_TOL:
        _update_stage_frontier_joint(store, [int(graph_source_id)], guide_point, robot)
        return int(graph_source_id), []

    exact_result = explore_joint_manifold(
        manifold=store.manifold,
        start=source_q,
        goal=target_q_arr,
        collision_fn=collision_fn,
    )
    if not exact_result.success:
        return None, []
    store.explored_edges = ex66.merge_edges(store.explored_edges, list(getattr(exact_result, "explored_edges", [])))
    store.chart_centers = _merge_chart_centers_joint(
        store.chart_centers,
        robot,
        np.asarray(exact_result.path, dtype=float),
    )
    target_node_id, path_nodes, _ = ex66.connect_path_to_stage_graph(
        store=store,
        source_node_id=int(graph_source_id),
        path=np.asarray(exact_result.path, dtype=float),
        kind=edge_kind,
    )
    _update_stage_frontier_joint(store, list(path_nodes) + [int(target_node_id)], guide_point, robot)
    return int(target_node_id), list(path_nodes)


def _add_left_plane_hypotheses_joint(
    source_stage: str,
    source_store: ex66.StageEvidenceStore,
    plane_store: ex66.StageEvidenceStore,
    left_store: ex66.StageEvidenceStore,
    left_geom,
    plane_geom,
    result,
    source_node_id: int | None,
    path_node_ids: list[int],
    guide_point: np.ndarray,
    hypotheses: list[ex66.TransitionHypothesis],
    robot,
    collision_fn=None,
) -> tuple[int, int]:
    if result is None or source_node_id is None:
        return 0, 0

    hits = detect_transitions_jointspace(
        robot=robot,
        current_manifold=source_store.manifold,
        target_manifold=plane_store.manifold if source_stage == ex66.LEFT_STAGE else left_store.manifold,
        path_configs=np.asarray(result.path, dtype=float),
        collision_fn=collision_fn,
    )
    if source_stage == ex66.PLANE_STAGE:
        hits = detect_transitions_jointspace(
            robot=robot,
            current_manifold=source_store.manifold,
            target_manifold=left_store.manifold,
            path_configs=np.asarray(result.path, dtype=float),
            collision_fn=collision_fn,
        )
    if len(hits) == 0:
        return 0, 0

    added = 0
    eval_count = 0
    known_points = [hyp.q for hyp in hypotheses]
    ranked_hits = ex66.rank_transition_hits(
        np.asarray([hit[2] for hit in hits], dtype=float),
        guide_point,
        known_points,
    )
    if len(ranked_hits) == 0:
        return 0, 0

    for hit_task in ranked_hits[: min(12, len(ranked_hits))]:
        matching = None
        for source_q_hit, target_q_hit, ee_hit in hits:
            if float(np.linalg.norm(np.asarray(ee_hit, dtype=float) - np.asarray(hit_task, dtype=float))) <= ex66.TRANSITION_DEDUP_TOL:
                matching = (source_q_hit, target_q_hit, ee_hit)
                break
        if matching is None:
            continue
        source_q_hit, target_q_hit, ee_hit = matching
        if any(float(np.linalg.norm(np.asarray(ee_hit, dtype=float) - hyp.q)) <= ex66.TRANSITION_DEDUP_TOL for hyp in hypotheses):
            continue
        refined_task, refined_ok = ex66.refine_intersection_on_both_manifolds(
            left_geom,
            plane_geom,
            0.5 * (np.asarray(ee_hit, dtype=float) + _task_point(robot, target_q_hit)),
            tol=1e-8,
            max_iters=25,
        )
        transition_task = np.asarray(refined_task if refined_ok else ee_hit, dtype=float)
        if not bool(plane_geom.within_bounds(transition_task)):
            continue
        use_shared = False
        if source_stage == ex66.LEFT_STAGE:
            if use_shared:
                left_node_id, _left_path_nodes = _connect_source_to_transition_joint(
                    store=left_store,
                    source_node_id=int(source_node_id),
                    path_node_ids=path_node_ids,
                    target_q=q_shared,
                    guide_point=guide_point,
                    robot=robot,
                    edge_kind=ex66.LEFT_MOTION,
                    collision_fn=collision_fn,
                )
                if left_node_id is not None:
                    plane_node_id = ex66.add_stage_node(plane_store, q_shared, seeded_from_proposal=True)
                    _update_stage_frontier_joint(plane_store, [plane_node_id], guide_point, robot)
                else:
                    use_shared = False
            if not use_shared:
                if collision_fn is not None and (bool(collision_fn(source_q_hit)) or bool(collision_fn(target_q_hit))):
                    continue
                candidate_source_ids = [int(source_node_id)] + [int(node_id) for node_id in path_node_ids]
                graph_source_id = _nearest_node_id(source_store, candidate_source_ids, source_q_hit)
                if graph_source_id is None:
                    continue
                left_node_id = int(graph_source_id)
                plane_node_id = ex66.add_stage_node(plane_store, target_q_hit, seeded_from_proposal=True)
                _update_stage_frontier_joint(plane_store, [plane_node_id], guide_point, robot)
        else:
            if use_shared:
                plane_node_id, _plane_path_nodes = _connect_source_to_transition_joint(
                    store=plane_store,
                    source_node_id=int(source_node_id),
                    path_node_ids=path_node_ids,
                    target_q=q_shared,
                    guide_point=guide_point,
                    robot=robot,
                    edge_kind=ex66.PLANE_MOTION,
                    collision_fn=collision_fn,
                )
                if plane_node_id is not None:
                    left_node_id = ex66.add_stage_node(left_store, q_shared, seeded_from_proposal=True)
                    _update_stage_frontier_joint(left_store, [left_node_id], guide_point, robot)
                else:
                    use_shared = False
            if not use_shared:
                if collision_fn is not None and (bool(collision_fn(source_q_hit)) or bool(collision_fn(target_q_hit))):
                    continue
                candidate_source_ids = [int(source_node_id)] + [int(node_id) for node_id in path_node_ids]
                graph_source_id = _nearest_node_id(source_store, candidate_source_ids, source_q_hit)
                if graph_source_id is None:
                    continue
                plane_node_id = int(graph_source_id)
                left_node_id = ex66.add_stage_node(left_store, target_q_hit, seeded_from_proposal=True)
                _update_stage_frontier_joint(left_store, [left_node_id], guide_point, robot)
        hypotheses.append(
            ex66.TransitionHypothesis(
                left_node_id=int(left_node_id),
                plane_node_id=int(plane_node_id),
                q=np.asarray(transition_task, dtype=float),
                provenance=f"{source_stage}_jointspace",
                score=float(np.linalg.norm(np.asarray(transition_task, dtype=float) - np.asarray(guide_point, dtype=float))),
            )
        )
        added += 1
        ex66.prune_transition_hypotheses(hypotheses)
    return added, eval_count


def _add_plane_right_hypotheses_joint(
    source_stage: str,
    source_store: ex66.StageEvidenceStore,
    plane_store: ex66.StageEvidenceStore,
    right_store: ex66.StageEvidenceStore,
    plane_geom,
    right_geom,
    result,
    source_node_id: int | None,
    path_node_ids: list[int],
    guide_point: np.ndarray,
    hypotheses: list[ex66.TransitionHypothesis],
    robot,
    collision_fn=None,
) -> tuple[int, int]:
    if result is None or source_node_id is None:
        return 0, 0

    hits = detect_transitions_jointspace(
        robot=robot,
        current_manifold=source_store.manifold,
        target_manifold=right_store.manifold if source_stage == ex66.PLANE_STAGE else plane_store.manifold,
        path_configs=np.asarray(result.path, dtype=float),
        collision_fn=collision_fn,
    )
    if source_stage == ex66.RIGHT_STAGE:
        hits = detect_transitions_jointspace(
            robot=robot,
            current_manifold=source_store.manifold,
            target_manifold=plane_store.manifold,
            path_configs=np.asarray(result.path, dtype=float),
            collision_fn=collision_fn,
        )
    if len(hits) == 0:
        return 0, 0

    added = 0
    eval_count = 0
    known_points = [hyp.q for hyp in hypotheses]
    ranked_hits = ex66.rank_transition_hits(
        np.asarray([hit[2] for hit in hits], dtype=float),
        guide_point,
        known_points,
    )
    if len(ranked_hits) == 0:
        return 0, 0

    for hit_task in ranked_hits[: min(12, len(ranked_hits))]:
        matching = None
        for source_q_hit, target_q_hit, ee_hit in hits:
            if float(np.linalg.norm(np.asarray(ee_hit, dtype=float) - np.asarray(hit_task, dtype=float))) <= ex66.TRANSITION_DEDUP_TOL:
                matching = (source_q_hit, target_q_hit, ee_hit)
                break
        if matching is None:
            continue
        source_q_hit, target_q_hit, ee_hit = matching
        if any(float(np.linalg.norm(np.asarray(ee_hit, dtype=float) - hyp.q)) <= ex66.TRANSITION_DEDUP_TOL for hyp in hypotheses):
            continue
        refined_task, refined_ok = ex66.refine_intersection_on_both_manifolds(
            plane_geom,
            right_geom,
            0.5 * (np.asarray(ee_hit, dtype=float) + _task_point(robot, target_q_hit)),
            tol=1e-8,
            max_iters=25,
        )
        transition_task = np.asarray(refined_task if refined_ok else ee_hit, dtype=float)
        if source_stage == ex66.PLANE_STAGE and not bool(plane_geom.within_bounds(transition_task)):
            continue
        use_shared = False
        if source_stage == ex66.PLANE_STAGE:
            if use_shared:
                plane_node_id, _plane_path_nodes = _connect_source_to_transition_joint(
                    store=plane_store,
                    source_node_id=int(source_node_id),
                    path_node_ids=path_node_ids,
                    target_q=q_shared,
                    guide_point=guide_point,
                    robot=robot,
                    edge_kind=ex66.PLANE_MOTION,
                    collision_fn=collision_fn,
                )
                if plane_node_id is not None:
                    right_node_id = ex66.add_stage_node(right_store, q_shared, seeded_from_proposal=True)
                    _update_stage_frontier_joint(right_store, [right_node_id], guide_point, robot)
                else:
                    use_shared = False
            if not use_shared:
                if collision_fn is not None and (bool(collision_fn(source_q_hit)) or bool(collision_fn(target_q_hit))):
                    continue
                candidate_source_ids = [int(source_node_id)] + [int(node_id) for node_id in path_node_ids]
                graph_source_id = _nearest_node_id(source_store, candidate_source_ids, source_q_hit)
                if graph_source_id is None:
                    continue
                plane_node_id = int(graph_source_id)
                right_node_id = ex66.add_stage_node(right_store, target_q_hit, seeded_from_proposal=True)
                _update_stage_frontier_joint(right_store, [right_node_id], guide_point, robot)
        else:
            if use_shared:
                right_node_id, _right_path_nodes = _connect_source_to_transition_joint(
                    store=right_store,
                    source_node_id=int(source_node_id),
                    path_node_ids=path_node_ids,
                    target_q=q_shared,
                    guide_point=guide_point,
                    robot=robot,
                    edge_kind=ex66.RIGHT_MOTION,
                    collision_fn=collision_fn,
                )
                if right_node_id is not None:
                    plane_node_id = ex66.add_stage_node(plane_store, q_shared, seeded_from_proposal=True)
                    _update_stage_frontier_joint(plane_store, [plane_node_id], guide_point, robot)
                else:
                    use_shared = False
            if not use_shared:
                if collision_fn is not None and (bool(collision_fn(source_q_hit)) or bool(collision_fn(target_q_hit))):
                    continue
                candidate_source_ids = [int(source_node_id)] + [int(node_id) for node_id in path_node_ids]
                graph_source_id = _nearest_node_id(source_store, candidate_source_ids, source_q_hit)
                if graph_source_id is None:
                    continue
                right_node_id = int(graph_source_id)
                plane_node_id = ex66.add_stage_node(plane_store, target_q_hit, seeded_from_proposal=True)
                _update_stage_frontier_joint(plane_store, [plane_node_id], guide_point, robot)
        hypotheses.append(
            ex66.TransitionHypothesis(
                plane_node_id=int(plane_node_id),
                right_node_id=int(right_node_id),
                q=np.asarray(transition_task, dtype=float),
                provenance=f"{source_stage}_jointspace",
                score=float(np.linalg.norm(np.asarray(transition_task, dtype=float) - np.asarray(guide_point, dtype=float))),
            )
        )
        added += 1
        ex66.prune_transition_hypotheses(hypotheses)
    return added, eval_count


def _bridge_left_hypotheses_to_start_joint(
    left_store: ex66.StageEvidenceStore,
    start_node_id: int,
    left_plane_hypotheses: list[ex66.TransitionHypothesis],
    guide_point: np.ndarray,
    robot,
    collision_fn=None,
) -> tuple[int, int]:
    left_dist, _prev_node, _prev_edge = ex66.shortest_paths_in_stage(left_store, int(start_node_id))
    reachable_ids = sorted(int(node_id) for node_id in left_dist)
    target_ids = sorted(
        {
            int(hyp.left_node_id)
            for hyp in left_plane_hypotheses
            if hyp.left_node_id is not None and int(hyp.left_node_id) not in left_dist
        }
    )
    if len(reachable_ids) == 0 or len(target_ids) == 0:
        return 0, 0

    candidate_pairs: list[tuple[float, int, int]] = []
    for target_id in target_ids:
        target_task = _task_point(robot, left_store.graph.nodes[target_id].q)
        nearest_sources = sorted(
            reachable_ids,
            key=lambda source_id: float(np.linalg.norm(_task_point(robot, left_store.graph.nodes[source_id].q) - target_task)),
        )
        for source_id in nearest_sources[: min(6, len(nearest_sources))]:
            candidate_pairs.append(
                (
                    float(
                        np.linalg.norm(
                            _task_point(robot, left_store.graph.nodes[source_id].q) - target_task
                        )
                    ),
                    source_id,
                    target_id,
                )
            )
    candidate_pairs.sort(key=lambda item: item[0])

    eval_count = 0
    node_gain = 0
    tried_targets: set[int] = set()
    bridge_try_limit = max(ex66.LEFT_BRIDGE_TRY_LIMIT, 12)
    for _score, source_id, target_id in candidate_pairs:
        if len(tried_targets) >= bridge_try_limit:
            break
        if int(target_id) in tried_targets:
            continue
        tried_targets.add(int(target_id))
        exact_result = explore_joint_manifold(
            manifold=left_store.manifold,
            start=left_store.graph.nodes[source_id].q,
            goal=left_store.graph.nodes[target_id].q,
            collision_fn=collision_fn,
        )
        eval_count += 1
        if not exact_result.success:
            continue
        left_store.explored_edges = ex66.merge_edges(left_store.explored_edges, list(getattr(exact_result, "explored_edges", [])))
        left_store.chart_centers = _merge_chart_centers_joint(left_store.chart_centers, robot, np.asarray(exact_result.path, dtype=float))
        _, node_ids, _ = ex66.connect_path_to_stage_graph(
            store=left_store,
            source_node_id=int(source_id),
            path=np.asarray(exact_result.path, dtype=float),
            kind=ex66.LEFT_MOTION,
            terminal_node_id=int(target_id),
        )
        _update_stage_frontier_joint(left_store, node_ids + [int(target_id)], guide_point, robot)
        node_gain += max(0, len(node_ids) - 2)
        break
    return node_gain, eval_count


def _bridge_plane_hypothesis_components_joint(
    plane_store: ex66.StageEvidenceStore,
    left_plane_hypotheses: list[ex66.TransitionHypothesis],
    plane_right_hypotheses: list[ex66.TransitionHypothesis],
    robot,
    collision_fn=None,
) -> tuple[int, int]:
    left_ids = sorted({int(hyp.plane_node_id) for hyp in left_plane_hypotheses if hyp.plane_node_id is not None})
    right_ids = sorted({int(hyp.plane_node_id) for hyp in plane_right_hypotheses if hyp.plane_node_id is not None})
    if len(left_ids) == 0 or len(right_ids) == 0:
        return 0, 0

    candidate_pairs: list[tuple[float, int, int]] = []
    for left_id in left_ids:
        q_left = plane_store.graph.nodes[left_id].q
        ee_left = _task_point(robot, q_left)
        for right_id in right_ids:
            if left_id == right_id:
                continue
            ee_right = _task_point(robot, plane_store.graph.nodes[right_id].q)
            candidate_pairs.append((float(np.linalg.norm(ee_left - ee_right)), left_id, right_id))
    candidate_pairs.sort(key=lambda item: item[0])

    def guided_plane_path(q_start: np.ndarray, q_goal: np.ndarray) -> np.ndarray | None:
        p_start = _task_point(robot, q_start)
        p_goal = _task_point(robot, q_goal)
        q_path = [np.asarray(q_start, dtype=float)]
        current = np.asarray(q_start, dtype=float)
        lower = np.asarray(plane_store.manifold.joint_lower, dtype=float)
        upper = np.asarray(plane_store.manifold.joint_upper, dtype=float)
        for alpha in np.linspace(0.125, 1.0, 8):
            target = (1.0 - float(alpha)) * p_start + float(alpha) * p_goal
            q_guess = inverse_kinematics_start(
                robot,
                target,
                warm_start=current,
                joint_lower=lower,
                joint_upper=upper,
                tol=8e-2,
            )
            if q_guess is None:
                return None
            projection = plane_store.manifold.project(np.asarray(q_guess, dtype=float), tol=1e-6, max_iters=80)
            if not projection.success:
                return None
            q_next = np.asarray(projection.x_projected, dtype=float)
            if collision_fn is not None and bool(collision_fn(q_next)):
                return None
            if float(np.linalg.norm(_task_point(robot, q_next) - np.asarray(target, dtype=float))) > 1e-1:
                return None
            q_path.append(q_next)
            current = q_next
        return np.asarray(q_path, dtype=float)

    eval_count = 0
    node_gain = 0
    tried = 0
    bridge_try_limit = max(ex66.PLANE_BRIDGE_TRY_LIMIT, min(40, len(candidate_pairs)))
    for _, left_id, right_id in candidate_pairs:
        if tried >= bridge_try_limit:
            break
        tried += 1
        exact_result = explore_joint_manifold(
            manifold=plane_store.manifold,
            start=plane_store.graph.nodes[left_id].q,
            goal=plane_store.graph.nodes[right_id].q,
            collision_fn=collision_fn,
        )
        eval_count += 1
        if not exact_result.success:
            fallback_path = guided_plane_path(
                np.asarray(plane_store.graph.nodes[left_id].q, dtype=float),
                np.asarray(plane_store.graph.nodes[right_id].q, dtype=float),
            )
            if fallback_path is None or len(fallback_path) < 2:
                continue
            class _FallbackResult:
                success = True
                def __init__(self, path):
                    self.path = np.asarray(path, dtype=float)
                    self.explored_edges = [
                        (np.asarray(path[idx], dtype=float), np.asarray(path[idx + 1], dtype=float))
                        for idx in range(len(path) - 1)
                    ]
            exact_result = _FallbackResult(fallback_path)
        plane_store.explored_edges = ex66.merge_edges(plane_store.explored_edges, list(getattr(exact_result, "explored_edges", [])))
        plane_store.chart_centers = _merge_chart_centers_joint(plane_store.chart_centers, robot, np.asarray(exact_result.path, dtype=float))
        _, node_ids, _ = ex66.connect_path_to_stage_graph(
            store=plane_store,
            source_node_id=left_id,
            path=np.asarray(exact_result.path, dtype=float),
            kind=ex66.PLANE_MOTION,
            terminal_node_id=right_id,
        )
        _update_stage_frontier_joint(plane_store, node_ids + [left_id, right_id], _task_point(robot, plane_store.graph.nodes[right_id].q), robot)
        node_gain += max(0, len(node_ids) - 2)
        break
    return node_gain, eval_count


def _connect_right_hypothesis_to_goal_joint(
    right_store: ex66.StageEvidenceStore,
    plane_right_hypotheses: list[ex66.TransitionHypothesis],
    goal_node_id: int,
    guide_point: np.ndarray,
    robot,
    collision_fn=None,
) -> tuple[int, int]:
    right_ids = sorted({int(hyp.right_node_id) for hyp in plane_right_hypotheses if hyp.right_node_id is not None})
    if len(right_ids) == 0:
        return 0, 0

    scored = []
    for node_id in right_ids:
        q = right_store.graph.nodes[node_id].q
        scored.append((float(np.linalg.norm(_task_point(robot, q) - np.asarray(guide_point, dtype=float))), node_id))
    scored.sort(key=lambda item: item[0])

    eval_count = 0
    node_gain = 0
    bridge_try_limit = max(ex66.RIGHT_BRIDGE_TRY_LIMIT, 8)
    for _, node_id in scored[: min(bridge_try_limit, len(scored))]:
        exact_result = explore_joint_manifold(
            manifold=right_store.manifold,
            start=right_store.graph.nodes[node_id].q,
            goal=right_store.graph.nodes[goal_node_id].q,
            collision_fn=collision_fn,
        )
        eval_count += 1
        if not exact_result.success:
            continue
        right_store.explored_edges = ex66.merge_edges(right_store.explored_edges, list(getattr(exact_result, "explored_edges", [])))
        right_store.chart_centers = _merge_chart_centers_joint(right_store.chart_centers, robot, np.asarray(exact_result.path, dtype=float))
        _, node_ids, _ = ex66.connect_path_to_stage_graph(
            store=right_store,
            source_node_id=node_id,
            path=np.asarray(exact_result.path, dtype=float),
            kind=ex66.RIGHT_MOTION,
            terminal_node_id=goal_node_id,
        )
        _update_stage_frontier_joint(right_store, node_ids + [goal_node_id], guide_point, robot)
        node_gain += max(0, len(node_ids) - 2)
        break
    return node_gain, eval_count


def _build_stage_task_path(store: ex66.StageEvidenceStore, node_ids: list[int], edge_ids: list[int], robot) -> np.ndarray:
    return joint_path_to_task_path(robot, ex66.build_stage_raw_path(store, node_ids, edge_ids))


def _extract_committed_route_joint(
    left_store: ex66.StageEvidenceStore,
    plane_store: ex66.StageEvidenceStore,
    right_store: ex66.StageEvidenceStore,
    start_node_id: int,
    goal_node_id: int,
    left_plane_hypotheses: list[ex66.TransitionHypothesis],
    plane_right_hypotheses: list[ex66.TransitionHypothesis],
    robot,
) -> tuple[ex66.SequentialRouteCandidate | None, bool, bool, int]:
    left_dist, left_prev_node, left_prev_edge = ex66.shortest_paths_in_stage(left_store, start_node_id)
    right_dist, right_prev_node, right_prev_edge = ex66.shortest_paths_in_stage(right_store, goal_node_id)

    entry_candidates = [hyp for hyp in left_plane_hypotheses if hyp.left_node_id in left_dist]
    exit_candidates = [hyp for hyp in plane_right_hypotheses if hyp.right_node_id in right_dist]
    has_committed_entry = len(entry_candidates) > 0
    has_committed_exit = len(exit_candidates) > 0
    if not has_committed_entry or not has_committed_exit:
        return None, has_committed_entry, has_committed_exit, 0

    best: ex66.SequentialRouteCandidate | None = None
    pairs_evaluated = 0
    for entry_hyp in entry_candidates:
        plane_dist, plane_prev_node, plane_prev_edge = ex66.shortest_paths_in_stage(plane_store, int(entry_hyp.plane_node_id))
        for exit_hyp in exit_candidates:
            pairs_evaluated += 1
            plane_exit_id = int(exit_hyp.plane_node_id)
            if plane_exit_id not in plane_dist:
                continue

            left_node_path, left_edge_path = ex66.reconstruct_stage_path(
                left_store,
                start_node_id,
                int(entry_hyp.left_node_id),
                left_prev_node,
                left_prev_edge,
            )
            plane_node_path, plane_edge_path = ex66.reconstruct_stage_path(
                plane_store,
                int(entry_hyp.plane_node_id),
                plane_exit_id,
                plane_prev_node,
                plane_prev_edge,
            )
            right_goal_to_entry_nodes, right_goal_to_entry_edges = ex66.reconstruct_stage_path(
                right_store,
                goal_node_id,
                int(exit_hyp.right_node_id),
                right_prev_node,
                right_prev_edge,
            )
            if len(left_node_path) == 0 or len(plane_node_path) == 0 or len(right_goal_to_entry_nodes) == 0:
                continue
            right_node_path = list(reversed(right_goal_to_entry_nodes))
            right_edge_path = list(reversed(right_goal_to_entry_edges))

            left_joint_path = ex66.build_stage_raw_path(left_store, left_node_path, left_edge_path)
            plane_joint_path = ex66.build_stage_raw_path(plane_store, plane_node_path, plane_edge_path)
            right_joint_path = ex66.build_stage_raw_path(right_store, right_node_path, right_edge_path)
            left_task_path = _build_stage_task_path(left_store, left_node_path, left_edge_path, robot)
            plane_task_path = _build_stage_task_path(plane_store, plane_node_path, plane_edge_path, robot)
            right_task_path = _build_stage_task_path(right_store, right_node_path, right_edge_path, robot)
            raw_path = ex66.concatenate_paths(
                left_task_path,
                np.asarray([entry_hyp.q], dtype=float),
                plane_task_path,
                np.asarray([exit_hyp.q], dtype=float),
                right_task_path,
            )
            import traceback

            try:
                left_center = left_store.manifold.center
                left_radius = left_store.manifold.radius
                p_left_start = end_effector_point(robot, left_store.graph.nodes[left_node_path[0]].q)
                p_left_end = np.asarray(entry_hyp.q, dtype=float)
                left_disp = ex66.smooth_sphere_arc(
                    left_center,
                    left_radius,
                    p_left_start,
                    p_left_end,
                    num=ex66.DISPLAY_SPHERE_SAMPLES,
                )

                p_right_start = np.asarray(exit_hyp.q, dtype=float)
                p_right_end = end_effector_point(robot, right_store.graph.nodes[right_node_path[-1]].q)
                right_center = right_store.manifold.center
                right_radius = right_store.manifold.radius
                right_disp = ex66.smooth_sphere_arc(
                    right_center,
                    right_radius,
                    p_right_start,
                    p_right_end,
                    num=ex66.DISPLAY_SPHERE_SAMPLES,
                )

                plane_entry_anchor = (
                    np.asarray(plane_task_path[0], dtype=float)
                    if len(plane_task_path) > 0
                    else np.asarray(entry_hyp.q, dtype=float)
                )
                plane_exit_anchor = (
                    np.asarray(plane_task_path[-1], dtype=float)
                    if len(plane_task_path) > 0
                    else np.asarray(exit_hyp.q, dtype=float)
                )
                plane_entry_connector = ex66.smooth_plane_segment(
                    np.asarray(entry_hyp.q, dtype=float),
                    plane_entry_anchor,
                    num=max(4, ex66.DISPLAY_PLANE_SAMPLES // 2),
                )
                plane_exit_connector = ex66.smooth_plane_segment(
                    plane_exit_anchor,
                    np.asarray(exit_hyp.q, dtype=float),
                    num=max(4, ex66.DISPLAY_PLANE_SAMPLES // 2),
                )

                display_path_smooth = ex66.concatenate_paths(
                    left_disp,
                    plane_entry_connector,
                    plane_task_path,
                    plane_exit_connector,
                    right_disp,
                )
                if len(display_path_smooth) == 0:
                    raise RuntimeError("smoothed display path is empty")
                display_path_final = np.asarray(display_path_smooth, dtype=float)
            except Exception:
                print("Warning: display smoothing failed, using raw linear path. Details:", traceback.format_exc())
                display_path_final = np.asarray(raw_path, dtype=float)

            joint_path = ex66.concatenate_paths(
                left_joint_path,
                plane_joint_path,
                right_joint_path,
            )
            total_cost = ex66.path_cost(raw_path)
            candidate = ex66.SequentialRouteCandidate(
                total_cost=float(total_cost),
                left_node_path=left_node_path,
                left_edge_path=left_edge_path,
                plane_node_path=plane_node_path,
                plane_edge_path=plane_edge_path,
                right_node_path=right_node_path,
                right_edge_path=right_edge_path,
                committed_nodes={
                    ex66.LEFT_STAGE: set(left_node_path),
                    ex66.PLANE_STAGE: set(plane_node_path),
                    ex66.RIGHT_STAGE: set(right_node_path),
                },
                raw_path=np.asarray(raw_path, dtype=float),
                display_path=display_path_final,
                joint_path=np.asarray(joint_path, dtype=float),
            )
            if best is None or candidate.total_cost + 1e-9 < best.total_cost:
                best = candidate
    return best, has_committed_entry, has_committed_exit, pairs_evaluated


def _empty_route(message: str, path_point: np.ndarray) -> ex66.FixedPlaneRoute:
    point = np.asarray(path_point, dtype=float).reshape(1, 3)
    return ex66.FixedPlaneRoute(
        success=False,
        message=message,
        total_rounds=0,
        candidate_evaluations=0,
        left_evidence_nodes=0,
        plane_evidence_nodes=0,
        right_evidence_nodes=0,
        committed_nodes=0,
        evidence_only_nodes=0,
        shared_proposals_processed=0,
        proposals_used_by_multiple_stages=0,
        plane_evidence_before_first_committed_entry=0,
        right_evidence_before_first_committed_exit=0,
        transition_hypotheses_left_plane=0,
        transition_hypotheses_plane_right=0,
        first_solution_round=None,
        best_solution_round=None,
        continued_after_first_solution=False,
        path=point,
        raw_path=point,
        certified_path_points=1,
        display_path_points=1,
        route_cost_raw=0.0,
        route_cost_display=0.0,
        graph_route_edges=0,
        obstacles=[],
        joint_path=np.zeros((0, 3), dtype=float),
    )


def plan_fixed_manifold_multimodal_route_jointspace(
    families,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    robot,
    serial_mode: bool = False,
    obstacles=None,
) -> ex66.FixedPlaneRoute:
    left_family, plane_family, right_family = families
    left_geom = left_family.manifold(float(left_family.sample_lambdas()[0]))
    plane_geom = plane_family.manifold(float(plane_family.sample_lambdas()[0]))
    right_geom = right_family.manifold(float(right_family.sample_lambdas()[0]))

    left_base = ex66.unwrap_manifold(left_geom)
    plane_base = ex66.unwrap_manifold(plane_geom)
    right_base = ex66.unwrap_manifold(right_geom)
    if not isinstance(left_base, ex66.SphereManifold) or not isinstance(right_base, ex66.SphereManifold):
        return _empty_route("Robot joint-space mode requires sphere support manifolds on the left and right.", start_q)
    if not isinstance(plane_base, ex66.PlaneManifold):
        return _empty_route("Robot joint-space mode requires a plane transfer manifold in the middle stage.", start_q)

    joint_lower = -np.pi * np.ones(3, dtype=float)
    joint_upper = np.pi * np.ones(3, dtype=float)
    left_manifold = RobotSphereManifold(robot=robot, center=left_base.center, radius=left_base.radius, name="robot_left_sphere", joint_lower=joint_lower, joint_upper=joint_upper)
    plane_task_validity = (
        (lambda ee, geom=plane_geom: bool(geom.within_bounds(np.asarray(ee, dtype=float))))
        if isinstance(plane_geom, ex66.MaskedManifold)
        else None
    )
    plane_manifold = RobotPlaneManifold(
        robot=robot,
        point=plane_base.point,
        normal=plane_base.normal,
        name="robot_transfer_plane",
        joint_lower=joint_lower,
        joint_upper=joint_upper,
        task_space_validity_fn=plane_task_validity,
    )
    right_manifold = RobotSphereManifold(robot=robot, center=right_base.center, radius=right_base.radius, name="robot_right_sphere", joint_lower=joint_lower, joint_upper=joint_upper)
    active_obstacles = default_example_66_obstacles() if obstacles is None else list(obstacles)
    collision_fn = lambda joint_angles: configuration_in_collision(robot, joint_angles, active_obstacles)

    start_theta_seed = inverse_kinematics_start(robot, np.asarray(start_q, dtype=float), joint_lower=joint_lower, joint_upper=joint_upper)
    if start_theta_seed is None:
        return _empty_route("Failed to find a start joint configuration whose end-effector lies on the left support sphere.", start_q)
    goal_theta_seed = inverse_kinematics_start(robot, np.asarray(goal_q, dtype=float), warm_start=start_theta_seed, joint_lower=joint_lower, joint_upper=joint_upper)
    if goal_theta_seed is None:
        return _empty_route("Failed to find a goal joint configuration whose end-effector lies on the right support sphere.", goal_q)

    start_projection = left_manifold.project(start_theta_seed)
    goal_projection = right_manifold.project(goal_theta_seed)
    if not start_projection.success:
        return _empty_route("Failed to project the start IK seed onto the left robot-sphere manifold.", start_q)
    if not goal_projection.success:
        return _empty_route("Failed to project the goal IK seed onto the right robot-sphere manifold.", goal_q)
    start_theta = np.asarray(start_projection.x_projected, dtype=float)
    goal_theta = np.asarray(goal_projection.x_projected, dtype=float)
    if collision_fn(start_theta):
        return _empty_route("Start joint configuration collides with an obstacle.", start_q)
    if collision_fn(goal_theta):
        return _empty_route("Goal joint configuration collides with an obstacle.", goal_q)

    stores = {
        ex66.LEFT_STAGE: ex66.StageEvidenceStore(stage=ex66.LEFT_STAGE, manifold=left_manifold),
        ex66.PLANE_STAGE: ex66.StageEvidenceStore(stage=ex66.PLANE_STAGE, manifold=plane_manifold),
        ex66.RIGHT_STAGE: ex66.StageEvidenceStore(stage=ex66.RIGHT_STAGE, manifold=right_manifold),
    }

    start_node_id = ex66.add_stage_node(stores[ex66.LEFT_STAGE], start_theta, seeded_from_proposal=False)
    goal_node_id = ex66.add_stage_node(stores[ex66.RIGHT_STAGE], goal_theta, seeded_from_proposal=False)
    stores[ex66.LEFT_STAGE].frontier_ids = [start_node_id]
    stores[ex66.RIGHT_STAGE].frontier_ids = [goal_node_id]

    left_plane_hypotheses: list[ex66.TransitionHypothesis] = []
    plane_right_hypotheses: list[ex66.TransitionHypothesis] = []

    candidate_evaluations = 0
    total_rounds = 0
    shared_proposals_processed = 0
    proposals_used_by_multiple_stages = 0
    useful_stage_total = 0
    multi_stage_update_total = 0
    proposal_rounds_with_plane_updates = 0
    proposal_rounds_with_multi_stage_updates = 0
    committed_route_changes_after_first_solution = 0
    alternative_hypothesis_pairs_evaluated = 0

    stage_node_gains = {stage: [] for stage in ex66.STAGES}
    stage_transition_gains = {stage: [] for stage in ex66.STAGES}
    stage_route_gains = {stage: [] for stage in ex66.STAGES}
    best_cost_history: list[float] = []

    best_candidate: ex66.SequentialRouteCandidate | None = None
    first_solution_round: int | None = None
    best_solution_round: int | None = None
    first_committed_entry_round: int | None = None
    first_committed_exit_round: int | None = None
    plane_evidence_before_first_committed_entry = 0
    right_evidence_before_first_committed_exit = 0
    plane_evidence_at_first_solution = 0
    right_evidence_at_first_solution = 0

    guides_task = {
        ex66.LEFT_STAGE: np.asarray(start_q, dtype=float),
        ex66.PLANE_STAGE: 0.5 * (np.asarray(start_q, dtype=float) + np.asarray(goal_q, dtype=float)),
        ex66.RIGHT_STAGE: np.asarray(goal_q, dtype=float),
    }

    mode_counts = {
        "shared_proposal_round": 0,
        "right_goal_bias_updates": 0,
    }
    if serial_mode:
        mode_counts["serial_round"] = 0

    for round_idx in range(1, ex66.SAFETY_MAX_TOTAL_ROUNDS + 1):
        current_stage_counts = ex66.stage_evidence_counts(stores)
        total_rounds = round_idx
        if ex66.should_stop_exploration(
            first_solution_round=first_solution_round,
            total_rounds=round_idx,
            stage_node_gains=stage_node_gains,
            stage_transition_gains=stage_transition_gains,
            stage_route_gains=stage_route_gains,
            current_stage_counts=current_stage_counts,
        ):
            break

        proposal_count = ex66.effective_proposals_per_round(round_idx, first_solution_round, stores)
        mode_counts["shared_proposal_round"] += 1

        round_node_gain = {stage: 0 for stage in ex66.STAGES}
        round_transition_gain = {stage: 0 for stage in ex66.STAGES}
        route_improved_this_round = 0
        plane_updated_this_round = False
        multi_stage_updated_this_round = False
        active_serial_stage = ex66.greedy_stage_for_serial_round(stores) if serial_mode else None
        if serial_mode:
            mode_counts["serial_round"] += 1
            mode_counts[f"serial_active_{active_serial_stage}"] = mode_counts.get(f"serial_active_{active_serial_stage}", 0) + 1

        guide_joint_pool: list[np.ndarray] = [start_theta.copy(), goal_theta.copy()]
        for stage in ex66.STAGES:
            for node_id in stores[stage].frontier_ids[: min(8, len(stores[stage].frontier_ids))]:
                guide_joint_pool.append(np.asarray(stores[stage].graph.nodes[node_id].q, dtype=float))
        proposals = generate_joint_proposals(
            round_idx=round_idx,
            start_q=start_theta,
            goal_q=goal_theta,
            guides=guide_joint_pool,
            proposal_count=proposal_count,
            joint_lower=joint_lower,
            joint_upper=joint_upper,
        )

        adaptive_budget = ex66.adaptive_stage_update_budget(stores, first_solution_round)

        for proposal in proposals:
            shared_proposals_processed += 1
            stage_candidates: list[tuple[float, str, np.ndarray]] = []
            for stage in ex66.STAGES:
                projection = stores[stage].manifold.project(np.asarray(proposal, dtype=float))
                if not projection.success:
                    continue
                projected_q = np.asarray(projection.x_projected, dtype=float)
                score = _proposal_stage_utility_joint(
                    stage=stage,
                    projected_q=projected_q,
                    store=stores[stage],
                    guide_point=guides_task[stage],
                    stores=stores,
                    robot=robot,
                )
                stage_candidates.append((score, stage, projected_q))

            if len(stage_candidates) == 0:
                continue
            stage_candidates.sort(key=lambda item: item[0], reverse=True)
            useful_stages = [stage for _, stage, _ in stage_candidates]
            useful_stage_total += len(useful_stages)
            if len(useful_stages) > 1:
                proposals_used_by_multiple_stages += 1
                if not serial_mode:
                    multi_stage_update_total += len(useful_stages)
                    multi_stage_updated_this_round = True

            active_candidates = stage_candidates
            if serial_mode:
                active_candidates = [candidate for candidate in stage_candidates if candidate[1] == active_serial_stage]
            updates_used = 0
            for _score, stage, projected_q in active_candidates:
                if not serial_mode and updates_used >= adaptive_budget:
                    break
                node_gain, evals, result, source_node_id, path_node_ids = _update_stage_evidence_from_proposal_joint(
                    store=stores[stage],
                    target_q=projected_q,
                    guide_point=guides_task[stage],
                    robot=robot,
                    collision_fn=collision_fn,
                )
                candidate_evaluations += evals
                round_node_gain[stage] += node_gain
                updates_used += 1
                if stage == ex66.PLANE_STAGE and (node_gain > 0 or result is not None):
                    plane_updated_this_round = True

                if stage in [ex66.LEFT_STAGE, ex66.PLANE_STAGE]:
                    transition_gain, evals = _add_left_plane_hypotheses_joint(
                        source_stage=stage,
                        source_store=stores[stage],
                        plane_store=stores[ex66.PLANE_STAGE],
                        left_store=stores[ex66.LEFT_STAGE],
                        left_geom=left_geom,
                        plane_geom=plane_geom,
                        result=result,
                        source_node_id=source_node_id,
                        path_node_ids=path_node_ids,
                        guide_point=guides_task[ex66.PLANE_STAGE],
                        hypotheses=left_plane_hypotheses,
                        robot=robot,
                        collision_fn=collision_fn,
                    )
                    candidate_evaluations += evals
                    round_transition_gain[ex66.LEFT_STAGE] += transition_gain
                    round_transition_gain[ex66.PLANE_STAGE] += transition_gain
                    if stage == ex66.PLANE_STAGE and transition_gain > 0:
                        plane_updated_this_round = True

                if stage in [ex66.PLANE_STAGE, ex66.RIGHT_STAGE]:
                    transition_gain, evals = _add_plane_right_hypotheses_joint(
                        source_stage=stage,
                        source_store=stores[stage],
                        plane_store=stores[ex66.PLANE_STAGE],
                        right_store=stores[ex66.RIGHT_STAGE],
                        plane_geom=plane_geom,
                        right_geom=right_geom,
                        result=result,
                        source_node_id=source_node_id,
                        path_node_ids=path_node_ids,
                        guide_point=guides_task[ex66.RIGHT_STAGE],
                        hypotheses=plane_right_hypotheses,
                        robot=robot,
                        collision_fn=collision_fn,
                    )
                    candidate_evaluations += evals
                    round_transition_gain[ex66.PLANE_STAGE] += transition_gain
                    round_transition_gain[ex66.RIGHT_STAGE] += transition_gain
                    if stage == ex66.PLANE_STAGE and transition_gain > 0:
                        plane_updated_this_round = True

        if not serial_mode or active_serial_stage == ex66.LEFT_STAGE:
            left_bridge_gain, left_bridge_evals = _bridge_left_hypotheses_to_start_joint(
                left_store=stores[ex66.LEFT_STAGE],
                start_node_id=start_node_id,
                left_plane_hypotheses=left_plane_hypotheses,
                guide_point=guides_task[ex66.PLANE_STAGE],
                robot=robot,
                collision_fn=collision_fn,
            )
            candidate_evaluations += left_bridge_evals
            round_node_gain[ex66.LEFT_STAGE] += left_bridge_gain

        if not serial_mode or active_serial_stage == ex66.PLANE_STAGE:
            plane_bridge_gain, plane_bridge_evals = _bridge_plane_hypothesis_components_joint(
                plane_store=stores[ex66.PLANE_STAGE],
                left_plane_hypotheses=left_plane_hypotheses,
                plane_right_hypotheses=plane_right_hypotheses,
                robot=robot,
                collision_fn=collision_fn,
            )
            candidate_evaluations += plane_bridge_evals
            round_node_gain[ex66.PLANE_STAGE] += plane_bridge_gain
            if plane_bridge_gain > 0:
                plane_updated_this_round = True

        if not serial_mode or active_serial_stage == ex66.RIGHT_STAGE:
            right_bridge_gain, right_bridge_evals = _connect_right_hypothesis_to_goal_joint(
                right_store=stores[ex66.RIGHT_STAGE],
                plane_right_hypotheses=plane_right_hypotheses,
                goal_node_id=goal_node_id,
                guide_point=guides_task[ex66.RIGHT_STAGE],
                robot=robot,
                collision_fn=collision_fn,
            )
            candidate_evaluations += right_bridge_evals
            round_node_gain[ex66.RIGHT_STAGE] += right_bridge_gain

        if len(stores[ex66.RIGHT_STAGE].frontier_ids) > 0 and (not serial_mode or active_serial_stage == ex66.RIGHT_STAGE):
            mode_counts["right_goal_bias_updates"] += 1
            for source_node_id in _ranked_stage_sources_joint(stores[ex66.RIGHT_STAGE], guides_task[ex66.RIGHT_STAGE], limit=3, robot=robot):
                ex66.increment_stage_node_expansion(stores[ex66.RIGHT_STAGE], source_node_id)
                source_q = stores[ex66.RIGHT_STAGE].graph.nodes[source_node_id].q
                exact_result = explore_joint_manifold(
                    manifold=stores[ex66.RIGHT_STAGE].manifold,
                    start=source_q,
                    goal=goal_theta,
                    collision_fn=collision_fn,
                )
                candidate_evaluations += 1
                if not exact_result.success:
                    continue
                stores[ex66.RIGHT_STAGE].explored_edges = ex66.merge_edges(
                    stores[ex66.RIGHT_STAGE].explored_edges,
                    list(getattr(exact_result, "explored_edges", [])),
                )
                stores[ex66.RIGHT_STAGE].chart_centers = _merge_chart_centers_joint(
                    stores[ex66.RIGHT_STAGE].chart_centers,
                    robot,
                    np.asarray(exact_result.path, dtype=float),
                )
                _, path_nodes, _ = ex66.connect_path_to_stage_graph(
                    store=stores[ex66.RIGHT_STAGE],
                    source_node_id=source_node_id,
                    path=np.asarray(exact_result.path, dtype=float),
                    kind=ex66.RIGHT_MOTION,
                    terminal_node_id=goal_node_id,
                )
                _update_stage_frontier_joint(stores[ex66.RIGHT_STAGE], path_nodes + [goal_node_id], guides_task[ex66.RIGHT_STAGE], robot)
                round_node_gain[ex66.RIGHT_STAGE] += max(0, len(path_nodes) - 1)
                break

        candidate, has_committed_entry, has_committed_exit, pairs_evaluated = _extract_committed_route_joint(
            left_store=stores[ex66.LEFT_STAGE],
            plane_store=stores[ex66.PLANE_STAGE],
            right_store=stores[ex66.RIGHT_STAGE],
            start_node_id=start_node_id,
            goal_node_id=goal_node_id,
            left_plane_hypotheses=left_plane_hypotheses,
            plane_right_hypotheses=plane_right_hypotheses,
            robot=robot,
        )
        alternative_hypothesis_pairs_evaluated += pairs_evaluated

        if has_committed_entry and first_committed_entry_round is None:
            first_committed_entry_round = round_idx
            plane_evidence_before_first_committed_entry = len(stores[ex66.PLANE_STAGE].graph.nodes)
        if has_committed_exit and first_committed_exit_round is None:
            first_committed_exit_round = round_idx
            right_evidence_before_first_committed_exit = len(stores[ex66.RIGHT_STAGE].graph.nodes)

        if candidate is not None:
            if first_solution_round is None:
                first_solution_round = round_idx
                plane_evidence_at_first_solution = len(stores[ex66.PLANE_STAGE].graph.nodes)
                right_evidence_at_first_solution = len(stores[ex66.RIGHT_STAGE].graph.nodes)
            if best_candidate is None or candidate.total_cost + 1e-9 < best_candidate.total_cost:
                if first_solution_round is not None and best_candidate is not None and round_idx > first_solution_round:
                    committed_route_changes_after_first_solution += 1
                best_candidate = candidate
                best_solution_round = round_idx
                route_improved_this_round = 1

        if plane_updated_this_round:
            proposal_rounds_with_plane_updates += 1
        if multi_stage_updated_this_round:
            proposal_rounds_with_multi_stage_updates += 1

        best_cost_history.append(best_candidate.total_cost if best_candidate is not None else 1e12)
        for stage in ex66.STAGES:
            stage_node_gains[stage].append(int(round_node_gain[stage]))
            stage_transition_gains[stage].append(int(round_transition_gain[stage]))
            stage_route_gains[stage].append(int(route_improved_this_round))

    success = best_candidate is not None
    committed_nodes = {stage: set() for stage in ex66.STAGES}
    raw_path = np.asarray([np.asarray(start_q, dtype=float)], dtype=float)
    display_path = np.asarray([np.asarray(start_q, dtype=float)], dtype=float)
    joint_path = np.zeros((0, 3), dtype=float)
    route_cost_raw = 0.0
    route_cost_display = 0.0
    graph_route_edges = 0
    if best_candidate is not None:
        committed_nodes = best_candidate.committed_nodes
        raw_path = np.asarray(best_candidate.raw_path, dtype=float)
        display_path = np.asarray(best_candidate.display_path, dtype=float)
        joint_path = np.asarray(best_candidate.joint_path, dtype=float)
        route_cost_raw = ex66.path_cost(raw_path)
        route_cost_display = ex66.path_cost(display_path)
        graph_route_edges = (
            len(best_candidate.left_edge_path)
            + len(best_candidate.plane_edge_path)
            + len(best_candidate.right_edge_path)
        )

    total_evidence_nodes = sum(len(stores[stage].graph.nodes) for stage in ex66.STAGES)
    committed_node_count = sum(len(committed_nodes[stage]) for stage in ex66.STAGES)
    evidence_only_nodes = max(0, total_evidence_nodes - committed_node_count)
    continued_after_first_solution = bool(first_solution_round is not None and total_rounds > first_solution_round)
    plane_evidence_growth_after_first_solution = (
        max(0, len(stores[ex66.PLANE_STAGE].graph.nodes) - plane_evidence_at_first_solution)
        if first_solution_round is not None
        else 0
    )
    right_evidence_growth_after_first_solution = (
        max(0, len(stores[ex66.RIGHT_STAGE].graph.nodes) - right_evidence_at_first_solution)
        if first_solution_round is not None
        else 0
    )
    multi_stage_updates_per_round = multi_stage_update_total / max(total_rounds, 1)
    average_useful_stages_per_proposal = useful_stage_total / max(shared_proposals_processed, 1)
    stage_stagnation_flags = {
        stage: ex66.stage_stagnating(stage_node_gains[stage], stage_transition_gains[stage], stage_route_gains[stage])
        for stage in ex66.STAGES
    }
    saturated_before_solution = bool(not success and total_rounds > 0)
    stagnation_stage = None
    if saturated_before_solution:
        stagnant = [stage for stage in ex66.STAGES if stage_stagnation_flags.get(stage, False)]
        if len(stagnant) == 1:
            stagnation_stage = stagnant[0]
        elif len(stagnant) == len(ex66.STAGES):
            stagnation_stage = "all"
        elif len(stagnant) > 1:
            stagnation_stage = ",".join(stagnant)
        else:
            stagnation_stage = ex66.greedy_stage_for_serial_round(stores) if serial_mode else None

    return ex66.FixedPlaneRoute(
        success=success,
        message=(
            (
                "Serial stage evidence accumulated enough certified sequential structure to extract a fixed left-plane-right robot joint-space route."
                if serial_mode
                else "Parallel stage evidence accumulated enough certified sequential structure to extract a fixed left-plane-right robot joint-space route."
            )
            if success
            else (
                "Robot joint-space serial exploration accumulated evidence across the fixed left / plane / right manifolds, but no certified sequential route was extracted before exploration saturated."
                if serial_mode
                else "Robot joint-space evidence accumulated across the fixed left / plane / right manifolds, but no certified sequential route was extracted before exploration saturated."
            )
        ),
        total_rounds=total_rounds,
        candidate_evaluations=candidate_evaluations,
        left_evidence_nodes=len(stores[ex66.LEFT_STAGE].graph.nodes),
        plane_evidence_nodes=len(stores[ex66.PLANE_STAGE].graph.nodes),
        right_evidence_nodes=len(stores[ex66.RIGHT_STAGE].graph.nodes),
        committed_nodes=committed_node_count,
        evidence_only_nodes=evidence_only_nodes,
        shared_proposals_processed=shared_proposals_processed,
        proposals_used_by_multiple_stages=proposals_used_by_multiple_stages,
        plane_evidence_before_first_committed_entry=plane_evidence_before_first_committed_entry,
        right_evidence_before_first_committed_exit=right_evidence_before_first_committed_exit,
        transition_hypotheses_left_plane=len(left_plane_hypotheses),
        transition_hypotheses_plane_right=len(plane_right_hypotheses),
        first_solution_round=first_solution_round,
        best_solution_round=best_solution_round,
        continued_after_first_solution=continued_after_first_solution,
        path=np.asarray(display_path, dtype=float),
        raw_path=np.asarray(raw_path, dtype=float),
        certified_path_points=int(len(raw_path)),
        display_path_points=int(len(display_path)),
        route_cost_raw=route_cost_raw,
        route_cost_display=route_cost_display,
        graph_route_edges=graph_route_edges,
        stage_evidence_points={stage: _stage_evidence_points_joint(stores[stage], robot) for stage in ex66.STAGES},
        stage_evidence_edges={stage: _task_edge_segments(stores[stage], robot) for stage in ex66.STAGES},
        stage_frontier_points={stage: _task_frontier_points(stores[stage], robot) for stage in ex66.STAGES},
        stage_chart_centers={stage: np.asarray(stores[stage].chart_centers, dtype=float) for stage in ex66.STAGES},
        stage_frontier_counts={stage: len(stores[stage].frontier_ids) for stage in ex66.STAGES},
        stage_stagnation_flags=stage_stagnation_flags,
        recent_graph_node_gain=sum(ex66.stage_recent_sum(stage_node_gains[stage]) for stage in ex66.STAGES),
        recent_transition_gain=sum(ex66.stage_recent_sum(stage_transition_gains[stage]) for stage in ex66.STAGES),
        recent_route_improvement_gain=ex66.recent_route_improvement(best_cost_history),
        plane_evidence_growth_after_first_solution=plane_evidence_growth_after_first_solution,
        right_evidence_growth_after_first_solution=right_evidence_growth_after_first_solution,
        multi_stage_updates_per_round=float(multi_stage_updates_per_round),
        average_useful_stages_per_proposal=float(average_useful_stages_per_proposal),
        proposal_rounds_with_plane_updates=proposal_rounds_with_plane_updates,
        proposal_rounds_with_multi_stage_updates=proposal_rounds_with_multi_stage_updates,
        committed_route_changes_after_first_solution=committed_route_changes_after_first_solution,
        alternative_hypothesis_pairs_evaluated=alternative_hypothesis_pairs_evaluated,
        left_plane_hypothesis_points=ex66.deduplicate_points([hyp.q for hyp in left_plane_hypotheses], tol=ex66.TRANSITION_DEDUP_TOL),
        plane_right_hypothesis_points=ex66.deduplicate_points([hyp.q for hyp in plane_right_hypotheses], tol=ex66.TRANSITION_DEDUP_TOL),
        committed_stage_nodes={
            ex66.LEFT_STAGE: np.asarray([_task_point(robot, stores[ex66.LEFT_STAGE].graph.nodes[node_id].q) for node_id in sorted(committed_nodes[ex66.LEFT_STAGE])], dtype=float)
            if len(committed_nodes[ex66.LEFT_STAGE]) > 0
            else np.zeros((0, 3), dtype=float),
            ex66.PLANE_STAGE: np.asarray([_task_point(robot, stores[ex66.PLANE_STAGE].graph.nodes[node_id].q) for node_id in sorted(committed_nodes[ex66.PLANE_STAGE])], dtype=float)
            if len(committed_nodes[ex66.PLANE_STAGE]) > 0
            else np.zeros((0, 3), dtype=float),
            ex66.RIGHT_STAGE: np.asarray([_task_point(robot, stores[ex66.RIGHT_STAGE].graph.nodes[node_id].q) for node_id in sorted(committed_nodes[ex66.RIGHT_STAGE])], dtype=float)
            if len(committed_nodes[ex66.RIGHT_STAGE]) > 0
            else np.zeros((0, 3), dtype=float),
        },
        mode_counts=mode_counts,
        serial_mode=serial_mode,
        saturated_before_solution=saturated_before_solution,
        stagnation_stage=stagnation_stage,
        obstacles=active_obstacles,
        joint_path=np.asarray(joint_path, dtype=float),
    )
