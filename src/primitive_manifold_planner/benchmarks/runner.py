from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, List, Optional
import time
import csv
import os
import numpy as np

from primitive_manifold_planner.families.leaf_state import LeafState
from primitive_manifold_planner.planners.leaf_graph import build_leaf_graph, shortest_leaf_route
from primitive_manifold_planner.planners.realize_leaf_route import realize_leaf_route


@dataclass
class BenchmarkCase:
    name: str
    families: list
    start_state: LeafState
    goal_point: np.ndarray
    goal_family_name: str
    goal_lam: object
    seed_points_fn: Callable


@dataclass
class BenchmarkResult:
    case_name: str
    success: bool
    route_found: bool
    route_num_edges: int
    realized_num_steps: int
    chosen_leaf_sequence: str
    transition_points: str
    path_length: float
    runtime_sec: float
    message: str


def compute_path_length(realized) -> float:
    total = 0.0
    if realized is None:
        return total

    for step in realized.steps:
        pts = np.asarray(step.path)
        if len(pts) < 2:
            continue
        total += float(sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)))
    return total


def route_to_string(route) -> str:
    if route is None:
        return ""
    if len(route) == 0:
        return ""
    nodes = [route[0].src] + [e.dst for e in route]
    return " -> ".join([f"{fam}[{lam}]" for fam, lam in nodes])


def transitions_to_string(route) -> str:
    if route is None:
        return ""
    pts = []
    for e in route:
        p = np.round(e.transition_point, 4)
        pts.append(f"({p[0]}, {p[1]})")
    return " ; ".join(pts)


def run_benchmark_case(
    case: BenchmarkCase,
    project_newton,
    constrained_interpolate,
    max_candidates_per_pair: int = 3,
) -> BenchmarkResult:
    t0 = time.perf_counter()

    goal_family = next(f for f in case.families if f.name == case.goal_family_name)

    start_key = (case.start_state.family_name, case.start_state.lam)
    goal_key = (case.goal_family_name, case.goal_lam)

    graph = build_leaf_graph(
        families=case.families,
        project_newton=project_newton,
        seed_points_fn=case.seed_points_fn,
        goal_point=case.goal_point,
        max_candidates_per_pair=max_candidates_per_pair,
    )

    route = shortest_leaf_route(graph, start=start_key, goal=goal_key)

    if route is None:
        t1 = time.perf_counter()
        return BenchmarkResult(
            case_name=case.name,
            success=False,
            route_found=False,
            route_num_edges=0,
            realized_num_steps=0,
            chosen_leaf_sequence="",
            transition_points="",
            path_length=0.0,
            runtime_sec=t1 - t0,
            message="No leaf route found.",
        )

    realized = realize_leaf_route(
        start_state=case.start_state,
        goal_point=case.goal_point,
        goal_family=goal_family,
        goal_lam=case.goal_lam,
        families=case.families,
        route_edges=route,
        constrained_interpolate=constrained_interpolate,
        step_size=0.08,
    )

    t1 = time.perf_counter()

    return BenchmarkResult(
        case_name=case.name,
        success=bool(realized.success),
        route_found=True,
        route_num_edges=len(route),
        realized_num_steps=len(realized.steps),
        chosen_leaf_sequence=route_to_string(route),
        transition_points=transitions_to_string(route),
        path_length=compute_path_length(realized),
        runtime_sec=t1 - t0,
        message=realized.message,
    )


def save_results_csv(results: List[BenchmarkResult], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_name",
                "success",
                "route_found",
                "route_num_edges",
                "realized_num_steps",
                "chosen_leaf_sequence",
                "transition_points",
                "path_length",
                "runtime_sec",
                "message",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))