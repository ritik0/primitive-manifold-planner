from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from primitive_manifold_planner.families.leaf_state import LeafState
from primitive_manifold_planner.families.standard import PlaneFamily, SphereFamily
from primitive_manifold_planner.manifolds import PlaneManifold, SphereManifold
from primitive_manifold_planner.planners.admissibility import (
    build_semantic_target_progress_point_fn,
    ProgressTargetSelectionConfig,
)
from primitive_manifold_planner.planners.mode_semantics import PlanningSemanticContext
from primitive_manifold_planner.planners.plan_foliated_route import plan_foliated_route
from primitive_manifold_planner.planners.semantic_templates import (
    build_support_transfer_goal_semantic_model,
)
from primitive_manifold_planner.projection import project_newton


@dataclass
class ConnectorCase:
    start_family_name: str
    start_lam: float
    start_point: np.ndarray
    goal_family_name: str
    goal_lam: float
    goal_point: np.ndarray
    title: str


def sphere_point(center: np.ndarray, radius: float, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = np.deg2rad(float(azimuth_deg))
    el = np.deg2rad(float(elevation_deg))
    direction = np.array(
        [
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ],
        dtype=float,
    )
    return np.asarray(center, dtype=float) + float(radius) * direction


def build_competing_connector_benchmark():
    left_support = SphereFamily(
        name="left_support_3d",
        center=np.array([-1.70, 0.0, 0.45], dtype=float),
        radii={1.10: 1.10},
    )
    lower_transfer = PlaneFamily(
        name="lower_transfer_plane_3d",
        base_point=np.array([0.0, 0.0, 0.0], dtype=float),
        normal=np.array([0.0, 0.0, 1.0], dtype=float),
        offsets=[-0.08],
        anchor_span=1.0,
        feasibility_fn=lambda lam, point, goal, meta: bool(-1.1 <= float(point[0]) <= 1.1 and -0.85 <= float(point[1]) <= 0.85),
    )
    upper_transfer = PlaneFamily(
        name="upper_transfer_plane_3d",
        base_point=np.array([0.0, 0.0, 0.0], dtype=float),
        normal=np.array([0.0, 0.0, 1.0], dtype=float),
        offsets=[0.26],
        anchor_span=1.0,
        feasibility_fn=lambda lam, point, goal, meta: bool(-1.1 <= float(point[0]) <= 1.1 and -0.85 <= float(point[1]) <= 0.85),
    )
    right_support = SphereFamily(
        name="right_support_3d",
        center=np.array([1.70, 0.0, 0.45], dtype=float),
        radii={1.10: 1.10},
    )

    families = [left_support, lower_transfer, upper_transfer, right_support]

    case = ConnectorCase(
        start_family_name="left_support_3d",
        start_lam=1.10,
        start_point=sphere_point(left_support.center, 1.10, azimuth_deg=-28.0, elevation_deg=-34.0),
        goal_family_name="right_support_3d",
        goal_lam=1.10,
        goal_point=sphere_point(right_support.center, 1.10, azimuth_deg=208.0, elevation_deg=-34.0),
        title="3D competing transfer connectors",
    )

    allowed_family_transitions = {
        "left_support_3d": ["lower_transfer_plane_3d", "upper_transfer_plane_3d"],
        "lower_transfer_plane_3d": ["right_support_3d"],
        "upper_transfer_plane_3d": ["right_support_3d"],
        "right_support_3d": [],
    }

    return families, case, allowed_family_transitions


def build_semantic_model():
    def semantic_feasibility(context: PlanningSemanticContext) -> bool:
        if context.point is None:
            return True
        involved = {context.source_family_name, context.target_family_name}
        if not {"lower_transfer_plane_3d", "upper_transfer_plane_3d"} & involved:
            return True
        q = np.asarray(context.point, dtype=float)
        return bool(-1.1 <= float(q[0]) <= 1.1 and -0.85 <= float(q[1]) <= 0.85)

    def semantic_admissibility(context: PlanningSemanticContext) -> float:
        if context.point is None:
            return 0.0
        q = np.asarray(context.point, dtype=float)
        goal = None if context.goal_point is None else np.asarray(context.goal_point, dtype=float)
        cost = 0.0
        if "upper_transfer_plane_3d" in {context.source_family_name, context.target_family_name}:
            cost += 0.45
        if goal is not None:
            cost += 0.04 * float(np.linalg.norm(q - goal))
        return float(cost)

    return build_support_transfer_goal_semantic_model(
        support_families=["left_support_3d"],
        transfer_families=["lower_transfer_plane_3d", "upper_transfer_plane_3d"],
        goal_support_families=["right_support_3d"],
        transition_feasibility_fn=semantic_feasibility,
        transition_admissibility_fn=semantic_admissibility,
    )


def concatenate_steps(steps) -> np.ndarray:
    pts = []
    for i, step in enumerate(steps):
        path = np.asarray(step.path)
        if len(path) == 0:
            continue
        if i == 0:
            pts.extend(list(path))
        else:
            pts.extend(list(path[1:]))
    if not pts:
        return np.zeros((0, 3))
    return np.asarray(pts)


def total_path_length_from_steps(steps) -> float:
    total = 0.0
    for step in steps:
        path = np.asarray(step.path)
        if len(path) >= 2:
            total += float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    return total


def summarize_leaf_sequence(result) -> str:
    seq = []
    for step in result.steps:
        node = f"{step.family_name}[{step.lam}]"
        if not seq or seq[-1] != node:
            seq.append(node)
    return " -> ".join(seq)


def selected_transfer_family(result) -> str | None:
    for step in result.steps:
        if "transfer_plane" in step.family_name:
            return step.family_name
    return None


def plot_manifold(ax, manifold, color="lightgray", alpha=0.16):
    if isinstance(manifold, SphereManifold):
        u = np.linspace(0.0, 2.0 * np.pi, 36)
        v = np.linspace(0.0, np.pi, 20)
        x = manifold.center[0] + manifold.radius * np.outer(np.cos(u), np.sin(v))
        y = manifold.center[1] + manifold.radius * np.outer(np.sin(u), np.sin(v))
        z = manifold.center[2] + manifold.radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(x, y, z, color=color, alpha=alpha, rstride=2, cstride=2, linewidth=0.6)
    elif isinstance(manifold, PlaneManifold):
        xx = np.linspace(-1.1, 1.1, 14)
        yy = np.linspace(-0.85, 0.85, 10)
        X, Y = np.meshgrid(xx, yy)
        Z = np.zeros_like(X) + manifold.point[2]
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0.0, shade=False)


def main():
    np.random.seed(23)

    families, case, allowed_family_transitions = build_competing_connector_benchmark()
    family_map = {f.name: f for f in families}
    semantic_model = build_semantic_model()

    result = plan_foliated_route(
        start_state=LeafState(case.start_family_name, case.start_lam, case.start_point.copy()),
        goal_point=case.goal_point.copy(),
        goal_family=family_map[case.goal_family_name],
        goal_lam=case.goal_lam,
        families=families,
        project_newton=project_newton,
        max_switches=3,
        local_step_size=0.10,
        progress_tol=1e-4,
        local_planner_name="projection",
        local_planner_kwargs=dict(
            goal_tol=1e-3,
            max_iters=900,
            projection_tol=1e-10,
            projection_max_iters=60,
            projection_damping=1.0,
        ),
        allowed_family_transitions=allowed_family_transitions,
        semantic_model=semantic_model,
        goal_leaf_mismatch_penalty=10.0,
        target_progress_point_fn=build_semantic_target_progress_point_fn(
            ProgressTargetSelectionConfig(goal_distance_weight=1.0, admissibility_weight=1.8)
        ),
    )

    print(f"\nRunning {case.title}")
    print(f"success = {result.success}")
    print(f"message = {result.message}")
    print(f"path_length = {total_path_length_from_steps(result.steps):.4f}")
    print(f"leaf sequence = {summarize_leaf_sequence(result)}")
    print(f"selected transfer family = {selected_transfer_family(result)}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = {
        "left_support_3d": "#c58b4c",
        "lower_transfer_plane_3d": "#7fa7c6",
        "upper_transfer_plane_3d": "#9fb7d0",
        "right_support_3d": "#c58b4c",
    }
    for fam in families:
        for lam in fam.sample_lambdas():
            plot_manifold(ax, fam.manifold(lam), color=colors.get(fam.name, "lightgray"))

    path = concatenate_steps(result.steps)
    if len(path) > 0:
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color="#1565c0", linewidth=3.0, label="realized path")

    for step in result.steps:
        if step.transition is not None:
            tp = np.asarray(step.transition.x, dtype=float)
            ax.scatter(tp[0], tp[1], tp[2], s=65, marker="x", color="black")

    ax.scatter(case.start_point[0], case.start_point[1], case.start_point[2], s=90, marker="s", color="black", label="start")
    ax.scatter(case.goal_point[0], case.goal_point[1], case.goal_point[2], s=130, marker="*", color="gold", edgecolor="black", label="goal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Example 60: competing 3D transfer connectors")
    ax.legend(loc="upper right")
    ax.view_init(elev=24, azim=-58)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
