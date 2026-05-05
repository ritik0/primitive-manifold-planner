from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from ompl import base as ob
from ompl import geometric as og

from primitive_manifold_planner.examplesupport.planar3link import (
    Planar3LinkKinematics,
    torus_distance,
    wrap_q,
)
from primitive_manifold_planner.examplesupport.planar3link_families import (
    EndEffectorCircleFamily3Link,
    EndEffectorEllipseFamily3Link,
)
from primitive_manifold_planner.families.leaf_state import LeafState
from primitive_manifold_planner.planners.component_discovery import (
    ComponentModelRegistry,
    build_component_model_registry,
)
from primitive_manifold_planner.planners.mode_semantics import PlanningSemanticContext
from primitive_manifold_planner.planners.multimodal_component_planner import (
    MultimodalComponentPlanner,
    PlannerConfig,
)
from primitive_manifold_planner.planners.semantic_templates import (
    build_support_transfer_goal_semantic_model,
)
from primitive_manifold_planner.projection import project_newton


@dataclass
class RobotTransferCase:
    start_state: LeafState
    goal_family_name: str
    goal_lam: float
    goal_q: np.ndarray
    title: str


def project_to_leaf(manifold, x0: np.ndarray) -> np.ndarray:
    result = project_newton(
        manifold=manifold,
        x0=np.asarray(x0, dtype=float),
        tol=1e-10,
        max_iters=60,
        damping=1.0,
    )
    if not result.success:
        raise RuntimeError(f"Projection to leaf failed from seed {np.round(x0, 4)}.")
    return np.asarray(result.x_projected, dtype=float)


def build_robot_scene():
    robot = Planar3LinkKinematics(l1=1.05, l2=0.85, l3=0.65)

    left_support = EndEffectorCircleFamily3Link(
        name="left_support_circle",
        robot=robot,
        center_xy=np.array([-1.05, -0.35], dtype=float),
        radii=[0.60],
    )
    transfer = EndEffectorEllipseFamily3Link(
        name="transfer_ellipse",
        robot=robot,
        center_xy=np.array([0.0, -0.05], dtype=float),
        a_scales={1.0: 1.55},
        b_scales={1.0: 0.55},
    )
    right_support = EndEffectorCircleFamily3Link(
        name="right_support_circle",
        robot=robot,
        center_xy=np.array([1.05, -0.35], dtype=float),
        radii=[0.60],
    )

    # Chosen away from the support/transfer intersection heights so the example
    # clearly contains nontrivial motion on the support manifolds.
    start_seed = np.array([-2.55, 1.35, 0.95], dtype=float)
    goal_seed = np.array([0.55, -1.65, -0.95], dtype=float)

    start_q = wrap_q(project_to_leaf(left_support.manifold(0.60), start_seed))
    goal_q = wrap_q(project_to_leaf(right_support.manifold(0.60), goal_seed))

    case = RobotTransferCase(
        start_state=LeafState("left_support_circle", 0.60, start_q.copy()),
        goal_family_name="right_support_circle",
        goal_lam=0.60,
        goal_q=goal_q.copy(),
        title="Planar 3-link robot on implicit support-transfer-support manifolds",
    )

    families = [left_support, transfer, right_support]
    return robot, families, case


def sample_leaf_points(family, lam: float, n_random: int = 120, max_keep: int = 60):
    manifold = family.manifold(lam)
    rng = np.random.default_rng(17 + int(round(100 * float(lam))))
    seeds = [np.asarray(q, dtype=float).copy() for q in family.transition_seed_anchors(lam)]
    for _ in range(n_random):
        seeds.append(rng.uniform(-np.pi, np.pi, size=3))

    points = []
    for seed in seeds:
        try:
            proj = project_newton(
                manifold=manifold,
                x0=np.asarray(seed, dtype=float),
                tol=1e-10,
                max_iters=60,
                damping=1.0,
            )
        except Exception:
            continue
        if not proj.success:
            continue
        q = wrap_q(np.asarray(proj.x_projected, dtype=float))
        if any(torus_distance(q, p) < 0.16 for p in points):
            continue
        points.append(q)
        if len(points) >= max_keep:
            break
    if not points:
        return np.zeros((0, 3))
    return np.asarray(points)


def build_component_registry(families):
    registry, discoveries = build_component_model_registry(
        families=families,
        seed_samples_for_leaf_fn=lambda fam, lam: sample_leaf_points(fam, lam),
        should_discover_fn=lambda fam, lam: True,
        assignment_distance_fn=torus_distance,
        local_planner_name="ompl_projection",
        local_planner_kwargs=dict(
            goal_tol=1e-3,
            max_iters=700,
            projection_tol=1e-10,
            projection_max_iters=60,
            solve_time=0.20,
            ompl_planner_name="RRTConnect",
            bounds_min=np.array([-np.pi, -np.pi, -np.pi], dtype=float),
            bounds_max=np.array([np.pi, np.pi, np.pi], dtype=float),
        ),
        step_size=0.12,
        neighbor_radius=0.45,
    )
    return registry, discoveries


def build_semantic_model():
    def semantic_feasibility(context: PlanningSemanticContext) -> bool:
        if context.point is None:
            return True
        q = np.asarray(context.point, dtype=float)
        involved = {context.source_family_name, context.target_family_name}
        if "transfer_ellipse" in involved:
            return bool(-2.75 <= float(q[0]) <= 2.75 and -1.8 <= float(q[1]) <= 1.8)
        return True

    def semantic_admissibility(context: PlanningSemanticContext) -> float:
        if context.point is None:
            return 0.0
        q = np.asarray(context.point, dtype=float)
        goal = None if context.goal_point is None else np.asarray(context.goal_point, dtype=float)
        involved = {context.source_family_name, context.target_family_name}
        cost = 0.0
        if "transfer_ellipse" in involved:
            cost += 0.10 * abs(float(q[2]))
        if goal is not None:
            cost += 0.03 * float(np.linalg.norm(q - goal))
        return float(cost)

    return build_support_transfer_goal_semantic_model(
        support_families=["left_support_circle"],
        transfer_families=["transfer_ellipse"],
        goal_support_families=["right_support_circle"],
        transition_feasibility_fn=semantic_feasibility,
        transition_admissibility_fn=semantic_admissibility,
    )


def seed_points_fn(source_family, source_lam, target_family, target_lam):
    seeds = []
    seeds.extend(np.asarray(q, dtype=float).copy() for q in source_family.transition_seed_anchors(source_lam))
    seeds.extend(np.asarray(q, dtype=float).copy() for q in target_family.transition_seed_anchors(target_lam))
    return seeds


def concatenate_realization(realization):
    pts = []
    for i, step in enumerate(realization.steps):
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


def total_path_length(path: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def plot_constraint_geometry(ax, robot: Planar3LinkKinematics, q_path: np.ndarray, result):
    support_color = "#c58b4c"
    transfer_color = "#88a8c6"

    reach = robot.l1 + robot.l2 + robot.l3 + 0.15
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    left_center = np.array([-1.05, -0.35], dtype=float)
    right_center = np.array([1.05, -0.35], dtype=float)
    support_radius = 0.60

    ax.plot(
        left_center[0] + support_radius * np.cos(theta),
        left_center[1] + support_radius * np.sin(theta),
        color=support_color,
        linewidth=4.0,
        alpha=0.90,
        label="left support: ||p_ee(q)-c_L|| = 0.60",
    )
    ax.plot(
        1.55 * np.cos(theta),
        -0.05 + 0.55 * np.sin(theta),
        color=transfer_color,
        linewidth=4.0,
        alpha=0.90,
        label="transfer: ((x/a)^2 + ((y+0.05)/b)^2 = 1)",
    )
    ax.plot(
        right_center[0] + support_radius * np.cos(theta),
        right_center[1] + support_radius * np.sin(theta),
        color=support_color,
        linewidth=4.0,
        alpha=0.90,
        label="right support: ||p_ee(q)-c_R|| = 0.60",
    )

    if len(q_path) > 0:
        ee_path = np.asarray([robot.fk(q) for q in q_path], dtype=float)
        ax.plot(ee_path[:, 0], ee_path[:, 1], color="#1565c0", linewidth=2.8, label="end-effector path")
        ax.scatter(ee_path[0, 0], ee_path[0, 1], s=80, marker="s", color="black", label="start")
        ax.scatter(ee_path[-1, 0], ee_path[-1, 1], s=130, marker="*", color="gold", edgecolor="black", label="goal")

    if result.realization is not None:
        for step in result.realization.steps:
            if step.transition_point is None:
                continue
            ee = robot.fk(step.transition_point)
            ax.scatter(ee[0], ee[1], color="black", marker="x", s=55)

    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Defined Constraint Geometry")


def plot_workspace_scene(ax, robot: Planar3LinkKinematics, q_path: np.ndarray, result):
    ee_path = np.zeros((0, 2))
    if len(q_path) > 0:
        ee_path = np.asarray([robot.fk(q) for q in q_path], dtype=float)
        ax.plot(ee_path[:, 0], ee_path[:, 1], color="#1565c0", linewidth=2.8, label="end-effector path")

        snapshot_ids = np.linspace(0, len(q_path) - 1, min(8, len(q_path)), dtype=int)
        for idx in snapshot_ids:
            joints = robot.joint_positions(q_path[idx])
            ax.plot(joints[:, 0], joints[:, 1], color="#5e81ac", alpha=0.28, linewidth=2.0)
            ax.scatter(joints[:, 0], joints[:, 1], color="#5e81ac", alpha=0.28, s=14)

    if result.realization is not None:
        for step in result.realization.steps:
            if step.transition_point is None:
                continue
            ee = robot.fk(step.transition_point)
            ax.scatter(ee[0], ee[1], color="black", marker="x", s=55)

    # Always show start/goal robot configurations, even if realization fails.
    if len(q_path) > 0:
        start_q = q_path[0]
        goal_q = q_path[-1]
    else:
        start_q = getattr(result, "start_state", None)
        goal_q = getattr(result, "goal_state", None)
        start_q = None if start_q is None else np.asarray(start_q.x, dtype=float)
        goal_q = None if goal_q is None else np.asarray(goal_q.x, dtype=float)

    if start_q is not None:
        start_joints = robot.joint_positions(start_q)
        ax.plot(start_joints[:, 0], start_joints[:, 1], color="#2e3440", linewidth=2.4, alpha=0.85)
        ax.scatter(start_joints[:, 0], start_joints[:, 1], color="#2e3440", s=18, alpha=0.85)
        start_ee = start_joints[-1]
        ax.scatter(start_ee[0], start_ee[1], s=80, marker="s", color="black", label="start")

    if goal_q is not None:
        goal_joints = robot.joint_positions(goal_q)
        ax.plot(goal_joints[:, 0], goal_joints[:, 1], color="#6b7280", linewidth=2.2, alpha=0.7)
        ax.scatter(goal_joints[:, 0], goal_joints[:, 1], color="#6b7280", s=18, alpha=0.7)
        goal_ee = goal_joints[-1]
        ax.scatter(goal_ee[0], goal_ee[1], s=130, marker="*", color="gold", edgecolor="black", label="goal")

    reach = robot.l1 + robot.l2 + robot.l3 + 0.15
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Robot Workspace")


def plot_joint_discoveries(ax, discoveries):
    palette = ["#c58b4c", "#88a8c6", "#8fbc8f", "#bf616a", "#b48ead", "#d08770"]
    first = True
    for (family_name, lam), discovery in discoveries.items():
        for comp in discovery.components:
            pts = np.asarray(comp.samples, dtype=float)
            if len(pts) == 0:
                continue
            color = palette[comp.component_id % len(palette)]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                s=16,
                alpha=0.38,
                color=color,
                label=f"{family_name}[{lam}] comp {comp.component_id}" if first else None,
            )
            first = False


def plot_joint_path(ax, q_path: np.ndarray, result, start_q: np.ndarray, goal_q: np.ndarray, discoveries):
    plot_joint_discoveries(ax, discoveries)

    if len(q_path) > 0:
        ax.plot(q_path[:, 0], q_path[:, 1], q_path[:, 2], color="#1565c0", linewidth=3.0, label="realized path")

    ax.scatter(start_q[0], start_q[1], start_q[2], s=80, marker="s", color="black", label="start")
    ax.scatter(goal_q[0], goal_q[1], goal_q[2], s=125, marker="*", color="gold", edgecolor="black", label="goal")

    if result.realization is not None:
        for step in result.realization.steps:
            if step.nominal_transition_point is not None:
                q_nom = np.asarray(step.nominal_transition_point, dtype=float)
                ax.scatter(
                    q_nom[0], q_nom[1], q_nom[2],
                    s=85,
                    marker="o",
                    facecolors="none",
                    edgecolors="#d62728",
                    linewidths=1.5,
                    label="nominal transition",
                )
            if step.transition_point is not None:
                q_real = np.asarray(step.transition_point, dtype=float)
                ax.scatter(
                    q_real[0], q_real[1], q_real[2],
                    s=55,
                    marker="x",
                    color="black",
                    label="realized transition",
                )

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_zlim(-np.pi, np.pi)
    ax.set_xlabel("q1")
    ax.set_ylabel("q2")
    ax.set_zlabel("q3")
    ax.set_title("Joint-Space Path")
    ax.grid(True, alpha=0.35)
    ax.view_init(elev=24, azim=-56)
    ax.text2D(
        0.02,
        0.98,
        (
            "Implicit manifolds in joint space:\n"
            "circle and ellipse constraints in workspace induce\n"
            "nontrivial implicit manifolds in joint space.\n"
            "Their exact shape in q-space is not explicitly prescribed."
        ),
        transform=ax.transAxes,
        fontsize=8.2,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.86, edgecolor="#bbbbbb"),
    )


def main():
    np.random.seed(37)

    robot, families, case = build_robot_scene()
    registry, discoveries = build_component_registry(families)
    semantic_model = build_semantic_model()

    planner = MultimodalComponentPlanner(
        families=families,
        project_newton=project_newton,
        seed_points_fn=seed_points_fn,
        component_model_registry=registry,
        config=PlannerConfig(
            semantic_model=semantic_model,
            wrap_state_fn=wrap_q,
            state_distance_fn=torus_distance,
            local_planner_name="ompl_projection",
            local_planner_kwargs=dict(
                goal_tol=1e-3,
                max_iters=1000,
                projection_tol=1e-10,
                projection_max_iters=60,
                solve_time=0.24,
                ompl_planner_name="RRTConnect",
                bounds_min=np.array([-np.pi, -np.pi, -np.pi], dtype=float),
                bounds_max=np.array([np.pi, np.pi, np.pi], dtype=float),
            ),
            step_size=0.10,
            max_candidates_per_pair=10,
        ),
    )

    result = planner.plan_with_inferred_components(
        start_state=case.start_state,
        goal_family_name=case.goal_family_name,
        goal_lam=case.goal_lam,
        goal_q=case.goal_q.copy(),
    )
    start_comp = planner.infer_component(case.start_state.family_name, case.start_state.lam, case.start_state.x)
    goal_comp = planner.infer_component(case.goal_family_name, case.goal_lam, case.goal_q)

    start_ee = robot.fk(case.start_state.x)
    goal_ee = robot.fk(case.goal_q)
    theta = np.linspace(0.0, 2.0 * np.pi, 2000)
    left_curve = np.column_stack(
        [
            -1.05 + 0.60 * np.cos(theta),
            -0.35 + 0.60 * np.sin(theta),
        ]
    )
    transfer_curve = np.column_stack(
        [
            1.55 * np.cos(theta),
            -0.05 + 0.55 * np.sin(theta),
        ]
    )
    right_curve = np.column_stack(
        [
            1.05 + 0.60 * np.cos(theta),
            -0.35 + 0.60 * np.sin(theta),
        ]
    )
    left_intersections = left_curve[
        np.min(np.linalg.norm(left_curve[:, None, :] - transfer_curve[None, :, :], axis=2), axis=1) < 0.03
    ]
    right_intersections = right_curve[
        np.min(np.linalg.norm(right_curve[:, None, :] - transfer_curve[None, :, :], axis=2), axis=1) < 0.03
    ]
    left_transition_dist = float(np.min(np.linalg.norm(left_intersections - start_ee, axis=1))) if len(left_intersections) > 0 else float("nan")
    right_transition_dist = float(np.min(np.linalg.norm(right_intersections - goal_ee, axis=1))) if len(right_intersections) > 0 else float("nan")

    print(f"\nRunning {case.title}")
    print(f"OMPL backend = {og.RRTConnect.__name__} on ProjectedStateSpace (delta={ob.CONSTRAINED_STATE_SPACE_DELTA})")
    print(f"start_q = {np.round(case.start_state.x, 4)}")
    print(f"goal_q  = {np.round(case.goal_q, 4)}")
    print(f"start_component = {start_comp}")
    print(f"goal_component  = {goal_comp}")
    print(f"start_ee = {np.round(start_ee, 4)}")
    print(f"goal_ee  = {np.round(goal_ee, 4)}")
    print(f"distance from start to nearest support-transfer intersection  = {left_transition_dist:.4f}")
    print(f"distance from goal to nearest transfer-goal intersection      = {right_transition_dist:.4f}")
    print(f"route_found = {result.route_found}")
    print(f"realization_success = {result.realization_success}")
    print(f"message = {result.message}")
    print(f"route = {result.diagnostics.selected_route_string}")
    print(f"nominal candidate indices = {result.diagnostics.selected_candidate_indices}")
    print(f"realized candidate indices = {result.diagnostics.realized_candidate_indices}")
    print(f"realized deviations = {result.diagnostics.realized_transition_deviations}")
    print(
        "discovered components = "
        f"{ {f'{name}[{lam}]': len(discovery.components) for (name, lam), discovery in discoveries.items()} }"
    )

    q_path = np.zeros((0, 3))
    if result.realization is not None and result.realization.success:
        q_path = concatenate_realization(result.realization)
        print(f"path_length = {total_path_length(q_path):.4f}")

    # Expose start/goal states to the plotting helper even on failure.
    result.start_state = case.start_state
    result.goal_state = LeafState(case.goal_family_name, case.goal_lam, case.goal_q.copy())

    fig = plt.figure(figsize=(16.5, 5.8))

    ax0 = fig.add_subplot(131)
    plot_constraint_geometry(ax0, robot=robot, q_path=q_path, result=result)
    handles0, labels0 = ax0.get_legend_handles_labels()
    unique0 = {}
    for handle, label in zip(handles0, labels0):
        if label not in unique0:
            unique0[label] = handle
    ax0.legend(unique0.values(), unique0.keys(), loc="upper left", fontsize=8)

    ax1 = fig.add_subplot(132)
    plot_workspace_scene(ax1, robot=robot, q_path=q_path, result=result)
    handles1, labels1 = ax1.get_legend_handles_labels()
    unique1 = {}
    for handle, label in zip(handles1, labels1):
        if label not in unique1:
            unique1[label] = handle
    ax1.legend(unique1.values(), unique1.keys(), loc="upper left", fontsize=8)

    ax2 = fig.add_subplot(133, projection="3d")
    plot_joint_path(
        ax2,
        q_path=q_path,
        result=result,
        start_q=case.start_state.x,
        goal_q=case.goal_q,
        discoveries=discoveries,
    )
    handles2, labels2 = ax2.get_legend_handles_labels()
    unique2 = {}
    for handle, label in zip(handles2, labels2):
        if label not in unique2:
            unique2[label] = handle
    ax2.legend(unique2.values(), unique2.keys(), loc="upper left", fontsize=8)

    plt.suptitle("Example 63: planar 3-link robot on implicit support-transfer-support manifolds")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
