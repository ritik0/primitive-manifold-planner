from __future__ import annotations

import numpy as np

from primitive_manifold_planner.examplesupport.planar2link import (
    wrap_q,
    torus_diff,
    component_of_q,
    two_link_ik_solutions,
    Planar2LinkKinematics,
)


def make_seed_points_fn(robot: Planar2LinkKinematics):
    def seed_points_fn(fam_a, lam_a, fam_b, lam_b):
        seeds = []
        vals = np.linspace(-np.pi, np.pi, 9)
        for q1 in vals:
            for q2 in vals:
                seeds.append(np.array([q1, q2], dtype=float))

        def add_xy_ik_seeds(x_target: float, y_target: float):
            sols = two_link_ik_solutions(robot, x_target, y_target)
            eps = 0.05
            for q in sols:
                seeds.append(q.copy())
                seeds.append(q + np.array([eps, 0.0]))
                seeds.append(q + np.array([-eps, 0.0]))
                seeds.append(q + np.array([0.0, eps]))
                seeds.append(q + np.array([0.0, -eps]))

        def add_q2_zero_x_seeds(x_target: float):
            if abs(x_target / 2.0) <= 1.0 + 1e-10:
                c = np.clip(x_target / 2.0, -1.0, 1.0)
                q1a = np.arccos(c)
                q1b = -np.arccos(c)
                for q1 in [q1a, q1b]:
                    q = np.array([q1, 0.0], dtype=float)
                    seeds.append(q.copy())
                    seeds.append(q + np.array([0.05, 0.02]))
                    seeds.append(q + np.array([-0.05, -0.02]))

        def add_q2_zero_y_seeds(y_target: float):
            if abs(y_target / 2.0) <= 1.0 + 1e-10:
                s = np.clip(y_target / 2.0, -1.0, 1.0)
                q1a = np.arcsin(s)
                q1b = np.pi - q1a
                for q1 in [q1a, q1b]:
                    q = np.array([q1, 0.0], dtype=float)
                    seeds.append(q.copy())
                    seeds.append(q + np.array([0.05, 0.02]))
                    seeds.append(q + np.array([-0.05, -0.02]))

        if fam_a.name.endswith("x_family") and fam_b.name.endswith("y_family"):
            add_xy_ik_seeds(float(lam_a), float(lam_b))
        if fam_b.name.endswith("x_family") and fam_a.name.endswith("y_family"):
            add_xy_ik_seeds(float(lam_b), float(lam_a))

        if fam_a.name.endswith("x_family") and fam_b.name == "switch_q2_family":
            add_q2_zero_x_seeds(float(lam_a))
        if fam_b.name.endswith("x_family") and fam_a.name == "switch_q2_family":
            add_q2_zero_x_seeds(float(lam_b))

        if fam_a.name.endswith("y_family") and fam_b.name == "switch_q2_family":
            add_q2_zero_y_seeds(float(lam_a))
        if fam_b.name.endswith("y_family") and fam_a.name == "switch_q2_family":
            add_q2_zero_y_seeds(float(lam_b))

        uniq = []
        for s in seeds:
            s = wrap_q(s)
            if not any(np.linalg.norm(torus_diff(s, t)) < 1e-8 for t in uniq):
                uniq.append(s)

        return uniq

    return seed_points_fn


def component_ids_for_family(fam, lam: float):
    if fam.name == "switch_q2_family":
        return ["switch"]
    return ["up", "down"]


def compatible_components_for_leaf(fam, lam: float, q_transition):
    q_comp = component_of_q(q_transition)
    if fam.name == "switch_q2_family":
        return ["switch"] if q_comp == "switch" else []
    if q_comp == "switch":
        return ["up", "down"]
    return [q_comp]


def allowed_unaware_pair(a: str, b: str) -> bool:
    allowed = {
        ("left_x_family", "bridge_y_family"),
        ("bridge_y_family", "left_x_family"),
        ("bridge_y_family", "right_x_family"),
        ("right_x_family", "bridge_y_family"),
    }
    return (a, b) in allowed


def allowed_switch_pair(a: str, b: str) -> bool:
    allowed = {
        ("left_x_family", "switch_q2_family"),
        ("switch_q2_family", "left_x_family"),
        ("switch_q2_family", "bridge_y_family"),
        ("bridge_y_family", "switch_q2_family"),
        ("bridge_y_family", "right_x_family"),
        ("right_x_family", "bridge_y_family"),
    }
    return (a, b) in allowed 