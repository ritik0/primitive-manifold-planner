from __future__ import annotations

import numpy as np


class Planar2LinkKinematics:
    def __init__(self, l1: float = 1.0, l2: float = 1.0):
        self.l1 = float(l1)
        self.l2 = float(l2)

    def fk(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        q1, q2 = q
        x = self.l1 * np.cos(q1) + self.l2 * np.cos(q1 + q2)
        y = self.l1 * np.sin(q1) + self.l2 * np.sin(q1 + q2)
        return np.array([x, y], dtype=float)

    def jacobian_xy(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        q1, q2 = q
        dx_dq1 = -self.l1 * np.sin(q1) - self.l2 * np.sin(q1 + q2)
        dx_dq2 = -self.l2 * np.sin(q1 + q2)
        dy_dq1 = self.l1 * np.cos(q1) + self.l2 * np.cos(q1 + q2)
        dy_dq2 = self.l2 * np.cos(q1 + q2)
        return np.array(
            [
                [dx_dq1, dx_dq2],
                [dy_dq1, dy_dq2],
            ],
            dtype=float,
        )


def wrap_to_pi(angle: float) -> float:
    return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi


def wrap_q(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).copy()
    q[0] = wrap_to_pi(q[0])
    q[1] = wrap_to_pi(q[1])
    return q


def torus_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    d[0] = wrap_to_pi(d[0])
    d[1] = wrap_to_pi(d[1])
    return d


def torus_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(torus_diff(a, b)))


def component_of_q(q: np.ndarray) -> str:
    q = np.asarray(q, dtype=float)
    q2 = float(wrap_to_pi(q[1]))
    if abs(q2) < 1e-6:
        return "switch"
    return "up" if q2 > 0.0 else "down"


def two_link_ik_solutions(robot: Planar2LinkKinematics, x: float, y: float):
    l1 = float(robot.l1)
    l2 = float(robot.l2)

    r2 = x * x + y * y
    cos_q2 = (r2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)

    if cos_q2 < -1.0 - 1e-10 or cos_q2 > 1.0 + 1e-10:
        return []

    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    q2_candidates = [np.arccos(cos_q2), -np.arccos(cos_q2)]

    sols = []
    for q2 in q2_candidates:
        k1 = l1 + l2 * np.cos(q2)
        k2 = l2 * np.sin(q2)
        q1 = np.arctan2(y, x) - np.arctan2(k2, k1)

        q = np.array([wrap_to_pi(q1), wrap_to_pi(q2)], dtype=float)

        duplicate = False
        for s in sols:
            if np.linalg.norm(torus_diff(q, s)) < 1e-8:
                duplicate = True
                break
        if not duplicate:
            sols.append(q)

    return sols