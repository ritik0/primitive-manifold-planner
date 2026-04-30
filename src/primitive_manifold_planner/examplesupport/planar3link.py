from __future__ import annotations

import numpy as np


class Planar3LinkKinematics:
    def __init__(self, l1: float = 1.0, l2: float = 0.9, l3: float = 0.7):
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.l3 = float(l3)

    def fk(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        q1, q2, q3 = q[:3]
        a1 = q1
        a2 = q1 + q2
        a3 = q1 + q2 + q3
        x = self.l1 * np.cos(a1) + self.l2 * np.cos(a2) + self.l3 * np.cos(a3)
        y = self.l1 * np.sin(a1) + self.l2 * np.sin(a2) + self.l3 * np.sin(a3)
        return np.array([x, y], dtype=float)

    def joint_positions(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        q1, q2, q3 = q[:3]
        a1 = q1
        a2 = q1 + q2
        a3 = q1 + q2 + q3

        p0 = np.array([0.0, 0.0], dtype=float)
        p1 = np.array([self.l1 * np.cos(a1), self.l1 * np.sin(a1)], dtype=float)
        p2 = p1 + np.array([self.l2 * np.cos(a2), self.l2 * np.sin(a2)], dtype=float)
        p3 = p2 + np.array([self.l3 * np.cos(a3), self.l3 * np.sin(a3)], dtype=float)
        return np.vstack([p0, p1, p2, p3])

    def jacobian_xy(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        q1, q2, q3 = q[:3]
        a1 = q1
        a2 = q1 + q2
        a3 = q1 + q2 + q3

        dx_dq1 = -self.l1 * np.sin(a1) - self.l2 * np.sin(a2) - self.l3 * np.sin(a3)
        dx_dq2 = -self.l2 * np.sin(a2) - self.l3 * np.sin(a3)
        dx_dq3 = -self.l3 * np.sin(a3)

        dy_dq1 = self.l1 * np.cos(a1) + self.l2 * np.cos(a2) + self.l3 * np.cos(a3)
        dy_dq2 = self.l2 * np.cos(a2) + self.l3 * np.cos(a3)
        dy_dq3 = self.l3 * np.cos(a3)

        return np.array(
            [
                [dx_dq1, dx_dq2, dx_dq3],
                [dy_dq1, dy_dq2, dy_dq3],
            ],
            dtype=float,
        )


def wrap_to_pi(angle: float) -> float:
    return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi


def wrap_q(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).copy().reshape(-1)
    q[:3] = np.array([wrap_to_pi(q[0]), wrap_to_pi(q[1]), wrap_to_pi(q[2])], dtype=float)
    return q


def torus_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    d = a - b
    d[:3] = np.array([wrap_to_pi(d[0]), wrap_to_pi(d[1]), wrap_to_pi(d[2])], dtype=float)
    return d


def torus_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(torus_diff(a, b)))


def ik_position_solutions(
    robot: Planar3LinkKinematics,
    target_xy: np.ndarray,
    seeds: list[np.ndarray] | None = None,
    tol: float = 1e-6,
    max_iters: int = 60,
    damping: float = 1e-3,
) -> list[np.ndarray]:
    target_xy = np.asarray(target_xy, dtype=float).reshape(-1)
    if target_xy.shape[0] != 2:
        raise ValueError("target_xy must be 2D.")

    if seeds is None:
        seeds = []
        for q1 in np.linspace(-2.4, 2.4, 3):
            for q2 in np.linspace(-2.0, 2.0, 3):
                for q3 in np.linspace(-1.4, 1.4, 3):
                    seeds.append(np.array([q1, q2, q3], dtype=float))

    sols: list[np.ndarray] = []
    for seed in seeds:
        q = wrap_q(np.asarray(seed, dtype=float))
        success = False
        for _ in range(max_iters):
            err = robot.fk(q) - target_xy
            if np.linalg.norm(err) <= tol:
                success = True
                break
            J = robot.jacobian_xy(q)
            JJt = J @ J.T + (damping**2) * np.eye(2)
            step = J.T @ np.linalg.solve(JJt, err)
            q = wrap_q(q - step)
        if not success:
            continue
        duplicate = False
        for s in sols:
            if torus_distance(q, s) < 1e-3:
                duplicate = True
                break
        if not duplicate:
            sols.append(q.copy())
    return sols
