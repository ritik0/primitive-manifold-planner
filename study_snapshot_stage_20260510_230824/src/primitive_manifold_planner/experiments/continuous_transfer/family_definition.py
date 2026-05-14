"""Family geometry, obstacle profiles, and scene construction for the continuous-transfer experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import numpy as np

from primitive_manifold_planner.families.base import ManifoldFamily
from primitive_manifold_planner.families.standard import SphereFamily
from primitive_manifold_planner.manifolds import MaskedManifold, PlaneManifold

from .support import sphere_point

if TYPE_CHECKING:
    from .adapters import FamilyLeafManifoldAdapter, FamilyLeafValidityRegion, SphereManifoldAdapter


@dataclass(frozen=True)
class FamilyObstacle:
    """A simple blocked local region on the foliated family, used for falsification experiments."""

    name: str
    lambda_min: float
    lambda_max: float
    u_min: float
    u_max: float
    v_min: float
    v_max: float

    def contains(self, lam: float, u_coord: float, v_coord: float, tol: float = 1e-9) -> bool:
        return (
            float(self.lambda_min) - tol <= float(lam) <= float(self.lambda_max) + tol
            and float(self.u_min) - tol <= float(u_coord) <= float(self.u_max) + tol
            and float(self.v_min) - tol <= float(v_coord) <= float(self.v_max) + tol
        )

    def summary(self) -> str:
        return (
            f"{self.name}: lambda=[{self.lambda_min:.2f}, {self.lambda_max:.2f}], "
            f"u=[{self.u_min:.2f}, {self.u_max:.2f}], v=[{self.v_min:.2f}, {self.v_max:.2f}]"
        )


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=float)
    if abs(n[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    u = np.cross(n, ref)
    u = u / max(np.linalg.norm(u), 1e-15)
    v = np.cross(n, u)
    v = v / max(np.linalg.norm(v), 1e-15)
    return u, v


class ContinuousMaskedPlaneFamily(ManifoldFamily):
    def __init__(
        self,
        name: str,
        base_point: np.ndarray,
        normal: np.ndarray,
        lambda_min: float,
        lambda_max: float,
        half_u: float,
        half_v: float,
        nominal_lambda: float = 0.0,
        obstacles: list[FamilyObstacle] | None = None,
    ):
        super().__init__(name=name)
        self.base_point = np.asarray(base_point, dtype=float)
        self.normal = np.asarray(normal, dtype=float)
        self.normal = self.normal / max(np.linalg.norm(self.normal), 1e-15)
        self.lambda_min = float(min(lambda_min, lambda_max))
        self.lambda_max = float(max(lambda_min, lambda_max))
        self.half_u = float(half_u)
        self.half_v = float(half_v)
        self.nominal_lambda = float(nominal_lambda)
        self.obstacles = list(obstacles or [])
        self._basis_u, self._basis_v = _plane_basis(self.normal)

    def sample_lambdas(self, context=None) -> Iterable[float]:
        count = 7
        if isinstance(context, dict):
            count = max(2, int(context.get("count", count)))
        return [float(v) for v in np.linspace(self.lambda_min, self.lambda_max, count)]

    def manifold(self, lam):
        lam = float(lam)
        point = self.point_on_leaf(lam)
        base = PlaneManifold(point=point, normal=self.normal.copy(), name=f"{self.name}_lambda_{lam:g}")
        return MaskedManifold(
            base_manifold=base,
            validity_fn=lambda x, lam=lam: self.within_patch(lam, x),
            name=f"{self.name}_lambda_{lam:g}",
        )

    def point_on_leaf(self, lam: float) -> np.ndarray:
        return np.asarray(self.base_point, dtype=float) + float(lam) * np.asarray(self.normal, dtype=float)

    def infer_lambda(self, q: np.ndarray) -> float:
        qq = np.asarray(q, dtype=float)
        return float(np.dot(self.normal, qq - self.base_point))

    def lambda_in_range(self, lam: float, tol: float = 1e-9) -> bool:
        return self.lambda_min - tol <= float(lam) <= self.lambda_max + tol

    def patch_coords(self, lam: float, q: np.ndarray) -> tuple[float, float]:
        rel = np.asarray(q, dtype=float) - self.point_on_leaf(lam)
        u_coord = float(np.dot(rel, self._basis_u))
        v_coord = float(np.dot(rel, self._basis_v))
        return u_coord, v_coord

    def within_patch(self, lam: float, q: np.ndarray, tol: float = 1e-9) -> bool:
        if not self.lambda_in_range(lam, tol=tol):
            return False
        u_coord, v_coord = self.patch_coords(lam, q)
        if not (abs(u_coord) <= self.half_u + tol and abs(v_coord) <= self.half_v + tol):
            return False
        return not self.blocked_by_obstacle(lam, u_coord, v_coord, tol=tol)

    def patch_margin(self, lam: float, q: np.ndarray) -> float:
        u_coord, v_coord = self.patch_coords(lam, q)
        return float(min(self.half_u - abs(u_coord), self.half_v - abs(v_coord)))

    def blocked_by_obstacle(
        self,
        lam: float,
        u_coord: float,
        v_coord: float,
        tol: float = 1e-9,
    ) -> bool:
        return any(obstacle.contains(lam, u_coord, v_coord, tol=tol) for obstacle in self.obstacles)

    def obstacle_summaries(self) -> list[str]:
        return [obstacle.summary() for obstacle in self.obstacles]

    def leaf_adapter(self, lam: float) -> FamilyLeafManifoldAdapter:
        """Return a thin constrained-manifold adapter for one family leaf."""

        from .adapters import FamilyLeafManifoldAdapter

        return FamilyLeafManifoldAdapter.from_family(self, float(lam))

    def leaf_validity_region(self, lam: float) -> FamilyLeafValidityRegion:
        """Return a thin validity-region adapter for one family leaf patch."""

        from .adapters import FamilyLeafValidityRegion

        return FamilyLeafValidityRegion(transfer_family=self, lambda_value=float(lam))


def build_family_obstacles(profile: str, half_u: float, half_v: float) -> list[FamilyObstacle]:
    profile_name = str(profile).strip().lower()
    full_v = float(half_v)
    center_u = 0.32 * float(half_u)
    narrow_u = 0.24 * float(half_u)
    if profile_name in {"", "none", "baseline"}:
        return []
    if profile_name == "block_nominal_corridor":
        return [
            FamilyObstacle(
                name="block_nominal_corridor",
                lambda_min=-0.08,
                lambda_max=0.10,
                u_min=-center_u,
                u_max=center_u,
                v_min=-full_v,
                v_max=full_v,
            )
        ]
    if profile_name == "block_positive_corridor":
        return [
            FamilyObstacle(
                name="block_positive_corridor",
                lambda_min=0.18,
                lambda_max=0.34,
                u_min=-center_u,
                u_max=center_u,
                v_min=-full_v,
                v_max=full_v,
            )
        ]
    if profile_name == "block_negative_corridor":
        return [
            FamilyObstacle(
                name="block_negative_corridor",
                lambda_min=-0.34,
                lambda_max=-0.18,
                u_min=-center_u,
                u_max=center_u,
                v_min=-full_v,
                v_max=full_v,
            )
        ]
    if profile_name == "split_nominal_window":
        return [
            FamilyObstacle(
                name="split_nominal_window_upper",
                lambda_min=-0.06,
                lambda_max=0.12,
                u_min=-narrow_u,
                u_max=narrow_u,
                v_min=-0.35,
                v_max=full_v,
            ),
            FamilyObstacle(
                name="split_nominal_window_lower",
                lambda_min=-0.06,
                lambda_max=0.12,
                u_min=-narrow_u,
                u_max=narrow_u,
                v_min=-full_v,
                v_max=-0.95,
            ),
        ]
    raise ValueError(
        "Unknown continuous-transfer obstacle profile. "
        "Expected one of: none, block_nominal_corridor, block_positive_corridor, "
        "block_negative_corridor, split_nominal_window."
    )


def build_bounds(family_count: int) -> tuple[np.ndarray, np.ndarray]:
    if family_count <= 3:
        return (
            np.array([-3.2, -1.8, -0.6], dtype=float),
            np.array([3.2, 1.8, 1.8], dtype=float),
        )
    return (
        np.array([-6.8, -2.8, -0.9], dtype=float),
        np.array([6.8, 2.8, 2.0], dtype=float),
    )


def build_continuous_scene(obstacle_profile: str = "none"):
    left_support = SphereFamily(
        name="left_support_3d",
        center=np.array([-2.15, -0.55, 0.48], dtype=float),
        radii={1.05: 1.05},
    )
    transfer_family = ContinuousMaskedPlaneFamily(
        name="transfer_foliation_3d",
        base_point=np.array([0.0, 0.0, 0.0], dtype=float),
        normal=np.array([0.0, 0.0, 1.0], dtype=float),
        lambda_min=-0.55,
        lambda_max=0.55,
        half_u=0.95,
        half_v=2.15,
        nominal_lambda=0.0,
        obstacles=build_family_obstacles("none" if obstacle_profile is None else str(obstacle_profile), half_u=0.95, half_v=2.15),
    )
    right_support = SphereFamily(
        name="right_support_3d",
        center=np.array([2.15, 0.55, 0.48], dtype=float),
        radii={1.05: 1.05},
    )
    start_q = sphere_point(left_support.center, 1.05, azimuth_deg=0.0, elevation_deg=-90.0)
    goal_q = sphere_point(right_support.center, 1.05, azimuth_deg=0.0, elevation_deg=90.0)
    return left_support, transfer_family, right_support, start_q, goal_q


def sphere_adapter_from_family(family: SphereFamily, radius: float | None = None) -> SphereManifoldAdapter:
    """Build a thin constrained-manifold adapter for the current sphere support family."""

    from .adapters import SphereManifoldAdapter

    return SphereManifoldAdapter.from_family(family, radius=radius)


def sphere_arc_length(center: np.ndarray, radius: float, q_a: np.ndarray, q_b: np.ndarray) -> float:
    u0 = (np.asarray(q_a, dtype=float) - np.asarray(center, dtype=float)) / float(radius)
    u1 = (np.asarray(q_b, dtype=float) - np.asarray(center, dtype=float)) / float(radius)
    u0 = u0 / max(np.linalg.norm(u0), 1e-12)
    u1 = u1 / max(np.linalg.norm(u1), 1e-12)
    dot = float(np.clip(np.dot(u0, u1), -1.0, 1.0))
    return float(radius) * float(np.arccos(dot))


def plane_leaf_patch(family: ContinuousMaskedPlaneFamily, lam: float) -> np.ndarray:
    center = family.point_on_leaf(lam)
    u = np.asarray(family._basis_u, dtype=float)
    v = np.asarray(family._basis_v, dtype=float)
    return np.asarray(
        [
            center - family.half_u * u - family.half_v * v,
            center + family.half_u * u - family.half_v * v,
            center + family.half_u * u + family.half_v * v,
            center - family.half_u * u + family.half_v * v,
        ],
        dtype=float,
    )


def obstacle_patch_on_leaf(
    family: ContinuousMaskedPlaneFamily,
    obstacle: FamilyObstacle,
    lam: float,
) -> np.ndarray:
    center = family.point_on_leaf(lam)
    u = np.asarray(family._basis_u, dtype=float)
    v = np.asarray(family._basis_v, dtype=float)
    return np.asarray(
        [
            center + obstacle.u_min * u + obstacle.v_min * v,
            center + obstacle.u_max * u + obstacle.v_min * v,
            center + obstacle.u_max * u + obstacle.v_max * v,
            center + obstacle.u_min * u + obstacle.v_max * v,
        ],
        dtype=float,
    )
