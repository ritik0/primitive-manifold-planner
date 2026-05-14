from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional
import numpy as np

from primitive_manifold_planner.manifolds import (
    DoubleSphereManifold,
    EllipseManifold,
    MaskedManifold,
    PlaneManifold,
    RoundedBoxManifold,
    RoundedRectangleManifold,
    SphereManifold,
)

from .base import ManifoldFamily


class DiscreteFamily(ManifoldFamily):
    """
    Simple reusable family with an explicit finite lambda set.
    """

    def __init__(self, name: str, lambdas: List[float]):
        super().__init__(name=name)
        self._lambdas = [float(v) for v in lambdas]

    def sample_lambdas(self, context=None) -> Iterable[float]:
        _ = context
        return list(self._lambdas)


class MaskedFamily(ManifoldFamily):
    """
    Wrap a base family and restrict each leaf to a valid subset.
    """

    def __init__(
        self,
        base_family: ManifoldFamily,
        validity_mask_fn: Callable[[float, np.ndarray], bool],
        name: Optional[str] = None,
    ):
        super().__init__(name=name or f"{base_family.name}_masked")
        self.base_family = base_family
        self.validity_mask_fn = validity_mask_fn

    def manifold(self, lam):
        base = self.base_family.manifold(lam)
        return MaskedManifold(
            base_manifold=base,
            validity_fn=lambda x: bool(self.validity_mask_fn(float(lam), np.asarray(x, dtype=float))),
            name=f"{self.name}_lambda_{float(lam):g}",
        )

    def sample_lambdas(self, context=None) -> Iterable:
        return self.base_family.sample_lambdas(context=context)

    def nearby_lambdas(self, lam, radius: float) -> Iterable:
        return self.base_family.nearby_lambdas(lam, radius)

    def lambda_distance(self, lam_a, lam_b) -> float:
        return float(self.base_family.lambda_distance(lam_a, lam_b))

    def transition_seed_anchors(self, lam, goal_point: np.ndarray | None = None):
        anchors_fn = getattr(self.base_family, "transition_seed_anchors", None)
        if not callable(anchors_fn):
            return []
        anchors = []
        for anchor in anchors_fn(lam, goal_point=goal_point):
            q = np.asarray(anchor, dtype=float)
            if self.validity_mask_fn(float(lam), q):
                anchors.append(q.copy())
        return anchors

    def transition_admissibility_cost(
        self,
        lam,
        point: np.ndarray,
        goal_point: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> float:
        if not self.validity_mask_fn(float(lam), np.asarray(point, dtype=float)):
            return float("inf")
        cost_fn = getattr(self.base_family, "transition_admissibility_cost", None)
        if callable(cost_fn):
            return float(cost_fn(lam, point, goal_point=goal_point, metadata=metadata))
        return 0.0

    def transition_feasibility(
        self,
        lam,
        point: np.ndarray,
        goal_point: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> bool:
        q = np.asarray(point, dtype=float)
        if not self.validity_mask_fn(float(lam), q):
            return False
        feasibility_fn = getattr(self.base_family, "transition_feasibility", None)
        if callable(feasibility_fn):
            return bool(feasibility_fn(lam, q, goal_point=goal_point, metadata=metadata))
        return True


class EllipseFamily(DiscreteFamily):
    def __init__(
        self,
        name: str,
        center: np.ndarray,
        a_scales: Dict[float, float],
        b_scales: Dict[float, float],
        admissibility_cost_fn: Optional[Callable[[float, np.ndarray, np.ndarray | None, dict | None], float]] = None,
        feasibility_fn: Optional[Callable[[float, np.ndarray, np.ndarray | None, dict | None], bool]] = None,
    ):
        super().__init__(name=name, lambdas=list(a_scales.keys()))
        self.center = np.asarray(center, dtype=float)
        self.a_scales = {float(k): float(v) for k, v in a_scales.items()}
        self.b_scales = {float(k): float(v) for k, v in b_scales.items()}
        self.admissibility_cost_fn = admissibility_cost_fn
        self.feasibility_fn = feasibility_fn

    def manifold(self, lam: float):
        lam = float(lam)
        return EllipseManifold(
            center=self.center.copy(),
            a=self.a_scales[lam],
            b=self.b_scales[lam],
            name=f"{self.name}_lambda_{lam:g}",
        )

    def transition_seed_anchors(self, lam, goal_point: np.ndarray | None = None):
        lam = float(lam)
        manifold = self.manifold(lam)
        angles = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
        anchors = [manifold.point_from_angle(theta) for theta in angles]
        if goal_point is not None:
            goal = np.asarray(goal_point, dtype=float)
            theta = np.arctan2(goal[1] - self.center[1], goal[0] - self.center[0])
            anchors.append(manifold.point_from_angle(theta))
        return anchors

    def transition_admissibility_cost(
        self,
        lam,
        point: np.ndarray,
        goal_point: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> float:
        if self.admissibility_cost_fn is None:
            return 0.0
        return float(
            self.admissibility_cost_fn(
                float(lam),
                np.asarray(point, dtype=float),
                None if goal_point is None else np.asarray(goal_point, dtype=float),
                metadata,
            )
        )

    def transition_feasibility(
        self,
        lam,
        point: np.ndarray,
        goal_point: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> bool:
        if self.feasibility_fn is None:
            return True
        return bool(
            self.feasibility_fn(
                float(lam),
                np.asarray(point, dtype=float),
                None if goal_point is None else np.asarray(goal_point, dtype=float),
                metadata,
            )
        )


class RoundedRectangleConnectorFamily(DiscreteFamily):
    def __init__(
        self,
        name: str,
        center: np.ndarray,
        a_scales: Dict[float, float],
        b_scales: Dict[float, float],
        power: float = 4.0,
        admissibility_cost_fn: Optional[Callable[[float, np.ndarray, np.ndarray | None, dict | None], float]] = None,
        feasibility_fn: Optional[Callable[[float, np.ndarray, np.ndarray | None, dict | None], bool]] = None,
    ):
        super().__init__(name=name, lambdas=list(a_scales.keys()))
        self.center = np.asarray(center, dtype=float)
        self.a_scales = {float(k): float(v) for k, v in a_scales.items()}
        self.b_scales = {float(k): float(v) for k, v in b_scales.items()}
        self.power = float(power)
        self.admissibility_cost_fn = admissibility_cost_fn
        self.feasibility_fn = feasibility_fn

    def manifold(self, lam: float):
        lam = float(lam)
        return RoundedRectangleManifold(
            center=self.center.copy(),
            a=self.a_scales[lam],
            b=self.b_scales[lam],
            power=self.power,
            name=f"{self.name}_lambda_{lam:g}",
        )

    def transition_seed_anchors(self, lam, goal_point: np.ndarray | None = None):
        lam = float(lam)
        manifold = self.manifold(lam)
        anchors = [
            manifold.point_from_angle(0.0),
            manifold.point_from_angle(0.5 * np.pi),
            manifold.point_from_angle(np.pi),
            manifold.point_from_angle(1.5 * np.pi),
        ]
        if goal_point is not None:
            goal = np.asarray(goal_point, dtype=float)
            direction = goal - self.center
            if np.linalg.norm(direction) > 1e-12:
                theta = np.arctan2(direction[1], direction[0])
                anchors.append(manifold.point_from_angle(theta))
        return anchors

    def transition_admissibility_cost(
        self,
        lam,
        point: np.ndarray,
        goal_point: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> float:
        if self.admissibility_cost_fn is None:
            return 0.0
        return float(
            self.admissibility_cost_fn(
                float(lam),
                np.asarray(point, dtype=float),
                None if goal_point is None else np.asarray(goal_point, dtype=float),
                metadata,
            )
        )

    def transition_feasibility(
        self,
        lam,
        point: np.ndarray,
        goal_point: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> bool:
        if self.feasibility_fn is None:
            return True
        return bool(
            self.feasibility_fn(
                float(lam),
                np.asarray(point, dtype=float),
                None if goal_point is None else np.asarray(goal_point, dtype=float),
                metadata,
            )
        )


class RoundedBoxFamily(DiscreteFamily):
    def __init__(
        self,
        name: str,
        center: np.ndarray,
        a_scales: Dict[float, float],
        b_scales: Dict[float, float],
        c_scales: Dict[float, float],
        power: float = 4.0,
        admissibility_cost_fn: Optional[Callable[[float, np.ndarray, np.ndarray | None, dict | None], float]] = None,
        feasibility_fn: Optional[Callable[[float, np.ndarray, np.ndarray | None, dict | None], bool]] = None,
    ):
        super().__init__(name=name, lambdas=list(a_scales.keys()))
        self.center = np.asarray(center, dtype=float)
        self.a_scales = {float(k): float(v) for k, v in a_scales.items()}
        self.b_scales = {float(k): float(v) for k, v in b_scales.items()}
        self.c_scales = {float(k): float(v) for k, v in c_scales.items()}
        self.power = float(power)
        self.admissibility_cost_fn = admissibility_cost_fn
        self.feasibility_fn = feasibility_fn

    def manifold(self, lam: float):
        lam = float(lam)
        return RoundedBoxManifold(
            center=self.center.copy(),
            a=self.a_scales[lam],
            b=self.b_scales[lam],
            c=self.c_scales[lam],
            power=self.power,
            name=f"{self.name}_lambda_{lam:g}",
        )

    def transition_seed_anchors(self, lam, goal_point: np.ndarray | None = None):
        lam = float(lam)
        a = self.a_scales[lam]
        b = self.b_scales[lam]
        c = self.c_scales[lam]
        center = self.center
        anchors = [
            center + np.array([a, 0.0, 0.0], dtype=float),
            center + np.array([-a, 0.0, 0.0], dtype=float),
            center + np.array([0.0, b, 0.0], dtype=float),
            center + np.array([0.0, -b, 0.0], dtype=float),
            center + np.array([0.0, 0.0, c], dtype=float),
            center + np.array([0.0, 0.0, -c], dtype=float),
        ]
        if goal_point is not None:
            goal = np.asarray(goal_point, dtype=float)
            direction = goal - center
            scaled = np.array(
                [
                    direction[0] / max(a, 1e-12),
                    direction[1] / max(b, 1e-12),
                    direction[2] / max(c, 1e-12),
                ],
                dtype=float,
            )
            norm = float(np.linalg.norm(scaled))
            if norm > 1e-12:
                anchors.append(
                    center + np.array([a * scaled[0], b * scaled[1], c * scaled[2]], dtype=float) / norm
                )
        return anchors

    def transition_admissibility_cost(
        self,
        lam,
        point: np.ndarray,
        goal_point: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> float:
        if self.admissibility_cost_fn is None:
            return 0.0
        return float(
            self.admissibility_cost_fn(
                float(lam),
                np.asarray(point, dtype=float),
                None if goal_point is None else np.asarray(goal_point, dtype=float),
                metadata,
            )
        )

    def transition_feasibility(
        self,
        lam,
        point: np.ndarray,
        goal_point: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> bool:
        if self.feasibility_fn is None:
            return True
        return bool(
            self.feasibility_fn(
                float(lam),
                np.asarray(point, dtype=float),
                None if goal_point is None else np.asarray(goal_point, dtype=float),
                metadata,
            )
        )


class SphereFamily(DiscreteFamily):
    def __init__(
        self,
        name: str,
        center: np.ndarray,
        radii: Dict[float, float],
        admissibility_cost_fn: Optional[Callable[[float, np.ndarray, np.ndarray | None, dict | None], float]] = None,
        feasibility_fn: Optional[Callable[[float, np.ndarray, np.ndarray | None, dict | None], bool]] = None,
    ):
        super().__init__(name=name, lambdas=list(radii.keys()))
        self.center = np.asarray(center, dtype=float)
        self.radii = {float(k): float(v) for k, v in radii.items()}
        self.admissibility_cost_fn = admissibility_cost_fn
        self.feasibility_fn = feasibility_fn

    def manifold(self, lam: float):
        lam = float(lam)
        return SphereManifold(
            center=self.center.copy(),
            radius=self.radii[lam],
            name=f"{self.name}_lambda_{lam:g}",
        )

    def transition_seed_anchors(self, lam, goal_point: np.ndarray | None = None):
        lam = float(lam)
        radius = self.radii[lam]
        anchors = [
            self.center + radius * np.array([1.0, 0.0, 0.0]),
            self.center + radius * np.array([-1.0, 0.0, 0.0]),
            self.center + radius * np.array([0.0, 1.0, 0.0]),
            self.center + radius * np.array([0.0, -1.0, 0.0]),
            self.center + radius * np.array([0.0, 0.0, 1.0]),
            self.center + radius * np.array([0.0, 0.0, -1.0]),
        ]
        if goal_point is not None:
            goal = np.asarray(goal_point, dtype=float)
            direction = goal - self.center
            norm = float(np.linalg.norm(direction))
            if norm > 1e-12:
                anchors.append(self.center + radius * direction / norm)
        return anchors

    def transition_admissibility_cost(
        self,
        lam,
        point: np.ndarray,
        goal_point: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> float:
        if self.admissibility_cost_fn is None:
            return 0.0
        return float(
            self.admissibility_cost_fn(
                float(lam),
                np.asarray(point, dtype=float),
                None if goal_point is None else np.asarray(goal_point, dtype=float),
                metadata,
            )
        )

    def transition_feasibility(
        self,
        lam,
        point: np.ndarray,
        goal_point: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> bool:
        if self.feasibility_fn is None:
            return True
        return bool(
            self.feasibility_fn(
                float(lam),
                np.asarray(point, dtype=float),
                None if goal_point is None else np.asarray(goal_point, dtype=float),
                metadata,
            )
        )


class PlaneFamily(DiscreteFamily):
    def __init__(
        self,
        name: str,
        base_point: np.ndarray,
        normal: np.ndarray,
        offsets: List[float],
        anchor_span: float = 1.0,
        admissibility_cost_fn: Optional[Callable[[float, np.ndarray, np.ndarray | None, dict | None], float]] = None,
        feasibility_fn: Optional[Callable[[float, np.ndarray, np.ndarray | None, dict | None], bool]] = None,
    ):
        super().__init__(name=name, lambdas=offsets)
        self.base_point = np.asarray(base_point, dtype=float)
        self.normal = np.asarray(normal, dtype=float)
        self.normal = self.normal / max(np.linalg.norm(self.normal), 1e-15)
        self.anchor_span = float(anchor_span)
        self.admissibility_cost_fn = admissibility_cost_fn
        self.feasibility_fn = feasibility_fn
        self._basis_u, self._basis_v = _plane_basis(self.normal)

    def manifold(self, lam: float):
        lam = float(lam)
        point = self.base_point + lam * self.normal
        return PlaneManifold(
            point=point,
            normal=self.normal.copy(),
            name=f"{self.name}_lambda_{lam:g}",
        )

    def transition_seed_anchors(self, lam, goal_point: np.ndarray | None = None):
        lam = float(lam)
        point = self.base_point + lam * self.normal
        anchors = [
            point.copy(),
            point + self.anchor_span * self._basis_u,
            point - self.anchor_span * self._basis_u,
            point + self.anchor_span * self._basis_v,
            point - self.anchor_span * self._basis_v,
        ]
        if goal_point is not None:
            goal = np.asarray(goal_point, dtype=float)
            signed = float(np.dot(self.normal, goal - point))
            anchors.append(goal - signed * self.normal)
        return anchors

    def transition_admissibility_cost(
        self,
        lam,
        point: np.ndarray,
        goal_point: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> float:
        if self.admissibility_cost_fn is None:
            return 0.0
        return float(
            self.admissibility_cost_fn(
                float(lam),
                np.asarray(point, dtype=float),
                None if goal_point is None else np.asarray(goal_point, dtype=float),
                metadata,
            )
        )

    def transition_feasibility(
        self,
        lam,
        point: np.ndarray,
        goal_point: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> bool:
        if self.feasibility_fn is None:
            return True
        return bool(
            self.feasibility_fn(
                float(lam),
                np.asarray(point, dtype=float),
                None if goal_point is None else np.asarray(goal_point, dtype=float),
                metadata,
            )
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


class DoubleSphereFamily(DiscreteFamily):
    def __init__(
        self,
        name: str,
        center_a: np.ndarray,
        center_b: np.ndarray,
        radii: Dict[float, float],
        admissibility_cost_fn: Optional[Callable[[float, np.ndarray, np.ndarray | None, dict | None], float]] = None,
        feasibility_fn: Optional[Callable[[float, np.ndarray, np.ndarray | None, dict | None], bool]] = None,
    ):
        super().__init__(name=name, lambdas=list(radii.keys()))
        self.center_a = np.asarray(center_a, dtype=float)
        self.center_b = np.asarray(center_b, dtype=float)
        self.radii = {float(k): float(v) for k, v in radii.items()}
        self.admissibility_cost_fn = admissibility_cost_fn
        self.feasibility_fn = feasibility_fn

    def manifold(self, lam: float):
        lam = float(lam)
        return DoubleSphereManifold(
            center_a=self.center_a.copy(),
            center_b=self.center_b.copy(),
            radius=self.radii[lam],
            name=f"{self.name}_lambda_{lam:g}",
        )

    def transition_seed_anchors(self, lam, goal_point: np.ndarray | None = None):
        lam = float(lam)
        radius = self.radii[lam]
        anchors = [
            self.center_a + radius * np.array([1.0, 0.0, 0.0]),
            self.center_a + radius * np.array([-1.0, 0.0, 0.0]),
            self.center_a + radius * np.array([0.0, 0.0, 1.0]),
            self.center_b + radius * np.array([1.0, 0.0, 0.0]),
            self.center_b + radius * np.array([-1.0, 0.0, 0.0]),
            self.center_b + radius * np.array([0.0, 0.0, 1.0]),
        ]
        if goal_point is not None:
            goal = np.asarray(goal_point, dtype=float)
            for center in (self.center_a, self.center_b):
                direction = goal - center
                norm = float(np.linalg.norm(direction))
                if norm > 1e-12:
                    anchors.append(center + radius * direction / norm)
        return anchors

    def transition_admissibility_cost(
        self,
        lam,
        point: np.ndarray,
        goal_point: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> float:
        if self.admissibility_cost_fn is None:
            return 0.0
        return float(
            self.admissibility_cost_fn(
                float(lam),
                np.asarray(point, dtype=float),
                None if goal_point is None else np.asarray(goal_point, dtype=float),
                metadata,
            )
        )

    def transition_feasibility(
        self,
        lam,
        point: np.ndarray,
        goal_point: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> bool:
        if self.feasibility_fn is None:
            return True
        return bool(
            self.feasibility_fn(
                float(lam),
                np.asarray(point, dtype=float),
                None if goal_point is None else np.asarray(goal_point, dtype=float),
                metadata,
            )
        )
