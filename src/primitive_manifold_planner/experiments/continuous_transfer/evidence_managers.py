"""Leaf-aware evidence store helpers for continuous-transfer planning.

These utilities provide a thin bridge from continuous manifold families to the
stage-local evidence stores introduced in Example 66. They are intentionally
incremental so Example 65 can adopt them in phases without destabilizing the
existing staged pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from primitive_manifold_planner.projection import project_newton

from .config import LAMBDA_SOURCE_TOL

import example_66_multimodal_graph_search as ex66


def _deduplicate_points(points: list[np.ndarray], tol: float = 1e-9) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 3), dtype=float)
    unique: list[np.ndarray] = []
    for point in points:
        qq = np.asarray(point, dtype=float).reshape(-1)
        if all(float(np.linalg.norm(qq - existing)) > tol for existing in unique):
            unique.append(qq.copy())
    return np.asarray(unique, dtype=float)


@dataclass
class FamilyEvidenceManager:
    """Unify fixed-manifold and foliated-manifold evidence ownership."""

    family: object
    name: str
    is_foliation: bool = False
    manager: "LeafStoreManager | None" = None
    manifold: object | None = None
    store: ex66.StageEvidenceStore | None = None

    def __init__(self, family, name: str, is_foliation: bool = False, **kwargs: Any) -> None:
        self.family = family
        self.name = str(name)
        self.is_foliation = bool(is_foliation)
        if self.is_foliation:
            self.manager = LeafStoreManager(family=family, family_name=self.name, **kwargs)
            self.manifold = None
            self.store = None
            return

        self.manager = None
        self.manifold = _build_family_manifold(family, None, kwargs)
        self.store = ex66.StageEvidenceStore(stage=self.name, manifold=self.manifold)

    def get_all_stores(self) -> list[ex66.StageEvidenceStore]:
        if self.is_foliation:
            return [] if self.manager is None else self.manager.all_stores()
        return [] if self.store is None else [self.store]


def _build_family_manifold(family, lam: float | None, manifold_base_kwargs: dict[str, Any]):
    kwargs = dict(manifold_base_kwargs)
    if lam is None:
        lam = kwargs.pop("lam", kwargs.pop("lambda_value", kwargs.pop("radius", None)))
    else:
        kwargs.pop("lam", None)
        kwargs.pop("lambda_value", None)
        kwargs.pop("radius", None)
    if lam is None:
        sample_lambdas = getattr(family, "sample_lambdas", None)
        if callable(sample_lambdas):
            sampled = list(sample_lambdas())
            if len(sampled) > 0:
                lam = float(sampled[0])
    if lam is None:
        return family.manifold(**kwargs)
    return family.manifold(float(lam), **kwargs)


class LeafStoreManager:
    """Map lambda values to lambda-locked Example 66 evidence stores."""

    def __init__(self, family, family_name: str, **manifold_base_kwargs: Any) -> None:
        self.family = family
        self.family_name = str(family_name)
        self.manifold_base_kwargs = dict(manifold_base_kwargs)
        self.lambda_tol = float(self.manifold_base_kwargs.pop("lambda_tol", LAMBDA_SOURCE_TOL))
        self.stores: dict[float, ex66.StageEvidenceStore] = {}

    def nearest_lambda(self, lam: float) -> float | None:
        if len(self.stores) == 0:
            return None
        target = float(lam)
        return min(self.stores, key=lambda existing: abs(float(existing) - target))

    def _canonical_lambda(self, lam: float) -> float:
        nearest = self.nearest_lambda(float(lam))
        if nearest is not None and abs(float(nearest) - float(lam)) <= self.lambda_tol:
            return float(nearest)
        return float(lam)

    def _make_manifold(self, lam: float):
        return _build_family_manifold(self.family, float(lam), self.manifold_base_kwargs)

    def _project_points_to_leaf(self, lam: float, points: np.ndarray) -> np.ndarray:
        arr = np.asarray(points, dtype=float)
        if len(arr) == 0:
            return np.zeros((0, 3), dtype=float)
        manifold = self._make_manifold(float(lam))
        projected: list[np.ndarray] = []
        for point in arr:
            projection = project_newton(
                manifold=manifold,
                x0=np.asarray(point, dtype=float),
                tol=1e-10,
                max_iters=60,
                damping=1.0,
            )
            if projection.success:
                projected.append(np.asarray(projection.x_projected, dtype=float).reshape(-1))
        return _deduplicate_points(projected)

    def get_or_create_store(
        self,
        lam: float,
        seed_q: np.ndarray,
        guide: np.ndarray,
    ) -> tuple[ex66.StageEvidenceStore, bool]:
        lambda_value = self._canonical_lambda(float(lam))
        guide_point = np.asarray(guide, dtype=float).reshape(-1)
        seed = np.asarray(seed_q, dtype=float).reshape(-1)
        if lambda_value in self.stores:
            store = self.stores[lambda_value]
            node_id = ex66.add_stage_node(store, seed, seeded_from_proposal=True)
            ex66.update_stage_frontier(store, [int(node_id)], guide_point)
            return store, False

        store = ex66.StageEvidenceStore(stage=self.family_name, manifold=self._make_manifold(lambda_value))
        nearest = self.nearest_lambda(lambda_value)
        warped_centers: list[np.ndarray] = []
        if nearest is not None:
            source_store = self.stores[nearest]
            projected_centers = self._project_points_to_leaf(lambda_value, source_store.chart_centers)
            for center in projected_centers:
                center_id = ex66.add_stage_node(store, center, seeded_from_proposal=True)
                ex66.update_stage_frontier(store, [int(center_id)], guide_point)
                warped_centers.append(np.asarray(center, dtype=float).reshape(-1))
        node_id = ex66.add_stage_node(store, seed, seeded_from_proposal=True)
        ex66.update_stage_frontier(store, [int(node_id)], guide_point)
        if len(warped_centers) > 0:
            store.chart_centers = _deduplicate_points(warped_centers)
        self.stores[lambda_value] = store
        return store, True

    def representative_store(self) -> ex66.StageEvidenceStore:
        if len(self.stores) > 0:
            lam = sorted(self.stores)[0]
            return self.stores[lam]
        sample_lambdas = getattr(self.family, "sample_lambdas", None)
        lam = 0.0
        if callable(sample_lambdas):
            sampled = list(sample_lambdas())
            if len(sampled) > 0:
                lam = float(sampled[0])
        return ex66.StageEvidenceStore(stage=self.family_name, manifold=self._make_manifold(float(lam)))

    def all_stores(self) -> list[ex66.StageEvidenceStore]:
        return [self.stores[lam] for lam in sorted(self.stores)]
