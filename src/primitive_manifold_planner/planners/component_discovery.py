from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Tuple
import numpy as np

from primitive_manifold_planner.planning.local import run_local_planner


LeafKey = Tuple[str, float]


class LeafComponentModel(Protocol):
    def component_ids(self) -> List[str]:
        ...

    def compatible_components(self, q: np.ndarray) -> List[str]:
        ...

    def infer_component(self, q: np.ndarray) -> str:
        ...

    def transition_seed_anchors(self, goal_point: Optional[np.ndarray] = None) -> List[np.ndarray]:
        ...


@dataclass
class DiscoveredComponent:
    component_id: int
    samples: np.ndarray
    representative: np.ndarray
    sample_indices: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=int))


@dataclass
class LeafComponentDiscoveryResult:
    success: bool
    components: List[DiscoveredComponent]
    sample_points: np.ndarray
    labels: np.ndarray
    connectivity_edges: List[Tuple[int, int]] = field(default_factory=list)
    message: str = ""


@dataclass
class StaticComponentModel:
    ids: List[str]
    compatibility_fn: Optional[Callable[[np.ndarray], List[str]]] = None

    def component_ids(self) -> List[str]:
        return list(self.ids)

    def compatible_components(self, q: np.ndarray) -> List[str]:
        if self.compatibility_fn is None:
            return list(self.ids)
        return [str(v) for v in self.compatibility_fn(np.asarray(q, dtype=float))]

    def infer_component(self, q: np.ndarray) -> str:
        compatible = self.compatible_components(q)
        if len(compatible) == 0:
            raise ValueError("No compatible component found for query point.")
        return str(compatible[0])

    def transition_seed_anchors(self, goal_point: Optional[np.ndarray] = None) -> List[np.ndarray]:
        if goal_point is None:
            return []
        return [np.asarray(goal_point, dtype=float).copy()]


@dataclass
class DiscoveredLeafComponentModel:
    discovery: LeafComponentDiscoveryResult
    distance_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None

    def component_ids(self) -> List[str]:
        return [str(comp.component_id) for comp in self.discovery.components]

    def compatible_components(self, q: np.ndarray) -> List[str]:
        return [self.infer_component(q)]

    def infer_component(self, q: np.ndarray) -> str:
        cid = assign_point_to_discovered_component(
            q=np.asarray(q, dtype=float),
            discovery=self.discovery,
            distance_fn=self.distance_fn,
        )
        if cid < 0:
            raise ValueError(
                f"Failed to infer component for query point because discovery has "
                f"{len(self.discovery.components)} component(s)."
            )
        return str(cid)

    def transition_seed_anchors(self, goal_point: Optional[np.ndarray] = None) -> List[np.ndarray]:
        anchors: List[np.ndarray] = []
        for comp in self.discovery.components:
            anchors.append(np.asarray(comp.representative, dtype=float).copy())
            if len(comp.samples) > 0:
                anchors.append(np.asarray(comp.samples[0], dtype=float).copy())

        if goal_point is not None and len(self.discovery.components) > 0:
            cid = assign_point_to_discovered_component(
                q=np.asarray(goal_point, dtype=float),
                discovery=self.discovery,
                distance_fn=self.distance_fn,
            )
            for comp in self.discovery.components:
                if comp.component_id == cid:
                    anchors.extend(np.asarray(sample, dtype=float).copy() for sample in comp.samples[:4])
                    break

        return anchors


def _pairwise_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _union_find(n: int):
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    return find, union


class ComponentModelRegistry:
    """
    Per-leaf registry for component models.

    This keeps component semantics modular: discovery can populate the registry
    automatically, while hand-authored connector leaves can still be registered
    explicitly.
    """

    def __init__(self):
        self._models: Dict[LeafKey, LeafComponentModel] = {}

    @staticmethod
    def _leaf_key(family_name: str, lam: float) -> LeafKey:
        return (str(family_name), float(lam))

    def register_model(self, family_name: str, lam: float, model: LeafComponentModel) -> None:
        self._models[self._leaf_key(family_name, lam)] = model

    def register_static_components(
        self,
        family_name: str,
        lam: float,
        component_ids: List[str],
        compatibility_fn: Optional[Callable[[np.ndarray], List[str]]] = None,
    ) -> None:
        self.register_model(
            family_name=family_name,
            lam=lam,
            model=StaticComponentModel(
                ids=[str(v) for v in component_ids],
                compatibility_fn=compatibility_fn,
            ),
        )

    def register_discovered_components(
        self,
        family_name: str,
        lam: float,
        discovery: LeafComponentDiscoveryResult,
        distance_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ) -> None:
        self.register_model(
            family_name=family_name,
            lam=lam,
            model=DiscoveredLeafComponentModel(
                discovery=discovery,
                distance_fn=distance_fn,
            ),
        )

    def get_model(self, family_name: str, lam: float) -> LeafComponentModel:
        key = self._leaf_key(family_name, lam)
        if key not in self._models:
            raise KeyError(f"No component model registered for leaf {key}.")
        return self._models[key]

    def has_model(self, family_name: str, lam: float) -> bool:
        return self._leaf_key(family_name, lam) in self._models

    def component_ids_for_family(self, fam, lam: float) -> List[str]:
        return self.get_model(fam.name, float(lam)).component_ids()

    def compatible_components_for_leaf(self, fam, lam: float, q: np.ndarray) -> List[str]:
        return self.get_model(fam.name, float(lam)).compatible_components(q)

    def infer_component(self, family_name: str, lam: float, q: np.ndarray) -> str:
        return self.get_model(family_name, float(lam)).infer_component(q)

    def transition_seed_anchors(
        self,
        family_name: str,
        lam: float,
        goal_point: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        model = self.get_model(family_name, float(lam))
        anchors_fn = getattr(model, "transition_seed_anchors", None)
        if callable(anchors_fn):
            return [np.asarray(anchor, dtype=float).copy() for anchor in anchors_fn(goal_point=goal_point)]
        return []


def discover_leaf_components(
    manifold,
    seed_samples: np.ndarray,
    local_planner_name: str = "projection",
    local_planner_kwargs: Optional[dict] = None,
    step_size: float = 0.08,
    neighbor_radius: float = 0.35,
    require_bidirectional: bool = False,
):
    """
    Discover connected components of a single leaf by:
    1. taking already-valid samples on the leaf,
    2. trying constrained local connections between nearby samples,
    3. clustering the successfully connected samples.
    """
    local_planner_kwargs = dict(local_planner_kwargs or {})
    seed_samples = np.asarray(seed_samples, dtype=float)

    if len(seed_samples) == 0:
        return LeafComponentDiscoveryResult(
            success=False,
            components=[],
            sample_points=np.zeros((0, 0)),
            labels=np.zeros((0,), dtype=int),
            connectivity_edges=[],
            message="No seed samples provided.",
        )

    n = len(seed_samples)
    find, union = _union_find(n)
    connectivity_edges: List[Tuple[int, int]] = []

    for i in range(n):
        for j in range(i + 1, n):
            if _pairwise_dist(seed_samples[i], seed_samples[j]) > neighbor_radius:
                continue

            fwd = run_local_planner(
                manifold=manifold,
                x_start=seed_samples[i],
                x_goal=seed_samples[j],
                planner_name=local_planner_name,
                step_size=step_size,
                **local_planner_kwargs,
            )

            if not fwd.success:
                continue

            if require_bidirectional:
                bwd = run_local_planner(
                    manifold=manifold,
                    x_start=seed_samples[j],
                    x_goal=seed_samples[i],
                    planner_name=local_planner_name,
                    step_size=step_size,
                    **local_planner_kwargs,
                )
                if not bwd.success:
                    continue

            union(i, j)
            connectivity_edges.append((i, j))

    root_to_indices: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        root_to_indices.setdefault(r, []).append(i)

    labels = np.zeros((n,), dtype=int)
    components: List[DiscoveredComponent] = []

    for cid, (_, idxs) in enumerate(root_to_indices.items()):
        pts = seed_samples[idxs]
        rep = pts[0].copy()
        for idx in idxs:
            labels[idx] = cid
        components.append(
            DiscoveredComponent(
                component_id=cid,
                samples=pts.copy(),
                representative=rep,
                sample_indices=np.asarray(idxs, dtype=int),
            )
        )

    return LeafComponentDiscoveryResult(
        success=True,
        components=components,
        sample_points=seed_samples.copy(),
        labels=labels,
        connectivity_edges=connectivity_edges,
        message=f"Discovered {len(components)} connected components.",
    )


def build_component_model_registry(
    families,
    seed_samples_for_leaf_fn: Callable[[object, float], np.ndarray],
    should_discover_fn: Optional[Callable[[object, float], bool]] = None,
    static_model_fn: Optional[Callable[[object, float], Optional[LeafComponentModel]]] = None,
    assignment_distance_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    local_planner_name: str = "projection",
    local_planner_kwargs: Optional[dict] = None,
    step_size: float = 0.08,
    neighbor_radius: float = 0.35,
    require_bidirectional: bool = False,
) -> Tuple[ComponentModelRegistry, Dict[LeafKey, LeafComponentDiscoveryResult]]:
    registry = ComponentModelRegistry()
    discoveries: Dict[LeafKey, LeafComponentDiscoveryResult] = {}

    for fam in families:
        for lam_raw in fam.sample_lambdas():
            lam = float(lam_raw)

            if static_model_fn is not None:
                static_model = static_model_fn(fam, lam)
                if static_model is not None:
                    registry.register_model(fam.name, lam, static_model)
                    continue

            if should_discover_fn is not None and not should_discover_fn(fam, lam):
                continue

            seed_samples = np.asarray(seed_samples_for_leaf_fn(fam, lam), dtype=float)
            discovery = discover_leaf_components(
                manifold=fam.manifold(lam),
                seed_samples=seed_samples,
                local_planner_name=local_planner_name,
                local_planner_kwargs=local_planner_kwargs,
                step_size=step_size,
                neighbor_radius=neighbor_radius,
                require_bidirectional=require_bidirectional,
            )
            if not discovery.success or len(discovery.components) == 0:
                raise ValueError(
                    f"Automatic component discovery failed for leaf "
                    f"({fam.name}, {lam}): {discovery.message}"
                )
            discoveries[(fam.name, lam)] = discovery
            registry.register_discovered_components(
                family_name=fam.name,
                lam=lam,
                discovery=discovery,
                distance_fn=assignment_distance_fn,
            )

    return registry, discoveries


def assign_point_to_discovered_component(
    q: np.ndarray,
    discovery: LeafComponentDiscoveryResult,
    distance_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> int:
    """
    Assign a point to the nearest discovered component sample.

    Using all samples is more robust than using a single representative, which
    matters once leaves become more curved, wrapped, or higher-dimensional.
    """
    if distance_fn is None:
        distance_fn = _pairwise_dist

    q = np.asarray(q, dtype=float)
    best_cid = -1
    best_d = float("inf")

    for comp in discovery.components:
        if len(comp.samples) == 0:
            continue

        comp_best = min(distance_fn(q, sample) for sample in comp.samples)
        if comp_best < best_d:
            best_d = comp_best
            best_cid = comp.component_id

    return best_cid
