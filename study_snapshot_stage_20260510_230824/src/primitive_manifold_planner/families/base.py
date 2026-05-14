from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Any
import numpy as np


@dataclass(frozen=True)
class LeafDescriptor:
    family_name: str
    lam: Any


class ManifoldFamily(ABC):
    """
    A family of manifolds M_lambda.

    Example:
      - parallel lines indexed by y = lambda
      - concentric circles indexed by radius = lambda
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def manifold(self, lam):
        """Return the manifold object for a given lambda."""
        raise NotImplementedError

    @abstractmethod
    def sample_lambdas(self, context=None) -> Iterable:
        """Return candidate lambda values for this family."""
        raise NotImplementedError

    def nearby_lambdas(self, lam, radius: float) -> Iterable:
        """
        Optional: return lambdas near lam.
        Default fallback: just return all sampled lambdas.
        """
        return self.sample_lambdas(context={"center": lam, "radius": radius})

    def lambda_distance(self, lam_a, lam_b) -> float:
        """Used for preference / penalty. Override if lambda is not scalar."""
        try:
            return abs(float(lam_a) - float(lam_b))
        except Exception:
            return 0.0

    def describe_leaf(self, lam) -> LeafDescriptor:
        return LeafDescriptor(family_name=self.name, lam=lam)

    def transition_seed_anchors(self, lam, goal_point: np.ndarray | None = None) -> Iterable[np.ndarray]:
        """
        Optional reusable geometric anchor points for transition seeding.

        Families can override this to provide a small set of characteristic
        points on a leaf that are useful for transition discovery.
        """
        _ = lam, goal_point
        return []

    def transition_admissibility_cost(
        self,
        lam,
        point: np.ndarray,
        goal_point: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> float:
        """
        Optional reusable semantic cost for switching on or through this leaf.

        This is intended for problem-level meaning such as transfer difficulty,
        preferred connector regions, or support-vs-transfer bias. The default
        implementation is neutral.
        """
        _ = lam, point, goal_point, metadata
        return 0.0

    def transition_feasibility(
        self,
        lam,
        point: np.ndarray,
        goal_point: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> bool:
        """
        Optional reusable semantic feasibility predicate for switching on or
        through this leaf. The default implementation imposes no extra
        restriction beyond geometry.
        """
        _ = lam, point, goal_point, metadata
        return True
