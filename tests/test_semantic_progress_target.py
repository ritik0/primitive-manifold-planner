import numpy as np

from primitive_manifold_planner.manifolds import CircleManifold
from primitive_manifold_planner.planners.admissibility import (
    choose_semantic_progress_target,
    ProgressTargetSelectionConfig,
)
from primitive_manifold_planner.projection import project_newton


class AnchorFamily:
    def __init__(self, manifold, admissibility_cost_fn=None):
        self.name = "anchor_family"
        self._manifold = manifold
        self._admissibility_cost_fn = admissibility_cost_fn

    def manifold(self, lam):
        _ = lam
        return self._manifold

    def transition_seed_anchors(self, lam, goal_point=None):
        _ = lam, goal_point
        return [
            np.array([0.0, 2.0]),
            np.array([0.0, -2.0]),
        ]

    def transition_admissibility_cost(self, lam, point, goal_point=None, metadata=None):
        if self._admissibility_cost_fn is None:
            return 0.0
        return float(
            self._admissibility_cost_fn(
                lam,
                np.asarray(point, dtype=float),
                None if goal_point is None else np.asarray(goal_point, dtype=float),
                metadata,
            )
        )


def test_semantic_progress_target_can_override_goal_projection_with_leaf_semantics():
    manifold = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0, name="circle")
    family = AnchorFamily(
        manifold=manifold,
        admissibility_cost_fn=lambda lam, point, goal, meta: 10.0 if float(point[1]) > 0.0 else 0.0,
    )

    selected = choose_semantic_progress_target(
        fam=family,
        lam=0.0,
        target_leaf=manifold,
        goal_point=np.array([0.0, 3.0]),
        project_newton=project_newton,
        config=ProgressTargetSelectionConfig(
            goal_distance_weight=1.0,
            admissibility_weight=2.0,
        ),
    )

    assert selected is not None
    assert selected[1] < 0.0
