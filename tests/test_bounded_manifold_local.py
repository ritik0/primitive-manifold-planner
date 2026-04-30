import numpy as np

from primitive_manifold_planner.manifolds import ImplicitManifold
from primitive_manifold_planner.planning.local import constrained_interpolate


class BoundedLineManifold(ImplicitManifold):
    def __init__(self, y: float = 0.0, x_min: float = -1.0, x_max: float = 1.0):
        super().__init__(ambient_dim=2, codim=1, name="bounded_line")
        self.y = float(y)
        self.x_min = float(x_min)
        self.x_max = float(x_max)

    def residual(self, x: np.ndarray) -> np.ndarray:
        x = self._coerce_point(x)
        return np.array([x[1] - self.y], dtype=float)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        _ = self._coerce_point(x)
        return np.array([[0.0, 1.0]], dtype=float)

    def within_bounds(self, x: np.ndarray, tol: float = 1e-9) -> bool:
        x = self._coerce_point(x)
        return self.x_min - tol <= x[0] <= self.x_max + tol


def test_constrained_interpolate_rejects_goal_outside_bounded_manifold():
    manifold = BoundedLineManifold(y=0.0, x_min=-1.0, x_max=1.0)

    result = constrained_interpolate(
        manifold=manifold,
        x_start=np.array([0.0, 0.0]),
        x_goal=np.array([1.8, 0.0]),
        step_size=0.1,
        goal_tol=1e-3,
        max_iters=200,
    )

    assert not result.success
    assert "outside bounded manifold limits" in result.message


def test_constrained_interpolate_succeeds_inside_bounded_manifold():
    manifold = BoundedLineManifold(y=0.0, x_min=-1.0, x_max=1.0)

    result = constrained_interpolate(
        manifold=manifold,
        x_start=np.array([-0.8, 0.0]),
        x_goal=np.array([0.8, 0.0]),
        step_size=0.1,
        goal_tol=1e-3,
        max_iters=200,
    )

    assert result.success
    assert len(result.path) > 1
