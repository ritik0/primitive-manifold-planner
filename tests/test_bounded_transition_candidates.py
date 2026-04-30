import numpy as np

from primitive_manifold_planner.manifolds import CircleManifold, ImplicitManifold
from primitive_manifold_planner.planners.transition_manager import TransitionGenerator
from primitive_manifold_planner.projection import project_newton


class SingleLeafFamily:
    def __init__(self, name: str, manifold):
        self.name = name
        self._manifold = manifold

    def sample_lambdas(self):
        return [0.0]

    def manifold(self, lam: float):
        _ = lam
        return self._manifold

    def lambda_distance(self, lam_a, lam_b):
        return abs(float(lam_a) - float(lam_b))


class BoundedVerticalLineManifold(ImplicitManifold):
    def __init__(self, x_value: float = 0.0, y_min: float = -0.5, y_max: float = 0.5):
        super().__init__(ambient_dim=2, codim=1, name="bounded_vertical_line")
        self.x_value = float(x_value)
        self.y_min = float(y_min)
        self.y_max = float(y_max)

    def residual(self, x: np.ndarray) -> np.ndarray:
        x = self._coerce_point(x)
        return np.array([x[0] - self.x_value], dtype=float)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        _ = self._coerce_point(x)
        return np.array([[1.0, 0.0]], dtype=float)

    def within_bounds(self, x: np.ndarray, tol: float = 1e-9) -> bool:
        x = self._coerce_point(x)
        return self.y_min - tol <= x[1] <= self.y_max + tol


def test_transition_generator_filters_out_of_bounds_intersections():
    circle = CircleManifold(center=np.array([0.0, 0.0]), radius=2.0, name="circle")
    bounded_line = BoundedVerticalLineManifold(x_value=0.0, y_min=-0.5, y_max=0.5)

    circle_family = SingleLeafFamily("circle_family", circle)
    line_family = SingleLeafFamily("bounded_line_family", bounded_line)

    def seed_points_fn(_fam_a, _lam_a, _fam_b, _lam_b):
        return [
            np.array([0.1, 1.9]),
            np.array([0.1, -1.9]),
            np.array([0.0, 0.0]),
        ]

    generator = TransitionGenerator(
        seed_points_fn=seed_points_fn,
        project_newton=project_newton,
    )

    result = generator.generate_transitions(
        source_family=circle_family,
        source_lam=0.0,
        target_family=line_family,
        target_lam=0.0,
        goal_point=np.array([0.0, 0.0]),
    )

    assert not result.success
    assert len(result.candidates) == 0
