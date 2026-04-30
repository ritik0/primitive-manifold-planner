from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from primitive_manifold_planner.manifolds import ImplicitManifold
from primitive_manifold_planner.projection import project_newton

try:
    from ompl import base as ob
    from ompl import geometric as og
except Exception:
    ob = None
    og = None


@dataclass
class LocalPathResult:
    """
    Result of local constrained path generation on a manifold.
    """

    success: bool
    path: np.ndarray
    iterations: int
    reached_goal: bool
    message: str
    planner_name: str = "projection"
    chart_count: int = 0
    explored_edges: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)


def _validate_local_inputs(
    manifold: ImplicitManifold,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    step_size: float,
    goal_tol: float,
    max_iters: int,
):
    if step_size <= 0.0:
        raise ValueError(f"step_size must be positive, got {step_size}")
    if goal_tol <= 0.0:
        raise ValueError(f"goal_tol must be positive, got {goal_tol}")
    if max_iters <= 0:
        raise ValueError(f"max_iters must be positive, got {max_iters}")

    x_start = manifold._coerce_point(x_start)
    x_goal = manifold._coerce_point(x_goal)

    return x_start, x_goal


def _failed_result(
    x_start: np.ndarray,
    message: str,
    planner_name: str,
) -> LocalPathResult:
    return LocalPathResult(
        success=False,
        path=np.array([x_start]),
        iterations=0,
        reached_goal=False,
        message=message,
        planner_name=planner_name,
        chart_count=0,
    )


def _extract_planner_edges(ss, csi, ambient_dim: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if ob is None or ss is None or csi is None:
        return []

    try:
        planner_data = ob.PlannerData(csi)
        ss.getPlannerData(planner_data)
        planner_data.decoupleFromPlanner()
    except Exception:
        return []

    try:
        vertices = [
            _state_to_numpy(planner_data.getVertex(i).getState(), ambient_dim)
            for i in range(planner_data.numVertices())
        ]
        edges: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(planner_data.numVertices()):
            for j in planner_data.getEdges(i):
                jj = int(j)
                if 0 <= jj < len(vertices):
                    edges.append((vertices[i].copy(), vertices[jj].copy()))
        return edges
    except Exception:
        return []


def _point_is_within_bounds(manifold: ImplicitManifold, x: np.ndarray, tol: float = 1e-9) -> bool:
    within_fn = getattr(manifold, "within_bounds", None)
    if callable(within_fn):
        return bool(within_fn(np.asarray(x, dtype=float), tol=tol))
    return True


def _segment_is_within_bounds(
    manifold: ImplicitManifold,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    max_step: float = 0.05,
    tol: float = 1e-9,
) -> bool:
    within_fn = getattr(manifold, "within_bounds", None)
    if not callable(within_fn):
        return True

    q_start = np.asarray(x_start, dtype=float).reshape(-1)
    q_goal = np.asarray(x_goal, dtype=float).reshape(-1)
    seg_len = float(np.linalg.norm(q_goal - q_start))
    samples = max(2, int(np.ceil(seg_len / max(max_step, 1e-6))) + 1)
    for t in np.linspace(0.0, 1.0, samples):
        q = (1.0 - float(t)) * q_start + float(t) * q_goal
        if not bool(within_fn(q, tol=tol)):
            return False
    return True


def _path_is_within_bounds(
    manifold: ImplicitManifold,
    path: np.ndarray,
    max_step: float = 0.05,
    tol: float = 1e-9,
) -> bool:
    arr = np.asarray(path, dtype=float)
    if len(arr) == 0:
        return True
    if not all(_point_is_within_bounds(manifold, q, tol=tol) for q in arr):
        return False
    if len(arr) == 1:
        return True
    return all(
        _segment_is_within_bounds(manifold, arr[idx], arr[idx + 1], max_step=max_step, tol=tol)
        for idx in range(len(arr) - 1)
    )


def _filter_edges_within_bounds(
    manifold: ImplicitManifold,
    edges: list[tuple[np.ndarray, np.ndarray]],
    max_step: float = 0.05,
    tol: float = 1e-9,
) -> list[tuple[np.ndarray, np.ndarray]]:
    filtered: list[tuple[np.ndarray, np.ndarray]] = []
    for q_start, q_goal in edges:
        if _segment_is_within_bounds(manifold, q_start, q_goal, max_step=max_step, tol=tol):
            filtered.append((np.asarray(q_start, dtype=float).copy(), np.asarray(q_goal, dtype=float).copy()))
    return filtered


def _manifold_ambient_dim(manifold: ImplicitManifold, x_ref: np.ndarray) -> int:
    ambient_dim = getattr(manifold, "ambient_dim", None)
    if ambient_dim is not None:
        return int(ambient_dim)
    return int(np.asarray(x_ref, dtype=float).reshape(-1).shape[0])


def _manifold_codim(manifold: ImplicitManifold, x_ref: np.ndarray) -> int:
    codim = getattr(manifold, "codim", None)
    if codim is not None:
        return int(codim)
    residual = np.asarray(manifold.residual(np.asarray(x_ref, dtype=float)), dtype=float).reshape(-1)
    return int(residual.shape[0])


def _state_to_numpy(state: Any, ambient_dim: int) -> np.ndarray:
    return np.asarray([float(state[i]) for i in range(ambient_dim)], dtype=float)


class _OmplConstraintAdapter(ob.Constraint if ob is not None else object):
    def __init__(self, manifold: ImplicitManifold, x_ref: np.ndarray):
        self._manifold = manifold
        ambient_dim = _manifold_ambient_dim(manifold, x_ref)
        codim = _manifold_codim(manifold, x_ref)
        super().__init__(ambient_dim, codim)

    def function(self, x, out):
        residual = np.asarray(self._manifold.residual(np.asarray(x, dtype=float)), dtype=float).reshape(-1)
        out[:] = residual

    def jacobian(self, x, out):
        jacobian = np.asarray(self._manifold.jacobian(np.asarray(x, dtype=float)), dtype=float)
        out[:, :] = jacobian


def _build_ompl_ambient_space(
    ambient_dim: int,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    bounds_min: np.ndarray | None,
    bounds_max: np.ndarray | None,
    padding: float,
):
    if ob is None:
        raise RuntimeError("OMPL Python bindings are not available.")

    if bounds_min is None or bounds_max is None:
        lo = np.minimum(x_start, x_goal) - float(padding)
        hi = np.maximum(x_start, x_goal) + float(padding)
    else:
        lo = np.asarray(bounds_min, dtype=float).reshape(-1)
        hi = np.asarray(bounds_max, dtype=float).reshape(-1)

    if lo.shape[0] != ambient_dim or hi.shape[0] != ambient_dim:
        raise ValueError("OMPL ambient bounds dimension mismatch.")

    bounds = ob.RealVectorBounds(ambient_dim)
    for i in range(ambient_dim):
        if hi[i] <= lo[i]:
            raise ValueError("OMPL ambient bounds must satisfy high > low in every dimension.")
        bounds.setLow(i, float(lo[i]))
        bounds.setHigh(i, float(hi[i]))

    space = ob.RealVectorStateSpace(ambient_dim)
    space.setBounds(bounds)
    return space


def _ompl_path_on_manifold(
    manifold: ImplicitManifold,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    planner_name: str,
    constrained_space_name: str,
    step_size: float,
    goal_tol: float,
    max_iters: int,
    projection_tol: float,
    projection_max_iters: int,
    solve_time: float | None = None,
    bounds_min: np.ndarray | None = None,
    bounds_max: np.ndarray | None = None,
    ambient_padding: float = 0.5,
    ompl_planner_name: str = "RRTConnect",
    ompl_delta: float | None = None,
    ompl_lambda: float | None = None,
    atlas_exploration: float | None = None,
    atlas_epsilon: float | None = None,
    atlas_rho: float | None = None,
    atlas_alpha: float | None = None,
    atlas_max_charts_per_extension: int | None = None,
    atlas_separated: bool = True,
) -> LocalPathResult:
    if ob is None or og is None:
        return _failed_result(
            x_start=x_start,
            message="OMPL Python bindings are not available in this environment.",
            planner_name=planner_name,
        )

    ambient_dim = _manifold_ambient_dim(manifold, x_start)
    constraint = _OmplConstraintAdapter(manifold=manifold, x_ref=x_start)

    if solve_time is None:
        solve_time = min(0.5, max(0.1, float(max_iters) / 4000.0))

    constraint_tol = max(float(projection_tol), min(1e-4, float(goal_tol)))
    constraint.setTolerance(constraint_tol)
    constraint.setMaxIterations(int(projection_max_iters))

    try:
        space = _build_ompl_ambient_space(
            ambient_dim=ambient_dim,
            x_start=x_start,
            x_goal=x_goal,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            padding=ambient_padding,
        )
    except Exception as exc:
        return _failed_result(
            x_start=x_start,
            message=f"Failed to build OMPL ambient space: {exc}",
            planner_name=planner_name,
        )

    try:
        if constrained_space_name == "projection":
            css = ob.ProjectedStateSpace(space, constraint)
        elif constrained_space_name == "atlas":
            css = ob.AtlasStateSpace(space, constraint)
        else:
            raise ValueError(f"Unknown constrained_space_name '{constrained_space_name}'.")

        csi = ob.ConstrainedSpaceInformation(css)
        css.setSpaceInformation(csi)
        css.setDelta(float(step_size if ompl_delta is None else ompl_delta))
        css.setLambda(float(ob.CONSTRAINED_STATE_SPACE_LAMBDA if ompl_lambda is None else ompl_lambda))

        if constrained_space_name == "atlas":
            css.setExploration(
                float(ob.ATLAS_STATE_SPACE_EXPLORATION if atlas_exploration is None else atlas_exploration)
            )
            css.setEpsilon(float(ob.ATLAS_STATE_SPACE_EPSILON if atlas_epsilon is None else atlas_epsilon))
            css.setRho(
                float(
                    step_size * ob.ATLAS_STATE_SPACE_RHO_MULTIPLIER
                    if atlas_rho is None
                    else atlas_rho
                )
            )
            css.setAlpha(float(ob.ATLAS_STATE_SPACE_ALPHA if atlas_alpha is None else atlas_alpha))
            css.setMaxChartsPerExtension(
                int(
                    ob.ATLAS_STATE_SPACE_MAX_CHARTS_PER_EXTENSION
                    if atlas_max_charts_per_extension is None
                    else atlas_max_charts_per_extension
                )
            )
            css.setSeparated(bool(atlas_separated))
        css.setup()
        ss = og.SimpleSetup(csi)
        ss.setStateValidityChecker(
            lambda state: _point_is_within_bounds(
                manifold,
                _state_to_numpy(state, ambient_dim),
            )
        )

        start = css.allocState()
        goal = css.allocState()
        start.copy(x_start.tolist())
        goal.copy(x_goal.tolist())

        if constrained_space_name == "atlas":
            css.anchorChart(start)
            css.anchorChart(goal)

        try:
            ss.setStartAndGoalStates(start, goal, float(goal_tol))
        except TypeError:
            ss.setStartAndGoalStates(start, goal)

        planner_cls = getattr(og, ompl_planner_name, None)
        if planner_cls is None:
            return _failed_result(
                x_start=x_start,
                message=f"Unknown OMPL planner '{ompl_planner_name}'.",
                planner_name=planner_name,
            )

        planner = planner_cls(csi)
        if hasattr(planner, "setRange"):
            planner.setRange(float(step_size))
        ss.setPlanner(planner)
        ss.setup()

        status = ss.solve(float(solve_time))
        explored_edges = _extract_planner_edges(ss, csi, ambient_dim)
        if not bool(status) or not ss.haveSolutionPath():
            return LocalPathResult(
                success=False,
                path=np.asarray([x_start]),
                iterations=int(max_iters),
                reached_goal=False,
                message=f"OMPL {planner_name} failed to find a local path within {solve_time:.3f}s.",
                planner_name=planner_name,
                chart_count=int(css.getChartCount()) if hasattr(css, "getChartCount") else 0,
                explored_edges=explored_edges,
            )

        path = ss.getSolutionPath()
        path.interpolate()
        path_np = np.asarray(
            [_state_to_numpy(path.getState(i), ambient_dim) for i in range(path.getStateCount())],
            dtype=float,
        )
        explored_edges = _filter_edges_within_bounds(manifold, explored_edges, max_step=max(step_size * 0.5, 0.03))
        if not _path_is_within_bounds(manifold, path_np, max_step=max(step_size * 0.5, 0.03)):
            return LocalPathResult(
                success=False,
                path=np.asarray([x_start]),
                iterations=int(max(0, path.getStateCount() - 1)),
                reached_goal=False,
                message=(
                    f"OMPL {planner_name} returned a path that left the bounded manifold region between samples."
                ),
                planner_name=planner_name,
                chart_count=int(css.getChartCount()) if hasattr(css, "getChartCount") else 0,
                explored_edges=explored_edges,
            )
        final_dist = float(np.linalg.norm(path_np[-1] - x_goal))
        reached_goal = final_dist <= goal_tol

        return LocalPathResult(
            success=reached_goal,
            path=path_np,
            iterations=max(0, int(path.getStateCount()) - 1),
            reached_goal=reached_goal,
            message=(
                f"OMPL {planner_name} reached the goal on the manifold."
                if reached_goal
                else f"OMPL {planner_name} returned an approximate path with final distance {final_dist:.4e}."
            ),
            planner_name=planner_name,
            chart_count=int(css.getChartCount()) if hasattr(css, "getChartCount") else 0,
            explored_edges=explored_edges,
        )
    except Exception as exc:
        return LocalPathResult(
            success=False,
            path=np.asarray([x_start]),
            iterations=0,
            reached_goal=False,
            message=f"OMPL {planner_name} failed with exception: {exc}",
            planner_name=planner_name,
            chart_count=0,
            explored_edges=[],
        )


def ompl_sample_state_on_manifold(
    manifold: ImplicitManifold,
    x_seed: np.ndarray,
    constrained_space_name: str,
    bounds_min: np.ndarray | None = None,
    bounds_max: np.ndarray | None = None,
    ambient_padding: float = 0.5,
    projection_tol: float = 1e-10,
    projection_max_iters: int = 60,
    ompl_delta: float | None = None,
    ompl_lambda: float | None = None,
    atlas_exploration: float | None = None,
    atlas_epsilon: float | None = None,
    atlas_rho: float | None = None,
    atlas_alpha: float | None = None,
    atlas_max_charts_per_extension: int | None = None,
    atlas_separated: bool = True,
) -> np.ndarray | None:
    if ob is None:
        return None

    x_seed = manifold._coerce_point(x_seed)
    ambient_dim = _manifold_ambient_dim(manifold, x_seed)
    constraint = _OmplConstraintAdapter(manifold=manifold, x_ref=x_seed)
    constraint.setTolerance(float(projection_tol))
    constraint.setMaxIterations(int(projection_max_iters))

    try:
        space = _build_ompl_ambient_space(
            ambient_dim=ambient_dim,
            x_start=x_seed,
            x_goal=x_seed,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            padding=ambient_padding,
        )
        if constrained_space_name == "projection":
            css = ob.ProjectedStateSpace(space, constraint)
        elif constrained_space_name == "atlas":
            css = ob.AtlasStateSpace(space, constraint)
        else:
            raise ValueError(f"Unknown constrained_space_name '{constrained_space_name}'.")

        csi = ob.ConstrainedSpaceInformation(css)
        css.setSpaceInformation(csi)
        css.setDelta(float(0.1 if ompl_delta is None else ompl_delta))
        css.setLambda(float(ob.CONSTRAINED_STATE_SPACE_LAMBDA if ompl_lambda is None else ompl_lambda))
        if constrained_space_name == "atlas":
            css.setExploration(
                float(ob.ATLAS_STATE_SPACE_EXPLORATION if atlas_exploration is None else atlas_exploration)
            )
            css.setEpsilon(float(ob.ATLAS_STATE_SPACE_EPSILON if atlas_epsilon is None else atlas_epsilon))
            css.setRho(float(0.1 * ob.ATLAS_STATE_SPACE_RHO_MULTIPLIER if atlas_rho is None else atlas_rho))
            css.setAlpha(float(ob.ATLAS_STATE_SPACE_ALPHA if atlas_alpha is None else atlas_alpha))
            css.setMaxChartsPerExtension(
                int(
                    ob.ATLAS_STATE_SPACE_MAX_CHARTS_PER_EXTENSION
                    if atlas_max_charts_per_extension is None
                    else atlas_max_charts_per_extension
                )
            )
            css.setSeparated(bool(atlas_separated))

        css.setup()
        seed_state = css.allocState()
        seed_state.copy(x_seed.tolist())
        if constrained_space_name == "atlas":
            css.anchorChart(seed_state)
        sampler = css.allocStateSampler()
        sampled = css.allocState()
        sampler.sampleUniform(sampled)
        return _state_to_numpy(sampled, ambient_dim)
    except Exception:
        return None


def constrained_interpolate(
    manifold: ImplicitManifold,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    step_size: float = 0.1,
    goal_tol: float = 1e-3,
    max_iters: int = 500,
    projection_tol: float = 1e-10,
    projection_max_iters: int = 50,
    projection_damping: float = 1.0,
) -> LocalPathResult:
    """
    Projection-style local constrained motion.

    Strategy:
      - move directly toward goal in ambient space
      - project each trial step back onto the manifold
    """
    x_start, x_goal = _validate_local_inputs(
        manifold=manifold,
        x_start=x_start,
        x_goal=x_goal,
        step_size=step_size,
        goal_tol=goal_tol,
        max_iters=max_iters,
    )

    if not manifold.is_valid(x_start, tol=1e-6):
        return _failed_result(
            x_start=x_start,
            message="Start point is not on the manifold.",
            planner_name="projection",
        )

    if not manifold.is_valid(x_goal, tol=1e-6):
        return _failed_result(
            x_start=x_start,
            message="Goal point is not on the manifold.",
            planner_name="projection",
        )

    if not _point_is_within_bounds(manifold, x_start):
        return _failed_result(
            x_start=x_start,
            message="Start point is outside bounded manifold limits.",
            planner_name="projection",
        )

    if not _point_is_within_bounds(manifold, x_goal):
        return _failed_result(
            x_start=x_start,
            message="Goal point is outside bounded manifold limits.",
            planner_name="projection",
        )

    x_current = x_start.copy()
    path = [x_current.copy()]

    for k in range(max_iters):
        to_goal = x_goal - x_current
        dist_to_goal = float(np.linalg.norm(to_goal))

        if dist_to_goal <= goal_tol:
            return LocalPathResult(
                success=True,
                path=np.asarray(path),
                iterations=k,
                reached_goal=True,
                message="Reached goal tolerance on the manifold.",
                planner_name="projection",
                chart_count=0,
            )

        direction = to_goal / max(dist_to_goal, 1e-15)
        step = min(step_size, dist_to_goal)
        x_trial = x_current + step * direction

        proj = project_newton(
            manifold=manifold,
            x0=x_trial,
            tol=projection_tol,
            max_iters=projection_max_iters,
            damping=projection_damping,
        )

        if not proj.success:
            return LocalPathResult(
                success=False,
                path=np.asarray(path),
                iterations=k,
                reached_goal=False,
                message=f"Projection failed during local planning: {proj.message}",
                planner_name="projection",
                chart_count=0,
            )

        x_next = proj.x_projected

        if not _point_is_within_bounds(manifold, x_next):
            return LocalPathResult(
                success=False,
                path=np.asarray(path),
                iterations=k,
                reached_goal=False,
                message="Projection left the bounded manifold region.",
                planner_name="projection",
                chart_count=0,
            )

        move_norm = float(np.linalg.norm(x_next - x_current))
        if move_norm <= 1e-12:
            return LocalPathResult(
                success=False,
                path=np.asarray(path),
                iterations=k,
                reached_goal=False,
                message="Local planning stalled: projected step made no progress.",
                planner_name="projection",
                chart_count=0,
            )

        path.append(x_next.copy())
        x_current = x_next

    final_dist = float(np.linalg.norm(x_goal - x_current))
    reached_goal = final_dist <= goal_tol

    return LocalPathResult(
        success=reached_goal,
        path=np.asarray(path),
        iterations=max_iters,
        reached_goal=reached_goal,
        message=(
            "Local planning reached max iterations and hit goal tolerance."
            if reached_goal
            else "Local planning reached max iterations without reaching the goal."
        ),
        planner_name="projection",
        chart_count=0,
    )


def _choose_tangent_direction(
    manifold: ImplicitManifold,
    x_current: np.ndarray,
    x_goal: np.ndarray,
    preview_step: float,
) -> np.ndarray:
    """
    Choose a tangent direction whose sign better reduces distance to x_goal.

    For codimension-1 manifolds, the tangent projector applied to the goal
    direction gives a good local continuation direction.
    """
    to_goal = x_goal - x_current
    to_goal_norm = float(np.linalg.norm(to_goal))
    if to_goal_norm <= 1e-15:
        return np.zeros_like(x_current)

    ambient_dir = to_goal / to_goal_norm
    tangent_dir = manifold.project_tangent(x_current, ambient_dir)
    tangent_norm = float(np.linalg.norm(tangent_dir))

    # If goal direction is almost normal to the manifold, build a fallback tangent
    # from the Jacobian normal in 2D codim-1 cases.
    if tangent_norm <= 1e-12 and manifold.ambient_dim == 2 and manifold.codim == 1:
        J = manifold.jacobian(x_current)
        n = J.reshape(-1)
        tangent_dir = np.array([-n[1], n[0]], dtype=float)
        tangent_norm = float(np.linalg.norm(tangent_dir))

    if tangent_norm <= 1e-12:
        return np.zeros_like(x_current)

    tangent_dir = tangent_dir / tangent_norm

    x_plus = x_current + preview_step * tangent_dir
    x_minus = x_current - preview_step * tangent_dir

    if np.linalg.norm(x_plus - x_goal) <= np.linalg.norm(x_minus - x_goal):
        return tangent_dir
    return -tangent_dir


def atlas_like_interpolate(
    manifold: ImplicitManifold,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    step_size: float = 0.1,
    goal_tol: float = 1e-3,
    max_iters: int = 500,
    projection_tol: float = 1e-10,
    projection_max_iters: int = 50,
    projection_damping: float = 1.0,
    preview_fraction: float = 0.5,
    min_progress_tol: float = 1e-10,
    max_step_growth: float = 2.5,
) -> LocalPathResult:
    """
    Atlas-like continuation local planner.

    Strategy:
      - compute a local tangent direction at current point
      - step along tangent direction
      - project back to the manifold
      - treat each accepted projected point as a new local chart center

    This is not a full AtlasRRT chart atlas, but it is a continuation-style
    local planner built from tangent stepping plus projection correction.
    """
    x_start, x_goal = _validate_local_inputs(
        manifold=manifold,
        x_start=x_start,
        x_goal=x_goal,
        step_size=step_size,
        goal_tol=goal_tol,
        max_iters=max_iters,
    )

    if not manifold.is_valid(x_start, tol=1e-6):
        return _failed_result(
            x_start=x_start,
            message="Start point is not on the manifold.",
            planner_name="atlas_like",
        )

    if not manifold.is_valid(x_goal, tol=1e-6):
        return _failed_result(
            x_start=x_start,
            message="Goal point is not on the manifold.",
            planner_name="atlas_like",
        )

    if not _point_is_within_bounds(manifold, x_start):
        return _failed_result(
            x_start=x_start,
            message="Start point is outside bounded manifold limits.",
            planner_name="atlas_like",
        )

    if not _point_is_within_bounds(manifold, x_goal):
        return _failed_result(
            x_start=x_start,
            message="Goal point is outside bounded manifold limits.",
            planner_name="atlas_like",
        )

    x_current = x_start.copy()
    path = [x_current.copy()]
    chart_count = 1

    for k in range(max_iters):
        to_goal = x_goal - x_current
        dist_to_goal = float(np.linalg.norm(to_goal))

        if dist_to_goal <= goal_tol:
            return LocalPathResult(
                success=True,
                path=np.asarray(path),
                iterations=k,
                reached_goal=True,
                message="Reached goal tolerance on the manifold.",
                planner_name="atlas_like",
                chart_count=chart_count,
            )

        tangent_dir = _choose_tangent_direction(
            manifold=manifold,
            x_current=x_current,
            x_goal=x_goal,
            preview_step=max(step_size * preview_fraction, 1e-6),
        )

        tangent_norm = float(np.linalg.norm(tangent_dir))
        if tangent_norm <= 1e-12:
            return LocalPathResult(
                success=False,
                path=np.asarray(path),
                iterations=k,
                reached_goal=False,
                message="Atlas-like local planning stalled: no usable tangent direction.",
                planner_name="atlas_like",
                chart_count=chart_count,
            )

        step = min(step_size, dist_to_goal)
        x_trial = x_current + step * tangent_dir

        proj = project_newton(
            manifold=manifold,
            x0=x_trial,
            tol=projection_tol,
            max_iters=projection_max_iters,
            damping=projection_damping,
        )

        if not proj.success:
            return LocalPathResult(
                success=False,
                path=np.asarray(path),
                iterations=k,
                reached_goal=False,
                message=f"Projection failed during atlas-like continuation: {proj.message}",
                planner_name="atlas_like",
                chart_count=chart_count,
            )

        x_next = proj.x_projected
        if not _point_is_within_bounds(manifold, x_next):
            return LocalPathResult(
                success=False,
                path=np.asarray(path),
                iterations=k,
                reached_goal=False,
                message="Atlas-like continuation left the bounded manifold region.",
                planner_name="atlas_like",
                chart_count=chart_count,
            )
        move_norm = float(np.linalg.norm(x_next - x_current))

        # retry with a shorter tangent step once if the continuation behaved badly
        if move_norm <= min_progress_tol or move_norm > max_step_growth * step_size:
            x_trial_small = x_current + 0.5 * step * tangent_dir
            proj_small = project_newton(
                manifold=manifold,
                x0=x_trial_small,
                tol=projection_tol,
                max_iters=projection_max_iters,
                damping=projection_damping,
            )

            if not proj_small.success:
                return LocalPathResult(
                    success=False,
                    path=np.asarray(path),
                    iterations=k,
                    reached_goal=False,
                    message=(
                        "Atlas-like continuation failed: projection failed on both full and reduced tangent steps."
                    ),
                    planner_name="atlas_like",
                    chart_count=chart_count,
                )

            x_next = proj_small.x_projected
            if not _point_is_within_bounds(manifold, x_next):
                return LocalPathResult(
                    success=False,
                    path=np.asarray(path),
                    iterations=k,
                    reached_goal=False,
                    message="Reduced atlas-like step left the bounded manifold region.",
                    planner_name="atlas_like",
                    chart_count=chart_count,
                )
            move_norm = float(np.linalg.norm(x_next - x_current))

        if move_norm <= min_progress_tol:
            return LocalPathResult(
                success=False,
                path=np.asarray(path),
                iterations=k,
                reached_goal=False,
                message="Atlas-like local planning stalled: projected tangent step made no progress.",
                planner_name="atlas_like",
                chart_count=chart_count,
            )

        path.append(x_next.copy())
        x_current = x_next
        chart_count += 1

    final_dist = float(np.linalg.norm(x_goal - x_current))
    reached_goal = final_dist <= goal_tol

    return LocalPathResult(
        success=reached_goal,
        path=np.asarray(path),
        iterations=max_iters,
        reached_goal=reached_goal,
        message=(
            "Atlas-like local planning reached max iterations and hit goal tolerance."
            if reached_goal
            else "Atlas-like local planning reached max iterations without reaching the goal."
        ),
        planner_name="atlas_like",
        chart_count=chart_count,
    )


def ompl_projected_interpolate(
    manifold: ImplicitManifold,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    step_size: float = 0.1,
    goal_tol: float = 1e-3,
    max_iters: int = 500,
    projection_tol: float = 1e-10,
    projection_max_iters: int = 50,
    **kwargs: Any,
) -> LocalPathResult:
    x_start, x_goal = _validate_local_inputs(
        manifold=manifold,
        x_start=x_start,
        x_goal=x_goal,
        step_size=step_size,
        goal_tol=goal_tol,
        max_iters=max_iters,
    )

    if not manifold.is_valid(x_start, tol=1e-6):
        return _failed_result(
            x_start=x_start,
            message="Start point is not on the manifold.",
            planner_name="ompl_projection",
        )

    if not manifold.is_valid(x_goal, tol=1e-6):
        return _failed_result(
            x_start=x_start,
            message="Goal point is not on the manifold.",
            planner_name="ompl_projection",
        )

    if not _point_is_within_bounds(manifold, x_start):
        return _failed_result(
            x_start=x_start,
            message="Start point is outside bounded manifold limits.",
            planner_name="ompl_projection",
        )

    if not _point_is_within_bounds(manifold, x_goal):
        return _failed_result(
            x_start=x_start,
            message="Goal point is outside bounded manifold limits.",
            planner_name="ompl_projection",
        )

    return _ompl_path_on_manifold(
        manifold=manifold,
        x_start=x_start,
        x_goal=x_goal,
        planner_name="ompl_projection",
        constrained_space_name="projection",
        step_size=step_size,
        goal_tol=goal_tol,
        max_iters=max_iters,
        projection_tol=projection_tol,
        projection_max_iters=projection_max_iters,
        **kwargs,
    )


def ompl_atlas_interpolate(
    manifold: ImplicitManifold,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    step_size: float = 0.1,
    goal_tol: float = 1e-3,
    max_iters: int = 500,
    projection_tol: float = 1e-10,
    projection_max_iters: int = 50,
    **kwargs: Any,
) -> LocalPathResult:
    x_start, x_goal = _validate_local_inputs(
        manifold=manifold,
        x_start=x_start,
        x_goal=x_goal,
        step_size=step_size,
        goal_tol=goal_tol,
        max_iters=max_iters,
    )

    if not manifold.is_valid(x_start, tol=1e-6):
        return _failed_result(
            x_start=x_start,
            message="Start point is not on the manifold.",
            planner_name="ompl_atlas",
        )

    if not manifold.is_valid(x_goal, tol=1e-6):
        return _failed_result(
            x_start=x_start,
            message="Goal point is not on the manifold.",
            planner_name="ompl_atlas",
        )

    if not _point_is_within_bounds(manifold, x_start):
        return _failed_result(
            x_start=x_start,
            message="Start point is outside bounded manifold limits.",
            planner_name="ompl_atlas",
        )

    if not _point_is_within_bounds(manifold, x_goal):
        return _failed_result(
            x_start=x_start,
            message="Goal point is outside bounded manifold limits.",
            planner_name="ompl_atlas",
        )

    return _ompl_path_on_manifold(
        manifold=manifold,
        x_start=x_start,
        x_goal=x_goal,
        planner_name="ompl_atlas",
        constrained_space_name="atlas",
        step_size=step_size,
        goal_tol=goal_tol,
        max_iters=max_iters,
        projection_tol=projection_tol,
        projection_max_iters=projection_max_iters,
        **kwargs,
    )


def run_local_planner(
    manifold: ImplicitManifold,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    planner_name: str = "projection",
    **kwargs: Any,
) -> LocalPathResult:
    """
    Dispatch local constrained motion generation by planner name.
    """
    if planner_name == "projection":
        return constrained_interpolate(
            manifold=manifold,
            x_start=x_start,
            x_goal=x_goal,
            **kwargs,
        )

    if planner_name == "atlas_like":
        return atlas_like_interpolate(
            manifold=manifold,
            x_start=x_start,
            x_goal=x_goal,
            **kwargs,
        )

    if planner_name == "ompl_projection":
        return ompl_projected_interpolate(
            manifold=manifold,
            x_start=x_start,
            x_goal=x_goal,
            **kwargs,
        )

    if planner_name == "ompl_atlas":
        return ompl_atlas_interpolate(
            manifold=manifold,
            x_start=x_start,
            x_goal=x_goal,
            **kwargs,
        )

    raise ValueError(
        f"Unknown local planner '{planner_name}'. Expected one of: ['projection', 'atlas_like', 'ompl_projection', 'ompl_atlas']"
    )
