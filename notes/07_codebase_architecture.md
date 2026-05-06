# Codebase Architecture

This repository is organized so the examples are easy to run while the reusable thesis code lives inside the installable package.

## Workflow

The `examples/` folder contains entry-point scripts. An example should build or select a scene, parse command-line flags, call package functions, print a summary, and optionally open a visualization. Large planner implementations should live under `src/primitive_manifold_planner/`.

The package namespace is the source of reusable code:

- `src/primitive_manifold_planner/manifolds/` defines constraint manifolds. These include primitive geometric constraints such as spheres, planes, masks, rounded boxes, and robot task-space manifolds.
- `src/primitive_manifold_planner/thesis/` contains the Example 66-style parallel evidence planner. This is the main thesis workflow for accumulating stage-local evidence, discovering transition hypotheses, and extracting committed multimodal routes.
- `src/primitive_manifold_planner/scenes/` contains reusable scene builders for harder environments, such as the blocked-plane stress scene.
- `src/primitive_manifold_planner/visualization/` displays planner results, graph evidence, task-space paths, and robot motion.
- `src/primitive_manifold_planner/experiments/` contains comparison scripts and continuous-transfer modules, including serial-vs-parallel sweeps and obstacle-profile sweeps.

## Import Rule

Reusable code should use package-qualified imports, for example:

```python
from primitive_manifold_planner.thesis.parallel_evidence_planner import StageEvidenceStore
from primitive_manifold_planner.scenes.stress_scene import build_stress_scene
from primitive_manifold_planner.visualization.display import show_route
from primitive_manifold_planner.manifolds.robot import RobotSphereManifold
```

Top-level helper packages such as `planner`, `manifolds`, `scenes`, `visualization`, and `experiments` are intentionally avoided. Keeping one package namespace makes the thesis code easier to read, test, and package.

## Example 66-Style Planner

The parallel evidence planner maintains one evidence graph per active stage or leaf. Ambient proposals are projected onto candidate manifolds, scored for usefulness, and used to grow local graph evidence. Transition hypotheses connect stages only at certified manifold intersections. A committed route is extracted after enough evidence exists to connect the desired sequence.

The joint-space helper module adapts the same planner interface to robot configurations by scoring and validating motion in task space while preserving joint-space feasibility.

`examples/multimodal_graph_search.py` is the reference task-space fixed-sequence multimodal evidence planner.

`examples/three_dof_robot_pyvista_demo.py` contains the Example 66.1 robot workflows:

- Example 66.1A, `--taskspace-planning`: task-space multimodal evidence planner plus IK robot execution of the selected route.
- Example 66.1B, `--jointspace-planning`: robot-aware constrained multimodal planning in q-space, with FK(q) constraints and dense joint-route certification.
- Example 66.1C, `--compare-taskspace-jointspace`: comparison mode that ranks task-space and joint-space candidates by robot executability.

For the robot demo, planner evidence is not robot motion. The planner may continue growing left-sphere, plane, and right-sphere evidence after the first feasible route; the robot execution layer receives only the selected final route. In task-space mode, IK is a post-processing/tracking layer. In joint-space mode, the planner explores robot configurations directly, and FK is used to visualize explored evidence in task space.

Joint-space route smoothing is a certified, opt-in post-processing layer. The normal demo keeps smoothing disabled so planning diagnostics appear immediately. When `--smooth-final-route` is used, smoothing is bounded by `--smoothing-time-limit` and `--smoothing-max-connector-calls`; if the smoothed joint path fails constraint, collision, or joint-step certification, the original certified dense joint route remains the source of truth.

Task-space planning remains the clearest way to inspect multimodal manifold evidence. Joint-space planning is closer to deployment, and can change route selection because robot feasibility, joint jumps, dense constraint residuals, and IK tracking quality become part of the explanation.

## Continuous Transfer

The continuous-transfer workflow keeps lambda-specific leaves locked once selected. This prevents physically invalid leaf jumping while still allowing parallel cross-family evidence accumulation and top-k certified route extraction.

## Validation

Recommended lightweight checks:

```powershell
.\.venv\Scripts\python.exe examples\multimodal_graph_search.py --help
.\.venv\Scripts\python.exe examples\multimodal_graph_search.py --fast --no-viz
.\.venv\Scripts\python.exe examples\multimodal_graph_search_stress_test.py --help
.\.venv\Scripts\python.exe examples\three_dof_robot_pyvista_demo.py --help
.\.venv\Scripts\python.exe examples\check_example66_robot_parity.py
.\.venv\Scripts\python.exe examples\three_dof_robot_pyvista_demo.py --jointspace-planning --without-obstacles --max-rounds 30 --joint-max-step 0.12 --no-smooth-final-route --no-viz
.\.venv\Scripts\python.exe examples\three_dof_robot_pyvista_demo.py --jointspace-planning --without-obstacles --max-rounds 30 --joint-max-step 0.12 --smooth-final-route --smoothing-iters 20 --smoothing-passes 1 --smoothing-time-limit 5 --no-viz
.\.venv\Scripts\python.exe examples\continuous_transfer_family.py --help
.\.venv\Scripts\python.exe -m pytest
```
