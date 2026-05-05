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

## Continuous Transfer

The continuous-transfer workflow keeps lambda-specific leaves locked once selected. This prevents physically invalid leaf jumping while still allowing parallel cross-family evidence accumulation and top-k certified route extraction.

## Validation

Recommended lightweight checks:

```powershell
.\.venv\Scripts\python.exe examples\multimodal_graph_search.py --help
.\.venv\Scripts\python.exe examples\multimodal_graph_search.py --fast --no-viz
.\.venv\Scripts\python.exe examples\multimodal_graph_search_stress_test.py --help
.\.venv\Scripts\python.exe examples\three_dof_robot_pyvista_demo.py --help
.\.venv\Scripts\python.exe examples\continuous_transfer_family.py --help
.\.venv\Scripts\python.exe -m pytest
```
