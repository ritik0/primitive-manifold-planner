# Primitive Manifold Planner

A Python research codebase for a robotics master's thesis on multimodal constrained trajectory planning.

The project studies how a planner can accumulate evidence on several constraint manifolds in parallel, detect valid transition regions, and then extract a certified route across a sequence of modes. The examples start with interpretable geometric constraints such as spheres and planes, then extend the same ideas toward robot joint-space planning and continuous transfer families.

## Repository Layout

`examples/` contains runnable entry-point scripts. These files are intentionally thin: they parse command-line arguments, build a scene, call package code, and optionally visualize the result.

`src/primitive_manifold_planner/manifolds/` defines reusable constraint manifolds: primitive geometric manifolds, masks/bounds, and robot task-space constraint manifolds.

`src/primitive_manifold_planner/thesis/` contains the thesis-focused Example 66-style planner code: parallel evidence stores, transition hypotheses, route extraction, and joint-space helpers.

`src/primitive_manifold_planner/scenes/` contains reusable scene builders for harder benchmark environments, including the multimodal stress-test scene.

`src/primitive_manifold_planner/visualization/` contains Matplotlib and PyVista display helpers for planner graphs, task-space routes, and robot demos.

`src/primitive_manifold_planner/experiments/` contains reusable experiment runners and the continuous-transfer planning modules used by Example 65-style workflows.

`tests/` contains regression tests for manifolds, projection, transitions, planners, and visualization helpers.

`notes/` contains thesis notes, method roadmaps, validation checklists, and architecture documentation.

## Main Workflows

Use the repository virtual environment when available:

```powershell
.\.venv\Scripts\python.exe examples\multimodal_graph_search.py --fast --no-viz
.\.venv\Scripts\python.exe examples\multimodal_graph_search_stress_test.py --help
.\.venv\Scripts\python.exe examples\three_dof_robot_pyvista_demo.py --help
.\.venv\Scripts\python.exe examples\continuous_transfer_family.py --help
```

If the package is installed into your active Python environment, the same commands can be run with `python` instead of the explicit venv path.

## Key Examples

`examples/multimodal_graph_search.py` runs the fixed left-sphere / plane / right-sphere parallel-evidence planner.

`examples/three_dof_robot_pyvista_demo.py` runs the same route-planning workflow with a simple 3DOF robot and PyVista visualization.

`examples/multimodal_graph_search_stress_test.py` runs a harder blocked-plane stress scene using the same planner interface.

`examples/continuous_transfer_family.py` runs the continuous-transfer family planner with lambda-locked leaves and optional top-k route reporting.

## Development Checks

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Some visualization and robot demos require optional dependencies such as OMPL, PyVista, scipy, or trimesh. Missing optional dependencies should be treated as environment issues, not planner-logic failures.

## More Context

See [notes/05_thesis_method_roadmap.md](notes/05_thesis_method_roadmap.md), [notes/06_thesis_validation_checklist.md](notes/06_thesis_validation_checklist.md), and [notes/07_codebase_architecture.md](notes/07_codebase_architecture.md) for the thesis method direction and streamlined code workflow.
