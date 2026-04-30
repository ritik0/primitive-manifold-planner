# Primitive Manifold Planner

A research-oriented Python framework for planning on and between primitive implicitly defined manifolds.

## Goal

This project studies motion planning on constraint manifolds in the simplest possible setting.

Instead of starting with robot kinematics, the framework focuses on primitive geometric manifolds such as circles, spheres, planes, and cylinders. The main objective is to develop and compare general methods for:

1. Planning on a single smooth manifold
2. Planning across multiple manifolds
3. Finding and using manifold intersections as transitions
4. Extending the setup to parameterized manifold families (leaf families / foliations)

The project is intentionally designed to stay as general and minimal as possible before introducing robot-specific structure.

---

## Motivation

Many constrained motion planning problems can be understood as motion in lower-dimensional spaces embedded in a higher-dimensional ambient space.

Examples:
- A point constrained to a circle in 2D
- A point constrained to a sphere in 3D
- Motion restricted to a plane
- Switching from one manifold to another through their intersection

This project builds the mathematical and algorithmic foundations for such problems first, using primitive examples, before applying similar ideas to robotic systems.

---

## Project Roadmap

The project is developed in the following order:

### Stage 1 — Single Smooth Manifold
Examples:
- circle in 2D
- sphere in 3D
- plane in 3D
- cylinder in 3D

Goal:
- define manifolds implicitly
- project points onto them
- plan local and global paths constrained to one manifold

### Stage 2 — Multiple Manifolds
Examples:
- sphere + plane
- sphere + cylinder
- two spheres

Goal:
- detect whether intersections exist
- compute transition samples
- plan across manifold switches

### Stage 3 — Transition-Based Planning
Goal:
- build a graph of manifolds
- connect manifolds through valid transition regions
- perform graph search over manifold sequences
- solve continuous path segments inside each manifold

### Stage 4 — Parameterized Leaf Families
Examples:
- concentric circles
- parallel planes
- variable-radius spheres

Goal:
- represent a family of manifolds indexed by a parameter lambda
- study planning on one leaf versus planning across multiple leaves

### Stage 5 — Toward Foliation-Inspired Planning
Goal:
- understand how a planner can reason over families of manifolds
- compare discrete leaf selection with continuous augmented-state formulations

---

## Core Mathematical View

A manifold is represented implicitly using equality constraints:

M = { x in R^n : h(x) = 0 }

Optionally, valid regions may also include inequality constraints:

M = { x in R^n : h(x) = 0, g(x) <= 0 }

For a family of manifolds:

M_lambda = { x in R^n : h(x, lambda) = 0, g(x, lambda) <= 0 }

The framework focuses on the following primitives:

- residual evaluation
- Jacobian evaluation
- projection to the manifold
- tangent-space computation
- constrained local planning
- transition detection between manifolds

---

## Design Principles

- Keep geometry primitive and interpretable
- Separate mathematical definitions from planning algorithms
- Avoid robot-specific assumptions
- Use small, testable modules
- Prefer explicit understanding over premature complexity

---

## Planned Algorithms

The framework will support and compare:

- projection-based manifold adherence
- local constrained interpolation
- constrained RRT-style planning
- intersection-based transition search
- graph search over manifolds
- later: tangent-space / atlas-based local planning

For the current thesis-oriented method direction and validation priorities, see:

- [notes/05_thesis_method_roadmap.md](C:/dev/mmp/primitive-manifold-planner/notes/05_thesis_method_roadmap.md)

---

## Initial Dependencies

- numpy
- scipy
- matplotlib

Optional later:
- networkx
- pytest
- plotly or pyvista

---

## First Milestone

Implement a single-manifold pipeline for a circle and a sphere:
- implicit residual
- Jacobian
- projection
- constrained local step
- simple visualization

---

## Long-Term Direction

After validating all methods on primitive manifolds, the same architecture can later be reused for more structured constrained planning problems, including robotic systems.

The emphasis, however, is on getting the geometric and algorithmic foundations correct first.
