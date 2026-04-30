# Thesis Method Roadmap

## Purpose

This note defines the current thesis-method direction after the recent framework upgrades.

The goal is no longer only to build more examples. The goal is to make clear:

- what the planner method now *is*
- what it can reasonably *claim*
- what still needs validation
- what should remain future work rather than being overclaimed

---

## Current Method Identity

The framework has moved beyond simple manifold-switching demos.

The current method is best described as:

**A multimodal constrained planning framework that searches over component-aware manifold leaves and realizes routes through explicit, semantically filtered transition structures.**

More concretely, the planner now contains:

- implicit manifold representations
- projection-based constraint satisfaction
- local constrained motion on a leaf
- exact transition discovery between leaves
- connected-component-aware leaf reasoning
- transition candidate caching and ranking
- nominal route vs realized transition distinction
- bounded connector semantics
- admissibility costs
- feasibility filtering
- unified semantic models over mode roles
- reusable semantic templates
- initial 3D-compatible planning structure

This is already a thesis-worthy method core if the claims are framed correctly.

---

## Strong Thesis Claims

The framework can now support the following claims with a straight face:

### Claim 1

**Multimodal constrained planning benefits from explicit connector semantics instead of implicit branch jumping.**

Why this is now strong:

- branch changes are no longer accidental
- connector leaves are explicit planning objects
- semantic feasibility and admissibility can restrict or bias their use

### Claim 2

**Component-aware leaf reasoning is important when manifold leaves are disconnected.**

Why this is now strong:

- start and goal can be assigned to discovered leaf components
- graph nodes are effectively `(family, lambda, component)`
- route search can succeed or fail for meaningful topological reasons

### Claim 3

**Exact transition generation should be separated from route realization.**

Why this is now strong:

- the planner stores multiple exact candidates
- the nominal route selects node pairs
- realization may choose a different reachable exact candidate
- this behavior is now explicit rather than accidental

### Claim 4

**Semantic filtering should sit on top of geometric transition generation.**

Why this is now strong:

- geometry determines what transitions exist
- semantics determines what transitions are allowed or preferred
- feasibility and admissibility are now distinct ideas in the framework

### Claim 5

**The planner architecture can be prepared for richer 3D state spaces before full robot models are introduced.**

Why this is now reasonable:

- the local planner and transition logic are already dimension-agnostic
- 3D pilot examples exist
- semantic and connector abstractions survive the dimensional jump

---

## Claims To Avoid Overstating

The framework is stronger now, but the thesis should still avoid claiming:

- full probabilistic completeness for multimodal transition discovery
- robust high-dimensional constrained planning in the general case
- robot-ready manipulation planning with realistic torque/collision semantics
- automatic semantic discovery from problem structure
- atlas-level coverage guarantees

These are better framed as:

- future work
- partial architectural readiness
- initial stepping stones

---

## Validation Priorities

The next phase should validate the method, not just add infrastructure.

### Priority A: Structural Validation

Show that the method behaves differently and more correctly when:

- connected components are respected
- connectors are explicit
- semantics restrict transitions
- semantics bias transitions without forbidding them
- bounded connectors are treated as finite admissible objects

### Priority B: Route Semantics Validation

Show that:

- the discrete route is selected at the node-pair / leaf level
- realization can reselect the exact transition candidate
- this reselection is meaningful and not a bug

### Priority C: Generalization Validation

Show the same architecture works across:

- 2D primitive families
- component-aware 2-link joint-space examples
- bounded support-transfer-support examples
- minimal 3D state-space pilots

The exact geometry may differ, but the method structure should stay the same.

---

## Recommended Validation Set

The thesis does not need many examples if each one validates a different method point.

Recommended core set:

1. **Component-aware 2-link benchmark**

Use:

- `examples/example_55_auto_component_discovery_demo.py`

Validates:

- automatic component discovery
- component-aware routing
- explicit switch connector semantics
- semantic model integration

2. **Bounded support-transfer-support benchmark**

Use:

- `examples/example_56_two_table_transfer_zone.py`

Validates:

- bounded connector semantics
- semantic feasibility
- semantic admissibility
- support -> transfer -> goal support structure

3. **Minimal 3D pilot**

Use:

- `examples/example_58_joint_space_3d_transfer_pilot.py`

Validates:

- dimension-agnostic planner structure
- semantic and bounded connector concepts surviving in 3D state space

This is enough for a strong method chapter if discussed carefully.

---

## Missing Method Pieces

These are the main remaining research gaps before claiming a stronger general planner.

### 1. Transition Discovery Robustness

Current status:

- exact intersections are found by seed + projection + deduplication
- adaptive seeding and component-aware anchors now help a lot

Still missing:

- stronger local exploration near difficult intersections
- continuation along intersection structure
- more systematic discovery in narrow or higher-dimensional cases

### 2. Bounded Subset Semantics

Current status:

- bounded connectors are much better represented now

Still missing:

- a first-class abstraction for “valid subset of a manifold”
- not only manifold-specific `within_bounds(...)` or connector-local logic

### 3. Stateful Semantic Models

Current status:

- semantic models can express role-based permission, feasibility, and cost

Still missing:

- richer state dependence
- semantics that depend on current planning context, previous support, or mode history

### 4. Atlas / Chart-Based Coverage

Current status:

- there is already an `atlas_like` local planner direction

Still missing:

- systematic chart coverage
- a stronger answer to hard manifold exploration in higher dimensions

### 5. Robot-Specific Feasibility

Current status:

- the framework is ready for semantically filtered transfer logic

Still missing:

- real robot kinematics/dynamics semantics
- collision checks
- torque or wrench feasibility models
- grasp/contact state modeling

---

## Recommended Next Development Order

The next development steps should stay method-centered.

### Step 1

**Stabilize the semantic-model architecture**

Meaning:

- use semantic templates wherever possible
- reduce remaining example-local semantic wiring
- ensure planners consume one semantic interface consistently

### Step 2

**Improve transition discovery in a more geometric way**

Meaning:

- local transition continuation
- richer seed refinement
- better component-pair-local search

### Step 3

**Introduce bounded subset semantics as a reusable abstraction**

Meaning:

- distinguish manifold geometry from valid region on that manifold

### Step 4

**Start one real low-DOF robot-style 3D benchmark**

Meaning:

- not just synthetic 3D state manifolds
- a small actual configuration-space example with meaningful semantic constraints

A useful validation experiment for this benchmark is a serial-vs-parallel evidence ablation. The scene stays the same fixed left-sphere -> plane -> right-sphere problem, but the planner is run once in its default parallel evidence mode and once in a serial mode where only one underexplored stage is expanded per round. The expected outcome is that the parallel mode should accumulate transition hypotheses and committed route structure faster, improving either success rate, planning time, or both, while the serial mode serves as a clean baseline showing what is lost when concurrent evidence growth is removed.

---

## Suggested Thesis Framing

A good framing for the thesis is:

> This work develops a primitive multimodal manifold planning framework in which discrete routing, connected-component structure, exact transition generation, and semantic connector handling are treated as separate but interacting layers. The objective is not to solve full robot manipulation in general, but to establish a methodologically clean and extensible planning architecture that can later support richer robotic state spaces and 3D problems.

That framing is honest and strong.

---

## Immediate Working Rule

From this point on, new code changes should answer one of these questions:

- Does this make the semantic layer cleaner or more reusable?
- Does this improve transition-discovery robustness?
- Does this improve bounded-subset semantics?
- Does this make a future low-DOF 3D robot benchmark easier to express?

If not, it is probably not a priority for the thesis method anymore.

For a concrete mapping from method claims to scripts and expected observations, see:

- [06_thesis_validation_checklist.md](C:/dev/mmp/primitive-manifold-planner/notes/06_thesis_validation_checklist.md)
