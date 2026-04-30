# Thesis Validation Checklist

## Purpose

This note turns the thesis-method roadmap into a concrete validation checklist.

Each item below links:

- a method claim
- the most relevant script(s)
- the expected observation
- what would count as weakness or failure

The point is to keep validation aligned with the method rather than collecting many loosely related examples.

---

## Claim A

**Explicit connector semantics improve multimodal planning compared with implicit branch changes.**

Primary scripts:

- `examples/example_55_auto_component_discovery_demo.py`
- optionally earlier planner-v1 benchmark scripts if needed for comparison

Expected observation:

- routes that require branch change succeed only when explicit connector structure is present
- the route passes through the connector mode rather than “jumping” directly between disconnected supports
- the selected route string clearly reflects connector use

What would count as weakness:

- the planner succeeds equally well without connector structure
- branch change appears to happen implicitly without meaningful connector semantics
- the connector exists in the code but does not materially affect routing

Useful things to record:

- route found / not found
- route string
- path length change when connector is required
- semantic-route diagnostics if relevant

---

## Claim B

**Connected-component-aware reasoning is necessary on disconnected leaves.**

Primary scripts:

- `examples/example_55_auto_component_discovery_demo.py`

Expected observation:

- start and goal are assigned to discovered components
- routes are possible or impossible for topological reasons tied to component structure
- the planner behaves differently from a component-agnostic version

What would count as weakness:

- component labels are discovered but do not change planning behavior
- routes ignore disconnectedness
- start/goal component inference is unstable or obviously inconsistent

Useful things to record:

- inferred start component
- inferred goal component
- whether route exists
- whether route failure/success is explained by component structure

---

## Claim C

**Geometric transition existence should be separated from semantic feasibility and semantic preference.**

Primary scripts:

- `examples/example_56_two_table_transfer_zone.py`

Expected observation:

- a geometrically valid connector region exists
- semantic feasibility restricts which transition points are admissible
- semantic admissibility biases which connector usage is preferred
- the planner still finds a route when the semantic window is reasonable

What would count as weakness:

- feasibility and admissibility have no visible effect
- the planner succeeds only because semantic constraints are too loose
- the planner fails under slight semantic filtering because the method is too brittle

Useful things to record:

- success/failure under different semantic windows
- chosen connector leaf
- path shape through the connector
- whether narrowing the admissible region changes the route or invalidates it

---

## Claim D

**Bounded connectors should be treated as finite admissible planning objects, not only as visual trims of infinite manifolds.**

Primary scripts:

- `examples/example_56_two_table_transfer_zone.py`

Expected observation:

- switching and local continuation happen only on the bounded connector
- the path does not continue indefinitely along an unbounded surrogate manifold
- boundedness affects planning behavior, not only plotting

What would count as weakness:

- bounded connectors only change the visualization
- realization continues outside the intended connector region
- transition discovery ignores the bounded semantics in practice

Useful things to record:

- selected connector leaf
- transition locations
- whether any failure occurs when boundedness is tightened

---

## Claim E

**Nominal route choice and realized transition choice are distinct and should be reported separately.**

Primary scripts:

- `examples/example_55_auto_component_discovery_demo.py`
- planner result diagnostics through `multimodal_component_planner`

Expected observation:

- discrete search selects a nominal route over node pairs
- realization may reselect a different exact transition candidate for the same pair
- this reselection is visible in diagnostics

What would count as weakness:

- realized transition always equals nominal only because there is effectively one candidate
- reselection happens but is not observable in outputs
- reselection changes semantics in a confusing or inconsistent way

Useful things to record:

- nominal candidate indices
- realized candidate indices
- whether deviations occurred
- interpretation of why deviations happened

---

## Claim F

**The planner architecture generalizes from 2D to 3D state spaces without changing its core logic.**

Primary scripts:

- `examples/example_57_minimal_3d_transfer_pilot.py`
- `examples/example_58_joint_space_3d_transfer_pilot.py`

Expected observation:

- the same core planning structure works in 3D
- explicit support/transfer/support semantics survive the jump
- bounded and semantic connector ideas still make sense

What would count as weakness:

- 3D scripts require a different planner logic than 2D
- only the visualization changes, while method structure silently breaks
- 3D examples succeed only because they are trivial and do not exercise transitions meaningfully

Useful things to record:

- route success
- leaf sequence
- transition sequence
- whether semantics and boundedness remain active in 3D

---

## Secondary Validation Items

These are useful if time allows, but they are not necessary for a strong main method story.

### Item 1

**Competing connectors should produce semantically different routes.**

Candidate scripts:

- future extension of the 2D bounded transfer case
- future 3D competing-connector case

Expected observation:

- multiple connectors are geometrically possible
- semantics selects one over the others

### Item 2

**Disconnected components on 3D leaves should still be discoverable and usable.**

Candidate scripts:

- future dedicated 3D component-discovery example

Expected observation:

- component discovery still works in 3D
- planner decisions depend on discovered components

### Item 3

**Transition reuse and ranking should improve repeated planning queries.**

Candidate scripts:

- planner-v1 benchmark scripts

Expected observation:

- cache hits increase on repeated related queries
- attempt statistics start influencing candidate ordering

---

## Minimal Thesis Validation Set

If the thesis needs a compact experimental core, the minimal set is:

1. `example_55_auto_component_discovery_demo.py`
   Supports:
   - explicit connector semantics
   - component-aware planning
   - nominal vs realized transition semantics

2. `example_56_two_table_transfer_zone.py`
   Supports:
   - bounded connectors
   - semantic feasibility
   - semantic admissibility

3. `example_58_joint_space_3d_transfer_pilot.py`
   Supports:
   - 3D-compatible architecture
   - semantic and bounded transfer structure in 3D state space

This is enough for a clean and honest thesis method section if discussed carefully.

---

## Suggested Output Table Structure

When you later summarize results, a compact structure could be:

- Example / scenario
- Main method property being validated
- Route found
- Realization success
- Component-aware needed?
- Explicit connector needed?
- Semantic feasibility active?
- Bounded connector active?
- 3D state space?
- Key observation

This keeps the results focused on method claims rather than raw benchmark volume.

---

## Failure Interpretation Guide

If a validation case fails, classify it before changing code.

### Case 1: Geometric failure

Meaning:

- transition does not exist
- manifold projection fails
- no local constrained continuation

Interpretation:

- likely a transition-generation or geometry issue

### Case 2: Topological failure

Meaning:

- disconnected components prevent route existence

Interpretation:

- often a valid and informative method outcome

### Case 3: Semantic overconstraint

Meaning:

- route existed geometrically but semantic feasibility blocked everything

Interpretation:

- not necessarily a planner bug
- may indicate an overly strict semantic rule or a genuinely infeasible scenario

### Case 4: Realization mismatch

Meaning:

- discrete route exists but realization cannot execute it as selected

Interpretation:

- useful evidence for why nominal vs realized transition semantics matter

---

## Immediate Next Work Pattern

For every next experiment or change, answer:

1. Which claim from this checklist does it support?
2. What observation do we expect?
3. What failure would be informative?
4. Does it strengthen the method, or only add another example?

If those answers are not clear, the task is probably not a priority for the thesis.
