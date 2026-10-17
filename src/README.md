# Solver source

Nimbus separates the simulation lifecycle, numerical operators, physical
closures, diagnostics and compute backends. The public interfaces live under
[`include/`](../include/README.md); built-in implementations live here.
[Architecture](../docs/ARCHITECTURE.pdf) explains ownership and execution, and
[Mathematics](../docs/MATHEMATICS.pdf) explains equations and assumptions.

`core/runtime` owns configuration, state and the headless loop.
`core/orchestration` composes operators into accepted model steps. Physical
modules own their closures, cadences and persistent private state; `numerics`
owns transport, diffusion, integration and stability policy. `compute`, `vulkan`
and `metal` implement the declared execution contracts without defining a
separate scientific model merely because memory is on a different device.

A factory entry does not establish public support. Schema parsing, runtime
projection, construction, native capability, checkpoint state and exact-profile
admission must agree. The authoritative status is the executable's
`--scheme-catalog=json` output and
[`core/runtime/readiness.cpp`](core/runtime/readiness.cpp). Preserve guarded
unfinished capabilities and distinguish bounded testing from campaigns and
independent scientific review.

When adding an operator, state its units, physical domain, conserved inventory,
source/destination ownership and failure behavior first. Extend the existing
module interface, explicitly register build inputs, and test the independent
reference, invalid inputs, rollback, continuation and applicable backends.
Do not infer a new module's lifecycle from a neighboring factory alone.

Use [`docs/BUILDING.md`](../docs/BUILDING.md) for build instructions. Generated
objects, local timing runs and historical qualification receipts are not source
dependencies. Permanent reference data and regression fixtures belong in their
maintained data/test locations.
