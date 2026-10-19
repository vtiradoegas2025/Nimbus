# Public C++ interfaces

These headers define the contracts shared by solver modules, orchestration,
compute backends and supported extensions. Built-in algorithm details normally
remain under `src/`. See [the source overview](../src/README.md) and
[architecture guide](../docs/ARCHITECTURE.pdf).

`core/field` defines field storage; `core/infra` defines physical domains,
geometry and constants; `core/runtime` defines owning storage and explicit
state views. New orchestration should receive a `SimulationStateView` rather
than invent another global owner. Compatibility bindings remain separate from
those explicit interfaces.

The physics directories expose scheme, configuration and runtime contracts.
`numerics` defines transport/integration interfaces, `compute` defines native
operation/resource ownership, and `diagnostics` defines versioned inventories
and output contracts. A public header is not an assertion that every schema,
backend or coupling combination is admitted.

Document units and representation at interfaces: dry density versus mixture
mass, mixing ratio versus volume concentration, cell versus face location,
physical cells versus closure storage, and caller versus retained buffers.
The [mathematics guide](../docs/MATHEMATICS.pdf) explains why these distinctions
change the equations. Any new persistent state also needs an explicit copy,
rollback and checkpoint contract.
