# Atmospheric boundary-layer closures

This module owns column mixing, surface exchange and their held runtime
sources. It is separate from geometric boundary conditions and from the
resolved-grid SGS operator. Combining those closures requires a compatible
profile rather than an assumption that their tendencies can always be added.
The [mathematics guide](../../docs/MATHEMATICS.pdf) explains the modeling levels.

The selectable families are disabled, slab, YSU and MYNN. Slab retains a mixed
layer; YSU includes nonlocal transport and entrainment; MYNN uses prognostic TKE
and stability functions. Scheme selection does not establish support for every
terrain, cloud, cadence or backend combination. Consult the compiled readiness
registry and ordinary configuration admission.

`config_validation.cpp` owns the common initialization/restore contract.
`column_evaluation.cpp` validates physical columns and converts completed
results into publication units. `native_batch_runtime.cpp` owns bounded native
column submission/finalization, and `runtime.cpp` publishes the complete refresh
transaction. Face momentum forcing uses the appropriate live dual masses;
closure storage is not a physical atmospheric column.

The implicit column solves publish realized interval-mean fluxes, not merely
the requested coefficients. A failed refresh must preserve previous held
sources and diagnostics. Checkpoint continuity includes the state needed to
resume the cadence. Native-required execution must not silently substitute a
CPU evaluation.

See [`test_boundary_layer.cpp`](../../tests/physics/test_boundary_layer.cpp)
and [`test_boundary_layer_runtime_state.cpp`](../../tests/physics/test_boundary_layer_runtime_state.cpp)
for reference columns, closure, rejection, threading and runtime ownership.
