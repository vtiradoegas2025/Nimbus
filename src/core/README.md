# Simulation orchestration

Core owns the simulation lifecycle, storage, cross-module phase order,
checkpointing and output. It composes scientific operators; it is not a
selectable physics closure. [Architecture](../../docs/ARCHITECTURE.pdf) describes
its ownership model and [Mathematics](../../docs/MATHEMATICS.pdf) explains the
quantities exchanged between operators.

`runtime/nimbus_sim.cpp` is the process entry point; `headless_runtime.cpp`
advances accepted model steps. The typed configuration path and runtime
projection establish the selected profile. `simulation_state.cpp` provides
views over caller-owned storage, while compatibility aliases remain separate.
`simulation_checkpoint.cpp` owns versioned, checked and transactional restore.

`orchestration/dynamics/dynamics.cpp` owns acoustic/slow-stage composition and
physics coupling. `orchestration/physics/microphysics_step.cpp` owns source
application and its private-state/report transaction. Module runtimes retain
their own closure state and cadences. A failed stage must not leave a mixture
of old fields and new private state, clocks, precipitation or ledgers.

`infra` owns physical geometry, nesting, remap/recycling and shared support.
Parent/child fluxes, held sources, origins and private state are part of the
same accepted domain transaction. Public support is profile-specific; source
presence does not admit arbitrary moving terrain or AMR combinations.

`output` publishes immutable field/geometry snapshots and indexes;
`io` supplies shared readers. Output and diagnostic work may be asynchronous,
but retained snapshots must own their data and record the correct generation.
The public contracts are under `include/core`; implementation changes should
preserve their copy, rollback, restart and native-execution semantics.
