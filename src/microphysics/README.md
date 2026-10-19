# Bulk cloud microphysics

This module owns water-category sources, particle moments and sedimentation.
The selectable families include Kessler warm rain, Lin mixed phase, Thompson
and ordinary Milbrandt. Their category, spectrum, activation and collection
assumptions differ. [Mathematics](../../docs/MATHEMATICS.pdf) explains the dry-air
mixing-ratio basis, spectrum moments and source/EOS limitations.

`runtime.cpp` owns scheme initialization. Core's
`orchestration/physics/microphysics_step.cpp` owns accepted application to primary
fields and the accompanying private-state, stochastic, precipitation and report
transaction. Private particle fields explicitly declare per-dry-mass or
per-volume storage; transport must use the corresponding conserved inventory.

Sedimentation uses actual physical layer thickness and records bottom fallout.
Source and sedimentation subcycling respect the scheme's live stability and
donor limits. A failed interval must restore primary/private fields and reports.
Nonnegative water alone is not proof of conserved water, heat or realizable
particle moments. Repair/report policy is explicit; do not add a silent
zero-tendency fallback to strict execution.

Thompson startup requires pinned activation, collection and cold-process data;
see [its data contract](schemes/thompson/README.md). Required native operation
ownership is checked separately from CPU implementation availability.

Public Milbrandt triple-moment selection remains **Engineering incomplete**.
Its internal moment, wet-surface, original-vapor and source/EOS helpers are
bounded development owners. They do not complete the live full cold composition,
all transitions, native paths or public admission. Ordinary double behavior and
its collision policies remain separate. Permanent tests under `tests/physics`
and reference headers under `tests/support` exercise the declared assumptions
without depending on local build experiments or archived reports.
