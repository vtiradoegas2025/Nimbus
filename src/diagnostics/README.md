# Diagnostics and scientific inventories

Diagnostics validate fields, calculate derived products and account for
accepted state changes. They are services rather than alternative physical
closures. The [mathematics guide](../../docs/MATHEMATICS.pdf) distinguishes a
closed accounting residual from a validated physical energy transfer.

`contracts` owns field membership, bounds and validation. Query the registry
and its tests rather than copying a historical field count into a new tool.
`budgets` owns dry mass, water, momentum and versioned energy inventories.
`derived` owns physical gradients, pressure decomposition and other products;
`trajectories` owns parcel integration and continuation state.

Energy basis v2 uses dry-air water mixing ratios, mixture-mass resolved kinetic
and gravitational energy, dry-carrier subgrid TKE, and physical C-grid face
inventories including the mapped ground contribution. The legacy proxy is
retained separately. Do not compare quantities with different reference states
or call a resolved energy change numerical drift without examining its sources,
boundary exchanges and storage work.

Mapped derivatives and reference profiles use physical height. Exported
buoyancy matches the solver's density/vapor/loading convention. SRH uses signed
Earth/storm-relative hodograph segments; UH uses the declared physical AGL
layer and available cell overlap. Missing vertical coverage must not silently
be represented as a complete layer integral.

Output snapshots own selected diagnostic maps. Parcel integration and domain
recycling must preserve their time/origin ordering. The maintained suites under
`tests/diagnostics`, `tests/core` and `tests/dynamics` exercise these contracts;
archived campaign receipts are not runtime input dependencies.

The standalone `bin/field_validator` reads exported fields and their declared
contracts. `make test-diagnostics` checks the current implementation; successful
bounded tests do not by themselves establish campaign validation or independent
scientific review.
