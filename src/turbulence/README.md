# Resolved-grid turbulence closures

This module represents unresolved stress and scalar transport on the resolved
grid. Smagorinsky and prognostic TKE are different closure models; they are
also distinct from a boundary-layer column scheme. Their combination requires
an explicitly compatible profile. [Mathematics](../../docs/MATHEMATICS.pdf)
explains the filter, strain, buoyancy and energy assumptions.

The standard Smagorinsky branch uses strain-dependent eddy viscosity. Dynamic
and stability-corrected variants have separate selection and averaging
contracts; the dynamic coefficient is bounded without SGS backscatter.
A configured physical filter width changes the closure rather than simply
changing output resolution.

The TKE branch uses Deardorff-Moeng-style mixing length, shear/buoyancy
production, dissipation and transport. Its prognostic TKE is per dry carrier.
Stable mixing length and scalar transport ratios must be evaluated consistently
with the selected correction. Coefficients and diagnostic production fields
are staged with the accepted tendencies.

The C-grid tensor/flux locations, physical spacings and live-density control
volumes are part of the equations. Collocated and mapped operators require their
own contracts. Boundary work and the relation between stress work and TKE
production must survive changes to interpolation, threading or batching.
Native-required execution cannot silently return through a CPU substitute.

`runtime.cpp` owns cadence/state; `base` contains common coefficient and
transport machinery; `schemes` contains the closures. The permanent SGS,
topology and runtime-state tests under `tests/physics` are the starting point
for changes. Consult the compiled catalog for the supported exact profiles;
historical diary IDs and timing results are not admission rules.
