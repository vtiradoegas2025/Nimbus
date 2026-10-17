# Terrain and mapped geometry

This module supplies topography, physical heights, Jacobians and conservative
face geometry. Built-in shapes include flat, Bell and Schär terrain. A mapped
coordinate is part of the numerical operator; it is not a visual displacement
of otherwise Cartesian fields.

`base/topography.*` builds shapes and metrics;
`base/conservative_transport.cpp` owns metric-consistent flux operations;
`runtime.cpp` and `factory.cpp` own configuration and lifecycle. Conservative
face metrics derive from shared interface geometry so related discrete
operators use consistent areas and volumes.

Terrain affects pressure gradients, physical reference-profile sampling,
scalar/momentum fluxes, sedimentation thickness, surface traction and energy
inventories. Zero normal flow at sloping ground can have a nonzero physical
vertical velocity. Mapped C-grid energy therefore includes the diagnosed
boundary-face half-mass contribution.

Grid-attached terrain and Earth-fixed terrain have different motion semantics.
Relocation/recycling must update geometry, state, held sources, origins and
ledgers together. Nesting adds parent/child interface and feedback contracts.
The [mathematical guide](../../docs/MATHEMATICS.pdf) explains these geometric
and domain assumptions.

Support is exact-profile specific across staggering, motion, nesting, physics
and backend. It is no longer accurately described by a blanket CPU-only terrain
rule, and the existence of a native terrain kernel does not imply all such
combinations are admitted. The compiled readiness registry and strict guards
are the authority. Permanent geometry, mapped reconstruction, nested production
and energy tests cover their declared scopes; broader combinations remain
Engineering incomplete.
