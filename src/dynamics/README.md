# Resolved atmospheric dynamics

Nimbus has separate collocated A-grid and staggered C-grid Cartesian
implementations of the compressible, nonhydrostatic dynamics. Terrain mapping,
rotation, boundary conditions and backend ownership are independently constrained
by the selected profile. Do not infer support for a cross-product from an
available scheme name.

The built-in implementations live under `schemes/klemp_wilhelmson_split`.
`factory.cpp` and `runtime.cpp` own construction/lifecycle; core dynamics
orchestration owns complete phase ordering and integration. The C-grid stores
normal wind components on faces, while the A-grid is a separate collocated
operator, not a reinterpretation of those locations.

Pressure uses the dry rho–theta EOS. The vertical tendencies combine a
perturbation-pressure/density force with the declared vapor-buoyancy and
condensate-loading approximations. An extra theta-buoyancy correction would
count part of the response twice. Conservative transport and pressure work
must use the actual physical cell/face domains and the declared carrier.
See [Mathematics](../../docs/MATHEMATICS.pdf) for the equations and limitations.

Split-explicit acoustic stepping, coupled SSPRK3/RK4, boundary closure, held
physics and native execution have different ownership contracts. Changing
one stage can alter the whole-step result even if a standalone tendency still
matches. The permanent suites under `tests/dynamics`, `tests/numerics` and
`tests/diagnostics` cover operators, coupled sequences, rollback and parity.
The current compiled catalog is the authority for ordinary admission; a bounded
profile is not a universal qualification of all dynamics combinations.
