# Boundary conditions

This module enforces domain-edge conditions at the locations owned by the
selected grid. The factory dispatches to collocated or C-grid implementations;
the user selects compatible boundary types. Those types are physical modeling
choices: periodic, rigid, zero-gradient and radiative boundaries describe
different exchanges with the exterior.

The public contracts are in `include/boundary_conditions`; `runtime.cpp` owns
application and the Cartesian implementation files own the edge stencils.
The compatibility spelling `open` is resolved by strict loading rather than
being treated as one universal runtime equation. The admissible boundary/profile
cross-product is checked by configuration and runtime guards.

Boundary conditions are distinct from atmospheric boundary-layer mixing.
A geometric boundary constrains edge values or fluxes; a PBL closure supplies
physical column and surface tendencies. On sloping ground, impermeability
means zero relative normal flux, not necessarily zero physical vertical wind.
See the [mathematical guide](../../docs/MATHEMATICS.pdf) for these geometric
assumptions.

Acoustic staging must apply the appropriate closures between dependent updates.
Boundary-owned faces and halo storage are not extra prognostic cells. Preserve
this ownership when adding a kernel or changing threading. Permanent checks
include [`test_cartesian_boundary_conditions.cpp`](../../tests/dynamics/test_cartesian_boundary_conditions.cpp)
and [`test_native_cgrid_boundary.cpp`](../../tests/dynamics/test_native_cgrid_boundary.cpp).
