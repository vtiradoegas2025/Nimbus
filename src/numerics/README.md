# Numerical operators

This module owns scalar transport, diffusion, time integration and timestep
policy. [Mathematics](../../docs/MATHEMATICS.pdf) follows the equations from their
physical inventories to the grid; [Architecture](../../docs/ARCHITECTURE.pdf)
explains the runtime ownership.

`advection` contains TVD/MC and WENO5 reconstruction and the Cartesian transport
owners. Production conservative scalar recovery uses continuity's retained
beginning density and interval mass flux. A mixing ratio must not be advanced
with an unrelated density update. Private fields are converted according to
their declared storage units before the same transport contract is applied.

`diffusion` contains explicit horizontal/vertical flux-form operators and
vertical implicit backward-Euler solves. Density, physical spacings and
cell/face control volumes participate in the fluxes and stability limits.
The implicit solve does not remove advective/acoustic constraints or make a
large timestep accurate. Boundary-owned closure planes are not extra cells.

`time_stepping` contains split-explicit acoustic integration, SSPRK3 and RK4.
The complete coupled callback state includes primary fields, private moments,
held sources and reports. Invalid intermediate stages reject transactionally.
`runtime.cpp` derives timestep restrictions from the live flow, acoustic speed,
diffusion and physical metrics rather than a fixed assumed sound speed.

Scheme presence does not admit every grid, boundary, physics or backend
combination. Consult ordinary strict configuration admission and the compiled
readiness registry. The permanent `tests/numerics` suite covers conservative
transport, diffusion, timestep selection and integration; coupled dynamics and
physics tests are also required when stage ordering changes.
