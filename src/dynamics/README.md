# Atmospheric Dynamics Module

This module provides the dynamics-scheme layer used by the core runtime.

## Purpose

`src/dynamics/` supplies scheme implementations for momentum tendencies and diagnostic fields. The runtime coordinator in `src/core/orchestration/dynamics/dynamics.cpp` owns the timestep coupling, while this module provides scheme-specific calculations.

## Layout

```text
src/dynamics/
├── factory.cpp/.hpp
└── schemes/
    ├── cartesian/
    ├── supercell/
    └── tornado/

src/core/orchestration/dynamics/dynamics.cpp   # runtime coordinator and coupling
include/dynamics/dynamics_base.hpp             # scheme interface
```

## Supported Schemes

- `tornado` — axisymmetric cylindrical; default when `dynamics.scheme` is unset.
- `supercell` — non-axisymmetric cylindrical.
- `cartesian` — Cartesian (x, y, z) backend. Required for non-axisymmetric
  hodographs (e.g. WK2002 supercell). Supports both CPU and GPU compute paths
  with dedicated Cartesian acoustic shaders.

Alias handling (factory canonicalization):
- `axisymmetric` -> `tornado`
- `mesocyclone` -> `supercell`
- `cart` / `cartesian_cpu` -> `cartesian`

If `dynamics.scheme` is not set, runtime defaults to `tornado`.

## Runtime Coupling

Main coupling points:
- `initialize_dynamics(...)` initializes the selected scheme.
- `step_dynamics(simulation_time)` applies coordinated dynamics updates.
- `src/core/runtime/headless_runtime.cpp` calls `step_dynamics(...)` each model step.

`src/core/orchestration/dynamics/dynamics.cpp` also couples in:
- microphysics tendencies
- turbulence SGS tendencies
- boundary-layer tendencies
- chaos perturbation hooks
- terrain-aware metric handling

## Diagnostics

Dynamics-related diagnostics exported by the runtime include (when enabled in export path):
- vorticity components (`vorticity_r`, `vorticity_theta`, `vorticity_z`)
- stretching/tilting/baroclinic terms
- pressure decomposition (`p_prime`, `dynamic_pressure`, `buoyancy_pressure`)
- angular momentum diagnostics

Field bounds and strict-guard behavior are checked through the validation contract.

## Configuration

The key runtime selector is:

```yaml
dynamics:
  scheme: tornado   # or supercell
```

## Validation

Recommended checks:

```bash
make test-guards
make test-backend-physics
```

These cover strict exported-field checks and integration behavior under multiple configs.

## Notes

This module is focused on scheme-level dynamics logic. Numerical advection/diffusion/time-stepping orchestration is documented in `src/numerics/README.md` and coordinated through `src/core/orchestration/dynamics/numerics.cpp` and `src/core/orchestration/dynamics/dynamics.cpp`.
