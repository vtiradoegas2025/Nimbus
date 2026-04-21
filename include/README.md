# Include Directory

Public headers for Nimbus. Each subdirectory maps to the corresponding `src/` module.

## Layout

```
include/
  boundary_conditions/   BC scheme base class and factory declarations
  boundary_layer/        PBL parameterization base class
  chaos/                 Stochastic perturbation base class
  compute/               GPU compute backend and kernel dispatch
  core/                  Simulation state, Field3D, runtime config, constants
    field/               Field3D utilities (pool, snapshot, sanitization)
    output/              Output writers (NPY, SHM, config)
  data/                  Sounding data structures and ingestion
  diagnostics/           Conservation budget, field contract, field validation
  dynamics/              Dynamics scheme base class and split-explicit mixin
  microphysics/          Cloud microphysics base class
  numerics/              Shared numerical types (numerics_base)
    advection/           Scalar transport interfaces and Cartesian kernels
    derivatives/         Coordinate-system derivative operators
    diffusion/           Diffusion algorithm base class
    time_stepping/       Time integration base class and callbacks
  radar/                 Radar forward operator base class and API
  radiation/             Radiative transfer base class
  terrain/               Terrain/orographic forcing base class
  turbulence/            Sub-grid turbulence closure base class
  util/                  Logging, grid metrics, scheme factory, SIMD, strings
```

## Conventions

- Each physics/numerics module defines a base class here; implementations live in `src/<module>/schemes/`.
- Schemes are registered via `util/scheme_factory.hpp` and created through factory functions.
- `core/simulation.hpp` declares all global state (Field3D arrays, grid dimensions, config globals).
- `core/field3d.hpp` provides the contiguous 3D storage used by every module.

## Adding a new module

1. Create `include/<module>/<module>_base.hpp` with the abstract interface.
2. Create `src/<module>/schemes/<name>/` with the implementation.
3. Create `src/<module>/factory.cpp` using `tmv::SchemeRegistry`.
4. Add a test in `tests/`.
