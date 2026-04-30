# Include Directory

Public headers for Nimbus. Each subdirectory maps to the corresponding `src/` module.

## Layout

```
include/
  boundary_conditions/   BC dispatch API and base definitions
  boundary_layer/        PBL parameterization base class
  chaos/                 Stochastic perturbation base class
  compute/               GPU compute backend interface and kernel dispatch
  core/                  Simulation state, Field3D, runtime config, constants
    field/               Field3D utilities (pool, snapshot, sanitization)
    output/              Output writers (NPY, SHM, config)
  data/                  Sounding data structures and ingestion
  diagnostics/           Conservation budget, field contract, field validation
  dynamics/              Dynamics scheme base class
  microphysics/          Cloud microphysics base class
  numerics/              Shared numerical types
    advection/           Scalar transport interfaces and Cartesian kernels
    derivatives/         Coordinate-system derivative operators
    diffusion/           Diffusion algorithm base class
    time_stepping/       Time integration base class and split-explicit callbacks
  radar/                 Radar forward operator base class and API
  radiation/             Radiative transfer base class
  terrain/               Terrain/orographic forcing base class
  turbulence/            Sub-grid turbulence closure base class
  util/                  Logging, grid metrics, scheme factory, SIMD, strings
```

## Conventions

- Each physics/numerics module defines a base class here as `<module>/<module>_base.hpp`. Implementations live in `src/<module>/schemes/`.
- Schemes are registered via `util/scheme_factory.hpp` and created through factory functions in `src/<module>/factory.cpp`.
- `core/simulation.hpp` declares all global state (Field3D arrays, grid dimensions, config globals).
- `core/field/field3d.hpp` provides the contiguous 3D storage used by every module.
- Infrastructure modules (boundary_conditions, compute, diagnostics) define their interfaces here but do not use the factory pattern. See `src/README.md` for why.

## Adding a New Module

1. Create `include/<module>/<module>_base.hpp` with the abstract interface.
2. Create `src/<module>/factory.cpp` using `tmv::SchemeRegistry`.
3. Create `src/<module>/schemes/<name>/` with the implementation.
4. Add a test in `tests/`.
5. See `src/README.md` for the full guide including orchestration wiring.
