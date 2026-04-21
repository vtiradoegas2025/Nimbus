# Source Code

Implementation of the Nimbus atmospheric simulation framework. Organized by module, with each physics/numerics domain following a factory pattern.

## Directory Layout

```
src/
  boundary_conditions/     Coordinate-specific BC scheme implementations
  boundary_layer/          PBL parameterizations (slab, YSU, MYNN)
  chaos/                   Stochastic perturbation schemes (IC, BL, full SPPT)
  compute/                 GPU compute backend and kernel dispatch infrastructure
  core/                    Simulation engine (see below)
  diagnostics/             Conservation budget, field contract, field validation
  dynamics/                Dynamics schemes (cartesian, supercell, tornado)
  microphysics/            Cloud microphysics (Kessler, Lin, Thompson, Milbrandt-Yau)
  numerics/
    advection/             Scalar transport: horizontal kernels + vertical schemes (TVD, WENO5)
    diffusion/             Diffusion schemes (explicit, implicit)
    time_stepping/         Time integration (RK3, RK4, split-explicit)
  radar/                   Radar forward operators (reflectivity, velocity, ZDR)
  radiation/               Radiative transfer (simple grey)
  soundings/               Sounding ingestion (SHARPY)
  terrain/                 Terrain schemes (bell, Schar, none)
  tools/                   Standalone utilities (field_validator)
  turbulence/              Sub-grid turbulence (Smagorinsky, TKE)
```

## Core (src/core/)

The simulation engine, organized into four areas:

```
core/
  runtime/                 Entry point and simulation loop
    tornado_sim.cpp          main() -- program entry
    headless_runtime.cpp     main simulation loop (time stepping, output, diagnostics)
    runtime_config.cpp       YAML config parsing into global state
    gui.cpp                  SFML viewer (conditional, GUI=1)

  orchestration/           Wires modules into the simulation loop
    dynamics/                Equation solver and initialization
      dynamics.cpp             dynamics step: tendency computation + time integration
      equations.cpp            base state init, field resize, equation wrappers
      initial_conditions_cartesian.cpp   Cartesian wind + bubble helpers
      numerics.cpp             numerics scheme init and coordination
    physics/                 Physics parameterization callers
      boundary_layer.cpp       PBL step
      diffusion_step.cpp       diffusion step
      microphysics_step.cpp    microphysics step
      radar.cpp                radar orchestration
      radar_step.cpp           radar computation + init
      radiation.cpp            radiation step
      terrain.cpp              terrain step
      turbulence.cpp           turbulence step

  output/                  Data export and IO
    npy_writer.cpp           NPY format writer
    output_config.cpp        output field selection
    output_writer.cpp        file output with manifests
    shm_writer.cpp           shared memory transport for live visualization

  infra/                   Low-level support
    coordinate_system.cpp    CoordinateSystem enum
    field_sanitization.cpp   NaN/bounds guards on prognostic fields
    hardware_info.cpp        CPU/GPU capability detection
    nested_grid.cpp          grid nesting support
    simd_utils.cpp           runtime SIMD detection
```

## Module Pattern

Every physics/numerics module follows the same structure:

```
module/
  factory.cpp/.hpp         Scheme registry and creation
  base/                    Shared utilities (thermodynamics, surface fluxes, etc.)
  schemes/
    scheme_name/
      scheme_name.cpp/.hpp   Self-contained implementation
```

Interfaces live in `include/<module>/` as `*_base.hpp`. Implementations are selected at runtime through `util/scheme_factory.hpp`.

## Adding a New Scheme

1. Create `src/<module>/schemes/<name>/<name>.{cpp,hpp}` implementing the base class.
2. Register in `src/<module>/factory.cpp` via `tmv::SchemeRegistry`.
3. Add test in `tests/`.
4. Wire into orchestration if the module needs a step caller in `core/orchestration/physics/`.

## Build

Makefile-based with incremental object-file compilation. Key targets:

- `make` -- build the main binary
- `make test` -- run the full test suite
- `make vulkan` -- build the Vulkan viewer
- `make clean` -- remove build artifacts
