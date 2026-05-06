# Source Code

Implementation of the Nimbus atmospheric simulation framework.

---

## Directory Layout

```
src/
  boundary_conditions/     Coordinate-specific boundary enforcement (Cartesian, cylindrical)
  boundary_layer/          PBL parameterizations (slab, YSU, MYNN)
  chaos/                   Stochastic perturbation schemes (IC, BL, full SPPT)
  compute/                 GPU compute backend and kernel dispatch infrastructure
  core/                    Simulation engine: runtime, orchestration, output, infra
  diagnostics/             Conservation budget, field contract, field validation
  dynamics/                Dynamics schemes (cartesian, supercell, tornado)
  microphysics/            Cloud microphysics (Kessler, Lin, Thompson, Milbrandt-Yau)
  numerics/
    advection/             Scalar transport schemes (TVD, WENO5) + coordinate kernels
    diffusion/             Diffusion schemes (explicit, implicit)
    time_stepping/         Time integration (RK3, RK4, split-explicit)
  radar/                   Radar forward operators (reflectivity, velocity, ZDR)
  radiation/               Radiative transfer (simple grey)
  soundings/               Sounding ingestion (SHARPY)
  terrain/                 Terrain schemes (bell, Schar, none)
  tools/                   Standalone utilities (field_validator)
  turbulence/              Sub-grid turbulence (Smagorinsky, TKE)
```

---

## Architecture Patterns

Not every module uses the same structure, and the differences are intentional. This section explains the three patterns and why each module uses the one it does.

### Pattern 1: Factory Modules (configurable physics/numerics)

Most physics and numerics modules follow a factory pattern. The user selects a scheme by name in YAML config, and a factory function instantiates the right implementation at runtime.

```
module/
  factory.cpp/.hpp         Scheme registry: maps string names to constructors
  base/                    Shared utilities used by multiple schemes
  schemes/
    scheme_a/
      scheme_a.cpp/.hpp    Self-contained implementation of the base class
    scheme_b/
      scheme_b.cpp/.hpp
```

Base classes live in `include/<module>/<module>_base.hpp`. Factories use `util/scheme_factory.hpp` for name canonicalization (case-insensitive, alias handling).

**Modules using this pattern:** microphysics, dynamics, boundary_layer, turbulence, radiation, radar, terrain, chaos, soundings, numerics/advection, numerics/diffusion, numerics/time_stepping.

**Why factories:** These modules represent scientific choices. A researcher picks Thompson vs. Kessler microphysics, or WENO5 vs. TVD advection, based on what they're studying. The factory pattern lets them swap implementations via config without touching code.

### Pattern 2: Infrastructure Modules (no factory, no user selection)

Some modules provide fixed infrastructure. There is nothing to select. They do one job determined by the coordinate system or hardware.

**boundary_conditions/** -- Two files: `boundary_conditions_cartesian.cpp` and `boundary_conditions_cylindrical.cpp`. The active coordinate system determines which BC functions are called. There is no factory because BCs are not a scientific choice -- they are a mathematical requirement of the domain geometry. BCs are applied by the time-stepping loop, not selected by the user.

**Why boundary_conditions is separate from boundary_layer:** These are orthogonal concerns despite the similar names. Boundary conditions (`boundary_conditions/`) enforce mathematical constraints at grid edges: zero-gradient, periodic, rigid-lid. Boundary layer (`boundary_layer/`) is atmospheric physics: vertical mixing, surface fluxes, PBL height. A simulation always needs both, and they are configured independently.

**compute/** -- GPU backend abstraction with a `ComputeBackend` virtual interface. Selection is `cpu` vs `vulkan` (and eventually `metal`), but this isn't a factory-pattern module -- it's an infrastructure backend with its own initialization lifecycle, buffer management, and shader routing. The selection logic lives in `compute_backend.cpp`; the Vulkan implementation in `vulkan/src/compute/compute_backend_vulkan.cpp`.

**diagnostics/** -- Field validation, conservation budgets, and contract checking. These are runtime analysis tools, not configurable schemes. They operate on the simulation state without modifying it.

### Pattern 3: Core (the orchestrator)

`core/` is the simulation engine itself. It does not have a factory because it is not a pluggable component. It's the fixed control flow that wires everything else together. See `src/core/README.md` for its internal structure.

### Numerics: Nested Factories

The `numerics/` directory contains three independent submodules, each with its own factory:

```
numerics/
  advection/       factory + schemes (tvd, weno5) + coordinate kernels
  diffusion/       factory + schemes (explicit, implicit)
  time_stepping/   factory + schemes (rk3, rk4, split_explicit)
```

These are grouped under `numerics/` because they share a common concern (discretization methods) but are independently selectable. You can combine WENO5 advection with implicit diffusion and split-explicit time stepping.

The advection submodule also contains coordinate-specific transport kernels (`advection.cpp` for cylindrical, `advection_cartesian.cpp` for Cartesian) that are not scheme-selectable -- they implement the directional splitting logic that calls into whichever scheme the factory selected.

---

## Core (src/core/)

The simulation engine, organized into four areas:

```
core/
  runtime/                 Entry point and simulation loop
    tornado_sim.cpp          main() -- program entry, initialization sequence
    headless_runtime.cpp     main simulation loop (time stepping, output, diagnostics)
    runtime_config.cpp       YAML config parsing into global state

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
    coordinate_system.cpp    CoordinateSystem enum and global
    field_sanitization.cpp   NaN/bounds guards on prognostic fields
    hardware_info.cpp        CPU/GPU capability detection
    nested_grid.cpp          grid nesting support
    rayleigh_damping.cpp     upper-boundary sponge layer
    simd_utils.cpp           runtime SIMD detection (SSE/AVX/NEON)
```

Each file in `orchestration/physics/` is a thin coordinator: it calls `initialize_<module>()` at startup, then `step_<module>()` each timestep at the configured cadence. The actual physics lives in the module's `schemes/` directory.

---

## Adding a New Scheme

1. Create `src/<module>/schemes/<name>/<name>.{cpp,hpp}` implementing the base class in `include/<module>/`.
2. Register in `src/<module>/factory.cpp` via `tmv::SchemeRegistry`.
3. Add a test in `tests/`.
4. Wire into orchestration if the module needs a step caller in `core/orchestration/physics/`.

## Adding a New Module

1. Create `include/<module>/<module>_base.hpp` with the abstract interface.
2. Create `src/<module>/factory.cpp` using `tmv::SchemeRegistry`.
3. Create `src/<module>/schemes/<name>/` with the first implementation.
4. Add an orchestration coordinator in `core/orchestration/physics/` if it needs per-timestep calling.
5. Add config parsing in `core/runtime/runtime_config.cpp`.
6. Add a test in `tests/`.

## Build

Makefile-based with incremental object-file compilation. Key targets:

- `make` -- build the simulation binary + compile GPU shaders
- `make test` -- run the full test suite
- `make vulkan` -- build the Vulkan viewer
- `make check-deps` -- verify dependencies are installed
- `make clean` -- remove build artifacts

See [BUILDING.md](../BUILDING.md) for platform-specific setup.
