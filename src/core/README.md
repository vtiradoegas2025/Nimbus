# Core (Simulation Engine)

The simulation engine. Owns the main loop, field allocation, module wiring, output, and low-level infrastructure.

## Why No Factory

Core is the orchestrator, not a pluggable component. Physics modules (microphysics, turbulence, etc.) are interchangeable scientific choices -- the user picks one via config. Core is the fixed control flow that initializes those modules, calls them in the right order each timestep, and writes output. There is nothing to select.

## Structure

```
core/
  runtime/           Entry point and main loop
  orchestration/     Module wiring (dynamics + physics step callers)
  output/            Data export (NPY files, shared memory, manifests)
  infra/             Low-level support (SIMD, hardware, coordinates, damping)
```

### runtime/

The simulation entry point and main loop.

| File | Purpose |
|------|---------|
| `tornado_sim.cpp` | `main()` -- parses CLI args, loads config, runs initialization sequence, enters main loop |
| `headless_runtime.cpp` | Main simulation loop: timestep selection, physics/dynamics stepping, output, diagnostics |
| `runtime_config.cpp` | YAML config parsing into global state structs |
| `gui.cpp` | SFML windowed viewer (conditional, built with `make GUI=1`) |

### orchestration/

Thin coordinators that wire physics/dynamics modules into the simulation loop. Each file follows the same pattern: an `initialize_<module>()` function called at startup, and a `step_<module>()` function called each timestep at the configured cadence.

**orchestration/dynamics/**

| File | Purpose |
|------|---------|
| `dynamics.cpp` | Dynamics step: computes tendencies, integrates, applies BCs. Owns the split-explicit acoustic substep loop. |
| `equations.cpp` | Base state initialization, field allocation, equation-level wrappers |
| `initial_conditions_cartesian.cpp` | Cartesian-specific IC helpers (thermal bubble, shear profile) |
| `numerics.cpp` | Numerics scheme initialization and timestep coordination |

**orchestration/physics/**

| File | Purpose |
|------|---------|
| `boundary_layer.cpp` | PBL step (cadence, tendency fill, guards) |
| `diffusion_step.cpp` | Diffusion tendency application |
| `microphysics_step.cpp` | Microphysics step (tendency computation, theta limiting, state clamping) |
| `radar.cpp` | Radar forward operator orchestration |
| `radar_step.cpp` | Radar computation and initialization |
| `radiation.cpp` | Radiation step (cadence, dT/dt to dtheta/dt conversion) |
| `terrain.cpp` | Terrain initialization and metric computation |
| `turbulence.cpp` | SGS turbulence step (cadence, tendency sanitization) |

### output/

| File | Purpose |
|------|---------|
| `npy_writer.cpp` | NPY format writer for 3D field export |
| `output_config.cpp` | Output field selection and filtering |
| `output_writer.cpp` | Per-step directory output with manifests |
| `shm_writer.cpp` | POSIX shared memory transport for live visualization |

### infra/

| File | Purpose |
|------|---------|
| `coordinate_system.cpp` | `CoordinateSystem` enum and global coordinate system state |
| `field_sanitization.cpp` | NaN/Inf guards and bounds clamping on prognostic fields |
| `hardware_info.cpp` | CPU feature detection (cores, cache, SIMD) and GPU info |
| `nested_grid.cpp` | Grid nesting support |
| `rayleigh_damping.cpp` | Upper-boundary sponge layer for gravity wave absorption |
| `simd_utils.cpp` | Runtime SIMD capability detection (SSE, AVX, AVX-512, NEON) |

## Simulation Loop

The high-level control flow in `headless_runtime.cpp`:

```
for each timestep:
    choose_runtime_timestep()          # CFL-adaptive dt
    step_boundary_layer(t)             # PBL tendencies (at cadence)
    step_chaos_noise(dt)               # stochastic perturbations
    apply_chaos_tendencies()           # inject perturbations
    step_dynamics(t)                   # advection + pressure + momentum + BCs
      step_microphysics()              # cloud physics (within dynamics)
      step_turbulence()                # SGS mixing (within dynamics)
      step_diffusion()                 # explicit/implicit diffusion
    step_radiation(t)                  # radiative heating (at cadence)
    apply_field_sanitization()         # NaN/bounds guards
    write_output()                     # NPY export + SHM transport
    log_diagnostics()                  # conservation, contract checks
```
