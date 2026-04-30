# Chaos Module (Stochastic Perturbations)

This module injects controlled stochastic variability into initial conditions and physics tendencies.

## Scope and Naming

`chaos.scheme: boundary_layer` means **stochastic perturbations of PBL tendencies**.
It does **not** replace the deterministic physical PBL scheme selected by `boundary_layer.scheme`.

## Directory Structure

```text
src/chaos/
├── chaos.cpp
├── factory.cpp
├── factory.hpp
├── base/
│   ├── random_generator.cpp
│   ├── random_generator.hpp
│   ├── perturbation_field.cpp
│   ├── perturbation_field.hpp
│   ├── correlation_filter.cpp
│   └── correlation_filter.hpp
└── schemes/
    ├── none/
    ├── initial_conditions/
    ├── boundary_layer/
    └── full_stochastic/
```

## Runtime Boundaries and Call Sites

- Initialization path:
  - `src/core/runtime/tornado_sim.cpp` calls `initialize_chaos(global_chaos_config)` after numerics init.
  - `src/core/runtime/tornado_sim.cpp` then calls `apply_chaos_initial_conditions()` once.
- Time-stepping path (headless runtime):
  - `src/core/runtime/headless_runtime.cpp` calls, in order:
    - `step_boundary_layer(simulation_time)`
    - `step_chaos_noise(dt)`
    - `apply_chaos_tendencies()`
    - `step_dynamics(simulation_time)`
- Additional tendency hook points:
  - `src/core/orchestration/dynamics/equations.cpp` calls `apply_chaos_to_microphysics_tendencies(...)`.
  - `src/core/orchestration/dynamics/dynamics.cpp` calls `apply_chaos_to_turbulence_tendencies(...)`.

## Supported Config Keys

Primary keys parsed in `src/core/runtime/runtime_config.cpp`:

```yaml
chaos:
  scheme: "none" | "initial_conditions" | "boundary_layer" | "full_stochastic"
  seed: 42
  member_id: 0
  dt_chaos: 60.0
  tau_t: 21600.0
  Lx: 50000.0
  Ly: 50000.0
  filter_id: "recursive_gaussian" | "spectral_gaussian"
  xi_max: 2.0
  taper_id: "none" | "pbl_only" | "cosine"
  taper_z1: 1000.0
  taper_z2: 3000.0
  apply_to_ic: ["u", "v", "w", "theta", "qv"]
  apply_to_tendencies: ["microphysics", "pbl", "diffusion"]
```

Per-variable/block amplitudes:

```yaml
chaos:
  sigma_ic:
    u: 0.5
    theta: 0.5
  alpha_tend:
    microphysics: 0.3
    pbl: 0.2
    diffusion: 0.1
```

## Aliases and Canonicalization

- Scheme aliases accepted by factory include:
  - `pbl`, `pbl_perturbation`, `boundary_layer_perturbation` -> `boundary_layer`
  - `ic` -> `initial_conditions`
  - `full` -> `full_stochastic`
- Tendency block aliases normalized in `src/chaos/chaos.cpp`:
  - `boundary_layer`, `bl` -> `pbl`
  - `micro` -> `microphysics`
  - `turbulence`, `sgs`, `subgrid` -> `diffusion`

## Safety and Guardrails

- Config sanitization at chaos initialization:
  - Invalid `dt_chaos`, `tau_t`, `xi_max`, `taper_z1`, `taper_z2` are corrected to safe defaults.
  - `taper_z2 <= taper_z1` is corrected to avoid invalid cosine tapering.
- Perturbation bounding now defends against invalid clip amplitudes (no divide-by-zero/NaN path for `xi_max`).
- Perturbed tendency fields are sanitized for non-finite values before returning to physics.
- PBL chaos perturbations are applied from deterministic base tendencies each update (prevents accidental multiplicative compounding between PBL refreshes).

