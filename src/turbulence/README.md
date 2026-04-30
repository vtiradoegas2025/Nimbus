# Sub-Grid Turbulence Module

This module computes sub-grid-scale (SGS) momentum and scalar mixing tendencies
for the backend solver.

## Overview

Implemented closures:
- `smagorinsky`: algebraic eddy-viscosity closure
- `tke`: prognostic TKE closure

Primary outputs:
- momentum SGS tendencies (`dudt_sgs`, `dvdt_sgs`, `dwdt_sgs`)
- scalar SGS tendencies (`dthetadt_sgs`, optional `dqvdt_sgs`)
- TKE tendency (`dtkedt_sgs`, TKE scheme)

## Code Layout

```text
src/core/orchestration/physics/turbulence.cpp   # runtime coordinator, cadence, guards
src/turbulence/factory.cpp/.hpp                 # scheme construction
src/turbulence/base/eddy_viscosity.cpp/.hpp
src/turbulence/schemes/smagorinsky/*.cpp
src/turbulence/schemes/tke/*.cpp
```

## Runtime Coupling

1. `initialize_turbulence(...)` is called from
   `src/core/runtime/tornado_sim.cpp`.
2. `step_turbulence(...)` is called from `src/core/orchestration/dynamics/dynamics.cpp`.
3. Chaos perturbations can modify SGS tendencies through
   `apply_chaos_to_turbulence_tendencies(...)`.
4. SGS tendencies are added to resolved state tendencies in the dynamics step.
5. For TKE updates, `src/core/orchestration/dynamics/dynamics.cpp` applies
   `dtkedt_sgs + dtke_dt_pbl` to the shared `tke` field.

## Scheme Notes

### Smagorinsky

- Uses `nu_t = (Cs * Delta)^2 * |S|`.
- Optional Richardson-based stability correction (`ri`).
- Computes `K_theta` and `K_q` from `Pr_t` and `Sc_t`.

### TKE

- Uses `nu_t = c_k * l * sqrt(e)`.
- Prognostic tendency form:
  - `dtke = P_s + P_b - eps + diffusion`
- Reads the shared runtime `tke` field when available, so closure and model
  state remain synchronized.
- Uses a simple vertical second-derivative term for TKE diffusion.

## Grid and Metrics

- Turbulence uses `global_grid_metrics` when available; otherwise it builds a
  fallback grid from `dr`, `dtheta`, and `dz`.
- Local `dx`, `dy`, and `dz` come from `grid_metric_utils.hpp`.
- Brunt-Vaisala frequency and diffusion terms use local vertical spacing
  (terrain-aware when terrain metrics are active).

## Configuration

Example:

```yaml
turbulence:
  scheme: tke                # smagorinsky | tke
  dt_sgs: 1.0                # > 0 (alias: turbulence.dt)
  Cs: 0.18                   # >= 0
  Pr_t: 0.7                  # > 0
  Sc_t: 0.7                  # > 0
  nu_t_max: 1000.0           # >= 0
  K_max: 1000.0              # >= 0
  mode: 3d                   # 3d | horizontal_only | vertical_only
  filter_width: cubic_root   # dx | cubic_root | user
  stability_correction: none # none | ri | tke_based
  dynamic_Cs: false
  enable_moist_buoyancy: false
  enable_diagnostics: false
```

Supported aliases:
- `turbulence.dt` for `turbulence.dt_sgs`
- `turbulence.smagorinsky.filter_width`
- `turbulence.smagorinsky.stability_correction`
- `turbulence.smagorinsky.dynamic_procedure`
- `smag`, `smagorinsky-lilly` accepted as scheme aliases
- `1.5`, `1.5-order` accepted as `tke` aliases

## Validation and Guarding

Recommended checks:

```bash
make bin/tornado_sim
./bin/tornado_sim --headless --config=configs/simulation/physical_supercell.yaml --duration=1 --write-every=1 --outdir=/tmp/turbulence_smoke --log-profile quiet
./bin/field_validator --input /tmp/turbulence_smoke --contract cm1 --mode strict --scope exported --json /tmp/turbulence_smoke/offline_validation.json
make test-guards
make test-backend-physics
```

Runtime safety behavior:
- non-finite SGS tendency values are sanitized to zero in
  `src/core/orchestration/physics/turbulence.cpp`
- invalid turbulence config values are clamped/fallbacked at initialization

## Current Limitations

- TKE diffusion is intentionally simple (vertical Laplacian contribution only),
  not a full anisotropic flux-divergence closure.
