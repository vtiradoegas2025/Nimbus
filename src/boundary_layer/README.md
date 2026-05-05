# Planetary Boundary Layer (PBL) Module

This module computes planetary boundary layer tendencies for momentum, potential temperature, moisture, and (for MYNN) TKE.

## Purpose

The PBL module couples near-surface turbulence and vertical mixing into the core solver through:

- `du_dt_pbl`
- `dv_dt_pbl`
- `dtheta_dt_pbl`
- `dqv_dt_pbl`
- `dtke_dt_pbl` (MYNN only)

These tendencies are produced at a configurable cadence and then consumed by the dynamics/microphysics update paths.

## Code Layout

```text
src/boundary_layer/
├── factory.cpp/.hpp
├── base/
│   └── surface_fluxes.cpp/.hpp
└── schemes/
    ├── slab/
    ├── ysu/
    └── mynn/
```

Module coordination lives in:

- `src/core/orchestration/physics/boundary_layer.cpp` (initialize, cadence stepping, tendency field fill)

Shared interfaces and config structures live in:

- `include/boundary_layer/boundary_layer_base.hpp`

## Available Schemes

### `slab`

Simple mixed-layer model with:

- Prognostic mixed-layer state (`h`, `theta_m`, `qv_m`)
- Surface-flux forcing
- Entrainment-like update and in-layer relaxation tendencies

### `ysu`

YSU-style profile diffusion with:

- Diagnosed PBL height
- K-profile eddy diffusivities
- Optional nonlocal transport term (`enable_nonlocal_transport`)

### `mynn`

MYNN-style TKE-based closure with:

- Column TKE-aware diffusivities
- Prognostic TKE tendency (`dtke_dt_pbl`)
- TKE-informed PBL-top diagnosis path

## Surface Flux Options

Surface fluxes are selected by boundary-layer config:

- `bulk`
- `monin_obukhov` (aliases: `monin-obukhov`, `moninobukhov`, `most`, `mo`)

The selected method is applied through `compute_surface_fluxes(...)` in `src/boundary_layer/base/surface_fluxes.cpp`.

`min_ustar` enforces a configurable lower bound for friction velocity.

## Runtime Configuration (Supported Keys)

The keys below are parsed in `src/core/runtime/runtime_config.cpp`.

### `boundary_layer.*`

- `scheme`: `slab | ysu | mynn`
- `apply_surface_fluxes`: bool
- `dt` (preferred) or `dt_pbl` (alias), positive
- `surface_layer` or `surface_layer_id`: `bulk | monin_obukhov`
- `enable_nonlocal_transport`: bool
- `enable_tke`: bool
- `pbl_top_method`: `ri | theta | tke`
- `pbl_max_height`: positive
- `min_ustar`: non-negative
- `clear_sky_only`: accepted and stored

### `surface.*`

- `z0m`: positive
- `z0h`: positive
- `albedo`: [0, 1]
- `emissivity`: [0, 1]
- `tsfc` or `Tsfc`: finite
- `qsfc` or `qv_sfc`: non-negative

Note:

- In current PBL flux computations, `z0m` and `z0h` are actively used.
- `albedo`, `emissivity`, `Tsfc`, and `qsfc` are parsed into `SurfaceConfig` for shared surface state, but are not directly used by current `compute_surface_fluxes(...)` calls.

## Configuration Examples

### YSU with bulk fluxes

```yaml
boundary_layer:
  scheme: ysu
  apply_surface_fluxes: true
  dt: 30.0
  surface_layer: bulk
  enable_nonlocal_transport: true
  pbl_max_height: 2500.0
  min_ustar: 0.01

surface:
  z0m: 0.05
  z0h: 0.01
```

### YSU with Monin-Obukhov fluxes

```yaml
boundary_layer:
  scheme: ysu
  apply_surface_fluxes: true
  dt: 30.0
  surface_layer: monin_obukhov
  min_ustar: 0.05
```

### MYNN

```yaml
boundary_layer:
  scheme: mynn
  apply_surface_fluxes: true
  dt: 30.0
  surface_layer: bulk
  enable_tke: true
  pbl_top_method: tke
  pbl_max_height: 3000.0
  min_ustar: 0.01
```

## Integration Notes

- Initialization entrypoint: `initialize_boundary_layer(...)`
- Time stepping entrypoint: `step_boundary_layer(current_time)`
- The core module applies guards for invalid config values and malformed tendency outputs before writing tendency fields.

## Quick Smoke Check

```bash
make bin/tornado_sim
./bin/tornado_sim --headless --config=configs/test/physics_full_stack.yaml --duration=1 --write-every=1 --outdir=/tmp/tmv_pbl_smoke --log-profile quiet
```

