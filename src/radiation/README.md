# Atmospheric Radiation Module

This module computes radiative heating/cooling tendencies used by the backend thermodynamics step.

## Architecture

```
src/core/orchestration/physics/radiation.cpp   # Runtime coordinator + integration
src/radiation/factory.cpp/.hpp                 # Scheme factory
src/radiation/base/radiative_transfer.*        # Transfer/math utilities
src/radiation/schemes/simple_grey/*            # Implemented scheme
```

`rrtmg` is recognized as a canonical planned scheme id for platform-fidelity config compatibility, but current runtime execution remains `simple_grey`.

## Implemented Scheme

### `simple_grey`
- Broadband grey-gas approximation.
- Longwave: two-stream update with layer optical depths.
- Shortwave: Beer-Lambert attenuation through each layer.
- Heating tendency is computed from vertical net-flux divergence:

```
dT/dt = -(1 / (rho * cp)) * dFnet/dz
```

## Runtime Integration Notes

- Radiation is stepped in `src/core/orchestration/physics/radiation.cpp` on cadence `radiation.dt` (or `radiation.dt_radiation` alias).
- Surface properties used by radiation come from shared surface config (`surface.albedo`, `surface.emissivity`, `surface.tsfc`/`surface.Tsfc`).
- Radiation tendencies are converted from `dT/dt` to `dtheta/dt` before entering the main physics tendency sum.

## Configuration

### Supported keys

```yaml
radiation:
  scheme: simple_grey        # implemented runtime scheme (default)
  # scheme: rrtmg            # recognized planned id; currently falls back to simple_grey
  dt: 300.0                  # cadence (s); alias: dt_radiation
  do_lw: true
  do_sw: true

  tau_lw_ref: 6.0            # non-negative
  tau_sw_ref: 0.22           # non-negative
  n_lw: 4.0                  # positive
  n_sw: 2.0                  # positive

  clear_sky_only: true       # parsed/stored
  use_subcycling: false      # parsed/stored
  enable_diagnostics: false  # parsed/stored
```

Surface coupling:

```yaml
surface:
  albedo: 0.2
  emissivity: 0.95
  tsfc: 288.0
```

## Regression Coverage

Focused radiation regression checks are provided by:

```bash
make test-radiation-regression
```

This runs `tests/radiation_regression.cpp`, which validates:
- Optical-depth profile monotonicity and layer-depth consistency.
- Per-layer shortwave attenuation behavior.
- Flux-divergence shape guards.

For full backend validation:

```bash
make test-guards
make test-backend-physics
```
