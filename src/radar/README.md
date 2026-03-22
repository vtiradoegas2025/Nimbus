# Radar Forward Operators

This module computes synthetic radar observables from the model state for backend diagnostics and export.

## Current Scope

Implemented observables:
- Reflectivity in linear units: `Ze_linear` (mm^6/m^3)
- Reflectivity in dBZ: `Z_dBZ`
- Radial velocity: `Vr` (m/s)
- Polarimetric reflectivity components: `ZH_dBZ`, `ZV_dBZ`, `ZDR_dB`

Implemented scheme IDs:
- `reflectivity`
- `velocity`
- `zdr`

## File Layout

Core interfaces:
- `include/radar_base.hpp`
- `include/radar.hpp`

Factory and shared utilities:
- `src/radar/factory.cpp`
- `src/radar/base/radar_base.cpp`

Scheme implementations:
- `src/radar/schemes/reflectivity/reflectivity.cpp`
- `src/radar/schemes/velocity/velocity.cpp`
- `src/radar/schemes/zdr/zdr.cpp`

Runtime integration points:
- `src/core/radar.cpp`
- `src/core/equations.cpp`

## Scheme Behavior

### Reflectivity (`scheme_id="reflectivity"`)

Operator tiers:
- `fast_da`
- `psd_moment`

Inputs:
- Hydrometeor mixing ratios: `qr`, `qs`, `qg`, `qh`, `qi`
- Optional number concentrations for moment-aware paths: `Nr`, `Ns`, `Ng`, `Nh`, `Ni`

Outputs written:
- `Ze_rain`, `Ze_snow`, `Ze_graupel`, `Ze_hail`, `Ze_ice`
- `Ze_linear`, `Z_dBZ`, `ZH_dBZ`, `ZV_dBZ`

### Velocity (`scheme_id="velocity"`)

Sampling modes:
- `point`
- `beam_volume` (local 3x3x3 averaging)

Inputs:
- Winds: `u`, `v`, `w`
- Optional hydrometeors for scatterer correction
- Radar location: `radar_x`, `radar_y`, `radar_z`

Outputs written:
- `Vr`

Optional behavior:
- `apply_scatterer_correction=true` applies a bulk fall-speed correction when hydrometeors are present.

### Differential Reflectivity (`scheme_id="zdr"`)

Operator tiers:
- `simple`
- `polarimetric_fo`

Inputs:
- Rain for simple tier (`qr`)
- Rain/ice species for polarimetric tier (`qr`, `qs`, `qg`, `qh`, `qi`)

Outputs written:
- `ZH_dBZ`, `ZV_dBZ`, `ZDR_dB`, `Ze_linear`, `Z_dBZ`

## Runtime Guards and Data Safety

- Each scheme now reshapes and clears its owned output fields at the start of `compute(...)`.
- Each scheme validates state/output dimensions; on mismatch it logs a warning once and returns zeroed output for that scheme.
- `src/core/radar.cpp` sanitizes and clamps output ranges before returning:
  - `Ze_*`: `[0, 1e12]`
  - `Z*_dBZ`: `[-40, 100]`
  - `ZDR_dB`: `[-12, 12]`
  - `Vr`: `[-250, 250]`
- `src/core/equations.cpp` has guarded fallback behavior in `calculate_radar_reflectivity()`:
  - Uses radar scheme when available.
  - Falls back to microphysics reflectivity if radar scheme is unavailable or throws.
  - Keeps previous reflectivity field if fallback input/output shape is invalid.

## API Usage Example

```cpp
#include "diagnostics/radar.hpp"

RadarStateView state;
state.NR = NR;
state.NTH = NTH;
state.NZ = NZ;
state.u = &u;
state.v = &v_theta;
state.w = &w;
state.qr = &qr;
state.qs = &qs;
state.qg = &qg;
state.qh = &qh;
state.qi = &qi;

RadarConfig config;
config.scheme_id = "reflectivity";
config.operator_tier = "fast_da";
config.has_qr = true;
config.has_qs = true;
config.has_qg = true;
config.has_qh = true;
config.has_qi = true;

auto scheme = create_radar_scheme(config.scheme_id);
scheme->initialize(config, state.NR, state.NTH, state.NZ);

RadarOut out;
out.initialize(state.NR, state.NTH, state.NZ);
scheme->compute(config, state, out);
```

## Validation

Recommended backend checks:

```bash
make test-backend-physics
make test-guards
```

These checks verify export contracts, numeric bounds, non-finite guards, and renderer-facing field compatibility.
