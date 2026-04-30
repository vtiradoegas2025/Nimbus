# Terrain/Orography Module

This module handles topography generation and terrain-coordinate metrics for the simulation runtime.

## Purpose

Terrain support provides:
- 2D topography fields (`h`, `hx`, `hy`)
- terrain-following metric fields (`z`, `J`, `mx`, `my`, `zeta`)
- runtime diagnostics about terrain height/slopes/Jacobian validity

## Layout

```text
src/core/orchestration/physics/terrain.cpp      # runtime coordinator and config sanitation
src/terrain/factory.cpp/.hpp                    # scheme factory
src/terrain/base/topography.cpp/.hpp            # topography + metric utilities
src/terrain/schemes/bell/
src/terrain/schemes/schar/
src/terrain/schemes/none/
include/terrain/terrain_base.hpp                # public terrain interfaces/config
```

## Supported Schemes

- `none` (flat terrain)
- `bell` (axisymmetric bell mountain)
- `schar` (ridge-style Schar profile)

Accepted aliases:
- `flat` -> `none`
- `schaer` -> `schar`

## Coordinate Modes

`terrain.coord_id` accepts:
- `btf` (terrain-following)
- `smoothed`
- `cartesian`

Aliases such as `terrain_following`, `terrain-following`, and `sigma` map to `btf`.

## Runtime Flow

1. Runtime parses/sanitizes terrain config in `src/core/runtime/runtime_config.cpp` and `src/core/orchestration/physics/terrain.cpp`.
2. `initialize_terrain(...)` selects and initializes the scheme.
3. `build_terrain_fields()` builds topography and metrics for the active grid.
4. Terrain diagnostics are logged (max height, Jacobian range, warnings).

## Configuration Keys

Supported keys include:

```yaml
terrain:
  scheme: none            # none | bell | schar
  coord_id: btf           # btf | smoothed | cartesian
  ztop: 20000.0
  compute_derivatives: true
  compute_metrics: true
  smoothing_decay: 0.1

  bell:
    h0: 1000.0
    a: 5000.0
    axisymmetric: false

  schar:
    H: 1000.0
    a: 5000.0
    ell: 4000.0
```

Invalid/nonphysical values are sanitized to safe defaults at runtime.

## Validation

Recommended checks:

```bash
make test-terrain-regression
make test-backend-physics
```

`test-terrain-regression` validates terrain-field/metric generation behavior and guardrails.

## Current Status

Terrain is integrated in runtime initialization and export workflows. Remaining work is primarily broader science calibration/case validation rather than integration plumbing.
