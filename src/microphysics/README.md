# Cloud Microphysics Module

This module provides bulk cloud microphysics parameterizations for the storm model. It computes tendencies for water species, latent-heating tendencies, and scheme-level radar reflectivity estimates.

## Scope

Microphysics handles:
- Warm-rain conversion and accretion
- Mixed-phase processes (freezing, melting, deposition, sublimation)
- Sedimentation/fallout terms for precipitating species
- Scheme-local moment variables (for schemes that predict number concentrations)

## Current Architecture

```text
src/microphysics/
├── base/
│   └── thermodynamics.cpp/.hpp
├── factory.cpp/.hpp
└── schemes/
    ├── kessler/
    ├── lin/
    ├── thompson/
    └── milbrandt/
```

Runtime coupling is in:
- `src/core/orchestration/dynamics/equations.cpp`
  - `initialize_microphysics(...)`
  - `step_microphysics(...)`
  - `calculate_radar_reflectivity(...)`
- `src/core/orchestration/dynamics/dynamics.cpp`
  - calls `step_microphysics(...)` each dynamics step

## Prognostic Fields

Model state fields passed to microphysics:
- `qv` water vapor
- `qc` cloud water
- `qr` rain
- `qi` cloud ice
- `qs` snow
- `qg` graupel
- `qh` hail
- `theta`, `p` for thermodynamic conversions

All fields are `Field3D` and are expected to match global grid dimensions (`NR`, `NTH`, `NZ`).

## Scheme Inventory

- `kessler`
  - Warm-rain core with extended graupel/hail pathways in this codebase.
  - `get_num_prognostic_vars() = 5` (`qv`, `qc`, `qr`, `qg`, `qh`).

- `lin`
  - Mixed-phase bulk scheme with ice/snow/graupel/hail pathways.
  - `get_num_prognostic_vars() = 7`.

- `thompson`
  - Mixed-phase scheme with internal prognostic ice number concentration (`Ni`).
  - `get_num_prognostic_vars() = 8`.

- `milbrandt`
  - Double-moment style scheme with internal number concentrations (`Nr`, `Ni`, `Ns`, `Ng`, `Nh`) and optional third-moment reflectivity storage.
  - `get_num_prognostic_vars() = 12` (or `17` when triple-moment is enabled).

## Numerical Safety and Guards

Coupling-side protections in `step_microphysics(...)`:
- Reused tendency buffers with explicit shape checks
- Non-finite tendency sanitization
- Per-step theta increment limiter
- State clamping after tendency application (`theta`, `qv`, all hydrometeors)
- Exception guard around scheme tendency computation with zero-tendency fallback for that step

Radar-side protections:
- Exception guard for microphysics radar fallback
- Reflectivity finite/bounds sanitization before storing to model fields

Scheme-side protections include:
- Denominator floors for RH calculations in Lin/Thompson/Milbrandt
- Shape-consistent reinitialization of internal moment fields when grid size changes (Thompson/Milbrandt)

## Validation

Recommended backend checks:

```bash
make test-backend-physics
make test-guards
```

These run integrated dynamics+physics regression gates and field-contract guard checks.

## Notes

This is still a research/prototype microphysics stack. Some process rates are intentionally simplified relative to full operational implementations, but coupling and runtime guards are in place to keep integrated runs stable.
