# Nimbus Technical Reference

This document is the technical entry point for the active Nimbus codebase.

For current project state, gaps, and recent validation outcomes, see [STATUS.md](STATUS.md).

## Scope

Nimbus is a research atmospheric model for supercell/tornado dynamics with:
- C++17 simulation runtime (`bin/tornado_sim`)
- Modular physics factories (microphysics, turbulence, PBL, radar, terrain, chaos, soundings)
- Native Vulkan renderer (`bin/vulkan_viewer`) for direct NPY export ingest

## Active Runtime Paths

### Simulation
- Main executable: `src/core/tornado_sim.cpp`
- Runtime loop entry: `src/core/headless_runtime.cpp`
- Core coordinators: `src/core/*.cpp`

### Rendering
- Active renderer: `vulkan/`
- Legacy Python/OpenGL documentation is no longer an active runtime path in this workspace.

## Build

### Prerequisites
- C++17 compiler (`clang++`/`g++`)
- GNU Make
- Optional OpenMP runtime (auto-detected)
- Vulkan toolchain for viewer path (headers/loader + platform runtime)

### Common Targets

```bash
# Build simulation
make

# Optional GUI build path
make GUI=1

# Build Vulkan viewer
make vulkan

# Run simulation
./bin/tornado_sim --headless --config=configs/classic.yaml --duration=60

# Run Vulkan dry-run smoke check
./bin/vulkan_viewer --dry-run
```

## Validation and QA

Primary test targets:

```bash
# Fast baseline suite
make test

# Strict backend physics/export validation across config matrix
make test-backend-physics

# Soundings regression coverage
make test-soundings

# Focused radiation regression
make test-radiation-regression

# Focused terrain regression
make test-terrain-regression
```

`make test-backend-physics` validates:
- strict exported-field contract checks (`nonfinite=0`, `bounds=0`, missing required/exported = 0)
- integration compatibility (manifest + core slice shape checks)
- multi-config runs (`lp`, `hp`, `cyclic`, `sharpy_lp`)

## Physics/Module Inventory

| Module | Implemented Schemes | Primary Docs |
|---|---|---|
| Dynamics | `tornado`, `supercell` | `src/dynamics/README.md` |
| Microphysics | `kessler`, `lin`, `thompson`, `milbrandt` | `src/microphysics/README.md` |
| Boundary Layer | `slab`, `ysu`, `mynn` | `src/boundary_layer/README.md` |
| Turbulence | `smagorinsky`, `tke` | `src/turbulence/README.md` |
| Radiation | `simple_grey` | `src/radiation/README.md` |
| Radar | `reflectivity`, `velocity`, `zdr` | `src/radar/README.md` |
| Terrain | `none`, `bell`, `schar` | `src/terrain/README.md` |
| Chaos | `none`, `initial_conditions`, `boundary_layer`, `full_stochastic` | `src/chaos/README.md` |
| Soundings | `sharpy` | `src/soundings/README.md` |
| Numerics | advection/diffusion/time-stepping factories | `src/numerics/README.md` |

## Configuration Overview

Configuration is loaded via `src/core/runtime_config.cpp` from YAML files in `configs/`.

Common production/test config entries:
- `configs/classic.yaml`
- `configs/lp.yaml`
- `configs/hp.yaml`
- `configs/cyclic.yaml`
- `configs/sharpy_lp.yaml`
- `configs/physical_supercell.yaml`
- `configs/physical_supercell_storm_tuned.yaml`

Typical run command:

```bash
./bin/tornado_sim \
  --headless \
  --config=configs/physical_supercell.yaml \
  --duration=300 \
  --write-every=5 \
  --outdir=data/exports
```

## Export and Validation Contracts

Backend exports are written as per-step directories (`step_XXXXXX`) containing NPY slices and a `manifest.json`.

Validation contract sources:
- Contract definitions: `src/validation/field_contract.cpp`
- Runtime checks: `src/validation/field_validation.cpp`
- Offline validator tool: `src/tools/field_validator.cpp` (`bin/field_validator`)

Example offline validation:

```bash
./bin/field_validator \
  --input data/exports \
  --contract cm1 \
  --mode strict \
  --scope exported \
  --json data/exports/offline_validation.json
```

## Soundings Ingestion Notes

The soundings pipeline currently supports:
- Native in-process NetCDF classic parsing (CDF1/CDF2)
- Native NetCDF C API ingest for NetCDF4/HDF-backed layouts
- Native HDF5 (HL API) ingest for SHARPY-style dataset layouts

See `src/soundings/README.md` for full details and test workflow.

## Vulkan Rendering Notes

Renderer docs and command options are maintained in `vulkan/README.md`.

Typical commands:

```bash
make vulkan
./bin/vulkan_viewer --dry-run
./bin/vulkan_viewer --window-test --render-backend volume --input data/exports --field theta
```

## Known Gaps (CM1-lite context)

From the current contract/QA baseline:
- Required-now exported field set is covered.
- Remaining backlog is primarily diagnostic breadth (surface, column, trajectory, synthetic sweep products).
- Terrain/chaos are integrated, but broader science calibration remains ongoing.
- Radiation runtime remains `simple_grey` by default; `rrtmg` is recognized as a planned interface target.

See [STATUS.md](STATUS.md) for the dated audit snapshot.

## Additional Documentation

- Project status and gap audit: [STATUS.md](STATUS.md)
- Scientific foundations and references: [foundationalScience.md](foundationalScience.md)
- Public headers and interfaces: [../include/README.md](../include/README.md)
- Source layout: [../src/README.md](../src/README.md)
