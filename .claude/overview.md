# CLAUDE.md — TornadoModel / SupercellModel

## Project Overview
Research-grade compressible atmospheric simulation in cylindrical (r, θ, z) coordinates.
C++17 codebase (~45k LOC) with modular physics, factory-pattern scheme selection, and optional Vulkan visualization.

## Build & Run
```bash
make                    # Build bin/tornado_sim (C++17, auto-detects OpenMP)
make vulkan             # Build bin/vulkan_viewer (requires Vulkan SDK + GLFW)
make test               # Full test suite
make run CONFIG=configs/classic.yaml DURATION=3600
```

### Key Build Flags
- `GUI=1` — Enable SFML GUI
- `EXPORT_NPY=1` — NumPy export (default on)
- Compiler: g++/clang++ with `-std=c++17 -O3 -march=native`
- OpenMP: Auto-detected (libomp via Homebrew on macOS)

## Architecture

```
CLI (tornado_sim / vulkan_viewer)
  → Runtime (headless_runtime, runtime_config)
    → Physics Modules (factory pattern, 12 schemes)
      → Numerics (advection, diffusion, time-stepping)
        → Data (Field3D, GridMetrics, FieldContract)
```

### Module Map
| Module | Path | Schemes | Status |
|--------|------|---------|--------|
| Microphysics | src/microphysics/ | kessler, thompson, lin, milbrandt | Stable |
| Boundary Layer | src/boundary_layer/ | ysu, mynn, slab | Stable |
| Turbulence | src/turbulence/ | smagorinsky, tke | Stable |
| Radiation | src/radiation/ | simple_grey (RRTMG planned) | Beta — grey-body only |
| Terrain | src/terrain/ | bell, schar, none | Stable |
| Chaos | src/chaos/ | none, initial_conditions, boundary_layer, full_stochastic | Stable |
| Soundings | src/soundings/ | sharpy | Stable — hand-rolled NetCDF parser |
| Radar | src/radar/ | reflectivity, velocity, zdr | Stable |
| Advection | src/numerics/advection/ | tvd, weno5 | Stable |
| Time-Stepping | src/numerics/time_stepping/ | rk3, rk4 | Stable |
| Diffusion | src/numerics/diffusion/ | explicit, implicit (unused) | Stable |
| Dynamics | src/dynamics/ | supercell, tornado | Stable |

### Entry Points
- **Main sim**: [tornado_sim.cpp](src/core/tornado_sim.cpp) — `main()` at line 315
- **Vulkan viewer**: [main.cpp](vulkan/src/main.cpp)
- **Field validator**: [field_validator.cpp](src/tools/field_validator.cpp)

### Key Data Structures
- `Field3D` ([field3d.hpp](include/field3d.hpp)) — Flattened 3D array container
- `FieldContract` ([field_contract.hpp](include/field_contract.hpp)) — 99-field metadata spec
- `RuntimeConfig` parsed from YAML configs in `configs/`

## Configurations
Pre-built configs in `configs/`: classic, cyclic, elevated, hp, lp, physical_supercell, physical_supercell_storm_tuned, sharpy_lp

---

# CODEBASE AUDIT — Current State

## Critical Issues

### 1. Vulkan Compute: Shell Without Engine
The GPU compute pipeline is **architecturally designed but not implemented**.

- **What works**: Vulkan instance/device initialization, graphics rendering (volume visualization), backend abstraction
- **What's missing**: Compute shaders, VkPipeline creation, buffer management, host↔device transfers, synchronization, command buffer recording
- `backend_dispatch_ready = true` in [compute_kernel_template.cpp](src/core/compute_kernel_template.cpp) is **misleading** — dispatch always falls back to CPU
- [compute_backend_vulkan.cpp](vulkan/src/compute_backend_vulkan.cpp) (1064 lines) handles device init but has **zero compute dispatch code**
- No `.comp` shader files exist anywhere in the project
- The graphics rendering path (volume visualization) is fully working and separate from compute

### 2. .gitignore Breaks Test Reproducibility
```gitignore
tests/          # ignored globally
!tests/         # un-ignored
tests/*         # re-ignored
!tests/test_compute_backend.sh
!tests/test_vulkan_smoke.sh
```
Most test source files (point2_kernel_parity.cpp, radiation_regression.cpp, contract_coverage_guard.cpp, etc.) are **untracked** but required by `make test`. CI will break on a fresh clone.

### 3. CI/CD is Minimal
- Single Ubuntu runner, no macOS testing (despite Homebrew/libomp support)
- No Vulkan smoke tests in CI
- No compiler matrix (clang vs gcc)
- No performance regression tracking

## Duplicated Logic

### lower_copy() — String Lowering
Independently defined in 3+ locations:
- [tornado_sim.cpp](src/core/tornado_sim.cpp) ~line 92
- [dynamics.cpp](src/core/dynamics.cpp) ~line 91
- [advection.cpp](src/advection/advection.cpp) ~line 182

**Fix**: Move to [string_utils.hpp](include/string_utils.hpp) (already exists but not used everywhere).

### Finite-Value Sanitization
Repeated `if (!std::isfinite(x)) x = fallback;` pattern scattered across:
- [dynamics.cpp](src/core/dynamics.cpp) — lines 226-228, 631-641, 686-696
- [radiation.cpp](src/core/radiation.cpp) — lines 181-195 (silent profile substitution)
- Multiple microphysics scheme files
- 476 isnan/isinf checks across codebase, only 3 centralized in field_validation

**Fix**: Centralize into a `sanitize_field()` utility or use FieldValidation consistently.

### Bounds Clamping
572 occurrences of `std::min/std::max/clamp` with repeated patterns like:
```cpp
theta = std::max(theta, 250.0);
theta = std::min(theta, 400.0);
```
[simulation.hpp](include/simulation.hpp) defines shared bounds helpers but they're **not consistently used**.

### Chaos Module: Dual Data Structures
[perturbation_field.hpp](src/chaos/base/perturbation_field.hpp) has **duplicate implementations** for `std::vector<vector<vector<double>>>` AND `Field3D` with identical semantics. Bug fixes in one don't propagate to the other.

## Global State Problems
- Global unique_ptrs for scheme pointers (radiation_scheme, dynamics_scheme, sounding_scheme)
- Global config variables (global_sounding_enabled, global_wind_profile, global_radiation_config, global_chaos_config)
- 233 std::cerr/cout calls with no unified logging framework
- Prevents concurrent model instances and complicates testing

## Large/Complex Files
| File | Lines | Issue |
|------|-------|-------|
| [runtime_config.cpp](src/core/runtime_config.cpp) | ~2500 | Monolithic config parser, needs decomposition |
| [sharpy_sounding.cpp](src/soundings/schemes/sharpy/sharpy_sounding.cpp) | ~2900 | Hand-rolled NetCDF parser (should use libnetcdf) |
| [tornado_sim.cpp](src/core/tornado_sim.cpp) | ~730 | 220 lines of sounding init mixed with main() |
| [headless_runtime.cpp](src/core/headless_runtime.cpp) | Large | Entire sim loop (advection, dynamics, radiation, PBL, turbulence) in one function |

## Radiation Module Gaps
- Only `simple_grey` scheme implemented — clear-sky grey-body only
- RRTMG recognized in factory but throws "not implemented"
- [radiation.cpp](src/core/radiation.cpp) lines 181-195: **silently substitutes** default lapse rate profile when theta/p are out of bounds — should warn
- RadiationColumnStateView allows nullptr for fields without documenting which are required

## Missing Infrastructure
- **No C++ test framework** — shell-based tests only, no Google Test / Catch2
- **No linting/formatting** — no clang-tidy, clang-format config
- **No documentation generation** — no Doxygen
- **SIMD**: [simd_utils.hpp](include/simd_utils.hpp) exists but **no actual SIMD kernel integration** found
- **Implicit diffusion**: Built but unused

## Documentation Drift
- Naming inconsistency: "SupercellModel" vs "TornadoModel" used interchangeably
- [STATUS.md](docs/STATUS.md) doesn't reflect new compute_backend/kernel_template files
- [POINT2_FIRST_KERNEL_OFFLOAD.md](docs/POINT2_FIRST_KERNEL_OFFLOAD.md) claims GPU speedup numbers that **cannot be reproduced** — dispatch always falls back to CPU
- Missing READMEs: src/core/, src/chaos/, src/numerics/

## Positive Patterns
- Factory pattern consistently applied across all physics modules
- Field3D abstraction properly used with flattened memory layout
- FieldContract system (99 fields, 87 exported) provides strong validation foundation
- OpenMP pragmas correctly guarded with `#ifdef _OPENMP`
- Vulkan graphics rendering (volume visualization) is fully functional
- Good config-driven design with 8 pre-built YAML configurations
- Coordinate system (cylindrical) is well-implemented throughout

---

## Conventions
- C++17 standard, no external C++ physics dependencies
- **Allman brace style** — opening braces on their own line, always. Enforced via `.clang-format`. Never use K&R or one-line blocks.
- Factory pattern for all scheme selection (register in factory.cpp per module)
- Field3D for all 3D data; flattened `std::vector<double>` internally
- Configs in `configs/*.yaml`, parsed by runtime_config.cpp
- Tests via `make test` (shell + C++ regression)
- Binaries output to `bin/`
- Data exports to `data/exports/` (NPY format)
- Format code with `clang-format -i <file>` before committing
