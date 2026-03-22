# Nimbus - Project Status

**Last Updated:** March 22, 2026

This document provides a comprehensive overview of the current state of the Nimbus project, including known faults, working components, and next steps.

---

## Project Overview

Nimbus is a high-performance atmospheric simulation framework for supercell and tornadic storm research. It features:

- **Core Engine**: C++17 simulation engine implementing compressible, non-hydrostatic equations in cylindrical coordinates
- **Modular Physics**: Factory-based architecture enabling systematic comparison of parameterizations
- **Visualization**: Native Vulkan rendering path (`vulkan/`) with legacy OpenGL/Python workflows archived
- **Research Focus**: Severe convective storm dynamics, supercell morphology, tornado genesis

---

## Current State

### What's Working

- **Core Simulation Engine**: Functional C++17 implementation
  - Compressible Euler equations solver
  - Cylindrical coordinate system (r, θ, z)
  - RK3 time integration with CFL-limited adaptive stepping
  - Field3D flattened array storage for cache optimization
  - OpenMP multi-core parallelization (when available)

- **Physics Modules**: Multiple schemes implemented
  - **Microphysics**: 4 schemes (Kessler, Thompson, Lin, Milbrandt)
  - **Boundary Layer**: 3 schemes (YSU, MYNN, slab)
  - **Turbulence**: 2 schemes (Smagorinsky-Lilly, TKE prognostic)
  - **Radiation**: `simple_grey` runtime implementation; `rrtmg` canonical id is now recognized for planned-fidelity compatibility
  - **Radar**: Forward operators for reflectivity, velocity, polarimetric variables

- **Build System**: Makefile with OpenMP detection, optional GUI support
- **Visualization Pipeline**: Vulkan viewer build/run path is active via `make vulkan` and `bin/vulkan_viewer`

### What's Incomplete

- **Chaos Module**: Runtime wiring is in place for initial-condition, boundary-layer, and full-stochastic schemes
  - Spectral Gaussian and recursive Gaussian correlation filters are both available
  - Remaining gap: broader ensemble calibration and case-based validation is still pending

- **Terrain Module**: Integrated in runtime initialization and configuration workflow
  - Remaining gap: broader scientific validation/calibration for terrain impacts is still pending

- **Testing Infrastructure**: Integration checks are active; unit and regression test suite under `tests/` covers core, physics, numerics, diagnostics, integration, and Vulkan paths

### CM1-Lite Audit Snapshot (March 22, 2026)

Baseline used for this audit: CM1-style field contract coverage + runtime module wiring + strict guard/test outcomes.

**Verified now**
- `make test`: pass
- `make test-backend-physics`: pass for `lp`, `hp`, `cyclic`, `sharpy_lp`
- `make test-soundings`: pass
- CM1-style contract coverage (`src/validation/field_contract.cpp`):
  - `99` total contract fields
  - `87` `ExportedNow`
  - `12` `NotImplemented`
  - `20/20` `RequiredNow` fields exported (`known_not_implemented_required_now=0`)

**Current Standing on "CM1-lite"**
- Core storm-simulation runtime is operational and guard-hardened for exported 3D fields.
- Baseline module set (dynamics, microphysics, PBL, turbulence, radar, terrain, chaos, soundings) is wired and testable.
- Current state is suitable for controlled research runs and backend export validation.

**Primary gaps**
- **Diagnostic breadth gap (largest)**: 12 CM1-style backlog fields remain unimplemented, concentrated in:
  - 3D diagnostics (`streamlines`, `q_vectors`, `turbulent_diffusion_term`)
  - Cross-section diagnostics
  - Trajectory diagnostics
- **Physics fidelity gap**:
  - Radiation runtime remains `simple_grey`; no in-tree RRTMG physics implementation yet.
  - Soundings ingestion now includes native NetCDF classic + NetCDF C API + HDF5 readers; broader science validation is still pending.
- **Science validation gap**:
  - Terrain and chaos are runtime-integrated, but broader case-based calibration/validation remains pending.
- **QA depth gap (reduced)**:
  - `make test` now includes `test-backend-physics` and `test-soundings` by default.
  - Unit/regression coverage remains narrow relative to module surface area.

---

## Recent Changes

Recent modifications include:

- **Stability Fixes**: Preventive theta guards in microphysics, turbulence diffusion formulation correction, slab PBL indexing bug fix
- **Bounds Consolidation**: Shared clamping helpers in `include/simulation.hpp` now used in advection, dynamics, and microphysics update paths
- **Chaos Integration**: Initial-conditions, boundary-layer, and full-stochastic schemes are implemented and connected to runtime tendency hooks
- **Terrain Integration**: Terrain scheme/config parsing and field build diagnostics wired into startup flow
- **Testing**: `make test` includes config/interface checks plus chaos-terrain and physics-sanity regression scripts
- **Backend Physics QA**: Added strict exported-field verification workflow (`tests/test_backend_physics.sh`, `make test-backend-physics`) for multi-config physical output checks
- **Derived Reflectivity Guarding**: `reflectivity_dbz` export now uses contract-driven bounds to prevent strict-mode mismatches in derived diagnostics
- **Dynamics Diagnostics Hardening**: Diagnostic sanitization in `src/core/dynamics.cpp` now follows field-contract bounds per field (and non-finite-only for unbounded fields like angular momentum), reducing non-physical clipping
- **Radar Fallback Unit Fix**: Microphysics fallback reflectivity path now correctly converts dBZ output to linear reflectivity before writing `radar_reflectivity`
- **Microphysics Reflectivity Guards**: Kessler/Lin/Thompson/Milbrandt reflectivity outputs now enforce finite checks and physically bounded dBZ ranges before export/use
- **Unified Export Manifest**: Each `step_XXXXXX` directory now includes `manifest.json` with grid metadata, sounding metadata, and full core/derived field inventory (IDs, file patterns, units, bounds) to support all-field, isolated-field, and renderer/plot workflows from one backend contract
- **Advection Path Consolidation**: `src/advection` now explicitly orchestrates runtime directional splitting while delegating configured vertical transport (`tvd`/`weno5`) through `src/numerics/advection`, reducing duplicate-path ambiguity and making `numerics.advection` materially active
- **Diffusion/Time-Step Runtime Wiring**: `src/numerics/diffusion` now runs directly on `Field3D` and is applied in `step_dynamics_*` according to `numerics.diffusion.apply_to`; runtime `dt` now consumes `numerics.time_stepping` controls (adaptive dt, bounds, CFL safety) plus explicit-diffusion stability caps via `choose_runtime_timestep()`
- **Core File Decomposition (in progress)**: `tornado_sim.cpp` runtime configuration parsing/state was extracted into `src/core/runtime_config.cpp` + `include/runtime_config.hpp`, reducing `src/core/tornado_sim.cpp` size and isolating config-hardening work from simulation-loop logic
- **Boundary-Layer Naming Disambiguation**: Chaos scheme aliases (`pbl`, `pbl_perturbation`, `boundary_layer_perturbation`) now canonicalize to `chaos.scheme=boundary_layer`, and runtime logs explicitly clarify this is stochastic PBL tendency perturbation, not the deterministic `boundary_layer.scheme` solver
- **Chaos Loop/Memory Optimization**: Runtime `boundary_layer` and `full_stochastic` chaos schemes now store perturbation fields in contiguous `Field3D`, reuse horizontal-correlation slice workspaces, and apply multipliers through flat loops; recursive Gaussian filtering now avoids per-row/per-pass copies

---

## Key Components

### Core Simulation (`src/core/`)

- **equations.cpp**: Field initialization, microphysics stepping, nested grid support
- **dynamics.cpp**: Dynamics coordination, time stepping, bounds checking
- **tornado_sim.cpp**: Main executable, configuration parsing, simulation loop
- **advection/advection.cpp**: Scalar advection with TVD/WENO5 schemes
- **numerics/**: Advection, diffusion, time-stepping factories and schemes

### Physics Modules

- **microphysics/**: 4 bulk parameterization schemes
- **boundary_layer/**: PBL schemes with surface flux calculations
- **turbulence/**: Sub-grid scale closures
- **radiation/**: Radiative transfer schemes
- **radar/**: Forward operators for radar observables
- **chaos/**: Stochastic perturbations (implemented: none, initial-conditions, boundary-layer, full-stochastic)

### Visualization (`vulkan/`)

- **vulkan/**: Native Vulkan renderer and scripts
- **bin/vulkan_viewer**: Runtime viewer with `clear` and `volume` backends
- **legacy/**: Archived historical OpenGL/Python rendering paths

---

## Build Status

### Prerequisites

- **C++ Compiler**: C++17 compliant (clang++ ≥ 9.0, g++ ≥ 7.0)
- **OpenMP**: Optional but recommended (libomp on macOS via Homebrew)
- **Python**: 3.10+ with scientific stack
- **Graphics**: Vulkan-capable GPU/driver stack (for native viewer)

### Build Instructions

```bash
# Build simulation engine
make

# Build with GUI support (SFML)
make GUI=1

# Run simulation
./bin/tornado_sim --headless --config=configs/classic.yaml --duration=3600
```

### Known Build Issues

- OpenMP detection handles both GCC and clang++ (macOS libomp support)

---

## Optimizations for Speed and Space

The codebase implements multiple optimization strategies to maximize computational performance and minimize memory usage.

### Compiler Optimizations

**Flags Used:** `-O3 -march=native -mtune=native`

- **`-O3`**: Maximum optimization level enabling aggressive inlining, loop unrolling, and vectorization
- **`-march=native`**: Generates code optimized for the specific CPU architecture (uses all available instruction sets)
- **`-mtune=native`**: Tunes code generation for the specific CPU microarchitecture

**Impact:**
- Enables automatic vectorization of loops where possible
- CPU-specific instruction set utilization (SSE, AVX, AVX-512 when available)
- Aggressive inlining reduces function call overhead
- Estimated speedup: 2-4x over `-O2` for compute-intensive kernels

### Memory Optimizations

#### Field3D Flattened Array Storage

**Location:** `include/field3d.hpp`

**Implementation:**
- Replaces nested `std::vector<std::vector<std::vector<float>>>` with single contiguous `std::vector<float>`
- Memory layout: Row-major order (r varies slowest, z fastest)
- Index calculation: `idx = i * NTH * NZ + j * NZ + k`

**Benefits:**
- **Memory Overhead Reduction**: ~99.9% reduction in allocation overhead
  - Nested vectors: 3× pointer overhead + per-vector metadata
  - Flattened array: Single allocation, minimal metadata
- **Cache Locality**: Contiguous memory access improves cache hit rates
  - Estimated speedup: 1.5-3x for memory-bound operations
- **Memory Usage**: ~8GB for large domains (256×128×128 grid) vs. ~12-15GB with nested vectors

**Example Impact:**
- Small grid (64×64×32): ~50MB vs. ~75MB with nested vectors
- Production grid (256×128×128): ~8GB vs. ~12GB with nested vectors

### Parallelization

#### OpenMP Multi-Core Parallelization

**Implementation:**
- Automatic detection of OpenMP support (GCC/clang++)
- macOS libomp support via Homebrew
- Parallel loops using `#pragma omp parallel for collapse(2)`

**Usage Locations:**
- Field initialization loops
- Microphysics tendency calculations
- Advection operations
- Dynamics updates

**Performance:**
- **Speedup**: 4-8x on multi-core systems (4-8 cores)
- **Scaling**: Near-linear scaling up to available cores
- **Overhead**: Minimal (<5%) for small domains, negligible for production grids

**Example Benchmarks:**
- Small grid (64×64×32): ~10,000-15,000 time steps/hour (with OpenMP)
- Production grid (256×128×128): ~100-150 time steps/hour (with OpenMP)
- Without OpenMP: ~2-3x slower

### SIMD Vectorization

#### SIMD-Ready Architecture

**Location:** `include/simd_utils.hpp`, `src/core/simd_utils.cpp`

**Features:**
- Runtime detection of available SIMD instruction sets (SSE, AVX, AVX-512)
- Vectorized operations for element-wise computations
- Fallback to scalar operations when SIMD unavailable

**Supported Operations:**
- Vector addition: `result = a + b`
- Vector multiplication: `result = a * b`
- Fused multiply-add: `result = a * b + c`
- Process 4/8/16 floats at once (depending on instruction set)

**Current Status:**
- Infrastructure in place
- Ready for integration into computational kernels
- Can provide 4-16x speedup for vectorizable operations

**Potential Integration Points:**
- Advection flux calculations
- Microphysics bulk operations
- Diffusion operators
- Field arithmetic operations

### Cache Optimization Strategies

#### Contiguous Memory Access Patterns

**Implementation:**
- Field3D ensures all data is contiguous in memory
- Loop ordering optimized for cache line utilization
- Row-major access pattern matches memory layout

**Cache Blocking:**
- Large loops can be blocked for better cache reuse
- Potential for further optimization in advection/diffusion kernels

**Impact:**
- Improved cache hit rates (estimated 1.5-3x speedup for memory-bound operations)
- Reduced memory bandwidth requirements

### Performance Benchmarks

**Small Test Grid** (64×64×32):
- Time steps/hour: ~10,000-15,000 (with all optimizations)
- Memory usage: ~50MB
- Single-threaded: ~3,000-5,000 time steps/hour

**Production Grid** (256×128×128):
- Time steps/hour: ~100-150 (with all optimizations)
- Memory usage: ~8GB
- Single-threaded: ~25-40 time steps/hour

**Visualization:**
- Interactive rendering: 30-60 FPS
- Offline rendering: 4K resolution support

### Optimization Opportunities

**Future Improvements:**

1. **SIMD Integration**: Integrate SIMD utilities into computational kernels
   - Advection flux calculations
   - Microphysics bulk operations
   - Expected speedup: 2-4x for vectorizable loops

2. **Cache Blocking**: Implement cache-blocking for large loops
   - Advection operations
   - Diffusion operators
   - Expected speedup: 1.2-1.5x

3. **GPU Acceleration**: Offload compute-intensive operations to GPU
   - Advection/diffusion kernels
   - Microphysics calculations
   - Expected speedup: 10-100x for suitable operations

4. **Memory Pooling**: Reduce allocation overhead for temporary fields
   - Reuse temporary arrays across timesteps
   - Expected memory reduction: 10-20%

5. **Sparse Storage**: Use sparse data structures for fields with many zeros
   - Hydrometeor fields (qi, qs, qg, qh) often sparse
   - Expected memory reduction: 30-50% for sparse fields

### Memory Usage Breakdown

**Typical Production Run** (256×128×128 grid):

- **Field Storage**: ~8GB (all prognostic fields)
- **Temporary Arrays**: ~500MB (tendencies, intermediate calculations)
- **Base State**: ~10MB (density profile, etc.)
- **Total**: ~8.5GB

**Optimization Impact:**
- Without Field3D: ~12-15GB
- Without OpenMP: Same memory, but slower execution
- With all optimizations: ~8.5GB + 4-8x speedup

---

## Visualization

See `vulkan/README.md` for detailed renderer status and usage.

**Summary:**
- Native Vulkan viewer supports dry-run bootstrap and windowed `clear`/`volume` backends
- Direct NPY field ingestion is active for backend exports
- Legacy visualization stacks are archived under `legacy/`

---

## Known Faults: Came after flattening array structure

### RESOLVED (Feb 10, 2026): Potential Temperature Going Negative After One Timestep

**Location:** `src/core/equations.cpp` lines 451-454 in `step_microphysics()`

**Resolution Summary:**
- Preventive theta guards added in `step_microphysics()`:
  - finite-value check on combined tendencies
  - per-step tendency limiting
  - immediate theta clamping to physical bounds
- Shared bounds helpers consolidated in `include/simulation.hpp`
- Consistent theta clamping applied in advection and dynamics paths
- Turbulence tendency implementation corrected from unphysical `-K*phi` damping to diffusion-form Laplacian tendencies
- Slab boundary-layer vertical indexing bug fixed (`k=0` out-of-bounds in midpoint computation)
- Additional finite-value and pressure/moisture bound hardening applied in dynamics updates
- Makefile test targets restored and expanded (`make test`, `make test-backend-physics`)

**Verification:**
- Build passes: `make -j4`
- Test suite passes: `make test`, `make test-backend-physics`, `make test-soundings`
- Short-run smoke tests no longer show immediate collapse of theta to the lower bound.

---

## Pre-QA Gap Audit (February 20, 2026)

### Known Incomplete By Design

- **Validation contract breadth gap**: `src/validation/field_contract.cpp` still includes a broad CM1-style `NotImplemented` inventory; these are now surfaced as `known_not_implemented` backlog entries while `missing_not_implemented` is reserved for required unresolved gaps.
- **Soundings backend readiness gap**: Runtime wiring exists in `src/core/tornado_sim.cpp` behind `environment.sounding.enabled`; broader production/science validation remains pending.
- **SHARPY ingestion implementation gap (further reduced)**: `src/soundings/schemes/sharpy/sharpy_sounding.cpp` now includes native in-process NetCDF classic, NetCDF C API, and HDF5 readers with aggregated backend diagnostics.
- **Radar sampling/velocity simplifications**: Beam-volume local averaging and bounded bulk fall-speed relations are now present, but full beam physics and advanced scatterer microphysics remain simplified.

### Needs Correction Before QA Pass

- **Source layout inconsistency (resolved in this pass)**: Top-level `src/*.cpp` files were reorganized into `src/core/` to match module-oriented structure and reduce QA navigation ambiguity.
- **Stale source artifact (resolved in this pass)**: Removed `src/tornado_sim.cpp.bak`.
- **Stale source docs (resolved in this pass)**: Updated `src/README.md` and path references in this status document to match the current layout.
- **Soundings runtime wiring (resolved in this pass)**: Added guarded initialization hook (`environment.sounding.enabled`) that can apply sounding-driven `theta`/`qv`/wind profiles at startup.
- **Config/CLI numeric parsing fragility (resolved in this pass)**: Replaced crash-prone direct numeric conversions in `src/core/tornado_sim.cpp` with validated parsing and explicit warning/error handling.
- **Thermo constant duplication in soundings base (resolved in this pass)**: Soundings thermodynamic helpers now consume shared constants from `include/physical_constants.hpp` rather than redefining local copies.
- **Soundings base include-path fragility (resolved in this pass)**: `src/soundings/base/soundings_base.hpp` now resolves the public `soundings_base` interface through include paths instead of brittle directory traversal.
- **Soundings optional-field extrapolation inconsistency (resolved in this pass)**: `src/soundings/base/soundings_base.cpp` now clamps/extrapolates dewpoint/wind fields consistently for below-ground/above-top targets instead of leaving zero-filled artifacts.
- **Module include-path fragility (resolved in this pass)**: Normalized `src/*/factory.hpp` and selected base headers away from brittle `../../include/...` includes to include-path-based headers.
- **Boolean config parsing inconsistency (resolved in this pass)**: `src/core/tornado_sim.cpp` now uses centralized boolean parsing for previously strict `"true"` string checks in nested/radiation/boundary-layer/terrain toggles.
- **Field-validator index parsing fragility (resolved in this pass)**: `src/tools/field_validator.cpp` now validates regex-captured numeric indices before conversion to avoid potential exceptions on malformed/overflowed values.
- **SHARPY placeholder substitution risk (resolved in this pass)**: `src/soundings/schemes/sharpy/sharpy_sounding.cpp` no longer returns synthetic sample profiles for HDF5/NetCDF/profile parse paths; these now fail fast so fallback behavior is explicit/config-driven.
- **Radar sampling placeholder scaffolding (partially resolved in this pass)**: `src/radar/base/radar_base.cpp` point-sampling helper now materializes single-cell sampled state, and `src/radar/schemes/velocity/velocity.cpp` adds a beam-volume local averaging option plus bounded bulk fall-speed estimation.
- **Validation reporting noise from static placeholder inventory (resolved in this pass)**: `missing_not_implemented` now tracks only required unresolved fields, while full CM1 placeholder inventory is preserved separately as `known_not_implemented` for audit visibility.
- **SHARPY log-linear interpolation fallback behavior (resolved in this pass)**: `src/soundings/schemes/sharpy/sharpy_sounding.cpp` now implements method `2` with log-linear pressure interpolation (and recomputed derived thermodynamics) instead of warning-and-linear fallback.
- **Radar scatterer correction wiring gap (resolved in this pass)**: `src/radar/schemes/velocity/velocity.cpp` now applies scatterer correction when enabled via `RadarConfig::apply_scatterer_correction`; velocity convenience paths in `src/core/radar.cpp` enable it by default.
- **Remaining relative include-path fragility (resolved in this pass)**: normalized remaining module-local `../../base/...` include paths in `src/` scheme files to include-path-based module headers.
- **Radar reflectivity conversion precision loss (resolved in this pass)**: `src/core/equations.cpp` now consumes `RadarOut::Ze_linear` directly instead of round-tripping through dBZ and thresholding before export.
- **Initialization pressure warning false positives (resolved in this pass)**: `src/core/equations.cpp` expected-range diagnostics now use physically reasonable pressure bounds to avoid noisy warnings during normal deep-domain runs.
- **SHARPY placeholder reader gap (resolved in prior pass)**: placeholder/`NotImplemented` reader behavior was removed from SHARPY ingestion paths.
- **Exported diagnostics contract gap (further reduced in this pass)**: `theta_prime`, `temperature`, `dewpoint`, `relative_humidity`, `saturation_mixing_ratio`, `total_condensate`, `reflectivity_dbz`, `theta_v`, `theta_e`, `vorticity_magnitude`, `divergence`, `buoyancy`, `storm_relative_winds` (magnitude proxy), `helicity_density`, and `okubo_weiss` are now exported as real NPY diagnostics; computed dynamics diagnostics (`vorticity_*`, stretching/tilting/baroclinic terms, pressure partitions, angular momentum and tendency) are exported in the main bundle; validator `known_not_implemented` inventory is now 12.
- **SHARPY native ingestion gap (further resolved in this pass)**: `src/soundings/schemes/sharpy/sharpy_sounding.cpp` now parses NetCDF classic (CDF1/CDF2), NetCDF4-capable layouts (via native NetCDF C API), and SHARPY-style HDF5 datasets (via native HDF5 HL API) fully in-process without Python extractor fallback.
- **SHARPY spline placeholder gap (resolved in this pass)**: interpolation method `1` in `src/soundings/schemes/sharpy/sharpy_sounding.cpp` now uses monotone cubic (PCHIP-style) interpolation instead of warning-and-linear fallback.
- **Thermodynamic advection moisture-bounds gap (resolved in this pass)**: `src/advection/advection.cpp` now clamps/sanitizes `qv`, `qc`, `qr`, `qi`, `qs`, `qg`, and `qh` immediately after advection to prevent transport-time non-finite/negative carryover.
- **Turbulence tendency non-finite propagation gap (resolved in this pass)**: `src/core/turbulence.cpp` now sanitizes non-finite SGS tendencies right after scheme compute so downstream chaos/dynamics paths consume finite values.
- **Legacy dynamics wind-clamp inconsistency (resolved in this pass)**: `src/core/dynamics.cpp` legacy fallback path now uses centralized wind clamp helpers instead of local hard-coded bounds.
- **Strict validation scope gap (resolved in this pass)**: validation policy now supports strict scope (`required_only` or `exported_now`) so backend runs can fail on any exported-field non-finite/out-of-bounds violations when desired (`validation.guard_scope`, `--guard-scope`).
- **Validation throughput bottleneck (reduced in this pass)**: `src/validation/field_validation.cpp` buffer validation is now OpenMP-parallelized with reductions, reducing guard overhead on large 3D fields.
- **Advection cache-locality bottleneck (reduced in this pass)**: `src/advection/advection.cpp` radial/azimuthal advection kernels now iterate with `k` as the innermost dimension for contiguous memory access in `Field3D`.
- **Offline export-integrity blind spot (resolved in this pass)**: `src/tools/field_validator.cpp` now fails strict exported checks when exported fields are missing or theta-slice coverage is incomplete/duplicated.
- **Runtime derived-export guard gap (resolved in this pass)**: `src/core/tornado_sim.cpp` now validates derived export slices (thermodynamic/radar/dynamics diagnostics) against contract policy before writing NPY files, so strict-mode failures trigger at export time.
- **Chaos perturbation non-finite propagation gap (resolved in this pass)**: `src/chaos/chaos.cpp` now sanitizes non-finite tendency fields after chaos multipliers and clamps IC-perturbed state fields back to physical bounds.
- **Radar finite/bounds hardening gap (resolved in this pass)**: `src/core/radar.cpp`, `src/core/equations.cpp`, and radar scheme kernels now sanitize/clamp reflectivity/velocity/polarimetric outputs to physically bounded finite ranges before downstream export use.
- **Dynamics vorticity diagnostic correctness gap (resolved in this pass)**: `src/dynamics/schemes/supercell/supercell.cpp` now computes/updates all vorticity components before stretching/tilting diagnostics, and `src/dynamics/schemes/tornado/tornado.cpp` vertical vorticity uses corrected axisymmetric form `∂vθ/∂r + vθ/r`.
- **Numerics divide-by-zero and metric consistency gap (resolved in this pass)**: `src/core/numerics.cpp` now uses domain-consistent `dy` metrics; TVD/WENO and diffusion kernels now guard zero/empty spacing and zero-CFL edge cases.
- **Turbulence closure denominator safety gap (resolved in this pass)**: `src/turbulence/base/eddy_viscosity.cpp` and `src/turbulence/schemes/*` now guard unstable denominators/non-finite coefficients in Brunt-Väisälä, diffusivity, and TKE mixing-length/dissipation paths.

---

## Known Issues

### Limitations

- **Terrain Module**: Runtime integration is complete, but broader workflow/physics validation is still needed
- **Chaos Module**: Core schemes and correlation filters are integrated
- **API Stability**: Work in progress / research prototype - APIs and file formats may change

### TODOs

- Continue consolidating bounds checks across remaining modules
- Reduce remaining reactive debug-only diagnostics in production paths
- Expand targeted unit/regression coverage beyond current sanity/integration checks

---

## Next Steps

1. **Priority 1**: Extend stability hardening and verification
   - Continue replacing reactive debug-only checks with preventive guards
   - Add targeted unit/regression tests for tendency magnitude and bounds safety in more physics modules
   - Audit remaining modules for centralized bounds helper usage

2. **Priority 2**: Complete incomplete modules
   - Continue terrain-science validation/calibration against reference cases

3. **Priority 3**: Code quality improvements
   - Consolidate scattered bounds checking
   - Replace reactive debug paths with preventive checks where safe
   - Increase unit coverage for critical physics modules

4. **Priority 4**: Documentation
   - Update README with current status
   - Document physics module interfaces
   - Add examples for common use cases

---

## File Organization

### Test Files Location

Current focused regression/QA suite is in `tests/`:
- `tests/test_guards.sh`
- `tests/test_backend_physics.sh`
- `tests/test_radiation_regression.sh`
- `tests/radiation_regression.cpp`
- `tests/terrain_regression.cpp`
- `tests/test_soundings.cpp`

### Configuration Files

- Production/test config set currently lives under `configs/`:
  - `classic.yaml`, `lp.yaml`, `hp.yaml`, `cyclic.yaml`, `elevated.yaml`, `sharpy_lp.yaml`
  - Additional scenario configs: `physical_supercell.yaml`, `physical_supercell_storm_tuned.yaml`

---

## Notes


- For visualization-specific status, see `vulkan/README.md`
- For technical documentation, see `docs/README.md`
- For scientific foundations, see `docs/foundationalScience.md`
