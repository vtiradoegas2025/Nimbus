# Phase 2 Audit — Performance, Efficiency & Accessibility

**Date:** 2026-03-18
**Scope:** Verification of all Phase 2 sub-sections (2.1–2.9) + codebase quality audit

---

## Phase 2 Completion Matrix

| Section | Description | Status | Notes |
|---------|-------------|--------|-------|
| **2.1** SIMD Library | simd_utils.hpp | **COMPLETE** | 11 primitives (add, subtract, multiply, fma, scale, scale_add, clamp, sanitize_nonfinite, reduce_sum, reduce_min_max, diffuse_1d) with NEON/AVX/SSE/scalar paths |
| **2.1** SIMD Wiring | Hot path integration | **COMPLETE** | 7 call sites: diffusion (2), chaos perturbation (2), sanitization (3). See `jiggly-dreaming-globe.md` |
| **2.1** Field3D Direct Access | row_ptr(i,j) | **COMPLETE** | Bypasses proxy object chain for SIMD-friendly k-column pointer access |
| **2.1** FieldPool | Memory recycling | **COMPLETE** | Used in diffusion, TKE, microphysics |
| **2.1** Cache Blocking | Tiled iteration | **COMPLETE** | TILE_J=32 in advection + explicit diffusion |
| **2.1** OpenMP | Parallel loops | **COMPLETE** | 79+ pragmas; TKE `collapse(3)` fixed; Thompson 4-loop gap fixed |
| **2.2** GPU Shader Wiring | All 7 pipelines | **COMPLETE** | All dispatches wired: supercell, tornado, kessler×2, diffusion, advection×3 |
| **2.3** GPU-Resident Fields | Unified memory | **PARTIAL** | Buffer pool + zero-copy path exist; fields still copied back after **every** dispatch (no cross-timestep residency). Deferred to 2R. |
| **2.4** Async I/O | Double-buffering | **COMPLETE** | AsyncOutputWriter with background thread, condition variables, backpressure |
| **2.5** ZFP Compression | Scientific compress | **PARTIAL** | ZFP library integrated + 3 modes work; **delta encoding + keyframe intervals NOT implemented**. Deferred to 2R. |
| **2.6** Configurable Output | Presets, formats, budget | **COMPLETE** | 5 field presets, 4 output formats, disk budget estimation, YAML parsing |
| **2.7** Shared Memory | In-situ rendering | **COMPLETE** | POSIX SHM transport, ShmWriter, ShmDataset, Vulkan viewer integration |
| **2.8** Hardware Scaling | Detection + safety | **MOSTLY COMPLETE** | CPU/RAM/cache/GPU/SIMD detection; memory-safe grid warnings; auto-config is suggestions only. Deferred to 2R. |
| **2.9** Accessibility | Student builds | **COMPLETE** | Zero-dep CPU build, smoke test, 3 teaching configs, student.yaml, CSV/NPY/ZFP export |

**Overall: ~92% complete. Remaining items deferred to Phase 2R (Architecture Review).**

---

## Gap 1: SIMD Not Wired Into Hot Paths (Section 2.1)

### What exists
- `include/util/simd_utils.hpp` (104 lines) — multi-ISA header with compile-time dispatch
- `src/core/simd_utils.cpp` (362 lines) — full implementations:
  - `add_vectors()` — element-wise addition (AVX: 8 floats/iter, SSE: 4 floats/iter)
  - `multiply_vectors()` — element-wise multiplication
  - `fma_vectors()` — fused multiply-add (with `__FMA__` conditional for AVX)
  - `diffuse_1d()` — 1D Laplacian kernel for diffusion stencils
- Scalar fallbacks in `simd_utils::scalar::` namespace

### What's missing
- **Zero calls** to any SIMD function from any compute kernel across the entire `src/` directory
- Only reference is in `src/core/hardware_info.cpp` for SIMD type detection/reporting
- Microphysics inner loops (lin.cpp, kessler.cpp, milbrandt.cpp, thompson.cpp) do scalar FMA operations
- `diffuse_1d()` is purpose-built for the diffusion stencil but never invoked from diffusion schemes

### Impact
- ~4–8x potential speedup (AVX) on element-wise operations left dormant
- Most impactful in microphysics tendencies (largest inner loops) and explicit diffusion

### Recommended fix
- Profile the top 3 hottest loops (likely: microphysics tendencies, advection flux calc, diffusion operator)
- Refactor loop bodies to batch operations for SIMD width
- Apply `fma_vectors()` to microphysics element-wise FMA operations
- Apply `diffuse_1d()` to explicit diffusion stencil in `src/numerics/diffusion/schemes/explicit/explicit.cpp`

---

## Gap 2: TKE Missing OpenMP Parallelization (Section 2.1)

### What exists
- 51+ `#pragma omp` directives across the codebase covering advection, dynamics, microphysics, diffusion, validation, chaos
- Consistent use of `collapse(2)` for 2D-loop collapse
- Thread-local vectors in compute_kernel_template.cpp to avoid false-sharing

### What's missing
- `TKEScheme::compute_eddy_coefficients_from_tke()` in `src/turbulence/schemes/tke/tke.cpp` (lines 186–231) — triple-nested loop over NR×NTH×NZ with **no OpenMP pragma**
- Runs on a single thread for ~1M+ iterations at production grid sizes

### Impact
- ~8–16x speedup on 16-core systems for TKE eddy coefficient computation
- TKE scheme may not be the default turbulence scheme, but when selected it represents a meaningful fraction of step time

### Recommended fix
- Add `#pragma omp parallel for collapse(3)` to the triple-nested loop
- Verify no loop-carried dependencies (the computation reads from TKE field and writes to separate eddy coefficient fields — should be safe)

---

## Gap 3: GPU Field Residency Is Per-Dispatch Only (Section 2.3)

### What exists
- **Unified memory detection** in `vulkan/src/compute/compute_backend_vulkan.cpp` (lines 2504–2531)
  - Checks for HOST_VISIBLE | DEVICE_LOCAL memory type
  - Sets `has_unified_memory_` flag
  - Logs: "unified memory detected — zero-copy buffer path enabled"
- **Zero-copy path** on unified memory (lines 2453–2470)
  - Uses single HOST_VISIBLE | DEVICE_LOCAL buffer (no staging buffer)
- **GPU buffer pool** in `vulkan/include/compute/gpu_buffer_pool.hpp` (224 lines)
  - Persistent pool with acquire/release semantics
  - Grows on demand, bounded to high-water mark
  - Dual buffer support (staging + device) for discrete GPUs
- **Dispatch helpers** (`dispatch_field3_kernel`, `dispatch_multi_field_kernel`)
  - Handle both unified and discrete memory paths
  - Proper memory barriers for coherency

### What's missing
- **GPU-resident field snapshots between timesteps** — fields are copied back to CPU after every dispatch
- **No persistent GPU field objects** — no Field3D equivalent living on GPU across timestep boundaries
- **No GPU-to-GPU field chains** — each dispatch reads from CPU, writes to CPU (H2D/D2H each time)
- **No GPU state accumulation** — tendencies must be downloaded before the next kernel can use them

### Architecture note
The current buffer pool is optimized for **latency-bound single-kernel dispatch**, not **throughput-bound multi-kernel pipelines**. For true GPU residency:
1. Simulation state would need to reside in GPU memory between kernels
2. GPU-side accumulation (no H2D/D2H between dependent computations)
3. Only export results when needed for I/O or diagnostics

### Impact
- On discrete GPUs: PCIe round-trips per kernel dispatch (~5–15μs each)
- On unified memory (Apple Silicon): the copies are nearly free (shared physical RAM), so the practical impact is minimal on the primary dev platform
- Becomes meaningful if/when multiple kernels chain within a single timestep

### Recommended fix
- Introduce a `GPUFieldSnapshot` holding all simulation state on GPU
- Load at timestep start, unload at end (or when I/O is needed)
- Chain kernel dispatches GPU-side within a timestep
- Priority: **Low on Apple Silicon** (zero-copy already eliminates the main bottleneck), **High on discrete GPUs**

---

## Gap 4: ZFP Delta Encoding + Keyframes Not Implemented (Section 2.5)

### What exists
- **ZFP library integration** — conditional compilation via `ZFP=1` build flag
  - Makefile lines 68–78: detects brew install, links `-lzfp`, defines `-DHAVE_ZFP`
- **`write_zfp_3d()` function** in `src/core/output_writer.cpp` (lines 160–291)
  - Supports 3 ZFP modes: accuracy, precision, rate
  - Binary file format with magic "ZFP3", version, mode, dimensions, parameters
  - Output path: `field_name.zfp3d`
- **ZFP configuration** in `include/core/output_config.hpp` (lines 61–68)
  - `zfp_tolerance` (double, default 1.0e-4)
  - `zfp_mode` (string: "accuracy" | "precision" | "rate")
  - `zfp_rate_bps` (int, default 8)
- **3D chunked format** available via `OutputFormat::npy_3d` (one 3D file per field instead of 25,344 per-theta 2D files)

### What's missing
- **No delta encoding between timesteps**
  - No keyframe interval tracking in OutputConfig
  - No delta computation logic in output_writer.cpp
  - No fields storing previous timestep snapshot for delta calculation
  - No per-timestep compression mode selection (keyframe vs delta)
- **No `keyframe_interval` config parameter** (plan line 99: `keyframe_interval: 10`)
- **No temporal compression infrastructure**
  - No mechanism to compare consecutive snapshots
  - No delta field creation or selective storage
- **Default format is still `npy_2d_slices`** — users must explicitly opt into `npy_3d` or `zfp`

### Impact on compression targets (from plan.md)

| Method | Target Ratio | 2hr Run | Current Status |
|--------|-------------|---------|----------------|
| Raw NPY (current default) | 1× | 2.2 TB | Available |
| ZFP alone (ε=1e-5) | 10–50× | 44–220 GB | **Implemented** |
| ZFP + delta encoding | 30–150× | 15–75 GB | **NOT implemented** |
| ZFP + delta + selective fields | 200–1000× | 2–10 GB | **NOT implemented** |

Without delta encoding, the achievable compression is ~5–10× (ZFP raw), not the 30–150× target.

### Recommended fix
1. Add `keyframe_interval` to `OutputConfig` (default: 10)
2. Track previous snapshot in `AsyncOutputWriter` (or at export time)
3. Compute delta: `delta[i] = current[i] - previous[i]` for non-keyframe steps
4. Store keyframe flag in file header or naming convention
5. Implement delta decompression reader for analysis tools
6. Change default format from `npy_2d_slices` to `npy_3d` (256× fewer files)

---

## Code Quality Audit Findings

### High Priority

#### 1. Factory Boilerplate Duplication (12 instances)

**Location:** All `src/*/factory.cpp` files (dynamics, microphysics, turbulence, boundary_layer, radiation, terrain, chaos, soundings, radar, advection, diffusion, time_stepping)

**Issue:** Each factory repeats near-identical code:
- String normalization function (trim_and_lower with scheme-specific aliases)
- CSV builder function (`available_schemes_csv()`)
- Exception handling with identical messaging patterns
- Schema/id lookup via if-else chains

**Examples:**
- `src/dynamics/factory.cpp` (lines 23–54): normalize + available_schemes_csv
- `src/microphysics/factory.cpp` (lines 25–43): nearly identical structure
- `src/turbulence/factory.cpp` (lines 18–35): uses raw lambda instead of string_utils — inconsistent

**Estimated duplication:** ~100 lines of boilerplate across 12 files

**Recommended fix:** Extract a template-based factory builder or common factory trait parameterized by scheme type, name map, and alias map. Normalize all to use `tmv::strutil::trim_and_lower()`.

---

#### 2. Mixed Error Handling Patterns

**Issue:** No consistent error contract across modules:
- **Factories:** ~104 sites use `throw std::runtime_error(...)` — mostly consistent
- **I/O routines:** ~130 sites use `return false` or `return nullptr`
- **Terrain factory:** Falls back with warning instead of throwing (inconsistent with all other factories)
- `src/radiation/factory.cpp` (lines 58–67): throws for unimplemented schemes AND unknown schemes (two different throw paths)

**Risk:** Silent failures possible where a factory silently falls back vs throws.

**Recommended fix:** Document and enforce contract: factories throw on unknown/unimplemented schemes; I/O routines return bool. Update terrain factory to throw.

---

#### 3. Logging Inconsistency

**Issue:** ~75+ sites use raw `std::cout`/`std::cerr` with ad-hoc string prefixes:
- `src/core/dynamics.cpp:965,971` — `"[DYNAMICS DEBUG]"` to std::cout
- `src/chaos/chaos.cpp:493` — `"[CHAOS DEBUG]"` to std::cout
- `src/core/boundary_layer.cpp:46,53,75–77` — mixed std::cerr and std::cout
- `src/core/headless_runtime.cpp:62–66,317` — std::cout for exports

**`include/util/log.hpp` exists** but is not used uniformly across the codebase.

**Recommended fix:** Replace ad-hoc logging with centralized logger using `include/util/log.hpp`. Support log levels (DEBUG, INFO, WARN, ERROR). Allow suppression in headless/benchmark mode.

---

### Medium Priority

#### 4. Turbulence Factory Uses Inline Lambda

**Location:** `src/turbulence/factory.hpp` (lines 21–24)

**Issue:** Uses an inline lambda for string normalization (`std::tolower` in a loop) while all other 11 factories use `tmv::strutil::trim_and_lower()`.

**Recommended fix:** Replace with `tmv::strutil::trim_and_lower()` for consistency.

---

#### 5. Very Large Files

| File | Lines | Issue |
|------|-------|-------|
| `src/core/headless_runtime.cpp` | ~2936 | `run_headless_simulation()` spans ~1800 lines |
| `src/core/runtime_config.cpp` | ~2509 | Monolithic config parsing |
| `src/soundings/schemes/sharpy/sharpy_sounding.cpp` | ~2893 | NetCDF parsing + interpolation mixed |

**Note:** These are already flagged in Phase 5.1 of the plan as refactoring targets. Not a Phase 2 blocker.

---

#### 6. Microphysics Scheme Initialization Duplication

**Location:** `src/microphysics/schemes/{kessler,lin,thompson,milbrandt}/`

**Issue:** Each scheme repeats identical initialization (~10 lines):
- Resize output field tensors to (NR, NTH, NZ)
- Call `thermodynamics::convert_theta_to_temperature_field()`

**Recommended fix:** Factor into a base class helper method (e.g., `MicrophysicsScheme::init_output_fields()`).

---

### Low Priority / Style

#### 7. simd_utils.hpp Include Guard Inconsistency

**Location:** `include/util/simd_utils.hpp`

**Issue:** Uses `#ifndef SIMD_UTILS_HPP` / `#define SIMD_UTILS_HPP` while all other headers (~35+ files) use `#pragma once`.

**Recommended fix:** Replace with `#pragma once`.

---

#### 8. Soundings Dual Header

**Location:**
- `include/data/soundings_base.hpp` (4149 bytes) — full SoundingData struct and SoundingScheme interface
- `src/soundings/base/soundings_base.hpp` (1216 bytes) — thin wrapper that includes the above

**Issue:** Both exist after the directory restructure. The `src/` version wraps the `include/` version — functionally correct but confusing.

**Recommended fix:** Verify consumers and consider consolidating or renaming to make the relationship explicit.

---

## Priority Order for Remediation

1. **Gap 2: TKE OpenMP** — lowest effort, direct speedup (add one pragma, verify safety)
2. **Finding 4: Turbulence factory lambda** — trivial consistency fix
3. **Finding 7: simd_utils include guard** — trivial style fix
4. **Finding 2: Error handling contract** — document + fix terrain factory
5. **Finding 3: Logging unification** — replace ad-hoc logging with log.hpp
6. **Finding 1: Factory deduplication** — extract template helper (medium effort)
7. **Gap 1: SIMD hot-path wiring** — profile first, then apply to top 3 loops
8. **Gap 4: ZFP delta encoding** — implement keyframe + delta infrastructure
9. **Finding 6: Microphysics init duplication** — factor to base class
10. **Gap 3: GPU field residency** — low priority on Apple Silicon, design for future discrete GPU support
