# Further Optimization

Compute and wall-clock optimization opportunities beyond the compression stack (Tiers 0-3). All optimizations apply to both 2D axisymmetric (NTH=1) and full 3D cylindrical modes unless noted.

**Last updated:** 2026-03-27
**Baseline:** Apple M4 Max (16 cores, 48 GB RAM), Kessler microphysics, RK3 time stepping.

---

## Measured Bottleneck Profile

| Phase | % of Step Time | Source |
|-------|---------------|--------|
| Dynamics + Advection | 86.4% | Profiled 2026-03-20 |
| Diffusion | 7-15% | Within dynamics call |
| Radiation | 1.6% | Per-step overhead |
| Boundary layer | 0.3% | Per-step overhead |
| Output (async) | 0% (background) | Async writer absorbs cost |

**Key finding:** Vertical advection (z-direction) consumes **93% of advection time**. This is the single highest-value optimization target in the entire codebase.

---

## Priority 1: Advection Hot Path

**Files:** `src/numerics/advection/schemes/tvd/tvd.cpp`, `src/numerics/advection/schemes/weno5/weno5.cpp`

### Current State

The TVD advection kernel iterates over (NR x NTH) columns, performing 1D vertical advection per column:

```
for i in 0..NR:
    for j in 0..NTH:
        allocate vector<double> q_col(NZ)     // dynamic allocation
        allocate vector<double> w_col(NZ)     // dynamic allocation
        convert float -> double (NZ elements)  // type conversion
        build_vertical_spacing_column(...)     // per-column GridMetrics lookup
        advect_1d(q_col, w_col, dz, dt, dqdt) // scalar 1D kernel
        convert double -> float (NZ elements)  // type conversion back
        compute CFL (separate NZ loop)         // 4th pass through data
```

### Problems

| Issue | Impact |
|-------|--------|
| `std::vector<double>` allocated per column | malloc/free x NR x NTH per step |
| Float-to-double conversion per element | 2x memory bandwidth, prevents SIMD |
| `build_vertical_spacing_column()` per column | Cache miss + function call overhead when dz is uniform |
| No `#pragma omp parallel` on outer loops | Single-threaded on all cores |
| No SIMD in 1D advection kernel | Scalar throughput on NZ loop |
| CFL computed in separate loop | 4th pass through same data; poor cache reuse |

### Optimization Plan

#### 1A. Eliminate float-to-double conversion (~1.3x)

TVD and WENO5 are numerically robust in float32. The 1D advection kernel should operate directly on `float*` via `row_ptr(i, j)`:

```cpp
const float* q_col = state.q->row_ptr(i, j);
const float* w_col = state.w->row_ptr(i, j);
float* dqdt_col = tendencies.dqdt_adv.row_ptr(i, j);
advect_1d_f32(q_col, w_col, dz, dt, dqdt_col);
```

Eliminates all dynamic allocation and type conversion. Data stays in L1 cache.

#### 1B. Pre-cache vertical spacing (~1.15x)

For uniform grids (no terrain), `dz` is constant across all columns. Compute once at step start:

```cpp
if (!has_terrain) {
    const float dz = grid.dz;  // Single value, reused NR*NTH times
} else {
    // Per-column lookup only when terrain is active
}
```

#### 1C. Add OpenMP to outer loops (~Nx on N cores)

The outer (i, j) loops are embarrassingly parallel — each column's advection is independent:

```cpp
#pragma omp parallel for collapse(2) reduction(max:max_cfl)
for (int i = 0; i < NR; ++i) {
    for (int j = 0; j < NTH; ++j) {
        // 1D advection on this column (no shared state)
    }
}
```

**2D mode (NTH=1):** collapse(2) degenerates to parallelizing over NR columns, which is still effective for NR >= 64.

#### 1D. SIMD vectorization of 1D kernel (~3-5x)

The SIMD infrastructure (`simd_utils.cpp`) already supports AVX2/SSE/NEON with scalar fallback. Extend it with a vectorized TVD MUSCL reconstruction:

```cpp
void tvd_muscl_reconstruct_f32(
    const float* q, int n, float* q_left, float* q_right, LimiterType limiter);
```

With `row_ptr()` providing stride-1 access on the z-axis, the entire NZ loop vectorizes naturally. On AVX2: 8 cells per iteration. On NEON (Apple Silicon): 4 cells per iteration.

#### 1E. Fuse advection + CFL into single pass (~1.2x)

Compute CFL diagnostics inside the advection kernel rather than as a separate loop:

```cpp
advect_1d_fused(q_col, w_col, dz, dt, dqdt_col, &col_max_cfl);
```

Keeps q and w in registers/L1 for the CFL computation.

#### Combined advection impact

On a 16-core M4 Max with NEON:

| Optimization | Speedup | Cumulative |
|-------------|---------|------------|
| Float32 + no alloc | 1.3x | 1.3x |
| Pre-cache dz | 1.15x | 1.5x |
| OpenMP (16 cores) | ~10x | 15x |
| SIMD (NEON 4-wide) | 3x | 45x |
| Fused CFL | 1.2x | 54x |

**Conservative realistic estimate: 8-15x** (not all gains are multiplicative due to Amdahl's law and memory bandwidth limits).

**Wall-clock impact for Production grid (512x256x128):**
- Current: 1,467 ms/step -> ~1,267 ms in advection
- After: ~100-160 ms in advection -> ~300-360 ms/step total
- **2-hour sim: 27h -> 6-8h (overnight run instead of multi-day)**

---

## Priority 2: Dynamics Tendencies

**Files:** `src/dynamics/schemes/supercell/supercell.cpp`, `src/dynamics/schemes/tornado/tornado.cpp`

### Current State

The dynamics schemes compute 5 tendency fields (du_r, du_theta, du_z, drho, dp) using 9 finite-difference stencil evaluations per grid point. OpenMP `collapse(2)` is applied. Vulkan compute shaders exist but may silently fall back to CPU.

### Problems

| Issue | Impact |
|-------|--------|
| `field[i][j][k]` proxy chain accessor (3 proxy objects per access) | ~1.2-1.5x overhead vs `operator()` |
| Stencil points loaded individually (up to 7 loads per derivative) | Cache thrashing; no register reuse |
| GPU dispatch fallback is silent | May be running CPU path without logging |
| Tornado scheme: redundant NTH replication loop (2D -> 3D broadcast) | Wastes bandwidth when NTH > 1 |

### Optimization Plan

#### 2A. Replace proxy chain with direct accessor (~1.2x)

```cpp
// Before: 3 proxy instantiations per access
double val = field[i+1][j][k];

// After: single flatten_index call
double val = field(i+1, j, k);
```

Systematic replacement across dynamics hot paths.

#### 2B. Cache stencil neighborhood (~1.3x)

Load the 7-point stencil (center + 6 neighbors) once per grid point, compute all 9 derivatives from cached values:

```cpp
float ur_im = u_r(i-1, j, k), ur_ip = u_r(i+1, j, k);
float ur_jm = u_r(i, jm, k),  ur_jp = u_r(i, jp, k);
float ur_km = u_r(i, j, k-1),  ur_kp = u_r(i, j, k+1);
// 6 loads serve all 3 derivatives of u_r
```

#### 2C. Validate and instrument GPU dispatch

Add per-kernel logging for dispatch success/failure. For flat domains (common case), GPU dispatch should always succeed:

```
[GPU] supercell_tendencies: dispatched (512x256x128, 16.8M cells)
[GPU] tvd_vertical_flux: dispatched
```

Ensure the Vulkan backend initializes correctly on Apple Silicon via MoltenVK.

#### 2D. Eliminate tornado NTH replication

For 3D tornado mode: compute at j=0 and set all other j columns to reference the same tendency memory, or use a broadcast write:

```cpp
// Instead of copying NTH-1 times:
std::memcpy(du_r_dt.row_ptr(i, jj), du_r_dt.row_ptr(i, 0), NZ * sizeof(float));
```

---

## Priority 3: Time Stepping

**Files:** `src/numerics/time_stepping/schemes/rk3/rk3.cpp`, `src/numerics/time_stepping/schemes/rk4/rk4.cpp`

### Current State

RK3 requires 3 full-field evaluations per step. The stage assembly (linear combination of fields) runs scalar with float-to-double conversion per element.

### Optimization Plan

#### 3A. SIMD field assembly (~2x)

The stage assembly pattern is: `q_new = a * q_old + b * (q_stage + dt * tendency)`

This is a fused multiply-add (FMA) on stride-1 data — ideal for SIMD. Use existing `simd_utils::fma_vectors()`:

```cpp
simd_utils::scale_add_vectors(q_old_ptr, tendency_ptr, dt, q_new_ptr, count);
```

#### 3B. Work in float32 (~1.3x)

RK3/RK4 stage assembly doesn't need double precision. The truncation error from float32 is well below the scheme's order of accuracy for atmospheric fields.

---

## Priority 4: Field3D Access Patterns

**File:** `include/core/field3d.hpp`

### Assessment

The row-major layout `index = i*NTH*NZ + j*NZ + k` is **correct for the dominant access pattern** (z-sweeps in advection and diffusion). No layout change is needed.

The `row_ptr(i, j)` accessor provides zero-overhead stride-1 access to z-columns. This should be the standard accessor in all hot paths.

### Action: Deprecate proxy chain

The `field[i][j][k]` accessor creates 3 temporary proxy objects per access. It should be systematically replaced with `field(i, j, k)` or `field.row_ptr(i, j)[k]` in all performance-critical code:

- `src/dynamics/schemes/supercell/supercell.cpp`
- `src/dynamics/schemes/tornado/tornado.cpp`
- `src/numerics/advection/schemes/tvd/tvd.cpp`
- `src/numerics/advection/schemes/weno5/weno5.cpp`

---

## Priority 5: GPU Compute Offload

### Existing Infrastructure

Vulkan compute shaders already exist for every major kernel:

| Shader | File | Lines |
|--------|------|-------|
| TVD vertical flux | `tvd_vertical_flux.comp` | 207 |
| Radial advection | `advect_radial.comp` | 80 |
| Azimuthal advection | `advect_azimuthal.comp` | 85 |
| Diffusion (cylindrical) | `diffusion.comp` | 85 |
| Supercell tendencies | `supercell_tendencies.comp` | 166 |
| Tornado tendencies | `tornado_tendencies.comp` | 158 |
| Kessler microphysics | `kessler_pointwise.comp` | 202 |
| Kessler sedimentation | `kessler_sedimentation.comp` | 101 |

### Optimization Plan

#### 5A. Validate all kernel dispatches

Ensure the compute backend initializes on target hardware and all kernels dispatch to GPU rather than silently falling back to CPU.

#### 5B. Overlap GPU + CPU work

Dispatch GPU kernels (advection, diffusion) asynchronously while radiation and boundary layer run on CPU:

```
GPU: dispatch_advection()  ──────────────────> wait_advection()
CPU:                        step_radiation()    step_dynamics()
                            step_boundary()
```

Hides GPU dispatch latency behind ~2% of step time.

#### 5C. Unified memory for Apple Silicon

On M4 Max, GPU and CPU share the same memory. Eliminate explicit buffer copies between CPU and GPU fields by mapping Field3D buffers directly as Vulkan storage buffers.

---

## Cartesian Grid: Benefits and Tradeoffs

### Current Cylindrical Advantage

The cylindrical grid (r, theta, z) is purpose-built for vortex-centric simulation:

- **Natural vortex alignment:** Tornado/mesocyclone axis at r=0; finest resolution where it matters most
- **2D efficiency:** Axisymmetric mode (NTH=1) eliminates azimuthal dimension entirely; 50-250x faster than 3D
- **Memory advantage:** 300x128 = 38,400 horizontal points covers a 15 km domain at 50m resolution vs 600x600 = 360,000 for equivalent Cartesian
- **Cylindrical-specific overhead:** ~5-25% (1/r factors, metric terms, centrifugal/Coriolis) — well worth the savings

### What Cartesian Would Enable

| Capability | Cylindrical | Cartesian |
|-----------|-------------|-----------|
| Single vortex (tornado, mesocyclone) | Optimal | Possible but wasteful |
| Multi-vortex interaction | Impossible (r=0 singularity) | Natural |
| Storm splitting / deviant motion | Awkward at domain edge | Natural |
| Mesoscale convective systems | Domain too small | Scalable |
| Topographic forcing (real terrain) | Limited | Standard |
| Nesting into larger models (WRF, MPAS) | Incompatible | Standard |
| Environmental wind shear studies | Constrained | Natural |

### Recommendation

Cartesian support would be a significant expansion (~2-3 weeks to implement) that opens the model to a broader class of phenomena. The recommended approach:

1. **Abstract the coordinate system** behind an interface (metric terms, cell volumes, flux areas)
2. **Keep cylindrical as the default** for vortex work
3. **Add Cartesian as an option** via `grid.coordinates: cartesian` in config
4. **Share all physics/numerics code** — only the metric terms and flux geometry change

The cylindrical grid is not a performance bottleneck — it's a scientific advantage for the model's primary use case. The 5-25% computational overhead from 1/r factors is far less than the memory savings from not needing a 600x600 Cartesian grid.

---

## Estimated Impact Summary

Optimizations ranked by wall-clock impact on a 2-hour production run (512x256x128):

| Priority | Optimization | Est. Speedup | Effort | Current Wall-Clock | After |
|----------|-------------|-------------|--------|-------------------|-------|
| **1** | Advection: OpenMP + SIMD + float32 | 8-15x on advection | High | 27h 24min | **6-8h** |
| **2** | Dynamics: accessor + stencil cache | 1.5x on dynamics | Low | — | **5-6h** |
| **3** | Time stepping: SIMD assembly | 1.3x on RK stages | Low | — | **4-5h** |
| **4** | Field3D: deprecate proxy chain | 1.2x on hot paths | Medium | — | **4h** |
| **5** | GPU offload: validate + overlap | 2-5x if GPU active | Medium | — | **2-3h** |

**Conservative combined estimate: 3-5x total speedup (CPU only), 5-10x with GPU.**

A production 2-hour supercell sim that currently takes 27 hours could run in **5-8 hours** with CPU optimizations alone, or **3-5 hours** with GPU offload validated. This makes overnight production runs routine and opens the door to 4-8 hour storm lifecycle simulations within a 24-hour wall-clock window.
