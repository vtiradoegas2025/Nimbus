# Grid Architecture -- Analysis and Known Issues

**Last Updated:** April 27, 2026

This document examines the grid system's contributions to both computational overhead and numerical instability in Nimbus. The grid is not the sole source of either problem, but it is a structural root cause that amplifies both.

---

## Current Grid Architecture

### Storage

Every prognostic and diagnostic variable is stored in a `Field3D` -- a flat `std::vector<float>` in row-major order:

```
index = i * NTH * NZ + j * NZ + k
```

- Stride-1 direction: vertical (k). Vertical column operations (microphysics, TVD/WENO advection) access contiguous memory.
- Horizontal neighbors (i-1, i+1, j-1, j+1) require strides of NTH\*NZ and NZ respectively, crossing cache lines on large grids.

### Coordinate System

Two backends share the same storage layout:

- **Cylindrical (r, theta, z):** For axisymmetric tornado-vortex simulations. Introduces 1/r metric terms in every momentum equation and a coordinate singularity at r = 0.
- **Cartesian (x, y, z):** For supercell simulations. No metric terms, no singularity. Added in Phase A of the Coordinate Backend Plan (April 2026).

### Collocation

All variables (u, v, w, p, rho, theta, moisture) are stored on the same grid points -- an **Arakawa A-grid**. There is no staggering between velocity and thermodynamic variables.

### Dimensions and Spacing

Grid dimensions (`NR`, `NTH`, `NZ`) and spacings (`dr`, `dz`, `dtheta`) are global variables defined in `equations.cpp`. There is no `Grid` object. Coordinate values (radial position, vertical height, angular position) are not precomputed -- they are recomputed from index arithmetic in every loop that needs them.

---

## Overhead Analysis

### Startup: Field Allocation

There are **136 `resize(NR, NTH, NZ, 0.0f)` calls** across the codebase and **80+ distinct Field3D instances** declared across all modules. Each `resize()` call does two things: allocates memory via `std::vector`, then zero-fills every element. Many fields are then immediately overwritten with actual values in `initialize()`, making the zero-fill a wasted pass.

Memory cost by grid tier:

| Grid | Points | Per Field | 80 Fields |
|------|--------|-----------|-----------|
| Student (64x64x32) | 131K | 0.5 MB | ~40 MB |
| Research (256x128x128) | 4.2M | 16 MB | ~1.3 GB |
| Production (800x800x200) | 128M | 512 MB | ~40 GB |

On the production grid, startup involves ~40 GB of allocation and zero-fill before the first timestep. Many of these fields are not needed at initialization -- diagnostic fields (vorticity components, stretching, tilting, baroclinic, angular momentum, pressure decomposition), radar fields (11 Field3D instances), and turbulence buffers (15 Field3D instances) are not used until the simulation loop begins.

### Per-Timestep: Coordinate Recomputation

No coordinate values are precomputed. Every hot loop recomputes them from scratch:

```cpp
// Repeated in dynamics, advection, diffusion -- every timestep, every grid point
double r = i * dr_ + dynamics_constants::eps;     // multiply + add
double advective_r = -ur * dur_dr - (uth / r) * dur_dth;  // division
double coriolis_th = -ur * uth / r;                        // division
double pressure_grad_th = -dp_dth / (rho_safe * r);       // division
```

In the supercell dynamics inner loop, `1/r` appears as a **floating-point division 4-5 times per grid point**. Division costs 10-20 cycles on most CPUs (3-5x more expensive than multiplication). For the production grid at 128M points, this amounts to ~500-640M unnecessary divisions per dynamics call per timestep.

Precomputing `r[i]`, `r_inv[i]`, `z[k]`, `sin_th[j]`, `cos_th[j]` into 1D lookup arrays would eliminate all of these. The memory cost is trivial (NR + NTH + NZ doubles = kilobytes). Every NWP model at scale (CM1, WRF, MPAS) does this.

### Per-Timestep: Inconsistent Singularity Handling

The cylindrical coordinate singularity at r = 0 is handled by adding a small epsilon to avoid division by zero. But the epsilon value is not consistent:

| Location | Value |
|----------|-------|
| `dynamics/schemes/tornado/tornado.cpp` | `dynamics_constants::eps` |
| `dynamics/schemes/supercell/supercell.cpp` | `dynamics_constants::eps` |
| `numerics/advection/advection.cpp` | `1.0e-6` (hardcoded literal) |

This means the effective radial position at i = 0 differs between dynamics and advection, introducing a subtle inconsistency in the computed tendencies near the axis.

---

## Stability Analysis

### Arakawa A-Grid: 2-Delta-x Computational Mode

The collocated grid is the single largest structural contributor to numerical instability. On an A-grid, the centered difference pressure gradient `(p[i+1] - p[i-1]) / (2*dx)` is blind to the value at point `i`. This admits a **checkerboard pattern** -- pressure can oscillate sign between adjacent grid points and the centered stencil sees it as zero gradient.

This 2-delta-x mode is a well-documented problem in computational fluid dynamics (Arakawa & Lamb 1977). It manifests as:
- Spurious pressure oscillations that grow over time
- Non-physical grid-scale noise in velocity fields
- Degraded pressure-velocity coupling, requiring artificial diffusion to suppress

The standard fix is Arakawa C-grid staggering: store velocity components at cell faces and thermodynamic variables at cell centers. The staggered pressure gradient `(p[i] - p[i-1]) / dx` uses adjacent points and cannot support the checkerboard mode. This is documented in STATUS.md as a planned improvement.

### Cylindrical Singularity at r = 0

The cylindrical coordinate system has a geometric singularity at the polar axis. Terms like `u_theta / r`, `u_theta^2 / r`, and `dp/(rho * r)` diverge as r approaches zero. The current mitigation adds a small epsilon:

```cpp
double r = i * dr_ + eps;
```

This avoids NaN/Inf but introduces a small but non-physical radial offset. At i = 0, the effective radius is epsilon rather than zero, which means:
- The centrifugal force `u_theta^2 / r` is finite but artificially large near the axis
- The pressure gradient in theta, `dp_dth / (rho * r)`, is similarly amplified
- The axis boundary condition `u[0] = -u[1]` (antisymmetric ghost cell) further distorts gradients at i = 1

For axisymmetric flows (where fields do not depend on theta), this is manageable -- the symmetry eliminates the problematic terms. For non-axisymmetric flows (supercell hodographs), the Bug 7 analysis in `docs/Journey.md` showed that identical configurations with the wind hodograph zeroed were **17x more stable** than with the standard 35 m/s shear. The Cartesian backend was built specifically to eliminate this class of instability for supercell cases.

### NaN Guards as Symptom Masking

The dynamics code contains **30+ `if (!isfinite(x)) x = 0.0` guards** across all three schemes (tornado, supercell, cartesian). These guards prevent simulation crashes but mask the underlying instability rather than addressing it. If the numerics were stable, non-finite values would never be produced. Each zeroed tendency represents a grid point where the physics was replaced with "do nothing" -- a silent loss of conservation and accuracy.

The guards are a necessary safety net today, but their presence indicates that the grid and numerics produce non-finite values under conditions that should be numerically tractable.

### float Precision and Perturbation Quantities

The dynamics solve for perturbation quantities (p', theta') that are small deviations from a large base state. With float32 (~7 decimal digits):

```
p_base  = 50000.0 Pa   (at ~5 km altitude)
p_prime =    37.5 Pa
p_total = 50037.5       --> only 2-3 significant digits remain for p'
```

Over thousands of timesteps, accumulated round-off in the perturbation quantities propagates through the pressure solver and advection, introducing non-physical noise. This noise can seed or amplify the 2-delta-x mode from the collocated grid. The TVD advection code already mitigates this by casting to double for internal computation, but this practice is not consistent across all tendency accumulation paths.

---

## What the Grid Does Not Cause

Not all overhead or instability traces back to the grid. For completeness:

- **Physics cost:** Microphysics (~15-20% of step time), radiation, PBL, and turbulence are physics-limited, not grid-limited. Their overhead is inherent to the parameterization complexity.
- **GPU transfer cost:** Host-to-device memory copies are proportional to field size but are a Vulkan pipeline concern, not a grid architecture concern.
- **Time stepping stability:** The split-explicit scheme's acoustic substep count and CFL constraint are time-integration properties, not grid properties (though the A-grid's weak pressure-velocity coupling makes the acoustic solve less robust than it would be on a C-grid).
- **Diffusion tuning:** Explicit diffusion coefficients control how aggressively small-scale noise is damped. Under-diffusion allows grid-scale modes to grow regardless of grid type.

---

## Path Forward

Listed in order of impact relative to implementation cost:

### 1. Precomputed Coordinate Lookup Tables

Add 1D arrays for `r[i]`, `r_inv[i]`, `z[k]`, `sin_th[j]`, `cos_th[j]` and derived constants (`inv_dr`, `inv_dz`, `inv_dr2`, `inv_dz2`). Replace all per-grid-point coordinate arithmetic with lookups. Consolidate the epsilon singularity treatment to a single definition.

**Impact:** Eliminates ~500M+ unnecessary divisions per timestep on production grids. Zero fidelity risk (identical values computed once instead of repeatedly). Standardizes singularity handling.

### 2. Lazy Field Allocation

Defer allocation of diagnostic, tendency, and radar fields until first use. The `field_matches_domain()` guard pattern already exists in some modules -- extend it to all 80+ fields. For fields that are immediately overwritten in `initialize()`, skip the zero-fill pass.

**Impact:** Reduces startup allocation from ~40 GB to ~7 GB (14 prognostic fields only) on the production grid. Remaining fields amortize across the first few timesteps.

### 3. Arakawa C-Grid Staggering

Migrate velocity components to cell faces (u at i+1/2, v at j+1/2, w at k+1/2) while keeping thermodynamic variables at cell centers. This eliminates the 2-delta-x computational mode and improves pressure-velocity coupling in the acoustic solve.

**Impact:** Addresses the root structural cause of grid-scale oscillations. Required for long-duration simulations and research-grade fidelity. Significant implementation scope -- touches dynamics, advection, diffusion, boundary conditions, output, and GPU shaders.

### 4. Consistent Double-Precision Accumulation

Ensure all tendency accumulation and time integration paths compute in double precision, even though fields are stored as float. The TVD advection code already does this; dynamics and microphysics tendency application should follow the same pattern.

**Impact:** Reduces round-off noise that seeds grid-scale instabilities. Moderate implementation scope -- audit and modify tendency accumulation loops.

---

## References

- Arakawa, A., and V. R. Lamb (1977): Computational Design of the Basic Dynamical Processes of the UCLA General Circulation Model. **Methods in Computational Physics**, 17, 173-265.
- Klemp, J. B., and R. B. Wilhelmson (1978): The Simulation of Three-Dimensional Convective Storm Dynamics. **Journal of the Atmospheric Sciences**.
- Bryan, G. H., and J. M. Fritsch (2002): A Benchmark Simulation for Moist Nonhydrostatic Numerical Models. **Monthly Weather Review**.
- Nimbus Bug 7 analysis: `docs/Journey.md` (cylindrical singularity impact on supercell stability).
- Coordinate Backend Plan: `docs/CoordinateBackend_Plan.md` (Cartesian backend rationale and implementation).
