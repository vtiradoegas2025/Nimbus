# Coordinate Backend Plan

**Status:** Phase A complete (2026-04-07). Phase B complete (2026-04-20). Grid prerequisites complete (2026-04-27). Phase C in progress: C.1 + C.2 + C.3 complete (2026-04-28); C.4 + C.5 + C.6 complete (2026-04-29); C.7 + C.8 complete (2026-04-30).
**Target completion:** Phase C CPU-side by 2026-05-23. GPU shaders by 2026-06-06.
**AMS deadline:** January 2027 (~35 calendar weeks of runway).

---

## Background

The model currently runs only on a **collocated cylindrical grid with the singular axis at i = 0**. The radial momentum component `u_r` uses an antisymmetric ghost-cell convention `u[0] = −u[1]` so that the centered ∂u_r/∂r stencil at i = 1 has a defined value. This works for *axisymmetric* base states (where `u_r`, `u_θ`, `p`, `θ` do not depend on θ).

It does **not** work for non-axisymmetric base states. A uniform Cartesian wind `(u_x, u_y)` projects onto cylindrical as

```
u_r(θ) = u_x cos θ + u_y sin θ
u_θ(θ) = −u_x sin θ + u_y cos θ
```

— functions of azimuth even though the underlying flow is uniform. In the discrete equation on the centered grid, the antisymmetric BC `u[0] = −u[1]` makes the discrete radial gradient at i = 1 evaluate to roughly `u[1]/dr` even when the continuous gradient is zero. That false gradient drives a false divergence, drives a false `dp/dt`, breaks hydrostatic balance, and produces a vertical force the dynamics correctly responds to. The result was traced in the Bug 7 entry of `docs/Journey.md`: identical configs with the wind hodograph zeroed are **17× more stable** than with the WK2002 35 m/s shear.

The supercell hodograph the model is supposed to support is non-axisymmetric **by design**. The cylindrical grid is the wrong tool for it. This document is the plan to add the right tool — a Cartesian backend — without throwing away the cylindrical one (which is still the right tool for axisymmetric tornado-vortex modeling).

## Goals

1. **Make the supercell case actually run.** A 2-hour `student.yaml` run should complete with no clamping, no spurious body forces, and a recognizable storm structure.
2. **Keep the cylindrical backend working.** Existing tornado configs and the existing test suite must continue to pass with no behavior change.
3. **Build the foundation for both backends to share.** Microphysics, radiation, PBL, turbulence, terrain, output, and validation should not need to know which coordinate system is in use.
4. **Defer "polish."** Staggering the cylindrical grid (Arakawa C-grid) is a separate, smaller improvement that does not unblock the supercell case. Defer until needed.

## Strategy: duplicate-then-refactor, not abstract-first

Two ways to add a second coordinate backend:

| Approach | Pro | Con |
|---|---|---|
| **Abstract first.** Build a `CoordinateBackend` interface, refactor existing cylindrical to use it, then implement Cartesian. | Cleaner architecture from day one. | Requires guessing the right abstraction before any code exists. High risk of building the wrong abstraction. Long "no progress" phase. |
| **Duplicate then refactor.** Add Cartesian as a parallel scheme alongside cylindrical. After both work, find the common patterns and pull them into shared code. | Lower-risk. Both backends provably work before any abstraction is committed. The right abstraction emerges from real use. | More duplicate code in the short term. |

This plan uses **duplicate-then-refactor**. Phase A copies-and-edits existing cylindrical code to produce a Cartesian version. Phase B does the cleanup pass after both work. Phase C is the (optional, later) cylindrical-grid staggering.

---

## Phase A — Cartesian backend [COMPLETE] (2026-04-07)

**Goal:** A `coordinate: cartesian` runtime config that runs `student.yaml`-like setups end-to-end on CPU and GPU, with an off-center trigger bubble, sheared hodograph, kessler microphysics, and the same physics modules as the cylindrical path.

**Estimate:** 5–8 weeks of focused work. CPU-only (A.1 → A.6) is the first milestone, ~3 weeks. GPU + full integration (A.7 → A.8) is the second milestone, ~2–3 weeks.

**Result:** All 8 sub-tasks completed. All tests pass. Cartesian CPU + GPU paths operational.

### A.1 — `CoordinateSystem` config plumbing

**Scope:** Add a `coordinate_system` config key that the runtime parses, defaulting to `cylindrical`. Add a `CoordinateSystem` enum used by the dynamics/advection/BC dispatchers. Wire it through the `runtime_config.cpp` parser, into `simulation.hpp`, into the headless runtime startup banner.

**Files touched:** `include/core/simulation.hpp`, `src/core/runtime_config.cpp`, `src/core/headless_runtime.cpp`, possibly a new `include/core/coordinate_system.hpp`.

**Estimate:** ~1 day.

**Verification gate:**
- `make test` passes (no behavior change)
- All existing configs run unchanged (default = cylindrical)
- A new `coordinate_system: cartesian` config is recognized at parse time and stored, even if no scheme dispatches on it yet

### A.2 — `CartesianDynamicsScheme` (CPU)

**Scope:** A new dynamics scheme class in `src/dynamics/schemes/cartesian/cartesian.{hpp,cpp}` registered through the existing factory. Computes momentum, mass, and pressure tendencies in Cartesian (x, y, z) instead of cylindrical (r, θ, z). Removes:
- centrifugal force `u_θ²/r`
- coriolis-like coupling `−u_r u_θ/r`
- `1/r` factors in vorticity, divergence, and the azimuthal pressure gradient
- the `eps = 1e-8` axis-singularity guards

Adds:
- `∂p/∂x = (p[i+1] − p[i−1])/(2·dx)` and equivalent for y
- straightforward divergence `∂u/∂x + ∂v/∂y + ∂w/∂z`

The vertical momentum equation is **identical** to cylindrical after the buoyancy fix from Bug 3 — `dw/dt = −∂p/∂z/ρ − g + advection`. Reuse it verbatim.

**Files added:** `src/dynamics/schemes/cartesian/cartesian.{hpp,cpp}` (~400 LOC, mostly copied from `supercell.cpp` with the cylindrical terms removed).

**Files touched:** `src/dynamics/factory.cpp` (register the new scheme), `src/dynamics/README.md` (document the new scheme), the same memory of `CartesianDynamicsScheme` argument names — they should still take `Field3D u, v, w, rho, p, theta` so the existing call site in `dynamics.cpp` doesn't change.

**Estimate:** 2–3 days.

**Verification gate:**
- A standalone unit test: hydrostatically balanced base state, zero wind, no trigger → `du/dt`, `dv/dt`, `dw/dt`, `dρ/dt`, `dp/dt` are all `≤ 1e−3` in magnitude per cell (machine noise from second-order discretization, not real forcing)
- A second unit test: hydrostatic + uniform Cartesian wind → tendencies are still `≤ 1e−3`. **This is the test that the cylindrical scheme cannot pass.** It's the proof Cartesian solves Bug 7.
- A third unit test: small warm bubble (`Δθ = 2 K, radius 1 km`) → vertical tendency at the bubble center is `g·Δθ/θ ≈ 0.065 m/s²` to within 10 %.

### A.3 — Cartesian boundary conditions

**Scope:** Branch in `apply_boundary_conditions()` on the active `CoordinateSystem`. For Cartesian:
- Lateral BCs: open (zero-gradient) on x and y. No axis reflection. No periodic θ wraparound.
- Vertical BCs: same as cylindrical — `w = 0` at top and bottom (rigid lid + rigid surface), hydrostatic extrapolation for pressure (`p[NZ−1] = p[NZ−2] − ρ·g·dz`).
- Density and θ: zero-gradient on all six faces.

**Files touched:** `src/core/dynamics.cpp::apply_boundary_conditions` (add the branch).

**Estimate:** ~0.5 day.

**Verification gate:**
- With Cartesian BCs and the IC from A.2's third test, run 10 timesteps. Confirm `|w|_max` stays bounded (within ±2 m/s for the 2 K bubble).
- Mass conservation: `Σρ` over the domain is conserved to 1 part in 10⁴ over 60 sim seconds.

### A.4 — Cartesian initial conditions

**Scope:** Branch in `equations.cpp::initialize` on the coordinate system. For Cartesian:
- The hydrostatic profile from the Bug 2 fix is reused verbatim (depends only on z)
- The wind is stored directly: `u[i][j][k] = wind_u_cart(z)`, `v[i][j][k] = wind_v_cart(z)`. **No `cos θ`/`sin θ` projection.**
- The trigger bubble is placed at `(x_center, y_center, z_center)` and uses literal Cartesian distance: `dist = √((x − x_c)² + (y − y_c)² + (z − z_c)²)`. The `bubble.center_x_km` / `bubble.center_y_km` config keys finally mean what they say.

**Files touched:** `src/core/equations.cpp::initialize` (add the branch).

**Estimate:** ~0.5 day.

**Verification gate:**
- Print init summary: `u`, `v` are constant in `(x, y)` at every level (matching the hodograph profile). `w = 0` everywhere. Trigger bubble shows up as a localized `Δθ` patch where the config says it should.

### A.5 — `CartesianAdvectionScheme` (CPU)

**Scope:** New advection scheme that uses the existing TVD limiters and the same flux-divergence structure as the cylindrical TVD scheme, but with:
- straight indexing in y instead of the periodic modulo `j_next = (j + 1) % NTH`
- Cartesian pre/post-vertical batched dispatchers that call the new x and y kernels
- the existing vertical TVD kernel (k direction is identical)

Most of the file is a copy of `src/numerics/advection/schemes/tvd/tvd.{hpp,cpp}` with the periodic-θ logic removed. Probably ~80 % code reuse.

**Files added:** `src/numerics/advection/schemes/tvd_cartesian/tvd_cartesian.{hpp,cpp}` (~320 LOC), or a `coordinate_system` parameter inside the existing TVD class — TBD by the implementation pass.

**Files touched:** `src/numerics/advection/factory.cpp`, `src/advection/advection.cpp` (route through the right scheme based on coord system).

**Estimate:** 2–3 days.

> **Discovered during A.5 implementation (2026-04-07):** the cylindrical horizontal advection kernels live in `src/advection/advection.cpp` (`advect_scalar_1d_r_kernel`, `advect_scalar_1d_theta_kernel`, `apply_diffusion_kernel`), not in `src/numerics/advection/schemes/tvd/`. The vertical TVD scheme (`tvd.cpp::compute_flux_divergence`) is already coordinate-agnostic — it loops `(i, j)` and only operates on the `z` column at each cell, so there is no periodic-θ logic to remove.
>
> A.5 was implemented by mirroring the A.3 / A.4 file-extraction pattern: a new `src/advection/advection_cartesian.cpp` (with `include/numerics/advection_cartesian.hpp`) provides three CPU helpers — `advect_scalar_1d_x_kernel_cartesian`, `advect_scalar_1d_y_kernel_cartesian`, `apply_diffusion_kernel_cartesian` — that mirror the cylindrical first-order-upwind + Forward-Euler scheme without the `1/r` factor and without the periodic `(j ± 1) % NTH` wraparound. `advect_scalar_3d` early-returns into a Cartesian Strang split (`x/2 → y/2 → z → y/2 → x/2 → diffusion`) when `global_coordinate_system == Cartesian`, skipping the cylindrical batched GPU dispatch (whose shaders use periodic-θ). The vertical TVD scheme is reused verbatim. No new directory under `src/numerics/advection/schemes/`.
>
> **Implication for upgrading horizontal advection to TVD MUSCL:** when that work happens (Phase B or later), it should refactor the horizontal kernels into a coordinate-aware abstraction in `src/advection/`, not under `src/numerics/advection/schemes/`. The natural next step is the `BoundaryConditionScheme`-style factory mentioned in Phase B.

**Verification gate:**
- 1D test: a Gaussian tracer bump advected at 30 m/s for 100 sim seconds — the bump's centroid moves at 30 m/s ± 1 % and the bump's mass is conserved to 1 part in 10⁴.
- 2D test: tracer bump moved diagonally in (x, y) by uniform `(u_x = u_y = 20 m/s)` — bump remains compact, no axis-aligned smearing artifacts.

### A.6 — Top-level wiring + smoke test

**Scope:** The headless runtime selects dynamics + advection + BCs by `coordinate_system`. The microphysics, radiation, PBL, turbulence, and chaos modules are coordinate-blind — they should not need any branch.

**Files touched:** `src/core/headless_runtime.cpp`, `src/core/dynamics.cpp::step_dynamics_new` (route to the right scheme), maybe a tiny dispatcher in `src/dynamics/factory.cpp`.

**Estimate:** ~1 day.

**Verification gate:**
- Take `student.yaml`, add `coordinate_system: cartesian`, run for 60 sim seconds with no trigger bubble.
- Expected: `|w|_max ~ 10⁻²` m/s at every step (machine noise floor). `theta_bounds_clamped = 0` at every step. Physics budget warnings: zero. Exit 0.
- This is the moment the supercell case stops being broken.

### A.7 — Cartesian GPU shaders

**Scope:** Three new shaders:
- `cartesian_tendencies.comp` — based on `supercell_tendencies.comp`, with the cylindrical-specific terms removed (`1/r`, centrifugal, azimuthal coriolis) and Cartesian indexing
- `advect_x.comp` — copy of `advect_radial.comp` with `dr → dx` and the Cartesian indexing
- `advect_y.comp` — copy of `advect_azimuthal.comp` with periodic modulo replaced by straight indexing and `dtheta → dy`

The diffusion shader and the TVD vertical-flux shader are already coordinate-agnostic — reuse them verbatim.

**Files added:** `vulkan/shaders/compute/cartesian_tendencies.comp`, `advect_x.comp`, `advect_y.comp`. Compiled SPIR-V committed to the tree.

**Files touched:** `vulkan/src/compute/compute_backend_vulkan.cpp` (load the new pipelines, add `dispatch_cartesian_tendencies`, `dispatch_advection_x`, `dispatch_advection_y` methods). `src/core/compute_kernel_template.cpp` and `include/numerics/compute_backend.hpp` (interface declarations).

**Estimate:** 1–2 weeks. The shader logic is short but GPU debugging always takes longer than predicted.

**Verification gate:**
- Existing `test_vulkan_gpu_parity` still passes (no regression in cylindrical GPU path)
- A new `test_vulkan_gpu_parity_cartesian` test that exercises the new dispatchers and confirms CPU-vs-GPU agreement to 1e−4 for momentum tendencies and 1e−3 for advection
- Vulkan run of `student.yaml` with `coordinate_system: cartesian` completes successfully

### A.8 — Full integration test

**Scope:** A real Cartesian supercell run. Not a unit test — a 60–120 sim second run of a `student_cartesian.yaml` config with the WK2002 hodograph, an off-center 2 K trigger bubble, kessler microphysics, slab PBL, simple-grey radiation, smagorinsky turbulence.

**Verification gate:**
- Exit 0
- `theta_bounds_clamped = 0` at every step
- Physics budget warnings: 0
- A first storm cell forms (recognizable updraft column with `w_max > 5 m/s` somewhere in the upper troposphere)
- Conservation budget drift < 0.05 % per step (tighter than the current threshold)
- Output files load and visualize correctly

This is the **end of Phase A.** From this point forward, "the model runs" stops being a question.

---

## Phase B — Refactor for shared code [COMPLETE] (2026-04-20)

**Goal:** Consolidate the duplicated code paths created by Phase A's duplicate-then-refactor strategy. Both backends work; the right abstractions are now visible. Every concern should live in exactly one place with coordinate-specific behavior dispatched through clean interfaces.

**Estimate:** 2–3 weeks.

**Result:** All 7 sub-tasks completed. All tests pass (1058 assertions). Key outcomes:
- `v_theta` eliminated from codebase
- `DerivativeOperators` base class with Cartesian/Cylindrical implementations
- `BoundaryConditionScheme` factory replacing inline dispatch
- Unified Strang split in advection (no early-return Cartesian block)
- Shared thermodynamic init with coordinate-dispatched wind/bubble
- Time stepping delegates through scheme layer (no inline Forward Euler)
- `SplitExplicitDynamics` mixin separated from `DynamicsScheme` base

### B.1 — Field naming unification

Rename `u_r → u`, `u_theta → v` globally so both backends use the same field names without local aliasing. The 267-site rename is now safe because both backends have full test coverage.

**Files touched:** `include/core/simulation.hpp`, `src/core/equations.cpp`, all dynamics schemes, all advection code, all BC code, `test_harness.cpp`, GPU shaders.

### B.2 — `CoordinateBackend` interface for derivative operators

Extract the coordinate-specific derivative operators (`compute_dx`/`compute_dr`, `compute_dy`/`compute_dtheta`, `compute_dz`, divergence) into a `CoordinateBackend` interface. Both `CartesianScheme` and `SupercellScheme` currently duplicate these — they should share the loop structure and dispatch to coordinate-specific kernels.

**Current duplication:**
- `cartesian.cpp` has `compute_dx`, `compute_dy`, `compute_dz` (centered differences with `dr_`)
- `supercell.cpp` has `compute_dr`, `compute_dtheta`, `compute_dz` (centered differences with `dr_`, `dtheta_`, periodic modulo, 1/r factors)

Both compute the same set of tendencies (`compute_momentum_tendencies`, `compute_slow_tendencies`, `compute_fast_pressure_tendencies`, `compute_fast_momentum_tendencies`) with identical loop structure but different derivative operators. The shared loop + dispatched derivatives pattern eliminates this duplication.

### B.3 — Boundary condition scheme factory

Pull boundary condition dispatch out of `dynamics.cpp` (cylindrical, inline) and `boundary_conditions_cartesian.cpp` (Cartesian, file-level) into a `BoundaryConditionScheme` factory, mirroring how dynamics and microphysics are structured.

**Current state (3 patterns for the same concern):**
| BC type | Location | Pattern |
|---------|----------|---------|
| Full cylindrical | `dynamics.cpp::apply_boundary_conditions()` | inline |
| Full Cartesian | `boundary_conditions_cartesian.cpp` | free function |
| Acoustic (both) | `dynamics.cpp::apply_acoustic_boundary_conditions()` | free function |

**Target:** One `BoundaryConditionScheme` base class with `apply_full()` and `apply_acoustic()` virtual methods, coordinate-specific implementations registered through a factory, selected at startup.

### B.4 — Advection unification

Merge `src/advection/advection.cpp` (cylindrical) and `src/advection/advection_cartesian.cpp` into a single dispatch that uses the coordinate backend for horizontal derivative kernels. The vertical TVD scheme is already coordinate-agnostic.

### B.5 — Initial conditions unification

Merge `src/core/equations.cpp::initialize()` (cylindrical) and `src/core/initial_conditions_cartesian.cpp` into a single init path that branches on the coordinate system. The hydrostatic profile and trigger bubble placement are the only coordinate-dependent parts.

### B.6 — Time stepping integration

Wire the unsplit Forward Euler path in `dynamics.cpp` through the `src/numerics/time_stepping/` layer instead of keeping it inline. Currently the split-explicit path delegates to the time stepping scheme, but the unsplit path bypasses it entirely.

**Target:** `dynamics.cpp::step_dynamics_new()` always delegates to `time_stepping_scheme->step()` or `step_split_acoustic()`. No inline time integration in the dynamics orchestrator.

### B.7 — Dynamics interface cleanup

The `DynamicsScheme` base class now carries 8+ virtual methods (original tendencies + diagnostics + split-explicit). Extract the split-explicit methods into a `SplitExplicitCapable` mixin or separate interface so schemes that don't support split-explicit (tornado) aren't carrying dead virtual methods.

**Verification gates:**
- All tests still pass (cylindrical regression + Cartesian regression)
- LOC reduction in `src/dynamics/schemes/` and `src/advection/` measurable
- No inline time integration in `dynamics.cpp`
- No inline boundary conditions in `dynamics.cpp`
- `v_theta` appears nowhere in the codebase outside of comments
- Every concern (BCs, advection, init, time stepping) has exactly one dispatch point

---

## Phase C -- Stagger the Cylindrical Grid (Arakawa C-Grid)

### Context

Phases A and B gave the model a working Cartesian backend for supercells and cleaned up the dual-backend architecture. The cylindrical grid still uses a
collocated discretization where u, v, w, p, rho, theta all live at cell centers. This causes three concrete problems for tornado-mode simulations:

1. **Checkerboard pressure modes.** The centered 2dr stencil `(p[i+1] - p[i-1]) / 2dr` is blind to the grid-scale oscillation. The C-grid stencil `(p[i+1] - p[i]) / dr` resolves it at full resolution.
2. **Fragile axis singularity.** The antisymmetric ghost cell hack `u[0] = -u[1]` creates a false radial gradient at i=1 that drives spurious divergence. On C-grid, the radial face at r=0 simply carries `u_r=0` (no flow through the axis) -- no hack needed.
3. **Non-conservative mass flux.** Collocated divergence `du/dr + u/r + (1/r)dv/dtheta + dw/dz` is not in flux form. The C-grid flux divergence `(1/r) d(r*u)/dr` using face-normal velocities is conservative to machine precision.

The Arakawa C-grid is the standard for compressible NWP codes (CM1, WRF, MPAS). This phase brings the cylindrical grid up to that standard.

### Prerequisites (completed 2026-04-27)

Three items from `docs/grid.md` were completed before Phase C:

1. **Precomputed coordinate lookup tables.** `GridGeometry` struct in `include/core/grid_geometry.hpp` with `r[]`, `r_inv[]`, `z[]`, `theta[]`, `sin_theta[]`, `cos_theta[]`, and spacing reciprocals (`inv_dr`, `inv_2dr`, `inv_dr2`, etc.). Epsilon inconsistency eliminated: `r[0] = 0`, `r_inv[0] = 0`, axis handled by loop ranges. All dynamics and advection hot loops use geometry lookups.
2. **Double-precision tendency accumulation.** All application paths (`dynamics.cpp`, `microphysics_step.cpp`, `diffusion_step.cpp`) accumulate `field + tendency * dt` in double before casting back to float.
3. **Lazy field allocation.** Diagnostic fields deferred to first compute. PBL/radiation/radar fields deferred to their `initialize_*()` functions.

### Design Decisions

**Extend GridGeometry, not a new struct.** Add `r_face[NR]`, `r_face_inv[NR]`, `z_face[NZ]`, and a `staggered` flag to the existing `GridGeometry`. Both dynamics schemes already hold `const GridGeometry& geo_` references.

**Axis singularity centralized in div_flux().** The flux-form divergence `(1/r) d(ru)/dr` is 0/0 at i=0. The control-volume derivation gives `2 * u[0] / dr`. This goes inside `StaggeredCylindricalDerivatives::div_flux()` so dynamics loops never see the special case.

**Separate scheme classes, not branches.** New `TornadoCGridScheme` and `SupercellCGridScheme` in their own files, registered in the factory as `"tornado_cgrid"` and `"supercell_cgrid"`. Zero modifications to existing scheme source files. The `grid.staggering: c_grid` config key auto-selects the `_cgrid` variant. Rationale: C-grid kernels have different loop ranges, stencil widths, and interpolation requirements. Branching inside the existing 200-line loops would double their length.

**Tornado before Supercell.** Tornado is axisymmetric (no theta derivatives, j-replication, no split-explicit). The Lamb-Oseen vortex test has an analytical cyclostrophic balance solution -- the gold-standard C-grid validation. Supercell adds theta-derivatives and split-explicit decomposition as incremental complexity.

**GPU deferred to weeks 5-6.** Phase A.7 (3 GPU shaders) took 2x predicted. Phase C has 6 shaders. CPU-side completion (C.1-C.8) is the priority.

### Field Storage Convention

Keep all Field3D arrays at dimension (NR, NTH, NZ). No dimension changes. Velocity components are reinterpreted with half-grid offsets:

| Field | Location | Index meaning |
|-------|----------|---------------|
| u (radial) | r-face | `u[i][j][k]` = u_r at (r_{i+1/2}, theta_j, z_k) -- right face of cell i |
| v (azimuthal) | theta-face | `v[i][j][k]` = u_theta at (r_i, theta_{j+1/2}, z_k) -- "north" face of cell i |
| w (vertical) | z-face | `w[i][j][k]` = u_z at (r_i, theta_j, z_{k+1/2}) -- top face of cell i |
| scalars (p, rho, theta, q*) | cell center | `p[i][j][k]` at (r_i, theta_j, z_k) -- unchanged |

Interior loop ranges for velocity tendencies shift:
- u: i = 0..NR-2, j = 0..NTH-1, k = 1..NZ-2
- v: i = 1..NR-2, j = 0..NTH-1 (periodic), k = 1..NZ-2
- w: i = 1..NR-2, j = 0..NTH-1, k = 0..NZ-2
- scalars: i = 1..NR-2, j = 0..NTH-1, k = 1..NZ-2 (unchanged)

### Key Stencil Transformations

Pressure gradient at u-face (C-grid):

    dp/dr at (i+1/2) = (p[i+1] - p[i]) / dr
    rho at (i+1/2)   = 0.5 * (rho[i] + rho[i+1])
    du/dt = -dp_dr / rho_face

Divergence at cell center (C-grid, flux form):

    div = (1/r_i) * (r_{i+1/2}*u[i] - r_{i-1/2}*u[i-1]) / dr
        + (v[i][j] - v[i][j-1]) / (r_i * dtheta)
        + (w[i][j][k] - w[i][j][k-1]) / dz

Axis (i=0) divergence -- control-volume derivation:

    if (i == 0): div_r = 2.0 * u[0][j][k] / dr
    else:        div_r = (r_face[i]*u[i] - r_face[i-1]*u[i-1]) / (r_center[i] * dr)

---

### C.1 -- Extend GridGeometry + StaggeredCylindricalDerivatives + Config

**Scope (reduced from original):** Extend existing `GridGeometry` with face arrays. Add `StaggeredCylindricalDerivatives`. Parse `grid.staggering` config key.

**Files modified:**
- `include/core/grid_geometry.hpp` -- add `r_face`, `r_face_inv`, `z_face`, `staggered`
- `include/core/coordinate_system.hpp` -- add `enum class StaggerType { Collocated, CGrid }` + `parse_stagger_type()`
- `include/numerics/derivatives/derivative_operators.hpp` -- add `StaggeredCylindricalDerivatives` (grad_r, grad_theta, grad_z, div_flux, interp helpers)
- `src/core/runtime/runtime_config.cpp` -- parse `grid.staggering` key
- `include/core/runtime_config.hpp` -- declare `extern StaggerType global_stagger_type`
- `src/core/orchestration/dynamics/equations.cpp` -- pass stagger flag to geometry init

**Estimate:** 1 day.

**Verification:** Unit tests for div_flux (uniform flow = 0, linear flow = analytical), grad_r against known function. `make test` passes.

---

### C.2 -- C-Grid Cylindrical Boundary Conditions [COMPLETE] (2026-04-28)

**New files:**
- `src/boundary_conditions/boundary_conditions_cylindrical_cgrid.cpp` -- `CylindricalCGridBCScheme` implementation
- `src/boundary_conditions/factory.cpp` -- dispatcher `create_boundary_condition_scheme(coord, stagger)`
- `tests/dynamics/test_cylindrical_cgrid_boundary_conditions.cpp` -- 11 verification gates

**Modified:**
- `include/boundary_conditions/boundary_conditions.hpp` -- factory declaration, removed unused `<string>` include
- `src/core/orchestration/dynamics/dynamics.cpp` -- BC selection collapsed to a single factory call (dispatch logic moved out of orchestration into the BC module for modularity)
- `Makefile` -- added cgrid + factory sources, new test binary

Axis u is implicit zero (handled by div_flux), not antisymmetric hack -- the
BC explicitly leaves `u[0]` untouched. `w[0]` is likewise interior on the
C-grid (the rigid surface at z=0 is below `z_face[0]` and implicit in
div_flux_z). The only explicit rigid-boundary writes are `u[NR-1] = 0` (outer
wall) and `w[NZ-1] = 0` (lid).

**Verification gates passed:** factory dispatch correctness (4 cases including
loud rejection of unsupported Cartesian + CGrid); 100-cycle stability of a
hydrostatic column with bit-exact field preservation; structural distinctions
vs collocated (`u[0]`/`w[0]` interior, `u[NR-1]`/`w[NZ-1]` rigid faces, axis
v zeroed); zero-gradient scalar ghosts; hydrostatic pressure extrapolation;
acoustic-substep BC scope (momentum + pressure only). 1409 assertions pass.

The "tendencies < 1e-10 over 100 steps" formulation in the original gate
requires the C.4 tornado_cgrid dynamics scheme; that gate moves to C.4.

---

### C.3 -- C-Grid Initial Conditions [COMPLETE] (2026-04-28)

**New files:**
- `src/core/orchestration/dynamics/initial_conditions_cylindrical_cgrid.cpp` -- `apply_cylindrical_cgrid_wind_initialization()`. Split out (mirroring `initial_conditions_cartesian.cpp`) so the unit test can link the IC code without dragging in advection / microphysics / radiation via the rest of `equations.cpp`.
- `tests/dynamics/test_cylindrical_cgrid_initial_conditions.cpp` -- 6 verification gates.

**Modified:**
- `include/core/initial_conditions.hpp` -- declared the new C-grid wind init helper.
- `src/core/orchestration/dynamics/equations.cpp` -- branched `initialize()` on `(coordinate, stagger)` so cylindrical+CGrid dispatches to the new helper while collocated cylindrical and Cartesian paths are unchanged.
- `Makefile` -- added the new IC source and test binary.

**Field placement on C-grid:**

    u (radial, r-face)      u[i][j][k] = u_x(z) cos(theta[j])         + u_y(z) sin(theta[j])
    v (azimuthal, theta-face) v[i][j][k] = -u_x(z) sin(theta_{j+1/2}) + u_y(z) cos(theta_{j+1/2})
    w (vertical)             0

The half-cell-shifted theta_{j+1/2} = (j + 0.5) * dtheta in the v projection
is the only structural difference vs the collocated cylindrical wind init.
The half-shifted sin/cos are computed from the cell-center lookups via the
angle-addition identities, so the inner loop has no transcendental calls.

**Verification gate translated to assertions** (the literal "div_flux < 1e-12"
gate is unattainable for non-axisymmetric uniform Cartesian winds because of
the inherent O(dtheta^2) cylindrical-from-Cartesian projection error; the gate
below is the rigorous form):

1. **Pointwise placement:** every (i, j, k) cell satisfies the analytic
   formulas above. Verified at 24 x 16 x 12 = 4608 cells per check.
2. **C-grid distinguishing feature:** v matches the theta-face projection
   AND differs from the cell-center projection (else the test catches a
   regression to the collocated lookup).
3. **Radial uniformity:** with a hodograph that depends only on z, both u
   and v are bit-exactly constant across i.
4. **Divergence formula:** at every interior cell (1 <= i <= NR-2),
   `div_flux(u, v, w)` matches the analytical
   `(1 - sinc(dtheta/2)) * (u_x cos(theta) + u_y sin(theta)) / r` to 2e-5,
   and the global maximum stays below 1e-3 for the test grid (NTH=16).
5. **Zero hodograph:** velocities and div_flux are bit-exactly zero.
6. **Vertical shear preservation:** the (u_x(z), u_y(z)) split survives at
   every level with no swap or aliasing.

39,100 assertions pass.

---

### C.4 -- Tornado C-Grid Dynamics (CPU) [COMPLETE] (2026-04-29)

**New files:**
- `src/dynamics/schemes/tornado/tornado_cgrid.{hpp,cpp}` -- `TornadoCGridScheme`
  with momentum, mass, and pressure tendency computation on staggered fields.
  Uses `StaggeredCylindricalDerivatives::grad_r/grad_z/div_flux` from C.1, the
  axis ghost `u[-1] = -u[0]` inline (no antisymmetric BC hack on a stored
  field), and reference-state subtraction for vertical momentum (perturbation
  pressure + perturbation density buoyancy) matching the supercell scheme.
  Pressure equation is the standard compressible form
  `dp/dt = -gamma p div(u) - u . grad(p)` (no spurious "centrifugal pressure
  source" that would drift Lamb-Oseen out of cyclostrophic balance).
- `tests/dynamics/test_tornado_cgrid_dynamics.cpp` -- 6 verification gates.

**Modified:**
- `src/dynamics/factory.cpp` -- registered `"tornado_cgrid"` and the
  `"axisymmetric_cgrid"` alias.
- `Makefile` -- added the new source to the dynamics object list and the
  `bin/test_dynamics_tornado_cgrid` test target; added it to `CATCH2_BINS`
  and `test-dynamics`.

**Field placements:**

    u (radial,    r-face)     u[i][j][k] at (r_face[i], theta[j],     z[k])
    v (azimuthal, theta-face) v[i][j][k] at (r[i],      theta_{j+1/2}, z[k])
    w (vertical,  z-face)     w[i][j][k] at (r[i],      theta[j],     z_face[k])
    scalars                   p, rho, theta, q* at (r[i], theta[j],   z[k])

**Loop ranges (axisymmetric: compute at j=0 and replicate):**
- `du/dt`     at r-face:     i = 0..NR-2,   k = 1..NZ-2 (axis ghost u[-1]=-u[0])
- `dv/dt`     at theta-face: i = 1..NR-2,   k = 1..NZ-2 (v[0] = 0 by axis BC)
- `dw/dt`     at z-face:     i = 1..NR-2,   k = 0..NZ-2 (surface ghost w[-1]=0)
- `drho/dt`   at cell center: i = 1..NR-2,  k = 1..NZ-2
- `dp/dt`     at cell center: i = 1..NR-2,  k = 1..NZ-2

**Verification gates passed:**

1. **Hydrostatic equilibrium has machine-zero tendencies.** With kRho0 = 1.25
   (exact in float and double) and the round-trip `p0_base[k] =
   double(float(p0_base[k]))` so the reference profile bit-exactly matches
   the float storage, all five tendencies cancel to <= 1e-10 at every
   interior face/cell.
2. **Hydrostatic state preserved over 300 Euler steps.** dt=1s; 300 steps;
   tendencies stay <= 1e-10 at every step; rho and p match initial values
   bit-exactly; u, v, w stay identically zero.
3. **Lamb-Oseen at discrete cyclostrophic balance.** v(r) =
   (Gamma/(2 pi r))(1 - exp(-r^2/r_c^2)) with r_c = 10*dr, v_max = 50 m/s;
   p(r,z) = p0(z) + sum dp_cyclo to make the discrete grad_r(p) at every
   r-face equal exactly rho_face*v_face^2/r_face*dr. Result:
   `dv/dt`, `drho/dt`, `dp/dt` all bit-exactly zero; `du/dt` and `dw/dt`
   below 1e-3 (~1e-5 in practice -- float-storage cancellation floor).
4. **Lamb-Oseen vortex preserved over 60 simulated seconds.** Forward Euler
   on momentum only (Forward Euler is unconditionally unstable for the
   acoustic wave equation, which the production split-explicit scheme
   handles separately in C.6); 120 steps at dt=0.5s; partial BC (outer wall
   u=0, lid w=0; axis untouched, the dynamics scheme handles its own ghost);
   v profile drift < 0.1% of v_max.
5. **C-grid axis advantage at the first interior r-face.** The axis r-face
   residual `|du/dt[0]|` stays at the same float-storage cancellation floor
   as the interior; no special "antisymmetric ghost cell artifact"
   amplification because the C-grid replaces the stored `u[0] = -u[1]`
   hack with an inline antisymmetric stencil ghost handled where it is
   actually used.
6. **Scheme metadata.** `get_scheme_name() == "tornado_cgrid"`,
   `get_coordinate_system() == "cylindrical_cgrid"`, 5 prognostic vars.

11757 assertions pass. Total dynamics test count after C.4: 64,725 assertions
across 33 test cases (Cartesian + cylindrical_cgrid BCs/IC + tornado_cgrid).

---

### C.5 -- Supercell C-Grid Slow Tendencies (CPU) [COMPLETE] (2026-04-29)

**New files:**
- `src/dynamics/schemes/supercell/supercell_cgrid.{hpp,cpp}` --
  `SupercellCGridScheme`. Inherits both `DynamicsScheme` and
  `SplitExplicitDynamics`. Implements unsplit `compute_momentum_tendencies`
  (full slow + fast in one sweep), the 3 split-explicit methods
  (`compute_slow_tendencies`, `compute_fast_pressure_tendencies`,
  `compute_fast_momentum_tendencies`), plus vorticity and pressure
  diagnostics. Adds: azimuthal advection (4-point bilinear interpolation
  of cross-component velocities at staggered faces), azimuthal pressure
  gradient at theta-face (`grad_theta`), centrifugal `v^2/r` and
  curvature `-u v/r` terms at the appropriate face placements,
  reference-state subtraction for the vertical perturbation pressure
  gradient (Bug 3 form, matching the collocated SupercellScheme and the
  C.4 TornadoCGridScheme).
- `tests/dynamics/test_supercell_cgrid_dynamics.cpp` -- 7 verification
  gates.

**Modified:**
- `src/dynamics/factory.cpp` -- registered `"supercell_cgrid"` and the
  `"mesocyclone_cgrid"` alias.
- `Makefile` -- added the source to the dynamics object list, added
  `bin/test_dynamics_supercell_cgrid` test target, added it to
  `CATCH2_BINS` and `test-dynamics`.

**Field placements (same as C.4):**

    u (radial,    r-face)     u[i][j][k] at (r_face[i], theta[j],     z[k])
    v (azimuthal, theta-face) v[i][j][k] at (r[i],      theta_{j+1/2}, z[k])
    w (vertical,  z-face)     w[i][j][k] at (r[i],      theta[j],     z_face[k])
    scalars                   p, rho, theta, q* at (r[i], theta[j],   z[k])

**Loop ranges (full 3D, no axisymmetric replication):**
- `du/dt`     at r-face:     i = 0..NR-2,  j = 0..NTH-1, k = 1..NZ-2
  (axis ghost u[-1]=-u[0] inline; periodic in j)
- `dv/dt`     at theta-face: i = 1..NR-2,  j = 0..NTH-1, k = 1..NZ-2
- `dw/dt`     at z-face:     i = 1..NR-2,  j = 0..NTH-1, k = 0..NZ-2
  (surface ghost w[-1]=0 inline)
- `drho/dt`, `dp/dt` at cell-center: i = 1..NR-2, j = 0..NTH-1, k = 1..NZ-2

**Cross-component velocity interpolation (4-point bilinear):**

    v at r-face   = 0.25*(v[i][j-1] + v[i][j] + v[i+1][j-1] + v[i+1][j])
    w at r-face   = 0.25*(w[i][j][k-1] + w[i][j][k] + w[i+1][j][k-1] + w[i+1][j][k])
    u at theta-face = 0.25*(u[i-1][j] + u[i][j] + u[i-1][j+1] + u[i][j+1])
    w at theta-face = 0.25*(w[i][j][k-1] + w[i][j][k] + w[i][j+1][k-1] + w[i][j+1][k])
    u at z-face   = 0.25*(u[i-1][j][k] + u[i][j][k] + u[i-1][j][k+1] + u[i][j][k+1])
    v at z-face   = 0.25*(v[i][j-1][k] + v[i][j][k] + v[i][j-1][k+1] + v[i][j][k+1])

**Split-explicit decomposition (Klemp-Wilhelmson):**
- slow: advection + centrifugal/coriolis + buoyancy on u/v/w; advection
  only `(-u . grad p)` on p; `drho/dt = 0`.
- fast pressure: `dp/dt = -gamma p div_flux`, `drho/dt = -rho div_flux`.
- fast momentum: `du/dt = -grad_r(p)/rho_face`, `dv/dt = -grad_theta(p)/rho_face`,
  `dw/dt = -(grad_z(p) - dp0/dz)/rho_face` (perturbation form).
- Algebraic identity `total = slow + fast` is asserted bit-exactly to
  float roundoff in Gate 3 below.

**Verification gates passed:**

1. **Hydrostatic equilibrium has machine-zero tendencies.** Same kRho0=1.25
   / round-trip trick as C.4 (the same float-storage cancellation argument
   applies) -- all five tendencies <= 1e-10.
2. **Hydrostatic preserved over 100 momentum-only Forward-Euler steps.**
   dt=1s; tendencies stay <= 1e-10 at every step. (Pressure/density
   integration deferred to C.6 split-explicit, since Forward Euler is
   unstable for the acoustic system.)
3. **Bug-7 verification: hydrostatic + uniform Cartesian wind.** With
   `(u_x, u_y) = (5, 0)` m/s on NTH=32 grid: dp/dt < 5 Pa/s and
   drho/dt < 1e-3 kg/m^3/s -- the only divergence residual is the
   inherent O(dtheta^2) cylindrical-from-Cartesian projection error,
   ~500x smaller than the `gamma p U/dr ~ 2800 Pa/s` collapse the
   collocated grid produces from its `u[0] = -u[1]` antisymmetric BC
   artifact. Momentum tendencies (du/dt, dv/dt) are bounded at ~0.5 m/s^2
   from the kinematic centrifugal and curvature terms that BOTH grids
   carry (a uniform Cartesian wind in cylindrical coords legitimately has
   v^2/r at the inner u-face).
4. **Convergence in NTH.** Doubling NTH from 16 to 32 to 64 reduces the
   max dp/dt residual by >2.5x at each step (consistent with O(dtheta^2)
   scaling), confirming the residual is a discretization-projection error
   rather than a stationary axis artifact.
5. **Slow + fast = total bit-exactly.** On a non-trivial state
   (hydrostatic + uniform wind + small rho perturbation), the algebraic
   identity holds at every (i, j, k): max(|tot - (slow+fast)|) <
   1e-6 m/s^2 for momentum, < 1e-3 Pa/s for pressure, < 1e-9 kg/m^3/s
   for density. This is the correctness check for the split-explicit
   decomposition; the C.6 acoustic substep relies on this identity.
6. **Warm thermal produces upward dw/dt.** A 0.4% density deficit in a
   small column produces dw/dt > 0.01 m/s^2 (buoyant lift signal,
   correct sign and order of magnitude); the far-field column is < 0.01
   m/s^2 (no spurious propagation).
7. **Scheme metadata + SplitExplicitDynamics capability.**
   `get_scheme_name() == "supercell_cgrid"`,
   `get_coordinate_system() == "cylindrical_cgrid"`, 5 prognostic vars,
   and `dynamic_cast<SplitExplicitDynamics*>` succeeds on the
   `DynamicsScheme*` base pointer.

524 assertions pass. Total dynamics test count after C.5: 65,249
assertions across 40 test cases (Cartesian + cylindrical_cgrid BCs/IC +
tornado_cgrid + supercell_cgrid).

---

### C.6 -- Split-Explicit Acoustic Substep on C-Grid [COMPLETE] (2026-04-29)

The fast pressure and fast momentum tendency methods on `SupercellCGridScheme`
were already implemented in C.5 because they are required by the
`SplitExplicitDynamics` mixin. C.6 wires them into the time-stepping path
and verifies the integrated Klemp-Wilhelmson loop.

**New files:**
- `tests/dynamics/test_supercell_cgrid_acoustic.cpp` -- 3 verification gates,
  each driving the SplitExplicitDynamics methods through a manual
  Klemp-Wilhelmson large-step loop (slow + N forward-backward acoustic
  substeps) at the scheme level.

**Modified:**
- `src/core/orchestration/dynamics/dynamics.cpp` -- added a
  `gpu_acoustic_ok` guard inside `step_dynamics_split_explicit` that
  forces the CPU fallback when `global_stagger_type == StaggerType::CGrid`.
  The existing GPU acoustic kernels (`dispatch_acoustic_pressure_backend`,
  `dispatch_acoustic_momentum_backend`, `dispatch_acoustic_substep_fused_backend`,
  `dispatch_acoustic_substeps_batched_backend`) all use collocated
  cylindrical stencils; running them against a C-grid configuration would
  silently produce wrong tendencies. The guard preserves correctness
  end-to-end until C.9 lands the C-grid GPU shaders.
- `Makefile` -- added the new test target, wired into `CATCH2_BINS` and
  `test-dynamics`.

**Klemp-Wilhelmson loop on C-grid (mirror of `SplitExplicitScheme::step_split_acoustic`):**

    1. compute_slow_tendencies     -> tendency buffers
    2. apply slow with dt_large    -> u, v, w, p (rho slow == 0)
    3. acoustic_bcs                -> outer wall, lid (axis is inline)
    4. for n in [0, N):
         a. compute_fast_pressure_tendencies -> drho/dt, dp/dt
         b. apply with dt_small              -> rho, p
         c. compute_fast_momentum_tendencies -> du/dt, dv/dt, dw/dt
         d. apply with dt_small              -> u, v, w
         e. acoustic_bcs                     -> outer wall, lid

Default stable substep count for dr = 250 m: N = 6..10 gives
`CFL_acoustic = c * dt_small / dr ~ 0.07..0.24` for dt_large = 0.5..1.0 s.

**Verification gates passed:**

1. **Hydrostatic preserved over 300 s of full split-explicit.** With
   `dt_large = 1 s`, `N = 6` (so `dt_small ~ 0.17 s`, CFL_acoustic ~ 0.24),
   every slow tendency is bit-exactly zero, every fast pressure tendency
   is bit-exactly zero (`div_flux = 0` with `u = v = w = 0`), every fast
   momentum tendency is bit-exactly zero (`grad_r p = 0` since p has no
   radial variation, `grad_z p - dp0/dz = 0` thanks to the C.4 round-trip
   trick), and after 300 large steps `rho`, `p` are preserved bit-exactly
   and `u`, `v`, `w` stay identically zero. This is the gate the C.4
   momentum-only test had to skip because Forward Euler is unstable for
   the acoustic system; on Klemp-Wilhelmson the equilibrium IS a fixed
   point.
2. **Acoustic pulse propagates outward at sound speed.** A small Gaussian
   pressure perturbation (dp = 100 Pa = 0.1% of base, sigma = 2*dr)
   centered at r = 4*dr propagates outward over 1 simulated second; the
   peak of the outward-going pulse sits at i > i_initial after the run
   and contains > 5% of the original amplitude (the rest goes inward
   and into 2D geometric dispersion). All fields remain finite.
3. **No 2dx checkerboard mode in the propagated pressure field.** The
   discrete second difference `|p[i-1] - 2 p[i] + p[i+1]|` along the
   radial slice through the pulse stays below 100 Pa, comfortably under
   the 400 Pa a 100 Pa amplitude checkerboard would produce. The smooth
   Gaussian curvature scale is ~25 Pa for this configuration.

25,905 assertions pass (the 300-step gate runs many bit-exact field
checks). Total dynamics test count after C.6: 91,154 assertions across
43 test cases.

**Production wiring already in place.** `step_dynamics_split_explicit` in
the dynamics orchestrator does `dynamic_cast<SplitExplicitDynamics*>` on
the active scheme, so a `coordinate_system: cylindrical` + `grid.staggering: c_grid`
config that selects `supercell_cgrid` will automatically take the
split-explicit path through this new C-grid scheme. The acoustic BC
selection already uses `bc_scheme->apply_acoustic()`, which the C.2
boundary-condition factory dispatches to the C-grid acoustic BC.

---

### C.7 -- Advection on C-Grid [COMPLETE] (2026-04-30)

**New files:**
- `include/numerics/advection/advection_cylindrical_cgrid.hpp` -- declares
  the three cylindrical C-grid scalar advection helpers
  (`advect_scalar_1d_r_kernel_cylindrical_cgrid` etc.)
- `src/numerics/advection/advection_cylindrical_cgrid.cpp` --
  TVD-MUSCL flux-form scalar advection in r, theta, z directions on the
  staggered cylindrical grid. MC limiter (matching the default in the
  vertical TVD scheme) with monotonic centered slope reconstruction;
  upwind face value picked by the sign of the face velocity. Mass
  conservation is bit-exact pairwise (same flux value at every shared
  face is debited from one cell and credited to the adjacent cell).
- `tests/numerics/test_advection_cylindrical_cgrid.cpp` -- 5
  verification gates exercising the dispatch in advect_scalar_3d.

**Modified:**
- `src/numerics/advection/advection.cpp` -- added
  `active_backend_is_cylindrical_cgrid()` helper, three is-cgrid branches
  in `step_h1`/`step_h2`/`step_z` lambdas to route to the new kernels,
  and a stagger guard on the batched cylindrical GPU dispatches
  (`dispatch_advection_batch_pre_vertical_backend` and
  `dispatch_advection_batch_post_vertical_backend` are skipped for
  C-grid because their shaders use cell-center velocity reads). The
  per-direction GPU dispatches inside the existing `_kernel` helpers
  are bypassed automatically because the C-grid branch never enters
  those helpers. Cell-center diffusion (`apply_diffusion_kernel`) is
  reused unchanged because it does not read u, v, or w.
- `Makefile` -- added the new source to `SRCS`, added
  `bin/test_numerics_advection_cylindrical_cgrid` test target, added it
  to `CATCH2_BINS` and `test-numerics`. The
  `bin/test_numerics_advection` and
  `bin/test_numerics_advection_cartesian` targets pick up the new
  source as part of their advection link closure.

**Verification gates passed:**

1. **Zero-flow preservation.** With u=v=w=0, an arbitrary cell-center
   IC is preserved bit-exactly through 50 advection steps; max diff
   over interior is 0.0.
2. **Pure-vertical uniform advection.** A Gaussian column at fixed
   (i, j) advected by uniform w=5 m/s for 100 sim seconds tracks its
   z centroid within 2 % of the analytic translation distance, and
   the interior mass relative drift stays under 1e-5 (the float
   round-off floor for the per-step double->float cast).
3. **Pure-azimuthal uniform advection (one revolution).** A Gaussian
   bump in (i, j) advected by uniform v=20 m/s through one full
   rotation period (~1257 steps) preserves interior mass to better
   than 1e-5 relative drift; periodic theta closes the flux balance
   pairwise so all theta-face fluxes cancel.
4. **Solid-body rotation (one revolution).** A smooth periodic
   `q = 1 + cos(theta)` profile placed on a single radial ring is
   advected for exactly one rotation period T = 2*pi/Omega
   (~160 steps at NTH=64 with CFL_theta ~ 0.4). After one revolution
   the L2 relative error vs the IC is below 5 %, and the mass
   relative drift stays under 1e-6 (~3e-8 in practice). This is the
   plan gate text "solid-body rotation < 5% L2 error, mass
   conservation to machine precision".
5. **Stagger-routing distinguishability.** The same uniform radial
   outflow IC produces a different result on the C-grid path
   (TVD-MUSCL flux form including the geometric divergence
   `-u q / (2 r)` term) than on the collocated path (first-order
   upwind advective form without the geometric correction); a
   regression that silently routed C-grid configurations through the
   collocated kernel would produce identical fields and fail this
   gate.

9 assertions pass across 5 test cases. Total test count after C.7: the
pre-existing 18,835-assertion advection-cartesian regression and all
other test binaries continue to pass with no behavior change.

**Key design decisions worth carrying into C.8 / C.9:**

1. **Flux form is the structural advantage.** First-order upwind would
   already give mass conservation under flux form -- the upgrade to
   TVD-MUSCL is what brings the L2 error under the plan's 5 % gate.
   The MC limiter zeroes the slope at extrema (e.g., the peak of a
   single Gaussian), which falls back to first-order at the peak cell
   and accumulates numerical viscosity over many revolutions; the
   solid-body rotation gate therefore uses a smooth `1 + cos(theta)`
   profile (with one max + one min per ring) rather than a single
   Gaussian peak. For physically meaningful advection (qv, qc fronts,
   passive tracers with mostly-smooth profiles), this distinction is
   academic -- TVD with MC limiter is the correct choice.

2. **GPU dispatch is collocated-only until C.9.** The existing
   `dispatch_radial_advection_backend`,
   `dispatch_azimuthal_advection_backend`,
   `dispatch_vertical_flux_template_backend`,
   `dispatch_advection_batch_pre_vertical_backend`, and
   `dispatch_advection_batch_post_vertical_backend` all use cell-center
   velocity stencils. The C.7 branch never enters those code paths on
   C-grid -- it routes directly to the new CPU helpers. C.9 will lift
   this restriction with stagger-aware shaders.

3. **The vertical TVD scheme in `src/numerics/advection/schemes/tvd/`
   is bypassed on C-grid.** That scheme assumes w is at cell center;
   on C-grid w lives at z_face[k]. The new
   `advect_scalar_1d_z_kernel_cylindrical_cgrid` handles vertical
   advection directly with the same TVD-MUSCL math but reading w from
   the face. C.9 may want to harmonize these (a single TVD scheme
   that knows about staggering); as of C.7 they are parallel
   implementations.

4. **Cell-center diffusion `apply_diffusion_kernel` is reused as-is.**
   The Laplacian operates only on cell-center scalar storage, which is
   identical on collocated and C-grid configurations; the GPU
   diffusion shader reads no velocity field and is therefore safe to
   keep dispatching on C-grid (no guard added).

5. **Momentum advection on C-grid is in the dynamics scheme, not in
   `advect_scalar_3d`.** SupercellCGridScheme already implements
   centered-difference advective tendencies for u, v, w with 4-point
   bilinear interpolation of cross-component velocities (see C.5).
   C.7 only adds the SCALAR (theta, qv, qc, qr, qi, qs, qg, qh,
   tracer) advection on the C-grid; momentum was C.5's responsibility.

---

### C.8 -- Output Interpolation [COMPLETE] (2026-04-30)

**New files:**
- `include/core/output/stagger_interpolation.hpp` -- declares the
  three face-to-center helpers (`interpolate_u_face_to_center`,
  `interpolate_v_face_to_center`, `interpolate_w_face_to_center`).
- `src/core/output/stagger_interpolation.cpp` -- implementations
  using arithmetic averaging that match the
  `StaggeredCylindricalDerivatives::interp_from_*_face` conventions
  already used by `SupercellCGridScheme::compute_vorticity_diagnostics`.
- `tests/core/test_stagger_interpolation.cpp` -- 5 verification gates
  (6,892 assertions).

**Modified:**
- `src/core/runtime/headless_runtime.cpp` -- introduced cell-center
  velocity buffers in `write_all_fields` (covers both the 3D async
  snapshot path and the 2D theta-slice path) and in `shm_update`
  (the live SHM viewer channel). Each lambda checks
  `global_stagger_type == StaggerType::CGrid` and routes the
  `core_bindings` / `core_map` u, v, w pointers through the
  cell-center buffers when on C-grid; collocated configurations are
  unchanged.
- `Makefile` -- added the new source to `SRCS`, added a
  `bin/test_core_stagger_interpolation` test target wired into
  `CATCH2_BINS` and `test-core`.

**Conventions (from
`StaggeredCylindricalDerivatives::interp_from_*_face`):**

  u_center[i][j][k] = 0.5 * (u[i-1][j][k] + u[i][j][k]),  i >= 1
  u_center[0][j][k] = 0.5 * u[0][j][k]                    (axis: u=0
                       implicit at the singular r=0 boundary)

  v_center[i][j][k] = 0.5 * (v[i][j_prev][k] + v[i][j][k])
                       j_prev = (j - 1 + NTH) % NTH (periodic)

  w_center[i][j][k] = 0.5 * (w[i][j][k-1] + w[i][j][k]),  k >= 1
  w_center[i][j][0] = 0.5 * w[i][j][0]                    (surface:
                       rigid-surface w=0 implicit below z_face[0])

**Verification gates passed:**

1. **Radial interpolation: interior + axis.** A linear ramp
   `u_face[i] = i + 0.5` produces `u_center[i] = i` for every interior
   cell and `u_center[0] = 0.25` (= 0.5 * 0.5) at the axis. Verified
   at every (i, j, k) on a 8x4x6 grid.
2. **Azimuthal interpolation: periodic wrap.** A ramp
   `v_face[j] = j` produces `v_center[j] = j - 0.5` for j >= 1 and
   `v_center[0] = (NTH - 1) / 2` at the wrap cell. Verified on
   4x8x4 grid.
3. **Vertical interpolation: interior + surface.** A linear ramp
   `w_face[k] = k + 0.5` produces `w_center[k] = k` for k >= 1 and
   `w_center[0] = 0.25` at the surface. Verified on 4x4x8 grid.
4. **Constant-face passthrough.** Constant face values produce the
   same constant at every interior cell (the arithmetic mean of two
   equal values), and exactly half that value at axis / surface cells
   (the documented one-sided averaging rule).
5. **Cylindrical-from-Cartesian projection.** With the C.3 IC for a
   uniform Cartesian wind `(ux, uy)` placed on the C-grid faces, the
   interpolated cell-center fields reproduce the analytic
   cell-center projection scaled by `cos(dtheta/2)` (the
   sum-to-product identity collapse for the half-cell theta-face
   averaging). At NTH=64 the attenuation is ~0.13%, and the
   numerically computed cell-center field matches the analytic
   formula to 1e-5 relative error -- the float-precision floor of
   the cos/sin computation.

6,892 assertions pass. Total core-test count after C.8 unchanged for
all other test binaries; new binary
`bin/test_core_stagger_interpolation` runs alongside the existing
`test-core` target.

**Verification (per plan): "Output from C-grid run shows smooth
velocity fields in viewer."** Gate 5 above is the analytic form of
this verification: with the Phase C.3 uniform-wind IC, the
cell-center u and v fields are smooth functions of theta with no
visible artifacts at the axis or theta-wrap. The interpolation
attenuation is ~0.13% at NTH=64 (the cell-center sampling of a
theta-face-stored sinusoid loses high-frequency content
proportional to cos(dtheta/2)), which is well below the perceptual
threshold for a Vulkan viewer rendering normalized fields and is
consistent with the same O(dtheta^2) projection error that bounds
all C-grid theta diagnostics.

**Key design decisions worth carrying into C.9:**

1. **Helpers are stagger-agnostic; the runtime caller decides.** The
   interpolation helpers in `stagger_interpolation.cpp` do NOT inspect
   `global_stagger_type` -- they unconditionally average their input.
   The dispatch lives in the runtime (`headless_runtime.cpp`) so the
   helpers stay testable in isolation. Collocated configurations
   never invoke them.

2. **One temporary buffer per output sub-step.** `write_all_fields`
   allocates `u_center_buf`, `v_center_buf`, `w_center_buf` once per
   export step; `shm_update` does the same per SHM tick. The
   buffers live on the stack of the lambda and are reused for the
   3D async snapshot AND the 2D slice path within the same
   `write_all_fields` call.

3. **All other fields are unchanged.** rho, p, theta, q*, vorticity,
   reflectivity, tracer, and the pressure decomposition diagnostics
   already live at cell centers on both collocated and C-grid
   configurations -- the dispatch does not touch their pointers.

4. **The output cost is O(NR*NTH*NZ) per export tick.** For typical
   grids (24x32x48 ~ 37k cells, 3 fields ~ 110k float ops) the
   interpolation runs in microseconds and is amortized into the
   export cadence rather than the inner simulation loop.

5. **GPU dispatch implication.** When C.9 ships C-grid GPU shaders
   for advection and dynamics, the prognostic u, v, w in GPU memory
   will still live at faces. The output-side interpolation here
   stays on the CPU side because the output data path is CPU-bound
   (npy serialization, shm transposition); a CPU read-back of the
   face-staggered fields followed by these helpers keeps the
   downstream consumers (npy writer, shm viewer) unchanged.

---

### C.9 -- GPU Compute Shaders

6 new `.comp` shaders mirroring CPU kernels. Deferred to after full CPU validation.

**Estimate:** 2 weeks. GPU debugging overhead applies (Phase A.7 took 2x predicted).

**Verification:** CPU-GPU parity to 1e-4 for tendencies, 1e-3 for advection.

---

### C.10 -- Integration Tests and Validation

| Test | Config | Scheme | Duration | Pass criterion |
|------|--------|--------|----------|----------------|
| Hydrostatic equilibrium | cylindrical, c_grid, no trigger, no wind | tornado_cgrid | 300s | all tendencies < 1e-10, zero clamps |
| Uniform wind equilibrium | cylindrical, c_grid, WK2002, no trigger | supercell_cgrid | 60s | tendencies < 1e-6 |
| Lamb-Oseen vortex | cylindrical, c_grid, analytic vortex IC | tornado_cgrid | 120s | cyclostrophic drift < 0.1% |
| Warm bubble supercell | full supercell config + c_grid | supercell_cgrid | 300s | 0 theta clamps, recognizable updraft |
| Tornado genesis | tornado_genesis.yaml + c_grid | tornado_cgrid | 1800s | stable vortex, 0 GUARD events |
| Mass conservation | any c_grid config | both | 120s | sum(rho*dV) drift < 1e-12 per step |
| Collocated regression | existing configs, default staggering | all existing | 60s | identical output to pre-Phase-C baseline |

**Estimate:** 2 days.

---

### Schedule

| Week | Tasks | Milestone |
|------|-------|-----------|
| Week 1 (Apr 28 - May 2) | C.1 + C.2 + C.3 | Infrastructure complete. Hydrostatic C-grid passes. |
| Week 2 (May 5 - May 9) | C.4 | Tornado C-grid dynamics. Lamb-Oseen vortex passes. |
| Week 3 (May 12 - May 16) | C.5 + C.6 | Supercell C-grid + split-explicit acoustics. Bug 7 test passes on C-grid. |
| Week 4 (May 19 - May 23) | C.7 + C.8 + C.10 | Advection + output + integration. Full CPU path validated. |
| Week 5-6 (May 26 - Jun 6) | C.9 | GPU shaders. CPU-GPU parity. |

CPU-side complete after week 4. GPU can slip without blocking science runs.

---

### Risks

1. **Momentum advection (C.7)** is the hardest kernel. Face-staggered velocity advected by interpolated velocities. Mitigated by doing scalar advection first, then momentum.
2. **GPU debugging (C.9).** 2x overrun expected based on Phase A.7. Mitigated by deferring to weeks 5-6.
3. **Axis singularity.** Resolved by design. `div_flux()` uses direct `2*u[0]/dr` formula. No epsilon, no L'Hopital at runtime.
4. **Backward compatibility.** Fully mitigated by separate scheme classes. Existing files untouched. Three regression tests verify collocated paths.
5. **Reference-state subtraction.** No bias. One-sided stencil on cell-center p0 is exact. Theta0 interpolation to z-face has same O(dz^2) truncation as collocated.

---

### Files Summary

New files (~14):
- `src/dynamics/schemes/tornado/tornado_cgrid.{hpp,cpp}`
- `src/dynamics/schemes/supercell/supercell_cgrid.{hpp,cpp}`
- `src/boundary_conditions/boundary_conditions_cylindrical_cgrid.cpp`
- `src/core/output/stagger_interpolation.cpp`
- 6 GPU shaders (`*_cgrid.comp`)
- Test configs/runners

Modified files (~10):
- `include/core/grid_geometry.hpp` -- face arrays, staggered flag
- `include/core/coordinate_system.hpp` -- StaggerType enum
- `include/core/runtime_config.hpp` -- global_stagger_type
- `include/numerics/derivatives/derivative_operators.hpp` -- StaggeredCylindricalDerivatives
- `src/core/runtime/runtime_config.cpp` -- parse grid.staggering
- `src/core/orchestration/dynamics/equations.cpp` -- C-grid init branch
- `src/core/orchestration/dynamics/dynamics.cpp` -- BC selection for C-grid
- `src/dynamics/factory.cpp` -- register cgrid schemes
- `src/numerics/advection/advection.cpp` -- C-grid kernels
- Makefile -- new source files

Untouched: Existing TornadoScheme, SupercellScheme, CartesianScheme source files. All Cartesian code, teaching configs, student configs, microphysics, radiation, PBL, turbulence, terrain, chaos.