# Coordinate Backend Plan

**Status:** Phase A complete (2026-04-07). Phase B complete (2026-04-20). Grid prerequisites complete (2026-04-27). Phase C in progress: C.1 + C.2 + C.3 complete (2026-04-28); C.4 complete (2026-04-29).
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

### C.5 -- Supercell C-Grid Slow Tendencies (CPU)

**New files:** `src/dynamics/schemes/supercell/supercell_cgrid.{hpp,cpp}`
**Modified:** `src/dynamics/factory.cpp` (register `"supercell_cgrid"`), Makefile

Adds: azimuthal advection (4-point interpolation), azimuthal pressure gradient at theta-face, cross-derivative terms, reference-state subtraction. Inherits both `DynamicsScheme` and `SplitExplicitDynamics`.

**Estimate:** 2-3 days.

**Verification:** Hydrostatic + WK2002 wind, tendencies < 1e-6 (Bug 7 on C-grid). Warm bubble: no checkerboard in pressure.

---

### C.6 -- Split-Explicit Acoustic Substep on C-Grid

**Modified:** `src/dynamics/schemes/supercell/supercell_cgrid.cpp` -- implement fast pressure and fast momentum tendencies.

Fast pressure: flux-form divergence at cell center. Fast momentum: one-sided pressure gradient at faces. Callback interface in `dynamics.cpp` unchanged.

**Estimate:** 2 days.

**Verification:** Acoustic pulse, 10 substeps, clean circular propagation without checkerboard.

---

### C.7 -- Advection on C-Grid

**Modified:** `src/numerics/advection/advection.cpp` -- C-grid branch with new `_cgrid` static kernels.

Scalar advection: face velocities already available. Momentum advection: staggered positions with interpolated advecting velocities. Highest technical risk (index bookkeeping).

**Estimate:** 2-3 days.

**Verification:** Solid-body rotation < 5% L2 error. Mass conservation to machine precision.

---

### C.8 -- Output Interpolation

**New file:** `src/core/output/stagger_interpolation.cpp`
**Modified:** Output paths in `npy_writer.cpp`, `shm_writer.cpp`, `headless_runtime.cpp`

Face-to-center: `u_center[i] = 0.5*(u[i] + u[i-1])`. Axis: `u_center[0] = 0.5*u[0]`.

**Estimate:** 1 day.

**Verification:** Output from C-grid run shows smooth velocity fields in viewer.

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