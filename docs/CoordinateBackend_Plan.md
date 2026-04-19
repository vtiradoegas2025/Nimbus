# Coordinate Backend Plan

**Status:** Active — Phase A (Cartesian backend) starting 2026-04-06.
**Target completion:** Phase A + B by mid-June 2026; Phase C deferred until tornado-mode work resumes.
**AMS deadline:** January 2027 (~36 calendar weeks of runway).

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

## Phase A — Cartesian backend

**Goal:** A `coordinate: cartesian` runtime config that runs `student.yaml`-like setups end-to-end on CPU and GPU, with an off-center trigger bubble, sheared hodograph, kessler microphysics, and the same physics modules as the cylindrical path.

**Estimate:** 5–8 weeks of focused work. CPU-only (A.1 → A.6) is the first milestone, ~3 weeks. GPU + full integration (A.7 → A.8) is the second milestone, ~2–3 weeks.

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

## Phase B — Refactor for shared code

**Goal:** Consolidate the duplicated code paths created by Phase A's duplicate-then-refactor strategy. Both backends work; the right abstractions are now visible. Every concern should live in exactly one place with coordinate-specific behavior dispatched through clean interfaces.

**Estimate:** 2–3 weeks.

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

## Phase C — Stagger the cylindrical grid (deferred)

**Goal:** Move the cylindrical grid from collocated to Arakawa C-grid (`u_r` at radial faces, `u_θ` at azimuthal faces, `u_z` at vertical faces, scalars at cell centers). This is the standard discretization for compressible cloud models (CM1, WRF, MPAS) and improves accuracy of pressure-gradient handling and mass-flux accounting.

**Why deferred:** Phase A already gives us a working backend for the supercell case (the broken thing). Phase C would improve accuracy of the **tornado** case (which already works for axisymmetric vortex modeling). It's polish, not a fix. Deferring it preserves runway for the supercell-side work that is on the critical path for the AMS presentation.

**When to revisit:** After Phase A + B are done, if there's time before AMS, or whenever tornado-mode work resumes after AMS.

**Estimate (when revisited):** 3–4 weeks.

---

## Verification gates (summary)

| Gate | Phase | Pass criterion |
|---|---|---|
| Existing tests still pass | A.1 | `make test` green |
| Cartesian dynamics tendencies are zero in equilibrium | A.2 | `|du/dt|, |dw/dt| ≤ 1e−3` at every cell with hydrostatic IC + uniform Cartesian wind |
| Cartesian BCs preserve hydrostatic balance | A.3 | 60 s sim with 2 K bubble: `|w|_max < 2 m/s`, mass conserved to 1e−4 |
| Cartesian IC uses literal Cartesian coordinates | A.4 | bubble at config-specified `(x_c, y_c)` |
| Cartesian advection conserves and translates | A.5 | 1D + 2D Gaussian-bump tests |
| Smoke test: Cartesian student.yaml is in equilibrium | A.6 | 60 s no-trigger run, `|w|_max ~ 1e−2`, no clamps |
| GPU parity for Cartesian | A.7 | new `test_vulkan_gpu_parity_cartesian` passes to 1e−4 |
| First real storm | A.8 | 60–120 s sim, recognizable updraft, 0 clamps, conservation drift < 0.05 %/step |
| Field rename: `v_theta` gone | B.1 | `v_theta` appears nowhere except comments |
| Derivative operators shared | B.2 | dynamics schemes share loop structure |
| BCs in factory | B.3 | no inline BCs in `dynamics.cpp` |
| Advection unified | B.4 | single dispatch in `src/advection/` |
| Init unified | B.5 | single init path in `equations.cpp` |
| Time stepping wired | B.6 | no inline Forward Euler in `dynamics.cpp` |
| DynamicsScheme slim | B.7 | split-explicit methods on separate interface |

---

## Risks

1. **GPU debugging is slow.** Every GPU shader port in the project so far has taken 2× the predicted time. Plan A.7 is intentionally ranged 1–2 weeks, not "1 week".
2. **Refactoring two-backend code post-hoc is harder than abstracting up front *if* you guess right.** I'm betting we won't guess right, and that the post-hoc cost is lower than the up-front cost. If Phase B turns out to be > 2 weeks of work, that's still acceptable on the timeline.
3. **The 267 `v_theta` references.** The first instinct is to rename them all up front. Resist. Doing the rename after both backends work is much safer because we'll have tests on both sides confirming correctness — without that, the rename is a 267-site change with no safety net.
4. **A latent BC bug surfaces in the Cartesian path.** The Field3D no-op fix from Bug 5 means several BC patterns that were silently dead are now live. If a Cartesian-specific BC bug shows up, treat it as Bug 8 and add it to Journey.md, do not absorb it silently.

---

## Open decisions (asked of Victor on 2026-04-06)

1. **Field naming.** Keep `u, v_theta, w` for the existing cylindrical scheme and introduce *new* field arrays `u_x, u_y, u_z` for Cartesian (parallel-fields-then-unify), or rename `v_theta → v` everywhere up front (267 sites)? **Recommendation: parallel-fields-then-unify.** Lower short-term blast radius, the rename happens in Phase B with both backends green.
2. **Where to start.** A.1 (config plumbing) is small, mechanical, and immediately verifiable. A.2 (the actual scheme) is the first piece that does real work but is also where mistakes have cost. **Recommendation: A.1 today, A.2 starting tomorrow once A.1 is verified green.**

These are flagged for Victor's call, not to be decided unilaterally.

---

## What "this week" means

April 6 → April 12, focused-but-realistic pace:

| Day | Target |
|---|---|
| Mon Apr 6 | Plan doc (this file) + numerical-output audit |
| Tue Apr 7 | A.1 (config plumbing) + start A.2 |
| Wed Apr 8 | Finish A.2 (CartesianDynamicsScheme CPU) |
| Thu Apr 9 | A.3 (BCs) + A.4 (IC) |
| Fri Apr 10 | A.5 (CartesianAdvectionScheme CPU) |
| Sat–Sun | A.6 (wiring + smoke test). End-of-week milestone: **Cartesian CPU path is in equilibrium for `student.yaml`-no-trigger.** |

Following week: A.7 (GPU shaders) and A.8 (real storm test).

If anything slips, A.5 is the most likely day to absorb it — TVD cylindrical → TVD Cartesian is the largest mechanical edit.
