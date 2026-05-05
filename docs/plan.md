# TornadoModel — Road to AMS January 2027

**Deadline:** January 2027 AMS Conference (~9 months)
**Goal:** Complete, efficient simulation on personal hardware with reproducible results

**Architecture as of 2026-04-30:** Dual coordinate backend. Cartesian for storm-scale supercell simulations (matches CM1 discretization). Cylindrical Arakawa C-grid for tornado-vortex modeling (CM1/WRF/MPAS staggering applied to a vortex-native grid). Shared physics modules (microphysics, radiation, PBL, turbulence, terrain) are coordinate-blind and run on either backend. See [docs/CoordinateBackend_Plan.md](CoordinateBackend_Plan.md) for the full Phase A / B / C history.

---

## ~~Phase 0: Foundation Cleanup (Weeks 1–3)~~ COMPLETE

---

## ~~Phase 1: Vulkan Compute — Make It Real (Weeks 3–10)~~ COMPLETE

---

## Phase CB: Coordinate Backend Architecture (Apr 2026)
*The work that made the supercell case actually run, and brought the cylindrical grid up to CM1/WRF/MPAS standard.*

This phase was not in the original plan. It emerged from Bug 7: a uniform Cartesian wind on the collocated cylindrical grid produced a false radial gradient at i = 1 (driven by the antisymmetric ghost-cell hack `u[0] = -u[1]`) that drove false divergence, false `dp/dt`, broke hydrostatic balance, and made the supercell hodograph case 17× more unstable than the zero-wind case. The cylindrical grid was the wrong tool for non-axisymmetric base states. The fix was to add a Cartesian backend (right tool for supercells) and stagger the cylindrical grid (right tool for tornadoes). Detailed plan: [docs/CoordinateBackend_Plan.md](CoordinateBackend_Plan.md).

| Phase | Status | Date | Outcome |
|-------|--------|------|---------|
| **A — Cartesian backend** | COMPLETE | 2026-04-07 | `coordinate_system: cartesian` runs `student.yaml`-class configs end-to-end on CPU + GPU. 8 sub-tasks (config plumbing, dynamics, BCs, ICs, advection, top-level wiring, GPU shaders, integration). Bug 7 resolved. |
| **B — Refactor for shared code** | COMPLETE | 2026-04-20 | 7 sub-tasks. `v_theta`→`v` rename (267 sites). `DerivativeOperators` base class with Cartesian/Cylindrical specializations. `BoundaryConditionScheme` factory. Unified Strang split. Shared thermodynamic init. Time stepping delegates through scheme layer. `SplitExplicitDynamics` mixin separated from `DynamicsScheme` base. Every concern has exactly one dispatch point. |
| **C — Cylindrical Arakawa C-grid** | C.1–C.8 COMPLETE | 2026-04-27 → 2026-04-30 | Velocity components on r/θ/z faces; scalars at cell centers. Axis singularity centralized in `StaggeredCylindricalDerivatives::div_flux()` (control-volume `2·u[0]/dr`, no antisymmetric hack). New schemes `TornadoCGridScheme` and `SupercellCGridScheme` registered alongside existing collocated schemes. Klemp-Wilhelmson split-explicit on staggered grid verified (300 s hydrostatic preservation, acoustic pulse at sound speed, no 2dx checkerboard). TVD-MUSCL flux-form scalar advection (solid-body rotation < 5% L2 per revolution; mass conservation pairwise to float roundoff). Face-to-center output interpolation for downstream consumers. |
| **C.9 — GPU compute shaders** | PENDING | target 2026-06-06 | 6 stagger-aware shaders (`*_cgrid.comp`) mirroring CPU C-grid kernels. Until they ship, the existing GPU acoustic and advection dispatches are guarded off when `global_stagger_type == StaggerType::CGrid` so production runs silently fall back to the CPU path on C-grid configs. |
| **C.10 — Integration + validation** | PENDING | target 2026-05-23 | 7 integration tests (hydrostatic equilibrium, uniform-wind equilibrium, Lamb-Oseen vortex, warm bubble supercell, tornado genesis, mass conservation, collocated regression). |

**Aggregate test count** (after C.8): 91,154+ dynamics assertions across 43 test cases plus 6,892 stagger-interpolation assertions. Full suite green. The C-grid acceptance gates from `docs/CoordinateBackend_Plan.md` for C.1 → C.8 are all closed; only C.9 (GPU) and C.10 (validation runs) remain in Phase C.

**Why this matters for everything else:**
- Bug 7 is no longer an open question. The supercell hodograph case now runs.
- The cylindrical backend is no longer "the only option, with caveats." It is the right tool for tornado-vortex modeling, with the same gold-standard discretization (C-grid + Klemp-Wilhelmson) that CM1 / WRF / MPAS use.
- Phase 4 (Physics) and the Scientific Enhancements section (S1, S5 in particular) can now be planned against the right grid for the right problem.

---

## Phase 2: Performance, Efficiency & Accessibility (Weeks 6–12)
*Make it fast on high-end hardware. Make it possible on a student's laptop.*

**Design principle:** The same binary should run a meaningful simulation on a base MacBook Air with 8GB RAM *and* saturate an M3/M5(Future) Ultra with 512GB unified memory or a 64-core Threadripper. The user configures the knobs; the software adapts to the hardware.

**#1 Priority — Target Machine Floor:**
- MacBook Air M1 (8GB RAM, 256GB SSD) — must run a meaningful 64×64×32 simulation
- Ubuntu laptop with integrated Intel graphics — must run CPU-only without Vulkan
- Windows WSL2 — must build and run (CPU-only)
- Raspberry Pi 5 (8GB) — stretch goal, would be remarkable for outreach
- **Any post-2018 machine with a C++17 compiler must build and run with zero external dependencies.**

**GPU strategy:** Metal support coming once Vulkan is 100% utilized.
- MoltenVK translates Vulkan → Metal on macOS automatically (~95% native perf)
- Vulkan covers Windows + Linux natively (NVIDIA/AMD/Intel)
- CPU-only fallback covers everything else
- Metal in certain scenarios can see performance gains once all of Vulkan is realized

### 2.1 CPU hot-path optimization — ~~COMPLETE~~

| Sub-item | Status | Evidence |
|----------|--------|----------|
| **SIMD library** | **COMPLETE** | 11 SIMD primitives (add, subtract, multiply, fma, scale, scale_add, clamp, sanitize_nonfinite, reduce_sum, reduce_min_max, diffuse_1d) with NEON/AVX/SSE/scalar paths. See `include/util/simd_utils.hpp`. |
| **SIMD wiring** | **COMPLETE** | 7 call sites wired: diffusion (2), chaos perturbation (2), sanitization (3). See `.claude/plans/jiggly-dreaming-globe.md` for full map. |
| **Field3D direct access** | **COMPLETE** | `row_ptr(i,j)` bypasses proxy object chain for SIMD-friendly k-column access. |
| **FieldPool** | **COMPLETE** | RAII `scoped_acquire` in diffusion (explicit+implicit), microphysics (kessler), turbulence (TKE). |
| **Cache blocking** | **COMPLETE** | TILE_J=32 in advection + explicit diffusion (scalar + momentum). |
| **OpenMP** | **COMPLETE** | 79+ pragmas across 19+ files. TKE `collapse(3)` gap fixed. Thompson 4-loop gap fixed. |

**Remaining SIMD opportunities (deferred to Phase 2R):**
- Double-precision SIMD overloads (unlocks TVD advection, dynamics intermediates)
- Masked SIMD for microphysics conditional loops
- `isfinite` pre-validation pass (removes branches from ~15 loops)

### 2.2 GPU shader wiring — ~~COMPLETE~~
All 7 Vulkan compute pipelines dispatched: supercell, tornado, kessler×2, diffusion, advection×3. Kernel-level 4.4× on TVD vertical advection (validated March 2026).

### 2.3 GPU-resident field management — DEPRIORITIZED with measurement
Buffer pool + zero-copy path exist for unified memory. Cross-timestep residency *not* implemented and *not* worth implementing on Apple Silicon: zero-copy transfer overhead is ≈ 1–2 µs/dispatch (< 0.1% of step time). `GPUFieldSnapshot` interface exists but has zero call sites. Revisit only for discrete-GPU support.

### 2.4 Async I/O — ~~COMPLETE~~
AsyncOutputWriter with background thread, condition variables, backpressure. Enabled in `production.yaml`.

### 2.5 ZFP compression — ~~COMPLETE~~
ZFP integrated with 3 modes (accuracy, precision, rate). Delta encoding + keyframe intervals + reader + roundtrip tests all shipped (commit `2cfb4f1`, 2026-04-21). Predictive delta and float16 prefilter active in production / research configs. Per-field tolerance map drives per-field compression. Tier 2b *measurement* validation moved to Phase 3 §3.1.

### 2.6 Configurable output — ~~COMPLETE~~
5 field presets, 4 output formats (NPY 2D slices, NPY 3D, CSV, ZFP), disk budget estimation, YAML parsing, tiered cadence (`WriteCadenceTier` fast/medium/slow).

### 2.7 Shared memory transport — ~~COMPLETE~~
POSIX SHM transport, ShmWriter, ShmDataset, Vulkan viewer integration. 1058 assertions passing.

### 2.8 Hardware-aware scaling — ~~COMPLETE~~
CPU/RAM/cache/GPU/SIMD detection and logging. Memory-safe grid warnings. `apply_hardware_defaults()` runs at the 80% RAM threshold *before* field allocation so student-laptop configs no longer OOM between detection and allocation. 6 tests, 31 assertions.

### 2.9 Accessibility — ~~COMPLETE~~
Zero-dep CPU build, smoke test, 3 teaching configs, student.yaml, CSV/NPY/ZFP export.

### Phase 2 Acceptance Criteria

- [x] `make` produces a working binary with zero external dependencies on any C++17 system
- [x] Student config (64×64×32) runs in <1GB RAM
- [x] All SIMD primitives have multi-ISA implementations (NEON/AVX/SSE/scalar)
- [x] SIMD is wired into real compute paths (not dormant)
- [x] All parallelizable physics loops have OpenMP pragmas
- [x] FieldPool prevents per-timestep allocation spikes
- [x] Cache blocking on hottest 3D loops
- [x] All 7 Vulkan compute shaders dispatched
- [x] Async I/O eliminates export blocking
- [x] ZFP compression available (3 modes)
- [x] Configurable output with presets
- [x] Shared memory in-situ rendering works
- [x] Hardware detection + memory-safe grid warnings
- [x] ZFP delta encoding — Writer + reader + roundtrip tests all shipped (commit 2cfb4f1, 2026-04-21). Enabled via `zfp_keyframe_interval` in `production.yaml` and `research.yaml`.
- [~] GPU cross-timestep field residency — **Deprioritized with measurement.** Zero-copy on Apple Silicon (unified memory) means transfer overhead is ~1-2 µs/dispatch (< 0.1% of step time). `GPUFieldSnapshot` interface exists but unwired. Not worth the architectural cost on unified memory. Revisit only for discrete GPU support.
- [x] Auto-config applies hardware-based defaults — `apply_hardware_defaults()` implemented and wired into `headless_runtime.cpp`. 6 tests (31 assertions). Auto-scales grid to fit 80% RAM, logs what changed, calls `resize_fields()` and recomputes disk budget. The 80% threshold check now runs *before* field allocation (Cycles 0-2 infrastructure, 2026-04-21) so student-laptop configs can no longer OOM between detection and allocation.

**Overall: COMPLETE.** The 2026-04-21 infrastructure landing (commit `2cfb4f1`) closed the remaining gaps: ZFP delta reader + roundtrip tests shipped, hardware auto-config moved to pre-allocation, logging ratchet test in place (raw print count ≤ 274), TVD v2 float-precision kernel with 8-column batching, OpenMP on RK3/RK4 + implicit diffusion + radiation columns, `__restrict__` on all 51 simd_utils signatures. The remaining work is *measurement* validation (Phase 3 §3.1) and *coordinate backend* (Phase CB), neither of which belongs in Phase 2 itself.

---

## Phase 3: Validation & Measurement (Weeks 12–14)
*Phase 2 shipped the infrastructure. Phase 3 measures whether it does what it claims.*

The earlier draft of this plan carried a heavyweight "Phase 2R+3 audit-test cycles" framework (Cycles 0–5, infrastructure gates, per-module audit loops). The actual workflow that has driven the project is much simpler: phase-based execution with sub-tasks (Phase A → B → C, each with verification gates). The cycle framework was retired on 2026-04-30 because the work it described had already happened — Cycles 0–2 infrastructure landed in commit `2cfb4f1` (2026-04-21), Cycles 3–5 audit findings were absorbed into Phase B's refactor and Phase C's verification gates, and the only items that remain are *measurement* validations of the spatial/temporal pipeline.

### Test Baseline (2026-04-30)

Catch2 integrated. **41,000+ assertions across 25+ test suites — all passing.**

| Suite | Tests | Assertions | Notes |
|-------|-------|------------|-------|
| Core (field, output, hardware, npy, writer, shm, stagger interp) | 65+ | 8,289+ | + `test_core_stagger_interpolation` (C.8): 6,892 assertions |
| Dynamics (Cartesian + cylindrical_cgrid + tornado_cgrid + supercell_cgrid) | 43 | 91,154+ | Phase A/B/C dynamics gates |
| Numerics (advection collocated, advection cartesian, advection cgrid, diffusion, timestepping, staggered derivatives) | 20+ | 18,830+ | + `test_numerics_advection_cylindrical_cgrid` (C.7) |
| Physics (microphysics, radiation, terrain) | 27 | 8,381 | unchanged |
| Data (soundings) | 7 | 41 | unchanged |
| Vulkan (backend, gpu_parity) | 11 | 18,844 | awaiting C.9 cgrid shader parity tests |
| Integration (performance) | 12 | 23 | unchanged |
| SHM E2E | 3 | 1,058 | unchanged |

### 3.1 Tier 2b Compression Measurement (Phase 4 gate)

The output pipeline already implements:

- **ZFP delta encoding** (writer + reader + roundtrip tests; commit `2cfb4f1`)
- **Per-field ZFP tolerance map** (`config.zfp_field_tolerances` in `output_config.hpp`)
- **Tiered write cadence** (`WriteCadenceTier` enum: fast / medium / slow)
- **Predictive delta + sparse thresholding + float16 prefilter** (production / research configs)

What is *not* yet validated: that this pipeline actually meets the storage targets in §3.3 on a production-grade run. Until that measurement lands, every Phase 4 physics addition risks blowing the storage budget for long-duration runs. Earlier projections (effective compression ratios, GB-per-2h-run, wall-clock breakdowns) were estimates from before the Phase B refactor and the C-grid path landed; they were never verified end-to-end and should not be cited as facts. Phase 3's job is to replace those estimates with measurements.

**Compression tiers — implementation status (not measurements):**

| Tier | What | Status |
|------|------|--------|
| **0** | Default ZFP, uniform tolerance, uniform cadence | shipped |
| **1** | Delta encoding enabled in shipped configs | shipped (commit `2cfb4f1`, 2026-04-21) |
| **2** | Per-field ZFP tolerance map (field contract bounds drive tolerance) | shipped |
| **2b** | + Tiered write cadence (`WriteCadenceTier` fast / medium / slow) | shipped — **measurement pending (Phase 4 gate)** |
| **3** | Sparse + predictive delta | predictive shipped; sparse pending |

**Measurement steps:**
1. Run a 120 s production simulation with the shipped configs (`production.yaml`, Tier 2b active).
2. Measure GB per export, per tier, per field type. Record per-field compression ratios.
3. Extrapolate to a 2 h run and compare against the §3.3 acceptance criteria.
4. If §3.3 cleared: Phase 4 unblocked.
5. If not: tighten per-field tolerances or cadence and re-measure before adding new physics.

### 3.2 Wall-clock baseline

Phase B's refactor and the C-grid path both touched the dynamics hot loop and the advection batching, so a fresh wall-clock baseline is part of Phase 3 (no prior numbers carry forward as facts):

- Profile the production grid with Instruments on macOS. Identify the dominant hotspot in the current code; in particular, confirm whether vertical advection (now batched in TVD v2) is still the leading contributor or has been displaced.
- Document ms/step for student / research / production tiers, on both Cartesian and cylindrical-cgrid backends.
- Decide whether GPU vertical-advection dispatch is worth wiring for the cgrid path now or after Phase C.9 ships.

### 3.3 Phase 3 Acceptance Criteria

- [ ] Tier 2b measurement: 2 h production run ≤ 30 GB **(Phase 4 gate)**
- [ ] 8 h run extrapolation: ≤ 100 GB at Tier 2b with 10 s core cadence
- [ ] Student config (64×64×32): ≤ 60 MB output, runs in < 6 minutes wall-clock
- [ ] ms/step baseline documented for student / research / production tiers, on both backends
- [ ] Measured per-field compression ratios captured in a freshly-written benchmark doc

### 3.4 Carry-forward items (low priority, opportunistic)

- **Logging ratchet:** guardrail test asserts raw print count ≤ 274. Migrate to `tmv::log_*` opportunistically when touching a file; no scheduled cycle.
- **GPU field residency:** deprioritized with measurement. Zero-copy on unified memory ≈ 1–2 µs/dispatch (< 0.1% of step time). `GPUFieldSnapshot` interface exists but has zero call sites. Revisit only for discrete-GPU support.
- **Double-precision SIMD overloads:** TVD v2 float-batched path is the new baseline; double overloads gated on measurement justification.
- **CI matrix expansion** (macOS runner, clang + gcc, Vulkan smoke test) and **`.clang-format` / `.clang-tidy`** configs: Phase 5 polish.
- **Microphysics init duplication, `RadiationColumnStateView` field validation, radiation silent lapse-rate fallback, turbulence factory inline lambda:** small Phase 4 prerequisites. Bundle them into the radiation/microphysics work in 4.1 and 4.4 rather than scheduling a standalone audit.

---

## Phase 4: Physics Completeness (Weeks 14–20)
*Fill the science gaps for a credible AMS presentation. Begins after Phase 3 §3.1 validates the output pipeline (Tier 2b ≤ 30 GB for 2 h) and §3.2 documents wall-clock for the current dynamics + advection paths. Adding more physics without that confirmation risks blowing the storage / compute budget for long-duration runs.*

**Prerequisites (from Phase 3):**
- Tier 2b compression measurement validated on a 2 h production run
- Wall-clock baseline established per backend per grid tier
- Every new physics field added in Phase 4 must be classified into a compression tier and cadence group before it ships

### 4.1 Radiation: Beyond grey-body
`simple_grey` is too simplistic for realistic storm simulations. RRTMG is recognized in the factory but throws "not implemented."

**Options (pick one based on time budget):**
- **Option A (lower effort):** Implement a simplified broadband radiation scheme (e.g., Dudhia shortwave + RRTM-like longwave with lookup tables). Good enough for diurnal cycle effects.
- **Option B (higher fidelity):** Full RRTMG port. This is substantial work but gets you closest to CM1 parity.

**Action:** Also fix the silent fallback in `radiation.cpp` that substitutes a default lapse rate without warning.

### 4.2 Remaining field contract fields
12 CM1-style fields remain `NotImplemented`:
- 3D diagnostics: streamlines, q_vectors, turbulent_diffusion_term
- Cross-section diagnostics
- Trajectory diagnostics

**Action:** Prioritize by what you'll show at AMS. Streamlines and q_vectors are high visual impact. Trajectory diagnostics can wait.

### 4.3 Terrain and chaos validation
Both modules are runtime-integrated but lack case-based scientific validation.

**Action:** Run against 2–3 reference cases from literature and document agreement/disagreement. This is research work, not code work.

### 4.4 Fix radiation column state validation
`RadiationColumnStateView` allows nullptr for fields without documenting which are required. `compute_column()` doesn't validate. Add guards.

---

## Phase 5: Presentation Readiness (Weeks 20–26)
*Polish for AMS.*

### 5.1 Refactor monolithic files
| File | Lines | Action |
|------|-------|--------|
| `sharpy_sounding.cpp` | ~2900 | Extract NetCDF parsing into a dedicated reader class |
| `runtime_config.cpp` | ~2500 | Split into config sections (grid, physics, output, validation) |
| `headless_runtime.cpp` | ~2936 | Extract per-module stepping into separate functions |
| `tornado_sim.cpp` | ~730 | Extract 220-line sounding init into its own function |

### 5.2 Documentation refresh
- Resolve SupercellModel vs TornadoModel naming
- Update STATUS.md with current compute backend state and benchmark numbers
- Add missing READMEs where needed

### 5.3 Demo workflow
Create a turnkey script that:
1. Builds the project
2. Runs a compelling simulation case
3. Launches the Vulkan viewer on the output
4. Produces publication-quality output frames

### 5.4 First-run experience
**The acid test:** A meteorology student who has never used the command line should be able to:
1. Download the repo (GitHub zip or `git clone`)
2. Run `make` (or a build script that detects their platform)
3. Run the simulation with a default config
4. See results

No CUDA toolkit installation. No Vulkan SDK setup. No OpenMP library hunting. The build system detects what's available and adapts silently. If nothing is available, the CPU-only path works.

---

## Phase Summary

| Phase | Weeks | Focus | Impact |
|-------|-------|-------|--------|
| ~~**0: Foundation**~~ | ~~1–3~~ | ~~Gitignore, dedup, globals, chaos cleanup~~ | ~~COMPLETE~~ |
| ~~**1: Vulkan Compute**~~ | ~~3–10~~ | ~~Real GPU dispatch pipeline~~ | ~~COMPLETE~~ |
| ~~**2: Performance, Efficiency & Accessibility**~~ | ~~6–12~~ | ~~SIMD library, OpenMP, FieldPool, cache blocking, ZFP delta + reader, async I/O, configurable output, SHM transport, hardware auto-config, TVD v2 batched kernel, logging ratchet~~ | ~~COMPLETE — see Phase 3 for the measurement validation that confirms it~~ |
| ~~**CB-A: Cartesian backend**~~ | — (2026-04-07) | ~~Second coordinate backend; resolves Bug 7 for non-axisymmetric base states~~ | ~~COMPLETE — supercell case runs~~ |
| ~~**CB-B: Refactor**~~ | — (2026-04-20) | ~~Consolidate dual-backend code; one dispatch point per concern~~ | ~~COMPLETE — `v_theta`→`v`, derivative operators, BC factory, time stepping unified~~ |
| **CB-C: Cylindrical Arakawa C-grid** | Apr 28 – Jun 6 | Bring cylindrical grid to CM1/WRF/MPAS standard staggering. C.1–C.8 done; C.9 GPU shaders + C.10 integration tests remaining | Tornado-vortex modeling on the right discrete grid |
| **3: Validation & Measurement** | 12–14 | Tier 2b compression measurement on a real 2 h production run. Wall-clock baseline per backend × grid tier. Replaces the retired Phase 2R+3 audit-test cycle framework. | Output pipeline proven before more physics lands |
| **4: Physics** | 14–20 | Radiation, field contract, validation. **Blocked until Phase 3 §3.1 measurement clears AND Phase CB-C complete.** Every new field gets a compression tier + cadence group AND a coordinate backend classification. | Scientific credibility for AMS |
| **5: Polish** | 20–26 | Refactoring, docs, demo script, first-run experience, CI matrix, clang-format/tidy | Presentation readiness |

---


## Scientific Enhancements — What This Architecture Can Uniquely Do

*Maintained section: every architectural milestone should propagate into this list. If a Phase X completion adds a new uniquely-enabled capability or strengthens an existing one, add or update an entry here. The point of this section is to articulate — for AMS reviewers, for collaborators, and for future Victor — what the model can do that nothing else can.*

**The dual-backend architecture (as of 2026-04-30):** Cartesian backend (Phase A) for storm-scale supercell simulations on the same discretization CM1 uses. Cylindrical Arakawa C-grid backend (Phase C, in progress) for tornado-vortex modeling on a vortex-native grid with CM1/WRF/MPAS-standard staggering and Klemp-Wilhelmson split-explicit time stepping. Shared physics modules (microphysics, radiation, PBL, turbulence, terrain) are coordinate-blind and run on either backend. Each problem class gets the right discretization for the right physics, without duplicating the physics.

**What that combination uniquely enables on personal hardware:**

- *For storm-scale studies:* a working CM1-class Cartesian model (no false body forces from antisymmetric axis hacks; no centrifugal artifacts from non-axisymmetric base states) running unattended on a MacBook Air through a Threadripper.
- *For tornado-vortex studies:* the same staggered, split-explicit, conservative compressible NWP discretization that CM1/WRF/MPAS use, applied to a cylindrical grid where the singular axis is treated rigorously by control-volume divergence (`2·u[0]/dr` at i = 0, no antisymmetric ghost) instead of avoided. Discrete cyclostrophic balance is a fixed point of the dynamics, not a goal you drift away from.
- *For coupled studies:* the same trigger bubble, sounding, microphysics scheme, radiation scheme, PBL parameterization, and turbulence closure can be flipped between backends to isolate which features come from the physics and which come from the coordinate system.

These are scientifically motivated extensions that play to the model's strengths.

### S1. Vortex-Centric Analysis (Natural Fit for Cylindrical C-Grid)

CM1 uses Cartesian coordinates and requires post-hoc coordinate transforms for vortex analysis. Your native (r, θ, z) **Arakawa C-grid** makes the following **first-class operations** instead of post-processing hacks:

- **Azimuthal decomposition of vortex structure** — Decompose any field into wavenumber-0 (axisymmetric mean) and wavenumber-1,2,...,n (asymmetric perturbations) directly on the native grid. This is trivial in cylindrical coords and expensive/lossy in Cartesian.
- **Angular momentum budgets** — Track absolute angular momentum flux through cylindrical surfaces natively. No interpolation artifacts. Mass conservation on the C-grid is **pairwise to machine precision** (the same face flux is debited from one cell and credited to its neighbor in floating point), so circulation budgets close exactly within the discretization.
- **Vortex Rossby wave dynamics** — Diagnose VRW propagation, reflection, and wave-mean flow interaction using azimuthal wavenumber decomposition. Literature: Montgomery & Kallenbach (1997), Nolan & Montgomery (2002).
- **Secondary circulation diagnostics** — Compute transverse (r,z) streamfunction for the axisymmetric overturning circulation directly.
- **Discrete cyclostrophic balance is a fixed point.** The Phase C.4 verification: a Lamb-Oseen vortex with `v(r) = (Γ/(2πr))(1 − exp(−r²/r_c²))` and analytically-balanced `p(r,z)` produces tendencies bit-exactly zero (`dv/dt`, `drho/dt`, `dp/dt` = 0) on the C-grid; over 60 simulated seconds of Forward-Euler-on-momentum the v profile drifts < 0.1% of v_max. The collocated cylindrical grid could not pass this test because the antisymmetric axis ghost `u[0] = −u[1]` produces a structural false-divergence floor that corrupts cyclostrophic balance over any meaningful integration time. **The C-grid backend is what finally makes cylindrical vortex modeling on personal hardware numerically rigorous.**

**What to build:** An `azimuthal_decomposition` diagnostic module that performs real-time FFT decomposition in θ and exports wavenumber-resolved fields. This is a unique selling point vs. CM1.

### S2. Ensemble Sensitivity Analysis on Personal Hardware

Your chaos module already supports IC perturbations, BL perturbations, and full stochastic physics. The next step is making this scientifically useful:

- **Ensemble Sensitivity Analysis (ESA)** — Correlate initial condition perturbations with outcome variables (e.g., tornado intensity, track, timing) across small ensembles (10-30 members, feasible on personal hardware). Literature: Torn & Hakim (2008), Bednarczyk & Ancell (2011).
- **Adjoint-free sensitivity** — Your stochastic perturbation framework can approximate adjoint sensitivity without needing an adjoint model. Run N perturbation members, regress outcomes against perturbation patterns.
- **Stochastic physics calibration** — Use the factory pattern to systematically compare deterministic vs. stochastic PBL/turbulence and quantify spread in storm-scale features.

**What to build:** Ensemble output collection (parallel member runs → shared analysis), statistical correlation utilities, perturbation-response diagnostics.

### S3. GPU-Accelerated Parameter Space Exploration

Once Vulkan compute works (Phase 1), the GPU becomes useful for more than single-run acceleration:

- **Rapid parameter sweeps** — Run many short simulations varying microphysics parameters (e.g., N₀ rain, graupel density, ice nucleation threshold) to map sensitivity surfaces. GPU makes each run fast enough to sweep in hours instead of days.
- **Convective mode classification** — Systematically vary CAPE, shear, hodograph curvature across a parameter grid. Classify resulting storm mode (supercell, multicell, squall line, tornado). Literature: Weisman & Klemp (1982, 1984), Thompson et al. (2007).
- **Microphysics intercomparison** — Your 4 microphysics schemes can be systematically compared across identical environmental profiles. Quantify which scheme aspects drive the largest storm-structure differences.

**What to build:** A batch runner that takes a parameter sweep config, distributes across GPU/CPU, collects summary statistics.

### S4. Forward Radar Operator Enhancement

Your radar module already computes Z, V_r, Z_DR. Extensions that are scientifically impactful and architecturally clean:

- **Specific differential phase (K_DP)** — Already listed in README as a capability but needs validation. K_DP is particularly useful for heavy rain rate estimation.
- **Dual-wavelength ratio** — Simulate S-band and C-band reflectivity differences for hail detection algorithms.
- **Beam propagation effects** — Add refraction, attenuation, and beam broadening for more realistic radar simulation. The cylindrical grid actually simplifies radial beam propagation when the radar is near the domain center.
- **Synthetic PAR scans** — Phased-array radar scanning strategies (rapid-update volumetric) using your 3D fields. Relevant for NSSL's PAR research. Literature: Zrnić et al. (2007), Heinselman & Torres (2011).

**What to build:** Beam propagation module with configurable radar position, refraction model, and multi-frequency support.

### S5. Tornadogenesis Process Diagnostics

Your dynamics module computes vorticity components and stretching/tilting/baroclinic terms. Extensions for tornadogenesis research, *now with the conservation properties to do them quantitatively*:

- **Circulation budget analysis** — Compute material circulation tendencies around circuits in (r,z) plane. Track how circulation concentrates from mesocyclone to tornado scale. The C-grid pairwise mass-conservation property means that circulation budgets close at the discretization floor, not at the "did we lose mass to the axis hack?" floor. Literature: Rotunno & Klemp (1985), Markowski & Richardson (2014).
- **Dynamic pressure decomposition** — Separate perturbation pressure into linear and nonlinear dynamic components plus buoyancy pressure. Diagnose which drives the low-level updraft. The Klemp-Wilhelmson split-explicit pressure equation on the C-grid (`dp/dt = −γp · div_flux − u·grad p` with one-sided face gradients) gives the same partition CM1 uses, with no checkerboard pressure modes contaminating the diagnosis. Literature: Klemp & Rotunno (1983).
- **Downdraft provenance tracking** — Tag air parcels in the RFD/FFD and track their thermodynamic history (origin height, θ_e deficit, angular momentum). Directly addresses the "RFD surge" hypothesis. Literature: Markowski et al. (2002, 2003).
- **Critical angle diagnostics** — Compute the angle between storm-relative inflow and low-level shear vector in real time. Literature: Esterheld & Giuliano (2008).

**What to build:** A `tornadogenesis_diagnostics` module that computes circulation tendencies, pressure partitioning, and parcel tracking on the native grid.

### S6. Surface Layer and Near-Ground Resolution

Personal hardware runs mean you can afford finer vertical resolution near the surface than operational models:

- **Logarithmic wind profile near surface** — Refine the surface layer representation with Monin-Obukhov similarity + roughness length variation. Your slab PBL already has surface fluxes; enhance with stability-corrected profiles.
- **Near-surface vorticity dynamics** — The tornado near-surface wind field is poorly resolved in most models. Finer vertical spacing in the lowest 500m would be a contribution. Literature: Lewellen et al. (1997), Nolan et al. (2017).
- **Frictional vorticity generation** — Surface drag generates horizontal vorticity that can be tilted into vertical vorticity by the updraft. This is a key tornadogenesis mechanism. Your cylindrical grid resolves this naturally.

**What to build:** Stretched vertical grid option (fine spacing near surface, coarser aloft) and enhanced surface layer physics.

### S7. High-Fidelity Scientific Visualization (Vulkan Viewer)

The Vulkan viewer has a solid single-pass volume ray-marcher with Beer-Lambert transmittance, Henyey-Greenstein phase function, and 8-field compositing. The goal is to approach CM1/VisIt/VAPOR-quality rendering — reflectivity floors, vorticity streamlines, isosurfaces, terrain meshes, self-shadowing clouds — running on personal hardware (M3 Pro / MoltenVK).

Each sub-phase is independently shippable. Together they prove that research-grade atmospheric visualization doesn't require a supercomputer.

#### S7.1 Reflectivity Floor + Scientific Colormaps

A dBZ reflectivity heatmap on the ground plane beneath the volume, plus proper colormap LUTs (viridis, coolwarm, NWS reflectivity scale) replacing the hardcoded 8-color palette.

**Approach:** Add directly into `volume.frag` as a ground-plane ray intersection. When the ray exits the volume box at the bottom face, sample a 2D reflectivity texture and map through a 1D colormap LUT. No mesh rendering needed.

**New files:**
| File | Purpose |
|------|---------|
| `vulkan/include/rendering/colormap_lut.hpp` | Generate 1D RGBA8 LUT textures (viridis, coolwarm, inferno, NWS reflectivity) |
| `vulkan/src/rendering/colormap_lut.cpp` | LUT generation + Vulkan texture upload |

**Modified files:**
| File | Change |
|------|--------|
| `vulkan/shaders/rendering/volume.frag` | Add `sampler2D u_floor` + `sampler1D u_colormap` bindings. After ray-march, if ray exits bottom face, sample floor, map through LUT, blend as ground color |
| `vulkan/src/rendering/volume_backend.cpp` | Load reflectivity 2D slice, upload as R32_SFLOAT 2D texture. Create LUT texture. Bind both to new descriptor bindings |
| `vulkan/include/rendering/volume_backend.hpp` | Members for floor texture, LUT texture, new descriptor bindings |

**New Vulkan resources:** 2D reflectivity texture (R32_SFLOAT) + 1D colormap LUT (RGBA8, 256 texels) + 2 new descriptor bindings.

**Verification:** Floor aligns with volume bottom. Colormap matches published NWS dBZ reference values. No performance regression (~1 extra texture lookup per pixel).

---

#### S7.2 CompositeBackend + Depth Buffer + Terrain Surface Mesh

Introduce mesh rendering infrastructure via a `RenderLayer` abstraction. Render terrain (bell/schar topography) as a colored surface beneath the volume with proper depth compositing. This establishes the pattern for all subsequent mesh features (isosurfaces, streamlines).

**Architecture — RenderLayer abstraction:**
```cpp
class RenderLayer {
    virtual bool needs_own_render_pass() const = 0;
    virtual bool record_commands(VkCommandBuffer, VkExtent2D, std::string&) = 0;
    virtual bool initialize(VulkanContext&, VkRenderPass, VkExtent2D, std::string&) = 0;
    virtual bool on_swapchain_recreated(VkRenderPass, VkExtent2D, std::string&) = 0;
    virtual void shutdown(VkDevice) = 0;
};
```

A `CompositeBackend` implements `RenderBackend` and sequences `RenderLayer` objects. The existing `VolumeBackend` is refactored into a `VolumeLayer`. The render pass gains a D32_SFLOAT depth attachment.

**Terrain mesh generation:** A compute shader converts the cylindrical height field `h(r,θ)` to a Cartesian triangle mesh (`x = r*cos(θ), z = r*sin(θ), y = h`). No geometry shaders (MoltenVK limitation) — compute shader writes vertices + normals to an SSBO, vertex shader reads from it.

**Render order:** Terrain (writes depth + color) → Volume (ray-march, writes depth at first opaque hit, composites over terrain) → Overlays (Phase S7.5, depth test disabled).

**Compute-graphics sync:** Pipeline barrier between mesh gen dispatch and vertex input read (`srcStage=COMPUTE, dstStage=VERTEX_INPUT`).

**New files:**
| File | Purpose |
|------|---------|
| `vulkan/include/rendering/render_layer.hpp` | Abstract RenderLayer interface |
| `vulkan/include/rendering/composite_backend.hpp` | CompositeBackend: sequences layers within one command buffer |
| `vulkan/src/rendering/composite_backend.cpp` | Layer orchestration, render pass begin/end, barrier insertion |
| `vulkan/include/rendering/volume_layer.hpp` | Refactored VolumeBackend internals as a RenderLayer |
| `vulkan/src/rendering/volume_layer.cpp` | Volume layer implementation |
| `vulkan/include/rendering/terrain_layer.hpp` | Terrain mesh layer |
| `vulkan/src/rendering/terrain_layer.cpp` | Terrain mesh management + compute dispatch |
| `vulkan/shaders/rendering/terrain_mesh_gen.comp` | Compute: cylindrical height field → Cartesian triangle mesh |
| `vulkan/shaders/rendering/terrain.vert` | Read SSBO vertices, transform by VP matrix |
| `vulkan/shaders/rendering/terrain.frag` | Lambert shading with height-based coloring |
| `vulkan/include/rendering/view_projection_ubo.hpp` | Shared VP matrix UBO (reused by all mesh layers) |

**Modified files:** `window_renderer.cpp` (depth attachment + composite backend factory), `volume.frag` (write `gl_FragDepth` at first opaque hit).

**Performance budget:** 256x256 terrain = ~393K vertices, ~9.4 MB SSBO. One-time compute dispatch. Depth buffer = ~7 MB at 1280x720.

**Verification:** Bell-mountain silhouette matches analytical profile. Depth compositing renders terrain/volume overlap correctly. Flat terrain produces a circular disk (cylindrical projection check).

---

#### S7.3 Isosurface Rendering (GPU Marching Cubes)

Render isosurfaces of vorticity (or any scalar field) at user-specified thresholds, colored by a secondary field. Two-pass GPU marching cubes — classify cells + prefix sum, then emit vertices. Use `vkCmdDrawIndirect` to avoid CPU readback.

**Compute passes:**
1. **mc_classify.comp** — Classify each grid cell's 8 corners against threshold, look up triangle count from MC edge table, write per-cell count
2. **mc_prefix_sum.comp** — Parallel Blelloch scan for output offset computation
3. **mc_emit.comp** — Interpolate vertex positions along edges, compute normals via central differences, write to output SSBO

**New files:**
| File | Purpose |
|------|---------|
| `vulkan/include/rendering/isosurface_layer.hpp` | IsosurfaceLayer : RenderLayer |
| `vulkan/src/rendering/isosurface_layer.cpp` | MC dispatch orchestration, threshold control |
| `vulkan/include/rendering/mc_tables.hpp` | Marching cubes edge/triangle lookup tables (256 entries) |
| `vulkan/shaders/rendering/mc_classify.comp` | Classify cells, count triangles |
| `vulkan/shaders/rendering/mc_prefix_sum.comp` | Parallel prefix sum |
| `vulkan/shaders/rendering/mc_emit.comp` | Emit interpolated triangle vertices |
| `vulkan/shaders/rendering/isosurface.vert` | SSBO read, VP transform |
| `vulkan/shaders/rendering/isosurface.frag` | Phong shading + colormap LUT + optional alpha blending |

**Performance:** 256^3 grid → 16M cells, <5ms classification on M3 Pro. Typical output: 100K-500K triangles. Extraction only on threshold/data change — mesh is cached between frames.

**Verification:** Synthetic sphere field produces correct spherical isosurface. Threshold sweep animates smoothly. Indirect draw count matches expected triangle count.

---

#### S7.4 Streamline / Vortex Tube Rendering

3D streamlines following the velocity field, displayed as tubes colored by vorticity magnitude. This is the single most visually striking feature from the CM1/VisIt renders.

**Compute passes:**
1. **streamline_integrate.comp** — RK4 integration through the velocity field (sampled from 3D textures). Handles cylindrical-to-Cartesian velocity conversion:
   ```glsl
   float r = sqrt(x*x + z*z);
   float theta = atan(z, x);
   vec3 v_cyl = sample_velocity(r, theta, y);  // (v_r, v_theta, v_z)
   vec3 v_cart = vec3(
       v_cyl.x * cos(theta) - v_cyl.y * sin(theta),
       v_cyl.z,
       v_cyl.x * sin(theta) + v_cyl.y * cos(theta)
   );
   ```
2. **streamline_tubes.comp** — Generate tube cross-sections (8 vertices per ring) + triangle strip connectivity from polyline data

**Seed point placement:** CPU-side generator places seeds in regions of high vorticity, uniformly in a region, or at user-specified positions.

**Quality controls (scales from laptop to workstation):**
| Parameter | Student (128^3) | Research (256^3) | Production (512^3) |
|-----------|----------------|-----------------|-------------------|
| Seed count | 256 | 1024 | 4096 |
| Integration steps | 100 | 300 | 500 |
| Tube segments | 4 | 8 | 12 |

**New files:**
| File | Purpose |
|------|---------|
| `vulkan/include/rendering/streamline_layer.hpp` | StreamlineLayer : RenderLayer |
| `vulkan/src/rendering/streamline_layer.cpp` | Seed generation, compute dispatch, quality controls |
| `vulkan/include/rendering/seed_point_generator.hpp` | CPU-side seed placement strategies |
| `vulkan/shaders/rendering/streamline_integrate.comp` | RK4 integration |
| `vulkan/shaders/rendering/streamline_tubes.comp` | Tube mesh generation from polylines |
| `vulkan/shaders/rendering/streamline.vert` | SSBO read, VP transform |
| `vulkan/shaders/rendering/streamline.frag` | Phong + colormap by vorticity magnitude |

**Performance:** 1024 streamlines x 300 steps x 8 segments = ~2.4M vertices, ~77 MB. Integration: ~1.2M texture lookups, <5ms on M3 Pro. Re-integrate only on data/seed change.

**Verification:** Rankine vortex field produces concentric circular streamlines. Streamlines never intersect in divergence-free fields. Performance scales linearly with seed count.

---

#### S7.5 Text Overlays + Self-Shadowing + Multi-Scattering

HUD annotations (timestamps, colorbars, scale bars) and volume rendering quality improvements that bridge the gap to photorealistic cloud rendering.

**Overlay system:**
- Bitmap font atlas (16x16 ASCII, 8x16 glyphs) — no external font library needed
- `OverlayLayer : RenderLayer` with depth test disabled, alpha blend enabled
- Colorbar: vertical gradient quad + tick marks + labels

**New files:**
| File | Purpose |
|------|---------|
| `vulkan/include/rendering/overlay_layer.hpp` | OverlayLayer : RenderLayer |
| `vulkan/src/rendering/overlay_layer.cpp` | Text layout, colorbar rendering |
| `vulkan/include/rendering/font_atlas.hpp` | Bitmap font atlas generation |
| `vulkan/src/rendering/font_atlas.cpp` | Atlas texture creation |
| `vulkan/include/rendering/colorbar_renderer.hpp` | Colorbar geometry + label layout |
| `vulkan/shaders/rendering/overlay.vert` | Screen-space quad positioning |
| `vulkan/shaders/rendering/overlay.frag` | Font atlas sampling + tinting |

**Volume rendering improvements (modified `volume.frag`):**

1. **Self-shadowing** — At each sample point with significant density, march 4-8 secondary steps toward the sun. Accumulate extinction along the shadow ray. Apply as multiplier on in-scatter. Single biggest quality improvement for photorealistic clouds.

2. **Multi-scattering approximation** — Wrenninge/Hillaire model: after shadow attenuation, add back a fraction as isotropic scatter. Prevents unrealistically dark cloud interiors.

3. **Transfer function LUT** — Replace hardcoded `density_from_sample()` with a 1D texture lookup, enabling per-field opacity curves.

**Performance impact:** Self-shadowing with 8 steps ≈ 9x texture samples per march step. At 192 steps on 1280x720: ~1.6B lookups/frame. 30+ FPS on 256^3, may need quality slider for 512^3. Quality presets: shadow steps (0/4/8), march steps (96/192/384).

**Verification:** Shadow on uniform density cube forms correctly on light-opposite side. Multi-scatter doesn't exceed energy conservation. Text readable at 720p. Colorbar gradient matches volume LUT.

---

#### S7 Dependency Graph

```
S7.1 (floor + colormaps) ─────────────────────────────────────────┐
                                                                    │
S7.2 (composite + depth + terrain) ──┬── S7.3 (isosurfaces) ──────┤
                                     │                              │
                                     └── S7.4 (streamlines) ───────┤
                                                                    │
S7.5 (overlays + shadows) ────────────────────────────────────────┘
```

S7.1 is independent. S7.2 enables S7.3 and S7.4 (which are independent of each other). S7.5 overlay system needs S7.2's composite layers; volume shader improvements are standalone.

#### S7 Key Existing Files

- `vulkan/shaders/rendering/volume.frag` — Core shader, modified in S7.1, S7.2, S7.5
- `vulkan/src/rendering/volume_backend.cpp` — 1500+ lines, refactored to layer in S7.2
- `vulkan/include/rendering/render_backend.hpp` — Interface extended in S7.2
- `vulkan/src/rendering/window_renderer.cpp` — Render pass + framebuffer changes in S7.2
- `vulkan/include/core/field3d.hpp` — Cylindrical grid structure (NR, NTH, NZ)

---

### S8. Multi-Scale Storm-to-Tornado Workflow (Dual-Backend Capability)

The dual-backend architecture, combined with shared coordinate-blind physics modules from Phase B, opens a workflow that no single-coordinate model makes convenient:

1. Run a **Cartesian supercell** simulation (Phase A backend) with the full WK2002 hodograph, microphysics, PBL, radiation. Identify the mesocyclone region (e.g., low-level rotation maximum, tornadic vortex signature in the synthetic radar field).
2. Re-run the inner subdomain in **cylindrical C-grid** (Phase C backend) centered on the mesocyclone, with the same physics modules and the same trigger / sounding. The cylindrical grid resolves the vortex-scale dynamics (cyclostrophic balance, axisymmetric secondary circulation, axis-region surface friction) at the discretization the physics actually wants.
3. Compare features that are stable across the coordinate change (mass flux, vorticity magnitude, thermodynamic profile) vs. features that change (azimuthal asymmetry, axis-region wind structure). Anything stable across the change is a real feature of the physics. Anything that changes is partly numerical and worth flagging.

This is the inverse of the standard nested-grid approach — instead of refining resolution, you **change the discretization to match the local geometry of the flow**. Cartesian for the storm scale where the inflow is rectilinear; cylindrical C-grid for the tornado scale where the flow is locally rotational and the grid singularity coincides with a physical feature.

**What to build:** A subdomain extraction utility that reads a Cartesian output snapshot, identifies the rotation center, samples onto a cylindrical C-grid, and writes a re-runnable initial-condition file. Shared physics modules already work on either backend (no code change needed).

**Why this is unique:** CM1 only has Cartesian. WRF only has Cartesian (with optional curvilinear maps that aren't vortex-native). Cylindrical-only research codes (e.g., axisymmetric tornado-vortex chambers in the Lewellen / Nolan tradition) cannot ingest a non-axisymmetric storm-scale supercell as IC. The dual-backend architecture bridges that gap.

### S9. Tornado-Native Compressible NWP-Class Discretization

This is a *capability claim* about what the architecture itself enables, not a future build. Document it carefully because it is what most distinguishes the project at AMS:

- **Arakawa C-grid staggering on a cylindrical grid.** Velocity components on r-faces (u), θ-faces (v), z-faces (w); scalars at cell centers. The standard for compressible NWP — what CM1, WRF, and MPAS all use — applied for the first time (to the author's knowledge, in a personal-hardware code) to a cylindrical mesh with the singular axis treated rigorously.
- **Klemp-Wilhelmson split-explicit time stepping.** Slow tendencies (advection, buoyancy, centrifugal/curvature) advanced with the large dt; fast acoustic tendencies (pressure, mass-divergence-driven momentum) substepped at small dt within each large step. The slow + fast = total identity is asserted bit-exactly to float roundoff in the test suite (Phase C.5 Gate 5). Equilibrium states (hydrostatic, cyclostrophic) are fixed points of the discrete operator over hundreds of large steps.
- **Axis singularity by construction, not by patching.** The 0/0 of `(1/r)·d(ru)/dr` at i = 0 is replaced by the control-volume derivation `2·u[0]/dr`, computed inside `StaggeredCylindricalDerivatives::div_flux()`. Dynamics scheme loops never see the special case. The collocated antisymmetric ghost `u[0] = −u[1]` (which created Bug 7's false-divergence drive) is gone.
- **Pairwise machine-precision mass conservation.** Same face flux value on both sides of every interior face, in floating point, debited from one cell and credited to the adjacent one. Total scalar mass drifts only at the per-step double→float storage cast level (~1e-7 relative per thousand steps in the C.7 tests).
- **Bug 7 is structurally impossible on this discretization.** Verified directly: a hydrostatic state plus uniform Cartesian wind on the C-grid produces dp/dt < 5 Pa/s on a 32-cell-θ grid (the only residual is the inherent O(dθ²) cylindrical-from-Cartesian projection error), versus ~2800 Pa/s on the collocated grid that the same hodograph collapses.

**What to build:** A *validation document* that captures this in one place — Lamb-Oseen preservation gate, hydrostatic-300s gate, acoustic pulse propagation gate, mass conservation gate, Bug-7 negative test gate — with the actual measured numbers from `tests/dynamics/test_*_cgrid_*.cpp`. This becomes part of the AMS submission as evidence that the model is doing the right thing on the right grid for the right physics.

---

### Priority for AMS 2027

| Enhancement | Effort | Impact | Priority |
|-------------|--------|--------|----------|
| **S9: Document tornado-native compressible NWP discretization** | Low (writing, not coding) | Very High — central AMS credibility claim | Do first (alongside C.10 integration tests) |
| **S1: Azimuthal decomposition** | Medium | High — unique capability, now numerically rigorous on C-grid | Do first |
| **S5: Tornadogenesis diagnostics** | Medium | High — core science, conservation properties now exact | Do first |
| **S7.1: Reflectivity floor + colormaps** | Low | High — immediate visual upgrade | Do first |
| **S7.4: Streamline / vortex tubes** | Medium | Very High — the "wow" feature | Do first |
| **S8: Storm-to-tornado workflow** | Medium | High — uniquely enabled by dual-backend | Do second (after C.9 + C.10 ship) |
| **S4: Radar operator enhancement** | Low-Medium | Medium — validates output | Do second |
| **S7.2: Composite + terrain mesh** | Medium | High — enables S7.3/S7.4 | Do second |
| **S7.3: Isosurfaces** | Medium | High — vorticity structure | Do second |
| **S7.5: Overlays + self-shadowing** | Medium | High — photorealism + polish | Do second |
| **S2: Ensemble sensitivity** | Medium | High — research utility | Do third |
| **S6: Near-surface resolution** | Low | Medium — publication quality | Do third |
| **S3: GPU parameter sweeps** | High | Medium — requires Phase C.9 GPU shaders for cgrid | After C.9 ships |

Literature references are placeholders — provide your specific citations when ready and I'll integrate them into the implementation.
