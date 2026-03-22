# TornadoModel — Road to AMS January 2027

**Deadline:** January 2027 AMS Conference (~10 months)
**Goal:** Complete, efficient simulation on personal hardware with reproducible results

---

## ~~Phase 0: Foundation Cleanup (Weeks 1–3)~~ COMPLETE

---

## ~~Phase 1: Vulkan Compute — Make It Real (Weeks 3–10)~~ COMPLETE

---

## Phase 2: Performance, Efficiency & Accessibility (Weeks 6–12)
*Make it fast on high-end hardware. Make it possible on a student's laptop.*

**Design principle:** The same binary should run a meaningful simulation on a base MacBook Air with 8GB RAM *and* saturate an M4 Ultra with 512GB unified memory or a 64-core Threadripper. The user configures the knobs; the software adapts to the hardware.

**#1 Priority — Target Machine Floor:**
- MacBook Air M1 (8GB RAM, 256GB SSD) — must run a meaningful 64×64×32 simulation
- Ubuntu laptop with integrated Intel graphics — must run CPU-only without Vulkan
- Windows WSL2 — must build and run (CPU-only)
- Raspberry Pi 5 (8GB) — stretch goal, would be remarkable for outreach
- **Any post-2018 machine with a C++17 compiler must build and run with zero external dependencies.**

**GPU strategy: Vulkan only.** No CUDA, no Metal.
- MoltenVK translates Vulkan → Metal on macOS automatically (~95% native perf)
- Vulkan covers Windows + Linux natively (NVIDIA/AMD/Intel)
- CPU-only fallback covers everything else
- Adding CUDA (NVIDIA-only) or Metal (Apple-only) would triple the compute backend maintenance for marginal gains and add friction to the build process — the opposite of our goal.

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

### 2.3 GPU-resident field management — PARTIAL
Buffer pool + zero-copy path exist for unified memory. Fields still copied back after every dispatch (no cross-timestep residency). **Low priority on Apple Silicon** (zero-copy already eliminates the main bottleneck). Deferred to Phase 2R.

### 2.4 Async I/O — ~~COMPLETE~~
AsyncOutputWriter with background thread, condition variables, backpressure.

### 2.5 ZFP compression — PARTIAL
ZFP integrated with 3 modes (accuracy, precision, rate). **Delta encoding + keyframe intervals NOT implemented.** Current compression: 10-50×. Target with delta: 30-150×. Deferred to Phase 2R.

### 2.6 Configurable output — ~~COMPLETE~~
5 field presets, 4 output formats (NPY 2D slices, NPY 3D, CSV, ZFP), disk budget estimation, YAML parsing.

### 2.7 Shared memory transport — ~~COMPLETE~~
POSIX SHM transport, ShmWriter, ShmDataset, Vulkan viewer integration. 1058 assertions passing.

### 2.8 Hardware-aware scaling — MOSTLY COMPLETE
CPU/RAM/cache/GPU/SIMD detection and logging. Memory-safe grid warnings. Auto-config is suggestions only (not automatic). **Needs: auto-config that actually applies defaults based on detected hardware.** Deferred to Phase 2R.

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
- [~] ZFP delta encoding — **Write path COMPLETE** (`output_writer.cpp:327-391`). Keyframe intervals, delta computation, v2 header with delta flag all implemented. **Needs: tests, reader, enable in a shipped config.** Resolved in Cycle 1.
- [~] GPU cross-timestep field residency — **Deprioritized with measurement.** Zero-copy on Apple Silicon (unified memory) means transfer overhead is ~1-2 µs/dispatch (< 0.1% of step time). `GPUFieldSnapshot` interface exists but unwired. Not worth the architectural cost on unified memory. Revisit only for discrete GPU support.
- [x] Auto-config applies hardware-based defaults — `apply_hardware_defaults()` implemented and wired into `headless_runtime.cpp`. 6 new tests (31 assertions). Auto-scales grid to fit 80% RAM, logs what changed, calls `resize_fields()` and recomputes disk budget.

**Overall: ~97% complete. ZFP write path was already done (plan was stale). GPU residency deprioritized with evidence. Hardware auto-config implemented and tested (2026-03-20). Remaining: ZFP needs tests + reader + config enablement.**

---

## Phases 2R+3: Audit-Test Cycles (Weeks 12–18)
*Every finding gets a test. Every test gets a fix. Every fix gets re-audited.*

Phase 1 and Phase 2 are foundational. Every future addition builds on this architecture. If the foundation has cracks, they compound. Phases 2R and 3 are not sequential — they are a single interleaved loop that ensures nothing slips through.

### How the Loop Works

```
Audit module → find gap or risk
       ↓
Write test that exposes it (RED)
       ↓
Fix the code (GREEN)
       ↓
Test passes — fix is proven
       ↓
Re-audit — did the fix introduce anything new?
       ↓
(repeat until module is clean)
```

**Why this matters:** A standalone audit finds issues it can't prove. Standalone tests cover what you *think* the code does, not what it *actually* does. The loop forces contact with reality at every step. If the plan says something isn't done but the code has it (see ZFP delta below), the audit catches the stale assumption and the test proves the real state.

### Test Baseline (as of 2026-03-20)

Catch2 integrated. **29,833 assertions across 148 test cases — all passing.**

| Suite | Tests | Assertions |
|-------|-------|------------|
| Core (field, output, hardware, npy, writer, shm) | 64 | 1,397 |
| Diagnostics (contract, validation) | 19 | 446 |
| Numerics (advection, diffusion, timestepping) | 16 | 20 |
| Physics (microphysics, radiation, terrain) | 27 | 8,381 |
| Data (soundings) | 7 | 41 |
| Vulkan (backend, gpu_parity) | 11 | 18,844 |
| Integration (performance) | 12 | 23 |
| SHM E2E | 3 | 1,058 |

---

### Infrastructure Gate (do first — enables all cycles)

These are prerequisites. They make every subsequent cycle faster and more trustworthy.

**CI matrix expansion**
- Add macOS runner (the actual dev platform)
- Add clang + gcc matrix
- Add Vulkan smoke test (at least `--dry-run`)
- Add performance baseline comparison (flag regressions)

**Code quality tooling**
- `.clang-format` config — format the codebase once
- `.clang-tidy` config — static analysis
- CI checks (warnings, not blockers initially)

**Logging ratchet** (cross-cutting guardrail)
- Current state: **279 raw `std::cout`/`std::cerr` across 33 files**, despite `log.hpp` existing with `log_info`/`log_warn`/`log_error`/`log_debug` and level gating
- Worst offenders: `runtime_config.cpp` (61), `headless_runtime.cpp` (49), `tornado_sim.cpp` (35)
- Write a guardrail test: grep `src/` for raw `std::cout`/`std::cerr`, assert count ≤ threshold
- Start at threshold = 279 (current), ratchet down as each cycle migrates a module
- Re-audit after each batch: did we lose any user-facing status messages? Are error paths now capturable?

---

### Cycle 0: Spatial & Temporal Optimization (PRIORITY — gates Phase 4)

**This cycle is the architectural bridge between a working simulation and a scientifically useful one.** Without it, production runs produce terabytes of output and take days of wall-clock time. With it, 8–24 hour storm lifecycle studies fit in 100 GB and overnight compute budgets. **Phase 4 (Physics Completeness) does not begin until Cycle 0 acceptance criteria are met** — adding more physics to an unoptimized pipeline just makes the problem worse.

**Two problems, one cycle:**
1. **Space:** A 2-hour production run currently produces **243 GB** (30x ZFP, uniform 5s cadence, global tolerance). Target: **26 GB** (Tier 2b).
2. **Time:** A 2-hour production run takes **27 hours** wall-clock on M4 Max. Dynamics is 86% of step time; vertical advection is 93% of that. Reducing per-step cost directly unlocks longer simulations.

**The compression tiers from `docs/SpatialBenchmarks.md` are the implementation milestones:**

| Tier | What | Code Change | Eff. Ratio | 2h Production | Gate |
|------|------|-------------|-----------|---------------|------|
| **0** | Current state | None | 30x | 243 GB | — |
| **1** | Enable delta + relax tolerance | Config only | 50-70x | 104-146 GB | Validated 30x assumption |
| **2** | Per-field ZFP tolerance map | ~200 LOC | 80x | 91 GB | Field contract bounds used |
| **2b** | + Tiered write cadence | ~300 LOC | 80x + cadence | **26 GB** | **Phase 4 gate** |
| **3** | Sparse + predictive delta | ~750 LOC | 150x + cadence | **14 GB** | 24h diurnal cycle feasible |

**What each tier unlocks (100 GB budget, production grid):**
- Tier 1: **4-hour** supercell lifecycle at 13s cadence
- Tier 2b: **8-hour** storm lifecycle at 10s core cadence (82 GB), **12-hour** at 15s core (100 GB)
- Tier 3: **24-hour** full diurnal cycle at 15s core (70 GB)

**Wall-clock reduction targets (companion to space optimization):**

The spatial benchmarks show dynamics dominates (86% of step time). Vertical advection alone is 93% of advection time. Two paths to improve wall-clock:

| Optimization | Target | Expected Speedup | Where |
|-------------|--------|------------------|-------|
| GPU-accelerated vertical advection | Move z-sweep to Vulkan compute | 2-4x on step time | `src/advection/`, Vulkan shaders |
| Reduced field count mode | Export subset during compute, full set at cadence | Eliminates I/O contention | `headless_runtime.cpp` |
| 2D axisymmetric for parameter sweeps | `grid.ny=1` already works | 50-250x faster per sim-second | Config only |
| Adaptive output cadence | Write more during active tornadogenesis, less during spinup | Reduces total exports 30-50% | `output_config.hpp` |

**Combined impact:** Tier 2b compression + GPU vertical advection could bring a 2-hour production run from 27h/243GB down to ~10h/26GB — making overnight runs practical and 8-hour storm lifecycle studies achievable over a weekend.

**Audit targets:** `output_writer.cpp`, `output_config.cpp`, `output_config.hpp`, `headless_runtime.cpp`, `npy_writer.cpp`, `field_contract.cpp`

#### Step 0.1: Validate the 30x assumption (benchmark, not code)
- Run a 60s production simulation with ZFP+delta enabled
- Measure actual compression ratio per field type (dynamics, moisture, diagnostics)
- Compare against the assumed 30x in `output_config.cpp:234`
- Document results in `docs/SpatialBenchmarks.md`
- **Test:** Assert measured ratio matches estimate within 2x (if not, update estimate)

#### Step 0.2: Enable delta encoding in shipped configs (config only, zero code)
- Add `zfp_keyframe_interval: 10` to `production.yaml` and `research.yaml`
- **Test:** Run production config, verify delta frames are written (header flag = 0x01)
- **Test:** Verify keyframe at frame 0, delta at frames 1-9, keyframe at frame 10
- **Test:** Verify delta files are smaller than keyframe files for slowly-varying fields
- **Re-audit:** Does enabling delta break any existing test?

#### Step 0.3: Per-field ZFP tolerance map (~200 LOC)
The field contract already has `min`/`max` bounds for every field. Use them.

**New code:**
- Add `std::unordered_map<std::string, double> field_tolerances` to `OutputConfig`
- Add `output.zfp_field_tolerances` YAML section (optional, overrides global)
- Add built-in tolerance tiers using field contract bounds:

| Tier | Fields | Tolerance | Why |
|------|--------|-----------|-----|
| Tight (1e-4) | u, v, w, rho, p, theta, vorticity_{r,th,z} | Must preserve dynamics | Core science fields |
| Moderate (1e-3) | temperature, moisture species, derived thermo, TKE | Sub-millikelvin, sub-ppm precision | More than adequate for analysis |
| Loose (1e-2) | diagnostics, visualization, indices, radar synth | Bounded fields with known ranges | Visual/statistical use only |

**Implementation in `output_writer.cpp`:** Before calling `write_zfp_3d`, look up `field_tolerances[entry.name]` and pass it instead of the global `config_.zfp_tolerance`.

**Tests:**
- Tight field compressed at 1e-4 → verify error < 1e-4 on known input
- Moderate field compressed at 1e-3 → verify error < 1e-3
- Loose field achieves higher compression ratio than tight field on same data
- Roundtrip: write 3 fields at 3 tolerances, read back, verify each meets its bound
- Per-field map overrides global tolerance

**Re-audit:** Does per-field tolerance change disk budget estimates? (Yes — update `estimate_disk_budget` to use weighted average.)

#### Step 0.4: Tiered write cadence (~300 LOC)
Not all fields need the same temporal resolution.

**New code:**
- Add `FieldCadence` struct to `OutputConfig`: `{fast_interval_s, medium_interval_s, slow_interval_s}`
- Add `output.cadence.fast`, `output.cadence.medium`, `output.cadence.slow` YAML keys
- Classify fields into tiers (hardcoded, matching SpatialBenchmarks analysis):
  - **Fast** (15): u, v, w, rho, p, theta, qv, qc, qr, qi, qs, qh, qg, tke, tracer
  - **Medium** (32): vorticity, stretching/tilting/baroclinic, pressures, derived thermo, radar pol.
  - **Slow** (34): surface, column, cross-section, radar synthetic, trajectories, indices
- In `headless_runtime.cpp` export loop: check if current time matches each tier's cadence

**Tests:**
- Fast fields written every 5s, medium every 30s, slow every 60s (verify file existence)
- Fields in each tier are correct (no field dropped, no field duplicated)
- Total output size matches `docs/SpatialBenchmarks.md` Tier 2b estimate (within 10%)
- Uniform cadence mode still works (all tiers set to same interval)

**Re-audit:** Does tiered cadence break SHM transport? (SHM writes all fields every export — need to handle partial exports.) Does it break the Vulkan viewer? (Viewer reads from SHM, not files — should be unaffected.)

#### Step 0.5: ZFP roundtrip reader (~200 LOC)
Delta-compressed output cannot be verified without a reader. This is a test infrastructure gap.

**New code:**
- `read_zfp_3d()` function: reads v2 header, decompresses payload, handles delta flag
- Delta reconstruction state machine: track last keyframe per field, `reconstructed = keyframe + delta`
- Add to `npy_reader.hpp` or new `zfp_reader.hpp`

**Tests:**
- Write keyframe → read back → values match within tolerance
- Write keyframe + 3 deltas → reconstruct all 4 frames → verify each matches original
- Corrupted header → graceful error (not crash)
- Wrong version → clear error message
- Reader handles missing keyframe before delta → error

#### Step 0.6: Eliminate per-frame delta allocation
- `output_writer.cpp:375`: `std::vector<float> delta(n)` allocates on every non-keyframe export
- Replace with persistent `delta_buffer_` member, resize only when field size changes
- **Test:** Run 100 exports, verify no allocation spikes (measure with custom allocator or timing)

#### Step 0.7: Real compression benchmark
After Steps 0.2-0.4 are implemented:
- Run full 120s production simulation with Tier 2b configuration
- Measure actual GB per export, per tier, per field type
- Compare against `docs/SpatialBenchmarks.md` projections
- Update the document with measured (not estimated) ratios
- **This is the gate for Phase 4.** If measured ratios are worse than projected, iterate on tolerance tuning before adding new physics.

#### Step 0.8: Wall-clock profiling and vertical advection optimization
After compression pipeline is validated:
- Profile production grid (512x256x128) with Instruments — confirm dynamics is 86%+ and vertical advection dominates
- Identify whether GPU vertical advection dispatch is feasible (shader already exists for horizontal advection)
- If vertical advection shader yields ≥ 2x speedup on z-sweep: wire it into the main loop
- If not: document why and identify next-best optimization target
- **Test:** Benchmark before/after — measure ms/step improvement
- **Test:** GPU vertical advection produces identical results to CPU path (add to `test_vulkan_gpu_parity`)

#### Cycle 0 Acceptance Criteria

**Space (Tier progression — each must be validated before moving to next):**
- [ ] **Tier 0 validated:** Measured actual compression ratio with current ZFP settings. Document baseline.
- [ ] **Tier 1 achieved:** Delta enabled in production.yaml + research.yaml. Measured ratio ≥ 50x.
- [ ] **Tier 2 achieved:** Per-field ZFP tolerance map using field contract bounds. Measured weighted ratio ≥ 80x.
- [ ] **Tier 2b achieved:** Tiered write cadence (5s/30s/60s). 2-hour production run ≤ 30 GB (target: 26 GB). **This is the Phase 4 gate.**
- [ ] ZFP reader with delta reconstruction — roundtrip verified
- [ ] Per-frame delta allocation eliminated (persistent buffer)
- [ ] `docs/SpatialBenchmarks.md` updated with measured (not assumed) ratios at each tier

**Time (wall-clock reduction):**
- [ ] Production grid profiled — dynamics/advection breakdown documented
- [ ] Vertical advection GPU shader evaluated (≥ 2x speedup or documented why not)
- [ ] 2D axisymmetric mode (ny=1) validated for parameter sweeps
- [ ] ms/step baseline established for each config tier (student, research, production)

**Combined gate for Phase 4:**
- [ ] 2-hour production run: ≤ 30 GB storage AND measured wall-clock documented
- [ ] 8-hour run feasibility confirmed: ≤ 100 GB at Tier 2b with 10s core cadence
- [ ] Student config (64x64x32): ≤ 60 MB output, runs in < 6 minutes wall-clock

---

### Cycle 1: Core Infrastructure (output, hardware, config)

**Audit targets:** `output_writer.cpp`, `hardware_info.cpp`, `runtime_config.cpp`, `output_config.cpp`

**Known findings (from 2026-03-20 audit):**

| Finding | Source | Status |
|---------|--------|--------|
| ZFP delta encoding | Plan says "NOT implemented" | **Actually COMPLETE** — `output_writer.cpp:327-391` has keyframe intervals, delta computation, previous-field storage. Plan was stale. |
| Delta per-frame allocation | `output_writer.cpp:375` | `std::vector<float> delta(n)` allocates every non-keyframe. Should use FieldPool or a persistent buffer. |
| Hardware auto-config doesn't apply | `hardware_info.cpp` | `check_grid_memory_safety()` warns but never adjusts. Student on 8GB MacBook Air gets a log warning, then OOMs. |
| Logging: `runtime_config.cpp` | 61 raw `std::cout`/`std::cerr` | Migrate to `log.hpp`. Ratchet threshold down. |

**Tests to write:**

- ZFP delta: delta frame produces smaller output than keyframe for slowly-varying data
- ZFP delta: keyframe interval=5 produces a keyframe every 5th frame
- ZFP delta: empty `previous_fields_` on non-keyframe falls back to keyframe (line 360 guard)
- Hardware auto-config: given 8GB RAM + 256³ config → system downsizes to safe grid
- Hardware auto-config: given 48GB RAM + 256³ config → grid unchanged
- Hardware auto-config: suggested grid has positive dims and preserves aspect ratio
- Hardware auto-config: downscale is logged (student needs to know what happened)

**Fixes to apply:**
- Mark ZFP delta encoding COMPLETE in deferred items
- Implement `apply_hardware_defaults()` that modifies `RuntimeConfig&`
- Replace delta per-frame allocation with persistent buffer
- Migrate `runtime_config.cpp` logging (61 → 0)

**Re-audit checkpoint:** Tests pass. Delta allocation eliminated. Auto-config proven on simulated hardware. Logging ratchet threshold reduced.

---

### Cycle 2: Numerics (advection, diffusion, time-stepping, SIMD)

**Audit targets:** `src/numerics/`, `src/advection/`, `include/util/simd_utils.hpp`

**Audit questions:**
- Are TVD limiters (MC, van Leer, superbee, universal) producing monotone results?
- Does implicit diffusion preserve positivity of density, moisture fields?
- Is RK3/RK4 order of accuracy correct? (Richardson extrapolation test)
- Double-precision SIMD: only float overloads exist — TVD advection operates on doubles and misses SIMD entirely
- Are there stability violations at CFL > 0.5 for explicit schemes?

**Tests to write:**
- Advection: top-hat profile advected one full domain — verify monotonicity for each limiter
- Advection: smooth Gaussian — measure L2 error, verify convergence rate matches scheme order
- Diffusion: constant field → zero tendency; linear gradient → uniform tendency
- Diffusion: positivity test — field with small positive values stays positive after diffusion step
- Time-stepping: integrate `dy/dt = -y` — verify RK3 is 3rd order, RK4 is 4th order via Richardson extrapolation
- SIMD: double-precision overloads — parity with scalar path for add, multiply, fma, reduce_sum
- SIMD: NaN input → NaN output (no silent corruption)
- SIMD: count < SIMD width → correct scalar tail handling

**Deferred Phase 2 item to resolve:**

| Item | Action |
|------|--------|
| Double-precision SIMD | Implement `double*` overloads for NEON/AVX/SSE/scalar. Wire into TVD advection + dynamics intermediates. Test proves parity. |

**Re-audit checkpoint:** All limiters proven monotone. Order of accuracy verified. Double-precision SIMD wired and tested. No stability violations in edge cases.

---

### Cycle 3: Physics (microphysics, boundary layer, turbulence, radiation, dynamics)

**Audit targets:** `src/microphysics/`, `src/boundary_layer/`, `src/turbulence/`, `src/radiation/`, `src/dynamics/`

**Audit questions:**
- Are microphysics tendencies physically bounded? (no negative mixing ratios, no runaway temperatures)
- Do 4 microphysics schemes agree on sign and order-of-magnitude for identical inputs?
- Does the radiation silent fallback (`radiation.cpp` substituting default lapse rate without warning) still exist?
- Does `RadiationColumnStateView` validate required fields before `compute_column()`?
- Are boundary layer surface fluxes sensible for extreme stability (very stable, very unstable)?
- Does TKE stay non-negative?
- Are dynamics vorticity budget terms (stretching, tilting, baroclinic) internally consistent?

**Known findings:**
- Microphysics init duplication: 4 schemes repeat identical resize+convert → factor to base class helper
- `RadiationColumnStateView` allows nullptr for fields without documenting which are required
- Turbulence factory inline lambda in `factory.hpp` → use `tmv::strutil::trim_and_lower()`

**Tests to write:**
- Each microphysics scheme: known-input/known-output (warm rain: autoconversion rate from Kessler formula, etc.)
- Each microphysics scheme: conservation test — total water mass before = after tendency application
- Microphysics edge cases: zero mixing ratios → zero tendencies, saturated column, extreme temperatures
- Radiation: column with known optical depth → verify heating rate against analytical Beer-Lambert solution
- Radiation: nullptr field in `RadiationColumnStateView` → test that guard rejects/throws (after fix)
- Boundary layer: neutral profile → flux matches surface layer theory
- Turbulence: TKE non-negativity after 100 timesteps of decaying turbulence
- Dynamics: solid body rotation → zero stretching/tilting terms

**Fixes to apply:**
- Factor microphysics init to base class helper
- Add `RadiationColumnStateView` field validation with clear error messages
- Fix radiation silent lapse rate fallback → `log_warn` at minimum
- Turbulence factory: replace inline lambda

**Re-audit checkpoint:** All schemes produce bounded, conserving results. Edge cases handled. Radiation guards in place. Init duplication eliminated.

---

### Cycle 4: Data & Diagnostics (soundings, field contract, chaos)

**Audit targets:** `src/soundings/`, `src/validation/`, `src/chaos/`

**Audit questions:**
- Does sounding QC handle malformed input gracefully? (missing levels, non-monotonic pressure, NaN values)
- Are all field contract fields either Implemented or explicitly NotImplemented with justification?
- Do chaos perturbation modes produce reproducible results given the same seed?
- Is the random generator thread-safe?

**Tests to write:**
- Soundings: malformed input (missing levels, duplicate pressures, NaN temperature) → graceful rejection
- Soundings: interpolated profile matches analytical atmosphere (isothermal, constant lapse rate)
- Field contract: roundtrip — every Implemented field can be computed and validated without error
- Chaos: same seed → identical perturbation field (reproducibility)
- Chaos: different seeds → different fields (non-degeneracy)
- Chaos: perturbation magnitudes within configured bounds

**Re-audit checkpoint:** Sounding QC robust. Field contract inventory current. Chaos reproducibility proven.

---

### Cycle 5: Compute & I/O (Vulkan, SHM, async output)

**Audit targets:** `src/core/compute_backend.cpp`, `src/core/shm_writer.cpp`, `vulkan/`

**Audit questions:**
- Is GPU dispatch overhead net-positive at every call site? (profile, don't assume)
- Does SHM transport handle reader disconnect gracefully?
- Does async output writer handle backpressure without data loss?
- Are Vulkan buffer lifetimes correct? (no use-after-free on swapchain recreate)

**Deferred Phase 2 item:**

| Item | Action |
|------|--------|
| GPU field residency | Profile cross-timestep copy overhead. On Apple Silicon (unified memory), verify zero-copy path means this is a non-issue. On discrete GPU, measure copy cost vs. residency benefit. Deprioritize if < 2% of step time. |

**Tests to write:**
- Vulkan: dispatch + readback matches CPU reference (existing `test_vulkan_gpu_parity` — 18,835 assertions, already strong)
- SHM: writer closes → reader detects disconnect without crash
- SHM: reader attaches after writer starts → gets current data (late-join)
- Async output: submit faster than write speed → backpressure blocks without data loss
- GPU residency: benchmark cross-timestep copy cost (measurement, not assertion)

**Re-audit checkpoint:** GPU dispatch proven net-positive or removed. SHM lifecycle robust. Async output handles all pressure scenarios.

---

### Cross-Cutting: Performance Profiling

Runs **after each cycle**, not as a separate phase. Catches regressions introduced by fixes.

- Profile benchmark config (256×256×64, 60s) with Instruments on macOS
- Track top 10 hottest functions
- Compare against previous cycle's baseline
- Flag any function that moved into the top 10 after a fix
- Verify FieldPool and buffer reuse are working (allocation profile)
- Verify OpenMP thread balance (no false sharing)

---

### Deferred Phase 2 Items — Resolved Status (audited 2026-03-20)

| Item | Original Status | Resolved By | Actual State |
|------|----------------|-------------|--------------|
| ZFP delta encoding | Plan said "NOT implemented" | Cycle 1 audit | **Write path COMPLETE** — `output_writer.cpp:327-391` has v2 header, keyframe intervals, delta computation, previous-field state tracking. Plan was stale. **Gaps:** zero tests, no reader, disabled in all configs, per-frame `vector<float>` alloc at line 375. |
| Double-precision SIMD | Not started | Cycle 2 | Implement + test + wire into TVD advection + dynamics intermediates |
| Hardware auto-config | Detection only | Cycle 1 audit | `check_grid_memory_safety()` warns at 80% RAM threshold and suggests safe grid, but **suggestion is discarded**. Grid allocates regardless. Student laptop OOMs. **Fix:** implement `apply_hardware_defaults()`, respect explicit config overrides, add `--no-auto-scale` flag. |
| GPU field residency | Not measured | Cycle 5 audit | **Deprioritized with measurement.** Zero-copy on unified memory (Apple Silicon) = ~1-2 µs/dispatch transfer overhead (< 0.1% of step time). `GPUFieldSnapshot` interface exists (buffer pool, dirty bits, upload/download) but has **zero call sites**. Not worth wiring for < 0.1% gain. Revisit for discrete GPU support only. |
| Factory duplication | 12 files | N/A | **Already largely solved** by `SchemeRegistry<Base>` template. Each factory is a thin wrapper (create + available). Further reduction is micro-optimization — deprioritized. |
| Logging inconsistency | Plan said ~75 sites | Infrastructure gate | **Actually 279 across 33 files.** Worst: `runtime_config.cpp` (61), `headless_runtime.cpp` (49), `tornado_sim.cpp` (35). `log.hpp` exists with full API. Ratchet down via guardrail test. |

---

### Phase 2R+3 Acceptance Criteria

- [ ] **Cycle 0 complete (Phase 4 gate):** Tier 2b compression validated. 2h production ≤ 30 GB. 8h run ≤ 100 GB. Wall-clock profiled. Vertical advection GPU path evaluated.
- [ ] Every module audited with findings documented per cycle
- [ ] Every finding has a corresponding test
- [ ] Every test passes after fix
- [ ] Every fix re-audited for collateral damage
- [ ] Profiling data captured after each cycle — no performance regressions
- [ ] All deferred Phase 2 items resolved or explicitly deprioritized with measurement
- [ ] Logging ratchet: 279 → < 50 raw `std::cout`/`std::cerr` calls
- [ ] CI runs on macOS + Linux with clang + gcc
- [ ] `.clang-format` and `.clang-tidy` configs in place
- [ ] Build + all tests pass on macOS (ARM) and Linux (x86)

---

## Phase 4: Physics Completeness (Weeks 18–24)
*Fill the science gaps for a credible AMS presentation. Only begins after Cycle 0 proves the output pipeline can handle the load (Tier 2b: ≤ 30 GB for 2h, ≤ 100 GB for 8h) and wall-clock times are documented. Adding more physics without optimized I/O just compounds the storage/compute problem.*

**Prerequisites (from Cycle 0):**
- Tier 2b compression validated and enabled in production/research configs
- Per-field tolerance map active (field contract bounds drive ZFP)
- Tiered cadence operational (fast/medium/slow field groups)
- Wall-clock baseline established for each grid tier
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
| **2: Performance** | 6–12 | SIMD, memory pools, cache blocking, output pipeline, GPU shaders, hardware scaling | **~92% COMPLETE** |
| **2R+3: Audit-Test Cycles** | 12–18 | **Cycle 0 (spatial/temporal optimization) gates Phase 4.** Tier 2b: 26 GB/2h, 82 GB/8h. Wall-clock profiled. Then Cycles 1-5 audit each module. | Foundation proven bulletproof with evidence |
| **4: Physics** | 18–24 | Radiation, field contract, validation. **Blocked until Cycle 0 Tier 2b validated.** Every new field gets a compression tier + cadence group. | Scientific credibility for AMS |
| **5: Polish** | 22–26 | Refactoring, docs, demo script, first-run experience | Presentation readiness |

---

## Claude Code Skills for This Project

Create these in `.claude/commands/` (one markdown file per skill):

### Essential Skills

**`.claude/commands/build.md`** — Build targets
```
Build the project. Run `make -j$(sysctl -n hw.ncpu)` for the main sim, `make vulkan` for the viewer. Report any errors clearly. If a specific target is mentioned like "radiation" or "tests", build only that target.
```

**`.claude/commands/test.md`** — Run tests
```
Run the test suite with `make test`. If a specific module is mentioned (e.g., "radiation", "soundings", "terrain"), run only that test target (e.g., `make test-radiation-regression`). Report pass/fail for each test. If tests fail, read the output and diagnose.
```

**`.claude/commands/bench.md`** — Benchmark
```
Run performance benchmarks. Build with optimizations first (`make clean && make -j$(sysctl -n hw.ncpu)`), then run `./bin/tornado_sim --headless --config=configs/lp.yaml --duration=60` and report wall-clock time, memory usage, and timesteps completed. Compare against previous results if available in data/benchmark/.
```

**`.claude/commands/validate.md`** — Field validation
```
Run field contract validation: `make validate-fields`. Check the output against the contract in `src/validation/field_contract.cpp`. Report: total fields, exported count, not-implemented count, any strict-mode violations. If violations exist, trace them to the source module.
```

**`.claude/commands/check-vulkan.md`** — Vulkan status
```
Audit the Vulkan compute and rendering state. Check:
1. Does `make vulkan` build cleanly?
2. Does `./bin/vulkan_viewer --dry-run` succeed?
3. What's the state of compute dispatch in `src/core/compute_kernel_template.cpp` — is `backend_dispatch_ready` true or false?
4. Are there any `.comp` shader files in vulkan/shaders/?
5. Report what's working vs stubbed.
```

### Development Workflow Skills

**`.claude/commands/add-physics-scheme.md`** — Scaffold a new physics scheme
```
Create a new physics scheme following the factory pattern. The user will specify: module (microphysics, boundary_layer, turbulence, radiation, terrain, chaos), scheme name, and basic parameters.

Steps:
1. Create `src/{module}/schemes/{scheme_name}/{scheme_name}.cpp` and `.hpp`
2. Register in `src/{module}/factory.cpp`
3. Add config parsing in `src/core/runtime_config.cpp`
4. Follow the pattern of existing schemes in that module
5. Add a basic test case
```

**`.claude/commands/audit-module.md`** — Deep-dive a specific module
```
Perform a detailed audit of the specified module. Read all source files in the module directory. Report:
1. What's implemented vs stubbed
2. Code quality issues (duplicated logic, missing error handling, dead code)
3. Header/implementation consistency
4. TODO/FIXME comments
5. Suggested improvements prioritized by impact
```

**`.claude/commands/profile.md`** — Profile a simulation run
```
Profile a simulation run to find performance bottlenecks. Build with debug symbols (`make clean && make CFLAGS="-O2 -g"`), run a short simulation, and analyze where time is spent. On macOS, use `sample` or Instruments. Report the top 5 hottest functions and suggest optimization targets.
```

### To create these skills, run:
```bash
mkdir -p .claude/commands
# Then create each .md file listed above
```

Invoke them with `/build`, `/test`, `/bench`, `/validate`, `/check-vulkan`, `/add-physics-scheme`, `/audit-module`, `/profile` in Claude Code.

---

## Scientific Enhancements — What This Architecture Can Uniquely Do

Your cylindrical-coordinate, modular-physics architecture on personal hardware opens research avenues that CM1 either doesn't prioritize or makes inconvenient. These are scientifically motivated extensions that play to your model's strengths.

### S1. Vortex-Centric Analysis (Natural Fit for Cylindrical Grid)

CM1 uses Cartesian coordinates and requires post-hoc coordinate transforms for vortex analysis. Your native (r, θ, z) grid makes the following **first-class operations** instead of post-processing hacks:

- **Azimuthal decomposition of vortex structure** — Decompose any field into wavenumber-0 (axisymmetric mean) and wavenumber-1,2,...,n (asymmetric perturbations) directly on the native grid. This is trivial in cylindrical coords and expensive/lossy in Cartesian.
- **Angular momentum budgets** — Track absolute angular momentum flux through cylindrical surfaces natively. No interpolation artifacts.
- **Vortex Rossby wave dynamics** — Diagnose VRW propagation, reflection, and wave-mean flow interaction using azimuthal wavenumber decomposition. Literature: Montgomery & Kallenbach (1997), Nolan & Montgomery (2002).
- **Secondary circulation diagnostics** — Compute transverse (r,z) streamfunction for the axisymmetric overturning circulation directly.

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

Your dynamics module computes vorticity components and stretching/tilting/baroclinic terms. Extensions for tornadogenesis research:

- **Circulation budget analysis** — Compute material circulation tendencies around circuits in (r,z) plane. Track how circulation concentrates from mesocyclone to tornado scale. Literature: Rotunno & Klemp (1985), Markowski & Richardson (2014).
- **Dynamic pressure decomposition** — Separate perturbation pressure into linear and nonlinear dynamic components plus buoyancy pressure. Diagnose which drives the low-level updraft. Literature: Klemp & Rotunno (1983).
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

### Priority for AMS 2027

| Enhancement | Effort | Impact | Priority |
|-------------|--------|--------|----------|
| **S1: Azimuthal decomposition** | Medium | High — unique capability | Do first |
| **S5: Tornadogenesis diagnostics** | Medium | High — core science | Do first |
| **S7.1: Reflectivity floor + colormaps** | Low | High — immediate visual upgrade | Do first |
| **S7.4: Streamline / vortex tubes** | Medium | Very High — the "wow" feature | Do first |
| **S4: Radar operator enhancement** | Low-Medium | Medium — validates output | Do second |
| **S7.2: Composite + terrain mesh** | Medium | High — enables S7.3/S7.4 | Do second |
| **S7.3: Isosurfaces** | Medium | High — vorticity structure | Do second |
| **S7.5: Overlays + self-shadowing** | Medium | High — photorealism + polish | Do second |
| **S2: Ensemble sensitivity** | Medium | High — research utility | Do third |
| **S6: Near-surface resolution** | Low | Medium — publication quality | Do third |
| **S3: GPU parameter sweeps** | High | Medium — requires Phase 1 | After Vulkan compute works |

Literature references are placeholders — provide your specific citations when ready and I'll integrate them into the implementation.
