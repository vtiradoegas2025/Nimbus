# Physics and Numerical Gaps

Systematic inventory of known gaps in the Nimbus simulation framework. Each gap includes what it affects, why it matters, and what the fix looks like. Items are ordered by impact within each category.

Verified April 20, 2026 through warm bubble testing on the Cartesian backend (64x64x32, student grid, all 12 components active and coupled).

---

## Physics Gaps (Missing Science)

### 1. Precipitation Loading

**What:** The vertical momentum equation uses `-g(rho - rho0)/rho` for buoyancy but does not include the weight of hydrometeors (rain, graupel, hail). The condensate mass acts as ballast that opposes the updraft and drives downdrafts.

**Why it matters:** In real storms, precipitation loading provides 30-50% of the downdraft forcing. Without it, the updraft has no self-limiting mechanism -- convection converts CAPE to kinetic energy without bound, hitting field clamps at ~5 minutes of simulation time.

**Literature:**
- Klemp, J. B., and R. B. Wilhelmson (1978): *The Simulation of Three-Dimensional Convective Storm Dynamics.* Journal of the Atmospheric Sciences, 35, 1070-1096.
- Bryan, G. H., and J. M. Fritsch (2002): *A Benchmark Simulation for Moist Nonhydrostatic Numerical Models.* Monthly Weather Review, 130, 2917-2928.

**Fix:** Add hydrometeor loading to the buoyancy term:
```
B = -g * (rho - rho0) / rho - g * (qc + qr + qi + qs + qg + qh)
```
This is a single-line change to the vertical momentum equation in each dynamics scheme (cartesian.cpp, supercell.cpp, tornado.cpp) and their corresponding GPU shaders.

**Location:** `src/dynamics/schemes/*/compute_momentum_tendencies()` and `compute_slow_tendencies()`

---

### 2. Coriolis Force and Static Environment

**What:** The Cartesian dynamics scheme does not include Earth's rotation. The Coriolis acceleration (`f x u`, where f = 2*Omega*sin(lat)) is absent from the horizontal momentum equations. The environmental sounding (rho0_base, p0_base, theta0) is fixed at initialization and never updated, so convection draws from an infinite CAPE reservoir.

**Why it matters:** Coriolis is essential for supercell mesocyclone formation. It provides storm-relative helicity, drives the rightward deviant motion of supercells, and enables storm splitting. Without it, the simulation cannot produce realistic supercell structure. The static environment means convection is physically unbounded.

These two simplifications are linked: once Coriolis is included, the Taylor-Proudman theorem makes a static horizontally uniform veering wind profile unphysical unless an external body force is added. The standard practice is to apply Coriolis only to wind perturbations after establishing an antitriptic balance base state.

**Literature:**
- Davies-Jones, R. (2021): *Invented Forces in Supercell Models.* Journal of the Atmospheric Sciences, 78, 2927-2939.
- Rotunno, R., and J. B. Klemp (1982): *The Influence of the Shear-Induced Pressure Gradient on Thunderstorm Motion.* Monthly Weather Review, 110, 136-151.

**Fix:** Add Coriolis terms to the horizontal momentum equations, applied to perturbation winds:
```
du/dt += f * (v - v0)
dv/dt += -f * (u - u0)
```
where `f = 2 * 7.292e-5 * sin(latitude)` and `u0, v0` are the base-state wind profile. Requires base-state wind storage and a configurable latitude parameter.

**Location:** `src/dynamics/schemes/cartesian/cartesian.cpp` (both unsplit and split-explicit paths)

---

---

## Numerical Gaps (Stability/Accuracy)

### 4. ~~GPU Saturation Adjustment Mismatch~~ RESOLVED

Verified April 20, 2026: Not a gap. The saturation adjustment runs on CPU before the GPU dispatch. The GPU shader receives already-adjusted qv/qc fields (`qv_adj.data()`, `qc_adj.data()`). Both CPU and GPU paths produce consistent results. The GPU shader also has the corrected saturation mixing ratio functions (pressure-dependent Buck equation, zero hardcoded values).

---

### 5. Weak Environmental Stratification and Laminar PBL

**What:** The initial theta profile is 302-315K over 16km (~0.8 K/km). A real troposphere has dtheta/dz of 3-5 K/km with a sharp tropopause inversion above 12km. The stratosphere should have dtheta/dz of 10-15 K/km. Additionally, the environment is initialized with laminar flow and a subgrid turbulence closure is applied to that laminar flow, which produces unrealistic near-surface shear once surface drag is activated.

**Why it matters:** The weak stratification produces excessive CAPE and provides no stable "lid" at the tropopause. The updraft encounters minimal resistance as it rises, leading to unrealistically rapid convective development. The Rayleigh sponge partially compensates but is a numerical fix, not a physical one. The laminar PBL treatment is known to produce unrealistic storm evolution.

**Literature:**
- Markowski, P. M., and G. H. Bryan (2016): *LES of Laminar Flow in the PBL: A Potential Problem for Convective Storm Simulations.* Monthly Weather Review, 144, 1841-1850.
- Nowotarski, C. J., P. M. Markowski, Y. P. Richardson, and G. H. Bryan (2014): *Properties of a Simulated Convective Boundary Layer in an Idealized Supercell Thunderstorm Environment.* Monthly Weather Review, 142, 3955-3976.

**Fix:** Improve the sounding initialization to produce a realistic theta profile with proper tropospheric lapse rate and tropopause inversion. This may require adjustments to the environmental profile construction in the config parser or sounding ingestion.

**Location:** `src/core/orchestration/dynamics/equations.cpp` (base state initialization) and `src/core/runtime/runtime_config.cpp` (YAML parsing for environment parameters)

---

### 6. Collocated Grid

**What:** All prognostic variables (u, v, w, rho, p, theta) are stored at cell centers (Arakawa A-grid). Pressure gradients and velocity divergence use the same 2-delta-x centered stencil.

**Why it matters:** The collocated arrangement admits a 2-delta-x computational mode (checkerboard pattern) that centered differences cannot see. This mode can grow and contaminate the solution. It also means the discrete pressure gradient does not exactly balance hydrostatic equilibrium, requiring explicit reference-state subtraction. At 2km resolution, the mode manifests as downdraft/updraft asymmetry (measured at 2.17:1 on the 2km grid, converging to 1.0:1 at 1km).

**Literature:**
- Arakawa, A., and V. R. Lamb (1977): *Computational Design of the Basic Dynamical Processes of the UCLA General Circulation Model.* Methods in Computational Physics, Vol. 17, Academic Press, 173-265.
- Randall, D. A. (1994): *Geostrophic Adjustment and the Finite-Difference Shallow-Water Equations.* Monthly Weather Review, 122, 1371-1377.
- Adcroft, A., C. Hill, and J. Marshall (1999): *A New Treatment of the Coriolis Terms in C-Grid Models at Both High and Low Resolutions.* Monthly Weather Review, 127, 1928-1936.

**Fix:** Long-term: migrate to an Arakawa C-grid (staggered) where u, v, w are at cell faces and scalars at cell centers. This eliminates the computational mode and improves pressure-velocity coupling. Short-term: 4th-order hyperdiffusion to selectively damp 2-delta-x waves without affecting resolved scales.

**Location:** Fundamental data structure change affecting `include/core/field3d.hpp`, all dynamics schemes, all advection schemes, and boundary conditions. Major refactor.

---

## Verified Non-Gaps

These initially appeared to be problems but were confirmed through testing to be either physically correct behavior or resolution-dependent artifacts.

- **Downdraft asymmetry at 2km grid** -- The downdraft/updraft ratio of 2.17 at dx=2000m converges to 1.0 at dx=1000m. This is a resolution artifact from the collocated grid, not a physics error.

- **Simulation blow-up at ~5 minutes** -- The convection converts CAPE (3000 J/kg) to kinetic energy (theoretical w_max = 77 m/s) with no precipitation loading to arrest growth. The blow-up is physically correct behavior for a model without dissipation mechanisms, not a numerical instability.

- **Radiation/BL "zero effect"** -- Both components are active and producing non-zero tendencies (verified via debug-level conservation budgets). Their contributions are physically small relative to dry dynamics at the student grid scale and 5-minute timescale.

- **Microphysics "no feedback"** -- Fixed April 20, 2026. Root causes were missing EOS density closure (theta changes did not update rho for buoyancy) and missing saturation adjustment in Kessler/Milbrandt (vapor never condensed). All 4 schemes now produce differentiated storm evolution.
