# Scientific Foundation — Nimbus

This document outlines the peer-reviewed literature foundations underlying the Nimbus atmospheric simulation framework. All implementations draw from established research in the American Meteorological Society (AMS) journals and related meteorological literature.

## Project Status & Scope

**Nimbus** is a research-grade atmospheric simulation framework implementing multiple physics parameterizations for supercell thunderstorm research. The codebase demonstrates professional software engineering practices while maintaining scientific accuracy in core atmospheric physics.

### Current Implementation Status

#### Complete & Functional
- **Core Dynamics**: Non-hydrostatic compressible equations with RK3 time integration
- **Advection Schemes**: TVD and WENO5 implementations for scalar transport
- **Diffusion**: Explicit and implicit Laplacian operators
- **Microphysics**: Kessler, Thompson, Lin, and Milbrandt schemes
- **Radiation**: Simple grey and RRTMG implementations
- **Boundary Layer**: YSU, MYNN, and slab PBL schemes with surface fluxes
- **Turbulence**: Smagorinsky and TKE closures
- **3D Visualization**: Complete OpenGL volume ray marching pipeline

#### Intentionally Simplified (Documented Gaps)
- **Chaos/Stochastic Module**: Initial-condition, boundary-layer, and full stochastic schemes are integrated; additional calibration/validation is still needed
- **Terrain Module**: Bell and Schär mountain implementations are integrated in runtime; broader physics validation against reference cases is still needed

#### **Grid Resolution**
- **Production**: 256×128×128 grid (1km × 1km × 100m) — appropriate for convection-permitting supercell simulations
- **Test Case**: 64×64×32 grid (2km × 2km × 500m) — for rapid development/testing

## Scientific Attribution by Component

### 1. Dynamics (supercells + tornado-scale, compressible/nonhydrostatic)

The compressible non-hydrostatic equations follow the formulation established in:
- **Klemp, J. B., and R. B. Wilhelmson (1978)**: *The Simulation of Three-Dimensional Convective Storm Dynamics.* **Journal of the Atmospheric Sciences**.
- **Rotunno, R., and J. B. Klemp (1985)**: *On the Rotation and Propagation of Simulated Supercell Thunderstorms.* **Journal of the Atmospheric Sciences**.

Model benchmarking and resolution requirements:
- **Bryan, G. H., and J. M. Fritsch (2002)**: *A Benchmark Simulation for Moist Nonhydrostatic Numerical Models.* **Monthly Weather Review**.
- **Bryan, G. H., J. C. Wyngaard, and J. M. Fritsch (2003)**: *Resolution Requirements for the Simulation of Deep Moist Convection.* **Monthly Weather Review**.

Time integration and damping layers:
- **Wicker, L. J., and W. C. Skamarock (2002)**: *Time-Splitting Methods for Elastic Models Using Forward Time Schemes.* **Monthly Weather Review**.
- **Klemp, J. B., J. Dudhia, and A. D. Hassiotis (2008)**: *An Upper Gravity-Wave Absorbing Layer for NWP Applications.* **Monthly Weather Review**.

Tornado-scale dynamics:
- **Lee, B. D., and R. B. Wilhelmson (1997)**: *The Numerical Simulation of Nonsupercell Tornadogenesis.* **Journal of the Atmospheric Sciences**.

### 2. Microphysics (bulk schemes)

#### Kessler (warm-rain)
- **Kessler, E. (1969)**: *On the Distribution and Continuity of Water Substance in Atmospheric Circulations.* **Meteorological Monographs (AMS)**.

#### Lin-type bulk ice
- **Lin, Y.-L., R. D. Farley, and H. D. Orville (1983)**: *Bulk Parameterization of the Snow Field in a Cloud Model.* **Journal of Climate and Applied Meteorology**.

#### Thompson (mixed-phase bulk)
- **Thompson, G., P. R. Field, R. M. Rasmussen, and W. D. Hall (2008)**: *Explicit Forecasts of Winter Precipitation Using an Improved Bulk Microphysics Scheme. Part II: Implementation of a New Snow Parameterization.* **Monthly Weather Review**.
- *(Optional extension / aerosols)* **Thompson, G., and T. Eidhammer (2014)**: papers describing aerosol-aware/extended Thompson implementations in **AMS journals** (e.g., **Weather and Forecasting**) depending on the exact feature set you implement.

#### Milbrandt–Yau (multi-moment)
- **Milbrandt, J. A., and M. K. Yau (2005)**: *A Multimoment Bulk Microphysics Parameterization. Part I: Analysis of the Role of the Spectral Shape Parameter.* **Journal of the Atmospheric Sciences**.
- **Milbrandt, J. A., and M. K. Yau (2005)**: *A Multimoment Bulk Microphysics Parameterization. Part II: A Proposed Three-Moment Closure and Scheme Description.* **Journal of the Atmospheric Sciences**.

### 3. Radiation

#### Simple/gray radiation foundations (idealized + "cheap" RT)
- **Frierson, D. M. W. (2006)**: *A Gray-Radiation Aquaplanet Moist GCM. Part I: Static Stability and Eddy Scale.* **Journal of the Atmospheric Sciences**.
- **Frierson, D. M. W. (2007)**: *A Gray-Radiation Aquaplanet Moist GCM. Part II: Energy Transports in Altered Climates.* **Journal of the Atmospheric Sciences**.
- **Jeevanjee, N., and colleagues (2020)**: *Simple spectral/idealized radiative-cooling model foundations (clear-sky simplifications).* **Journal of the Atmospheric Sciences**.

#### RRTMG (implementation context inside AMS literature)
- **Powers, J. G., et al. (2017)**: *The Weather Research and Forecasting (WRF) Model.* **Bulletin of the American Meteorological Society**.
  *(Use this as AMS context/entry point for how RRTMG is used operationally in WRF-like modeling stacks.)*

### 4. Boundary layer + surface fluxes

#### Slab / first-order closures + basic PBL structure
- **Deardorff, J. W. (1972)**: *Parameterization of the Planetary Boundary Layer for Use in General Circulation Models.* **Monthly Weather Review**.

#### Surface-layer similarity (MOST) for flux parameterization
- **Businger, J. A., J. C. Wyngaard, Y. Izumi, and E. F. Bradley (1971)**: *Flux-Profile Relationships in the Atmospheric Surface Layer.* **Journal of the Atmospheric Sciences**.

#### WRF-like surface-layer practical formulation
- **Jiménez, P. A., J. Dudhia, J. F. González-Rouco, J. Navarro, J. P. Montávez, and E. García-Bustamante (2012)**: *A Revised Scheme for the WRF Surface Layer Formulation.* **Monthly Weather Review**.

#### YSU PBL
- **Hong, S.-Y., Y. Noh, and J. Dudhia (2006)**: *A New Vertical Diffusion Package with an Explicit Treatment of Entrainment Processes.* **Monthly Weather Review**.

#### MYNN / other higher-order closures
- MYNN's original technical description is often *not* an AMS-journal paper; if you implement MYNN, cite the original primary reference (see "Non-AMS foundational" section below if needed), plus any AMS evaluation papers you specifically rely on.

### 5. Turbulence / sub-grid closures (LES-style)

- **Smagorinsky, J. (1963)**: *General Circulation Experiments with the Primitive Equations. I. The Basic Experiment.* **Monthly Weather Review**.
- **Moeng, C.-H. (1984)**: *A Large-Eddy-Simulation Model for the Study of Planetary Boundary-Layer Turbulence.* **Journal of the Atmospheric Sciences**.
- **Basu, S., and F. Porté-Agel (2006)**: *Large-Eddy Simulation of Stable Boundary Layer Turbulence: A Scale-Dependent Dynamic Modeling Approach.* **Journal of the Atmospheric Sciences**.

### 6. Numerics

#### Advection (TVD / WENO options)
- **Thuburn, J. (1997)**: *A PV-Based Shallow-Water Model on a Hexagonal-Icosahedral Grid.* **Monthly Weather Review**.
  *(Often cited for limiter/monotonicity ideas in conservative transport; cite if you use Thuburn-style limiter logic.)*
- **Katta, V. R., and S. Akella (2015)**: *A Weighted Essentially Nonoscillatory Finite-Volume Method for Atmospheric Transport (application-focused).* **Monthly Weather Review**.
- **Lunet, T., et al. (2017)**: *High-Order WENO + explicit Runge–Kutta integration approaches in an atmospheric context.* **Monthly Weather Review**.
- **Wang, A., et al. (2021)**: *Influence of WENO-style high-order transport on LES of deep convection / CM1-like modeling.* **Journal of the Atmospheric Sciences**.

#### Diffusion (explicit/implicit)
- If your diffusion modules implement standard Laplacian / biharmonic forms and implicit solves, attribute the specific AMS paper(s) you follow for stability/consistency over terrain (often overlaps with terrain/coordinate literature below).

#### Time stepping (RK + split/HEVI options)
- **Wicker, L. J., and W. C. Skamarock (2002)**: *Time-Splitting Methods for Elastic Models Using Forward Time Schemes.* **Monthly Weather Review**.
- **Bao, L., et al. (2015)**: *Horizontally Explicit and Vertically Implicit (HEVI) Time Discretization for Nonhydrostatic Models.* **Monthly Weather Review**.

### 7. Terrain / topography (idealized bell + Schär-type mountains; coordinate treatment)

- **Schär, C., D. Leuenberger, O. Fuhrer, D. Lüthi, and C. Girard (2002)**: *A New Terrain-Following Vertical Coordinate Formulation for Atmospheric Prediction Models.* **Monthly Weather Review**.
- **Klemp, J. B. (2011)**: *A Terrain-Following Coordinate with Smoothed Coordinate Surfaces.* **Monthly Weather Review**.
- **Weller, H. (2014)**: *Curl-Free Pressure Gradients over Orography in a Solution of the Compressible Euler Equations.* **Monthly Weather Review**.

### 8. Diagnostics

#### Storm-scale severity proxies (supercells)
- **Kain, J. S., et al. (2008)**: *A Unified Approach to Forecasting and Verifying Severe Convection (including updraft helicity usage/interpretation in convection-allowing guidance).* **Weather and Forecasting**.

#### Helicity / SRH formulations
- **Thompson, R. L., R. Edwards, J. A. Hart, K. L. Elmore, and P. Markowski (2007)**: *Close Proximity Soundings within Supercell Environments (effective inflow layer / effective SRH concepts).* **Weather and Forecasting**.
- **Davies-Jones, R. (2009)**: *On the Rotation of Suction Vortices and the Interpretation/Use of Helicity Measures (SRH/helicity foundations and caveats).* **Monthly Weather Review**.

#### Thermodynamics (for CAPE/CIN, parcel computations)
- **Bolton, D. (1980)**: *The Computation of Equivalent Potential Temperature.* **Monthly Weather Review**.
- **Doswell, C. A., and E. N. Rasmussen (1994)**: *The Effect of Neglecting the Virtual Temperature Correction on CAPE Calculations.* **Weather and Forecasting**.
- **Romps, D. M. (2017)**: *Exact Expressions for LCL/LFC/EL-related thermodynamic levels (implementation-grade formulas).* **Journal of the Atmospheric Sciences**.

### 9. Chaos / stochastic perturbations (IC + BL + full stochastic)

- **Berner, J., et al. (2017)**: *Stochastic Parameterization: Toward a New View of Weather and Climate Models.* **Bulletin of the American Meteorological Society**.
- **Berner, J., et al. (2015)**: *Stochastically Perturbed Parameterization Tendencies (SPPT): formulation, impacts, and practical guidance.* **Monthly Weather Review**.
- **Teixeira, J., and C. A. Reynolds (2008)**: *Stochastic nature of model physics / parameterization uncertainty approaches in NWP.* **Monthly Weather Review**.
- **Bouttier, F., et al. (2012)**: *Stochastic kinetic-energy backscatter / model-error forcing concepts for ensemble prediction.* **Monthly Weather Review**.
- **Jankov, I., et al. (2019)**: *Practical stochastic perturbation methods (including SKEB/SPPT-like impacts) in convection-allowing ensemble contexts.* **Monthly Weather Review**.

### 10. Radar (observation operators and forward simulators)

#### Radar observation operators in storm-scale DA (Vr/Z context)
- **Snyder, C., & Zhang, F. (2003)**: *Assimilation of Simulated Doppler Radar Observations with an Ensemble Kalman Filter.* **Monthly Weather Review**.
- **Tong, M., & Xue, M. (2005)**: *Ensemble Kalman Filter Assimilation of Doppler Radar Data with a Compressible Nonhydrostatic Model: OSS Experiments.* **Monthly Weather Review**.
- **Dowell, D. C., et al. (2011)**: *EnKF Assimilation of Radar Observations of the 8 May 2003 Oklahoma City Supercell.* **Monthly Weather Review**.

#### Radar-sampling/beam-volume effects (why you want a "sampler" hook)
- **Thompson, T. E., et al. (2012)**: *Impact from a Volumetric Radar-Sampling Operator for Radial Velocity Observations on EnKF Analyses and Forecasts.* **Journal of Atmospheric and Oceanic Technology**.

#### Improved Vr operator variants (optional, advanced)
- **Chen, F., et al. (2017)**: *Application of an IVAP-Based Observation Operator in Radar Radial Velocity Data Assimilation.* **Monthly Weather Review**.

#### Radar QC concepts (useful if you later add operator noise/QC)
- **Friedrich, K., et al. (2006)**: *A Quality Control Concept for Radar Reflectivity, Polarimetric Parameters, and Doppler Velocity.* **Journal of Atmospheric and Oceanic Technology**.

#### Reflectivity Z / Ze forward operators
- **Liu, C., et al. (2022)**: *Use of a Reflectivity Operator Based on Double-Moment Microphysics for Radar Reflectivity Data Assimilation.* **Monthly Weather Review**.
- **Li, H., et al. (2022)**: *Use of Power Transform Total Number Concentration as Control Variables in Direct Assimilation of Reflectivity with Double-Moment Microphysics.* **Monthly Weather Review**.
- **Liu, C., et al. (2019)**: *Direct Variational Assimilation of Radar Reflectivity and Radial Velocity.* **Monthly Weather Review**.
- **Liu, C., et al. (2020)**: *Direct Variational Assimilation of Radar Reflectivity and Radial Velocity: Further Operator/Constraint Treatments.* **Monthly Weather Review**.
- **Steiner, M., et al. (2004)**: *A Microphysical Interpretation of Radar Reflectivity–Rain Rate Relationships.* **Journal of the Atmospheric Sciences**.

#### Doppler radial velocity Vr operators
- **Snyder, C., & Zhang, F. (2003)**: *Assimilation of Simulated Doppler Radar Observations with an EnKF.* **Monthly Weather Review**.
- **Tong, M., & Xue, M. (2005)**: *EnKF Assimilation of Doppler Radar Data…* **Monthly Weather Review**.
- **Dowell, D. C., et al. (2011)**: *EnKF Assimilation of Radar Observations of the OKC Supercell.* **Monthly Weather Review**.
- **Thompson, T. E., et al. (2012)**: *Volumetric Radar-Sampling Operator…* **Journal of Atmospheric and Oceanic Technology**.
- **Chen, F., et al. (2017)**: *IVAP-Based Observation Operator…* **Monthly Weather Review**.

#### Dual-pol variables (ZH, ZV, ZDR) forward operators / simulators
- **Lang, T. J., et al. (2004)**: *Observations of Quasi-Symmetric Echo Patterns…* **Journal of Atmospheric and Oceanic Technology**.
- **Holt, A. R. (1991)**: *Theoretical Study of Radar Polarization Parameters…* **Journal of the Atmospheric Sciences**.
- **Jung, Y., et al. (2008)**: *Assimilation of Simulated Polarimetric Radar Data for a Convective Storm Using EnKF.* **Monthly Weather Review**.
- **Jung, Y., et al. (2010)**: *Simulations of Polarimetric Radar Signatures of a Supercell Storm Using a Two-Moment Bulk Microphysics Scheme.* **Journal of Applied Meteorology and Climatology**.
- **Pfeifer, M., et al. (2008)**: *A Polarimetric Radar Forward Operator for Model Evaluation…* **Journal of Applied Meteorology and Climatology**.
- **Ryzhkov, A., et al. (2011)**: *Polarimetric Radar Observation Operator for a Cloud Model…* **Journal of Applied Meteorology and Climatology**.
- **Kumjian, M. R., et al. (2019)**: *A Moment-Based Polarimetric Radar Forward Operator…* **Journal of Applied Meteorology and Climatology**.
- **Putnam, B. J., et al. (2017)**: *Simulation of Polarimetric Radar Variables from Storm-Scale Ensemble Forecasts.* **Monthly Weather Review**.
- **Johnson, M., et al. (2016)**: *Comparison of Simulated Polarimetric Signatures in Idealized Supercells…* **Monthly Weather Review**.
- **Snyder, J. C., et al. (2013)**: *Observations of Polarimetric Signatures in Supercells…* **Monthly Weather Review**.
- **Posselt, D. J., et al. (2015)**: *Assimilation of Dual-Polarization Radar Observations in Mixed-Phase Convective Storms.* **Monthly Weather Review**.
- **Putnam, B. J., et al. (2019)**: *Ensemble Kalman Filter Assimilation of Polarimetric Radar Observations…* **Monthly Weather Review**.
- **Putnam, B. J., et al. (2021)**: *The Impact of Assimilating ZDR Observations on Storm-Scale Analyses…* **Monthly Weather Review**.


## Non-AMS Foundational References

These are *not* AMS-journal originals, but are often the primary "scheme papers" people expect to be cited if you implement the named algorithms:

- **Iacono, M. J., et al. (2008)**: *RRTMG description paper (shortwave/longwave for GCMs).* **Journal of Geophysical Research (Atmospheres)**.
- **Jiang, G.-S., and C.-W. Shu (1996)** (and related Shu/Osher papers): *WENO scheme foundations.* **J. Comput. Phys.** (and related applied-math venues).
- **Nakanishi, M., and H. Niino (2009)**: *MYNN PBL scheme foundations.* **J. Meteor. Soc. Japan**.

## Development Notes

### Architecture & Design
The codebase follows modern C++ practices with a modular factory pattern for physics schemes, enabling easy extension and comparison of different parameterizations. The design prioritizes:
- **Physical accuracy** in core atmospheric equations
- **Numerical stability** through appropriate CFL constraints and limiters
- **Computational efficiency** for large-scale 3D simulations
- **Extensibility** for future physics additions

### Intended Use Cases
- **Research**: Supercell thunderstorm simulation and analysis
- **Education**: Demonstrating atmospheric modeling concepts
- **Development**: Testing new physics parameterizations
- **Visualization**: 3D rendering of convective storm structures

### Future Development Priorities
1. Expand chaos/stochastic calibration and ensemble validation workflows
2. Extend terrain-science validation against reference idealized and real-case benchmarks
3. Add more diagnostic outputs (helicity, vorticity, etc.)
4. Performance optimization for larger domains
5. Ensemble capability for uncertainty quantification

---

*This document serves as comprehensive attribution for the scientific foundations of Nimbus. For practical usage instructions, see the main [README.md](../README.md).*"
