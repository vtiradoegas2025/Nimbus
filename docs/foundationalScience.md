# Scientific references and implementation provenance

[Mathematics](MATHEMATICS.pdf) explains Nimbus's actual equations, assumptions
and numerical representation. The references below supply scientific context;
they do not imply that Nimbus implements every part of a publication or shares
all of its discretization and thermodynamic choices. Current source, independent
reference tests and configuration admission determine the implemented scope.
Nimbus has been an independent codebase from its beginning, not a CM1 port or
fork. A shared scientific reference or CM1-compatible input format does not
establish a shared solver implementation.

## Dynamics and unresolved motion

[Klemp and Wilhelmson (1978), *The Simulation of Three-Dimensional Convective
Storm Dynamics*](https://journals.ametsoc.org/abstract/journals/atsc/35/6/1520-0469_1978_035_1070_tsotdc_2_0_co_2.xml),
Journal of the Atmospheric Sciences, 35, 1070–1096, supplies the compressible
storm-model and acoustic-splitting background. Nimbus's dry rho–theta EOS,
moisture-loading approximation, stage ordering and energy attribution must be
read in the current dynamics code rather than inferred from that citation.

[Moeng (1984), *A Large-Eddy-Simulation Model for the Study of Planetary
Boundary-Layer Turbulence*](https://journals.ametsoc.org/view/journals/atsc/41/13/1520-0469_1984_041_2052_alesmf_2_0_co_2.xml),
Journal of the Atmospheric Sciences, 41, 2052–2062, provides LES context.
Nimbus's discrete stress, TKE, averaging, clipping and boundary-work contracts
are specified in `src/turbulence` and its permanent tests; Nimbus does not
thereby inherit the paper's complete spatial discretization.

[Hong, Noh and Dudhia (2006), *A New Vertical Diffusion Package with an Explicit
Treatment of Entrainment Processes*](https://journals.ametsoc.org/abstract/journals/mwre/134/9/mwr3199.1.xml),
Monthly Weather Review, 134, 2318–2341, describes the YSU nonlocal diffusion and
explicit entrainment formulation. The source under `src/boundary_layer` defines
the retained column equations, surface treatment and admitted Nimbus couplings.

## Resolution and experiment design

[Skamarock (2004), *Evaluating Mesoscale NWP Models Using Kinetic Energy
Spectra*](https://doi.org/10.1175/MWR2830.1), Monthly Weather Review, 132,
3019–3032, provides context for assessing effective resolution through resolved
spectra. Its findings do not determine Nimbus's effective resolution without
corresponding tests of Nimbus's numerical operators and configuration.

[Bryan, Wyngaard and Fritsch (2003), *Resolution Requirements for the Simulation
of Deep Moist Convection*](https://journals.ametsoc.org/view/journals/mwre/131/10/1520-0493_2003_131_2394_rrftso_2.0.co_2.xml),
Monthly Weather Review, 131, 2394–2416, relates resolution to turbulence-closure
assumptions. The mathematics guide uses this context to explain refinement and
closure sensitivity, not to prescribe a universal spacing for Nimbus.

[NASA/NPARC, *Examining Spatial (Grid) Convergence*](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html)
describes conditional convergence and extrapolation estimates. The new
resolution chapter states the smoothness, asymptotic-error and matched-problem
assumptions before applying those estimates, and distinguishes them from
statistics of long nonlinear convective simulations.

## Bulk microphysics

[Lin, Farley and Orville (1983), *Bulk Parameterization of the Snow Field in a
Cloud Model*](https://www.atmos.albany.edu/facstaff/rfovell/ATM562/lin-etal-1983.pdf),
Journal of Climate and Applied Meteorology, 22, 1065–1092, is a primary
mixed-phase bulk reference. Its wet-growth treatment and Nimbus's newer guarded
material-temperature/vapor-mass extensions are explicitly distinguished in the
mathematics guide.

[Milbrandt and Yau (2005), *A Multimoment Bulk Microphysics Parameterization.
Part II: A Proposed Three-Moment Closure and Scheme Description*](https://www2.mmm.ucar.edu/wrf/users/physics/phys_refs/MICRO_PHYS/Milbrandt_Yau_part2.pdf),
Journal of the Atmospheric Sciences, 62, 3065–3081, provides the spectral moment
and process-closure background. Public triple-moment admission in Nimbus remains
Engineering incomplete; a paper and an internal moment test do not complete its
live source, energy, transport, restart and native-backend couplings.

[Thompson, Field, Rasmussen and Hall (2008), *Explicit Forecasts of Winter
Precipitation Using an Improved Bulk Microphysics Scheme. Part II:
Implementation of a New Snow Parameterization*](https://journals.ametsoc.org/abstract/journals/mwre/136/12/2008mwr2387.1.xml),
Monthly Weather Review, 136, 5095–5115, describes the snow parameterization
background, including its distribution and particle-shape assumptions.
Nimbus's pinned Thompson implementation and offline table identities are
specified by [its maintained data guide](../src/microphysics/schemes/thompson/README.md).
A scheme name alone is insufficient to identify a reproducible implementation.

## Coefficient data and numerical contracts

RRTMG coefficient files are scientific data supplied by the AER model family;
the native Nimbus implementation is C++, not a runtime call to reference
Fortran. Keep coefficient identity and the official reference-column fixtures
with the implementation. See [the RRTMG guide](../src/radiation/schemes/rrtmg/README.md)
and [radiation tests](../tests/physics/test_radiation.cpp).

For numerical changes, the permanent tests are the executable reference index:
[advection](../tests/numerics/test_advection_cartesian.cpp),
[diffusion](../tests/numerics/test_diffusion.cpp),
[time stepping](../tests/numerics/test_time_stepping.cpp),
[EOS](../tests/core/test_eos_pressure.cpp), and
[energy inventories](../tests/diagnostics/test_energy_budget.cpp).
These checks supplement physical references rather than replacing convergence,
observational validation or independent scientific review.

When adding a reference, identify the actual equation or assumption used and
its Nimbus source/test owner. Avoid vague citations to an author's body of work,
unsupported claims of identical schemes, or treating an archived campaign as a
current runtime dependency. The repository readiness contract distinguishes
Engineering incomplete, Ready, Campaign validated, Externally reviewed and
Fully qualified; bibliography entries do not change those states.
