# Native RRTMG

Nimbus implements RRTMG in C++. The AER netCDF files are coefficient data;
ordinary execution does not compile or call the reference Fortran solver.
Required default inputs are `data/rrtmg/rrtmg_lw.nc` and
`data/rrtmg/rrtmg_sw.nc` for the selected spectral components. Explicit
`NIMBUS_RRTMG_LW_COEFFICIENTS` and `NIMBUS_RRTMG_SW_COEFFICIENTS` overrides change
those scientific inputs and must retain the loader's schema/identity contract.

`rrtmg_coefficients.*` owns typed loading; `rrtmg_atmosphere.*` validates column
state and units; preprocessing owns pressure/temperature/reference-grid
interpolation. The optics files implement reduced spectral quadrature and
molecular/continuum contributions. `rrtmg_cloud_optics.*` and checked-in generated
tables supply the resolved liquid/ice optics. Longwave and shortwave transfer
have separate solvers, assembled by `rrtmg.*` and the radiation runtime.

The resolved-condensate treatment assumes a fully cloudy cell rather than a
subgrid cloud-fraction or overlap model. Explicit gas composition, solar
geometry, surface state and physical interfaces are part of the scientific
input. Invalid state or unavailable required native execution must reject;
RRTMG must not silently become simple-grey.

Permanent reference-column, flux, runtime-state, continuation and backend tests
live under `tests/physics` and the maintained qualification fixtures. Current
ordinary admission is profile-specific and recorded by the compiled catalog;
it is not accurately described by the historical claim that RRTMG is available
only in exploratory mode. Column agreement alone does not establish every
terrain/cadence/physics composition or full campaign qualification.

See [radiation](../../README.md),
[Mathematics](../../../../docs/MATHEMATICS.pdf) and
[scientific references](../../../../docs/foundationalScience.md).
