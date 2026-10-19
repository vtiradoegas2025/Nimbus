# Radiation

Radiation supplies heating from flux divergence using explicit atmospheric,
optical, surface and solar inputs. The runtime owns refresh cadence and held
source publication. Its temperature tendency is converted to the model's
potential-temperature convention before coupling. See
[Mathematics](../../docs/MATHEMATICS.pdf) for the assumptions and energy meaning.

`simple_grey` provides broadband idealizations. The pressure clear-sky profile
and absorbing water/cloud-path profile are distinct choices. The latter uses
resolved vapor/liquid/ice absorption paths and does not represent cloud
scattering. Its collimated solar beam and Lambertian reflected surface flux
must not be treated as the same angular distribution.

`rrtmg` is a native C++ implementation using explicit AER coefficients and
separate longwave/shortwave transfer. Its resolved-condensate cloud treatment
is not a subgrid cloud-fraction/overlap model. See
[the RRTMG data and implementation guide](schemes/rrtmg/README.md).

Terrain columns require actual interface heights and thicknesses. Slope-relative
illumination, terrain visibility, horizontal ray transport and angular transport
add separate geometry and discretization assumptions; they are not implied by
a working column solver. CPU, Metal and Vulkan support is exact-profile specific.
The source registry and strict runtime admission remain authoritative, including
unfinished broader terrain and output compositions.

`runtime.cpp` owns initialization, cadence and transactional publication;
`base/radiative_transfer.*` supplies common transfer operators; `schemes` owns
the selected closure. Invalid source or native dispatch must not leave a
partly updated radiation field, and RRTMG must not silently become simple-grey.

Permanent tests include `tests/physics/test_radiation.cpp` and
`test_radiation_runtime_state.cpp`, with independent reference data and coupled
fixtures. Keep coefficient identities, physical flux units, held-source work,
restart state and backend parity explicit. Passing a reference column is not a
claim of full coupled or campaign qualification.
