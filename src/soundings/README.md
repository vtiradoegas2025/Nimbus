# Sounding readers

`src/init/sounding` selects the model-column source; this module reads and
interpolates file-backed profiles. Public data and runtime interfaces live in
`include/soundings`. A sounding is a scientific initial condition, not merely
a plotting input.

The retained readers handle native CM1 text and SHARPY-related text/netCDF/HDF5
formats. Automatic routing uses content and file signatures. Validation checks
finite values, ordering, required variables, physical ranges and model-top
coverage. Cartesian wind components are interpolated directly to avoid angular
wrap artifacts. CM1 theta and dry-air vapor mixing ratio retain their declared
meaning through the conversion.

Strict-v1 file initialization uses the automatic reader, linear interpolation,
no extrapolation and no silent procedural fallback. Lower-level compatibility
controls must not be mistaken for ordinary strict-v1 behavior. The parametric
sounding branches have separate requested-versus-realized diagnostics; selecting
a CAPE target does not guarantee that every other environmental diagnostic is
independently prescribed.

`base` contains shared quality-control/interpolation algorithms, `schemes`
contains readers, and `factory.cpp`/`soundings.cpp` own construction and the API.
The maintained sounding/init tests and their small profile fixtures are required
source inputs. User observations and large local storm experiments belong
outside the source tree.

See [Mathematics](../../docs/MATHEMATICS.pdf) for units, reference-state
assumptions, physical height,
Earth-relative wind and model coverage.
