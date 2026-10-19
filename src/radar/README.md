# Radar forward operators

Radar products are diagnostic transformations of model state. They are not
additional prognostic water moments or direct observations. `radar.products`
selects a complete product set, and `RadarSystem` stages all selected outputs
before publication. A late failure must preserve the previous product set.

Reflectivity depends on the selected particle-distribution and dielectric
assumptions. Radial velocity projects reconstructed physical wind onto the
instrument line of sight and, where selected, corrects for hydrometeor terminal
fall speed. The instrument is Earth-referenced, so moving domains must update
coordinates and invalidate stale generations.

The retained rain-only polarimetric operator has explicit species, diameter,
unit and output bounds. It must reject unsupported frozen material rather than
quietly ignoring it. Legacy reflectivity or axis-ratio approximations are not
interchangeable with that operator. Product output bounds are part of the
scientific contract and do not establish observational accuracy.

`system.cpp` and `runtime.cpp` own product publication/state;
`schemes` contains the forward operators. Current exact-profile fixtures and
gates include `cgrid_radar_ready.yaml`, `cgrid_radar_polarimetric_ready.yaml`,
`test_radar_ready.py` and `test_radar_polarimetric_ready.py` under the maintained
test trees. Consult the compiled catalog for supported combinations. Bounded
reference/continuation checks do not establish full qualification.

See [Mathematics](../../docs/MATHEMATICS.pdf) for moment and unit conventions
and physical coordinates.
