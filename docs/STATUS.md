# Nimbus — Project Status

**Last Updated:** March 22, 2026

---

## Current State

### Working

- **Core Simulation Engine** — Compressible Euler equations in cylindrical coordinates, RK3/RK4 time integration with CFL-adaptive stepping, Field3D contiguous storage, OpenMP parallelization
- **Physics Modules** — All wired and testable through factory architecture:
  - Microphysics: Kessler, Thompson, Lin, Milbrandt-Yau
  - Boundary Layer: YSU, MYNN, Slab
  - Turbulence: Smagorinsky-Lilly, TKE prognostic
  - Radiation: `simple_grey` (RRTMG recognized as planned target)
  - Dynamics: tornado (axisymmetric), supercell (full 3D)
  - Radar: reflectivity, Doppler velocity, Z_DR forward operators
  - Terrain: bell mountain, Schar
  - Chaos: IC perturbation, BL stochastic, full stochastic
  - Soundings: SHARPY (NetCDF classic, NetCDF C API, HDF5 native readers)
- **Numerics** — TVD/WENO5 advection, explicit/implicit diffusion, RK3/RK4 time stepping
- **Vulkan Viewer** — Native volume renderer with NPY field ingestion, orbital camera, cylindrical-to-Cartesian transform
- **Field Contract** — CM1-style validation with 99 defined fields, 87 exported, 20/20 required-now covered
- **Test Suite** — `make test` (aggregate), `make test-backend-physics`, `make test-soundings`, `make test-radiation-regression`, `make test-terrain-regression`

### Incomplete

- **Diagnostic breadth**: 12 CM1-style contract fields remain unimplemented (streamlines, q-vectors, cross-section/trajectory diagnostics)
- **Radiation fidelity**: Runtime is `simple_grey` only; no in-tree RRTMG implementation yet
- **Science validation**: Terrain and chaos are runtime-integrated but lack broader case-based calibration
- **Soundings validation**: Native ingestion pipeline works; broader science validation pending

---

## Known Limitations

- Terrain module is integrated but not calibrated against reference orographic cases
- Chaos/ensemble workflows need broader calibration and case-based validation
- Radar beam physics and advanced scatterer microphysics remain simplified
- APIs and file formats may change (research prototype)

---

## Roadmap

1. **Stability & verification** — Extend guard coverage, add targeted regression tests for tendency bounds across more physics modules
2. **Science validation** — Terrain and chaos calibration against published reference cases
3. **Diagnostic breadth** — Implement remaining 12 contract fields (streamlines, q-vectors, trajectory diagnostics)
4. **Radiation fidelity** — RRTMG implementation
5. **GPU compute** — Vulkan compute kernel offload for advection/diffusion hot paths

---

## References

- Visualization details: [`vulkan/README.md`](../vulkan/README.md)
- Technical reference: [`docs/README.md`](README.md)
- Scientific foundations: [`docs/foundationalScience.md`](foundationalScience.md)
