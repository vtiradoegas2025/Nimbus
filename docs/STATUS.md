# Nimbus -- Project Status

**Last Updated:** April 22, 2026

---

## Current State

### Working

- **Core Simulation Engine** -- Compressible non-hydrostatic Euler equations in both cylindrical (r, theta, z) and Cartesian (x, y, z) coordinates. Split-explicit time stepping (Klemp-Wilhelmson 1978), RK3/RK4 with CFL-adaptive stepping. Field3D contiguous storage with OpenMP parallelization.
- **Physics Modules** -- All wired and testable through factory architecture:
  - Microphysics: Kessler, Thompson, Lin, Milbrandt-Yau (warm-rain through double-moment)
  - Boundary Layer: YSU, MYNN, Slab (with bulk and Monin-Obukhov surface fluxes)
  - Turbulence: Smagorinsky-Lilly, TKE prognostic
  - Radiation: simple_grey (broadband two-stream longwave, Beer-Lambert shortwave)
  - Dynamics: tornado (axisymmetric cylindrical), supercell (full 3D cylindrical), cartesian (x, y, z)
  - Radar: reflectivity, Doppler velocity, Z_DR forward operators
  - Terrain: bell mountain, Schar
  - Chaos: IC perturbation, BL stochastic, full stochastic (SPPT-style)
  - Soundings: SHARPY (native NetCDF classic, NetCDF C API, HDF5 readers)
- **Numerics** -- TVD/WENO5 advection with coordinate-specific kernels, explicit/implicit diffusion, split-explicit/RK3/RK4 time stepping
- **GPU Compute** -- Vulkan compute backend with 15 shaders covering advection, acoustic substeps (cylindrical and Cartesian), microphysics, dynamics tendencies, and diffusion. Fused and batched acoustic substep dispatch. Coordinate-aware shader routing.
- **Vulkan Viewer** -- Native volume ray marcher with NPY field ingestion, live shared-memory ingest from running simulations, orbital and freefly cameras, cinematic rendering styles
- **Field Contract** -- CM1-style validation with 99 defined fields, 87 exported, 20/20 required-now covered
- **Test Suite** -- 26 test binaries, 42,000+ assertions across 9 test categories:
  - `make test` (full suite), `make test-core`, `make test-diagnostics`, `make test-dynamics`, `make test-numerics`, `make test-physics`, `make test-data`, `make test-vulkan`, `make test-integration`, `make test-shm-e2e`
- **Cross-Platform Build** -- Automated setup script (`scripts/setup.sh`) for macOS, Ubuntu/Debian/Pop!_OS, Fedora, Arch. Auto-detecting compiler, `make check-deps` target, BUILDING.md with per-OS instructions.

### Known Limitations

- **Collocated grid** -- Arakawa A-grid admits 2-delta-x computational mode. C-grid staggering planned.
- **Radiation fidelity** -- Runtime is simple_grey only; RRTMG recognized as a planned target.

### Incomplete

- **Diagnostic breadth**: 12 CM1-style contract fields remain unimplemented (streamlines, q-vectors, trajectory diagnostics)
- **Radiation fidelity**: Runtime is simple_grey only; RRTMG recognized as a planned target
- **Science validation**: Terrain and chaos are runtime-integrated but lack broader case-based calibration

---

## Recent Changes (April 2026)

- Cartesian coordinate system: full dynamics scheme, BCs, ICs, GPU acoustic shaders
- Split-explicit time stepping with GPU-accelerated acoustic substeps
- Fused and batched GPU dispatch: single H2D/D2H for all acoustic substeps per timestep
- Shader compilation optimization (-Os flag on all 15 compute shaders)
- Removed redundant GPU buffer copies in advection batch dispatch
- Cross-platform build system: setup.sh, check-deps, BUILDING.md
- Documentation overhaul: all src/ READMEs updated with correct paths and architecture guide

---

## Roadmap

1. **C-grid staggering** -- Arakawa C-grid migration for improved pressure-velocity coupling
2. **Science validation** -- Terrain and chaos calibration against published reference cases
3. **Diagnostic breadth** -- Remaining 12 contract fields (streamlines, q-vectors, trajectories)
4. **Radiation fidelity** -- RRTMG implementation

---

## References

- Build instructions: [BUILDING.md](../BUILDING.md)
- Source architecture: [src/README.md](../src/README.md)
- Visualization: [vulkan/README.md](../vulkan/README.md)
- Technical reference: [docs/README.md](README.md)
- Scientific foundations: [foundationalScience.md](foundationalScience.md)
