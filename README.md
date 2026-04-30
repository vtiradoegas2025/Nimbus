# Nimbus

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Tests](https://img.shields.io/badge/Tests-passing-brightgreen.svg)](#testing--validation)

**A modern approach to supercell modeling.** High-performance atmospheric simulation built in **C++17** with a modular physics architecture, **GPU-accelerated compute via Vulkan**, and a **native 3D volume renderer** for real-time visualization.

> **Status:** Active development -- research prototype. See **[docs/STATUS.md](docs/STATUS.md)** for the latest audit.

---

## Motivation

Nimbus implements a **CM1-lite style research model** designed to run on personal hardware (laptops to workstations) while maintaining a modular, testable physics architecture. It is not a replacement for CM1 -- the goal is a pragmatic subset with strong developer ergonomics, strict field validation, and reproducible workflows.

**Why C++ over Fortran:**
- Explicit memory-layout control for large 3D fields (contiguous `Field3D` storage)
- Factory-based scheme architecture for rapid physics experimentation
- Direct integration with Vulkan for GPU compute and native rendering
- Modern tooling ecosystem (profiling, testing, CI)

---

## Architecture

### Governing Equations

Compressible, non-hydrostatic Euler equations in either cylindrical (r, theta, z) or Cartesian (x, y, z) coordinates:

```
du/dt + (u . nabla)u + (1/rho_0) nabla(p') + g(theta'/theta_0) k = -nabla . tau + F_buoyancy
dtheta/dt + nabla . (u theta) = Q_radiation + Q_microphysics + nabla . (K_theta nabla(theta))
dq_v/dt + nabla . (u q_v) = -C - E + nabla . (K_q nabla(q_v))
```

### Physics Modules

Every physics module follows a factory pattern -- schemes are selected by name in YAML config and swapped without touching code. See [src/README.md](src/README.md) for the full architecture guide.

| Module | Schemes | Notes |
|--------|---------|-------|
| **Microphysics** | Kessler, Thompson, Lin, Milbrandt-Yau | Warm-rain through double-moment |
| **Boundary Layer** | YSU, MYNN, Slab | Non-local and TKE-based closures |
| **Turbulence** | Smagorinsky-Lilly, TKE prognostic | Eddy-viscosity and prognostic closures |
| **Radiation** | Simple grey | Longwave/shortwave transfer |
| **Dynamics** | Tornado, Supercell, Cartesian | Axisymmetric, full 3D cylindrical, and Cartesian |
| **Radar** | Reflectivity, Doppler velocity, Z_DR | Forward operators for synthetic obs |
| **Terrain** | Bell mountain, Schar | Idealized orographic forcing |
| **Chaos** | IC perturbation, BL stochastic, Full | Ensemble-ready stochastic physics |
| **Soundings** | SHARPY | NetCDF/HDF5 profile ingestion |

### Numerics

- **Time integration:** Split-explicit (Klemp-Wilhelmson 1978), RK3, RK4 with CFL-adaptive stepping
- **Advection:** TVD and WENO5 directional splitting with coordinate-specific kernels
- **Diffusion:** Explicit and implicit Laplacian operators
- **Coordinate systems:** Cylindrical (r, theta, z) and Cartesian (x, y, z)

### GPU Compute

Vulkan compute backend with 15 GPU shaders covering advection, acoustic substeps, microphysics, dynamics tendencies, and diffusion. Optimizations include fused and batched acoustic substep dispatch (single GPU submission for all substeps) and coordinate-aware shader routing.

```yaml
numerics:
  compute:
    backend: vulkan    # or "cpu"
```

### Vulkan Viewer

Native GPU-accelerated 3D visualization:
- Volume ray marching through atmospheric fields
- Real-time interactive camera (orbit and freefly modes)
- Live shared-memory ingest from running simulations
- NPY field ingestion from simulation exports

---

## Quick Start

### Prerequisites

- **C++17 compiler** (clang++ >= 9.0 or g++ >= 7.0)
- **OpenMP** (optional; enables parallel computation)
- **Vulkan SDK** (optional; for GPU compute and 3D viewer)

For detailed per-platform installation instructions, see **[BUILDING.md](BUILDING.md)**.

### Build & Run

```bash
git clone https://github.com/vtiradoegas2025/Nimbus.git
cd Nimbus

# Auto-detect OS and install dependencies
./scripts/setup.sh

# Build simulation + GPU shaders
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

# Run the test suite
make test

# Run a simulation
./bin/tornado_sim --headless --config=configs/teaching/supercell_30min.yaml

# Build and launch the Vulkan viewer (optional)
make vulkan
./bin/vulkan_viewer --dry-run

# One-command demo: sim + live 3D viewer
./scripts/demo.sh --preset quick
```

### Example Configurations

**Supercell** (Cartesian, Kessler microphysics, 800x800x200 grid):
```yaml
coordinate_system: cartesian
grid:
  nx: 800
  ny: 800
  nz: 200
  dx: 250.0
  dy: 250.0
  dz: 75.0
microphysics:
  scheme: kessler
dynamics:
  scheme: cartesian
numerics:
  time_stepping:
    scheme: split_explicit
```

**Tornado** (cylindrical, Milbrandt-Yau microphysics, high resolution):
```yaml
coordinate_system: cylindrical
grid:
  nr: 512
  nth: 256
  nz: 256
microphysics:
  scheme: milbrandt
dynamics:
  scheme: tornado
```

Configs live in `configs/` organized by purpose: `student/` (small grids for learning), `teaching/` (classroom demos), `simulation/` (production runs), `benchmark/` (performance testing).

---

## Testing & Validation

### Test Suite

```bash
make test               # Full suite (26 binaries, 42,000+ assertions)
make test-core          # Field3D, output, hardware, SHM, coordinate system
make test-diagnostics   # Field contracts, validation, logging
make test-dynamics      # Cartesian dynamics, BCs, initial conditions
make test-numerics      # Advection, diffusion, time stepping
make test-physics       # Microphysics, radiation, terrain
make test-data          # Sounding ingestion
make test-vulkan        # GPU compute backend, CPU/GPU parity
make test-integration   # Config presets, performance
make test-shm-e2e       # End-to-end shared memory transport
make smoke-test         # Quick headless simulation
```

### Field Contract System

A CM1-style validation contract tracks every exported 3D field:
- **99** total contract fields defined
- **87** exported with runtime validation
- **20/20** required-now fields covered
- Strict mode: fails on any non-finite value or out-of-bounds violation

### Physical Validation

- Mass conservation: <0.1% drift over 2-hour simulations
- Energy conservation: <5% total energy change in stable cases
- Realistic supercell morphology against literature benchmarks
- Radar Z-V_r relationships match theoretical expectations

---

## Performance

| Grid | Steps/hr | Memory | Notes |
|------|----------|--------|-------|
| 64x64x32 (student) | ~10,000-15,000 | ~50 MB | CPU with OpenMP |
| 256x128x128 (research) | ~100-150 | ~8 GB | CPU with OpenMP |
| 800x800x200 (production) | ~20-30 | ~30 GB | Vulkan GPU compute |

Key optimizations:
- **Field3D** contiguous storage with row-major cache-aligned layout
- **OpenMP** parallelization: 4-8x speedup on multi-core systems
- **Vulkan GPU compute**: batched acoustic substeps, fused dispatch, 15 compute shaders
- **SIMD-ready** architecture (SSE/AVX/AVX-512/NEON runtime detection)
- Compiler flags: `-O3 -march=native -mtune=native`

---

## Repository Structure

```
Nimbus/
  src/                       Core simulation source (C++17)
    core/                    Simulation engine: runtime, orchestration, output, infra
    dynamics/                Dynamics schemes (tornado, supercell, cartesian)
    microphysics/            Cloud physics (Kessler, Thompson, Lin, Milbrandt)
    boundary_layer/          PBL schemes (YSU, MYNN, slab)
    turbulence/              SGS closures (Smagorinsky, TKE)
    radiation/               Radiative transfer (simple_grey)
    radar/                   Forward operators (reflectivity, velocity, ZDR)
    chaos/                   Stochastic perturbation schemes
    terrain/                 Orographic forcing (bell, schar)
    soundings/               Atmospheric profile ingestion (SHARPY)
    numerics/                Advection, diffusion, time-stepping factories
    boundary_conditions/     Coordinate-specific edge enforcement
    compute/                 GPU compute backend and kernel dispatch
    diagnostics/             Conservation budget, field contracts, validation
  include/                   Public headers organized by module
  tests/                     Test suite (unit, integration, regression)
  configs/                   Simulation configurations (YAML)
  vulkan/                    Vulkan viewer + GPU compute shaders
  scripts/                   Setup, demo, and utility scripts
  docs/                      Technical documentation and references
  data/                      Simulation exports (generated at runtime)
```

See [src/README.md](src/README.md) for the module architecture guide (factory patterns, infrastructure modules, and why each module is structured the way it is).

---

## Documentation

- **[Building from Source](BUILDING.md)** -- Platform-specific setup and build instructions
- **[Source Architecture](src/README.md)** -- Module patterns, factory design, adding new schemes
- **[Technical Reference](docs/README.md)** -- System documentation and runtime details
- **[Project Status](docs/STATUS.md)** -- Current audit, gaps, and roadmap
- **[Scientific Foundation](docs/foundationalScience.md)** -- Literature references and theoretical basis
- **[Header Interfaces](include/README.md)** -- Public API and base class conventions
- **[Vulkan Viewer](vulkan/README.md)** -- Rendering pipeline, GPU compute, and usage

---

## Scientific Foundation

This model builds on established atmospheric modeling research:

- Klemp & Wilhelmson (1978) -- Compressible storm dynamics, split-explicit time stepping
- Weisman & Klemp (1982) -- Supercell simulation foundations
- Bryan et al. (2003) -- Model resolution requirements
- Thompson et al. (2008) -- Aerosol-aware microphysics
- Hong et al. (2006) -- YSU boundary layer scheme
- Jung et al. (2008) -- Radar forward operators

Full references in **[docs/foundationalScience.md](docs/foundationalScience.md)**.

---

## Contributing

Nimbus welcomes contributions from atmospheric scientists and computational researchers.

### Guidelines

1. Fork and create a feature branch
2. Follow C++17 style conventions already in the codebase
3. Include tests for new physics parameterizations
4. Add scientific references for new schemes
5. Validate against existing test suite before submitting

### Areas for Contribution

- New physics schemes (microphysics, PBL, turbulence, radiation)
- Enhanced radar forward operators
- Terrain validation against reference cases
- Visualization features and rendering improvements
- Additional sounding data format support

---

## License

MIT License -- see [LICENSE](LICENSE) for details.
