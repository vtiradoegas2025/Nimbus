# Nimbus

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Tests](https://img.shields.io/badge/Tests-passing-brightgreen.svg)](#testing--validation)

**A modern approach to supercell modeling.** High-performance atmospheric simulation built in **C++17** with a modular physics architecture and a **native Vulkan rendering path** for real-time 3D visualization.

> **Status:** Active development — research prototype. See **[docs/STATUS.md](docs/STATUS.md)** for the latest audit.

---

## Motivation

Nimbus implements a **CM1-lite style research model** designed to run on personal hardware (laptops to workstations) while maintaining a modular, testable physics architecture. It is not a replacement for CM1 — the goal is a pragmatic subset with strong developer ergonomics, strict field validation, and reproducible workflows.

**Why C++ over Fortran:**
- Explicit memory-layout control for large 3D fields (contiguous `Field3D` storage)
- Factory-based scheme architecture for rapid physics experimentation
- Direct integration with Vulkan for native GPU rendering
- Modern tooling ecosystem (profiling, testing, CI)

---

## Architecture

### Governing Equations

Compressible, non-hydrostatic Euler equations in cylindrical coordinates (r, θ, z):

```
∂u/∂t + ∇·(u⊗u) + (1/ρ₀)∇p' + gθ'/θ₀ k̂ = -∇·τ + F_buoyancy
∂θ/∂t + ∇·(uθ) = Q_radiation + Q_microphysics + ∇·(K_θ ∇θ)
∂q_v/∂t + ∇·(uq_v) = -C - E + ∇·(K_q ∇q_v)
```

### Physics Modules

| Module | Schemes | Notes |
|--------|---------|-------|
| **Microphysics** | Kessler, Thompson, Lin, Milbrandt-Yau | Warm-rain through double-moment |
| **Boundary Layer** | YSU, MYNN, Slab | Non-local and TKE-based closures |
| **Turbulence** | Smagorinsky-Lilly, TKE prognostic | Eddy-viscosity and prognostic closures |
| **Radiation** | Simple grey (RRTMG planned) | Longwave/shortwave transfer |
| **Dynamics** | Tornado, Supercell | Axisymmetric and full 3D modes |
| **Radar** | Reflectivity, Doppler velocity, Z_DR | Forward operators for synthetic obs |
| **Terrain** | Bell mountain, Schar | Idealized orographic forcing |
| **Chaos** | IC perturbation, BL stochastic, Full | Ensemble-ready stochastic physics |
| **Soundings** | SHARPY | NetCDF/HDF5 profile ingestion |

### Numerics

- **Time integration:** RK3/RK4 with CFL-adaptive stepping
- **Advection:** TVD and WENO5 directional splitting
- **Diffusion:** Explicit and implicit Laplacian operators
- **Grid:** Cylindrical (r, θ, z) with configurable resolution up to 512×256×256

### Vulkan Rendering

Native GPU-accelerated visualization pipeline:
- Volume ray marching through atmospheric fields
- Cylindrical-to-Cartesian coordinate transformation
- Real-time interactive orbital camera
- Direct NPY field ingestion from simulation exports

---

## Testing & Validation

### Test Suite

```bash
make test                       # Full baseline suite
make test-backend-physics       # Strict field contract checks across config matrix
make test-soundings             # Sounding ingestion regression
make test-radiation-regression  # Radiation scheme regression
make test-terrain-regression    # Terrain scheme regression
make test-guards                # Field contract guard verification
```

### Field Contract System

A CM1-style validation contract tracks every exported 3D field:
- **99** total contract fields defined
- **87** exported with runtime validation
- **20/20** required-now fields covered
- Strict mode: fails on any non-finite value or out-of-bounds violation in exported fields

### Physical Validation

- Mass conservation: <0.1% drift over 2-hour simulations
- Energy conservation: <5% total energy change in stable cases
- Realistic supercell morphology against literature benchmarks
- Radar Z-V_r relationships match theoretical expectations

---

## Repository Structure

```
Nimbus/
├── src/                    # Core simulation source (C++17)
│   ├── core/              # Runtime coordinators, main executable, output writers
│   ├── dynamics/          # Dynamics schemes (tornado, supercell)
│   ├── microphysics/      # Cloud physics (Kessler, Thompson, Lin, Milbrandt)
│   ├── boundary_layer/    # PBL schemes (YSU, MYNN, slab)
│   ├── turbulence/        # SGS closures (Smagorinsky, TKE)
│   ├── radiation/         # Radiative transfer (simple_grey)
│   ├── radar/             # Forward operators (reflectivity, velocity, ZDR)
│   ├── chaos/             # Stochastic perturbation schemes
│   ├── terrain/           # Orographic forcing (bell, schar)
│   ├── soundings/         # Atmospheric profile ingestion
│   ├── numerics/          # Advection, diffusion, time-stepping factories
│   └── validation/        # Field contract and guard infrastructure
├── include/               # Public headers organized by domain
│   ├── core/              # Simulation state, Field3D, runtime config
│   ├── physics/           # Physics module base classes
│   ├── numerics/          # Numerical method interfaces
│   ├── diagnostics/       # Radar, field validation contracts
│   ├── data/              # Sounding data structures
│   └── util/              # SIMD, string, grid metric utilities
├── tests/                 # Test suite (unit, integration, regression)
├── configs/               # Simulation configurations (YAML)
├── vulkan/                # Native Vulkan renderer (shaders, pipeline, viewer)
├── docs/                  # Technical documentation and references
└── data/                  # Simulation exports (generated at runtime)
```

---

## Quick Start

### Prerequisites

- **C++17 compiler** (clang++ >= 9.0 or g++ >= 7.0)
- **OpenMP** (optional; `brew install libomp` on macOS)
- **Vulkan SDK** (optional; for the native viewer)

### Build & Run

```bash
git clone https://github.com/vtiradoegas2025/Nimbus.git
cd Nimbus

# Build simulation engine
make

# Run a quick smoke test
./bin/tornado_sim --headless --config=configs/classic.yaml --duration=60

# Run the test suite
make test

# Build and run the Vulkan viewer (optional)
make vulkan
./bin/vulkan_viewer --dry-run
```

### Example Configurations

**Classic Supercell** (Weisman-Klemp):
```yaml
microphysics:
  scheme: thompson
boundary_layer:
  scheme: ysu
turbulence:
  scheme: smagorinsky
  Cs: 0.18
environment:
  cape_target_jkg: 2500
```

**High-Resolution Tornado**:
```yaml
grid:
  nr: 512
  nth: 256
  nz: 256
microphysics:
  scheme: milbrandt
radiation:
  scheme: simple_grey
```

---

## Performance

| Grid | Steps/hr | Memory | Notes |
|------|----------|--------|-------|
| 64×64×32 (test) | ~10,000–15,000 | ~50 MB | With OpenMP |
| 256×128×128 (production) | ~100–150 | ~8 GB | With OpenMP |

Key optimizations:
- **Field3D** contiguous storage: ~99.9% reduction in allocation overhead vs nested vectors
- **OpenMP** parallelization: 4–8x speedup on multi-core systems
- **Cache-aligned** memory layout with row-major access patterns
- **SIMD-ready** architecture (SSE/AVX/AVX-512 runtime detection)
- Compiler flags: `-O3 -march=native -mtune=native`

---

## Documentation

- **[Technical Reference](docs/README.md)** — System documentation and build details
- **[Project Status](docs/STATUS.md)** — Current audit, gaps, and roadmap
- **[Scientific Foundation](docs/foundationalScience.md)** — Literature references and theoretical basis
- **[API Reference](include/README.md)** — Header interfaces and design patterns
- **[Vulkan Viewer](vulkan/README.md)** — Rendering pipeline and usage

---

## Scientific Foundation

This model builds on established atmospheric modeling research:

- Klemp & Wilhelmson (1978) — Compressible storm dynamics
- Weisman & Klemp (1982) — Supercell simulation foundations
- Bryan et al. (2003) — Model resolution requirements
- Thompson et al. (2008) — Aerosol-aware microphysics
- Hong et al. (2006) — YSU boundary layer scheme
- Jung et al. (2008) — Radar forward operators

Full references in **[docs/foundationalScience.md](docs/foundationalScience.md)**.

---

## Contributing

Nimbus welcomes contributions from atmospheric scientists and computational researchers. See the [Contributing Guide](#contributing-guidelines) below.

### Contributing Guidelines

1. Fork and create a feature branch
2. Follow C++17 style conventions already in the codebase
3. Include tests for new physics parameterizations
4. Add scientific references for new schemes
5. Validate against existing test suite before submitting

### Areas for Contribution

- New physics schemes (microphysics, PBL, turbulence)
- Enhanced radar forward operators
- Terrain validation against reference cases
- GPU compute kernel offload
- Visualization features

---

## License

MIT License — see [LICENSE](LICENSE) for details.

*Built with modern C++17 and a Vulkan-first rendering path. A modern approach to reproducible, well-validated severe storm research.*
