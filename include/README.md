# Header Files

This directory contains the header files defining interfaces, base classes, and type definitions for Nimbus. Headers are organized by domain and provide the public API for the simulation framework.

## Directory Layout

Headers are organized into domain-specific subdirectories:

```
include/
├── core/           # Simulation state, Field3D, runtime config, output
├── physics/        # Physics module base classes
├── numerics/       # Advection, diffusion, time-stepping interfaces
├── diagnostics/    # Radar, field validation, field contracts
├── data/           # Sounding data structures
└── util/           # SIMD, string, grid metric utilities
```

## Core Headers (`core/`)

- **`simulation.hpp`** - Primary simulation class, `SimulationState`, `WindProfile`, grid metrics
- **`field3d.hpp`** - Contiguous 3D field storage with flattened indexing
- **`runtime_config.hpp`** - Runtime configuration globals and parser-facing declarations
- **`physical_constants.hpp`** - Shared constants consumed across physics modules
- **`headless_runtime.hpp`** - Headless simulation loop entry point
- **`output_writer.hpp`** - NPY export and manifest generation

## Diagnostics Headers (`diagnostics/`)

- **`field_contract.hpp`** - CM1-style export/validation contract metadata
- **`field_validation.hpp`** - Field-level guard/validation result structures
- **`radar.hpp`** / **`radar_base.hpp`** - Radar forward operator interfaces

## Numerics Headers (`numerics/`)

- **`advection.hpp`** / **`advection_base.hpp`** - Scalar transport interfaces
- **`diffusion_base.hpp`** - Diffusion algorithm interface
- **`time_stepping_base.hpp`** - Time integration interface
- **`numerics_base.hpp`** - Numerical methods coordinator
- **`compute_backend.hpp`** - Compute backend abstraction
- **`compute_kernel_template.hpp`** - Kernel-template registry/dispatch API

## Physics Module Base Classes (`physics/`)

Each physics module provides a base class defining the interface for all schemes:

- **`boundary_layer_base.hpp`** - PBL parameterization interface (MOST surface fluxes, non-local mixing)
- **`chaos_base.hpp`** - Stochastic parameterization interface (SPPT, perturbation fields)
- **`dynamics_base.hpp`** - Compressible Euler equations interface
- **`microphysics_base.hpp`** - Cloud microphysics interface (phase changes, precipitation fallout)
- **`radiation_base.hpp`** - Radiative transfer interface (LW/SW heating)
- **`terrain_base.hpp`** - Terrain/orographic forcing interface
- **`turbulence_base.hpp`** - Sub-grid turbulence closures (eddy viscosity, TKE prognostic)

## Data Headers (`data/`)

- **`soundings_base.hpp`** - Vertical profile ingestion interface
- **`soundings.hpp`** - Sounding data containers and profile utilities

## Utility Headers (`util/`)

- **`simd_utils.hpp`** - Runtime SIMD detection and vectorized operations
- **`grid_metric_utils.hpp`** - Cylindrical grid metric computations
- **`string_utils.hpp`** - String parsing utilities
- **`log.hpp`** - Logging infrastructure

## Design Principles

### Interface Consistency
All base classes follow consistent patterns:
- **Factory registration** via `register_*_scheme()` functions
- **Initialization** through `initialize()` methods
- **Application** via `apply_*()` or `compute_*()` methods
- **Configuration** through parameter structures

### Memory Management
- **RAII principles** throughout
- **Smart pointer usage** where appropriate
- **No raw pointers** in public interfaces
- **Exception safety** guaranteed

### Type Safety
- **Strong typing** with custom structs/enums
- **Template metaprogramming** for compile-time optimization
- **Runtime type checking** with assertions
- **Unit-safe quantities** where beneficial

### Extensibility
- **Plugin architecture** via factory registration
- **Forward-compatible interfaces**
- **Versioned base classes**
- **Optional feature flags**

## Usage Patterns

### Implementing a New Scheme
```cpp
#include "physics/microphysics_base.hpp"

class MyMicrophysics : public MicrophysicsScheme {
public:
    void initialize(const MicrophysicsConfig& config) override { /* ... */ }
    void compute_tendencies(/* params */) override { /* ... */ }
};
```

### Using Physics Modules
```cpp
#include "core/simulation.hpp"
#include "physics/microphysics_base.hpp"

SimulationState state;
initialize_simulation(state, config);

auto microphysics = create_microphysics_scheme("kessler");
microphysics->compute_tendencies(state, tendencies, dt);
```

## Build System Integration

Headers are included via the main Makefile:
- **Automatic dependency generation**
- **Precompiled headers** support
- **Include path management**
- **Conditional compilation** flags

## Testing

Headers include extensive contracts:
- **Preconditions** checked via assertions
- **Postconditions** validated
- **Invariant maintenance**
- **Error handling** with descriptive messages

## Documentation

Each header contains:
- **Doxygen-compatible comments**
- **Mathematical formulations**
- **Algorithm references**
- **Usage examples**
- **Performance characteristics**

This header organization provides a clean separation between interface and implementation, enabling modular development and testing of atmospheric physics parameterizations.
