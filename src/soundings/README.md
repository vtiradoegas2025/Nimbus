# Atmospheric Soundings Module

This module provides support for loading and interpolating atmospheric sounding data from external sources, enabling the use of real atmospheric profiles for model initialization instead of procedural profiles.

## Architecture

The soundings module follows the same factory pattern as other Nimbus components (microphysics, terrain, radiation, etc.):

```
src/soundings/
├── base/                          # Base classes and utilities
│   ├── soundings_base.hpp/cpp     # Data structures and base class
├── schemes/                       # Sounding implementations
│   └── sharpy/                    # SHARPY sounding reader
│       ├── sharpy_sounding.hpp/cpp
├── factory.hpp/cpp                # Factory pattern implementation
├── soundings.cpp                  # Main API
├── integration_example.hpp        # Integration guide
└── README.md                      # This file
```

## Features

- **Multiple data formats**: SHARPY-style ingestion with native NetCDF classic, NetCDF4, and HDF5 support
- **Quality control**: Automatic validation and filtering of sounding data
- **Optional-field safety**: Incomplete dewpoint/wind vectors are discarded during QC/interpolation to avoid mismatched profile application
- **Interpolation**: Linear, monotone spline (PCHIP-style), and log-linear (pressure) interpolation to model grid heights
- **Fallback support**: Graceful degradation to procedural profiles
- **Extensible design**: Easy to add new sounding formats

## Usage

### Basic Usage

```cpp
#include "data/soundings.hpp"

// Configure sounding system
SoundingConfig config;
config.scheme_id = "sharpy";
config.file_path = "path/to/sounding.h5";
config.use_fallback_profiles = true;

// Initialize
initialize_soundings(config);

// Load sounding data
SoundingData data = load_sounding_data();

// Interpolate to model grid
std::vector<double> model_heights = {0, 100, 200, ..., 15000};
SoundingData interpolated = interpolate_sounding_to_grid(data, model_heights);
```

### Configuration

Add to YAML config files:

```yaml
environment:
  sounding:
    scheme_id: "sharpy"           # "sharpy", "none", or empty
    file_path: "data/sounding.h5"  # Path to sounding file
    use_fallback_profiles: true    # Use procedural if sounding fails
    interpolation_method: 0        # 0=linear, 1=spline, 2=log-linear
    extrapolate_below_ground: false
    extrapolate_above_top: false
```

## SHARPY Integration

SHARPY is a Python package for analyzing atmospheric sounding data. This module can read SHARPY-exported files containing:

- Height/pressure profiles
- Temperature and dewpoint
- Wind speed and direction
- Derived quantities (potential temperature, mixing ratio, etc.)

### Expected SHARPY HDF5 Structure

```
/profiles/
  height_m                 # Height above ground (m)
  pressure_hpa            # Pressure (hPa)
  temperature_k           # Temperature (K)
  dewpoint_k              # Dewpoint temperature (K)
  wind_speed_ms           # Wind speed (m/s)
  wind_direction_deg      # Wind direction (degrees)
/metadata/
  station_id              # Station identifier
  timestamp_utc           # Observation time
  latitude_deg            # Station latitude
  longitude_deg           # Station longitude
  elevation_m             # Station elevation
```

## Implementation Status

### Current Implementation
- ✅ Data structures and base classes
- ✅ Factory pattern and API
- ✅ Quality control and validation
- ✅ Linear interpolation
- ✅ Monotone spline interpolation (PCHIP-style)
- ✅ Log-linear pressure interpolation
- ✅ Native in-process NetCDF classic reader path (CDF1/CDF2), including record-variable layouts
- ✅ Native in-process NetCDF C API reader path for NetCDF4/HDF-backed layouts
- ✅ Native in-process HDF5 (HL API) reader path for SHARPY-style HDF5 datasets
- ✅ Packaging/runtime library validation report in backend regression (`tests/test_backend_physics.sh`)
- ✅ Integration examples and documentation

### Still Incomplete / Deferred
- 🔄 Broader real-case atmospheric validation suite across multiple observed events

## Dependencies

### Runtime Dependencies
- **No Python dependency is required** for sounding ingestion.
- Native readers use shared libraries loaded at runtime:
  - **libnetcdf**: NetCDF classic/64-bit-offset/NetCDF4-compatible paths
  - **libhdf5 + libhdf5_hl**: SHARPY-style HDF5 dataset paths

### Build Integration

No additional C++ link-time libraries are required for default build; readers resolve shared libraries at runtime.

## Testing

Run focused soundings regression tests:

```bash
make test-soundings
```

Run backend packaging validation report (and strict fail mode if desired):

```bash
make test-backend-physics
# Optional strict packaging gate:
# SOUNDINGS_PACKAGING_STRICT=1 make test-backend-physics
```

To test with a synthetic NetCDF profile:

1. Create a file:
   ```python
   import numpy as np
   import xarray as xr

   z = np.array([0, 500, 1000, 1500, 2000], dtype=float)
   ds = xr.Dataset(
       data_vars={
           "height_m": (("level",), z),
           "pressure_hpa": (("level",), np.array([1000, 950, 900, 850, 800], dtype=float)),
           "temperature_k": (("level",), np.array([300, 296, 292, 288, 284], dtype=float)),
           "dewpoint_k": (("level",), np.array([294, 290, 286, 282, 278], dtype=float)),
           "wind_speed_ms": (("level",), np.array([5, 8, 11, 14, 17], dtype=float)),
           "wind_direction_deg": (("level",), np.array([180, 190, 200, 210, 220], dtype=float)),
       },
       attrs={"station_id": "KTEST"}
   )
   ds.to_netcdf("sounding.nc", engine="scipy")
   ```
2. Point `environment.sounding.file_path` at that file and run headless.

## Future Enhancements

- Expanded metadata extraction coverage across non-canonical SHARPY exports
- Support for multiple sounding sources
- Sounding data assimilation
- Real-time sounding updates
- Additional quality control checks
