#!/usr/bin/env python3
"""
Generate a synthetic SHARPY sounding file in NetCDF classic (CDF1) format.

Produces a meteorologically plausible warm-season severe convective profile
with CAPE ~3000 J/kg, suitable for supercell/tornado simulations.

Output: /tmp/tmv_sharpy_baseline.nc (matches configs/sharpy_lp.yaml)

Usage:
    python3 tools/generate_test_sounding.py [output_path]
"""

import sys
import numpy as np
from scipy.io import netcdf_file


def build_profile():
    """Build a 20-level severe convective sounding profile."""

    # Heights from surface to ~16 km (meters AGL)
    height_m = np.array([
        0, 100, 250, 500, 750, 1000, 1500, 2000, 3000,
        4000, 5000, 6000, 7000, 8000, 9000, 10000,
        11000, 12000, 14000, 16000,
    ], dtype=np.float64)

    # Pressure profile (hPa) — standard-ish atmosphere with moist low levels
    pressure_hpa = np.array([
        1013, 1001, 983, 955, 928, 902, 852, 805, 718,
        638, 564, 496, 435, 380, 330, 285,
        245, 210, 150, 105,
    ], dtype=np.float64)

    # Temperature profile (K)
    # Warm, moist boundary layer with steep mid-level lapse rate.
    # Surface ~301 K (~28 C), tropopause ~210 K (~-63 C) at 12 km.
    temperature_k = np.array([
        301.0, 300.0, 298.5, 296.0, 293.5, 291.0, 286.0, 281.0, 271.0,
        261.0, 252.0, 243.0, 235.0, 228.0, 221.0, 215.0,
        212.0, 210.0, 210.0, 210.0,
    ], dtype=np.float64)

    # Dewpoint profile (K)
    # Moist boundary layer (small dewpoint depression), rapid drying above 2 km.
    # This produces realistic CAPE with a well-defined LFC.
    dewpoint_k = np.array([
        296.0, 295.0, 293.5, 291.0, 288.0, 285.0, 277.0, 270.0, 256.0,
        244.0, 234.0, 225.0, 218.0, 212.0, 206.0, 200.0,
        197.0, 195.0, 190.0, 185.0,
    ], dtype=np.float64)

    # Wind speed profile (m/s)
    # Strong low-level shear (SFC→1km), continued increase to 6 km.
    # 0-6 km shear ~35 m/s (matches config hodograph params).
    wind_speed_ms = np.array([
        4.1, 5.0, 7.0, 10.0, 12.5, 15.0, 20.0, 24.0, 30.0,
        34.0, 36.0, 37.0, 36.0, 34.0, 32.0, 30.0,
        28.0, 26.0, 22.0, 18.0,
    ], dtype=np.float64)

    # Wind direction profile (degrees, meteorological convention)
    # Veering with height: southerly at surface → westerly above 6 km.
    wind_direction_deg = np.array([
        170, 175, 185, 195, 205, 215, 225, 235, 245,
        250, 255, 258, 260, 262, 264, 265,
        266, 268, 270, 270,
    ], dtype=np.float64)

    return (height_m, pressure_hpa, temperature_k,
            dewpoint_k, wind_speed_ms, wind_direction_deg)


def write_sounding(output_path):
    """Write the sounding to a NetCDF classic (CDF1) file."""

    (height_m, pressure_hpa, temperature_k,
     dewpoint_k, wind_speed_ms, wind_direction_deg) = build_profile()

    n_levels = len(height_m)

    with netcdf_file(output_path, "w", version=1) as f:
        # Dimension
        f.createDimension("level", n_levels)

        # Required variables
        v_h = f.createVariable("height_m", np.float64, ("level",))
        v_h[:] = height_m
        v_h.units = "m"
        v_h.long_name = "Height above ground level"

        v_p = f.createVariable("pressure_hpa", np.float64, ("level",))
        v_p[:] = pressure_hpa
        v_p.units = "hPa"
        v_p.long_name = "Atmospheric pressure"

        v_t = f.createVariable("temperature_k", np.float64, ("level",))
        v_t[:] = temperature_k
        v_t.units = "K"
        v_t.long_name = "Air temperature"

        # Optional variables
        v_td = f.createVariable("dewpoint_k", np.float64, ("level",))
        v_td[:] = dewpoint_k
        v_td.units = "K"
        v_td.long_name = "Dewpoint temperature"

        v_ws = f.createVariable("wind_speed_ms", np.float64, ("level",))
        v_ws[:] = wind_speed_ms
        v_ws.units = "m/s"
        v_ws.long_name = "Wind speed"

        v_wd = f.createVariable("wind_direction_deg", np.float64, ("level",))
        v_wd[:] = wind_direction_deg
        v_wd.units = "degrees"
        v_wd.long_name = "Wind direction (meteorological)"

        # Global metadata attributes
        f.station_id = "KTEST"
        f.timestamp_utc = "2024-05-20 18:00:00"
        f.latitude_deg = 35.5
        f.longitude_deg = -97.5
        f.elevation_m = 350.0
        f.history = "Synthetic profile for TMV integration testing"

    print(f"[sounding] Wrote {n_levels}-level sounding to {output_path}")
    print(f"  Height: {height_m[0]:.0f} – {height_m[-1]:.0f} m")
    print(f"  Pressure: {pressure_hpa[-1]:.0f} – {pressure_hpa[0]:.0f} hPa")
    print(f"  Temperature: {temperature_k[-1]:.0f} – {temperature_k[0]:.0f} K")
    print(f"  Wind: {wind_speed_ms[0]:.0f} – {wind_speed_ms.max():.0f} m/s")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tmv_sharpy_baseline.nc"
    write_sounding(path)
