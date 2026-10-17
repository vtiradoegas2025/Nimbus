#!/usr/bin/env python3
"""
Extract SHARPY-style sounding profiles from NetCDF/HDF-like files.

Output format (stdout):
  TMV_SHARPY_PROFILE_V1
  station_id<TAB>...
  timestamp_utc<TAB>...
  latitude_deg<TAB>...
  longitude_deg<TAB>...
  elevation_m<TAB>...
  levels<TAB>N
  h<TAB>p<TAB>t<TAB>td<TAB>wspd<TAB>wdir
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def _find_array(ds: Any, candidates: Iterable[str]) -> Optional[np.ndarray]:
    names = list(getattr(ds, "variables", {}).keys())
    lower_map = {name.lower(): name for name in names}
    norm_map = {name.lower().replace("/", "_"): name for name in names}

    def get_by_name(name: str) -> Optional[np.ndarray]:
        if name in getattr(ds, "variables", {}):
            values = np.asarray(ds[name].values)
            return values.squeeze()
        return None

    for cand in candidates:
        key = cand.lower()
        values = get_by_name(cand)
        if values is not None:
            return values
        if key in lower_map:
            values = get_by_name(lower_map[key])
            if values is not None:
                return values
        key_norm = key.replace("/", "_")
        if key_norm in norm_map:
            values = get_by_name(norm_map[key_norm])
            if values is not None:
                return values
        for var_name in names:
            ln = var_name.lower()
            if ln.endswith(f"/{key}") or ln.endswith(f"_{key}") or ln == key:
                values = get_by_name(var_name)
                if values is not None:
                    return values
    return None


def _to_profile_1d(values: Optional[np.ndarray], n: int) -> List[float]:
    if values is None:
        return [math.nan] * n
    arr = np.asarray(values, dtype=float).squeeze()
    if arr.ndim == 0:
        return [float(arr)] * n
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if arr.size < n:
        out = np.full(n, np.nan, dtype=float)
        out[: arr.size] = arr
        arr = out
    if arr.size > n:
        arr = arr[:n]
    return [float(x) for x in arr]


def _decode_attr(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        first = value.reshape(-1)[0]
        return _decode_attr(first)
    return str(value)


def _open_dataset(path: Path):
    import xarray as xr

    errors: List[str] = []
    for engine in (None, "scipy", "h5netcdf", "netcdf4"):
        try:
            if engine is None:
                return xr.open_dataset(path)
            return xr.open_dataset(path, engine=engine)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"engine={engine or 'auto'}: {exc}")
    raise RuntimeError(
        "unable to open dataset with xarray backends. "
        "Install netCDF/HDF readers (e.g. scipy for NetCDF3, h5py/h5netcdf/netCDF4 for HDF5/NetCDF4). "
        + " | ".join(errors)
    )


def _extract(path: Path) -> Dict[str, Any]:
    ds = _open_dataset(path)
    try:
        height = _find_array(ds, ("profiles/height_m", "height_m", "height", "z", "altitude"))
        pressure = _find_array(ds, ("profiles/pressure_hpa", "pressure_hpa", "pressure", "pres", "p"))
        temperature = _find_array(ds, ("profiles/temperature_k", "temperature_k", "temperature", "temp", "t"))
        dewpoint = _find_array(ds, ("profiles/dewpoint_k", "dewpoint_k", "dewpoint", "td"))
        wind_speed = _find_array(ds, ("profiles/wind_speed_ms", "wind_speed_ms", "wind_speed", "wspd"))
        wind_dir = _find_array(
            ds, ("profiles/wind_direction_deg", "wind_direction_deg", "wind_direction", "wdir")
        )

        if height is None or pressure is None or temperature is None:
            raise RuntimeError(
                "missing required profile variables. "
                "Required: height_m, pressure_hpa, temperature_k (or supported aliases)."
            )

        h = np.asarray(height, dtype=float).reshape(-1)
        p = np.asarray(pressure, dtype=float).reshape(-1)
        t = np.asarray(temperature, dtype=float).reshape(-1)
        n = int(min(h.size, p.size, t.size))
        if n < 2:
            raise RuntimeError("insufficient profile levels after extraction")
        h = h[:n]
        p = p[:n]
        t = t[:n]

        td = _to_profile_1d(dewpoint, n)
        ws = _to_profile_1d(wind_speed, n)
        wd = _to_profile_1d(wind_dir, n)

        order = np.argsort(h)
        h = h[order]
        p = p[order]
        t = t[order]
        td = [td[int(i)] for i in order]
        ws = [ws[int(i)] for i in order]
        wd = [wd[int(i)] for i in order]

        attrs = dict(getattr(ds, "attrs", {}))
        return {
            "station_id": _decode_attr(attrs.get("station_id", "")),
            "timestamp_utc": _decode_attr(attrs.get("timestamp_utc", "")),
            "latitude_deg": float(attrs.get("latitude_deg", math.nan))
            if attrs.get("latitude_deg", None) is not None
            else math.nan,
            "longitude_deg": float(attrs.get("longitude_deg", math.nan))
            if attrs.get("longitude_deg", None) is not None
            else math.nan,
            "elevation_m": float(attrs.get("elevation_m", math.nan))
            if attrs.get("elevation_m", None) is not None
            else math.nan,
            "height_m": [float(x) for x in h],
            "pressure_hpa": [float(x) for x in p],
            "temperature_k": [float(x) for x in t],
            "dewpoint_k": td,
            "wind_speed_ms": ws,
            "wind_direction_deg": wd,
        }
    finally:
        ds.close()


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("Usage: sharpy_extract.py <input_file>", file=sys.stderr)
        return 2

    path = Path(argv[1])
    if not path.exists():
        print(f"Input file does not exist: {path}", file=sys.stderr)
        return 2

    try:
        data = _extract(path)
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1

    print("TMV_SHARPY_PROFILE_V1")
    print(f"station_id\t{data['station_id']}")
    print(f"timestamp_utc\t{data['timestamp_utc']}")
    print(f"latitude_deg\t{data['latitude_deg']}")
    print(f"longitude_deg\t{data['longitude_deg']}")
    print(f"elevation_m\t{data['elevation_m']}")
    print(f"levels\t{len(data['height_m'])}")

    for i in range(len(data["height_m"])):
        print(
            f"{data['height_m'][i]}\t{data['pressure_hpa'][i]}\t{data['temperature_k'][i]}"
            f"\t{data['dewpoint_k'][i]}\t{data['wind_speed_ms'][i]}\t{data['wind_direction_deg'][i]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
