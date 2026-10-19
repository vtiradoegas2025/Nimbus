#!/usr/bin/env python3
"""Generate reproducible classic CDF sounding limit witnesses without libnetcdf."""

from __future__ import annotations

import math
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests/init/fixtures"


def u32(value: int) -> bytes:
    return struct.pack(">I", value)


def name(value: str) -> bytes:
    encoded = value.encode("ascii")
    return u32(len(encoded)) + encoded + b"\0" * (-len(encoded) % 4)


def attribute(key: str, kind: int, values: bytes, count: int) -> bytes:
    return name(key) + u32(kind) + u32(count) + values + b"\0" * (-len(values) % 4)


def profile(version: int, levels: int, *, record: bool = False,
            metadata: tuple[bytes, ...] = (), optional_levels: int | None = None,
            omit_payload: bool = False) -> bytes:
    fields = ("height_m", "pressure_hpa", "temperature_k", "dewpoint_k",
              "wind_speed_ms", "wind_direction_deg")
    dimensions = [("level", 0 if record else levels)]
    if optional_levels is not None:
        dimensions.append(("optional_level", optional_levels))
    prefix = (b"CDF" + bytes([version]) + u32(levels if record else 0) +
              u32(10) + u32(len(dimensions)) +
              b"".join(name(key) + u32(count) for key, count in dimensions) +
              (u32(12) + u32(len(metadata)) + b"".join(metadata)
               if metadata else u32(0) + u32(0)))
    sizes = [8 if record else 8 * levels for _ in fields]
    if optional_levels is not None:
        sizes[3] = 8 * optional_levels

    def header(offsets: list[int]) -> bytes:
        variables = []
        for index, key in enumerate(fields):
            dimension = 1 if index == 3 and optional_levels is not None else 0
            offset = struct.pack(">Q", offsets[index]) if version == 2 else u32(offsets[index])
            variables.append(name(key) + u32(1) + u32(dimension) + u32(0) + u32(0) +
                             u32(6) + u32(sizes[index]) + offset)
        return prefix + u32(11) + u32(len(fields)) + b"".join(variables)

    start = len(header([0] * len(fields)))
    offsets = [start + index * 8 for index in range(len(fields))] if record else []
    if not record:
        for size in sizes:
            offsets.append(start)
            start += size
    encoded_header = header(offsets)
    if omit_payload:
        return encoded_header

    def row(index: int, count: int) -> tuple[float, ...]:
        height = 16000.0 * index / (count - 1)
        temperature = 301.0 - 0.006 * height
        return (height, 1000.0 * math.exp(-height / 8200.0), temperature,
                temperature - 5.0, 20.0, 270.0)

    if record:
        payload = b"".join(struct.pack(">6d", *row(index, levels)) for index in range(levels))
    else:
        chunks = []
        for field in range(len(fields)):
            count = optional_levels if field == 3 and optional_levels is not None else levels
            chunks.append(b"".join(struct.pack(">d", row(index, count)[field])
                                   for index in range(count)))
        payload = b"".join(chunks)
    return encoded_header + payload


def main() -> None:
    for version in (1, 2):
        cases = {
            "fixed_100000": profile(version, 100000),
            "fixed_100001": profile(version, 100001),
            "record_33": profile(version, 33, record=True),
            "record_100001": profile(version, 100001, record=True, omit_payload=True),
            "optional_100001": profile(version, 33, optional_levels=100001),
            "station_1023": profile(version, 33, metadata=(attribute("station_id", 2, b"M" * 1023, 1023),)),
            "station_1024": profile(version, 33, metadata=(attribute("station_id", 2, b"M" * 1024, 1024),)),
            "timestamp_1023": profile(version, 33, metadata=(attribute("timestamp_utc", 2, b"T" * 1023, 1023),)),
            "timestamp_1024": profile(version, 33, metadata=(attribute("timestamp_utc", 2, b"T" * 1024, 1024),)),
            "latitude_scalar": profile(version, 33, metadata=(attribute("latitude_deg", 6, struct.pack(">d", 18.5), 1),)),
            "latitude_array": profile(version, 33, metadata=(attribute("latitude_deg", 6, struct.pack(">2d", 18.5, 19.5), 2),)),
        }
        for key, payload in cases.items():
            target = FIXTURES / f"sharpy_classic_limits_{key}_cdf{version}.nc"
            if target.exists():
                if target.read_bytes() != payload:
                    raise RuntimeError(f"preserve differing fixture: {target}")
            else:
                target.write_bytes(payload)
    print("classic CDF1/CDF2 limit fixtures generated and byte-verified")


if __name__ == "__main__":
    main()
