#!/usr/bin/env python3
"""Fail unless a NumPy v1/v2 little-endian float32 array has signal."""

from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path
import struct


def read_header(handle) -> tuple[dict[str, object], int]:
    if handle.read(6) != b"\x93NUMPY":
        raise ValueError("not an NPY file")
    major, minor = handle.read(2)
    if (major, minor) == (1, 0):
        header_length = struct.unpack("<H", handle.read(2))[0]
    elif major in (2, 3):
        header_length = struct.unpack("<I", handle.read(4))[0]
    else:
        raise ValueError(f"unsupported NPY version {major}.{minor}")
    header = ast.literal_eval(handle.read(header_length).decode("latin1"))
    shape = header.get("shape")
    if not isinstance(shape, tuple) or not shape:
        raise ValueError("missing or invalid NPY shape")
    count = math.prod(shape)
    return header, count


def max_abs_finite(path: Path) -> tuple[float, int]:
    with path.open("rb") as handle:
        header, count = read_header(handle)
        if header.get("descr") not in ("<f4", "|f4"):
            raise ValueError(f"expected float32 data, got {header.get('descr')!r}")
        if header.get("fortran_order") is not False:
            raise ValueError("Fortran-order arrays are not supported")
        payload = handle.read()
    expected_bytes = count * 4
    if len(payload) != expected_bytes:
        raise ValueError(
            f"payload size mismatch: expected {expected_bytes}, got {len(payload)}"
        )
    maximum = 0.0
    finite_count = 0
    for (value,) in struct.iter_unpack("<f", payload):
        if math.isfinite(value):
            finite_count += 1
            maximum = max(maximum, abs(value))
    return maximum, finite_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()
    maximum, finite_count = max_abs_finite(args.path)
    if finite_count == 0:
        raise SystemExit(f"{args.path}: array has no finite values")
    if not maximum > args.threshold:
        raise SystemExit(
            f"{args.path}: max_abs={maximum:g} does not exceed {args.threshold:g}"
        )
    print(f"{args.path}: finite={finite_count}, max_abs={maximum:.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
