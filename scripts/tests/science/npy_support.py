from __future__ import annotations

import array
import ast
import math
from pathlib import Path
import struct


def read_float32_npy(
    path: Path,
) -> tuple[tuple[int, ...], bytes, array.array[float]]:
    payload = path.read_bytes()
    if payload[:6] != b"\x93NUMPY" or len(payload) < 10:
        raise RuntimeError(f"invalid NPY file: {path}")
    major = payload[6]
    if major == 1:
        header_size = struct.unpack_from("<H", payload, 8)[0]
        header_start = 10
    elif major in (2, 3):
        header_size = struct.unpack_from("<I", payload, 8)[0]
        header_start = 12
    else:
        raise RuntimeError(f"unsupported NPY version {major}: {path}")
    header = ast.literal_eval(
        payload[header_start:header_start + header_size].decode("latin1").strip()
    )
    if header.get("descr") not in ("<f4", "=f4", "|f4"):
        raise RuntimeError(f"expected little-endian float32 NPY data: {path}")
    if header.get("fortran_order"):
        raise RuntimeError(f"Fortran-order NPY data is unsupported: {path}")
    shape = tuple(int(value) for value in header["shape"])
    raw = payload[header_start + header_size:]
    expected_bytes = math.prod(shape) * 4
    if len(raw) != expected_bytes:
        raise RuntimeError(f"NPY payload size mismatch: {path}")
    values = array.array("f")
    values.frombytes(raw)
    return shape, raw, values
