#!/usr/bin/env python3
"""Qualify Vulkan residency science, copy savings, and memory equivalence."""

from __future__ import annotations

import argparse
import ast
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile


FIELDS = ("u", "v", "w", "rho", "p", "theta", "qv", "qr")
TELEMETRY_KEYS = (
    "model_steps",
    "command_buffers",
    "dispatches",
    "host_upload_bytes",
    "host_download_bytes",
    "device_upload_bytes",
    "device_download_bytes",
    "resident_hits",
    "resident_upload_bytes_saved",
    "staging_current_bytes",
    "staging_high_water_bytes",
    "device_current_bytes",
    "device_high_water_bytes",
    "staging_memory_allocations_high_water",
    "device_memory_allocations_high_water",
    "staging_memory_reserved_high_water_bytes",
    "device_memory_reserved_high_water_bytes",
)


def read_npy_float32(path: Path) -> tuple[tuple[int, ...], tuple[float, ...]]:
    with path.open("rb") as handle:
        if handle.read(6) != b"\x93NUMPY":
            raise RuntimeError(f"{path}: not an NPY file")
        major, minor = handle.read(2)
        if (major, minor) == (1, 0):
            header_size = struct.unpack("<H", handle.read(2))[0]
        elif major in (2, 3):
            header_size = struct.unpack("<I", handle.read(4))[0]
        else:
            raise RuntimeError(f"{path}: unsupported NPY version {major}.{minor}")
        encoding = "utf-8" if major == 3 else "latin1"
        header = ast.literal_eval(handle.read(header_size).decode(encoding))
        if header.get("descr") not in ("<f4", "|f4"):
            raise RuntimeError(f"{path}: expected little-endian float32")
        if header.get("fortran_order") is not False:
            raise RuntimeError(f"{path}: Fortran-order arrays are unsupported")
        shape = header.get("shape")
        if not isinstance(shape, tuple) or not shape:
            raise RuntimeError(f"{path}: invalid shape")
        payload = handle.read()
    count = math.prod(shape)
    if len(payload) != count * 4:
        raise RuntimeError(f"{path}: payload size mismatch")
    values = tuple(value for (value,) in struct.iter_unpack("<f", payload))
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"{path}: non-finite value")
    return shape, values


def final_frame(run_dir: Path) -> Path:
    frames = sorted(
        path for path in run_dir.glob("step_[0-9][0-9][0-9][0-9][0-9][0-9]")
        if path.is_dir()
    )
    if not frames:
        raise RuntimeError(f"{run_dir}: no output frames")
    return frames[-1]


def parse_telemetry(stdout: str, label: str) -> dict[str, int]:
    lines = [line for line in stdout.splitlines()
             if "[GPU TELEMETRY]" in line]
    if not lines:
        raise RuntimeError(f"{label}: missing final GPU telemetry")
    values: dict[str, int] = {}
    for key in TELEMETRY_KEYS:
        match = re.search(rf"\b{re.escape(key)}=([0-9]+)", lines[-1])
        if match is None:
            raise RuntimeError(f"{label}: telemetry missing {key}")
        values[key] = int(match.group(1))
    return values


def run(binary: Path, config: Path, duration: int, resident: bool,
        output_dir: Path) -> dict[str, int]:
    env = os.environ.copy()
    env["NIMBUS_GPU_RESIDENT"] = "1" if resident else "0"
    env["NIMBUS_GPU_LIN_BATCH"] = "1"
    env["NIMBUS_GPU_KERNEL_TIMING"] = "1"
    result = subprocess.run(
        [
            str(binary),
            "--config", str(config),
            "--headless",
            "--compute-backend", "vulkan",
            "--guard-mode", "off",
            f"--duration={duration}",
            f"--write-every={duration}",
            f"--outdir={output_dir}",
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"resident={'on' if resident else 'off'} run failed "
            f"with exit {result.returncode}\n{result.stdout[-4000:]}"
        )
    if "selected device" not in result.stdout:
        raise RuntimeError(
            f"resident={'on' if resident else 'off'} run did not select "
            "a physical Vulkan device"
        )
    if "Falling back to CPU backend" in result.stdout:
        raise RuntimeError(
            f"resident={'on' if resident else 'off'} run fell back to CPU"
        )
    return parse_telemetry(
        result.stdout, f"resident={'on' if resident else 'off'}")


def require_equivalent_resource_shape(
    resident_on: dict[str, int], resident_off: dict[str, int]
) -> None:
    same_keys = (
        "model_steps",
        "command_buffers",
        "dispatches",
        "host_download_bytes",
        "staging_current_bytes",
        "staging_high_water_bytes",
        "device_current_bytes",
        "device_high_water_bytes",
        "staging_memory_allocations_high_water",
        "device_memory_allocations_high_water",
        "staging_memory_reserved_high_water_bytes",
        "device_memory_reserved_high_water_bytes",
    )
    for key in same_keys:
        if resident_on[key] != resident_off[key]:
            raise RuntimeError(
                f"residency changed {key}: "
                f"on={resident_on[key]} off={resident_off[key]}")
    if resident_on["resident_hits"] <= 0:
        raise RuntimeError("resident-on execution recorded no reuse hits")
    if resident_on["resident_upload_bytes_saved"] <= 0:
        raise RuntimeError("resident-on execution avoided no upload bytes")
    if resident_off["resident_hits"] != 0:
        raise RuntimeError("resident-off execution unexpectedly recorded hits")
    if resident_off["resident_upload_bytes_saved"] != 0:
        raise RuntimeError("resident-off execution unexpectedly saved uploads")
    if resident_on["host_upload_bytes"] >= resident_off["host_upload_bytes"]:
        raise RuntimeError(
            "resident-on execution did not reduce host upload traffic")
    avoided = resident_off["host_upload_bytes"] - resident_on["host_upload_bytes"]
    if avoided != resident_on["resident_upload_bytes_saved"]:
        raise RuntimeError(
            "host upload delta does not equal resident saved-byte telemetry: "
            f"delta={avoided} saved={resident_on['resident_upload_bytes_saved']}")
    print(
        "[gpu-residency-equivalence] COPY PASS "
        f"host_upload_bytes_on={resident_on['host_upload_bytes']} "
        f"host_upload_bytes_off={resident_off['host_upload_bytes']} "
        f"avoided={avoided} resident_hits={resident_on['resident_hits']}"
    )
    print(
        "[gpu-residency-equivalence] MEMORY PASS "
        f"device_high_water_bytes={resident_on['device_high_water_bytes']} "
        "device_memory_allocations_high_water="
        f"{resident_on['device_memory_allocations_high_water']} "
        f"staging_high_water_bytes={resident_on['staging_high_water_bytes']} "
        "staging_memory_allocations_high_water="
        f"{resident_on['staging_memory_allocations_high_water']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("bin/nimbus_sim"))
    parser.add_argument(
        "--config", type=Path,
        default=Path("tests/fixtures/configs/qualification/gpu_residency.yaml"),
    )
    # Four seconds exercises two complete large steps in the maintained
    # profile. A one-step endpoint cannot expose a stale binding that is
    # published by one operator and consumed after a later host writer.
    parser.add_argument("--duration", type=int, default=4)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be a positive integer")

    with tempfile.TemporaryDirectory(
        prefix="nimbus_gpu_residency_equivalence_"
    ) as root_text:
        root = Path(root_text)
        on_dir = root / "resident_on"
        off_dir = root / "resident_off"
        resident_on = run(
            args.binary, args.config, args.duration, True, on_dir)
        resident_off = run(
            args.binary, args.config, args.duration, False, off_dir)
        require_equivalent_resource_shape(resident_on, resident_off)
        on_frame = final_frame(on_dir)
        off_frame = final_frame(off_dir)

        failed = False
        for field in FIELDS:
            shape_on, values_on = read_npy_float32(on_frame / f"{field}.npy")
            shape_off, values_off = read_npy_float32(off_frame / f"{field}.npy")
            if shape_on != shape_off:
                raise RuntimeError(
                    f"{field}: shape mismatch {shape_on} != {shape_off}"
                )
            changed = 0
            max_abs = 0.0
            for resident_value, baseline_value in zip(values_on, values_off):
                difference = abs(resident_value - baseline_value)
                if difference != 0.0:
                    changed += 1
                    max_abs = max(max_abs, difference)
            status = "PASS" if changed == 0 else "FAIL"
            print(
                f"[gpu-residency-equivalence] {status} field={field} "
                f"changed={changed}/{len(values_on)} max_abs={max_abs:.9e}"
            )
            failed = failed or changed != 0

    if failed:
        print("[gpu-residency-equivalence] RESULT: FAIL")
        return 1
    print("[gpu-residency-equivalence] RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
