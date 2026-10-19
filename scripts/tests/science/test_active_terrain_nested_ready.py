#!/usr/bin/env python3
"""Bounded ordinary gate for stationary mapped Bell/Schär nesting on CPU."""

from __future__ import annotations

import argparse
import csv
import array
import ast
import json
import math
from pathlib import Path
import re
import subprocess
import struct
import sys
import tempfile

from npy_support import read_float32_npy


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.qualification.checkpointing.qualify_vulkan_restart import (  # noqa: E402
    LEDGERS,
    require_report_owner_identities,
    require_restart_suffix,
)

QUALIFICATION = ROOT / "tests/fixtures/configs/qualification"
PROFILES = (
    ("bell", "one_way", 2, QUALIFICATION / "stationary_nested_active_bell_cpu_one_way_ready.yaml"),
    ("bell", "two_way", 2, QUALIFICATION / "stationary_nested_active_terrain_cpu_two_way_ready.yaml"),
    ("schar", "one_way", 2, QUALIFICATION / "stationary_nested_active_schar_cpu_one_way_ready.yaml"),
    ("schar", "two_way", 2, QUALIFICATION / "stationary_nested_active_schar_cpu_two_way_ready.yaml"),
    ("bell", "one_way", 3, QUALIFICATION / "stationary_nested_active_bell_cpu_one_way_3to1_ready.yaml"),
    ("bell", "two_way", 3, QUALIFICATION / "stationary_nested_active_bell_cpu_two_way_3to1_ready.yaml"),
    ("schar", "one_way", 3, QUALIFICATION / "stationary_nested_active_schar_cpu_one_way_3to1_ready.yaml"),
    ("schar", "two_way", 3, QUALIFICATION / "stationary_nested_active_schar_cpu_two_way_3to1_ready.yaml"),
)
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
FIELDS = ("rho", "theta", "p", "u", "v", "w")
CURRENCIES = ("dry_mass", "rho_theta", "x_momentum", "y_momentum", "z_momentum")
CLOSURE = re.compile(
    r"\[NESTING TWO-WAY CLOSURE\] currency=(\w+).*?mismatch=(\S+)"
    r".*?reflux=(\S+).*?dynamics_storage=(\S+).*?normalized_residual=(\S+)"
)


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )


def require_success(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed:\n{result.stdout}")
    for marker in ("[SCIENTIFIC INVALIDITY]", "Science profile rejected"):
        if marker in result.stdout:
            raise RuntimeError(f"{label} reported {marker}:\n{result.stdout}")


def require_rejection(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode == 0:
        raise RuntimeError(f"{label} unexpectedly succeeded:\n{result.stdout}")
    if "terrain" not in result.stdout.lower() and "nesting" not in result.stdout.lower():
        raise RuntimeError(f"{label} omitted terrain/nesting scope evidence:\n{result.stdout}")


def payload(
    output: Path, second: int, field: str, child: bool = False,
    child_size: int = 10,
) -> bytes:
    base = output / "nests/child_0000" if child else output
    path = base / f"step_{second:06d}" / f"{field}.npy"
    if not path.is_file():
        raise RuntimeError(f"missing output field: {path}")
    shape, raw, values = read_float32_npy(path)
    expected = (child_size, child_size, 24) if child else (16, 16, 24)
    if shape != expected:
        raise RuntimeError(f"{path} has shape {shape}, expected {expected}")
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError(f"{path} contains non-finite values")
    return raw


def require_nonflat_geometry(
    output: Path, second: int, child: bool, child_size: int,
) -> None:
    base = output / "nests/child_0000" if child else output
    manifest_path = base / f"step_{second:06d}" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vertical = manifest.get("grid", {}).get("vertical_geometry", {})
    if vertical.get("storage") != "versioned_npy_geometry":
        raise RuntimeError(f"{manifest_path} omitted versioned mapped geometry")
    center_path = vertical.get("center_path")
    if not isinstance(center_path, str) or not center_path:
        raise RuntimeError(f"{manifest_path} omitted its mapped center path")
    path = (manifest_path.parent / center_path).resolve()
    if not path.is_relative_to(base.resolve()):
        raise RuntimeError(f"{manifest_path} mapped geometry escaped its series")
    payload = path.read_bytes()
    if payload[:6] != b"\x93NUMPY" or payload[6] != 1:
        raise RuntimeError(f"invalid mapped geometry NPY: {path}")
    header_size = struct.unpack_from("<H", payload, 8)[0]
    header_start = 10
    header = ast.literal_eval(
        payload[header_start:header_start + header_size].decode("latin1").strip()
    )
    if header.get("descr") != "<f8" or header.get("fortran_order"):
        raise RuntimeError(f"expected row-major float64 mapped geometry: {path}")
    shape = tuple(int(value) for value in header["shape"])
    values = array.array("d")
    values.frombytes(payload[header_start + header_size:])
    expected = (child_size, child_size, 24) if child else (16, 16, 24)
    if shape != expected or any(not math.isfinite(value) for value in values):
        raise RuntimeError(f"invalid mapped geometry output: {path}")
    nz = shape[2]
    ground_centers = values[0::nz]
    if max(ground_centers) - min(ground_centers) <= 0.01:
        raise RuntimeError(f"{path} did not retain non-flat mapped geometry")


def closure_rows(
    output: str, label: str, require_two_way: bool,
) -> dict[str, list[tuple[float, ...]]]:
    observed: dict[str, list[tuple[float, ...]]] = {}
    for currency, mismatch, reflux, storage, normalized in CLOSURE.findall(output):
        values = tuple(float(value) for value in (mismatch, reflux, storage, normalized))
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"{label} published non-finite {currency} closure")
        if values[0] != values[1]:
            raise RuntimeError(f"{label} did not exactly reflux {currency}: {values}")
        if abs(values[3]) > 1.0e-8:
            raise RuntimeError(f"{label} exceeded {currency} normalized closure: {values[3]}")
        observed.setdefault(currency, []).append(values)
    if not require_two_way:
        if observed:
            raise RuntimeError(f"{label} published two-way closure for one-way feedback")
        return observed
    if set(observed) != set(CURRENCIES):
        raise RuntimeError(f"{label} omitted currencies: {set(CURRENCIES) - set(observed)}")
    if not any(abs(row[0]) > 0.0 for rows in observed.values() for row in rows):
        raise RuntimeError(f"{label} did not exercise nonzero mapped reflux")
    if not any(abs(row[2]) > 0.0 for row in observed["x_momentum"]):
        raise RuntimeError(f"{label} omitted mapped endpoint storage ownership")
    return observed


def ordinary_run(
    binary: Path, config: Path, terrain: str, feedback: str,
    ratio: int, output: Path, threads: int,
) -> tuple[dict[str, list[tuple[float, ...]]], str]:
    result = run([
        str(binary), "--headless", "--config", str(config),
        f"--outdir={output}", "--write-every=1", "--checkpoint-every=1",
        f"--threads={threads}", "--log-profile=normal", "--no-timing",
    ])
    require_success(result, f"active-terrain nested {threads}-thread run")
    for marker in (
        "Execution mode: qualified", f"stationary {feedback.replace('_', '-')} child",
        f"Terrain: {terrain}", "[COMPUTE] backend=cpu", "fallback=off",
        f"refinement={ratio}x{ratio}, time_ratio={ratio}",
    ):
        if marker not in result.stdout:
            raise RuntimeError(f"ordinary run omitted {marker!r}")
    return closure_rows(
        result.stdout, f"{threads}-thread run", feedback == "two_way",
    ), result.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    binary = args.binary.resolve()
    for path in (binary, *(profile[3] for profile in PROFILES)):
        if not path.is_file():
            raise RuntimeError(f"required engineering input is missing: {path}")

    source = PROFILES[1][3].read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="nimbus-active-terrain-nested-") as temporary:
        root = Path(temporary)
        mutations = (
            ("height", "    h0: 20\n", "    h0: 21\n"),
            ("placement", "  horizontal_placement: grid_attached\n", "  horizontal_placement: earth_fixed\n"),
            ("ratio", "    parent_ratio_x: 2\n", "    parent_ratio_x: 3\n"),
            ("backend", "    backend: cpu\n", "    backend: metal\n"),
        )
        for label, old, new in mutations:
            mutated = source.replace(old, new, 1)
            if mutated == source:
                raise RuntimeError(f"{label} mutation did not alter the fixture")
            config = root / f"invalid-{label}.yaml"
            config.write_text(mutated, encoding="utf-8")
            require_rejection(
                run([str(binary), "--validate-only", "--config", str(config)]),
                f"{label} validation guard",
            )
            rejected_output = root / f"invalid-{label}-output"
            require_rejection(
                run([str(binary), "--headless", "--duration=0", "--write-every=0",
                     f"--outdir={rejected_output}", "--config", str(config)]),
                f"{label} startup guard",
            )
            if (rejected_output / "nimbus_output.json").exists():
                raise RuntimeError(f"{label} rejection published output")

        for terrain, feedback, ratio, config in PROFILES:
            label = f"{terrain}-{feedback}-{ratio}to1"
            child_size = (8 if ratio == 2 else 9) + 2
            preflight = run([
                str(binary), "--validate-only", "--config", str(config),
            ])
            require_success(preflight, f"{label} preflight")
            if "[VALIDATE] PASS" not in preflight.stdout:
                raise RuntimeError(f"{label} preflight omitted PASS")

            serial = root / label / "serial"
            parallel = root / label / "parallel"
            resumed = root / label / "resumed"
            serial_closure, _ = ordinary_run(
                binary, config, terrain, feedback, ratio, serial, 1,
            )
            parallel_closure, _ = ordinary_run(
                binary, config, terrain, feedback, ratio, parallel, 4,
            )
            if serial_closure != parallel_closure:
                raise RuntimeError(
                    f"{label} mapped closure differs between workers"
                )
            evolved = False
            for field in FIELDS:
                initial = payload(parallel, 0, field)
                serial_final = payload(serial, 2, field)
                parallel_final = payload(parallel, 2, field)
                if serial_final != parallel_final:
                    raise RuntimeError(
                        f"{label} root {field} differs between workers"
                    )
                if payload(serial, 2, field, True, child_size) != payload(
                    parallel, 2, field, True, child_size,
                ):
                    raise RuntimeError(
                        f"{label} child {field} differs between workers"
                    )
                evolved = evolved or initial != parallel_final
            if not evolved:
                raise RuntimeError(f"{label} did not evolve")
            require_nonflat_geometry(parallel, 2, False, child_size)
            require_nonflat_geometry(parallel, 2, True, child_size)
            for level in (Path(), Path("nests/child_0000")):
                for ledger in LEDGERS:
                    serial_path = serial / level / ledger
                    parallel_path = parallel / level / ledger
                    if not serial_path.is_file() or not parallel_path.is_file():
                        raise RuntimeError(f"{label} {level} omitted {ledger}")
                    if serial_path.read_bytes() != parallel_path.read_bytes():
                        raise RuntimeError(f"{label} {level} {ledger} differs between workers")
                serial_checks = require_report_owner_identities(serial / level)
                parallel_checks = require_report_owner_identities(parallel / level)
                if serial_checks == 0 or serial_checks != parallel_checks:
                    raise RuntimeError(f"{label} {level} owner-identity coverage is incomplete")
                with (parallel / level / "radiation_budget.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                expected_steps = 4 if level == Path() else 4 * ratio
                if [int(row["step"]) for row in rows] != list(range(1, expected_steps + 1)):
                    raise RuntimeError(f"{label} {level} mixes radiation step histories")
                times = [float(row["sim_time_s"]) for row in rows]
                if any(right <= left for left, right in zip(times, times[1:])):
                    raise RuntimeError(f"{label} {level} mixes radiation clocks")

            checkpoint = (
                parallel /
                "checkpoints/checkpoint_step_000000002.nimbus"
            )
            if (not checkpoint.is_file() or
                    not Path(f"{checkpoint}.json").is_file()):
                raise RuntimeError(
                    f"{label} omitted synchronized midpoint checkpoint"
                )
            continuation = run([
                str(binary), "--headless", "--config", str(config),
                f"--resume={checkpoint}", "--duration=2", "--write-every=1",
                "--checkpoint-every=0", f"--outdir={resumed}", "--threads=4",
                "--log-profile=normal", "--no-timing",
            ])
            require_success(continuation, f"{label} checkpoint continuation")
            if "[CHECKPOINT] resumed step=2, time_s=1" not in continuation.stdout:
                raise RuntimeError(f"{label} did not restore its midpoint")
            resumed_closure = closure_rows(
                continuation.stdout, f"{label} continuation",
                feedback == "two_way",
            )
            if feedback == "two_way":
                for currency in CURRENCIES:
                    if resumed_closure[currency] != parallel_closure[currency][
                        -len(resumed_closure[currency]):
                    ]:
                        raise RuntimeError(
                            f"{label} {currency} closure changed across restart"
                        )
            for field in FIELDS:
                if payload(parallel, 2, field) != payload(resumed, 1, field):
                    raise RuntimeError(f"{label} root {field} restart differs")
                if payload(parallel, 2, field, True, child_size) != payload(
                    resumed, 1, field, True, child_size,
                ):
                    raise RuntimeError(f"{label} child {field} restart differs")
            for level in (Path(), Path("nests/child_0000")):
                require_report_owner_identities(resumed / level)
                require_restart_suffix(parallel / level, resumed / level)

    print(
        "[active-terrain-nested-ready] PASS: stationary isotropic 2:1/3:1 Bell/Schär "
        "one-/two-way CPU nesting retained non-flat mapped geometry, closed "
        "all applicable currencies, matched 1/4 workers and restart bitwise, "
        "and rejected neighboring profiles"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
