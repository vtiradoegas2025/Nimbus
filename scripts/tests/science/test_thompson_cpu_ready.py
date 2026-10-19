#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "tests/fixtures/configs/qualification/cgrid_thompson_cpu_ready.yaml"
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
STATE_FIELDS = (
    "p", "rho", "theta", "u", "v", "w", "qv", "qc", "qr", "qi", "qs", "qg",
)
LEDGERS = (
    "budget.csv", "energy_budget.csv", "moisture_budget.csv",
    "momentum_budget.csv", "radiation_budget.csv",
    "thermodynamic_source_budget.csv",
)
ARTIFACTS = {
    "ccn_activation.ntact": "6f72d69fcedb89b3d9e978e3f5b2a0eda204c5652114a4f4e7e0d033a73f66bf",
    "collection_v1.ntctbl": "b01efcfe79208b820ea43d8c0d59b4557fb94706586c600dbd43b70e36e83547",
    "cold_process_v1.ntcold": "c4bdf82a41bc22cd4bd2f7c6c2c5f7f6d75d527941437ad57e00bbbb53a9df53",
}


def run(command: list[str], threads: int = 1,
        cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["OMP_DYNAMIC"] = "FALSE"
    return subprocess.run(
        command, cwd=cwd, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )


def require_success(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed:\n{result.stdout}")
    for marker in ("[SCIENTIFIC INVALIDITY]", "Science profile rejected"):
        if marker in result.stdout:
            raise RuntimeError(f"{label} reported {marker}:\n{result.stdout}")


def final_frame(output: Path) -> Path:
    frames = sorted(path for path in output.glob("step_*") if path.is_dir())
    if not frames:
        raise RuntimeError(f"{output}: no published frames")
    return frames[-1]


def require_state_identity(reference: Path, candidate: Path, label: str) -> None:
    lhs_frame = final_frame(reference)
    rhs_frame = final_frame(candidate)
    for field in STATE_FIELDS:
        if (lhs_frame / f"{field}.npy").read_bytes() != (
                rhs_frame / f"{field}.npy").read_bytes():
            raise RuntimeError(f"{label}: final {field} bytes differ")


def require_ledger_identity(reference: Path, candidate: Path, label: str) -> None:
    for filename in LEDGERS:
        if (reference / filename).read_bytes() != (candidate / filename).read_bytes():
            raise RuntimeError(f"{label}: {filename} bytes differ")


def require_restart_suffix(continuous: Path, resumed: Path) -> None:
    for filename in LEDGERS:
        reference = (continuous / filename).read_text(encoding="utf-8").splitlines()
        candidate = (resumed / filename).read_text(encoding="utf-8").splitlines()
        if not reference or len(candidate) < 2 or reference[0] != candidate[0]:
            raise RuntimeError(f"restart: malformed {filename}")
        if reference[-(len(candidate) - 1):] != candidate[1:]:
            raise RuntimeError(f"restart: {filename} is not an exact suffix")


def npy_float_values(path: Path) -> tuple[float, ...]:
    data = path.read_bytes()
    if data[:6] != b"\x93NUMPY":
        raise RuntimeError(f"{path}: invalid NPY magic")
    major = data[6]
    if major == 1:
        header_length = struct.unpack_from("<H", data, 8)[0]
        offset = 10 + header_length
    elif major in (2, 3):
        header_length = struct.unpack_from("<I", data, 8)[0]
        offset = 12 + header_length
    else:
        raise RuntimeError(f"{path}: unsupported NPY version")
    payload = data[offset:]
    if len(payload) % 4 != 0:
        raise RuntimeError(f"{path}: unexpected non-float32 payload")
    return tuple(value[0] for value in struct.iter_unpack("<f", payload))


def require_process_activity(output: Path) -> None:
    with (output / "thermodynamic_source_budget.csv").open(
            encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("Thompson bounded run published no source ledger")
    activity = max(
        abs(float(row[f"microphysics_{name}_requested_kg"]))
        for row in rows for name in ("qv", "qc", "qr", "qi", "qs", "qg")
    )
    if activity <= 0.0:
        raise RuntimeError("Thompson bounded run exercised no microphysics process")
    if any(int(row["microphysics_conservation_active"]) != 1 for row in rows):
        raise RuntimeError("Thompson bounded run omitted conservation admission")
    if any(
        abs(float(row["microphysics_conservation_residual_kg"]))
        / max(abs(float(row["microphysics_baseline_water_kg"])), 1.0e-30)
        > 1.0e-6 for row in rows
    ):
        raise RuntimeError("Thompson bounded run exceeded water closure")
    frame = final_frame(output)
    frozen_maximum = max(
        abs(value)
        for field in ("qi", "qs", "qg")
        for value in npy_float_values(frame / f"{field}.npy")
    )
    if frozen_maximum <= 0.0:
        raise RuntimeError("Thompson bounded run produced no frozen hydrometeor")


def run_profile(binary: Path, output: Path, threads: int) -> None:
    result = run([
        str(binary), "--headless", "--config", str(CONFIG),
        f"--outdir={output}", "--write-every=4", "--checkpoint-every=2",
        f"--threads={threads}", "--log-profile=normal",
    ], threads)
    require_success(result, f"Thompson {threads}-thread evolution")
    for marker in (
        "Execution mode: qualified",
        "Initialized microphysics scheme: thompson",
        "Initialized radiation scheme: rrtmg",
        "Initialized boundary layer scheme: ysu",
        "[COMPUTE] backend=cpu",
        "fallback=off",
    ):
        if marker not in result.stdout:
            raise RuntimeError(
                f"Thompson {threads}-thread evolution omitted {marker!r}")


def require_scope_rejection(binary: Path, source: str, old: str, new: str,
                            root: Path, label: str) -> None:
    if old not in source:
        raise RuntimeError(f"{label}: mutation anchor is missing")
    config = root / f"{label}.yaml"
    config.write_text(source.replace(old, new, 1), encoding="utf-8")
    expected = "microphysics.scheme=thompson requires a maintained CPU profile or the native C-grid Thompson reference"
    for mode, arguments in (
        ("validation", ["--validate-only"]),
        ("startup", ["--headless", "--duration=0", "--write-every=0",
                     f"--outdir={root / (label + '-output')}"]),
    ):
        result = run([str(binary), *arguments, "--config", str(config)])
        if result.returncode == 0 or expected not in result.stdout:
            raise RuntimeError(
                f"{label} {mode} did not fail with exact scope:\n{result.stdout}")
    if (root / f"{label}-output/nimbus_output.json").exists():
        raise RuntimeError(f"{label}: rejected startup published output")


def require_packaged_data_identity(binary: Path, source: str, root: Path) -> None:
    data_root = ROOT / "data/thompson"
    for filename, expected in ARTIFACTS.items():
        path = data_root / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"packaged Thompson hash mismatch: {path}")
    isolated = root / "isolated"
    (isolated / "data").mkdir(parents=True)
    (isolated / "data/rrtmg").symlink_to(ROOT / "data/rrtmg")
    (isolated / "tests").symlink_to(ROOT / "tests")
    isolated_config = isolated / "thompson.yaml"
    isolated_config.write_text(source, encoding="utf-8")
    result = run([
        str(binary), "--headless", "--duration=0", "--write-every=0",
        f"--outdir={isolated / 'missing-output'}", "--config", str(isolated_config),
    ], cwd=isolated)
    recovery = "run `make thompson-data` to prepare the pinned tables"
    if result.returncode == 0 or recovery not in result.stdout:
        raise RuntimeError(f"missing packaged data did not fail closed:\n{result.stdout}")
    isolated_tables = isolated / "data/thompson"
    isolated_tables.mkdir()
    for filename in ARTIFACTS:
        shutil.copy2(data_root / filename, isolated_tables / filename)
    corrupt = isolated_tables / "cold_process_v1.ntcold"
    payload = bytearray(corrupt.read_bytes())
    payload[-1] ^= 1
    corrupt.write_bytes(payload)
    result = run([
        str(binary), "--headless", "--duration=0", "--write-every=0",
        f"--outdir={isolated / 'corrupt-output'}", "--config", str(isolated_config),
    ], cwd=isolated)
    if result.returncode == 0 or recovery not in result.stdout:
        raise RuntimeError(f"corrupt packaged data did not fail closed:\n{result.stdout}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    binary = args.binary.resolve()
    required = [binary, CONFIG, ROOT / "data/rrtmg/rrtmg_lw.nc",
                ROOT / "data/rrtmg/rrtmg_sw.nc"]
    required.extend(ROOT / "data/thompson" / name for name in ARTIFACTS)
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"required Thompson input is missing: {path}")
    source = CONFIG.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="nimbus-thompson-cpu-ready-") as temporary:
        root = Path(temporary)
        validated = run([str(binary), "--validate-only", "--config", str(CONFIG)])
        require_success(validated, "Thompson qualified preflight")
        if "[VALIDATE] PASS" not in validated.stdout:
            raise RuntimeError("Thompson qualified preflight omitted PASS")
        one = root / "one-thread"
        four = root / "four-threads"
        run_profile(binary, one, 1)
        run_profile(binary, four, 4)
        require_state_identity(one, four, "thread determinism")
        require_ledger_identity(one, four, "thread determinism")
        require_process_activity(four)
        checkpoint = four / "checkpoints/checkpoint_step_000000010.nimbus"
        if not checkpoint.is_file() or not Path(f"{checkpoint}.json").is_file():
            raise RuntimeError("Thompson run omitted split checkpoint")
        resumed = root / "resumed"
        result = run([
            str(binary), "--headless", "--config", str(CONFIG),
            f"--resume={checkpoint}", "--duration=4", "--write-every=4",
            "--checkpoint-every=0", f"--outdir={resumed}",
            "--threads=4", "--log-profile=normal",
        ], 4)
        require_success(result, "Thompson checkpoint continuation")
        if "[CHECKPOINT] resumed step=10, time_s=2" not in result.stdout:
            raise RuntimeError("Thompson continuation omitted restored step")
        require_state_identity(four, resumed, "restart continuation")
        require_restart_suffix(four, resumed)
        require_scope_rejection(
            binary, source,
            "    water_friendly_aerosol_surface_m3: 350000000\n",
            "    water_friendly_aerosol_surface_m3: 340000000\n",
            root, "noncanonical-thompson-aerosol")
        require_scope_rejection(
            binary, source, "  scheme: ysu\n", "  scheme: slab\n",
            root, "noncanonical-boundary-layer")
        require_scope_rejection(
            binary, source,
            "    allow_fallback: false\n", "    allow_fallback: true\n",
            root, "fallback-enabled")
        require_packaged_data_identity(binary, source, root)
    print(
        "[thompson-cpu-ready] PASS: exact CPU profile, packaged identity, "
        "nonzero cold processes, 1/4 identity, restart, and negative gates passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
