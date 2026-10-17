#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from npy_support import read_float32_npy


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "tests/fixtures/configs/qualification/cgrid_radar_ready.yaml"
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
FIELDS = (
    "p", "rho", "theta", "u", "v", "w", "qv", "qr",
    "reflectivity_dbz", "radial_velocity",
)
LEDGERS = (
    "budget.csv", "energy_budget.csv", "moisture_budget.csv",
    "momentum_budget.csv", "radiation_budget.csv",
    "thermodynamic_source_budget.csv",
)


def run(binary: Path, arguments: list[str], threads: int = 1):
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["OMP_DYNAMIC"] = "FALSE"
    return subprocess.run(
        [str(binary), *arguments], cwd=ROOT, env=environment,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False,
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


def require_identity(reference: Path, candidate: Path, label: str,
                     compare_ledgers: bool = True) -> None:
    lhs = final_frame(reference)
    rhs = final_frame(candidate)
    for field in FIELDS:
        if (lhs / f"{field}.npy").read_bytes() != (
                rhs / f"{field}.npy").read_bytes():
            _, _, lhs_values = read_float32_npy(lhs / f"{field}.npy")
            _, _, rhs_values = read_float32_npy(rhs / f"{field}.npy")
            max_difference = max(
                abs(left - right)
                for left, right in zip(lhs_values, rhs_values))
            raise RuntimeError(
                f"{label}: final {field} bytes differ; max_abs={max_difference}")
    if compare_ledgers:
        for ledger in LEDGERS:
            if (reference / ledger).read_bytes() != (candidate / ledger).read_bytes():
                raise RuntimeError(f"{label}: {ledger} bytes differ")


def require_restart_suffix(continuous: Path, resumed: Path) -> None:
    for ledger in LEDGERS:
        reference = (continuous / ledger).read_text().splitlines()
        candidate = (resumed / ledger).read_text().splitlines()
        if not reference or len(candidate) < 2 or candidate[0] != reference[0]:
            raise RuntimeError(f"restart: malformed {ledger}")
        if reference[-(len(candidate) - 1):] != candidate[1:]:
            raise RuntimeError(f"restart: {ledger} is not an exact suffix")


def require_radar_products(output: Path) -> None:
    frame = final_frame(output)
    _, _, reflectivity = read_float32_npy(frame / "reflectivity_dbz.npy")
    _, _, velocity = read_float32_npy(frame / "radial_velocity.npy")
    if not all(math.isfinite(value) for value in reflectivity) or not all(
            math.isfinite(value) for value in velocity):
        raise RuntimeError("radar products contain non-finite values")
    if min(reflectivity) < -30.0 or max(reflectivity) > 120.0:
        raise RuntimeError("reflectivity_dbz is outside its publication units/range")
    if max(abs(value) for value in velocity) > 325.0:
        raise RuntimeError("radial_velocity is outside its m/s range")
    if max(reflectivity) <= -29.9:
        raise RuntimeError("bounded run did not produce a resolved radar echo")
    if max(abs(value) for value in velocity) <= 0.1:
        raise RuntimeError("bounded run did not produce line-of-sight velocity")

    manifest = json.loads((frame / "manifest.json").read_text())
    provenance = manifest.get("radar_provenance")
    if provenance is None:
        raise RuntimeError("radar output omitted operator provenance")
    if provenance.get("products") != ["reflectivity_dbz", "radial_velocity"]:
        raise RuntimeError("radar provenance products do not match publication")
    if provenance.get("reflectivity_operator") != (
            "marshall_palmer_lfo83_smith1984_fast_da_v1"):
        raise RuntimeError("reflectivity provenance does not identify the maintained operator")
    if provenance.get("velocity_operator") != (
            "earth_relative_cgrid_line_of_sight_point_v1"):
        raise RuntimeError("velocity provenance does not identify the maintained operator")
    if provenance.get("instrument_location_earth_m") != [-1500, 750, 250]:
        raise RuntimeError("radar provenance lost Earth-relative instrument geometry")
    if not provenance.get("scatterer_correction"):
        raise RuntimeError("radar provenance lost scatterer-correction ownership")


def require_scope_rejection(binary: Path, source: str, old: str, new: str,
                            root: Path, label: str) -> None:
    changed = source.replace(old, new, 1)
    if changed == source:
        raise RuntimeError(f"{label}: mutation anchor is missing")
    config = root / f"invalid-{label}.yaml"
    config.write_text(changed)
    for mode, arguments in (
        ("validate", ["--validate-only"]),
        ("startup", ["--headless", "--duration=0", "--write-every=0",
                     f"--outdir={root / ('invalid-' + label)}"]),
    ):
        result = run(binary, [*arguments, "--config", str(config)])
        if result.returncode == 0 or "radar" not in result.stdout.lower():
            raise RuntimeError(
                f"{label} {mode} did not fail closed as radar scope:\n{result.stdout}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    binary = args.binary.resolve()
    for path in (binary, CONFIG):
        if not path.is_file():
            raise RuntimeError(f"required radar input is missing: {path}")

    source = CONFIG.read_text()
    with tempfile.TemporaryDirectory(prefix="nimbus-radar-ready-") as temporary:
        root = Path(temporary)
        result = run(binary, ["--validate-only", "--config", str(CONFIG)])
        require_success(result, "radar preflight")
        if "[VALIDATE] PASS" not in result.stdout:
            raise RuntimeError("radar preflight omitted PASS")

        outputs: dict[int, Path] = {}
        for threads in (1, 4):
            output = root / f"threads-{threads}"
            result = run(binary, [
                "--headless", "--config", str(CONFIG), f"--outdir={output}",
                "--write-every=4", "--checkpoint-every=2",
                f"--threads={threads}", "--log-profile=normal",
            ], threads)
            require_success(result, f"radar {threads}-thread evolution")
            if "Initialized radar products: reflectivity=1 radial_velocity=1 zdr=0" not in result.stdout:
                raise RuntimeError("ordinary runtime did not select the exact radar pair")
            outputs[threads] = output

        require_identity(outputs[1], outputs[4], "thread determinism")
        require_radar_products(outputs[4])

        checkpoint = outputs[4] / "checkpoints/checkpoint_step_000000010.nimbus"
        resumed = root / "resumed"
        result = run(binary, [
            "--headless", "--config", str(CONFIG), f"--resume={checkpoint}",
            "--duration=4", "--write-every=4", "--checkpoint-every=0",
            f"--outdir={resumed}", "--threads=4", "--log-profile=normal",
        ], 4)
        require_success(result, "radar checkpoint continuation")
        if "[CHECKPOINT] resumed step=10, time_s=2" not in result.stdout:
            raise RuntimeError("radar continuation omitted the restore marker")
        require_identity(outputs[4], resumed, "restart continuation", False)
        require_restart_suffix(outputs[4], resumed)
        require_radar_products(resumed)

        require_scope_rejection(
            binary, source,
            "  products: [reflectivity_dbz, radial_velocity]\n",
            "  products: [reflectivity_dbz, zdr]\n",
            root, "synthetic-zdr")
        require_scope_rejection(
            binary, source, "  reflectivity_tier: fast_da\n",
            "  reflectivity_tier: psd_moment\n", root, "noncanonical-tier")
        require_scope_rejection(
            binary, source, "    x_m: -1500\n",
            "    x_m: -1400\n", root, "noncanonical-location")

        require_scope_rejection(
            binary, source,
            "  products: [reflectivity_dbz, radial_velocity]\n",
            "  products: [reflectivity_dbz, kdp]\n",
            root, "unsupported-product-tuple")

        mismatch = root / "checkpoint-mismatch.yaml"
        mismatch.write_text(source.replace(
            "    mode: qualified\n", "    mode: exploratory\n", 1
        ).replace(
            "    mode: qualified\n",
            "    mode: qualified\n    acknowledge_maturity_limits: false\n", 1
        ).replace("    x_m: -1500\n", "    x_m: -1400\n", 1))
        result = run(binary, [
            "--headless", "--config", str(mismatch), f"--resume={checkpoint}",
            "--duration=4", "--write-every=0", f"--outdir={root / 'mismatch'}",
        ])
        mismatch_output = result.stdout.lower()
        if result.returncode == 0 or (
                "fingerprint" not in mismatch_output and
                "checkpoint radar configuration" not in mismatch_output and
                "checkpoint scientific configuration" not in mismatch_output):
            raise RuntimeError(
                "checkpoint accepted different radar geometry:\n" + result.stdout)

    print(
        "[radar-ready] PASS: exact CPU reflectivity/velocity profile, units, "
        "geometry provenance, 1/4-thread identity, restart/config identity, "
        "and negative scope gates passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
