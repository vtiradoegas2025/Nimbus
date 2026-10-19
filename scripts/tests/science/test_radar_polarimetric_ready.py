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
CONFIG = ROOT / "tests/fixtures/configs/qualification/cgrid_radar_polarimetric_ready.yaml"
FIELDS = ("p", "rho", "theta", "u", "v", "w", "qv", "qc", "qr", "zdr", "kdp", "rhohv")


def run(binary: Path, args: list[str], threads: int = 1):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_DYNAMIC"] = "FALSE"
    return subprocess.run([str(binary), *args], cwd=ROOT, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def require_success(result, label: str) -> None:
    if result.returncode or "[SCIENTIFIC INVALIDITY]" in result.stdout:
        raise RuntimeError(f"{label} failed:\n{result.stdout}")


def final_frame(path: Path) -> Path:
    frames = sorted(item for item in path.glob("step_*") if item.is_dir())
    if not frames:
        raise RuntimeError(f"no output frames in {path}")
    return frames[-1]


def require_identity(left: Path, right: Path, label: str) -> None:
    lhs, rhs = final_frame(left), final_frame(right)
    for field in FIELDS:
        if (lhs / f"{field}.npy").read_bytes() != (rhs / f"{field}.npy").read_bytes():
            raise RuntimeError(f"{label}: {field} is not bitwise identical")


def require_products(path: Path) -> None:
    frame = final_frame(path)
    values = {}
    for field in ("zdr", "kdp", "rhohv"):
        _, _, values[field] = read_float32_npy(frame / f"{field}.npy")
        if not all(math.isfinite(value) for value in values[field]):
            raise RuntimeError(f"{field} contains non-finite samples")
    if min(values["zdr"]) < 0.0 or max(values["zdr"]) > 6.0:
        raise RuntimeError("ZDR is outside the maintained dB range")
    if min(values["kdp"]) < 0.0 or max(values["kdp"]) > 4.0:
        raise RuntimeError("KDP is outside the maintained deg/km range")
    if min(values["rhohv"]) < 0.8 or max(values["rhohv"]) > 1.0:
        raise RuntimeError("rhohv is outside its dimensionless range")
    if max(values["zdr"]) <= 0.0 or max(values["kdp"]) <= 0.0 or min(values["rhohv"]) >= 1.0:
        raise RuntimeError("bounded evolution did not exercise active rain polarimetry")
    provenance = json.loads((frame / "manifest.json").read_text())["radar_provenance"]
    if provenance["products"] != ["zdr", "kdp", "rhohv"]:
        raise RuntimeError("manifest radar products do not match the selected tuple")
    if provenance["polarimetric_operator"] != "zhang_gao_du_2021_rain_mp_v1":
        raise RuntimeError("manifest omitted maintained polarimetric provenance")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    args = parser.parse_args()
    binary = args.binary.resolve()
    source = CONFIG.read_text()
    with tempfile.TemporaryDirectory(prefix="nimbus-radar-polarimetric-") as temp:
        root = Path(temp)
        result = run(binary, ["--validate-only", "--config", str(CONFIG)])
        require_success(result, "preflight")
        outputs = {}
        for threads in (1, 4):
            output = root / f"threads-{threads}"
            result = run(binary, ["--headless", "--config", str(CONFIG),
                                  f"--outdir={output}", "--write-every=4",
                                  "--checkpoint-every=2", f"--threads={threads}"], threads)
            require_success(result, f"{threads}-thread evolution")
            outputs[threads] = output
        require_identity(outputs[1], outputs[4], "thread determinism")
        require_products(outputs[4])

        checkpoint = outputs[4] / "checkpoints/checkpoint_step_000000010.nimbus"
        resumed = root / "resumed"
        result = run(binary, ["--headless", "--config", str(CONFIG),
                              f"--resume={checkpoint}", "--duration=4",
                              "--write-every=4", "--checkpoint-every=0",
                              f"--outdir={resumed}", "--threads=4"], 4)
        require_success(result, "checkpoint continuation")
        require_identity(outputs[4], resumed, "checkpoint continuation")
        require_products(resumed)

        for label, old, new in (
            ("missing-rhohv", "  products: [zdr, kdp, rhohv]\n", "  products: [zdr, kdp]\n"),
            ("exploratory-tier", "  zdr_tier: zhang2021_rain_mp\n", "  zdr_tier: polarimetric_fo\n"),
            ("mixed-phase-scheme", "  scheme: kessler\n", "  scheme: lin\n"),
        ):
            changed = root / f"invalid-{label}.yaml"
            changed.write_text(source.replace(old, new, 1))
            for mode in (["--validate-only"], ["--headless", "--duration=0", "--write-every=0"]):
                result = run(binary, [*mode, "--config", str(changed)])
                if result.returncode == 0 or "radar" not in result.stdout.lower():
                    raise RuntimeError(f"{label} did not fail closed:\n{result.stdout}")

    print("[radar-polarimetric-ready] PASS: Zhang-2021 rain-only ZDR/KDP/rhohv, units, ranges, 1/4-thread identity, restart, provenance, and negative scope gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
