#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import sys
import tempfile

from npy_support import read_float32_npy


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

SOURCE_CONFIG = (
    ROOT
    / "tests/fixtures/configs/qualification/"
    "cgrid_stationary_bell_pbl_implicit_ready.yaml"
)
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
FIELDS = ("u", "v", "w", "theta", "p", "rho", "qv", "qr")


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def require_success(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed:\n{result.stdout}")
    for marker in ("[SCIENTIFIC INVALIDITY]", "Science profile rejected"):
        if marker in result.stdout:
            raise RuntimeError(f"{label} reported {marker}:\n{result.stdout}")


def field_payload(output: Path, second: int, field: str):
    path = output / f"step_{second:06d}" / f"{field}.npy"
    if not path.is_file():
        raise RuntimeError(f"missing {field} output: {path}")
    shape, raw, values = read_float32_npy(path)
    if shape != (12, 12, 18):
        raise RuntimeError(f"{field} has unexpected shape {shape}")
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError(f"{field} contains a non-finite value")
    return raw


def execute(binary: Path, config: Path, output: Path, threads: int, label: str):
    result = run([
        str(binary),
        "--headless",
        "--config",
        str(config),
        f"--outdir={output}",
        "--write-every=1",
        f"--threads={threads}",
        "--log-profile=normal",
    ])
    require_success(result, label)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    binary = args.binary.resolve()
    for path in (binary, SOURCE_CONFIG):
        if not path.is_file():
            raise RuntimeError(f"required Earth-fixed startup input is missing: {path}")

    source = SOURCE_CONFIG.read_text(encoding="utf-8")
    earth_source = source.replace(
        "  scheme: bell\n",
        "  scheme: bell\n"
        "  horizontal_placement: earth_fixed\n"
        "  feature_center_earth_x_m: 3000\n"
        "  feature_center_earth_y_m: 3000\n",
        1,
    ).replace(
        "    mode: qualified\n", "    mode: exploratory\n", 1
    ).replace(
        "    mode: qualified\n",
        "    mode: qualified\n    acknowledge_maturity_limits: false\n",
        1,
    )
    if earth_source == source:
        raise RuntimeError("Earth-fixed startup mutation did not alter the fixture")

    with tempfile.TemporaryDirectory(prefix="nimbus-earth-fixed-startup-") as temporary:
        root = Path(temporary)
        earth_config = root / "earth-fixed.yaml"
        earth_config.write_text(earth_source, encoding="utf-8")
        preflight = run([
            str(binary), "--validate-only", "--config", str(earth_config)
        ])
        require_success(preflight, "Earth-fixed strict preflight")
        if "[VALIDATE] PASS" not in preflight.stdout:
            raise RuntimeError("Earth-fixed strict preflight omitted PASS")

        reference = root / "grid-attached"
        earth_one = root / "earth-one"
        earth_four = root / "earth-four"
        resumed = root / "earth-resumed"
        execute(binary, SOURCE_CONFIG, reference, 4, "grid-attached reference")
        one_result = execute(binary, earth_config, earth_one, 1, "Earth-fixed one-worker run")
        four_result = execute(binary, earth_config, earth_four, 4, "Earth-fixed four-worker run")
        marker = "placement=earth_fixed, generation=1"
        for label, result in (("one-worker", one_result), ("four-worker", four_result)):
            if marker not in result.stdout:
                raise RuntimeError(f"Earth-fixed {label} run omitted placement provenance")

        for field in FIELDS:
            reference_raw = field_payload(reference, 2, field)
            one_raw = field_payload(earth_one, 2, field)
            four_raw = field_payload(earth_four, 2, field)
            if one_raw != four_raw:
                raise RuntimeError(f"Earth-fixed {field} differs between workers")
            if four_raw != reference_raw:
                raise RuntimeError(
                    f"Earth-fixed {field} does not reduce to grid-attached geometry at the same origin"
                )

        checkpoint = earth_four / "checkpoints/checkpoint_step_000000005.nimbus"
        if not checkpoint.is_file() or not Path(f"{checkpoint}.json").is_file():
            raise RuntimeError("Earth-fixed run omitted its midpoint checkpoint pair")
        continuation = run([
            str(binary),
            "--headless",
            "--config",
            str(earth_config),
            f"--resume={checkpoint}",
            "--duration=2",
            "--write-every=1",
            "--checkpoint-every=0",
            f"--outdir={resumed}",
            "--threads=4",
            "--log-profile=normal",
        ])
        require_success(continuation, "Earth-fixed midpoint continuation")
        if "[CHECKPOINT] resumed step=5, time_s=1" not in continuation.stdout:
            raise RuntimeError("Earth-fixed continuation did not restore the midpoint")
        for field in FIELDS:
            if field_payload(earth_four, 2, field) != field_payload(resumed, 1, field):
                raise RuntimeError(f"Earth-fixed {field} restart is not bitwise exact")

    print(
        "[earth-fixed-terrain-startup] PASS: strict Earth-fixed placement reached "
        "ordinary runtime after frame initialization, reduced exactly at the common "
        "origin, matched one/four workers, and resumed bitwise"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
