"""Shared strict field, ledger and execution checks for native admission gates."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import subprocess

from scripts.tests.science.npy_support import read_float32_npy

ROOT = Path(__file__).resolve().parents[2]

FIELDS = ("p", "rho", "theta", "u", "v", "w", "qv", "qr")

LEDGERS = ("budget.csv", "energy_budget.csv", "moisture_budget.csv",
           "momentum_budget.csv", "radiation_budget.csv", "thermodynamic_source_budget.csv")

NATIVE_TVD_OPERATION = "advection/advect_conservative_update_cartesian_cgrid.comp"

def identity(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"expected one configuration anchor: {old!r}")
    return source.replace(old, new, 1)

def compare_fields(reference: Path, candidate: Path, exact: bool) -> dict:
    left, right = sorted(reference.glob("step_*"))[-1], sorted(candidate.glob("step_*"))[-1]
    result = {}
    for field in FIELDS:
        a, b = left / f"{field}.npy", right / f"{field}.npy"
        shape, _, expected = read_float32_npy(a)
        actual_shape, _, actual = read_float32_npy(b)
        if shape != actual_shape or not expected or any(
                not math.isfinite(value) for values in (expected, actual) for value in values):
            raise RuntimeError(f"invalid {field} field")
        scale = max(map(abs, expected))
        ulp = math.ldexp(1, math.frexp(scale)[1] - 24) if scale else math.ldexp(1, -149)
        limit = 32 * 10 * ulp
        if field in ("u", "v", "w"):
            limit = max(limit, 10 * 5e-5)
        error = max(abs(x - y) for x, y in zip(expected, actual))
        passed = a.read_bytes() == b.read_bytes() if exact else error <= limit
        result[field] = {"error": error, "limit": 0 if exact else limit, "passed": passed}
    return result

def run(binary: Path, config: Path, output: Path, threads: int,
        duration: int = 1, checkpoint_every: int = 0, resume: Path | None = None) -> dict:
    command = [str(binary), "--headless", "--config", str(config), f"--duration={duration}",
               f"--outdir={output}", "--write-every=1", f"--checkpoint-every={checkpoint_every}",
               f"--threads={threads}", "--log-profile=normal", "--timing"]
    if resume is not None:
        command.append(f"--resume={resume}")
    result = subprocess.run(
        command,
        cwd=ROOT, env={**os.environ, "OMP_NUM_THREADS": str(threads), "OMP_DYNAMIC": "FALSE"},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
    log = output.with_suffix(".log")
    log.write_text(result.stdout)
    return {"returncode": result.returncode, "log": identity(log),
            "invalidScience": "[SCIENTIFIC INVALIDITY]" in result.stdout,
            "errors": [line for line in result.stdout.splitlines()
                       if "ERROR" in line or "INVALIDITY" in line]}

def ledger_suffix(continuous: Path, resumed: Path) -> dict[str, bool]:
    result = {}
    for name in LEDGERS:
        reference = (continuous / name).read_text().splitlines()
        candidate = (resumed / name).read_text().splitlines()
        result[name] = bool(reference and len(candidate) > 1 and reference[0] == candidate[0]
                            and reference[-(len(candidate) - 1):] == candidate[1:])
    return result
