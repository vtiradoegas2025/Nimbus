#!/usr/bin/env python3
"""Bounded public Lilly-Smagorinsky selection, closure and continuation."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.qualification.checkpointing.qualify_vulkan_restart import (
    LEDGERS, require_report_owner_identities, require_restart_suffix,
)
from npy_support import read_float32_npy

CONFIG = ROOT / "tests/fixtures/configs/qualification/cgrid_richardson_smagorinsky_ready.yaml"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(binary, config, output, threads, resume=None):
    command = [str(binary), "--headless", "--config", str(config), "--duration=2",
               "--write-every=1", "--checkpoint-every=1", f"--threads={threads}",
               f"--checkpoint-dir={output}/checkpoints", f"--outdir={output}", "--log-profile=normal"]
    if resume is not None:
        command.append(f"--resume={resume}")
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    output.with_suffix(".log").write_text(result.stdout+result.stderr)
    require(result.returncode == 0, result.stdout[-3000:]+result.stderr)
    return result.stdout


def preflight(binary, config):
    return subprocess.run([str(binary), "--validate-only", "--config", str(config)],
                          cwd=ROOT, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    temporary = tempfile.TemporaryDirectory(prefix="nimbus-richardson-") if args.output is None else None
    root = Path(temporary.name) if temporary else args.output.resolve()
    require(not root.exists() or not any(root.iterdir()), "evidence directory must be empty")
    root.mkdir(parents=True, exist_ok=True)
    admitted = preflight(binary, CONFIG)
    require(admitted.returncode == 0 and "[VALIDATE] PASS" in admitted.stdout,
            admitted.stdout+admitted.stderr)
    source = CONFIG.read_text()
    for name, old, new in (
        ("backend", "backend: cpu", "backend: metal"),
        ("mode", "mode: 3d", "mode: 2d"),
        ("prandtl", "Pr_t: 0.3333333333333333", "Pr_t: 0.7"),
        ("coefficient", "Cs: 0.23", "Cs: 0.18"),
        ("dynamic", "dynamic_coefficient: false", "dynamic_coefficient: true"),
    ):
        require(old in source, f"missing negative fixture anchor {old}")
        path = root/f"invalid-{name}.yaml"
        path.write_text(source.replace(old, new, 1))
        require(preflight(binary, path).returncode != 0, f"unsupported profile admitted: {name}")
    one, four, resumed = (root/name for name in ("one", "four", "resumed"))
    for output, threads in ((one, 1), (four, 4)):
        log = run(binary, CONFIG, output, threads)
        require("Stability correction = Lilly equilibrium" in log, "Lilly producer was not selected")
        require("backend=cpu" in log and "fallback=off" in log, "wrong compute ownership")
        require(require_report_owner_identities(output) > 0, "energy owner checks did not execute")
        for field in ("rho", "p", "theta", "qv", "u", "v", "w"):
            shape, _, values = read_float32_npy(output/f"step_000002/{field}.npy")
            require(shape == (12,12,16) and all(math.isfinite(x) for x in values), f"invalid {field}")
        with (output/"thermodynamic_source_budget.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        require(any(float(row["sgs_committed_momentum_work_j"]) != 0.0 for row in rows), "SGS momentum owner is inert")
    for name in LEDGERS:
        require((one/name).read_bytes() == (four/name).read_bytes(), f"thread-dependent ledger: {name}")
    for step in (0,1,2):
        for path in (one/f"step_{step:06d}").glob("*.npy"):
            require(path.read_bytes() == (four/path.relative_to(one)).read_bytes(), f"thread-dependent field: {path.name}")
    checkpoints = sorted((one/"checkpoints").glob("checkpoint_step_*.nimbus"))
    require(len(checkpoints) == 2, "missing checkpoint sequence")
    for path in checkpoints:
        require(path.read_bytes() == (four/"checkpoints"/path.name).read_bytes(), "thread-dependent checkpoint")
    run(binary, CONFIG, resumed, 4, checkpoints[0])
    require_restart_suffix(one, resumed)
    require(require_report_owner_identities(resumed) > 0, "restart energy checks did not execute")
    for path in (one/"step_000002").glob("*.npy"):
        require(path.read_bytes() == (resumed/"step_000001"/path.name).read_bytes(), f"restart changed {path.name}")
    require(checkpoints[-1].read_bytes() == (resumed/"checkpoints"/checkpoints[-1].name).read_bytes(), "restart checkpoint differs")
    neutral = root/"neutral.yaml"
    neutral.write_text(source.replace("stability_correction: ri", "stability_correction: none", 1))
    neutral_output = root/"neutral"
    run(binary, neutral, neutral_output, 1)
    changed = [name for name in ("u", "v", "theta")
               if (one/f"step_000002/{name}.npy").read_bytes() != (neutral_output/f"step_000002/{name}.npy").read_bytes()]
    require(changed, "Richardson correction did not change the coupled solution")
    (root/"result.json").write_text(json.dumps({
        "status": "pass", "model": "ri_lilly_v1", "threads": [1,4], "duration_s": 2,
        "restart": "byte-identical", "neutral_control_changed": changed,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
    }, indent=2)+"\n")
    print("[richardson-ready] PASS: public Lilly selection, active conservative SGS, exact 1/4 fields/ledgers/checkpoints, restart, neutral control and negative admission")


if __name__ == "__main__":
    main()
