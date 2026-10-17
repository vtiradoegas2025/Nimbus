#!/usr/bin/env python3
"""Closed-wall SGS threading, restart and CPU/Vulkan field regression."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.qualification.domain_motion.qualify_domain_recycling import runtime_environment
from scripts.qualification.checkpointing.qualify_vulkan_restart import require_restart_suffix, LEDGERS
from test_nested_grid_recycling_ready import checkpoint_at
from npy_support import read_float32_npy

FIELDS = ("u", "v", "w", "theta", "qv", "rho", "p")
TUPLE = ("grid=c_grid|backend=cpu|qualified=true|microphysics=none|radiation=none|"
         "boundary_layer=none|turbulence=smagorinsky|chaos=initial_conditions|diffusion=explicit|"
         "terrain=none|dynamics=klemp_wilhelmson_split_cartesian_cgrid|advection=tvd|"
         "time_stepping=split_explicit|turbulence_mode=2d|surface=none|surface_fluxes=false|"
         "sounding=parametric|output=npy_3d\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frames(directory):
    return {round(json.loads(path.read_text())["simulation_time_s"], 8): path.parent
            for path in directory.glob("step_*/manifest.json")}


def require_identity(reference, candidate, resumed=False):
    expected, actual = frames(reference), frames(candidate)
    if 2 not in expected or 2 not in actual or (not resumed and expected.keys() != actual.keys()):
        raise RuntimeError("closed-wall SGS field history is incomplete")
    for time, directory in actual.items():
        if time not in expected:
            raise RuntimeError("closed-wall SGS continuation published an unexpected time")
        files = {path.name for path in expected[time].glob("*.npy")}
        if files != {path.name for path in directory.glob("*.npy")}:
            raise RuntimeError("closed-wall SGS field set changed")
        for name in files:
            if (expected[time] / name).read_bytes() != (directory / name).read_bytes():
                raise RuntimeError(f"closed-wall SGS field identity changed: {time}/{name}")


def qualify(binary, native_tool, root):
    root.mkdir(parents=True, exist_ok=False)
    materialized = subprocess.run([str(native_tool), "--yaml"], input=TUPLE, text=True,
                                  capture_output=True, check=True, cwd=ROOT)
    config = root / "closed.yaml"
    config.write_text(materialized.stdout)
    if any(f"{side}: rigid" not in materialized.stdout for side in ("west", "east", "south", "north")):
        raise RuntimeError("closed-wall reference no longer selects four rigid sides")
    if "adaptive_dt: false" not in materialized.stdout:
        raise RuntimeError("closed-wall reference requires a fixed timestep")
    timestep = float(re.search(r"^  dt_large: ([^\n]+)", materialized.stdout, re.MULTILINE)[1])
    shader_root = Path(runtime_environment("vulkan").get("TMV_SHADER_DIR", ROOT / "vulkan/shaders/compute"))
    paths = [binary, native_tool, config, Path(__file__).resolve(), *sorted(shader_root.rglob("*.spv"))]
    identities = {str(path): sha(path) for path in paths}

    def run(backend, label, threads, resume=None):
        output = root / f"{backend}-{label}"
        command = [str(binary), "--headless", "--config", str(config), f"--compute-backend={backend}",
                   "--duration=2", "--write-every=1", "--checkpoint-every=1", f"--threads={threads}",
                   f"--outdir={output}", "--no-timing"]
        if resume:
            command.append(f"--resume={resume}")
        environment = runtime_environment(backend)
        environment["OMP_DYNAMIC"] = "FALSE"
        result = subprocess.run(command, env=environment, cwd=ROOT, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        (root / f"{backend}-{label}.log").write_text(result.stdout)
        if result.returncode or "Execution mode: qualified" not in result.stdout or "fallback=off" not in result.stdout:
            raise RuntimeError(f"closed-wall {backend}/{label} failed public runtime execution")
        if f"[COMPUTE] backend={backend}" not in result.stdout:
            raise RuntimeError("closed-wall case did not select its requested backend")
        if backend == "vulkan" and "turbulence/smagorinsky_tendencies.comp:" not in result.stdout:
            raise RuntimeError("closed-wall case omitted native SGS transport")
        if backend == "vulkan":
            expected = round((1 if resume else 2) / timestep) * 6
            for stage in ("pressure", "momentum"):
                match = re.search(rf"acoustic/acoustic_{stage}_cartesian_cgrid\.comp:(\d+)", result.stdout)
                if match is None or int(match[1]) != expected:
                    raise RuntimeError(f"closed-wall strict validation omitted native acoustic {stage}")
        return output

    report = {}
    for backend in ("cpu", "vulkan"):
        serial = run(backend, "t1", 1)
        threaded = run(backend, "t4", 4)
        resumed = run(backend, "resumed", 4, checkpoint_at(serial / "checkpoints", 1))
        require_identity(serial, threaded)
        require_identity(serial, resumed, resumed=True)
        for name in LEDGERS:
            if (serial / name).read_bytes() != (threaded / name).read_bytes():
                raise RuntimeError(f"closed-wall threading changed {name}")
        require_restart_suffix(serial, resumed)
        report[backend] = {"thread_identity": True, "restart_identity": True, "ledger_identity": True}

    worst = 0.0
    native = frames(root / "vulkan-t1")
    for time, directory in frames(root / "cpu-t1").items():
        for name in FIELDS:
            shape, _, expected = read_float32_npy(directory / f"{name}.npy")
            actual_shape, _, actual = read_float32_npy(native[time] / f"{name}.npy")
            if shape != actual_shape or any(not math.isfinite(value) for values in (expected, actual) for value in values):
                raise RuntimeError("invalid closed-wall field comparison")
            scale = max(map(abs, expected))
            ulp = math.ldexp(1, math.frexp(scale)[1] - 24) if scale else math.ldexp(1, -149)
            steps = max(1, round(time / timestep))
            limit = 32 * steps * ulp
            if name in ("u", "v", "w"):
                limit = max(limit, steps * 5e-5)
            error = max(abs(a - b) for a, b in zip(expected, actual))
            if error > limit:
                raise RuntimeError(f"closed-wall parity failed: {name}/{time}: {error} > {limit}")
            worst = max(worst, error / limit)
    if identities != {str(path): sha(path) for path in paths}:
        raise RuntimeError("closed-wall regression inputs changed during execution")
    report.update(inputs=identities, parity_maximum_error_to_limit=worst,
                  claimBoundary="Bounded CPU/Vulkan SGS regression; not whole-backend readiness.")
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"closed-wall SGS regression PASS: {root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--native-tool", type=Path, default=ROOT / "bin/nimbus_native_admission_batch")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="nimbus-closed-sgs-") as temporary:
        output = args.output.resolve() if args.output else Path(temporary) / "evidence"
        qualify(args.binary.resolve(), args.native_tool.resolve(), output)


if __name__ == "__main__":
    main()
