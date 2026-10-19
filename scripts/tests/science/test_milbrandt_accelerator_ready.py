#!/usr/bin/env python3
"""Bounded public Milbrandt CPU/Vulkan/Metal coupling, ownership and continuation gate."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile

import test_lin_rrtmg_accelerator_ready as runtime
from npy_support import read_float32_npy
from test_native_mapped_terrain_ready import require_ledgers

ROOT = runtime.ROOT
FIELDS = ("p", "rho", "theta", "u", "v", "w", "qv", "qc", "qr", "qi", "qs", "qg", "qh", "tke", "tracer")


def require_dispatch(log: str, backend: str, steps: int) -> None:
    for marker in ("Execution mode: qualified", f"[COMPUTE] backend={backend}", "fallback=off",
                   "Initialized microphysics scheme: milbrandt", "Initialized boundary layer scheme: ysu",
                   "Initialized radiation scheme: rrtmg"):
        if marker not in log:
            raise RuntimeError(f"Milbrandt run omitted {marker}")
    if backend == "cpu":
        return
    lines = [line for line in log.splitlines() if "[COMPUTE DISPATCH]" in line]
    if not lines:
        raise RuntimeError("Milbrandt omitted native dispatch telemetry")
    counts = {name: int(count) for name, count in re.findall(r"([^,{} ]+\.comp):(\d+)", lines[-1])}
    expected = {f"microphysics/milbrandt_{name}.comp": steps for name in ("cloud_adjustment", "warm_rain", "cold", "melting")}
    expected["microphysics/milbrandt_sedimentation.comp"] = 5 * steps
    for name in ("longwave", "shortwave"):
        expected[f"radiation/rrtmg_{name}_transfer_batch.comp"] = 2 * steps // 5
    for kernel, count in expected.items():
        if counts.get(kernel) != count:
            raise RuntimeError(f"Milbrandt expected {count} dispatches of {kernel}: {lines[-1]}")


def require_activity(output: Path) -> int:
    checks = require_ledgers(output)
    with (output / "thermodynamic_source_budget.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(int(row["microphysics_conservation_active"]) != 1 for row in rows):
        raise RuntimeError("Milbrandt omitted conservation admission")
    if not any(abs(float(row["microphysics_qv_requested_kg"])) > 0 for row in rows):
        raise RuntimeError("Milbrandt vapor processes remained inactive")
    for row in rows:
        residual = abs(float(row["microphysics_conservation_residual_kg"]))
        baseline = max(abs(float(row["microphysics_baseline_water_kg"])), 1e-30)
        if residual / baseline > 1e-6:
            raise RuntimeError("Milbrandt water closure exceeded 1e-6")
    frame = runtime.final_frame(output)
    if not any(max(read_float32_npy(frame / f"{field}.npy")[2]) > 0 for field in ("qi", "qs", "qg", "qh")):
        raise RuntimeError("Milbrandt mixed-phase reference produced no frozen water")
    return checks


def require_identity(a: Path, b: Path, label: str) -> None:
    first, second = runtime.final_frame(a), runtime.final_frame(b)
    for field in FIELDS:
        if (first / f"{field}.npy").read_bytes() != (second / f"{field}.npy").read_bytes():
            raise RuntimeError(f"Milbrandt {label}: {field} differs")


def require_parity(a: Path, b: Path) -> dict:
    water_fields = ("qv", "qc", "qr", "qi", "qs", "qg", "qh")
    reference_water = [read_float32_npy(runtime.final_frame(a) / f"{f}.npy")[2] for f in water_fields]
    candidate_water = [read_float32_npy(runtime.final_frame(b) / f"{f}.npy")[2] for f in water_fields]
    reference_total = [sum(values) for values in zip(*reference_water)]
    candidate_total = [sum(values) for values in zip(*candidate_water)]
    total_error = max(abs(x-y) for x,y in zip(reference_total,candidate_total))
    if total_error > 8e-9:
        raise RuntimeError(f"Milbrandt total-water parity: {total_error} > 8e-9")
    water_scale = max(map(abs, reference_total))
    water_limit = max(8e-9,80*math.ldexp(1.0,math.frexp(water_scale)[1]-24)) if water_scale else 8e-9
    result = {"totalWater": {"maxAbsoluteError":total_error,"absoluteLimit":8e-9},
              "waterPartitionReferenceScale":water_scale}
    for field in FIELDS:
        shape, _, expected = read_float32_npy(runtime.final_frame(a) / f"{field}.npy")
        other_shape, _, actual = read_float32_npy(runtime.final_frame(b) / f"{field}.npy")
        if shape != other_shape or any(not math.isfinite(v) for values in (expected, actual) for v in values):
            raise RuntimeError(f"Milbrandt invalid parity field {field}")
        scale = max(abs(v) for v in expected)
        limit = 80 * math.ldexp(1.0, math.frexp(scale)[1] - 24) if scale else 0
        if field in ("u", "v", "w"):
            limit = 1e-3
        if field.startswith("q"):
            limit = water_limit
        error = max(abs(x-y) for x, y in zip(expected, actual))
        result[field] = {"maxAbsoluteError": error, "absoluteLimit": limit}
        if error > limit:
            raise RuntimeError(f"Milbrandt {field} parity: {error} > {limit}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=runtime.DEFAULT_BINARY)
    parser.add_argument("--backend", choices=("all", "metal", "vulkan"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    identity = hashlib.sha256(binary.read_bytes()).hexdigest()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        if any(args.output.iterdir()):
            raise RuntimeError("Milbrandt evidence directory must be empty")
    context = nullcontext(args.output.resolve()) if args.output else tempfile.TemporaryDirectory(prefix="nimbus-milbrandt-ready-")
    backends = ("cpu", "metal", "vulkan") if args.backend == "all" else ("cpu", args.backend)
    with context as temporary:
        root = Path(temporary)
        results = {}
        for hail in (True, False):
            variant = "hail" if hail else "no-hail"
            outputs = {}
            for backend in backends:
                source = (ROOT / f"tests/fixtures/configs/qualification/cgrid_milbrandt_{backend}_ready.yaml").read_text()
                if not hail:
                    source = source.replace("hail_processes: true", "hail_processes: false")
                config = root / f"{variant}-{backend}.yaml"
                config.write_text(source)
                preflight = runtime.run(binary, backend, ["--validate-only", "--config", str(config)])
                runtime.require_success(preflight, f"{variant}-{backend} preflight")
                if "[VALIDATE] PASS" not in preflight.stdout:
                    raise RuntimeError("Milbrandt preflight omitted PASS")
                if backend != "cpu" and hail:
                    for old, new, label in (("allow_fallback: false", "allow_fallback: true", "fallback"),
                                            ("gpu_apply_microphysics: true", "gpu_apply_microphysics: false", "ownership"),
                                            ("    west: periodic\n    east: periodic", "    west: radiative\n    east: radiative", "boundary")):
                        changed = source.replace(old, new, 1)
                        if changed == source:
                            raise RuntimeError(f"Milbrandt {label} mutation drifted")
                        invalid = root / f"{backend}-{label}.yaml"
                        invalid.write_text(changed)
                        for options in (("--validate-only",), ("--headless", "--duration=0", f"--outdir={root / (backend+'-'+label)}")):
                            rejected = runtime.run(binary, backend, ["--config", str(invalid), *options])
                            if rejected.returncode == 0 or "microphysics.scheme=milbrandt requires" not in rejected.stdout:
                                raise RuntimeError(f"Milbrandt {label} bypassed ordinary admission:\n{rejected.stdout}")
                for workers in (1, 4):
                    output = root / f"{variant}-{backend}-{workers}"
                    result = runtime.run(binary, backend, ["--headless", "--config", str(config), f"--outdir={output}",
                        "--write-every=4", "--checkpoint-every=2", f"--threads={workers}", "--log-profile=normal"], workers)
                    (root / f"{variant}-{backend}-{workers}.log").write_text(result.stdout)
                    runtime.require_success(result, f"Milbrandt {variant}-{backend}-{workers}")
                    require_dispatch(result.stdout, backend, 20)
                    checks = require_activity(output)
                    outputs[backend, workers] = output
                require_identity(outputs[backend, 1], outputs[backend, 4], f"{variant}-{backend} threading")
                runtime.require_ledger_identity(outputs[backend, 1], outputs[backend, 4], f"{variant}-{backend} threading")
                for step in (10, 20):
                    name = f"checkpoints/checkpoint_step_{step:09d}.nimbus"
                    if (outputs[backend, 1] / name).read_bytes() != (outputs[backend, 4] / name).read_bytes():
                        raise RuntimeError("Milbrandt private checkpoint changed across workers")
                resumed = root / f"{variant}-{backend}-resumed"
                checkpoint = outputs[backend, 4] / "checkpoints/checkpoint_step_000000010.nimbus"
                result = runtime.run(binary, backend, ["--headless", "--config", str(config), f"--resume={checkpoint}",
                    f"--outdir={resumed}", "--duration=4", "--write-every=4", "--checkpoint-every=0", "--threads=4", "--log-profile=normal"], 4)
                (root / f"{variant}-{backend}-resumed.log").write_text(result.stdout)
                runtime.require_success(result, "Milbrandt restart")
                if "[CHECKPOINT] resumed step=10, time_s=2" not in result.stdout:
                    raise RuntimeError("Milbrandt omitted resumed checkpoint provenance")
                require_dispatch(result.stdout, backend, 10)
                require_identity(outputs[backend, 4], resumed, f"{variant}-{backend} restart")
                runtime.require_restart_suffix(outputs[backend, 4], resumed)
                results[f"{variant}-{backend}"] = {"ownerChecks": checks, "threading": "exact", "restart": "exact"}
                if backend != "cpu":
                    results[f"{variant}-{backend}"]["cpuParity"] = require_parity(outputs["cpu", 1], outputs[backend, 1])
                    cross = {}
                    for source_backend, target_backend in (("cpu", backend), (backend, "cpu")):
                        checkpoint = outputs[source_backend, 4] / "checkpoints/checkpoint_step_000000010.nimbus"
                        target_config = root / f"{variant}-{target_backend}.yaml"
                        target_output = root / f"{variant}-{source_backend}-to-{target_backend}"
                        result = runtime.run(binary, target_backend, ["--headless", "--config", str(target_config),
                            f"--resume={checkpoint}", f"--outdir={target_output}", "--duration=4", "--write-every=4",
                            "--checkpoint-every=0", "--threads=4", "--log-profile=normal"], 4)
                        (root / f"{variant}-{source_backend}-to-{target_backend}.log").write_text(result.stdout)
                        runtime.require_success(result, "Milbrandt cross-backend restart")
                        require_dispatch(result.stdout, target_backend, 10)
                        cross[f"{source_backend}-to-{target_backend}"] = {
                            "ownerChecks": require_activity(target_output),
                            "targetParity": require_parity(outputs[target_backend, 4], target_output)}
                    results[f"{variant}-{backend}"]["crossBackendRestart"] = cross
        if hashlib.sha256(binary.read_bytes()).hexdigest() != identity:
            raise RuntimeError("Milbrandt binary changed during evidence collection")
        (root / "summary.json").write_text(json.dumps({"binarySha256": identity, "bounded": True, "results": results}, indent=2)+"\n")
    print(f"[milbrandt-native-ready] PASS: public {'/'.join(backends)} selection, all native processes, water/owner closure, parity, exact threading/same-backend restart and bounded cross-backend continuation; both hail settings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
