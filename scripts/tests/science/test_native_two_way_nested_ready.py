#!/usr/bin/env python3
"""Bounded public native two-way nesting, conservation, parity and continuation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from npy_support import read_float32_npy

ROOT = Path(__file__).resolve().parents[3]
WATER = ("qv", "qc", "qr", "qi", "qs", "qg", "qh")
PRIMARY = ("rho", "theta", "p", "u", "v", "w", *WATER)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(cpu, device, seconds, dt, exact=False, refinement_ratio=1):
    records = {}
    for level in (Path(""), Path("nests/child_0000")):
        for second in range(seconds + 1):
            expected = cpu / level / f"step_{second:06d}"
            actual = device / level / f"step_{second:06d}"
            water = [read_float32_npy(expected / f"{name}.npy")[2]
                     for name in WATER if (expected / f"{name}.npy").exists()]
            water_scale = max((math.fsum(values) for values in zip(*water)), default=0)
            for name in PRIMARY:
                path = expected / f"{name}.npy"
                if not path.exists():
                    continue
                shape, raw, reference = read_float32_npy(path)
                other_shape, other_raw, candidate = read_float32_npy(actual / path.name)
                if shape != other_shape or any(not math.isfinite(v) for v in candidate):
                    raise RuntimeError(f"invalid native field: {actual / path.name}")
                if name in WATER and min(candidate) < 0:
                    raise RuntimeError(f"negative native water: {actual / path.name}")
                if exact or second == 0:
                    if raw != other_raw:
                        raise RuntimeError(f"field identity failed: {actual / path.name}")
                    continue
                scale = water_scale if name in WATER else max(map(abs, reference))
                ulp = math.ldexp(1.0, math.frexp(scale)[1] - 24) if scale else math.ldexp(1.0, -149)
                steps = round(second / dt) * (refinement_ratio if level.parts else 1)
                limit = 32 * steps * ulp
                if name in ("u", "v", "w"):
                    limit = max(limit, steps * 5e-5)
                error = max(abs(a-b) for a,b in zip(reference,candidate))
                records[f"{level}/{second}s/{name}"] = {"max_absolute_error":error, "absolute_limit":limit}
                if error > limit:
                    raise RuntimeError(f"native parity exceeded its limit: {actual / path.name}: {error} > {limit}")
    return records


def compare_private(probe: Path, reference: Path, candidate: Path, seconds: int, dt: float, ratio: int, exact: bool = False, *, checkpoint_samples=None, checkpoint_directories=None) -> dict:
    records = {}
    def extract(run: Path, step: int) -> Path:
        destination = run / ("private_fields_support" if checkpoint_samples is not None else "private_fields") / str(step)
        if not (destination / "manifest.json").exists():
            checkpoint_root = (checkpoint_directories[0 if run == reference else 1]
                               if checkpoint_directories is not None else run / "checkpoints")
            checkpoint = checkpoint_root / f"checkpoint_step_{step:09d}.nimbus"
            command = [str(probe), str(checkpoint), str(destination)]
            if checkpoint_samples is not None:
                command.append("--support-fields")
            result = subprocess.run(command, cwd=ROOT, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"private checkpoint probe failed: {result.stdout}")
        return destination
    samples = checkpoint_samples if checkpoint_samples is not None else ((second, round(second / dt)) for second in range(1, seconds + 1))
    for second, step in samples:
        expected = extract(reference, step)
        actual = extract(candidate, step)
        if checkpoint_samples is not None:
            for run, extracted in ((reference, expected), (candidate, actual)):
                manifest = json.loads((run / f"step_{second:06d}" / "manifest.json").read_text())
                checkpoint_time = json.loads((extracted / "manifest.json").read_text())["time_s"]
                if manifest["simulation_time_s"] != checkpoint_time:
                    raise RuntimeError("private checkpoint and primary snapshot times differ")
        def moment_names(directory):
            return sorted(p.relative_to(directory) for level in ("parent", "child")
                          for p in (directory / level).glob("*.npy"))
        names = moment_names(expected)
        if not names or names != moment_names(actual):
            raise RuntimeError("private checkpoint field layout differs")
        for name in names:
            shape, raw, values = read_float32_npy(expected / name)
            other_shape, other_raw, other = read_float32_npy(actual / name)
            if shape != other_shape or any(not math.isfinite(v) or v < 0 for population in (values, other) for v in population):
                raise RuntimeError(f"invalid private moment {actual/name}")
            moment = name.name.split(".")[1].split("_")[0]
            species = {"Nc":"qc", "Nr":"qr", "Ni":"qi", "Ns":"qs", "Ng":"qg", "Nh":"qh"}.get(moment)
            if species:
                for run, population in ((reference, values), (candidate, other)):
                    level = Path("nests/child_0000") if name.parts[0] == "child" else Path()
                    support = (extract(run, step) / "support" / name.parts[0] if checkpoint_samples is not None
                               else run / level / f"step_{second:06d}")
                    mass = read_float32_npy(support / f"{species}.npy")[2]
                    density = read_float32_npy(support / "rho.npy")[2]
                    for i in range(1, shape[0]-1):
                        for j in range(1, shape[1]-1):
                            for k in range(shape[2]):
                                index = (i*shape[1]+j)*shape[2]+k
                                if density[index]*mass[index] > 1e-12 and population[index] <= 0:
                                    raise RuntimeError(f"active {species} has no particles: {run}: {name}: {(i,j,k)}")
            if exact:
                if raw != other_raw:
                    raise RuntimeError(f"private direct/staged identity failed: {name}")
                continue
            scale = max(map(abs, values))
            ulp = math.ldexp(1.0, math.frexp(scale)[1] - 24) if scale else math.ldexp(1.0, -149)
            process_steps = step * (ratio if name.parts[0] == "child" else 1)
            transport_limit = 32 * process_steps * ulp
            physics_limit = .05 + 5e-4 * process_steps * scale
            limit = max(transport_limit, physics_limit)
            error = max(abs(a-b) for a,b in zip(values, other))
            records[f"{step * dt:g}s/{name}"] = {"max_absolute_error":error, "absolute_limit":limit,
                "transport_absolute_limit":transport_limit, "physics_absolute_limit":physics_limit}
            if error > limit:
                raise RuntimeError(f"private moment parity exceeded limit: {name}: {error} > {limit}")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-probe", type=Path, default=ROOT / "bin/nimbus_private_checkpoint_probe")
    parser.add_argument("--family", choices=("all", "dry", "kessler", "lin", "thompson", "milbrandt", "milbrandt_no_hail"), default="all")
    parser.add_argument("--advection", choices=("weno", "tvd"), default="weno")
    parser.add_argument("--terrain", choices=("flat", "bell", "schar", "bell3", "schar3"), default="flat")
    parser.add_argument("--initialization", choices=("seeded", "unseeded"), default="seeded")
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise RuntimeError("native two-way evidence directory must be empty")
    families = ("dry", "kessler", "lin") if args.family == "all" else (args.family,)
    configs = {f"{family}-{backend}": ROOT / "tests/fixtures/configs/qualification" /
               (f"stationary_nested_{family}_{args.advection}_{backend}_ready.yaml" if args.terrain == "flat" else
                f"stationary_mapped_nested_{family}_{args.terrain}_{args.advection}_{backend}_ready.yaml")
               for family in families for backend in ("cpu", "metal", "vulkan")}
    if args.initialization == "unseeded":
        configs = {name: path.with_name(path.name.replace("_ready.yaml", "_unseeded_ready.yaml"))
                   for name, path in configs.items()}
    hashes = {name: sha(path) for name,path in configs.items()}
    binary = args.binary.resolve()
    identity = sha(binary)
    private_family = any(f in ("thompson", "milbrandt", "milbrandt_no_hail") for f in families)
    probe_identity = sha(args.private_probe.resolve()) if private_family else None
    results = {}
    with tempfile.TemporaryDirectory(prefix=".native-two-way-", dir=binary.parent) as temporary:
        snapshot = Path(temporary) / "nimbus_sim"
        shutil.copy2(binary, snapshot)
        for family in families:
            for backend in ("cpu", "metal", "vulkan", "vulkan-staged"):
                label = f"{family}-{backend}"
                config = configs[f"{family}-{backend.removesuffix('-staged')}"]
                text = config.read_text()
                seconds = int(float(re.search(r"(?m)^  run_time: ([^\n]+)", text)[1]))
                dt = float(re.search(r"(?m)^  dt_large: ([^\n]+)", text)[1])
                environment = os.environ.copy()
                environment.pop("NIMBUS_VULKAN_MEMORY_POLICY", None)
                environment["NIMBUS_GPU_KERNEL_TIMING"] = "1"
                if backend == "vulkan-staged":
                    environment["NIMBUS_VULKAN_MEMORY_POLICY"] = "staged"
                command = [sys.executable, str(ROOT / "scripts/tests/science/test_two_way_nested_ready.py"),
                           "--binary", str(snapshot), "--config", str(config), "--output", str(destination/label)]
                result = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                (destination/f"{label}.log").write_text(result.stdout)
                if result.returncode:
                    raise RuntimeError(f"{label} failed:\n{result.stdout[-10000:]}")
                evidence = json.loads((destination/label/"summary.json").read_text())
                if evidence["binary_sha256"] != identity:
                    raise RuntimeError("native two-way binary identity changed")
                if backend != "cpu":
                    log = (destination/label/"serial.log").read_text()
                    if "[NESTED_NATIVE]" not in log:
                        raise RuntimeError("native ownership declaration is missing")
                    counts = {name: int(value) for name,value in re.findall(r"([^,{} ]+\.comp):(\d+)",log)}
                    if args.terrain == "flat":
                        ratio = int(re.search(r"(?m)^    parent_time_step_ratio: (\d+)", text)[1])
                        substeps = int(re.search(r"(?m)^  acoustic_substeps: (\d+)", text)[1])
                        expected_pressure = round(seconds / dt) * (1 + ratio) * substeps
                        if counts.get("acoustic/acoustic_pressure_cartesian_cgrid.comp") != expected_pressure:
                            raise RuntimeError(f"{label} omitted native nested continuity updates")
                    transport = ("advection/advect_conservative_update_cartesian_cgrid.comp",) if args.terrain == "flat" else (
                        "terrain/mapped_scalar_update.comp", "terrain/mapped_slow_tendencies.comp", "terrain/mapped_fast_momentum.comp")
                    for kernel in ("dynamics/conservative_momentum.comp", "dynamics/conservative_reduction.comp", *transport):
                        if counts.get(kernel,0) <= 0:
                            raise RuntimeError(f"{label} did not execute {kernel}")
                    if family in ("thompson", "milbrandt", "milbrandt_no_hail"):
                        scheme = family.split("_")[0]
                        kernels = ("cloud_adjustment", "warm_rain", "deposition", "freezing", "completion", "cloud_phase", "moment_sedimentation") if scheme == "thompson" else ("cloud_adjustment", "warm_rain", "cold", "melting", "sedimentation")
                        ratio = int(re.search(r"(?m)^    parent_time_step_ratio: (\d+)", text)[1])
                        steps = round(seconds / dt) * (1 + ratio)
                        for name in kernels:
                            expected = 5 * steps if "sedimentation" in name else steps
                            if counts.get(f"microphysics/{scheme}_{name}.comp", 0) != expected:
                                raise RuntimeError(f"private native coupling omitted {scheme}/{name}")
                        scalar_kernel = "advection/advect_conservative_update_cartesian_cgrid.comp" if args.terrain == "flat" else "terrain/mapped_scalar_update.comp"
                        minimum = steps * (9 + (5 if scheme == "thompson" else 6))
                        if counts.get(scalar_kernel, 0) < minimum:
                            raise RuntimeError(f"private native transport omitted fields: {counts.get(scalar_kernel)} < {minimum}")
                    evidence["dispatches"] = counts
                    ratio = int(re.search(r"(?m)^    parent_time_step_ratio: (\d+)", text)[1]) if private_family else 1
                    evidence["parity"] = compare(destination/f"{family}-cpu"/"serial", destination/label/"serial",seconds,dt, refinement_ratio=ratio)
                    if family in ("thompson", "milbrandt", "milbrandt_no_hail"):
                        evidence["private_parity"] = compare_private(args.private_probe.resolve(), destination/f"{family}-cpu"/"serial", destination/label/"serial", seconds, dt,
                            int(re.search(r"(?m)^    parent_time_step_ratio: (\d+)", text)[1]))
                    if backend == "vulkan-staged":
                        if "memory plan=staged_forced" not in log:
                            raise RuntimeError("forced staging was not exercised")
                        compare(destination/f"{family}-vulkan"/"serial",destination/label/"serial",seconds,dt,True)
                        if family in ("thompson", "milbrandt", "milbrandt_no_hail"):
                            compare_private(args.private_probe.resolve(), destination/f"{family}-vulkan"/"serial", destination/label/"serial", seconds, dt,
                                int(re.search(r"(?m)^    parent_time_step_ratio: (\d+)", text)[1]), True)
                if family == "thompson":
                    evidence["private_integrity"] = compare_private(args.private_probe.resolve(), destination/label/"serial", destination/label/"parallel",seconds,dt,
                        int(re.search(r"(?m)^    parent_time_step_ratio: (\d+)", text)[1]), True)
                results[label] = evidence
                print(f"passed {label}: closure, threading, restart and native ownership", flush=True)
    if (sha(binary) != identity or any(sha(configs[name]) != value for name,value in hashes.items()) or
            (private_family and sha(args.private_probe.resolve()) != probe_identity)):
        raise RuntimeError("native two-way qualification inputs changed")
    (destination/"summary.json").write_text(json.dumps({"status":"Ready", "binary_sha256":identity, "private_probe_sha256":probe_identity,
        "configuration_sha256":hashes, "advection":args.advection, "terrain":args.terrain, "results":results, "campaign_validated":False,
        "externally_reviewed":False},indent=2)+"\n")


if __name__ == "__main__":
    main()
