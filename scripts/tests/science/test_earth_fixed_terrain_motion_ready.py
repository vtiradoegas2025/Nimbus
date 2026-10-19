#!/usr/bin/env python3
"""Bounded public continuous Earth-fixed motion, budgets and restart gate."""

from __future__ import annotations

import argparse
import array
import ast
import csv
import json
import hashlib
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.qualification.checkpointing.qualify_vulkan_restart import (
    LEDGERS, require_report_owner_identities, require_restart_suffix,
)

CONFIG = ROOT / "tests/fixtures/configs/qualification/earth_fixed_terrain_motion_ready.yaml"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run(binary: Path, output: Path, threads: int, *, resume: Path | None = None,
        config: Path = CONFIG, duration: int = 65) -> None:
    args = [str(binary), "--headless", "--config", str(config), f"--threads={threads}",
            "--log-profile=quiet", "--no-timing", f"--duration={duration}",
            f"--write-every={duration}", "--checkpoint-every=32",
            f"--checkpoint-dir={output}/checkpoints", f"--outdir={output}"]
    if resume is not None:
        args.append(f"--resume={resume}")
    native = re.search(r"^    backend: (metal|vulkan)\s*$", config.read_text(), re.MULTILINE)
    environment = os.environ.copy()
    if native:
        args.remove("--log-profile=quiet")
        environment["NIMBUS_GPU_KERNEL_TIMING"] = "1"
    result = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".log").write_text(result.stdout)
    require(result.returncode == 0, f"public moving run failed:\n{result.stdout[-5000:]}")
    require("Falling back" not in result.stdout, "moving run substituted a backend")
    if native:
        for kernel in ("terrain/mapped_slow_tendencies.comp", "terrain/mapped_mass_continuity.comp",
                       "terrain/mapped_fast_momentum.comp", "terrain/mapped_scalar_update.comp",
                       "dynamics/acoustic_bc_face_cartesian_cgrid.comp", "terrain/mapped_boundary.comp",
                       "terrain/earth_geometry.comp", "radiation/simple_grey.comp", "boundary_layer/slab_column.comp",
                       "terrain/held_sources.comp",
                       "microphysics/lin_sedimentation.comp" if "  scheme: lin\n" in config.read_text() else "microphysics/kessler_sedimentation.comp",
                       "chaos/earth_control_evolution.comp"):
            require(re.search(re.escape(kernel)+r":[1-9][0-9]*", result.stdout) is not None,
                    f"ordinary native Earth-fixed execution omitted {kernel}")



def read_npy(path: Path) -> tuple[tuple[int, ...], array.array]:
    raw = path.read_bytes()
    require(raw[:6] == b"\x93NUMPY", f"invalid NPY file: {path}")
    width = 2 if raw[6] == 1 else 4
    length = int.from_bytes(raw[8:8+width], "little")
    start = 8+width
    header = ast.literal_eval(raw[start:start+length].decode("latin1"))
    require(not header["fortran_order"], "unexpected Fortran layout")
    dtype = header["descr"]
    require(dtype[-2:] in ("f4", "f8"), f"unsupported numerical dtype {dtype}")
    values = array.array("f" if dtype[-1] == "4" else "d")
    values.frombytes(raw[start+length:])
    if (dtype[0] == ">") != (sys.byteorder == "big"):
        values.byteswap()
    shape = tuple(header["shape"])
    require(math.prod(shape) == len(values), "NPY shape and payload differ")
    return shape, values


def surface_reference(origin: tuple[float, float], i: int, j: int) -> float:
    def height(x: float, y: float) -> float:
        a, b = (x-4500.0)/500.0, (y-4500.0)/500.0
        ia, ib = math.floor(a), math.floor(b)
        fa, fb = a-ia, b-ib
        def node(di: int, dj: int) -> float:
            radius2 = ((ia+di)*500.0)**2 + ((ib+dj)*500.0)**2
            return 300.0*1800.0**3/(1800.0**2+radius2)**1.5
        return ((1-fa)*node(0,0)+fa*node(1,0))*(1-fb) + ((1-fa)*node(0,1)+fa*node(1,1))*fb
    x, y = origin[0]+500.0*i, origin[1]+500.0*j
    def cuts(start: float) -> list[float]:
        knot = 4500.0+(math.floor((start-4500.0)/500.0)+1)*500.0
        return [start, knot, start+500.0] if start < knot < start+500.0 else [start, start+500.0]
    xs, ys = cuts(x), cuts(y)
    integral = 0.0
    for left, right in zip(xs, xs[1:]):
        for lower, upper in zip(ys, ys[1:]):
            integral += (right-left)*(upper-lower)*sum(height(a,b) for a in (left,right) for b in (lower,upper))/4.0
    return integral/250000.0


def final_step(output: Path) -> Path:
    return sorted(path for path in output.glob("step_*") if path.is_dir())[-1]


def compare_stationary_reference(moving: Path, stationary: Path) -> dict[str, float]:
    tolerances = {"rho": 0.001, "theta": 0.05, "qv": 1e-5, "u": 0.1, "v": 0.1, "w": 0.05}
    errors = {}
    for name, tolerance in tolerances.items():
        _, actual = read_npy(final_step(moving) / f"{name}.npy")
        _, reference = read_npy(final_step(stationary) / f"{name}.npy")
        differences = []
        for i in range(3,14):
            for j in range(3,14):
                x, y = i+5.0*65.0/500.0, j-2.0*65.0/500.0
                ix, iy = math.floor(x), math.floor(y)
                sx, sy = x-ix, y-iy
                for k in range(2,14):
                    expected = math.fsum(wx*wy*reference[((ix+di)*18+iy+dj)*18+k]
                        for di, wx in ((0,1-sx),(1,sx)) for dj, wy in ((0,1-sy),(1,sy)))
                    differences.append(abs(actual[(i*18+j)*18+k]-expected))
        errors[name] = math.fsum(differences)/len(differences)
        require(errors[name] < tolerance, f"Earth-frame reference mismatch for {name}: {errors[name]}")
    return errors


def verify(output: Path, duration: int = 65, velocity: tuple[float, float] = (5.0, -2.0)) -> dict:
    step = final_step(output)
    manifest = json.loads((step / "manifest.json").read_text())
    geometry = manifest["grid"]["vertical_geometry"]
    require(geometry["sampling"] == "cell_average_bilinear_earth_v1", "wrong terrain representation")
    require(geometry["coordinate_remap_order"] == 1, "remap order is not declared")
    frame = manifest["grid"]["horizontal_frame"]
    origin = tuple(frame["grid_origin_earth_m"])
    require(all(abs(origin[k]-velocity[k]*duration) <= 1e-9 for k in range(2)),
            "frame origin did not advance with physical geometry")
    shape, rho = read_npy(step / "rho.npy")
    zshape, zface = read_npy((step / geometry["interface_path"]).resolve())
    require(shape == (18,18,18) and zshape == (18,18,19), "incorrect field/geometry shape")
    measured_mass = math.fsum(rho[(i*18+j)*18+k]
        * (zface[(i*18+j)*19+k+1]-zface[(i*18+j)*19+k])*250000.0
        for i in range(1,17) for j in range(1,17) for k in range(18))
    for i in range(1,17):
        for j in range(1,17):
            require(abs(zface[(i*18+j)*19]-surface_reference(origin,i,j)) < 1e-10,
                    "published ground does not match the fixed Earth surface")
    for path in step.glob("*.npy"):
        require(all(math.isfinite(x) for x in read_npy(path)[1]), f"invalid published field {path.name}")
    remaps = rows(output / "earth_fixed_remap.csv")
    require(remaps, "no coordinate-remapping ledger was published")
    require(abs(measured_mass-float(remaps[-1]["dry_mass_after_kg"])) / measured_mass < 1e-12,
            "remapping ledger does not match published density and geometry")
    energy = {round(float(row["sim_time_s"]), 9): row for row in rows(output / "energy_budget.csv")
              if row["stage"] == "total_step"}
    sources = {round(float(row["time_s"]), 9): row
               for row in rows(output / "thermodynamic_source_budget.csv")}
    largest = {"dry_mass": 0.0, "water": 0.0, "energy_storage": 0.0, "whole_step_energy": 0.0}
    for row in remaps:
        t = round(float(row["time_s"]), 9)
        for prefix, before, after, outward, storage in (
            ("dry_mass", "dry_mass_before_kg", "dry_mass_after_kg", "outward_dry_mass_kg", "dry_mass_storage_kg"),
            ("water", "water_before_kg", "water_after_kg", "outward_water_kg", "water_storage_kg"),
        ):
            scale = max(1.0, abs(float(row[before])), abs(float(row[after])))
            residual = abs(float(row[after])-float(row[before])+float(row[outward])-float(row[storage])) / scale
            require(residual < 1e-12, f"{prefix} remapping is not conservative at {t}")
            require(abs(float(row[storage]))/scale < 5e-7, f"unbounded {prefix} storage error")
            largest[prefix] = max(largest[prefix], residual)
        for component in "xyz":
            initial=float(row[f"momentum_{component}_before"])
            final=float(row[f"momentum_{component}_after"])
            outward=float(row[f"momentum_{component}_outward"])
            rounding=float(row[f"momentum_{component}_storage"])
            scale=(float(row["dry_mass_before_kg"])+float(row["dry_mass_after_kg"]))*180.0
            require(abs(final-initial+outward-rounding)/scale < 1e-12, "momentum remapping does not close")
            require(abs(rounding)/scale < 1e-6, "unbounded momentum remapping error")
        e = energy[t]
        scale = max(1.0, sum(abs(float(e[f"moist_v2_{when}_{name}_j"]))
                    for when in ("before", "after")
                    for name in ("dry_internal", "water_sensible", "phase_reference",
                                 "resolved_kinetic", "gravitational", "subgrid_tke")))
        storage = abs(float(row["energy_storage_j"])) / scale
        require(storage < 5e-7, f"coordinate remapping lost energy at {t}")
        require(abs(float(row["energy_change_j"])+float(row["outward_energy_j"])
                    -float(row["energy_storage_j"])) < 128*math.ulp(scale), "energy ledger does not close")
        observed = float(e["moist_v2_after_total_j"])-float(e["moist_v2_before_total_j"])
        endpoint = round(float(row["time_s"])+float(row["dt_s"]), 9)
        authored = float(sources[endpoint]["completed_step_energy_step_committed_owner_sum_j"])
        closure = abs(observed-authored) / scale
        require(closure < 2e-12, f"whole-step energy has an unowned change at {t}")
        largest["energy_storage"] = max(largest["energy_storage"], storage)
        largest["whole_step_energy"] = max(largest["whole_step_energy"], closure)
    require(require_report_owner_identities(output) > 0, "owner report validation did not execute")
    return largest


def compare_backend_reference(output: Path, reference: Path, steps: int = 325) -> dict:
    errors = {}
    for field in ("u", "v", "w", "rho", "p", "theta", "qv", "qr"):
        shape, actual = read_npy(final_step(output) / f"{field}.npy")
        expected_shape, expected = read_npy(final_step(reference) / f"{field}.npy")
        require(shape == expected_shape == (18,18,18), f"backend parity shape mismatch: {field}")
        require(all(math.isfinite(value) for values in (actual, expected) for value in values),
                f"backend parity contains nonfinite {field}")
        if field in ("u", "v", "w"):
            tolerance = steps * 5e-5
        elif field in ("qv", "qr"):
            tolerance = steps * 5e-9
        else:
            exponent = math.frexp(max(abs(value) for value in expected))[1]
            tolerance = steps * 4 * math.ldexp(1.0, exponent - 24)
        error = max(abs(a-b) for a,b in zip(actual, expected))
        require(error <= tolerance, f"CPU/native Earth-fixed {field} error {error} exceeds {tolerance}")
        errors[field] = {"maximum_absolute_error": error, "absolute_limit": tolerance}
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--cpu-reference", type=Path)
    args = parser.parse_args()
    config = args.config.resolve()
    binary = args.binary.resolve()
    preflight = subprocess.run([str(binary), "--validate-only", "--config", str(config)],
                               cwd=ROOT, capture_output=True, text=True)
    require(preflight.returncode == 0 and "[VALIDATE] PASS" in preflight.stdout,
            f"public moving preflight failed: {preflight.stdout}{preflight.stderr}")
    temporary = tempfile.TemporaryDirectory(prefix="nimbus-earth-motion-") if args.output is None else None
    root = Path(temporary.name) if temporary else args.output.resolve()
    require(not root.exists() or not any(root.iterdir()), "evidence directory must be empty")
    root.mkdir(parents=True, exist_ok=True)
    one, four = root / "one", root / "four"
    run(binary, one, 1, config=config)
    run(binary, four, 4, config=config)
    results = {"one": verify(one), "four": verify(four)}
    for name in [*LEDGERS, "earth_fixed_remap.csv"]:
        require((one/name).read_bytes() == (four/name).read_bytes(), f"thread-dependent ledger: {name}")
    for path in final_step(one).glob("*.npy"):
        require(path.read_bytes() == (final_step(four)/path.name).read_bytes(), f"thread-dependent field: {path.name}")
    checkpoints = sorted((one / "checkpoints").glob("checkpoint_step_*.nimbus"))
    require(len(checkpoints) >= 2, "missing moving-grid continuation checkpoint")
    resumed = root / "resumed"
    run(binary, resumed, 4, resume=checkpoints[0], config=config)
    results["resumed"] = verify(resumed)
    require_restart_suffix(one, resumed)
    continuous_rows = (one / "earth_fixed_remap.csv").read_text().splitlines()
    resumed_rows = (resumed / "earth_fixed_remap.csv").read_text().splitlines()
    require(continuous_rows[-(len(resumed_rows)-1):] == resumed_rows[1:], "remapping restart suffix differs")
    final_checkpoint = sorted((resumed / "checkpoints").glob("checkpoint_step_*.nimbus"))[-1]
    require(checkpoints[-1].read_bytes() == final_checkpoint.read_bytes(), "moving checkpoint restart is not exact")
    require(checkpoints[-1].read_bytes() == sorted((four / "checkpoints").glob("checkpoint_step_*.nimbus"))[-1].read_bytes(),
            "moving checkpoint depends on thread count")
    for path in final_step(one).glob("*.npy"):
        require(path.read_bytes() == (final_step(resumed)/path.name).read_bytes(), f"restart changed {path.name}")
    text = config.read_text()
    for label, velocity in (("positive-limit", (180, 180)), ("negative-limit", (-180, -180))):
        changed = root / f"{label}.yaml"
        changed.write_text(text.replace("u_ms: 5", f"u_ms: {velocity[0]}", 1)
                          .replace("v_ms: -2", f"v_ms: {velocity[1]}", 1))
        output = root / label
        run(binary, output, 1, config=changed, duration=2)
        results[label] = verify(output, duration=2, velocity=velocity)
    stationary = root / "stationary.yaml"
    stationary.write_text(text.replace("u_ms: 5", "u_ms: 0", 1).replace("v_ms: -2", "v_ms: 0", 1))
    run(binary, root / "stationary", 1, config=stationary)
    manifest = json.loads((final_step(root / "stationary") / "manifest.json").read_text())
    require(manifest["grid"]["vertical_geometry"]["sampling"] == "cell_average_bilinear_earth_v1",
            "stationary reference changed terrain representation")
    results["stationary_reference_mean_absolute_error"] = compare_stationary_reference(one, root / "stationary")
    mutations = [
        ("speed", "u_ms: 5", "u_ms: 181"),
        ("tracking", "enabled: false\n    recycle_domain: false", "enabled: true\n    recycle_domain: false"),
        ("terrain", "h0: 300", "h0: 301"),
        (("backend", "backend: cpu", "backend: vulkan") if "backend: cpu" in text else
         ("native ownership", "gpu_apply_slow: true", "gpu_apply_slow: false")),
    ]
    if "  scheme: lin\n" in text:
        mutations.append(("Lin parameters", "accretion_coefficient: 2.2", "accretion_coefficient: 2.21"))
    for name, old, new in mutations:
        changed = root / f"invalid-{name}.yaml"
        require(old in text, f"negative fixture anchor missing: {old}")
        changed.write_text(text.replace(old, new, 1))
        result = subprocess.run([str(binary), "--headless", "--validate-only", "--config", str(changed)],
                                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        require(result.returncode != 0, f"unsupported moving profile admitted: {name}")
    if args.cpu_reference:
        results["cpu_parity"] = compare_backend_reference(one, args.cpu_reference.resolve())
    results["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    results["configuration_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    (root / "result.json").write_text(json.dumps(results, indent=2)+"\n")
    print("[earth-fixed-motion] PASS: public evolution, physical mass/water/energy closure, "
          "1/4 identity, exact restart and negative admission")


if __name__ == "__main__":
    main()
