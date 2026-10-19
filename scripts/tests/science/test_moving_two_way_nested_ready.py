#!/usr/bin/env python3
"""Public moving-child feedback, relocation closure and continuation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from npy_support import read_float32_npy
from test_nested_grid_recycling_ready import checkpoint_at, checkpoint_section
from test_two_way_nested_ready import require_closure, require_water_closure, require_feedback_output_budget
from test_earth_fixed_terrain_motion_ready import read_npy
from scripts.qualification.checkpointing.qualify_vulkan_restart import LEDGERS, require_restart_suffix

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "tests/fixtures/configs/qualification/moving_nested_two_way_cpu_ready.yaml"
FIELDS = ("rho", "theta", "p", "u", "v", "w", "qv", "qc", "qr")
ENERGY = ("dry_internal", "water_sensible", "phase_reference", "resolved_kinetic", "gravitational", "subgrid_tke", "total")
CURRENCIES = ("dry_mass", "rho_theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh",
              "physical_x_momentum", "physical_y_momentum", "physical_z_momentum",
              "dry_internal_energy", "water_sensible_energy", "phase_reference_energy", "resolved_kinetic_energy",
              "gravitational_energy", "subgrid_tke_energy", "total_energy")


def execute(binary: Path, root: Path, label: str, threads: int, *, duration: float = 120,
            cadence: float = 8, resume: Path | None = None, config: Path = CONFIG) -> tuple[Path, str]:
    output = root / label
    command = [str(binary), "--headless", "--config", str(config), f"--duration={duration:g}",
               f"--write-every={duration:g}", f"--checkpoint-every={cadence:g}", f"--threads={threads}",
               f"--outdir={output}", "--log-profile=normal", "--no-timing"]
    if resume:
        command.append(f"--resume={resume}")
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (root / f"{label}.log").write_text(result.stdout)
    if result.returncode or "[NESTING SCIENTIFIC VALIDITY]" in result.stdout or "[SCIENTIFIC INVALIDITY]" in result.stdout:
        raise RuntimeError(f"{label} failed:\n{result.stdout[-8000:]}")
    backend = re.search(r"(?m)^    backend: (\w+)", config.read_text())[1]
    for marker in ("Execution mode: qualified", f"[COMPUTE] backend={backend}", "fallback=off"):
        if marker not in result.stdout:
            raise RuntimeError(f"{label} omitted {marker}")
    water = tuple(name for name in ("qv", "qc", "qr", "qi", "qs", "qg", "qh")
                  if (output / "step_000001" / f"{name}.npy").exists())
    require_feedback_output_budget(output, 1, water)
    if "preset: dynamics_full" in config.read_text():
        for level in (Path(), Path("nests/child_0000")):
            for name in ("dynamic_pressure", "buoyancy_pressure", "angular_momentum", "angular_momentum_tendency"):
                path = output / level / "step_000001" / f"{name}.npy"
                values = read_float32_npy(path)[2]
                if not all(math.isfinite(value) for value in values) or max(map(abs, values), default=0.0) == 0.0:
                    raise RuntimeError(f"moving thermal omitted its resolved diagnostic: {path}")
    return output, result.stdout


def movements(output: Path) -> list[dict]:
    result = []
    for path in sorted((output / "storm_track").glob("track_step_*.json")):
        record = json.loads(path.read_text())
        validity = record["moving_nest_scientific_validity"]
        if validity["status"] == "invalid":
            raise RuntimeError(f"invalid child history: {path}")
        if not record["state_mutation"]:
            continue
        authority = record["movement_authority"]
        if validity["trusted_interior"] is not True or authority["nested_movement_sequence"] is None:
            raise RuntimeError("child moved without trusted-interior ownership")
        inventory = authority["two_way_inventory"]
        if set(inventory["currencies"]) != set(CURRENCIES):
            raise RuntimeError("relocation omitted a conserved currency")
        for name, values in inventory["currencies"].items():
            if not all(math.isfinite(value) for value in values.values()):
                raise RuntimeError(f"nonfinite relocation {name}")
            residual = values["after"] - values["before"] + values["retired"] - values["admitted"] - values["boundary_source"]
            tolerance = 128 * math.ulp(1.0) * max(1.0, values["arithmetic_scale"])
            if abs(residual) > tolerance:
                raise RuntimeError(f"relocation {name} does not close: {residual} > {tolerance}")
        result.append({"time_s": record["simulation_time_s"], "sequence": authority["nested_movement_sequence"],
                       "shift_cells": authority["committed_shift_cells"],
                       "child_origin": authority["post_shift_child_grid_origin_t0_earth_m"], "inventory": inventory})
    with (output / "nests/child_0000/energy_budget.csv").open() as stream:
        energy_rows = {float(row["sim_time_s"]): row for row in csv.DictReader(stream) if row["stage"] == "total_step"}
    for event in result:
        row = energy_rows.get(event["time_s"])
        if row is None:
            continue
        for quantity in ENERGY:
            after = event["inventory"]["currencies"][f"{quantity}_energy"]["after"]
            actual = float(row[f"moist_v2_before_{quantity}_j"])
            if abs(after - actual) > 64 * math.ulp(max(1.0, abs(after), abs(actual))):
                raise RuntimeError(f"relocation {quantity} differs from the next child energy state")
    return result


def verify_frame(output: Path, events: list[dict]) -> None:
    parent = json.loads((output / "step_000001/manifest.json").read_text())
    child = json.loads((output / "nests/child_0000/step_000001/manifest.json").read_text())
    ratio_x = (child["grid"]["dimensions"]["x"] - 2) // 10
    ratio_y = (child["grid"]["dimensions"]["y"] - 2) // 10
    expected = [6000 - 1000 / ratio_x + 1000 * sum(event["shift_cells"]["i"] for event in events),
                7000 - 1000 / ratio_y + 1000 * sum(event["shift_cells"]["j"] for event in events)]
    if parent["grid"]["horizontal_frame"]["grid_origin_t0_earth_m"] != [0, 0]:
        raise RuntimeError("moving-child profile moved the parent frame")
    actual = child["grid"]["horizontal_frame"]["grid_origin_t0_earth_m"]
    if any(abs(a - b) > 8 * math.ulp(max(abs(a), abs(b))) for a, b in zip(actual, expected)):
        raise RuntimeError("child output geometry disagrees with committed shifts")


def verify_relocation_output_momentum(output: Path, event: dict) -> None:
    frame = output / "nests/child_0000/step_000001"
    manifest = json.loads((frame / "manifest.json").read_text())
    if manifest["simulation_time_s"] != event["time_s"]:
        raise RuntimeError("relocation momentum reference is not the immediate post-move output")
    grid = manifest["grid"]
    geometry = grid.get("vertical_geometry", {})
    if geometry.get("storage") != "versioned_npy_geometry":
        return
    shape, _, rho = read_float32_npy(frame / "rho.npy")
    _, zface = read_npy((frame / geometry["interface_path"]).resolve())
    nx, ny, nz = shape
    area = grid["spacing_m"]["x"] * grid["spacing_m"]["y"]
    for field, component in (("u", "x"), ("v", "y"), ("w", "z")):
        values = read_float32_npy(frame / f"{field}.npy")[2]
        terms = []
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(nz):
                    scalar = (i * ny + j) * nz + k
                    face = (i * ny + j) * (nz + 1) + k
                    terms.append(rho[scalar] * area * (zface[face + 1] - zface[face]) * values[scalar])
        actual = math.fsum(terms)
        expected = event["inventory"]["currencies"][f"physical_{component}_momentum"]["after"]
        limit = 8 * 2**-23 * max(1.0, math.fsum(map(abs, terms)))
        if abs(actual - expected) > limit:
            raise RuntimeError(f"relocation {component} momentum differs from physical output: {actual} vs {expected}")


def compare(first: Path, second: Path, *, resumed: bool = False) -> None:
    for location in (Path(), Path("nests/child_0000")):
        frames = [root / location / "step_000001" for root in (first, second)]
        geometries = [json.loads((frame / "manifest.json").read_text())["grid"].get("vertical_geometry", {})
                      for frame in frames]
        for key in ("kind", "storage", "content_sha256"):
            if geometries[0].get(key) != geometries[1].get(key):
                raise RuntimeError(f"terrain continuation/threading mismatch: {location}/{key}")
        for key in ("center_path", "interface_path"):
            if key in geometries[0]:
                paths = [frame / geometry[key] for frame, geometry in zip(frames, geometries)]
                if paths[0].read_bytes() != paths[1].read_bytes():
                    raise RuntimeError(f"terrain continuation/threading mismatch: {location}/{key}")
        published = [{path.name for path in frame.glob("*.npy")} for frame in frames]
        if published[0] != published[1] or not {f"{field}.npy" for field in FIELDS}.issubset(published[0]):
            raise RuntimeError(f"published field set changed: {location}")
        for field in sorted(published[0]):
            a = frames[0] / field
            b = frames[1] / field
            if a.read_bytes() != b.read_bytes():
                raise RuntimeError(f"field continuation/threading mismatch: {location}/{field}")
        if resumed:
            require_restart_suffix(first / location, second / location)
        else:
            for name in LEDGERS:
                if (first / location / name).read_bytes() != (second / location / name).read_bytes():
                    raise RuntimeError(f"ledger threading mismatch: {location}/{name}")
    a = checkpoint_at(first / "checkpoints", 120)
    b = checkpoint_at(second / "checkpoints", 120)
    for section in range(1, 8):
        if checkpoint_section(a, section) != checkpoint_section(b, section):
            raise RuntimeError(f"checkpoint continuation/threading mismatch in section {section}")


def verify_environment(output: Path) -> float:
    base = output / "step_000000"
    shape, _, theta = read_float32_npy(base / "theta.npy")
    fields = {name: read_float32_npy(base / f"{name}.npy")[2] for name in ("p", "qv", "qc")}
    nx, ny, nz = shape
    humidity = []
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(nz):
                index = (i * ny + j) * nz + k
                if fields["qc"][index] != 0:
                    continue
                pressure = fields["p"][index]
                temperature = theta[index] * (pressure / 100000.0) ** (287.0 / 1004.0)
                es = 611.2 * math.exp(17.67 * (temperature - 273.15) / (temperature - 29.65))
                humidity.append(fields["qv"][index] / (.622 * es / (pressure - es)))
    if not humidity or max(humidity) >= .99 or max(fields["qc"]) <= 0:
        raise RuntimeError("reference did not initialize a subsaturated environment and localized cloud seed")
    return max(humidity)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--restart-coverage", choices=("full", "synchronized"), default="full")
    args = parser.parse_args()
    selected_config = args.config.resolve()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise RuntimeError("moving-child evidence directory must be empty")
    with tempfile.TemporaryDirectory(prefix="nimbus-moving-two-way-") as temporary:
        binary = Path(temporary) / "nimbus_sim"
        shutil.copy2(args.binary.resolve(), binary)
        identity = hashlib.sha256(binary.read_bytes()).hexdigest()
        preflight = subprocess.run([str(binary), "--validate-only", "--config", str(selected_config)], cwd=ROOT,
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (root / "preflight.log").write_text(preflight.stdout)
        if preflight.returncode or "[VALIDATE] PASS" not in preflight.stdout:
            raise RuntimeError(f"public preflight failed: {preflight.stdout}")
        source = selected_config.read_text()
        backend = re.search(r"(?m)^    backend: (\w+)", source)[1]
        sounding = ROOT / "tests/fixtures/configs/qualification/soundings/moving_two_way_thermal.txt"
        configuration_hash = hashlib.sha256(selected_config.read_bytes()).hexdigest()
        sounding_hash = hashlib.sha256(sounding.read_bytes()).hexdigest()
        child_width = int(re.search(r"(?m)^    child_cells_x: (\d+)", source)[1])
        for label, before, after in (
            ("fallback", "    allow_fallback: false\n", "    allow_fallback: true\n"),
            ("child-width", f"    child_cells_x: {child_width}\n", f"    child_cells_x: {child_width - 2}\n"),
            ("feedback", "    feedback: two_way\n", "    feedback: one_way\n"),
        ):
            text = source.replace(before, after, 1)
            if text == source:
                raise RuntimeError(f"invalid mutation {label}")
            config = root / f"invalid-{label}.yaml"
            config.write_text(text)
            for mode in ("--validate-only", "--headless"):
                invalid_output = root / f"invalid-{label}-output"
                command = [str(binary), mode, "--config", str(config)]
                if mode == "--headless":
                    command += ["--duration=0", "--write-every=0", f"--outdir={invalid_output}"]
                result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                (root / f"invalid-{label}-{mode[2:]}.log").write_text(result.stdout)
                if result.returncode == 0 or (invalid_output / "nimbus_output.json").exists():
                    raise RuntimeError(f"{label} {mode} bypassed ordinary admission")
        serial, serial_log = execute(binary, root, "serial", 1, config=selected_config)
        serial_moves = movements(serial)
        if not serial_moves:
            raise RuntimeError("ordinary moving-child profile never relocated")
        parallel, parallel_log = execute(binary, root, "parallel", 4, config=selected_config)
        if movements(parallel) != serial_moves:
            raise RuntimeError("relocation reports changed with worker count")
        compare(serial, parallel)
        verify_frame(serial, serial_moves)
        verify_frame(parallel, serial_moves)
        max_background_rh = verify_environment(serial)
        water = require_water_closure(serial_log)
        dry = require_closure(serial_log, "moving child")
        if require_water_closure(parallel_log) != water or require_closure(parallel_log, "parallel moving child") != dry:
            raise RuntimeError("feedback ledgers changed with worker count")
        first_move = float(serial_moves[0]["time_s"])
        times = sorted(json.loads(path.read_text())["simulation_time_s"] for path in (serial / "checkpoints").glob("*.json"))
        if args.restart_coverage == "full":
            before_move = max(time for time in times if time < first_move)
            after_move = min(time for time in times if time > first_move)
            immediate, _ = execute(binary, root, "immediate-move", 1, duration=first_move, cadence=first_move, config=selected_config)
            if movements(immediate) != [serial_moves[0]]:
                raise RuntimeError("immediate checkpoint branch changed the first relocation")
            verify_relocation_output_momentum(immediate, serial_moves[0])
            checkpoints = [(before_move, checkpoint_at(serial / "checkpoints", before_move)),
                           (first_move, checkpoint_at(immediate / "checkpoints", first_move)),
                           (after_move, checkpoint_at(serial / "checkpoints", after_move))]
        else:
            synchronized = min(time for time in times if first_move <= time < 120)
            checkpoints = [(synchronized, checkpoint_at(serial / "checkpoints", synchronized))]
        for time, checkpoint in checkpoints:
            for threads in ((1, 4) if args.restart_coverage == "full" else (4,)):
                resumed, log = execute(binary, root, f"resumed-{time:g}-t{threads}", threads, cadence=120, resume=checkpoint, config=selected_config)
                compare(serial, resumed, resumed=True)
                verify_frame(resumed, serial_moves)
                if movements(resumed) != [move for move in serial_moves if move["time_s"] > time]:
                    raise RuntimeError("relocation history changed across restart")
                for observed, reference in ((require_water_closure(log), water), (require_closure(log, "resumed moving child"), dry)):
                    for name, values in observed.items():
                        if values != reference[name][-len(values):]:
                            raise RuntimeError(f"{name} feedback ledger changed across restart")
                print(f"passed restart at {time:g}s with {threads} workers", flush=True)
        if hashlib.sha256(selected_config.read_bytes()).hexdigest() != configuration_hash or hashlib.sha256(sounding.read_bytes()).hexdigest() != sounding_hash:
            raise RuntimeError("moving-child reference inputs changed during the gate")
        (root / "summary.json").write_text(json.dumps({
            "state": "ready", "backend": backend, "binary_sha256": identity, "configuration_sha256": configuration_hash, "sounding_sha256": sounding_hash,
            "duration_s": 120, "threading": "exact", "restart": "exact", "restart_times_s": [time for time, _ in checkpoints],
            "restart_coverage": args.restart_coverage,
            "movements": serial_moves, "maximum_background_relative_humidity": max_background_rh,
            "negative_admission": True, "campaign_validated": False, "externally_reviewed": False,
        }, indent=2) + "\n")
    print("[moving-two-way-ready] PASS: public relocation, mass/water/momentum/energy closure, exact threading and restart")


if __name__ == "__main__":
    main()
