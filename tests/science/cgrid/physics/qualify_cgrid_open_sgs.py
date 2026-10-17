#!/usr/bin/env python3
"""Qualify sustained vertical/full SGS with open lateral and upper faces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = ROOT / "tests/fixtures/configs/qualification/cgrid_open_sgs_coupled.yaml"
CORE_FIELDS = ("p", "qr", "qv", "rho", "theta", "u", "v", "w")
LEDGERS = (
    "budget.csv",
    "energy_budget.csv",
    "momentum_budget.csv",
    "moisture_budget.csv",
    "radiation_budget.csv",
    "thermodynamic_source_budget.csv",
)
DURATION_S = 80.0
CHECKPOINT_S = 40.0
STEPS = 160
CHECKPOINT_STEP = 80
GPU_TELEMETRY_PATTERN = re.compile(
    r"\[GPU TELEMETRY\].*model_steps=([0-9]+).*dispatches=([0-9]+)"
)
GPU_SGS_STAGE = "turbulence/smagorinsky_coefficients.comp.spv"
GPU_SGS_APPLY_STAGE = "turbulence/sgs_apply.comp.spv"


class QualificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(binary: Path, config: Path, outdir: Path, threads: int,
        backend: str, resume: Path | None = None) -> dict[str, int]:
    command = [
        str(binary), "--headless", f"--config={config}",
        f"--duration={DURATION_S:g}", f"--outdir={outdir}",
        f"--compute-backend={backend}", "--guard-mode=strict",
        f"--threads={threads}",
    ]
    if backend == "cpu":
        command.append("--log-profile=quiet")
    if resume is not None:
        command.append(f"--resume={resume}")
    environment = os.environ.copy()
    if backend == "vulkan":
        environment["NIMBUS_GPU_KERNEL_TIMING"] = "1"
        environment["NIMBUS_GPU_RESIDENT"] = "1"
        environment["NIMBUS_GPU_LIN_BATCH"] = "1"
    result = subprocess.run(
        command, cwd=ROOT, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    require(result.returncode == 0,
            f"{outdir.name}: simulator exited {result.returncode}\n"
            f"{result.stdout[-5000:]}")
    if backend == "cpu":
        return {"model_steps": (STEPS if resume is None
                                else STEPS - CHECKPOINT_STEP),
                "dispatches": 0,
                "sgs_coefficient_dispatches": 0,
                "sgs_apply_dispatches": 0}

    require("[COMPUTE][VULKAN] selected device[" in result.stdout,
            f"{outdir.name}: no physical Vulkan device selected")
    require("Falling back to CPU backend" not in result.stdout,
            f"{outdir.name}: Vulkan fell back to CPU")
    telemetry = GPU_TELEMETRY_PATTERN.findall(result.stdout)
    require(telemetry, f"{outdir.name}: missing GPU telemetry")
    model_steps, dispatches = (int(value) for value in telemetry[-1])
    require(model_steps > 0 and dispatches > 0,
            f"{outdir.name}: Vulkan execution was vacuous")
    stage_line = next(
        (line for line in result.stdout.splitlines()
         if line.startswith(f"[GPU STAGE TELEMETRY] {GPU_SGS_STAGE} ")),
        "")
    require(stage_line, f"{outdir.name}: stochastic SGS coefficient stage "
            "did not execute on Vulkan")
    match = re.search(r"dispatches=([0-9]+)", stage_line)
    require(match is not None and int(match.group(1)) > 0,
            f"{outdir.name}: stochastic SGS device dispatch was vacuous")
    apply_line = next(
        (line for line in result.stdout.splitlines()
         if line.startswith(
             f"[GPU STAGE TELEMETRY] {GPU_SGS_APPLY_STAGE} ")),
        "")
    require(apply_line, f"{outdir.name}: SGS state commit did not execute "
            "on Vulkan")
    apply_match = re.search(r"dispatches=([0-9]+)", apply_line)
    require(apply_match is not None and int(apply_match.group(1)) > 0,
            f"{outdir.name}: SGS state-commit dispatch was vacuous")
    return {
        "model_steps": model_steps,
        "dispatches": dispatches,
        "sgs_coefficient_dispatches": int(match.group(1)),
        "sgs_apply_dispatches": int(apply_match.group(1)),
    }


def checkpoint_records(run_dir: Path) -> list[dict[str, object]]:
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_dir / "checkpoints").glob("*.nimbus.json"))
    ]
    require(records, f"{run_dir.name}: no checkpoint sidecars")
    require(all(record.get("boundary") == "completed_timestep"
                for record in records),
            f"{run_dir.name}: checkpoint outside completed timestep")
    return records


def checkpoint_path(run_dir: Path, step: int) -> Path:
    path = run_dir / "checkpoints" / f"checkpoint_step_{step:09d}.nimbus"
    require(path.is_file(), f"{run_dir.name}: missing checkpoint step {step}")
    return path


def final_frame(run_dir: Path) -> Path:
    frames = sorted(path for path in run_dir.glob("step_*") if path.is_dir())
    require(frames, f"{run_dir.name}: no output frames")
    return frames[-1]


def require_field_identity(reference: Path, candidate: Path,
                           label: str) -> None:
    ref_frame = final_frame(reference)
    candidate_frame = final_frame(candidate)
    for field in CORE_FIELDS:
        require((ref_frame / f"{field}.npy").read_bytes()
                == (candidate_frame / f"{field}.npy").read_bytes(),
                f"{label}: final {field} differs")


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(rows, f"{path}: empty ledger")
    for row in rows:
        for key, value in row.items():
            if key == "stage":
                continue
            require(value is not None and math.isfinite(float(value)),
                    f"{path}: non-finite {key}")
    return rows


def require_ledger_identity(reference: Path, parallel: Path,
                            resumed: Path) -> None:
    for name in LEDGERS:
        ref_bytes = (reference / name).read_bytes()
        require(ref_bytes == (parallel / name).read_bytes(),
                f"{name}: thread count changed full ledger bytes")
        with (reference / name).open(
                "r", encoding="utf-8", newline="") as handle:
            ref_reader = csv.DictReader(handle)
            ref_fields = ref_reader.fieldnames
            ref_rows = list(ref_reader)
        with (resumed / name).open(
                "r", encoding="utf-8", newline="") as handle:
            resumed_reader = csv.DictReader(handle)
            resumed_fields = resumed_reader.fieldnames
            resumed_rows = list(resumed_reader)
        require(ref_fields == resumed_fields,
                f"{name}: restart changed the ledger schema")
        if ref_fields is not None and "step" in ref_fields:
            expected = [row for row in ref_rows
                        if int(row["step"]) > CHECKPOINT_STEP]
        elif ref_fields is not None and "sim_time_s" in ref_fields:
            expected = [row for row in ref_rows
                        if float(row["sim_time_s"]) >= CHECKPOINT_S]
        elif ref_fields is not None and "time_s" in ref_fields:
            expected = [row for row in ref_rows
                        if float(row["time_s"]) > CHECKPOINT_S]
        else:
            raise QualificationError(
                f"{name}: no scientific sequence coordinate")
        require(resumed_rows == expected,
                f"{name}: restart is not the exact post-checkpoint suffix")


def inspect_open_stress(run_dir: Path,
                        require_nonzero_boundary_work: bool) -> dict[str, float | int]:
    rows = csv_rows(run_dir / "thermodynamic_source_budget.csv")
    require(len(rows) == STEPS, "source ledger has the wrong step count")
    require([int(row["sgs_momentum_work_step_count"]) for row in rows]
            == list(range(1, STEPS + 1)),
            "SGS work ledger sequence is discontinuous")
    require([int(row["sgs_open_boundary_stability_step_count"])
             for row in rows] == list(range(1, STEPS + 1)),
            "open-stress stability provenance is discontinuous")

    max_step_closure = 0.0
    max_cumulative_closure = 0.0
    maximum_boundary_work = 0.0
    minimum_dissipation = math.inf
    for row in rows:
        boundary = float(row["sgs_open_boundary_work_j"])
        dissipation = float(row["sgs_irreversible_dissipation_j"])
        residual = float(row["sgs_stress_closure_residual_j"])
        cumulative_boundary = float(
            row["sgs_cumulative_open_boundary_work_j"])
        cumulative_dissipation = float(
            row["sgs_cumulative_irreversible_dissipation_j"])
        cumulative_residual = float(
            row["sgs_cumulative_stress_closure_residual_j"])
        max_step_closure = max(
            max_step_closure,
            abs(residual) / max(1.0, abs(boundary), abs(dissipation)))
        max_cumulative_closure = max(
            max_cumulative_closure,
            abs(cumulative_residual)
            / max(1.0, abs(cumulative_boundary),
                  abs(cumulative_dissipation)))
        maximum_boundary_work = max(maximum_boundary_work, abs(boundary))
        minimum_dissipation = min(minimum_dissipation, dissipation)

    require(minimum_dissipation > 0.0,
            "open-stress SGS did not remain irreversibly dissipative")
    if require_nonzero_boundary_work:
        require(maximum_boundary_work > 0.0,
                "open-face SGS traction work was never exercised")
    require(max_step_closure <= 1.0e-12,
            f"per-step open-stress identity failed: {max_step_closure}")
    require(max_cumulative_closure <= 1.0e-12,
            "cumulative open-stress identity failed: "
            f"{max_cumulative_closure}")

    radiation = csv_rows(run_dir / "radiation_budget.csv")
    require(len(radiation) == STEPS, "radiation ledger is incomplete")
    max_radiation_residual = max(
        abs(float(row["relative_booking_residual"])) for row in radiation)
    require(max_radiation_residual <= 1.0e-10,
            "radiation booking residual exceeded its bound")

    max_microphysics_residual = max(
        abs(float(row["microphysics_conservation_residual_kg"]))
        / max(1.0, abs(float(row["microphysics_baseline_water_kg"])))
        for row in rows)
    require(max_microphysics_residual <= 1.0e-5,
            "microphysics conservation residual exceeded its bound")
    require(all(int(row["sgs_moist_energy_step_complete"]) == 1
                for row in rows),
            "SGS moist-energy attribution is incomplete")

    return {
        "maximum_step_stress_closure_relative_residual": max_step_closure,
        "maximum_cumulative_stress_closure_relative_residual":
            max_cumulative_closure,
        "maximum_absolute_boundary_work_j": maximum_boundary_work,
        "minimum_irreversible_dissipation_j": minimum_dissipation,
        "maximum_radiation_booking_relative_residual":
            max_radiation_residual,
        "maximum_microphysics_conservation_relative_residual":
            max_microphysics_residual,
        "open_boundary_steps": STEPS,
    }


def qualify_mode(binary: Path, config: Path, root: Path,
                 mode: str, backend: str) -> dict[str, object]:
    serial = root / f"{mode}_serial"
    parallel = root / f"{mode}_parallel"
    resumed = root / f"{mode}_resumed"
    serial_telemetry = run(binary, config, serial, 1, backend)
    records = checkpoint_records(serial)
    require([int(record["completed_steps"]) for record in records]
            == [CHECKPOINT_STEP, STEPS],
            f"{mode}: wrong checkpoint sequence")
    branch = checkpoint_path(serial, CHECKPOINT_STEP)
    parallel_telemetry = run(binary, config, parallel, 4, backend)
    resumed_telemetry = run(binary, config, resumed, 4, backend, branch)

    require_field_identity(serial, parallel, f"{mode} one/four thread")
    require_field_identity(serial, resumed, f"{mode} restart")
    final_checkpoint = checkpoint_path(serial, STEPS).read_bytes()
    require(final_checkpoint == checkpoint_path(parallel, STEPS).read_bytes(),
            f"{mode}: thread count changed final checkpoint bytes")
    require(final_checkpoint == checkpoint_path(resumed, STEPS).read_bytes(),
            f"{mode}: restart changed final checkpoint bytes")
    require_ledger_identity(serial, parallel, resumed)
    return {
        "mode": mode,
        "continuous_steps": STEPS,
        "restart_step": CHECKPOINT_STEP,
        "exact_fields": list(CORE_FIELDS),
        "exact_ledgers": list(LEDGERS),
        "execution": {
            "backend": backend,
            "serial": serial_telemetry,
            "parallel": parallel_telemetry,
            "resumed": resumed_telemetry,
        },
        "metrics": inspect_open_stress(
            serial, require_nonzero_boundary_work=mode == "3d"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path,
                        default=Path("bin/nimbus_sim_cpu"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--backend", choices=("cpu", "vulkan"),
                        default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    binary = args.binary if args.binary.is_absolute() else ROOT / args.binary
    config = args.config if args.config.is_absolute() else ROOT / args.config
    require(binary.is_file(), f"missing simulator: {binary}")
    require(config.is_file(), f"missing configuration: {config}")

    owned_temp = args.output is None
    work = (Path(tempfile.mkdtemp(prefix="nimbus_open_sgs_"))
            if owned_temp else args.output.resolve())
    work.mkdir(parents=True, exist_ok=True)
    try:
        source = config.read_text(encoding="utf-8")
        marker = "  mode: 3d\n"
        require(source.count(marker) == 1,
                "qualification source must contain exactly one 3d SGS mode")
        vertical_config = work / "vertical_only.yaml"
        vertical_config.write_text(
            source.replace(marker, "  mode: vertical_only\n"),
            encoding="utf-8")
        result = {
            "status": "PASS",
            "scope": (f"{args.backend}_flat_cgrid_open_faces_"
                      "stochastic_smagorinsky"),
            "source_config": str(config.relative_to(ROOT)),
            "provenance": {
                "binary": str(binary.relative_to(ROOT)),
                "binary_sha256": sha256(binary),
                "source_config_sha256": sha256(config),
                "derived_vertical_config_sha256": sha256(vertical_config),
            },
            "modes": [
                qualify_mode(binary, vertical_config, work, "vertical_only",
                             args.backend),
                qualify_mode(binary, config, work, "3d", args.backend),
            ],
        }
        (work / "qualification.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            f"[open-sgs] PASS ({args.backend}): vertical_only + 3d; "
            "6 sustained branches; "
            "exact one/four-thread fields/checkpoints/ledgers; exact "
            "40s restart suffixes; 320 open-stress identities")
        return 0
    except (QualificationError, subprocess.CalledProcessError) as error:
        print(f"[open-sgs] FAIL: {error}")
        return 1
    finally:
        if owned_temp and not args.keep:
            shutil.rmtree(work, ignore_errors=True)
        elif args.keep or not owned_temp:
            print(f"[open-sgs] artifacts: {work}")


if __name__ == "__main__":
    raise SystemExit(main())
