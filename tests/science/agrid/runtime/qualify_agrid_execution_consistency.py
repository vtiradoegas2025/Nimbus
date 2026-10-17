#!/usr/bin/env python3
"""Qualify sustained A-grid checkpoint and CPU-thread consistency."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Sequence


COMMON_TOOLS = Path(__file__).resolve().parents[2] / "common"
if str(COMMON_TOOLS) not in sys.path:
    sys.path.insert(0, str(COMMON_TOOLS))

from check_hydrostatic_hold import (
    QualificationError,
    STEP_DIR_PATTERN,
    parse_frame,
    read_json_object,
    require,
    valid_generation,
)


CORE_FIELDS = ("p", "qr", "qv", "rho", "theta", "u", "v", "w")
LEDGER_FILES = (
    "budget.csv",
    "energy_budget.csv",
    "moisture_budget.csv",
    "momentum_budget.csv",
    "radiation_budget.csv",
    "thermodynamic_source_budget.csv",
)


def validate_strict_summary(run_dir: Path) -> None:
    path = run_dir / "validation_summary.json"
    summary = read_json_object(path)
    require(summary.get("context") == "run_summary",
            f"{path}: wrong context")
    require(summary.get("guard_mode") == "strict",
            f"{path}: strict guard was not active")
    require(summary.get("failed") is False,
            f"{path}: validation summary failed")
    for key in (
        "failed_report_count", "total_violations", "total_nonfinite",
        "total_bounds", "total_sanitized_nonfinite",
        "total_sanitized_bounds",
    ):
        require(summary.get(key) == 0, f"{path}: {key} must be zero")
    report_count = summary.get("report_count")
    require(isinstance(report_count, int) and not isinstance(report_count, bool)
            and report_count >= 1,
            f"{path}: expected at least one strict validation report")


def run(binary: Path, config: Path, outdir: Path, duration_s: int,
        checkpoint_every_s: int, threads: int,
        resume: Path | None = None) -> None:
    command = [
        str(binary), "--headless", f"--config={config}",
        f"--duration={duration_s}", f"--outdir={outdir}",
        f"--write-every={duration_s}",
        "--compute-backend=cpu", "--guard-mode=strict",
        "--log-profile=quiet",
        f"--checkpoint-every={checkpoint_every_s}",
    ]
    if resume is not None:
        command.append(f"--resume={resume}")
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["OMP_DYNAMIC"] = "FALSE"
    subprocess.run(command, check=True, env=environment)


def frames(run_dir: Path):
    marker = read_json_object(run_dir / "nimbus_output.json")
    require(marker.get("schema") == "nimbus.output",
            f"{run_dir}: wrong output schema")
    require(marker.get("publication_state") == "ready",
            f"{run_dir}: output is not ready")
    generation = marker.get("publication_generation")
    require(valid_generation(generation), f"{run_dir}: invalid generation")
    assert isinstance(generation, str)
    result = sorted(
        (parse_frame(path, generation) for path in run_dir.iterdir()
         if path.is_dir() and STEP_DIR_PATTERN.fullmatch(path.name)),
        key=lambda frame: frame.time_s,
    )
    validate_strict_summary(run_dir)
    return result


def checkpoint_sidecars(run_dir: Path) -> list[dict[str, object]]:
    paths = sorted((run_dir / "checkpoints").glob("*.nimbus.json"))
    result = [read_json_object(path) for path in paths]
    require(result, f"{run_dir}: no checkpoints")
    require(all(item.get("boundary") == "completed_timestep" for item in result),
            f"{run_dir}: checkpoint outside completed timestep")
    return result


def checkpoint_path(run_dir: Path, sidecar: dict[str, object]) -> Path:
    filename = sidecar.get("checkpoint_file")
    require(isinstance(filename, str) and filename,
            f"{run_dir}: malformed checkpoint sidecar")
    result = run_dir / "checkpoints" / filename
    require(result.is_file(), f"{run_dir}: missing checkpoint payload {filename}")
    return result


def require_field_identity(reference_dir: Path, candidate_dir: Path,
                           duration_s: int, label: str) -> None:
    reference_frames = frames(reference_dir)
    candidate_frames = frames(candidate_dir)
    require(reference_frames and candidate_frames,
            f"{label}: missing output frames")
    require(math.isclose(reference_frames[-1].time_s, duration_s,
                         abs_tol=1.0e-8)
            and math.isclose(candidate_frames[-1].time_s, duration_s,
                             abs_tol=1.0e-8),
            f"{label}: final frame is not at {duration_s} s")
    for field in CORE_FIELDS:
        reference = reference_frames[-1].path / f"{field}.npy"
        candidate = candidate_frames[-1].path / f"{field}.npy"
        require(reference.read_bytes() == candidate.read_bytes(),
                f"{label}: final {field} bytes differ")


def require_ledger_identity(reference_dir: Path, candidate_dir: Path,
                            label: str) -> None:
    for filename in LEDGER_FILES:
        reference = reference_dir / filename
        candidate = candidate_dir / filename
        require(reference.read_bytes() == candidate.read_bytes(),
                f"{label}: {filename} bytes differ")


def require_restart_ledger_suffix(continuous_dir: Path, resumed_dir: Path,
                                  label: str) -> None:
    for filename in LEDGER_FILES:
        continuous = (continuous_dir / filename).read_text(
            encoding="utf-8").splitlines()
        resumed = (resumed_dir / filename).read_text(
            encoding="utf-8").splitlines()
        require(continuous and resumed,
                f"{label}: {filename} is empty")
        require(continuous[0] == resumed[0],
                f"{label}: {filename} header changed on restart")
        if len(resumed) == 1:
            require(len(continuous) == 1,
                    f"{label}: {filename} lost diagnostic rows on restart")
            continue
        require(continuous[-(len(resumed) - 1):] == resumed[1:],
                f"{label}: {filename} did not continue with an exact suffix")


def qualify(work_root: Path, duration_s: int,
            parallel_threads: int) -> dict[str, object]:
    repo = Path(__file__).resolve().parents[4]
    binary = repo / "bin/nimbus_sim"
    config = repo / "tests/fixtures/configs/qualification/agrid_sustained_periodic_none.yaml"
    require(binary.is_file(), f"missing simulator: {binary}")
    require(config.is_file(), f"missing config: {config}")

    midpoint_s = duration_s // 2
    serial = work_root / "serial_continuous"
    serial_resumed = work_root / "serial_resumed"
    parallel = work_root / "parallel_continuous"
    parallel_resumed = work_root / "parallel_resumed"

    run(binary, config, serial, duration_s, midpoint_s, 1)
    serial_sidecars = checkpoint_sidecars(serial)
    require([int(item["simulation_time_s"]) for item in serial_sidecars]
            == [midpoint_s, duration_s],
            f"{serial}: expected midpoint and final checkpoints")
    branch_checkpoint = checkpoint_path(serial, serial_sidecars[0])

    run(binary, config, serial_resumed, duration_s, midpoint_s, 1,
        branch_checkpoint)
    run(binary, config, parallel, duration_s, midpoint_s, parallel_threads)
    run(binary, config, parallel_resumed, duration_s, midpoint_s,
        parallel_threads, branch_checkpoint)

    require_field_identity(serial, serial_resumed, duration_s,
                           "same-thread restart")
    require_field_identity(serial, parallel, duration_s,
                           "one-vs-N-thread continuous")
    require_field_identity(serial, parallel_resumed, duration_s,
                           "one-to-N-thread restart")
    require_ledger_identity(serial, parallel,
                            "one-vs-N-thread continuous")
    require_ledger_identity(serial_resumed, parallel_resumed,
                            "one-vs-N-thread resumed")
    require_restart_ledger_suffix(serial, serial_resumed,
                                  "same-thread restart")
    require_restart_ledger_suffix(parallel, parallel_resumed,
                                  "one-to-N-thread restart")

    final_serial = checkpoint_path(serial, checkpoint_sidecars(serial)[-1])
    checkpoint_comparisons = {
        "same_thread_restart": checkpoint_path(
            serial_resumed, checkpoint_sidecars(serial_resumed)[-1]),
        "one_vs_n_thread_continuous": checkpoint_path(
            parallel, checkpoint_sidecars(parallel)[-1]),
        "one_to_n_thread_restart": checkpoint_path(
            parallel_resumed, checkpoint_sidecars(parallel_resumed)[-1]),
    }
    reference_bytes = final_serial.read_bytes()
    for label, candidate in checkpoint_comparisons.items():
        require(reference_bytes == candidate.read_bytes(),
                f"{label}: final checkpoint bytes differ")

    return {
        "status": "PASS",
        "scope": "production_agrid_cpu_execution_consistency",
        "duration_s": duration_s,
        "checkpoint_s": midpoint_s,
        "serial_threads": 1,
        "parallel_threads": parallel_threads,
        "bitwise_final_fields": list(CORE_FIELDS),
        "bitwise_final_checkpoint": True,
        "bitwise_thread_ledgers": list(LEDGER_FILES),
        "exact_restart_ledger_suffixes": list(LEDGER_FILES),
        "comparisons": list(checkpoint_comparisons),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--parallel-threads", type=int, default=4)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args(argv)
    if args.duration < 2 or args.duration % 2 != 0:
        parser.error("duration must be a positive even integer of at least 2 s")
    if args.parallel_threads < 2 or args.parallel_threads > 256:
        parser.error("parallel-threads must be between 2 and 256")
    if args.analyze_only and args.work_dir is None:
        parser.error("--analyze-only requires --work-dir")

    created_temp = args.work_dir is None
    work_root = args.work_dir or Path(tempfile.mkdtemp(prefix="nimbus-agrid-execution-"))
    work_root.mkdir(parents=True, exist_ok=True)
    try:
        if args.analyze_only:
            # Analysis is deterministic and does not need to retain the source
            # checkpoint path used to create an existing four-run artifact.
            result = qualify_existing(
                work_root, args.duration, args.parallel_threads)
        else:
            result = qualify(work_root, args.duration, args.parallel_threads)
    except (QualificationError, OSError, subprocess.CalledProcessError) as error:
        if created_temp and os.environ.get("NIMBUS_QUALIFICATION_KEEP") != "1":
            shutil.rmtree(work_root, ignore_errors=True)
        parser.exit(1, f"[agrid-execution] FAIL: {error}\n")

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if created_temp and os.environ.get("NIMBUS_QUALIFICATION_KEEP") == "1":
        print(f"[agrid-execution] retained artifacts: {work_root}")
    elif created_temp:
        shutil.rmtree(work_root, ignore_errors=True)
    return 0


def qualify_existing(work_root: Path, duration_s: int,
                     parallel_threads: int) -> dict[str, object]:
    serial = work_root / "serial_continuous"
    comparisons = {
        "same_thread_restart": work_root / "serial_resumed",
        "one_vs_n_thread_continuous": work_root / "parallel_continuous",
        "one_to_n_thread_restart": work_root / "parallel_resumed",
    }
    for label, candidate in comparisons.items():
        require_field_identity(serial, candidate, duration_s, label)
    require_ledger_identity(
        serial, comparisons["one_vs_n_thread_continuous"],
        "one-vs-N-thread continuous")
    require_ledger_identity(
        comparisons["same_thread_restart"],
        comparisons["one_to_n_thread_restart"],
        "one-vs-N-thread resumed")
    require_restart_ledger_suffix(
        serial, comparisons["same_thread_restart"],
        "same-thread restart")
    require_restart_ledger_suffix(
        comparisons["one_vs_n_thread_continuous"],
        comparisons["one_to_n_thread_restart"],
        "one-to-N-thread restart")
    final_serial = checkpoint_path(serial, checkpoint_sidecars(serial)[-1])
    reference_bytes = final_serial.read_bytes()
    for label, candidate_dir in comparisons.items():
        candidate = checkpoint_path(
            candidate_dir, checkpoint_sidecars(candidate_dir)[-1])
        require(reference_bytes == candidate.read_bytes(),
                f"{label}: final checkpoint bytes differ")
    return {
        "status": "PASS",
        "scope": "production_agrid_cpu_execution_consistency",
        "duration_s": duration_s,
        "checkpoint_s": duration_s // 2,
        "serial_threads": 1,
        "parallel_threads": parallel_threads,
        "bitwise_final_fields": list(CORE_FIELDS),
        "bitwise_final_checkpoint": True,
        "bitwise_thread_ledgers": list(LEDGER_FILES),
        "exact_restart_ledger_suffixes": list(LEDGER_FILES),
        "comparisons": list(comparisons),
    }


if __name__ == "__main__":
    raise SystemExit(main())
