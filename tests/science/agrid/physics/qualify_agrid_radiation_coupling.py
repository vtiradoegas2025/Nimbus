#!/usr/bin/env python3
"""Qualify production-coupled simple-grey radiation on both Cartesian paths."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict, dataclass
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
    eos_relative_residual,
    load_fields,
    parse_frame,
    physical_indices,
    read_json_object,
    require,
    valid_generation,
    validate_summary,
)


@dataclass(frozen=True)
class Thresholds:
    # One-step attribution passes through an FP32 dtheta/dt buffer; the
    # cumulative bound below rejects bias at an order of magnitude tighter.
    maximum_step_ledger_relative_residual: float = 1.0e-5
    maximum_cumulative_ledger_relative_residual: float = 5.0e-6
    maximum_timestep_theta_difference_k: float = 5.0e-3
    maximum_cadence_theta_difference_k: float = 1.0e-2
    # A ~200 W m-2 column imbalance over 120 s implies millikelvin-scale
    # resolved response. A smaller signal means source increments were lost to
    # FP32 storage even when the flux-to-request ledger itself closes.
    minimum_theta_response_k: float = 2.0e-3
    maximum_horizontal_theta_spread_k: float = 2.0e-5
    maximum_wind_ms: float = 0.1
    maximum_eos_relative_residual: float = 2.0e-6


def thresholds_for_duration(duration_s: float) -> Thresholds:
    if duration_s < 3600.0:
        return Thresholds()
    # The short gate's timestep/cadence differences accumulate approximately
    # linearly with simulated duration. Preserve their error-rate envelope for
    # the one-hour gate, while keeping balance/symmetry and wind limits fixed:
    # those are physical invariants, not truncation-error allowances.
    return Thresholds(
        maximum_timestep_theta_difference_k=0.1,
        maximum_cadence_theta_difference_k=0.05,
        minimum_theta_response_k=0.05,
    )


CASES = (
    ("dt0p2_c10", 0.2, 10.0),
    ("dt0p1_c10", 0.1, 10.0),
    ("dt0p2_c60", 0.2, 60.0),
)


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    require(text.count(old) == 1, f"{label}: expected exactly one {old!r}")
    return text.replace(old, new)


def materialize_config(template: Path, destination: Path,
                       dt_s: float, cadence_s: float) -> None:
    text = template.read_text(encoding="utf-8")
    text = replace_exact(
        text, "  dt_large: 0.2\n", f"  dt_large: {dt_s:g}\n", str(template))
    text = replace_exact(
        text, "  cadence_s: 10\n", f"  cadence_s: {cadence_s:g}\n", str(template))
    destination.write_text(text, encoding="utf-8")


def run_case(binary: Path, config: Path, outdir: Path, duration_s: float,
             omp_threads: int) -> None:
    command = [
        str(binary), "--headless", f"--config={config}",
        f"--duration={duration_s:g}", f"--write-every={duration_s:g}",
        f"--outdir={outdir}", "--compute-backend=cpu",
        "--guard-mode=strict", "--log-profile=quiet",
    ]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(omp_threads)
    environment["OMP_DYNAMIC"] = "FALSE"
    subprocess.run(command, check=True, env=environment)


def load_run(run_dir: Path):
    marker = read_json_object(run_dir / "nimbus_output.json")
    require(marker.get("schema") == "nimbus.output", f"{run_dir}: wrong output schema")
    require(marker.get("publication_state") == "ready", f"{run_dir}: output is not ready")
    generation = marker.get("publication_generation")
    require(valid_generation(generation), f"{run_dir}: invalid publication generation")
    assert isinstance(generation, str)
    frames = sorted(
        (parse_frame(path, generation) for path in run_dir.iterdir()
         if path.is_dir() and STEP_DIR_PATTERN.fullmatch(path.name)),
        key=lambda frame: frame.time_s,
    )
    require(len(frames) == 2, f"{run_dir}: expected initial and final frames")
    require(abs(frames[0].time_s) <= 1.0e-9, f"{run_dir}: initial frame is not t=0")
    validate_summary(run_dir)
    return frames, [load_fields(frame) for frame in frames]


def read_ledger(run_dir: Path, dt_s: float, cadence_s: float,
                duration_s: float, thresholds: Thresholds) -> dict[str, object]:
    path = run_dir / "radiation_budget.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_steps = int(round(duration_s / dt_s))
    require(len(rows) == expected_steps,
            f"{path}: expected {expected_steps} applied steps, got {len(rows)}")
    require([int(row["step"]) for row in rows] == list(range(1, expected_steps + 1)),
            f"{path}: ledger step sequence is discontinuous")

    numeric_columns = tuple(key for key in rows[0] if key != "step")
    for row in rows:
        require(all(math.isfinite(float(row[key])) for key in numeric_columns),
                f"{path}: non-finite ledger value")

    samples: list[float] = []
    for row in rows:
        value = float(row["last_radiation_sample_s"])
        if not samples or value != samples[-1]:
            samples.append(value)
    expected_samples = [
        cadence_s * index
        for index in range(int(math.ceil(duration_s / cadence_s)))
    ]
    require(len(samples) == len(expected_samples),
            f"{path}: cadence produced {samples}, expected {expected_samples}")
    require(all(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-8)
                for actual, expected in zip(samples, expected_samples)),
            f"{path}: radiation samples drifted from the nominal cadence: {samples}")

    max_step_relative = max(float(row["relative_booking_residual"]) for row in rows)
    final = rows[-1]
    cumulative_expected = float(final["cumulative_expected_energy_j"])
    cumulative_residual = float(final["cumulative_booking_residual_j"])
    cumulative_relative = abs(cumulative_residual) / max(1.0, abs(cumulative_expected))
    require(max_step_relative <= thresholds.maximum_step_ledger_relative_residual,
            f"{path}: step ledger residual {max_step_relative:.9g} exceeds bound")
    require(cumulative_relative <= thresholds.maximum_cumulative_ledger_relative_residual,
            f"{path}: cumulative ledger residual {cumulative_relative:.9g} exceeds bound")
    require(cumulative_expected < 0.0,
            f"{path}: prescribed nighttime LW case did not cool the atmosphere")
    return {
        "row_count": len(rows),
        "sample_count": len(samples),
        "first_sample_time_s": samples[0],
        "last_sample_time_s": samples[-1],
        "maximum_step_relative_residual": max_step_relative,
        "cumulative_relative_residual": cumulative_relative,
        "cumulative_expected_energy_j": cumulative_expected,
        "cumulative_booked_heat_j": float(final["cumulative_booked_heat_j"]),
    }


def max_field_difference(lhs, rhs, field: str, indices) -> float:
    return max(abs(lhs[field].at(*index) - rhs[field].at(*index)) for index in indices)


def analyze_path(root: Path, path_name: str, duration_s: float,
                 thresholds: Thresholds) -> dict[str, object]:
    loaded: dict[str, tuple[object, object, list[tuple[int, int, int]]]] = {}
    ledgers: dict[str, object] = {}
    for case_name, dt_s, cadence_s in CASES:
        run_dir = root / f"{path_name}_{case_name}"
        frames, fields = load_run(run_dir)
        require(math.isclose(frames[-1].time_s, duration_s, abs_tol=1.0e-8),
                f"{run_dir}: final frame time differs from requested duration")
        indices = list(physical_indices(frames[0]))
        require(indices, f"{run_dir}: empty physical domain")
        loaded[case_name] = (frames, fields, indices)
        ledgers[case_name] = read_ledger(
            run_dir, dt_s, cadence_s, duration_s, thresholds)

    baseline_frames, baseline_fields, indices = loaded["dt0p2_c10"]
    reference_frames, reference_fields, reference_indices = loaded["dt0p1_c10"]
    cadence_frames, cadence_fields, cadence_indices = loaded["dt0p2_c60"]
    require(indices == reference_indices == cadence_indices,
            f"{path_name}: variant physical domains differ")
    del reference_frames, cadence_frames

    timestep_theta_error = max_field_difference(
        baseline_fields[-1], reference_fields[-1], "theta", indices)
    cadence_theta_error = max_field_difference(
        baseline_fields[-1], cadence_fields[-1], "theta", indices)
    require(timestep_theta_error <= thresholds.maximum_timestep_theta_difference_k,
            f"{path_name}: timestep theta difference {timestep_theta_error:.9g} K exceeds bound")
    require(cadence_theta_error <= thresholds.maximum_cadence_theta_difference_k,
            f"{path_name}: cadence theta difference {cadence_theta_error:.9g} K exceeds bound")

    initial, final = baseline_fields
    theta_response = max(abs(final["theta"].at(*index) - initial["theta"].at(*index))
                         for index in indices)
    require(theta_response >= thresholds.minimum_theta_response_k,
            f"{path_name}: radiation response is below the resolvable floor")
    levels: dict[int, list[float]] = {}
    for index in indices:
        levels.setdefault(index[2], []).append(final["theta"].at(*index))
    horizontal_spread = max(max(values) - min(values) for values in levels.values())
    require(horizontal_spread <= thresholds.maximum_horizontal_theta_spread_k,
            f"{path_name}: horizontally uniform forcing produced theta spread "
            f"{horizontal_spread:.9g} K")
    max_wind = max(math.sqrt(
        final["u"].at(*index) ** 2 + final["v"].at(*index) ** 2
        + final["w"].at(*index) ** 2) for index in indices)
    require(max_wind <= thresholds.maximum_wind_ms,
            f"{path_name}: coupled column wind {max_wind:.9g} m/s exceeds bound")
    max_eos = max(
        eos_relative_residual(initial["rho"], initial["p"], initial["theta"], baseline_frames[0]),
        eos_relative_residual(final["rho"], final["p"], final["theta"], baseline_frames[-1]),
    )
    require(max_eos <= thresholds.maximum_eos_relative_residual,
            f"{path_name}: EOS residual {max_eos:.9g} exceeds bound")
    return {
        "physical_cell_count": len(indices),
        "ledger": ledgers,
        "metrics": {
            "maximum_timestep_theta_difference_k": timestep_theta_error,
            "maximum_cadence_theta_difference_k": cadence_theta_error,
            "maximum_theta_response_k": theta_response,
            "maximum_horizontal_theta_spread_k": horizontal_spread,
            "maximum_wind_ms": max_wind,
            "maximum_eos_relative_residual": max_eos,
        },
    }


def analyze_agrid_dynamics_control(
    root: Path, duration_s: float, thresholds: Thresholds
) -> dict[str, object]:
    run_dir = root / "agrid_no_radiation_periodic"
    frames, fields = load_run(run_dir)
    require(math.isclose(frames[-1].time_s, duration_s, abs_tol=1.0e-8),
            f"{run_dir}: final frame time differs from requested duration")
    indices = list(physical_indices(frames[0]))
    require(indices, f"{run_dir}: empty physical domain")
    initial, final = fields

    horizontal_spreads: dict[str, float] = {}
    for field_name in ("rho", "p", "theta", "qv", "u", "v", "w"):
        levels: dict[int, list[float]] = {}
        for index in indices:
            levels.setdefault(index[2], []).append(final[field_name].at(*index))
        horizontal_spreads[field_name] = max(
            max(values) - min(values) for values in levels.values())
        require(horizontal_spreads[field_name]
                <= thresholds.maximum_horizontal_theta_spread_k,
                f"agrid no-radiation periodic control produced horizontal "
                f"{field_name} spread {horizontal_spreads[field_name]:.9g}")

    max_wind = max(math.sqrt(
        final["u"].at(*index) ** 2 + final["v"].at(*index) ** 2
        + final["w"].at(*index) ** 2) for index in indices)
    require(max_wind <= 0.01,
            f"agrid no-radiation periodic wind {max_wind:.9g} m/s exceeds bound")

    drift_limits = {
        "rho": 1.0e-4,
        "p": 2.0,
        "theta": 2.0e-2,
        "qv": 1.0e-6,
    }
    maximum_drifts = {
        field_name: max(abs(final[field_name].at(*index)
                            - initial[field_name].at(*index))
                        for index in indices)
        for field_name in drift_limits
    }
    for field_name, limit in drift_limits.items():
        require(maximum_drifts[field_name] <= limit,
                f"agrid no-radiation periodic {field_name} drift "
                f"{maximum_drifts[field_name]:.9g} exceeds {limit:.9g}")

    return {
        "physical_cell_count": len(indices),
        "maximum_wind_ms": max_wind,
        "maximum_horizontal_spreads": horizontal_spreads,
        "maximum_absolute_drifts": maximum_drifts,
        "drift_limits": drift_limits,
    }


def qualify(root: Path, duration_s: float,
            thresholds: Thresholds | None = None) -> dict[str, object]:
    active_thresholds = thresholds or thresholds_for_duration(duration_s)
    return {
        "status": "PASS",
        "scope": "production_coupled_clear_sky_lw_cartesian_paths_with_agrid_dynamics_control",
        "duration_s": duration_s,
        "paths": {
            "agrid": analyze_path(root, "agrid", duration_s, active_thresholds),
            "cgrid": analyze_path(root, "cgrid", duration_s, active_thresholds),
        },
        "agrid_no_radiation_periodic_control":
            analyze_agrid_dynamics_control(root, duration_s, active_thresholds),
        "thresholds": asdict(active_thresholds),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="number of independent simulator cases to run concurrently")
    parser.add_argument(
        "--threads-per-case", type=int, default=1,
        help="fixed OpenMP thread count for each simulator process")
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="validate an existing --work-dir without launching simulations")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration <= 60.0:
        parser.error("duration must be finite and greater than 60 seconds")
    maximum_jobs = len(CASES) * 2 + 1
    if args.jobs < 1 or args.jobs > maximum_jobs:
        parser.error(f"jobs must be between 1 and {maximum_jobs}")
    if args.threads_per_case < 1 or args.threads_per_case > 256:
        parser.error("threads-per-case must be between 1 and 256")
    if args.analyze_only and args.work_dir is None:
        parser.error("--analyze-only requires --work-dir")

    repo = Path(__file__).resolve().parents[4]
    created_temp = args.work_dir is None
    work_root = args.work_dir or Path(tempfile.mkdtemp(prefix="nimbus-radiation-"))
    work_root.mkdir(parents=True, exist_ok=True)
    try:
        paths = {
            "agrid": (repo / "bin/nimbus_sim",
                      repo / "tests/fixtures/configs/qualification/agrid_radiation_coupled.yaml"),
            "cgrid": (repo / "bin/nimbus_sim",
                      repo / "tests/fixtures/configs/qualification/cgrid_radiation_coupled.yaml"),
        }
        if not args.analyze_only:
            run_specs: list[tuple[Path, Path, Path, float, int]] = []
            for path_name, (binary, template) in paths.items():
                require(binary.is_file(), f"missing simulator binary: {binary}")
                for case_name, dt_s, cadence_s in CASES:
                    config = work_root / f"{path_name}_{case_name}.yaml"
                    outdir = work_root / f"{path_name}_{case_name}"
                    materialize_config(template, config, dt_s, cadence_s)
                    run_specs.append((
                        binary, config, outdir, args.duration,
                        args.threads_per_case))
            run_specs.append((
                repo / "bin/nimbus_sim",
                repo / "tests/fixtures/configs/qualification/agrid_sustained_periodic_none.yaml",
                work_root / "agrid_no_radiation_periodic",
                args.duration,
                args.threads_per_case))
            if args.jobs == 1:
                for spec in run_specs:
                    run_case(*spec)
            else:
                # Cases have disjoint configuration/output trees and the
                # checker consumes them in a fixed order after every process
                # completes. Bounded concurrency changes wall time, not
                # numerical ordering.
                with ThreadPoolExecutor(max_workers=args.jobs) as executor:
                    futures = [
                        executor.submit(run_case, *spec) for spec in run_specs]
                    for future in futures:
                        future.result()
        result = qualify(work_root, args.duration)
    except (QualificationError, subprocess.CalledProcessError) as error:
        if created_temp and os.environ.get("NIMBUS_QUALIFICATION_KEEP") != "1":
            shutil.rmtree(work_root, ignore_errors=True)
        parser.exit(1, f"[cartesian-radiation] FAIL: {error}\n")

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if created_temp and os.environ.get("NIMBUS_QUALIFICATION_KEEP") == "1":
        print(f"[cartesian-radiation] retained artifacts: {work_root}")
    elif created_temp:
        shutil.rmtree(work_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
