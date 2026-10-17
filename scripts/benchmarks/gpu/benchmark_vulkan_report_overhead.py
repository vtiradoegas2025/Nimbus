#!/usr/bin/env python3
"""Paired physical-device benchmark for Vulkan scientific-report overhead."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import struct
import subprocess
import tempfile
import time


CORE_FIELDS = ("p", "qr", "qv", "rho", "theta", "u", "v", "w")
STEP_PATTERN = re.compile(r"step_[0-9]{6}$")
PERF_PATTERN = re.compile(
    r"\[PERF SUMMARY\].*measured_steps=([0-9]+).*"
    r"step_wall_s=([0-9.eE+-]+).*avg_step_ms=([0-9.eE+-]+)"
)
ACOUSTIC_PROMOTION_PATTERN = re.compile(
    r"promoting to ([0-9]+)\. Positive acoustic_substeps"
)
DYNAMICS_PERF_PATTERN = re.compile(r"\[DYNAMICS PERF\]\s*(.*)")
STAGE_TELEMETRY_PATTERN = re.compile(
    r"\[GPU STAGE TELEMETRY\]\s+(\S+)\s+(.*)"
)
TELEMETRY_KEYS = (
    "model_steps",
    "command_buffers",
    "dispatches",
    "host_upload_bytes",
    "host_download_bytes",
    "device_upload_bytes",
    "device_download_bytes",
    "resident_hits",
    "resident_misses",
    "resident_upload_bytes_saved",
    "writeonly_output_upload_bytes_saved",
    "batched_command_buffers",
    "synchronous_waits",
    "descriptor_cache_hits",
    "descriptor_cache_misses",
    "staging_high_water_bytes",
    "device_high_water_bytes",
    "staging_memory_allocations_high_water",
    "device_memory_allocations_high_water",
    "staging_memory_reserved_high_water_bytes",
    "device_memory_reserved_high_water_bytes",
)
STAGE_TELEMETRY_KEYS = (
    "command_buffers",
    "dispatches",
    "host_upload_bytes",
    "host_download_bytes",
    "device_upload_bytes",
    "device_download_bytes",
    "resident_hits",
    "resident_misses",
    "resident_upload_bytes_saved",
    "writeonly_output_upload_bytes_saved",
    "descriptor_cache_hits",
    "descriptor_cache_misses",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def normalized_config_contract(path: Path) -> tuple[list[str], int]:
    lines = []
    cadence: int | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        content = raw_line.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        match = re.fullmatch(
            r"(\s*)scientific_every_steps:\s*([0-9]+)\s*", content
        )
        if match:
            require(cadence is None,
                    f"{path}: duplicate scientific_every_steps")
            cadence = int(match.group(2))
            content = f"{match.group(1)}scientific_every_steps: <REPORT_MODE>"
        lines.append(content)
    require(cadence is not None, f"{path}: missing scientific_every_steps")
    return lines, cadence


def require_config_pair_contract(
    report_free: Path,
    report_bearing: Path,
) -> None:
    free_lines, free_cadence = normalized_config_contract(report_free)
    report_lines, report_cadence = normalized_config_contract(report_bearing)
    require(
        free_lines == report_lines,
        "report-free/report-bearing configs differ outside "
        "scientific_every_steps",
    )
    require(
        free_cadence == 0 and report_cadence == 1,
        "expected scientific_every_steps 0 versus 1",
    )


def final_frame(run_dir: Path) -> Path:
    frames = sorted(
        path for path in run_dir.iterdir()
        if path.is_dir() and STEP_PATTERN.fullmatch(path.name)
    )
    require(frames, f"{run_dir.name}: no field frame")
    return frames[-1]


def read_npy_float32(path: Path) -> tuple[tuple[int, ...], tuple[float, ...]]:
    with path.open("rb") as handle:
        require(handle.read(6) == b"\x93NUMPY", f"{path}: not an NPY file")
        major, minor = handle.read(2)
        if (major, minor) == (1, 0):
            header_size = struct.unpack("<H", handle.read(2))[0]
        elif major in (2, 3):
            header_size = struct.unpack("<I", handle.read(4))[0]
        else:
            raise RuntimeError(
                f"{path}: unsupported NPY version {major}.{minor}")
        encoding = "utf-8" if major == 3 else "latin1"
        header = ast.literal_eval(handle.read(header_size).decode(encoding))
        require(
            header.get("descr") in ("<f4", "|f4")
            and header.get("fortran_order") is False,
            f"{path}: expected C-order little-endian float32",
        )
        shape = header.get("shape")
        require(
            isinstance(shape, tuple) and bool(shape),
            f"{path}: invalid shape",
        )
        payload = handle.read()
    count = math.prod(shape)
    require(len(payload) == count * 4, f"{path}: payload size mismatch")
    values = tuple(value for (value,) in struct.iter_unpack("<f", payload))
    require(
        all(math.isfinite(value) for value in values),
        f"{path}: non-finite value",
    )
    return shape, values


def sample_rss_kib(pid: int) -> int:
    result = subprocess.run(
        ("ps", "-o", "rss=", "-p", str(pid)),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return 0
    return int(result.stdout.strip().splitlines()[-1])


def parse_telemetry(stdout: str, label: str) -> dict[str, int]:
    lines = [
        line for line in stdout.splitlines()
        if "[GPU TELEMETRY]" in line
    ]
    require(lines, f"{label}: missing final GPU telemetry")
    line = lines[-1]
    values: dict[str, int] = {}
    for key in TELEMETRY_KEYS:
        match = re.search(rf"\b{re.escape(key)}=([0-9]+)", line)
        require(match is not None, f"{label}: telemetry missing {key}")
        values[key] = int(match.group(1))
    return values


def parse_dynamics_perf(stdout: str, label: str) -> dict[str, float]:
    matches = DYNAMICS_PERF_PATTERN.findall(stdout)
    require(matches, f"{label}: missing dynamics performance summary")
    values: dict[str, float] = {}
    for item in matches[-1].split(","):
        key, separator, raw_value = item.strip().partition("=")
        require(separator == "=" and key, f"{label}: malformed dynamics metric")
        value = float(raw_value)
        require(math.isfinite(value) and value >= 0.0,
                f"{label}: invalid dynamics metric {key}")
        values[key] = value
    return values


def parse_stage_telemetry(
    stdout: str,
    label: str,
) -> dict[str, dict[str, int]]:
    stages: dict[str, dict[str, int]] = {}
    for stage, payload in STAGE_TELEMETRY_PATTERN.findall(stdout):
        require(
            stage not in stages,
            f"{label}: duplicate GPU stage telemetry for {stage}",
        )
        values: dict[str, int] = {}
        for item in payload.split():
            key, separator, raw_value = item.partition("=")
            require(
                separator == "=" and key and raw_value.isdigit(),
                f"{label}: malformed GPU stage metric for {stage}",
            )
            values[key] = int(raw_value)
        missing = set(STAGE_TELEMETRY_KEYS) - set(values)
        require(
            not missing,
            f"{label}: GPU stage {stage} missing {sorted(missing)}",
        )
        stages[stage] = {
            key: values[key] for key in STAGE_TELEMETRY_KEYS
        }
    require(stages, f"{label}: missing GPU stage telemetry")
    return stages


def reconcile_stage_telemetry(
    stages: dict[str, dict[str, int]],
    telemetry: dict[str, int],
    label: str,
) -> dict[str, int]:
    residual: dict[str, int] = {}
    for key in STAGE_TELEMETRY_KEYS:
        stage_total = sum(stage[key] for stage in stages.values())
        require(
            stage_total <= telemetry[key],
            f"{label}: GPU stage {key} total {stage_total} exceeds "
            f"global value {telemetry[key]}",
        )
        residual[key] = telemetry[key] - stage_total
    require(
        residual["command_buffers"] == 0
        and residual["dispatches"] == 0,
        f"{label}: GPU command/dispatch work exists outside named stages",
    )
    return residual


def run_trial(
    binary: Path,
    config: Path,
    outdir: Path,
    duration: int,
    label: str,
    report_bearing: bool,
    acoustic_substeps: int | None,
    dt_large: float | None,
) -> dict[str, object]:
    command = [
        str(binary),
        "--headless",
        f"--config={config}",
        "--compute-backend=vulkan",
        "--guard-mode=off",
        f"--duration={duration}",
        f"--write-every={duration}",
        f"--outdir={outdir}",
        "--timing",
        "--timing-every=0",
    ]
    if acoustic_substeps is not None:
        command.append(f"--acoustic-substeps={acoustic_substeps}")
    if dt_large is not None:
        command.append(f"--dt-large={dt_large:g}")
    environment = os.environ.copy()
    environment["NIMBUS_GPU_KERNEL_TIMING"] = "1"
    environment["NIMBUS_GPU_RESIDENT"] = "1"
    environment["NIMBUS_GPU_LIN_BATCH"] = "1"

    started = time.perf_counter()
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=environment,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        peak_rss_kib = 0
        while process.poll() is None:
            peak_rss_kib = max(
                peak_rss_kib, sample_rss_kib(process.pid))
            time.sleep(0.02)
        process.wait()
        log.seek(0)
        stdout = log.read()
    process_wall_s = time.perf_counter() - started

    require(
        process.returncode == 0,
        f"{label}: simulator exited {process.returncode}\n{stdout[-5000:]}",
    )
    selected_device_lines = [
        line.strip() for line in stdout.splitlines()
        if "selected device" in line
    ]
    require(selected_device_lines,
            f"{label}: no physical Vulkan device selected")
    require("Falling back to CPU backend" not in stdout,
            f"{label}: Vulkan fell back to CPU")
    expected_diagnostic = (
        "Scientific diagnostics: every 1 steps"
        if report_bearing else
        "Scientific diagnostics: off"
    )
    require(
        expected_diagnostic in stdout,
        f"{label}: unexpected scientific-report cadence",
    )

    perf_matches = PERF_PATTERN.findall(stdout)
    require(perf_matches, f"{label}: missing performance summary")
    measured_steps, step_wall_s, avg_step_ms = perf_matches[-1]
    telemetry = parse_telemetry(stdout, label)
    dynamics_perf = parse_dynamics_perf(stdout, label)
    stage_telemetry = parse_stage_telemetry(stdout, label)
    unattributed_stage_telemetry = reconcile_stage_telemetry(
        stage_telemetry, telemetry, label)
    if any(unattributed_stage_telemetry.values()):
        stage_telemetry["__unattributed_runtime__"] = (
            unattributed_stage_telemetry
        )
    promotions = ACOUSTIC_PROMOTION_PATTERN.findall(stdout)
    realized_acoustic_substeps = (
        int(promotions[-1])
        if promotions else acoustic_substeps
    )
    if acoustic_substeps is not None:
        require(
            realized_acoustic_substeps == acoustic_substeps,
            f"{label}: requested {acoustic_substeps} acoustic substeps but "
            f"runtime realized {realized_acoustic_substeps}",
        )
    require(
        int(measured_steps) > 0
        and telemetry["model_steps"] > 0
        and telemetry["dispatches"] > 0,
        f"{label}: physical-device work was vacuous",
    )
    return {
        "label": label,
        "measured_steps": int(measured_steps),
        "step_wall_s": float(step_wall_s),
        "avg_step_ms": float(avg_step_ms),
        "process_wall_s": process_wall_s,
        "peak_rss_kib": peak_rss_kib,
        "telemetry": telemetry,
        "stage_telemetry": stage_telemetry,
        "dynamics_perf_s": dynamics_perf,
        "final_frame": str(final_frame(outdir)),
        "command": command,
        "selected_device_evidence": selected_device_lines[-1],
        "realized_acoustic_substeps": realized_acoustic_substeps,
    }


def require_same_endpoint(
    report_free: dict[str, object],
    report_bearing: dict[str, object],
) -> None:
    free_frame = Path(str(report_free["final_frame"]))
    report_frame = Path(str(report_bearing["final_frame"]))
    for field in CORE_FIELDS:
        free_path = free_frame / f"{field}.npy"
        report_path = report_frame / f"{field}.npy"
        if free_path.read_bytes() == report_path.read_bytes():
            continue
        free_shape, free_values = read_npy_float32(free_path)
        report_shape, report_values = read_npy_float32(report_path)
        require(
            free_shape == report_shape
            and len(free_values) == len(report_values),
            f"reporting changed final {field} shape",
        )
        differences = [
            (abs(report - free), index, free, report)
            for index, (free, report) in enumerate(
                zip(free_values, report_values))
            if free != report
        ]
        worst_abs, worst_index, free_value, report_value = max(differences)
        scale = max(abs(free_value), abs(report_value), 1.0e-30)
        raise RuntimeError(
            f"reporting changed final {field} bytes: "
            f"mismatches={len(differences)}/{len(free_values)} "
            f"worst_index={worst_index} worst_abs={worst_abs:.9g} "
            f"worst_rel={worst_abs / scale:.9g} "
            f"free={free_value:.9g} report={report_value:.9g}"
        )


def median(values: list[float | int]) -> float:
    return float(statistics.median(values))


def robust_statistics(values: list[float]) -> dict[str, object]:
    require(bool(values), "cannot summarize an empty sample")
    center = median(values)
    absolute_deviations = [
        abs(value - center) for value in values
    ]
    mad = median(absolute_deviations)
    return {
        "samples": values,
        "median": center,
        "median_absolute_deviation": mad,
        "three_robust_sigma": 3.0 * 1.4826 * mad,
    }


def summarize(
    free_trials: list[dict[str, object]],
    report_trials: list[dict[str, object]],
) -> dict[str, object]:
    def values(
        trials: list[dict[str, object]], key: str
    ) -> list[float]:
        return [float(trial[key]) for trial in trials]

    free_step = median(values(free_trials, "avg_step_ms"))
    report_step = median(values(report_trials, "avg_step_ms"))
    require(free_step > 0.0 and report_step > 0.0,
            "invalid non-positive median step time")
    measured_step_counts = {
        int(trial["measured_steps"])
        for trial in free_trials + report_trials
    }
    require(
        len(measured_step_counts) == 1,
        "measured step count changed across paired trials",
    )
    measured_steps = next(iter(measured_step_counts))
    require(measured_steps > 0, "measured step count must be positive")
    require(
        all(
            int(trial["telemetry"]["model_steps"]) == measured_steps  # type: ignore[index]
            for trial in free_trials + report_trials
        ),
        "GPU telemetry model-step count does not match performance summary",
    )
    free_step_samples = values(free_trials, "avg_step_ms")
    report_step_samples = values(report_trials, "avg_step_ms")
    paired_delta_samples = [
        report - free
        for free, report in zip(free_step_samples, report_step_samples)
    ]
    paired_ratio_samples = [
        report / free
        for free, report in zip(free_step_samples, report_step_samples)
    ]

    telemetry: dict[str, dict[str, float]] = {}
    for key in TELEMETRY_KEYS:
        free_median = median([
            int(trial["telemetry"][key])  # type: ignore[index]
            for trial in free_trials
        ])
        report_median = median([
            int(trial["telemetry"][key])  # type: ignore[index]
            for trial in report_trials
        ])
        telemetry[key] = {
            "report_free_median": free_median,
            "report_bearing_median": report_median,
            "delta": report_median - free_median,
        }

    dynamics_keys = set(free_trials[0]["dynamics_perf_s"])  # type: ignore[arg-type]
    require(
        all(
            set(trial["dynamics_perf_s"]) == dynamics_keys  # type: ignore[arg-type]
            for trial in free_trials + report_trials
        ),
        "dynamics performance keys changed across trials",
    )
    dynamics_perf: dict[str, dict[str, float | str]] = {}
    for key in sorted(dynamics_keys):
        free_median = median([
            float(trial["dynamics_perf_s"][key])  # type: ignore[index]
            for trial in free_trials
        ])
        report_median = median([
            float(trial["dynamics_perf_s"][key])  # type: ignore[index]
            for trial in report_trials
        ])
        if key.endswith("_s"):
            dynamics_perf[key] = {
                "unit": "seconds_per_run",
                "report_free_median": free_median,
                "report_bearing_median": report_median,
                "delta": report_median - free_median,
                "report_free_median_ms_per_step":
                    1000.0 * free_median / measured_steps,
                "report_bearing_median_ms_per_step":
                    1000.0 * report_median / measured_steps,
                "delta_ms_per_step":
                    1000.0 * (report_median - free_median) / measured_steps,
            }
        else:
            require(
                key.endswith("_fraction"),
                f"unclassified dynamics performance metric {key}",
            )
            dynamics_perf[key] = {
                "unit": "fraction",
                "report_free_median": free_median,
                "report_bearing_median": report_median,
                "delta": report_median - free_median,
            }

    stage_names = sorted({
        stage
        for trial in free_trials + report_trials
        for stage in trial["stage_telemetry"]  # type: ignore[union-attr]
    })
    stage_telemetry: dict[
        str, dict[str, dict[str, float]]
    ] = {}
    for stage in stage_names:
        stage_summary: dict[str, dict[str, float]] = {}
        for key in STAGE_TELEMETRY_KEYS:
            free_median = median([
                int(
                    trial["stage_telemetry"]  # type: ignore[index]
                        .get(stage, {}).get(key, 0)
                )
                for trial in free_trials
            ])
            report_median = median([
                int(
                    trial["stage_telemetry"]  # type: ignore[index]
                        .get(stage, {}).get(key, 0)
                )
                for trial in report_trials
            ])
            stage_summary[key] = {
                "report_free_median": free_median,
                "report_bearing_median": report_median,
                "delta": report_median - free_median,
                "report_free_median_per_step":
                    free_median / measured_steps,
                "report_bearing_median_per_step":
                    report_median / measured_steps,
                "delta_per_step":
                    (report_median - free_median) / measured_steps,
            }
        stage_telemetry[stage] = stage_summary

    return {
        "measured_steps_per_trial": measured_steps,
        "report_free_median_avg_step_ms": free_step,
        "report_bearing_median_avg_step_ms": report_step,
        "report_bearing_step_time_ratio": report_step / free_step,
        "report_bearing_step_time_delta_ms": report_step - free_step,
        "step_time_statistics": {
            "report_free_ms": robust_statistics(free_step_samples),
            "report_bearing_ms": robust_statistics(report_step_samples),
            "paired_delta_ms": robust_statistics(paired_delta_samples),
            "paired_ratio": robust_statistics(paired_ratio_samples),
        },
        "report_free_median_process_wall_s": median(
            values(free_trials, "process_wall_s")),
        "report_bearing_median_process_wall_s": median(
            values(report_trials, "process_wall_s")),
        "report_free_median_peak_rss_kib": median(
            values(free_trials, "peak_rss_kib")),
        "report_bearing_median_peak_rss_kib": median(
            values(report_trials, "peak_rss_kib")),
        "telemetry": telemetry,
        "dynamics_perf": dynamics_perf,
        "stage_telemetry": stage_telemetry,
    }


def serializable_trial(trial: dict[str, object]) -> dict[str, object]:
    result = dict(trial)
    result.pop("final_frame", None)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("bin/nimbus_sim"))
    parser.add_argument(
        "--report-free-config",
        type=Path,
        default=Path("tests/fixtures/configs/qualification/gpu_report_free.yaml"),
    )
    parser.add_argument(
        "--report-bearing-config",
        type=Path,
        default=Path("tests/fixtures/configs/qualification/gpu_report_bearing.yaml"),
    )
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument(
        "--acoustic-substeps",
        type=int,
        default=None,
        help="Optional runtime acoustic-substep override.",
    )
    parser.add_argument(
        "--dt-large",
        type=float,
        default=None,
        help="Optional fixed outer-timestep override.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "reports/benchmarks/vulkan-report-overhead-current.json"),
    )
    args = parser.parse_args()

    require(args.binary.is_file(), f"missing simulator: {args.binary}")
    require(
        args.report_free_config.is_file(),
        f"missing report-free config: {args.report_free_config}",
    )
    require(
        args.report_bearing_config.is_file(),
        f"missing report-bearing config: {args.report_bearing_config}",
    )
    require_config_pair_contract(
        args.report_free_config, args.report_bearing_config)
    require(args.duration > 0, "duration must be positive")
    require(args.trials >= 3 and args.trials % 2 == 1,
            "trials must be an odd integer of at least three")
    require(
        args.acoustic_substeps is None or args.acoustic_substeps > 0,
        "acoustic substeps must be positive",
    )
    require(
        args.dt_large is None
        or (math.isfinite(args.dt_large) and args.dt_large > 0.0),
        "dt-large must be positive and finite",
    )

    free_trials: list[dict[str, object]] = []
    report_trials: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(
        prefix="nimbus_vulkan_report_pareto_"
    ) as root_text:
        root = Path(root_text)
        for trial_index in range(args.trials):
            order = (
                ("free", "report")
                if trial_index % 2 == 0 else
                ("report", "free")
            )
            pair: dict[str, dict[str, object]] = {}
            for mode in order:
                report_bearing = mode == "report"
                label = f"trial-{trial_index + 1}-{mode}"
                result = run_trial(
                    args.binary,
                    (
                        args.report_bearing_config
                        if report_bearing else
                        args.report_free_config
                    ),
                    root / label,
                    args.duration,
                    label,
                    report_bearing,
                    args.acoustic_substeps,
                    args.dt_large,
                )
                pair[mode] = result
                (
                    report_trials if report_bearing else free_trials
                ).append(result)
            require_same_endpoint(pair["free"], pair["report"])
            print(
                "[vulkan-report-pareto] "
                f"trial={trial_index + 1}/{args.trials} "
                f"free={float(pair['free']['avg_step_ms']):.6f} ms "
                f"report={float(pair['report']['avg_step_ms']):.6f} ms"
            )

        summary = summarize(free_trials, report_trials)
        device_evidence = {
            str(trial["selected_device_evidence"])
            for trial in free_trials + report_trials
        }
        require(
            len(device_evidence) == 1,
            "physical-device identity changed across trials",
        )
        realized_substeps = {
            trial["realized_acoustic_substeps"]
            for trial in free_trials + report_trials
        }
        require(
            len(realized_substeps) == 1,
            "realized acoustic-substep count changed across trials",
        )
        report = {
            "schema": "nimbus.vulkan-report-overhead.v3",
            "status": "measured",
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "physical_device_required": True,
            "selected_device_evidence": next(iter(device_evidence)),
            "identity": {
                "binary": str(args.binary.resolve()),
                "binary_sha256": sha256_file(args.binary),
                "report_free_config": str(args.report_free_config.resolve()),
                "report_free_config_sha256": sha256_file(
                    args.report_free_config),
                "report_bearing_config": str(
                    args.report_bearing_config.resolve()),
                "report_bearing_config_sha256": sha256_file(
                    args.report_bearing_config),
            },
            "duration_s": args.duration,
            "trials": args.trials,
            "runtime_overrides": {
                "acoustic_substeps": args.acoustic_substeps,
                "realized_acoustic_substeps": next(
                    iter(realized_substeps)),
                "dt_large_s": args.dt_large,
            },
            "measurement_scope": (
                "minimum-two-state acoustic plus non-acoustic/system "
                "report overhead"
                if args.acoustic_substeps == 2
                else "complete configured report overhead"
            ),
            "configuration_difference_contract":
                "scientific_every_steps 0 versus 1 only",
            "endpoint_fields_bitwise_equal": list(CORE_FIELDS),
            "summary": summary,
            "report_free_trials": [
                serializable_trial(trial) for trial in free_trials
            ],
            "report_bearing_trials": [
                serializable_trial(trial) for trial in report_trials
            ],
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    ratio = float(summary["report_bearing_step_time_ratio"])
    delta_ms = float(summary["report_bearing_step_time_delta_ms"])
    rss_delta = (
        float(summary["report_bearing_median_peak_rss_kib"])
        - float(summary["report_free_median_peak_rss_kib"])
    )
    require(math.isfinite(ratio) and math.isfinite(delta_ms),
            "non-finite benchmark summary")
    print(
        "[vulkan-report-pareto] RESULT: MEASURED "
        f"trials={args.trials} duration={args.duration}s "
        f"acoustic_substeps={args.acoustic_substeps or 'config'} "
        f"dt_large={args.dt_large or 'config'} "
        f"endpoint_fields={len(CORE_FIELDS)} "
        f"step_ratio={ratio:.6f} delta_ms={delta_ms:.6f} "
        f"rss_delta_kib={rss_delta:.0f} report={args.report}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"[vulkan-report-pareto] RESULT: FAIL: {error}")
        raise SystemExit(1)
