#!/usr/bin/env python3
"""Measure the exact Nimbus CPU workload across an OpenMP thread frontier."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Sequence


REPORT_SCHEMA = "nimbus.cpu-scaling.v3"
CPU_REPORT_SCHEMA = "nimbus.cpu-benchmark.v1"
CLEARED_ENVIRONMENT_PREFIXES = ("NIMBUS_", "OMP_", "KMP_", "GOMP_")
OPENMP_THREAD_REPORT = re.compile(r"OpenMP threads:\s*(\d+)")


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def nonnegative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return result


def default_thread_counts() -> list[int]:
    logical = os.cpu_count() or 1
    counts = [1]
    value = 2
    while value < logical:
        counts.append(value)
        value *= 2
    counts.append(logical)
    if platform.system() == "Darwin":
        try:
            completed = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                text=True,
                capture_output=True,
                timeout=5,
                check=False,
            )
            performance_cores = int(completed.stdout.strip())
            if 0 < performance_cores < logical:
                counts.append(performance_cores)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return sorted(set(counts))


def thread_counts(value: str) -> list[int]:
    try:
        counts = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "thread counts must be comma-separated integers"
        ) from exc
    if not counts or any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError(
            "thread counts must contain positive integers"
        )
    if len(set(counts)) != len(counts):
        raise argparse.ArgumentTypeError("thread counts must be unique")
    return sorted(counts)


def runtime_default_thread_count(
    binary_value: str,
    config_value: str,
    extra_args: Sequence[str],
) -> int | None:
    """Ask Nimbus for the same topology-aware default used by real runs."""
    binary = Path(binary_value).expanduser().resolve()
    config = Path(config_value).expanduser().resolve()
    if not binary.is_file() or not config.is_file():
        return None
    command = [
        str(binary),
        "--headless",
        f"--config={config}",
        *extra_args,
        "--duration=0",
        "--write-every=0",
        "--compute-backend=cpu",
        "--log-profile=normal",
        "--no-timing",
    ]
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith(CLEARED_ENVIRONMENT_PREFIXES):
            environment.pop(key, None)
    try:
        completed = subprocess.run(
            command,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    matches = OPENMP_THREAD_REPORT.findall(
        completed.stdout + "\n" + completed.stderr
    )
    if not matches:
        return None
    count = int(matches[-1])
    return count if count > 0 else None


def nested(report: dict[str, Any], path: str) -> Any:
    value: Any = report
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"CPU report is missing {path!r}")
        value = value[part]
    return value


def median_absolute_deviation(values: Sequence[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_cpu_report(
    report: dict[str, Any],
    expected_threads: int,
    minimum_trials: int,
) -> None:
    if report.get("schema") != CPU_REPORT_SCHEMA:
        raise ValueError(f"CPU report does not use {CPU_REPORT_SCHEMA}")
    trials = report.get("trials")
    if not isinstance(trials, list) or len(trials) < minimum_trials:
        raise ValueError(
            f"CPU report has fewer than {minimum_trials} trials"
        )
    if nested(report, "run.threads") != expected_threads:
        raise ValueError("CPU report thread count disagrees with its request")
    if nested(report, "run.trials") != len(trials):
        raise ValueError("CPU report trial count is internally inconsistent")
    for trial in trials:
        for metric in ("avg_step_ms", "peak_rss_bytes"):
            value = float(trial[metric])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"CPU report has invalid {metric}")


def run_thread_count(
    args: argparse.Namespace,
    count: int,
    trials: int | None = None,
) -> dict[str, Any]:
    trial_count = args.trials if trials is None else trials
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("run_cpu_benchmark.py")),
        "--binary",
        args.binary,
        "--config",
        args.config,
        "--trials",
        str(trial_count),
        "--threads",
        str(count),
        "--timeout",
        str(args.timeout),
        "--compiler",
        args.compiler,
        "--cxxflags",
        args.cxxflags,
        "--ldflags",
        args.ldflags,
        "--build-profile",
        args.build_profile,
    ]
    for extra_arg in args.extra_arg:
        command.append(f"--extra-arg={extra_arg}")
    for setting in args.env:
        command.extend(("--env", setting))
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{count}-thread CPU benchmark failed:\n{completed.stderr}"
        )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{count}-thread CPU benchmark returned invalid JSON: {exc}"
        ) from exc
    validate_cpu_report(report, count, trial_count)
    return report


def identity_signature(report: dict[str, Any]) -> dict[str, Any]:
    paths = (
        "host.system",
        "host.machine",
        "host.logical_cpu_count",
        "build.profile",
        "build.compiler",
        "build.cxxflags",
        "build.ldflags",
        "build.binary_sha256",
        "run.config_sha256",
        "run.metric_scope",
        "run.measured_steps",
        "run.warmup_steps",
        "run.measured_simulated_s",
        "run.memory_plan",
    )
    return {path: nested(report, path) for path in paths}


def summarize_point(report: dict[str, Any]) -> dict[str, Any]:
    step_values = [float(trial["avg_step_ms"]) for trial in report["trials"]]
    rss_values = [int(trial["peak_rss_bytes"]) for trial in report["trials"]]
    step_median = statistics.median(step_values)
    step_mad = median_absolute_deviation(step_values)
    return {
        "threads": int(report["run"]["threads"]),
        "avg_step_ms": {
            "samples": step_values,
            "median": step_median,
            "mad": step_mad,
            "three_robust_sigma": 3.0 * 1.4826 * step_mad,
            "min": min(step_values),
            "max": max(step_values),
        },
        "peak_rss_bytes": {
            "samples": rss_values,
            "median": statistics.median(rss_values),
            "max": max(rss_values),
        },
    }


def nearest_rank_p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def merge_single_trial_reports(
    reports: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not reports:
        raise ValueError("cannot merge an empty CPU report sequence")
    identity = identity_signature(reports[0])
    if any(identity_signature(report) != identity for report in reports[1:]):
        raise ValueError("cannot merge CPU reports with different identities")

    merged = copy.deepcopy(reports[0])
    trials = []
    for trial_number, report in enumerate(reports, start=1):
        if len(report["trials"]) != 1:
            raise ValueError("interleaved source reports must contain one trial")
        trial = copy.deepcopy(report["trials"][0])
        trial["trial"] = trial_number
        trials.append(trial)
    merged["generated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    merged["run"]["trials"] = len(trials)
    merged["trials"] = trials
    aggregate = {}
    for metric in (
        "step_wall_s",
        "avg_step_ms",
        "sim_s_per_wall_s",
        "process_wall_s",
        "peak_rss_bytes",
    ):
        values = [float(trial[metric]) for trial in trials]
        aggregate[metric] = {
            "median": statistics.median(values),
            "p95_nearest_rank": nearest_rank_p95(values),
            "min": min(values),
            "max": max(values),
        }
    merged["aggregate"] = aggregate
    return merged


def pareto_frontier(points: Sequence[dict[str, Any]]) -> list[int]:
    frontier = []
    for candidate in points:
        candidate_time = candidate["avg_step_ms"]["median"]
        candidate_rss = candidate["peak_rss_bytes"]["max"]
        dominated = any(
            other is not candidate
            and other["avg_step_ms"]["median"] <= candidate_time
            and other["peak_rss_bytes"]["max"] <= candidate_rss
            and (
                other["avg_step_ms"]["median"] < candidate_time
                or other["peak_rss_bytes"]["max"] < candidate_rss
            )
            for other in points
        )
        if not dominated:
            frontier.append(candidate["threads"])
    return frontier


def endpoint_environment(
    count: int,
    settings: Sequence[str],
) -> dict[str, str]:
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith(CLEARED_ENVIRONMENT_PREFIXES):
            environment.pop(key, None)
    environment["OMP_NUM_THREADS"] = str(count)
    for setting in settings:
        if "=" not in setting:
            raise ValueError(f"invalid endpoint environment setting: {setting!r}")
        key, value = setting.split("=", 1)
        environment[key] = value
    return environment


def qualify_exact_endpoints(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if args.endpoint_duration == 0.0:
        return None

    binary = Path(args.binary).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()
    duration = f"{args.endpoint_duration:g}"
    results = []
    with tempfile.TemporaryDirectory(prefix="nimbus_cpu_scaling_endpoint_") as temp:
        root = Path(temp)
        for count in args.thread_counts:
            checkpoint_dir = root / f"threads_{count}"
            checkpoint_dir.mkdir()
            command = [
                str(binary),
                "--headless",
                f"--config={config}",
                *args.extra_arg,
                "--compute-backend=cpu",
                "--log-profile=quiet",
                "--no-timing",
                f"--duration={duration}",
                "--write-every=0",
                f"--checkpoint-every={duration}",
                f"--checkpoint-dir={checkpoint_dir}",
            ]
            completed = subprocess.run(
                command,
                env=endpoint_environment(count, args.env),
                text=True,
                capture_output=True,
                timeout=args.timeout or None,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{count}-thread endpoint qualification failed:\n"
                    f"{completed.stderr}"
                )
            checkpoints = sorted(checkpoint_dir.glob("*.nimbus"))
            if len(checkpoints) != 1:
                raise RuntimeError(
                    f"{count}-thread endpoint qualification produced "
                    f"{len(checkpoints)} binary checkpoints; expected one"
                )
            checkpoint = checkpoints[0]
            results.append(
                {
                    "threads": count,
                    "checkpoint_bytes": checkpoint.stat().st_size,
                    "checkpoint_sha256": sha256_file(checkpoint),
                }
            )

    hashes = {result["checkpoint_sha256"] for result in results}
    sizes = {result["checkpoint_bytes"] for result in results}
    exact = len(hashes) == 1 and len(sizes) == 1
    return {
        "duration_s": args.endpoint_duration,
        "comparison": "bitwise-identical portable binary checkpoint",
        "exact_across_thread_counts": exact,
        "reference_sha256": results[0]["checkpoint_sha256"],
        "reference_bytes": results[0]["checkpoint_bytes"],
        "results": results,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--binary", default="bin/nimbus_sim_cpu_benchmark")
    result.add_argument("--config", default="tests/fixtures/configs/benchmark/benchmark.yaml")
    result.add_argument("--trials", type=positive_int, default=7)
    result.add_argument(
        "--thread-counts",
        type=thread_counts,
        default=None,
        help=(
            "Comma-separated frontier. By default, powers of two, logical "
            "cores, and Nimbus's topology-aware runtime default are measured."
        ),
    )
    result.add_argument("--timeout", type=nonnegative_float, default=0.0)
    result.add_argument(
        "--endpoint-duration",
        type=nonnegative_float,
        default=0.0,
        help=(
            "Run one isolated checkpoint discriminator per thread count at "
            "this simulated duration; zero disables endpoint qualification."
        ),
    )
    result.add_argument(
        "--output",
        default="reports/benchmarks/cpu-scaling-current.json",
    )
    result.add_argument("--compiler", default="unknown")
    result.add_argument("--cxxflags", default="unknown")
    result.add_argument("--ldflags", default="unknown")
    result.add_argument("--build-profile", default="unspecified")
    result.add_argument("--extra-arg", action="append", default=[])
    result.add_argument("--env", action="append", default=[])
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.thread_counts is None:
        counts = default_thread_counts()
        runtime_default = runtime_default_thread_count(
            args.binary, args.config, args.extra_arg
        )
        if runtime_default is not None:
            counts.append(runtime_default)
        args.thread_counts = sorted(set(counts))
    interleaved: dict[int, list[dict[str, Any]]] = {
        count: [] for count in args.thread_counts
    }
    for trial_index in range(args.trials):
        order = (
            args.thread_counts
            if trial_index % 2 == 0
            else list(reversed(args.thread_counts))
        )
        for count in order:
            report = run_thread_count(args, count, trials=1)
            interleaved[count].append(report)
            print(
                f"[cpu-scaling] trial={trial_index + 1}/{args.trials} "
                f"threads={count}: "
                f"{report['trials'][0]['avg_step_ms']:.4f} ms/step",
                file=sys.stderr,
            )
    reports = [
        merge_single_trial_reports(interleaved[count])
        for count in args.thread_counts
    ]
    identity = identity_signature(reports[0])
    for report in reports[1:]:
        if identity_signature(report) != identity:
            raise SystemExit(
                "refusing to aggregate thread runs with different "
                "host/build/workload identity"
            )

    points = [summarize_point(report) for report in reports]
    one_thread = next(
        (point for point in points if point["threads"] == 1),
        None,
    )
    if one_thread is None:
        raise SystemExit("thread frontier must include the one-thread reference")
    baseline_ms = one_thread["avg_step_ms"]["median"]
    previous_ms = baseline_ms
    for point in points:
        current_ms = point["avg_step_ms"]["median"]
        point["speedup_over_one_thread"] = baseline_ms / current_ms
        point["parallel_efficiency"] = (
            point["speedup_over_one_thread"] / point["threads"]
        )
        point["incremental_speedup_over_previous"] = previous_ms / current_ms
        previous_ms = current_ms

    fastest = min(points, key=lambda point: point["avg_step_ms"]["median"])
    endpoint_equivalence = qualify_exact_endpoints(args)
    output_report = {
        "schema": REPORT_SCHEMA,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "measurement_contract": (
            "identical binary/config/equations/metric scope; inherited "
            "Nimbus/OpenMP tuning cleared; only OMP_NUM_THREADS changes; "
            "trial rounds alternate ascending/descending thread order"
        ),
        "identity": identity,
        "run": {
            "thread_counts": args.thread_counts,
            "trials_per_thread_count": args.trials,
            "trial_order": "alternating ascending/descending frontier",
            "endpoint_duration_s": args.endpoint_duration,
        },
        "frontier": {
            "recommended_threads": fastest["threads"],
            "fastest_median_step_ms": fastest["avg_step_ms"]["median"],
            "pareto_thread_counts": pareto_frontier(points),
        },
        "points": points,
        "endpoint_equivalence": endpoint_equivalence,
        "source_reports": reports,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(output_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[cpu-scaling] recommended_threads={fastest['threads']}, "
        f"fastest={fastest['avg_step_ms']['median']:.4f} ms/step, "
        f"wrote {output}",
        file=sys.stderr,
    )
    if (
        endpoint_equivalence is not None
        and not endpoint_equivalence["exact_across_thread_counts"]
    ):
        print(
            "[cpu-scaling] FAIL: checkpoints differ across thread counts",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
