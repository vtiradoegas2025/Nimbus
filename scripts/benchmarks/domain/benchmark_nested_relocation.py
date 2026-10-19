#!/usr/bin/env python3
"""Measure warm production nested-grid relocation cost and process peak RSS."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import statistics
import subprocess
import sys
import time
from typing import Any, Sequence


BENCHMARK_LINE = re.compile(
    r"\[NESTED RELOCATION BENCHMARK\]\s*"
    r"profile=(?P<profile>[a-z_]+),\s*"
    r"child_nx=(?P<child_nx>\d+),\s*"
    r"child_ny=(?P<child_ny>\d+),\s*"
    r"nz=(?P<nz>\d+),\s*"
    r"warmup_moves=(?P<warmup_moves>\d+),\s*"
    r"measured_moves=(?P<measured_moves>\d+),\s*"
    r"total_ms=(?P<total_ms>[-+0-9.eE]+),\s*"
    r"avg_ms=(?P<avg_ms>[-+0-9.eE]+),\s*"
    r"heap_allocations=(?P<heap_allocations>\d+),\s*"
    r"heap_requested_bytes=(?P<heap_requested_bytes>\d+),\s*"
    r"physics_field_allocations=(?P<physics_field_allocations>\d+),\s*"
    r"physics_field_allocated_bytes=(?P<physics_field_allocated_bytes>\d+),\s*"
    r"rollback_journal_bytes=(?P<rollback_journal_bytes>\d+),\s*"
    r"full_primary_owner_values=(?P<full_primary_owner_values>\d+)"
)
FILTER = "warm production nested relocation benchmark"
CLEARED_ENVIRONMENT_PREFIXES = ("NIMBUS_", "OMP_", "KMP_", "GOMP_")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_rss_bytes(raw_maxrss: int) -> int:
    return raw_maxrss if sys.platform == "darwin" else raw_maxrss * 1024


def parse_benchmark_output(stdout: str, stderr: str = "") -> dict[str, Any] | None:
    matches = list(BENCHMARK_LINE.finditer(stdout + "\n" + stderr))
    if not matches:
        return None
    raw = matches[-1].groupdict()
    integer_keys = (
        "child_nx", "child_ny", "nz", "warmup_moves", "measured_moves",
        "heap_allocations", "heap_requested_bytes",
        "physics_field_allocations", "physics_field_allocated_bytes",
        "rollback_journal_bytes", "full_primary_owner_values",
    )
    result: dict[str, Any] = {key: int(raw[key]) for key in integer_keys}
    result["profile"] = raw["profile"]
    result["total_ms"] = float(raw["total_ms"])
    result["avg_ms"] = float(raw["avg_ms"])
    return result


def validate_measurement(
    measurement: dict[str, Any], expected: dict[str, Any]
) -> str | None:
    for key, expected_value in expected.items():
        if measurement.get(key) != expected_value:
            return f"{key} was {measurement.get(key)!r}, expected {expected_value}"
    if measurement["measured_moves"] <= 0 or measurement["total_ms"] <= 0.0:
        return "timing and measured move count must be positive"
    if not math.isfinite(measurement["total_ms"]) or not math.isfinite(
        measurement["avg_ms"]
    ):
        return "timing must be finite"
    expected_average = measurement["total_ms"] / measurement["measured_moves"]
    if not math.isclose(
        measurement["avg_ms"], expected_average, rel_tol=2.0e-5, abs_tol=1.0e-9
    ):
        return "avg_ms is inconsistent with total_ms / measured_moves"
    for key in (
        "heap_allocations", "heap_requested_bytes",
        "physics_field_allocations", "physics_field_allocated_bytes",
    ):
        if measurement[key] != 0:
            return f"warm relocation allocated: {key}={measurement[key]}"
    if measurement["rollback_journal_bytes"] <= 0:
        return "rollback journal byte count must be positive"
    if measurement["full_primary_owner_values"] <= 0:
        return "full primary owner count must be positive"
    return None


def output_tail(stdout: str, stderr: str, line_count: int = 80) -> str:
    lines = [f"stdout: {line}" for line in stdout.splitlines()]
    lines.extend(f"stderr: {line}" for line in stderr.splitlines())
    return "\n".join(lines[-line_count:])


def run_worker() -> int:
    request = json.load(sys.stdin)
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith(CLEARED_ENVIRONMENT_PREFIXES):
            environment.pop(key, None)
    environment.update(request["environment"])
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            request["command"], env=environment, text=True,
            capture_output=True, timeout=request.get("timeout_s") or None,
            check=False,
        )
        timed_out = False
        stdout = completed.stdout
        stderr = completed.stderr
        returncode: int | None = completed.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = None
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
    elapsed = time.perf_counter() - started
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    json.dump(
        {
            "returncode": returncode,
            "timed_out": timed_out,
            "process_wall_s": elapsed,
            "peak_rss_bytes": peak_rss_bytes(usage.ru_maxrss),
            "measurement": parse_benchmark_output(stdout, stderr),
            "log_tail": output_tail(stdout, stderr),
        },
        sys.stdout,
    )
    return 0


def nearest_rank_p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def summarize(trials: Sequence[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for key in ("total_ms", "avg_ms", "process_wall_s", "peak_rss_bytes"):
        values = [float(trial[key]) for trial in trials]
        aggregate[key] = {
            "median": statistics.median(values),
            "p95_nearest_rank": nearest_rank_p95(values),
            "min": min(values),
            "max": max(values),
        }
    return aggregate


def positive_integer(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--child-nx", type=positive_integer, default=128)
    parser.add_argument("--child-ny", type=positive_integer, default=128)
    parser.add_argument("--nz", type=positive_integer, default=64)
    parser.add_argument("--warmup-moves", type=positive_integer, default=4)
    parser.add_argument("--measured-moves", type=positive_integer, default=8)
    parser.add_argument("--trials", type=positive_integer, default=5)
    parser.add_argument("--threads", type=positive_integer, default=1)
    parser.add_argument(
        "--profile",
        choices=(
            "baseline",
            "flat_active",
            "flat_milbrandt",
            "flat_active_milbrandt",
        ),
        default="baseline",
    )
    parser.add_argument("--timeout", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compiler", default="")
    parser.add_argument("--cxxflags", default="")
    args = parser.parse_args()
    if args.worker:
        return run_worker()
    if args.binary is None:
        parser.error("--binary is required")
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"benchmark binary does not exist: {binary}")
    if args.child_nx < 4 or args.child_ny < 4:
        parser.error("child dimensions must be at least four")
    if args.child_nx % 2 or args.child_ny % 2:
        parser.error("child dimensions must be divisible by the refinement ratio 2")

    expected = {
        "profile": args.profile,
        "child_nx": args.child_nx,
        "child_ny": args.child_ny,
        "nz": args.nz,
        "warmup_moves": args.warmup_moves,
        "measured_moves": args.measured_moves,
    }
    environment = {
        "OMP_NUM_THREADS": str(args.threads),
        "NIMBUS_NESTED_BENCH_PROFILE": args.profile,
        "NIMBUS_NESTED_BENCH_CHILD_NX": str(args.child_nx),
        "NIMBUS_NESTED_BENCH_CHILD_NY": str(args.child_ny),
        "NIMBUS_NESTED_BENCH_NZ": str(args.nz),
        "NIMBUS_NESTED_BENCH_WARMUP_MOVES": str(args.warmup_moves),
        "NIMBUS_NESTED_BENCH_MEASURED_MOVES": str(args.measured_moves),
    }
    request = {
        "command": [str(binary), FILTER],
        "environment": environment,
        "timeout_s": args.timeout,
    }
    trials: list[dict[str, Any]] = []
    for trial_index in range(args.trials):
        worker = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            input=json.dumps(request), text=True, capture_output=True, check=False,
        )
        if worker.returncode != 0:
            raise RuntimeError(f"measurement worker failed: {worker.stderr}")
        result = json.loads(worker.stdout)
        if result["timed_out"]:
            raise RuntimeError(f"trial {trial_index + 1} timed out\n{result['log_tail']}")
        if result["returncode"] != 0:
            raise RuntimeError(
                f"trial {trial_index + 1} exited {result['returncode']}\n"
                f"{result['log_tail']}"
            )
        measurement = result["measurement"]
        if measurement is None:
            raise RuntimeError(
                f"trial {trial_index + 1} emitted no benchmark record\n"
                f"{result['log_tail']}"
            )
        error = validate_measurement(measurement, expected)
        if error:
            raise RuntimeError(f"trial {trial_index + 1}: {error}")
        trials.append(
            {
                "trial": trial_index + 1,
                **measurement,
                "process_wall_s": result["process_wall_s"],
                "peak_rss_bytes": result["peak_rss_bytes"],
            }
        )

    report = {
        "schema": "nimbus.nested-relocation-benchmark.v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": (
            "release-built CPU relocation transaction after scratch warmup; "
            "peak RSS covers complete benchmark-process initialization"
        ),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "artifact": {
            "path": str(binary),
            "sha256": sha256_file(binary),
            "compiler": args.compiler,
            "cxxflags": args.cxxflags,
        },
        "run": {**expected, "threads": args.threads, "trials": args.trials},
        "trials": trials,
        "aggregate": summarize(trials),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
