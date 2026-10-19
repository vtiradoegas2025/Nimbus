#!/usr/bin/env python3
"""Repeatable CPU terrain-shortwave runtime/allocation qualification.

This is the runtime half of the terrain-shortwave Pareto program. It requires
byte-identical final checkpoints across repetitions and worker counts before
reporting timing, and it rejects any post-warm-up radiation allocation. Angular
quality/convergence is a separate gate; timing alone cannot select quadrature.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
from typing import Any, Sequence

from benchmark_active_physics_allocations import run_one


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def thread_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "thread counts must be comma-separated positive integers"
        ) from error
    if not counts or any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError(
            "thread counts must be comma-separated positive integers"
        )
    if len(set(counts)) != len(counts):
        raise argparse.ArgumentTypeError("thread counts must be unique")
    return counts


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric(values: Sequence[float | int]) -> dict[str, float]:
    converted = [float(value) for value in values]
    return {
        "median": statistics.median(converted),
        "min": min(converted),
        "max": max(converted),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--binary", required=True)
    result.add_argument(
        "--config",
        default="tests/fixtures/configs/benchmark/terrain_shortwave_cpu.yaml",
    )
    result.add_argument("--duration", type=positive_int, default=10)
    result.add_argument("--trials", type=positive_int, default=5)
    result.add_argument("--thread-counts", type=thread_counts, default=(1, 4))
    result.add_argument(
        "--output",
        default="reports/benchmarks/terrain-shortwave-cpu-current.json",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    binary = Path(args.binary).resolve()
    config = Path(args.config).resolve()
    if not binary.is_file():
        raise SystemExit(f"benchmark binary does not exist: {binary}")
    if not config.is_file():
        raise SystemExit(f"benchmark config does not exist: {config}")

    runs_by_threads: dict[str, list[dict[str, Any]]] = {}
    for trial in range(args.trials):
        order = args.thread_counts
        if trial % 2 == 1:
            order = tuple(reversed(order))
        for threads in order:
            run = run_one(
                f"terrain-shortwave-{threads}t-trial-{trial + 1}",
                binary,
                config,
                threads,
                args.duration,
                require_dynamics_step_coverage=False,
            )
            radiation_allocations = run["allocation_components"].get(
                "radiation"
            )
            if radiation_allocations is None:
                raise SystemExit("benchmark log omitted radiation allocation data")
            if any(
                radiation_allocations[name] != 0
                for name in (
                    "heap_allocations",
                    "heap_requested_bytes",
                    "physics_field_allocations",
                    "physics_field_allocated_bytes",
                )
            ):
                raise SystemExit(
                    "terrain-shortwave radiation allocated after warm-up: "
                    f"threads={threads}, trial={trial + 1}, "
                    f"allocation={radiation_allocations}"
                )
            if run["radiation_s"] <= 0.0 or not math.isfinite(
                run["radiation_s"]
            ):
                raise SystemExit("benchmark produced invalid radiation timing")
            runs_by_threads.setdefault(str(threads), []).append(run)

    endpoint_hashes = {
        run["checkpoint_sha256"]
        for runs in runs_by_threads.values()
        for run in runs
    }
    endpoint_sizes = {
        run["checkpoint_bytes"]
        for runs in runs_by_threads.values()
        for run in runs
    }
    if len(endpoint_hashes) != 1 or len(endpoint_sizes) != 1:
        raise SystemExit(
            "terrain-shortwave endpoint changed across repetitions or workers"
        )

    summaries: dict[str, Any] = {}
    for threads, runs in runs_by_threads.items():
        measured_steps = {run["measured_steps"] for run in runs}
        warmup_steps = {run["warmup_steps"] for run in runs}
        if len(measured_steps) != 1 or len(warmup_steps) != 1:
            raise SystemExit("benchmark cadence metadata changed across trials")
        cadence_count = next(iter(measured_steps))
        summaries[threads] = {
            "trials": len(runs),
            "measured_cadences_per_trial": cadence_count,
            "warmup_cadences": next(iter(warmup_steps)),
            "radiation_total_s": metric(
                [run["radiation_s"] for run in runs]
            ),
            "radiation_ms_per_cadence": metric(
                [1000.0 * run["radiation_s"] / cadence_count for run in runs]
            ),
            "model_step_ms": metric([run["avg_step_ms"] for run in runs]),
            "peak_rss_bytes": metric([run["peak_rss_bytes"] for run in runs]),
            "radiation_allocations_after_warmup": {
                "heap_allocations": 0,
                "heap_requested_bytes": 0,
                "physics_field_allocations": 0,
                "physics_field_allocated_bytes": 0,
            },
        }

    report = {
        "schema": "nimbus.terrain_shortwave_cpu_benchmark.v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "qualification": {
            "passed": True,
            "scope": (
                "runtime/allocation/repeatability only; angular quality and "
                "production-resolution scaling remain separate gates"
            ),
            "requirements": {
                "byte_identical_checkpoint_across_trials_and_workers": True,
                "zero_post_warmup_radiation_allocations": True,
                "finite_positive_radiation_timing": True,
            },
        },
        "inputs": {
            "binary": str(binary),
            "binary_sha256": sha256(binary),
            "config": str(config),
            "config_sha256": sha256(config),
            "duration_s": args.duration,
            "trials": args.trials,
            "thread_counts": list(args.thread_counts),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "logical_cpus": os.cpu_count(),
        },
        "endpoint": {
            "checkpoint_sha256": next(iter(endpoint_hashes)),
            "checkpoint_bytes": next(iter(endpoint_sizes)),
        },
        "workers": summaries,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
