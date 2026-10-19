#!/usr/bin/env python3
"""Cross-size exact-endpoint CPU scaling evidence for terrain shortwave."""

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
import statistics
import sys
import tempfile
from typing import Any, Sequence

from benchmark_active_physics_allocations import run_one
from benchmark_terrain_shortwave_cpu import metric, positive_int, thread_counts


GRID_NX = re.compile(r"(?m)^  nx: [0-9]+$")
GRID_NY = re.compile(r"(?m)^  ny: [0-9]+$")


def horizontal_sizes(value: str) -> tuple[int, ...]:
    sizes = thread_counts(value)
    if any(size < 8 for size in sizes):
        raise argparse.ArgumentTypeError("horizontal sizes must be at least 8")
    return sizes


def scaled_config_text(base: str, size: int) -> str:
    updated, nx_count = GRID_NX.subn(f"  nx: {size}", base, count=1)
    updated, ny_count = GRID_NY.subn(f"  ny: {size}", updated, count=1)
    if nx_count != 1 or ny_count != 1:
        raise ValueError("base config must contain exactly one grid nx/ny pair")
    return updated


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def robust_metric(values: Sequence[float]) -> dict[str, float]:
    if len(values) < 3:
        raise ValueError("robust metric requires at least three samples")
    if any(not isinstance(value, (float, int)) for value in values):
        raise ValueError("robust metric samples must be numeric")
    converted = [float(value) for value in values]
    if any(not math.isfinite(value) for value in converted):
        raise ValueError("robust metric samples must be finite")
    center = float(statistics.median(converted))
    mad = float(statistics.median(
        abs(value - center) for value in converted
    ))
    return {
        "median": center,
        "median_absolute_deviation": mad,
        "three_robust_sigma": 3.0 * 1.4826 * mad,
        "min": min(converted),
        "max": max(converted),
    }


def paired_scaling_evidence(
    one_worker_ms: Sequence[float],
    candidate_ms: Sequence[float],
    minimum_speedup_percent: float = 2.0,
) -> dict[str, Any]:
    if len(one_worker_ms) != len(candidate_ms):
        raise ValueError("paired scaling samples must have equal length")
    if any(value <= 0.0 for value in (*one_worker_ms, *candidate_ms)):
        raise ValueError("paired scaling times must be positive")
    if minimum_speedup_percent < 0.0:
        raise ValueError("minimum speedup percent must be nonnegative")
    improvement = [
        float(one) - float(candidate)
        for one, candidate in zip(one_worker_ms, candidate_ms)
    ]
    ratios = [
        float(one) / float(candidate)
        for one, candidate in zip(one_worker_ms, candidate_ms)
    ]
    improvement_metric = robust_metric(improvement)
    one_metric = robust_metric(one_worker_ms)
    material_improvement_ms = (
        one_metric["median"] * minimum_speedup_percent / 100.0
    )
    required_magnitude = max(
        material_improvement_ms,
        improvement_metric["three_robust_sigma"],
    )
    median_improvement = improvement_metric["median"]
    if median_improvement > required_magnitude:
        classification = "accelerates"
    elif -median_improvement > required_magnitude:
        classification = "regresses"
    else:
        classification = "unresolved"
    return {
        "classification": classification,
        "minimum_speedup_percent": minimum_speedup_percent,
        "minimum_material_improvement_ms": material_improvement_ms,
        "required_improvement_magnitude_ms": required_magnitude,
        "improvement_ms": improvement_metric,
        "speedup_ratio": robust_metric(ratios),
        "one_worker_ms_samples": [float(value) for value in one_worker_ms],
        "candidate_ms_samples": [float(value) for value in candidate_ms],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--binary", required=True)
    result.add_argument(
        "--config",
        default="tests/fixtures/configs/benchmark/terrain_shortwave_cpu.yaml",
    )
    result.add_argument("--duration", type=positive_int, default=10)
    result.add_argument("--trials", type=positive_int, default=7)
    result.add_argument("--thread-counts", type=thread_counts, default=(1, 4))
    result.add_argument(
        "--horizontal-sizes",
        type=horizontal_sizes,
        default=(16, 32, 64),
    )
    result.add_argument(
        "--output",
        default=(
            "reports/benchmarks/"
            "terrain-shortwave-cpu-scaling-current.json"
        ),
    )
    return result


def ensure_zero_radiation_allocations(
    run: dict[str, Any], size: int, threads: int, trial: int
) -> None:
    allocations = run["allocation_components"].get("radiation")
    if allocations is None or any(value != 0 for value in allocations.values()):
        raise RuntimeError(
            "post-warm-up radiation allocation: "
            f"size={size}, threads={threads}, trial={trial}, "
            f"allocation={allocations}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    binary = Path(args.binary).resolve()
    base_config = Path(args.config).resolve()
    if not binary.is_file() or not base_config.is_file():
        raise SystemExit("benchmark binary and base config must exist")
    base_text = base_config.read_text()

    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(
        prefix="nimbus-terrain-shortwave-scaling-"
    ) as temporary:
        temporary_root = Path(temporary)
        for size in args.horizontal_sizes:
            config_text = scaled_config_text(base_text, size)
            config = temporary_root / f"terrain-shortwave-{size}.yaml"
            config.write_text(config_text)
            runs_by_threads: dict[str, list[dict[str, Any]]] = {}
            for trial in range(args.trials):
                order = args.thread_counts
                if trial % 2 == 1:
                    order = tuple(reversed(order))
                for threads in order:
                    run = run_one(
                        f"terrain-sw-{size}-{threads}t-{trial + 1}",
                        binary,
                        config,
                        threads,
                        args.duration,
                        require_dynamics_step_coverage=False,
                    )
                    ensure_zero_radiation_allocations(
                        run, size, threads, trial + 1
                    )
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
                raise RuntimeError(
                    f"size {size} endpoint differs by trial or worker count"
                )

            workers: dict[str, Any] = {}
            cadence_samples: dict[str, list[float]] = {}
            for threads, runs in runs_by_threads.items():
                cadence_counts = {run["measured_steps"] for run in runs}
                if len(cadence_counts) != 1:
                    raise RuntimeError("measured cadence count changed")
                cadences = next(iter(cadence_counts))
                cadence_ms = [
                    1000.0 * run["radiation_s"] / cadences for run in runs
                ]
                cadence_samples[threads] = cadence_ms
                workers[threads] = {
                    "trials": len(runs),
                    "measured_cadences_per_trial": cadences,
                    "radiation_ms_per_cadence": metric(cadence_ms),
                    "radiation_ns_per_physical_cell_per_cadence": metric(
                        [
                            value * 1.0e6 / (size * size * 20)
                            for value in cadence_ms
                        ]
                    ),
                    "peak_rss_bytes": metric(
                        [run["peak_rss_bytes"] for run in runs]
                    ),
                    "radiation_allocations_after_warmup": {
                        "heap_allocations": 0,
                        "heap_requested_bytes": 0,
                        "physics_field_allocations": 0,
                        "physics_field_allocated_bytes": 0,
                    },
                }
            one_thread = workers.get("1")
            if one_thread is not None:
                one_median = one_thread["radiation_ms_per_cadence"]["median"]
                for threads, summary in workers.items():
                    summary["speedup_vs_one_worker"] = (
                        one_median
                        / summary["radiation_ms_per_cadence"]["median"]
                    )
                    if threads != "1":
                        summary["paired_vs_one_worker"] = (
                            paired_scaling_evidence(
                                cadence_samples["1"],
                                cadence_samples[threads],
                            )
                        )

            results[str(size)] = {
                "nx": size,
                "ny": size,
                "nz": 20,
                "active_lambertian_rays_per_bounce": size * size * 32,
                "config_sha256": sha256_bytes(config_text.encode()),
                "endpoint": {
                    "checkpoint_sha256": next(iter(endpoint_hashes)),
                    "checkpoint_bytes": next(iter(endpoint_sizes)),
                },
                "workers": workers,
            }

    binary_bytes = binary.read_bytes()
    report = {
        "schema": "nimbus.terrain_shortwave_cpu_scaling.v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "qualification": {
            "passed": True,
            "scope": (
                "cross-size CPU runtime/allocation/repeatability; angular "
                "accuracy and other hardware remain separate"
            ),
        },
        "inputs": {
            "binary": str(binary),
            "binary_sha256": sha256_bytes(binary_bytes),
            "base_config": str(base_config),
            "base_config_sha256": sha256_bytes(base_text.encode()),
            "duration_s": args.duration,
            "trials": args.trials,
            "thread_counts": list(args.thread_counts),
            "horizontal_sizes": list(args.horizontal_sizes),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "sizes": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
