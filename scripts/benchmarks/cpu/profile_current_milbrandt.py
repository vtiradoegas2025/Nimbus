#!/usr/bin/env python3
"""Bounded current-binary Milbrandt stage attribution, without an optimization claim."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

import benchmark_active_physics_allocations as benchmark_parser
from benchmark_active_physics_allocations import PERF_SUMMARY, parse_dynamics_perf


SUBPHASES = (
    "pbl_reprojection",
    "input_validation",
    "transaction_snapshot",
    "scheme",
    "chaos",
    "tendency_contract",
    "validation",
    "private_state",
    "precipitation",
    "projection",
    "apply",
    "finalize",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trials", type=positive_int, default=7)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--duration", type=positive_int, default=40)
    parser.add_argument("--timeout", type=positive_int, default=20)
    args = parser.parse_args()
    if args.trials < 7:
        parser.error("at least seven steady-state trials are required")

    binary = args.binary.resolve(strict=True)
    config = args.config.resolve(strict=True)
    report_path = args.output.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    binary_hash = sha256(binary)
    config_hash = sha256(config)
    script_hash = sha256(Path(__file__))
    runs = []

    for trial in range(1, args.trials + 1):
        with tempfile.TemporaryDirectory(prefix="nimbus-c6-milbrandt-") as temp:
            temp_path = Path(temp)
            checkpoint_dir = temp_path / "checkpoints"
            command = [
                str(binary),
                "--headless",
                f"--config={config}",
                "--compute-backend=cpu",
                "--log-profile=normal",
                "--timing",
                "--timing-every=0",
                f"--duration={args.duration}",
                f"--checkpoint-every={args.duration}",
                f"--checkpoint-dir={checkpoint_dir}",
                f"--outdir={temp_path / 'output'}",
            ]
            environment = os.environ.copy()
            for name in list(environment):
                if name.startswith(("NIMBUS_", "OMP_", "KMP_", "GOMP_")):
                    del environment[name]
            environment["OMP_NUM_THREADS"] = str(args.threads)
            try:
                completed = subprocess.run(
                    command,
                    env=environment,
                    text=True,
                    capture_output=True,
                    timeout=args.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise SystemExit(f"trial {trial} exceeded {args.timeout}s") from error
            log = completed.stdout + "\n" + completed.stderr
            log_path = report_path.parent / f"milbrandt_profile_trial_{trial:02d}.log"
            log_path.write_text(log, encoding="utf-8")
            if completed.returncode != 0:
                raise SystemExit(f"trial {trial} failed: exit {completed.returncode}; {log_path}")
            if sha256(binary) != binary_hash or sha256(config) != config_hash:
                raise SystemExit("binary or configuration changed during profiling")
            perf_matches = list(PERF_SUMMARY.finditer(log))
            if len(perf_matches) != 1:
                raise SystemExit(f"trial {trial} has no unique performance summary")
            perf = perf_matches[0].groupdict()
            dynamics = parse_dynamics_perf(log)
            step_wall = float(perf["wall"])
            if int(perf["warmup"]) < 20 or int(perf["steps"]) < 20:
                raise SystemExit(f"trial {trial} is not a steady-state measurement")
            if dynamics["profiled"] / step_wall < 0.95:
                raise SystemExit(f"trial {trial} has less than 95% named wall coverage")
            microphysics = dynamics["microphysics"]
            subphases = {name: dynamics[f"microphysics_{name}"] for name in SUBPHASES}
            subphase_residual = microphysics - sum(subphases.values())
            if abs(subphase_residual - dynamics["microphysics_residual"]) > max(
                1.0e-6, 5.0e-4 * microphysics
            ):
                raise SystemExit(f"trial {trial} has inconsistent microphysics ownership")
            checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.nimbus"))
            if len(checkpoints) != 1:
                raise SystemExit(f"trial {trial} has {len(checkpoints)} final checkpoints")
            runs.append({
                "trial": trial,
                "log": str(log_path),
                "log_sha256": sha256(log_path),
                "warmup_steps": int(perf["warmup"]),
                "measured_steps": int(perf["steps"]),
                "step_wall_s": step_wall,
                "avg_step_ms": float(perf["step_ms"]),
                "named_wall_fraction": dynamics["profiled"] / step_wall,
                "dynamics_s": dynamics["total"],
                "microphysics_s": microphysics,
                "microphysics_subphases_s": subphases,
                "microphysics_residual_s": subphase_residual,
                "checkpoint_sha256": sha256(checkpoints[0]),
                "checkpoint_bytes": checkpoints[0].stat().st_size,
            })
            print(f"[c6-milbrandt] trial {trial}/{args.trials}: {runs[-1]['avg_step_ms']:.5f} ms/step", file=sys.stderr)

    endpoint_hashes = {run["checkpoint_sha256"] for run in runs}
    endpoint_sizes = {run["checkpoint_bytes"] for run in runs}
    if len(endpoint_hashes) != 1 or len(endpoint_sizes) != 1:
        raise SystemExit("repeated endpoints are not byte-identical")
    if sha256(binary) != binary_hash:
        raise SystemExit("binary changed during profiling")
    median_step = statistics.median(run["avg_step_ms"] for run in runs)
    report = {
        "schema": "nimbus.c6-current-milbrandt-stage-profile.v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {
            "binary": str(binary),
            "binary_sha256": binary_hash,
            "config": str(config),
            "config_sha256": config_hash,
            "script_sha256": script_hash,
            "parser_sha256": sha256(Path(benchmark_parser.__file__)),
        },
        "contract": {
            "trials": args.trials,
            "threads": args.threads,
            "duration_s": args.duration,
            "per_trial_timeout_s": args.timeout,
            "minimum_named_wall_fraction": 0.95,
            "checkpoint_exact_across_trials": True,
            "scientific_ledgers_enabled": False,
            "optimization_comparator": None,
        },
        "endpoint": {
            "checkpoint_sha256": next(iter(endpoint_hashes)),
            "checkpoint_bytes": next(iter(endpoint_sizes)),
        },
        "runs": runs,
        "summary": {
            "median_step_ms": median_step,
            "step_mad_ms": statistics.median(abs(run["avg_step_ms"] - median_step) for run in runs),
            "minimum_named_wall_fraction": min(run["named_wall_fraction"] for run in runs),
            "median_microphysics_fraction": statistics.median(
                run["microphysics_s"] / run["step_wall_s"] for run in runs
            ),
            "median_subphase_fraction_of_microphysics": {
                name: statistics.median(
                    run["microphysics_subphases_s"][name] / run["microphysics_s"]
                    for run in runs
                ) for name in SUBPHASES
            },
        },
        "disposition": "attribution only; no science or throughput change accepted",
    }
    if any(not math.isfinite(value) for value in report["summary"].values() if isinstance(value, float)):
        raise SystemExit("non-finite summary")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[c6-milbrandt] wrote {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
