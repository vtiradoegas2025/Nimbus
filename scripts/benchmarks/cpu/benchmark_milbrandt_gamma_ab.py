#!/usr/bin/env python3
"""Bounded science-identical A/B gate for the Milbrandt cold gamma experiment."""

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

from benchmark_active_physics_allocations import PERF_SUMMARY, parse_dynamics_perf


LEDGERS = (
    "budget.csv",
    "energy_budget.csv",
    "moisture_budget.csv",
    "momentum_budget.csv",
    "radiation_budget.csv",
    "thermodynamic_source_budget.csv",
)
FIELDS = ("p", "rho", "theta", "u", "v", "w", "qv", "qr")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def run_one(binary: Path, config: Path, case: Path, timeout: int) -> dict:
    case.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = case / "checkpoints"
    output_dir = case / "output"
    command = [
        str(binary),
        "--headless",
        f"--config={config}",
        "--threads=1",
        "--duration=40",
        "--write-every=40",
        "--checkpoint-every=40",
        f"--checkpoint-dir={checkpoint_dir}",
        f"--outdir={output_dir}",
        "--timing",
        "--timing-every=0",
        "--log-profile=normal",
    ]
    environment = os.environ.copy()
    for name in list(environment):
        if name.startswith(("NIMBUS_", "OMP_", "KMP_", "GOMP_")):
            del environment[name]
    environment["OMP_NUM_THREADS"] = "1"
    try:
        completed = subprocess.run(
            command, env=environment, text=True, capture_output=True,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"trial exceeded {timeout}s: {case}") from error
    log_path = case / "runtime.log"
    log_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"trial failed with exit {completed.returncode}: {log_path}")
    log = log_path.read_text(encoding="utf-8")
    perf_matches = list(PERF_SUMMARY.finditer(log))
    if len(perf_matches) != 1:
        raise RuntimeError(f"missing unique performance summary: {log_path}")
    perf = perf_matches[0].groupdict()
    if int(perf["warmup"]) != 20 or int(perf["steps"]) != 60:
        raise RuntimeError(f"unexpected warmup/measurement window: {log_path}")
    dynamics = parse_dynamics_perf(log)
    step_wall = float(perf["wall"])
    named_coverage = dynamics["profiled"] / step_wall
    if not math.isfinite(named_coverage) or named_coverage < 0.95:
        raise RuntimeError(f"named wall coverage below 95%: {log_path}")
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.nimbus"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"expected one final checkpoint: {checkpoint_dir}")
    final_frames = sorted(output_dir.glob("step_*/manifest.json"))
    if len(final_frames) != 2:
        raise RuntimeError(f"expected initial and final field frames: {output_dir}")
    final_frame = final_frames[-1].parent
    field_paths = {name: final_frame / f"{name}.npy" for name in FIELDS}
    ledger_paths = {name: output_dir / name for name in LEDGERS}
    if any(not path.is_file() for path in (*field_paths.values(), *ledger_paths.values())):
        raise RuntimeError(f"missing field or ledger: {output_dir}")
    if any(len(path.read_text(encoding="utf-8").splitlines()) < 2 for path in ledger_paths.values()):
        raise RuntimeError(f"empty science ledger: {output_dir}")
    validation_path = output_dir / "validation_summary.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("failed") is not False or validation.get("total_violations") != 0:
        raise RuntimeError(f"validation failed: {validation_path}")
    return {
        "binary_sha256": sha256(binary),
        "log": str(log_path),
        "log_sha256": sha256(log_path),
        "step_wall_s": step_wall,
        "avg_step_ms": float(perf["step_ms"]),
        "named_wall_fraction": named_coverage,
        "microphysics_scheme_s": dynamics["microphysics_scheme"],
        "microphysics_s": dynamics["microphysics"],
        "checkpoint_sha256": sha256(checkpoints[0]),
        "checkpoint_bytes": checkpoints[0].stat().st_size,
        "field_sha256": {name: sha256(path) for name, path in field_paths.items()},
        "ledger_sha256": {name: sha256(path) for name, path in ledger_paths.items()},
        "validation_sha256": sha256(validation_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trials", type=positive_int, default=7)
    parser.add_argument("--timeout", type=positive_int, default=20)
    args = parser.parse_args()
    if args.trials < 7:
        parser.error("at least seven paired trials are required")
    baseline = args.baseline.resolve(strict=True)
    candidate = args.candidate.resolve(strict=True)
    config = args.config.resolve(strict=True)
    raw_root = args.raw_root.resolve()
    raw_root.mkdir(parents=True, exist_ok=False)
    source = {
        "baseline_binary": str(baseline),
        "baseline_sha256": sha256(baseline),
        "candidate_binary": str(candidate),
        "candidate_sha256": sha256(candidate),
        "config": str(config),
        "config_sha256": sha256(config),
        "benchmark_sha256": sha256(Path(__file__)),
        "timing_parser_sha256": sha256(Path(__file__).with_name("benchmark_active_physics_allocations.py")),
    }
    pairs = []
    report = {
        "schema": "nimbus.c6-milbrandt-gamma-ab.v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "in_progress",
        "source": source,
        "contract": {
            "trials": args.trials,
            "threads": 1,
            "duration_s": 40,
            "warmup_steps": 20,
            "measured_steps": 60,
            "timeout_per_run_s": args.timeout,
            "minimum_named_wall_fraction": 0.95,
            "minimum_time_gain_percent": 2.0,
            "minimum_noise_multiple_mad": 3.0,
            "field_count": len(FIELDS),
            "ledger_count": len(LEDGERS),
        },
        "pairs": pairs,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    for index in range(args.trials):
        order = ("candidate", "baseline") if index % 2 == 0 else ("baseline", "candidate")
        runs = {}
        for variant in order:
            binary = candidate if variant == "candidate" else baseline
            runs[variant] = run_one(binary, config, raw_root / f"pair_{index + 1:02d}_{variant}", args.timeout)
            if sha256(binary) != source[f"{variant}_sha256"] or sha256(config) != source["config_sha256"]:
                raise RuntimeError("A/B input changed during execution")
        first = runs["baseline"]
        second = runs["candidate"]
        exact = (
            first["checkpoint_sha256"] == second["checkpoint_sha256"]
            and first["checkpoint_bytes"] == second["checkpoint_bytes"]
            and first["field_sha256"] == second["field_sha256"]
            and first["ledger_sha256"] == second["ledger_sha256"]
        )
        pair = {
            "pair": index + 1,
            "order": list(order),
            "baseline": first,
            "candidate": second,
            "exact_state_fields_and_ledgers": exact,
            "speedup_percent": 100.0 * (first["avg_step_ms"] - second["avg_step_ms"]) / first["avg_step_ms"],
        }
        pairs.append(pair)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[c6-gamma] pair {index + 1}/{args.trials}: {pair['speedup_percent']:.3f}% exact={exact}", file=sys.stderr)
        if not exact:
            raise RuntimeError(f"science identity failed in pair {index + 1}")
    checkpoint_hashes = {run["checkpoint_sha256"] for pair in pairs for run in (pair["baseline"], pair["candidate"])}
    field_sets = {tuple(sorted(run["field_sha256"].items())) for pair in pairs for run in (pair["baseline"], pair["candidate"])}
    ledger_sets = {tuple(sorted(run["ledger_sha256"].items())) for pair in pairs for run in (pair["baseline"], pair["candidate"])}
    if len(checkpoint_hashes) != 1 or len(field_sets) != 1 or len(ledger_sets) != 1:
        raise RuntimeError("science identity changed across trial pairs")
    speedups = [pair["speedup_percent"] for pair in pairs]
    median = statistics.median(speedups)
    mad = statistics.median(abs(value - median) for value in speedups)
    time_gate = median > 2.0 and median > 3.0 * mad
    report["status"] = "complete"
    report["aggregate"] = {
        "paired_speedups_percent": speedups,
        "median_speedup_percent": median,
        "paired_mad_percent": mad,
        "three_mad_percent": 3.0 * mad,
        "time_gate_passed": time_gate,
        "minimum_named_wall_fraction": min(
            run["named_wall_fraction"] for pair in pairs for run in (pair["baseline"], pair["candidate"])
        ),
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "field_sha256": dict(next(iter(field_sets))),
        "ledger_sha256": dict(next(iter(ledger_sets))),
    }
    report["disposition"] = "time_gain_accepted" if time_gate else "time_gain_rejected"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[c6-gamma] median={median:.4f}%, three_MAD={3.0 * mad:.4f}%, accepted={time_gate}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
