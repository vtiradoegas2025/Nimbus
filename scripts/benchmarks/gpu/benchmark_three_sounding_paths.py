#!/usr/bin/env python3
"""Paired strict-guard timing with exact checkpoints and scientific ledgers."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

CASES = ("01_classic_supercell", "02_pulse_decay", "03_moist_supercell")
LEDGERS = ("budget.csv", "energy_budget.csv", "moisture_budget.csv", "momentum_budget.csv",
           "radiation_budget.csv", "thermodynamic_source_budget.csv")
ROOT = Path(__file__).resolve().parents[3]


def gpu_snapshot(args) -> dict | None:
    if args.max_other_gpu_memory_mib is None:
        return None
    command = ["nvidia-smi", f"--id={args.gpu_index}",
               "--query-compute-apps=pid,process_name,used_gpu_memory",
               "--format=csv,noheader,nounits"]
    response = subprocess.run(command, check=True, text=True, capture_output=True, timeout=10)
    processes = []
    for row in csv.reader(response.stdout.splitlines()):
        if not row:
            continue
        processes.append(dict(pid=int(row[0]), name=row[1].strip(), memory_mib=float(row[2])))
    occupied = sum(row["memory_mib"] for row in processes)
    if occupied > args.max_other_gpu_memory_mib:
        raise RuntimeError(f"other GPU processes occupy {occupied} MiB; limit is {args.max_other_gpu_memory_mib}")
    info = subprocess.run(["nvidia-smi", f"--id={args.gpu_index}",
        "--query-gpu=timestamp,name,driver_version,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True, timeout=10)
    return dict(processes=processes, gpu_csv=info.stdout.strip())


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def metrics(line: str) -> dict[str, float | str]:
    result = {}
    for key, value in re.findall(r"([a-z_]+)=([^,\s]+)", line):
        try:
            result[key] = float(value)
        except ValueError:
            result[key] = value
    return result


def run(binary: Path, config: Path, directory: Path, args) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    output = directory / "output"
    command = [str(binary), "--headless", "--config=" + str(config), "--compute-backend=" + args.backend,
               "--threads=" + str(args.threads), "--outdir=" + str(output)]
    environment = {k: v for k, v in os.environ.items()
                   if not k.startswith(("NIMBUS_", "OMP_", "KMP_", "GOMP_"))}
    gpu_before = gpu_snapshot(args)
    start = time.monotonic()
    with (directory / "run.log").open("w") as log:
        process = subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                                 stderr=subprocess.STDOUT, timeout=args.timeout)
    elapsed = time.monotonic() - start
    if process.returncode:
        raise RuntimeError(f"{directory}: simulator exit {process.returncode}")
    gpu_after = gpu_snapshot(args)
    text = (directory / "run.log").read_text()
    expects_cuda = bool(re.search(r"(?m)^    cuda_assist: true\s*$", config.read_text()))
    if ("CUDA assist active:" in text) != expects_cuda:
        raise RuntimeError("CUDA runtime activation disagrees with the measured configuration")
    if not re.search(rf"\[COMPUTE\] backend={args.backend},[^\n]*fallback=off", text):
        raise RuntimeError("requested backend did not execute without fallback")
    if not re.search(r"\[RUN SETTINGS\][^\n]*guard_mode=strict", text):
        raise RuntimeError("strict guards must remain enabled")
    summaries = re.findall(r"\[PERF SUMMARY\] (.*)", text)
    if len(summaries) != 1:
        raise RuntimeError("one complete performance summary is required")
    perf = metrics(summaries[0])
    if perf["measured_steps"] < 10 or perf["warmup_steps"] < 1:
        raise RuntimeError("insufficient measured work or missing warmup")
    validation = json.loads((output / "validation_summary.json").read_text())
    if validation["failed"] or validation["total_violations"] or not validation["report_count"]:
        raise RuntimeError("scientific guards failed or did not run")
    checkpoints = sorted((output / "checkpoints").glob("*.nimbus"))
    if not checkpoints:
        raise RuntimeError("a final checkpoint is required")
    frames = sorted(output.glob("step_*"))
    if not frames:
        raise RuntimeError("saved frames are required")
    fields = {p.name: sha256(p) for p in frames[-1].glob("*.npy")}
    if not fields:
        raise RuntimeError("the final frame is empty")
    dynamics = re.findall(r"\[DYNAMICS PERF\] (.*)", text)
    return dict(command=command, process_seconds=elapsed, perf=perf,
                gpu_before=gpu_before, gpu_after=gpu_after,
                dynamics=metrics(dynamics[-1]) if dynamics else {},
                checkpoint_sha256=sha256(checkpoints[-1]),
                ledgers={name: sha256(output / name) for name in LEDGERS},
                fields=fields, validation=validation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--candidate-config-dir", type=Path,
                        help="Compare CUDA assist with identical physics; only cuda_assist may differ")
    parser.add_argument("--comparison-only", action="store_true",
                        help="Complete all measurements even when a speedup gate fails; never relax equivalence")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "vulkan"), default="vulkan")
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--minimum-speedup-percent", type=float, default=5)
    parser.add_argument("--max-other-gpu-memory-mib", type=float,
                        help="Require and record NVIDIA process memory headroom before and after every run")
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()
    if args.trials < 3 or args.threads < 1 or args.timeout <= 0 or args.minimum_speedup_percent < 0:
        parser.error("use at least three paired trials and positive workers/timeouts")
    if args.gpu_index < 0 or (args.max_other_gpu_memory_mib is not None and args.max_other_gpu_memory_mib < 0):
        parser.error("GPU index and process memory limit must be nonnegative")
    binaries = {"baseline": args.baseline.resolve(strict=True), "candidate": args.candidate.resolve(strict=True)}
    hashes = {name: sha256(path) for name, path in binaries.items()}
    if hashes["baseline"] == hashes["candidate"] and args.candidate_config_dir is None:
        parser.error("baseline and candidate must be distinct builds")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    configs = {case: (args.config_dir / (case + ".yaml")).resolve(strict=True) for case in CASES}
    config_hashes = {case: sha256(path) for case, path in configs.items()}
    candidate_configs = {case: ((args.candidate_config_dir or args.config_dir) / (case + ".yaml")).resolve(strict=True) for case in CASES}
    candidate_hashes = {case: sha256(path) for case, path in candidate_configs.items()}
    if args.candidate_config_dir:
        def physics_text(path):
            return re.sub(r"(?m)^    cuda_assist: (?:true|false)\s*\n", "", path.read_text())
        for case in CASES:
            if physics_text(configs[case]) != physics_text(candidate_configs[case]):
                parser.error(f"{case}: candidate configuration changes more than CUDA assist")
    sounding_dir = ROOT / "configs/custom/rtx5090_lifecycle_50m/soundings"
    sounding_hashes = {case: sha256(sounding_dir / (case + ".txt")) for case in CASES}
    evidence = dict(backend=args.backend, workers=args.threads, binary_sha256=hashes,
                    config_sha256=config_hashes, candidate_config_sha256=candidate_hashes,
                    sounding_sha256=sounding_hashes, cases={}, passed=False, measurements_complete=False)
    try:
        for case in CASES:
            rows = []
            evidence["cases"][case] = dict(pairs=rows)
            for trial in range(args.trials):
                pair = {}
                for name in (("baseline", "candidate") if trial % 2 == 0 else ("candidate", "baseline")):
                    config = candidate_configs[case] if name == "candidate" else configs[case]
                    expected_config = candidate_hashes[case] if name == "candidate" else config_hashes[case]
                    if sha256(binaries[name]) != hashes[name] or sha256(config) != expected_config:
                        raise RuntimeError("benchmark binary or configuration changed")
                    if sha256(sounding_dir / (case + ".txt")) != sounding_hashes[case]:
                        raise RuntimeError("benchmark sounding changed")
                    pair[name] = run(binaries[name], config, args.output / case / f"pair-{trial + 1}" / name, args)
                for key in ("checkpoint_sha256", "fields", "ledgers"):
                    if pair["baseline"][key] != pair["candidate"][key]:
                        raise RuntimeError(f"{case}: {key} changed")
                for key in ("measured_steps", "warmup_steps", "measured_simulated_s", "scope"):
                    if pair["baseline"]["perf"][key] != pair["candidate"]["perf"][key]:
                        raise RuntimeError(f"{case}: timing denominator {key} changed")
                rows.append(pair)
                print(case, "pair", trial + 1, "ms/step", pair["baseline"]["perf"]["avg_step_ms"],
                      pair["candidate"]["perf"]["avg_step_ms"], "exact checkpoint/fields/ledgers", flush=True)
                (args.output / "results.json").write_text(json.dumps(evidence, indent=2) + "\n")
            before = [r["baseline"]["perf"]["avg_step_ms"] for r in rows]
            after = [r["candidate"]["perf"]["avg_step_ms"] for r in rows]
            improvements = [b - a for b, a in zip(before, after)]
            delta = statistics.median(improvements)
            noise = 3 * 1.4826 * statistics.median(abs(x - delta) for x in improvements)
            percent = 100 * delta / statistics.median(before)
            passed = delta > noise and percent >= args.minimum_speedup_percent
            evidence["cases"][case].update(baseline_median_ms=statistics.median(before),
                candidate_median_ms=statistics.median(after), paired_speedup_percent=percent,
                paired_three_mad_ms=noise, passes_speedup_and_noise=passed)
            if not passed and not args.comparison_only:
                raise RuntimeError(f"{case}: speedup did not exceed its threshold and measured noise")
        evidence["measurements_complete"] = True
        evidence["passed"] = all(case["passes_speedup_and_noise"] for case in evidence["cases"].values())
    except Exception as error:
        evidence["error"] = str(error)
        raise
    finally:
        (args.output / "results.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
