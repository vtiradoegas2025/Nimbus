#!/usr/bin/env python3
"""Compare serial and isolated field-parallel ZFP runtime publication."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
import platform
import re


ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "bin" / ("nimbus_sim.exe" if os.name == "nt" else "nimbus_sim")
FIXTURE = ROOT / "tests/fixtures/configs/qualification/zfp_core_history.yaml"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def artifact_map(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.glob("step_*/*.zfp3d"))
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(config: Path, output: Path) -> tuple[float, str]:
    started = time.perf_counter()
    result = subprocess.run(
        [
            str(BINARY), "--headless", "--config", str(config),
            "--duration=6", "--write-every=2", "--threads=1",
            f"--outdir={output}", "--log-profile=normal",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    elapsed = time.perf_counter() - started
    require(
        result.returncode == 0,
        f"benchmark run failed\n{result.stdout}\n{result.stderr}",
    )
    return elapsed, result.stdout + result.stderr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/benchmarks/zfp-field-parallel-current.json",
    )
    args = parser.parse_args()
    require(args.repeats > 0, "repeats must be positive")
    source = FIXTURE.read_text(encoding="utf-8")
    require("  zfp_field_execution: serial" in source, "fixture lacks serial policy")
    benchmark_source = (source
        .replace("  nx: 16", "  nx: 128", 1)
        .replace("  ny: 16", "  ny: 128", 1)
        .replace("  nz: 12", "  nz: 32", 1)
        .replace("  ztop: 6000", "  ztop: 16000", 1))
    parallel = benchmark_source.replace(
        "  zfp_field_execution: serial",
        "  zfp_field_execution: field_parallel",
    ).replace("  zfp_field_workers: 0", "  zfp_field_workers: 4")

    serial_times: list[float] = []
    parallel_times: list[float] = []
    with tempfile.TemporaryDirectory(prefix="nimbus_zfp_field_benchmark_") as raw:
        temporary = Path(raw)
        serial_config = temporary / "serial.yaml"
        parallel_config = temporary / "parallel.yaml"
        serial_config.write_text(benchmark_source, encoding="utf-8")
        parallel_config.write_text(parallel, encoding="utf-8")
        serial_reference: dict[str, bytes] | None = None
        codec_versions: set[str] = set()
        runtime_cpu_names: set[str] = set()
        for repeat in range(args.repeats):
            serial_output = temporary / f"serial-{repeat}"
            parallel_output = temporary / f"parallel-{repeat}"
            serial_elapsed, serial_log = run_case(serial_config, serial_output)
            parallel_elapsed, parallel_log = run_case(parallel_config, parallel_output)
            serial_times.append(serial_elapsed)
            parallel_times.append(parallel_elapsed)
            for log in (serial_log, parallel_log):
                match = re.search(
                    r"ZFP codec provenance: version=(.*?), libzfp_execution_policy=",
                    log,
                )
                require(match is not None, "runtime omitted linked libzfp version")
                codec_versions.add(match.group(1).strip())
                cpu_match = re.search(r"\[HARDWARE\] CPU: ([^\n]+)", log)
                require(cpu_match is not None, "runtime omitted CPU identity")
                runtime_cpu_names.add(cpu_match.group(1).strip())
            current_serial = artifact_map(serial_output)
            current_parallel = artifact_map(parallel_output)
            require(current_serial, "serial benchmark emitted no ZFP artifacts")
            require(current_parallel == current_serial, "parallel artifacts differ from serial")
            if serial_reference is None:
                serial_reference = current_serial
            else:
                require(current_serial == serial_reference, "serial repeat is not deterministic")

        serial_median = statistics.median(serial_times)
        parallel_median = statistics.median(parallel_times)
        require(len(codec_versions) == 1, "serial/parallel runs reported different libzfp versions")
        require(len(runtime_cpu_names) == 1, "serial/parallel runs reported different CPUs")
        source_paths = [
            ROOT / "src/core/output/output_writer.cpp",
            ROOT / "include/core/output/output_writer.hpp",
            ROOT / "src/core/output/output_config.cpp",
            ROOT / "include/core/output/output_config.hpp",
            ROOT / "src/core/runtime/headless_runtime.cpp",
        ]
        source_digest = hashlib.sha256()
        for path in source_paths:
            source_digest.update(str(path.relative_to(ROOT)).encode())
            source_digest.update(path.read_bytes())
        try:
            git_head = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                capture_output=True, check=True,
            ).stdout.strip()
            worktree = subprocess.run(
                ["git", "status", "--porcelain=v1"], cwd=ROOT, text=True,
                capture_output=True, check=True,
            ).stdout.encode()
            worktree_status_sha256 = hashlib.sha256(worktree).hexdigest()
        except (OSError, subprocess.CalledProcessError):
            git_head = "unavailable"
            worktree_status_sha256 = "unavailable"
        cpu = next(iter(runtime_cpu_names))
        artifact_digests = {
            relative: hashlib.sha256(payload).hexdigest()
            for relative, payload in sorted((serial_reference or {}).items())
        }
        evidence = {
            "schema": "nimbus.benchmark.zfp-field-parallel",
            "schema_version": 1,
            "generated_at_utc": datetime.datetime.now(
                datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            "fixture": str(FIXTURE.relative_to(ROOT)),
            "fixture_sha256": sha256(FIXTURE),
            "binary": str(BINARY.relative_to(ROOT)),
            "binary_sha256": sha256(BINARY),
            "source_fingerprint_sha256": source_digest.hexdigest(),
            "git_head": git_head,
            "worktree_status_sha256": worktree_status_sha256,
            "platform": platform.platform(),
            "cpu": cpu,
            "libzfp_version": next(iter(codec_versions)),
            "repeats": args.repeats,
            "serial_execution_policy": "serial",
            "parallel_execution_policy": "field_parallel",
            "serial_workers": 1,
            "parallel_workers": 4,
            "parallel_scratch_budget_mib": 64,
            "serial_wall_s": serial_times,
            "parallel_wall_s": parallel_times,
            "serial_median_wall_s": serial_median,
            "parallel_median_wall_s": parallel_median,
            "median_speedup": serial_median / parallel_median,
            "artifact_count": len(serial_reference or {}),
            "artifact_bytes_exact": True,
            "artifact_sha256": artifact_digests,
            "claim_boundary": (
                "Bounded local end-to-end comparison only; no universal throughput "
                "claim and no change to the default serial qualified policy."
            ),
        }
        required = {
            "binary_sha256", "fixture_sha256", "source_fingerprint_sha256",
            "platform", "cpu", "libzfp_version", "artifact_sha256",
            "generated_at_utc", "claim_boundary",
        }
        require(required.issubset(evidence), "benchmark evidence schema is incomplete")
        require(
            len(artifact_digests) == evidence["artifact_count"] and
            all(re.fullmatch(r"[0-9a-f]{64}", value) for value in artifact_digests.values()),
            "benchmark artifact digest set is invalid",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
