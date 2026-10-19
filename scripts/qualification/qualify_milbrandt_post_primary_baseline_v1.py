#!/usr/bin/env python3
"""Capture a bounded ordinary Milbrandt CPU payload baseline for hook replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/configs/qualification/cgrid_milbrandt_cpu_ready.yaml"
SOURCES = (
    "src/microphysics/schemes/milbrandt/milbrandt.cpp",
    "src/microphysics/schemes/milbrandt/milbrandt.hpp",
    "src/microphysics/schemes/milbrandt/warm_rain.hpp",
    "include/microphysics/microphysics_base.hpp",
    "src/core/orchestration/physics/microphysics_step.cpp",
    "tests/fixtures/configs/qualification/cgrid_milbrandt_cpu_ready.yaml",
)
LEDGERS = (
    "budget.csv",
    "momentum_budget.csv",
    "energy_budget.csv",
    "radiation_budget.csv",
    "moisture_budget.csv",
    "thermodynamic_source_budget.csv",
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    raw = args.raw_root.resolve()
    if raw.exists() or args.report_output.exists():
        parser.error("use new raw/report paths to preserve prior evidence")
    raw.mkdir(parents=True)
    source_before = {name: digest(ROOT / name) for name in SOURCES}
    binary_before = digest(binary)
    output = raw / "ordinary_cpu"
    command = [
        str(binary), "--headless", "--config", str(FIXTURE),
        "--duration=4", "--write-every=4", "--checkpoint-every=2",
        "--threads=1", f"--outdir={output}",
        "--log-profile=normal", "--no-timing",
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True,
            timeout=30, check=False)
    except subprocess.TimeoutExpired as error:
        (raw / "run.log").write_text(str(error), encoding="utf-8")
        raise RuntimeError("Milbrandt control exceeded 30-second cap") from error
    log = raw / "run.log"
    content = completed.stdout + completed.stderr
    log.write_text(content, encoding="utf-8")
    if completed.returncode != 0 or \
            "backend=cpu, requested=cpu" not in content or \
            "Initialized microphysics scheme: milbrandt" not in content:
        raise RuntimeError("ordinary Milbrandt CPU control failed; preserve raw")
    if not output.is_dir():
        raise RuntimeError("ordinary Milbrandt CPU output is missing")
    for name in LEDGERS:
        if not (output / name).is_file():
            raise RuntimeError(f"Milbrandt ledger missing: {name}")
    checkpoints = sorted((output / "checkpoints").glob("*.nimbus"))
    if len(checkpoints) != 2:
        raise RuntimeError("Milbrandt did not publish two bounded checkpoints")
    files = sorted(path for path in output.rglob("*") if path.is_file())
    if len(files) > 4096:
        raise RuntimeError("Milbrandt control exceeded 4096 output files")
    total_bytes = sum(path.stat().st_size for path in files)
    if total_bytes > 128 * 1024 * 1024 or log.stat().st_size > 20 * 1024 * 1024:
        raise RuntimeError("Milbrandt control exceeded byte caps")
    npy = [path for path in files if path.suffix == ".npy"]
    if not npy:
        raise RuntimeError("Milbrandt control published no NPY fields")
    source_after = {name: digest(ROOT / name) for name in SOURCES}
    if source_before != source_after or binary_before != digest(binary):
        raise RuntimeError("Milbrandt source or simulator drifted")
    records = [{
        "path": path.relative_to(output).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    } for path in files]
    aggregate = hashlib.sha256()
    for record in records:
        aggregate.update((record["path"] + "\t" + str(record["bytes"]) +
                          "\t" + record["sha256"] + "\n").encode())
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    report = {
        "schema": "nimbus.milbrandt-post-primary-baseline.v1",
        "status": "pass",
        "claimBoundary": "Ordinary qualified double-moment CPU payload baseline, not triple readiness.",
        "simulatorSha256": binary_before,
        "sourceSha256": source_before,
        "runnerSha256": digest(Path(__file__)),
        "fixtureSha256": digest(FIXTURE),
        "logSha256": digest(log),
        "wallSeconds": time.monotonic() - started,
        "peakChildRssRaw": rss,
        "outputRoot": str(output),
        "fileCount": len(files),
        "outputBytes": total_bytes,
        "npyCount": len(npy),
        "checkpointCount": len(checkpoints),
        "ledgerNames": list(LEDGERS),
        "memberManifestSha256": aggregate.hexdigest(),
        "files": records,
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "files": len(files),
                      "bytes": total_bytes, "simulatorSha256": binary_before}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
