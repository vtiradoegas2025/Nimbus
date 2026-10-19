#!/usr/bin/env python3
"""Bound the retained Milbrandt triple selector without opening admission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tests/fixtures/configs/qualification/cgrid_milbrandt_cpu_ready.yaml"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(binary: Path, name: str, config: Path, outdir: Path,
             validate_only: bool, log: Path) -> dict[str, object]:
    args = [str(binary), "--config", str(config)]
    if validate_only:
        args.append("--validate-only")
    else:
        args += ["--headless", "--duration=1", "--write-every=0",
                 "--checkpoint-every=0", f"--outdir={outdir}", "--no-timing"]
    start = time.monotonic()
    try:
        completed = subprocess.run(
            args, cwd=ROOT, capture_output=True, text=True, timeout=20, check=False)
    except subprocess.TimeoutExpired as error:
        log.write_text(str(error), encoding="utf-8")
        raise RuntimeError(f"{name} exceeded the 20-second cap") from error
    content = completed.stdout + completed.stderr
    log.write_text(content, encoding="utf-8")
    expected_failure = name != "double_control"
    if expected_failure:
        if completed.returncode == 0 or "triple_moment" not in content or \
                "engineering incomplete" not in content:
            raise RuntimeError(f"{name} did not fail closed with keyed diagnostic")
        if outdir.exists() and any(outdir.rglob("*")):
            raise RuntimeError(f"{name} published output after rejection")
    elif completed.returncode != 0 or "Initialized microphysics scheme: milbrandt" not in content:
        raise RuntimeError("double-moment ordinary control did not execute")
    return {
        "name": name,
        "returnCode": completed.returncode,
        "expectedFailure": expected_failure,
        "wallSeconds": time.monotonic() - start,
        "configSha256": digest(config),
        "logSha256": digest(log),
        "publicationFiles": sum(1 for path in outdir.rglob("*") if path.is_file())
        if outdir.exists() else 0,
    }


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
    source = BASE.read_text(encoding="utf-8")
    needle = "  milbrandt:\n    hail_processes: true"
    if source.count(needle) != 1 or source.count("  scheme: milbrandt") != 1:
        raise RuntimeError("base Milbrandt fixture changed")
    inputs = raw / "inputs"
    logs = raw / "logs"
    inputs.mkdir()
    logs.mkdir()
    strict_true = source.replace(needle, "  milbrandt:\n    triple_moment: true\n    hail_processes: true")
    strict_false = source.replace(needle, "  milbrandt:\n    triple_moment: false\n    hail_processes: true")
    wrong_scheme = strict_true.replace("  scheme: milbrandt", "  scheme: kessler", 1)
    legacy_true = (
        "grid:\n  nx: 12\n  ny: 12\n  nz: 16\n  dx: 1000\n  dy: 1000\n"
        "  dz: 250\nmicrophysics:\n  scheme: milbrandt\n  milbrandt:\n"
        "    triple_moment: true\n"
    )
    legacy_wrong_scheme = legacy_true.replace("  scheme: milbrandt", "  scheme: kessler", 1)
    configurations = {
        "strict_true": strict_true,
        "strict_false": strict_false,
        "strict_wrong_scheme_true": wrong_scheme,
        "legacy_true": legacy_true,
        "legacy_wrong_scheme_true": legacy_wrong_scheme,
        "double_control": source,
    }
    cases = []
    for name, text in configurations.items():
        config = inputs / f"{name}.yaml"
        config.write_text(text, encoding="utf-8")
        cases.append(run_case(
            binary, name, config, raw / "runs" / name,
            name.startswith("strict"), logs / f"{name}.log"))
    report = {
        "schema": "nimbus.milbrandt-triple-guard.v1",
        "status": "pass",
        "claimBoundary": "Retained triple selector remains fail-closed; isolated CPU state is not integrated or publicly admitted.",
        "simulatorSha256": digest(binary),
        "baseFixtureSha256": digest(BASE),
        "runnerSha256": digest(Path(__file__)),
        "rawRoot": str(raw),
        "cases": cases,
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "cases": len(cases), "simulatorSha256": digest(binary)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
