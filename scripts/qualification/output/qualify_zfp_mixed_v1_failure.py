#!/usr/bin/env python3
"""Bound mixed-v1 ZFP post-manifest and resource failure publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
FALLBACK = {"k_index", "lifted_index", "showalter_index", "total_totals"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise RuntimeError(f"expected one source token: {before!r}")
    return source.replace(before, after, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    config = args.config.resolve()
    raw = args.raw_root.resolve()
    report = args.report_output.resolve()
    if not binary.is_file() or not config.is_file() or raw.exists() or report.exists():
        parser.error("binary/config must exist and output targets must be new")
    identity = {"simulatorSha256": sha256(binary),
                "sourceConfigSha256": sha256(config),
                "runnerSha256": sha256(Path(__file__))}
    raw.mkdir(parents=True)
    inputs = raw / "inputs"
    runs = raw / "runs"
    logs = raw / "logs"
    inputs.mkdir()
    runs.mkdir()
    logs.mkdir()
    source = config.read_text(encoding="utf-8")
    free_space = inputs / "minimum_free_space_exhausted.yaml"
    free_space.write_text(replace_once(
        source, "  minimum_free_space_mib: 4096\n",
        "  minimum_free_space_mib: 2147483647\n"), encoding="utf-8")
    file_ceiling = inputs / "retained_file_ceiling.yaml"
    file_ceiling.write_text(replace_once(
        source, "  maximum_retained_history_files: 4096\n",
        "  maximum_retained_history_files: 1\n"), encoding="utf-8")
    cases = (("post_manifest_index", config, True),
             ("minimum_free_space", free_space, False),
             ("retained_file_ceiling", file_ceiling, False))
    rows: list[dict[str, object]] = []
    status = "pass"
    started = time.monotonic()
    for name, candidate, inject_index in cases:
        outdir = runs / name
        command = [str(binary), "--headless", "--config", str(candidate),
                   "--duration=1", "--write-every=1", "--checkpoint-every=0",
                   "--threads=1", "--log-profile=quiet", f"--outdir={outdir}"]
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        if inject_index:
            environment["NIMBUS_TEST_HISTORY_INDEX_FAIL_AFTER_MANIFEST"] = "1"
        try:
            completed = subprocess.run(command, cwd=ROOT, env=environment,
                                       capture_output=True, text=True,
                                       timeout=30, check=False)
            code = completed.returncode
            message = completed.stdout + "\n" + completed.stderr
        except subprocess.TimeoutExpired as error:
            code = 124
            stdout = error.stdout or b""
            stderr = error.stderr or b""
            message = stdout.decode(errors="replace") + "\n" + stderr.decode(errors="replace")
        log = logs / f"{name}.log"
        log.write_text(message, encoding="utf-8")
        contract = outdir / "nimbus_output.json"
        failed_state = contract.is_file() and '"publication_state": "failed"' in \
            contract.read_text(encoding="utf-8")
        t0 = outdir / "step_000000"
        t0_manifest = t0 / "manifest.json"
        no_index = not (outdir / "history_index.json").exists()
        no_later = not (outdir / "step_000001/manifest.json").exists()
        fallback_congruent = True
        same_generation = False
        if inject_index and t0_manifest.is_file():
            manifest = json.loads(t0_manifest.read_text(encoding="utf-8"))
            if contract.is_file():
                run_contract = json.loads(contract.read_text(encoding="utf-8"))
                same_generation = run_contract.get("publication_generation") == \
                    manifest.get("publication_generation")
            entries = {item["field_id"]: item for item in manifest["fields"]}
            for field in FALLBACK:
                item = entries[field]
                fallback_congruent = fallback_congruent and \
                    item.get("file_pattern") == f"{field}.npy" and \
                    item.get("storage_format") == "npy_2d" and \
                    "compression" not in item and \
                    (t0 / f"{field}.npy").is_file() and \
                    not (t0 / f"{field}.zfp3d").exists()
        passed = code not in (0, 124) and no_index and no_later and \
            (t0_manifest.is_file() if inject_index else not t0_manifest.exists()) and \
            (failed_state and same_generation if inject_index else True) and \
            fallback_congruent
        rows.append({"name": name, "configSha256": sha256(candidate),
                     "exitCode": code, "logSha256": sha256(log),
                     "sameGenerationFailedState": failed_state,
                     "sameGenerationIdentity": same_generation,
                     "t0ManifestDurable": t0_manifest.is_file(),
                     "noHistoryIndex": no_index, "noLaterManifest": no_later,
                     "fallbackCongruent": fallback_congruent, "pass": passed})
        if not passed:
            status = "failure_contract_red"
            break
    files = [path for path in raw.rglob("*") if path.is_file()]
    bytes_used = sum(path.stat().st_size for path in files)
    if len(files) >= 2000 or bytes_used >= 64 * 1024 * 1024 or \
            time.monotonic() - started >= 90:
        status = "resource_cap"
    if identity != {"simulatorSha256": sha256(binary),
                    "sourceConfigSha256": sha256(config),
                    "runnerSha256": sha256(Path(__file__))}:
        status = "source_identity_drift"
    summary = {
        "schema": "nimbus.zfp-mixed-v1-failure.v1",
        "status": status,
        "claimBoundary": "Ordinary mapped Bell mixed-v1 late index and resource failures; current-binary broad Ready remains separate.",
        **identity,
        "cases": rows,
        "wallSeconds": time.monotonic() - started,
        "rawFiles": len(files),
        "rawBytes": bytes_used,
        "rawRoot": str(raw),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"{status}: {len(rows)}/3 mixed-v1 failure/resource cases")
    return 0 if status == "pass" and len(rows) == 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
