#!/usr/bin/env python3
"""Bound an ordinary mixed ZFP/NPY missing-value history publication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
FALLBACK = {"k_index", "lifted_index", "showalter_index", "total_totals"}
LEDGERS = ("budget.csv", "momentum_budget.csv", "energy_budget.csv",
           "radiation_budget.csv", "thermodynamic_source_budget.csv",
           "moisture_budget.csv")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--zfp-config", type=Path, required=True)
    parser.add_argument("--npy-config", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    zfp_config = args.zfp_config.resolve()
    npy_config = args.npy_config.resolve()
    raw = args.raw_root.resolve()
    report = args.report_output.resolve()
    if not all(path.is_file() for path in (binary, zfp_config, npy_config)) or \
            raw.exists() or report.exists():
        parser.error("binary/configs must exist and output targets must be new")
    catalog_bytes = subprocess.check_output(
        [str(binary), "--scheme-catalog=json"], cwd=ROOT
    )
    catalog = json.loads(catalog_bytes)
    readiness = next(item for item in catalog["readiness"]["items"] if
                     item["id"] == "output.zfp_bounded_history")
    if readiness["state"] != "engineering_incomplete":
        raise RuntimeError("broad ZFP readiness unexpectedly changed")
    identity = {"simulatorSha256": sha256(binary),
                "zfpConfigSha256": sha256(zfp_config),
                "npyConfigSha256": sha256(npy_config),
                "runnerSha256": sha256(Path(__file__))}
    raw.mkdir(parents=True)
    runs = raw / "runs"
    logs = raw / "logs"
    runs.mkdir()
    logs.mkdir()
    rows: list[dict[str, object]] = []
    started = time.monotonic()
    status = "pass_storage_contract"
    for format_name, config in (("npy", npy_config), ("zfp", zfp_config)):
        outdir = runs / format_name
        command = [str(binary), "--headless", "--config", str(config),
                   "--duration=2", "--write-every=2", "--checkpoint-every=0",
                   "--threads=1", "--log-profile=quiet", f"--outdir={outdir}"]
        try:
            completed = subprocess.run(command, cwd=ROOT, capture_output=True,
                                       text=True, timeout=30, check=False)
            code = completed.returncode
            message = completed.stdout + "\n" + completed.stderr
        except subprocess.TimeoutExpired as error:
            code = 124
            stdout = error.stdout or b""
            stderr = error.stderr or b""
            message = stdout.decode(errors="replace") + "\n" + stderr.decode(errors="replace")
        log = logs / f"{format_name}.log"
        log.write_text(message, encoding="utf-8")
        rows.append({"format": format_name, "command": command,
                     "exitCode": code, "logSha256": sha256(log)})
        if code != 0:
            status = "runtime_failure"
            break
    if status == "pass_storage_contract":
        for ledger in LEDGERS:
            if sha256(runs / "npy" / ledger) != sha256(runs / "zfp" / ledger):
                status = "ledger_mismatch"
                break
    if status == "pass_storage_contract":
        for index in (0, 1):
            step = f"step_{index:06d}"
            npy_step = runs / "npy" / step
            zfp_step = runs / "zfp" / step
            manifest = json.loads((zfp_step / "manifest.json").read_text(
                encoding="utf-8"))
            control = json.loads((npy_step / "manifest.json").read_text(
                encoding="utf-8"))
            fields = manifest["published_fields"]
            if fields != control["published_fields"] or not FALLBACK.issubset(fields) or \
                    manifest["data_product"].get("lossy_compression_contract") != \
                    "nimbus.zfp.per_field.v1":
                status = "inventory_or_product_contract_mismatch"
                break
            metadata = {entry["field_id"]: entry for entry in manifest["fields"]}
            for field in fields:
                entry = metadata[field]
                if field in FALLBACK:
                    fallback = zfp_step / f"{field}.npy"
                    reference = npy_step / f"{field}.npy"
                    if entry.get("file_pattern") != f"{field}.npy" or \
                            entry.get("storage_format") != "npy_2d" or \
                            entry.get("missing_value") != "NaN" or \
                            "compression" in entry or not fallback.is_file() or \
                            (zfp_step / f"{field}.zfp3d").exists() or \
                            fallback.read_bytes() != reference.read_bytes():
                        status = "lossless_fallback_contract_mismatch"
                        break
                else:
                    compression = entry.get("compression")
                    if entry.get("file_pattern") != f"{field}.zfp3d" or \
                            entry.get("storage_format") is not None or \
                            not isinstance(compression, dict) or \
                            compression.get("codec") != "zfp" or \
                            compression.get("container_version") != 3 or \
                            not (zfp_step / f"{field}.zfp3d").is_file() or \
                            (zfp_step / f"{field}.npy").exists():
                        status = "compressed_field_contract_mismatch"
                        break
            rows.append({"step": step, "publishedFields": len(fields),
                         "fallbackFields": sorted(FALLBACK),
                         "manifestSha256": sha256(zfp_step / "manifest.json"),
                         "status": status})
            if status != "pass_storage_contract":
                break
    files = [path for path in raw.rglob("*") if path.is_file()]
    bytes_used = sum(path.stat().st_size for path in files)
    if len(files) >= 4000 or bytes_used >= 128 * 1024 * 1024 or \
            time.monotonic() - started >= 90:
        status = "resource_cap"
    if identity != {"simulatorSha256": sha256(binary),
                    "zfpConfigSha256": sha256(zfp_config),
                    "npyConfigSha256": sha256(npy_config),
                    "runnerSha256": sha256(Path(__file__))}:
        status = "source_identity_drift"
    summary = {
        "schema": "nimbus.zfp-mixed-v1-fallback.v1",
        "status": status,
        "claimBoundary": "Ordinary mapped Bell dynamics_full per-field mixed storage and NPY fallback identity; compressed decoded-field Studio science and broad Ready are separate.",
        **identity,
        "catalogSha256": hashlib.sha256(catalog_bytes).hexdigest(),
        "cases": rows,
        "wallSeconds": time.monotonic() - started,
        "rawFiles": len(files),
        "rawBytes": bytes_used,
        "rawRoot": str(raw),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"{status}: {len(rows)} rows, {len(files)} files/{bytes_used} bytes")
    return 0 if status == "pass_storage_contract" else 1


if __name__ == "__main__":
    raise SystemExit(main())
