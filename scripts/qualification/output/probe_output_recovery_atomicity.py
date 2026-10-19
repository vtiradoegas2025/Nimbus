#!/usr/bin/env python3
"""Bounded current-binary A-grid output, midpoint resume, and corruption probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/tests/science"))
from npy_support import read_float32_npy


FIELDS = ("p", "rho", "theta", "u", "v", "w", "qv", "qr")
UNITS = {"p": "Pa", "rho": "kg/m^3", "theta": "K", "u": "m/s", "v": "m/s", "w": "m/s",
         "qv": "kg/kg", "qr": "kg/kg"}
LEDGERS = ("budget.csv", "energy_budget.csv", "moisture_budget.csv",
           "momentum_budget.csv", "radiation_budget.csv", "thermodynamic_source_budget.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(binary: Path, config: Path, case: Path, resume: Path | None = None) -> dict:
    case.mkdir(parents=True, exist_ok=False)
    output = case / "output"
    checkpoints = case / "checkpoints"
    command = [
        str(binary), "--headless", f"--config={config}", "--duration=2",
        "--write-every=1", "--checkpoint-every=1", "--threads=1",
        "--log-profile=normal", f"--outdir={output}", f"--checkpoint-dir={checkpoints}",
    ]
    if resume is not None:
        command.append(f"--resume={resume}")
    environment = os.environ.copy()
    for name in list(environment):
        if name.startswith(("NIMBUS_", "OMP_", "KMP_", "GOMP_")):
            del environment[name]
    environment["OMP_NUM_THREADS"] = "1"
    completed = subprocess.run(
        command, cwd=ROOT, env=environment, text=True, capture_output=True,
        timeout=30, check=False,
    )
    log = case / "runtime.log"
    log.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    return {"returncode": completed.returncode, "log": str(log), "logSha256": sha256(log),
            "output": str(output), "checkpoints": str(checkpoints)}


def require_clean_run(result: dict, label: str) -> None:
    log = Path(result["log"]).read_text(encoding="utf-8")
    if result["returncode"] != 0 or "[SCIENTIFIC INVALIDITY]" in log or "fallback=on" in log:
        raise RuntimeError(f"{label} did not complete cleanly: {result['log']}")
    validation_path = Path(result["output"]) / "validation_summary.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("failed") is not False or validation.get("total_violations") != 0:
        raise RuntimeError(f"{label} has invalid science: {validation_path}")
    result["validationSha256"] = sha256(validation_path)


def frame(output: Path, index: int) -> dict:
    directory = output / f"step_{index:06d}"
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (set(manifest.get("published_fields", [])) != set(FIELDS)
            or manifest.get("data_product", {}).get("storage_format") != "npy_3d"
            or manifest.get("publication_state") != "complete"
            or manifest.get("grid", {}).get("dimensions") != {"x": 18, "y": 18, "z": 18}
            or manifest.get("grid", {}).get("axis_order") != ["x", "y", "z"]):
        raise RuntimeError(f"A-grid output membership or geometry differs: {manifest_path}")
    metadata = {item["field_id"]: item for item in manifest["fields"]}
    result = {"manifestSha256": sha256(manifest_path), "fieldSha256": {}, "shape": [18, 18, 18]}
    for field in FIELDS:
        if metadata[field]["units"] != UNITS[field] or metadata[field]["grid_location"] != "cell_center":
            raise RuntimeError(f"{field} output units or location differ")
        path = directory / f"{field}.npy"
        shape, _, values = read_float32_npy(path)
        if shape != (18, 18, 18) or not values or any(not math.isfinite(value) for value in values):
            raise RuntimeError(f"{field} NPY shape or values are invalid")
        result["fieldSha256"][field] = sha256(path)
    return result


def checkpoint_record(path: Path) -> dict:
    sidecar = Path(str(path) + ".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    identity = sha256(path)
    if metadata.get("checkpoint_sha256") != identity or metadata.get("checkpoint_bytes") != path.stat().st_size:
        raise RuntimeError(f"checkpoint sidecar identity differs: {path}")
    return {"path": str(path), "sha256": identity, "bytes": path.stat().st_size,
            "sidecarSha256": sha256(sidecar), "completedSteps": metadata["completed_steps"],
            "simulationTimeS": metadata["simulation_time_s"], "runId": metadata["run_id"],
            "parentRunId": metadata.get("parent_run_id")}


def ledger_suffix(continuous: Path, resumed: Path) -> dict:
    results = {}
    for name in LEDGERS:
        source = (continuous / name).read_bytes().splitlines(keepends=True)
        branch = (resumed / name).read_bytes().splitlines(keepends=True)
        exact = len(source) > len(branch) > 1 and source[0] == branch[0] and source[-(len(branch) - 1):] == branch[1:]
        if not exact:
            raise RuntimeError(f"resumed {name} ledger suffix differs")
        results[name] = {"continuousSha256": sha256(continuous / name),
                         "resumedSha256": sha256(resumed / name), "suffixByteExact": True,
                         "resumedRows": len(branch) - 1}
    return results


def corrupt_pair(original: Path, case: Path, mode: str) -> Path:
    case.mkdir(parents=True, exist_ok=False)
    candidate = case / original.name
    shutil.copy2(original, candidate)
    sidecar = Path(str(candidate) + ".json")
    shutil.copy2(Path(str(original) + ".json"), sidecar)
    if mode == "sidecar":
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        metadata["checkpoint_sha256"] = "0" * 64
        sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif mode == "payload":
        payload = bytearray(candidate.read_bytes())
        payload[-1] ^= 1
        candidate.write_bytes(payload)
    else:
        raise ValueError(f"unknown corruption mode: {mode}")
    return candidate


def rejected_publication(result: dict, label: str) -> dict:
    output = Path(result["output"])
    checkpoints = Path(result["checkpoints"])
    expected_error = {
        "sidecar": "checkpoint sidecar integrity mismatch",
        "payload": "simulation checkpoint checksum mismatch",
    }[label]
    log = Path(result["log"]).read_text(encoding="utf-8")
    frames = sorted(path.name for path in output.glob("step_*")) if output.exists() else []
    completed = sorted(path.name for path in checkpoints.glob("checkpoint_*.nimbus")) if checkpoints.exists() else []
    ledger_rows = {
        name: max(0, len((output / name).read_text(encoding="utf-8").splitlines()) - 1)
        if (output / name).is_file() else 0 for name in LEDGERS
    }
    if (result["returncode"] != 1 or expected_error not in log or
            frames or completed or any(ledger_rows.values())):
        raise RuntimeError(f"{label} did not reject specifically and without normal science publication")
    return {**result, "frames": frames, "checkpoints": completed,
            "scienceLedgerRows": ledger_rows, "rejectionReason": expected_error,
            "normalSciencePublished": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "tests/fixtures/configs/qualification/agrid_flat_cpu_core_ready.yaml")
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    config = args.config.resolve(strict=True)
    raw_root = args.raw_root.resolve()
    raw_root.mkdir(parents=True, exist_ok=False)
    binary_sha = sha256(binary)
    config_sha = sha256(config)
    continuous = run(binary, config, raw_root / "continuous")
    require_clean_run(continuous, "continuous")
    continuous_output = Path(continuous["output"])
    midpoint = raw_root / "continuous/checkpoints/checkpoint_step_000000010.nimbus"
    final = raw_root / "continuous/checkpoints/checkpoint_step_000000020.nimbus"
    midpoint_record = checkpoint_record(midpoint)
    final_record = checkpoint_record(final)
    if midpoint_record["completedSteps"] != 10 or final_record["completedSteps"] != 20:
        raise RuntimeError("A-grid checkpoint cadence differs")
    continuous_frame = frame(continuous_output, 2)

    resumed = run(binary, config, raw_root / "resumed", midpoint)
    require_clean_run(resumed, "resumed")
    resumed_output = Path(resumed["output"])
    resumed_frame = frame(resumed_output, 1)
    if continuous_frame["fieldSha256"] != resumed_frame["fieldSha256"]:
        raise RuntimeError("A-grid midpoint continuation changed final field bytes")
    ledgers = ledger_suffix(continuous_output, resumed_output)
    branch_path = resumed_output / "nimbus_branch.json"
    branch = json.loads(branch_path.read_text(encoding="utf-8"))
    if branch.get("parent_run_id") != midpoint_record["runId"]:
        raise RuntimeError("resumed output lineage does not name midpoint run")
    resumed_final = checkpoint_record(raw_root / "resumed/checkpoints/checkpoint_step_000000020.nimbus")
    if resumed_final["parentRunId"] != midpoint_record["runId"]:
        raise RuntimeError("resumed checkpoint lineage does not name midpoint run")

    rejected = {}
    for mode in ("sidecar", "payload"):
        corrupted = corrupt_pair(midpoint, raw_root / f"corrupt_{mode}", mode)
        rejected[mode] = rejected_publication(
            run(binary, config, raw_root / f"rejected_{mode}", corrupted), mode)
    if sha256(binary) != binary_sha or sha256(config) != config_sha:
        raise RuntimeError("A-grid output/recovery probe inputs changed during execution")
    report = {
        "schema": "nimbus.r4-output-recovery-atomicity-probe.v1",
        "status": "pass_bounded_agrid_cpu_only",
        "claimBoundary": "One ordinary qualified flat 18-cubed A-grid CPU profile, exact midpoint output/ledger continuation and two corrupt-checkpoint rejection cases; not all retained output, native resource, or backend combinations.",
        "inputs": {"binary": {"path": str(binary), "sha256": binary_sha},
                   "config": {"path": str(config), "sha256": config_sha},
                   "probeScriptSha256": sha256(Path(__file__))},
        "continuous": {**continuous, "midpoint": midpoint_record, "finalCheckpoint": final_record,
                       "finalFrame": continuous_frame},
        "resumed": {**resumed, "finalCheckpoint": resumed_final, "finalFrame": resumed_frame,
                    "branchSha256": sha256(branch_path), "parentRunId": branch["parent_run_id"]},
        "fieldByteExact": True,
        "ledgerSuffixes": ledgers,
        "corruptRejections": rejected,
    }
    output = args.output.resolve()
    if output.exists():
        parser.error("use a new receipt path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "binarySha256": binary_sha,
                      "exactFields": len(FIELDS), "exactLedgerSuffixes": len(LEDGERS),
                      "corruptRejections": list(rejected)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
