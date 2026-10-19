#!/usr/bin/env python3
"""Measure detached reversible ZFP against archived active radar float fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import struct


ROOT = Path(__file__).resolve().parents[3]
FIELDS = ("reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv")
STEPS = ("step_000000", "step_000001")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototype", type=Path, required=True)
    parser.add_argument("--npy-run", type=Path, required=True)
    parser.add_argument("--historical-accuracy-zfp-run", type=Path, required=True)
    parser.add_argument("--input-provenance-report", type=Path, required=True)
    parser.add_argument("--zfp-library", type=Path, required=True)
    parser.add_argument("--zfp-header", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prototype = args.prototype.resolve(strict=True)
    npy_run = args.npy_run.resolve(strict=True)
    zfp_run = args.historical_accuracy_zfp_run.resolve(strict=True)
    input_provenance_path = args.input_provenance_report.resolve(strict=True)
    input_provenance = json.loads(input_provenance_path.read_text())
    if (input_provenance.get("npyRun") != str(npy_run) or
            input_provenance.get("zfpRun") != str(zfp_run) or
            input_provenance.get("status") != "fail_decoded_science_contract"):
        raise RuntimeError("prototype input runs are not bound to the decoded red receipt")
    library = args.zfp_library.resolve(strict=True)
    header = args.zfp_header.resolve(strict=True)
    if args.output.exists():
        parser.error("use a new report path")
    rows = []
    for step in STEPS:
        npy_manifest = json.loads((npy_run / step / "manifest.json").read_text())
        zfp_manifest = json.loads((zfp_run / step / "manifest.json").read_text())
        if (npy_manifest["simulation_time_s"] != zfp_manifest["simulation_time_s"]
                or not set(FIELDS).issubset(set(npy_manifest["published_fields"]))
                or not set(FIELDS).issubset(set(zfp_manifest["published_fields"]))):
            raise RuntimeError(f"prototype inputs are not same-time selected fields: {step}")
        for field in FIELDS:
            npy = npy_run / step / f"{field}.npy"
            historical = zfp_run / step / f"{field}.zfp3d"
            command = [str(prototype), field, step, str(npy)]
            if step == STEPS[-1]:
                command.append(str(npy_run / STEPS[0] / f"{field}.npy"))
            result = subprocess.run(
                command, cwd=ROOT,
                text=True, capture_output=True, timeout=30, check=True)
            row = json.loads(result.stdout)
            if (row["field"] != field or row["step"] != step or
                    row["zfpVersion"] != "1.0.1" or row["brickCount"] != 2 or
                    row["rawBytes"] != 12 * 12 * 48 * 4 or
                    not row["byteIdentical"] or not row["withinApiMaximumSize"]):
                raise RuntimeError(f"reversible ZFP prototype failed exact comparison: {field}/{step}")
            row["simulationTimeS"] = npy_manifest["simulation_time_s"]
            row["npySha256"] = sha256(npy)
            row["historicalAccuracyContainerSha256"] = sha256(historical)
            row["historicalAccuracyContainerBytes"] = historical.stat().st_size
            row["reversibleContainerToRawRatio"] = (
                row["estimatedNimbusV3ContainerBytes"] / row["rawBytes"])
            row["reversibleToHistoricalAccuracyContainerRatio"] = (
                row["estimatedNimbusV3ContainerBytes"] /
                row["historicalAccuracyContainerBytes"])
            rows.append(row)
    prior_zdr = f32(3.8186869621276855)
    next_zdr = f32(0.011046011932194233)
    delta = f32(next_zdr - prior_zdr)
    reconstructed_zdr = f32(prior_zdr + delta)
    if reconstructed_zdr == next_zdr:
        raise RuntimeError("maintained ZDR-range simple-delta counterexample disappeared")
    full_brick = json.loads(subprocess.run(
        [str(prototype), "--full-brick-bound"], cwd=ROOT, text=True,
        capture_output=True, timeout=30, check=True).stdout)
    if (full_brick["shape"] != [64, 64, 32] or
            full_brick["rawBytes"] != 64 * 64 * 32 * 4 or
            not full_brick["byteIdentical"] or
            full_brick["maximumSizeApiBytes"] > full_brick["parallelWorkerCodecReserveBytes"] or
            full_brick["maximumSizeApiBytes"] > full_brick["readerBrickLimitBytes"]):
        raise RuntimeError("installed reversible full-brick resource bound does not fit owners")
    report = {
        "schema": "nimbus.r4-radar-zfp-reversible-detached-prototype.v4",
        "status": "pass_detached_exact_float_bytes",
        "claimBoundary": "Isolated CPU libzfp 1.0.1 reversible mode over two 64x64x32-style bricks per selected 12x12x48 float32 field; no Nimbus container, manifest, production writer or native/Rust reader was modified. Simple-delta arithmetic mismatch counts describe only this exact 0-to-4-second pair, not a general guarantee; a within-range ZDR counterexample proves bitwise reconstruction can fail despite a lossless residual codec. Historical accuracy files at 4 s are delta frames, so size ratios there are not same-encoding benchmarks.",
        "prototypeBinarySha256": sha256(prototype),
        "prototypeSourceSha256": sha256(ROOT / "scripts/qualification/output/prototype_reversible_radar_zfp.cpp"),
        "driverSourceSha256": sha256(Path(__file__)),
        "zfpLibrarySha256": sha256(library),
        "zfpHeaderSha256": sha256(header),
        "inputGeneratorBinarySha256": input_provenance["binarySha256"],
        "inputProvenanceReportSha256": sha256(input_provenance_path),
        "currentGuardBinarySha256": sha256(ROOT / "bin/nimbus_sim"),
        "npyRun": str(npy_run),
        "historicalAccuracyZfpRun": str(zfp_run),
        "fieldComparisons": rows,
        "allFloatBytesIdentical": True,
        "totalRawBytes": sum(row["rawBytes"] for row in rows),
        "totalReversibleEstimatedContainerBytes": sum(
            row["estimatedNimbusV3ContainerBytes"] for row in rows),
        "totalHistoricalAccuracyContainerBytes": sum(
            row["historicalAccuracyContainerBytes"] for row in rows),
        "maximumPrototypePeakResidentBytes": max(row["peakResidentBytes"] for row in rows),
        "maximumPrototypeOwnedBytes": max(row["estimatedPeakOwnedBytes"] for row in rows),
        "installedFullBrickResourceBound": full_brick,
        "withinRangeZdrSimpleDeltaCounterexample": {
            "previous": prior_zdr,
            "target": next_zdr,
            "float32Residual": delta,
            "float32Reconstructed": reconstructed_zdr,
            "targetBitwiseRecovered": False,
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "comparisons": len(rows),
                      "totalRawBytes": report["totalRawBytes"],
                      "totalReversibleEstimatedContainerBytes": report["totalReversibleEstimatedContainerBytes"],
                      "totalHistoricalAccuracyContainerBytes": report["totalHistoricalAccuracyContainerBytes"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
