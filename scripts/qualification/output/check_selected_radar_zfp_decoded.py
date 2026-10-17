#!/usr/bin/env python3
"""Compare native-decoded selected radar ZFP fields to same-time NPY exactly."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/tests/science"))
from npy_support import read_float32_npy


FIELDS = ("reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv")
STEPS = ("step_000000", "step_000001")
LEDGERS = ("budget.csv", "energy_budget.csv", "moisture_budget.csv",
           "momentum_budget.csv", "radiation_budget.csv",
           "thermodynamic_source_budget.csv")
UNITS = {"reflectivity_dbz": "dBZ", "radial_velocity": "m/s",
         "zdr": "dB", "kdp": "deg/km", "rhohv": "1"}
REGISTRY_BOUNDS = {
    "reflectivity_dbz": (-30.0, 120.0),
    "radial_velocity": (-325.0, 325.0),
    "zdr": (-5.0, 10.0),
    "kdp": (0.0, 4.0),
    "rhohv": (0.8, 1.0),
}
RAIN_OPERATOR_BOUNDS = {
    "zdr": (0.0, 6.0),
    "kdp": (0.0, 4.0),
    "rhohv": (0.8, 1.0),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def catalog_budgets(binary: Path) -> tuple[dict[str, float], str]:
    result = subprocess.run([str(binary), "--scheme-catalog=json"], cwd=ROOT,
                            text=True, capture_output=True, timeout=30, check=True)
    catalog = json.loads(result.stdout)
    policy = catalog["runtimeCapabilities"]["interactionPolicy"]
    budgets = policy["qualifiedZfpFieldBudgets"]
    units = policy["qualifiedZfpFieldUnits"]
    if any(field not in budgets or not math.isfinite(float(budgets[field])) or
           float(budgets[field]) <= 0.0 or units.get(field) != UNITS[field]
           for field in FIELDS):
        raise RuntimeError("exact catalog omitted a selected radar budget or unit")
    return {field: float(budgets[field]) for field in FIELDS}, hashlib.sha256(result.stdout.encode()).hexdigest()


def decoded_values(path: Path, elements: int) -> array.array:
    raw = path.read_bytes()
    if len(raw) != 4 * elements:
        raise RuntimeError(f"native decoded payload has wrong byte count: {path}")
    values = array.array("f")
    values.frombytes(raw)
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError(f"native decoded payload is nonfinite: {path}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--native-decoder", type=Path, required=True)
    parser.add_argument("--zfp-run", type=Path, required=True)
    parser.add_argument("--npy-run", type=Path, required=True)
    parser.add_argument("--native-decoded", type=Path, required=True)
    parser.add_argument("--native-decode-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-active", action="store_true")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    decoder = args.native_decoder.resolve(strict=True)
    zfp_run = args.zfp_run.resolve(strict=True)
    npy_run = args.npy_run.resolve(strict=True)
    decoded = args.native_decoded.resolve(strict=True)
    decode_log = args.native_decode_log.resolve(strict=True)
    budgets, catalog_sha = catalog_budgets(binary)
    log_text = decode_log.read_text(encoding="utf-8")
    if len(log_text.splitlines()) != len(FIELDS) * len(STEPS):
        raise RuntimeError("native reader omitted a selected radar field or frame")
    results = []
    codec_violations = []
    registry_range_violations = []
    rain_operator_range_violations = []
    erased_signals = []
    for step_index, step in enumerate(STEPS):
        zfp_manifest_path = zfp_run / step / "manifest.json"
        npy_manifest_path = npy_run / step / "manifest.json"
        zfp_manifest = json.loads(zfp_manifest_path.read_text(encoding="utf-8"))
        npy_manifest = json.loads(npy_manifest_path.read_text(encoding="utf-8"))
        if (zfp_manifest["simulation_time_s"] != npy_manifest["simulation_time_s"] or
                set(zfp_manifest["published_fields"]) != set(npy_manifest["published_fields"]) or
                not set(FIELDS).issubset(set(zfp_manifest["published_fields"])) or
                zfp_manifest["grid"]["dimensions"] != {"x": 12, "y": 12, "z": 48}):
            raise RuntimeError(f"selected radar frames are not same-time/same-shape at {step}")
        catalog = {entry["field_id"]: entry for entry in zfp_manifest["fields"]}
        for field in FIELDS:
            metadata = catalog[field]
            compression = metadata.get("compression", {})
            budget = budgets[field]
            if (metadata["units"] != UNITS[field] or compression.get("codec") != "zfp" or
                    compression.get("mode") != "accuracy" or
                    compression.get("mode_parameter") != budget or
                    compression.get("parent_step_indices") != ([] if step_index == 0 else [0]) or
                    compression.get("frame_role") != ("keyframe" if step_index == 0 else "simple_delta")):
                raise RuntimeError(f"selected {field} compression or lineage differs at {step}")
            shape, _, reference = read_float32_npy(npy_run / step / f"{field}.npy")
            if shape != (12, 12, 48) or any(not math.isfinite(value) for value in reference):
                raise RuntimeError(f"selected {field} NPY reference is malformed at {step}")
            actual_path = decoded / step / f"{field}.f32"
            actual = decoded_values(actual_path, len(reference))
            error = max(abs(value - target) for value, target in zip(actual, reference))
            if error > budget:
                codec_violations.append({"field": field, "step": step, "error": error, "budget": budget})
            decoded_min = min(actual)
            decoded_max = max(actual)
            registry_low, registry_high = REGISTRY_BOUNDS[field]
            operator_bounds = RAIN_OPERATOR_BOUNDS.get(field)
            if decoded_min < registry_low or decoded_max > registry_high:
                registry_range_violations.append({
                    "field": field, "step": step, "decodedMin": decoded_min,
                    "decodedMax": decoded_max, "bounds": [registry_low, registry_high],
                })
            if operator_bounds is not None and (
                    decoded_min < operator_bounds[0] or decoded_max > operator_bounds[1]):
                rain_operator_range_violations.append({
                    "field": field, "step": step, "decodedMin": decoded_min,
                    "decodedMax": decoded_max, "bounds": list(operator_bounds),
                })
            positive_reference = sum(value > 0.0 for value in reference)
            positive_decoded = sum(value > 0.0 for value in actual)
            if positive_reference and not positive_decoded:
                erased_signals.append({
                    "field": field, "step": step,
                    "positiveReferenceSamples": positive_reference,
                    "positiveDecodedSamples": positive_decoded,
                })
            results.append({
                "step": step,
                "timeS": zfp_manifest["simulation_time_s"],
                "field": field,
                "units": UNITS[field],
                "budget": budget,
                "maximumAbsoluteError": error,
                "errorToBudget": error / budget,
                "referenceMin": min(reference),
                "referenceMax": max(reference),
                "decodedMin": decoded_min,
                "decodedMax": decoded_max,
                "positiveReferenceSamples": positive_reference,
                "positiveDecodedSamples": positive_decoded,
                "decodedNonzeroSamples": sum(value != 0.0 for value in actual),
                "withinRegistryBounds": registry_low <= decoded_min and decoded_max <= registry_high,
                "withinRainOperatorBounds": None if operator_bounds is None else
                operator_bounds[0] <= decoded_min and decoded_max <= operator_bounds[1],
                "npySha256": sha256(npy_run / step / f"{field}.npy"),
                "zfpSha256": sha256(zfp_run / step / f"{field}.zfp3d"),
                "decodedSha256": sha256(actual_path),
                "lineage": compression["frame_role"],
            })
    if args.require_active:
        final = {row["field"]: row for row in results if row["step"] == STEPS[-1]}
        if (final["reflectivity_dbz"]["referenceMax"] <= -29.9 or
                max(abs(final["radial_velocity"]["referenceMin"]),
                    abs(final["radial_velocity"]["referenceMax"])) <= 0.1 or
                final["zdr"]["referenceMax"] <= 0.0 or
                final["kdp"]["referenceMax"] <= 0.0 or
                final["rhohv"]["referenceMin"] >= 1.0):
            raise RuntimeError("four-second radar fields are vacuous")
    ledgers = {}
    for name in LEDGERS:
        npy = npy_run / name
        zfp = zfp_run / name
        if not npy.is_file() or npy.read_bytes() != zfp.read_bytes():
            raise RuntimeError(f"{name} science ledger differs across NPY/ZFP")
        ledgers[name] = sha256(npy)
    for run in (npy_run, zfp_run):
        validation = json.loads((run / "validation_summary.json").read_text())
        if validation.get("failed") is not False or validation.get("total_violations") != 0:
            raise RuntimeError(f"run has invalid science: {run}")
    passed = not codec_violations and not registry_range_violations and not rain_operator_range_violations
    report = {
        "schema": "nimbus.r4-selected-radar-zfp-decoded-error.v2",
        "status": "pass_decoded_science_contract" if passed else "fail_decoded_science_contract",
        "claimBoundary": "Five selected radar fields, two same-time CPU frames. Exact catalog absolute-error parity, field-registry bounds, and maintained rain-operator bounds are separate checks; no tolerance slack or automatic clipping.",
        "binarySha256": sha256(binary),
        "catalogSha256": catalog_sha,
        "nativeDecoderSha256": sha256(decoder),
        "nativeDecoderSourceSha256": sha256(ROOT / "scripts/qualification/output/native_decode_radar_zfp.cpp"),
        "nativeReaderSourceSha256": sha256(ROOT / "src/core/output/zfp_reader.cpp"),
        "fieldContractSourceSha256": sha256(ROOT / "src/diagnostics/contracts/field_contract.cpp"),
        "rainOperatorSourceSha256": sha256(ROOT / "include/radar/polarimetric/rain_mp.hpp"),
        "radarScopeSourceSha256": sha256(ROOT / "src/radar/README.md"),
        "comparatorSha256": sha256(Path(__file__)),
        "zfpRun": str(zfp_run),
        "npyRun": str(npy_run),
        "nativeDecodeLogSha256": sha256(decode_log),
        "referenceActiveRequired": args.require_active,
        "catalogBudgetPassed": not codec_violations,
        "codecViolations": codec_violations,
        "registryRangeViolations": registry_range_violations,
        "rainOperatorRangeViolations": rain_operator_range_violations,
        "erasedPositiveSignals": erased_signals,
        "fieldComparisons": results,
        "scienceLedgers": ledgers,
    }
    output = args.output.resolve()
    if output.exists():
        parser.error("use a new report path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "comparisons": len(results),
                      "maximumErrorToBudget": max(row["errorToBudget"] for row in results),
                      "registryRangeViolations": len(registry_range_violations),
                      "rainOperatorRangeViolations": len(rain_operator_range_violations),
                      "erasedPositiveSignals": len(erased_signals)}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
