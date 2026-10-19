#!/usr/bin/env python3
"""Bounded strict-v1 file-sounding and radar publication probe on one CPU profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tests/science"))
from npy_support import read_float32_npy


BASE_CONFIG = ROOT / "tests/fixtures/configs/qualification/cgrid_radar_ready.yaml"
AGRID_CONFIG = ROOT / "tests/fixtures/configs/qualification/agrid_flat_cpu_core_ready.yaml"
CM1_PROFILE = ROOT / "tests/fixtures/configs/qualification/soundings/catalog_cm1_profile.txt"
SHARPY_PROFILE = ROOT / "tests/fixtures/configs/qualification/soundings/lin_cloudy_profile.txt"
FILE_LINE = "    file_path: tests/fixtures/configs/qualification/soundings/lin_cloudy_profile.txt\n"
FIELDS = {"p", "rho", "theta", "u", "v", "w", "qv", "qr",
          "reflectivity_dbz", "radial_velocity"}
AGRID_SOUNDING = (
    "  sounding:\n"
    "    type: parametric_cape\n"
    "    cape_jkg: 0\n"
    "    tropopause_m: 8000\n"
    "    surface_theta_k: 300\n"
    "    moisture_qv_kgkg: 0.004\n"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_one(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"fixture anchor is not unique: {old!r}")
    return text.replace(old, new, 1)


def execute(binary: Path, config: Path, directory: Path, runtime: bool) -> dict:
    command = [str(binary), "--headless", f"--config={config}"]
    if runtime:
        command.extend(("--duration=1", "--write-every=1", "--checkpoint-every=0",
                        "--threads=1", "--log-profile=normal", f"--outdir={directory / 'output'}"))
    else:
        command.append("--validate-only")
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith(("NIMBUS_", "OMP_", "KMP_", "GOMP_")):
            del environment[key]
    environment["OMP_NUM_THREADS"] = "1"
    completed = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                               capture_output=True, timeout=30, check=False)
    log = directory / ("runtime.log" if runtime else "admission.log")
    log.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    return {"returncode": completed.returncode, "log": str(log), "logSha256": sha256(log),
            "configSha256": sha256(config)}


def inspect_publication(output: Path) -> dict:
    frames = sorted(path for path in output.glob("step_*") if path.is_dir())
    if len(frames) != 2:
        raise RuntimeError(f"expected initial and final radar frames, found {len(frames)}")
    result = {"frames": len(frames), "finalManifestSha256": "", "fields": {}, "radarProvenance": {}}
    final = frames[-1]
    manifest_path = final / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (set(manifest.get("published_fields", [])) != FIELDS
            or manifest.get("grid", {}).get("dimensions") != {"x": 12, "y": 12, "z": 48}
            or manifest.get("data_product", {}).get("storage_format") != "npy_3d"):
        raise RuntimeError("radar field membership, geometry or format differs")
    metadata = {row["field_id"]: row for row in manifest["fields"]}
    for field in FIELDS:
        path = final / f"{field}.npy"
        shape, _, values = read_float32_npy(path)
        if shape != (12, 12, 48) or not values or any(not math.isfinite(value) for value in values):
            raise RuntimeError(f"{field} publication is malformed")
        result["fields"][field] = {
            "sha256": sha256(path), "shape": list(shape), "units": metadata[field]["units"],
            "min": min(values), "max": max(values),
        }
    if result["fields"]["reflectivity_dbz"]["units"] != "dBZ" or \
            result["fields"]["radial_velocity"]["units"] != "m/s":
        raise RuntimeError("radar publication units differ")
    result["finalManifestSha256"] = sha256(manifest_path)
    result["radarProvenance"] = manifest.get("radar_provenance", {})
    if result["radarProvenance"].get("products") != ["reflectivity_dbz", "radial_velocity"]:
        raise RuntimeError("radar operator provenance differs from selected products")
    return result


def inspect_agrid_publication(output: Path) -> dict:
    frames = sorted(path for path in output.glob("step_*") if path.is_dir())
    if len(frames) != 2:
        raise RuntimeError(f"expected initial and final A-grid frames, found {len(frames)}")
    final = frames[-1]
    manifest_path = final / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = FIELDS - {"reflectivity_dbz", "radial_velocity"}
    if (set(manifest.get("published_fields", [])) != expected
            or manifest.get("grid", {}).get("dimensions") != {"x": 18, "y": 18, "z": 18}
            or manifest.get("data_product", {}).get("storage_format") != "npy_3d"):
        raise RuntimeError("A-grid field membership, geometry or format differs")
    fields = {}
    for field in expected:
        path = final / f"{field}.npy"
        shape, _, values = read_float32_npy(path)
        if shape != (18, 18, 18) or not values or any(not math.isfinite(value) for value in values):
            raise RuntimeError(f"{field} A-grid publication is malformed")
        fields[field] = {"sha256": sha256(path), "shape": list(shape)}
    return {"frames": len(frames), "finalManifestSha256": sha256(manifest_path), "fields": fields}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    binary_sha = sha256(binary)
    source = BASE_CONFIG.read_text(encoding="utf-8")
    agrid_source = AGRID_CONFIG.read_text(encoding="utf-8")
    raw_root = args.raw_root.resolve()
    raw_root.mkdir(parents=True, exist_ok=False)

    malformed_cm1 = raw_root / "malformed_cm1.txt"
    malformed_cm1.write_text(replace_one(CM1_PROFILE.read_text(encoding="utf-8"),
                                         "4000 314 5 12 4\n", "1000 314 5 12 4\n"), encoding="utf-8")
    malformed_sharpy = raw_root / "malformed_sharpy.txt"
    malformed_sharpy.write_text(replace_one(SHARPY_PROFILE.read_text(encoding="utf-8"),
                                            "TMV_SHARPY_PROFILE_V1\n", "NOT_A_SOUNDING\n"), encoding="utf-8")
    variants = {
        "sharpy_auto": source,
        "cm1_auto": replace_one(source, FILE_LINE, f"    file_path: {CM1_PROFILE}\n"),
        "explicit_cm1_selector": replace_one(source, FILE_LINE,
            f"    file_path: {CM1_PROFILE}\n    file:\n      scheme_id: cm1\n"),
        "malformed_cm1": replace_one(source, FILE_LINE, f"    file_path: {malformed_cm1}\n"),
        "malformed_sharpy": replace_one(source, FILE_LINE, f"    file_path: {malformed_sharpy}\n"),
        "agrid_cm1_auto": replace_one(agrid_source, AGRID_SOUNDING,
            f"  sounding:\n    type: file\n    file_path: {CM1_PROFILE}\n"),
        "agrid_sharpy_auto": replace_one(agrid_source, AGRID_SOUNDING,
            f"  sounding:\n    type: file\n    file_path: {SHARPY_PROFILE}\n"),
        "agrid_malformed_cm1": replace_one(agrid_source, AGRID_SOUNDING,
            f"  sounding:\n    type: file\n    file_path: {malformed_cm1}\n"),
        "agrid_malformed_sharpy": replace_one(agrid_source, AGRID_SOUNDING,
            f"  sounding:\n    type: file\n    file_path: {malformed_sharpy}\n"),
        "agrid_explicit_cm1_selector": replace_one(agrid_source, AGRID_SOUNDING,
            f"  sounding:\n    type: file\n    file_path: {CM1_PROFILE}\n"
            "    file:\n      scheme_id: cm1\n"),
    }
    cases = {}
    for name, yaml in variants.items():
        directory = raw_root / name
        directory.mkdir()
        config = directory / "config.yaml"
        config.write_text(yaml, encoding="utf-8")
        admission = execute(binary, config, directory, False)
        case = {"config": str(config), "configSha256": sha256(config), "admission": admission}
        if admission["returncode"] == 0 and name in (
            "sharpy_auto", "cm1_auto", "agrid_cm1_auto", "agrid_sharpy_auto"
        ):
            runtime = execute(binary, config, directory, True)
            case["runtime"] = runtime
            if runtime["returncode"] == 0:
                case["publication"] = (
                    inspect_agrid_publication(directory / "output") if name.startswith("agrid_")
                    else inspect_publication(directory / "output")
                )
        cases[name] = case
    if sha256(binary) != binary_sha:
        raise RuntimeError("simulator changed during sounding/radar probe")
    expected_positive = {
        "sharpy_auto": "sharpy",
        "agrid_cm1_auto": "cm1",
        "agrid_sharpy_auto": "sharpy",
    }
    for name, reader in expected_positive.items():
        case = cases[name]
        runtime = case.get("runtime", {})
        log = Path(runtime["log"]).read_text(encoding="utf-8") if runtime else ""
        if (
            case["admission"]["returncode"] != 0 or runtime.get("returncode") != 0 or
            "publication" not in case or f"Auto-detected sounding reader: {reader}" not in log or
            "fallback=off" not in log or "[SCIENTIFIC INVALIDITY]" in log
        ):
            raise RuntimeError(f"{name} did not reach its ordinary {reader} reader and output")
        validation = json.loads((Path(runtime["log"]).parent / "output/validation_summary.json").read_text())
        if validation.get("failed") is not False or validation.get("total_violations") != 0:
            raise RuntimeError(f"{name} produced invalid science")
        case["validationSummarySha256"] = sha256(Path(runtime["log"]).parent / "output/validation_summary.json")
    expected_negative = {
        "cm1_auto": "qualified radar execution is limited",
        "malformed_cm1": "qualified radar execution is limited",
        "malformed_sharpy": "qualified radar execution is limited",
        "explicit_cm1_selector": "initial_conditions.sounding.file.scheme_id: unrecognized v1 key",
        "agrid_explicit_cm1_selector": "initial_conditions.sounding.file.scheme_id: unrecognized v1 key",
        "agrid_malformed_cm1": "heights must increase strictly",
        "agrid_malformed_sharpy": "cannot auto-detect sounding format",
    }
    for name, marker in expected_negative.items():
        case = cases[name]
        log = Path(case["admission"]["log"]).read_text(encoding="utf-8")
        if case["admission"]["returncode"] != 2 or marker not in log or "runtime" in case:
            raise RuntimeError(f"{name} did not reject at its expected admission owner")
        case["rejectionOwnerMarker"] = marker
    report = {
        "schema": "nimbus.r4-sounding-radar-public-probe.v1",
        "status": "pass_bounded_cpu_sounding_and_radar",
        "binary": {"path": str(binary), "sha256": binary_sha},
        "inputs": {
            "baseConfigSha256": sha256(BASE_CONFIG), "agridConfigSha256": sha256(AGRID_CONFIG),
            "cm1ProfileSha256": sha256(CM1_PROFILE),
            "sharpyProfileSha256": sha256(SHARPY_PROFILE), "probeScriptSha256": sha256(Path(__file__)),
        },
        "cases": cases,
        "claimBoundary": "One qualified 12x12x48 C-grid Lin/YSU/RRTMG CPU radar fixture with auto-detected CM1 and SHARPy file branches plus malformed-input and direct-selector probes; not all soundings, radar products, outputs, or backends.",
    }
    output = args.output.resolve()
    if output.exists():
        parser.error("use a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({name: {"admission": case["admission"]["returncode"],
                             "runtime": case.get("runtime", {}).get("returncode")}
                      for name, case in cases.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
