#!/usr/bin/env python3
"""Bound ordinary qualified static vertical-only admission neighbors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/fixtures/configs/vertical_only/static_cpu.yaml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise RuntimeError(f"expected one source token: {before!r}")
    return source.replace(before, after, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    raw = args.raw_root.resolve()
    report = args.report_output.resolve()
    if not binary.is_file() or raw.exists() or report.exists():
        parser.error("binary must exist and outputs must be new")
    source = SOURCE.read_text(encoding="utf-8")
    catalog_bytes = subprocess.check_output([str(binary), "--scheme-catalog=json"], cwd=ROOT)
    policy = json.loads(catalog_bytes)["runtimeCapabilities"]["interactionPolicy"]
    if policy["version"] != 14 or policy["qualifiedVerticalOnlySgs"]["admitted"] is not True:
        raise RuntimeError("exact binary does not advertise the expected public static vertical-only scope")
    identity = {"simulatorSha256": sha256(binary),
                "canonicalConfigSha256": sha256(SOURCE),
                "runnerSha256": sha256(Path(__file__))}
    raw.mkdir(parents=True)
    inputs = raw / "inputs"
    logs = raw / "logs"
    outputs = raw / "rejected_outputs"
    inputs.mkdir()
    logs.mkdir()
    outputs.mkdir()
    cases = {
        "grid_nx_13": replace_once(source, "  nx: 12\n", "  nx: 13\n"),
        "dt_large_0_1": replace_once(source, "  dt_large: 0.2\n", "  dt_large: 0.1\n"),
        "acoustic_5": replace_once(source, "  acoustic_substeps: 6\n", "  acoustic_substeps: 5\n"),
        "active_pbl": replace_once(source, "boundary_layer:\n  scheme: none\n",
                                   "boundary_layer:\n  scheme: slab\n"),
        "output_zfp": replace_once(source, "  format: npy_3d\n", "  format: zfp_high\n"),
        "async_output": replace_once(source, "  async_io: false\n", "  async_io: true\n"),
        "dynamic_vertical": replace_once(source, "  dynamic_coefficient: false\n",
                                         "  dynamic_coefficient: true\n"),
        "terrain_bell": replace_once(
            source, "surface:\n",
            "terrain:\n  scheme: bell\n  coordinate: smoothed\n"
            "  compute_derivatives: true\n  compute_metrics: true\n"
            "  smoothing_decay: 0.1\n  bell:\n    h0: 300\n"
            "    a: 2000\n    axisymmetric: true\n\nsurface:\n"),
    }
    rows: list[dict[str, object]] = []
    status = "pass"
    started = time.monotonic()
    for name, contents in (("canonical", source), *cases.items()):
        config = inputs / f"{name}.yaml"
        config.write_text(contents, encoding="utf-8")
        positive = name == "canonical"
        for mode in (("validate",) if positive else ("validate", "runtime")):
            outdir = outputs / name
            command = [str(binary), "--config", str(config)]
            if mode == "validate":
                command.append("--validate-only")
            else:
                command += ["--headless", "--duration=1", "--write-every=1",
                            "--checkpoint-every=0", "--threads=1",
                            f"--outdir={outdir}", "--log-profile=quiet"]
            result = subprocess.run(command, cwd=ROOT, capture_output=True,
                                    text=True, timeout=30, check=False)
            log = logs / f"{name}_{mode}.log"
            log.write_text(result.stdout + "\n" + result.stderr,
                           encoding="utf-8")
            no_publication = not outdir.exists() or not any(outdir.rglob("*"))
            passed = (result.returncode == 0) == positive and \
                (positive or no_publication)
            if name == "dynamic_vertical":
                diagnostic = log.read_text(encoding="utf-8")
                passed = passed and ("requires turbulence.mode=3d" in diagnostic or
                                     "dynamic Smagorinsky is qualified only" in diagnostic)
            rows.append({"name": name, "mode": mode, "exitCode": result.returncode,
                         "configSha256": sha256(config), "logSha256": sha256(log),
                         "noPublication": no_publication, "pass": passed})
            if not passed:
                status = "neighbor_deviation"
                break
        if status != "pass":
            break
    if status == "pass":
        command = [str(binary), "--config", str(SOURCE), "--validate-only",
                   "--compute-backend=metal"]
        result = subprocess.run(command, cwd=ROOT, capture_output=True,
                                text=True, timeout=30, check=False)
        log = logs / "backend_metal_validate.log"
        log.write_text(result.stdout + "\n" + result.stderr,
                       encoding="utf-8")
        rows.append({"name": "backend_metal", "mode": "validate",
                     "exitCode": result.returncode,
                     "configSha256": sha256(SOURCE), "logSha256": sha256(log),
                     "noPublication": True, "pass": result.returncode != 0})
        if result.returncode == 0:
            status = "backend_neighbor_deviation"
    if identity != {"simulatorSha256": sha256(binary),
                    "canonicalConfigSha256": sha256(SOURCE),
                    "runnerSha256": sha256(Path(__file__))}:
        status = "source_identity_drift"
    files = [item for item in raw.rglob("*") if item.is_file()]
    bytes_used = sum(item.stat().st_size for item in files)
    if len(files) >= 200 or bytes_used >= 8 * 1024 * 1024 or \
            time.monotonic() - started >= 90:
        status = "resource_cap"
    report.parent.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "nimbus.vertical-only-static-public-neighbors.v1",
               "status": status,
               "claimBoundary": "Exact ordinary qualified flat-CPU static vertical-only admission and prepublication neighbors; numerical Ready, backend parity and campaigns separate.",
               **identity,
               "catalogSha256": hashlib.sha256(catalog_bytes).hexdigest(),
               "cases": rows,
               "wallSeconds": time.monotonic() - started,
               "rawFiles": len(files), "rawBytes": bytes_used,
               "rawRoot": str(raw)}
    report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"{status}: {len(rows)} ordinary admission/neighbors")
    return 0 if status == "pass" and len(rows) == 18 else 1


if __name__ == "__main__":
    raise SystemExit(main())
