#!/usr/bin/env python3
"""Check guarded Metal/ZFP and qualified CPU/ZFP invalid/resource neighbors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "tests/fixtures/reference/catalog/generated_metal_zfp_candidate_v1.yaml"
NEIGHBORS = {
    "precision_mode": ("  zfp_mode: accuracy\n", "  zfp_mode: precision\n", "requires accuracy mode"),
    "sparse_prefilter": ("  zfp_sparse_threshold: 0\n", "  zfp_sparse_threshold: 1.0e-6\n",
                         "forbids sparse thresholding"),
    "float16_prefilter": ("  zfp_float16_prefilter: false\n", "  zfp_float16_prefilter: true\n",
                          "forbids the float16 prefilter"),
    "excess_theta_error": ("    theta: 1.0e-3\n", "    theta: 1.0e-2\n", "exceeds qualified maximum"),
    "inactive_predictive": ("  zfp_keyframe_interval: 4\n", "  zfp_keyframe_interval: 2\n",
                            "predictive delta requires a keyframe interval of at least three"),
    "missing_per_field_policy": ("  zfp_per_field_tolerances: true\n",
                                 "  zfp_per_field_tolerances: false\n", "requires explicit per-field tolerances"),
    "unbounded_retention": ("  keep_last_frames: 3\n", "  keep_last_frames: 0\n",
                            "requires positive finite frame retention"),
    "disabled_free_space": ("  minimum_free_space_mib: 4096\n", "  minimum_free_space_mib: 0\n",
                            "requires a positive retained free-space reserve"),
    "disabled_file_ceiling": ("  maximum_retained_history_files: 256\n",
                              "  maximum_retained_history_files: 0\n",
                              "requires a positive retained history file ceiling"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}


def replace_one(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"invalid ZFP mutation anchor: {old!r}")
    return text.replace(old, new, 1)


def validate(binary: Path, config: Path, log: Path) -> tuple[int, str]:
    result = subprocess.run([str(binary), "--validate-only", "--config", str(config)],
                            cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=60, check=False)
    log.write_text(result.stdout)
    return result.returncode, result.stdout


def runtime_resource_failure(binary: Path, source: str, output: Path, name: str,
                             old: str, new: str, evidence: str) -> dict:
    case = output / name
    case.mkdir()
    config = case / "config.yaml"
    config.write_text(replace_one(source, old, new))
    directory = case / "run"
    result = subprocess.run([str(binary), "--headless", "--config", str(config),
                             "--duration=1", "--write-every=1", "--checkpoint-every=0",
                             f"--outdir={directory}", "--threads=1", "--log-profile=quiet"],
                            cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=60, check=False,
                            env={**os.environ, "OMP_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE"})
    log = case / "runtime.log"
    log.write_text(result.stdout)
    published = list(directory.glob("step_*")) if directory.exists() else []
    if result.returncode == 0 or evidence not in result.stdout or published:
        raise RuntimeError(f"{name} did not fail closed without output publication: {result.stdout[-2500:]}")
    return {"status": "rejected_without_publication", "config": identity(config),
            "log": identity(log), "returncode": result.returncode, "publishedSteps": 0,
            "requiredDiagnostic": evidence}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve(strict=True)
    source = CANDIDATE.read_text()
    qualified_metal = replace_one(source, "    mode: exploratory\n", "    mode: qualified\n")
    qualified_cpu = replace_one(qualified_metal, "    backend: metal\n", "    backend: cpu\n")
    inputs = {"binary": identity(binary), "candidate": identity(CANDIDATE),
              "checker": identity(Path(__file__))}
    positive = output / "qualified-cpu.yaml"
    positive.write_text(qualified_cpu)
    code, diagnostic = validate(binary, positive, output / "qualified-cpu-validation.log")
    if code != 0 or "[VALIDATE] PASS:" not in diagnostic:
        raise RuntimeError("qualified CPU/ZFP control did not pass ordinary admission")
    guarded = output / "qualified-metal-guarded.yaml"
    guarded.write_text(qualified_metal)
    code, diagnostic = validate(binary, guarded, output / "qualified-metal-validation.log")
    if code == 0 or "qualified Metal execution is limited to exact public" not in diagnostic or \
            "output.format=zfp:" in diagnostic:
        raise RuntimeError("qualified Metal/ZFP candidate did not retain exact public guard")
    results = {}
    for name, (old, new, expected) in NEIGHBORS.items():
        config = output / f"invalid-{name}.yaml"
        config.write_text(replace_one(qualified_cpu, old, new))
        log = output / f"invalid-{name}.log"
        code, diagnostic = validate(binary, config, log)
        if code == 0 or expected not in diagnostic or "[VALIDATE] PASS:" in diagnostic:
            raise RuntimeError(f"{name} escaped qualified ZFP preflight: {diagnostic[-2500:]}")
        results[name] = {"status": "rejected", "config": identity(config),
                         "log": identity(log), "requiredDiagnostic": expected}
    results["retained_file_ceiling"] = runtime_resource_failure(
        binary, qualified_cpu, output, "retained-file-ceiling",
        "  maximum_retained_history_files: 256\n",
        "  maximum_retained_history_files: 1\n",
        "exceeds output.maximum_retained_history_files")
    results["impossible_free_space"] = runtime_resource_failure(
        binary, qualified_cpu, output, "impossible-free-space",
        "  minimum_free_space_mib: 4096\n",
        "  minimum_free_space_mib: 2147483647\n",
        "insufficient free space")
    if any(identity(Path(value["path"])) != value for value in inputs.values()):
        raise RuntimeError("ZFP invalid/resource inputs changed during execution")
    report = {"schema": "nimbus.generated-metal-zfp-invalid.v1",
              "status": "complete_guarded", "claimBoundary":
              "One exact candidate's qualified CPU ZFP output policy and guarded Metal neighbor; "
              "negative resource/selection checks do not establish qualified Metal admission.",
              "inputs": inputs, "qualifiedCpuControl": identity(positive),
              "qualifiedMetalGuard": identity(guarded),
              "results": [{"id": name, **value} for name, value in sorted(results.items())]}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "invalidNeighbors": len(results),
                      "metalPublicGuard": "closed"}), flush=True)


if __name__ == "__main__":
    main()
