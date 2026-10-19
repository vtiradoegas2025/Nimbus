#!/usr/bin/env python3
"""Seal bounded exact-binary Studio parity for guarded Milbrandt policy13."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
UI = ROOT / "ui"
SOURCES = {
    "snapshot": UI / "src/lib/config/nativeCatalogSnapshot.json",
    "browserParser": UI / "src/lib/config/schemeCatalog.ts",
    "browserTests": UI / "../tests/ui/unit/config/test_scheme_catalog.ts",
    "typescriptPreflight": UI / "src/lib/config/configPreflight.ts",
    "typescriptPreflightTests": UI / "../tests/ui/unit/config/test_config_preflight.ts",
    "rustPreflight": UI / "src-tauri/src/commands/config_preflight.rs",
    "formalParity": ROOT / "scripts/tests/catalog/test_studio_native_parity.py",
    "qualifier": Path(__file__),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--expected-binary-sha256", required=True)
    parser.add_argument("--cargo-target-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    raw = args.raw_root.resolve()
    report_path = args.report_output.resolve()
    if raw.exists() or report_path.exists():
        parser.error("raw root and report must be new paths")
    raw.mkdir(parents=True)
    binary_before = sha256(binary)
    sources_before = {name: sha256(path) for name, path in SOURCES.items()}
    if binary_before != args.expected_binary_sha256:
        raise RuntimeError("simulator binary changed before the policy13 gate")
    catalog_bytes = subprocess.check_output([str(binary), "--scheme-catalog=json"], cwd=ROOT)
    (raw / "native_catalog_stdout.json").write_bytes(catalog_bytes)
    catalog = json.loads(catalog_bytes)
    policy = catalog["runtimeCapabilities"]["interactionPolicy"]
    triple = policy.get("milbrandtTripleMoment")
    rows = [row for row in catalog["readiness"]["items"]
            if row["id"] == "microphysics.milbrandt_triple_moment"]
    if (catalog["catalogVersion"] != 9 or policy["version"] != 13 or
            not isinstance(triple, dict) or triple.get("retained") is not True or
            triple.get("admitted") is not False or
            triple.get("strictV1ExplicitKeyAdmitted") is not False or
            len(rows) != 1 or rows[0]["state"] != "engineering_incomplete" or
            not rows[0]["blocker"] or rows[0]["campaignEvidence"] or
            rows[0]["externalReviewEvidence"] or
            json.loads(SOURCES["snapshot"].read_text(encoding="utf-8")) != catalog):
        raise RuntimeError("policy13 native catalog or browser snapshot is not the reviewed guarded scope")

    environment = os.environ.copy()
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "CARGO_TARGET_DIR": str(args.cargo_target_dir.resolve()),
        "CARGO_BUILD_JOBS": "2",
        "NIMBUS_MILBRANDT_TRIPLE_BINARY_SHA256": binary_before,
    })
    cases = [
        ("native_authority", ["python3", "scripts/tests/catalog/test_catalog_authority.py",
                               "--binary", str(binary)], ROOT, None),
        ("browser_snapshot", ["python3", "scripts/tests/catalog/test_browser_catalog_snapshot.py",
                              "--binary", str(binary)], ROOT, "90 readiness items"),
        ("typescript_catalog", ["npm", "run", "test:scheme-catalog"], UI,
         "scheme catalog: 142 checks passed"),
        ("typescript_preflight", ["npm", "run", "test:config-preflight"], UI,
         "config preflight tests passed"),
        ("typescript_typecheck", ["npm", "run", "typecheck"], UI, None),
        ("rust_preflight", ["cargo", "test", "--offline", "--lib",
                            "commands::config_preflight::tests::", "--", "--test-threads=1"],
         UI / "src-tauri", "31 passed; 0 failed; 5 ignored"),
        ("rust_policy13_exact_source", ["cargo", "test", "--offline", "--lib",
                                        "policy13_real_exact_source_keeps_triple_guard_and_double_control",
                                        "--", "--ignored", "--nocapture"],
         UI / "src-tauri", "1 passed; 0 failed"),
        ("rust_east_radar_hidden_neighbors", ["cargo", "test", "--offline", "--lib",
                                             "v9_exact_native_source_distinguishes_east_and_radar_hidden_neighbors",
                                             "--", "--ignored", "--nocapture"],
         UI / "src-tauri", "1 passed; 0 failed"),
        ("formal_715_plus_440", ["python3", "scripts/tests/catalog/test_studio_native_parity.py",
                                 "--binary", str(binary)], ROOT,
         "715 qualified fixtures and 440 shipped presets"),
    ]
    results = []
    status = "pass"
    for label, command, cwd, expected in cases:
        try:
            result = subprocess.run(command, cwd=cwd, env=environment, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    timeout=360, check=False)
            output = result.stdout
            exit_code = result.returncode
        except subprocess.TimeoutExpired as error:
            output = str(error)
            exit_code = -1
        log = raw / f"{label}.log"
        log.write_text(output, encoding="utf-8")
        results.append({"case": label, "exitCode": exit_code,
                        "logPath": log.relative_to(ROOT).as_posix(),
                        "logSha256": sha256(log)})
        if (exit_code != 0 or expected is not None and expected not in output or
                sha256(binary) != binary_before or
                any(sha256(path) != sources_before[name] for name, path in SOURCES.items())):
            status = "red"
            break
    report = {
        "schema": "nimbus.policy13-guarded-studio-parity.v1",
        "status": status,
        "claimBoundary": "Policy13 Milbrandt triple moment remains Engineering incomplete and unadmitted; descriptor-only parity does not prove arbitrary YAML equivalence or whole-backend readiness.",
        "binarySha256Before": binary_before,
        "binarySha256After": sha256(binary),
        "catalogStdoutSha256": hashlib.sha256(catalog_bytes).hexdigest(),
        "sourceSha256Before": sources_before,
        "sourceSha256After": {name: sha256(path) for name, path in SOURCES.items()},
        "qualifiedFixtureCount": 715,
        "shippedPresetCount": 440,
        "cases": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if status != "pass":
        raise RuntimeError(f"policy13 Studio gate stopped red after {results[-1]['case']}")
    print(json.dumps({"status": status, "binarySha256": binary_before,
                      "caseCount": len(results)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
