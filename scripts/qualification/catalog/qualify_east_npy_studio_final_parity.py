#!/usr/bin/env python3
"""Seal exact-binary browser/Rust parity for the east-NPY Metal extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.qualification.catalog_inventory import inventory_counts

UI = ROOT / "ui"
SOURCES = {
    "browserSnapshot": ROOT / "ui/src/lib/config/nativeCatalogSnapshot.json",
    "browserCatalogParser": ROOT / "ui/src/lib/config/schemeCatalog.ts",
    "browserCatalogTests": ROOT / "tests/ui/unit/config/test_scheme_catalog.ts",
    "typescriptPreflight": ROOT / "ui/src/lib/config/configPreflight.ts",
    "typescriptPreflightTests": ROOT / "tests/ui/unit/config/test_config_preflight.ts",
    "rustPreflight": ROOT / "ui/src-tauri/src/commands/config_preflight.rs",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--cargo-target-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    if args.report_output.exists():
        parser.error("use a new report path")
    raw = args.raw_root.resolve()
    raw.mkdir(parents=True, exist_ok=False)
    binary_sha = sha256(binary)
    source_shas = {label: sha256(path) for label, path in SOURCES.items()}
    catalog_bytes = subprocess.check_output([str(binary), "--scheme-catalog=json"], cwd=ROOT)
    catalog = json.loads(catalog_bytes)
    if json.loads(SOURCES["browserSnapshot"].read_text()) != catalog:
        raise RuntimeError("browser snapshot differs from exact simulator catalog")
    scope = catalog["runtimeCapabilities"]["interactionPolicy"]["generatedMetalCanonicalScope"]
    if (catalog["runtimeCapabilities"]["interactionPolicy"]["version"] != 10 or
            scope["version"] != 4 or scope["tupleCount"] != 6049 or
            scope["distinctConfigCount"] != 4537 or
            scope["additionalBoundaryNpyScope"]["outputFormats"] != ["npy_3d"]):
        raise RuntimeError("binary no longer publishes the reviewed east-NPY-only extension")
    inventory = inventory_counts()
    if inventory["qualifiedFixtures"] != 715:
        raise RuntimeError("qualified fixture count changed before parity")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CARGO_TARGET_DIR"] = str(args.cargo_target_dir.resolve())
    cases = [
        ("catalog_authority", ["python3", "scripts/tests/catalog/test_catalog_authority.py",
                               "--binary", str(binary)], ROOT, None),
        ("browser_snapshot", ["python3", "scripts/tests/catalog/test_browser_catalog_snapshot.py",
                              "--binary", str(binary)], ROOT, "[browser-catalog-snapshot] PASS"),
        ("typescript_catalog", ["npm", "run", "test:scheme-catalog"], UI,
         "scheme catalog: 89 checks passed"),
        ("typescript_preflight", ["npm", "run", "test:config-preflight"], UI,
         "config preflight tests passed"),
        ("typescript_typecheck", ["npm", "run", "typecheck"], UI, None),
        ("rust_preflight", ["cargo", "test", "--locked", "--manifest-path",
                            str(UI / "src-tauri/Cargo.toml"), "commands::config_preflight", "--lib"],
         ROOT, "23 passed; 0 failed; 1 ignored"),
        ("native_studio_parity", ["python3", "scripts/tests/catalog/test_studio_native_parity.py",
                                  "--binary", str(binary)], ROOT,
         "matched native admission for 715 qualified fixtures and 440 shipped presets"),
    ]
    results = []
    for label, command, cwd, expected in cases:
        result = subprocess.run(command, cwd=cwd, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=360)
        log = raw / f"{label}.log"
        log.write_text(result.stdout)
        if result.returncode != 0 or (expected is not None and expected not in result.stdout):
            raise RuntimeError(f"{label} failed or expected result absent: {log}")
        results.append({"case": label, "exitCode": result.returncode,
                        "expectedSummary": expected, "logPath": str(log),
                        "logSha256": sha256(log)})
        if sha256(binary) != binary_sha or any(sha256(path) != source_shas[name]
                                               for name, path in SOURCES.items()):
            raise RuntimeError(f"binary or Studio source changed during {label}")
    report = {
        "schema": "nimbus.east-npy-studio-exact-parity.v1",
        "status": "pass_exact_binary_715_plus_440_parity",
        "claimBoundary": "Browser catalog and v8 TypeScript/Rust preflight matched this exact simulator's native admission for 715 maintained qualified fixtures and 440 shipped presets. East netCDF/ZFP/async/preset/cadence negatives are focused parser tests; Studio still lacks the raw strict-v1 YAML digest, so this does not prove exhaustive hash parity or whole-backend Ready.",
        "binarySha256": binary_sha,
        "catalogStdoutSha256": hashlib.sha256(catalog_bytes).hexdigest(),
        "canonicalCatalogSha256": hashlib.sha256(json.dumps(
            catalog, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "sourceSha256": source_shas,
        "qualifierSha256": sha256(Path(__file__)),
        "qualifiedFixtureCount": inventory["qualifiedFixtures"],
        "shippedPresetCount": 440,
        "rawRoot": str(raw),
        "tests": results,
        "rawStrictYamlHashParity": "not_available_in_studio_request",
    }
    output = args.report_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "binarySha256": binary_sha,
                      "tests": len(results)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
