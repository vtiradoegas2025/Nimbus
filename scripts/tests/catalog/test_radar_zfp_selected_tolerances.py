#!/usr/bin/env python3
"""Bound explicit qualified ZFP tolerances for selected radar output fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.qualification.output import qualify_radar_product_sets as qualifier
from scripts.qualification.output.qualify_radar_product_sets import (
    BASE_CONFIG, PRODUCTS, catalog_zfp_budgets, configured_source, with_zfp_output,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(binary: Path, config: Path) -> tuple[int, str]:
    result = subprocess.run(
        [str(binary), "--validate-only", "--config", str(config)],
        cwd=ROOT, text=True, capture_output=True, timeout=30, check=False,
    )
    return result.returncode, result.stdout + result.stderr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    budgets, catalog_sha = catalog_zfp_budgets(binary)
    source = configured_source(BASE_CONFIG.read_text(encoding="utf-8"), PRODUCTS)
    source = with_zfp_output(source, budgets)
    cases = []
    with tempfile.TemporaryDirectory(prefix="nimbus-radar-selected-zfp-") as temporary:
        root = Path(temporary)
        valid = root / "valid.yaml"
        valid.write_text(source, encoding="utf-8")
        code, log = validate(binary, valid)
        cases.append({"variant": "all_explicit", "returncode": code,
                      "passed": code == 0 and "[VALIDATE] PASS" in log})
        for field in PRODUCTS:
            anchor = f"    {field}: {budgets[field]:.17g}\n"
            if source.count(anchor) != 1:
                raise RuntimeError(f"selected {field} has no unique qualified budget")
            missing = root / f"missing_{field}.yaml"
            missing.write_text(source.replace(anchor, "", 1), encoding="utf-8")
            code, log = validate(binary, missing)
            expected = f"field '{field}' is missing its explicit ZFP tolerance"
            cases.append({"variant": f"missing_{field}", "returncode": code,
                          "expectedReason": expected,
                          "passed": code != 0 and expected in log})
    failed = [case for case in cases if not case["passed"]]
    print(json.dumps({
        "schema": "nimbus.radar-selected-zfp-tolerances.v1",
        "status": "pass" if not failed else "fail",
        "binarySha256": sha256(binary),
        "catalogSha256": catalog_sha,
        "fixtureSha256": sha256(BASE_CONFIG),
        "configSha256": hashlib.sha256(source.encode()).hexdigest(),
        "qualifierSha256": sha256(Path(qualifier.__file__)),
        "testSha256": sha256(Path(__file__)),
        "cases": cases,
        "claimBoundary": "One five-product qualified CPU ZFP admission and five missing-field negatives; not decoded codec parity or all output presets.",
    }, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
