#!/usr/bin/env python3
"""Require qualified ZFP presets to reject omitted per-field YAML tolerances."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests/fixtures/configs/qualification/zfp_core_history.yaml"
TIERS = ("zfp_low", "zfp_medium", "zfp_high")
FORMAT_ANCHOR = "  format: zfp_high\n"
TOLERANCE_ANCHOR = "    theta: 1.0e-3\n"
MISSING_ERROR = "field 'theta' is missing its explicit ZFP tolerance"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(binary: Path, config: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(binary), "--validate-only", "--config", str(config), "--duration=1"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    source = FIXTURE.read_text(encoding="utf-8")
    if source.count(FORMAT_ANCHOR) != 1 or source.count(TOLERANCE_ANCHOR) != 1:
        raise RuntimeError("qualified ZFP fixture mutation anchors are not unique")

    cases = []
    with tempfile.TemporaryDirectory(prefix="nimbus-zfp-explicit-tolerances-") as temporary:
        root = Path(temporary)
        for tier in TIERS:
            valid = root / f"{tier}-valid.yaml"
            valid.write_text(source.replace(FORMAT_ANCHOR, f"  format: {tier}\n", 1),
                             encoding="utf-8")
            valid_result = validate(binary, valid)
            valid_pass = valid_result.returncode == 0 and "[VALIDATE] PASS:" in valid_result.stdout
            cases.append({"tier": tier, "variant": "explicit", "passed": valid_pass,
                          "exitCode": valid_result.returncode,
                          "diagnostic": valid_result.stdout[-500:] if not valid_pass else ""})

            missing = root / f"{tier}-missing-theta.yaml"
            missing.write_text(valid.read_text(encoding="utf-8").replace(
                TOLERANCE_ANCHOR, "", 1), encoding="utf-8")
            missing_result = validate(binary, missing)
            missing_pass = missing_result.returncode != 0 and MISSING_ERROR in missing_result.stdout
            cases.append({"tier": tier, "variant": "missing_theta", "passed": missing_pass,
                          "exitCode": missing_result.returncode,
                          "diagnostic": missing_result.stdout[-500:] if not missing_pass else MISSING_ERROR})

    failures = [case for case in cases if not case["passed"]]
    print(json.dumps({
        "schema": "nimbus.qualified-zfp-explicit-tolerance-test.v1",
        "claimBoundary": "CPU ordinary validate-only per-field input contract for three qualified ZFP tiers; not codec parity or whole-backend readiness.",
        "binarySha256": sha(binary),
        "fixtureSha256": sha(FIXTURE),
        "testSha256": sha(Path(__file__)),
        "caseCount": len(cases),
        "status": "pass" if not failures else "fail",
        "cases": cases,
    }, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
