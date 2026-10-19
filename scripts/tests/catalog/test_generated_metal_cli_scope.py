#!/usr/bin/env python3
"""Require exact generated Metal CLI validation to match effective runtime owners."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
NPY = ROOT / "tests/fixtures/reference/catalog/generated_metal_canonical_v1.yaml"
NETCDF = ROOT / "tests/fixtures/reference/catalog/generated_metal_netcdf_v1.yaml"
ZFP = ROOT / "tests/fixtures/reference/catalog/generated_metal_zfp_candidate_v1.yaml"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    cases = (
        ("npy_base", NPY, (), True),
        ("npy_duration_short", NPY, ("--duration=1",), True),
        ("npy_timestep_same", NPY, ("--dt-large=0.1",), True),
        ("npy_substeps_same", NPY, ("--acoustic-substeps=6",), True),
        ("npy_write_one", NPY, ("--write-every=1",), True),
        ("npy_write_cadence_301", NPY, ("--write-every=301",), True),
        ("npy_long_cadence", NPY, ("--duration=301", "--write-every=301"), True),
        ("npy_guard_strict", NPY, ("--guard-mode=strict",), True),
        ("npy_guard_fail_both", NPY, ("--guard-fail-on=both",), True),
        ("npy_required_guard_scope", NPY, ("--guard-scope=required",), True),
        ("npy_timestep_changed", NPY, ("--dt-large=0.2",), False),
        ("npy_substeps_changed", NPY, ("--acoustic-substeps=7",), False),
        ("npy_write_unreviewed", NPY, ("--write-every=2",), False),
        ("npy_write_disabled", NPY, ("--write-every=0",), False),
        ("npy_live_shm", NPY, ("--live-shm",), False),
        ("npy_trajectories", NPY, ("--trajectories",), False),
        ("npy_exported_guard_scope", NPY, ("--guard-scope=exported",), False),
        ("npy_guard_off", NPY, ("--guard-mode=off",), False),
        ("npy_guard_nonfinite_only", NPY, ("--guard-fail-on=nonfinite",), False),
        ("netcdf_base", NETCDF, (), True),
        ("netcdf_duration_short", NETCDF, ("--duration=1",), True),
        ("netcdf_write_cadence_301", NETCDF, ("--write-every=301",), True),
        ("netcdf_long_cadence", NETCDF, ("--duration=301", "--write-every=301"), True),
        ("netcdf_timestep_changed", NETCDF, ("--dt-large=0.2",), False),
        ("netcdf_write_unreviewed", NETCDF, ("--write-every=2",), False),
    )
    failures = []
    for name, config, flags, expected in cases:
        result = subprocess.run([str(binary), "--config", str(config), "--validate-only", *flags],
                                cwd=ROOT, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=30, check=False)
        passed = result.returncode == 0 and "[VALIDATE] PASS:" in result.stdout and \
            "[VALIDATE WARNING]" not in result.stdout
        if passed != expected:
            failures.append({"name": name, "expectedAdmission": expected,
                             "actualAdmission": passed, "exitCode": result.returncode,
                             "diagnostic": result.stdout[-600:]})
    with tempfile.TemporaryDirectory(prefix="nimbus-generated-metal-zfp-cli-") as temporary:
        source = ZFP.read_text(encoding="utf-8")
        if source.count("    mode: exploratory\n") != 1:
            raise RuntimeError("ZFP qualified-mode mutation anchor is not unique")
        qualified_zfp = Path(temporary) / "qualified-zfp.yaml"
        qualified_zfp.write_text(
            source.replace("    mode: exploratory\n", "    mode: qualified\n", 1),
            encoding="utf-8",
        )
        zfp_cases = (
            ("zfp_base", (), True),
            ("zfp_duration_short", ("--duration=1",), True),
            ("zfp_write_one", ("--write-every=1",), True),
            ("zfp_write_301", ("--write-every=301",), False),
            ("zfp_write_unreviewed", ("--write-every=2",), False),
            ("zfp_timestep_changed", ("--dt-large=0.2",), False),
            ("zfp_guard_off", ("--guard-mode=off",), False),
        )
        for name, flags, expected in zfp_cases:
            result = subprocess.run(
                [str(binary), "--config", str(qualified_zfp), "--validate-only", *flags],
                cwd=ROOT, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=30, check=False,
            )
            passed = result.returncode == 0 and "[VALIDATE] PASS:" in result.stdout and \
                "[VALIDATE WARNING]" not in result.stdout
            if passed != expected:
                failures.append({"name": name, "expectedAdmission": expected,
                                 "actualAdmission": passed, "exitCode": result.returncode,
                                 "diagnostic": result.stdout[-600:]})
    print(json.dumps({"schema": "nimbus.generated-metal-cli-scope-test.v1",
                      "claimBoundary": "Exact generated Metal NPY/netCDF/ZFP-high validate-only versus supported operational CLI envelope; source regression test, not whole-backend readiness.",
                      "binarySha256": sha(binary),
                      "testSha256": sha(Path(__file__)),
                      "configSha256": {"npy": sha(NPY), "netcdf": sha(NETCDF),
                                       "zfpCandidate": sha(ZFP)},
                      "cases": len(cases) + len(zfp_cases),
                      "status": "pass" if not failures else "fail", "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
