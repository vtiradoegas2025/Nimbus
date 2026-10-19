#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
OWNER_PATTERN = re.compile(
    r'\{"([a-z0-9_]+)",\s*Action::([A-Za-z0-9_]+),\s*'
    r'Ready::([A-Za-z0-9_]+),\s*"([^"]+)"\}',
    re.MULTILINE,
)
EXPECTED_ACTIONS = {
    "primary_atmosphere": "ShiftWithLeadingState",
    "base_state_profiles": "Preserve",
    "simulation_clock_and_constants": "Preserve",
    "retiring_physical_ledgers": "ShiftAndExportEarthTile",
    "surface_instantaneous_diagnostics": "ShiftWithLeadingState",
    "surface_accumulated_rainfall": "ShiftAndExportEarthTile",
    "nested_grid": "TranslateEarthCoordinates",
    "derived_diagnostic_fields": "InvalidateAndRecompute",
    "dynamics_operator": "InvalidateAndRecompute",
    "boundary_closure_and_kw78_history": "RebuildFromShiftedState",
    "radiation_runtime": "ShiftPersistentInvalidateTransient",
    "turbulence_runtime": "InvalidateAndRecompute",
    "boundary_layer_runtime": "ShiftPersistentInvalidateTransient",
    "chaos_runtime": "ShiftPersistentInvalidateTransient",
    "rayleigh_runtime": "Preserve",
    "microphysics_runtime": "ShiftPersistentInvalidateTransient",
    "advection_runtime": "InvalidateAndRecompute",
    "diffusion_runtime": "InvalidateAndRecompute",
    "time_stepping_runtime": "InvalidateAndRecompute",
    "terrain_runtime": "ShiftWithLeadingState",
    "radar_runtime": "InvalidateAndRecompute",
    "compute_residency": "InvalidateDeviceResidency",
    "parity_history": "Preserve",
    "storm_tracking_and_partition": "TranslateEarthCoordinates",
    "lagrangian_trajectories": "TranslateEarthCoordinates",
    "domain_frame": "TranslateEarthCoordinates",
    "durable_global_swath_output": "ShiftAndExportEarthTile",
    "checkpoint_transaction_provenance": "TranslateEarthCoordinates",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {
        "path": resolved.relative_to(ROOT).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def parse_contracts(source: str) -> list[dict[str, str]]:
    marker = "domain_recycling_owner_contracts() noexcept"
    start = source.find(marker)
    end = source.find("std::size_t remaining_domain_recycling_owner_contracts", start)
    if start < 0 or end < 0:
        raise RuntimeError("domain-recycling owner contract block is missing")
    records = [
        {"id": match[0], "action": match[1], "readiness": match[2], "scope": match[3]}
        for match in OWNER_PATTERN.findall(source[start:end])
    ]
    return records


def validate_contracts(records: list[dict[str, str]]) -> dict[str, int]:
    if len(records) != 28:
        raise RuntimeError(f"expected 28 domain-recycling owners, got {len(records)}")
    identities = [record["id"] for record in records]
    if len(set(identities)) != len(identities):
        raise RuntimeError("domain-recycling owner identities are duplicated")
    if set(identities) != set(EXPECTED_ACTIONS):
        raise RuntimeError("domain-recycling owner identities drifted")
    counts: dict[str, int] = {}
    for record in records:
        identity = record["id"]
        if record["action"] != EXPECTED_ACTIONS[identity]:
            raise RuntimeError(f"domain-recycling owner action drifted: {identity}")
        if record["readiness"] != "ReadyForTransaction":
            raise RuntimeError(f"domain-recycling owner is not ready for transaction: {identity}")
        if not record["scope"].strip():
            raise RuntimeError(f"domain-recycling owner scope is empty: {identity}")
        counts[record["action"]] = counts.get(record["action"], 0) + 1
    return dict(sorted(counts.items()))


def write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument(
        "--test-binary", type=Path, default=ROOT / "bin/test_core_domain_recycling",
    )
    parser.add_argument(
        "--source", type=Path, default=ROOT / "src/core/infra/domain_recycling_state.cpp",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    binary = args.binary.resolve(strict=True)
    test_binary = args.test_binary.resolve(strict=True)
    source = args.source.resolve(strict=True)
    records = parse_contracts(source.read_text(encoding="utf-8"))
    action_counts = validate_contracts(records)
    command = [str(test_binary), "[core][domain_recycling][owners][D5]"]
    result = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("compiled domain-recycling owner gate failed:\n" + result.stdout)
    match = re.search(r"All tests passed \((\d+) assertions in (\d+) test case", result.stdout)
    if match is None or tuple(map(int, match.groups())) != (463, 1):
        raise RuntimeError("compiled domain-recycling owner gate summary drifted")

    if args.output is not None:
        output = args.output.resolve()
        evidence = {
            "schema": "nimbus.domain-recycling-owner-evidence.v1",
            "version": 1,
            "binary": record(binary),
            "testBinary": record(test_binary),
            "source": record(source),
            "receipt": {
                "command": [record(test_binary)["path"], command[1]],
                "exitStatus": result.returncode,
                "assertions": 463,
                "testCases": 1,
                "skipped": 0,
            },
            "summary": {
                "owners": len(records),
                "readyForTransaction": len(records),
                "remaining": 0,
                "actions": action_counts,
            },
            "owners": records,
            "claimBoundary": (
                "Root D5 domain-recycling transaction ownership; not universal motion/nesting "
                "composition, a long campaign, or external review."
            ),
        }
        write_atomic(output, evidence)
    print(
        "[domain-recycling-owner-contracts] PASS: 28/28 owners ReadyForTransaction; "
        f"remaining=0; binary_sha256={sha256_file(binary)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
