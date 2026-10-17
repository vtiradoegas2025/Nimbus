#!/usr/bin/env python3
"""Check bounded replay tests and an explicitly supplied evolving-history record."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SYNTHETIC_HISTORY = ROOT / "tests/qualification/fixtures/storm_history_synthetic.csv"
CM1_SOUNDINGS = ROOT / "tests/fixtures/soundings/cm1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_nimbus_history(history: Path) -> dict:
    require(history.is_file(), f"missing Nimbus history evidence: {history}")
    record = json.loads(history.read_text(encoding="utf-8"))
    require(record.get("schema") == "nimbus.qualification.constant_velocity_storm_lifetime", "wrong Nimbus history schema")
    require(record.get("schema_version") == 2, "wrong Nimbus history schema version")
    require(record.get("status") == "pass", "Nimbus D2 history did not pass")
    samples = record.get("storm_samples")
    coherent = record.get("coherent_samples")
    motion = record.get("motion_available_samples")
    require(isinstance(samples, int) and samples >= 100, "Nimbus history is too short for D4 evidence")
    require(isinstance(coherent, int) and coherent >= int(0.9 * samples), "Nimbus history lost coherent selection too often")
    require(isinstance(motion, int) and motion >= int(0.75 * coherent), "Nimbus history lacks sufficient fitted-motion evidence")
    require(record.get("branch_candidate_overflow_samples") == 0, "Nimbus history contains candidate overflow")
    require(record.get("dominant_branch_overflow_samples") == 0, "Nimbus dominant branch overflowed")
    require(record.get("maximum_consecutive_unselected_samples", samples) <= 1, "Nimbus history has a long selection outage")
    require(record.get("maximum_reorganization_recovery_samples", samples) <= 4, "Nimbus history recovers too slowly after reorganization")
    require(record.get("minimum_storm_boundary_distance_m", 0.0) > 0.0, "Nimbus history contacted a boundary")
    reasons = record.get("continuity_reason_counts")
    require(isinstance(reasons, dict) and sum(reasons.values()) == samples, "Nimbus continuity reasons do not cover every sample")
    return {
        "status": "pass",
        "path": str(history.resolve()),
        "sha256": sha256(history),
        "samples": samples,
        "coherent_samples": coherent,
        "motion_available_samples": motion,
        "topology_reorganization_samples": record.get("topology_reorganization_samples"),
        "maximum_reorganization_recovery_samples": record.get("maximum_reorganization_recovery_samples"),
        "minimum_boundary_distance_m": record.get("minimum_storm_boundary_distance_m"),
        "scope": "real_nimbus_evolving_history_aggregate",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/test_qualification_storm_history_replay")
    parser.add_argument("--history-report", type=Path, required=True)
    parser.add_argument("--evidence-out", type=Path, required=True)
    args = parser.parse_args()

    binary = args.binary.resolve()
    require(binary.is_file(), f"missing replay binary: {binary}")
    require(SYNTHETIC_HISTORY.is_file(), f"missing synthetic history: {SYNTHETIC_HISTORY}")
    result = subprocess.run(
        [str(binary)], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    require(result.returncode == 0, f"synthetic storm history replay failed\n{result.stdout[-6000:]}")
    assertions = re.search(r"All tests passed \((\d+) assertions in (\d+) test case", result.stdout)
    require(assertions is not None, "synthetic replay did not publish its assertion identity")
    require(int(assertions.group(1)) == 55 and int(assertions.group(2)) == 1, "synthetic replay coverage changed")

    sounding_paths = sorted(path for path in CM1_SOUNDINGS.glob("*") if path.is_file())
    require(len(sounding_paths) == 4, "expected the four supplied real CM1 soundings")
    evidence = {
        "schema": "nimbus.qualification.storm_history_replay",
        "schema_version": 1,
        "status": "partial",
        "synthetic_history": {
            "status": "pass",
            "path": str(SYNTHETIC_HISTORY.relative_to(ROOT)),
            "sha256": sha256(SYNTHETIC_HISTORY),
            "rows": 9,
            "assertions": 55,
            "scope": "continuity_plus_diagnostic_movement_authority",
        },
        "nimbus_history": validate_nimbus_history(args.history_report.resolve()),
        "cm1_history": {
            "status": "unavailable",
            "reason": "repository contains environmental CM1 soundings but no time-evolving CM1 history output",
            "soundings_present": [str(path.relative_to(ROOT)) for path in sounding_paths],
            "soundings_are_not_history_evidence": True,
        },
        "remaining_gate": "ingest and replay at least one real time-evolving CM1 history with declared scalar coordinates, times, winds, density, and condensate species",
    }
    output = args.evidence_out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "[storm-history-replay] PARTIAL: synthetic continuity/movement PASS; "
        f"Nimbus {evidence['nimbus_history']['samples']}-sample aggregate PASS; real CM1 history unavailable"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        raise SystemExit(f"[storm-history-replay] FAIL: {exc}") from exc
