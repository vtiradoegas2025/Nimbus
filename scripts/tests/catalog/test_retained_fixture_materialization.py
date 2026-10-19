#!/usr/bin/env python3
"""Gate exact-loader maintained YAML identity, sharding, and stale-input rejection."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / "tests/fixtures/reference/catalog/maintained_canonical_baseline_v1.json"
MATERIALIZE = ROOT / "scripts/qualification/materialize_retained_fixtures.py"
MERGE = ROOT / "scripts/qualification/merge_retained_fixture_shards.py"
AXIS_AUDIT = ROOT / "scripts/qualification/audit_loader_categorical_axes.py"


def execute(command: list[str], should_pass: bool) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    if (result.returncode == 0) != should_pass:
        raise RuntimeError(f"retained-fixture command had the wrong outcome: {result.stderr[-1200:]}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--native", type=Path, default=ROOT / "bin/nimbus_native_admission_batch")
    args = parser.parse_args()
    baseline = json.loads(BASELINE.read_text())
    if baseline["schema"] != "nimbus.maintained-canonical-baseline.v1":
        raise RuntimeError("maintained canonical baseline changed schema")
    binary = args.binary.resolve(strict=True)
    native = args.native.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="nimbus-retained-canonical-") as temporary:
        root = Path(temporary)

        def shard(index: int) -> Path:
            output = root / f"shard_{index}"
            execute([sys.executable, str(MATERIALIZE), "--binary", str(binary),
                     "--native", str(native), "--shard-index", str(index),
                     "--shard-count", "8", "--workers", "4", "--output", str(output)], True)
            return output / "summary.json"

        with ThreadPoolExecutor(max_workers=8) as pool:
            summaries = list(pool.map(shard, range(8)))

        def merge(selected: list[Path], name: str, selected_native: Path = native,
                  should_pass: bool = True) -> subprocess.CompletedProcess[str]:
            command = [sys.executable, str(MERGE), "--binary", str(binary),
                       "--native", str(selected_native), "--output", str(root / name)]
            for summary in selected:
                command.extend(("--shard-summary", str(summary)))
            return execute(command, should_pass)

        merge(summaries, "merged")
        result = json.loads((root / "merged/summary.json").read_text())
        execute([sys.executable, str(AXIS_AUDIT), "--canonical-summary",
                 str(root / "merged/summary.json"), "--output", str(root / "loader_axis_audit.json")], True)
        axes = json.loads((root / "loader_axis_audit.json").read_text())
        expected = {
            "sourceCount": baseline["sourceCount"],
            "uniqueEffectiveConfigs": baseline["uniqueEffectiveConfigs"],
            "duplicateEffectiveConfigGroups": baseline["duplicateEffectiveConfigGroups"],
        }
        if any(result[key] != value for key, value in expected.items()) or \
                len(result["familyCounts"]) != baseline["familyCount"] or \
                result["conditionalResolutionCounts"].get("conditional_auto_by_grid_staggering") != \
                baseline["conditionalAutoAdvectionResolutions"] or \
                result["activeProductSetPaths"] != baseline["activeProductSetPaths"] or \
                result["selectedRadarProductCounts"] != baseline["selectedRadarProductCounts"] or \
                result["physicalBackendCounts"] != baseline["physicalBackendCounts"] or \
                result["backendAliasCounts"] != baseline["backendAliasCounts"] or \
                result["records"]["uncompressedSha256"] != baseline["uncompressedRecordSha256"]:
            raise RuntimeError("maintained strict-loader canonical baseline drifted")
        if axes["axisCount"] != baseline["literalLoaderEnumAxes"] or \
                axes["missingEffectiveAxisCount"] != baseline["axesWithUnseenEffectiveValues"] or \
                axes["unrepresentedEffectiveAxisCount"] != baseline["axesWithUnrepresentedEffectiveChoices"] or \
                axes["axisRowsSha256"] != baseline["loaderAxisRowsSha256"]:
            raise RuntimeError("strict-loader categorical axis baseline drifted")
        missing = merge(summaries[:-1], "missing_shard", should_pass=False)
        stale = merge(summaries, "stale_native", selected_native=binary, should_pass=False)
        if "skipped or extra shard" not in missing.stderr or "stale binary or native materializer" not in stale.stderr:
            raise RuntimeError("maintained canonical mutation checks failed for the wrong reason")
        print(f"[retained-fixture-materialization] PASS: {result['sourceCount']} sources, "
              f"{result['uniqueEffectiveConfigs']} effective configs, 8 complete shards; "
              "skipped-shard and stale-native mutations rejected")


if __name__ == "__main__":
    main()
