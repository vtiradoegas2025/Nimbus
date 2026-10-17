#!/usr/bin/env python3
"""Validate C6's source-bound draft policy without promoting release readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "nimbus.c6-release-threshold-candidate.v1"
FIELDS = {"p", "rho", "theta", "u", "v", "w", "qv", "qr"}
LEDGERS = {
    "budget.csv",
    "energy_budget.csv",
    "moisture_budget.csv",
    "momentum_budget.csv",
    "radiation_budget.csv",
    "thermodynamic_source_budget.csv",
}
REQUIRED_WORKLOADS = {
    "cpu_milbrandt_24x24x20",
    "cpu_active_32x32x24",
    "cpu_student_64x64x48",
    "cpu_workstation_128x128x64",
    "metal_npy_18x18x18",
    "metal_netcdf_18x18x18",
    "vulkan_direct_12x12x48",
    "vulkan_staged_12x12x48",
    "zfp_history_64x64x48",
    "retained_terrain_postimplementation",
}


class C6PolicyError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise C6PolicyError(message)


def finite_positive(value: object) -> bool:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(converted) and converted > 0.0


def close(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=1.0e-10, abs_tol=1.0e-10)


def validate_release_binary(policy: dict, binary: Path) -> str:
    expected = policy["releaseBinary"]["sha256"]
    observed = sha256(binary)
    require(observed == expected, f"stale release binary: expected {expected}, found {observed}")
    return observed


def qualifies_memory_gain(
    full_field_bytes: int,
    admitted_peak_bytes: int,
    live_full_fields_removed: int,
    peak_reduction_bytes: int,
    cost_shifted_outside_measurement: bool,
) -> bool:
    require(full_field_bytes > 0 and admitted_peak_bytes > 0, "invalid memory envelope")
    require(live_full_fields_removed >= 0 and peak_reduction_bytes >= 0, "negative memory saving")
    return (
        not cost_shifted_outside_measurement
        and (
            live_full_fields_removed >= 1
            or peak_reduction_bytes >= math.ceil(0.02 * admitted_peak_bytes)
        )
    )


def validate_paired_ab(report: dict, observation: dict, acceptance: dict) -> dict:
    require(report.get("schema") == "nimbus.c6-milbrandt-gamma-ab.v1", "wrong A/B schema")
    require(report.get("status") == "complete", "incomplete A/B report")
    source = report.get("source", {})
    require(source.get("baseline_sha256") == observation["baselineBinarySha256"], "stale A/B baseline binary")
    require(source.get("candidate_sha256") == observation["candidateBinarySha256"], "stale A/B candidate binary")
    contract = report.get("contract", {})
    pairs = report.get("pairs", [])
    minimum_trials = acceptance["minimumSteadyStateTrials"]
    require(isinstance(pairs, list) and len(pairs) >= minimum_trials, "fewer than seven paired trials")
    require(contract.get("trials") == len(pairs), "A/B trial count disagrees with report")
    require(contract.get("warmup_steps", 0) >= 1 and contract.get("measured_steps", 0) >= 1,
            "missing steady-state measurement window")
    require(contract.get("field_count") == len(FIELDS) and contract.get("ledger_count") == len(LEDGERS),
            "A/B field or ledger contract changed")

    observed_endpoints = set()
    observed_fields = set()
    observed_ledgers = set()
    speedups = []
    named_coverages = []
    for index, pair in enumerate(pairs, start=1):
        require(pair.get("pair") == index, "A/B pair sequence is incomplete")
        require(pair.get("exact_state_fields_and_ledgers") is True, "missing exact-science identity")
        baseline = pair.get("baseline", {})
        candidate = pair.get("candidate", {})
        require(baseline.get("binary_sha256") == source["baseline_sha256"], "A/B baseline run binary drifted")
        require(candidate.get("binary_sha256") == source["candidate_sha256"], "A/B candidate run binary drifted")
        for run in (baseline, candidate):
            coverage = run.get("named_wall_fraction")
            require(finite_positive(coverage) and float(coverage) >= acceptance["minimumNamedWallFraction"],
                    "less than 95% named wall coverage")
            require(finite_positive(run.get("avg_step_ms")) and finite_positive(run.get("step_wall_s")),
                    "invalid A/B step time")
            require(isinstance(run.get("checkpoint_sha256"), str) and run.get("checkpoint_bytes", 0) > 0,
                    "missing exact checkpoint")
            require(set(run.get("field_sha256", {})) == FIELDS, "missing exact endpoint fields")
            require(set(run.get("ledger_sha256", {})) == LEDGERS, "missing exact science ledgers")
            require(isinstance(run.get("validation_sha256"), str), "missing validation artifact")
            observed_endpoints.add((run["checkpoint_sha256"], run["checkpoint_bytes"]))
            observed_fields.add(tuple(sorted(run["field_sha256"].items())))
            observed_ledgers.add(tuple(sorted(run["ledger_sha256"].items())))
            named_coverages.append(float(coverage))
        gain = 100.0 * (baseline["avg_step_ms"] - candidate["avg_step_ms"]) / baseline["avg_step_ms"]
        require(close(gain, float(pair.get("speedup_percent", math.nan))), "A/B paired gain is inconsistent")
        speedups.append(gain)
    require(len(observed_endpoints) == len(observed_fields) == len(observed_ledgers) == 1,
            "A/B exact science changed across trial pairs")
    median = statistics.median(speedups)
    mad = statistics.median(abs(value - median) for value in speedups)
    minimum_gain = acceptance["minimumWholeStepGainPercentExclusive"]
    mad_multiple = acceptance["minimumPairedMadMultipleExclusive"]
    passes = median > minimum_gain and median > mad_multiple * mad
    aggregate = report.get("aggregate", {})
    require(close(median, float(aggregate.get("median_speedup_percent", math.nan))), "A/B median is inconsistent")
    require(close(mad, float(aggregate.get("paired_mad_percent", math.nan))), "A/B MAD is inconsistent")
    require(aggregate.get("time_gate_passed") is passes, "A/B time gate contradicts samples")
    require(report.get("disposition") == ("time_gain_accepted" if passes else "time_gain_rejected"),
            "A/B disposition violates >2% and >3-MAD gates")
    require(observation["disposition"] == report["disposition"], "policy disposition disagrees with A/B evidence")
    return {"pairs": len(pairs), "medianGainPercent": median, "threeMadPercent": 3.0 * mad,
            "minimumNamedWallFraction": min(named_coverages), "timeGainAccepted": passes}


def validate_attribution(report: dict, observation: dict, acceptance: dict) -> None:
    if observation["id"] == "cpu_milbrandt_production_stage_8cc":
        require(report.get("schema") == "nimbus.c6-current-milbrandt-stage-profile.v1", "wrong stage schema")
        require(report.get("source", {}).get("binary_sha256") == observation["boundBinarySha256"],
                "stage profile binary drifted")
        runs = report.get("runs", [])
        require(len(runs) >= acceptance["minimumSteadyStateTrials"], "too few stage trials")
        require(len({run.get("checkpoint_sha256") for run in runs}) == 1, "stage checkpoints differ")
        require(all(float(run.get("named_wall_fraction", 0)) >= acceptance["minimumNamedWallFraction"] for run in runs),
                "stage named wall coverage failed")
    else:
        require(report.get("schema") == "nimbus.c6-milbrandt-process-attribution.v1", "wrong process schema")
        require(report.get("binary_sha256") == observation["boundBinarySha256"], "process binary drifted")
        trials = report.get("trials", [])
        require(len(trials) >= acceptance["minimumSteadyStateTrials"], "too few process trials")
        require(all(trial.get("scheme_process_coverage", 0) >= 0.95 for trial in trials),
                "process attribution coverage failed")


def validate_historical(report: dict, observation: dict) -> None:
    identifier = observation["id"]
    if identifier.startswith("historical_sci175_"):
        require(report.get("run", {}).get("trials", 0) >= 7, "historical allocation has too few trials")
        require(report.get("endpoint", {}).get("all_trials_bitwise_identical") is True,
                "historical allocation endpoint is not exact")
    elif identifier == "historical_sci177_smagorinsky":
        require(report.get("aggregate", {}).get("speedup_gate_passed") is True,
                "historical accepted timing gate no longer passes")
        require(report.get("endpoint", {}).get("all_trials_bitwise_identical") is True,
                "historical accepted endpoint is not exact")
    elif identifier == "historical_sci180_cpu_student_frontier":
        require(report.get("run", {}).get("trials_per_thread_count", 0) >= 7,
                "historical CPU frontier has too few trials")
    elif identifier in {
        "historical_sci240_report_attribution",
        "historical_sci241_endpoint_moisture",
        "historical_sci242_microphysics_gate",
    }:
        require(report.get("status") == "rejected" and report.get("gate", {}).get("passes_required_gates") is False,
                "historically rejected timing result was promoted")
    elif identifier == "historical_sci252_codec_workspace":
        require(report.get("status") == "accepted_lifecycle_only" and
                report.get("gate", {}).get("throughput_promoted") is False,
                "historical codec throughput was promoted")
    else:
        raise C6PolicyError(f"unknown historical observation {identifier}")


def validate_workloads(workloads: list[dict], root: Path) -> list[str]:
    ids = [row.get("id") for row in workloads]
    require(len(ids) == len(set(ids)) and set(ids) == REQUIRED_WORKLOADS,
            "release workload classes are missing, duplicated, or unreviewed")
    missing = []
    for row in workloads:
        grid = row.get("grid")
        if grid is not None:
            require(len(grid) == 3 and all(isinstance(value, int) and value > 0 for value in grid),
                    f"invalid grid for {row['id']}")
            require(row.get("fullFieldBytes") == math.prod(grid) * 4,
                    f"incorrect full-field memory equivalent for {row['id']}")
        else:
            require(row["id"] == "retained_terrain_postimplementation" and row.get("fullFieldBytes") is None,
                    "terrain grid cannot be inferred before retained scope closes")
        config = row.get("config")
        if config is not None:
            require((root / config).is_file(), f"missing workload config: {config}")
        require(isinstance(row.get("missing"), list) and row["missing"], f"missing C6 task list: {row['id']}")
        if row["missing"] or row.get("receipt") is None or row.get("thresholds") is None:
            missing.append(row["id"])
    return missing


def validate_policy(policy: dict, root: Path, binary: Path) -> dict:
    require(policy.get("schema") == SCHEMA, "wrong C6 policy schema")
    require(policy.get("status") == "draft_not_activated", "C6 draft was promoted without review")
    binary_sha = validate_release_binary(policy, binary)
    acceptance = policy.get("acceptance", {})
    require(acceptance.get("minimumSteadyStateTrials") == 7, "seven-trial rule was weakened")
    require(acceptance.get("minimumNamedWallFraction") == 0.95, "named-wall rule was weakened")
    require(acceptance.get("minimumWholeStepGainPercentExclusive") == 2.0, "whole-step gain rule was weakened")
    require(acceptance.get("minimumPairedMadMultipleExclusive") == 3.0, "MAD rule was weakened")
    require(acceptance.get("requireExactCheckpoint") is True, "checkpoint identity rule was removed")
    require(set(acceptance.get("requiredExactFields", [])) == FIELDS, "exact field contract changed")
    require(set(acceptance.get("requiredExactLedgers", [])) == LEDGERS, "exact ledger contract changed")
    require(acceptance.get("memoryAlternative") == "at_least_one_live_full_field_or_two_percent_of_admitted_peak",
            "peak-memory gate changed")
    seen = set()
    paired = None
    for row in policy.get("observations", []):
        identifier = row.get("id")
        require(identifier not in seen, "duplicate C6 observation")
        seen.add(identifier)
        path = root / row["path"]
        require(path.is_file() and sha256(path) == row["sha256"], f"stale C6 evidence: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        kind = row.get("kind")
        if kind == "paired_ab":
            paired = validate_paired_ab(report, row, acceptance)
        elif kind == "attribution_only":
            validate_attribution(report, row, acceptance)
        elif kind == "historical":
            validate_historical(report, row)
        else:
            raise C6PolicyError(f"unknown observation kind: {kind}")
    require(paired is not None and len(seen) >= 11, "required C6 evidence classes are missing")
    missing = validate_workloads(policy.get("workloadClasses", []), root)
    return {
        "status": "validated_draft_not_release_ready",
        "releaseBinarySha256": binary_sha,
        "observationCount": len(seen),
        "pairedAb": paired,
        "missingWorkloadClasses": missing,
        "releaseReady": not missing and policy["status"] != "draft_not_activated",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=ROOT / "tests/fixtures/reference/performance/threshold_policy_v1.json")
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--require-release-ready", action="store_true")
    args = parser.parse_args()
    try:
        result = validate_policy(json.loads(args.policy.read_text(encoding="utf-8")), ROOT, args.binary)
        if args.require_release_ready:
            require(result["releaseReady"], "C6 release thresholds are not activated")
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"C6 threshold candidate rejected: {error}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
