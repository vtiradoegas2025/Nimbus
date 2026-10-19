#!/usr/bin/env python3
"""Compare a Nimbus CPU benchmark report with a versioned baseline policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence


REPORT_SCHEMA = "nimbus.cpu-benchmark.v1"
POLICY_SCHEMA = "nimbus.cpu-benchmark-policy.v1"
RESULT_SCHEMA = "nimbus.cpu-benchmark-comparison.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def nested_value(value: dict[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"report is missing identity field {dotted_path!r}")
        current = current[part]
    return current


def median_absolute_deviation(values: Sequence[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def validate_report(report: dict[str, Any], label: str) -> None:
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError(f"{label} does not use {REPORT_SCHEMA}")
    trials = report.get("trials")
    run = report.get("run")
    if not isinstance(trials, list) or not trials:
        raise ValueError(f"{label} has no trials")
    if not isinstance(run, dict):
        raise ValueError(f"{label} has no run metadata")
    if run.get("trials") != len(trials):
        raise ValueError(f"{label} trial count does not match its metadata")


def aggregate(values: Sequence[float], name: str) -> float:
    if name == "median":
        return statistics.median(values)
    if name == "max":
        return max(values)
    raise ValueError(f"unsupported aggregate {name!r}")


def compare_reports(
    baseline: dict[str, Any], current: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    validate_report(baseline, "baseline")
    validate_report(current, "current report")
    if policy.get("schema") != POLICY_SCHEMA:
        raise ValueError(f"policy does not use {POLICY_SCHEMA}")

    minimum_trials = int(policy.get("minimum_trials", 0))
    minimum_steps = int(policy.get("minimum_measured_steps", 0))
    if minimum_trials < 2 or minimum_steps < 1:
        raise ValueError("policy minimums must require at least two trials and one step")
    for label, report in (("baseline", baseline), ("current report", current)):
        if len(report["trials"]) < minimum_trials:
            raise ValueError(f"{label} has fewer than {minimum_trials} trials")
        if int(report["run"].get("measured_steps", 0)) < minimum_steps:
            raise ValueError(f"{label} has fewer than {minimum_steps} measured steps")

    identity_fields = policy.get("identity_fields")
    if not isinstance(identity_fields, list) or not identity_fields:
        raise ValueError("policy must declare identity_fields")
    mismatches = []
    for field in identity_fields:
        if not isinstance(field, str):
            raise ValueError("identity field names must be strings")
        expected = nested_value(baseline, field)
        observed = nested_value(current, field)
        if expected != observed:
            mismatches.append(
                {"field": field, "baseline": expected, "current": observed}
            )

    comparisons = []
    metrics = policy.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("policy must declare metrics")
    for metric, rule in metrics.items():
        if not isinstance(rule, dict):
            raise ValueError(f"metric policy {metric!r} must be an object")
        try:
            baseline_values = [float(trial[metric]) for trial in baseline["trials"]]
            current_values = [float(trial[metric]) for trial in current["trials"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"metric {metric!r} is missing or non-numeric") from exc
        if not all(math.isfinite(value) for value in baseline_values + current_values):
            raise ValueError(f"metric {metric!r} contains a non-finite value")

        aggregate_name = str(rule.get("aggregate", "median"))
        baseline_value = aggregate(baseline_values, aggregate_name)
        current_value = aggregate(current_values, aggregate_name)
        relative_allowance = abs(baseline_value) * float(
            rule.get("max_relative_regression", 0.0)
        )
        absolute_allowance = float(rule.get("absolute_allowance", 0.0))
        # 1.4826 scales normal-distribution MAD to a robust sigma estimate.
        noise_allowance = (
            float(rule.get("baseline_mad_multiplier", 0.0))
            * 1.4826
            * median_absolute_deviation(baseline_values)
        )
        allowed_regression = max(
            relative_allowance, absolute_allowance, noise_allowance
        )

        direction = rule.get("direction")
        if direction == "lower_is_better":
            observed_regression = current_value - baseline_value
        elif direction == "higher_is_better":
            observed_regression = baseline_value - current_value
        else:
            raise ValueError(f"metric {metric!r} has invalid direction {direction!r}")
        passed = observed_regression <= allowed_regression
        comparisons.append(
            {
                "metric": metric,
                "aggregate": aggregate_name,
                "direction": direction,
                "baseline": baseline_value,
                "current": current_value,
                "observed_regression": observed_regression,
                "allowed_regression": allowed_regression,
                "relative_allowance": relative_allowance,
                "absolute_allowance": absolute_allowance,
                "baseline_noise_allowance": noise_allowance,
                "passed": passed,
            }
        )

    status = "pass"
    if mismatches:
        status = "incompatible"
    elif any(not comparison["passed"] for comparison in comparisons):
        status = "regression"
    return {
        "schema": RESULT_SCHEMA,
        "policy_id": policy.get("policy_id"),
        "status": status,
        "identity_mismatches": mismatches,
        "comparisons": comparisons,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        baseline = load_object(args.baseline)
        current = load_object(args.current)
        policy = load_object(args.policy)
        result = compare_reports(baseline, current, policy)
    except ValueError as exc:
        print(f"[benchmark-policy] {exc}", file=sys.stderr)
        return 2
    result["baseline"] = {
        "path": str(args.baseline), "sha256": sha256_file(args.baseline)
    }
    result["current"] = {
        "path": str(args.current), "sha256": sha256_file(args.current)
    }
    result["policy"] = {
        "path": str(args.policy), "sha256": sha256_file(args.policy)
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
