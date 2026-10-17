#!/usr/bin/env python3
"""Seal and attribute macro-gated Milbrandt process timers from bounded trials."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


PROCESS_LINE = re.compile(r"\[MILBRANDT PROCESS PERF\] (?P<values>[^\n]+)")
PHASES = (
    "prepare",
    "saturation",
    "activation",
    "warm",
    "warm_commit",
    "cold",
    "melting",
    "sedimentation",
    "finalize",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_process_samples(log: str) -> list[dict[str, float | int]]:
    samples = []
    expected = {"call", *(f"{phase}_s" for phase in PHASES)}
    for match in PROCESS_LINE.finditer(log):
        fields = {}
        for raw_field in match.group("values").split(","):
            name, separator, value = raw_field.strip().partition("=")
            if separator != "=" or name in fields:
                raise ValueError("malformed or duplicate Milbrandt process field")
            fields[name] = value
        if set(fields) != expected:
            raise ValueError("Milbrandt process field set changed")
        call = int(fields["call"])
        if call != len(samples):
            raise ValueError("Milbrandt process call sequence is incomplete")
        sample: dict[str, float | int] = {"call": call}
        for phase in PHASES:
            value = float(fields[f"{phase}_s"])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid Milbrandt {phase} timer")
            sample[phase] = value
        samples.append(sample)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    profile_path = args.profile.resolve(strict=True)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("schema") != "nimbus.c6-current-milbrandt-stage-profile.v1":
        parser.error("unexpected stage-profile schema")
    if len(profile.get("runs", [])) < 7:
        parser.error("at least seven trials are required")
    trials = []
    for run in profile["runs"]:
        log_path = Path(run["log"])
        if sha256(log_path) != run["log_sha256"]:
            raise ValueError(f"Milbrandt raw log changed: {log_path}")
        samples = parse_process_samples(log_path.read_text(encoding="utf-8"))
        warmup = run["warmup_steps"]
        measured = run["measured_steps"]
        if len(samples) != warmup + measured:
            raise ValueError(
                f"trial {run['trial']} has {len(samples)} process calls, expected {warmup + measured}"
            )
        measured_samples = samples[warmup:]
        phase_totals = {
            phase: sum(float(sample[phase]) for sample in measured_samples)
            for phase in PHASES
        }
        scheme_s = run["microphysics_subphases_s"]["scheme"]
        covered_s = sum(phase_totals.values())
        coverage = covered_s / scheme_s
        if not 0.95 <= coverage <= 1.05:
            raise ValueError(f"trial {run['trial']} scheme-process coverage is {coverage:.6f}")
        trials.append({
            "trial": run["trial"],
            "raw_log_sha256": run["log_sha256"],
            "measured_calls": measured,
            "scheme_s": scheme_s,
            "process_covered_s": covered_s,
            "scheme_process_coverage": coverage,
            "phase_totals_s": phase_totals,
            "phase_fraction_of_scheme": {
                phase: phase_totals[phase] / scheme_s for phase in PHASES
            },
            "phase_fraction_of_step": {
                phase: phase_totals[phase] / run["step_wall_s"] for phase in PHASES
            },
        })
    report = {
        "schema": "nimbus.c6-milbrandt-process-attribution.v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage_profile": str(profile_path),
        "stage_profile_sha256": sha256(profile_path),
        "analyzer_sha256": sha256(Path(__file__)),
        "binary_sha256": profile["source"]["binary_sha256"],
        "config_sha256": profile["source"]["config_sha256"],
        "endpoint": profile["endpoint"],
        "trials": trials,
        "summary": {
            "minimum_scheme_process_coverage": min(
                trial["scheme_process_coverage"] for trial in trials
            ),
            "median_phase_fraction_of_scheme": {
                phase: statistics.median(
                    trial["phase_fraction_of_scheme"][phase] for trial in trials
                ) for phase in PHASES
            },
            "median_phase_fraction_of_step": {
                phase: statistics.median(
                    trial["phase_fraction_of_step"][phase] for trial in trials
                ) for phase in PHASES
            },
            "maximum_phase_fraction_of_step": {
                phase: max(
                    trial["phase_fraction_of_step"][phase] for trial in trials
                ) for phase in PHASES
            },
        },
        "candidateDisposition": {
            "candidate": "pure removal of the warm-commit traversal",
            "status": "rejected_without_implementation",
            "maximum_entire_pass_removal_percent": 100.0 * max(
                trial["phase_fraction_of_step"]["warm_commit"] for trial in trials
            ),
            "required_time_gain_percent": 2.0,
            "reason": "Even removing the complete measured traversal cannot reach the time gate; the pass also materializes the warm-before-cold donor state.",
        },
        "disposition": "process attribution and bounded pass-removal rejection; no optimization accepted",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[c6-milbrandt] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
