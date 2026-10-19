#!/usr/bin/env python3
"""Paired exact-endpoint Vulkan binary A/B with robust speedup gating."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import tempfile

import benchmark_vulkan_report_overhead as benchmark


LEDGER_FILES = (
    "budget.csv",
    "energy_budget.csv",
    "moisture_budget.csv",
    "momentum_budget.csv",
    "radiation_budget.csv",
    "thermodynamic_source_budget.csv",
)


def require_same_ledgers(
    optimized: dict[str, object], comparator: dict[str, object]
) -> None:
    """Require byte-exact report-bearing scientific ledgers."""
    optimized_root = Path(str(optimized["final_frame"])).parent
    comparator_root = Path(str(comparator["final_frame"])).parent
    for name in LEDGER_FILES:
        optimized_path = optimized_root / name
        comparator_path = comparator_root / name
        benchmark.require(
            optimized_path.is_file() and comparator_path.is_file(),
            f"missing report-bearing ledger {name}",
        )
        benchmark.require(
            optimized_path.read_bytes() == comparator_path.read_bytes(),
            f"optimized/comparator report-bearing ledger differs: {name}",
        )


def evaluate_speedup_gate(
    optimized_report_ms: list[float],
    comparator_report_ms: list[float],
    minimum_speedup_percent: float,
) -> dict[str, object]:
    """Evaluate the paired whole-step speedup and robust-noise contract."""
    benchmark.require(
        len(optimized_report_ms) == len(comparator_report_ms),
        "optimized/comparator report sample counts differ",
    )
    benchmark.require(
        len(optimized_report_ms) >= 3,
        "speedup gate requires at least three paired samples",
    )
    benchmark.require(
        math.isfinite(minimum_speedup_percent)
        and minimum_speedup_percent >= 0.0,
        "minimum speedup must be finite and nonnegative",
    )
    benchmark.require(
        all(
            math.isfinite(value) and value > 0.0
            for value in optimized_report_ms + comparator_report_ms
        ),
        "speedup samples must be positive and finite",
    )

    improvement_samples_ms = [
        comparator - optimized
        for optimized, comparator in zip(
            optimized_report_ms, comparator_report_ms
        )
    ]
    improvement = benchmark.robust_statistics(improvement_samples_ms)
    comparator_median = float(
        benchmark.robust_statistics(comparator_report_ms)["median"]
    )
    median_improvement_ms = float(improvement["median"])
    median_speedup_percent = (
        100.0 * median_improvement_ms / comparator_median
    )
    minimum_improvement_ms = (
        comparator_median * minimum_speedup_percent / 100.0
    )
    passes_speedup = (
        median_improvement_ms > minimum_improvement_ms
        and median_improvement_ms
        > float(improvement["three_robust_sigma"])
    )
    return {
        "minimum_speedup_percent": minimum_speedup_percent,
        "minimum_improvement_ms": minimum_improvement_ms,
        "median_improvement_ms": median_improvement_ms,
        "median_speedup_percent": median_speedup_percent,
        "three_robust_sigma_ms": improvement["three_robust_sigma"],
        "passes_speedup_and_noise": passes_speedup,
        "improvement_statistics_ms": improvement,
    }


def evaluate_attribution_overhead_gate(
    optimized_free_ms: list[float],
    optimized_report_ms: list[float],
    comparator_free_ms: list[float],
    comparator_report_ms: list[float],
    minimum_speedup_percent: float,
) -> dict[str, object]:
    """Evaluate report-overhead improvement after paired free-run control."""
    sample_count = len(optimized_free_ms)
    benchmark.require(
        sample_count == len(optimized_report_ms)
        == len(comparator_free_ms) == len(comparator_report_ms),
        "attribution-overhead sample counts differ",
    )
    benchmark.require(
        sample_count >= 3,
        "attribution-overhead gate requires at least three paired samples",
    )
    benchmark.require(
        math.isfinite(minimum_speedup_percent)
        and minimum_speedup_percent >= 0.0,
        "minimum speedup must be finite and nonnegative",
    )
    benchmark.require(
        all(
            math.isfinite(value) and value > 0.0
            for value in (
                optimized_free_ms + optimized_report_ms
                + comparator_free_ms + comparator_report_ms
            )
        ),
        "attribution-overhead timing samples must be positive and finite",
    )

    optimized_overhead_ms = [
        report - free
        for free, report in zip(optimized_free_ms, optimized_report_ms)
    ]
    comparator_overhead_ms = [
        report - free
        for free, report in zip(comparator_free_ms, comparator_report_ms)
    ]
    improvement_samples_ms = [
        comparator - optimized
        for optimized, comparator in zip(
            optimized_overhead_ms, comparator_overhead_ms
        )
    ]
    improvement = benchmark.robust_statistics(improvement_samples_ms)
    comparator_overhead = benchmark.robust_statistics(
        comparator_overhead_ms
    )
    comparator_median_ms = float(comparator_overhead["median"])
    benchmark.require(
        comparator_median_ms > 0.0,
        "comparator median report overhead must be positive",
    )
    median_improvement_ms = float(improvement["median"])
    median_speedup_percent = (
        100.0 * median_improvement_ms / comparator_median_ms
    )
    minimum_improvement_ms = (
        comparator_median_ms * minimum_speedup_percent / 100.0
    )
    passes_speedup = (
        median_improvement_ms > minimum_improvement_ms
        and median_improvement_ms
        > float(improvement["three_robust_sigma"])
    )
    return {
        "minimum_speedup_percent": minimum_speedup_percent,
        "minimum_improvement_ms": minimum_improvement_ms,
        "median_improvement_ms": median_improvement_ms,
        "median_speedup_percent": median_speedup_percent,
        "three_robust_sigma_ms": improvement["three_robust_sigma"],
        "passes_speedup_and_noise": passes_speedup,
        "optimized_overhead_statistics_ms":
            benchmark.robust_statistics(optimized_overhead_ms),
        "comparator_overhead_statistics_ms": comparator_overhead,
        "improvement_statistics_ms": improvement,
    }


def passes_gate_policy(
    policy: str,
    whole_step_gate: dict[str, object],
    overhead_gate: dict[str, object],
) -> bool:
    """Apply the acceptance policy appropriate to the scoped experiment."""
    benchmark.require(
        policy in ("report-attribution", "whole-step"),
        f"unsupported benchmark gate policy: {policy}",
    )
    whole_step_passes = bool(
        whole_step_gate["passes_speedup_and_noise"]
    )
    if policy == "whole-step":
        return whole_step_passes
    return whole_step_passes and bool(
        overhead_gate["passes_speedup_and_noise"]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--comparator", type=Path, required=True)
    parser.add_argument(
        "--report-free-config",
        type=Path,
        default=Path("tests/fixtures/configs/qualification/gpu_report_free.yaml"),
    )
    parser.add_argument(
        "--report-bearing-config",
        type=Path,
        default=Path("tests/fixtures/configs/qualification/gpu_report_bearing.yaml"),
    )
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--acoustic-substeps", type=int, default=2)
    parser.add_argument("--dt-large", type=float, default=0.25)
    parser.add_argument("--minimum-speedup-percent", type=float, default=2.0)
    parser.add_argument(
        "--gate-policy",
        choices=("report-attribution", "whole-step"),
        default="report-attribution",
        help=(
            "report-attribution requires both raw whole-step and paired "
            "report-overhead wins; whole-step is for candidates that also "
            "execute when reports are disabled"
        ),
    )
    parser.add_argument(
        "--label",
        default="vulkan-scalar-attribution",
    )
    parser.add_argument(
        "--schema",
        default="nimbus.vulkan-scalar-attribution-prepass.v2",
    )
    parser.add_argument(
        "--comparison-contract",
        default=(
            "same equations, FP32 device commit, FP64 coefficients, "
            "fixed-slab report reduction, cadence, ledgers, and endpoint; "
            "comparator adds only the redundant read-only coefficient "
            "validation traversal"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/benchmarks/"
            "vulkan-scalar-attribution-prepass-current.json"
        ),
    )
    args = parser.parse_args()

    benchmark.require(args.optimized.is_file(), "missing optimized binary")
    benchmark.require(args.comparator.is_file(), "missing comparator binary")
    benchmark.require(
        args.optimized.resolve() != args.comparator.resolve(),
        "optimized and comparator binaries must differ",
    )
    benchmark.require_config_pair_contract(
        args.report_free_config, args.report_bearing_config)
    benchmark.require(args.duration > 0, "duration must be positive")
    benchmark.require(
        args.trials >= 3 and args.trials % 2 == 1,
        "trials must be an odd integer of at least three",
    )
    benchmark.require(
        args.acoustic_substeps > 0,
        "acoustic substeps must be positive",
    )
    benchmark.require(
        math.isfinite(args.dt_large) and args.dt_large > 0.0,
        "dt-large must be positive and finite",
    )
    benchmark.require(
        math.isfinite(args.minimum_speedup_percent)
        and args.minimum_speedup_percent >= 0.0,
        "minimum speedup must be finite and nonnegative",
    )

    binaries = {
        "optimized": args.optimized,
        "comparator": args.comparator,
    }
    trials: dict[str, dict[str, list[dict[str, object]]]] = {
        label: {"free": [], "report": []} for label in binaries
    }

    with tempfile.TemporaryDirectory(
        prefix="nimbus_vulkan_scalar_attribution_"
    ) as root_text:
        root = Path(root_text)
        for trial_index in range(args.trials):
            binary_order = (
                ("optimized", "comparator")
                if trial_index % 2 == 0 else
                ("comparator", "optimized")
            )
            paired: dict[str, dict[str, dict[str, object]]] = {}
            for binary_index, binary_label in enumerate(binary_order):
                mode_order = (
                    ("free", "report")
                    if (trial_index + binary_index) % 2 == 0 else
                    ("report", "free")
                )
                paired[binary_label] = {}
                for mode in mode_order:
                    report_bearing = mode == "report"
                    label = (
                        f"trial-{trial_index + 1}-"
                        f"{binary_label}-{mode}"
                    )
                    result = benchmark.run_trial(
                        binaries[binary_label],
                        (
                            args.report_bearing_config
                            if report_bearing else
                            args.report_free_config
                        ),
                        root / label,
                        args.duration,
                        label,
                        report_bearing,
                        args.acoustic_substeps,
                        args.dt_large,
                    )
                    paired[binary_label][mode] = result
                    trials[binary_label][mode].append(result)
                benchmark.require_same_endpoint(
                    paired[binary_label]["free"],
                    paired[binary_label]["report"],
                )

            for mode in ("free", "report"):
                benchmark.require_same_endpoint(
                    paired["optimized"][mode],
                    paired["comparator"][mode],
                )
            require_same_ledgers(
                paired["optimized"]["report"],
                paired["comparator"]["report"],
            )
            print(
                f"[{args.label}] "
                f"trial={trial_index + 1}/{args.trials} "
                f"optimized_report="
                f"{float(paired['optimized']['report']['avg_step_ms']):.6f} "
                "ms comparator_report="
                f"{float(paired['comparator']['report']['avg_step_ms']):.6f} "
                "ms"
            )

        summaries = {
            label: benchmark.summarize(
                trials[label]["free"], trials[label]["report"])
            for label in binaries
        }
        timing_samples = {
            label: {
                mode: [
                    float(trial["avg_step_ms"])
                    for trial in trials[label][mode]
                ]
                for mode in ("free", "report")
            }
            for label in binaries
        }
        whole_step_gate = evaluate_speedup_gate(
            timing_samples["optimized"]["report"],
            timing_samples["comparator"]["report"],
            args.minimum_speedup_percent,
        )
        overhead_gate = evaluate_attribution_overhead_gate(
            timing_samples["optimized"]["free"],
            timing_samples["optimized"]["report"],
            timing_samples["comparator"]["free"],
            timing_samples["comparator"]["report"],
            args.minimum_speedup_percent,
        )
        passes_speedup = passes_gate_policy(
            args.gate_policy,
            whole_step_gate,
            overhead_gate,
        )

        device_evidence = {
            str(trial["selected_device_evidence"])
            for binary_trials in trials.values()
            for mode_trials in binary_trials.values()
            for trial in mode_trials
        }
        benchmark.require(
            len(device_evidence) == 1,
            "physical-device identity changed across trials",
        )
        report = {
            "schema": args.schema,
            "status": "accepted" if passes_speedup else "rejected",
            "generated_at_utc":
                dt.datetime.now(dt.timezone.utc).isoformat(),
            "physical_device_required": True,
            "selected_device_evidence": next(iter(device_evidence)),
            "comparison_contract": args.comparison_contract,
            "identity": {
                "optimized_binary": str(args.optimized.resolve()),
                "optimized_binary_sha256":
                    benchmark.sha256_file(args.optimized),
                "comparator_binary": str(args.comparator.resolve()),
                "comparator_binary_sha256":
                    benchmark.sha256_file(args.comparator),
                "report_free_config":
                    str(args.report_free_config.resolve()),
                "report_free_config_sha256":
                    benchmark.sha256_file(args.report_free_config),
                "report_bearing_config":
                    str(args.report_bearing_config.resolve()),
                "report_bearing_config_sha256":
                    benchmark.sha256_file(args.report_bearing_config),
            },
            "duration_s": args.duration,
            "trials": args.trials,
            "runtime_overrides": {
                "acoustic_substeps": args.acoustic_substeps,
                "dt_large_s": args.dt_large,
            },
            "endpoint_fields_bitwise_equal":
                list(benchmark.CORE_FIELDS),
            "ledger_files_bitwise_equal": list(LEDGER_FILES),
            "gate": {
                "policy": args.gate_policy,
                "passes_required_gates": passes_speedup,
                "passes_whole_step_and_attribution_overhead":
                    bool(whole_step_gate["passes_speedup_and_noise"])
                    and bool(overhead_gate["passes_speedup_and_noise"]),
                "whole_step": {
                    key: value
                    for key, value in whole_step_gate.items()
                    if key != "improvement_statistics_ms"
                },
                "attribution_overhead": {
                    key: value
                    for key, value in overhead_gate.items()
                    if not key.endswith("_statistics_ms")
                },
            },
            "whole_step_improvement_statistics_ms":
                whole_step_gate["improvement_statistics_ms"],
            "attribution_overhead_statistics_ms": {
                key: value
                for key, value in overhead_gate.items()
                if key.endswith("_statistics_ms")
            },
            "optimized_summary": summaries["optimized"],
            "comparator_summary": summaries["comparator"],
            "optimized_trials": {
                mode: [
                    benchmark.serializable_trial(trial)
                    for trial in mode_trials
                ]
                for mode, mode_trials in trials["optimized"].items()
            },
            "comparator_trials": {
                mode: [
                    benchmark.serializable_trial(trial)
                    for trial in mode_trials
                ]
                for mode, mode_trials in trials["comparator"].items()
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(
        f"[{args.label}] RESULT: "
        f"{'ACCEPTED' if passes_speedup else 'REJECTED'} "
        f"gate_policy={args.gate_policy} "
        f"whole_step_speedup_percent="
        f"{float(whole_step_gate['median_speedup_percent']):.6f} "
        f"overhead_speedup_percent="
        f"{float(overhead_gate['median_speedup_percent']):.6f} "
        f"overhead_improvement_ms="
        f"{float(overhead_gate['median_improvement_ms']):.6f} "
        f"overhead_three_robust_sigma_ms="
        f"{float(overhead_gate['three_robust_sigma_ms']):.6f} "
        f"report={args.output}"
    )
    return 0 if passes_speedup else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"[vulkan-scalar-attribution] RESULT: FAIL: {error}")
        raise SystemExit(1)
