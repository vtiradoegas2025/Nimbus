#!/usr/bin/env python3
"""Exact-endpoint A/B for host versus Vulkan SGS state commit."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile

import benchmark_vulkan_report_overhead as benchmark
from benchmark_vulkan_scalar_attribution import evaluate_speedup_gate


LEDGERS = (
    "budget.csv",
    "energy_budget.csv",
    "momentum_budget.csv",
    "moisture_budget.csv",
    "radiation_budget.csv",
    "thermodynamic_source_budget.csv",
)
SGS_APPLY_STAGE = "turbulence/sgs_apply.comp.spv"
CONFIG_MARKER = "    gpu_apply_sgs: true"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def median(values: list[float | int]) -> float:
    return float(statistics.median(values))


def require_same_ledgers(
    host_trial: dict[str, object],
    device_trial: dict[str, object],
) -> None:
    host_root = Path(str(host_trial["final_frame"])).parent
    device_root = Path(str(device_trial["final_frame"])).parent
    for name in LEDGERS:
        host_path = host_root / name
        device_path = device_root / name
        benchmark.require(host_path.is_file(), f"host commit missing {name}")
        benchmark.require(
            device_path.is_file(), f"device commit missing {name}")
        if host_path.read_bytes() == device_path.read_bytes():
            continue
        with host_path.open("r", encoding="utf-8", newline="") as handle:
            host_reader = csv.DictReader(handle)
            host_fields = host_reader.fieldnames
            host_rows = list(host_reader)
        with device_path.open("r", encoding="utf-8", newline="") as handle:
            device_reader = csv.DictReader(handle)
            device_fields = device_reader.fieldnames
            device_rows = list(device_reader)
        benchmark.require(
            host_fields == device_fields,
            f"SGS commit placement changed {name} schema",
        )
        benchmark.require(
            len(host_rows) == len(device_rows),
            f"SGS commit placement changed {name} row count",
        )
        for row_index, (host_row, device_row) in enumerate(
            zip(host_rows, device_rows), start=1
        ):
            differences = [
                key for key in host_row
                if host_row[key] != device_row[key]
            ]
            if differences:
                details = ", ".join(
                    f"{key}:host={host_row[key]} device={device_row[key]}"
                    for key in differences[:8]
                )
                raise RuntimeError(
                    f"SGS commit placement changed {name} data row "
                    f"{row_index}: {details}"
                )
        raise RuntimeError(
            f"SGS commit placement changed {name} byte encoding"
        )


def telemetry_summary(
    host_trials: list[dict[str, object]],
    device_trials: list[dict[str, object]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for key in benchmark.TELEMETRY_KEYS:
        host_value = median([
            int(trial["telemetry"][key])  # type: ignore[index]
            for trial in host_trials
        ])
        device_value = median([
            int(trial["telemetry"][key])  # type: ignore[index]
            for trial in device_trials
        ])
        result[key] = {
            "host_commit_median": host_value,
            "device_commit_median": device_value,
            "device_minus_host": device_value - host_value,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("bin/nimbus_sim"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("tests/fixtures/configs/qualification/gpu_report_bearing.yaml"),
    )
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--acoustic-substeps", type=int, default=2)
    parser.add_argument("--dt-large", type=float, default=0.25)
    parser.add_argument("--minimum-speedup-percent", type=float, default=2.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/benchmarks/vulkan-sgs-apply-current.json"),
    )
    args = parser.parse_args()

    benchmark.require(args.binary.is_file(), "missing simulator")
    benchmark.require(args.config.is_file(), "missing benchmark config")
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

    device_config_text = args.config.read_text(encoding="utf-8")
    benchmark.require(
        device_config_text.count(CONFIG_MARKER) == 1,
        "benchmark config must enable gpu_apply_sgs exactly once",
    )
    host_config_text = device_config_text.replace(
        CONFIG_MARKER, "    gpu_apply_sgs: false")

    trials: dict[str, list[dict[str, object]]] = {
        "host": [],
        "device": [],
    }
    with tempfile.TemporaryDirectory(
        prefix="nimbus_vulkan_sgs_apply_"
    ) as root_text:
        root = Path(root_text)
        host_config = root / "host_commit.yaml"
        device_config = root / "device_commit.yaml"
        host_config.write_text(host_config_text, encoding="utf-8")
        device_config.write_text(device_config_text, encoding="utf-8")

        for trial_index in range(args.trials):
            order = (
                ("host", "device")
                if trial_index % 2 == 0 else
                ("device", "host")
            )
            pair: dict[str, dict[str, object]] = {}
            for mode in order:
                label = f"trial-{trial_index + 1}-{mode}"
                result = benchmark.run_trial(
                    args.binary,
                    device_config if mode == "device" else host_config,
                    root / label,
                    args.duration,
                    label,
                    True,
                    args.acoustic_substeps,
                    args.dt_large,
                )
                pair[mode] = result
                trials[mode].append(result)

            benchmark.require_same_endpoint(pair["host"], pair["device"])
            require_same_ledgers(pair["host"], pair["device"])
            host_stages = pair["host"]["stage_telemetry"]
            device_stages = pair["device"]["stage_telemetry"]
            benchmark.require(
                SGS_APPLY_STAGE not in host_stages,  # type: ignore[operator]
                "host comparator unexpectedly dispatched SGS apply",
            )
            device_apply = device_stages.get(  # type: ignore[union-attr]
                SGS_APPLY_STAGE)
            benchmark.require(
                device_apply is not None
                and int(device_apply["dispatches"]) > 0,
                "device branch did not dispatch SGS apply",
            )
            print(
                "[vulkan-sgs-apply] "
                f"trial={trial_index + 1}/{args.trials} "
                f"host={float(pair['host']['avg_step_ms']):.6f} ms "
                f"device={float(pair['device']['avg_step_ms']):.6f} ms"
            )

        host_times = [
            float(trial["avg_step_ms"]) for trial in trials["host"]
        ]
        device_times = [
            float(trial["avg_step_ms"]) for trial in trials["device"]
        ]
        gate = evaluate_speedup_gate(
            device_times, host_times, args.minimum_speedup_percent)
        accepted = bool(gate["passes_speedup_and_noise"])
        device_evidence = {
            str(trial["selected_device_evidence"])
            for mode_trials in trials.values()
            for trial in mode_trials
        }
        benchmark.require(
            len(device_evidence) == 1,
            "physical-device identity changed across trials",
        )
        measured_steps = {
            int(trial["measured_steps"])
            for mode_trials in trials.values()
            for trial in mode_trials
        }
        benchmark.require(
            len(measured_steps) == 1,
            "measured step count changed across trials",
        )
        report = {
            "schema": "nimbus.vulkan-sgs-apply-pareto.v1",
            "status": "accepted" if accepted else "rejected",
            "generated_at_utc":
                dt.datetime.now(dt.timezone.utc).isoformat(),
            "physical_device_required": True,
            "selected_device_evidence": next(iter(device_evidence)),
            "comparison_contract": (
                "same binary, equations, tendencies, per-step scientific "
                "reports, and output; only extensions.compute.gpu_apply_sgs "
                "changes false versus true"
            ),
            "identity": {
                "binary": str(args.binary.resolve()),
                "binary_sha256": benchmark.sha256_file(args.binary),
                "source_config": str(args.config.resolve()),
                "source_config_sha256": benchmark.sha256_file(args.config),
                "host_derived_config_sha256": sha256_text(host_config_text),
                "device_derived_config_sha256":
                    sha256_text(device_config_text),
            },
            "duration_s": args.duration,
            "trials": args.trials,
            "measured_steps_per_trial": next(iter(measured_steps)),
            "runtime_overrides": {
                "acoustic_substeps": args.acoustic_substeps,
                "dt_large_s": args.dt_large,
            },
            "endpoint_fields_bitwise_equal": list(benchmark.CORE_FIELDS),
            "ledger_files_bitwise_equal": list(LEDGERS),
            "gate": gate,
            "host_commit_step_time_ms":
                benchmark.robust_statistics(host_times),
            "device_commit_step_time_ms":
                benchmark.robust_statistics(device_times),
            "telemetry": telemetry_summary(
                trials["host"], trials["device"]),
            "host_trials": [
                benchmark.serializable_trial(trial)
                for trial in trials["host"]
            ],
            "device_trials": [
                benchmark.serializable_trial(trial)
                for trial in trials["device"]
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(
        "[vulkan-sgs-apply] RESULT: "
        f"{'ACCEPTED' if accepted else 'REJECTED'} "
        f"median_speedup_percent="
        f"{float(gate['median_speedup_percent']):.6f} "
        f"median_improvement_ms="
        f"{float(gate['median_improvement_ms']):.6f} "
        f"three_robust_sigma_ms="
        f"{float(gate['three_robust_sigma_ms']):.6f} "
        f"report={args.output}"
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"[vulkan-sgs-apply] RESULT: FAIL: {error}")
        raise SystemExit(1)
