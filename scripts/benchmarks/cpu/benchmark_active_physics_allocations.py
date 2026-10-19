#!/usr/bin/env python3
"""Paired exact-endpoint active-physics benchmark.

Every pair must publish a byte-identical final checkpoint before its timing or
allocation deltas are accepted. Callers describe the isolated implementation
difference and may require a minimum paired speedup above measured noise.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Sequence


PERF_SUMMARY = re.compile(
    r"\[PERF SUMMARY\]\s*measured_steps=(?P<steps>\d+),\s*"
    r"warmup_steps=(?P<warmup>\d+),\s*scope=(?P<scope>[^,]+),\s*"
    r"step_wall_s=(?P<wall>[-+0-9.eE]+),\s*"
    r"avg_step_ms=(?P<step_ms>[-+0-9.eE]+),\s*"
    r"sim_s_per_wall_s=(?P<rate>[-+0-9.eE]+),\s*"
    r"measured_simulated_s=(?P<simulated>[-+0-9.eE]+)"
)
ALLOCATION_SUMMARY = re.compile(
    r"\[ALLOCATION SUMMARY\]\s*"
    r"heap_allocations=(?P<heap_allocations>\d+),\s*"
    r"heap_requested_bytes=(?P<heap_requested_bytes>\d+),\s*"
    r"physics_field_allocations=(?P<field_allocations>\d+),\s*"
    r"physics_field_allocated_bytes=(?P<field_bytes>\d+)"
)
ALLOCATION_COMPONENT = re.compile(
    r"\[ALLOCATION COMPONENT\]\s*name=(?P<name>[a-z_]+),\s*"
    r"heap_allocations=(?P<heap_allocations>\d+),\s*"
    r"heap_requested_bytes=(?P<heap_requested_bytes>\d+),\s*"
    r"physics_field_allocations=(?P<field_allocations>\d+),\s*"
    r"physics_field_allocated_bytes=(?P<field_bytes>\d+)"
)
DYNAMICS_PERF = re.compile(r"\[DYNAMICS PERF\]\s*(.*)")
DYNAMICS_REQUIRED_KEYS = {
    "total",
    "core",
    "advection",
    "microphysics_transport",
    "microphysics",
    "turbulence",
    "turbulence_refresh",
    "turbulence_chaos",
    "turbulence_stability",
    "turbulence_apply",
    "turbulence_commit",
    "diffusion",
    "eos",
    "sponge",
    "final_boundary",
    "profiled",
    "residual",
    "profiled_fraction",
    "step_fraction",
}
TURBULENCE_PERF = re.compile(
    r"\[TURBULENCE PERF\]\s*"
    r"smagorinsky_coefficients_s=(?P<coefficients>[-+0-9.eE]+),\s*"
    r"smagorinsky_momentum_s=(?P<momentum>[-+0-9.eE]+),\s*"
    r"smagorinsky_theta_s=(?P<theta>[-+0-9.eE]+),\s*"
    r"smagorinsky_moisture_s=(?P<moisture>[-+0-9.eE]+),\s*"
    r"smagorinsky_profiled_s=(?P<profiled>[-+0-9.eE]+)"
)
COMPONENT = {
    "radiation_s": re.compile(
        r"^.*\bradiation_s=([-+0-9.eE]+)", re.MULTILINE
    ),
    "boundary_layer_s": re.compile(
        r"^.*\bboundary_layer_s=([-+0-9.eE]+)", re.MULTILINE
    ),
    "chaos_noise_s": re.compile(
        r"^.*\bchaos_noise_s=([-+0-9.eE]+)", re.MULTILINE
    ),
    "chaos_tendency_s": re.compile(
        r"^.*\bchaos_tendency_s=([-+0-9.eE]+)", re.MULTILINE
    ),
    "dynamics_s": re.compile(
        r"^.*\bdynamics_s=([-+0-9.eE]+)", re.MULTILINE
    ),
}
CLEARED_ENV_PREFIXES = ("NIMBUS_", "OMP_", "KMP_", "GOMP_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_rss_bytes(raw: int) -> int:
    return raw if sys.platform == "darwin" else raw * 1024


def parse_dynamics_perf(text: str) -> dict[str, float]:
    """Parse the extensible key/value dynamics timing record.

    New named subphases are evidence, not an ABI break. The ownership fields
    used by the benchmark remain mandatory while additional finite,
    non-negative ``dynamics_*`` metrics are retained in the artifact.
    """
    matches = DYNAMICS_PERF.findall(text)
    if not matches:
        raise ValueError("missing dynamics stage timing")
    values: dict[str, float] = {}
    for item in matches[-1].split(","):
        raw_key, separator, raw_value = item.strip().partition("=")
        if separator != "=" or not raw_key.startswith("dynamics_"):
            raise ValueError("malformed dynamics stage timing")
        key = raw_key.removeprefix("dynamics_")
        if key.endswith("_s"):
            key = key[:-2]
        if key in values:
            raise ValueError(f"duplicate dynamics stage timing '{key}'")
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ValueError(
                f"invalid dynamics stage timing '{key}'"
            ) from error
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid dynamics stage timing '{key}'")
        values[key] = value
    missing = DYNAMICS_REQUIRED_KEYS - values.keys()
    if missing:
        raise ValueError(
            f"missing dynamics stage timing fields: {sorted(missing)}"
        )
    return values


def parse_log(
    text: str,
    require_dynamics_step_coverage: bool = True,
) -> dict[str, Any]:
    perf_matches = list(PERF_SUMMARY.finditer(text))
    allocation_matches = list(ALLOCATION_SUMMARY.finditer(text))
    if not perf_matches or not allocation_matches:
        raise ValueError("missing performance or allocation summary")
    perf = perf_matches[-1].groupdict()
    allocations = allocation_matches[-1].groupdict()
    dynamics = parse_dynamics_perf(text)
    turbulence_matches = list(TURBULENCE_PERF.finditer(text))
    if not turbulence_matches:
        raise ValueError("missing turbulence operator timing")
    turbulence = turbulence_matches[-1].groupdict()
    result: dict[str, Any] = {
        "measured_steps": int(perf["steps"]),
        "warmup_steps": int(perf["warmup"]),
        "scope": perf["scope"].strip(),
        "step_wall_s": float(perf["wall"]),
        "avg_step_ms": float(perf["step_ms"]),
        "sim_s_per_wall_s": float(perf["rate"]),
        "measured_simulated_s": float(perf["simulated"]),
        "heap_allocations": int(allocations["heap_allocations"]),
        "heap_requested_bytes": int(allocations["heap_requested_bytes"]),
        "physics_field_allocations": int(allocations["field_allocations"]),
        "physics_field_allocated_bytes": int(allocations["field_bytes"]),
    }
    for name, pattern in COMPONENT.items():
        matches = list(pattern.finditer(text))
        if not matches:
            raise ValueError(f"missing {name} component timing")
        result[name] = float(matches[-1].group(1))
    result["dynamics_stages"] = {
        name: value
        for name, value in dynamics.items()
    }
    result["turbulence_operators"] = {
        name: float(value)
        for name, value in turbulence.items()
    }
    turbulence_numeric = tuple(result["turbulence_operators"].values())
    if any(
        not math.isfinite(value) or value < 0.0
        for value in turbulence_numeric
    ):
        raise ValueError("invalid turbulence operator timing")
    reconstructed_turbulence = sum(
        result["turbulence_operators"][name]
        for name in ("coefficients", "momentum", "theta", "moisture")
    )
    turbulence_operator_tolerance = max(
        1.0e-8, 2.0e-5 * result["turbulence_operators"]["profiled"]
    )
    if (
        abs(
            reconstructed_turbulence
            - result["turbulence_operators"]["profiled"]
        )
        > turbulence_operator_tolerance
    ):
        raise ValueError("inconsistent turbulence operator timing")
    dynamics_numeric = tuple(result["dynamics_stages"].values())
    if any(not math.isfinite(value) for value in dynamics_numeric):
        raise ValueError("non-finite dynamics stage timing")
    if any(
        result["dynamics_stages"][name] < 0.0
        for name in (
            "total",
            "core",
            "advection",
            "microphysics_transport",
            "microphysics",
            "turbulence",
            "turbulence_refresh",
            "turbulence_chaos",
            "turbulence_stability",
            "turbulence_apply",
            "turbulence_commit",
            "diffusion",
            "eos",
            "sponge",
            "final_boundary",
            "profiled",
        )
    ):
        raise ValueError("negative dynamics stage timing")
    reconstructed = (
        result["dynamics_stages"]["profiled"]
        + result["dynamics_stages"]["residual"]
    )
    timing_tolerance = max(
        1.0e-8, 2.0e-5 * result["dynamics_stages"]["total"]
    )
    if abs(reconstructed - result["dynamics_stages"]["total"]) > timing_tolerance:
        raise ValueError("inconsistent dynamics timing ownership")
    if not 0.0 <= result["dynamics_stages"]["profiled_fraction"] <= 1.0001:
        raise ValueError("invalid dynamics profiled fraction")
    turbulence_profiled = sum(
        result["dynamics_stages"][name]
        for name in (
            "turbulence_refresh",
            "turbulence_chaos",
            "turbulence_stability",
            "turbulence_apply",
            "turbulence_commit",
        )
    )
    turbulence_tolerance = max(
        1.0e-8, 2.0e-5 * result["dynamics_stages"]["turbulence"]
    )
    if (
        turbulence_profiled
        > result["dynamics_stages"]["turbulence"] + turbulence_tolerance
    ):
        raise ValueError("nested turbulence timing exceeds its parent stage")
    result["dynamics_stages"]["turbulence_profiled"] = turbulence_profiled
    result["dynamics_stages"]["turbulence_residual"] = (
        result["dynamics_stages"]["turbulence"] - turbulence_profiled
    )
    result["dynamics_stages"]["turbulence_profiled_fraction"] = (
        turbulence_profiled
        / max(result["dynamics_stages"]["turbulence"], 1.0e-12)
    )
    dynamics_outer_tolerance = max(1.0e-6, 5.0e-3 * result["dynamics_s"])
    if (
        abs(result["dynamics_stages"]["total"] - result["dynamics_s"])
        > dynamics_outer_tolerance
    ):
        raise ValueError("inner and outer dynamics timers disagree")
    result["dynamics_named_step_fraction"] = (
        result["dynamics_stages"]["profiled"] / result["step_wall_s"]
    )
    if (
        require_dynamics_step_coverage
        and result["dynamics_named_step_fraction"] < 0.95
    ):
        raise ValueError(
            "named dynamics stages cover less than 95% of solver-step time"
        )
    result["outer_profiled_s"] = sum(
        result[name]
        for name in (
            "radiation_s",
            "boundary_layer_s",
            "chaos_noise_s",
            "chaos_tendency_s",
            "dynamics_s",
        )
    )
    result["outer_profiled_fraction"] = (
        result["outer_profiled_s"] / result["step_wall_s"]
    )
    result["allocation_components"] = {
        match.group("name"): {
            "heap_allocations": int(match.group("heap_allocations")),
            "heap_requested_bytes": int(match.group("heap_requested_bytes")),
            "physics_field_allocations": int(match.group("field_allocations")),
            "physics_field_allocated_bytes": int(match.group("field_bytes")),
        }
        for match in ALLOCATION_COMPONENT.finditer(text)
    }
    numeric = (
        "step_wall_s",
        "avg_step_ms",
        "sim_s_per_wall_s",
        "measured_simulated_s",
        "radiation_s",
        "boundary_layer_s",
        "chaos_noise_s",
        "chaos_tendency_s",
        "dynamics_s",
        "dynamics_named_step_fraction",
        "outer_profiled_s",
        "outer_profiled_fraction",
    )
    if any(not math.isfinite(result[name]) for name in numeric):
        raise ValueError("non-finite benchmark metric")
    return result


def run_worker() -> int:
    request = json.load(sys.stdin)
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith(CLEARED_ENV_PREFIXES):
            environment.pop(key, None)
    environment["OMP_NUM_THREADS"] = str(request["threads"])
    completed = subprocess.run(
        request["command"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    combined = completed.stdout + "\n" + completed.stderr
    payload: dict[str, Any] = {
        "returncode": completed.returncode,
        "peak_rss_bytes": peak_rss_bytes(usage.ru_maxrss),
        "log_tail": "\n".join(combined.splitlines()[-100:]),
    }
    if completed.returncode == 0:
        try:
            payload.update(parse_log(
                combined,
                bool(request.get("require_dynamics_step_coverage", True)),
            ))
            checkpoints = sorted(
                Path(request["checkpoint_dir"]).glob("checkpoint_*.nimbus")
            )
            if len(checkpoints) != 1:
                raise ValueError(
                    f"expected one final checkpoint, found {len(checkpoints)}"
                )
            payload["checkpoint_bytes"] = checkpoints[0].stat().st_size
            payload["checkpoint_sha256"] = sha256(checkpoints[0])
        except (OSError, ValueError) as error:
            payload["parse_error"] = str(error)
    json.dump(payload, sys.stdout)
    return 0


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def nonnegative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--optimized", required=True)
    result.add_argument("--legacy", required=True)
    result.add_argument("--config", required=True)
    result.add_argument("--trials", type=positive_int, default=7)
    result.add_argument("--threads", type=positive_int, default=os.cpu_count() or 1)
    result.add_argument("--duration", type=positive_int, default=240)
    result.add_argument(
        "--comparison-contract",
        default=(
            "same equations/order/cadence/workers; comparator discards only "
            "replaceable scratch"
        ),
    )
    result.add_argument(
        "--minimum-speedup-percent",
        type=nonnegative_float,
        default=0.0,
        help=(
            "fail unless the median paired speedup reaches this threshold and "
            "exceeds the paired three-MAD noise bound; zero disables the gate"
        ),
    )
    result.add_argument(
        "--output",
        default="reports/benchmarks/active-physics-allocation-current.json",
    )
    return result


def run_one(
    label: str,
    binary: Path,
    config: Path,
    threads: int,
    duration: int,
    require_dynamics_step_coverage: bool = True,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"nimbus-{label}-") as temporary:
        root = Path(temporary)
        checkpoint_dir = root / "checkpoints"
        output_dir = root / "output"
        command = [
            str(binary),
            "--headless",
            f"--config={config}",
            "--compute-backend=cpu",
            "--log-profile=normal",
            "--timing",
            "--timing-every=0",
            f"--duration={duration}",
            f"--checkpoint-every={duration}",
            f"--checkpoint-dir={checkpoint_dir}",
            f"--outdir={output_dir}",
        ]
        request = {
            "command": command,
            "threads": threads,
            "checkpoint_dir": str(checkpoint_dir),
            "require_dynamics_step_coverage":
                require_dynamics_step_coverage,
        }
        worker = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--_worker"],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            check=False,
        )
        if worker.returncode != 0:
            raise RuntimeError(f"{label} benchmark worker failed:\n{worker.stderr}")
        result = json.loads(worker.stdout)
        if result["returncode"] != 0 or "parse_error" in result:
            detail = result.get("parse_error", f"exit {result['returncode']}")
            raise RuntimeError(
                f"{label} benchmark failed ({detail}):\n{result['log_tail']}"
            )
        result.pop("log_tail", None)
        result["variant"] = label
        return result


def metric_summary(values: Sequence[float | int]) -> dict[str, float]:
    converted = [float(value) for value in values]
    return {
        "median": statistics.median(converted),
        "min": min(converted),
        "max": max(converted),
    }


def summarize_variant(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "avg_step_ms",
        "step_wall_s",
        "radiation_s",
        "boundary_layer_s",
        "chaos_noise_s",
        "chaos_tendency_s",
        "dynamics_s",
        "dynamics_named_step_fraction",
        "outer_profiled_s",
        "outer_profiled_fraction",
        "heap_allocations",
        "heap_requested_bytes",
        "physics_field_allocations",
        "physics_field_allocated_bytes",
        "peak_rss_bytes",
    )
    summary = {
        metric: metric_summary([run[metric] for run in runs])
        for metric in metrics
    }
    summary["dynamics_stages"] = {
        metric: metric_summary(
            [run["dynamics_stages"][metric] for run in runs]
        )
        for metric in runs[0]["dynamics_stages"]
    }
    summary["turbulence_operators"] = {
        metric: metric_summary(
            [run["turbulence_operators"][metric] for run in runs]
        )
        for metric in runs[0]["turbulence_operators"]
    }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    optimized = Path(args.optimized).resolve()
    legacy = Path(args.legacy).resolve()
    config = Path(args.config).resolve()
    for path in (optimized, legacy, config):
        if not path.is_file():
            raise SystemExit(f"required input does not exist: {path}")

    pairs: list[dict[str, Any]] = []
    variants = {"optimized": optimized, "legacy": legacy}
    for pair_index in range(args.trials):
        order = ("optimized", "legacy") if pair_index % 2 == 0 else (
            "legacy", "optimized"
        )
        pair_runs: dict[str, dict[str, Any]] = {}
        for label in order:
            run = run_one(
                label, variants[label], config, args.threads, args.duration
            )
            pair_runs[label] = run
            print(
                f"[active-physics] pair {pair_index + 1}/{args.trials} "
                f"{label}: {run['avg_step_ms']:.4f} ms/step, "
                f"heap={run['heap_allocations']}, "
                f"fields={run['physics_field_allocations']}",
                file=sys.stderr,
            )
        if (
            pair_runs["optimized"]["checkpoint_sha256"]
            != pair_runs["legacy"]["checkpoint_sha256"]
            or pair_runs["optimized"]["checkpoint_bytes"]
            != pair_runs["legacy"]["checkpoint_bytes"]
        ):
            raise SystemExit(
                f"pair {pair_index + 1} final checkpoints are not byte-identical"
            )
        pairs.append(
            {
                "pair": pair_index + 1,
                "order": list(order),
                **pair_runs,
            }
        )

    optimized_runs = [pair["optimized"] for pair in pairs]
    legacy_runs = [pair["legacy"] for pair in pairs]
    endpoint_hashes = {
        run["checkpoint_sha256"] for run in optimized_runs + legacy_runs
    }
    if len(endpoint_hashes) != 1:
        raise SystemExit("final checkpoint changed across repeated trials")

    optimized_summary = summarize_variant(optimized_runs)
    legacy_summary = summarize_variant(legacy_runs)
    optimized_ms = optimized_summary["avg_step_ms"]["median"]
    legacy_ms = legacy_summary["avg_step_ms"]["median"]
    paired_speedups_percent = [
        100.0
        * (pair["legacy"]["avg_step_ms"] - pair["optimized"]["avg_step_ms"])
        / pair["legacy"]["avg_step_ms"]
        for pair in pairs
    ]
    paired_speedup_median = statistics.median(paired_speedups_percent)
    paired_speedup_mad = statistics.median(
        abs(value - paired_speedup_median)
        for value in paired_speedups_percent
    )
    paired_three_mad_percent = 3.0 * paired_speedup_mad
    speedup_gate_enabled = args.minimum_speedup_percent > 0.0
    speedup_gate_passed = (
        not speedup_gate_enabled
        or (
            paired_speedup_median >= args.minimum_speedup_percent
            and paired_speedup_median > paired_three_mad_percent
        )
    )
    report = {
        "schema": "nimbus.active-physics-allocation-benchmark.v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
        },
        "run": {
            "config": str(config),
            "config_sha256": sha256(config),
            "trials": args.trials,
            "threads": args.threads,
            "duration_s": args.duration,
            "order": "alternating optimized-first / legacy-first",
            "optimized_binary": str(optimized),
            "optimized_binary_sha256": sha256(optimized),
            "legacy_binary": str(legacy),
            "legacy_binary_sha256": sha256(legacy),
            "measurement_contract": (
                f"{args.comparison_contract}; warmup excluded; "
                "final checkpoints exact"
            ),
            "minimum_speedup_percent": args.minimum_speedup_percent,
        },
        "endpoint": {
            "all_trials_bitwise_identical": True,
            "checkpoint_sha256": next(iter(endpoint_hashes)),
            "checkpoint_bytes": optimized_runs[0]["checkpoint_bytes"],
        },
        "pairs": pairs,
        "aggregate": {
            "optimized": optimized_summary,
            "legacy": legacy_summary,
            "legacy_over_optimized_step_ratio": legacy_ms / optimized_ms,
            "optimized_step_time_reduction_percent": (
                100.0 * (legacy_ms - optimized_ms) / legacy_ms
            ),
            "paired_speedups_percent": paired_speedups_percent,
            "paired_speedup_median_percent": paired_speedup_median,
            "paired_speedup_mad_percent": paired_speedup_mad,
            "paired_three_mad_percent": paired_three_mad_percent,
            "speedup_gate_enabled": speedup_gate_enabled,
            "speedup_gate_passed": speedup_gate_passed,
            "median_allocation_delta_legacy_minus_optimized": {
                metric: (
                    legacy_summary[metric]["median"]
                    - optimized_summary[metric]["median"]
                )
                for metric in (
                    "heap_allocations",
                    "heap_requested_bytes",
                    "physics_field_allocations",
                    "physics_field_allocated_bytes",
                    "peak_rss_bytes",
                )
            },
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[active-physics] optimized={optimized_ms:.4f} ms/step, "
        f"legacy={legacy_ms:.4f} ms/step, "
        f"ratio={legacy_ms / optimized_ms:.5f}x, "
        f"endpoint={next(iter(endpoint_hashes))}",
        file=sys.stderr,
    )
    print(f"[active-physics] wrote {output}", file=sys.stderr)
    if not speedup_gate_passed:
        raise SystemExit(
            "paired speedup gate failed: "
            f"median={paired_speedup_median:.4f}%, "
            f"required={args.minimum_speedup_percent:.4f}%, "
            f"three_mad={paired_three_mad_percent:.4f}%"
        )
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--_worker":
        raise SystemExit(run_worker())
    raise SystemExit(main())
