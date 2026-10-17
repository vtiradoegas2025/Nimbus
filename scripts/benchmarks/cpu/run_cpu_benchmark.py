#!/usr/bin/env python3
"""Run repeatable whole-runtime Nimbus CPU benchmarks and emit JSON.

The script deliberately reports measurements only. It does not encode a wall
clock pass/fail threshold; regression policy belongs in a versioned baseline
that can account for the host, compiler, and build flags recorded here.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import shlex
import statistics
import subprocess
import sys
import time
from typing import Any, Sequence

try:
    import resource
except ImportError:  # Windows has no resource module.
    resource = None  # type: ignore[assignment]


PERF_SUMMARY = re.compile(
    r"\[PERF SUMMARY\]\s*"
    r"measured_steps=(?P<measured_steps>\d+),\s*"
    r"warmup_steps=(?P<warmup_steps>\d+),\s*"
    r"scope=(?P<scope>[^,]+),\s*"
    r"step_wall_s=(?P<step_wall_s>[-+0-9.eE]+),\s*"
    r"avg_step_ms=(?P<avg_step_ms>[-+0-9.eE]+),\s*"
    r"sim_s_per_wall_s=(?P<sim_s_per_wall_s>[-+0-9.eE]+),\s*"
    r"measured_simulated_s=(?P<measured_simulated_s>[-+0-9.eE]+)"
)
MEMORY_MILESTONE = re.compile(
    r"\[MEMORY MILESTONE\]\s*phase=(?P<phase>[A-Za-z0-9_\-]+),\s*"
    r"peak_rss_bytes=(?P<peak_rss_bytes>\d+)"
)
MEMORY_PLAN = re.compile(
    r"\[MEMORY PLAN\]\s*"
    r"cells=(?P<cells>\d+),\s*"
    r"field_bytes=(?P<field_bytes>\d+),\s*"
    r"predicted_fields=(?P<predicted_fields>\d+),\s*"
    r"fixed_overhead_bytes=(?P<fixed_overhead_bytes>\d+),\s*"
    r"predicted_bytes=(?P<predicted_bytes>\d+)"
)

CLEARED_ENVIRONMENT_PREFIXES = ("NIMBUS_", "OMP_", "KMP_", "GOMP_")
TUNING_ENVIRONMENT_PREFIXES = ("OMP_", "KMP_", "GOMP_")
ENVIRONMENT_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_tail(stdout: str, stderr: str, line_count: int = 80) -> str:
    combined = []
    if stdout:
        combined.extend(f"stdout: {line}" for line in stdout.splitlines())
    if stderr:
        combined.extend(f"stderr: {line}" for line in stderr.splitlines())
    return "\n".join(combined[-line_count:])


def parse_perf_summary(stdout: str, stderr: str) -> dict[str, Any] | None:
    matches = list(PERF_SUMMARY.finditer(stdout + "\n" + stderr))
    if not matches:
        return None
    values = matches[-1].groupdict()
    return {
        "measured_steps": int(values["measured_steps"]),
        "warmup_steps": int(values["warmup_steps"]),
        "scope": values["scope"].strip(),
        "step_wall_s": float(values["step_wall_s"]),
        "avg_step_ms": float(values["avg_step_ms"]),
        "sim_s_per_wall_s": float(values["sim_s_per_wall_s"]),
        "measured_simulated_s": float(values["measured_simulated_s"]),
    }


def parse_memory_milestones(stdout: str, stderr: str) -> list[dict[str, Any]]:
    return [
        {
            "phase": match.group("phase"),
            "peak_rss_bytes": int(match.group("peak_rss_bytes")),
        }
        for match in MEMORY_MILESTONE.finditer(stdout + "\n" + stderr)
    ]


def parse_memory_plan(stdout: str, stderr: str) -> dict[str, int] | None:
    matches = list(MEMORY_PLAN.finditer(stdout + "\n" + stderr))
    if not matches:
        return None
    values = matches[-1].groupdict()
    return {key: int(value) for key, value in values.items()}


def derive_memory_calibration(
    memory_plan: dict[str, int] | None,
    peak_rss: int,
    milestones: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    if memory_plan is None or memory_plan["field_bytes"] <= 0:
        return None
    field_bytes = memory_plan["field_bytes"]
    predicted_bytes = memory_plan["predicted_bytes"]
    phases = [
        {
            **milestone,
            "field_equivalents": milestone["peak_rss_bytes"] / field_bytes,
            "plan_margin_bytes": predicted_bytes - milestone["peak_rss_bytes"],
        }
        for milestone in milestones
    ]
    return {
        "observed_peak_field_equivalents": peak_rss / field_bytes,
        "plan_margin_bytes": predicted_bytes - peak_rss,
        "within_plan": peak_rss <= predicted_bytes,
        "phases": phases,
    }


def peak_rss_bytes(raw_maxrss: int) -> int:
    # getrusage(2): Darwin reports bytes; Linux and the common BSDs report KiB.
    return raw_maxrss if sys.platform in {"darwin", "win32"} else raw_maxrss * 1024


def windows_process_peak_rss(process: subprocess.Popen[str]) -> int:
    """Read PeakWorkingSetSize while Popen still owns the Windows handle."""
    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("page_fault_count", ctypes.c_ulong),
            ("peak_working_set_size", ctypes.c_size_t),
            ("working_set_size", ctypes.c_size_t),
            ("quota_peak_paged_pool_usage", ctypes.c_size_t),
            ("quota_paged_pool_usage", ctypes.c_size_t),
            ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
            ("quota_non_paged_pool_usage", ctypes.c_size_t),
            ("pagefile_usage", ctypes.c_size_t),
            ("peak_pagefile_usage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    try:
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        query = psapi.GetProcessMemoryInfo
        query.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        query.restype = ctypes.c_int
        handle = ctypes.c_void_p(int(process._handle))  # type: ignore[attr-defined]
        if not query(handle, ctypes.byref(counters), counters.cb):
            return 0
    except (AttributeError, OSError):
        return 0
    return int(counters.peak_working_set_size)


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if sys.platform == "win32":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                process.kill()
        process.wait()


def run_child(
    command: list[str], environment: dict[str, str], timeout_s: float
) -> tuple[int | None, bool, float, int, str, str]:
    launch_options: dict[str, Any]
    if sys.platform == "win32":
        launch_options = {
            "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        }
    else:
        launch_options = {"start_new_session": True}
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **launch_options,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_s or None)
        timed_out = False
    except subprocess.TimeoutExpired:
        terminate_process_tree(process)
        stdout, stderr = process.communicate()
        timed_out = True
    elapsed = time.perf_counter() - started
    if sys.platform == "win32":
        peak_rss = windows_process_peak_rss(process)
    elif resource is not None:
        peak_rss = peak_rss_bytes(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    else:
        peak_rss = 0
    return process.returncode, timed_out, elapsed, peak_rss, stdout, stderr


def run_worker() -> int:
    """Run exactly one child so RUSAGE_CHILDREN is a per-trial measurement."""
    request = json.load(sys.stdin)
    command = request["command"]
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith(tuple(request["clear_environment_prefixes"])):
            environment.pop(key, None)
    environment.update(request["environment"])

    returncode, timed_out, elapsed, peak_rss, stdout, stderr = run_child(
        command, environment, float(request.get("timeout_s") or 0.0)
    )

    if timed_out:
        payload = {
            "returncode": None,
            "timed_out": True,
            "process_wall_s": elapsed,
            "peak_rss_bytes": peak_rss,
            "summary": None,
            "memory_milestones": [],
            "memory_plan": None,
            "log_tail": output_tail(stdout, stderr),
        }
    else:
        payload = {
            "returncode": returncode,
            "timed_out": False,
            "process_wall_s": elapsed,
            "peak_rss_bytes": peak_rss,
            "summary": parse_perf_summary(stdout, stderr),
            "memory_milestones": parse_memory_milestones(
                stdout, stderr
            ),
            "memory_plan": parse_memory_plan(stdout, stderr),
            "log_tail": output_tail(stdout, stderr),
        }
    json.dump(payload, sys.stdout)
    return 0


def nearest_rank_p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def summarize(trials: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "step_wall_s",
        "avg_step_ms",
        "sim_s_per_wall_s",
        "process_wall_s",
        "peak_rss_bytes",
    )
    aggregate: dict[str, Any] = {}
    for metric in metrics:
        values = [float(trial[metric]) for trial in trials]
        aggregate[metric] = {
            "median": statistics.median(values),
            "p95_nearest_rank": nearest_rank_p95(values),
            "min": min(values),
            "max": max(values),
        }
    return aggregate


def compiler_metadata(compiler: str) -> dict[str, Any]:
    result: dict[str, Any] = {"command": compiler}
    try:
        command = shlex.split(compiler, posix=sys.platform != "win32")
        if sys.platform == "win32":
            command = [
                part[1:-1]
                if len(part) >= 2 and part[0] == part[-1] == '"'
                else part
                for part in command
            ]
        if command:
            completed = subprocess.run(
                command + ["--version"], text=True, capture_output=True,
                timeout=10, check=False,
            )
            version_lines = (completed.stdout or completed.stderr).splitlines()
            if version_lines:
                result["version"] = version_lines[0]
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return result


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def tuning_environment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("must use KEY=VALUE syntax")
    key, setting = value.split("=", 1)
    if not ENVIRONMENT_KEY.fullmatch(key):
        raise argparse.ArgumentTypeError(f"invalid environment key: {key!r}")
    if not key.startswith(TUNING_ENVIRONMENT_PREFIXES):
        allowed = ", ".join(f"{prefix}*" for prefix in TUNING_ENVIRONMENT_PREFIXES)
        raise argparse.ArgumentTypeError(
            f"{key} is not an allowed OpenMP tuning variable ({allowed})"
        )
    if key == "OMP_NUM_THREADS":
        raise argparse.ArgumentTypeError(
            "OMP_NUM_THREADS is controlled by --threads"
        )
    return key, setting


def validate_perf_summary(summary: dict[str, Any]) -> str | None:
    numeric_keys = (
        "step_wall_s", "avg_step_ms", "sim_s_per_wall_s",
        "measured_simulated_s",
    )
    if any(not math.isfinite(float(summary[key])) for key in numeric_keys):
        return "performance summary contains a non-finite metric"
    if (summary["measured_steps"] <= 0
            or summary["step_wall_s"] <= 0.0
            or summary["measured_simulated_s"] <= 0.0
            or summary["sim_s_per_wall_s"] <= 0.0):
        return ("performance summary must contain positive measured steps, "
                "wall time, simulated seconds, and simulated-seconds rate")
    expected_step_ms = (
        1000.0 * summary["step_wall_s"] / summary["measured_steps"]
    )
    if not math.isclose(
        summary["avg_step_ms"], expected_step_ms, rel_tol=5.0e-5, abs_tol=1.0e-8
    ):
        return "avg_step_ms is inconsistent with step_wall_s/measured_steps"
    expected_rate = summary["measured_simulated_s"] / summary["step_wall_s"]
    if not math.isclose(
        summary["sim_s_per_wall_s"], expected_rate,
        rel_tol=5.0e-5, abs_tol=1.0e-8,
    ):
        return "sim_s_per_wall_s is inconsistent with measured simulated/wall time"
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run repeated Nimbus CPU simulations and write benchmark JSON."
    )
    parser.add_argument("--binary", default="bin/nimbus_sim_cpu_benchmark")
    parser.add_argument("--config", default="tests/fixtures/configs/benchmark/benchmark.yaml")
    parser.add_argument("--trials", type=positive_int, default=7)
    parser.add_argument("--threads", type=positive_int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--timeout", type=non_negative_float, default=0.0,
        help="Per-trial timeout in seconds; zero disables the timeout.",
    )
    parser.add_argument("--output", help="Optional path for the JSON report.")
    parser.add_argument("--compiler", default="unknown")
    parser.add_argument("--cxxflags", default="unknown")
    parser.add_argument("--ldflags", default="unknown")
    parser.add_argument("--build-profile", default="unspecified")
    parser.add_argument(
        "--extra-arg", action="append", default=[],
        help="Additional argument passed to nimbus_sim (repeatable).",
    )
    parser.add_argument(
        "--env", action="append", default=[], type=tuning_environment,
        metavar="KEY=VALUE",
        help=("Controlled OpenMP tuning variable passed to each trial; inherited "
              "NIMBUS/OMP/KMP/GOMP variables are otherwise cleared."),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    binary = Path(args.binary).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()
    if not binary.is_file():
        raise SystemExit(f"benchmark binary does not exist: {binary}")
    if sys.platform != "win32" and not os.access(binary, os.X_OK):
        raise SystemExit(f"benchmark binary is not executable: {binary}")
    if not config.is_file():
        raise SystemExit(f"benchmark config does not exist: {config}")

    command = [
        str(binary),
        "--headless",
        f"--config={config}",
        *args.extra_arg,
        "--compute-backend=cpu",
        "--log-profile=normal",
        "--timing",
        "--timing-every=0",
    ]
    environment = {"OMP_NUM_THREADS": str(args.threads)}
    for key, value in args.env:
        environment[key] = value
    trials: list[dict[str, Any]] = []
    worker_request = {
        "command": command,
        "environment": environment,
        "clear_environment_prefixes": CLEARED_ENVIRONMENT_PREFIXES,
        "timeout_s": args.timeout,
    }

    for trial_index in range(1, args.trials + 1):
        worker = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--_worker"],
            input=json.dumps(worker_request),
            text=True,
            capture_output=True,
            check=False,
        )
        if worker.returncode != 0:
            raise SystemExit(
                f"benchmark worker failed on trial {trial_index}:\n{worker.stderr}"
            )
        try:
            result = json.loads(worker.stdout)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"benchmark worker returned invalid JSON on trial {trial_index}: {exc}"
            ) from exc
        if result["timed_out"]:
            raise SystemExit(
                f"benchmark trial {trial_index} timed out after {args.timeout:g}s:\n"
                f"{result['log_tail']}"
            )
        if result["returncode"] != 0:
            raise SystemExit(
                f"benchmark trial {trial_index} exited {result['returncode']}:\n"
                f"{result['log_tail']}"
            )
        if result["summary"] is None:
            raise SystemExit(
                f"benchmark trial {trial_index} produced no [PERF SUMMARY]:\n"
                f"{result['log_tail']}"
            )
        summary_error = validate_perf_summary(result["summary"])
        if summary_error is not None:
            raise SystemExit(
                f"benchmark trial {trial_index} produced an invalid "
                f"[PERF SUMMARY]: {summary_error}\n{result['log_tail']}"
            )

        trial = {"trial": trial_index, **result["summary"]}
        trial["process_wall_s"] = result["process_wall_s"]
        trial["peak_rss_bytes"] = result["peak_rss_bytes"]
        trial["memory_milestones"] = result["memory_milestones"]
        trial["memory_plan"] = result["memory_plan"]
        trial["memory_calibration"] = derive_memory_calibration(
            result["memory_plan"],
            result["peak_rss_bytes"],
            result["memory_milestones"],
        )
        trials.append(trial)
        print(
            f"[benchmark-cpu] {trial_index}/{args.trials}: "
            f"{trial['avg_step_ms']:.3f} ms/step, "
            f"{trial['sim_s_per_wall_s']:.3f} sim-s/wall-s, "
            f"process {trial['process_wall_s']:.3f}s",
            file=sys.stderr,
        )

    scopes = {trial["scope"] for trial in trials}
    measured_steps_set = {trial["measured_steps"] for trial in trials}
    warmup_steps_set = {trial["warmup_steps"] for trial in trials}
    memory_plans = {
        json.dumps(trial["memory_plan"], sort_keys=True)
        for trial in trials
    }
    if len(scopes) != 1 or len(measured_steps_set) != 1 or len(warmup_steps_set) != 1:
        raise SystemExit(
            "refusing to aggregate heterogeneous trials: scope, measured_steps, "
            "and warmup_steps must match"
        )
    if len(memory_plans) != 1:
        raise SystemExit(
            "refusing to aggregate heterogeneous trials: memory plans must match"
        )
    simulated_values = [trial["measured_simulated_s"] for trial in trials]
    if not all(
        math.isclose(value, simulated_values[0], rel_tol=1.0e-9, abs_tol=1.0e-9)
        for value in simulated_values[1:]
    ):
        raise SystemExit(
            "refusing to aggregate heterogeneous trials: measured simulated "
            "seconds do not match"
        )
    scope = next(iter(scopes))
    measured_steps = next(iter(measured_steps_set))
    warmup_steps = next(iter(warmup_steps_set))
    report = {
        "schema": "nimbus.cpu-benchmark.v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "build": {
            "profile": args.build_profile,
            "compiler": compiler_metadata(args.compiler),
            "cxxflags": args.cxxflags,
            "ldflags": args.ldflags,
            "binary": str(binary),
            "binary_sha256": sha256_file(binary),
        },
        "run": {
            "config": str(config),
            "config_sha256": sha256_file(config),
            "command": command,
            "trials": args.trials,
            "threads": args.threads,
            "environment": {
                **environment,
            },
            "cleared_inherited_environment_prefixes":
                list(CLEARED_ENVIRONMENT_PREFIXES),
            "metric_scope": scope,
            "measured_steps": measured_steps,
            "warmup_steps": warmup_steps,
            "measured_simulated_s": simulated_values[0],
            "memory_plan": trials[0]["memory_plan"],
        },
        "trials": trials,
        "aggregate": summarize(trials),
        "regression_policy": None,
    }

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"[benchmark-cpu] wrote {output}", file=sys.stderr)
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--_worker":
        raise SystemExit(run_worker())
    raise SystemExit(main())
