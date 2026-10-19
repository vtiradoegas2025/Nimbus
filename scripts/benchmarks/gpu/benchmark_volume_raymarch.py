#!/usr/bin/env python3
"""Repeat a fixed native-volume workload and report median GPU render time."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

GPU_RE = re.compile(r"GPU render: samples=(\d+) total=([0-9.]+)ms avg=([0-9.]+)ms")
SESSION_RE = re.compile(
    r"Session: (\d+) frames in ([0-9.]+)s \(([0-9.]+) avg FPS\)"
)


def parse_viewer_log(log: str) -> dict[str, float | int]:
    gpu_matches = list(GPU_RE.finditer(log))
    session_matches = list(SESSION_RE.finditer(log))
    if not gpu_matches or not session_matches:
        raise ValueError("viewer log lacks final session/GPU timestamp summaries")
    gpu = gpu_matches[-1]
    session = session_matches[-1]
    return {
        "gpu_samples": int(gpu.group(1)),
        "gpu_total_ms": float(gpu.group(2)),
        "gpu_average_ms": float(gpu.group(3)),
        "frames": int(session.group(1)),
        "wall_seconds": float(session.group(2)),
        "average_fps": float(session.group(3)),
    }


def build_command(args: argparse.Namespace, seconds: float) -> list[str]:
    return [
        str(args.viewer),
        "--validation",
        "--window-test",
        "--window-seconds", str(seconds),
        "--window-width", str(args.width),
        "--window-height", str(args.height),
        "--viewer-fps", str(args.viewer_fps),
        "--render-backend", "volume",
        "--input", str(args.input),
        "--fields", args.fields,
        "--volume-mode", args.volume_mode,
        "--texture-mode", args.texture_mode,
        "--camera-mode", "fixed",
        "--playback-fps", "0.01",
        "--ray-steps", str(args.ray_steps),
    ]


def run_trial(command: list[str]) -> tuple[dict[str, float | int], str]:
    env = os.environ.copy()
    env["NIMBUS_GPU_KERNEL_TIMING"] = "1"
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"viewer exited {completed.returncode}:\n{completed.stdout[-4000:]}"
        )
    if "[vulkan][validation][error]" in completed.stdout.lower():
        raise RuntimeError("viewer emitted a Vulkan validation error")
    return parse_viewer_log(completed.stdout), completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", type=Path, default=Path("bin/vulkan_viewer"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--fields", default="theta,w")
    parser.add_argument("--volume-mode", default="composite")
    parser.add_argument("--texture-mode", choices=("natural", "smooth"), default="natural")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--viewer-fps", type=int, default=30)
    parser.add_argument("--ray-steps", type=int, default=192)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--warmup-seconds", type=float, default=3.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.seconds <= 0 or args.warmup_seconds < 0:
        parser.error("repeats/seconds must be positive and warmup non-negative")

    if args.warmup_seconds > 0:
        run_trial(build_command(args, args.warmup_seconds))

    trials = []
    logs = []
    for _ in range(args.repeats):
        trial, log = run_trial(build_command(args, args.seconds))
        trials.append(trial)
        logs.append(log)

    gpu_values = [float(trial["gpu_average_ms"]) for trial in trials]
    fps_values = [float(trial["average_fps"]) for trial in trials]
    report = {
        "schema": "nimbus.volume-raymarch-benchmark.v1",
        "workload": {
            "resolution": [args.width, args.height],
            "fields": args.fields.split(","),
            "volume_mode": args.volume_mode,
            "texture_mode": args.texture_mode,
            "camera_mode": "fixed",
            "ray_step_ceiling": args.ray_steps,
            "seconds_per_trial": args.seconds,
            "repeats": args.repeats,
        },
        "trials": trials,
        "median_gpu_ms": statistics.median(gpu_values),
        "min_gpu_ms": min(gpu_values),
        "max_gpu_ms": max(gpu_values),
        "median_fps": statistics.median(fps_values),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        for index, log in enumerate(logs, start=1):
            args.output.with_suffix(f".trial-{index}.log").write_text(
                log, encoding="utf-8"
            )
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"[volume-benchmark] {exc}", file=sys.stderr)
        raise SystemExit(1)
