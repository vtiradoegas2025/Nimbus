#!/usr/bin/env python3
"""Short physical-device qualification for flat C-grid Vulkan translation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


DOMAIN_MOTION_SCRIPTS = Path(__file__).resolve().parent
CHECKPOINT_SCRIPTS = DOMAIN_MOTION_SCRIPTS.parent / "checkpointing"
for script_directory in (DOMAIN_MOTION_SCRIPTS, CHECKPOINT_SCRIPTS):
    if str(script_directory) not in sys.path:
        sys.path.insert(0, str(script_directory))

import qualify_constant_velocity_domain as domain_gate
import qualify_vulkan_restart as restart_gate


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "tests/fixtures/configs/qualification/constant_velocity_cgrid.yaml"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(
    binary: Path,
    config: Path,
    output: Path,
    duration_s: int,
    checkpoint_every_s: int = 0,
    resume: Path | None = None,
) -> dict[str, object]:
    command = [
        str(binary),
        "--headless",
        f"--config={config}",
        f"--outdir={output}",
        "--compute-backend=vulkan",
        "--threads=1",
        f"--duration={duration_s}",
        f"--write-every={duration_s}",
        f"--checkpoint-every={checkpoint_every_s}",
        # Backend selection is scientific provenance for this gate. Normal
        # logging keeps the physical-device record available to the harness;
        # stdout remains captured and is printed only on failure.
        "--log-profile=normal",
        "--no-timing",
    ]
    if resume is not None:
        command.append(f"--resume={resume}")
    environment = os.environ.copy()
    environment.update(
        {
            "NIMBUS_GPU_REQUIRED": "1",
            "NIMBUS_GPU_RESIDENT": "1",
            "NIMBUS_GPU_LIN_BATCH": "1",
            "NIMBUS_GPU_KERNEL_TIMING": "1",
        }
    )
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(
        result.returncode == 0,
        f"{output.name}: run exited {result.returncode}\n{result.stdout[-12000:]}",
    )
    selections = [
        line
        for line in result.stdout.splitlines()
        if "[COMPUTE][VULKAN] selected device" in line
    ]
    unique_selections = set(selections)
    require(
        len(unique_selections) <= 1,
        f"{output.name}: multiple different Vulkan devices were selected",
    )
    require(
        "Falling back to CPU backend" not in result.stdout,
        f"{output.name}: Vulkan fell back to CPU",
    )
    telemetry = [
        line for line in result.stdout.splitlines() if "[GPU TELEMETRY]" in line
    ]
    require(telemetry, f"{output.name}: missing GPU telemetry")
    telemetry_match = re.search(
        r"model_steps=([0-9]+).*dispatches=([0-9]+)", telemetry[-1]
    )
    require(telemetry_match is not None, f"{output.name}: malformed GPU telemetry")
    require(
        int(telemetry_match.group(1)) > 0
        and int(telemetry_match.group(2)) > 0,
        f"{output.name}: vacuous GPU execution",
    )
    return {
        "device_selection": next(iter(unique_selections))
        if unique_selections else None,
        "device_selection_reemitted": bool(selections),
        "final_gpu_telemetry": telemetry[-1],
    }


def require_rejected_preflight(
    binary: Path, config: Path, expected_message: str | tuple[str, ...]
) -> None:
    result = subprocess.run(
        [
            str(binary),
            "--headless",
            f"--config={config}",
            "--validate-only",
            "--log-profile=quiet",
            "--no-timing",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(result.returncode != 0, f"{config.name}: invalid profile was accepted")
    expected_messages = (
        (expected_message,) if isinstance(expected_message, str)
        else expected_message
    )
    require(
        any(message in result.stdout for message in expected_messages),
        f"{config.name}: wrong rejection\n{result.stdout[-8000:]}",
    )


def final_frame(run_dir: Path) -> Path:
    frames = sorted(
        path
        for path in run_dir.iterdir()
        if path.is_dir() and path.name.startswith("step_")
    )
    require(frames, f"{run_dir.name}: missing output frame")
    return frames[-1]


def inspect_galilean_pair(stationary: Path, moving: Path, duration_s: int) -> dict:
    stationary_frame = final_frame(stationary)
    moving_frame = final_frame(moving)
    result = {
        field: domain_gate.compare_field(
            stationary_frame / f"{field}.npy",
            moving_frame / f"{field}.npy",
            field,
        )
        for field in domain_gate.FIELDS
    }
    failures = [field for field, metrics in result.items() if not metrics["passed"]]
    require(not failures, "Galilean endpoint mismatch: " + ", ".join(failures))

    manifest = domain_gate.load_json(moving_frame / "manifest.json")
    frame = manifest.get("grid", {}).get("horizontal_frame", {})
    require(frame.get("motion_mode") == "constant_velocity", "moving frame mode missing")
    require(frame.get("frame_velocity_earth_ms") == [7, -3], "moving frame velocity wrong")
    expected_origin = [7 * duration_s, -3 * duration_s]
    origin = frame.get("grid_origin_earth_m")
    require(
        isinstance(origin, list)
        and len(origin) == 2
        and all(math.isclose(float(origin[n]), expected_origin[n], abs_tol=1.0e-9)
                for n in range(2)),
        f"moving Earth origin is wrong: {origin!r}",
    )
    return result


def require_field_identity(left: Path, right: Path, label: str) -> dict[str, bool]:
    left_frame = final_frame(left)
    right_frame = final_frame(right)
    result = {
        field: (left_frame / f"{field}.npy").read_bytes()
        == (right_frame / f"{field}.npy").read_bytes()
        for field in restart_gate.CORE_FIELDS
    }
    require(
        all(result.values()),
        f"{label} changed fields: "
        + ", ".join(field for field, identical in result.items() if not identical),
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin" / "nimbus_sim")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--duration", type=int, default=6)
    parser.add_argument("--checkpoint", type=int, default=3)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    config = args.config.resolve()
    require(binary.is_file(), f"missing simulator: {binary}")
    require(config.is_file(), f"missing configuration: {config}")
    require(0 < args.checkpoint < args.duration, "checkpoint must precede duration")

    with tempfile.TemporaryDirectory(prefix="nimbus_constant_velocity_vulkan_") as root_text:
        root = Path(root_text)
        moving_text = config.read_text(encoding="utf-8")
        stationary_text = domain_gate.stationary_counterpart(moving_text)
        report_free_text = domain_gate.replace_once(
            moving_text,
            "    scientific_every_steps: 1\n",
            "    scientific_every_steps: 0\n",
        )
        moving_config = root / "moving_report.yaml"
        stationary_config = root / "stationary_report.yaml"
        report_free_config = root / "moving_report_free.yaml"
        agrid_config = root / "moving_vulkan_agrid.yaml"
        auto_config = root / "moving_auto_cgrid.yaml"
        moving_config.write_text(moving_text, encoding="utf-8")
        stationary_config.write_text(stationary_text, encoding="utf-8")
        report_free_config.write_text(report_free_text, encoding="utf-8")
        agrid_config.write_text(
            domain_gate.replace_once(
                domain_gate.simple_agrid_variant(moving_text),
                "    backend: cpu\n",
                "    backend: vulkan\n",
            ),
            encoding="utf-8",
        )
        auto_config.write_text(
            domain_gate.replace_once(
                moving_text, "    backend: cpu\n", "    backend: auto\n"
            ),
            encoding="utf-8",
        )
        require_rejected_preflight(
            binary,
            agrid_config,
            (
                "klemp_wilhelmson_split_cartesian_agrid does not support backend=gpu",
                "microphysics.scheme=kessler is qualified only in an exact public reference profile",
            ),
        )
        require_rejected_preflight(
            binary,
            auto_config,
            "extensions.compute.backend: unknown value 'auto'",
        )

        stationary = root / "stationary"
        moving = root / "moving"
        report_free = root / "moving_report_free"
        resumed = root / "moving_resumed"
        branch_provenance: dict[str, dict[str, object]] = {}
        branch_provenance["stationary_report_bearing"] = run(
            binary, stationary_config, stationary, args.duration
        )
        branch_provenance["moving_report_bearing"] = run(
            binary, moving_config, moving, args.duration, args.checkpoint
        )
        branch_provenance["moving_report_free"] = run(
            binary, report_free_config, report_free, args.duration
        )
        branch = domain_gate.checkpoint_at(moving, float(args.checkpoint))
        branch_provenance["moving_resumed"] = run(
            binary,
            moving_config,
            resumed,
            args.duration,
            args.checkpoint,
            branch,
        )
        selections = {
            str(value["device_selection"])
            for value in branch_provenance.values()
            if value["device_selection"] is not None
        }
        require(len(selections) == 1, "fresh branches selected different devices")
        qualified_device_selection = next(iter(selections))

        galilean = inspect_galilean_pair(stationary, moving, args.duration)
        report_identity = require_field_identity(
            moving, report_free, "report-bearing/report-free execution"
        )
        restart_identity = require_field_identity(
            moving, resumed, "midpoint restart"
        )
        moving_end = domain_gate.checkpoint_at(moving, float(args.duration))
        resumed_end = domain_gate.checkpoint_at(resumed, float(args.duration))
        checkpoint_identical = moving_end.read_bytes() == resumed_end.read_bytes()
        require(checkpoint_identical, "midpoint restart changed final checkpoint bytes")
        restart_gate.require_restart_suffix(moving, resumed)
        owner_checks = restart_gate.require_report_owner_identities(moving)
        owner_checks += restart_gate.require_report_owner_identities(resumed)
        ledgers = domain_gate.inspect_completed_ledgers(moving)

        report = {
            "schema": "nimbus.qualification.constant_velocity_vulkan",
            "schema_version": 1,
            "status": "pass",
            "duration_s": args.duration,
            "checkpoint_s": args.checkpoint,
            "galilean_endpoint": galilean,
            "report_identity": report_identity,
            "restart_identity": restart_identity,
            "restart_checkpoint_bytes": checkpoint_identical,
            "owner_identity_checks": owner_checks,
            "completed_ledgers": ledgers,
            "fail_closed_preflights": {
                "vulkan_a_grid": True,
                "auto_backend": True,
            },
            "provenance": {
                "binary": str(binary),
                "binary_sha256": sha256_file(binary),
                "source_config": str(config),
                "source_config_sha256": sha256_file(config),
                "qualification_script_sha256": sha256_file(Path(__file__)),
                "shader_manifest_sha256": sha256_file(
                    ROOT / "vulkan" / "shaders" / "manifest.json"
                ),
                "qualified_device_selection": qualified_device_selection,
                "branches": branch_provenance,
            },
        }

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    suffix = f" report={args.report}" if args.report is not None else ""
    print(
        "[constant-velocity-vulkan] PASS: physical C-grid Galilean pair; "
        f"report-free identity; exact {args.checkpoint}s restart; "
        f"owner_identity_checks={owner_checks}; ledgers={len(ledgers)}; "
        f"fail_closed_preflights=2;{suffix}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"[constant-velocity-vulkan] FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
