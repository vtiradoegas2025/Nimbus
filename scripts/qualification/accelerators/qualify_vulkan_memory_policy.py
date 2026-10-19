#!/usr/bin/env python3
"""Qualify direct/staged Vulkan memory policy on one physical device."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[3]
WRAPPER = ROOT / "scripts" / "run_with_vulkan_env.sh"
REQUIRED = ROOT / "bin" / "test_vulkan_required_dispatch"
PARITY = ROOT / "bin" / "test_vulkan_gpu_parity"
DEFAULT_REPORT = (
    ROOT / "docs" / "evidence" / "vulkan_memory_policy_m4_v1_pass.json"
)
PASS_RE = re.compile(
    r"All tests passed \((?P<assertions>\d+) assertions in "
    r"(?P<cases>\d+) test cases?\)"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    binary: Path, policy: str, *, timing: bool = False
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["NIMBUS_VULKAN_MEMORY_POLICY"] = policy
    if timing:
        environment["NIMBUS_GPU_KERNEL_TIMING"] = "1"
    return subprocess.run(
        [str(WRAPPER), str(binary)],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def require_pass(
    result: subprocess.CompletedProcess[str], label: str
) -> dict[str, int]:
    match = PASS_RE.search(result.stdout)
    if result.returncode != 0 or match is None:
        raise RuntimeError(f"{label} failed:\n{result.stdout}")
    return {
        "test_cases": int(match.group("cases")),
        "assertions": int(match.group("assertions")),
    }


def require_telemetry(
    result: subprocess.CompletedProcess[str], label: str
) -> dict[str, int]:
    lines = [
        line for line in result.stdout.splitlines()
        if line.startswith("[GPU TELEMETRY]")
    ]
    if not lines:
        raise RuntimeError(f"{label} emitted no GPU telemetry:\n{result.stdout}")
    values = {
        key: int(value)
        for key, value in re.findall(r"([a-z_]+)=(\d+)", lines[-1])
    }
    required = {
        "command_buffers", "dispatches", "device_upload_bytes",
        "device_download_bytes", "staging_memory_allocations",
    }
    if not required.issubset(values):
        raise RuntimeError(f"{label} telemetry is incomplete: {lines[-1]}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    for required in (WRAPPER, REQUIRED, PARITY):
        if not required.is_file():
            raise RuntimeError(f"required input is missing: {required}")

    staged_required_run = run(REQUIRED, "staged", timing=True)
    staged_required = require_pass(
        staged_required_run, "forced-staged required dispatch"
    )
    staged_telemetry = require_telemetry(
        staged_required_run, "forced-staged required dispatch"
    )
    if (
        staged_telemetry["device_upload_bytes"] <= 0
        or staged_telemetry["device_download_bytes"] <= 0
        or staged_telemetry["staging_memory_allocations"] <= 0
    ):
        raise RuntimeError(
            f"forced-staged transfer telemetry is vacuous: {staged_telemetry}"
        )
    staged_parity = require_pass(
        run(PARITY, "staged"), "forced-staged full parity"
    )
    direct_required_run = run(REQUIRED, "direct", timing=True)
    direct_required = require_pass(
        direct_required_run, "forced-direct required dispatch"
    )
    direct_telemetry = require_telemetry(
        direct_required_run, "forced-direct required dispatch"
    )
    if (
        direct_telemetry["device_upload_bytes"] != 0
        or direct_telemetry["device_download_bytes"] != 0
        or direct_telemetry["staging_memory_allocations"] != 0
    ):
        raise RuntimeError(
            f"forced-direct transfer telemetry is not zero-copy: "
            f"{direct_telemetry}"
        )

    invalid = run(REQUIRED, "shared")
    invalid_marker = "invalid NIMBUS_VULKAN_MEMORY_POLICY='shared'"
    normalized_invalid = " ".join(invalid.stdout.split())
    if invalid.returncode == 0 or invalid_marker not in normalized_invalid:
        raise RuntimeError(
            "invalid memory policy did not fail at initialization:\n"
            f"{invalid.stdout}"
        )

    report = {
        "schema": "nimbus.vulkan.memory_policy_qualification",
        "schema_version": 1,
        "status": "pass",
        "inputs": {
            "required_dispatch_binary": {
                "path": str(REQUIRED.relative_to(ROOT)),
                "sha256": sha256_file(REQUIRED),
            },
            "parity_binary": {
                "path": str(PARITY.relative_to(ROOT)),
                "sha256": sha256_file(PARITY),
            },
        },
        "checks": {
            "forced_staged_required_dispatch": staged_required,
            "forced_staged_transfer_telemetry": staged_telemetry,
            "forced_staged_full_parity": staged_parity,
            "forced_direct_required_dispatch": direct_required,
            "forced_direct_transfer_telemetry": direct_telemetry,
            "invalid_policy_rejected_before_dispatch": True,
        },
        "claims": {
            "physical_vulkan_device_required": True,
            "forced_staged_transfer_path": True,
            "forced_direct_shared_path": True,
            "noncoherent_range_logic_unit_covered": True,
            "physical_noncoherent_memory_observed": False,
            "physical_nvidia_or_amd_qualified": False,
        },
    }
    output = args.report.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "[vulkan-memory-policy] PASS: forced staged/direct dispatch, "
        "full staged parity, and invalid-policy rejection"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
