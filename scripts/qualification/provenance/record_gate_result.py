#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Final


SCHEMA = "nimbus.release_gate_result"
VERSION = 2
LEGACY_VERSION = 1

REQUIRED_GATE_POLICY: Final[dict[str, dict[str, object]]] = {
    "test-catalog-authority": {
        "command": ("make", "test-catalog-authority"),
        "requiredHardware": False,
        "artifacts": (),
    },
    "test-catalog-reachable-matrix": {
        "command": ("make", "test-catalog-reachable-matrix"),
        "requiredHardware": False,
        "artifacts": ("reports/pre_beta/reachable_set_v1.json",),
    },
    "test-catalog-studio-native-parity": {
        "command": ("make", "test-catalog-studio-native-parity"),
        "requiredHardware": False,
        "artifacts": (),
    },
    "qualify-catalog-bounded-runs": {
        "command": ("make", "qualify-catalog-bounded-runs"),
        "requiredHardware": True,
        "artifacts": (),
    },
    "qualify-catalog-restart-threading": {
        "command": ("make", "qualify-catalog-restart-threading"),
        "requiredHardware": False,
        "artifacts": (),
    },
    "qualify-catalog-physical-backends": {
        "command": ("make", "qualify-catalog-physical-backends"),
        "requiredHardware": True,
        "artifacts": (
            "reports/pre_beta/physical/moving_nested_public_vulkan_admission_v1_pass.json",
            "reports/pre_beta/physical/moving_nested_vulkan_public_runtime_v1_pass.json",
            "reports/pre_beta/physical/moving_nested_metal_public_runtime_smoke_v1_pass.json",
            "bin/nimbus_sim_physical_qualification",
            "bin/nimbus_sim_internal_qualification",
            "tests/artifacts/moving_nested_reference_frontier_v12_20260818/moving_child/checkpoints/checkpoint_step_000002700.nimbus",
            "tests/artifacts/moving_nested_reference_frontier_v12_20260818/moving_child/checkpoints/checkpoint_step_000002700.nimbus.json",
        ),
    },
    "qualify-io": {
        "command": ("make", "qualify-io"),
        "requiredHardware": False,
        "artifacts": (),
    },
    "qualify-release-final-surface": {
        "command": ("make", "qualify-release-final-surface"),
        "requiredHardware": False,
        "artifacts": (),
    },
}


class GateResultError(RuntimeError):
    pass


def release_input_identity(
    root: Path,
    sources: list[Path],
) -> tuple[dict[str, object], dict[str, object]]:
    if not sources:
        raise GateResultError("receipt v2 requires release source roots")
    from scripts.qualification.provenance import release_manifest

    try:
        inventory = release_manifest.inventory_record(sources, root)
    except release_manifest.ManifestError as exc:
        raise GateResultError(str(exc)) from exc
    source = {
        "roots": inventory["roots"],
        "count": inventory["count"],
        "sha256": inventory["sha256"],
    }
    return release_manifest.git_identity(root), source


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_file(path: Path, root: Path) -> tuple[Path, str]:
    candidate = path if path.is_absolute() else root / path
    if candidate.is_symlink():
        raise GateResultError(f"input is a symbolic link: {candidate}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise GateResultError(f"input is not a regular file: {candidate}")
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise GateResultError(f"input is outside the receipt root: {candidate}") from exc
    return resolved, relative


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def forbidden_skip_lines(output: str) -> list[str]:
    return [
        line for line in output.splitlines()
        if "[SKIP]" in line.upper() or "[SKIPPED]" in line.upper()
    ]


def fallback_used(output: str) -> bool:
    lowered = output.lower()
    forbidden = (
        "fallback_active=yes",
        "fallback_active=true",
        "fallback=on",
        "fallback=true",
        "fallback=cpu",
        "falling back to cpu",
    )
    return any(marker in lowered for marker in forbidden)


def canonical_gate_command(target: str, command: list[str]) -> bool:
    policy = REQUIRED_GATE_POLICY.get(target)
    if policy is None or not command:
        return False
    expected = tuple(policy["command"])
    return (
        Path(command[0]).name == expected[0]
        and tuple(command[1:]) == expected[1:]
    )


def relative_artifact_paths(paths: list[Path], root: Path) -> list[str]:
    result: list[str] = []
    for artifact in paths:
        _, relative = checked_file(artifact, root)
        result.append(relative)
    if len(set(result)) != len(result):
        raise GateResultError("gate artifacts must be unique")
    return sorted(result)


def require_gate_policy(
    target: str,
    command: list[str],
    required_hardware: bool,
    artifacts: list[str],
) -> None:
    policy = REQUIRED_GATE_POLICY.get(target)
    if policy is None:
        raise GateResultError(f"gate target is not release-policy-owned: {target}")
    if not canonical_gate_command(target, command):
        raise GateResultError(f"gate command is not canonical for {target}")
    if required_hardware is not policy["requiredHardware"]:
        raise GateResultError(f"gate hardware policy does not match {target}")
    expected_artifacts = sorted(str(value) for value in policy["artifacts"])
    if artifacts != expected_artifacts:
        raise GateResultError(f"gate artifact policy does not match {target}")


def record_gate(
    *, root: Path, target: str, binary: Path, command: list[str],
    required_hardware: bool, log: Path, receipt: Path,
    artifacts: list[Path], sources: list[Path],
) -> dict[str, object]:
    root = root.resolve(strict=True)
    if not target.strip() or not command or not all(command):
        raise GateResultError("gate target and command must not be empty")
    binary_path, binary_relative = checked_file(binary, root)
    binary_sha256 = sha256_file(binary_path)
    artifact_paths = relative_artifact_paths(artifacts, root)
    require_gate_policy(
        target, command, required_hardware, artifact_paths,
    )
    revision, source = release_input_identity(root, sources)
    started = datetime.now(timezone.utc)
    monotonic_start = time.monotonic()
    result = subprocess.run(
        command, cwd=root, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    elapsed = time.monotonic() - monotonic_start
    write_atomic(log, result.stdout)
    output = result.stdout.decode("utf-8", errors="replace")
    skips = forbidden_skip_lines(output)
    used_fallback = fallback_used(output)
    if result.returncode != 0:
        raise GateResultError(
            f"gate {target} exited {result.returncode}; see {log}"
        )
    if skips:
        raise GateResultError(f"gate {target} reported a required skip")
    if used_fallback:
        raise GateResultError(f"gate {target} used a fallback backend")
    if not result.stdout:
        raise GateResultError(f"gate {target} produced an empty log")
    if sha256_file(binary_path) != binary_sha256:
        raise GateResultError(f"gate {target} changed its release binary")
    final_revision, final_source = release_input_identity(root, sources)
    if final_revision != revision or final_source != source:
        raise GateResultError(
            f"gate {target} changed its revision or release source inventory"
        )
    for artifact in artifacts:
        checked_file(artifact, root)
    try:
        log_relative = log.resolve(strict=True).relative_to(root).as_posix()
        receipt.resolve().relative_to(root)
    except ValueError as exc:
        raise GateResultError("receipt/log path is outside the receipt root") from exc
    record: dict[str, object] = {
        "schema": SCHEMA,
        "version": VERSION,
        "target": target,
        "command": command,
        "status": "pass",
        "startedUtc": started.isoformat(),
        "completedUtc": datetime.now(timezone.utc).isoformat(),
        "wallSeconds": elapsed,
        "requiredHardware": required_hardware,
        "skippedCount": 0,
        "fallbackUsed": False,
        "binary": binary_relative,
        "binarySha256": binary_sha256,
        "log": log_relative,
        "logSha256": sha256_file(log),
        "artifacts": artifact_paths,
        "revision": revision,
        "source": source,
    }
    write_atomic(
        receipt,
        (json.dumps(record, indent=2, sort_keys=True) + "\n").encode(),
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--target", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--required-hardware", action="store_true")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--artifact", action="append", type=Path, default=[])
    parser.add_argument("--source", action="append", type=Path, default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        record_gate(
            root=args.root, target=args.target, binary=args.binary,
            command=command, required_hardware=args.required_hardware,
            log=args.log, receipt=args.receipt, artifacts=args.artifact,
            sources=args.source,
        )
        print(f"[release-gate-receipt] PASS {args.target}: {args.receipt}")
    except (GateResultError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
