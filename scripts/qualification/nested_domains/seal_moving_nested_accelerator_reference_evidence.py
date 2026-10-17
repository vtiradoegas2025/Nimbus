#!/usr/bin/env python3
"""Seal same-backend moving/reference runtime evidence before comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import qualify_moving_nested_vulkan_reference as reference


ROOT = Path(__file__).resolve().parents[3]
SUPPORTED_BACKENDS = ("vulkan", "metal")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def sha256(path: Path) -> str:
    require(path.is_file(), f"missing artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": relative(path),
        "sha256": sha256(path),
    }


def build_evidence(compute_backend: str, binary: Path, config: Path,
                   moving_chain_path: Path,
                   reference_report_path: Path) -> dict[str, Any]:
    require(compute_backend in SUPPORTED_BACKENDS,
            f"unsupported compute backend: {compute_backend}")
    binary = binary.resolve()
    config = config.resolve()
    moving_chain_path = moving_chain_path.resolve()
    reference_report_path = reference_report_path.resolve()
    reference.validate_config_delta(config, compute_backend)
    binary_digest = reference.validate_binary_identity(
        binary, moving_chain_path, compute_backend)
    chain = load_json(moving_chain_path)
    report = load_json(reference_report_path)
    claims = report.get("claims", {})
    inputs = report.get("inputs", {})
    runtime = report.get("runtime", {})
    telemetry = runtime.get("gpu_telemetry", {})
    artifacts = report.get("artifacts", {})
    require(
        report.get("schema") ==
            f"nimbus.diagnostic.moving_nested_{compute_backend}_reference_runtime" and
        report.get("schema_version") == 1 and
        report.get("status") == "pass" and
        report.get("qualification_eligible") is False and
        report.get("scope") ==
            f"same_binary_oversized_physical_{compute_backend}_reference_runtime",
        "reference report is not the complete same-backend runtime",
    )
    require(
        inputs.get("binary_sha256") == binary_digest and
        inputs.get("config_sha256") == sha256(config) and
        inputs.get("moving_lifetime_chain_sha256") == sha256(moving_chain_path) and
        inputs.get("duration_s") == 4500,
        "reference report input identity changed",
    )
    require(
        claims.get(f"same_binary_as_moving_{compute_backend}_lifetime") is True and
        claims.get(f"fallback_off_physical_{compute_backend}") is True and
        claims.get("native_smagorinsky_sgs_dispatch") is True and
        claims.get("native_kessler_process_dispatch") is True and
        claims.get("host_authoritative_kessler_state_commit") is True and
        claims.get("strict_finite_bounded_outputs") is True and
        claims.get("formal_comparison_complete") is False,
        "reference report backend/science claims changed",
    )
    require(
        runtime.get("measured_steps") == telemetry.get("model_steps") and
        runtime.get("measured_simulated_seconds") == 4500.0 and
        runtime.get("tracker_samples") == 150 and
        runtime.get("diagnostic_samples") == 150 and
        type(telemetry.get("command_buffers")) is int and
        telemetry["command_buffers"] > 0 and
        type(telemetry.get("dispatches")) is int and
        telemetry["dispatches"] > 0 and
        type(telemetry.get("resident_hits")) is int and
        telemetry["resident_hits"] >= 0 and
        type(telemetry.get("resident_misses")) is int and
        telemetry["resident_misses"] >= 0 and
        isinstance(telemetry.get("stage_dispatches"), dict) and
        telemetry["stage_dispatches"],
        "reference runtime cadence or physical telemetry is incomplete",
    )
    require(
        artifacts.get("checkpoint_count") == 10 and
        artifacts.get("checkpoint_times_s") ==
            [float(value) for value in range(450, 4501, 450)] and
        isinstance(artifacts.get("final_checkpoint_sha256"), str) and
        len(artifacts["final_checkpoint_sha256"]) == 64,
        "reference checkpoint evidence is incomplete",
    )
    return {
        "schema":
            f"nimbus.evidence.moving_nested_{compute_backend}_reference_runtime",
        "schema_version": 1,
        "status": "pass",
        "qualification_eligible": False,
        "scope": (
            f"same_binary_oversized_physical_{compute_backend}_reference_runtime_"
            "before_blinded_comparison"),
        "candidate_metric_values_inspected": False,
        "inputs": {
            "binary_sha256": binary_digest,
            "reference_config": artifact(config),
            "moving_lifetime_chain": artifact(moving_chain_path),
        },
        "reference_runtime": {
            "report": artifact(reference_report_path),
            "fresh_configured_start": True,
            "uninterrupted_single_process": True,
            "final_time_s": 4500,
            "model_steps": telemetry["model_steps"],
            "tracker_samples": runtime["tracker_samples"],
            "diagnostic_samples": runtime["diagnostic_samples"],
            "portable_checkpoints": artifacts["checkpoint_count"],
            "simulated_seconds_per_solver_wall_second":
                runtime.get("simulated_seconds_per_solver_wall_second"),
            "final_checkpoint_sha256": artifacts["final_checkpoint_sha256"],
        },
        "device_evidence": {
            f"fallback_off_physical_{compute_backend}": True,
            "model_steps": telemetry["model_steps"],
            "command_buffers": telemetry["command_buffers"],
            "dispatches": telemetry["dispatches"],
            "resident_hits": telemetry["resident_hits"],
            "resident_misses": telemetry["resident_misses"],
            "stage_dispatches": telemetry["stage_dispatches"],
            "host_authoritative_kessler_state_commit": True,
        },
        "claims": {
            "same_binary_candidate_and_reference": True,
            f"same_{compute_backend}_backend_candidate_and_reference": True,
            "same_comparison_level_equations": True,
            "strict_finite_bounded_reference_outputs": True,
            "formal_comparison_complete": False,
            "public_moving_nest_admitted": False,
            "device_resident_relocation": False,
            "throughput_promotion": False,
        },
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compute-backend", choices=SUPPORTED_BACKENDS, default="vulkan")
    parser.add_argument("--binary", type=Path, default=reference.DEFAULT_BINARY)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--moving-chain", type=Path)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = (args.config or reference.default_config(args.compute_backend)).resolve()
    moving_chain = (
        args.moving_chain or reference.default_chain(args.compute_backend)).resolve()
    reference_run = reference.default_output(args.compute_backend)
    reference_report = (args.reference_report or reference_run / (
        f"moving_nested_{args.compute_backend}_reference_runtime_report.json"
    )).resolve()
    output = (args.output or ROOT / "reports/qualification" / (
        f"moving_nested_{args.compute_backend}_reference_runtime_v1_pass.json"
    )).resolve()
    evidence = build_evidence(
        args.compute_backend, args.binary.resolve(), config,
        moving_chain, reference_report)
    write_json(output, evidence)
    print(json.dumps({
        "status": evidence["status"],
        "qualification_eligible": evidence["qualification_eligible"],
        "candidate_metric_values_inspected":
            evidence["candidate_metric_values_inspected"],
        "output": relative(output),
        "sha256": sha256(output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"[moving-nested-accelerator-reference-evidence] FAIL: {exc}",
              file=sys.stderr)
        raise SystemExit(1)
