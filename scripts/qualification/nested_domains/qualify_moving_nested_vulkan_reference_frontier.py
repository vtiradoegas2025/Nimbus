#!/usr/bin/env python3
"""Extract and formally compare a same-backend accelerator moving frontier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import compare_moving_nested_supercell_reference as comparator
import extract_moving_nested_reference_series as extractor


ROOT = Path(__file__).resolve().parents[3]
BINARY = ROOT / "bin/nimbus_sim_internal_qualification"
MOVING_CONFIG = (
    ROOT / "tests/qualification/fixtures/"
    "moving_nested_supercell_frontier_vulkan_v1.yaml"
)
REFERENCE_CONFIG = (
    ROOT / "tests/fixtures/configs/qualification/"
    "moving_nested_supercell_reference_vulkan_v1.yaml"
)
POLICY = (
    ROOT / "tests/qualification/fixtures/"
    "moving_nested_supercell_reference_frontier_v13.json"
)
MOVING_RUN = (
    ROOT / "tests/artifacts/"
    "moving_nested_vulkan_lifetime_v2_20260819_segment_0000_4500"
)
MOVING_CHAIN = (
    ROOT / "tests/artifacts/"
    "moving_nested_vulkan_lifetime_v2_20260819_chain_report.json"
)
REFERENCE_RUN = (
    ROOT / "tests/artifacts/moving_nested_vulkan_reference_v1_20260819"
)
REFERENCE_RUNTIME_REPORT = (
    REFERENCE_RUN / "moving_nested_vulkan_reference_runtime_report.json"
)
DEFAULT_OUTPUT = (
    ROOT / "tests/artifacts/"
    "moving_nested_vulkan_reference_frontier_v13_20260819"
)
SCHEMA = "nimbus.qualification.moving_nested_vulkan_reference_frontier"
SUPPORTED_BACKENDS = ("vulkan", "metal")


def backend_paths(compute_backend: str) -> dict[str, Path]:
    require(compute_backend in SUPPORTED_BACKENDS,
            f"unsupported compute backend: {compute_backend}")
    if compute_backend == "vulkan":
        return {
            "moving_config": MOVING_CONFIG,
            "reference_config": REFERENCE_CONFIG,
            "policy": POLICY,
            "moving_run": MOVING_RUN,
            "moving_chain": MOVING_CHAIN,
            "reference_run": REFERENCE_RUN,
            "reference_report": REFERENCE_RUNTIME_REPORT,
            "output": DEFAULT_OUTPUT,
        }
    moving_run = (
        ROOT / "tests/artifacts/"
        "moving_nested_metal_lifetime_v1_segment_0000_4500")
    reference_run = ROOT / "tests/artifacts/moving_nested_metal_reference_v1"
    return {
        "moving_config": (
            ROOT / "tests/qualification/fixtures/"
            "moving_nested_supercell_frontier_metal_v1.yaml"),
        "reference_config": (
            ROOT / "tests/fixtures/configs/qualification/"
            "moving_nested_supercell_reference_metal_v1.yaml"),
        "policy": (
            ROOT / "tests/qualification/fixtures/"
            "moving_nested_supercell_reference_frontier_v14.json"),
        "moving_run": moving_run,
        "moving_chain": (
            ROOT / "tests/artifacts/"
            "moving_nested_metal_lifetime_v1_chain_report.json"),
        "reference_run": reference_run,
        "reference_report": (
            reference_run /
            "moving_nested_metal_reference_runtime_report.json"),
        "output": (
            ROOT / "tests/artifacts/"
            "moving_nested_metal_reference_frontier_v14"),
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    require(path.is_file(), f"missing artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_runtime_inputs(binary: Path, moving_run: Path,
                            moving_chain_path: Path, reference_run: Path,
                            reference_report_path: Path,
                            compute_backend: str = "vulkan",
                            moving_config: Path = MOVING_CONFIG,
                            reference_config: Path = REFERENCE_CONFIG
                            ) -> tuple[dict, dict, str]:
    binary_digest = sha256(binary)
    chain = load_json(moving_chain_path)
    require(chain.get("schema") ==
                f"nimbus.diagnostic.moving_nested_{compute_backend}_lifetime_chain" and
            chain.get("status") == "complete" and
            chain.get("progress", {}).get("completed_through_s") == 4500.0 and
            chain.get("progress", {}).get("tracker_samples") == 150 and
            chain.get("progress", {}).get("checkpoint_count") == 10 and
            chain.get("claims", {}).get("uninterrupted_single_process") is True and
            chain.get("claims", {}).get(
                f"fallback_off_physical_{compute_backend}") is True and
            chain.get("inputs", {}).get("binary_sha256") == binary_digest,
            f"moving {compute_backend} lifetime is not the completed "
            "same-binary campaign")
    reference = load_json(reference_report_path)
    claims = reference.get("claims", {})
    require(reference.get("status") == "pass" and
            reference.get("scope") ==
                f"same_binary_oversized_physical_{compute_backend}_reference_runtime" and
            reference.get("inputs", {}).get("binary_sha256") == binary_digest and
            reference.get("runtime", {}).get("tracker_samples") == 150 and
            reference.get("runtime", {}).get("diagnostic_samples") == 150 and
            reference.get("artifacts", {}).get("checkpoint_count") == 10 and
            claims.get(
                f"same_binary_as_moving_{compute_backend}_lifetime") is True and
            claims.get(f"fallback_off_physical_{compute_backend}") is True and
            claims.get("native_smagorinsky_sgs_dispatch") is True and
            claims.get("native_kessler_process_dispatch") is True,
            f"oversized {compute_backend} reference runtime is incomplete")
    require(sha256(moving_run / "qualification_input.yaml") ==
                sha256(moving_config) and
            sha256(reference_run / "qualification_input.yaml") ==
                sha256(reference_config),
            "runtime configuration snapshots changed")
    return chain, reference, binary_digest


def execute(output: Path, binary: Path, moving_run: Path,
            reference_run: Path, policy_path: Path,
            compute_backend: str = "vulkan",
            moving_config: Path = MOVING_CONFIG,
            reference_config: Path = REFERENCE_CONFIG) -> dict[str, Any]:
    policy = comparator.validate_policy(load_json(policy_path), policy_path)
    expected_policy_version = 13 if compute_backend == "vulkan" else 14
    expected_signature = (
        comparator.VULKAN_PHYSICS_SIGNATURE_SHA256
        if compute_backend == "vulkan" else
        comparator.METAL_PHYSICS_SIGNATURE_SHA256)
    require(policy.get("schema_version") == expected_policy_version and
            policy.get("expected_contract", {}).get("physics_signature_sha256") ==
                expected_signature,
            f"comparison policy is not the ratified {compute_backend} amendment")
    comparison_root = output / "comparison"
    comparison_root.mkdir(parents=True, exist_ok=True)
    moving_series = comparison_root / "moving_child.json"
    moving_source = comparison_root / "moving_child.source_manifest.json"
    reference_series = comparison_root / "oversized_reference.json"
    reference_source = comparison_root / "oversized_reference.source_manifest.json"
    comparison_report = comparison_root / "comparison_report.json"

    extractor.extract(
        moving_run, "moving_child", moving_config, binary, policy_path,
        moving_series, moving_source)
    extractor.extract(
        reference_run, "oversized_reference", reference_config, binary,
        policy_path, reference_series, reference_source, moving_series)
    comparison = comparator.compare_paths(
        moving_series, reference_series, policy_path)
    write_json(comparison_report, comparison)
    require(comparison.get("status") == "pass" and
            comparison.get("qualification_eligible") is True and
            len(comparison.get("checks", [])) == 21 and
            all(item.get("passed") is True for item in comparison["checks"]),
            f"same-backend {compute_backend} comparison did not pass all 21 checks")
    return {
        "comparison": comparison,
        "artifacts": {
            "moving_series": artifact(moving_series),
            "moving_source_manifest": artifact(moving_source),
            "reference_series": artifact(reference_series),
            "reference_source_manifest": artifact(reference_source),
            "comparison_report": artifact(comparison_report),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compute-backend", choices=SUPPORTED_BACKENDS, default="vulkan")
    parser.add_argument("--binary", type=Path, default=BINARY)
    parser.add_argument("--moving-run", type=Path)
    parser.add_argument("--moving-chain", type=Path)
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    defaults = backend_paths(args.compute_backend)
    binary = args.binary.resolve()
    moving_run = (args.moving_run or defaults["moving_run"]).resolve()
    moving_chain = (args.moving_chain or defaults["moving_chain"]).resolve()
    reference_run = (args.reference_run or defaults["reference_run"]).resolve()
    reference_report = (
        args.reference_report or defaults["reference_report"]).resolve()
    policy = (args.policy or defaults["policy"]).resolve()
    output = (args.output or defaults["output"]).resolve()
    moving_config = defaults["moving_config"].resolve()
    reference_config = defaults["reference_config"].resolve()
    for path in (binary, moving_chain, reference_report, policy,
                 moving_config, reference_config):
        require(path.is_file(), f"required input is missing: {path}")
    chain, reference, binary_digest = validate_runtime_inputs(
        binary, moving_run, moving_chain, reference_run, reference_report,
        args.compute_backend, moving_config, reference_config)
    result = execute(
        output, binary, moving_run, reference_run, policy,
        args.compute_backend, moving_config, reference_config)
    comparison = result["comparison"]
    report = {
        "schema":
            f"nimbus.qualification.moving_nested_{args.compute_backend}_reference_frontier",
        "schema_version": 1,
        "status": "pass",
        "qualification_eligible": True,
        "scope": (
            f"same_binary_same_backend_physical_{args.compute_backend}_"
            "moving_reference_frontier"),
        "claims": {
            "same_binary_candidate_and_reference": True,
            f"same_{args.compute_backend}_backend_candidate_and_reference": True,
            f"fallback_off_physical_{args.compute_backend}_both_branches": True,
            "formal_reference_comparison_complete": True,
            "public_moving_nest_admitted": False,
            "device_resident_relocation": False,
            "throughput_promotion": False,
        },
        "inputs": {
            "binary_sha256": binary_digest,
            "policy": artifact(policy),
            "moving_lifetime_chain": artifact(moving_chain),
            "moving_runtime_report": artifact(
                moving_run /
                f"moving_nested_{args.compute_backend}_runtime_report.json"),
            "reference_runtime_report": artifact(reference_report),
        },
        "coverage": comparison["coverage"],
        "metrics": comparison["metrics"],
        "checks": comparison["checks"],
        "comparison_artifacts": result["artifacts"],
        "runtime_summary": {
            "moving_tracker_samples": chain["progress"]["tracker_samples"],
            "moving_checkpoint_count": chain["progress"]["checkpoint_count"],
            "moving_relocation_count": chain["progress"]["relocation_count"],
            "reference_tracker_samples": reference["runtime"]["tracker_samples"],
            "reference_checkpoint_count": reference["artifacts"]["checkpoint_count"],
            "reference_gpu_dispatches":
                reference["runtime"]["gpu_telemetry"]["dispatches"],
        },
    }
    report_path = output / "campaign_report.json"
    write_json(report_path, report)
    print(json.dumps({
        "status": report["status"],
        "qualification_eligible": report["qualification_eligible"],
        "checks_passed": sum(item["passed"] for item in report["checks"]),
        "checks_total": len(report["checks"]),
        "output": relative(report_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError,
            comparator.ContractError, extractor.ExtractionError) as exc:
        print(f"[moving-nested-accelerator-reference-frontier] FAIL: {exc}")
        raise SystemExit(1)
