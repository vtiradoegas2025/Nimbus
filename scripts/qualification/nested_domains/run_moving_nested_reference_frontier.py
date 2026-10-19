#!/usr/bin/env python3
"""Run the hash-bound moving-child/oversized-reference frontier campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import compare_moving_nested_supercell_reference as comparator
import extract_moving_nested_reference_series as extractor


MOVING_CONFIG = ROOT / "tests/qualification/fixtures/moving_nested_supercell_frontier_v5.yaml"
REFERENCE_CONFIG = ROOT / "tests/fixtures/configs/qualification/moving_nested_supercell_reference_v5.yaml"
SCI325_TUPLE = ROOT / "tests/qualification/fixtures/moving_nested_supercell_lifetime.yaml"
POLICY = (
    ROOT / "tests/qualification/fixtures/"
    "moving_nested_supercell_reference_frontier_v12.json"
)
BINARY = ROOT / "bin/nimbus_sim_internal_qualification"
REPORT_SCHEMA = "nimbus.qualification.moving_nested_reference_frontier_campaign"
HEX64 = re.compile(r"[0-9a-f]{64}")
ALLOWED_MOVING_ADDITIONS = {
    "output.qualification_diagnostics.preset": "moving_reference_v1",
    "output.qualification_diagnostics.interval_s": "30",
}


class CampaignError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignError(message)


def sha256(path: Path) -> str:
    require(path.is_file(), f"missing artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"invalid JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def seal_report(report: dict[str, Any]) -> dict[str, Any]:
    require("campaign_payload_sha256" not in report,
            "campaign report was already sealed")
    sealed = dict(report)
    sealed["campaign_payload_sha256"] = canonical_sha256(report)
    return sealed


def verify_sealed_report(report: dict[str, Any]) -> bool:
    expected = report.get("campaign_payload_sha256")
    payload = dict(report)
    payload.pop("campaign_payload_sha256", None)
    return isinstance(expected, str) and expected == canonical_sha256(payload)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def exact_moving_tuple(moving: Path, baseline: Path) -> dict[str, Any]:
    current = extractor.flat_yaml_scalars(moving)
    expected = extractor.flat_yaml_scalars(baseline)
    additions = {key: current.pop(key, None) for key in ALLOWED_MOVING_ADDITIONS}
    require(additions == ALLOWED_MOVING_ADDITIONS,
            "moving frontier config lacks the exact 30-second moving_reference_v1 stream")
    missing = sorted(set(expected) - set(current))
    extra = sorted(set(current) - set(expected))
    changed = sorted(key for key in set(current) & set(expected)
                     if current[key] != expected[key])
    require(not missing and not extra and not changed,
            "moving frontier config drifted from the SCI-325 tuple: "
            f"missing={missing}, extra={extra}, changed={changed}")
    return {
        "baseline": artifact(baseline),
        "allowed_additions": additions,
        "flattened_tuple_sha256": canonical_sha256(current),
    }


def validate_binary_config(binary: Path, config: Path, moving: bool) -> None:
    command = [str(binary)]
    if moving:
        command.append("--qualification-moving-nest")
    command.extend(["--validate-only", "--config", str(config)])
    result = subprocess.run(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False)
    require(result.returncode == 0,
            f"{('moving' if moving else 'reference')} config validation failed:\n"
            + result.stdout[-16000:])


def validate_inputs(binary: Path, moving: Path, reference: Path,
                    policy_path: Path, validate_binary: bool = True) -> dict[str, Any]:
    for path in (binary, moving, reference, policy_path, SCI325_TUPLE,
                 Path(extractor.__file__), Path(comparator.__file__)):
        require(path.is_file(), f"missing campaign input: {path}")
    tuple_binding = exact_moving_tuple(moving, SCI325_TUPLE)
    policy = comparator.validate_policy(load_json(policy_path), policy_path.resolve())
    times, expected = extractor.expected_times(policy)
    require(policy.get("schema_version") == 12 and
            policy.get("policy_status") == "ratified",
            "production frontier runner requires the post-failure v12 parent-stencil-limit amendment")
    moving_flat = extractor.flat_yaml_scalars(moving)
    reference_flat = extractor.flat_yaml_scalars(reference)
    require(float(moving_flat.get("time.run_time", "nan")) == times[-1] and
            float(reference_flat.get("time.run_time", "nan")) == times[-1],
            "branch run durations differ from the frozen policy duration")
    for label, values in (("moving", moving_flat), ("reference", reference_flat)):
        require(values.get("output.qualification_diagnostics.preset") ==
                    "moving_reference_v1" and
                float(values.get("output.qualification_diagnostics.interval_s", "nan")) ==
                    float(expected["sample_interval_s"]),
                f"{label} config lost the policy diagnostic cadence")

    spacing = expected.get("horizontal_grid_spacing_m")
    require(isinstance(spacing, dict), "policy lacks comparison spacing")
    observed = {"x": float(spacing.get("x", math.nan)),
                "y": float(spacing.get("y", math.nan))}
    moving_values, moving_signature, moving_derivation = extractor.numerical_signature(
        moving, "moving_child", observed)
    reference_values, reference_signature, reference_derivation = (
        extractor.numerical_signature(
            reference, "oversized_reference", observed))
    require(moving_signature == reference_signature and
            moving_values == reference_values,
            "moving child and stationary reference lack one common comparison-level signature")
    require(moving_signature == expected.get("physics_signature_sha256"),
            "common comparison-level signature differs from the ratified policy")
    exclusion = expected.get("relaxation_exclusion_cells")
    trusted_shape = expected.get("trusted_region_shape_cells")
    geometry_plan = load_json(Path(policy["_verified_geometry_plan"]["path"]))
    aperture = load_json(Path(
        policy["_verified_qualification_aperture"]["path"]))
    extractor.validate_geometry_plan(geometry_plan)
    extractor.validate_geometry_plan_sources(
        Path(policy["_verified_geometry_plan"]["path"]), geometry_plan)
    extractor.validate_aperture(aperture)
    extractor.validate_configured_geometry(moving, "moving_child", geometry_plan)
    extractor.validate_configured_geometry(
        reference, "oversized_reference", geometry_plan)
    require(type(exclusion) is int and isinstance(trusted_shape, dict) and
            trusted_shape == aperture.get("shape_cells") and
            expected.get("trusted_region_cell_count") ==
                trusted_shape["x"] * trusted_shape["y"] ==
                aperture.get("cell_count"),
            "ratified trusted-region shape differs from the fixed aperture")
    require(expected.get("grid_phase_origin_m") == {"x": 0.0, "y": 0.0},
            "ratified policy lost the common Earth-grid phase")
    frozen_binary_sha = sha256(binary)
    if validate_binary:
        validate_binary_config(binary, moving, True)
        require(sha256(binary) == frozen_binary_sha,
                "binary changed during moving-config validation")
        validate_binary_config(binary, reference, False)
        require(sha256(binary) == frozen_binary_sha,
                "binary changed during reference-config validation")
    return {
        "policy_status": policy.get("policy_status"),
        "duration_s": times[-1],
        "sample_count": len(times),
        "sample_interval_s": expected["sample_interval_s"],
        "binary": artifact(binary),
        "moving_config": artifact(moving),
        "reference_config": artifact(reference),
        "policy": artifact(policy_path),
        "threshold_policy": policy["_verified_threshold_policy"],
        "prior_policy": policy["_verified_prior_policy"],
        "failure_evidence": policy["_verified_failure_evidence"],
        "geometry_plan": policy["_verified_geometry_plan"],
        "qualification_aperture":
            policy["_verified_qualification_aperture"],
        "extractor": artifact(Path(extractor.__file__)),
        "comparator": artifact(Path(comparator.__file__)),
        "diagnostic_implementation_sources": [
            artifact(ROOT / path) for path in extractor.SOURCE_FILES
        ],
        "moving_tuple": tuple_binding,
        "numerical_signature": {
            "sha256": moving_signature,
            "values": moving_values,
            "moving_derivation": moving_derivation,
            "reference_derivation": reference_derivation,
        },
    }


def require_inputs_unchanged(inputs: dict[str, Any]) -> None:
    records = [
        inputs["binary"], inputs["moving_config"], inputs["reference_config"],
        inputs["policy"], inputs["extractor"], inputs["comparator"],
        inputs["moving_tuple"]["baseline"],
        inputs["threshold_policy"], inputs["prior_policy"],
        inputs["failure_evidence"], inputs["geometry_plan"],
        inputs["qualification_aperture"],
        *inputs["diagnostic_implementation_sources"],
    ]
    for record in records:
        path = Path(str(record["path"]))
        require(path.is_file() and sha256(path) == record["sha256"],
                f"campaign input changed after validation: {path}")


def checkpoint_lifecycle(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing checkpoint: {path}")
    sidecar_path = Path(str(path) + ".json")
    sidecar = load_json(sidecar_path)
    require(sidecar.get("schema") == "nimbus.checkpoint.lifecycle" and
            sidecar.get("schema_version") == 1,
            f"wrong checkpoint lifecycle schema: {sidecar_path}")
    require(sidecar.get("checkpoint_file") == path.name and
            sidecar.get("checkpoint_bytes") == path.stat().st_size,
            f"checkpoint lifecycle identity mismatch: {sidecar_path}")
    step = sidecar.get("completed_steps")
    require(type(step) is int and step >= 0 and
            path.stem == f"checkpoint_step_{step:09d}",
            f"checkpoint step identity mismatch: {sidecar_path}")
    simulation_time = sidecar.get("simulation_time_s")
    require(type(simulation_time) in (int, float) and
            math.isfinite(float(simulation_time)) and float(simulation_time) >= 0.0,
            f"checkpoint time is invalid: {sidecar_path}")
    fnv = sidecar.get("checkpoint_fnv1a64")
    require(isinstance(fnv, str) and re.fullmatch(r"[0-9a-f]{16}", fnv) is not None,
            f"checkpoint FNV identity is invalid: {sidecar_path}")
    require(sidecar.get("boundary") == "completed_timestep" and
            sidecar.get("portable_across_compute_memory_topology") is True,
            f"checkpoint is not a restart-safe completed boundary: {sidecar_path}")
    run_id = sidecar.get("run_id")
    require(isinstance(run_id, str) and run_id,
            f"checkpoint lacks a run identity: {sidecar_path}")
    return {
        "simulation_time_s": float(simulation_time),
        "completed_steps": step,
        "run_id": run_id,
        "parent_run_id": sidecar.get("parent_run_id"),
        "parent_checkpoint": sidecar.get("parent_checkpoint"),
        "checkpoint_fnv1a64": fnv,
        "checkpoint": artifact(path),
        "sidecar": artifact(sidecar_path),
    }


def validate_checkpoints(run: Path, duration_s: float, cadence_s: int,
                         resume_record: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    paths = sorted((run / "checkpoints").glob("checkpoint_step_*.nimbus"))
    require(paths, f"branch produced no restart checkpoint: {run}")
    records = [checkpoint_lifecycle(path) for path in paths]
    start = (resume_record["simulation_time_s"] if resume_record is not None else 0.0)
    expected_times = []
    candidate = (math.floor(start / cadence_s) + 1) * cadence_s
    while candidate <= duration_s:
        expected_times.append(float(candidate))
        candidate += cadence_s
    if not expected_times or expected_times[-1] != duration_s:
        expected_times.append(float(duration_s))
    require([item["simulation_time_s"] for item in records] == expected_times,
            f"checkpoint cadence changed: expected {expected_times}, got "
            f"{[item['simulation_time_s'] for item in records]}")
    if resume_record is not None:
        split_path = str(Path(resume_record["checkpoint"]["path"]).resolve())
        for item in records:
            require(item["parent_run_id"] == resume_record["run_id"] and
                    str(Path(str(item["parent_checkpoint"])).resolve()) == split_path,
                    "resumed checkpoint lineage does not bind to the selected split")
    return records


def validate_output_cadence(run: Path, role: str, config: Path, binary: Path,
                            policy_path: Path) -> dict[str, Any]:
    manifest_path = run / "nimbus_output.json"
    manifest = load_json(manifest_path)
    require(manifest.get("schema") == "nimbus.output" and
            manifest.get("manifest_version") in {6, 7} and
            manifest.get("publication_state") == "ready",
            f"branch output manifest is incomplete: {manifest_path}")
    config_snapshot = run / "qualification_input.yaml"
    digest_snapshot = run / "qualification_binary.sha256"
    log_path = run / "qualification_run.log"
    require(log_path.is_file(), f"branch lacks its captured runtime log: {run}")
    require(sha256(config_snapshot) == sha256(config),
            f"branch config snapshot changed: {run}")
    recorded_binary = digest_snapshot.read_text(encoding="ascii").strip()
    require(HEX64.fullmatch(recorded_binary) is not None and
            recorded_binary == sha256(binary),
            f"branch binary identity changed: {run}")
    policy = comparator.validate_policy(
        load_json(policy_path), policy_path.resolve())
    times, _ = extractor.expected_times(policy)
    if role == "moving_child":
        history = run / "nests/child_0000/qualification_diagnostics"
        kind = "nested_child_qualification_diagnostics"
        require(not (run / "moving_nest_scientific_validity.json").exists(),
                "moving branch published a scientific-invalidity artifact")
    else:
        history = run / "qualification_diagnostics"
        kind = "qualification_diagnostics"
    products = manifest.get("data_products")
    require(isinstance(products, list) and kind in products,
            f"branch manifest does not advertise {kind}")
    steps = extractor.step_publications(history, kind)
    trackers = extractor.tracker_publications(run)
    require(sorted(steps) == times,
            f"{role} diagnostic cadence differs from policy")
    require(sorted(trackers) == times,
            f"{role} tracker cadence differs from policy")
    result = {
        "run_manifest": artifact(manifest_path),
        "config_snapshot": artifact(config_snapshot),
        "binary_digest": artifact(digest_snapshot),
        "runtime_log": artifact(log_path),
        "diagnostic_samples": len(steps),
        "tracker_samples": len(trackers),
        "first_sample_s": times[0],
        "final_sample_s": times[-1],
    }
    branch_marker = run / "nimbus_branch.json"
    if branch_marker.is_file():
        result["branch_marker"] = artifact(branch_marker)
    return result


def validate_run_identity(run: Path, config: Path, binary: Path) -> None:
    snapshot = run / "qualification_input.yaml"
    digest_path = run / "qualification_binary.sha256"
    require(snapshot.is_file() and sha256(snapshot) == sha256(config),
            f"run config identity differs from the frozen branch input: {run}")
    require(digest_path.is_file(), f"run lacks a binary identity: {run}")
    recorded = digest_path.read_text(encoding="ascii").strip()
    require(HEX64.fullmatch(recorded) is not None and recorded == sha256(binary),
            f"run binary identity differs from the frozen campaign binary: {run}")


def prepare_output(path: Path) -> None:
    require(not path.exists() or not any(path.iterdir()),
            f"output directory is not fresh: {path}")
    path.mkdir(parents=True, exist_ok=True)


def run_branch(binary: Path, config: Path, role: str, output: Path,
               duration_s: int, threads: int, checkpoint_every_s: int,
               frozen_binary_sha: str, resume: Path | None = None) -> dict[str, Any]:
    prepare_output(output)
    shutil.copyfile(config, output / "qualification_input.yaml")
    (output / "qualification_binary.sha256").write_text(
        frozen_binary_sha + "\n", encoding="ascii")
    command = [str(binary)]
    if role == "moving_child":
        command.append("--qualification-moving-nest")
    command.extend([
        "--headless", "--config", str(config), "--duration", str(duration_s),
        "--write-every=450", f"--checkpoint-every={checkpoint_every_s}",
        f"--checkpoint-dir={output / 'checkpoints'}", "--threads", str(threads),
        "--guard-mode", "strict", f"--outdir={output}",
        "--log-profile", "normal", "--timing",
    ])
    if resume is not None:
        command.append(f"--resume={resume.resolve()}")
    started = time.monotonic()
    result = subprocess.run(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False)
    wall = time.monotonic() - started
    (output / "qualification_run.log").write_text(result.stdout, encoding="utf-8")
    require(result.returncode == 0,
            f"{role} branch exited {result.returncode}:\n{result.stdout[-16000:]}")
    require(sha256(binary) == frozen_binary_sha,
            f"binary changed while the {role} branch was running")
    return {"command": command, "wall_seconds": wall}


def copy_timed_publications(prefix: Path, suffix: Path, destination: Path,
                            relative: Path, split_time_s: float,
                            manifests: bool) -> None:
    destination_root = destination / relative
    destination_root.mkdir(parents=True, exist_ok=True)
    pattern = "step_*" if manifests else "track_step_*.json"
    chosen: dict[float, Path] = {}
    for source, lower in ((prefix, True), (suffix, False)):
        root = source / relative
        for path in sorted(root.glob(pattern)):
            document_path = path / "manifest.json" if manifests else path
            if not document_path.is_file():
                continue
            time_s = float(load_json(document_path).get("simulation_time_s", math.nan))
            require(math.isfinite(time_s), f"publication lacks finite time: {document_path}")
            if (lower and time_s <= split_time_s) or (not lower and time_s > split_time_s):
                require(time_s not in chosen, f"duplicate composed publication at {time_s}s")
                chosen[time_s] = path
    for path in chosen.values():
        target = destination_root / path.name
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)


def compose_resumed_run(prefix: Path, suffix: Path, destination: Path,
                        role: str, split_time_s: float) -> None:
    prepare_output(destination)
    for name in ("nimbus_output.json", "qualification_input.yaml",
                 "qualification_binary.sha256", "qualification_run.log"):
        shutil.copy2(suffix / name, destination / name)
    branch_marker = suffix / "nimbus_branch.json"
    if branch_marker.is_file():
        shutil.copy2(branch_marker, destination / branch_marker.name)
    relative = (Path("nests/child_0000/qualification_diagnostics")
                if role == "moving_child" else Path("qualification_diagnostics"))
    copy_timed_publications(prefix, suffix, destination, relative,
                            split_time_s, True)
    copy_timed_publications(prefix, suffix, destination, Path("storm_track"),
                            split_time_s, False)
    invalidity = prefix / "moving_nest_scientific_validity.json"
    require(not invalidity.exists() and
            not (suffix / "moving_nest_scientific_validity.json").exists(),
            "cannot compose a scientifically invalid moving branch")


def resolve_resume(prefix: Path | None, checkpoint: Path | None,
                   role: str) -> tuple[dict[str, Any] | None, Path | None]:
    require((prefix is None) == (checkpoint is None),
            f"{role} checkpoint resume requires both prefix run and checkpoint")
    if prefix is None or checkpoint is None:
        return None, None
    prefix = prefix.resolve()
    checkpoint = checkpoint.resolve()
    try:
        relative = checkpoint.relative_to(prefix)
    except ValueError as exc:
        raise CampaignError(
            f"{role} resume checkpoint is outside its prefix run") from exc
    require(relative.parts and relative.parts[0] == "checkpoints",
            f"{role} resume checkpoint is outside the prefix checkpoint store")
    return checkpoint_lifecycle(checkpoint), checkpoint


def execute(args: argparse.Namespace, inputs: dict[str, Any]) -> dict[str, Any]:
    root = args.output.resolve()
    prepare_output(root)
    duration_s = int(inputs["duration_s"])
    frozen_binary_sha = str(inputs["binary"]["sha256"])
    branches: dict[str, Any] = {}

    definitions = (
        ("oversized_reference", args.reference_run, args.reference_prefix_run,
         args.reference_resume, args.reference_config),
        ("moving_child", args.moving_run, args.moving_prefix_run,
         args.moving_resume, args.moving_config),
    )
    evidence_runs: dict[str, Path] = {}
    for role, existing, prefix, resume_path, config in definitions:
        require_inputs_unchanged(inputs)
        require(existing is None or (prefix is None and resume_path is None),
                f"{role} cannot combine an existing run with checkpoint resume")
        resume_record, selected_checkpoint = resolve_resume(prefix, resume_path, role)
        if resume_record is not None:
            require(0.0 < resume_record["simulation_time_s"] < duration_s and
                    math.isclose(
                        math.remainder(
                            resume_record["simulation_time_s"],
                            float(inputs["sample_interval_s"])),
                        0.0, rel_tol=0.0, abs_tol=1.0e-9),
                    f"{role} split checkpoint is outside the strict sample cadence")
            validate_run_identity(prefix.resolve(), config, args.binary)
        if existing is not None:
            evidence_run = existing.resolve()
            runtime_run = evidence_run
            execution = {"mode": "existing_complete_run"}
        else:
            runtime_name = role + ("_suffix" if resume_record is not None else "")
            runtime_run = root / runtime_name
            execution = run_branch(
                args.binary, config, role, runtime_run, duration_s,
                args.threads, args.checkpoint_every, frozen_binary_sha,
                selected_checkpoint)
            execution["mode"] = ("checkpoint_continuation"
                                 if resume_record is not None else "fresh")
            if resume_record is not None:
                evidence_run = root / (role + "_evidence")
                compose_resumed_run(prefix.resolve(), runtime_run, evidence_run,
                                    role, resume_record["simulation_time_s"])
                execution["prefix_run"] = str(prefix.resolve())
                execution["split_checkpoint"] = resume_record
            else:
                evidence_run = runtime_run
        output_contract = validate_output_cadence(
            evidence_run, role, config, args.binary, args.policy)
        checkpoints = validate_checkpoints(
            runtime_run, duration_s, args.checkpoint_every, resume_record)
        branches[role] = {
            "execution": execution,
            "runtime_run": str(runtime_run),
            "evidence_run": str(evidence_run),
            "output_contract": output_contract,
            "checkpoints": checkpoints,
        }
        evidence_runs[role] = evidence_run
        progress = seal_report({
            "schema": REPORT_SCHEMA,
            "schema_version": 1,
            "status": "running",
            "qualification_eligible": False,
            "inputs": inputs,
            "completed_branches": branches,
        })
        write_json(root / "campaign_report.json", progress)

    require_inputs_unchanged(inputs)
    artifacts_root = root / "comparison"
    moving_series = artifacts_root / "moving_child.json"
    moving_source = artifacts_root / "moving_child.source_manifest.json"
    reference_series = artifacts_root / "oversized_reference.json"
    reference_source = artifacts_root / "oversized_reference.source_manifest.json"
    extractor.extract(
        evidence_runs["moving_child"], "moving_child", args.moving_config,
        args.binary, args.policy, moving_series, moving_source)
    extractor.extract(
        evidence_runs["oversized_reference"], "oversized_reference",
        args.reference_config, args.binary, args.policy, reference_series,
        reference_source, moving_series)
    comparison = comparator.compare_paths(
        moving_series, reference_series, args.policy)
    comparison_path = artifacts_root / "comparison_report.json"
    write_json(comparison_path, comparison)
    require_inputs_unchanged(inputs)
    status = comparison.get("status")
    require(status in {"pass", "measurement_only", "incomplete", "fail", "incompatible"},
            "comparator returned an unknown disposition")
    report = seal_report({
        "schema": REPORT_SCHEMA,
        "schema_version": 1,
        "status": status,
        "qualification_eligible": comparison.get("qualification_eligible") is True,
        "policy_status": inputs["policy_status"],
        "inputs": inputs,
        "branches": branches,
        "comparison_artifacts": {
            "moving_series": artifact(moving_series),
            "moving_source_manifest": artifact(moving_source),
            "reference_series": artifact(reference_series),
            "reference_source_manifest": artifact(reference_source),
            "comparison_report": artifact(comparison_path),
        },
        "comparison_disposition": comparison,
    })
    write_json(root / "campaign_report.json", report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--binary", type=Path, default=BINARY)
    result.add_argument("--moving-config", type=Path, default=MOVING_CONFIG)
    result.add_argument("--reference-config", type=Path, default=REFERENCE_CONFIG)
    result.add_argument("--policy", type=Path, default=POLICY)
    result.add_argument("--output", type=Path)
    result.add_argument("--threads", type=int, default=12)
    result.add_argument("--checkpoint-every", type=int, default=2250)
    result.add_argument("--validate-only", "--dry-run", action="store_true",
                        dest="validate_only")
    result.add_argument("--moving-run", type=Path)
    result.add_argument("--reference-run", type=Path)
    result.add_argument("--moving-prefix-run", type=Path)
    result.add_argument("--moving-resume", type=Path)
    result.add_argument("--reference-prefix-run", type=Path)
    result.add_argument("--reference-resume", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    require(args.threads > 0, "threads must be positive")
    require(args.checkpoint_every > 0, "checkpoint cadence must be positive")
    args.binary = args.binary.resolve()
    args.moving_config = args.moving_config.resolve()
    args.reference_config = args.reference_config.resolve()
    args.policy = args.policy.resolve()
    inputs = validate_inputs(
        args.binary, args.moving_config, args.reference_config, args.policy)
    if args.validate_only:
        report = seal_report({
            "schema": REPORT_SCHEMA,
            "schema_version": 1,
            "status": "validated",
            "qualification_eligible": False,
            "policy_status": inputs["policy_status"],
            "inputs": inputs,
        })
        if args.output is not None:
            output = args.output.resolve()
            require(not output.exists() or not any(output.iterdir()),
                    f"output directory is not fresh: {output}")
            write_json(output / "campaign_report.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    require(args.output is not None, "a fresh --output directory is required for a campaign")
    report = execute(args, inputs)
    print(json.dumps(report, indent=2, sort_keys=True))
    return {"pass": 0, "measurement_only": 3, "incomplete": 3,
            "fail": 1, "incompatible": 2}.get(str(report["status"]), 2)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CampaignError, extractor.ExtractionError,
            comparator.ContractError, OSError, ValueError) as exc:
        print(f"[moving-reference-campaign] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
