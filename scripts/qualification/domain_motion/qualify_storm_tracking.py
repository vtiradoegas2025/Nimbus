#!/usr/bin/env python3
"""Fail-closed local qualification for Cartesian storm-tracking evidence."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def runtime_environment(compute_backend: str) -> dict[str, str]:
    environment = os.environ.copy()
    if compute_backend != "vulkan" or platform.system() != "Darwin":
        return environment

    # Homebrew installs the loader, MoltenVK ICD, and validation layer in
    # separate prefixes that non-interactive processes do not always inherit.
    # Supply only missing discovery settings; explicit operator choices win.
    icd = Path("/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json")
    layer_dir = Path("/opt/homebrew/share/vulkan/explicit_layer.d")
    if "VK_ICD_FILENAMES" not in environment and icd.is_file():
        environment["VK_ICD_FILENAMES"] = str(icd)
    if "VK_LAYER_PATH" not in environment and layer_dir.is_dir():
        environment["VK_LAYER_PATH"] = str(layer_dir)

    library_dirs = [
        Path("/opt/homebrew/opt/vulkan-validationlayers/lib"),
        Path("/opt/homebrew/opt/vulkan-loader/lib"),
        Path("/opt/homebrew/opt/molten-vk/lib"),
    ]
    discovered = [str(path) for path in library_dirs if path.is_dir()]
    inherited = environment.get("DYLD_LIBRARY_PATH")
    if inherited:
        discovered.append(inherited)
    if discovered:
        environment["DYLD_LIBRARY_PATH"] = ":".join(discovered)
    return environment


def run(
    binary: Path,
    config: Path,
    outdir: Path,
    duration: int,
    compute_backend: str,
    require_vulkan: bool,
    checkpoint_every: int = 0,
    resume: Path | None = None,
) -> str:
    command = [
        str(binary),
        "--headless",
        f"--config={config}",
        f"--compute-backend={compute_backend}",
        f"--duration={duration}",
        "--write-every=0",
        f"--checkpoint-every={checkpoint_every}",
        f"--outdir={outdir}",
        f"--log-profile={'normal' if require_vulkan else 'quiet'}",
        "--no-timing",
    ]
    if resume is not None:
        command.append(f"--resume={resume}")
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=runtime_environment(compute_backend),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(
        result.returncode == 0,
        f"{config.name} exited {result.returncode}\n{result.stdout[-6000:]}",
    )
    if require_vulkan:
        require(
            "[COMPUTE][VULKAN] selected device[" in result.stdout,
            f"{config.name} did not select a physical Vulkan device\n{result.stdout[-6000:]}",
        )
        require(
            "[COMPUTE] backend=vulkan" in result.stdout,
            f"{config.name} did not retain the requested Vulkan backend\n{result.stdout[-6000:]}",
        )
        require(
            "Falling back to CPU" not in result.stdout and "fallback to CPU" not in result.stdout,
            f"{config.name} used CPU fallback\n{result.stdout[-6000:]}",
        )
    return result.stdout


def check_degenerate_branch_motion_rejected_before_run(
    binary: Path, source_config: Path, invalid_config: Path, compute_backend: str
) -> None:
    source = source_config.read_text(encoding="utf-8")
    u_surface = re.search(r"(?m)^\s*u_sfc_ms:\s*([^#\s]+)", source)
    v_surface = re.search(r"(?m)^\s*v_sfc_ms:\s*([^#\s]+)", source)
    require(u_surface is not None and v_surface is not None, "missing surface wind anchors")
    invalid = re.sub(
        r"(?m)^(\s*u_6km_ms:)\s*[^#\s]+",
        rf"\1 {u_surface.group(1)}",
        source,
    )
    invalid = re.sub(
        r"(?m)^(\s*v_6km_ms:)\s*[^#\s]+",
        rf"\1 {v_surface.group(1)}",
        invalid,
    )
    require(invalid != source, "failed to construct degenerate-shear branch fixture")
    invalid_config.write_text(invalid, encoding="utf-8")
    result = subprocess.run(
        [
            str(binary),
            "--headless",
            f"--config={invalid_config}",
            f"--compute-backend={compute_backend}",
            "--duration=1",
            "--write-every=0",
            f"--outdir={invalid_config.parent / 'invalid_out'}",
            "--log-profile=quiet",
            "--no-timing",
        ],
        cwd=ROOT,
        env=runtime_environment(compute_backend),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(result.returncode != 0, "degenerate-shear Bunkers preference unexpectedly started")
    require(
        "Bunkers storm motion is undefined for negligible 0--6 km shear" in result.stdout,
        f"degenerate-shear failure did not identify branch-motion science\n{result.stdout[-6000:]}",
    )
    require(
        not (invalid_config.parent / "invalid_out").exists(),
        "degenerate branch motion reached runtime output allocation",
    )


def load_latest(outdir: Path) -> dict:
    path = outdir / "storm_track_latest.json"
    require(path.is_file(), f"missing storm evidence: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid storm evidence {path}: {exc}") from exc
    require(isinstance(value, dict), f"storm evidence is not an object: {path}")
    return value


def check_weak_record(record: dict, completed_step: int, missed_samples: int) -> None:
    require(record.get("schema") == "nimbus.storm_feature", "wrong storm schema")
    require(record.get("schema_version") == 12, "wrong storm schema version")
    require(record.get("tracking_source") == "root_grid", "wrong storm tracking source")
    require(record.get("detector_coordinate_frame") == "grid_relative",
            "wrong storm detector coordinate frame")
    require(
        record.get("algorithm") == "updraft_core_watershed_condensate_rotation_branches_v5",
        "wrong feature algorithm",
    )
    require(
        record.get("updraft_helicity_algorithm") == "physical_height_midpoint_uh_2d_v1",
        "wrong UH algorithm",
    )
    require(
        record.get("continuity_algorithm") == "fixed_window_temporal_branch_association_v5",
        "wrong continuity algorithm",
    )
    require(record.get("completed_step") == completed_step, "wrong completed-step identity")
    require(record.get("status") == "weak", "calm fixture invented a storm")
    require(record.get("reported_branch_candidate_count") == 0, "weak record invented a branch candidate")
    require(record.get("branch_candidate_overflow") is False, "weak record claimed branch overflow")
    require(record.get("branch_candidates") == [], "weak record emitted branch evidence")
    require(record.get("selected_feature_available") is False, "weak record invented a selected feature")
    require(record.get("selected_feature_role") is None, "weak record invented a selected feature role")
    require(record.get("prognostic_state_mutation") is False, "tracker claimed prognostic mutation")
    require(record.get("diagnostic_history_mutation") is True, "tracker hid diagnostic mutation")
    for key in (
        "selected_component_cells",
        "selected_core_cells",
        "selected_rotating_core_cells",
        "maximum_updraft_ms",
        "condensate_mass_kg",
    ):
        require(record.get(key) is None, f"unavailable weak field {key} is not null")
    require(record.get("centroid_m", {}).get("x") is None, "weak centroid is not null")
    require(
        record.get("distance_to_physical_boundary_m", {}).get("nearest") is None,
        "weak boundary distance is not null",
    )
    require(record.get("boundary_contact") is None, "unavailable weak boundary contact is not null")
    require(record.get("inner_margin_breached") is None, "unavailable weak margin state is not null")
    face_motion = record.get("envelope_face_motion", {})
    require(face_motion.get("available") is False,
            "weak record invented envelope-face motion")
    for key in (
            "interval_s",
            "west_coordinate_velocity_ms",
            "east_coordinate_velocity_ms",
            "south_coordinate_velocity_ms",
            "north_coordinate_velocity_ms"):
        require(face_motion.get(key) is None,
                f"weak record invented envelope-face motion field {key}")
    movement = record.get("movement_authority", {})
    require(
        movement.get("algorithm") == "qualified_envelope_hysteresis_v14",
        "wrong movement-authority algorithm",
    )
    require(movement.get("mode") == "diagnostic_request_only", "movement authority is not diagnostic-only")
    require(movement.get("enabled") is True, "movement authority was not exercised")
    require(movement.get("recycling_enabled") is False, "diagnostic fixture gained recycling authority")
    require(movement.get("nested_movement_enabled") is False, "diagnostic fixture gained nested movement authority")
    require(movement.get("prognostic_state_mutation") is False, "movement authority claimed state mutation")
    require(movement.get("commit_performed") is False, "diagnostic tracker committed a domain move")
    require(movement.get("status") == "ineligible", "weak record gained movement authority")
    require(movement.get("reason") == "no_selected_feature", "weak movement rejection reason changed")
    require(movement.get("request_ready") is False, "weak record requested a domain move")
    require(movement.get("confirmation_count") == 0, "weak record retained movement confirmation")
    require(movement.get("armed") is True, "diagnostic movement authority unexpectedly disarmed")
    rotation = record.get("rotation", {})
    require(rotation.get("layer_reference") == "above_ground_level", "wrong UH height reference")
    require(rotation.get("layer_bottom_m_agl") == 2000, "wrong UH lower bound")
    require(rotation.get("layer_top_m_agl") == 5000, "wrong UH upper bound")
    require(rotation.get("sign_retained") is True, "UH sign provenance is missing")
    require(
        rotation.get("maximum_absolute_updraft_helicity_m2s2") is None,
        "weak UH maximum is not null",
    )
    continuity = record.get("continuity", {})
    require(continuity.get("missed_samples") == missed_samples, "wrong restart-exact miss count")
    require(continuity.get("topology") == "unknown", "weak record invented confirmed topology")
    require(continuity.get("topology_event") == "none", "weak record invented a topology event")
    require(
        continuity.get("selected_candidate_available") is False,
        "weak record invented a temporal branch association",
    )
    association = record.get("environmental_branch_association", {})
    require(association.get("available") is True, "resolved Bunkers branch motion is unavailable")
    require(association.get("preference") == "bunkers_right", "wrong runtime branch preference")
    require(
        association.get("bunkers_right_velocity_ms", {}).get("x") is not None,
        "Bunkers-right velocity was not published",
    )
    require(
        association.get("bunkers_left_velocity_ms", {}).get("x") is not None,
        "Bunkers-left velocity was not published",
    )
    require(
        association.get("selected_bunkers_right_mismatch_ms") is None
        and association.get("selected_bunkers_left_mismatch_ms") is None,
        "weak record invented selected branch-motion mismatch",
    )
    require(
        association.get("selection_uses_updraft_helicity_sign") is False,
        "branch selection incorrectly claims UH-sign authority",
    )


def record_count(outdir: Path) -> int:
    return len(list((outdir / "storm_track").glob("track_step_*.json")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary",
        type=Path,
        default=ROOT / "bin" / "nimbus_sim_cpu",
        help="Nimbus executable",
    )
    parser.add_argument(
        "--compute-backend",
        choices=("cpu", "vulkan"),
        default="cpu",
        help="runtime compute backend to qualify",
    )
    parser.add_argument(
        "--require-vulkan",
        action="store_true",
        help="fail unless every runtime selects and retains a physical Vulkan device",
    )
    parser.add_argument(
        "--grid-scope",
        choices=("all", "cgrid"),
        default="all",
        help="qualify both Cartesian staggerings or the mapped C-grid path only",
    )
    args = parser.parse_args()
    require(
        not args.require_vulkan or args.compute_backend == "vulkan",
        "--require-vulkan requires --compute-backend=vulkan",
    )
    binary = args.binary.resolve()
    require(binary.is_file(), f"missing simulator binary: {binary}")

    agrid_config = ROOT / "tests" / "fixtures" / "configs" / "test" / "storm_tracking_agrid_smoke.yaml"
    cgrid_config = ROOT / "tests" / "fixtures" / "configs" / "test" / "storm_tracking_smoke.yaml"
    with tempfile.TemporaryDirectory(prefix="nimbus_storm_qualification_") as temporary:
        base = Path(temporary)
        agrid_out = base / "agrid"
        cgrid_out = base / "cgrid"
        resumed_out = base / "cgrid_resumed"
        check_degenerate_branch_motion_rejected_before_run(
            binary,
            cgrid_config if args.compute_backend == "vulkan" else agrid_config,
            base / "invalid_branch_motion.yaml",
            args.compute_backend,
        )

        if args.grid_scope == "all":
            run(
                binary,
                agrid_config,
                agrid_out,
                duration=1,
                compute_backend=args.compute_backend,
                require_vulkan=args.require_vulkan,
            )
            check_weak_record(load_latest(agrid_out), completed_step=10, missed_samples=10)
            require(record_count(agrid_out) == 10, "A-grid cadence did not publish ten records")

        run(
            binary,
            cgrid_config,
            cgrid_out,
            duration=1,
            compute_backend=args.compute_backend,
            require_vulkan=args.require_vulkan,
            checkpoint_every=1,
        )
        check_weak_record(load_latest(cgrid_out), completed_step=10, missed_samples=10)
        require(record_count(cgrid_out) == 10, "mapped C-grid cadence did not publish ten records")
        checkpoints = sorted((cgrid_out / "checkpoints").glob("checkpoint_step_*.nimbus"))
        require(len(checkpoints) == 1, "mapped C-grid run did not publish exactly one checkpoint")

        run(
            binary,
            cgrid_config,
            resumed_out,
            duration=2,
            compute_backend=args.compute_backend,
            require_vulkan=args.require_vulkan,
            resume=checkpoints[0],
        )
        check_weak_record(load_latest(resumed_out), completed_step=20, missed_samples=20)
        require(record_count(resumed_out) == 10, "resumed branch did not publish records 11 through 20")

    print(
        f"[storm-tracking] PASS ({args.compute_backend}, {args.grid_scope}): "
        + ("flat A-grid 10 records; " if args.grid_scope == "all" else "")
        + "mapped C-grid 10 records; mapped restart continued miss 10 -> 20"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        raise SystemExit(f"[storm-tracking] FAIL: {exc}") from exc
