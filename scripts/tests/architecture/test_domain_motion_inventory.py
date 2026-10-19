#!/usr/bin/env python3
"""Fail-closed structural checks for the domain-motion impact inventory."""

import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "tests/fixtures/reference/domain_motion_inventory.json"

REQUIRED_CATEGORIES = {
    "state_owner",
    "coordinate_consumer",
    "velocity_consumer",
    "output_writer",
    "restart_field",
    "cpu_operator",
    "vulkan_operator",
}
ALLOWED_DISPOSITIONS = {
    "frame_primitive_ready",
    "contract_defined",
    "stationary_only",
    "translation_invariant",
    "unsupported_initial_release",
    "diagnostic_ready",
}
REQUIRED_PATHS = {
    "include/core/runtime/simulation_state.hpp",
    "src/core/orchestration/dynamics/initial_conditions_cartesian.cpp",
    "src/dynamics/schemes/klemp_wilhelmson_split/cartesian_agrid.cpp",
    "src/dynamics/schemes/klemp_wilhelmson_split/cartesian_cgrid_3d.cpp",
    "src/numerics/advection/advection_cartesian.cpp",
    "src/boundary_conditions/boundary_conditions_cartesian.cpp",
    "src/boundary_conditions/boundary_conditions_cartesian_cgrid.cpp",
    "src/boundary_layer/base/surface_fluxes.cpp",
    "src/core/infra/rayleigh_damping.cpp",
    "src/core/orchestration/physics/diffusion_step.cpp",
    "src/core/orchestration/dynamics/sgs_apply.cpp",
    "src/terrain/runtime.cpp",
    "src/radiation/runtime.cpp",
    "src/diagnostics/storm_tracker.cpp",
    "src/radar/runtime.cpp",
    "src/core/runtime/headless_runtime.cpp",
    "src/core/output/output_writer.cpp",
    "src/core/output/netcdf_history_writer.cpp",
    "src/core/output/shm_writer.cpp",
    "src/core/runtime/simulation_checkpoint.cpp",
    "src/core/infra/nested_grid.cpp",
    "include/compute/kernel_abi.hpp",
    "src/compute/dispatch/advection.cpp",
    "vulkan/src/compute/compute_backend_vulkan.cpp",
    "vulkan/shaders/compute/dynamics/cartesian_tendencies.comp",
    "vulkan/shaders/compute/dynamics/cartesian_cgrid_slow_tendencies.comp",
    "vulkan/shaders/compute/dynamics/acoustic_bc_face_cartesian_cgrid.comp",
    "vulkan/shaders/compute/advection/advect_conservative_mass_flux_cartesian_cgrid.comp",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> int:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        fail("domain-motion inventory schema_version must be 1")
    if data.get("contract_id") != "constant_translation_earth_wind_grid_transport_v1":
        fail("unexpected or missing domain-motion contract_id")

    storage = data.get("storage_contract", {})
    if storage.get("prognostic_horizontal_wind") != "earth_relative":
        fail("prognostic horizontal wind must remain Earth-relative")
    if storage.get("horizontal_transport_velocity") != "grid_relative":
        fail("horizontal transport velocity must be grid-relative")

    ids = set()
    categories = set()
    covered_paths = set()
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        fail("domain-motion inventory must contain entries")

    for entry in entries:
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            fail("every inventory entry needs a non-empty id")
        if entry_id in ids:
            fail(f"duplicate inventory id: {entry_id}")
        ids.add(entry_id)

        disposition = entry.get("disposition")
        if disposition not in ALLOWED_DISPOSITIONS:
            fail(f"{entry_id}: invalid disposition {disposition!r}")
        contract = entry.get("contract")
        if not isinstance(contract, str) or len(contract) < 24:
            fail(f"{entry_id}: missing substantive frame contract")

        entry_categories = entry.get("categories")
        if not isinstance(entry_categories, list) or not entry_categories:
            fail(f"{entry_id}: categories must be a non-empty list")
        categories.update(entry_categories)

        evidence = entry.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            fail(f"{entry_id}: evidence must be a non-empty list")
        for item in evidence:
            relative = item.get("path")
            marker = item.get("contains")
            if not isinstance(relative, str) or not isinstance(marker, str) or not marker:
                fail(f"{entry_id}: malformed source evidence")
            path = ROOT / relative
            if not path.is_file():
                fail(f"{entry_id}: missing inventoried source {relative}")
            text = path.read_text(encoding="utf-8", errors="strict")
            if marker not in text:
                fail(f"{entry_id}: source marker {marker!r} missing from {relative}")
            covered_paths.add(relative)

    missing_categories = REQUIRED_CATEGORIES - categories
    if missing_categories:
        fail(f"inventory omits required categories: {sorted(missing_categories)}")
    missing_paths = REQUIRED_PATHS - covered_paths
    if missing_paths:
        fail(f"inventory omits required source paths: {sorted(missing_paths)}")

    print(
        f"[domain-motion-inventory] PASS: {len(entries)} contracts, "
        f"{len(covered_paths)} source anchors, {len(categories)} categories"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[domain-motion-inventory] FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
