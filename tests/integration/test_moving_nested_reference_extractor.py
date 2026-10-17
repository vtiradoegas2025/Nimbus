#!/usr/bin/env python3
"""Synthetic production-contract tests for moving/reference extraction."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


extractor = load_module(
    "extract_moving_nested_reference_series",
    ROOT / "scripts" / "qualification" / "nested_domains" / "extract_moving_nested_reference_series.py",
)
comparator = load_module(
    "compare_moving_nested_supercell_reference",
    ROOT / "scripts" / "qualification" / "nested_domains" / "compare_moving_nested_supercell_reference.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_npy(path: Path, shape: tuple[int, int], values: list[float]) -> None:
    assert len(values) == shape[0] * shape[1]
    header = repr({"descr": "<f4", "fortran_order": False, "shape": shape})
    padding = 16 - ((10 + len(header) + 1) % 16)
    encoded = (header + " " * padding + "\n").encode("latin1")
    path.write_bytes(
        b"\x93NUMPY" + bytes((1, 0)) + struct.pack("<H", len(encoded)) + encoded +
        struct.pack("<" + "f" * len(values), *values)
    )


def config_text() -> str:
    return """schema: nimbus.v1
grid:
  coordinate_system: cartesian
  staggering: c_grid
  dx: 1
  dy: 1
  nz: 4
  dz: 250
  ztop: 1000
time:
  dt_large: 1
  acoustic_substeps: 4
  adaptive_dt: false
  cfl_safety: 0.7
domain_motion:
  nesting:
    parent_ratio_x: 1
    parent_ratio_y: 1
    parent_time_step_ratio: 1
dynamics:
  scheme: klemp_wilhelmson_split_cartesian_cgrid
damping:
  enabled: false
coriolis:
  enabled: false
microphysics:
  scheme: kessler
turbulence:
  scheme: none
diffusion:
  scheme: none
surface:
  scheme: none
boundary_layer:
  scheme: none
radiation:
  scheme: none
terrain:
  scheme: none
boundary_conditions:
  bottom: rigid_slip
  top: rigid
initial_conditions:
  sounding:
    type: parametric_cape
  hodograph:
    type: wk_param
  trigger:
    type: bubble
    bubble:
      dtheta_k: 4
      radius_km: 4
      center_x_m: 10
      center_y_m: 10
      center_z_m: 500
extensions:
  compute:
    backend: cpu
  diagnostics:
    deterministic_reductions: true
"""


def policy_document() -> dict:
    policy = json.loads((
        ROOT / "tests" / "qualification" / "fixtures" /
        "moving_nested_supercell_reference_frontier_v1.json"
    ).read_text(encoding="utf-8"))
    policy["expected_contract"].update({
        "duration_s": 60.0,
        "sample_start_s": 30.0,
        "sample_interval_s": 30.0,
        "relaxation_exclusion_cells": 1,
        "horizontal_grid_spacing_m": {"x": 1.0, "y": 1.0},
    })
    return policy


def step_manifest(role: str, time_s: float, nx: int, ny: int,
                  phase_x: float = 0.0, spacing_m: float = 1.0) -> dict:
    moving = role == "moving_child"
    lower = [1, 1, 0] if moving else [0, 0, 0]
    upper = [nx - 2, ny - 2, 3] if moving else [nx - 1, ny - 1, 3]
    return {
        "schema": "nimbus.step",
        "manifest_version": 6,
        "publication_state": "complete",
        "simulation_time_s": time_s,
        "data_product": {
            "kind": ("nested_child_qualification_diagnostics"
                     if moving else "qualification_diagnostics")},
        "grid": {
            "coordinate_system": "cartesian",
            "dimensions": {"x": nx, "y": ny, "z": 4},
            "spacing_m": {"x": spacing_m, "y": spacing_m, "z": 250.0},
            "horizontal_frame": {"grid_origin_earth_m": [phase_x, 0.0]},
            "grid_scalar_origin_m": [0.5 * spacing_m, 0.5 * spacing_m, 125.0],
            "valid_index_range": {"lower": lower, "upper": upper},
        },
        "published_fields": ["cold_pool_boundary", "accumulated_rainfall"],
        "fields": [
            {
                "field_id": "cold_pool_boundary", "units": "K",
                "description": "all species " + extractor.COLD_POOL_ALGORITHM,
            },
            {
                "field_id": "accumulated_rainfall", "units": "mm",
                "derived": True, "grid_location": "surface_cell_center",
                "state_semantic": "diagnostic",
            },
        ],
    }


def tracker(role: str, time_s: float,
            movement_algorithm: str = extractor.MOVEMENT_ALGORITHM) -> dict:
    moving = role == "moving_child"
    record = {
        "schema": "nimbus.storm_feature",
        "schema_version": 12,
        "algorithm": extractor.DETECTOR_ALGORITHM,
        "continuity_algorithm": extractor.CONTINUITY_ALGORITHM,
        "detector_coordinate_frame": "earth_absolute",
        "tracking_source": "synchronized_nested_child" if moving else "root_grid",
        "simulation_time_s": time_s,
        "selected_feature_available": True,
        "core_centroid_m": {"x": 3.0 + time_s / 30.0, "y": 3.0},
        "maximum_updraft_ms": 20.0 + time_s / 30.0,
        "rotation": {"maximum_absolute_updraft_helicity_m2s2": 100.0 + time_s},
    }
    if moving:
        record["movement_authority"] = {"algorithm": movement_algorithm}
        record["moving_nest_scientific_validity"] = {
            "owner": "physical_child_trusted_interior_v1", "status": "valid"
        }
    return record


def make_run(root: Path, role: str, config: Path, binary: Path,
             phase_x: float = 0.0, omit_time: float | None = None,
             surface_shape: tuple[int, int] | None = None,
             spacing_m: float = 1.0,
             times: tuple[float, ...] = (30.0, 60.0),
             movement_algorithm: str = extractor.MOVEMENT_ALGORITHM) -> Path:
    run = root / f"{role}_run"
    run.mkdir()
    (run / "qualification_input.yaml").write_bytes(config.read_bytes())
    (run / "qualification_binary.sha256").write_text(digest(binary) + "\n", encoding="ascii")
    write_json(run / "nimbus_output.json", {
        "schema": "nimbus.output", "manifest_version": 6,
        "publication_state": "ready",
        "data_products": ["nested_child_qualification_diagnostics"]
        if role == "moving_child" else ["qualification_diagnostics"],
    })
    history = (run / "nests" / "child_0000" / "qualification_diagnostics"
               if role == "moving_child" else run / "qualification_diagnostics")
    for ordinal, time_s in enumerate(times, 1):
        if time_s == omit_time:
            continue
        step = history / f"step_{ordinal:06d}"
        step.mkdir(parents=True)
        nx, ny = (surface_shape if surface_shape is not None else
                  ((8, 8) if role == "moving_child" else (10, 10)))
        write_json(step / "manifest.json", step_manifest(
            role, time_s, nx, ny, phase_x, spacing_m))
        cold = [0.0] * (nx * ny)
        rain = [0.0] * (nx * ny)
        for i in range(nx):
            for j in range(ny):
                cold[i * ny + j] = 1.0 + 0.25 * i + 0.1 * j
                rain[i * ny + j] = 0.5 + 0.2 * i + 0.1 * j + time_s / 60.0
        write_npy(step / "cold_pool_boundary.npy", (nx, ny), cold)
        write_npy(step / "accumulated_rainfall.npy", (nx, ny), rain)
        write_json(run / "storm_track" / f"track_step_{ordinal:09d}.json",
                   tracker(role, time_s, movement_algorithm))
    return run


class MovingReferenceExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="nimbus_reference_extract_")
        self.root = Path(self.temporary.name)
        self.binary = self.root / "nimbus_sim"
        self.binary.write_bytes(b"synthetic same Nimbus binary\n")
        self.config = self.root / "config.yaml"
        self.config.write_text(config_text(), encoding="utf-8")
        self.policy = self.root / "policy.json"
        write_json(self.policy, policy_document())

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def ratified_geometry_policy(self) -> tuple[dict, Path, Path]:
        policy_source = (
            ROOT / "tests" / "qualification" / "fixtures" /
            "moving_nested_supercell_reference_frontier_v3.json"
        )
        policy = json.loads(policy_source.read_text(encoding="utf-8"))
        moving_config = (
            ROOT / "tests" / "qualification" / "fixtures" /
            "moving_nested_supercell_lifetime.yaml"
        )
        reference_config = (
            ROOT / "tests" / "fixtures" / "configs" / "qualification" /
            "moving_nested_supercell_reference_v5.yaml"
        )
        _, signature, _ = extractor.numerical_signature(
            moving_config, "moving_child", {"x": 600.0, "y": 600.0})
        policy["expected_contract"]["physics_signature_sha256"] = signature
        for key in ("geometry_plan", "qualification_aperture"):
            record = policy["expected_contract"][key]
            record["path"] = str((policy_source.parent / record["path"]).resolve())
        for key in ("threshold_policy", "failure_evidence"):
            record = policy[key]
            record["path"] = str((policy_source.parent / record["path"]).resolve())
        for record in policy["ratification"]["reviewed_sources"]:
            record["path"] = str((policy_source.parent / record["path"]).resolve())
        return policy, moving_config, reference_config

    def ratified_projected_policy(self) -> tuple[dict, Path, Path]:
        policy_source = (
            ROOT / "tests" / "qualification" / "fixtures" /
            "moving_nested_supercell_reference_frontier_v4.json"
        )
        policy = json.loads(policy_source.read_text(encoding="utf-8"))
        moving_config = (
            ROOT / "tests" / "qualification" / "fixtures" /
            "moving_nested_supercell_lifetime.yaml"
        )
        reference_config = (
            ROOT / "tests" / "fixtures" / "configs" / "qualification" /
            "moving_nested_supercell_reference_v5.yaml"
        )
        _, signature, _ = extractor.numerical_signature(
            moving_config, "moving_child", {"x": 600.0, "y": 600.0})
        policy["expected_contract"]["physics_signature_sha256"] = signature
        for key in ("geometry_plan", "qualification_aperture"):
            record = policy["expected_contract"][key]
            record["path"] = str((policy_source.parent / record["path"]).resolve())
        for key in ("threshold_policy", "prior_policy", "failure_evidence"):
            record = policy[key]
            record["path"] = str((policy_source.parent / record["path"]).resolve())
        for record in policy["ratification"]["reviewed_sources"]:
            record["path"] = str((policy_source.parent / record["path"]).resolve())
        return policy, moving_config, reference_config

    def ratified_maximum_safe_projected_policy(self) -> tuple[dict, Path, Path]:
        policy_source = (
            ROOT / "tests" / "qualification" / "fixtures" /
            "moving_nested_supercell_reference_frontier_v5.json"
        )
        policy = json.loads(policy_source.read_text(encoding="utf-8"))
        moving_config = (
            ROOT / "tests" / "qualification" / "fixtures" /
            "moving_nested_supercell_lifetime.yaml"
        )
        reference_config = (
            ROOT / "tests" / "fixtures" / "configs" / "qualification" /
            "moving_nested_supercell_reference_v5.yaml"
        )
        _, signature, _ = extractor.numerical_signature(
            moving_config, "moving_child", {"x": 600.0, "y": 600.0})
        policy["expected_contract"]["physics_signature_sha256"] = signature
        for key in ("geometry_plan", "qualification_aperture"):
            record = policy["expected_contract"][key]
            record["path"] = str((policy_source.parent / record["path"]).resolve())
        for key in ("threshold_policy", "prior_policy", "failure_evidence"):
            record = policy[key]
            record["path"] = str((policy_source.parent / record["path"]).resolve())
        for record in policy["ratification"]["reviewed_sources"]:
            record["path"] = str((policy_source.parent / record["path"]).resolve())
        return policy, moving_config, reference_config

    def extract_pair(self, reference_phase: float = 0.0,
                     policy: Path | None = None,
                     moving_config: Path | None = None,
                     reference_config: Path | None = None,
                     qualified_geometry: bool = False,
                     movement_algorithm: str =
                         extractor.MOVEMENT_ALGORITHM) -> tuple[Path, Path]:
        policy = self.policy if policy is None else policy
        moving_config = self.config if moving_config is None else moving_config
        reference_config = moving_config if reference_config is None else reference_config
        full_times = tuple(float(value) for value in range(30, 4501, 30))
        moving_run = make_run(
            self.root, "moving_child", moving_config, self.binary,
            surface_shape=(158, 168) if qualified_geometry else None,
            spacing_m=600.0 if qualified_geometry else 1.0,
            times=full_times if qualified_geometry else (30.0, 60.0),
            movement_algorithm=movement_algorithm)
        moving_path = self.root / "moving.json"
        extractor.extract(
            moving_run, "moving_child", moving_config, self.binary, policy,
            moving_path, self.root / "moving.source.json")
        reference_run = make_run(
            self.root, "oversized_reference", reference_config, self.binary,
            reference_phase, surface_shape=(269, 215) if qualified_geometry else None,
            spacing_m=600.0 if qualified_geometry else 1.0,
            times=full_times if qualified_geometry else (30.0, 60.0))
        reference_path = self.root / "reference.json"
        extractor.extract(
            reference_run, "oversized_reference", reference_config, self.binary,
            policy, reference_path, self.root / "reference.source.json", moving_path)
        return moving_path, reference_path

    def test_pair_is_hash_bound_and_measurement_only(self) -> None:
        moving_path, reference_path = self.extract_pair()
        moving = json.loads(moving_path.read_text(encoding="utf-8"))
        self.assertEqual(moving["schema_version"], 2)
        self.assertEqual(
            moving["samples"][0]["precipitation"]["effective_depth_semantics"],
            "residual_inclusive_then_float32_published_v1",
        )
        source = json.loads((self.root / "moving.source.json").read_text(encoding="utf-8"))
        self.assertEqual(
            source["diagnostics"]["cold_pool"]["water_species"],
            ["qv", "qc", "qr", "qi", "qs", "qg", "qh"],
        )
        report = comparator.compare_paths(moving_path, reference_path, self.policy)
        self.assertEqual(report["status"], "measurement_only")
        self.assertFalse(report["qualification_eligible"])

    def test_hash_bound_pair_can_pass_only_an_explicit_v3_ratification(self) -> None:
        policy, moving_config, reference_config = self.ratified_geometry_policy()
        ratified_path = self.root / "ratified_policy.json"
        write_json(ratified_path, policy)
        moving_path, reference_path = self.extract_pair(
            policy=ratified_path, moving_config=moving_config,
            reference_config=reference_config, qualified_geometry=True)

        report = comparator.compare_paths(
            moving_path, reference_path, ratified_path)
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["qualification_eligible"])
        self.assertEqual(report["policy"]["schema_version"], 3)

        policy["policy_status"] = "measurement_only"
        write_json(ratified_path, policy)
        rejected = comparator.compare_paths(
            moving_path, reference_path, ratified_path)
        self.assertEqual(rejected["status"], "incompatible")
        self.assertFalse(rejected["qualification_eligible"])

    def test_ratified_v3_policy_is_bound_during_extraction(self) -> None:
        policy, moving_config, reference_config = self.ratified_geometry_policy()
        ratified_path = self.root / "ratified_policy.json"
        write_json(ratified_path, policy)

        moving_run = make_run(
            self.root, "moving_child", moving_config, self.binary,
            surface_shape=(158, 168), spacing_m=600.0,
            times=tuple(float(value) for value in range(30, 4501, 30)))
        moving_path = self.root / "moving_ratified.json"
        moving_source_path = self.root / "moving_ratified.source.json"
        extractor.extract(
            moving_run, "moving_child", moving_config, self.binary, ratified_path,
            moving_path, moving_source_path)
        reference_run = make_run(
            self.root, "oversized_reference", reference_config, self.binary,
            surface_shape=(269, 215), spacing_m=600.0,
            times=tuple(float(value) for value in range(30, 4501, 30)))
        reference_path = self.root / "reference_ratified.json"
        reference_source_path = self.root / "reference_ratified.source.json"
        extractor.extract(
            reference_run, "oversized_reference", reference_config, self.binary,
            ratified_path, reference_path, reference_source_path, moving_path)

        for source_path in (moving_source_path, reference_source_path):
            source = json.loads(source_path.read_text(encoding="utf-8"))
            self.assertEqual(source["policy_status"], "ratified")
            self.assertEqual(source["policy"]["sha256"], digest(ratified_path))
        report = comparator.compare_paths(
            moving_path, reference_path, ratified_path)
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["qualification_eligible"])

    def test_ratified_v4_policy_requires_v6_and_can_complete_comparison(self) -> None:
        policy, moving_config, reference_config = (
            self.ratified_projected_policy())
        ratified_path = self.root / "ratified_v4_policy.json"
        write_json(ratified_path, policy)
        moving_path, reference_path = self.extract_pair(
            policy=ratified_path,
            moving_config=moving_config,
            reference_config=reference_config,
            qualified_geometry=True,
            movement_algorithm=
                extractor.PROJECTED_CONTAINMENT_MOVEMENT_ALGORITHM)
        moving = json.loads(moving_path.read_text(encoding="utf-8"))
        source = json.loads(
            (self.root / "moving.source.json").read_text(encoding="utf-8"))
        self.assertEqual(
            moving["contract"]["movement_algorithm"],
            extractor.PROJECTED_CONTAINMENT_MOVEMENT_ALGORITHM)
        self.assertEqual(source["policy"]["schema_version"], 4)
        report = comparator.compare_paths(
            moving_path, reference_path, ratified_path)
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["qualification_eligible"])
        self.assertEqual(report["policy"]["schema_version"], 4)

    def test_ratified_v5_policy_requires_v7_and_can_complete_comparison(self) -> None:
        policy, moving_config, reference_config = (
            self.ratified_maximum_safe_projected_policy())
        ratified_path = self.root / "ratified_v5_policy.json"
        write_json(ratified_path, policy)
        moving_path, reference_path = self.extract_pair(
            policy=ratified_path,
            moving_config=moving_config,
            reference_config=reference_config,
            qualified_geometry=True,
            movement_algorithm=
                extractor.MAXIMUM_SAFE_PROJECTED_MOVEMENT_ALGORITHM)
        moving = json.loads(moving_path.read_text(encoding="utf-8"))
        source = json.loads(
            (self.root / "moving.source.json").read_text(encoding="utf-8"))
        self.assertEqual(
            moving["contract"]["movement_algorithm"],
            extractor.MAXIMUM_SAFE_PROJECTED_MOVEMENT_ALGORITHM)
        self.assertEqual(source["policy"]["schema_version"], 5)
        report = comparator.compare_paths(
            moving_path, reference_path, ratified_path)
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["qualification_eligible"])
        self.assertEqual(report["policy"]["schema_version"], 5)

    def test_fixed_aperture_is_explicit_and_reference_crop_is_identical(self) -> None:
        policy, moving_config, reference_config = self.ratified_geometry_policy()
        ratified_path = self.root / "ratified_policy.json"
        write_json(ratified_path, policy)
        moving_path, reference_path = self.extract_pair(
            policy=ratified_path, moving_config=moving_config,
            reference_config=reference_config, qualified_geometry=True)
        moving = json.loads(moving_path.read_text(encoding="utf-8"))
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        for moving_sample, reference_sample in zip(
                moving["samples"], reference["samples"]):
            region = moving_sample["region"]
            self.assertEqual(region["id"], extractor.APERTURE_REGION_ID)
            self.assertEqual(region["aperture_id"],
                             "moving_nested_fixed_100x96_v1")
            lower = region["source_index_range"]["lower"]
            upper = region["source_index_range"]["upper"]
            self.assertEqual((upper["i"] - lower["i"] + 1,
                              upper["j"] - lower["j"] + 1), (100, 96))
            operational = region[
                "operational_interior_source_index_range"]
            self.assertEqual((
                operational["upper"]["i"] - operational["lower"]["i"] + 1,
                operational["upper"]["j"] - operational["lower"]["j"] + 1,
            ), (140, 150))
            self.assertEqual(region["earth_bounds_m"],
                             reference_sample["region"]["earth_bounds_m"])

    def test_enlarged_child_without_aperture_binding_fails_closed(self) -> None:
        policy, moving_config, _ = self.ratified_geometry_policy()
        policy["expected_contract"].pop("qualification_aperture")
        ratified_path = self.root / "ratified_policy.json"
        write_json(ratified_path, policy)
        run = make_run(
            self.root, "moving_child", moving_config, self.binary,
            surface_shape=(158, 168), spacing_m=600.0)
        with self.assertRaisesRegex(
                extractor.ExtractionError,
                "lacks qualification_aperture binding"):
            extractor.extract(
                run, "moving_child", moving_config, self.binary,
                ratified_path, self.root / "moving.json",
                self.root / "moving.source.json")

    def test_aperture_hash_mismatch_fails_before_field_measurement(self) -> None:
        policy, moving_config, _ = self.ratified_geometry_policy()
        policy["expected_contract"]["qualification_aperture"]["sha256"] = "0" * 64
        ratified_path = self.root / "ratified_policy.json"
        write_json(ratified_path, policy)
        run = make_run(
            self.root, "moving_child", moving_config, self.binary,
            surface_shape=(158, 168), spacing_m=600.0)
        with self.assertRaisesRegex(
                extractor.ExtractionError,
                "qualification_aperture hash mismatch"):
            extractor.extract(
                run, "moving_child", moving_config, self.binary,
                ratified_path, self.root / "moving.json",
                self.root / "moving.source.json")

    def test_wrong_operational_interior_shape_fails_closed(self) -> None:
        policy, moving_config, _ = self.ratified_geometry_policy()
        ratified_path = self.root / "ratified_policy.json"
        write_json(ratified_path, policy)
        run = make_run(
            self.root, "moving_child", moving_config, self.binary,
            surface_shape=(158, 166), spacing_m=600.0,
            times=tuple(float(value) for value in range(30, 4501, 30)))
        with self.assertRaisesRegex(
                extractor.ExtractionError,
                "operational child interior differs"):
            extractor.extract(
                run, "moving_child", moving_config, self.binary,
                ratified_path, self.root / "moving.json",
                self.root / "moving.source.json")

    def test_geometry_plan_equations_and_replay_are_fail_closed(self) -> None:
        plan_path = (
            ROOT / "tests" / "qualification" / "fixtures" /
            "moving_nested_supercell_geometry_plan_v1.json"
        )
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        extractor.validate_geometry_plan(plan)
        extractor.validate_geometry_plan_sources(plan_path, plan)

        changed_span = json.loads(json.dumps(plan))
        changed_span["scientific_inputs"][
            "maximum_authoritative_envelope_span_m"]["y"] += 600.0
        with self.assertRaisesRegex(
                extractor.ExtractionError, "scientific inputs changed"):
            extractor.validate_geometry_plan(changed_span)

        changed_move = json.loads(json.dumps(plan))
        changed_move["reference_driven_movement_replay"][
            "expected_moves"][-1]["j"] += 1
        with self.assertRaisesRegex(
                extractor.ExtractionError, "movement replay changed"):
            extractor.validate_geometry_plan(changed_move)

    def test_recorded_binary_mismatch_fails_closed(self) -> None:
        run = make_run(self.root, "moving_child", self.config, self.binary)
        (run / "qualification_binary.sha256").write_text("a" * 64 + "\n", encoding="ascii")
        with self.assertRaisesRegex(extractor.ExtractionError, "binary digest differs"):
            extractor.extract(
                run, "moving_child", self.config, self.binary, self.policy,
                self.root / "moving.json", self.root / "moving.source.json")

    def test_missing_required_publication_fails_closed(self) -> None:
        run = make_run(
            self.root, "moving_child", self.config, self.binary, omit_time=60.0)
        with self.assertRaisesRegex(extractor.ExtractionError, "required diagnostic"):
            extractor.extract(
                run, "moving_child", self.config, self.binary, self.policy,
                self.root / "moving.json", self.root / "moving.source.json")

    def test_reference_grid_phase_mismatch_fails_closed(self) -> None:
        moving_run = make_run(self.root, "moving_child", self.config, self.binary)
        moving_path = self.root / "moving.json"
        extractor.extract(
            moving_run, "moving_child", self.config, self.binary, self.policy,
            moving_path, self.root / "moving.source.json")
        reference_run = make_run(
            self.root, "oversized_reference", self.config, self.binary, phase_x=0.5)
        with self.assertRaisesRegex(extractor.ExtractionError, "grid phase differs"):
            extractor.extract(
                reference_run, "oversized_reference", self.config, self.binary,
                self.policy, self.root / "reference.json",
                self.root / "reference.source.json", moving_path)

    def test_manifest_mutation_after_extraction_is_incompatible(self) -> None:
        moving_path, reference_path = self.extract_pair()
        manifest = next((self.root / "moving_child_run" / "nests" /
                         "child_0000" / "qualification_diagnostics").glob(
                             "step_*/manifest.json"))
        manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
        report = comparator.compare_paths(moving_path, reference_path, self.policy)
        self.assertEqual(report["status"], "incompatible")
        self.assertTrue(any("hash mismatch" in value for value in report["contract_errors"]))

    def test_pair_with_different_actual_binary_is_incompatible(self) -> None:
        moving_run = make_run(self.root, "moving_child", self.config, self.binary)
        moving_path = self.root / "moving.json"
        extractor.extract(
            moving_run, "moving_child", self.config, self.binary, self.policy,
            moving_path, self.root / "moving.source.json")
        reference_binary = self.root / "other_nimbus_sim"
        reference_binary.write_bytes(b"different Nimbus binary\n")
        reference_run = make_run(
            self.root, "oversized_reference", self.config, reference_binary)
        reference_path = self.root / "reference.json"
        extractor.extract(
            reference_run, "oversized_reference", self.config, reference_binary,
            self.policy, reference_path, self.root / "reference.source.json", moving_path)
        report = comparator.compare_paths(moving_path, reference_path, self.policy)
        self.assertEqual(report["status"], "incompatible")
        self.assertIn("moving/reference binaries differ", report["contract_errors"])

    def test_pair_with_different_numerical_signature_is_incompatible(self) -> None:
        moving_run = make_run(self.root, "moving_child", self.config, self.binary)
        moving_path = self.root / "moving.json"
        extractor.extract(
            moving_run, "moving_child", self.config, self.binary, self.policy,
            moving_path, self.root / "moving.source.json")
        reference_config = self.root / "reference_config.yaml"
        reference_config.write_text(
            config_text().replace("damping:\n  enabled: false",
                                  "damping:\n  enabled: true"),
            encoding="utf-8",
        )
        reference_run = make_run(
            self.root, "oversized_reference", reference_config, self.binary)
        reference_path = self.root / "reference.json"
        extractor.extract(
            reference_run, "oversized_reference", reference_config, self.binary,
            self.policy, reference_path, self.root / "reference.source.json", moving_path)
        report = comparator.compare_paths(moving_path, reference_path, self.policy)
        self.assertEqual(report["status"], "incompatible")
        self.assertIn("moving/reference physics signatures differ",
                      report["contract_errors"])

    def test_effective_child_timestep_matches_dedicated_reference(self) -> None:
        moving_config = self.root / "moving_config.yaml"
        moving_config.write_text(
            config_text()
            .replace("  dx: 1\n  dy: 1", "  dx: 2\n  dy: 2")
            .replace("  dt_large: 1\n", "  dt_large: 1.5\n")
            .replace("    parent_ratio_x: 1", "    parent_ratio_x: 2")
            .replace("    parent_ratio_y: 1", "    parent_ratio_y: 2")
            .replace("    parent_time_step_ratio: 1",
                     "    parent_time_step_ratio: 2"),
            encoding="utf-8",
        )
        reference_config = self.root / "reference_config.yaml"
        reference_config.write_text(
            config_text().replace("  dt_large: 1\n", "  dt_large: 0.75\n"),
            encoding="utf-8",
        )
        moving_run = make_run(self.root, "moving_child", moving_config, self.binary)
        moving_path = self.root / "moving.json"
        extractor.extract(
            moving_run, "moving_child", moving_config, self.binary, self.policy,
            moving_path, self.root / "moving.source.json")
        reference_run = make_run(
            self.root, "oversized_reference", reference_config, self.binary)
        reference_path = self.root / "reference.json"
        extractor.extract(
            reference_run, "oversized_reference", reference_config, self.binary,
            self.policy, reference_path, self.root / "reference.source.json", moving_path)
        report = comparator.compare_paths(moving_path, reference_path, self.policy)
        self.assertEqual(report["status"], "measurement_only")
        moving_source = json.loads(
            (self.root / "moving.source.json").read_text(encoding="utf-8"))
        reference_source = json.loads(
            (self.root / "reference.source.json").read_text(encoding="utf-8"))
        moving_signature = moving_source["numerical_signature"]
        reference_signature = reference_source["numerical_signature"]
        self.assertEqual(moving_signature["sha256"], reference_signature["sha256"])
        self.assertEqual(moving_signature["derivation"]["root_dt_large_s"], 1.5)
        self.assertEqual(moving_signature["derivation"]["effective_dt_large_s"], 0.75)
        self.assertEqual(reference_signature["derivation"]["root_dt_large_s"], 0.75)

    def test_effective_child_timestep_mismatch_is_incompatible(self) -> None:
        moving_config = self.root / "moving_config.yaml"
        moving_config.write_text(
            config_text()
            .replace("  dx: 1\n  dy: 1", "  dx: 2\n  dy: 2")
            .replace("  dt_large: 1\n", "  dt_large: 1.5\n")
            .replace("    parent_ratio_x: 1", "    parent_ratio_x: 2")
            .replace("    parent_ratio_y: 1", "    parent_ratio_y: 2")
            .replace("    parent_time_step_ratio: 1",
                     "    parent_time_step_ratio: 2"),
            encoding="utf-8",
        )
        reference_config = self.root / "reference_config.yaml"
        reference_config.write_text(
            config_text().replace("  dt_large: 1\n", "  dt_large: 0.8\n"),
            encoding="utf-8",
        )
        moving_run = make_run(self.root, "moving_child", moving_config, self.binary)
        moving_path = self.root / "moving.json"
        extractor.extract(
            moving_run, "moving_child", moving_config, self.binary, self.policy,
            moving_path, self.root / "moving.source.json")
        reference_run = make_run(
            self.root, "oversized_reference", reference_config, self.binary)
        reference_path = self.root / "reference.json"
        extractor.extract(
            reference_run, "oversized_reference", reference_config, self.binary,
            self.policy, reference_path, self.root / "reference.source.json", moving_path)
        report = comparator.compare_paths(moving_path, reference_path, self.policy)
        self.assertEqual(report["status"], "incompatible")
        self.assertIn("moving/reference physics signatures differ",
                      report["contract_errors"])

    def test_sub_ulp_residual_is_not_claimed_lossless_after_publication(self) -> None:
        run = make_run(self.root, "moving_child", self.config, self.binary)
        exact_depth = 1.0 + 2.0 ** -25
        published_depth = struct.unpack("<f", struct.pack("<f", exact_depth))[0]
        self.assertEqual(published_depth, 1.0)
        for path in (run / "nests" / "child_0000" /
                     "qualification_diagnostics").glob(
                "step_*/accumulated_rainfall.npy"):
            write_npy(path, (8, 8), [published_depth] * 64)
        output = self.root / "moving.json"
        extractor.extract(
            run, "moving_child", self.config, self.binary, self.policy,
            output, self.root / "moving.source.json")
        document = json.loads(output.read_text(encoding="utf-8"))
        precipitation = document["samples"][0]["precipitation"]
        self.assertEqual(precipitation["effective_depth_semantics"],
                         "residual_inclusive_then_float32_published_v1")
        self.assertEqual(precipitation["maximum_depth_mm"], 1.0)
        self.assertEqual(precipitation["accumulated_mass_kg"], 16.0)
        self.assertGreater(16.0 * exact_depth, precipitation["accumulated_mass_kg"])

    def test_production_reference_matches_moving_child_numerical_signature(self) -> None:
        moving_config = (
            ROOT / "tests" / "qualification" / "fixtures" /
            "moving_nested_supercell_lifetime.yaml"
        )
        reference_config = (
            ROOT / "tests" / "fixtures" / "configs" / "qualification" /
            "moving_nested_supercell_reference_v5.yaml"
        )
        spacing = {"x": 600.0, "y": 600.0}
        moving_values, moving_digest, moving_derivation = (
            extractor.numerical_signature(
                moving_config, "moving_child", spacing)
        )
        reference_values, reference_digest, reference_derivation = (
            extractor.numerical_signature(
                reference_config, "oversized_reference", spacing)
        )
        self.assertEqual(moving_values, reference_values)
        self.assertEqual(moving_digest, reference_digest)
        self.assertEqual(
            moving_derivation["effective_dt_large_s"], 0.75)
        self.assertEqual(
            reference_derivation["effective_dt_large_s"], 0.75)
        moving_flat = extractor.flat_yaml_scalars(moving_config)
        reference_flat = extractor.flat_yaml_scalars(reference_config)
        moving_nx = int(moving_flat["grid.nx"])
        moving_ny = int(moving_flat["grid.ny"])
        moving_dx = float(moving_flat["grid.dx"])
        moving_dy = float(moving_flat["grid.dy"])
        reference_nx = int(reference_flat["grid.nx"])
        reference_ny = int(reference_flat["grid.ny"])
        reference_dx = float(reference_flat["grid.dx"])
        reference_dy = float(reference_flat["grid.dy"])
        self.assertLessEqual(reference_dx, moving_dx)
        self.assertLessEqual(reference_dy, moving_dy)
        self.assertGreaterEqual(
            (reference_nx - 1) * reference_dx,
            (moving_nx - 1) * moving_dx,
        )
        self.assertGreaterEqual(
            (reference_ny - 1) * reference_dy,
            (moving_ny - 1) * moving_dy,
        )
        self.assertEqual(
            reference_flat["output.qualification_diagnostics.preset"],
            "moving_reference_v1",
        )
        self.assertEqual(
            float(reference_flat[
                "output.qualification_diagnostics.interval_s"]), 30.0)
        self.assertEqual(float(reference_flat["time.output_interval"]), 450.0)


if __name__ == "__main__":
    unittest.main()
