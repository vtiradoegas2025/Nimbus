#!/usr/bin/env python3

import importlib.util
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "qualification" / "nested_domains"
sys.path.insert(0, str(SCRIPTS))


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


campaign = load_script("qualify_moving_nested_supercell")
restart = load_script("qualify_moving_nested_supercell_restart")


def authority(**overrides):
    value = {
        "algorithm": campaign.MOVEMENT_ALGORITHM,
        "mode": "live_one_way_nested_child",
        "enabled": True,
        "nested_movement_enabled": True,
        "recycling_enabled": False,
        "inner_margin_m": campaign.INNER_MARGIN_M,
        "release_margin_m": campaign.RELEASE_MARGIN_M,
        "maximum_shift_cells": campaign.MAXIMUM_SHIFT_CELLS,
        "cooldown_s": 300.0,
        "confirmation_samples_required": 2,
        "legal_parent_shift_bounds_cells": {
            "available": True,
            "minimum_i": -52,
            "maximum_i": 0,
            "minimum_j": -21,
            "maximum_j": 0,
        },
        "commit_performed": False,
    }
    value.update(overrides)
    return value


def selected_record(envelope, distances, movement=None):
    return {
        "simulation_time_s": 720,
        "selected_feature_available": True,
        "motion_available": True,
        "envelope_faces_m": envelope,
        "distance_to_physical_boundary_m": distances,
        "continuity": {
            "status": "tracking",
            "reason": "none",
            "branch_id": 1,
            "selected_candidate_available": True,
            "selected_candidate_rank": 0,
            "selected_component_label": 7,
            "topology": "isolated",
            "pending_topology": "isolated",
            "pending_topology_samples": 0,
            "topology_event": "none",
            "topology_event_id": 0,
            "topology_reorganization": False,
            "history_samples": 4,
            "missed_samples": 0,
        },
        "movement_authority": movement or authority(),
    }


def write_checkpoint(directory: Path, step: int, time_s: float,
                     payload: bytes = b"checkpoint",
                     **sidecar_overrides) -> Path:
    path = directory / f"checkpoint_step_{step:09d}.nimbus"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    sidecar = {
        "schema": "nimbus.checkpoint.lifecycle",
        "schema_version": 1,
        "checkpoint_file": path.name,
        "checkpoint_bytes": len(payload),
        "checkpoint_fnv1a64": restart.fnv1a64(path),
        "run_id": "test-run",
        "parent_run_id": "",
        "parent_checkpoint": "",
        "completed_steps": step,
        "simulation_time_s": time_s,
        "boundary": "completed_timestep",
        "portable_across_compute_memory_topology": True,
        "domain_recycling": None,
    }
    sidecar.update(sidecar_overrides)
    Path(f"{path}.json").write_text(
        json.dumps(sidecar) + "\n", encoding="utf-8")
    return path


def write_reusable_continuous_authority(directory: Path,
                                        binary: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(campaign.CONFIG, directory / "qualification_input.yaml")
    (directory / "qualification_binary.sha256").write_text(
        campaign.digest(binary) + "\n", encoding="utf-8")
    (directory / "nimbus_output.json").write_text(
        json.dumps({
            "publication_state": "ready",
            "data_products": ["history", "nested_child_history"],
        }) + "\n", encoding="utf-8")
    write_checkpoint(
        directory / "checkpoints", 600, 900.0, b"pre-move",
        run_id="uninterrupted-authority")
    write_checkpoint(
        directory / "checkpoints", 900, 1350.0, b"endpoint",
        run_id="uninterrupted-authority")

    first = {
        "completed_step": 620,
        "simulation_time_s": 930.0,
        "selected_feature_available": True,
        "moving_nest_scientific_validity": {
            "owner": "physical_child_trusted_interior_v1",
            "status": "valid",
            "reason": "none",
            "emergency_diagnostic_only": False,
            "relaxation_width_m": {"x": 4800.0, "y": 4800.0},
            "trusted_interior": True,
        },
        "envelope_face_motion": {"available": True},
        "movement_authority": {"commit_performed": False},
    }
    moved = {
        "completed_step": 800,
        "simulation_time_s": 1200.0,
        "selected_feature_available": True,
        "moving_nest_scientific_validity": dict(
            first["moving_nest_scientific_validity"]),
        "movement_authority": {"commit_performed": True},
    }
    tracker = directory / "storm_track"
    tracker.mkdir()
    (tracker / "track_step_000000620.json").write_text(
        json.dumps(first) + "\n", encoding="utf-8")
    (tracker / "track_step_000000800.json").write_text(
        json.dumps(moved) + "\n", encoding="utf-8")


def write_resumed_lineage(directory: Path,
                          split: restart.CheckpointLifecycle,
                          branch_run_id: str = "resumed-branch") -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    parent_checkpoint = str(split.path.resolve())
    (directory / "nimbus_branch.json").write_text(
        json.dumps({
            "schema": "nimbus.checkpoint.branch",
            "schema_version": 1,
            "run_id": branch_run_id,
            "parent_run_id": split.run_id,
            "parent_checkpoint": parent_checkpoint,
            "resumed_step": split.completed_steps,
            "resumed_simulation_time_s": split.simulation_time_s,
        }) + "\n", encoding="utf-8")
    tracker = directory / "storm_track"
    tracker.mkdir()
    (tracker /
     f"resume_step_{split.completed_steps:09d}_{branch_run_id}.json").write_text(
        json.dumps({
            "schema": "nimbus.storm_track.resume",
            "schema_version": 1,
            "branch_run_id": branch_run_id,
            "parent_run_id": split.run_id,
            "parent_checkpoint": parent_checkpoint,
            "parent_checkpoint_fnv1a64": split.checkpoint_fnv1a64,
            "resumed_step": split.completed_steps,
            "resumed_simulation_time_s": split.simulation_time_s,
            "history_mode": "new_branch",
            "preserved_history_records": 0,
            "preserved_last_completed_step": None,
        }) + "\n", encoding="utf-8")
    return [
        write_checkpoint(
            directory / "checkpoints", 900, 1350.0, b"resumed-endpoint",
            run_id=branch_run_id, parent_run_id=split.run_id,
            parent_checkpoint=parent_checkpoint)
    ]


def rewrite_json(path: Path, **updates) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update(updates)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class MovingNestedQualificationTests(unittest.TestCase):
    @staticmethod
    def precipitation_relocation():
        return {
            "owner": "earth_fixed_surface_history_v1",
            "schema_version": 1,
            "active": True,
            "sample_time_s": 720.0,
            "newly_admitted_columns": 24,
            "previous_child_accumulated_water_m3": 9.0,
            "discarded_child_accumulated_water_m3": 2.0,
            "retained_child_accumulated_water_m3": 7.0,
            "newly_admitted_parent_accumulated_water_m3": 3.0,
            "committed_child_accumulated_water_m3": 10.0,
            "previous_child_precipitation_volume_rate_m3s": 0.9,
            "discarded_child_precipitation_volume_rate_m3s": 0.2,
            "retained_child_precipitation_volume_rate_m3s": 0.7,
            "newly_admitted_parent_precipitation_volume_rate_m3s": 0.3,
            "committed_child_precipitation_volume_rate_m3s": 1.0,
        }

    def test_science_inventory_retains_authoritative_cold_pool_inputs(self):
        self.assertEqual(
            campaign.VOLUME_FIELDS,
            {"p", "qc", "qr", "qv", "rho", "theta", "u", "v", "w"})
        self.assertEqual(
            campaign.PLANAR_FIELDS,
            {"cold_pool_boundary", "precip_rate", "accumulated_rainfall"})
        self.assertIn("  preset: science_validation\n",
                      campaign.CONFIG.read_text(encoding="utf-8"))

    def test_active_footprint_is_bound_to_observed_morphology(self):
        contract = campaign.movement_geometry_contract()
        self.assertEqual(
            contract["physical_child_extent_m"],
            {"x": 93600.0, "y": 99600.0})
        self.assertEqual(contract["parent_lattice_phase_allowance_m"], 1200.0)
        self.assertEqual(contract["inner_margin_m"], 15600.0)
        self.assertEqual(contract["release_margin_m"], 16800.0)
        self.assertEqual(contract["required_clean_shift_cells"], 10)
        self.assertEqual(contract["required_maximum_shift_cells"], 14)

    def test_reference_is_bound_to_committed_evidence_and_config(self):
        binding = campaign.reference_binding()
        self.assertEqual(binding["summary"], campaign.REFERENCE)
        self.assertEqual(binding["config_sha256"], campaign.digest(campaign.REFERENCE_CONFIG))
        self.assertEqual(len(binding["evidence_sha256"]), 64)

    def test_campaign_artifacts_require_exact_config_and_binary_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            shutil.copyfile(campaign.CONFIG, output / "qualification_input.yaml")
            (output / "qualification_binary.sha256").write_text("a" * 64 + "\n")
            _, binary_digest, _ = campaign.campaign_artifact_binding(output)
            self.assertEqual(binary_digest, "a" * 64)
            (output / "qualification_input.yaml").write_text("schema: nimbus.v1\n")
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                campaign.campaign_artifact_binding(output)

    def test_valid_nonmoving_envelope_stays_outside_relaxation(self):
        envelope = {"west": 20400.0, "east": 78600.0,
                    "south": 19800.0, "north": 84000.0}
        distances = {"west": 16800.0, "east": 18600.0,
                     "south": 17400.0, "north": 18000.0}
        record = selected_record(envelope, distances)
        campaign.require_authority_contract(record["movement_authority"])
        pre, post, span_x, span_y = campaign.selected_envelope_geometry(
            record, campaign.CHILD_INITIAL_STORAGE_ORIGIN_EARTH_M,
            campaign.CHILD_INITIAL_STORAGE_ORIGIN_EARTH_M)
        self.assertEqual(pre, post)
        self.assertEqual(span_x, 58200.0)
        self.assertEqual(span_y, 64200.0)

    def test_valid_commit_restores_complete_release_box(self):
        envelope = {"west": 24600.0, "east": 82800.0,
                    "south": 18600.0, "north": 82800.0}
        distances = {"west": 21000.0, "east": 14400.0,
                     "south": 16200.0, "north": 19200.0}
        movement = authority(commit_performed=True)
        record = selected_record(envelope, distances, movement)
        post_origin = (6600.0, 600.0)
        _, post, _, _ = campaign.selected_envelope_geometry(
            record, campaign.CHILD_INITIAL_STORAGE_ORIGIN_EARTH_M, post_origin)
        self.assertTrue(campaign.release_box_recovered(post))
        self.assertEqual(post["west"], 17400.0)
        self.assertEqual(post["east"], 18000.0)

    def test_rejects_pre_face_motion_movement_provenance(self):
        with self.assertRaisesRegex(RuntimeError, "v14"):
            campaign.require_authority_contract(
                authority(algorithm="qualified_envelope_hysteresis_v4"))

    def test_rejects_missing_or_malformed_parent_stencil_bounds(self):
        missing = authority()
        del missing["legal_parent_shift_bounds_cells"]
        with self.assertRaisesRegex(RuntimeError, "parent-stencil bounds"):
            campaign.require_authority_contract(missing)
        malformed = authority()
        malformed["legal_parent_shift_bounds_cells"]["minimum_i"] = 1
        with self.assertRaisesRegex(RuntimeError, "malformed"):
            campaign.require_authority_contract(malformed)

    def test_commit_origin_and_cumulative_shift_are_exact(self):
        movement = authority(
            committed_shift_cells={"i": 4, "j": -1},
            requested_shift_cells={"i": 4, "j": -1},
            post_shift_child_grid_origin_t0_earth_m={"x": 7800.0, "y": 600.0},
            cumulative_committed_shift_cells={"i": 7, "j": 1})
        origin, cumulative_i, cumulative_j = (
            campaign.committed_origin_and_cumulative(
                movement, campaign.CHILD_INITIAL_STORAGE_ORIGIN_EARTH_M,
                3, 2))
        self.assertEqual(origin, (7800.0, 600.0))
        self.assertEqual((cumulative_i, cumulative_j), (7, 1))
        movement["post_shift_child_grid_origin_t0_earth_m"]["x"] = 6600.0
        with self.assertRaisesRegex(RuntimeError, "integer-parent shift"):
            campaign.committed_origin_and_cumulative(
                movement, campaign.CHILD_INITIAL_STORAGE_ORIGIN_EARTH_M,
                3, 2)

    def test_precipitation_relocation_provenance_closes_surface_history(self):
        report = self.precipitation_relocation()
        campaign.require_precipitation_relocation(
            authority(commit_performed=True,
                      precipitation_relocation=report), 720.0)
        campaign.require_precipitation_relocation(
            authority(precipitation_relocation=None), 690.0)

        missing = authority(commit_performed=True,
                            precipitation_relocation=None)
        with self.assertRaisesRegex(RuntimeError, "lacks precipitation"):
            campaign.require_precipitation_relocation(missing, 720.0)

        unclosed = dict(report)
        unclosed["committed_child_accumulated_water_m3"] = 11.0
        with self.assertRaisesRegex(RuntimeError, "close committed"):
            campaign.require_precipitation_relocation(
                authority(commit_performed=True,
                          precipitation_relocation=unclosed), 720.0)

    def test_committed_shift_cap_is_exact(self):
        movement = authority(
            committed_shift_cells={"i": campaign.MAXIMUM_SHIFT_CELLS, "j": 0},
            requested_shift_cells={"i": campaign.MAXIMUM_SHIFT_CELLS, "j": 0},
            post_shift_child_grid_origin_t0_earth_m={
                "x": (campaign.CHILD_INITIAL_STORAGE_ORIGIN_EARTH_M[0] +
                      campaign.MAXIMUM_SHIFT_CELLS *
                      campaign.PARENT_SPACING_M[0]),
                "y": campaign.CHILD_INITIAL_STORAGE_ORIGIN_EARTH_M[1],
            },
            cumulative_committed_shift_cells={
                "i": campaign.MAXIMUM_SHIFT_CELLS, "j": 0})
        _, cumulative_i, cumulative_j = campaign.committed_origin_and_cumulative(
            movement, campaign.CHILD_INITIAL_STORAGE_ORIGIN_EARTH_M, 0, 0)
        self.assertEqual((cumulative_i, cumulative_j),
                         (campaign.MAXIMUM_SHIFT_CELLS, 0))

        movement["committed_shift_cells"] = {
            "i": campaign.MAXIMUM_SHIFT_CELLS + 1, "j": 0}
        movement["requested_shift_cells"] = movement["committed_shift_cells"]
        with self.assertRaisesRegex(RuntimeError, "exceeded its cap"):
            campaign.committed_origin_and_cumulative(
                movement, campaign.CHILD_INITIAL_STORAGE_ORIGIN_EARTH_M,
                0, 0)

    def test_rejects_every_boundary_sensitive_movement_reason(self):
        for reason in (
                "shift_limit_insufficient",
                "opposite_release_margin_insufficient",
                "opposite_faces_breached",
                "emergency_containment"):
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(RuntimeError, "boundary-sensitive"):
                    campaign.require_safe_movement_reasons({reason})
        campaign.require_safe_movement_reasons(
            {"request_ready", "confirmation_pending", "cooldown_active"})

    def test_envelope_face_rate_retains_confirmation_headroom(self):
        previous = {
            "simulation_time_s": 30.0,
            "envelope_faces_m": {
                "west": 10000.0, "east": 30000.0,
                "south": 12000.0, "north": 32000.0},
        }
        current = {
            "simulation_time_s": 60.0,
            "envelope_faces_m": {
                "west": 9400.0, "east": 31200.0,
                "south": 12000.0, "north": 32600.0},
        }
        speeds = campaign.outward_envelope_face_speeds(previous, current)
        self.assertEqual(speeds, {
            "west": 20.0, "east": 40.0,
            "south": 0.0, "north": 20.0})
        self.assertEqual(campaign.require_confirmation_headroom(40.0), 1200.0)
        with self.assertRaisesRegex(RuntimeError, "sizing authority"):
            campaign.require_confirmation_headroom(341.0)

    def test_reported_envelope_face_motion_is_signed_and_fail_closed(self):
        record = {
            "envelope_face_motion": {
                "available": True,
                "interval_s": 30.0,
                "west_coordinate_velocity_ms": -20.0,
                "east_coordinate_velocity_ms": 40.0,
                "south_coordinate_velocity_ms": 0.0,
                "north_coordinate_velocity_ms": 20.0,
            }
        }
        self.assertEqual(
            campaign.reported_envelope_face_motion(record),
            {"west": -20.0, "east": 40.0,
             "south": 0.0, "north": 20.0})
        unavailable = {
            "envelope_face_motion": {
                "available": False,
                "interval_s": None,
                "west_coordinate_velocity_ms": None,
                "east_coordinate_velocity_ms": None,
                "south_coordinate_velocity_ms": None,
                "north_coordinate_velocity_ms": None,
            }
        }
        self.assertIsNone(campaign.reported_envelope_face_motion(unavailable))
        unavailable["envelope_face_motion"]["east_coordinate_velocity_ms"] = 1.0
        with self.assertRaisesRegex(RuntimeError, "invented"):
            campaign.reported_envelope_face_motion(unavailable)

    def test_runtime_scientific_validity_owns_trusted_interior(self):
        valid = {
            "selected_feature_available": True,
            "movement_authority": {"status": "requested"},
            "moving_nest_scientific_validity": {
                "owner": "physical_child_trusted_interior_v1",
                "status": "valid",
                "reason": "none",
                "trusted_interior": True,
                "relaxation_width_m": {"x": 4800.0, "y": 4800.0},
                "emergency_diagnostic_only": False,
            },
        }
        campaign.require_runtime_scientific_validity(valid)
        invalid = dict(valid)
        invalid["moving_nest_scientific_validity"] = dict(
            valid["moving_nest_scientific_validity"],
            status="invalid", reason="relaxation_zone_entry",
            trusted_interior=False)
        with self.assertRaisesRegex(RuntimeError, "scientifically invalid"):
            campaign.require_runtime_scientific_validity(invalid)
        emergency = dict(valid)
        emergency["moving_nest_scientific_validity"] = dict(
            valid["moving_nest_scientific_validity"],
            emergency_diagnostic_only=True)
        with self.assertRaisesRegex(RuntimeError, "emergency"):
            campaign.require_runtime_scientific_validity(emergency)

    def test_startup_miss_is_unassessed_but_mature_track_interruption_is_invalid(self):
        startup = {
            "selected_feature_available": False,
            "movement_authority": {"status": "ineligible"},
            "moving_nest_scientific_validity": {
                "owner": "physical_child_trusted_interior_v1",
                "status": "not_assessed",
                "reason": "no_selected_feature",
                "trusted_interior": None,
                "relaxation_width_m": {"x": 4800.0, "y": 4800.0},
                "emergency_diagnostic_only": False,
            },
        }
        campaign.require_runtime_scientific_validity(startup)
        self.assertFalse(
            campaign.require_uninterrupted_authoritative_track(startup, False))

        tracking = {
            **startup,
            "selected_feature_available": True,
            "continuity": {"status": "tracking"},
        }
        self.assertTrue(
            campaign.require_uninterrupted_authoritative_track(tracking, False))
        with self.assertRaisesRegex(RuntimeError, "authority was interrupted"):
            campaign.require_uninterrupted_authoritative_track(startup, True)
        reorganized = {
            **tracking,
            "continuity": {
                "status": "tentative",
                "topology_reorganization": True,
            },
        }
        with self.assertRaisesRegex(RuntimeError, "authority was interrupted"):
            campaign.require_uninterrupted_authoritative_track(
                reorganized, True)

        interrupted = {
            **startup,
            "moving_nest_scientific_validity": {
                **startup["moving_nest_scientific_validity"],
                "status": "invalid",
                "reason": "authoritative_track_interrupted",
            },
        }
        with self.assertRaisesRegex(RuntimeError, "scientifically invalid"):
            campaign.require_runtime_scientific_validity(interrupted)

        interrupted_with_selection = {
            **interrupted,
            "selected_feature_available": True,
        }
        with self.assertRaisesRegex(RuntimeError, "scientifically invalid"):
            campaign.require_runtime_scientific_validity(
                interrupted_with_selection)

    def test_first_pending_relocation_remains_scientifically_valid(self):
        pending = {
            "selected_feature_available": True,
            "movement_authority": {"status": "pending_confirmation"},
            "moving_nest_scientific_validity": {
                "owner": "physical_child_trusted_interior_v1",
                "status": "valid_pending_relocation",
                "reason": "confirmation_pending_outside_relaxation",
                "trusted_interior": True,
                "relaxation_width_m": {"x": 4800.0, "y": 4800.0},
                "emergency_diagnostic_only": False,
            },
        }
        campaign.require_runtime_scientific_validity(pending)
        pending["moving_nest_scientific_validity"]["status"] = "valid"
        with self.assertRaisesRegex(RuntimeError, "pending"):
            campaign.require_runtime_scientific_validity(pending)

    def test_rejects_published_pre_shift_distance_mismatch(self):
        envelope = {"west": 20400.0, "east": 56400.0,
                    "south": 19800.0, "north": 52800.0}
        distances = {"west": 16800.0, "east": 16000.0,
                     "south": 17400.0, "north": 16800.0}
        with self.assertRaisesRegex(RuntimeError, "disagrees"):
            campaign.selected_envelope_geometry(
                selected_record(envelope, distances),
                campaign.CHILD_INITIAL_STORAGE_ORIGIN_EARTH_M,
                campaign.CHILD_INITIAL_STORAGE_ORIGIN_EARTH_M)

    def test_rejects_envelope_inside_boundary_relaxation(self):
        envelope = {"west": 34800.0, "east": 93000.0,
                    "south": 19800.0, "north": 84000.0}
        distances = {"west": 31200.0, "east": 4200.0,
                     "south": 17400.0, "north": 18000.0}
        with self.assertRaisesRegex(RuntimeError, "boundary-relaxation"):
            campaign.selected_envelope_geometry(
                selected_record(envelope, distances),
                campaign.CHILD_INITIAL_STORAGE_ORIGIN_EARTH_M,
                campaign.CHILD_INITIAL_STORAGE_ORIGIN_EARTH_M)

    def test_rejects_commit_without_release_recovery(self):
        envelope = {"west": 24600.0, "east": 82800.0,
                    "south": 18600.0, "north": 82800.0}
        distances = {"west": 21000.0, "east": 14400.0,
                     "south": 16200.0, "north": 19200.0}
        record = selected_record(
            envelope, distances, authority(commit_performed=True))
        with self.assertRaisesRegex(RuntimeError, "release box"):
            campaign.selected_envelope_geometry(
                record, campaign.CHILD_INITIAL_STORAGE_ORIGIN_EARTH_M,
                (4200.0, 1800.0))

    def test_restart_checkpoint_is_selected_from_observed_first_move(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            paths = [
                write_checkpoint(directory, 300, 455.0, b"first"),
                write_checkpoint(directory, 600, 910.0, b"second"),
            ]
            selected = restart.select_restart_checkpoint(paths, 780.0)
            self.assertEqual(selected.path, paths[0])
            self.assertEqual(selected.simulation_time_s, 455.0)
            self.assertEqual(
                restart.checkpoint_simulation_time_s(selected.path), 455.0)
            with self.assertRaisesRegex(RuntimeError, "precedes"):
                restart.select_restart_checkpoint(paths, 300.0)

    def test_restart_checkpoint_rejects_corrupt_lifecycle_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_checkpoint(Path(temporary), 300, 450.0)
            path.write_bytes(b"checkpoinu")
            with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
                restart.load_checkpoint_lifecycle(path)

    def test_final_restart_checkpoint_requires_exact_endpoint_sidecars(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            continuous = write_checkpoint(
                root / "continuous", 900, 1350.0, b"exact-state",
                run_id="continuous")
            resumed = write_checkpoint(
                root / "resumed", 900, 1350.0, b"exact-state",
                run_id="resumed", parent_run_id="continuous")
            lifecycle = restart.require_final_checkpoint_exact(
                [continuous], [resumed], 1350)
            self.assertEqual(lifecycle.completed_steps, 900)
            self.assertEqual(lifecycle.checkpoint_fnv1a64,
                             restart.fnv1a64(continuous))
            write_checkpoint(
                root / "resumed", 900, 1349.0, b"exact-state",
                run_id="resumed", parent_run_id="continuous")
            with self.assertRaisesRegex(RuntimeError, "requested endpoint"):
                restart.require_final_checkpoint_exact(
                    [continuous], [resumed], 1350)

    def test_resumed_lineage_binds_branch_marker_and_checkpoints_to_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_path = write_checkpoint(
                root / "continuous", 600, 900.0, b"split",
                run_id="continuous-run")
            split = restart.load_checkpoint_lifecycle(split_path)
            resumed_output = root / "resumed"
            resumed = write_resumed_lineage(resumed_output, split)
            lifecycles = restart.validate_resumed_lineage(
                resumed_output, split, resumed)
            self.assertEqual([item.run_id for item in lifecycles],
                             ["resumed-branch"])
            self.assertEqual(lifecycles[0].parent_run_id, "continuous-run")
            self.assertEqual(Path(lifecycles[0].parent_checkpoint),
                             split_path.resolve())

    def test_resumed_lineage_rejects_disagreement_at_each_provenance_layer(self):
        mutations = (
            ("branch parent", "parent run identity",
             lambda output, split, resumed: rewrite_json(
                 output / "nimbus_branch.json", parent_run_id="wrong-run")),
            ("branch checkpoint", "not bound to the selected split",
             lambda output, split, resumed: rewrite_json(
                 output / "nimbus_branch.json",
                 parent_checkpoint=str(resumed[0].resolve()))),
            ("marker digest", "marker digest",
             lambda output, split, resumed: rewrite_json(
                 next((output / "storm_track").glob("resume_step_*.json")),
                 parent_checkpoint_fnv1a64="0" * 16)),
            ("checkpoint branch", "checkpoint run identity",
             lambda output, split, resumed: rewrite_json(
                 Path(f"{resumed[0]}.json"), run_id="wrong-branch")),
            ("checkpoint parent", "checkpoint parent run identity",
             lambda output, split, resumed: rewrite_json(
                 Path(f"{resumed[0]}.json"), parent_run_id="wrong-run")),
            ("checkpoint path", "not bound to the selected split",
             lambda output, split, resumed: rewrite_json(
                 Path(f"{resumed[0]}.json"),
                 parent_checkpoint=str(resumed[0].resolve()))),
        )
        for label, error, mutate in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                split_path = write_checkpoint(
                    root / "continuous", 600, 900.0, b"split",
                    run_id="continuous-run")
                split = restart.load_checkpoint_lifecycle(split_path)
                resumed_output = root / "resumed"
                resumed = write_resumed_lineage(resumed_output, split)
                mutate(resumed_output, split, resumed)
                with self.assertRaisesRegex(RuntimeError, error):
                    restart.validate_resumed_lineage(
                        resumed_output, split, resumed)

    def test_resumed_lineage_rejects_duplicate_marker_and_unordered_lifecycle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_path = write_checkpoint(
                root / "continuous", 600, 900.0, b"split",
                run_id="continuous-run")
            split = restart.load_checkpoint_lifecycle(split_path)
            resumed_output = root / "resumed"
            resumed = write_resumed_lineage(resumed_output, split)
            marker = next((resumed_output / "storm_track").glob(
                "resume_step_*.json"))
            shutil.copyfile(
                marker, resumed_output / "storm_track" /
                "resume_step_000000600_duplicate.json")
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                restart.validate_resumed_lineage(
                    resumed_output, split, resumed)

            (resumed_output / "storm_track" /
             "resume_step_000000600_duplicate.json").unlink()
            earlier = write_checkpoint(
                resumed_output / "checkpoints", 800, 1400.0, b"earlier-name",
                run_id="resumed-branch", parent_run_id=split.run_id,
                parent_checkpoint=str(split.path.resolve()))
            with self.assertRaisesRegex(RuntimeError, "strictly ordered"):
                restart.validate_resumed_lineage(
                    resumed_output, split, sorted([earlier, *resumed]))

    def test_resumed_tracker_contract_accepts_complete_suffix_evidence(self):
        record = selected_record(
            {"west": 0.0, "east": 1.0, "south": 0.0, "north": 1.0},
            {"west": 1.0, "east": 1.0, "south": 1.0, "north": 1.0})
        record.update({
            "schema_version": 12,
            "tracking_source": "synchronized_nested_child",
            "detector_coordinate_frame": "earth_absolute",
            "algorithm":
                "updraft_core_watershed_condensate_rotation_branches_v5",
            "continuity_algorithm":
                "fixed_window_temporal_branch_association_v5",
            "completed_step": 480,
            "moving_nest_scientific_validity": {
                "owner": "physical_child_trusted_interior_v1",
                "status": "valid",
                "reason": "none",
                "emergency_diagnostic_only": False,
                "relaxation_width_m": {"x": 4800.0, "y": 4800.0},
                "trusted_interior": True,
            },
        })
        restart.require_resumed_tracker_contract([record])

        wrong_tracker = copy.deepcopy(record)
        wrong_tracker["algorithm"] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "detector algorithm"):
            restart.require_resumed_tracker_contract([wrong_tracker])

        wrong_authority = copy.deepcopy(record)
        wrong_authority["movement_authority"]["inner_margin_m"] = 1.0
        with self.assertRaisesRegex(RuntimeError, "inner margin"):
            restart.require_resumed_tracker_contract([wrong_authority])

        wrong_precipitation = copy.deepcopy(record)
        wrong_precipitation["movement_authority"].update({
            "commit_performed": True,
            "precipitation_relocation": None,
        })
        with self.assertRaisesRegex(RuntimeError, "precipitation relocation"):
            restart.require_resumed_tracker_contract([wrong_precipitation])

    def test_restart_suffix_requires_restored_face_motion_before_move(self):
        first = selected_record(
            {"west": 0.0, "east": 1.0, "south": 0.0, "north": 1.0},
            {"west": 1.0, "east": 1.0, "south": 1.0, "north": 1.0})
        first["simulation_time_s"] = 480.0
        first["envelope_face_motion"] = {
            "available": True,
            "interval_s": 30.0,
            "west_coordinate_velocity_ms": -1.0,
            "east_coordinate_velocity_ms": 2.0,
            "south_coordinate_velocity_ms": 0.0,
            "north_coordinate_velocity_ms": 0.0,
        }
        moved = selected_record(
            first["envelope_faces_m"], first["distance_to_physical_boundary_m"],
            authority(commit_performed=True))
        moved["simulation_time_s"] = 720.0
        self.assertEqual(
            restart.require_post_split_face_motion_before_move([first, moved]),
            480.0)
        first["envelope_face_motion"]["available"] = False
        with self.assertRaisesRegex(RuntimeError, "restored envelope-face motion"):
            restart.require_post_split_face_motion_before_move([first, moved])

    def test_resumed_branch_rejects_scientific_invalidity_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            restart.require_no_scientific_invalidity_artifact(output)
            (output / "moving_nest_scientific_validity.json").write_text(
                "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "scientific-invalidity"):
                restart.require_no_scientific_invalidity_artifact(output)

    def test_reuses_completed_hash_bound_continuous_authority(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "nimbus_sim_internal_qualification"
            binary.write_bytes(b"qualification binary")
            output = root / "continuous"
            write_reusable_continuous_authority(output, binary)
            expected_report = {
                "status": "pass",
                "binary_sha256": campaign.digest(binary),
                "config_sha256": campaign.digest(campaign.CONFIG),
            }
            with mock.patch.object(
                    restart.campaign, "inspect",
                    return_value=expected_report) as inspect:
                reusable = restart.validate_continuous_authority(
                    output, binary, 1350)
            self.assertEqual(reusable.output, output.resolve())
            self.assertEqual(reusable.split_checkpoint.simulation_time_s, 900.0)
            self.assertEqual(reusable.first_movement_time_s, 1200.0)
            self.assertEqual(
                reusable.first_face_motion_sample_time_s, 930.0)
            self.assertEqual(reusable.report, expected_report)
            inspect.assert_called_once_with(output.resolve(), 1350, None, True)

    def test_reuse_mode_runs_only_the_resumed_suffix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "binary"
            binary.write_bytes(b"qualification binary")
            continuous_output = root / "continuous"
            continuous_output.mkdir()
            split_path = continuous_output / "checkpoint_step_000000600.nimbus"
            split = restart.CheckpointLifecycle(
                split_path, 600, 900.0, 8, "0" * 16,
                "continuous", "", "")
            reusable = restart.ContinuousAuthority(
                continuous_output.resolve(),
                {
                    "status": "pass",
                    "binary_sha256": campaign.digest(binary),
                    "config_sha256": campaign.digest(campaign.CONFIG),
                },
                [split_path], [], split, [], 1200.0, 930.0)
            final_checkpoint = restart.CheckpointLifecycle(
                root / "checkpoint_step_000000900.nimbus",
                900, 1350.0, 8, "1" * 16,
                "continuous", "", "")
            output = root / "restart-result"
            argv = [
                "qualify_moving_nested_supercell_restart.py",
                "--binary", str(binary),
                "--continuous-authority", str(continuous_output),
                "--resume-threads", "4",
                "--output", str(output),
                "--final-time", "1350",
            ]
            with (mock.patch.object(sys, "argv", argv),
                  mock.patch.object(
                      restart.campaign, "assert_public_admission_closed"),
                  mock.patch.object(
                      restart.campaign, "assert_internal_admission_exact"),
                  mock.patch.object(
                      restart, "validate_continuous_authority",
                      return_value=reusable) as validate,
                  mock.patch.object(
                      restart, "run_branch", return_value=(2.0, "")) as run,
                  mock.patch.object(restart, "validate_resumed_lineage") as lineage,
                  mock.patch.object(
                      restart, "require_final_checkpoint_exact",
                      return_value=final_checkpoint),
                  mock.patch.object(restart, "require_endpoint_exact"),
                  mock.patch.object(
                      restart, "inspect_resumed", return_value={}),
                  mock.patch("builtins.print")):
                self.assertEqual(restart.main(), 0)
            validate.assert_called_once_with(
                continuous_output.resolve(), binary.resolve(), 1350)
            resolved_output = output.resolve()
            run.assert_called_once_with(
                binary.resolve(), resolved_output / "resumed_output",
                resolved_output / "resumed_output" / "checkpoints",
                1350, 4, restart.CHECKPOINT_INTERVAL_S, split_path)
            lineage.assert_called_once_with(
                resolved_output / "resumed_output", split, [])
            report = json.loads(
                (resolved_output / "qualification_report.json").read_text(
                    encoding="utf-8"))
            self.assertTrue(report["continuous_authority_reused"])
            self.assertIsNone(report["continuous_threads"])
            self.assertIsNone(report["changed_thread_count"])

    def test_reused_continuous_authority_rejects_stale_endpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "binary"
            binary.write_bytes(b"qualification binary")
            output = root / "continuous"
            write_reusable_continuous_authority(output, binary)
            write_checkpoint(
                output / "checkpoints", 900, 1349.0, b"endpoint",
                run_id="uninterrupted-authority")
            with self.assertRaisesRegex(RuntimeError, "stale"):
                restart.validate_continuous_authority(
                    output, binary, 1350)

    def test_reused_continuous_authority_rejects_wrong_binary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recorded_binary = root / "recorded-binary"
            recorded_binary.write_bytes(b"recorded")
            selected_binary = root / "selected-binary"
            selected_binary.write_bytes(b"selected")
            output = root / "continuous"
            write_reusable_continuous_authority(output, recorded_binary)
            with self.assertRaisesRegex(RuntimeError, "binary digest"):
                restart.validate_continuous_authority(
                    output, selected_binary, 1350)

    def test_reused_continuous_authority_rejects_wrong_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "binary"
            binary.write_bytes(b"qualification binary")
            output = root / "continuous"
            write_reusable_continuous_authority(output, binary)
            (output / "qualification_input.yaml").write_text(
                "schema: nimbus.v1\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                restart.validate_continuous_authority(
                    output, binary, 1350)

    def test_reused_continuous_authority_rejects_incomplete_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "binary"
            binary.write_bytes(b"qualification binary")
            output = root / "continuous"
            write_reusable_continuous_authority(output, binary)
            (output / "nimbus_output.json").write_text(
                json.dumps({
                    "publication_state": "writing",
                    "data_products": ["history", "nested_child_history"],
                }) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                restart.validate_continuous_authority(
                    output, binary, 1350)

    def test_restart_trace_preserves_multiple_dynamic_movements(self):
        records = []
        for time_s, sequence, shift in (
                (720, 1, {"i": 4, "j": 1}),
                (870, 2, {"i": 2, "j": -1})):
            movement = authority(
                status="requested", reason="request_ready",
                request_ready=True, requested_shift_cells=shift,
                confirmation_count=2, commit_performed=True,
                committed_shift_cells=shift,
                nested_movement_sequence=sequence,
                post_shift_child_grid_origin_t0_earth_m={"x": 0.0, "y": 0.0},
                armed=True,
                cumulative_committed_shift_cells={"i": 0, "j": 0})
            records.append(selected_record(
                {"west": 0.0, "east": 1.0, "south": 0.0, "north": 1.0},
                {"west": 1.0, "east": 1.0, "south": 1.0, "north": 1.0},
                movement))
            records[-1]["simulation_time_s"] = time_s
            records[-1]["envelope_face_motion"] = {
                "available": True,
                "interval_s": 30.0,
                "west_coordinate_velocity_ms": -1.0,
                "east_coordinate_velocity_ms": 2.0,
                "south_coordinate_velocity_ms": -3.0,
                "north_coordinate_velocity_ms": 4.0,
            }
            records[-1]["moving_nest_scientific_validity"] = {
                "owner": "physical_child_trusted_interior_v1",
                "status": "valid",
                "reason": "none",
            }
            records[-1]["movement_authority"]["precipitation_relocation"] = (
                self.precipitation_relocation())
        trace = campaign.movement_trace(records)
        self.assertEqual([item["nested_movement_sequence"] for item in trace], [1, 2])
        self.assertEqual([item["committed_shift_cells"] for item in trace],
                         [{"i": 4, "j": 1}, {"i": 2, "j": -1}])
        self.assertEqual(
            [item["envelope_face_motion"] for item in trace],
            [record["envelope_face_motion"] for record in records])
        self.assertEqual(
            [item["moving_nest_scientific_validity"] for item in trace],
            [record["moving_nest_scientific_validity"] for record in records])
        self.assertEqual(
            [item["precipitation_relocation"] for item in trace],
            [record["movement_authority"]["precipitation_relocation"]
             for record in records])
        self.assertEqual(
            [item["continuity_selected_component_label"] for item in trace],
            [7, 7])
        self.assertEqual(
            [item["continuity_topology_event"] for item in trace],
            ["none", "none"])


if __name__ == "__main__":
    unittest.main()
