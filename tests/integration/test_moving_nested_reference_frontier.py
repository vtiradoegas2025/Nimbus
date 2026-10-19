#!/usr/bin/env python3
"""Synthetic tests for the moving-nest reference-frontier comparator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
COMPARATOR_PATH = ROOT / "scripts" / "qualification" / "nested_domains" / "compare_moving_nested_supercell_reference.py"
POLICY_PATH = (
    ROOT
    / "tests"
    / "qualification"
    / "fixtures"
    / "moving_nested_supercell_reference_frontier_v1.json"
)
SPEC = importlib.util.spec_from_file_location(
    "compare_moving_nested_supercell_reference", COMPARATOR_PATH
)
assert SPEC is not None and SPEC.loader is not None
comparator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparator)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def derived_sample(index: int, offset: float) -> dict:
    return {
        "cold_pool": {
            "definition": "surface_density_potential_temperature_deficit_v1",
            "area_m2": 2.0e6 + index * 2.0e5 + offset * 1000.0,
            "integrated_deficit_k_m2": 4.0e6 + index * 4.0e5 + offset * 1000.0,
            "minimum_theta_rho_perturbation_k": -2.0 - 0.2 * index + offset * 0.001,
            "centroid_m": {"x": 10000.0 + 600.0 * index + offset,
                           "y": 12000.0 + 300.0 * index},
        },
        "precipitation": {
            "definition": "earth_aligned_accumulated_rainfall_v1",
            "accumulated_mass_kg": 1000.0 + 500.0 * index + offset,
            "footprint_area_m2": 1.0e6 + 2.0e5 * index + offset * 100.0,
            "p95_depth_mm": 2.0 + index + offset * 0.001,
            "maximum_depth_mm": 4.0 + 2.0 * index + offset * 0.001,
        },
    }


def make_series(root: Path, role: str, offset: float = 0.0) -> tuple[Path, dict]:
    artifacts: dict[str, dict[str, str]] = {}
    for name in sorted(comparator.REQUIRED_ARTIFACTS):
        artifact = root / f"{role}_{name}.dat"
        artifact.write_text(f"{role}:{name}\n", encoding="utf-8")
        artifacts[name] = {"path": artifact.name, "sha256": digest(artifact)}

    samples = []
    for index, time_s in enumerate((30.0, 60.0, 90.0)):
        sample = {
            "simulation_time_s": time_s,
            "selected": True,
            "core_centroid_m": {
                "x": 20000.0 + 900.0 * index + offset,
                "y": 18000.0 + 300.0 * index,
            },
            "maximum_updraft_ms": 20.0 + 2.0 * index + offset * 0.001,
            "maximum_absolute_uh_m2s2": 100.0 + 20.0 * index + offset * 0.01,
        }
        sample.update(derived_sample(index, offset))
        samples.append(sample)

    contract = {
        "duration_s": 90.0,
        "sample_start_s": 30.0,
        "sample_interval_s": 30.0,
        "coordinate_frame": "earth_absolute",
        "detector_algorithm": "updraft_core_watershed_condensate_rotation_branches_v5",
        "continuity_algorithm": "fixed_window_temporal_branch_association_v5",
        "physics_signature_sha256": "a" * 64,
        "tracking_source": (
            "synchronized_nested_child"
            if role == "moving_child" else "oversized_stationary_reference"
        ),
        "comparison_region": "moving_child_trusted_interior",
        "relaxation_exclusion_cells": 8,
        "horizontal_grid_spacing_m": {"x": 600.0, "y": 600.0},
        "grid_phase_origin_m": {"x": 0.0, "y": 0.0},
    }
    if role == "moving_child":
        contract["movement_algorithm"] = "qualified_envelope_hysteresis_v5"
    document = {
        "schema": comparator.INPUT_SCHEMA,
        "schema_version": 1,
        "role": role,
        "artifacts": artifacts,
        "contract": contract,
        "samples": samples,
    }
    path = root / f"{role}.json"
    write_json(path, document)
    return path, document


def make_policy(root: Path, status: str = "measurement_only") -> tuple[Path, dict]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policy["policy_status"] = status
    policy["expected_contract"]["duration_s"] = 90.0
    path = root / "policy.json"
    write_json(path, policy)
    return path, policy


class RetainedReferenceLocationTests(unittest.TestCase):
    def test_reviewed_policy_payloads_retain_their_original_hashes(self) -> None:
        for version in range(2, 14):
            with self.subTest(version=version):
                policy = ROOT / "tests/qualification/fixtures" / (
                    f"moving_nested_supercell_reference_frontier_v{version}.json")
                expected = (comparator.BLINDED_V2_POLICY_SHA256 if version == 2
                            else getattr(comparator, f"V{version}_POLICY_SHA256"))
                self.assertEqual(digest(policy), expected)

    def test_relocated_reference_still_requires_its_exact_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            retained = root / "tests/fixtures/reference/domain_history/witness.json"
            retained.parent.mkdir(parents=True)
            retained.write_bytes(b'{"status":"incomplete"}\n')
            record = {"path": "../../../docs/evidence/witness.json",
                      "sha256": digest(retained)}
            owner = root / "tests/qualification/fixtures/policy.json"
            with mock.patch.object(comparator, "ROOT", root):
                result = comparator.resolve_bound_record(record, owner, "witness")
                self.assertEqual(result, {"path": str(retained),
                                          "sha256": record["sha256"]})
                retained.write_bytes(b'{"status":"pass"}\n')
                with self.assertRaisesRegex(comparator.ContractError, "hash mismatch"):
                    comparator.resolve_bound_record(record, owner, "witness")

    def test_existing_mismatched_original_is_not_hidden_by_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            retained = root / "tests/fixtures/reference/domain_history/witness.json"
            original = root / "docs/evidence/witness.json"
            for path in (retained, original):
                path.parent.mkdir(parents=True)
            retained.write_bytes(b"retained original payload")
            original.write_bytes(b"changed original payload")
            with mock.patch.object(comparator, "ROOT", root):
                with self.assertRaisesRegex(comparator.ContractError, "hash mismatch"):
                    comparator.resolve_bound_record(
                        {"path": str(original), "sha256": digest(retained)},
                        root / "policy.json", "witness")

    def test_repository_alias_rejects_traversal_and_absolute_paths(self) -> None:
        for path in ("../docs/evidence/witness.json",
                     "docs/evidence/../../witness.json",
                     "docs/evidence/subdir/../witness.json",
                     str(ROOT / "docs/evidence/witness.json")):
            with self.subTest(path=path):
                with self.assertRaisesRegex(comparator.ContractError, "absolute or traverse"):
                    comparator.repository_evidence_path(path)

    def test_repository_alias_rejects_a_fixture_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            retained = root / "tests/fixtures/reference/domain_history/witness.json"
            retained.parent.mkdir(parents=True)
            outside = root.parent / f"{root.name}-external.json"
            retained.symlink_to(outside)
            with mock.patch.object(comparator, "ROOT", root):
                with self.assertRaisesRegex(comparator.ContractError, "leaves its fixture"):
                    comparator.repository_evidence_path("docs/evidence/witness.json")

    def test_bound_record_cannot_traverse_inside_a_retired_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with mock.patch.object(comparator, "ROOT", root):
                with self.assertRaisesRegex(comparator.ContractError, "must not traverse"):
                    comparator.resolve_bound_record(
                        {"path": "../../../docs/evidence/subdir/../witness.json",
                         "sha256": "0" * 64},
                        root / "tests/qualification/fixtures/policy.json", "witness")

    def test_bound_record_rejects_an_original_namespace_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            original = root / "docs/evidence/witness.json"
            original.parent.mkdir(parents=True)
            original.symlink_to(root.parent / f"{root.name}-external.json")
            with mock.patch.object(comparator, "ROOT", root):
                with self.assertRaisesRegex(comparator.ContractError, "leaves the repository"):
                    comparator.resolve_bound_record(
                        {"path": "../../../docs/evidence/witness.json", "sha256": "0" * 64},
                        root / "tests/qualification/fixtures/policy.json", "witness")

    def test_unrelated_missing_path_does_not_search_by_basename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            retained = root / "tests/fixtures/reference/domain_history/witness.json"
            retained.parent.mkdir(parents=True)
            retained.write_bytes(b"retained payload")
            with mock.patch.object(comparator, "ROOT", root):
                with self.assertRaisesRegex(comparator.ContractError, "is missing"):
                    comparator.resolve_bound_record(
                        {"path": "unrelated/witness.json", "sha256": digest(retained)},
                        root / "policy.json", "witness")


class MovingNestedReferenceFrontierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="nimbus_reference_frontier_")
        self.root = Path(self.temporary.name)
        self.moving_path, self.moving = make_series(self.root, "moving_child", 100.0)
        self.reference_path, self.reference = make_series(
            self.root, "oversized_reference", 0.0
        )
        self.policy_path, self.policy = make_policy(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def compare(self) -> dict:
        return comparator.compare_paths(
            self.moving_path, self.reference_path, self.policy_path
        )

    def rewrite_moving(self) -> None:
        write_json(self.moving_path, self.moving)

    def copied_v3_policy(self) -> tuple[Path, dict]:
        source = (
            ROOT / "tests" / "qualification" / "fixtures" /
            "moving_nested_supercell_reference_frontier_v3.json"
        )
        policy = json.loads(source.read_text(encoding="utf-8"))
        for key in ("geometry_plan", "qualification_aperture"):
            record = policy["expected_contract"][key]
            record["path"] = str((source.parent / record["path"]).resolve())
        for key in ("threshold_policy", "failure_evidence"):
            record = policy[key]
            record["path"] = str((source.parent / record["path"]).resolve())
        for record in policy["ratification"]["reviewed_sources"]:
            record["path"] = str((source.parent / record["path"]).resolve())
        path = self.root / "v3_policy.json"
        write_json(path, policy)
        return path, policy

    def copied_v4_policy(self) -> tuple[Path, dict]:
        source = (
            ROOT / "tests" / "qualification" / "fixtures" /
            "moving_nested_supercell_reference_frontier_v4.json"
        )
        policy = json.loads(source.read_text(encoding="utf-8"))
        for key in ("geometry_plan", "qualification_aperture"):
            record = policy["expected_contract"][key]
            record["path"] = str((source.parent / record["path"]).resolve())
        for key in ("threshold_policy", "prior_policy", "failure_evidence"):
            record = policy[key]
            record["path"] = str((source.parent / record["path"]).resolve())
        for record in policy["ratification"]["reviewed_sources"]:
            record["path"] = str((source.parent / record["path"]).resolve())
        path = self.root / "v4_policy.json"
        write_json(path, policy)
        return path, policy

    def copied_v5_policy(self) -> tuple[Path, dict]:
        source = (
            ROOT / "tests" / "qualification" / "fixtures" /
            "moving_nested_supercell_reference_frontier_v5.json"
        )
        policy = json.loads(source.read_text(encoding="utf-8"))
        for key in ("geometry_plan", "qualification_aperture"):
            record = policy["expected_contract"][key]
            record["path"] = str((source.parent / record["path"]).resolve())
        for key in ("threshold_policy", "prior_policy", "failure_evidence"):
            record = policy[key]
            record["path"] = str((source.parent / record["path"]).resolve())
        for record in policy["ratification"]["reviewed_sources"]:
            record["path"] = str((source.parent / record["path"]).resolve())
        path = self.root / "v5_policy.json"
        write_json(path, policy)
        return path, policy

    def copied_v6_policy(self) -> tuple[Path, dict]:
        source = (
            ROOT / "tests" / "qualification" / "fixtures" /
            "moving_nested_supercell_reference_frontier_v6.json"
        )
        policy = json.loads(source.read_text(encoding="utf-8"))
        for key in ("prior_policy", "failure_evidence"):
            record = policy[key]
            record["path"] = str((source.parent / record["path"]).resolve())
        path = self.root / "v6_policy.json"
        write_json(path, policy)
        return path, policy

    def copied_v7_policy(self) -> tuple[Path, dict]:
        source = (
            ROOT / "tests" / "qualification" / "fixtures" /
            "moving_nested_supercell_reference_frontier_v7.json"
        )
        policy = json.loads(source.read_text(encoding="utf-8"))
        for key in ("prior_policy", "failure_evidence"):
            record = policy[key]
            record["path"] = str((source.parent / record["path"]).resolve())
        path = self.root / "v7_policy.json"
        write_json(path, policy)
        return path, policy

    def copied_v8_policy(self) -> tuple[Path, dict]:
        source = (
            ROOT / "tests" / "qualification" / "fixtures" /
            "moving_nested_supercell_reference_frontier_v8.json"
        )
        policy = json.loads(source.read_text(encoding="utf-8"))
        for key in ("prior_policy", "failure_evidence"):
            record = policy[key]
            record["path"] = str((source.parent / record["path"]).resolve())
        path = self.root / "v8_policy.json"
        write_json(path, policy)
        return path, policy

    def copied_v9_policy(self) -> tuple[Path, dict]:
        source = (
            ROOT / "tests" / "qualification" / "fixtures" /
            "moving_nested_supercell_reference_frontier_v9.json"
        )
        policy = json.loads(source.read_text(encoding="utf-8"))
        for key in ("prior_policy", "failure_evidence"):
            record = policy[key]
            record["path"] = str((source.parent / record["path"]).resolve())
        path = self.root / "v9_policy.json"
        write_json(path, policy)
        return path, policy

    def copied_v10_policy(self) -> tuple[Path, dict]:
        source = (
            ROOT / "tests" / "qualification" / "fixtures" /
            "moving_nested_supercell_reference_frontier_v10.json"
        )
        policy = json.loads(source.read_text(encoding="utf-8"))
        for key in ("prior_policy", "failure_evidence"):
            record = policy[key]
            record["path"] = str((source.parent / record["path"]).resolve())
        path = self.root / "v10_policy.json"
        write_json(path, policy)
        return path, policy

    def copied_v11_policy(self) -> tuple[Path, dict]:
        source = (
            ROOT / "tests" / "qualification" / "fixtures" /
            "moving_nested_supercell_reference_frontier_v11.json"
        )
        policy = json.loads(source.read_text(encoding="utf-8"))
        for key in ("prior_policy", "failure_evidence"):
            record = policy[key]
            record["path"] = str((source.parent / record["path"]).resolve())
        path = self.root / "v11_policy.json"
        write_json(path, policy)
        return path, policy

    def copied_v12_policy(self) -> tuple[Path, dict]:
        source = (
            ROOT / "tests" / "qualification" / "fixtures" /
            "moving_nested_supercell_reference_frontier_v12.json"
        )
        policy = json.loads(source.read_text(encoding="utf-8"))
        for key in ("prior_policy", "failure_evidence"):
            record = policy[key]
            record["path"] = str((source.parent / record["path"]).resolve())
        path = self.root / "v12_policy.json"
        write_json(path, policy)
        return path, policy

    def copied_v13_policy(self) -> tuple[Path, dict]:
        source = (
            ROOT / "tests" / "qualification" / "fixtures" /
            "moving_nested_supercell_reference_frontier_v13.json"
        )
        policy = json.loads(source.read_text(encoding="utf-8"))
        for key in ("prior_policy", "backend_evidence"):
            record = policy[key]
            record["path"] = str((source.parent / record["path"]).resolve())
        path = self.root / "v13_policy.json"
        write_json(path, policy)
        return path, policy

    def test_v2_policy_is_immutable_and_v3_reuses_exact_thresholds(self) -> None:
        v2_path = (
            ROOT / "tests" / "qualification" / "fixtures" /
            "moving_nested_supercell_reference_frontier_v2.json"
        )
        v3_path = (
            ROOT / "tests" / "qualification" / "fixtures" /
            "moving_nested_supercell_reference_frontier_v3.json"
        )
        self.assertEqual(digest(v2_path), comparator.BLINDED_V2_POLICY_SHA256)
        v2 = json.loads(v2_path.read_text(encoding="utf-8"))
        v3 = json.loads(v3_path.read_text(encoding="utf-8"))
        for key in ("required_metric_families", "coverage_thresholds",
                    "metric_thresholds", "derived_definitions"):
            self.assertEqual(v3[key], v2[key])
        comparator.validate_policy(v3, v3_path)

    def test_v3_cannot_silently_relax_a_blinded_threshold(self) -> None:
        path, policy = self.copied_v3_policy()
        policy["metric_thresholds"]["path"]["rmse_m"] += 1.0
        write_json(path, policy)
        with self.assertRaisesRegex(
                comparator.ContractError,
                "differs from bound blinded v2 policy"):
            comparator.validate_policy(policy, path)

    def test_v3_requires_the_exact_blinded_v2_digest(self) -> None:
        path, policy = self.copied_v3_policy()
        policy["threshold_policy"]["sha256"] = "0" * 64
        write_json(path, policy)
        with self.assertRaisesRegex(
                comparator.ContractError,
                "threshold-policy binding changed"):
            comparator.validate_policy(policy, path)

    def test_v3_failure_evidence_hash_mismatch_is_incompatible(self) -> None:
        path, policy = self.copied_v3_policy()
        policy["failure_evidence"]["sha256"] = "0" * 64
        write_json(path, policy)
        with self.assertRaisesRegex(
                comparator.ContractError,
                "failure-evidence binding changed"):
            comparator.validate_policy(policy, path)

    def test_v4_preserves_v2_thresholds_and_v3_geometry(self) -> None:
        v2_path = (
            ROOT / "tests" / "qualification" / "fixtures" /
            "moving_nested_supercell_reference_frontier_v2.json"
        )
        v3_path = (
            ROOT / "tests" / "qualification" / "fixtures" /
            "moving_nested_supercell_reference_frontier_v3.json"
        )
        v4_path = (
            ROOT / "tests" / "qualification" / "fixtures" /
            "moving_nested_supercell_reference_frontier_v4.json"
        )
        self.assertEqual(digest(v2_path), comparator.BLINDED_V2_POLICY_SHA256)
        self.assertEqual(digest(v3_path), comparator.V3_POLICY_SHA256)
        v2 = json.loads(v2_path.read_text(encoding="utf-8"))
        v3 = json.loads(v3_path.read_text(encoding="utf-8"))
        v4 = json.loads(v4_path.read_text(encoding="utf-8"))
        for key in ("required_metric_families", "coverage_thresholds",
                    "metric_thresholds", "derived_definitions"):
            self.assertEqual(v4[key], v2[key])
        for key in ("geometry_plan", "qualification_aperture"):
            self.assertEqual(v4["expected_contract"][key],
                             v3["expected_contract"][key])
        self.assertEqual(
            v4["expected_contract"]["moving_movement_algorithm"],
            "qualified_envelope_hysteresis_v6")
        comparator.validate_policy(v4, v4_path)

    def test_v4_cannot_silently_relax_a_blinded_threshold(self) -> None:
        path, policy = self.copied_v4_policy()
        policy["metric_thresholds"]["path"]["maximum_m"] += 1.0
        write_json(path, policy)
        with self.assertRaisesRegex(
                comparator.ContractError,
                "differs from bound blinded v2 policy"):
            comparator.validate_policy(policy, path)

    def test_v4_failure_evidence_hash_mismatch_is_incompatible(self) -> None:
        path, policy = self.copied_v4_policy()
        policy["failure_evidence"]["sha256"] = "0" * 64
        write_json(path, policy)
        with self.assertRaisesRegex(
                comparator.ContractError,
                "v4 failure-evidence binding changed"):
            comparator.validate_policy(policy, path)

    def test_v4_measurement_only_cannot_silently_become_pass(self) -> None:
        path, policy = self.copied_v4_policy()
        policy["policy_status"] = "measurement_only"
        write_json(path, policy)
        with self.assertRaisesRegex(
                comparator.ContractError,
                "must be explicitly ratified"):
            comparator.validate_policy(policy, path)

    def test_v5_preserves_v2_thresholds_and_v4_geometry(self) -> None:
        fixtures = ROOT / "tests" / "qualification" / "fixtures"
        v2_path = fixtures / "moving_nested_supercell_reference_frontier_v2.json"
        v4_path = fixtures / "moving_nested_supercell_reference_frontier_v4.json"
        v5_path = fixtures / "moving_nested_supercell_reference_frontier_v5.json"
        self.assertEqual(digest(v4_path), comparator.V4_POLICY_SHA256)
        v2 = json.loads(v2_path.read_text(encoding="utf-8"))
        v4 = json.loads(v4_path.read_text(encoding="utf-8"))
        v5 = json.loads(v5_path.read_text(encoding="utf-8"))
        for key in ("required_metric_families", "coverage_thresholds",
                    "metric_thresholds", "derived_definitions"):
            self.assertEqual(v5[key], v2[key])
        for key in ("geometry_plan", "qualification_aperture"):
            self.assertEqual(v5["expected_contract"][key],
                             v4["expected_contract"][key])
        self.assertEqual(
            v5["expected_contract"]["moving_movement_algorithm"],
            "qualified_envelope_hysteresis_v7")
        comparator.validate_policy(v5, v5_path)

    def test_v5_failure_evidence_hash_mismatch_is_incompatible(self) -> None:
        path, policy = self.copied_v5_policy()
        policy["failure_evidence"]["sha256"] = "0" * 64
        write_json(path, policy)
        with self.assertRaisesRegex(
                comparator.ContractError,
                "v5 failure-evidence binding changed"):
            comparator.validate_policy(policy, path)

    def test_v6_inherits_v5_and_changes_only_trusted_containment(self) -> None:
        fixtures = ROOT / "tests" / "qualification" / "fixtures"
        v5_path = fixtures / "moving_nested_supercell_reference_frontier_v5.json"
        self.assertEqual(digest(v5_path), comparator.V5_POLICY_SHA256)
        path, policy = self.copied_v6_policy()
        effective = comparator.validate_policy(policy, path)
        v5 = comparator.validate_policy(
            json.loads(v5_path.read_text(encoding="utf-8")), v5_path)
        for key in ("required_metric_families", "coverage_thresholds",
                    "metric_thresholds", "derived_definitions"):
            self.assertEqual(effective[key], v5[key])
        inherited = dict(v5["expected_contract"])
        inherited["moving_movement_algorithm"] = (
            "qualified_envelope_hysteresis_v8")
        self.assertEqual(effective["expected_contract"], inherited)
        self.assertEqual(effective["schema_version"], 6)

    def test_v6_failure_evidence_hash_mismatch_is_incompatible(self) -> None:
        path, policy = self.copied_v6_policy()
        policy["failure_evidence"]["sha256"] = "0" * 64
        write_json(path, policy)
        with self.assertRaisesRegex(
                comparator.ContractError,
                "v6 failure-evidence binding changed"):
            comparator.validate_policy(policy, path)

    def test_v7_inherits_v6_and_adds_balanced_containment_state(self) -> None:
        fixtures = ROOT / "tests" / "qualification" / "fixtures"
        v6_path = fixtures / "moving_nested_supercell_reference_frontier_v6.json"
        self.assertEqual(digest(v6_path), comparator.V6_POLICY_SHA256)
        path, policy = self.copied_v7_policy()
        effective = comparator.validate_policy(policy, path)
        v6 = comparator.validate_policy(
            json.loads(v6_path.read_text(encoding="utf-8")), v6_path)
        for key in ("required_metric_families", "coverage_thresholds",
                    "metric_thresholds", "derived_definitions"):
            self.assertEqual(effective[key], v6[key])
        inherited = dict(v6["expected_contract"])
        inherited["moving_movement_algorithm"] = (
            "qualified_envelope_hysteresis_v9")
        self.assertEqual(effective["expected_contract"], inherited)
        self.assertEqual(effective["schema_version"], 7)

    def test_v7_failure_evidence_hash_mismatch_is_incompatible(self) -> None:
        path, policy = self.copied_v7_policy()
        policy["failure_evidence"]["sha256"] = "0" * 64
        write_json(path, policy)
        with self.assertRaisesRegex(
                comparator.ContractError,
                "v7 failure-evidence binding changed"):
            comparator.validate_policy(policy, path)

    def test_v8_inherits_v7_and_adds_orthogonal_containment_reserve(self) -> None:
        fixtures = ROOT / "tests" / "qualification" / "fixtures"
        v7_path = fixtures / "moving_nested_supercell_reference_frontier_v7.json"
        self.assertEqual(digest(v7_path), comparator.V7_POLICY_SHA256)
        path, policy = self.copied_v8_policy()
        effective = comparator.validate_policy(policy, path)
        v7 = comparator.validate_policy(
            json.loads(v7_path.read_text(encoding="utf-8")), v7_path)
        for key in ("required_metric_families", "coverage_thresholds",
                    "metric_thresholds", "derived_definitions"):
            self.assertEqual(effective[key], v7[key])
        inherited = dict(v7["expected_contract"])
        inherited["moving_movement_algorithm"] = (
            "qualified_envelope_hysteresis_v10")
        self.assertEqual(effective["expected_contract"], inherited)
        self.assertEqual(effective["schema_version"], 8)

    def test_v8_failure_evidence_hash_mismatch_is_incompatible(self) -> None:
        path, policy = self.copied_v8_policy()
        policy["failure_evidence"]["sha256"] = "0" * 64
        write_json(path, policy)
        with self.assertRaisesRegex(
                comparator.ContractError,
                "v8 failure-evidence binding changed"):
            comparator.validate_policy(policy, path)

    def test_v9_inherits_v8_and_adds_receding_trusted_face_state(self) -> None:
        fixtures = ROOT / "tests" / "qualification" / "fixtures"
        v8_path = fixtures / "moving_nested_supercell_reference_frontier_v8.json"
        self.assertEqual(digest(v8_path), comparator.V8_POLICY_SHA256)
        path, policy = self.copied_v9_policy()
        effective = comparator.validate_policy(policy, path)
        v8 = comparator.validate_policy(
            json.loads(v8_path.read_text(encoding="utf-8")), v8_path)
        for key in ("required_metric_families", "coverage_thresholds",
                    "metric_thresholds", "derived_definitions"):
            self.assertEqual(effective[key], v8[key])
        inherited = dict(v8["expected_contract"])
        inherited["moving_movement_algorithm"] = (
            "qualified_envelope_hysteresis_v11")
        self.assertEqual(effective["expected_contract"], inherited)
        self.assertEqual(effective["schema_version"], 9)

    def test_v9_failure_evidence_hash_mismatch_is_incompatible(self) -> None:
        path, policy = self.copied_v9_policy()
        policy["failure_evidence"]["sha256"] = "0" * 64
        write_json(path, policy)
        with self.assertRaisesRegex(
                comparator.ContractError,
                "v9 failure-evidence binding changed"):
            comparator.validate_policy(policy, path)

    def test_v10_inherits_v9_and_adds_quantized_trusted_balance(self) -> None:
        fixtures = ROOT / "tests" / "qualification" / "fixtures"
        v9_path = fixtures / "moving_nested_supercell_reference_frontier_v9.json"
        self.assertEqual(digest(v9_path), comparator.V9_POLICY_SHA256)
        path, policy = self.copied_v10_policy()
        effective = comparator.validate_policy(policy, path)
        v9 = comparator.validate_policy(
            json.loads(v9_path.read_text(encoding="utf-8")), v9_path)
        for key in ("required_metric_families", "coverage_thresholds",
                    "metric_thresholds", "derived_definitions"):
            self.assertEqual(effective[key], v9[key])
        inherited = dict(v9["expected_contract"])
        inherited["moving_movement_algorithm"] = (
            "qualified_envelope_hysteresis_v12")
        self.assertEqual(effective["expected_contract"], inherited)
        self.assertEqual(effective["schema_version"], 10)

    def test_v10_failure_evidence_hash_mismatch_is_incompatible(self) -> None:
        path, policy = self.copied_v10_policy()
        policy["failure_evidence"]["sha256"] = "0" * 64
        write_json(path, policy)
        with self.assertRaisesRegex(
                comparator.ContractError,
                "v10 failure-evidence binding changed"):
            comparator.validate_policy(policy, path)

    def test_v11_inherits_v10_and_adds_ordinary_trusted_improvement(self) -> None:
        fixtures = ROOT / "tests" / "qualification" / "fixtures"
        v10_path = fixtures / "moving_nested_supercell_reference_frontier_v10.json"
        self.assertEqual(digest(v10_path), comparator.V10_POLICY_SHA256)
        path, policy = self.copied_v11_policy()
        effective = comparator.validate_policy(policy, path)
        v10 = comparator.validate_policy(
            json.loads(v10_path.read_text(encoding="utf-8")), v10_path)
        for key in ("required_metric_families", "coverage_thresholds",
                    "metric_thresholds", "derived_definitions"):
            self.assertEqual(effective[key], v10[key])
        inherited = dict(v10["expected_contract"])
        inherited["moving_movement_algorithm"] = (
            "qualified_envelope_hysteresis_v13")
        self.assertEqual(effective["expected_contract"], inherited)
        self.assertEqual(effective["schema_version"], 11)

    def test_v11_failure_evidence_hash_mismatch_is_incompatible(self) -> None:
        path, policy = self.copied_v11_policy()
        policy["failure_evidence"]["sha256"] = "0" * 64
        write_json(path, policy)
        with self.assertRaisesRegex(
                comparator.ContractError,
                "v11 failure-evidence binding changed"):
            comparator.validate_policy(policy, path)

    def test_v12_inherits_v11_and_adds_parent_stencil_limit(self) -> None:
        fixtures = ROOT / "tests" / "qualification" / "fixtures"
        v11_path = fixtures / "moving_nested_supercell_reference_frontier_v11.json"
        self.assertEqual(digest(v11_path), comparator.V11_POLICY_SHA256)
        path, policy = self.copied_v12_policy()
        effective = comparator.validate_policy(policy, path)
        v11 = comparator.validate_policy(
            json.loads(v11_path.read_text(encoding="utf-8")), v11_path)
        for key in ("required_metric_families", "coverage_thresholds",
                    "metric_thresholds", "derived_definitions"):
            self.assertEqual(effective[key], v11[key])
        inherited = dict(v11["expected_contract"])
        inherited["moving_movement_algorithm"] = (
            "qualified_envelope_hysteresis_v14")
        self.assertEqual(effective["expected_contract"], inherited)
        self.assertEqual(effective["schema_version"], 12)

    def test_v12_failure_evidence_hash_mismatch_is_incompatible(self) -> None:
        path, policy = self.copied_v12_policy()
        policy["failure_evidence"]["sha256"] = "0" * 64
        write_json(path, policy)
        with self.assertRaisesRegex(
                comparator.ContractError,
                "v12 failure-evidence binding changed"):
            comparator.validate_policy(policy, path)

    def test_v13_inherits_v12_and_changes_only_backend_signature(self) -> None:
        fixtures = ROOT / "tests" / "qualification" / "fixtures"
        v12_path = fixtures / "moving_nested_supercell_reference_frontier_v12.json"
        self.assertEqual(digest(v12_path), comparator.V12_POLICY_SHA256)
        path, policy = self.copied_v13_policy()
        effective = comparator.validate_policy(policy, path)
        v12 = comparator.validate_policy(
            json.loads(v12_path.read_text(encoding="utf-8")), v12_path)
        for key in ("required_metric_families", "coverage_thresholds",
                    "metric_thresholds", "derived_definitions"):
            self.assertEqual(effective[key], v12[key])
        inherited = dict(v12["expected_contract"])
        inherited["physics_signature_sha256"] = (
            comparator.VULKAN_PHYSICS_SIGNATURE_SHA256)
        self.assertEqual(effective["expected_contract"], inherited)
        self.assertEqual(effective["schema_version"], 13)

    def test_v13_backend_evidence_hash_mismatch_is_incompatible(self) -> None:
        path, policy = self.copied_v13_policy()
        policy["backend_evidence"]["sha256"] = "0" * 64
        write_json(path, policy)
        with self.assertRaisesRegex(
                comparator.ContractError,
                "v13 backend-evidence binding changed"):
            comparator.validate_policy(policy, path)

    def test_complete_legacy_series_remains_measurement_only(self) -> None:
        report = self.compare()
        self.assertEqual(report["status"], "measurement_only")
        self.assertFalse(report["qualification_eligible"])
        self.assertEqual(
            set(report["metrics"]),
            {"path", "intensity", "cold_pool", "precipitation"},
        )
        self.assertTrue(all(check["passed"] for check in report["checks"]))

    def test_v1_measurement_policy_cannot_be_silently_flipped_to_ratified(self) -> None:
        self.policy["policy_status"] = "ratified"
        write_json(self.policy_path, self.policy)
        report = self.compare()
        self.assertEqual(report["status"], "incompatible")
        self.assertFalse(report["qualification_eligible"])
        self.assertTrue(any("permanently measurement-only" in item
                            for item in report["contract_errors"]))

    def test_ratified_v2_policy_rejects_legacy_unmanifested_series(self) -> None:
        policy_source = (
            ROOT / "tests" / "qualification" / "fixtures" /
            "moving_nested_supercell_reference_frontier_v2.json"
        )
        policy = json.loads(policy_source.read_text(encoding="utf-8"))
        policy["expected_contract"].update({
            "duration_s": 90.0,
            "physics_signature_sha256": "a" * 64,
        })
        policy["metric_thresholds"]["cold_pool"]["minimum_active_samples"] = 1
        for record in policy["ratification"]["reviewed_sources"]:
            record["path"] = str((policy_source.parent / record["path"]).resolve())
        write_json(self.policy_path, policy)
        report = self.compare()
        self.assertEqual(report["status"], "incompatible")
        self.assertFalse(report["qualification_eligible"])
        self.assertTrue(any("requires hash-bound v2 input series" in item
                            for item in report["contract_errors"]))

    def test_missing_cold_pool_is_explicitly_incomplete(self) -> None:
        self.moving["samples"][1].pop("cold_pool")
        self.rewrite_moving()
        report = self.compare()
        self.assertEqual(report["status"], "incomplete")
        self.assertFalse(report["qualification_eligible"])
        self.assertTrue(any("cold_pool inputs are incomplete" in item
                            for item in report["incomplete_reasons"]))
        self.assertNotIn("cold_pool", report["metrics"])

    def test_missing_precipitation_is_explicitly_incomplete(self) -> None:
        self.reference["samples"][0].pop("precipitation")
        write_json(self.reference_path, self.reference)
        report = self.compare()
        self.assertEqual(report["status"], "incomplete")
        self.assertTrue(any("precipitation inputs are incomplete" in item
                            for item in report["incomplete_reasons"]))

    def test_missing_primary_intensity_is_explicitly_incomplete(self) -> None:
        self.moving["samples"][1].pop("maximum_updraft_ms")
        self.rewrite_moving()
        report = self.compare()
        self.assertEqual(report["status"], "incomplete")
        self.assertTrue(any("intensity inputs are incomplete" in item
                            for item in report["incomplete_reasons"]))
        self.assertNotIn("intensity", report["metrics"])

    def test_missing_declared_time_is_incomplete(self) -> None:
        self.moving["samples"].pop(1)
        self.rewrite_moving()
        report = self.compare()
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["coverage"]["missing_moving_times_s"], [60.0])

    def test_artifact_hash_mismatch_is_incompatible(self) -> None:
        config = self.root / "moving_child_config.dat"
        config.write_text("mutated\n", encoding="utf-8")
        report = self.compare()
        self.assertEqual(report["status"], "incompatible")
        self.assertTrue(any("artifact hash mismatch" in item
                            for item in report["contract_errors"]))

    def test_coordinate_frame_mismatch_is_incompatible(self) -> None:
        self.moving["contract"]["coordinate_frame"] = "grid_relative"
        self.rewrite_moving()
        report = self.compare()
        self.assertEqual(report["status"], "incompatible")
        self.assertTrue(any("coordinate_frame" in item
                            for item in report["contract_errors"]))

    def test_threshold_violation_fails_without_becoming_incomplete(self) -> None:
        for sample in self.moving["samples"]:
            sample["core_centroid_m"]["x"] += 50000.0
        self.rewrite_moving()
        report = self.compare()
        self.assertEqual(report["status"], "fail")
        self.assertFalse(report["qualification_eligible"])
        self.assertFalse(next(check for check in report["checks"]
                              if check["name"] == "path.rmse_m")["passed"])

    def test_unpaired_reference_peak_cannot_be_hidden_from_intensity_gate(self) -> None:
        self.moving["samples"][1]["selected"] = False
        self.reference["samples"][1]["maximum_updraft_ms"] = 1000.0
        self.rewrite_moving()
        write_json(self.reference_path, self.reference)
        report = self.compare()
        peak = next(check for check in report["checks"]
                    if check["name"] ==
                    "intensity.maximum_updraft_peak_relative_error")
        self.assertFalse(peak["passed"])


if __name__ == "__main__":
    unittest.main()
