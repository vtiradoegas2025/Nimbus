import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
QUALIFICATION = ROOT / "scripts" / "qualification"
ACCELERATOR_SCRIPTS = QUALIFICATION / "accelerators"
DOMAIN_MOTION_SCRIPTS = QUALIFICATION / "domain_motion"
NESTED_DOMAIN_SCRIPTS = QUALIFICATION / "nested_domains"
for directory in (
    ACCELERATOR_SCRIPTS,
    DOMAIN_MOTION_SCRIPTS,
    NESTED_DOMAIN_SCRIPTS,
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
SPEC = importlib.util.spec_from_file_location(
    "qualify_metal", ACCELERATOR_SCRIPTS / "qualify_metal.py")
assert SPEC is not None and SPEC.loader is not None
QUALIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALIFY)

DOMAIN_SPEC = importlib.util.spec_from_file_location(
    "qualify_domain_recycling",
    DOMAIN_MOTION_SCRIPTS / "qualify_domain_recycling.py",
)
assert DOMAIN_SPEC is not None and DOMAIN_SPEC.loader is not None
DOMAIN_QUALIFY = importlib.util.module_from_spec(DOMAIN_SPEC)
DOMAIN_SPEC.loader.exec_module(DOMAIN_QUALIFY)

MOVING_SPEC = importlib.util.spec_from_file_location(
    "qualify_moving_nested_accelerator_runtime",
    NESTED_DOMAIN_SCRIPTS / "qualify_moving_nested_vulkan_runtime.py",
)
assert MOVING_SPEC is not None and MOVING_SPEC.loader is not None
MOVING_QUALIFY = importlib.util.module_from_spec(MOVING_SPEC)
MOVING_SPEC.loader.exec_module(MOVING_QUALIFY)

CHAIN_SPEC = importlib.util.spec_from_file_location(
    "inspect_moving_nested_accelerator_lifetime_chain",
    NESTED_DOMAIN_SCRIPTS / "inspect_moving_nested_vulkan_lifetime_chain.py",
)
assert CHAIN_SPEC is not None and CHAIN_SPEC.loader is not None
CHAIN = importlib.util.module_from_spec(CHAIN_SPEC)
CHAIN_SPEC.loader.exec_module(CHAIN)

REFERENCE_SPEC = importlib.util.spec_from_file_location(
    "qualify_moving_nested_accelerator_reference",
    NESTED_DOMAIN_SCRIPTS / "qualify_moving_nested_vulkan_reference.py",
)
assert REFERENCE_SPEC is not None and REFERENCE_SPEC.loader is not None
REFERENCE = importlib.util.module_from_spec(REFERENCE_SPEC)
REFERENCE_SPEC.loader.exec_module(REFERENCE)

EVIDENCE_SPEC = importlib.util.spec_from_file_location(
    "seal_moving_nested_accelerator_reference_evidence",
    NESTED_DOMAIN_SCRIPTS / "seal_moving_nested_accelerator_reference_evidence.py",
)
assert EVIDENCE_SPEC is not None and EVIDENCE_SPEC.loader is not None
EVIDENCE = importlib.util.module_from_spec(EVIDENCE_SPEC)
EVIDENCE_SPEC.loader.exec_module(EVIDENCE)

RATIFIER_SPEC = importlib.util.spec_from_file_location(
    "ratify_moving_nested_metal_reference_policy",
    NESTED_DOMAIN_SCRIPTS / "ratify_moving_nested_metal_reference_policy.py",
)
assert RATIFIER_SPEC is not None and RATIFIER_SPEC.loader is not None
RATIFIER = importlib.util.module_from_spec(RATIFIER_SPEC)
RATIFIER_SPEC.loader.exec_module(RATIFIER)

FRONTIER_SPEC = importlib.util.spec_from_file_location(
    "qualify_moving_nested_accelerator_reference_frontier",
    NESTED_DOMAIN_SCRIPTS / "qualify_moving_nested_vulkan_reference_frontier.py",
)
assert FRONTIER_SPEC is not None and FRONTIER_SPEC.loader is not None
FRONTIER = importlib.util.module_from_spec(FRONTIER_SPEC)
FRONTIER_SPEC.loader.exec_module(FRONTIER)

FRONTIER_EVIDENCE_SPEC = importlib.util.spec_from_file_location(
    "seal_moving_nested_accelerator_frontier_evidence",
    NESTED_DOMAIN_SCRIPTS / "seal_moving_nested_accelerator_frontier_evidence.py",
)
assert (FRONTIER_EVIDENCE_SPEC is not None and
        FRONTIER_EVIDENCE_SPEC.loader is not None)
FRONTIER_EVIDENCE = importlib.util.module_from_spec(FRONTIER_EVIDENCE_SPEC)
FRONTIER_EVIDENCE_SPEC.loader.exec_module(FRONTIER_EVIDENCE)


def valid_log() -> str:
    records = {
        label: "value=1" for label in QUALIFY.REQUIRED_EVIDENCE
    }
    records["metal-manifest-evidence"] = (
        "manifest=50 active=47 ready=47 deferred=0 orphan=3 "
        "canonical_fnv1a64=ced28fc195dbd340"
    )
    records["metal-evidence"] = (
        'device="Apple M4 Max" os="Version 26.5.2 (Build 25F84)" '
        'toolchain="clang 21" msl_fnv1a64=fc588d2c9c36af84 '
        "unified_memory=true parity=bitwise parity_cases=12"
    )
    records["metal-storage-evidence"] = (
        "field_bytes=2949120 private_heap_bytes=5902336 iterations=12 samples=7 "
        "shared_bitwise=true private_bitwise=true shared_device_median_ms=1.0 "
        "private_device_median_ms=1.1 winner=unclaimed msl_fnv1a64=fc588d2c9c36af84"
    )
    records["metal-workgroup-evidence"] = (
        "widths=32,64,128 field_bytes=786432 iterations=8 samples=7 all_bitwise=true "
        "w32_shared_median_ms=1.0 w64_shared_median_ms=0.9 w128_shared_median_ms=0.8 "
        "winner=unclaimed msl_fnv1a64=fc588d2c9c36af84"
    )
    records["metal-submission-evidence"] = (
        "depths=1,4,16,64 field_bytes=491520 samples=7 width=64 all_bitwise=true "
        "d1_shared_median_per_dispatch_ms=0.1 d64_shared_median_per_dispatch_ms=0.01 "
        "winner=unclaimed msl_fnv1a64=fc588d2c9c36af84"
    )
    lines = [f"[{label}] {records[label]}" for label in sorted(records)]
    lines.append("All tests passed (75655 assertions in 43 test cases)")
    return "\n".join(lines) + "\n"


class MetalQualificationParserTests(unittest.TestCase):
    def test_backend_stays_partitioned_by_maintenance_owner(self) -> None:
        composition_root = ROOT / "metal" / "src" / "compute" / "metal_compute_backend.mm"
        fragment_root = ROOT / "metal" / "src" / "compute" / "backend"
        expected_fragments = {
            'advection.inc',
            'boundary_layer.inc',
            'cgrid_boundary.inc',
            'chaos.inc',
            'collocated_sgs.inc',
            'column_remap.inc',
            'conservative_momentum.inc',
            'conservative_reduction.inc',
            'conservative_remap.inc',
            'diffusion.inc',
            'domain_recycling.inc',
            'dynamics/acoustic_agrid_complete.inc',
            'dynamics/acoustic_agrid_continuity.inc',
            'dynamics/acoustic_basic.inc',
            'dynamics/acoustic_cgrid.inc',
            'dynamics/acoustic_cgrid_batched.inc',
            'dynamics/acoustic_cgrid_fused.inc',
            'dynamics/cartesian_tendencies.inc',
            'dynamics/support.inc',
            'dynamics_acoustics.inc',
            'dynamics_slow.inc',
            'earth_geometry.inc',
            'foundation.inc',
            'held_sources.inc',
            'lifecycle.inc',
            'microphysics_kessler.inc',
            'microphysics_lin.inc',
            'microphysics_milbrandt.inc',
            'microphysics_thompson.inc',
            'moving_energy.inc',
            'moving_momentum.inc',
            'radiation.inc',
            'reference_atmosphere.inc',
            'rk_combination.inc',
            'rk_tendency.inc',
            'services.inc',
            'services/contracts.inc',
            'services/execution_evidence.inc',
            'services/manifest.inc',
            'services/storage_benchmark.inc',
            'shader_sources.inc',
            'simple_grey.inc',
            'slab_column.inc',
            'terrain.inc',
            'turbulence.inc',
        }
        fragments = {
            str(path.relative_to(fragment_root))
            for path in fragment_root.rglob("*.inc")
        }
        self.assertEqual(fragments, expected_fragments)
        self.assertLessEqual(len(composition_root.read_text().splitlines()), 120)
        for name in fragments:
            with self.subTest(fragment=name):
                lines = (fragment_root / name).read_text().splitlines()
                self.assertLessEqual(
                    len(lines),
                    {"terrain.inc": 1029}.get(name, 1000),
                )
                self.assertLessEqual(max(map(len, lines), default=0), 160)
        internal_root = ROOT / "metal" / "src" / "compute" / "internal"
        expected_internal = {
            'metal_abi_contracts.hpp',
            'metal_device_context.hpp',
            'metal_pipeline_registry.hpp',
            'metal_resident_state.hpp',
            'metal_shared_allocations.hpp',
            'metal_source_catalog.hpp',
            'metal_telemetry.hpp',
            'validation/common.hpp',
            'validation/domain.hpp',
            'validation/lin.hpp',
        }
        internal_headers = {
            str(path.relative_to(internal_root))
            for path in internal_root.rglob("*.hpp")
        }
        self.assertEqual(internal_headers, expected_internal)
        for name in internal_headers:
            with self.subTest(internal=name):
                lines = (internal_root / name).read_text().splitlines()
                counted_lines = lines
                if name == "metal_source_catalog.hpp":
                    # Catalog spacing is not implementation; shader bodies belong to their owners.
                    counted_lines = [line for line in lines if line.strip()]
                    self.assertNotIn('R"msl(', "\n".join(lines))
                self.assertLessEqual(
                    len(counted_lines),
                    {"metal_source_catalog.hpp": 347, "metal_telemetry.hpp": 368}.get(name, 300),
                )
                self.assertLessEqual(max(map(len, lines), default=0), 160)
        interface = (
            ROOT / "metal" / "include" / "compute" / "metal" /
            "metal_compute_backend.hpp"
        )
        self.assertLessEqual(len(interface.read_text().splitlines()), 327)

    def test_qualification_contract_covers_every_embedded_msl_source(self) -> None:
        embedded_sources = {
            str(path.relative_to(ROOT))
            for path in (ROOT / "metal" / "include" / "compute").rglob("*_msl.hpp")
        }
        implementation_fragments = {
            str(path.relative_to(ROOT))
            for path in (ROOT / "metal" / "src" / "compute" / "backend").rglob("*.inc")
        }
        internal_headers = {
            str(path.relative_to(ROOT))
            for path in (ROOT / "metal" / "src" / "compute" / "internal").rglob("*.hpp")
        }
        self.assertEqual(len(QUALIFY.REQUIRED_EVIDENCE), 30)
        self.assertTrue(embedded_sources)
        self.assertTrue(implementation_fragments)
        self.assertTrue(internal_headers)
        self.assertTrue(embedded_sources <= set(QUALIFY.SEALED_SOURCES))
        self.assertTrue(implementation_fragments <= set(QUALIFY.SEALED_SOURCES))
        self.assertTrue(internal_headers <= set(QUALIFY.SEALED_SOURCES))

    def test_accepts_complete_physical_evidence(self) -> None:
        summary, evidence = QUALIFY.parse_output(valid_log())
        self.assertEqual(summary, {"assertions": 75655, "test_cases": 43})
        self.assertEqual(evidence["metal-manifest-evidence"]["ready"], 47)
        self.assertTrue(evidence["metal-evidence"]["unified_memory"])

    def test_rejects_missing_family_evidence(self) -> None:
        log = valid_log().replace("[metal-lin-evidence] value=1\n", "")
        with self.assertRaisesRegex(QUALIFY.QualificationError, "missing evidence records"):
            QUALIFY.parse_output(log)

    def test_rejects_manifest_readiness_regression(self) -> None:
        log = valid_log().replace("active=47 ready=47", "active=47 ready=46")
        with self.assertRaisesRegex(QUALIFY.QualificationError, "manifest evidence ready"):
            QUALIFY.parse_output(log)

    def test_rejects_incomplete_workgroup_matrix(self) -> None:
        log = valid_log().replace("widths=32,64,128", "widths=64")
        with self.assertRaisesRegex(QUALIFY.QualificationError, "workgroup-width parity"):
            QUALIFY.parse_output(log)

    def test_rejects_incomplete_submission_matrix(self) -> None:
        log = valid_log().replace("depths=1,4,16,64", "depths=1,64")
        with self.assertRaisesRegex(QUALIFY.QualificationError, "submission-depth parity"):
            QUALIFY.parse_output(log)

    def test_report_seals_binary_and_source_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "metal-test"
            binary.write_bytes(b"qualified-binary")
            report = QUALIFY.build_report(binary, valid_log(), ROOT)
        self.assertEqual(report["result"], "pass")
        self.assertFalse(report["qualification_eligible"])
        self.assertEqual(report["manifest"]["ready"], 47)
        self.assertEqual(len(report["binary"]["sha256"]), 64)
        self.assertEqual(set(report["sealed_sources_sha256"]), set(QUALIFY.SEALED_SOURCES))

    def test_report_uses_repo_relative_binary_identity(self) -> None:
        binary = ACCELERATOR_SCRIPTS / "qualify_metal.py"
        report = QUALIFY.build_report(binary, valid_log(), ROOT)
        self.assertEqual(
            report["binary"]["path"],
            "scripts/qualification/accelerators/qualify_metal.py",
        )

    def test_atomic_report_is_published_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "metal-report.json"
            QUALIFY.write_atomic(output, "{}\n")
            self.assertEqual(output.read_text(encoding="utf-8"), "{}\n")
            self.assertEqual(output.stat().st_mode & 0o777, 0o644)

    def test_live_campaign_requires_physical_metal_authority(self) -> None:
        output = (
            "[HARDWARE] GPU: Apple M4 Max\n"
            "[COMPUTE] backend=metal, requested=metal, fallback=off\n"
        )
        DOMAIN_QUALIFY.require_selected_backend(
            output, "metal", "metal", "Metal campaign")
        with self.assertRaisesRegex(RuntimeError, "physical Metal device"):
            DOMAIN_QUALIFY.require_selected_backend(
                "[HARDWARE] GPU: not available\n[COMPUTE] backend=metal\n",
                "metal",
                "metal",
                "Metal campaign",
            )
        with self.assertRaisesRegex(RuntimeError, "used CPU fallback"):
            DOMAIN_QUALIFY.require_selected_backend(
                output + "Falling back to CPU\n",
                "metal",
                "metal",
                "Metal campaign",
            )

    def test_live_campaign_preserves_required_vulkan_check(self) -> None:
        output = (
            "[COMPUTE][VULKAN] selected device[0]: Apple M4 Max\n"
            "[COMPUTE] backend=vulkan, requested=vulkan, fallback=off\n"
        )
        DOMAIN_QUALIFY.require_selected_backend(
            output, "vulkan", "vulkan", "Vulkan campaign")

    def test_moving_nest_accepts_only_physical_metal_identity(self) -> None:
        MOVING_QUALIFY.require_device_log(
            "[COMPUTE] backend=metal, requested=metal, device_index=-1, "
            "fallback=off, fallback_allowed=off",
            "metal",
        )
        with self.assertRaisesRegex(RuntimeError, "physical metal"):
            MOVING_QUALIFY.require_device_log(
                "[COMPUTE] backend=vulkan, requested=vulkan, "
                "device_index=-1, fallback=off",
                "metal",
            )

    def test_moving_nest_bounded_window_cadence_and_validity(self) -> None:
        self.assertFalse(MOVING_QUALIFY.complete_cadence_window(4200, 4050))
        self.assertTrue(MOVING_QUALIFY.complete_cadence_window(4500, 4050))
        self.assertTrue(MOVING_QUALIFY.acceptable_final_validity(
            "valid_pending_relocation", True, 4200, 4050))
        self.assertFalse(MOVING_QUALIFY.acceptable_final_validity(
            "valid_pending_relocation", True, 4500, 4050))
        self.assertFalse(MOVING_QUALIFY.acceptable_final_validity(
            "invalid", True, 4200, 4050))

    def test_metal_lifetime_chain_uses_separate_fail_closed_defaults(self) -> None:
        self.assertEqual(
            CHAIN.default_segment_glob("metal"),
            "moving_nested_metal_lifetime_v1_segment_*",
        )
        self.assertEqual(
            CHAIN.default_output("metal").name,
            "moving_nested_metal_lifetime_v1_chain_report.json",
        )
        self.assertEqual(CHAIN.default_segment_glob("vulkan"), CHAIN.DEFAULT_GLOB)
        with self.assertRaisesRegex(CHAIN.ChainError, "unsupported compute backend"):
            CHAIN.default_segment_glob("cpu")

    def test_metal_reference_config_has_only_the_permitted_backend_delta(self) -> None:
        REFERENCE.validate_config_delta(REFERENCE.DEFAULT_METAL_CONFIG, "metal")
        self.assertEqual(
            REFERENCE.expected_ownership("metal")["extensions.compute.backend"],
            "metal",
        )
        with self.assertRaisesRegex(RuntimeError, "ownership tuple changed"):
            REFERENCE.validate_config_delta(REFERENCE.DEFAULT_CONFIG, "metal")

    def test_metal_reference_parser_uses_generic_dispatch_and_residency_telemetry(self) -> None:
        log = "\n".join((
            "[PERF SUMMARY] measured_steps=40, measured_simulated_s=30",
            "[COMPUTE DISPATCH] backend=metal, completed_submissions=240, "
            "kernel_dispatches=280, resident_uploads=14, resident_reuses=21, "
            "kernels={microphysics/kessler_pointwise.comp:40,"
            "turbulence/smagorinsky_coefficients.comp:40}",
        ))
        observed = REFERENCE.parse_gpu_telemetry(log, "metal")
        self.assertEqual(observed["model_steps"], 40)
        self.assertEqual(observed["command_buffers"], 240)
        self.assertEqual(observed["dispatches"], 280)
        self.assertEqual(observed["resident_hits"], 21)
        self.assertEqual(observed["resident_misses"], 14)

    def test_metal_moving_config_changes_only_the_backend_selector(self) -> None:
        vulkan = REFERENCE.extractor.flat_yaml_scalars(MOVING_QUALIFY.DEFAULT_CONFIG)
        metal = REFERENCE.extractor.flat_yaml_scalars(
            MOVING_QUALIFY.DEFAULT_METAL_CONFIG)
        differences = {
            key: (vulkan.get(key), metal.get(key))
            for key in vulkan.keys() | metal.keys()
            if vulkan.get(key) != metal.get(key)
        }
        self.assertEqual(
            differences,
            {"extensions.compute.backend": ("vulkan", "metal")},
        )

    def test_metal_reference_rejects_a_vulkan_lifetime_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "nimbus"
            binary.write_bytes(b"same-binary")
            digest = hashlib.sha256(binary.read_bytes()).hexdigest()
            chain = root / "chain.json"
            chain.write_text(json.dumps({
                "schema": "nimbus.diagnostic.moving_nested_vulkan_lifetime_chain",
                "status": "complete",
                "claims": {
                    "fallback_off_physical_vulkan": True,
                    "uninterrupted_single_process": True,
                },
                "inputs": {"binary_sha256": digest},
            }), encoding="utf-8")
            with self.assertRaisesRegex(
                    RuntimeError, "moving metal lifetime is not"):
                REFERENCE.validate_binary_identity(binary, chain, "metal")

    def test_metal_reference_evidence_is_sealed_before_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "nimbus"
            binary.write_bytes(b"metal-campaign-binary")
            binary_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
            chain = root / "moving-chain.json"
            chain.write_text(json.dumps({
                "schema": "nimbus.diagnostic.moving_nested_metal_lifetime_chain",
                "status": "complete",
                "claims": {
                    "fallback_off_physical_metal": True,
                    "uninterrupted_single_process": True,
                },
                "inputs": {"binary_sha256": binary_sha},
            }), encoding="utf-8")
            chain_sha = hashlib.sha256(chain.read_bytes()).hexdigest()
            config = REFERENCE.DEFAULT_METAL_CONFIG
            config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
            report = root / "reference-report.json"
            report.write_text(json.dumps({
                "schema": "nimbus.diagnostic.moving_nested_metal_reference_runtime",
                "schema_version": 1,
                "status": "pass",
                "qualification_eligible": False,
                "scope": "same_binary_oversized_physical_metal_reference_runtime",
                "claims": {
                    "same_binary_as_moving_metal_lifetime": True,
                    "fallback_off_physical_metal": True,
                    "native_smagorinsky_sgs_dispatch": True,
                    "native_kessler_process_dispatch": True,
                    "host_authoritative_kessler_state_commit": True,
                    "strict_finite_bounded_outputs": True,
                    "formal_comparison_complete": False,
                },
                "inputs": {
                    "binary_sha256": binary_sha,
                    "config_sha256": config_sha,
                    "moving_lifetime_chain_sha256": chain_sha,
                    "duration_s": 4500,
                },
                "runtime": {
                    "measured_steps": 6000,
                    "measured_simulated_seconds": 4500.0,
                    "simulated_seconds_per_solver_wall_second": 1.0,
                    "tracker_samples": 150,
                    "diagnostic_samples": 150,
                    "gpu_telemetry": {
                        "model_steps": 6000,
                        "command_buffers": 36000,
                        "dispatches": 42000,
                        "resident_hits": 72000,
                        "resident_misses": 54000,
                        "stage_dispatches": {
                            "smagorinsky_fused": 12000,
                            "kessler_pointwise": 6000,
                        },
                    },
                },
                "artifacts": {
                    "checkpoint_count": 10,
                    "checkpoint_times_s": [
                        float(value) for value in range(450, 4501, 450)
                    ],
                    "final_checkpoint_sha256": "a" * 64,
                },
            }), encoding="utf-8")
            evidence = EVIDENCE.build_evidence(
                "metal", binary, config, chain, report)
        self.assertEqual(
            evidence["schema"],
            "nimbus.evidence.moving_nested_metal_reference_runtime",
        )
        self.assertFalse(evidence["qualification_eligible"])
        self.assertFalse(evidence["candidate_metric_values_inspected"])
        self.assertFalse(evidence["claims"]["formal_comparison_complete"])

    def test_short_reference_smoke_does_not_claim_lifetime_binding(self) -> None:
        self.assertEqual(
            REFERENCE.reference_scope("metal", 30),
            "oversized_physical_metal_reference_smoke",
        )
        self.assertEqual(
            REFERENCE.reference_scope("metal", 4500),
            "same_binary_oversized_physical_metal_reference_runtime",
        )

    def test_metal_policy_is_ratified_from_uninspected_bound_evidence(self) -> None:
        bound_artifact = REFERENCE.DEFAULT_METAL_CONFIG.resolve()
        bound_repository_path = bound_artifact.relative_to(ROOT).as_posix()
        bound_sha = hashlib.sha256(bound_artifact.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_path = root / "metal-evidence.json"
            evidence_path.write_text(json.dumps({
                "schema": "nimbus.evidence.moving_nested_metal_reference_runtime",
                "schema_version": 1,
                "status": "pass",
                "qualification_eligible": False,
                "candidate_metric_values_inspected": False,
                "inputs": {
                    "reference_config": {
                        "path": bound_repository_path, "sha256": bound_sha},
                    "moving_lifetime_chain": {
                        "path": bound_repository_path, "sha256": bound_sha},
                },
                "reference_runtime": {
                    "report": {
                        "path": bound_repository_path, "sha256": bound_sha},
                    "fresh_configured_start": True,
                    "uninterrupted_single_process": True,
                    "final_time_s": 4500,
                    "tracker_samples": 150,
                    "diagnostic_samples": 150,
                    "portable_checkpoints": 10,
                },
                "device_evidence": {
                    "fallback_off_physical_metal": True,
                    "dispatches": 42000,
                    "host_authoritative_kessler_state_commit": True,
                    "stage_dispatches": {
                        "smagorinsky_fused": 12000,
                        "kessler_pointwise": 6000,
                    },
                },
                "claims": {
                    "same_binary_candidate_and_reference": True,
                    "same_metal_backend_candidate_and_reference": True,
                    "same_comparison_level_equations": True,
                    "strict_finite_bounded_reference_outputs": True,
                    "formal_comparison_complete": False,
                    "public_moving_nest_admitted": False,
                    "device_resident_relocation": False,
                    "throughput_promotion": False,
                },
            }), encoding="utf-8")
            policy_path = root / "policy-v14.json"
            policy = RATIFIER.build_policy(
                evidence_path, policy_path, "2026-08-21")
            policy_path.write_text(
                json.dumps(policy, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            effective = RATIFIER.comparator.validate_policy(policy, policy_path)
        self.assertEqual(effective["schema_version"], 14)
        self.assertEqual(
            effective["expected_contract"]["physics_signature_sha256"],
            RATIFIER.comparator.METAL_PHYSICS_SIGNATURE_SHA256,
        )

    def test_metal_frontier_defaults_are_backend_isolated(self) -> None:
        paths = FRONTIER.backend_paths("metal")
        self.assertEqual(
            paths["policy"].name,
            "moving_nested_supercell_reference_frontier_v14.json",
        )
        self.assertEqual(
            paths["moving_config"], MOVING_QUALIFY.DEFAULT_METAL_CONFIG)
        self.assertEqual(
            paths["reference_config"], REFERENCE.DEFAULT_METAL_CONFIG)

    def test_completed_metal_frontier_seals_all_comparison_artifacts(self) -> None:
        bound = REFERENCE.DEFAULT_METAL_CONFIG.resolve()
        bound_sha = hashlib.sha256(bound.read_bytes()).hexdigest()
        record = {"path": str(bound), "sha256": bound_sha}
        with tempfile.TemporaryDirectory() as directory:
            campaign_path = Path(directory) / "campaign.json"
            campaign_path.write_text(json.dumps({
                "schema":
                    "nimbus.qualification.moving_nested_metal_reference_frontier",
                "schema_version": 1,
                "status": "pass",
                "qualification_eligible": True,
                "claims": {
                    "same_binary_candidate_and_reference": True,
                    "same_metal_backend_candidate_and_reference": True,
                    "fallback_off_physical_metal_both_branches": True,
                    "formal_reference_comparison_complete": True,
                    "public_moving_nest_admitted": False,
                    "device_resident_relocation": False,
                    "throughput_promotion": False,
                },
                "inputs": {
                    "binary_sha256": "b" * 64,
                    "policy": record,
                    "moving_lifetime_chain": record,
                    "moving_runtime_report": record,
                    "reference_runtime_report": record,
                },
                "comparison_artifacts": {
                    "moving_series": record,
                    "moving_source_manifest": record,
                    "reference_series": record,
                    "reference_source_manifest": record,
                    "comparison_report": record,
                },
                "coverage": {"selected": 150},
                "metrics": {"path_rmse_m": 1.0},
                "checks": [
                    {"name": f"check-{index}", "passed": True}
                    for index in range(21)
                ],
                "runtime_summary": {"moving_tracker_samples": 150},
            }), encoding="utf-8")
            evidence = FRONTIER_EVIDENCE.build_evidence(
                "metal", campaign_path)
        self.assertTrue(evidence["qualification_eligible"])
        self.assertEqual(len(evidence["checks"]), 21)
        self.assertEqual(
            set(evidence["comparison_artifacts"]), {
                "moving_series", "moving_source_manifest",
                "reference_series", "reference_source_manifest",
                "comparison_report",
            })


if __name__ == "__main__":
    unittest.main()
