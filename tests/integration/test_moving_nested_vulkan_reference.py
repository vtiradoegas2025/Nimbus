from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "qualification" / "nested_domains"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import qualify_moving_nested_vulkan_reference as runtime
import qualify_moving_nested_vulkan_reference_frontier as frontier


RUNTIME_REPORT = (
    ROOT / "tests/artifacts/moving_nested_vulkan_reference_v1_20260819/"
    "moving_nested_vulkan_reference_runtime_report.json"
)
CAMPAIGN_REPORT = (
    ROOT / "tests/artifacts/"
    "moving_nested_vulkan_reference_frontier_v13_20260819/campaign_report.json"
)
EVIDENCE = (
    ROOT / "tests/fixtures/reference/domain_history/"
    "moving_nested_vulkan_reference_frontier_v13_pass.json"
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class MovingNestedVulkanReferenceTests(unittest.TestCase):
    def test_reference_config_delta_is_exact(self) -> None:
        runtime.validate_config_delta(runtime.DEFAULT_CONFIG)

    def test_telemetry_parser_uses_final_nonzero_backend(self) -> None:
        log = "\n".join((
            "[GPU TELEMETRY] model_steps=0 command_buffers=0 dispatches=0 "
            "resident_hits=0 resident_misses=0",
            "[GPU STAGE TELEMETRY] smagorinsky_fused command_buffers=2 "
            "dispatches=4",
            "[GPU STAGE TELEMETRY] microphysics/kessler_pointwise.comp.spv "
            "command_buffers=2 dispatches=2",
            "[GPU TELEMETRY] model_steps=2 command_buffers=4 dispatches=6 "
            "resident_hits=8 resident_misses=4",
        ))
        observed = runtime.parse_gpu_telemetry(log)
        self.assertEqual(observed["model_steps"], 2)
        self.assertEqual(observed["dispatches"], 6)

    @unittest.skipUnless(RUNTIME_REPORT.is_file(), "physical runtime artifact absent")
    def test_physical_reference_report_is_complete(self) -> None:
        report = json.loads(RUNTIME_REPORT.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["runtime"]["tracker_samples"], 150)
        self.assertEqual(report["runtime"]["diagnostic_samples"], 150)
        self.assertEqual(report["artifacts"]["checkpoint_count"], 10)
        self.assertEqual(report["runtime"]["gpu_telemetry"]["dispatches"], 42000)
        self.assertFalse(report["claims"]["formal_comparison_complete"])

    @unittest.skipUnless(CAMPAIGN_REPORT.is_file(), "formal campaign artifact absent")
    def test_formal_campaign_passes_all_checks_without_broadening(self) -> None:
        report = json.loads(CAMPAIGN_REPORT.read_text(encoding="utf-8"))
        self.assertEqual(report["schema"], frontier.SCHEMA)
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["qualification_eligible"])
        self.assertEqual(len(report["checks"]), 21)
        self.assertTrue(all(item["passed"] for item in report["checks"]))
        self.assertFalse(report["claims"]["public_moving_nest_admitted"])
        self.assertFalse(report["claims"]["device_resident_relocation"])
        self.assertFalse(report["claims"]["throughput_promotion"])

    @unittest.skipUnless(EVIDENCE.is_file() and CAMPAIGN_REPORT.is_file(),
                         "formal evidence artifact absent")
    def test_evidence_binds_formal_campaign_report(self) -> None:
        evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        record = evidence["comparison_artifacts"]["campaign_report"]
        self.assertEqual(ROOT / record["path"], CAMPAIGN_REPORT)
        self.assertEqual(record["sha256"], digest(CAMPAIGN_REPORT))
        self.assertTrue(evidence["qualification_eligible"])


if __name__ == "__main__":
    unittest.main()
