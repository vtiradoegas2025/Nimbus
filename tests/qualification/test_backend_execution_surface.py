from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/qualification/provenance/seal_backend_execution_surface.py"
SPEC = importlib.util.spec_from_file_location("backend_execution", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EXECUTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXECUTION)


class BackendExecutionSurfaceTests(unittest.TestCase):
    def test_bounded_inventory_is_exact(self) -> None:
        evidence = json.loads((ROOT / "reports/pre_beta/catalog_bounded_cpu_v1.json").read_text())
        EXECUTION.validate_bounded(evidence, EXECUTION.sha256_file(ROOT / "bin/nimbus_sim"))
        self.assertEqual(len(EXECUTION.expected_cpu_cases()), EXECUTION.inventory_counts()["cpuCases"])

    def test_restart_inventory_is_exact(self) -> None:
        evidence = json.loads((ROOT / "reports/pre_beta/catalog_restart_threading_v1.json").read_text())
        EXECUTION.validate_restart(evidence, EXECUTION.sha256_file(ROOT / "bin/nimbus_sim"))

    def test_current_inputs_are_coherent(self) -> None:
        records = EXECUTION.validate_inputs(EXECUTION.sha256_file(ROOT / "bin/nimbus_sim"))
        self.assertEqual(set(records), set(EXECUTION.input_paths()))

    def test_current_surface_verifies(self) -> None:
        evidence = json.loads((ROOT / "reports/pre_beta/backend_execution_surface_v1.json").read_text())
        EXECUTION.verify_evidence(evidence, EXECUTION.sha256_file(ROOT / "bin/nimbus_sim"))

    def test_mutated_summary_fails_closed(self) -> None:
        evidence = json.loads((ROOT / "reports/pre_beta/backend_execution_surface_v1.json").read_text())
        evidence["summary"]["cpuCases"] -= 1
        with self.assertRaisesRegex(RuntimeError, "summary drifted"):
            EXECUTION.verify_evidence(evidence, EXECUTION.sha256_file(ROOT / "bin/nimbus_sim"))


if __name__ == "__main__":
    unittest.main()
