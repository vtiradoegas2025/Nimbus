from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT / "scripts/qualification/accelerators/"
    "seal_kernel_dispatch_coverage.py"
)


class KernelDispatchCoverageSealTests(unittest.TestCase):
    def invoke(
        self,
        temporary: Path,
        *,
        ownership: Path | None = None,
        tuples: Path | None = None,
        metal: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "--binary",
                str(ROOT / "bin/nimbus_sim"),
                "--kernel-ownership",
                str(ownership or ROOT / "reports/pre_beta/backend_kernel_ownership_v1.json"),
                "--tuple-observation",
                str(tuples or ROOT / "reports/pre_beta/backend_tuple_operation_observation_v1.json"),
                "--vulkan-report",
                str(ROOT / "reports/qualification/vulkan-qualification-current.json"),
                "--vulkan-log",
                str(ROOT / "reports/qualification/vulkan-qualification-current.log"),
                "--metal-report",
                str(metal or ROOT / "reports/qualification/metal/bootstrap-current.json"),
                "--metal-log",
                str(ROOT / "reports/qualification/metal/bootstrap-current.log"),
                "--output",
                str(temporary / "coverage.json"),
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

    @staticmethod
    def write(path: Path, value: object) -> Path:
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_current_evidence_seals(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nimbus-kernel-seal-", dir=ROOT) as name:
            result = self.invoke(Path(name))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("47/47 active operations classified", result.stdout)

    def test_stale_binary_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nimbus-kernel-seal-", dir=ROOT) as name:
            temporary = Path(name)
            ownership = json.loads(
                (ROOT / "reports/pre_beta/backend_kernel_ownership_v1.json").read_text()
            )
            ownership["binary"]["sha256"] = "0" * 64
            result = self.invoke(
                temporary,
                ownership=self.write(temporary / "ownership.json", ownership),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not bind the current binary", result.stderr)

    def test_duplicate_tuple_operation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nimbus-kernel-seal-", dir=ROOT) as name:
            temporary = Path(name)
            tuples = json.loads(
                (ROOT / "reports/pre_beta/backend_tuple_operation_observation_v1.json").read_text()
            )
            operations = tuples["summary"]["observedOperationIds"]
            operations[-1] = operations[0]
            result = self.invoke(
                temporary,
                tuples=self.write(temporary / "tuples.json", tuples),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("operation set drifted", result.stderr)

    def test_malformed_metal_manifest_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nimbus-kernel-seal-", dir=ROOT) as name:
            temporary = Path(name)
            metal = json.loads(
                (ROOT / "reports/qualification/metal/bootstrap-current.json").read_text()
            )
            metal["manifest"]["canonical_fnv1a64"] = "not-a-hash"
            result = self.invoke(
                temporary,
                metal=self.write(temporary / "metal.json", metal),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Metal qualification evidence", result.stderr)


if __name__ == "__main__":
    unittest.main()
