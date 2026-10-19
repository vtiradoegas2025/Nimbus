from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/qualification/provenance/record_gate_result.py"
SPEC = importlib.util.spec_from_file_location("record_gate_result", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


class GateResultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="nimbus-gate-result-", dir=ROOT,
        )
        self.directory = Path(self.temporary.name)
        self.log = self.directory / "gate.log"
        self.receipt = self.directory / "gate.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def record(
        self, *, stdout: bytes = b"gate passed fallback=off\n",
        returncode: int = 0,
    ) -> dict[str, object]:
        result = GATE.subprocess.CompletedProcess(
            ["make", "qualify-io"], returncode, stdout=stdout,
        )
        real_run = GATE.subprocess.run
        def run(command: list[str], *args: object, **kwargs: object) -> object:
            if command and command[0] == "git":
                return real_run(command, *args, **kwargs)
            return result
        with mock.patch.object(GATE.subprocess, "run", side_effect=run):
            return GATE.record_gate(
                root=ROOT,
                target="qualify-io",
                binary=ROOT / "bin/nimbus_sim",
                command=["make", "qualify-io"],
                required_hardware=False,
                log=self.log,
                receipt=self.receipt,
                artifacts=[],
                sources=[ROOT / "README.md"],
            )

    def test_clean_gate_writes_typed_receipt(self) -> None:
        record = self.record()
        self.assertEqual(record["status"], "pass")
        self.assertEqual(record["skippedCount"], 0)
        self.assertFalse(record["fallbackUsed"])
        self.assertEqual(json.loads(self.receipt.read_text()), record)
        self.assertEqual(record["logSha256"], GATE.sha256_file(self.log))
        self.assertEqual(record["version"], 2)
        self.assertIn("tree", record["revision"])
        self.assertIn("tracked_diff_sha256", record["revision"])
        self.assertRegex(record["source"]["sha256"], r"^[0-9a-f]{64}$")

    def test_source_change_during_gate_rejects(self) -> None:
        source = self.directory / "source.txt"
        source.write_text("before\n")

        real_run = GATE.subprocess.run
        def mutate(command: list[str], *args: object, **kwargs: object) -> object:
            if command and command[0] == "git":
                return real_run(command, *args, **kwargs)
            source.write_text("after\n")
            return GATE.subprocess.CompletedProcess(
                ["make", "qualify-io"], 0,
                stdout=b"gate passed fallback=off\n",
            )

        with mock.patch.object(GATE.subprocess, "run", side_effect=mutate):
            with self.assertRaisesRegex(
                GATE.GateResultError, "revision or release source"
            ):
                GATE.record_gate(
                    root=ROOT,
                    target="qualify-io",
                    binary=ROOT / "bin/nimbus_sim",
                    command=["make", "qualify-io"],
                    required_hardware=False,
                    log=self.log,
                    receipt=self.receipt,
                    artifacts=[],
                    sources=[source],
                )

    def test_nonzero_empty_skip_and_fallback_fail_closed(self) -> None:
        cases = (
            (b"failed\n", 1, "exited"),
            (b"", 0, "empty log"),
            (b"[SKIP] device\n", 0, "required skip"),
            (b"falling back to CPU\n", 0, "fallback"),
        )
        for stdout, returncode, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(GATE.GateResultError, message):
                    self.record(stdout=stdout, returncode=returncode)

    def test_policy_rejects_forged_command_hardware_and_artifacts(self) -> None:
        common = dict(
            root=ROOT,
            binary=ROOT / "bin/nimbus_sim",
            log=self.log,
            receipt=self.receipt,
            sources=[ROOT / "README.md"],
        )
        with self.assertRaisesRegex(GATE.GateResultError, "canonical"):
            GATE.record_gate(
                **common,
                target="qualify-io",
                command=["echo", "pass"],
                required_hardware=False,
                artifacts=[],
            )
        with self.assertRaisesRegex(GATE.GateResultError, "hardware"):
            GATE.record_gate(
                **common,
                target="qualify-io",
                command=["make", "qualify-io"],
                required_hardware=True,
                artifacts=[],
            )
        artifact = self.directory / "unexpected.json"
        artifact.write_text("{}\n")
        with self.assertRaisesRegex(GATE.GateResultError, "artifact"):
            GATE.record_gate(
                **common,
                target="qualify-io",
                command=["make", "qualify-io"],
                required_hardware=False,
                artifacts=[artifact],
            )

    def test_final_surface_policy_records_only_its_canonical_aggregate(self) -> None:
        policy = GATE.REQUIRED_GATE_POLICY["qualify-release-final-surface"]
        self.assertEqual(
            policy["command"], ("make", "qualify-release-final-surface")
        )
        self.assertFalse(policy["requiredHardware"])
        self.assertEqual(policy["artifacts"], ())
        result = GATE.subprocess.CompletedProcess(
            ["make", "qualify-release-final-surface"], 0,
            stdout=b"[release-final-surface] PASS\n",
        )
        real_run = GATE.subprocess.run

        def run(command: list[str], *args: object, **kwargs: object) -> object:
            if command and command[0] == "git":
                return real_run(command, *args, **kwargs)
            return result

        with mock.patch.object(GATE.subprocess, "run", side_effect=run):
            record = GATE.record_gate(
                root=ROOT,
                target="qualify-release-final-surface",
                binary=ROOT / "bin/nimbus_sim",
                command=["make", "qualify-release-final-surface"],
                required_hardware=False,
                log=self.log,
                receipt=self.receipt,
                artifacts=[],
                sources=[ROOT / "README.md"],
            )
        self.assertEqual(record["target"], "qualify-release-final-surface")
        self.assertEqual(record["version"], 2)

    def test_external_artifact_and_symlink_binary_reject(self) -> None:
        (self.directory / "binary").write_bytes(b"binary")
        with self.assertRaisesRegex(GATE.GateResultError, "outside"):
            GATE.record_gate(
                root=self.directory,
                target="qualify-io",
                binary=self.directory / "binary",
                command=["make", "qualify-io"],
                required_hardware=False,
                log=self.log,
                receipt=self.receipt,
                artifacts=[ROOT / "README.md"],
                sources=[self.directory / "binary"],
            )


if __name__ == "__main__":
    unittest.main()
