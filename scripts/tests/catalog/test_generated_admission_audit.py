#!/usr/bin/env python3
"""Admission inventory identity, reader selection and publication regressions."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "bin/nimbus_native_admission_batch"
BINARY = ROOT / "bin/nimbus_sim"
spec = importlib.util.spec_from_file_location("generated_audit", ROOT / "scripts/qualification/audit_generated_admission.py")
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)
BASE = ("grid=c_grid|backend=cpu|qualified=false|microphysics=none|radiation=none|"
        "boundary_layer=none|turbulence=none|chaos=none|diffusion=explicit|terrain=none|"
        "dynamics=klemp_wilhelmson_split_cartesian_cgrid|advection=tvd|"
        "time_stepping=split_explicit|turbulence_mode=3d|surface=none|"
        "surface_fluxes=false|sounding={sounding}|output=npy_3d\n")


class GeneratedAdmissionAuditTests(unittest.TestCase):
    def invoke(self, payload, *args):
        return subprocess.run([str(TOOL), *args], input=payload, capture_output=True, cwd=ROOT)

    def identity(self, payload):
        return ("--expected-count", str(len(payload.splitlines())), "--expected-sha256",
                hashlib.sha256(payload).hexdigest())

    def test_distinct_reader_materialization_and_public_loading(self):
        payload = "".join(BASE.format(sounding=name) for name in ("auto", "cm1", "sharpy")).encode()
        result = self.invoke(payload, "--classify", *self.identity(payload))
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([row["admissionStatus"] for row in rows], ["preflight_admitted"] * 3)
        self.assertEqual(rows[0]["configSha256"], rows[2]["configSha256"])
        self.assertNotEqual(rows[0]["configSha256"], rows[1]["configSha256"])
        self.assertTrue(rows[1]["soundingPath"].endswith("catalog_cm1_profile.txt"))
        with tempfile.TemporaryDirectory(prefix="nimbus-reader-audit-") as temporary:
            for name in ("cm1", "sharpy"):
                result = self.invoke(BASE.format(sounding=name).encode(), "--yaml")
                self.assertEqual(result.returncode, 0, result.stderr)
                config = Path(temporary) / f"{name}.yaml"
                config.write_bytes(result.stdout)
                checked = subprocess.run([str(BINARY), "--validate-only", "--config", str(config)],
                                         cwd=ROOT, capture_output=True)
                self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)

    def test_invalid_qualified_value_is_not_exploratory(self):
        payload = BASE.format(sounding="cm1").replace("qualified=false", "qualified=garbage").encode()
        result = self.invoke(payload, "--yaml")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"invalid boolean qualified", result.stderr)

    def test_stream_contract_is_required_and_checked(self):
        payload = BASE.format(sounding="cm1").encode()
        cases = [(), ("--expected-count", "1"),
                 ("--expected-count", "-1", "--expected-sha256", "0" * 64),
                 ("--expected-count", "1", "--expected-sha256", "0" * 64),
                 ("--expected-count", "2", "--expected-sha256", hashlib.sha256(payload).hexdigest())]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                self.assertNotEqual(self.invoke(payload, "--classify", *arguments).returncode, 0)

    def generated_tool(self, directory, payload, digest=None):
        catalog = {"schema": "nimbus.generated-backend-tuples.v1",
                   "summary": {"accepted": len(payload.splitlines())},
                   "partitions": {"c_grid.cpu.exploratory": len(payload.splitlines())},
                   "acceptedTupleSha256": digest or hashlib.sha256(payload).hexdigest()}
        path = directory / "generator"
        path.write_text("#!/usr/bin/env python3\nimport sys\n"
                        f"sys.stdout.buffer.write({payload!r} if '--tuples' in sys.argv "
                        f"else {json.dumps(catalog).encode()!r})\n")
        path.chmod(0o755)
        return path

    def test_audit_rejects_bad_checksum_without_publication(self):
        with tempfile.TemporaryDirectory(prefix="nimbus-inventory-atomic-") as temporary:
            directory = Path(temporary)
            generated = self.generated_tool(directory, BASE.format(sounding="cm1").encode(), "0" * 64)
            output = directory / "evidence"
            with self.assertRaisesRegex(RuntimeError, "pipeline failed"):
                audit_module.audit(generated, TOOL, BINARY, output)
            self.assertFalse((output / "summary.json").exists())
            self.assertFalse((output / "classifications.jsonl.gz").exists())
            self.assertFalse((output / "classifications.jsonl.gz.partial").exists())

    def test_audit_rejects_duplicate_tuples_without_publication(self):
        with tempfile.TemporaryDirectory(prefix="nimbus-inventory-duplicate-") as temporary:
            directory = Path(temporary)
            generated = self.generated_tool(directory, (BASE.format(sounding="cm1") * 2).encode())
            output = directory / "evidence"
            with self.assertRaisesRegex(RuntimeError, "duplicate canonical tuple"):
                audit_module.audit(generated, TOOL, BINARY, output)
            self.assertFalse((output / "summary.json").exists())
            self.assertFalse((output / "classifications.jsonl.gz").exists())

    def test_complete_audit_accounts_for_each_tuple(self):
        with tempfile.TemporaryDirectory(prefix="nimbus-inventory-complete-") as temporary:
            directory = Path(temporary)
            payload = "".join(BASE.format(sounding=name) for name in ("auto", "cm1", "sharpy")).encode()
            generated = self.generated_tool(directory, payload)
            result = audit_module.audit(generated, TOOL, BINARY, directory / "evidence")
            self.assertEqual(result["inputTuples"], 3)
            self.assertEqual(result["counts"]["preflight_admitted"], 3)
            self.assertEqual(result["distinctSerializedConfigs"], 2)
            self.assertEqual(result["readinessStatus"], "Engineering incomplete")


if __name__ == "__main__":
    unittest.main()
