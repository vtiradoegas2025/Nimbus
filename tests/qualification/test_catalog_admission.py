import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.qualification import catalog_admission as catalog


def row(physics, surface="none"):
    return f"grid=c_grid|backend=cpu|qualified=false|microphysics={physics}|surface={surface}"


class CatalogAdmissionStreamTests(unittest.TestCase):
    def test_streaming_witnesses_preserve_all_owner_and_axis_witnesses(self):
        entries = [("first", row("none")), ("first", row("kessler")),
                   ("second", row("none", "slab")), ("second", row("kessler", "slab"))]
        lines = [f"{signature}\t{text}\n".encode() for signature, text in entries]
        selected, summary, digest = catalog.select_witnesses(iter(lines))
        self.assertEqual(selected, sorted(entries[:3], key=lambda entry: entry[1]))
        self.assertEqual(summary["inputTuples"], 4)
        self.assertEqual(summary["ownershipSignatures"], 2)
        self.assertEqual(digest, hashlib.sha256(b"".join(lines)).hexdigest())

    def test_signed_duplicates_and_truncation_reject(self):
        line = f"owner\t{row('none')}\n".encode()
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            catalog.select_witnesses(iter([line, line]))
        with self.assertRaisesRegex(RuntimeError, "terminator"):
            catalog.select_witnesses(iter([line[:-1]]))

    def test_admitted_stream_identity_and_mutations(self):
        payload = (row("none") + "\n" + row("kessler") + "\n").encode()
        digest = hashlib.sha256(payload).hexdigest()
        partitions = {"c_grid.cpu.exploratory": 2}
        self.assertEqual(catalog.stream_identity(io.BytesIO(payload), 2, digest, partitions),
                         (digest, partitions))
        for changed in (payload[:-1], payload + payload, b"".join(reversed(payload.splitlines(keepends=True)))):
            with self.subTest(changed=changed), self.assertRaises(RuntimeError):
                catalog.stream_identity(io.BytesIO(changed), 2, digest, partitions)

    def executable(self, directory, name, body):
        path = directory / name
        path.write_text("#!/usr/bin/env python3\n" + body)
        path.chmod(0o755)
        return path

    def test_large_stderr_does_not_block_the_pipeline(self):
        with tempfile.TemporaryDirectory(prefix="nimbus-admission-pipe-") as temporary:
            directory = Path(temporary)
            producer = self.executable(directory, "producer", "import sys\nsys.stderr.write('x'*262144)\nprint('tuple')\n")
            consumer = self.executable(directory, "consumer", "import sys\nsys.stdout.write(sys.stdin.read())\n")
            with catalog.native_stream(producer, consumer) as stream:
                self.assertEqual(stream.read(), b"tuple\n")

    def test_consumer_failure_is_not_successful_output(self):
        with tempfile.TemporaryDirectory(prefix="nimbus-admission-failure-") as temporary:
            directory = Path(temporary)
            producer = self.executable(directory, "producer", "print('tuple')\n")
            consumer = self.executable(directory, "consumer", "import sys\nprint(sys.stdin.read())\nsys.exit(7)\n")
            with self.assertRaisesRegex(RuntimeError, "pipeline failed"):
                with catalog.native_stream(producer, consumer) as stream:
                    stream.read()

    def test_changed_executable_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="nimbus-admission-mutation-") as temporary:
            directory = Path(temporary)
            producer = self.executable(directory, "producer", "print('tuple')\n")
            consumer = self.executable(directory, "consumer",
                "import sys\nfrom pathlib import Path\nprint(sys.stdin.read())\n"
                "with Path(__file__).open('a') as f: f.write('# mutated\\n')\n")
            with self.assertRaisesRegex(RuntimeError, "inputs changed"):
                with catalog.native_stream(producer, consumer) as stream:
                    stream.read()


class CatalogAdmissionEvidenceTests(unittest.TestCase):
    def test_valid_looking_owner_change_rejects(self):
        from scripts.tests.catalog import test_generated_backend_tuples as generated
        evidence = json.loads((catalog.ROOT / "reports/pre_beta/generated_backend_tuple_evidence_v1.json").read_text())
        evidence["catalog"]["ownershipGroups"][0]["signature"] += "|extra=host"
        with self.assertRaisesRegex(RuntimeError, "group identity drifted"):
            generated.validate_catalog(evidence["catalog"])

    def test_wrong_source_role_rejects(self):
        from scripts.tests.catalog import test_generated_backend_tuples as generated
        evidence = json.loads((catalog.ROOT / "reports/pre_beta/generated_backend_tuple_evidence_v1.json").read_text())
        evidence["source"] = generated.record(catalog.ROOT / "scripts/qualification/catalog_inventory.py")
        with self.assertRaisesRegex(RuntimeError, "source path drifted"):
            generated.verify_evidence(evidence, generated.sha256_file(catalog.ROOT / "bin/nimbus_sim"))

    def test_stale_upstream_binary_rejects_even_with_a_correct_file_hash(self):
        from scripts.tests.catalog import test_native_admission_batch as admission
        evidence = json.loads((catalog.ROOT / "reports/pre_beta/native_admission_evidence_v1.json").read_text())
        upstream = json.loads((catalog.ROOT / evidence["generatedEvidence"]["path"]).read_text())
        upstream["binary"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory(prefix="admission-upstream-", dir=catalog.ROOT / "build") as temporary:
            path = Path(temporary) / "generated.json"
            path.write_text(json.dumps(upstream))
            evidence["generatedEvidence"] = admission.record(path)
            with self.assertRaisesRegex(RuntimeError, "current binary"):
                admission.verify_evidence(evidence, admission.sha256_file(catalog.ROOT / "bin/nimbus_sim"))

    def test_changed_sounding_identity_rejects(self):
        from scripts.tests.catalog import test_native_full_preflight as preflight
        evidence = json.loads((catalog.ROOT / "reports/pre_beta/native_full_preflight_evidence_v1.json").read_text())
        evidence["soundingInputs"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "sounding input identities drifted"):
            preflight.verify_evidence(evidence, preflight.sha256_file(catalog.ROOT / "bin/nimbus_sim"))


if __name__ == "__main__":
    unittest.main()
