from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest
import warnings
import zipfile


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/qualification/provenance/release_manifest.py"
SPEC = importlib.util.spec_from_file_location("release_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RELEASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELEASE)


class ReleaseManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="nimbus-release-manifest-", dir=ROOT
        )
        self.root = Path(self.temporary.name)
        (self.root / "source").mkdir()
        (self.root / "source/main.cpp").write_text("int main() { return 0; }\n")
        (self.root / "source/model.hpp").write_text("#pragma once\n")
        (self.root / "nimbus_sim").write_bytes(b"binary")
        (self.root / "profile.yaml").write_text("schema: nimbus.v1\n")
        (self.root / "evidence.json").write_text('{"passed":true}\n')
        (self.root / "field.npy").write_bytes(b"artifact")
        (self.root / "package.bin").write_bytes(b"package")
        (self.root / "gate.log").write_text("PASS fallback=off\n")
        (self.root / "gate-artifact.json").write_text('{"bounded":true}\n')
        (self.root / "cross-product-bin").write_bytes(b"cross-product")
        (self.root / "cross-product.cpp").write_text("// cross product\n")
        (self.root / "reachable.log").write_text("all matrices passed\n")
        self.catalog_fingerprint = "a" * 64
        (self.root / "catalog-authority.json").write_text(json.dumps({
            "schema": "nimbus.scheme-catalog.v9",
            "catalogVersion": 9,
            "authorityFingerprintSha256": self.catalog_fingerprint,
            "moduleCount": 12,
            "schemeCount": 38,
            "modules": [],
        }))
        self.binary_sha256 = RELEASE.sha256_file(self.root / "nimbus_sim")
        authority = json.loads((self.root / "catalog-authority.json").read_text())
        matrices = [{"id": "maintained", "evaluated": 2}]
        scope = {
            "categoricalExactBinary": True,
            "continuousParametersEnumerated": False,
            "runtimeQualificationUniversal": False,
        }
        reachable = {
            "schema": RELEASE.REACHABLE_EVIDENCE_SCHEMA,
            "version": 1,
            "createdUtc": "2026-08-29T00:00:00+00:00",
            "scope": scope,
            "binary": RELEASE.file_record(self.root / "nimbus_sim", self.root),
            "catalogAuthority": RELEASE.file_record(
                self.root / "catalog-authority.json", self.root
            ),
            "catalog": authority,
            "crossProductBinary": RELEASE.file_record(
                self.root / "cross-product-bin", self.root
            ),
            "crossProductSource": RELEASE.file_record(
                self.root / "cross-product.cpp", self.root
            ),
            "matrices": matrices,
            "totalEvaluated": 2,
            "receipt": {
                "target": "test-catalog-reachable-matrix/cross-product",
                "command": ["cross-product-bin", "[cross-product]"],
                "exitStatus": 0,
                "skipped": 0,
                "fallbackUsed": False,
                "requiredHardware": False,
                "log": RELEASE.file_record(
                    self.root / "reachable.log", self.root
                ),
            },
        }
        reachable["reachableSetSha256"] = RELEASE.hashlib.sha256(
            RELEASE.canonical_bytes({
                "catalogAuthorityFingerprintSha256": self.catalog_fingerprint,
                "scope": scope,
                "matrices": matrices,
                "totalEvaluated": 2,
            })
        ).hexdigest()
        reachable["integritySealSha256"] = RELEASE.hashlib.sha256(
            RELEASE.canonical_bytes(reachable)
        ).hexdigest()
        (self.root / "reachable.json").write_text(json.dumps(reachable))
        (self.root / "gate.json").write_text(json.dumps({
            "schema": RELEASE.GATE_SCHEMA,
            "version": 1,
            "target": "test-catalog-authority",
            "command": ["make", "test-catalog-authority"],
            "status": "pass",
            "startedUtc": "2026-08-29T00:00:00+00:00",
            "completedUtc": "2026-08-29T00:00:01+00:00",
            "wallSeconds": 1.0,
            "requiredHardware": False,
            "skippedCount": 0,
            "fallbackUsed": False,
            "binary": "nimbus_sim",
            "binarySha256": self.binary_sha256,
            "log": "gate.log",
            "logSha256": RELEASE.sha256_file(self.root / "gate.log"),
            "artifacts": [],
        }))
        subprocess.run(
            ["git", "init", "-q", str(self.root)], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        subprocess.run(
            ["git", "-C", str(self.root), "add", "."], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        subprocess.run(
            [
                "git", "-C", str(self.root),
                "-c", "user.name=Nimbus Test",
                "-c", "user.email=nimbus-test@example.invalid",
                "commit", "-q", "-m", "fixture",
            ],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def arguments(self) -> dict[str, object]:
        return dict(
            root=self.root,
            binary=self.root / "nimbus_sim",
            config=self.root / "profile.yaml",
            backend="cpu",
            sources=[self.root / "source"],
            evidence=[self.root / "evidence.json"],
            catalog_authority=self.root / "catalog-authority.json",
            reachable_set=self.root / "reachable.json",
            gates=[self.root / "gate.json"],
            package=self.root / "package.bin",
            package_kind="test-bundle",
            package_platform="test-os",
            package_architecture="test-arch",
            artifacts=[self.root / "field.npy"],
            campaign_id="bounded-ready-gate",
            device_identity="test-cpu",
            toolchain_identity="test-c++ 1.0",
        )

    def create(self) -> dict[str, object]:
        return RELEASE.create_manifest(**self.arguments())

    def required_gate_paths(self, version: int = 2) -> list[Path]:
        template = json.loads((self.root / "gate.json").read_text())
        required_reachable = (
            self.root / "reports/pre_beta/reachable_set_v1.json"
        )
        required_reachable.parent.mkdir(parents=True, exist_ok=True)
        required_reachable.write_bytes((self.root / "reachable.json").read_bytes())
        gate_paths: list[Path] = []
        revision = RELEASE.git_identity(self.root)
        inventory = RELEASE.inventory_record([self.root / "source"], self.root)
        source = {
            "roots": inventory["roots"],
            "count": inventory["count"],
            "sha256": inventory["sha256"],
        }
        for target in RELEASE.REQUIRED_RELEASE_GATES:
            for artifact in RELEASE.REQUIRED_GATE_POLICY[target]["artifacts"]:
                artifact_path = self.root / str(artifact)
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                artifact_path.write_bytes(
                    f"fixture artifact for {target}: {artifact}\n".encode()
                )
        for index, target in enumerate(sorted(RELEASE.REQUIRED_RELEASE_GATES)):
            gate = dict(template)
            gate["version"] = version
            gate["target"] = target
            gate["command"] = ["make", target]
            policy = RELEASE.REQUIRED_GATE_POLICY[target]
            gate["requiredHardware"] = policy["requiredHardware"]
            gate["artifacts"] = list(policy["artifacts"])
            if version == 2:
                gate["revision"] = revision
                gate["source"] = source
            else:
                gate.pop("revision", None)
                gate.pop("source", None)
            path = self.root / f"required-gate-{index}.json"
            path.write_text(json.dumps(gate))
            gate_paths.append(path)
        return gate_paths

    def write_zip_bundle(
        self,
        path: Path,
        members: list[tuple[str, bytes, int]] | None = None,
    ) -> None:
        entries = members if members is not None else [
            (
                "Nimbus Studio.app/Contents/MacOS/nimbus_sim",
                (self.root / "nimbus_sim").read_bytes(),
                stat.S_IFREG | 0o755,
            ),
            (
                "Nimbus Studio.app/Contents/Resources/index.html",
                b"<html></html>\n",
                stat.S_IFREG | 0o644,
            ),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "w") as archive:
                for name, payload, mode in entries:
                    info = zipfile.ZipInfo(name)
                    info.create_system = 3
                    info.external_attr = mode << 16
                    archive.writestr(info, payload)

    def write_tar_bundle(self, path: Path) -> None:
        with tarfile.open(path, "w:gz") as archive:
            for name, payload, mode in (
                (
                    "Nimbus Studio.app/Contents/MacOS/nimbus_sim",
                    (self.root / "nimbus_sim").read_bytes(),
                    0o755,
                ),
                (
                    "Nimbus Studio.app/Contents/Resources/index.html",
                    b"<html></html>\n",
                    0o644,
                ),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                info.mode = mode
                archive.addfile(info, io.BytesIO(payload))

    def production_arguments(self, archive_format: str = "zip") -> dict[str, object]:
        suffix = ".zip" if archive_format == "zip" else ".tar.gz"
        package = self.root / f"Nimbus-Studio{suffix}"
        if archive_format == "zip":
            self.write_zip_bundle(package)
        else:
            self.write_tar_bundle(package)
        return {
            **self.arguments(),
            "package": package,
            "package_kind": "app-bundle-archive",
            "package_platform": "macos",
            "package_architecture": "arm64",
        }

    def signable_unsigned_manifest(self) -> dict[str, object]:
        manifest = RELEASE.create_manifest(**{
            **self.production_arguments(),
            "gates": self.required_gate_paths(),
        })
        manifest["eligibility"] = {
            "scope": "local_release_candidate",
            "releaseEligible": False,
            "gateScopeComplete": True,
            "missingPrerequisites": ["external_signature_absent"],
            "reason": "external_signature_absent",
        }
        self.reseal(manifest)
        RELEASE.verify_manifest(manifest, self.root)
        return manifest

    def ephemeral_rsa_keypair(self) -> tuple[Path, Path, str]:
        private_key = self.root / "ephemeral-private.pem"
        public_key = self.root / "ephemeral-public.pem"
        subprocess.run(
            [
                "openssl", "genpkey", "-algorithm", "RSA",
                "-pkeyopt", "rsa_keygen_bits:2048", "-out", str(private_key),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        subprocess.run(
            [
                "openssl", "pkey", "-in", str(private_key),
                "-pubout", "-out", str(public_key),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        identity = f"sha256:{RELEASE.sha256_file(public_key)}"
        return private_key, public_key, identity

    def reseal(self, manifest: dict[str, object]) -> None:
        manifest.pop("integritySealSha256", None)
        manifest["integritySealSha256"] = RELEASE.hashlib.sha256(
            RELEASE.canonical_bytes(manifest)
        ).hexdigest()

    def test_create_and_verify_bind_every_identity_layer(self) -> None:
        manifest = self.create()
        RELEASE.verify_manifest(manifest, self.root)
        self.assertEqual(manifest["schema"], "nimbus.release_provenance")
        self.assertEqual(manifest["version"], 2)
        self.assertEqual(manifest["backend"], "cpu")
        self.assertEqual(manifest["backend_device"], "test-cpu")
        self.assertEqual(manifest["source"]["count"], 2)
        self.assertEqual(len(manifest["evidence"]), 1)
        self.assertEqual(len(manifest["artifacts"]), 1)
        self.assertEqual(manifest["environment"]["toolchain"], "test-c++ 1.0")
        self.assertIn("recording_cxx", manifest["environment"])
        self.assertIn("commit", manifest["revision"])
        self.assertEqual(manifest["catalogAuthority"]["authorityFingerprintSha256"], self.catalog_fingerprint)
        self.assertEqual(manifest["reachableSet"]["count"], 2)
        self.assertEqual(
            manifest["gates"][0]["target"], "test-catalog-authority"
        )
        self.assertEqual(manifest["package"]["kind"], "test-bundle")
        self.assertFalse(manifest["eligibility"]["releaseEligible"])
        self.assertFalse(manifest["eligibility"]["gateScopeComplete"])
        self.assertEqual(
            manifest["eligibility"]["reason"],
            "required_release_gates_incomplete,"
            "receipt_v2_source_identity_incomplete,production_package_absent,"
            "external_signature_absent",
        )
        self.assertEqual(
            manifest["eligibility"]["missingPrerequisites"],
            [
                "required_release_gates_incomplete",
                "receipt_v2_source_identity_incomplete",
                "production_package_absent",
                "external_signature_absent",
            ],
        )
        self.assertEqual(manifest["signature"]["status"], "absent")
        self.assertRegex(manifest["integritySealSha256"], r"^[0-9a-f]{64}$")

    def test_changed_input_fails_closed(self) -> None:
        manifest = self.create()
        (self.root / "evidence.json").write_text('{"passed":false}\n')
        with self.assertRaisesRegex(RELEASE.ManifestError, "evidence.*no longer matches"):
            RELEASE.verify_manifest(manifest, self.root)

    def test_added_source_file_fails_closed(self) -> None:
        manifest = self.create()
        (self.root / "source/new.cpp").write_text("int added = 1;\n")
        with self.assertRaisesRegex(RELEASE.ManifestError, "source inventory"):
            RELEASE.verify_manifest(manifest, self.root)

    def test_manifest_mutation_breaks_the_seal(self) -> None:
        manifest = self.create()
        manifest["backend"] = "metal"
        with self.assertRaisesRegex(RELEASE.ManifestError, "seal"):
            RELEASE.verify_manifest(manifest, self.root)

    def test_path_escape_is_rejected_even_with_a_resealed_manifest(self) -> None:
        manifest = self.create()
        manifest["binary"]["path"] = "../nimbus_sim"
        self.reseal(manifest)
        with self.assertRaisesRegex(RELEASE.ManifestError, "escapes"):
            RELEASE.verify_manifest(manifest, self.root)

    def test_empty_evidence_is_rejected(self) -> None:
        arguments = self.arguments()
        arguments["evidence"] = []
        with self.assertRaisesRegex(RELEASE.ManifestError, "evidence"):
            RELEASE.create_manifest(**arguments)

    def test_empty_identity_and_external_sources_are_rejected(self) -> None:
        arguments = self.arguments()
        arguments.pop("device_identity")
        arguments.pop("toolchain_identity")
        with self.assertRaisesRegex(RELEASE.ManifestError, "device identity"):
            RELEASE.create_manifest(
                **arguments, device_identity="", toolchain_identity="test-c++"
            )
        with self.assertRaisesRegex(RELEASE.ManifestError, "toolchain identity"):
            RELEASE.create_manifest(
                **arguments, device_identity="test-cpu", toolchain_identity=""
            )
        with self.assertRaisesRegex(RELEASE.ManifestError, "outside"):
            RELEASE.create_manifest(
                **{**self.arguments(), "sources": [ROOT / "README.md"]}
            )

    def test_symbolic_link_input_is_rejected(self) -> None:
        link = self.root / "evidence-link.json"
        link.symlink_to(self.root / "evidence.json")
        with self.assertRaisesRegex(RELEASE.ManifestError, "symbolic link"):
            RELEASE.create_manifest(**{**self.arguments(), "evidence": [link]})

    def test_generated_cache_and_bytecode_cannot_enter_source_inventory(self) -> None:
        source = self.root / "generated-source"
        cases = (
            "__pycache__/module.cpython-314.pyc",
            "module.pyo",
            ".pytest_cache/state",
            "target/debug/generated.o",
            "node_modules/package/index.js",
            "dist/generated.js",
        )
        for relative in cases:
            with self.subTest(relative=relative):
                shutil.rmtree(source, ignore_errors=True)
                path = source / relative
                path.parent.mkdir(parents=True)
                path.write_bytes(b"mutable generated artifact")
                with self.assertRaisesRegex(
                    RELEASE.ManifestError,
                    "generated cache/build/report output",
                ):
                    RELEASE.inventory_record([source], self.root)

        for root_name in ("build", "reports"):
            with self.subTest(root=root_name):
                generated_root = self.root / root_name
                generated_root.mkdir()
                (generated_root / "artifact.bin").write_bytes(b"artifact")
                with self.assertRaisesRegex(
                    RELEASE.ManifestError,
                    "generated cache/build/report output",
                ):
                    RELEASE.inventory_record([generated_root], self.root)

        for parent, child in (
            ("tests", "artifacts"),
            ("tools/device_benchmark", "results"),
        ):
            with self.subTest(subtree=f"{parent}/{child}"):
                subtree = self.root / parent / child
                subtree.mkdir(parents=True)
                (subtree / "runtime-output.bin").write_bytes(b"output")
                with self.assertRaisesRegex(
                    RELEASE.ManifestError,
                    "generated cache/build/report output",
                ):
                    RELEASE.inventory_record([self.root / parent], self.root)

        maintained = self.root / "maintained-names"
        for relative in ("fixtures/artifacts/reference.bin", "baselines/results/policy.json"):
            path = maintained / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"maintained source")
        inventory = RELEASE.inventory_record([maintained], self.root)
        self.assertEqual(inventory["count"], 2)

    def test_missing_skipped_fallback_and_wrong_binary_gates_reject(self) -> None:
        with self.assertRaisesRegex(RELEASE.ManifestError, "gate"):
            RELEASE.create_manifest(**{**self.arguments(), "gates": []})
        original = json.loads((self.root / "gate.json").read_text())
        for key, value in (
            ("skippedCount", 1),
            ("fallbackUsed", True),
            ("status", "skipped"),
            ("binarySha256", "0" * 64),
        ):
            receipt = dict(original)
            receipt[key] = value
            (self.root / "gate.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(RELEASE.ManifestError, "gate result"):
                self.create()
        (self.root / "gate.json").write_text(json.dumps(original))

    def test_gate_policy_and_receipt_metadata_forgery_reject(self) -> None:
        original = json.loads((self.root / "gate.json").read_text())
        cases = (
            ({"command": ["echo", "pass"]}, "canonical"),
            ({"requiredHardware": True}, "hardware"),
            ({"artifacts": ["gate-artifact.json"]}, "artifact"),
            ({"binary": "package.bin"}, "binary"),
            ({"logSha256": "0" * 64}, "log SHA-256"),
            ({"startedUtc": "not-a-time"}, "startedUtc"),
            ({"completedUtc": "2026-08-28T00:00:00+00:00"}, "timestamps"),
            ({"wallSeconds": -1.0}, "policy|wall"),
        )
        for changes, message in cases:
            with self.subTest(changes=changes):
                forged = {**original, **changes}
                (self.root / "gate.json").write_text(json.dumps(forged))
                with self.assertRaisesRegex(RELEASE.ManifestError, message):
                    self.create()
        (self.root / "gate.json").write_text(json.dumps(original))

    def test_v2_rejects_legacy_reachable_set(self) -> None:
        entries = [{"id": "maintained:0"}]
        legacy = {
            "schema": RELEASE.REACHABLE_SCHEMA,
            "version": 1,
            "scope": "maintained",
            "catalogAuthorityFingerprintSha256": self.catalog_fingerprint,
            "binarySha256": self.binary_sha256,
            "count": 1,
            "entries": entries,
            "reachableSetSha256": RELEASE.hashlib.sha256(
                RELEASE.canonical_bytes(entries)
            ).hexdigest(),
        }
        path = self.root / "reachable-legacy.json"
        path.write_text(json.dumps(legacy))
        with self.assertRaisesRegex(
            RELEASE.ManifestError, "requires rich exact-binary"
        ):
            RELEASE.create_manifest(
                **{**self.arguments(), "reachable_set": path}
            )

    @unittest.skipUnless(shutil.which("openssl"), "openssl is required")
    def test_receipt_v1_remains_unsigned_local_only(self) -> None:
        manifest = RELEASE.create_manifest(**{
            **self.production_arguments(),
            "gates": self.required_gate_paths(version=1),
        })
        self.assertTrue(manifest["eligibility"]["gateScopeComplete"])
        self.assertIn(
            "receipt_v2_source_identity_incomplete",
            manifest["eligibility"]["missingPrerequisites"],
        )
        RELEASE.verify_manifest(manifest, self.root)
        _, public_key, identity = self.ephemeral_rsa_keypair()
        with self.assertRaisesRegex(
            RELEASE.ManifestError, "non-signature release prerequisite"
        ):
            RELEASE.signable_unsigned_candidate(
                manifest,
                self.root,
                public_key=public_key,
                key_identity=identity,
            )

    def test_receipt_v2_revision_and_source_forgery_reject(self) -> None:
        gate_paths = self.required_gate_paths()
        path = gate_paths[0]
        original = json.loads(path.read_text())
        for field, value in (
            ("revision", {**original["revision"], "tree": "0" * 40}),
            ("source", {**original["source"], "sha256": "0" * 64}),
        ):
            with self.subTest(field=field):
                forged = {**original, field: value}
                path.write_text(json.dumps(forged))
                with self.assertRaisesRegex(
                    RELEASE.ManifestError, "revision or source inventory"
                ):
                    RELEASE.create_manifest(**{
                        **self.production_arguments(),
                        "gates": gate_paths,
                    })
                path.write_text(json.dumps(original))

    def test_catalog_reachable_package_and_gate_drift_reject(self) -> None:
        cases = (
            ("catalog-authority.json", {"tampered": True}, "catalog authority"),
            ("reachable.json", {"tampered": True}, "reachable[- ]set"),
            ("package.bin", b"changed-package", "package"),
            ("gate.log", "changed log\n", "gate result"),
        )
        for name, replacement, message in cases:
            manifest = self.create()
            path = self.root / name
            original = path.read_bytes()
            if isinstance(replacement, bytes):
                path.write_bytes(replacement)
            elif isinstance(replacement, str):
                path.write_text(replacement)
            else:
                path.write_text(json.dumps(replacement))
            with self.assertRaisesRegex(RELEASE.ManifestError, message):
                RELEASE.verify_manifest(manifest, self.root)
            path.write_bytes(original)

    def test_zip_and_tar_app_bundles_bind_normalized_members_and_simulator(self) -> None:
        for archive_format in ("zip", "tar"):
            with self.subTest(archive_format=archive_format):
                manifest = RELEASE.create_manifest(
                    **self.production_arguments(archive_format)
                )
                package = manifest["package"]
                archive = package["archive"]
                self.assertEqual(archive["format"], archive_format)
                self.assertEqual(archive["memberCount"], 2)
                self.assertEqual(
                    archive["members"],
                    sorted(archive["members"], key=lambda item: item["path"]),
                )
                self.assertRegex(archive["membersSha256"], r"^[0-9a-f]{64}$")
                self.assertEqual(
                    archive["simulatorMember"]["sha256"], self.binary_sha256
                )
                RELEASE.verify_manifest(manifest, self.root)

    def test_app_bundle_archive_mutation_and_resealed_inventory_forgery_reject(self) -> None:
        arguments = self.production_arguments()
        manifest = RELEASE.create_manifest(**arguments)
        package_path = Path(arguments["package"])
        self.write_zip_bundle(
            package_path,
            [
                (
                    "Nimbus Studio.app/Contents/MacOS/nimbus_sim",
                    (self.root / "nimbus_sim").read_bytes(),
                    stat.S_IFREG | 0o755,
                ),
                (
                    "Nimbus Studio.app/Contents/Resources/index.html",
                    b"mutated\n",
                    stat.S_IFREG | 0o644,
                ),
            ],
        )
        with self.assertRaisesRegex(RELEASE.ManifestError, "package"):
            RELEASE.verify_manifest(manifest, self.root)

        arguments = self.production_arguments()
        manifest = RELEASE.create_manifest(**arguments)
        manifest["package"]["archive"]["members"][0]["sha256"] = "0" * 64
        manifest["package"]["archive"]["membersSha256"] = RELEASE.hashlib.sha256(
            RELEASE.canonical_bytes(manifest["package"]["archive"]["members"])
        ).hexdigest()
        self.reseal(manifest)
        with self.assertRaisesRegex(RELEASE.ManifestError, "package"):
            RELEASE.verify_manifest(manifest, self.root)

    def test_app_bundle_archive_rejects_unsafe_or_unbound_members(self) -> None:
        simulator = (self.root / "nimbus_sim").read_bytes()
        cases = (
            ("empty", []),
            (
                "traversal",
                [("../nimbus_sim", simulator, stat.S_IFREG | 0o755)],
            ),
            (
                "symlink",
                [("Nimbus.app/nimbus_sim", b"target", stat.S_IFLNK | 0o777)],
            ),
            (
                "duplicate",
                [
                    ("Nimbus.app/nimbus_sim", simulator, stat.S_IFREG | 0o755),
                    ("Nimbus.app/nimbus_sim", simulator, stat.S_IFREG | 0o755),
                ],
            ),
            (
                "junk",
                [
                    ("Nimbus.app/nimbus_sim", simulator, stat.S_IFREG | 0o755),
                    ("__MACOSX/.DS_Store", b"junk", stat.S_IFREG | 0o644),
                ],
            ),
            (
                "missing-simulator",
                [("Nimbus.app/index.html", b"page", stat.S_IFREG | 0o644)],
            ),
            (
                "two-simulators",
                [
                    ("Nimbus.app/MacOS/nimbus_sim", simulator, stat.S_IFREG | 0o755),
                    ("Nimbus.app/Resources/nimbus_sim", simulator, stat.S_IFREG | 0o755),
                ],
            ),
            (
                "wrong-simulator",
                [("Nimbus.app/nimbus_sim", b"wrong", stat.S_IFREG | 0o755)],
            ),
        )
        for label, members in cases:
            with self.subTest(label=label):
                package = self.root / f"invalid-{label}.zip"
                self.write_zip_bundle(package, members)
                with self.assertRaises(RELEASE.ManifestError):
                    RELEASE.create_manifest(**{
                        **self.arguments(),
                        "package": package,
                        "package_kind": "app-bundle-archive",
                        "package_platform": "macos",
                        "package_architecture": "arm64",
                    })

        tar_path = self.root / "invalid-link.tar"
        with tarfile.open(tar_path, "w") as archive:
            info = tarfile.TarInfo("Nimbus.app/nimbus_sim")
            info.type = tarfile.SYMTYPE
            info.linkname = "/tmp/nimbus_sim"
            archive.addfile(info)
        with self.assertRaisesRegex(RELEASE.ManifestError, "link"):
            RELEASE.create_manifest(**{
                **self.arguments(),
                "package": tar_path,
                "package_kind": "app-bundle-archive",
                "package_platform": "macos",
                "package_architecture": "arm64",
            })

    def test_catalog_and_reachable_identity_mismatch_rejects_creation(self) -> None:
        reachable = json.loads((self.root / "reachable.json").read_text())
        reachable["catalogAuthorityFingerprintSha256"] = "b" * 64
        (self.root / "reachable.json").write_text(json.dumps(reachable))
        with self.assertRaisesRegex(RELEASE.ManifestError, "reachable set"):
            self.create()

    def test_rich_reachable_set_evidence_schema_is_bound(self) -> None:
        matrices = [
            {"id": "gpu", "evaluated": 960},
            {"id": "boundary", "evaluated": 2592},
            {"id": "collocated", "evaluated": 6480},
            {"id": "cgrid", "evaluated": 1658880},
        ]
        authority = json.loads(
            (self.root / "catalog-authority.json").read_text()
        )
        evidence = {
            "schema": RELEASE.REACHABLE_EVIDENCE_SCHEMA,
            "version": 1,
            "createdUtc": "2026-08-29T00:00:00+00:00",
            "scope": {
                "categoricalExactBinary": True,
                "continuousParametersEnumerated": False,
                "runtimeQualificationUniversal": False,
            },
            "binary": RELEASE.file_record(self.root / "nimbus_sim", self.root),
            "catalogAuthority": RELEASE.file_record(
                self.root / "catalog-authority.json", self.root
            ),
            "catalog": authority,
            "crossProductBinary": RELEASE.file_record(
                self.root / "cross-product-bin", self.root
            ),
            "crossProductSource": RELEASE.file_record(
                self.root / "cross-product.cpp", self.root
            ),
            "matrices": matrices,
            "totalEvaluated": sum(matrix["evaluated"] for matrix in matrices),
            "receipt": {
                "target": "test-catalog-reachable-matrix/cross-product",
                "command": ["cross-product-bin", "[cross-product]"],
                "exitStatus": 0,
                "skipped": 0,
                "fallbackUsed": False,
                "requiredHardware": False,
                "log": RELEASE.file_record(self.root / "reachable.log", self.root),
            },
        }
        evidence["reachableSetSha256"] = RELEASE.hashlib.sha256(
            RELEASE.canonical_bytes({
                "catalogAuthorityFingerprintSha256":
                    authority["authorityFingerprintSha256"],
                "scope": evidence["scope"],
                "matrices": matrices,
                "totalEvaluated": evidence["totalEvaluated"],
            })
        ).hexdigest()
        evidence["integritySealSha256"] = RELEASE.hashlib.sha256(
            RELEASE.canonical_bytes(evidence)
        ).hexdigest()
        path = self.root / "reachable-rich.json"
        path.write_text(json.dumps(evidence))
        manifest = RELEASE.create_manifest(
            **{**self.arguments(), "reachable_set": path}
        )
        self.assertEqual(
            manifest["reachableSet"]["schema"],
            RELEASE.REACHABLE_EVIDENCE_SCHEMA,
        )
        self.assertEqual(manifest["reachableSet"]["count"], 1668912)
        RELEASE.verify_manifest(manifest, self.root)

    def test_duplicate_gate_targets_and_gate_symlinks_reject(self) -> None:
        duplicate = self.root / "gate-duplicate.json"
        duplicate.write_bytes((self.root / "gate.json").read_bytes())
        with self.assertRaisesRegex(RELEASE.ManifestError, "unique"):
            RELEASE.create_manifest(
                **{**self.arguments(), "gates": [self.root / "gate.json", duplicate]}
            )
        link = self.root / "gate-link.json"
        link.symlink_to(self.root / "gate.json")
        with self.assertRaisesRegex(RELEASE.ManifestError, "symbolic link"):
            RELEASE.create_manifest(**{**self.arguments(), "gates": [link]})

    def test_release_eligibility_rejects_absent_or_unverifiable_signature(self) -> None:
        gate_paths = self.required_gate_paths()
        manifest = RELEASE.create_manifest(
            **{**self.production_arguments(), "gates": gate_paths}
        )
        manifest["eligibility"] = {
            "scope": "externally_signed_release",
            "releaseEligible": True,
            "gateScopeComplete": True,
            "missingPrerequisites": [],
            "reason": "external_signature_verified",
        }
        self.reseal(manifest)
        with self.assertRaisesRegex(RELEASE.ManifestError, "detached signature"):
            RELEASE.verify_manifest(manifest, self.root)

        (self.root / "public.pem").write_text("not a public key\n")
        (self.root / "signature.bin").write_bytes(b"not a signature")
        manifest["signature"] = {
            "status": "detached",
            "algorithm": RELEASE.SIGNATURE_ALGORITHM,
            "keyIdentity": f"sha256:{RELEASE.sha256_file(self.root / 'public.pem')}",
            "publicKey": RELEASE.file_record(self.root / "public.pem", self.root),
            "artifact": RELEASE.file_record(self.root / "signature.bin", self.root),
            "signedPayloadSha256": "0" * 64,
        }
        self.reseal(manifest)
        with self.assertRaisesRegex(RELEASE.ManifestError, "pinned"):
            RELEASE.verify_manifest(manifest, self.root)
        with self.assertRaisesRegex(RELEASE.ManifestError, "payload identity|verification"):
            RELEASE.verify_manifest(
                manifest,
                self.root,
                trusted_key_identity=manifest["signature"]["keyIdentity"],
            )

    @unittest.skipUnless(shutil.which("openssl"), "openssl is required")
    def test_external_ephemeral_rsa_cli_roundtrip(self) -> None:
        unsigned = self.signable_unsigned_manifest()
        unsigned_path = self.root / "unsigned.json"
        RELEASE.write_manifest(unsigned_path, unsigned)
        private_key, public_key, identity = self.ephemeral_rsa_keypair()
        payload_path = self.root / "signature-payload.json"
        signature_path = self.root / "signature.bin"
        signed_path = self.root / "signed.json"

        payload_result = subprocess.run(
            [
                sys.executable, str(SCRIPT), "--root", str(self.root),
                "signature-payload", "--manifest", "unsigned.json",
                "--public-key", public_key.name,
                "--key-identity", identity,
                "--output", payload_path.name,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(payload_result.returncode, 0, payload_result.stdout)
        subprocess.run(
            [
                "openssl", "dgst", "-sha256", "-sign", str(private_key),
                "-out", str(signature_path), str(payload_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        attach_result = subprocess.run(
            [
                sys.executable, str(SCRIPT), "--root", str(self.root),
                "attach-signature", "--manifest", "unsigned.json",
                "--public-key", public_key.name,
                "--key-identity", identity,
                "--payload", payload_path.name,
                "--signature", signature_path.name,
                "--output", signed_path.name,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(attach_result.returncode, 0, attach_result.stdout)
        signed = RELEASE.load_manifest(signed_path)
        RELEASE.verify_manifest(
            signed, self.root, trusted_key_identity=identity,
        )
        self.assertTrue(signed["eligibility"]["releaseEligible"])
        self.assertEqual(signed["signature"]["keyIdentity"], identity)
        self.assertEqual(
            signed["signature"]["signedPayloadSha256"],
            RELEASE.hashlib.sha256(payload_path.read_bytes()).hexdigest(),
        )
        with self.assertRaisesRegex(RELEASE.ManifestError, "externally pinned"):
            RELEASE.verify_manifest(
                signed,
                self.root,
                trusted_key_identity="sha256:" + "0" * 64,
            )

    @unittest.skipUnless(shutil.which("openssl"), "openssl is required")
    def test_signature_workflow_rejects_payload_package_source_and_gate_mutation(self) -> None:
        unsigned = self.signable_unsigned_manifest()
        private_key, public_key, identity = self.ephemeral_rsa_keypair()
        _, payload = RELEASE.signable_unsigned_candidate(
            unsigned,
            self.root,
            public_key=public_key,
            key_identity=identity,
        )
        payload_path = self.root / "payload.json"
        signature_path = self.root / "signature.bin"
        RELEASE.write_signature_payload(payload_path, payload)
        subprocess.run(
            [
                "openssl", "dgst", "-sha256", "-sign", str(private_key),
                "-out", str(signature_path), str(payload_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        payload_path.write_bytes(payload + b"\n")
        with self.assertRaisesRegex(RELEASE.ManifestError, "canonical"):
            RELEASE.attach_detached_signature(
                unsigned,
                self.root,
                public_key=public_key,
                key_identity=identity,
                payload_path=payload_path,
                signature_path=signature_path,
            )
        payload_path.write_bytes(payload)
        signed = RELEASE.attach_detached_signature(
            unsigned,
            self.root,
            public_key=public_key,
            key_identity=identity,
            payload_path=payload_path,
            signature_path=signature_path,
        )

        mutation_paths = (
            Path(signed["package"]["file"]["path"]),
            self.root / "source/main.cpp",
            self.root / "gate.log",
            Path(signed["gates"][0]["receipt"]["path"]),
        )
        for path in mutation_paths:
            with self.subTest(path=str(path)):
                resolved = path if path.is_absolute() else self.root / path
                original = resolved.read_bytes()
                resolved.write_bytes(original + b"mutation")
                with self.assertRaises(RELEASE.ManifestError):
                    RELEASE.verify_manifest(
                        signed, self.root, trusted_key_identity=identity,
                    )
                resolved.write_bytes(original)

        forged = json.loads(json.dumps(signed))
        forged["campaign_id"] = "mutated-after-signing"
        self.reseal(forged)
        with self.assertRaisesRegex(
            RELEASE.ManifestError, "payload identity|verification"
        ):
            RELEASE.verify_manifest(
                forged, self.root, trusted_key_identity=identity,
            )

    @unittest.skipUnless(shutil.which("openssl"), "openssl is required")
    def test_signature_payload_refuses_incomplete_unsigned_candidate(self) -> None:
        _, public_key, identity = self.ephemeral_rsa_keypair()
        with self.assertRaisesRegex(
            RELEASE.ManifestError, "non-signature release prerequisite"
        ):
            RELEASE.signable_unsigned_candidate(
                self.create(),
                self.root,
                public_key=public_key,
                key_identity=identity,
            )

    def test_release_eligibility_requires_clean_revision_and_production_package(self) -> None:
        manifest = RELEASE.create_manifest(
            **{**self.arguments(), "gates": self.required_gate_paths()}
        )
        manifest["eligibility"] = {
            "scope": "externally_signed_release",
            "releaseEligible": True,
            "gateScopeComplete": True,
            "missingPrerequisites": [],
            "reason": "external_signature_verified",
        }
        dirty_revision = dict(manifest["revision"])
        dirty_revision["tracked_worktree_clean"] = False
        self.assertFalse(RELEASE.valid_release_revision(dirty_revision))
        self.reseal(manifest)
        with self.assertRaisesRegex(RELEASE.ManifestError, "production package"):
            RELEASE.verify_manifest(manifest, self.root)

    def test_legacy_v1_manifest_remains_verifiable(self) -> None:
        current = self.create()
        legacy = {
            key: current[key]
            for key in (
                "schema", "created_utc", "campaign_id", "backend",
                "backend_device", "binary", "configuration", "source",
                "evidence", "artifacts", "environment", "revision",
            )
        }
        legacy["version"] = 1
        legacy["seal_sha256"] = RELEASE.hashlib.sha256(
            RELEASE.canonical_bytes(legacy)
        ).hexdigest()
        RELEASE.verify_manifest(legacy, self.root)

    def test_atomic_writer_uses_readable_mode_and_round_trips(self) -> None:
        output = self.root / "reports/release.json"
        manifest = self.create()
        RELEASE.write_manifest(output, manifest)
        self.assertEqual(output.stat().st_mode & 0o777, 0o644)
        self.assertEqual(RELEASE.load_manifest(output), json.loads(output.read_text()))
        RELEASE.verify_manifest(RELEASE.load_manifest(output), self.root)


if __name__ == "__main__":
    unittest.main()
