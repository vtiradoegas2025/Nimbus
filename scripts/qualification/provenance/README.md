# Release provenance

`release_manifest.py` v2 binds one exact binary, strict-v1 configuration,
backend, catalog authority, reachable-set evidence, typed passing gate receipts,
package, source inventory, evidence, output artifacts, revision, toolchain,
hardware, and operating-system identity in an integrity-sealed JSON record.
Local creation is deliberately an unsigned candidate with
`releaseEligible=false`. Its `missingPrerequisites` list and comma-separated
`reason` enumerate every currently absent prerequisite: mandatory receipts, a
production package, a clean tracked revision, and the external signature.
Completing one prerequisite removes only that item; an unsigned local raw-
simulator candidate never becomes release eligible.
The SHA-256 integrity seal is never described as signing. Version-1 records
remain verifiable.

The focused provenance gate covers gate receipts, reachable evidence, release
manifests, package identity, and signing workflow. Passing those tests proves the
local fail-closed machinery described below; it does not provide current-binary
release receipts, construct or launch an installed Studio package, pass the
final documentation/UI/full-suite release gates, or substitute for the
external signing authority.

Inspect the active evidence namespace without modifying it:

```sh
make inspect-release-evidence
```

The inspection reports the exact current binary SHA and classifies every
mandatory receipt and local candidate as `current`, `stale`, `missing`, or
`invalid`. Use `inspect_release_evidence.py --require-current` when a nonzero
exit is required unless every active artifact is current and verifiable.

Create a record with explicit source roots and evidence files:

```sh
python3 scripts/qualification/provenance/release_manifest.py --root . create \
  --binary bin/nimbus_sim \
  --config tests/fixtures/configs/qualification/agrid_flat_cpu_core_ready.yaml \
  --backend cpu \
  --campaign-id agrid-bounded-ready \
  --device-identity 'Apple M4 Max CPU' \
  --toolchain-identity 'Apple clang 21.0.0 release build' \
  --source include \
  --source src \
  --source mk \
  --source Makefile \
  --catalog-authority tests/fixtures/reference/catalog/authority_v1.json \
  --reachable-set reports/pre_beta/reachable-set.json \
  --gate reports/pre_beta/gates/qualify-io.json \
  --package reports/pre_beta/Nimbus-Studio-macos-arm64.tar.zst \
  --package-kind app-bundle-archive \
  --package-platform macos \
  --package-architecture arm64 \
  --evidence reports/qualification/agrid/result.json \
  --artifact reports/qualification/agrid/final.npy \
  --output reports/qualification/agrid/release-provenance.json
```

Verify both the manifest seal and every bound file:

```sh
python3 scripts/qualification/provenance/release_manifest.py --root . verify \
  --manifest reports/qualification/agrid/release-provenance.json
```

A future externally signed record additionally requires
`--trusted-key-identity sha256:<pinned-public-key-sha256>` during verification.
Release eligibility is rejected unless the detached RSA/SHA-256 signature,
signature payload identity, signature artifact, public key, and externally
pinned key identity all verify. Merely inserting a self-declared public key does
not satisfy the contract.

The supported detached-signature workflow never accepts or generates a private
key. First place the externally governed public key under the provenance root,
obtain its independently pinned `sha256:<public-key-file-sha256>` identity, and
emit the exact canonical payload from an unsigned candidate whose only missing
prerequisite is `external_signature_absent`:

```sh
python3 scripts/qualification/provenance/release_manifest.py --root . \
  signature-payload \
  --manifest reports/pre_beta/unsigned-production-candidate.json \
  --public-key release-keys/nimbus-release-public.pem \
  --key-identity sha256:<externally-pinned-public-key-sha256> \
  --output reports/pre_beta/release-signature-payload.json
```

Transfer that payload to the external signing system and produce an RSA-SHA256
detached signature there. After returning only the signature artifact, attach
it to the unchanged unsigned manifest:

```sh
python3 scripts/qualification/provenance/release_manifest.py --root . \
  attach-signature \
  --manifest reports/pre_beta/unsigned-production-candidate.json \
  --public-key release-keys/nimbus-release-public.pem \
  --key-identity sha256:<externally-pinned-public-key-sha256> \
  --payload reports/pre_beta/release-signature-payload.json \
  --signature reports/pre_beta/release-signature.bin \
  --output reports/pre_beta/signed-production-release.json

python3 scripts/qualification/provenance/release_manifest.py --root . verify \
  --manifest reports/pre_beta/signed-production-release.json \
  --trusted-key-identity sha256:<externally-pinned-public-key-sha256>
```

Attachment reconstructs the canonical payload, requires byte identity with the
emitted payload, verifies the detached signature, and only then changes the
eligibility block to `externally_signed_release`. Any intervening source,
package, gate, reachable-evidence, configuration, payload, or manifest mutation
fails. Test-only ephemeral RSA keys exercise this path; no production private
key is generated or retained by Nimbus tooling.

Source directories are sealed as inventories. Adding, removing, or changing a
file under a recorded source root invalidates the record. Inputs must be regular
files within `--root`; symbolic links and path traversal fail closed.
Generated runtime/build material is never source evidence: `__pycache__`, Python
bytecode, dependency trees, compiler targets, distribution/cache directories,
coverage caches, and top-level `build/` or `reports/` roots make inventory
creation and verification fail. Remove those artifacts rather than sealing
machine-local mutable bytes. Directory names that are part of maintained source
paths, such as `scripts/dev/build` and `ui/src/components/reports`, remain valid;
only repository artifact roots and unambiguous generated components are banned.
Two repository-specific runtime-output subtrees are also always forbidden:
`tests/artifacts` and generated run directories. The names `artifacts` and
`results` remain legal elsewhere for maintained fixtures and policies.

Gate receipts use `nimbus.release_gate_result` v2. The receipt runner owns the
canonical policy for every mandatory target: its exact `make` target, physical-
hardware requirement, and exact artifact list. A caller cannot relabel another
command as a required gate. Receipts must report `status=pass`, zero skips,
`fallbackUsed=false`, UTC ordered timestamps consistent with finite nonnegative
wall time, the exact binary path and SHA-256, and a relative log whose recorded
SHA-256 matches its bytes. Artifact paths must match the target policy exactly.
The release manifest independently reapplies the same policy while ingesting
each receipt. Make owns one `RELEASE_SOURCE_ROOTS` list and passes it to both
receipt recording and manifest creation. Receipt v2 records the exact Git commit
and tree, tracked cleanliness/status/diff digests, and the configured source
inventory roots/count/SHA-256 before running the gate, recomputes them afterward,
and fails if any value changes. Manifest verification requires every mandatory
v2 receipt to equal its final revision and source identity. Receipt v1 remains
verifiable only in unsigned local candidates; it adds
`receipt_v2_source_identity_incomplete` and cannot enter the signing workflow.

Version-2 release provenance requires the canonical rich
`nimbus.catalog.reachable-set-evidence` v1 artifact. The earlier entries/count
schema cannot satisfy a version-2 release record.
The mandatory release receipt set is catalog authority, reachable matrix,
Studio/native parity, bounded runs, restart/threading, physical backends, I/O,
and the final release surface. `make qualify-release-seal` records all eight and creates an unsigned
candidate; it cannot produce a release-eligible result without the external
signature contract above. Even with a valid signature, release eligibility is
rejected unless `revision` identifies a clean tracked 40- or 64-hex commit and
the package is a recognized platform package (`app-bundle-archive`/DMG on
macOS, AppImage/DEB/RPM on Linux, or NSIS/MSI on Windows). `raw-simulator` and
`platform=local` remain valid descriptions of local candidates but are never
production-package evidence.

`make qualify-release-final-surface` is the canonical final-surface command.
It requires the complete repository `make test` aggregate plus
`test-release-docs-ui`, which reruns exact catalog/Studio parity, the static
Studio distribution/documentation contract, application and browser TypeScript
checks, UI unit tests, and locked Rust library tests. Its receipt remains bound
to the same clean revision/source identity as every other mandatory gate. The
target may be developed and its constituents run on a dirty tree, but a dirty
execution is not release evidence and must not be recorded or signed.

An `app-bundle-archive` must be a valid ZIP or TAR archive. Provenance records a
path-sorted SHA-256 inventory of every regular member and seals its count and
canonical inventory hash. Absolute, traversing, non-POSIX, duplicate, symlink,
hard-link, special, generated-junk, and empty archives reject. Exactly one
regular member must be named `nimbus_sim`, and its SHA-256 must equal the exact
binary bound by the reachable evidence and every gate receipt. The archive
inventory is reconstructed during verification, so changing either member
bytes or manifest inventory metadata fails closed. This is package-identity
foundation only; package construction, platform signing, installation, and
launch qualification remain separate workflows.
