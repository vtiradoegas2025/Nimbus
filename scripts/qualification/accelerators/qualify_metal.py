#!/usr/bin/env python3
"""Run and seal the isolated native-Metal physical qualification suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "bin" / "test_metal_compute_backend"
DEFAULT_OUTPUT = ROOT / "reports" / "qualification" / "metal" / "bootstrap-current.json"
DEFAULT_LOG = ROOT / "reports" / "qualification" / "metal" / "bootstrap-current.log"

SUMMARY_RE = re.compile(r"All tests passed \((\d+) assertions in (\d+) test cases\)")
EVIDENCE_RE = re.compile(
    r"^\[(metal(?:-[a-z0-9]+)*(?:-evidence|-continuity|-complete))\]\s+(.+)$"
)
HASH_RE = re.compile(r"^[0-9a-f]{16}$")

REQUIRED_EVIDENCE = {
    "metal-manifest-evidence",
    "metal-chaos-evidence",
    "metal-storage-evidence",
    "metal-workgroup-evidence",
    "metal-submission-evidence",
    "metal-evidence",
    "metal-smagorinsky-evidence",
    "metal-smagorinsky-tendencies-evidence",
    "metal-smagorinsky-fused-evidence",
    "metal-sgs-apply-evidence",
    "metal-diffusion-evidence",
    "metal-batch-evidence",
    "metal-kessler-pointwise-evidence",
    "metal-kessler-sedimentation-evidence",
    "metal-kessler-fused-evidence",
    "metal-lin-evidence",
    "metal-microphysics-apply-evidence",
    "metal-kessler-apply-residency-evidence",
    "metal-advection-xy-evidence",
    "metal-advection-pre-vertical-batch-evidence",
    "metal-tvd-vertical-flux-evidence",
    "metal-advection-post-vertical-batch-evidence",
    "metal-conservative-cgrid-evidence",
    "metal-dynamics-apply-evidence",
    "metal-cgrid-slow-evidence",
    "metal-agrid-evidence",
    "metal-agrid-continuity",
    "metal-agrid-complete",
    "metal-acoustic-cgrid-evidence",
    "metal-acoustic-batch-evidence",
}

SHARED_SEALED_SOURCES = (
    "include/compute/backend_services.hpp",
    "include/compute/compute_backend.hpp",
    "include/compute/compute_kernel_interfaces.hpp",
    "include/compute/compute_residency.hpp",
    "include/compute/dispatch_graph.hpp",
    "include/compute/kernel_abi.hpp",
    "include/compute/kernel_manifest.hpp",
    "include/compute/resident_field_store.hpp",
    "src/compute/kernel_manifest.cpp",
    "src/compute/compute_backend_factory.hpp",
)

METAL_SEALED_SOURCES = tuple(sorted(
    str(path.relative_to(ROOT))
    for directory, pattern in (
        (ROOT / "metal" / "include", "*.hpp"),
        (ROOT / "metal" / "src", "*.mm"),
        (ROOT / "metal" / "src", "*.inc"),
        (ROOT / "metal" / "src", "*.hpp"),
    )
    for path in directory.rglob(pattern)
))

SEALED_SOURCES = SHARED_SEALED_SOURCES + (
    "metal/Makefile",
    "scripts/qualification/accelerators/qualify_metal.py",
) + METAL_SEALED_SOURCES + (
    "tests/metal/test_metal_compute_backend.cpp",
    "tests/metal/test_qualify_metal.py",
)


class QualificationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value):
        return float(value)
    return value


def parse_fields(text: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for token in shlex.split(text):
        if "=" not in token:
            raise QualificationError(f"malformed evidence token: {token!r}")
        key, value = token.split("=", 1)
        if not key or key in fields:
            raise QualificationError(f"invalid or duplicate evidence key: {key!r}")
        fields[key] = scalar(value)
    return fields


def parse_output(output: str) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    summaries = SUMMARY_RE.findall(output)
    if len(summaries) != 1:
        raise QualificationError(f"expected one passing Catch summary, found {len(summaries)}")
    assertions, cases = (int(value) for value in summaries[0])
    if assertions < 75_000 or cases < 43:
        raise QualificationError(
            f"qualification coverage regressed: {assertions} assertions in {cases} cases"
        )

    evidence: dict[str, dict[str, Any]] = {}
    for line in output.splitlines():
        match = EVIDENCE_RE.match(line.strip())
        if not match:
            continue
        label = match.group(1)
        if label in evidence:
            raise QualificationError(f"duplicate evidence record: {label}")
        evidence[label] = parse_fields(match.group(2))

    missing = sorted(REQUIRED_EVIDENCE - evidence.keys())
    if missing:
        raise QualificationError("missing evidence records: " + ", ".join(missing))

    manifest = evidence["metal-manifest-evidence"]
    expected_manifest = {"manifest": 50, "active": 47, "ready": 47, "deferred": 0, "orphan": 3}
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise QualificationError(
                f"manifest evidence {key}={manifest.get(key)!r}, expected {expected}"
            )
    if not HASH_RE.fullmatch(str(manifest.get("canonical_fnv1a64", ""))):
        raise QualificationError("canonical manifest fingerprint is missing or malformed")

    bootstrap = evidence["metal-evidence"]
    if bootstrap.get("unified_memory") is not True:
        raise QualificationError("physical qualification did not report unified memory")
    if bootstrap.get("parity") != "bitwise" or bootstrap.get("parity_cases") != 12:
        raise QualificationError("domain-recycling bitwise parity evidence is incomplete")
    for key in ("device", "os", "toolchain"):
        if not str(bootstrap.get(key, "")).strip():
            raise QualificationError(f"physical identity field is missing: {key}")

    storage = evidence["metal-storage-evidence"]
    if storage.get("shared_bitwise") is not True or storage.get("private_bitwise") is not True:
        raise QualificationError("storage-mode bitwise evidence is incomplete")
    if storage.get("winner") != "unclaimed":
        raise QualificationError("storage benchmark must not make an unqualified winner claim")
    if int(storage.get("samples", 0)) < 7 or int(storage.get("iterations", 0)) < 1:
        raise QualificationError("storage benchmark sample coverage is incomplete")

    workgroup = evidence["metal-workgroup-evidence"]
    if workgroup.get("widths") != "32,64,128" or workgroup.get("all_bitwise") is not True:
        raise QualificationError("workgroup-width parity evidence is incomplete")
    if workgroup.get("winner") != "unclaimed":
        raise QualificationError("workgroup benchmark must not make an unqualified winner claim")
    if int(workgroup.get("samples", 0)) < 7 or int(workgroup.get("iterations", 0)) < 1:
        raise QualificationError("workgroup benchmark sample coverage is incomplete")

    submission = evidence["metal-submission-evidence"]
    if submission.get("depths") != "1,4,16,64" or submission.get("all_bitwise") is not True:
        raise QualificationError("submission-depth parity evidence is incomplete")
    if submission.get("width") != 64 or int(submission.get("samples", 0)) < 7:
        raise QualificationError("submission-depth sample coverage is incomplete")
    if submission.get("winner") != "unclaimed":
        raise QualificationError("submission benchmark must not make an unqualified winner claim")

    for label, fields in evidence.items():
        for key, value in fields.items():
            if "fnv1a64" in key and not HASH_RE.fullmatch(str(value)):
                raise QualificationError(f"malformed source hash {label}.{key}: {value!r}")

    return {"assertions": assertions, "test_cases": cases}, evidence


def build_report(binary: Path, output: str, repo_root: Path = ROOT) -> dict[str, Any]:
    summary, evidence = parse_output(output)
    if not binary.is_file():
        raise QualificationError(f"qualification binary is missing: {binary}")

    sources: dict[str, str] = {}
    for relative in SEALED_SOURCES:
        source = repo_root / relative
        if not source.is_file():
            raise QualificationError(f"sealed source is missing: {source}")
        sources[relative] = sha256_file(source)

    bootstrap = evidence["metal-evidence"]
    manifest = evidence["metal-manifest-evidence"]
    resolved_binary = binary.resolve()
    try:
        binary_identity = resolved_binary.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        binary_identity = str(resolved_binary)
    return {
        "schema": "nimbus.metal-bootstrap-qualification.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "pass",
        "qualification_eligible": False,
        "qualification_scope": "isolated_native_metal_compute",
        "non_qualification_reasons": [
            "This isolated kernel report does not contain the complete public-profile restart, threading, and exact operation receipts.",
            "No sustained same-Metal moving/reference campaign is included.",
            "No Metal performance-superiority claim is evaluated.",
        ],
        "hardware": {
            "device": bootstrap["device"],
            "operating_system": bootstrap["os"],
            "toolchain": bootstrap["toolchain"],
            "unified_memory": bootstrap["unified_memory"],
        },
        "manifest": {
            "canonical_fnv1a64": manifest["canonical_fnv1a64"],
            "kernels": manifest["manifest"],
            "active": manifest["active"],
            "ready": manifest["ready"],
            "deferred": manifest["deferred"],
            "orphan": manifest["orphan"],
        },
        "tests": summary,
        "binary": {"path": binary_identity, "sha256": sha256_file(binary)},
        "sealed_sources_sha256": sources,
        "test_output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "evidence": evidence,
    }


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        path.chmod(0o644)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()

    binary = args.binary.resolve()
    if not binary.is_file():
        raise QualificationError(f"qualification binary is missing: {binary}")
    completed = subprocess.run(
        [str(binary)], cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        raise QualificationError(f"Metal qualification binary exited {completed.returncode}")

    report = build_report(binary, completed.stdout)
    write_atomic(args.log.resolve(), completed.stdout)
    write_atomic(args.output.resolve(), json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "[metal-qualification] PASS: "
        f"{report['tests']['assertions']} assertions, "
        f"{report['manifest']['ready']}/{report['manifest']['active']} active kernels ready; "
        f"report={args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QualificationError as error:
        print(f"[metal-qualification] FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
