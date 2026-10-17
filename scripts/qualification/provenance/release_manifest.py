#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from pathlib import PurePosixPath
import shlex
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Iterable
import unicodedata
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.qualification.provenance.record_gate_result import (  # noqa: E402
    REQUIRED_GATE_POLICY,
    canonical_gate_command,
)


SCHEMA = "nimbus.release_provenance"
VERSION = 2
LEGACY_VERSION = 1
BACKENDS = ("cpu", "vulkan", "metal")
GATE_SCHEMA = "nimbus.release_gate_result"
REACHABLE_SCHEMA = "nimbus.catalog.reachable-set"
REACHABLE_EVIDENCE_SCHEMA = "nimbus.catalog.reachable-set-evidence"
SIGNATURE_ALGORITHM = "rsa-sha256"
REQUIRED_RELEASE_GATES = frozenset(REQUIRED_GATE_POLICY)
PRODUCTION_PACKAGE_PLATFORMS = {
    "app-bundle-archive": frozenset({"macos"}),
    "dmg": frozenset({"macos"}),
    "appimage": frozenset({"linux"}),
    "deb": frozenset({"linux"}),
    "rpm": frozenset({"linux"}),
    "nsis-installer": frozenset({"windows"}),
    "msi": frozenset({"windows"}),
}
GENERATED_SOURCE_COMPONENTS = frozenset({
    "__pycache__",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".vite",
    ".next",
    ".parcel-cache",
    "node_modules",
    "target",
    "dist",
    "htmlcov",
})
GENERATED_SOURCE_ROOTS = frozenset({"build", "reports"})
GENERATED_SOURCE_SUBTREES = frozenset({
    ("tests", "artifacts"),
    ("tools", "device_benchmark", "results"),
})
GENERATED_SOURCE_SUFFIXES = frozenset({".pyc", ".pyo"})
GENERATED_SOURCE_FILES = frozenset({".coverage", ".DS_Store"})
PACKAGE_JUNK_COMPONENTS = frozenset({
    "__MACOSX",
    "__pycache__",
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    "node_modules",
    "target",
})
PACKAGE_JUNK_SUFFIXES = frozenset({".pyc", ".pyo", ".tmp"})
PACKAGE_JUNK_FILES = frozenset({".DS_Store", "Thumbs.db"})


class ManifestError(RuntimeError):
    pass


def valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def checked_relative_file(path: Path, root: Path) -> tuple[Path, str]:
    path = path if path.is_absolute() else root / path
    if path.is_symlink():
        raise ManifestError(f"input is a symbolic link: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ManifestError(f"input is not a regular file: {path}")
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ManifestError(f"input is outside the provenance root: {path}") from exc
    return resolved, relative.as_posix()


def file_record(path: Path, root: Path) -> dict[str, object]:
    resolved, relative = checked_relative_file(path, root)
    return {
        "path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def load_json_file(path: Path, root: Path, label: str) -> tuple[Path, object]:
    resolved, _ = checked_relative_file(path, root)
    try:
        return resolved, json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"{label} is not valid JSON: {path}") from exc


def catalog_authority_record(path: Path, root: Path) -> dict[str, object]:
    resolved, value = load_json_file(path, root, "catalog authority")
    if not isinstance(value, dict):
        raise ManifestError("catalog authority root is not an object")
    fingerprint = value.get("authorityFingerprintSha256")
    if (
        not isinstance(value.get("schema"), str)
        or not isinstance(value.get("catalogVersion"), int)
        or value["catalogVersion"] <= 0
        or not valid_sha256(fingerprint)
        or not isinstance(value.get("moduleCount"), int)
        or value["moduleCount"] <= 0
        or not isinstance(value.get("schemeCount"), int)
        or value["schemeCount"] <= 0
    ):
        raise ManifestError("catalog authority identity is invalid")
    return {
        "file": file_record(resolved, root),
        "schema": value["schema"],
        "catalogVersion": value["catalogVersion"],
        "authorityFingerprintSha256": fingerprint,
        "moduleCount": value["moduleCount"],
        "schemeCount": value["schemeCount"],
    }


def reachable_set_record(
    path: Path,
    root: Path,
    *,
    catalog_fingerprint: str,
    binary_sha256: str,
) -> dict[str, object]:
    resolved, value = load_json_file(path, root, "reachable set")
    if not isinstance(value, dict):
        raise ManifestError("reachable set root is not an object")
    declared_hash = value.get("reachableSetSha256")
    if value.get("schema") == REACHABLE_EVIDENCE_SCHEMA:
        unsigned = dict(value)
        integrity_seal = unsigned.pop("integritySealSha256", None)
        catalog = value.get("catalog")
        binary = value.get("binary")
        matrices = value.get("matrices")
        receipt = value.get("receipt")
        scope = value.get("scope")
        if (
            value.get("version") != 1
            or not valid_sha256(declared_hash)
            or not valid_sha256(integrity_seal)
            or hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
                != integrity_seal
            or not isinstance(catalog, dict)
            or catalog.get("authorityFingerprintSha256") != catalog_fingerprint
            or not isinstance(binary, dict)
            or binary.get("sha256") != binary_sha256
            or not isinstance(matrices, list)
            or not matrices
            or not isinstance(value.get("totalEvaluated"), int)
            or value["totalEvaluated"] <= 0
            or value["totalEvaluated"] != sum(
                int(matrix.get("evaluated", -1))
                for matrix in matrices if isinstance(matrix, dict)
            )
            or not isinstance(scope, dict)
            or scope.get("categoricalExactBinary") is not True
            or not isinstance(receipt, dict)
            or receipt.get("exitStatus") != 0
            or receipt.get("skipped") != 0
            or receipt.get("fallbackUsed") is not False
        ):
            raise ManifestError(
                "reachable set evidence identity, receipt, count, or seal is invalid"
            )
        reachable_identity = {
            "catalogAuthorityFingerprintSha256": catalog_fingerprint,
            "scope": scope,
            "matrices": matrices,
            "totalEvaluated": value["totalEvaluated"],
        }
        if hashlib.sha256(canonical_bytes(reachable_identity)).hexdigest() != declared_hash:
            raise ManifestError("reachable set evidence fingerprint is invalid")
        authority_record = value.get("catalogAuthority")
        if not isinstance(authority_record, dict):
            raise ManifestError("reachable set evidence catalog authority is missing")
        _, authority_value = load_json_file(
            Path(str(authority_record.get("path", ""))),
            root,
            "reachable set catalog authority",
        )
        if catalog != authority_value:
            raise ManifestError(
                "reachable set embedded catalog differs from its authority artifact"
            )
        for key in (
            "binary", "catalogAuthority", "crossProductBinary",
            "crossProductSource",
        ):
            verify_file_record(value.get(key), root, f"reachableSet.{key}")
        verify_file_record(receipt.get("log"), root, "reachableSet.receipt.log")
        return {
            "file": file_record(resolved, root),
            "schema": REACHABLE_EVIDENCE_SCHEMA,
            "version": 1,
            "scope": scope,
            "count": value["totalEvaluated"],
            "reachableSetSha256": declared_hash,
            "catalogAuthorityFingerprintSha256": catalog_fingerprint,
            "binarySha256": binary_sha256,
        }

    raise ManifestError(
        "release provenance v2 requires rich exact-binary reachable-set evidence"
    )


def receipt_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"gate result {label} is missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ManifestError(f"gate result {label} is invalid") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ManifestError(f"gate result {label} must be UTC")
    return parsed


def gate_result_record(
    path: Path,
    root: Path,
    *,
    binary: Path,
    revision: dict[str, object],
    source: dict[str, object],
) -> dict[str, object]:
    resolved, value = load_json_file(path, root, "gate result")
    if not isinstance(value, dict):
        raise ManifestError("gate result root is not an object")
    binary_record = file_record(binary, root)
    binary_sha256 = str(binary_record["sha256"])
    target = value.get("target")
    command = value.get("command")
    log_path = value.get("log")
    artifact_paths = value.get("artifacts", [])
    policy = REQUIRED_GATE_POLICY.get(str(target))
    wall_seconds = value.get("wallSeconds")
    receipt_version = value.get("version")
    if (
        value.get("schema") != GATE_SCHEMA
        or receipt_version not in (1, 2)
        or not isinstance(target, str)
        or not target.strip()
        or not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) and part for part in command)
        or policy is None
        or not canonical_gate_command(target, command)
        or value.get("status") != "pass"
        or value.get("requiredHardware") is not policy["requiredHardware"]
        or not isinstance(value.get("skippedCount"), int)
        or value["skippedCount"] != 0
        or value.get("fallbackUsed") is not False
        or value.get("binary") != binary_record["path"]
        or value.get("binarySha256") != binary_sha256
        or not isinstance(log_path, str)
        or not log_path
        or not valid_sha256(value.get("logSha256"))
        or not isinstance(artifact_paths, list)
        or not all(isinstance(item, str) and item for item in artifact_paths)
        or sorted(artifact_paths) != sorted(str(item) for item in policy["artifacts"])
        or len(set(artifact_paths)) != len(artifact_paths)
        or isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, (int, float))
        or not math.isfinite(wall_seconds)
        or wall_seconds < 0.0
    ):
        raise ManifestError(
            "gate result violates its canonical command, hardware, artifact, log, or binary policy"
        )
    started = receipt_timestamp(value.get("startedUtc"), "startedUtc")
    completed = receipt_timestamp(value.get("completedUtc"), "completedUtc")
    elapsed = (completed - started).total_seconds()
    if elapsed < 0.0 or abs(elapsed - wall_seconds) > max(5.0, 0.05 * max(1.0, wall_seconds)):
        raise ManifestError("gate result timestamps and wall time are inconsistent")
    log_record = file_record(Path(log_path), root)
    if value["logSha256"] != log_record["sha256"]:
        raise ManifestError("gate result log SHA-256 does not match its log")
    expected_source = {
        "roots": source.get("roots"),
        "count": source.get("count"),
        "sha256": source.get("sha256"),
    }
    if receipt_version == 2 and (
        value.get("revision") != revision
        or value.get("source") != expected_source
    ):
        raise ManifestError(
            "gate result revision or source inventory differs from the manifest"
        )
    return {
        "receipt": file_record(resolved, root),
        "target": target,
        "receiptVersion": receipt_version,
        "command": command,
        "status": "pass",
        "requiredHardware": value["requiredHardware"],
        "skippedCount": 0,
        "fallbackUsed": False,
        "startedUtc": value["startedUtc"],
        "completedUtc": value["completedUtc"],
        "wallSeconds": wall_seconds,
        "binary": binary_record,
        "binarySha256": binary_sha256,
        "log": log_record,
        "logSha256": value["logSha256"],
        "artifacts": [file_record(Path(item), root) for item in artifact_paths],
        "revision": value.get("revision") if receipt_version == 2 else None,
        "source": value.get("source") if receipt_version == 2 else None,
    }


def normalized_archive_member_path(raw: str, *, directory: bool) -> str:
    if not isinstance(raw, str) or not raw or "\\" in raw or raw.startswith("/"):
        raise ManifestError("package archive contains an invalid absolute or non-POSIX member path")
    candidate = raw[:-1] if directory and raw.endswith("/") else raw
    normalized = unicodedata.normalize("NFC", candidate)
    parts = normalized.split("/")
    if (
        not normalized
        or any(part in ("", ".", "..") for part in parts)
        or (len(parts[0]) >= 2 and parts[0][1] == ":")
        or PurePosixPath(normalized).is_absolute()
    ):
        raise ManifestError("package archive member path escapes or is not normalized")
    if (
        any(part in PACKAGE_JUNK_COMPONENTS for part in parts)
        or parts[-1] in PACKAGE_JUNK_FILES
        or PurePosixPath(parts[-1]).suffix.lower() in PACKAGE_JUNK_SUFFIXES
    ):
        raise ManifestError(
            f"package archive contains generated junk: {normalized}"
        )
    return "/".join(parts)


def archive_member_record(
    path: str,
    stream: object,
    *,
    declared_size: int,
) -> dict[str, object]:
    digest = hashlib.sha256()
    size = 0
    while True:
        block = stream.read(1024 * 1024)
        if not block:
            break
        size += len(block)
        digest.update(block)
    if size != declared_size:
        raise ManifestError(f"package archive member size is inconsistent: {path}")
    return {"path": path, "bytes": size, "sha256": digest.hexdigest()}


def zip_member_inventory(path: Path) -> tuple[str, list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    names: set[str] = set()
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                directory = info.is_dir()
                name = normalized_archive_member_path(
                    info.filename, directory=directory,
                )
                if name in names:
                    raise ManifestError(
                        f"package archive contains a duplicate member: {name}"
                    )
                names.add(name)
                mode = (info.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(mode)
                if file_type == stat.S_IFLNK:
                    raise ManifestError(
                        f"package archive contains a symbolic link: {name}"
                    )
                if directory:
                    continue
                if file_type not in (0, stat.S_IFREG):
                    raise ManifestError(
                        f"package archive contains a non-regular member: {name}"
                    )
                with archive.open(info, "r") as stream:
                    records.append(archive_member_record(
                        name, stream, declared_size=info.file_size,
                    ))
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        if isinstance(exc, ManifestError):
            raise
        raise ManifestError(f"cannot read ZIP package archive: {path}") from exc
    return "zip", records


def tar_member_inventory(path: Path) -> tuple[str, list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    names: set[str] = set()
    try:
        with tarfile.open(path, mode="r:*") as archive:
            for info in archive.getmembers():
                name = normalized_archive_member_path(
                    info.name, directory=info.isdir(),
                )
                if name in names:
                    raise ManifestError(
                        f"package archive contains a duplicate member: {name}"
                    )
                names.add(name)
                if info.issym() or info.islnk():
                    raise ManifestError(
                        f"package archive contains a symbolic or hard link: {name}"
                    )
                if info.isdir():
                    continue
                if not info.isfile():
                    raise ManifestError(
                        f"package archive contains a non-regular member: {name}"
                    )
                stream = archive.extractfile(info)
                if stream is None:
                    raise ManifestError(
                        f"package archive member cannot be read: {name}"
                    )
                with stream:
                    records.append(archive_member_record(
                        name, stream, declared_size=info.size,
                    ))
    except (OSError, tarfile.TarError) as exc:
        raise ManifestError(f"cannot read TAR package archive: {path}") from exc
    return "tar", records


def archive_inventory(path: Path, binary_sha256: str) -> dict[str, object]:
    if zipfile.is_zipfile(path):
        archive_format, records = zip_member_inventory(path)
    elif tarfile.is_tarfile(path):
        archive_format, records = tar_member_inventory(path)
    else:
        raise ManifestError(
            "app-bundle-archive package is neither a valid ZIP nor TAR archive"
        )
    records.sort(key=lambda record: str(record["path"]))
    if not records:
        raise ManifestError("package archive contains no regular members")
    simulators = [
        record for record in records
        if PurePosixPath(str(record["path"])).name == "nimbus_sim"
    ]
    if len(simulators) != 1:
        raise ManifestError(
            "package archive must contain exactly one bundled nimbus_sim member"
        )
    if simulators[0]["sha256"] != binary_sha256:
        raise ManifestError(
            "bundled nimbus_sim does not match the gated release binary"
        )
    return {
        "format": archive_format,
        "memberCount": len(records),
        "membersSha256": hashlib.sha256(canonical_bytes(records)).hexdigest(),
        "members": records,
        "simulatorMember": simulators[0],
    }


def package_record(
    path: Path,
    root: Path,
    *,
    kind: str,
    platform_name: str,
    architecture: str,
    binary_sha256: str,
) -> dict[str, object]:
    if not all(value.strip() for value in (kind, platform_name, architecture)):
        raise ManifestError("package kind, platform, and architecture must not be empty")
    resolved, _ = checked_relative_file(path, root)
    record: dict[str, object] = {
        "file": file_record(resolved, root),
        "kind": kind,
        "platform": platform_name,
        "architecture": architecture,
    }
    if kind == "app-bundle-archive":
        record["archive"] = archive_inventory(resolved, binary_sha256)
    return record


def reject_generated_source_path(path: Path, root: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ManifestError(f"source is outside the provenance root: {path}") from exc
    if (
        (relative.parts and relative.parts[0] in GENERATED_SOURCE_ROOTS)
        or any(
            relative.parts[:len(prefix)] == prefix
            for prefix in GENERATED_SOURCE_SUBTREES
        )
        or any(part in GENERATED_SOURCE_COMPONENTS for part in relative.parts)
        or path.suffix.lower() in GENERATED_SOURCE_SUFFIXES
        or path.name in GENERATED_SOURCE_FILES
    ):
        raise ManifestError(
            f"source inventory contains generated cache/build/report output: {relative.as_posix()}"
        )


def is_production_package(package: object) -> bool:
    if not isinstance(package, dict):
        return False
    kind = package.get("kind")
    platform_name = package.get("platform")
    architecture = package.get("architecture")
    base_valid = (
        isinstance(kind, str)
        and isinstance(platform_name, str)
        and platform_name in PRODUCTION_PACKAGE_PLATFORMS.get(kind, ())
        and isinstance(architecture, str)
        and bool(architecture.strip())
        and architecture != "local"
    )
    if not base_valid:
        return False
    if kind == "app-bundle-archive":
        archive = package.get("archive")
        return (
            isinstance(archive, dict)
            and archive.get("format") in ("zip", "tar")
            and isinstance(archive.get("members"), list)
            and archive.get("memberCount") == len(archive["members"])
            and valid_sha256(archive.get("membersSha256"))
            and isinstance(archive.get("simulatorMember"), dict)
        )
    return True


def valid_release_revision(revision: object) -> bool:
    if not isinstance(revision, dict):
        return False
    commit = revision.get("commit")
    tree = revision.get("tree")
    return (
        revision.get("tracked_worktree_clean") is True
        and isinstance(commit, str)
        and len(commit) in (40, 64)
        and all(character in "0123456789abcdef" for character in commit)
        and isinstance(tree, str)
        and len(tree) in (40, 64)
        and all(character in "0123456789abcdef" for character in tree)
        and valid_sha256(revision.get("tracked_status_sha256"))
        and valid_sha256(revision.get("tracked_diff_sha256"))
    )


def unsigned_missing_prerequisites(
    *, gate_scope_complete: bool, receipt_v2_complete: bool,
    package: object, revision: object,
) -> list[str]:
    missing: list[str] = []
    if not gate_scope_complete:
        missing.append("required_release_gates_incomplete")
    if not receipt_v2_complete:
        missing.append("receipt_v2_source_identity_incomplete")
    if not is_production_package(package):
        missing.append("production_package_absent")
    if not valid_release_revision(revision):
        missing.append("clean_tracked_revision_absent")
    missing.append("external_signature_absent")
    return missing


def source_files(paths: Iterable[Path], root: Path) -> list[Path]:
    files: dict[str, Path] = {}
    for path in paths:
        path = path if path.is_absolute() else root / path
        if path.is_symlink():
            raise ManifestError(f"source inventory contains a symbolic link: {path}")
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ManifestError(f"source is outside the provenance root: {path}") from exc
        reject_generated_source_path(resolved, root)
        candidates = [resolved] if resolved.is_file() else sorted(resolved.rglob("*"))
        for candidate in candidates:
            if candidate.is_symlink():
                raise ManifestError(f"source inventory contains a symbolic link: {candidate}")
            reject_generated_source_path(candidate, root)
            if candidate.is_file():
                _, relative = checked_relative_file(candidate, root)
                files[relative] = candidate
    if not files:
        raise ManifestError("source inventory is empty")
    return [files[name] for name in sorted(files)]


def source_roots(paths: Iterable[Path], root: Path) -> list[dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for path in paths:
        path = path if path.is_absolute() else root / path
        if path.is_symlink():
            raise ManifestError(f"source inventory contains a symbolic link: {path}")
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ManifestError(f"source is outside the provenance root: {path}") from exc
        reject_generated_source_path(resolved, root)
        if not resolved.is_file() and not resolved.is_dir():
            raise ManifestError(f"source is not a regular file or directory: {path}")
        records[relative] = {
            "path": relative,
            "kind": "file" if resolved.is_file() else "directory",
        }
    return [records[name] for name in sorted(records)]


def command_identity(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    first_line = result.stdout.strip().splitlines()
    return first_line[0] if result.returncode == 0 and first_line else "unavailable"


def git_identity(root: Path) -> dict[str, object]:
    def run(args: list[str]) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return "unavailable"
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    commit = run(["rev-parse", "HEAD"])
    tree = run(["rev-parse", "HEAD^{tree}"])
    status = run(["status", "--porcelain", "--untracked-files=no"])
    diff = run(["diff", "--binary", "--no-ext-diff", "HEAD", "--"])
    return {
        "commit": commit,
        "tree": tree,
        "tracked_worktree_clean": status == "",
        "tracked_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
    }


def inventory_record(paths: Iterable[Path], root: Path) -> dict[str, object]:
    paths = list(paths)
    records = [file_record(path, root) for path in source_files(paths, root)]
    return {
        "roots": source_roots(paths, root),
        "files": records,
        "count": len(records),
        "sha256": hashlib.sha256(canonical_bytes(records)).hexdigest(),
    }


def create_manifest(
    *,
    root: Path,
    binary: Path,
    config: Path,
    backend: str,
    sources: Iterable[Path],
    evidence: Iterable[Path],
    catalog_authority: Path,
    reachable_set: Path,
    gates: Iterable[Path],
    package: Path,
    package_kind: str,
    package_platform: str,
    package_architecture: str,
    artifacts: Iterable[Path] = (),
    campaign_id: str,
    device_identity: str,
    toolchain_identity: str,
) -> dict[str, object]:
    root = root.resolve(strict=True)
    sources = list(sources)
    if backend not in BACKENDS:
        raise ManifestError(f"unsupported backend: {backend}")
    if not campaign_id.strip():
        raise ManifestError("campaign id must not be empty")
    if not device_identity.strip():
        raise ManifestError("device identity must not be empty")
    if not toolchain_identity.strip():
        raise ManifestError("toolchain identity must not be empty")
    evidence_records = [file_record(path, root) for path in evidence]
    if not evidence_records:
        raise ManifestError("at least one bounded-test or campaign evidence file is required")
    binary_record = file_record(binary, root)
    binary_sha256 = str(binary_record["sha256"])
    authority = catalog_authority_record(catalog_authority, root)
    reachable = reachable_set_record(
        reachable_set,
        root,
        catalog_fingerprint=str(authority["authorityFingerprintSha256"]),
        binary_sha256=binary_sha256,
    )
    source = inventory_record(sources, root)
    revision = git_identity(root)
    gate_records = [
        gate_result_record(
            path, root, binary=binary, revision=revision, source=source,
        )
        for path in gates
    ]
    if not gate_records:
        raise ManifestError("at least one typed passing gate result is required")
    targets = [str(record["target"]) for record in gate_records]
    if len(set(targets)) != len(targets):
        raise ManifestError("gate result targets must be unique")
    gate_scope_complete = REQUIRED_RELEASE_GATES.issubset(targets)
    receipt_v2_complete = gate_scope_complete and all(
        record["receiptVersion"] == 2
        for record in gate_records
        if record["target"] in REQUIRED_RELEASE_GATES
    )
    package = package_record(
        package,
        root,
        kind=package_kind,
        platform_name=package_platform,
        architecture=package_architecture,
        binary_sha256=binary_sha256,
    )
    missing_prerequisites = unsigned_missing_prerequisites(
        gate_scope_complete=gate_scope_complete,
        receipt_v2_complete=receipt_v2_complete,
        package=package,
        revision=revision,
    )
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": campaign_id,
        "backend": backend,
        "backend_device": device_identity,
        "binary": binary_record,
        "configuration": file_record(config, root),
        "catalogAuthority": authority,
        "reachableSet": reachable,
        "gates": gate_records,
        "package": package,
        "source": source,
        "evidence": evidence_records,
        "artifacts": [file_record(path, root) for path in artifacts],
        "environment": {
            "os": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unavailable",
            "python": platform.python_version(),
            "toolchain": toolchain_identity,
            "recording_cxx": command_identity(
                [*shlex.split(os.environ.get("CXX", "c++")), "--version"]
            ),
        },
        "revision": revision,
        "eligibility": {
            "scope": "local_release_candidate",
            "releaseEligible": False,
            "gateScopeComplete": gate_scope_complete,
            "missingPrerequisites": missing_prerequisites,
            "reason": ",".join(missing_prerequisites),
        },
        "signature": {
            "status": "absent",
            "algorithm": None,
            "keyIdentity": None,
            "publicKey": None,
            "artifact": None,
            "signedPayloadSha256": None,
        },
    }
    manifest["integritySealSha256"] = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    return manifest


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_signature_payload(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def verify_file_record(record: object, root: Path, label: str) -> None:
    if not isinstance(record, dict):
        raise ManifestError(f"{label} record is not an object")
    path_value = record.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ManifestError(f"{label} path is missing")
    relative = Path(path_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ManifestError(f"{label} path escapes the provenance root")
    actual = file_record(root / relative, root)
    if actual != record:
        raise ManifestError(f"{label} no longer matches its recorded bytes and SHA-256")


def _verify_common_manifest(manifest: dict[str, object], root: Path) -> None:
    root = root.resolve(strict=True)
    if manifest.get("backend") not in BACKENDS:
        raise ManifestError("manifest backend is invalid")
    if (not isinstance(manifest.get("backend_device"), str)
            or not manifest["backend_device"].strip()):
        raise ManifestError("manifest device identity is invalid")
    environment = manifest.get("environment")
    if (not isinstance(environment, dict)
            or not isinstance(environment.get("toolchain"), str)
            or not environment["toolchain"].strip()):
        raise ManifestError("manifest toolchain identity is invalid")
    verify_file_record(manifest.get("binary"), root, "binary")
    verify_file_record(manifest.get("configuration"), root, "configuration")
    for group in ("evidence", "artifacts"):
        records = manifest.get(group)
        if not isinstance(records, list) or (group == "evidence" and not records):
            raise ManifestError(f"manifest {group} inventory is invalid")
        for index, record in enumerate(records):
            verify_file_record(record, root, f"{group}[{index}]")
    source = manifest.get("source")
    if (not isinstance(source, dict)
            or not isinstance(source.get("roots"), list)
            or not isinstance(source.get("files"), list)):
        raise ManifestError("manifest source inventory is invalid")
    records = source["files"]
    if source.get("count") != len(records) or not records:
        raise ManifestError("manifest source count is invalid")
    if source.get("sha256") != hashlib.sha256(canonical_bytes(records)).hexdigest():
        raise ManifestError("manifest source inventory seal is invalid")
    for index, record in enumerate(records):
        verify_file_record(record, root, f"source.files[{index}]")
    roots: list[Path] = []
    for index, record in enumerate(source["roots"]):
        if not isinstance(record, dict):
            raise ManifestError(f"source.roots[{index}] is not an object")
        path_value = record.get("path")
        kind = record.get("kind")
        if not isinstance(path_value, str) or not path_value:
            raise ManifestError(f"source.roots[{index}] path is missing")
        relative = Path(path_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ManifestError(f"source.roots[{index}] path escapes the provenance root")
        path = root / relative
        actual_kind = "file" if path.is_file() else "directory" if path.is_dir() else "missing"
        if path.is_symlink() or kind != actual_kind:
            raise ManifestError(f"source.roots[{index}] type no longer matches")
        roots.append(path)
    current = inventory_record(roots, root)
    if current != source:
        raise ManifestError("source inventory no longer matches the recorded source roots")


def _verify_integrity_seal(manifest: dict[str, object], field: str) -> None:
    recorded = manifest.get(field)
    unsigned = dict(manifest)
    unsigned.pop(field, None)
    expected = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    if recorded != expected:
        raise ManifestError("manifest integrity seal does not match its contents")


def _signature_payload(manifest: dict[str, object]) -> bytes:
    payload = json.loads(json.dumps(manifest))
    payload.pop("integritySealSha256", None)
    signature = payload.get("signature")
    if isinstance(signature, dict):
        signature["artifact"] = None
        signature["signedPayloadSha256"] = None
    return canonical_bytes(payload)


def pinned_public_key_record(
    public_key: Path,
    root: Path,
    key_identity: str,
) -> dict[str, object]:
    record = file_record(public_key, root)
    expected = f"sha256:{record['sha256']}"
    if key_identity != expected:
        raise ManifestError(
            "supplied signing-key identity is not pinned to the public key"
        )
    return record


def signable_unsigned_candidate(
    manifest: object,
    root: Path,
    *,
    public_key: Path,
    key_identity: str,
) -> tuple[dict[str, object], bytes]:
    verify_manifest(manifest, root)
    if not isinstance(manifest, dict):
        raise ManifestError("unsigned signing candidate is invalid")
    eligibility = manifest.get("eligibility")
    signature = manifest.get("signature")
    if (
        not isinstance(eligibility, dict)
        or eligibility.get("releaseEligible") is not False
        or eligibility.get("gateScopeComplete") is not True
        or eligibility.get("missingPrerequisites") != [
            "external_signature_absent"
        ]
        or eligibility.get("reason") != "external_signature_absent"
        or signature != {
            "status": "absent",
            "algorithm": None,
            "keyIdentity": None,
            "publicKey": None,
            "artifact": None,
            "signedPayloadSha256": None,
        }
    ):
        raise ManifestError(
            "signature payload requires every non-signature release prerequisite"
        )
    public_key_record = pinned_public_key_record(
        public_key, root, key_identity,
    )
    candidate = json.loads(json.dumps(manifest))
    candidate.pop("integritySealSha256", None)
    candidate["eligibility"] = {
        "scope": "externally_signed_release",
        "releaseEligible": True,
        "gateScopeComplete": True,
        "missingPrerequisites": [],
        "reason": "external_signature_verified",
    }
    candidate["signature"] = {
        "status": "detached",
        "algorithm": SIGNATURE_ALGORITHM,
        "keyIdentity": key_identity,
        "publicKey": public_key_record,
        "artifact": None,
        "signedPayloadSha256": None,
    }
    payload = _signature_payload(candidate)
    return candidate, payload


def attach_detached_signature(
    manifest: object,
    root: Path,
    *,
    public_key: Path,
    key_identity: str,
    payload_path: Path,
    signature_path: Path,
) -> dict[str, object]:
    root = root.resolve(strict=True)
    candidate, expected_payload = signable_unsigned_candidate(
        manifest,
        root,
        public_key=public_key,
        key_identity=key_identity,
    )
    resolved_payload, _ = checked_relative_file(payload_path, root)
    if resolved_payload.read_bytes() != expected_payload:
        raise ManifestError(
            "detached-signature payload differs from the canonical manifest payload"
        )
    signature_record = file_record(signature_path, root)
    candidate["signature"]["artifact"] = signature_record
    candidate["signature"]["signedPayloadSha256"] = hashlib.sha256(
        expected_payload
    ).hexdigest()
    candidate["integritySealSha256"] = hashlib.sha256(
        canonical_bytes(candidate)
    ).hexdigest()
    verify_manifest(candidate, root, trusted_key_identity=key_identity)
    return candidate


def _verify_detached_signature(
    manifest: dict[str, object],
    root: Path,
    signature: dict[str, object],
    trusted_key_identity: str | None,
) -> None:
    if (
        signature.get("status") != "detached"
        or signature.get("algorithm") != SIGNATURE_ALGORITHM
        or not isinstance(signature.get("keyIdentity"), str)
        or not isinstance(signature.get("publicKey"), dict)
        or not isinstance(signature.get("artifact"), dict)
        or not valid_sha256(signature.get("signedPayloadSha256"))
    ):
        raise ManifestError("release eligibility requires a complete detached signature")
    if (
        not isinstance(trusted_key_identity, str)
        or not trusted_key_identity.startswith("sha256:")
        or signature.get("keyIdentity") != trusted_key_identity
    ):
        raise ManifestError(
            "release eligibility requires an externally pinned signing-key identity"
        )
    verify_file_record(signature["publicKey"], root, "signature public key")
    verify_file_record(signature["artifact"], root, "signature artifact")
    public_key = root / str(signature["publicKey"]["path"])
    artifact = root / str(signature["artifact"]["path"])
    key_hash = sha256_file(public_key)
    if signature["keyIdentity"] != f"sha256:{key_hash}":
        raise ManifestError("detached signature key identity is not pinned to its public key")
    payload = _signature_payload(manifest)
    if hashlib.sha256(payload).hexdigest() != signature["signedPayloadSha256"]:
        raise ManifestError("detached signature payload identity is invalid")
    descriptor, payload_path = tempfile.mkstemp(prefix="nimbus-release-signature-", suffix=".json")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        try:
            result = subprocess.run(
                [
                    "openssl", "dgst", "-sha256", "-verify", str(public_key),
                    "-signature", str(artifact), payload_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ManifestError("detached signature verifier is unavailable") from exc
        if result.returncode != 0:
            raise ManifestError("detached release signature verification failed")
    finally:
        try:
            os.unlink(payload_path)
        except FileNotFoundError:
            pass


def _verify_manifest_v1(manifest: dict[str, object], root: Path) -> None:
    _verify_integrity_seal(manifest, "seal_sha256")
    _verify_common_manifest(manifest, root)


def _verify_manifest_v2(
    manifest: dict[str, object],
    root: Path,
    trusted_key_identity: str | None,
) -> None:
    _verify_integrity_seal(manifest, "integritySealSha256")
    _verify_common_manifest(manifest, root)
    binary = manifest.get("binary")
    if not isinstance(binary, dict) or not valid_sha256(binary.get("sha256")):
        raise ManifestError("manifest binary identity is invalid")
    binary_sha256 = str(binary["sha256"])

    authority = manifest.get("catalogAuthority")
    if not isinstance(authority, dict) or not isinstance(authority.get("file"), dict):
        raise ManifestError("manifest catalog authority is missing")
    expected_authority = catalog_authority_record(
        Path(str(authority["file"].get("path", ""))), root
    )
    if authority != expected_authority:
        raise ManifestError("manifest catalog authority no longer matches")

    reachable = manifest.get("reachableSet")
    if not isinstance(reachable, dict) or not isinstance(reachable.get("file"), dict):
        raise ManifestError("manifest reachable set is missing")
    expected_reachable = reachable_set_record(
        Path(str(reachable["file"].get("path", ""))),
        root,
        catalog_fingerprint=str(authority["authorityFingerprintSha256"]),
        binary_sha256=binary_sha256,
    )
    if reachable != expected_reachable:
        raise ManifestError("manifest reachable set no longer matches")

    gates = manifest.get("gates")
    revision = manifest.get("revision")
    source = manifest.get("source")
    if not isinstance(revision, dict) or not isinstance(source, dict):
        raise ManifestError("manifest revision/source identity is missing")
    if not isinstance(gates, list) or not gates:
        raise ManifestError("manifest gate inventory is missing or empty")
    targets: set[str] = set()
    for index, gate in enumerate(gates):
        if not isinstance(gate, dict) or not isinstance(gate.get("receipt"), dict):
            raise ManifestError(f"manifest gates[{index}] is invalid")
        expected_gate = gate_result_record(
            Path(str(gate["receipt"].get("path", ""))),
            root,
            binary=Path(str(binary["path"])),
            revision=revision,
            source=source,
        )
        if gate != expected_gate:
            raise ManifestError(f"manifest gates[{index}] no longer matches")
        target = str(gate["target"])
        if target in targets:
            raise ManifestError("manifest gate targets are not unique")
        targets.add(target)

    package = manifest.get("package")
    if not isinstance(package, dict) or not isinstance(package.get("file"), dict):
        raise ManifestError("manifest package identity is missing")
    expected_package = package_record(
        Path(str(package["file"].get("path", ""))),
        root,
        kind=str(package.get("kind", "")),
        platform_name=str(package.get("platform", "")),
        architecture=str(package.get("architecture", "")),
        binary_sha256=binary_sha256,
    )
    if package != expected_package:
        raise ManifestError("manifest package no longer matches")

    eligibility = manifest.get("eligibility")
    signature = manifest.get("signature")
    if not isinstance(eligibility, dict) or not isinstance(signature, dict):
        raise ManifestError("manifest eligibility/signature block is missing")
    release_eligible = eligibility.get("releaseEligible")
    gate_scope_complete = REQUIRED_RELEASE_GATES.issubset(targets)
    receipt_v2_complete = gate_scope_complete and all(
        gate.get("receiptVersion") == 2
        for gate in gates
        if gate.get("target") in REQUIRED_RELEASE_GATES
    )
    if release_eligible is False:
        missing_prerequisites = unsigned_missing_prerequisites(
            gate_scope_complete=gate_scope_complete,
            receipt_v2_complete=receipt_v2_complete,
            package=package,
            revision=revision,
        )
        if (
            eligibility.get("scope") != "local_release_candidate"
            or eligibility.get("gateScopeComplete") is not gate_scope_complete
            or eligibility.get("missingPrerequisites") != missing_prerequisites
            or eligibility.get("reason") != ",".join(missing_prerequisites)
            or signature != {
                "status": "absent",
                "algorithm": None,
                "keyIdentity": None,
                "publicKey": None,
                "artifact": None,
                "signedPayloadSha256": None,
            }
        ):
            raise ManifestError("unsigned local candidate eligibility is invalid")
    elif release_eligible is True:
        if (
            eligibility.get("scope") != "externally_signed_release"
            or eligibility.get("gateScopeComplete") is not True
            or eligibility.get("missingPrerequisites") != []
            or eligibility.get("reason") != "external_signature_verified"
        ):
            raise ManifestError("release-eligible scope is invalid")
        if not gate_scope_complete:
            raise ManifestError(
                "release eligibility requires every mandatory release gate"
            )
        if not receipt_v2_complete:
            raise ManifestError(
                "release eligibility requires receipt-v2 source identity for every mandatory gate"
            )
        if not valid_release_revision(revision):
            raise ManifestError(
                "release eligibility requires a clean tracked revision"
            )
        if not is_production_package(package):
            raise ManifestError(
                "release eligibility requires an actual production package"
            )
        _verify_detached_signature(
            manifest, root, signature, trusted_key_identity
        )
    else:
        raise ManifestError("release eligibility must be boolean")


def verify_manifest(
    manifest: object,
    root: Path,
    trusted_key_identity: str | None = None,
) -> None:
    root = root.resolve(strict=True)
    if not isinstance(manifest, dict):
        raise ManifestError("manifest root is not an object")
    if manifest.get("schema") != SCHEMA:
        raise ManifestError("manifest schema is unsupported")
    version = manifest.get("version")
    if version == LEGACY_VERSION:
        _verify_manifest_v1(manifest, root)
    elif version == VERSION:
        _verify_manifest_v2(manifest, root, trusted_key_identity)
    else:
        raise ManifestError("manifest version is unsupported")


def load_manifest(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--binary", type=Path, required=True)
    create.add_argument("--config", type=Path, required=True)
    create.add_argument("--backend", choices=BACKENDS, required=True)
    create.add_argument("--campaign-id", required=True)
    create.add_argument("--device-identity", required=True)
    create.add_argument("--toolchain-identity", required=True)
    create.add_argument("--source", action="append", type=Path, required=True)
    create.add_argument("--evidence", action="append", type=Path, required=True)
    create.add_argument("--catalog-authority", type=Path, required=True)
    create.add_argument("--reachable-set", type=Path, required=True)
    create.add_argument("--gate", action="append", type=Path, required=True)
    create.add_argument("--package", type=Path, required=True)
    create.add_argument("--package-kind", required=True)
    create.add_argument("--package-platform", required=True)
    create.add_argument("--package-architecture", required=True)
    create.add_argument("--artifact", action="append", type=Path, default=[])
    create.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--trusted-key-identity")
    signature_payload = subparsers.add_parser("signature-payload")
    signature_payload.add_argument("--manifest", type=Path, required=True)
    signature_payload.add_argument("--public-key", type=Path, required=True)
    signature_payload.add_argument("--key-identity", required=True)
    signature_payload.add_argument("--output", type=Path, required=True)
    attach_signature = subparsers.add_parser("attach-signature")
    attach_signature.add_argument("--manifest", type=Path, required=True)
    attach_signature.add_argument("--public-key", type=Path, required=True)
    attach_signature.add_argument("--key-identity", required=True)
    attach_signature.add_argument("--payload", type=Path, required=True)
    attach_signature.add_argument("--signature", type=Path, required=True)
    attach_signature.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "create":
            result = create_manifest(
                root=args.root,
                binary=args.binary,
                config=args.config,
                backend=args.backend,
                sources=args.source,
                evidence=args.evidence,
                catalog_authority=args.catalog_authority,
                reachable_set=args.reachable_set,
                gates=args.gate,
                package=args.package,
                package_kind=args.package_kind,
                package_platform=args.package_platform,
                package_architecture=args.package_architecture,
                artifacts=args.artifact,
                campaign_id=args.campaign_id,
                device_identity=args.device_identity,
                toolchain_identity=args.toolchain_identity,
            )
            write_manifest(args.output, result)
            verify_manifest(load_manifest(args.output), args.root)
            print(f"[release-provenance] PASS create {args.output}")
        elif args.command == "verify":
            verify_manifest(
                load_manifest(args.manifest),
                args.root,
                trusted_key_identity=args.trusted_key_identity,
            )
            print(f"[release-provenance] PASS verify {args.manifest}")
        elif args.command == "signature-payload":
            manifest_path = (
                args.manifest if args.manifest.is_absolute()
                else args.root / args.manifest
            )
            _, payload = signable_unsigned_candidate(
                load_manifest(manifest_path),
                args.root,
                public_key=args.public_key,
                key_identity=args.key_identity,
            )
            output = (
                args.output if args.output.is_absolute()
                else args.root / args.output
            )
            write_signature_payload(output, payload)
            print(f"[release-provenance] PASS signature-payload {output}")
        else:
            manifest_path = (
                args.manifest if args.manifest.is_absolute()
                else args.root / args.manifest
            )
            signed = attach_detached_signature(
                load_manifest(manifest_path),
                args.root,
                public_key=args.public_key,
                key_identity=args.key_identity,
                payload_path=args.payload,
                signature_path=args.signature,
            )
            output = (
                args.output if args.output.is_absolute()
                else args.root / args.output
            )
            write_manifest(output, signed)
            print(f"[release-provenance] PASS attach-signature {output}")
    except (ManifestError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
