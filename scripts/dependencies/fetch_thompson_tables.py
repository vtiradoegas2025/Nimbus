#!/usr/bin/env python3
"""Install the existing byte-pinned Thompson inputs without host-dependent regeneration."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("thompson_tables.json")
CHUNK = 1024 * 1024


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(manifest: dict) -> None:
    if not isinstance(manifest, dict) or manifest.get("schema") != "nimbus.thompson-runtime-tables" or manifest.get("version") != 1:
        raise ValueError("Unsupported Thompson table manifest")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Missing Thompson table identities")
    seen = set()
    for entry in [manifest.get("archive", {}), *files]:
        if not isinstance(entry, dict):
            raise ValueError("Invalid Thompson asset identity")
        name = entry.get("name", "")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in seen:
            raise ValueError("Invalid or duplicate Thompson asset name")
        seen.add(name)
        if type(entry.get("bytes")) is not int or not 0 < entry["bytes"] <= 512 * 1024 * 1024:
            raise ValueError("Invalid Thompson asset byte count")
        if not isinstance(entry.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]):
            raise ValueError("Invalid Thompson asset digest")
    if sum(entry["bytes"] for entry in files) > 512 * 1024 * 1024:
        raise ValueError("Thompson manifest exceeds the installation byte limit")
    if urllib.parse.urlsplit(manifest["archive"].get("url", "")).scheme != "https":
        raise ValueError("Thompson download requires HTTPS")


def matches(path: Path, identity: dict) -> bool:
    return path.is_file() and path.stat().st_size == identity["bytes"] and file_hash(path) == identity["sha256"]


def copy_bounded(source, destination: Path, expected_bytes: int) -> None:
    count = 0
    with destination.open("wb") as target:
        for block in iter(lambda: source.read(CHUNK), b""):
            count += len(block)
            if count > expected_bytes:
                raise ValueError(f"Asset exceeds the declared size: {destination.name}")
            target.write(block)
    if count != expected_bytes:
        raise ValueError(f"Asset is truncated: {destination.name}")


def provision(destination: Path, manifest: dict, archive: Path | None = None) -> bool:
    validate_manifest(manifest)
    if all(matches(destination / entry["name"], entry) for entry in manifest["files"]):
        return False
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".thompson-install-", dir=destination) as temporary:
        stage = Path(temporary)
        packed = stage / manifest["archive"]["name"]
        if archive is not None:
            with archive.open("rb") as source:
                copy_bounded(source, packed, manifest["archive"]["bytes"])
        else:
            request = urllib.request.Request(manifest["archive"]["url"], headers={"User-Agent": "Nimbus-table-provisioner/1"})
            with urllib.request.urlopen(request, timeout=60) as source:
                copy_bounded(source, packed, manifest["archive"]["bytes"])
        if not matches(packed, manifest["archive"]):
            raise ValueError("Thompson archive SHA-256 does not match the pinned identity")

        expected = {entry["name"]: entry for entry in manifest["files"]}
        seen = set()
        with tarfile.open(packed, "r|gz") as bundle:
            for member in bundle:
                if member.name not in expected or member.name in seen or not member.isfile():
                    raise ValueError(f"Unexpected Thompson archive member: {member.name}")
                entry = expected[member.name]
                if member.size != entry["bytes"]:
                    raise ValueError(f"Wrong Thompson member size: {member.name}")
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError(f"Unreadable Thompson archive member: {member.name}")
                with source:
                    copy_bounded(source, stage / member.name, entry["bytes"])
                if not matches(stage / member.name, entry):
                    raise ValueError(f"Wrong Thompson member SHA-256: {member.name}")
                seen.add(member.name)
        if seen != expected.keys():
            raise ValueError("Thompson archive is missing required files")
        # Verify the whole set before replacing any installed input. Each file
        # has a fixed canonical identity, so an interrupted publication is retryable.
        for name in sorted(seen):
            target = destination / name
            if not matches(target, expected[name]):
                os.replace(stage / name, target)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "data/thompson")
    parser.add_argument("--archive", type=Path, help="Use a previously downloaded, byte-identical archive offline")
    args = parser.parse_args()
    try:
        manifest = json.loads(MANIFEST.read_text())
        changed = provision(args.destination, manifest, args.archive)
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f"Thompson table installation failed: {error}\n")
    print(f"Thompson tables {'installed and verified' if changed else 'already match pinned identities'}: {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
