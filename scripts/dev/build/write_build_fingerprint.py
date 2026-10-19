#!/usr/bin/env python3
"""Write a reproducible, machine-readable description of a Nimbus build."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone


def key_values(values: list[str]) -> dict[str, str | bool | int]:
    result: dict[str, str | bool | int] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or not key:
            raise ValueError(f"expected KEY=VALUE, got {value!r}")
        normalized = raw.strip()
        if normalized in {"0", "1"}:
            result[key] = normalized == "1"
        else:
            result[key] = normalized
    return result


def command_version(command: list[str]) -> str:
    if not command:
        return ""
    try:
        result = subprocess.run(
            [*command, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    output = result.stdout or result.stderr
    return output.splitlines()[0].strip() if output else ""


def git_value(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def build_definition_digest(root: Path) -> str:
    digest = hashlib.sha256()
    files = [root / "Makefile", *sorted((root / "mk").glob("*.mk"))]
    for path in files:
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifacts(root: Path, values: list[str]) -> dict[str, dict[str, str | bool | int]]:
    result: dict[str, dict[str, str | bool | int]] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name:
            raise ValueError(f"expected NAME=PATH, got {value!r}")
        path = Path(raw_path)
        resolved = path if path.is_absolute() else root / path
        entry: dict[str, str | bool | int] = {
            "path": path.as_posix(),
            "exists": resolved.is_file(),
        }
        if resolved.is_file():
            entry["bytes"] = resolved.stat().st_size
            entry["sha256"] = file_digest(resolved)
        result[name] = entry
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--cache-mode", default="auto")
    parser.add_argument("--cache", default="")
    parser.add_argument("--cxxflags", default="")
    parser.add_argument("--ldflags", default="")
    parser.add_argument("--setting", action="append", default=[])
    parser.add_argument("--dependency", action="append", default=[])
    parser.add_argument("--artifact", action="append", default=[])
    args = parser.parse_args()

    root = args.root.resolve()
    compiler_command = shlex.split(args.compiler)
    compiler_executable = (
        shutil.which(compiler_command[0]) if compiler_command else None
    )
    resolved_compiler = (
        [compiler_executable, *compiler_command[1:]]
        if compiler_executable
        else compiler_command
    )
    cache_executable = shutil.which(args.cache) if args.cache else None
    cache_command = [cache_executable or args.cache] if args.cache else []
    revision = git_value(root, "rev-parse", "HEAD")
    dirty = bool(git_value(root, "status", "--porcelain", "--untracked-files=no"))

    identity = {
        "source": {"revision": revision, "tracked_changes": dirty},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "compiler": {
            "requested": args.compiler,
            "resolved": " ".join(resolved_compiler),
            "version": command_version(resolved_compiler),
        },
        "compiler_cache": {
            "requested": args.cache_mode,
            "resolved": cache_executable or args.cache,
            "version": command_version(cache_command),
        },
        "flags": {"compile": args.cxxflags.strip(), "link": args.ldflags.strip()},
        "settings": key_values(args.setting),
        "dependencies": key_values(args.dependency),
        "artifacts": artifacts(root, args.artifact),
        "build_definition_sha256": build_definition_digest(root),
    }
    canonical_identity = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    document = {
        "schema": "nimbus.build-fingerprint.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint_sha256": hashlib.sha256(canonical_identity.encode()).hexdigest(),
        **identity,
    }

    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, delete=False
    ) as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
