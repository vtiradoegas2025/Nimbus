#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.qualification.catalog_inventory import INVENTORY, restart_class_scripts
DEFAULT_BINARY = ROOT / "bin" / "nimbus_sim"
CLASS_SCRIPTS = restart_class_scripts()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {
        "path": resolved.relative_to(ROOT).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--log", type=Path, help="Provenance log path within the repository")
    args = parser.parse_args()
    if args.log is not None:
        try:
            args.log.resolve().relative_to(ROOT)
        except ValueError as error:
            raise RuntimeError("restart provenance log must be inside the repository") from error
    source_binary = args.binary.resolve()
    if not source_binary.is_file():
        raise RuntimeError(f"missing simulator: {source_binary}")
    source_sha256 = sha256_file(source_binary)
    runner_path = Path(__file__).resolve()
    runner_sha256 = sha256_file(runner_path)
    scripts = tuple(ROOT / relative for relative in CLASS_SCRIPTS)
    missing = [str(path.relative_to(ROOT)) for path in scripts if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing restart/threading class scripts: {missing}")

    inventory_identity = sha256_file(INVENTORY)
    script_hashes = {script: sha256_file(script) for script in scripts}
    receipts: list[dict[str, object]] = []
    logs: list[str] = []
    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix="nimbus-catalog-restart-threading-"
    ) as temporary:
        pinned_binary = Path(temporary) / "nimbus_sim"
        shutil.copy2(source_binary, pinned_binary)
        if sha256_file(pinned_binary) != source_sha256:
            raise RuntimeError("failed to create an exact simulator snapshot")
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        for script in scripts:
            class_started = time.monotonic()
            result = subprocess.run(
                [sys.executable, str(script), "--binary", str(pinned_binary)],
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            print(result.stdout, end="", flush=True)
            logs.append(f"===== {script.relative_to(ROOT)} =====\n{result.stdout}")
            if result.returncode != 0:
                raise RuntimeError(
                    f"restart/threading class failed: {script.relative_to(ROOT)}"
                )
            if "] PASS" not in result.stdout:
                raise RuntimeError(
                    f"restart/threading class omitted PASS marker: {script.relative_to(ROOT)}"
                )
            receipts.append({
                "script": file_record(script),
                "exitStatus": result.returncode,
                "skipped": 0,
                "elapsedWallSeconds": time.monotonic() - class_started,
            })
        if sha256_file(pinned_binary) != source_sha256:
            raise RuntimeError("pinned simulator changed during qualification")

    if sha256_file(INVENTORY) != inventory_identity:
        raise RuntimeError("restart class inventory changed during execution")
    if sha256_file(source_binary) != source_sha256:
        raise RuntimeError("public simulator changed during restart/threading qualification")
    if sha256_file(runner_path) != runner_sha256:
        raise RuntimeError("restart/threading runner changed during qualification")
    changed_scripts = [
        script.relative_to(ROOT).as_posix()
        for script, identity in script_hashes.items()
        if sha256_file(script) != identity
    ]
    if changed_scripts:
        raise RuntimeError(f"restart/threading class scripts changed during qualification: {changed_scripts}")
    log_record = None
    if args.log is not None:
        log_path = args.log.resolve()
        write_atomic(log_path, "\n".join(logs).encode())
        log_record = file_record(log_path)
    if args.report is not None:
        report = {
            "schema": "nimbus.catalog-restart-threading-evidence.v1",
            "version": 1,
            "status": "pass",
            "completedUtc": datetime.now(timezone.utc).isoformat(),
            "elapsedWallSeconds": time.monotonic() - started,
            "binary": file_record(source_binary),
            "runner": file_record(runner_path),
            "log": log_record,
            "summary": {"classes": len(receipts), "passed": len(receipts), "skipped": 0},
            "classes": receipts,
            "claimBoundary": (
                f"Threading, checkpoint, and continuation evidence for {len(receipts)} maintained state-owner "
                "equivalence classes; not universal generated-tuple execution or a long campaign."
            ),
        }
        write_atomic(
            args.report.resolve(),
            (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(),
        )

    print(
        f"[qualify-catalog-restart-threading] PASS: {len(receipts)} maintained state-owner "
        "equivalence classes satisfied threading, checkpoint, and continuation "
        f"contracts on exact binary sha256={source_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
