#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "bin" / "nimbus_sim"
DEFAULT_SNAPSHOT = ROOT / "bin" / "nimbus_sim_physical_qualification"
TARGETS = (
    "test-agrid-native-ready",
    "test-lin-rrtmg-vulkan-ready",
    "test-lin-rrtmg-metal-ready",
    "test-native-chaos-vulkan-ready",
    "test-native-chaos-metal-ready",
    "test-variable-diffusion-ready",
    "test-native-mapped-terrain-ready",
    "qualify-constant-velocity-vulkan",
    "qualify-stationary-nested-vulkan-runtime",
    "qualify-stationary-nested-vulkan-smagorinsky-runtime",
    "qualify-stationary-nested-vulkan-kessler-smagorinsky-runtime",
    "qualify-stationary-nested-radiation-runtime",
    "qualify-moving-nested-vulkan-public-admission",
    "qualify-moving-nested-metal-public-admission",
    "qualify-moving-nested-vulkan-public-runtime",
    "qualify-moving-nested-metal-public-runtime-smoke",
    "qualify-catalog-backend-operation-observations",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    args = parser.parse_args()
    source = args.binary.resolve()
    snapshot = args.snapshot.resolve()
    if not source.is_file():
        raise RuntimeError(f"missing simulator: {source}")
    source_sha256 = sha256_file(source)
    copy_atomic(source, snapshot)
    if sha256_file(snapshot) != source_sha256:
        raise RuntimeError("physical qualification snapshot does not match source")
    try:
        snapshot_relative = snapshot.relative_to(ROOT).as_posix()
    except ValueError as error:
        raise RuntimeError("physical qualification snapshot must be inside the repository") from error

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for target in TARGETS:
        result = subprocess.run(
            [
                "make",
                f"PHYSICAL_QUALIFICATION_BINARY={snapshot_relative}",
                target,
            ],
            cwd=ROOT,
            env=environment,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"physical qualification target failed: {target}")
        if sha256_file(snapshot) != source_sha256:
            raise RuntimeError(f"physical qualification snapshot changed after {target}")
        if sha256_file(source) != source_sha256:
            raise RuntimeError(f"public simulator changed after {target}")

    print(
        "[qualify-catalog-physical-backends] PASS: 8 maintained accelerator "
        "equivalence classes executed on physical Vulkan/Metal devices with "
        f"fallback prohibited; exact binary sha256={source_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
