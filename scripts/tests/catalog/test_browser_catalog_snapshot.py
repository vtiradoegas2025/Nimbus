#!/usr/bin/env python3
"""Require the browser fallback catalog to match this simulator's native catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--snapshot", type=Path,
                        default=ROOT / "ui/src/lib/config/nativeCatalogSnapshot.json")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    snapshot = args.snapshot.resolve(strict=True)
    native = json.loads(subprocess.check_output([str(binary), "--scheme-catalog=json"], cwd=ROOT))
    browser = json.loads(snapshot.read_text())
    if native != browser:
        raise RuntimeError("browser fallback catalog differs from current native catalog")
    counts = {}
    for item in native["readiness"]["items"]:
        counts[item["state"]] = counts.get(item["state"], 0) + 1
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    print(f"[browser-catalog-snapshot] PASS: {len(native['modules'])} modules, "
          f"{len(native['readiness']['items'])} readiness items, {counts}, binary={digest}")


if __name__ == "__main__":
    main()
