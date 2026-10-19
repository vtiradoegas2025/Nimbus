#!/usr/bin/env python3
"""Check or regenerate the browser fallback from an exact simulator catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.tests.catalog.qualify_catalog_bounded_runs import write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--snapshot", type=Path,
                        default=ROOT / "ui/src/lib/config/nativeCatalogSnapshot.json")
    parser.add_argument("--write", action="store_true", help="regenerate after semantic review")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    snapshot = args.snapshot.resolve()
    catalog = json.loads(subprocess.check_output([str(binary), "--scheme-catalog=json"], cwd=ROOT))
    if catalog.get("schema") != "nimbus.scheme-catalog.v9" or \
            len(catalog.get("modules", [])) != 12 or not catalog.get("readiness", {}).get("items"):
        raise RuntimeError("native catalog is incomplete or changed schema")
    if args.write:
        if snapshot != (ROOT / "ui/src/lib/config/nativeCatalogSnapshot.json").resolve():
            parser.error("write mode accepts only the reviewed browser snapshot path")
        write_json_atomic(snapshot, catalog)
    else:
        if json.loads(snapshot.read_text()) != catalog:
            raise RuntimeError("browser fallback differs from the exact simulator catalog")
    print(f"[browser-catalog-generator] {'wrote' if args.write else 'checked'} "
          f"{len(catalog['readiness']['items'])} readiness items")


if __name__ == "__main__":
    main()
