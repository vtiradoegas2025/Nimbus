#!/usr/bin/env python3
"""Bounded regression of the maintained CPU implicit-diffusion profiles."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.qualification.catalog_inventory import INVENTORY, configured_backend, maintained_cases
from scripts.tests.catalog.qualify_catalog_bounded_runs import require_bounded_run, sha256_file, write_json_atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    qualified, representatives = maintained_cases()
    cases = [p for p in qualified + representatives if configured_backend(p) == "cpu" and
             re.search(r"(?m)^diffusion:\s*\n\s+scheme:\s*implicit\s*$", p.read_text())]
    if not cases:
        raise RuntimeError("no maintained CPU implicit profiles")
    identities = {p: sha256_file(p) for p in [binary, INVENTORY, Path(__file__),
        ROOT / "scripts/qualification/catalog_inventory.py",
        ROOT / "scripts/tests/catalog/qualify_catalog_bounded_runs.py", *cases]}
    records = []
    with tempfile.TemporaryDirectory(prefix="nimbus-implicit-maintained-") as directory:
        work = Path(directory)
        private_binary = work / "nimbus_sim"
        shutil.copy2(binary, private_binary)
        if sha256_file(private_binary) != identities[binary]:
            raise RuntimeError("private simulator copy drifted")

        def qualify(item):
            index, config = item
            output = work / f"case-{index}"
            require_bounded_run(private_binary, config, output, {}, False)
            shutil.rmtree(output)
            return {"config": str(config.relative_to(ROOT)), "sha256": identities[config], "passed": True}

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(qualify, item): item[1] for item in enumerate(cases)}
            for future in as_completed(futures):
                try:
                    records.append(future.result())
                except Exception as error:
                    raise RuntimeError(f"CPU implicit regression failed: {futures[future]}") from error
                if len(records) % 20 == 0 or len(records) == len(cases):
                    print(f"[implicit-maintained] {len(records)}/{len(cases)} passed", flush=True)
    if any(sha256_file(path) != identity for path, identity in identities.items()):
        raise RuntimeError("implicit regression inputs changed during execution")
    write_json_atomic(args.report, {
        "schema": "nimbus.implicit-maintained-regression.v1", "passed": True,
        "binarySha256": identities[binary], "inventorySha256": identities[INVENTORY],
        "runnerSha256": identities[Path(__file__)], "caseCount": len(records), "workers": 4, "threadsPerRun": 1,
        "records": sorted(records, key=lambda r: r["config"]),
        "claimBoundary": "Maintained CPU implicit-diffusion subset; not the whole catalog or long campaigns.",
    })


if __name__ == "__main__":
    main()
