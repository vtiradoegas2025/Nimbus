#!/usr/bin/env python3
"""Run the maintained CPU Thompson subset without issuing a full-catalog claim."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.qualification.catalog_inventory import INVENTORY, configured_backend, maintained_cases
from scripts.tests.catalog.qualify_catalog_bounded_runs import (
    require_bounded_run, requires_moving_reference_window, sha256_file, write_json_atomic,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    qualified, representatives = maintained_cases()
    cases = [path for path in qualified + representatives
             if configured_backend(path) == "cpu"
             and re.search(r"(?m)^microphysics:\s*\n\s+scheme:\s*thompson\s*$", path.read_text())]
    if not cases:
        raise RuntimeError("maintained inventory has no Thompson CPU profiles")
    identities = {path: sha256_file(path) for path in cases}
    binary_hash = sha256_file(binary)
    inventory_hash = sha256_file(INVENTORY)
    source_hashes = {path: sha256_file(path) for path in (
        Path(__file__), ROOT / "scripts/tests/catalog/qualify_catalog_bounded_runs.py",
        ROOT / "scripts/qualification/catalog_inventory.py",
    )}
    started = time.monotonic()
    completed = []
    with tempfile.TemporaryDirectory(prefix="nimbus-thompson-maintained-") as directory:
        work = Path(directory)
        private_binary = work / "nimbus_sim"
        shutil.copy2(binary, private_binary)
        if sha256_file(private_binary) != binary_hash:
            raise RuntimeError("private simulator copy differs from requested binary")

        def run(index, config):
            output = work / f"case-{index:03d}"
            case_started = time.monotonic()
            record = {"config": config.relative_to(ROOT).as_posix(),
                      "configSha256": identities[config], "backend": "cpu",
                      "durationSeconds": 30 if requires_moving_reference_window(config) else 1}
            try:
                require_bounded_run(private_binary, config, output, {}, False)
                record["passed"] = True
            except Exception as error:
                record.update(passed=False, error=str(error))
            finally:
                if output.exists():
                    shutil.rmtree(output)
            record["wallSeconds"] = time.monotonic() - case_started
            return record

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(run, index, path): path for index, path in enumerate(cases)}
            for future in as_completed(futures):
                record = future.result()
                completed.append(record)
                status = "PASS" if record["passed"] else "FAIL"
                print(f"[thompson-maintained] {len(completed)}/{len(cases)} {status} {record['config']}", flush=True)
    if (sha256_file(binary) != binary_hash or sha256_file(INVENTORY) != inventory_hash
            or any(sha256_file(path) != identity for path, identity in identities.items())
            or any(sha256_file(path) != identity for path, identity in source_hashes.items())):
        raise RuntimeError("Thompson qualification inputs changed while running")
    write_json_atomic(args.report, {
        "schema": "nimbus.thompson-maintained-regression.v1",
        "scope": "maintained CPU Thompson subset", "binarySha256": binary_hash,
        "inventorySha256": inventory_hash, "runnerSha256": source_hashes[Path(__file__)],
        "sourceSha256": {path.relative_to(ROOT).as_posix(): identity for path, identity in source_hashes.items()},
        "passed": all(record["passed"] for record in completed),
        "caseCount": len(completed), "workers": 4, "threadsPerRun": 1,
        "wallSeconds": time.monotonic() - started,
        "records": sorted(completed, key=lambda record: record["config"]),
    })
    return 0 if all(record["passed"] for record in completed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
