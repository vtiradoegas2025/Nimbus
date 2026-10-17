#!/usr/bin/env python3
"""Bounded stochastic Thompson regression, including private restart state."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

import test_collocated_mapped_terrain_ready as mapped
import test_agrid_chaos_ready as flat_chaos

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualify_flat(binary, output):
    config = flat_chaos.CONFIG
    serial, parallel, resumed = (output / name for name in ("serial", "parallel", "resumed"))
    for directory, threads in ((serial, 1), (parallel, 4)):
        log = flat_chaos.execute(binary, config, directory, threads, False)
        (output / (directory.name + ".log")).write_text(log)
    checkpoint = parallel / "checkpoints/checkpoint_step_000000010.nimbus"
    continuation = flat_chaos.run([
        str(binary), "--headless", "--config", str(config), f"--resume={checkpoint}",
        "--duration=2", "--write-every=1", "--checkpoint-every=1", f"--outdir={resumed}",
        "--threads=4", "--log-profile=normal",
    ])
    (output / "resumed.log").write_text(continuation.stdout)
    flat_chaos.require_success(continuation, "stochastic Thompson RK4 continuation")
    for field in flat_chaos.FIELDS:
        expected = flat_chaos.field_payload(serial, 2, field)
        if expected != flat_chaos.field_payload(parallel, 2, field) or expected != flat_chaos.field_payload(resumed, 1, field):
            raise RuntimeError(f"stochastic Thompson RK4 {field} is not thread/restart identical")
    for filename in flat_chaos.LEDGERS:
        if (serial / filename).read_bytes() != (parallel / filename).read_bytes():
            raise RuntimeError(f"stochastic Thompson RK4 {filename} differs across threads")
    for step in (10, 20):
        name = f"checkpoints/checkpoint_step_{step:09d}.nimbus"
        if (serial / name).read_bytes() != (parallel / name).read_bytes():
            raise RuntimeError(f"stochastic Thompson RK4 private checkpoint {step} differs across threads")
    name = "checkpoints/checkpoint_step_000000020.nimbus"
    if (parallel / name).read_bytes() != (resumed / name).read_bytes():
        raise RuntimeError("stochastic Thompson RK4 private restart state differs")
    mapped.require_restart_suffix(parallel, resumed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError("stochastic evidence directory must be empty")
    cases = [(terrain, advection, mapped.active_config_path(terrain, advection, "full_stochastic", "thompson"))
             for terrain in ("bell", "schar") for advection in mapped.ADVECTION_MODES]
    flat_runner = ROOT / "scripts/tests/science/test_agrid_chaos_ready.py"
    sources = [Path(__file__), Path(mapped.__file__), flat_runner,
               ROOT / "tests/fixtures/configs/qualification/agrid_full_stochastic_ready.yaml",
               *[case[2] for case in cases]]
    identities = {str(path.relative_to(ROOT)): sha(path) for path in sources}
    identity = sha(binary)
    snapshot = output / "nimbus_sim"
    shutil.copy2(binary, snapshot)
    if sha(snapshot) != identity:
        raise RuntimeError("stochastic regression executable copy differs")

    def qualify(case):
        terrain, advection, config = case
        label = terrain + "-" + advection
        try:
            mapped.qualify_active_microphysics_profile(
                snapshot, config, terrain, advection, "full_stochastic", "thompson", output / label, True)
            result = {"passed": True}
        except Exception as error:
            result = {"passed": False, "error": str(error)}
        print(label, "PASS" if result["passed"] else "FAIL", flush=True)
        return label, result

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = dict(executor.map(qualify, cases))
    flat_output = output / "flat-stochastic-rk4"
    flat_output.mkdir()
    try:
        qualify_flat(snapshot, flat_output)
        results["flat-stochastic-rk4"] = {"passed": True}
    except Exception as error:
        results["flat-stochastic-rk4"] = {"passed": False, "error": str(error)}
    if sha(binary) != identity or sha(snapshot) != identity or any(
            sha(ROOT / path) != value for path, value in identities.items()):
        raise RuntimeError("stochastic regression inputs changed during execution")
    passed = all(value["passed"] for value in results.values())
    (output / "summary.json").write_text(json.dumps({
        "status": "bounded_passed" if passed else "failed",
        "binary_sha256": identity, "source_sha256": identities, "results": results,
        "campaign_validated": False, "externally_reviewed": False,
    }, indent=2) + "\n")
    print("stochastic Thompson:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
