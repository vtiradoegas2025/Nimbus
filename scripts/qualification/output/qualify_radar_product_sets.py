#!/usr/bin/env python3
"""Exercise every ordinary radar product subset through real publication."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
BASE_CONFIG = (
    ROOT / "tests/fixtures/configs/qualification/"
    "cgrid_radar_polarimetric_ready.yaml"
)
ZFP_CONFIG = (
    ROOT / "tests/fixtures/configs/qualification/zfp_core_history.yaml"
)
PRODUCTS = (
    "reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv",
)
CORE_FIELDS = {"u", "v", "w", "theta", "p", "rho", "qv", "qr"}


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def product_sets() -> list[tuple[str, ...]]:
    return [
        selection
        for count in range(1, len(PRODUCTS) + 1)
        for selection in combinations(PRODUCTS, count)
    ]


def configured_source(source: str, selection: tuple[str, ...]) -> str:
    old_products = "  products: [zdr, kdp, rhohv]"
    if source.count(old_products) != 1:
        fail("radar qualification fixture has no unique product selection")
    result = source.replace(
        old_products, f"  products: [{', '.join(selection)}]", 1)
    result = result.replace(
        "  preset: radar_polarimetric_ready", "  preset: minimal", 1)
    return result


def catalog_zfp_budgets(binary: Path) -> tuple[dict[str, float], str]:
    result = subprocess.run([str(binary), "--scheme-catalog=json"], cwd=ROOT,
                            capture_output=True, text=True, timeout=30, check=False)
    if result.returncode != 0:
        fail("exact binary did not publish its ZFP field budget policy")
    catalog = json.loads(result.stdout)
    policy = catalog["runtimeCapabilities"]["interactionPolicy"]
    budgets = policy["qualifiedZfpFieldBudgets"]
    units = policy["qualifiedZfpFieldUnits"]
    if (set(policy["qualifiedZfpPresetFields"]["minimal"]) != CORE_FIELDS
            or any(field not in budgets or not math.isfinite(float(budgets[field]))
                   or float(budgets[field]) <= 0.0 for field in CORE_FIELDS)
            or any(field in budgets or field in units for field in PRODUCTS)
            or any(preset in policy["qualifiedZfpPresetIds"] for preset in
                   ("radar_ready", "radar_polarimetric_ready"))):
        fail("exact-binary qualified ZFP policy did not guard radar products")
    readiness = {item["id"]: item for item in catalog["readiness"]["items"]}
    if (readiness["output.zfp_bounded_history"]["state"] != "engineering_incomplete"
            or not readiness["output.zfp_bounded_history"]["blocker"]):
        fail("exact-binary radar ZFP readiness remains falsely Ready")
    return {field: float(budgets[field]) for field in CORE_FIELDS}, hashlib.sha256(result.stdout.encode()).hexdigest()


def with_zfp_output(source: str, budgets: dict[str, float]) -> str:
    zfp = ZFP_CONFIG.read_text(encoding="utf-8")
    begin = zfp.index("  zfp_field_tolerance:\n")
    end = zfp.index("  keep_last_frames:", begin)
    if set(budgets) != CORE_FIELDS:
        fail("exact-binary non-radar ZFP budget set changed")
    if zfp.count("  zfp_field_tolerance:\n") != 1:
        fail("qualified ZFP core tolerance block changed")
    tolerance_block = "  zfp_field_tolerance:\n" + "".join(
        f"    {field}: {budgets[field]:.17g}\n" for field in sorted(budgets)
    )
    zfp = zfp[:begin] + tolerance_block + zfp[end:]
    source_begin = source.index("output:\n")
    source_end = source.index("\nextensions:\n", source_begin)
    zfp_begin = zfp.index("output:\n")
    zfp_end = zfp.index("\nextensions:\n", zfp_begin)
    return source[:source_begin] + zfp[zfp_begin:zfp_end] + source[source_end:]


def run_case(
    binary: Path,
    config: Path,
    output: Path,
    threads: int,
) -> None:
    result = subprocess.run(
        [
            str(binary), "--headless", "--config", str(config),
            "--duration=1", "--write-every=1", "--checkpoint-every=0",
            f"--outdir={output}", f"--threads={threads}",
            "--log-profile=quiet",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    output.with_suffix(".log").write_text(
        result.stdout + "\n" + result.stderr, encoding="utf-8"
    )
    if result.returncode != 0:
        fail(
            f"{config.name} failed with {threads} thread(s)\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def inspect(output: Path, selection: tuple[str, ...]) -> None:
    expected = CORE_FIELDS | set(selection)
    steps = sorted(path for path in output.glob("step_*") if path.is_dir())
    if len(steps) != 2:
        fail(f"{output.name} did not publish initial and one-second frames")
    for step in steps:
        manifest = json.loads((step / "manifest.json").read_text(encoding="utf-8"))
        if set(manifest.get("published_fields", [])) != expected:
            fail(
                f"{output.name}/{step.name} field inventory differs from "
                f"selected products {selection}"
            )
        radar = manifest.get("radar_provenance", {})
        if radar.get("products") != list(selection):
            fail(
                f"{output.name}/{step.name} radar provenance differs from selection: "
                f"expected {list(selection)}, got {radar}"
            )
        for product in PRODUCTS:
            artifact = step / f"{product}.npy"
            if artifact.is_file() != (product in selection):
                fail(
                    f"{output.name}/{step.name} product artifact mismatch for {product}"
                )


def require_identity(one: Path, four: Path) -> None:
    one_files = {
        path.relative_to(one): path
        for path in one.rglob("*")
        if path.is_file() and path.suffix == ".npy"
    }
    four_files = {
        path.relative_to(four): path
        for path in four.rglob("*")
        if path.is_file() and path.suffix == ".npy"
    }
    if one_files.keys() != four_files.keys():
        fail("all-product one/four-thread artifact inventories differ")
    for relative in sorted(one_files):
        if one_files[relative].read_bytes() != four_files[relative].read_bytes():
            fail(f"all-product publication is not thread-identical: {relative}")


def require_zfp_rejection(binary: Path, config: Path, output: Path,
                          selection: tuple[str, ...]) -> dict[str, object]:
    log_hashes = {}
    for mode, command in (
        ("validate_only", [str(binary), "--validate-only", "--config", str(config)]),
        ("headless", [str(binary), "--headless", "--config", str(config),
                      "--duration=1", "--write-every=1", "--checkpoint-every=0",
                      f"--outdir={output}", "--threads=1", "--log-profile=quiet"]),
    ):
        result = subprocess.run(command, cwd=ROOT, text=True,
                                capture_output=True, timeout=60)
        log = output.with_suffix(f".{mode}.log")
        text = result.stdout + "\n" + result.stderr
        log.write_text(text, encoding="utf-8")
        if result.returncode == 0:
            fail(f"qualified ZFP radar selection unexpectedly admitted: {selection}, {mode}")
        for field in selection:
            if f"field '{field}' is not admitted to qualified ZFP" not in text:
                fail(f"qualified ZFP radar rejection lacks keyed {field}: {mode}")
        if output.exists() and any(output.rglob("*")):
            fail(f"qualified ZFP radar rejection published output: {selection}, {mode}")
        log_hashes[mode] = sha256(log)
    return {"selectedProducts": list(selection), "configSha256": sha256(config),
            "logSha256": log_hashes, "noPublishedOutput": True}


def require_non_radar_zfp_control(binary: Path, root: Path) -> dict[str, object]:
    output = root / "non_radar_zfp_control"
    logs = {}
    for mode, command in (
        ("validate_only", [str(binary), "--validate-only", "--config", str(ZFP_CONFIG)]),
        ("headless", [str(binary), "--headless", "--config", str(ZFP_CONFIG),
                      "--duration=2", "--write-every=2", "--checkpoint-every=0",
                      f"--outdir={output}", "--threads=1", "--log-profile=quiet"]),
    ):
        result = subprocess.run(command, cwd=ROOT, text=True,
                                capture_output=True, timeout=60)
        log = root / f"non_radar_zfp_control.{mode}.log"
        log.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
        if result.returncode != 0:
            fail(f"qualified non-radar ZFP control failed: {mode}, {result.stderr}")
        logs[mode] = sha256(log)
    manifests = []
    for step in ("step_000000", "step_000001"):
        manifest_path = output / step / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if set(manifest["published_fields"]) != CORE_FIELDS:
            fail(f"qualified non-radar ZFP control field set differs: {step}")
        if any(not (output / step / f"{field}.zfp3d").is_file()
               for field in CORE_FIELDS):
            fail(f"qualified non-radar ZFP control has an uncompressed substitution: {step}")
        manifests.append({"step": step, "sha256": sha256(manifest_path)})
    validation = output / "validation_summary.json"
    summary = json.loads(validation.read_text(encoding="utf-8"))
    if summary.get("failed") is not False or summary.get("total_violations") != 0:
        fail("qualified non-radar ZFP control has validation violations")
    return {"configSha256": sha256(ZFP_CONFIG), "logSha256": logs,
            "manifests": manifests, "validationSha256": sha256(validation),
            "fieldSet": sorted(CORE_FIELDS), "compressedPayloads": 16}


def require_restart(binary: Path, config: Path, root: Path) -> None:
    parent = root / "all_products_restart_parent"
    parent_result = subprocess.run(
        [
            str(binary), "--headless", "--config", str(config),
            "--duration=2", "--write-every=1", "--checkpoint-every=1",
            f"--outdir={parent}", "--threads=1", "--log-profile=quiet",
        ], cwd=ROOT, text=True, capture_output=True)
    parent.with_suffix(".log").write_text(
        parent_result.stdout + "\n" + parent_result.stderr, encoding="utf-8"
    )
    if parent_result.returncode != 0:
        fail(f"all-product restart parent failed\n{parent_result.stderr}")
    checkpoints = sorted((parent / "checkpoints").glob("*.nimbus"))
    if len(checkpoints) < 2:
        fail("all-product restart parent omitted its one-second checkpoint")
    resumed = root / "all_products_restart_resumed"
    resumed_result = subprocess.run(
        [
            str(binary), "--headless", "--config", str(config),
            f"--resume={checkpoints[0]}", "--duration=2", "--write-every=1",
            "--checkpoint-every=0", f"--outdir={resumed}", "--threads=4",
            "--log-profile=quiet",
        ], cwd=ROOT, text=True, capture_output=True)
    resumed.with_suffix(".log").write_text(
        resumed_result.stdout + "\n" + resumed_result.stderr, encoding="utf-8"
    )
    if resumed_result.returncode != 0:
        fail(f"all-product restart continuation failed\n{resumed_result.stderr}")
    def at_two_seconds(run: Path) -> Path:
        matches = []
        for step in run.glob("step_*"):
            manifest_path = step / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if abs(float(manifest.get("simulation_time_s", -1.0)) - 2.0) < 1.0e-9:
                matches.append(step)
        if len(matches) != 1:
            fail(f"{run.name} has no unique two-second publication")
        return matches[0]

    parent_step = at_two_seconds(parent)
    resumed_step = at_two_seconds(resumed)
    for field in CORE_FIELDS | set(PRODUCTS):
        left = parent_step / f"{field}.npy"
        right = resumed_step / f"{field}.npy"
        if (not left.is_file() or not right.is_file() or
                left.read_bytes() != right.read_bytes()):
            fail(f"all-product restart suffix differs for {field}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file() or not BASE_CONFIG.is_file() or not ZFP_CONFIG.is_file():
        fail("radar product-set qualification inputs are missing")
    if (args.raw_root is None) != (args.report_output is None):
        parser.error("--raw-root and --report-output must be supplied together")
    binary_sha256 = sha256(binary)
    script_sha256 = sha256(Path(__file__))

    source = BASE_CONFIG.read_text(encoding="utf-8")
    budgets, catalog_sha256 = catalog_zfp_budgets(binary)
    selections = product_sets()
    if len(selections) != 31:
        fail("radar product-set enumeration is incomplete")
    if args.raw_root is None:
        context = tempfile.TemporaryDirectory(prefix="nimbus_radar_product_sets_")
    else:
        raw_root = args.raw_root.resolve()
        raw_root.mkdir(parents=True, exist_ok=False)
        context = nullcontext(str(raw_root))
    with context as raw:
        temporary = Path(raw)
        all_one: Path | None = None
        all_four: Path | None = None
        subset_rows = []
        for index, selection in enumerate(selections):
            config = temporary / f"products_{index:02d}.yaml"
            config.write_text(configured_source(source, selection), encoding="utf-8")
            output = temporary / f"products_{index:02d}_t1"
            run_case(binary, config, output, 1)
            inspect(output, selection)
            subset_rows.append({
                "index": index,
                "selectedProducts": list(selection),
                "configSha256": sha256(config),
                "finalManifestSha256": sha256(output / "step_000001/manifest.json"),
            })
            if len(selection) == len(PRODUCTS):
                all_one = output
                all_four = temporary / "all_products_t4"
                run_case(binary, config, all_four, 4)
                inspect(all_four, selection)
        if all_one is None or all_four is None:
            fail("all-product representative was not exercised")
        require_identity(all_one, all_four)
        zfp_rejections = []
        for selection in [(field,) for field in PRODUCTS] + [PRODUCTS]:
            suffix = "all" if len(selection) == len(PRODUCTS) else selection[0]
            zfp_config = temporary / f"zfp_guard_{suffix}.yaml"
            zfp_config.write_text(with_zfp_output(
                configured_source(source, selection), budgets), encoding="utf-8")
            zfp_rejections.append(require_zfp_rejection(
                binary, zfp_config, temporary / f"zfp_guard_{suffix}_output", selection))
        non_radar_zfp_control = require_non_radar_zfp_control(binary, temporary)
        require_restart(binary, temporary / "products_30.yaml", temporary)
        if sha256(binary) != binary_sha256 or sha256(Path(__file__)) != script_sha256:
            fail("radar product gate binary or script changed during execution")
        if args.report_output is None:
            shutil.rmtree(all_four, ignore_errors=True)
        else:
            report = {
                "schema": "nimbus.r4-radar-product-set-cpu-v3",
                "status": "pass_bounded_cpu_product_sets_with_radar_zfp_guard",
                "claimBoundary": "All 31 nonempty selected radar product sets publish exact NPY membership and all-five NPY is one/four-thread/restart byte-identical. Each five singleton and the all-five qualified ZFP selection rejects before publication with keyed native errors. A direct minimal non-radar ZFP control admits and publishes 16 compressed field payloads. Historical radar ZFP publication on b400 remains red decoded-science evidence, not current admission.",
                "binarySha256": binary_sha256,
                "catalogSha256": catalog_sha256,
                "qualifierSha256": script_sha256,
                "baseConfigSha256": sha256(BASE_CONFIG),
                "zfpCoreConfigSha256": sha256(ZFP_CONFIG),
                "qualifiedNonRadarZfpBudgets": budgets,
                "rawRoot": str(temporary),
                "rawFileCount": sum(path.is_file() for path in temporary.rglob("*")),
                "subsets": subset_rows,
                "allFiveThreadByteExact": True,
                "allFiveRestartByteExact": True,
                "qualifiedRadarZfpRejections": zfp_rejections,
                "qualifiedNonRadarZfpControl": non_radar_zfp_control,
            }
            destination = args.report_output.resolve()
            if destination.exists():
                parser.error("use a new report path")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    print(
        "PASS all 31 non-empty radar product subsets publish exactly their "
        "selected fields; the five-product composition is one/four-thread "
        "and restart identical; qualified radar ZFP rejects before publication; "
        f"catalog_sha256={catalog_sha256}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"FAIL: {error}")
        raise SystemExit(1)
