#!/usr/bin/env python3
"""Qualify the ordinary bounded-error ZFP history path end to end."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from multiresolution_contract import inspect_durable_hierarchies


ROOT = Path(__file__).resolve().parents[3]
BINARY = ROOT / "bin" / ("nimbus_sim.exe" if os.name == "nt" else "nimbus_sim")
FIXTURE = ROOT / "tests" / "fixtures" / "configs" / "qualification" / "zfp_core_history.yaml"
MAPPED_FIXTURE = ROOT / "tests" / "fixtures" / "configs" / "qualification" / "cgrid_terrain_smagorinsky_coupled.yaml"
RADAR_FIXTURES = {
    "radar_ready": ROOT / "tests" / "fixtures" / "configs" / "qualification" / "cgrid_radar_ready.yaml",
    "radar_polarimetric_ready": ROOT / "tests" / "fixtures" / "configs" / "qualification" / "cgrid_radar_polarimetric_ready.yaml",
}
NON_RADAR_PRESETS = {
    "minimal", "presentation", "studio_full", "dynamics_full",
    "science_validation",
}
QUALIFIED_PRESETS = NON_RADAR_PRESETS
RADAR_PRODUCTS = {
    "radar_ready": ("reflectivity_dbz", "radial_velocity"),
    "radar_polarimetric_ready": ("zdr", "kdp", "rhohv"),
}
RADAR_READER = ROOT / "bin/test_core_radar_zfp_v4_integration"
LOSSLESS_MISSING_FIELDS = {
    "k_index", "lifted_index", "showalter_index", "total_totals",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                            capture_output=True, timeout=180)
    if result.returncode != 0:
        fail(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def run_simulator(config: Path, output: Path, threads: int, duration: int = 2) -> None:
    run(
        [
            str(BINARY),
            "--headless",
            "--config",
            str(config),
            f"--duration={duration}",
            "--write-every=2",
            f"--outdir={output}",
            f"--threads={threads}",
            "--log-profile=quiet",
        ]
    )


def inspect_publications(
    run_dir: Path,
    policy: str,
    workers: int,
    fields: set[str],
    budgets: dict[str, float],
) -> None:
    steps = sorted(path for path in run_dir.glob("step_*") if path.is_dir())
    if len(steps) != 2:
        fail(f"expected two ZFP publications, found {[path.name for path in steps]}")
    for index, step in enumerate(steps):
        manifest = json.loads((step / "manifest.json").read_text(encoding="utf-8"))
        data_product = manifest.get("data_product", {})
        if data_product.get("field_execution_policy") != policy:
            fail(f"{step.name} omitted ZFP field execution provenance")
        if data_product.get("field_workers") != workers:
            fail(f"{step.name} worker provenance is not {workers}")
        if manifest.get("published_fields") != sorted(fields):
            fail(
                f"{run_dir.name}/{step.name} did not publish the exact qualified field set: "
                f"expected {sorted(fields)}, got {manifest.get('published_fields')}"
            )
        catalog = {entry["field_id"]: entry for entry in manifest["fields"]}
        for field in fields:
            metadata = catalog[field]
            compression = metadata.get("compression")
            if compression is None:
                if field not in LOSSLESS_MISSING_FIELDS or \
                        metadata.get("file_pattern") != f"{field}.npy" or \
                        metadata.get("storage_format") != "npy_2d" or \
                        metadata.get("missing_value") != "NaN":
                    fail(f"{step.name}/{field} has an undeclared ZFP fallback")
                artifact = step / f"{field}.npy"
                if not artifact.is_file() or artifact.stat().st_size <= 16 or \
                        (step / f"{field}.zfp3d").exists():
                    fail(f"{step.name}/{field} lossless missing-value payload is incomplete")
                continue
            if metadata.get("file_pattern") != f"{field}.zfp3d" or \
                    metadata.get("storage_format") is not None:
                fail(f"{step.name}/{field} compressed artifact metadata is inconsistent")
            expected_role = "keyframe" if index == 0 else "simple_delta"
            expected_parents = [] if index == 0 else [0]
            expected = {
                "codec": "zfp",
                "mode": "accuracy",
                "mode_parameter": budgets[field],
                "error_contract": "zfp_absolute_accuracy_before_prefilters",
                "frame_role": expected_role,
                "parent_step_indices": expected_parents,
                "sparse_threshold": 0,
                "float16_prefilter": False,
                "decoded_dtype": "float32",
                "container_version": 3,
                "spatial_layout": "independently_decodable_bricks",
                "brick_shape": [64, 64, 32],
            }
            for key, value in expected.items():
                if compression.get(key) != value:
                    fail(
                        f"{step.name}/{field} compression {key}: "
                        f"expected {value!r}, got {compression.get(key)!r}"
                    )
            artifact = step / f"{field}.zfp3d"
            if not artifact.is_file() or artifact.stat().st_size <= 44:
                fail(f"missing non-empty real ZFP artifact {artifact}")
    inspect_durable_hierarchies(
        run_dir, expected_format="zfp", require_lod=False
    )


def require_artifact_identity(one_thread: Path, four_threads: Path) -> None:
    def payloads(root: Path) -> dict[Path, Path]:
        return {path.relative_to(root): path for path in root.glob("step_*/*")
                if path.is_file() and path.suffix in {".npy", ".zfp3d"} and
                ".lod" not in path.stem}

    one = payloads(one_thread)
    four = payloads(four_threads)
    if one.keys() != four.keys():
        fail("one-thread and four-thread ZFP artifact sets differ")
    for relative in sorted(one):
        if one[relative].read_bytes() != four[relative].read_bytes():
            fail(f"ZFP publication is not byte deterministic across threads: {relative}")


def published_fields(run_dir: Path) -> set[str]:
    publications = sorted(path for path in run_dir.glob("step_*") if path.is_dir())
    if len(publications) != 2:
        fail(f"{run_dir.name} has no two-publication field reference")
    inventories = {
        tuple(json.loads((step / "manifest.json").read_text(encoding="utf-8"))["published_fields"])
        for step in publications
    }
    if len(inventories) != 1:
        fail(f"{run_dir.name} changed its published field inventory between frames")
    return set(next(iter(inventories)))


def require_mapped_geometry_contract(run_dir: Path) -> None:
    identities: set[tuple[str, str]] = set()
    for step in sorted(path for path in run_dir.glob("step_*") if path.is_dir()):
        manifest = json.loads((step / "manifest.json").read_text(encoding="utf-8"))
        geometry = manifest.get("grid", {}).get("vertical_geometry", {})
        if geometry.get("kind") != "terrain_following" \
                or geometry.get("dtype") != "float64" \
                or geometry.get("storage") != "versioned_npy_geometry" \
                or geometry.get("contract") != "nimbus.terrain-geometry.v1":
            fail(f"{step.name} omitted the qualified mapped geometry contract")
        generation = geometry.get("publication_generation")
        digest = geometry.get("content_sha256")
        if not isinstance(generation, str) or not generation \
                or not isinstance(digest, str) or len(digest) != 64:
            fail(f"{step.name} has malformed mapped geometry identity")
        identities.add((generation, digest))
        for key in ("geometry_manifest", "center_path", "interface_path"):
            relative = geometry.get(key)
            if not isinstance(relative, str) or not (step / relative).resolve().is_file():
                fail(f"{step.name} has no durable mapped geometry artifact for {key}")
    if len(identities) != 1:
        fail(f"static mapped geometry identity changed across history: {identities}")


def require_retained_parent_closure(run_dir: Path) -> None:
    steps = sorted(path for path in run_dir.glob("step_*") if path.is_dir())
    indices = {int(path.name.removeprefix("step_")) for path in steps}
    if not indices or len(indices) > 6 or min(indices) == 0:
        fail(f"predictive retention did not produce a bounded pruned suffix: {sorted(indices)}")
    retained_files = sum(path.is_file() for path in run_dir.rglob("*"))
    if retained_files > 256:
        fail(f"retained ZFP series has {retained_files} files; declared ceiling is 256")
    for step in steps:
        manifest = json.loads((step / "manifest.json").read_text(encoding="utf-8"))
        catalog = {entry["field_id"]: entry for entry in manifest["fields"]}
        for field in manifest["published_fields"]:
            for parent in catalog[field]["compression"]["parent_step_indices"]:
                if parent not in indices:
                    fail(f"retained {step.name}/{field} is missing predictive parent {parent}")


def require_invalid_neighbors_rejected(source: str, temporary: Path) -> None:
    mutations = {
        "precision_mode": (
            "  zfp_mode: accuracy",
            "  zfp_mode: precision",
            "requires accuracy mode",
        ),
        "sparse_prefilter": (
            "  zfp_sparse_threshold: 0",
            "  zfp_sparse_threshold: 1.0e-6",
            "forbids sparse thresholding",
        ),
        "float16_prefilter": (
            "  zfp_float16_prefilter: false",
            "  zfp_float16_prefilter: true",
            "forbids the float16 prefilter",
        ),
        "excess_theta_error": (
            "    theta: 0.001",
            "    theta: 0.01",
            "exceeds qualified maximum",
        ),
        "unbounded_retention": (
            "  keep_last_frames: 3",
            "  keep_last_frames: 0",
            "requires positive finite frame retention",
        ),
        "disabled_free_space": (
            "  minimum_free_space_mib: 4096",
            "  minimum_free_space_mib: 0",
            "requires a positive retained free-space reserve",
        ),
        "disabled_file_ceiling": (
            "  maximum_retained_history_files: 4096",
            "  maximum_retained_history_files: 0",
            "requires a positive retained history file ceiling",
        ),
        "invalid_parallel_policy": (
            "  zfp_field_execution: serial",
            "  zfp_field_execution: shared_stream",
            "must be serial or field_parallel",
        ),
    }
    for label, (old, new, evidence) in mutations.items():
        if source.count(old) != 1:
            fail(f"cannot construct {label} neighbor from maintained fixture")
        config = temporary / f"invalid_{label}.yaml"
        config.write_text(source.replace(old, new), encoding="utf-8")
        result = subprocess.run(
            [
                str(BINARY),
                "--headless",
                "--config",
                str(config),
                "--duration=0",
                "--write-every=0",
                f"--outdir={temporary / ('invalid_' + label)}",
                "--threads=1",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        diagnostic = result.stdout + result.stderr
        if result.returncode == 0 or evidence not in diagnostic:
            fail(
                f"invalid qualified neighbor {label} was not rejected with {evidence!r}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        fail(f"cannot construct mapped ZFP {label}: expected one source block")
    return source.replace(old, new, 1)


def source_for_preset(
    source: str,
    preset: str,
    fields: set[str],
    budgets: dict[str, float],
) -> str:
    result = replace_once(source, "  preset: minimal", f"  preset: {preset}", "preset")
    if "  maximum_retained_history_files: 256" in result:
        result = replace_once(
            result,
            "  maximum_retained_history_files: 256",
            "  maximum_retained_history_files: 4096",
            "qualification retained-file budget",
        )
    elif result.count("  maximum_retained_history_files: 4096") != 1:
        fail("cannot establish the qualification retained-file budget")
    begin = result.index("  zfp_field_tolerance:\n")
    end = result.index("  keep_last_frames:", begin)
    tolerance_block = "  zfp_field_tolerance:\n" + "".join(
        f"    {field}: {budgets[field]:.17g}\n" for field in sorted(fields)
    )
    return result[:begin] + tolerance_block + result[end:]


def replace_output_block(source: str, output_source: str) -> str:
    output_begin = output_source.index("output:\n")
    output_end = output_source.index("\nextensions:\n", output_begin)
    source_begin = source.index("output:\n")
    source_end = source.index("\nextensions:\n", source_begin)
    return source[:source_begin] + output_source[output_begin:output_end] + source[source_end:]


def mapped_zfp_source(zfp_source: str, terrain_source: str, *, schar: bool) -> str:
    zfp_output_begin = zfp_source.index("output:\n")
    zfp_output_end = zfp_source.index("\nextensions:\n", zfp_output_begin)
    zfp_output = zfp_source[zfp_output_begin:zfp_output_end]
    terrain_output_begin = terrain_source.index("output:\n")
    terrain_output_end = terrain_source.index("\nextensions:\n", terrain_output_begin)
    result = terrain_source[:terrain_output_begin] + zfp_output + terrain_source[terrain_output_end:]
    result = replace_once(
        result, "microphysics:\n  scheme: kessler\n",
        "microphysics:\n  scheme: milbrandt\n", "microphysics")
    result = replace_once(result, "  K_h: 0\n", "  K_h: 80\n", "horizontal diffusion")
    result = replace_once(result, "  K_v: 0\n", "  K_v: 160\n", "vertical diffusion")
    result = replace_once(
        result, "  apply_to: scalars\n",
        "  apply_to: scalars_and_momentum\n", "diffusion target")
    result = replace_once(
        result,
        "dynamics:\n  scheme: klemp_wilhelmson_split_cartesian_cgrid\n  axisymmetric: false\n",
        "dynamics:\n  scheme: klemp_wilhelmson_split_cartesian_cgrid\n"
        "  axisymmetric: false\n\nadvection:\n  scheme: tvd\n",
        "advection")
    result = replace_once(result, "  do_sw: false\n", "  do_sw: true\n", "shortwave")
    result = replace_once(
        result, "  local_solar_time_at_start_h: 0\n",
        "  local_solar_time_at_start_h: 12\n", "solar time")
    result = replace_once(
        result, "  clear_sky_only: true\n",
        "  clear_sky_only: false\n"
        "  simple_grey_optical_model: absorbing_water_cloud_path_v1\n",
        "cloud optics")
    result = replace_once(
        result, "      dtheta_k: 1\n",
        "      dtheta_k: 1\n      cloud_water_kgkg: 0.002\n",
        "cloud seed")
    if schar:
        bell = """terrain:
  scheme: bell
  coordinate: smoothed
  compute_derivatives: true
  compute_metrics: true
  smoothing_decay: 0.1
  bell:
    h0: 300
    a: 2000
    axisymmetric: true
"""
        schar_block = """terrain:
  scheme: schar
  coordinate: smoothed
  compute_derivatives: true
  compute_metrics: true
  smoothing_decay: 0.1
  schar:
    H: 300
    a: 5000
    ell: 4000
"""
        result = replace_once(result, bell, schar_block, "Schär terrain")
    return result


def require_radar_zfp_guard(source: str, temporary: Path) -> list[dict[str, object]]:
    rows = []
    for preset, fixture in RADAR_FIXTURES.items():
        radar_source = replace_output_block(
            fixture.read_text(encoding="utf-8"), source)
        radar_source = replace_once(
            radar_source, "  preset: minimal", f"  preset: {preset}",
            "retained radar preset")
        zfp_config = temporary / f"{preset}_zfp_guard.yaml"
        npy_config = temporary / f"{preset}_npy_control.yaml"
        zfp_config.write_text(radar_source, encoding="utf-8")
        npy_config.write_text(
            radar_source.replace("  format: zfp_high", "  format: npy_3d", 1)
            .replace("  async_io: true", "  async_io: false", 1),
            encoding="utf-8")
        npy_output = temporary / f"{preset}_npy_control"
        run_simulator(npy_config, npy_output, 1)
        published = published_fields(npy_output)
        if not set(RADAR_PRODUCTS[preset]).issubset(published):
            fail(f"{preset} NPY control omitted selected radar products")
        logs = {}
        zfp_output = temporary / f"{preset}_zfp_guard_output"
        for mode, command in (
            ("validate_only", [str(BINARY), "--validate-only", "--config", str(zfp_config)]),
            ("headless", [str(BINARY), "--headless", "--config", str(zfp_config),
                          "--duration=2", "--write-every=2", f"--outdir={zfp_output}",
                          "--threads=1", "--log-profile=quiet"]),
        ):
            result = subprocess.run(command, cwd=ROOT, text=True,
                                    capture_output=True, timeout=60)
            log = temporary / f"{preset}_zfp_guard.{mode}.log"
            diagnostic = result.stdout + "\n" + result.stderr
            log.write_text(diagnostic, encoding="utf-8")
            if result.returncode == 0:
                fail(f"{preset} qualified ZFP unexpectedly admitted: {mode}")
            for field in RADAR_PRODUCTS[preset]:
                if f"field '{field}' is not admitted to qualified ZFP" not in diagnostic:
                    fail(f"{preset} {mode} omitted keyed {field} guard")
            if zfp_output.exists() and any(zfp_output.rglob("*")):
                fail(f"{preset} qualified ZFP published output before rejection")
            logs[mode] = sha256(log)
        rows.append({
            "preset": preset,
            "selectedProducts": list(RADAR_PRODUCTS[preset]),
            "zfpConfigSha256": sha256(zfp_config),
            "npyConfigSha256": sha256(npy_config),
            "npyFinalManifestSha256": sha256(npy_output / "step_000001/manifest.json"),
            "npyPublishedFields": sorted(published),
            "zfpRejectionLogSha256": logs,
            "noZfpPublication": True,
        })
    return rows


def require_radar_zfp_public(
    source: str, temporary: Path, budgets: dict[str, float]
) -> list[dict[str, object]]:
    if not RADAR_READER.is_file():
        fail(f"native radar ZFP reader test binary is missing: {RADAR_READER}")
    rows = []
    for preset, fixture in RADAR_FIXTURES.items():
        npy_source = fixture.read_text(encoding="utf-8")
        radar_source = replace_output_block(npy_source, source)
        radar_source = replace_once(
            radar_source, "  preset: minimal", f"  preset: {preset}",
            "retained radar preset",
        )
        if preset == "radar_polarimetric_ready":
            radar_source = replace_once(
                radar_source,
                "  keep_last_frames: 3",
                f"    qc: {budgets['qc']:.17g}\n  keep_last_frames: 3",
                "polarimetric cloud-water budget",
            )
        zfp_config = temporary / f"{preset}_zfp_public.yaml"
        npy_config = temporary / f"{preset}_npy_public.yaml"
        zfp_config.write_text(radar_source, encoding="utf-8")
        npy_config.write_text(npy_source, encoding="utf-8")
        npy_output = temporary / f"{preset}_npy_public"
        zfp_output = temporary / f"{preset}_zfp_public"
        commands = {
            "validate": [str(BINARY), "--validate-only", "--config", str(zfp_config),
                         "--duration=4"],
            "npy": [str(BINARY), "--headless", "--config", str(npy_config),
                    "--duration=4", "--write-every=4", "--checkpoint-every=0",
                    f"--outdir={npy_output}", "--threads=1", "--log-profile=quiet"],
            "zfp": [str(BINARY), "--headless", "--config", str(zfp_config),
                    "--duration=4", "--write-every=4", "--checkpoint-every=0",
                    f"--outdir={zfp_output}", "--threads=1", "--log-profile=quiet"],
        }
        logs = {}
        for mode, command in commands.items():
            result = subprocess.run(command, cwd=ROOT, text=True,
                                    capture_output=True, timeout=60, check=False)
            log = temporary / f"{preset}_{mode}.log"
            log.write_text(result.stdout + "\n" + result.stderr,
                           encoding="utf-8")
            if result.returncode != 0:
                fail(f"{preset} ordinary public {mode} failed: {log}")
            logs[mode] = sha256(log)
        expected_radar = set(RADAR_PRODUCTS[preset])
        npy_fields = published_fields(npy_output)
        zfp_fields = published_fields(zfp_output)
        if npy_fields != zfp_fields or not expected_radar.issubset(npy_fields):
            fail(f"{preset} public NPY/ZFP selected field sets disagree")
        environment = os.environ.copy()
        environment["NIMBUS_TEST_RADAR_V4_PUBLIC_PAIR_CONFIG"] = str(zfp_config)
        environment["NIMBUS_TEST_RADAR_V4_PUBLIC_PAIR_NPY"] = str(npy_output)
        environment["NIMBUS_TEST_RADAR_V4_PUBLIC_PAIR_ZFP"] = str(zfp_output)
        result = subprocess.run(
            [str(RADAR_READER), "[core][zfp][v4][public_pair_readback]"],
            cwd=ROOT, env=environment, text=True, capture_output=True,
            timeout=60, check=False,
        )
        reader_log = temporary / f"{preset}_native_readback.log"
        reader_log.write_text(result.stdout + "\n" + result.stderr,
                              encoding="utf-8")
        if result.returncode != 0 or "All tests passed" not in result.stdout:
            fail(f"{preset} ordinary public native decoded-field reader failed: {reader_log}")
        rows.append({
            "preset": preset,
            "selectedProducts": list(RADAR_PRODUCTS[preset]),
            "zfpConfigSha256": sha256(zfp_config),
            "npyConfigSha256": sha256(npy_config),
            "npyFinalManifestSha256": sha256(npy_output / "step_000001/manifest.json"),
            "zfpFinalManifestSha256": sha256(zfp_output / "step_000001/manifest.json"),
            "publishedFields": sorted(npy_fields),
            "runLogSha256": logs,
            "nativeReadbackLogSha256": sha256(reader_log),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radar-guard-only", action="store_true")
    parser.add_argument("--radar-public-only", action="store_true")
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args()
    radar_only = args.radar_guard_only or args.radar_public_only
    if args.radar_guard_only and args.radar_public_only:
        parser.error("select only one radar-only compatibility mode")
    if (args.raw_root is None) != (args.report_output is None):
        parser.error("--raw-root and --report-output must be supplied together")
    if args.raw_root is not None and \
            (args.raw_root.resolve().exists() or args.report_output.resolve().exists()):
        parser.error("raw root and report output must both be new")
    if not BINARY.is_file():
        fail(f"simulator is missing: {BINARY}")
    source = FIXTURE.read_text(encoding="utf-8")
    terrain_source = MAPPED_FIXTURE.read_text(encoding="utf-8")
    if "  async_io: true" not in source:
        fail("qualified ZFP fixture must exercise the bounded asynchronous writer")
    catalog_bytes = run([str(BINARY), "--scheme-catalog=json"]).stdout
    catalog = json.loads(catalog_bytes)
    policy = catalog["runtimeCapabilities"]["interactionPolicy"]
    radar_public = policy.get("qualifiedZfpReversibleRadarV4", {}).get("admitted") is True
    if args.radar_public_only and not radar_public:
        fail("exact simulator catalog keeps ordinary radar ZFP admission closed")
    preset_fields = {
        preset: set(fields)
        for preset, fields in policy["qualifiedZfpPresetFields"].items()
    }
    budgets = {
        field: float(value)
        for field, value in policy["qualifiedZfpFieldBudgets"].items()
    }
    if set(policy["qualifiedZfpPresetIds"]) != QUALIFIED_PRESETS \
            or set(preset_fields) != QUALIFIED_PRESETS:
        fail("exact binary does not publish every non-radar qualified ZFP preset")
    readiness = {item["id"]: item for item in catalog["readiness"]["items"]}
    if (readiness["output.zfp_bounded_history"]["state"] != "engineering_incomplete"
            or not readiness["output.zfp_bounded_history"]["blocker"]):
        fail("exact binary does not track guarded radar ZFP as Engineering incomplete")
    for preset, fields in preset_fields.items():
        if not fields or any(field not in budgets for field in fields):
            fail(f"{preset} has incomplete exact-binary field budgets")
    source = source_for_preset(source, "minimal", preset_fields["minimal"], budgets)

    if radar_only:
        if args.raw_root is None:
            context = tempfile.TemporaryDirectory(prefix="nimbus_zfp_radar_guard_")
        else:
            raw_root = args.raw_root.resolve()
            raw_root.mkdir(parents=True, exist_ok=False)
            context = nullcontext(str(raw_root))
        with context as raw:
            temporary = Path(raw)
            rows = (require_radar_zfp_public(source, temporary, budgets)
                    if radar_public else require_radar_zfp_guard(source, temporary))
            if args.report_output is not None:
                destination = args.report_output.resolve()
                if destination.exists():
                    parser.error("use a new report path")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps({
                    "schema": ("nimbus.zfp-core-history-radar-public.v2" if radar_public
                               else "nimbus.zfp-core-history-radar-guard.v1"),
                    "status": ("pass_public_radar_zfp_candidate" if radar_public
                               else "pass_retained_radar_zfp_fail_closed"),
                    "claimBoundary": ("Both named flat-CPU radar presets pass ordinary qualified ZFP v4 public admission, NPY/native decoded-field comparison and six-ledger identity. The 31-subset, Studio and full non-radar qualifier gates are separate; whole-backend Ready and scientific qualification are not claimed."
                                      if radar_public else
                                      "Both maintained radar presets publish selected products through NPY, while qualified ZFP rejects each selected field before publication through validate-only and headless entry paths. Non-radar ZFP full qualifier is a separate gate."),
                    "binarySha256": sha256(BINARY),
                    "qualifierSha256": sha256(Path(__file__)),
                    "catalogSha256": hashlib.sha256(catalog_bytes.encode()).hexdigest(),
                    "nativeReaderSha256": sha256(RADAR_READER) if radar_public else None,
                    "rawRoot": str(temporary),
                    "cases": rows,
                }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if radar_public:
            print("PASS both maintained radar presets through ordinary public ZFP v4 and native decoded-field readback")
        else:
            print("PASS maintained radar NPY controls and qualified ZFP keyed prepublication guards")
        return 0

    if args.raw_root is None:
        full_context = tempfile.TemporaryDirectory(
            prefix="nimbus_zfp_core_qualification_")
    else:
        full_root = args.raw_root.resolve()
        full_root.mkdir(parents=True, exist_ok=False)
        full_context = nullcontext(str(full_root))
    with full_context as raw_temporary:
        temporary = Path(raw_temporary)
        zfp_one = temporary / "zfp_t1"
        zfp_four = temporary / "zfp_t4"
        parallel_one = temporary / "parallel_w1"
        parallel_four_a = temporary / "parallel_w4_a"
        parallel_four_b = temporary / "parallel_w4_b"
        parallel_one_config = temporary / "parallel_w1.yaml"
        parallel_four_config = temporary / "parallel_w4.yaml"
        npy_one = temporary / "npy_t1"
        retained_zfp = temporary / "zfp_retained"
        npy_config = temporary / "npy_reference.yaml"
        mapped_bell_config = temporary / "mapped_bell_zfp.yaml"
        mapped_schar_config = temporary / "mapped_schar_zfp.yaml"
        mapped_bell_npy_config = temporary / "mapped_bell_npy.yaml"
        mapped_schar_npy_config = temporary / "mapped_schar_npy.yaml"
        if source.count("  format: zfp_high") != 1:
            fail("maintained fixture has no unique ZFP format selector")
        npy_config.write_text(source.replace("  format: zfp_high", "  format: npy_3d"), encoding="utf-8")
        mapped_bell = mapped_zfp_source(source, terrain_source, schar=False)
        mapped_schar = mapped_zfp_source(source, terrain_source, schar=True)
        mapped_bell_config.write_text(mapped_bell, encoding="utf-8")
        mapped_schar_config.write_text(mapped_schar, encoding="utf-8")
        mapped_bell_npy_config.write_text(
            mapped_bell.replace("  format: zfp_high", "  format: npy_3d", 1),
            encoding="utf-8")
        mapped_schar_npy_config.write_text(
            mapped_schar.replace("  format: zfp_high", "  format: npy_3d", 1),
            encoding="utf-8")
        if source.count("  zfp_field_execution: serial") != 1:
            fail("maintained fixture has no unique serial field policy")
        parallel_base = source.replace(
            "  zfp_field_execution: serial",
            "  zfp_field_execution: field_parallel",
        )
        parallel_one_config.write_text(
            parallel_base.replace("  zfp_field_workers: 0", "  zfp_field_workers: 1"),
            encoding="utf-8",
        )
        parallel_four_config.write_text(
            parallel_base.replace("  zfp_field_workers: 0", "  zfp_field_workers: 4"),
            encoding="utf-8",
        )

        run_simulator(FIXTURE, zfp_one, 1)
        run_simulator(FIXTURE, zfp_four, 4)
        run_simulator(npy_config, npy_one, 1)
        run_simulator(parallel_one_config, parallel_one, 1)
        run_simulator(parallel_four_config, parallel_four_a, 1)
        run_simulator(parallel_four_config, parallel_four_b, 4)
        run_simulator(FIXTURE, retained_zfp, 1, duration=12)
        mapped_runs: list[tuple[Path, Path, Path]] = []
        for terrain_name, zfp_config, npy_reference in (
            ("bell", mapped_bell_config, mapped_bell_npy_config),
            ("schar", mapped_schar_config, mapped_schar_npy_config),
        ):
            mapped_one = temporary / f"mapped_{terrain_name}_zfp_t1"
            mapped_four = temporary / f"mapped_{terrain_name}_zfp_t4"
            mapped_npy = temporary / f"mapped_{terrain_name}_npy_t1"
            run_simulator(zfp_config, mapped_one, 1)
            run_simulator(zfp_config, mapped_four, 4)
            run_simulator(npy_reference, mapped_npy, 1)
            inspect_publications(mapped_one, "serial", 1, preset_fields["minimal"], budgets)
            inspect_publications(mapped_four, "serial", 1, preset_fields["minimal"], budgets)
            require_mapped_geometry_contract(mapped_one)
            require_mapped_geometry_contract(mapped_four)
            require_artifact_identity(mapped_one, mapped_four)
            mapped_runs.append((mapped_one, mapped_four, mapped_npy))
        inspect_publications(zfp_one, "serial", 1, preset_fields["minimal"], budgets)
        inspect_publications(zfp_four, "serial", 1, preset_fields["minimal"], budgets)
        inspect_publications(parallel_one, "field_parallel", 1, preset_fields["minimal"], budgets)
        inspect_publications(parallel_four_a, "field_parallel", 4, preset_fields["minimal"], budgets)
        inspect_publications(parallel_four_b, "field_parallel", 4, preset_fields["minimal"], budgets)
        require_artifact_identity(zfp_one, zfp_four)
        require_artifact_identity(zfp_one, parallel_one)
        require_artifact_identity(zfp_one, parallel_four_a)
        require_artifact_identity(parallel_four_a, parallel_four_b)
        require_retained_parent_closure(retained_zfp)
        require_invalid_neighbors_rejected(source, temporary)

        env = os.environ.copy()
        env["NIMBUS_TEST_ZFP_RUN"] = str(parallel_four_a)
        env["NIMBUS_TEST_NPY_RUN"] = str(npy_one)
        run(
            [
                "cargo",
                "test",
                "--manifest-path",
                str(ROOT / "ui" / "src-tauri" / "Cargo.toml"),
                "qualified_simulator_zfp_matches_npy_reference",
                "--lib",
                "--",
                "--ignored",
            ],
            env=env,
        )
        env["NIMBUS_TEST_RETAINED_ZFP_RUN"] = str(retained_zfp)
        run(
            [
                "cargo",
                "test",
                "--manifest-path",
                str(ROOT / "ui" / "src-tauri" / "Cargo.toml"),
                "qualified_retained_predictive_zfp_latest_decodes",
                "--lib",
                "--",
                "--ignored",
            ],
            env=env,
        )
        for mapped_one, mapped_four, mapped_npy in mapped_runs:
            env["NIMBUS_TEST_ZFP_RUN"] = str(mapped_four)
            env["NIMBUS_TEST_NPY_RUN"] = str(mapped_npy)
            run(
                [
                    "cargo", "test", "--manifest-path",
                    str(ROOT / "ui" / "src-tauri" / "Cargo.toml"),
                    "qualified_simulator_zfp_matches_npy_reference",
                    "--lib", "--", "--ignored",
                ],
                env=env,
            )

        for terrain_name, schar in (("bell", False), ("schar", True)):
            mapped_minimal = mapped_zfp_source(source, terrain_source, schar=schar)
            for preset in sorted(NON_RADAR_PRESETS - {"minimal"}):
                fields = preset_fields[preset]
                preset_source = source_for_preset(mapped_minimal, preset, fields, budgets)
                zfp_config = temporary / f"mapped_{terrain_name}_{preset}_zfp.yaml"
                npy_reference = temporary / f"mapped_{terrain_name}_{preset}_npy.yaml"
                zfp_config.write_text(preset_source, encoding="utf-8")
                npy_reference.write_text(
                    preset_source.replace("  format: zfp_high", "  format: npy_3d", 1),
                    encoding="utf-8",
                )
                zfp_one = temporary / f"mapped_{terrain_name}_{preset}_zfp_t1"
                zfp_four = temporary / f"mapped_{terrain_name}_{preset}_zfp_t4"
                npy_one = temporary / f"mapped_{terrain_name}_{preset}_npy_t1"
                run_simulator(zfp_config, zfp_one, 1)
                run_simulator(zfp_config, zfp_four, 4)
                run_simulator(npy_reference, npy_one, 1)
                reference_fields = published_fields(npy_one)
                if not reference_fields or not reference_fields.issubset(fields):
                    fail(f"{terrain_name}/{preset} published fields outside its catalog contract")
                inspect_publications(zfp_one, "serial", 1, reference_fields, budgets)
                inspect_publications(zfp_four, "serial", 1, reference_fields, budgets)
                require_mapped_geometry_contract(zfp_one)
                require_mapped_geometry_contract(zfp_four)
                require_artifact_identity(zfp_one, zfp_four)
                env["NIMBUS_TEST_ZFP_RUN"] = str(zfp_four)
                env["NIMBUS_TEST_NPY_RUN"] = str(npy_one)
                run(
                    [
                        "cargo", "test", "--manifest-path",
                        str(ROOT / "ui" / "src-tauri" / "Cargo.toml"),
                        "qualified_simulator_zfp_matches_npy_reference",
                        "--lib", "--", "--ignored",
                    ],
                    env=env,
                )

        if radar_public:
            radar_rows = require_radar_zfp_public(source, temporary, budgets)
        else:
            radar_rows = require_radar_zfp_guard(source, temporary)
        if args.report_output is not None:
            files = [path for path in temporary.rglob("*") if path.is_file()]
            total_bytes = sum(path.stat().st_size for path in files)
            if len(files) >= 10000 or total_bytes >= 512 * 1024 * 1024:
                fail("full ZFP qualifier exceeded 10000-file/512-MiB evidence cap")
            destination = args.report_output.resolve()
            if destination.exists():
                parser.error("use a new report path")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps({
                "schema": "nimbus.zfp-core-history-full.v2",
                "status": "pass",
                "claimBoundary": "Bounded maintained non-radar ZFP history plus both ordinary public flat-CPU named radar presets; whole-backend readiness, campaigns and external review are separate.",
                "binarySha256": sha256(BINARY),
                "qualifierSha256": sha256(Path(__file__)),
                "catalogSha256": hashlib.sha256(catalog_bytes.encode()).hexdigest(),
                "nativeRadarReaderSha256": sha256(RADAR_READER) if radar_public else None,
                "zfpFixtureSha256": sha256(FIXTURE),
                "mappedFixtureSha256": sha256(MAPPED_FIXTURE),
                "radarPublic": radar_public,
                "radarCases": radar_rows,
                "rawFiles": len(files),
                "rawBytes": total_bytes,
                "rawRoot": str(temporary),
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "PASS qualified ordinary-runtime ZFP publication, decoded error budgets, "
        "bounded async ownership, serial/1/4-worker exact artifacts, repeated determinism, "
        "invalid-neighbor rejection, retained predictive-parent decode, "
        "all non-radar presets on mapped Bell/Schär geometry, radar NPY controls, "
        f"both radar qualified-ZFP {'public v4 preset checks' if radar_public else 'preset guards'}, "
        "and C++ writer to Rust reader parity"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
