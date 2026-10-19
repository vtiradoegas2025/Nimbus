#!/usr/bin/env python3
"""Qualify one real Nimbus netCDF frame across available local tooling."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile

from multiresolution_contract import inspect_durable_hierarchies


ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "bin" / "netcdf_history_fixture"
SIMULATOR = ROOT / "bin" / "nimbus_sim"
ZFP_CONFIG = ROOT / "tests" / "fixtures" / "configs" / "qualification" / "zfp_core_history.yaml"
MAPPED_CONFIG = ROOT / "tests" / "fixtures" / "configs" / "qualification" / "cgrid_terrain_smagorinsky_coupled.yaml"
EXPECTED_THETA = [float(value) for value in range(300, 306)] + [float(value) for value in range(310, 316)]
EXPECTED_RAIN = [float(value) for value in range(6)]


def fail(message: str) -> None:
    raise RuntimeError(message)


def run(command: list[str], *, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    if result.returncode != 0:
        fail(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout


def read_npy(path: Path) -> tuple[tuple[int, ...], list[float]]:
    data = path.read_bytes()
    if data[:8] != b"\x93NUMPY\x01\x00":
        fail(f"{path} is not a NumPy v1.0 file")
    header_length = struct.unpack_from("<H", data, 8)[0]
    header = ast.literal_eval(data[10 : 10 + header_length].decode("ascii").strip())
    if header.get("descr") != "<f4" or header.get("fortran_order") is not False:
        fail(f"{path} does not use little-endian C-order float32")
    shape = tuple(header["shape"])
    count = 1
    for dimension in shape:
        count *= dimension
    payload = data[10 + header_length :]
    if len(payload) != count * 4:
        fail(f"{path} payload length does not match shape {shape}")
    return shape, list(struct.unpack(f"<{count}f", payload))


def ncdump_values(dump: str, variable: str) -> list[float]:
    data = dump.split("\ndata:", 1)
    if len(data) != 2:
        fail("ncdump output has no data section")
    match = re.search(rf"\n\s*{re.escape(variable)}\s*=\s*(.*?);", data[1], re.DOTALL)
    if not match:
        fail(f"ncdump did not emit data for {variable}")
    return [float(token) for token in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?", match.group(1))]


def require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        fail(f"{label}: expected {expected!r}, got {actual!r}")


def run_simulator(config: Path, output: Path, threads: int, duration: int = 2) -> None:
    run([
        str(SIMULATOR), "--headless", "--config", str(config),
        f"--duration={duration}", "--write-every=2", f"--outdir={output}",
        f"--threads={threads}", "--log-profile=quiet",
    ])


def inspect_mapped_history(
        run_dir: Path, npy_reference: Path, ncdump: str) -> None:
    steps = sorted(path for path in run_dir.glob("step_*") if path.is_dir())
    npy_steps = sorted(path for path in npy_reference.glob("step_*") if path.is_dir())
    if len(steps) < 2 or len(steps) != len(npy_steps):
        fail("mapped netCDF/NPY qualification has inconsistent publications")
    for step, npy_step in zip(steps, npy_steps):
        manifest = json.loads((step / "manifest.json").read_text(encoding="utf-8"))
        npy_manifest = json.loads(
            (npy_step / "manifest.json").read_text(encoding="utf-8"))
        fields = manifest["published_fields"]
        if not fields:
            fail("mapped netCDF publication has no fields")
        require_equal(manifest["data_product"]["storage_format"], "netcdf", "mapped storage format")
        require_equal(fields, npy_manifest["published_fields"], "mapped netCDF/NPY field inventory")
        vertical = manifest["grid"]["vertical_geometry"]
        require_equal(vertical["kind"], "terrain_following", "mapped geometry kind")
        require_equal(vertical["dtype"], "float64", "mapped geometry dtype")
        require_equal(vertical["storage"], "netcdf_variable", "mapped geometry storage")
        history = step / "history.nc"
        header = run([ncdump, "-h", str(history)])
        for evidence in (
            "double physical_z_center_m(x, y, z)",
            "double physical_z_interface_m(x, y, z_interface)",
            ':vertical_geometry = "terrain_following"',
        ):
            if evidence not in header:
                fail(f"mapped netCDF header lacks {evidence!r}")
        for field in fields:
            values = ncdump_values(
                run([ncdump, "-p", "9,17", "-v", field, str(history)]),
                field)
            values = [
                struct.unpack("<f", struct.pack("<f", value))[0]
                for value in values
            ]
            shape, expected = read_npy(npy_step / f"{field}.npy")
            if len(shape) not in (2, 3):
                fail(f"mapped NPY {field} has unsupported rank {len(shape)}")
            require_equal(values, expected, f"mapped netCDF/NPY {field}")
    inspect_durable_hierarchies(
        run_dir, expected_format="netcdf", require_lod=False
    )
    inspect_durable_hierarchies(
        npy_reference, expected_format="npy_3d", require_lod=False
    )


def main() -> int:
    if not FIXTURE.is_file():
        fail(f"fixture generator is missing: {FIXTURE}; run make bin/netcdf_history_fixture")
    if not SIMULATOR.is_file():
        fail(f"simulator is missing: {SIMULATOR}")
    ncdump = shutil.which("ncdump")
    cargo = shutil.which("cargo")
    if not ncdump:
        fail("required external reader ncdump is unavailable")
    if not cargo:
        fail("required Studio qualification tool cargo is unavailable")

    with tempfile.TemporaryDirectory(prefix="nimbus_netcdf_qualification_") as temporary:
        run_dir = Path(temporary) / "run"
        run([str(FIXTURE), str(run_dir)])
        history = run_dir / "step_000004" / "history.nc"
        manifest_path = run_dir / "step_000004" / "manifest.json"
        if not history.is_file() or history.stat().st_size == 0:
            fail("Nimbus did not publish a non-empty history.nc")
        for suffix in ("history.nc.tmp", "manifest.json.tmp"):
            if (history.parent / suffix).exists():
                fail(f"atomic publication left {suffix} visible")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require_equal(manifest["manifest_version"], 7, "manifest version")
        require_equal(manifest["data_product"]["storage_format"], "netcdf", "storage format")
        require_equal(manifest["published_fields"], ["accumulated_rainfall", "theta"], "published fields")
        catalog = {entry["field_id"]: entry for entry in manifest["fields"]}
        for field in manifest["published_fields"]:
            entry = catalog[field]
            require_equal(entry["file_pattern"], "history.nc", f"{field} file mapping")
            require_equal(entry["storage_layout"], "shared_file", f"{field} storage layout")
            require_equal(entry["variable_name"], field, f"{field} variable mapping")

        header = run([ncdump, "-h", str(history)])
        for evidence in (
            'Conventions = "CF-1.10"',
            "time = UNLIMITED",
            "x = 2",
            "y = 3",
            "z = 2",
            "float theta(time, x, y, z)",
            'theta:units = "K"',
            "float accumulated_rainfall(time, x, y)",
            'accumulated_rainfall:units = "mm"',
            ':restart_complete = 0b',
        ):
            if evidence not in header:
                fail(f"ncdump header lacks {evidence!r}")
        dump = run([ncdump, "-v", "time,x,y,z,theta,accumulated_rainfall", str(history)])
        require_equal(ncdump_values(dump, "time"), [75.5], "time coordinate")
        require_equal(ncdump_values(dump, "x"), [500.0, 1500.0], "x coordinates")
        require_equal(ncdump_values(dump, "y"), [375.0, 1125.0, 1875.0], "y coordinates")
        require_equal(ncdump_values(dump, "z"), [125.0, 375.0], "z coordinates")
        netcdf_theta = ncdump_values(dump, "theta")
        netcdf_rain = ncdump_values(dump, "accumulated_rainfall")
        require_equal(netcdf_theta, EXPECTED_THETA, "netCDF theta values")
        require_equal(netcdf_rain, EXPECTED_RAIN, "netCDF rainfall values")
        theta_shape, npy_theta = read_npy(run_dir / "npy_reference" / "theta.npy")
        rain_shape, npy_rain = read_npy(run_dir / "npy_reference" / "accumulated_rainfall.npy")
        require_equal(theta_shape, (2, 3, 2), "NPY theta shape")
        require_equal(rain_shape, (2, 3), "NPY rainfall shape")
        require_equal(netcdf_theta, npy_theta, "netCDF/NPY theta parity")
        require_equal(netcdf_rain, npy_rain, "netCDF/NPY rainfall parity")
        print("PASS ncdump metadata, coordinates, units, dimensions, values, manifest mappings, and NPY parity")

        if importlib.util.find_spec("xarray") is None:
            print("SKIP xarray: Python package is not installed")
        else:
            import xarray as xr  # type: ignore

            with xr.open_dataset(history) as dataset:
                require_equal(tuple(dataset["theta"].shape), (1, 2, 3, 2), "xarray theta shape")
                require_equal(dataset["theta"].values.reshape(-1).tolist(), EXPECTED_THETA, "xarray theta values")
                require_equal(dataset["theta"].attrs.get("units"), "K", "xarray theta units")
            print("PASS xarray readback")

        ncks = shutil.which("ncks")
        if ncks:
            run([ncks, "-m", str(history)])
            run([ncks, "-H", "-C", "-v", "theta", str(history)])
            print("PASS NCO metadata and theta readback")
        else:
            print("SKIP NCO: ncks is not installed")

        cdo = shutil.which("cdo")
        if cdo:
            run([cdo, "-s", "sinfo", str(history)])
            print("PASS CDO stream inspection")
        else:
            print("SKIP CDO: cdo is not installed")

        env = os.environ.copy()
        env["NIMBUS_TEST_NETCDF_RUN"] = str(run_dir)
        run(
            [
                cargo,
                "test",
                "--manifest-path",
                str(ROOT / "ui" / "src-tauri" / "Cargo.toml"),
                "commands::outputs::tests::simulator_netcdf_history_reopens_in_studio",
                "--",
                "--ignored",
                "--exact",
            ],
            env=env,
        )
        print("PASS Studio run discovery and exact libnetcdf field readback")

        terrain_run = Path(temporary) / "terrain_run"
        run([str(FIXTURE), str(terrain_run), "--terrain"])
        terrain_history = terrain_run / "step_000004" / "history.nc"
        terrain_manifest = json.loads(
            (terrain_run / "step_000004" / "manifest.json").read_text(encoding="utf-8")
        )
        vertical = terrain_manifest["grid"]["vertical_geometry"]
        require_equal(vertical["kind"], "terrain_following", "terrain geometry kind")
        require_equal(vertical["dtype"], "float64", "terrain geometry dtype")
        require_equal(vertical["storage"], "netcdf_variable", "terrain geometry storage")
        terrain_header = run([ncdump, "-h", str(terrain_history)])
        for evidence in (
            "z_interface = 3",
            "double physical_z_center_m(x, y, z)",
            "double physical_z_interface_m(x, y, z_interface)",
            'theta:coordinates = "time x y physical_z_center_m"',
            ':vertical_geometry = "terrain_following"',
        ):
            if evidence not in terrain_header:
                fail(f"terrain ncdump header lacks {evidence!r}")
        env["NIMBUS_TEST_TERRAIN_NETCDF_RUN"] = str(terrain_run)
        run(
            [
                cargo,
                "test",
                "--manifest-path",
                str(ROOT / "ui" / "src-tauri" / "Cargo.toml"),
                "commands::outputs::tests::simulator_terrain_netcdf_window_uses_exact_geometry",
                "--",
                "--ignored",
                "--exact",
            ],
            env=env,
        )
        print("PASS exact terrain netCDF float64 geometry, bounded window, and volume weighting")

        from qualify_zfp_core_history import mapped_zfp_source

        zfp_source = ZFP_CONFIG.read_text(encoding="utf-8")
        mapped_source = MAPPED_CONFIG.read_text(encoding="utf-8")
        for terrain_name, schar in (("bell", False), ("schar", True)):
            netcdf_config = Path(temporary) / f"mapped_{terrain_name}_netcdf.yaml"
            npy_config = Path(temporary) / f"mapped_{terrain_name}_npy.yaml"
            source = mapped_zfp_source(zfp_source, mapped_source, schar=schar)
            source = source.replace("  format: zfp_high", "  format: netcdf", 1)
            netcdf_config.write_text(source, encoding="utf-8")
            npy_config.write_text(
                source.replace("  format: netcdf", "  format: npy_3d", 1),
                encoding="utf-8")
            netcdf_one = Path(temporary) / f"mapped_{terrain_name}_netcdf_t1"
            netcdf_four = Path(temporary) / f"mapped_{terrain_name}_netcdf_t4"
            npy_one = Path(temporary) / f"mapped_{terrain_name}_npy_t1"
            run_simulator(netcdf_config, netcdf_one, 1)
            run_simulator(netcdf_config, netcdf_four, 4)
            run_simulator(npy_config, npy_one, 1)
            inspect_mapped_history(netcdf_one, npy_one, ncdump)
            inspect_mapped_history(netcdf_four, npy_one, ncdump)
            env["NIMBUS_TEST_MAPPED_NETCDF_RUN"] = str(netcdf_four)
            env["NIMBUS_TEST_MAPPED_NPY_RUN"] = str(npy_one)
            run([
                cargo, "test", "--manifest-path",
                str(ROOT / "ui" / "src-tauri" / "Cargo.toml"),
                "commands::outputs::tests::simulator_mapped_netcdf_history_matches_npy_reference",
                "--", "--ignored", "--exact",
            ], env=env)
            if terrain_name == "bell":
                retained_netcdf = Path(temporary) / "mapped_bell_netcdf_retained"
                retained_npy = Path(temporary) / "mapped_bell_npy_retained"
                run_simulator(netcdf_config, retained_netcdf, 1, duration=8)
                run_simulator(npy_config, retained_npy, 1, duration=8)
                inspect_mapped_history(retained_netcdf, retained_npy, ncdump)
                retained_steps = sorted(
                    path for path in retained_netcdf.glob("step_*")
                    if path.is_dir())
                if len(retained_steps) != 3 or retained_steps[0].name == "step_000000":
                    fail(f"mapped netCDF retention did not publish a bounded suffix: {retained_steps}")
        for preset in (
            "presentation", "studio_full", "dynamics_full",
            "science_validation",
        ):
            preset_netcdf_config = Path(temporary) / f"mapped_bell_{preset}_netcdf.yaml"
            preset_npy_config = Path(temporary) / f"mapped_bell_{preset}_npy.yaml"
            source = mapped_zfp_source(zfp_source, mapped_source, schar=False)
            source = source.replace("  preset: minimal", f"  preset: {preset}", 1)
            source = source.replace("  format: zfp_high", "  format: netcdf", 1)
            source = source.replace(
                "  maximum_retained_history_files: 256",
                "  maximum_retained_history_files: 4096",
                1,
            )
            preset_netcdf_config.write_text(source, encoding="utf-8")
            preset_npy_config.write_text(
                source.replace("  format: netcdf", "  format: npy_3d", 1),
                encoding="utf-8")
            netcdf_run = Path(temporary) / f"mapped_bell_{preset}_netcdf"
            npy_run = Path(temporary) / f"mapped_bell_{preset}_npy"
            run_simulator(preset_netcdf_config, netcdf_run, 1)
            run_simulator(preset_npy_config, npy_run, 1)
            inspect_mapped_history(netcdf_run, npy_run, ncdump)
            env["NIMBUS_TEST_MAPPED_NETCDF_RUN"] = str(netcdf_run)
            env["NIMBUS_TEST_MAPPED_NPY_RUN"] = str(npy_run)
            run([
                cargo, "test", "--manifest-path",
                str(ROOT / "ui" / "src-tauri" / "Cargo.toml"),
                "commands::outputs::tests::simulator_mapped_netcdf_history_matches_npy_reference",
                "--", "--ignored", "--exact",
            ], env=env)
        print("PASS active mapped Bell/Schär netCDF threading, NPY parity, geometry, and Studio readback")

        large_base = ZFP_CONFIG.read_text(encoding="utf-8")
        large_base = large_base.replace("  nx: 16", "  nx: 130", 1)
        large_base = large_base.replace("  ny: 16", "  ny: 70", 1)
        large_base = large_base.replace("  nz: 12", "  nz: 35", 1)
        large_configs = {
            "zfp": Path(temporary) / "large_flat_zfp.yaml",
            "npy_3d": Path(temporary) / "large_flat_npy.yaml",
            "netcdf": Path(temporary) / "large_flat_netcdf.yaml",
        }
        large_configs["zfp"].write_text(large_base, encoding="utf-8")
        large_configs["npy_3d"].write_text(
            large_base.replace("  format: zfp_high", "  format: npy_3d", 1),
            encoding="utf-8",
        )
        large_configs["netcdf"].write_text(
            large_base.replace("  format: zfp_high", "  format: netcdf", 1),
            encoding="utf-8",
        )
        large_runs = {
            format_name: Path(temporary) / f"large_flat_{format_name}"
            for format_name in large_configs
        }
        hierarchy_reports = {}
        for format_name, config_path in large_configs.items():
            run_simulator(config_path, large_runs[format_name], 1)
            hierarchy_reports[format_name] = inspect_durable_hierarchies(
                large_runs[format_name],
                expected_format=format_name,
                require_lod=True,
            )
            env["NIMBUS_TEST_HIERARCHY_V2_RUN"] = str(large_runs[format_name])
            run([
                cargo, "test", "--manifest-path",
                str(ROOT / "ui" / "src-tauri" / "Cargo.toml"),
                "commands::outputs::tests::simulator_hierarchy_v2_history_reopens_with_bound_lod_dependencies",
                "--", "--ignored", "--exact",
            ], env=env)
        reference_topology = hierarchy_reports["npy_3d"]["topology"]
        for format_name, report in hierarchy_reports.items():
            require_equal(
                report["topology"], reference_topology,
                f"{format_name} hierarchy topology parity",
            )
        def lod_payloads(run_dir: Path) -> dict[Path, bytes]:
            return {
                path.relative_to(run_dir): path.read_bytes()
                for path in run_dir.glob("step_*/*.lod*.npy")
            }
        reference_lods = lod_payloads(large_runs["npy_3d"])
        for format_name, run_dir in large_runs.items():
            require_equal(
                lod_payloads(run_dir), reference_lods,
                f"{format_name} lossless LOD byte parity",
            )
        lod_count = hierarchy_reports["netcdf"]["validatedLods"]
        print(
            f"PASS large NPY/ZFP/netCDF hierarchy topology, {lod_count} checksummed "
            "lossless LOD payloads, byte parity, and bounded Studio reads"
        )

        retained_large_netcdf = Path(temporary) / "large_flat_netcdf_retained"
        run_simulator(large_configs["netcdf"], retained_large_netcdf, 1, duration=8)
        retained_steps = sorted(
            path for path in retained_large_netcdf.glob("step_*") if path.is_dir()
        )
        if len(retained_steps) != 3 or retained_steps[0].name == "step_000000":
            fail(f"large netCDF retention did not preserve a bounded newest suffix: {retained_steps}")
        retained_lods = inspect_durable_hierarchies(
            retained_large_netcdf, expected_format="netcdf", require_lod=True
        )["validatedLods"]
        env["NIMBUS_TEST_HIERARCHY_V2_RUN"] = str(retained_large_netcdf)
        run([
            cargo, "test", "--manifest-path",
            str(ROOT / "ui" / "src-tauri" / "Cargo.toml"),
            "commands::outputs::tests::simulator_hierarchy_v2_history_reopens_with_bound_lod_dependencies",
            "--", "--ignored", "--exact",
        ], env=env)
        print(
            f"PASS retained netCDF suffix preserves all {retained_lods} validated "
            "hierarchy LOD dependencies and bounded random reads"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"FAIL {error}", file=sys.stderr)
        raise SystemExit(1)
