#!/usr/bin/env python3
"""Capture bounded ordinary SHARPY NetCDF metadata type/cardinality admission."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "tests/init/fixtures/sharpy_netcdf_public_safe.nc"
YAML = ROOT / "tests/init/fixtures/validate_only_cm1_sounding.yaml"
FIXTURES = ROOT / "tests/init/fixtures"
CASES = (
    ("absent", {}, True, ""),
    ("station_char", {"station_id": (2, "S" * 1023)}, True, ""),
    ("station_string", {"station_id": (12, "METADATA_TYPES_VALID")}, True, ""),
    ("geo_alias", {"latitude": (6, [35.25]), "longitude": (4, [-97]), "elevation": (9, [324])}, True, ""),
    ("station_int", {"station_id": (4, [17])}, False, "metadata 'station_id' must be text"),
    ("timestamp_double", {"timestamp_utc": (6, [123.0])}, False, "metadata 'timestamp_utc' must be text"),
    ("latitude_char", {"latitude_deg": (2, "35.0")}, False, "metadata 'latitude_deg' must be scalar numeric"),
    ("longitude_string", {"longitude_deg": (12, "-97.0")}, False, "metadata 'longitude_deg' must be scalar numeric"),
    ("elevation_empty", {"elevation_m": (6, [])}, False, "numeric metadata must be scalar for 'elevation_m'"),
    ("invalid_primary_alias", {"latitude_deg": (2, "35.0"), "latitude": (6, [35.25])}, False,
     "metadata 'latitude_deg' must be scalar numeric"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(name: str) -> Path:
    return FIXTURES / f"sharpy_netcdf_metadata_types_{name}.nc"


def write_fixtures() -> None:
    library = ctypes.CDLL("/opt/homebrew/lib/libnetcdf.dylib")
    library.nc_strerror.restype = ctypes.c_char_p
    library.nc_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    library.nc_close.argtypes = [ctypes.c_int]
    library.nc_redef.argtypes = [ctypes.c_int]
    library.nc_enddef.argtypes = [ctypes.c_int]
    library.nc_put_att_text.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p]
    library.nc_put_att_string.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t,
                                        ctypes.POINTER(ctypes.c_char_p)]
    for function, scalar in (("nc_put_att_int", ctypes.c_int), ("nc_put_att_uint", ctypes.c_uint),
                             ("nc_put_att_double", ctypes.c_double)):
        getattr(library, function).argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                                              ctypes.c_size_t, ctypes.POINTER(scalar)]

    def checked(status: int, context: str) -> None:
        if status != 0:
            raise RuntimeError(context + ": " + library.nc_strerror(status).decode())

    for name, attributes, _, _ in CASES:
        path = fixture(name)
        if path.exists():
            raise RuntimeError("Refusing to overwrite fixture: " + str(path))
        shutil.copyfile(BASE, path)
        ncid = ctypes.c_int(-1)
        checked(library.nc_open(str(path).encode(), 1, ctypes.byref(ncid)), "open fixture")
        try:
            checked(library.nc_redef(ncid.value), "redefine fixture")
            for key, (kind, value) in attributes.items():
                encoded = key.encode()
                if kind == 2:
                    payload = value.encode()
                    status = library.nc_put_att_text(ncid.value, -1, encoded, len(payload), payload)
                elif kind == 12:
                    strings = (ctypes.c_char_p * 1)(value.encode())
                    status = library.nc_put_att_string(ncid.value, -1, encoded, 1, strings)
                else:
                    function, scalar = {4: ("nc_put_att_int", ctypes.c_int),
                                        9: ("nc_put_att_uint", ctypes.c_uint),
                                        6: ("nc_put_att_double", ctypes.c_double)}[kind]
                    values = (scalar * len(value))(*value)
                    status = getattr(library, function)(ncid.value, -1, encoded, kind, len(value), values)
                checked(status, "write " + key)
            checked(library.nc_enddef(ncid.value), "end fixture definition")
        finally:
            checked(library.nc_close(ncid.value), "close fixture")
        if path.stat().st_size > 65536:
            raise RuntimeError("Fixture exceeds 64-KiB bound")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-fixtures", action="store_true")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--expected-binary-sha256")
    parser.add_argument("--report-root", type=Path)
    args = parser.parse_args()
    if args.write_fixtures:
        write_fixtures()
        return 0
    if not args.binary or not args.expected_binary_sha256 or not args.report_root:
        parser.error("binary, expected identity and new report-root are required")
    binary = args.binary.resolve(strict=True)
    report = args.report_root.resolve()
    if report.exists() or digest(binary) != args.expected_binary_sha256:
        parser.error("use a new report and the exact expected binary")
    sources = {
        "reader": ROOT / "src/soundings/schemes/sharpy/sharpy_netcdf_dynamic_reader.cpp",
        "common": ROOT / "src/soundings/schemes/sharpy/sharpy_native_common.hpp",
        "routing": ROOT / "src/soundings/schemes/sharpy/sharpy_sounding.cpp",
        "typedInterface": ROOT / "src/soundings/schemes/sharpy/sharpy_native_reader.hpp",
        "baseFixture": BASE, "baseYaml": YAML, "qualifier": Path(__file__).resolve(), "binary": binary,
        **{name: fixture(name) for name, _, _, _ in CASES},
    }
    before = {key: digest(path) for key, path in sources.items()}
    report.mkdir(parents=True)
    inputs = report / "inputs"
    inputs.mkdir()
    marker = "file_path: tests/fixtures/soundings/cm1/HIGHCAPENEW_QUARTERCIRCLE"
    yaml = YAML.read_text()
    if yaml.count(marker) != 1:
        raise RuntimeError("Base YAML marker drifted")
    rows = []
    for name, _, expected, diagnostic in CASES:
        config = inputs / (name + ".yaml")
        config.write_text(yaml.replace(marker, 'file_path: "' + str(fixture(name)) + '"'))
        output = report / "outputs" / name
        command = [str(binary), "--headless", "--validate-only", "--config", str(config), "--outdir=" + str(output)]
        run = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=20, check=False)
        log = report / (name + ".log")
        log.write_text(run.stdout)
        published = [str(path.relative_to(report)) for path in output.rglob("*") if path.is_file()]
        accepted = run.returncode == 0
        matches = accepted == expected and not published
        matches = matches and (("[VALIDATE] PASS" in run.stdout and "netcdf_c_api" in run.stdout)
                               if expected else diagnostic in run.stdout)
        rows.append({"case": name, "command": command, "expectedAccepted": expected, "accepted": accepted,
                     "exitCode": run.returncode, "matches": matches, "diagnostic": diagnostic,
                     "published": published, "configSha256": digest(config), "logSha256": digest(log)})
    after = {key: digest(path) for key, path in sources.items()}
    status = "pass" if before == after and all(row["matches"] for row in rows) else "red"
    summary = {"schema": "nimbus.sharpy-netcdf-metadata-types.v1", "status": status,
               "binarySha256": before["binary"], "before": before, "after": after, "cases": rows,
               "claimBoundary": "Bounded native metadata type/cardinality and typed no-fallback contract only; NC_STRING preallocation and broad metadata/enumeration readiness remain open.",
               "wholeBackendReady": False, "fullyQualified": False}
    (report / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "cases": len(rows), "matches": sum(row["matches"] for row in rows)}))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
