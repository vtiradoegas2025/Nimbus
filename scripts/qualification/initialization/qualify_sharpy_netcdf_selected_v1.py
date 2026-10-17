#!/usr/bin/env python3
"""Write bounded selection fixtures or capture ordinary native NetCDF admission."""
import argparse
import ctypes
import ctypes.util
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "tests/init/fixtures/sharpy_netcdf_public_safe.nc"
YAML = ROOT / "tests/init/fixtures/validate_only_cm1_sounding.yaml"
FIXTURES = ROOT / "tests/init/fixtures"
FIELDS = ("height_m", "pressure_hpa", "temperature_k", "dewpoint_k", "wind_speed_ms", "wind_direction_deg")
LOWER = ("z", "pres", "temp", "td", "wspd", "wdir")
CASES = ("dense", "priority", "depth_first", "normalized", "selected_oversized")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(name):
    return FIXTURES / ("sharpy_netcdf_selected_" + name + ".nc")


def write_fixtures():
    found = ctypes.util.find_library("netcdf")
    lib = ctypes.CDLL(found or "/opt/homebrew/lib/libnetcdf.dylib")
    lib.nc_strerror.restype = ctypes.c_char_p
    signatures = {
        "nc_open": [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)],
        "nc_create": [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)],
        "nc_close": [ctypes.c_int], "nc_enddef": [ctypes.c_int],
        "nc_def_dim": [ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)],
        "nc_def_grp": [ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)],
        "nc_def_var": [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                       ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)],
        "nc_inq_varid": [ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)],
        "nc_get_var_double": [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)],
        "nc_put_var_double": [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)],
        "nc_put_att_string": [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t,
                              ctypes.POINTER(ctypes.c_char_p)],
    }
    for name, signature in signatures.items():
        getattr(lib, name).argtypes = signature

    def check(status):
        if status != 0:
            raise RuntimeError(lib.nc_strerror(status).decode())

    source = ctypes.c_int()
    check(lib.nc_open(str(BASE).encode(), 0, ctypes.byref(source)))
    values = []
    try:
        for field in FIELDS:
            var = ctypes.c_int()
            check(lib.nc_inq_varid(source.value, field.encode(), ctypes.byref(var)))
            data = (ctypes.c_double * 33)()
            check(lib.nc_get_var_double(source.value, var.value, data))
            values.append(list(data))
    finally:
        check(lib.nc_close(source.value))
    for name in CASES:
        path = fixture(name)
        if path.exists():
            raise RuntimeError("Refusing to overwrite fixture: " + str(path))
        ncid = ctypes.c_int()
        check(lib.nc_create(str(path).encode(), 0x1000 | 4, ctypes.byref(ncid)))
        writes = []
        try:
            level = ctypes.c_int()
            single = ctypes.c_int()
            check(lib.nc_def_dim(ncid.value, b"level", 100001 if name == "selected_oversized" else 33,
                                 ctypes.byref(level)))
            check(lib.nc_def_dim(ncid.value, b"single", 1, ctypes.byref(single)))

            def variables(group, names, altered=False):
                for index, field in enumerate(names):
                    var = ctypes.c_int()
                    dims = (ctypes.c_int * 1)(level.value)
                    check(lib.nc_def_var(group, field.encode(), 6, 1, dims, ctypes.byref(var)))
                    if name != "selected_oversized":
                        data = list(values[index])
                        if altered:
                            data = [value + (1 if index == 0 else .125) for value in data]
                        writes.append((group, var.value, data))

            if name == "dense":
                for index in range(256):
                    var = ctypes.c_int()
                    dims = (ctypes.c_int * 16)(*[single.value] * 15, level.value)
                    check(lib.nc_def_var(ncid.value, ("unused" + str(index)).encode(), 6, 16, dims, ctypes.byref(var)))
                variables(ncid.value, FIELDS)
            elif name == "priority":
                variables(ncid.value, LOWER, True)
                variables(ncid.value, FIELDS)
            elif name == "depth_first":
                variables(ncid.value, LOWER, True)
                first, deep, second = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
                check(lib.nc_def_grp(ncid.value, b"first", ctypes.byref(first)))
                check(lib.nc_def_grp(first.value, b"deep", ctypes.byref(deep)))
                variables(deep.value, FIELDS)
                check(lib.nc_def_grp(ncid.value, b"second", ctypes.byref(second)))
                variables(second.value, FIELDS, True)
            elif name == "normalized":
                variables(ncid.value, ["Sensor_" + field.upper() for field in FIELDS])
            else:
                variables(ncid.value, FIELDS)
            strings = (ctypes.c_char_p * 1)(b"STREAMING_NATIVE")
            check(lib.nc_put_att_string(ncid.value, -1, b"station_id", 1, strings))
            check(lib.nc_enddef(ncid.value))
            for group, var, data in writes:
                check(lib.nc_put_var_double(group, var, (ctypes.c_double * 33)(*data)))
        finally:
            check(lib.nc_close(ncid.value))
        if path.stat().st_size > 2 * 1024 * 1024:
            raise RuntimeError("Fixture exceeds two-MiB bound")


def main():
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
        parser.error("binary, identity and new report root required")
    binary, report = args.binary.resolve(strict=True), args.report_root.resolve()
    if report.exists() or sha(binary) != args.expected_binary_sha256:
        parser.error("new report/exact expected binary required")
    paths = {"binary": binary, "qualifier": Path(__file__).resolve(), "baseYaml": YAML,
             "reader": ROOT / "src/soundings/schemes/sharpy/sharpy_netcdf_dynamic_reader.cpp",
             "common": ROOT / "src/soundings/schemes/sharpy/sharpy_native_common.hpp",
             **{name: fixture(name) for name in CASES}}
    before = {name: sha(path) for name, path in paths.items()}
    report.mkdir(parents=True)
    marker = "file_path: tests/fixtures/soundings/cm1/HIGHCAPENEW_QUARTERCIRCLE"
    source = YAML.read_text()
    if source.count(marker) != 1:
        raise RuntimeError("Base YAML marker drift")
    rows = []
    for name in CASES:
        config = report / (name + ".yaml")
        config.write_text(source.replace(marker, 'file_path: "' + str(fixture(name)) + '"'))
        output = report / "outputs" / name
        command = [str(binary), "--headless", "--validate-only", "--config", str(config), "--outdir=" + str(output)]
        result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=20, check=False)
        log = report / (name + ".log")
        log.write_text(result.stdout)
        expected = name != "selected_oversized"
        diagnostic = "[VALIDATE] PASS" if expected else "100,000-level safety limit"
        published = [str(path.relative_to(report)) for path in output.rglob("*") if path.is_file()]
        matched = (result.returncode == 0) == expected and diagnostic in result.stdout and not published
        rows.append({"case": name, "command": command, "accepted": result.returncode == 0,
                     "expectedAccepted": expected, "exitCode": result.returncode, "matches": matched,
                     "configSha256": sha(config), "logSha256": sha(log), "published": published})
    after = {name: sha(path) for name, path in paths.items()}
    status = "pass" if before == after and all(row["matches"] for row in rows) else "red"
    summary = {"schema": "nimbus.sharpy-selected-netcdf-public.v1", "status": status,
               "before": before, "after": after, "cases": rows,
               "enumerationResourcesClosed": False, "ncStringPreallocationClosed": False,
               "wholeBackendReady": False, "fullyQualified": False}
    (report / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "cases": len(rows)}))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
