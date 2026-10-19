#!/usr/bin/env python3
"""Check actual public fields, units, geometry and finite values for Metal tuples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tests/science"))
sys.path.insert(0, str(ROOT / "scripts/qualification/output"))
sys.path.insert(0, str(ROOT))
from npy_support import read_float32_npy
from qualify_netcdf_history import ncdump_values
from scripts.tests.catalog.qualify_catalog_bounded_runs import write_json_atomic

FIELDS = {"p": "Pa", "qr": "kg/kg", "qv": "kg/kg", "rho": "kg/m^3",
          "theta": "K", "u": "m/s", "v": "m/s", "w": "m/s"}


def identity(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def check(row: dict, output_format: str) -> dict:
    output = Path(row["config"]["path"]).parent / "output"
    frame = output / "step_000001"
    manifest = frame / "manifest.json"
    document = json.loads(manifest.read_text())
    if document.get("schema") != "nimbus.step" or document.get("publication_state") != "complete" or \
            document.get("data_product", {}).get("storage_format") != output_format or \
            document.get("step_index") != 1 or document.get("simulation_time_s") != 1 or \
            set(document.get("published_fields", [])) != set(FIELDS) or \
            document.get("grid", {}).get("dimensions") != {"x": 18, "y": 18, "z": 18} or \
            document["grid"].get("spacing_m") != {"x": 500, "y": 500, "z": 250} or \
            document["grid"].get("native_model_staggering") != "c_grid" or \
            document["grid"].get("publication_staggering") != "cell_centered":
        raise RuntimeError(f"{row['id']} published a different field or coordinate contract")
    units = {field["field_id"]: field["units"] for field in document["fields"]}
    if any(units.get(name) != unit for name, unit in FIELDS.items()):
        raise RuntimeError(f"{row['id']} published inconsistent field units")
    fields = {}
    if output_format == "npy_3d":
        files = {path.name for path in frame.glob("*.npy")}
        if files != {name + ".npy" for name in FIELDS}:
            raise RuntimeError(f"{row['id']} published a different physical file set")
        for name in FIELDS:
            path = frame / f"{name}.npy"
            shape, _, values = read_float32_npy(path)
            if shape != (18, 18, 18) or any(not math.isfinite(value) for value in values):
                raise RuntimeError(f"{row['id']} published a malformed or nonfinite {name} field")
            fields[name] = identity(path)
        return {"id": row["id"], "status": "pass", "manifest": identity(manifest), "fields": fields}
    if output_format != "netcdf":
        raise RuntimeError("unknown generated Metal output format")
    history = frame / "history.nc"
    header = subprocess.run(["ncdump", "-h", str(history)], text=True,
                            capture_output=True, check=True, timeout=60).stdout
    dump = subprocess.run(["ncdump", "-p", "9,17", "-v", ",".join(FIELDS), str(history)],
                          text=True, capture_output=True, check=True, timeout=60).stdout
    for name, unit in FIELDS.items():
        if f"float {name}(time, x, y, z)" not in header or \
                f'{name}:units = "{unit}"' not in header or \
                f'{name}:grid_location = "cell_center"' not in header:
            raise RuntimeError(f"{row['id']} netCDF header lacks {name} metadata")
        values = ncdump_values(dump, name)
        if len(values) != 18 * 18 * 18 or any(not math.isfinite(value) for value in values):
            raise RuntimeError(f"{row['id']} netCDF {name} data are malformed or nonfinite")
        fields[name] = {"units": unit, "values": len(values)}
    if list(frame.glob("*.npy")):
        raise RuntimeError(f"{row['id']} netCDF publication contains NPY state fields")
    return {"id": row["id"], "status": "pass", "manifest": identity(manifest),
            "history": identity(history), "fields": fields}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.runtime_summary.read_text())
    if source.get("status") != "complete" or source.get("executionMode") != "qualified" or \
            source.get("counts") != {"pass": 2016}:
        raise RuntimeError("complete qualified Metal runtime evidence is required")
    output = args.output.resolve()
    if output.exists():
        parser.error("use a new output path to preserve existing output evidence")
    inputs = {"runtime": identity(args.runtime_summary.resolve()),
              "validator": identity(Path(__file__).resolve())}
    output_format = source.get("outputFormat", "npy_3d")
    records = []
    for index, row in enumerate(source["results"], start=1):
        records.append(check(row, output_format))
        if index % 100 == 0:
            print(f"checked {index}/2016 output contracts", flush=True)
    if len(records) != 2016 or len({record["id"] for record in records}) != 2016:
        raise RuntimeError("qualified Metal output coverage is incomplete")
    report = {"schema": "nimbus.generated-metal-output.v1", "status": "pass",
              "outputFormat": output_format,
              "claimBoundary": "Canonical minimal history fields, units and geometry on 2016 qualified Metal tuples; "
              "not all retained output formats, presets or product combinations.",
              "inputs": inputs, "cases": len(records), "results": records}
    write_json_atomic(output, report)
    print(f"qualified generated Metal {output_format} output: 2016 passed", flush=True)


if __name__ == "__main__":
    main()
