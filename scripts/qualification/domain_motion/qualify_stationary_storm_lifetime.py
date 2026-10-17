#!/usr/bin/env python3
"""Run or inspect the oversized stationary supercell lifetime control."""

from __future__ import annotations

import argparse
import array
import ast
import csv
import hashlib
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "tests/fixtures/configs/qualification/stationary_supercell_lifetime.yaml"
EXPECTED_PUBLICATIONS = tuple(range(11))
EXPECTED_STORM_SAMPLES = 150
EXPECTED_FIELDS = {"p", "qr", "qv", "rho", "theta", "u", "v", "w"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def run(binary: Path, output: Path) -> None:
    result = subprocess.run(
        [
            str(binary),
            "--headless",
            f"--config={CONFIG}",
            f"--outdir={output}",
            "--log-profile=quiet",
            "--no-timing",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(
        result.returncode == 0,
        f"stationary lifetime run exited {result.returncode}\n{result.stdout[-12000:]}",
    )


def load_json(path: Path) -> dict:
    require(path.is_file(), f"missing JSON evidence: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON evidence {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON evidence is not an object: {path}")
    return value


def finite_csv(path: Path) -> int:
    require(path.is_file() and path.stat().st_size > 0, f"missing or empty ledger: {path}")
    rows = 0
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None, f"missing ledger header: {path}")
        for row in reader:
            rows += 1
            for key, token in row.items():
                if token is None or token == "":
                    continue
                try:
                    value = float(token)
                except ValueError:
                    continue
                require(math.isfinite(value), f"non-finite ledger value {key} in {path}")
    require(rows > 0, f"ledger has no records: {path}")
    return rows


def npy_float32_extrema(path: Path) -> tuple[tuple[int, ...], float, float]:
    with path.open("rb") as stream:
        require(stream.read(6) == b"\x93NUMPY", f"invalid NPY magic: {path}")
        version = stream.read(2)
        require(len(version) == 2, f"truncated NPY version: {path}")
        if version[0] == 1:
            raw_length = stream.read(2)
            require(len(raw_length) == 2, f"truncated NPY header length: {path}")
            header_length = struct.unpack("<H", raw_length)[0]
        else:
            raw_length = stream.read(4)
            require(len(raw_length) == 4, f"truncated NPY header length: {path}")
            header_length = struct.unpack("<I", raw_length)[0]
        header = ast.literal_eval(stream.read(header_length).decode("latin1").strip())
        require(header.get("descr") in {"<f4", "=f4", "|f4"}, f"unsupported NPY dtype in {path}")
        require(header.get("fortran_order") is False, f"Fortran-order NPY is unsupported: {path}")
        shape = tuple(int(value) for value in header.get("shape", ()))
        count = math.prod(shape)
        payload = stream.read()
        require(len(payload) == count * 4, f"wrong NPY payload length: {path}")

    values = array.array("f")
    values.frombytes(payload)
    if sys.byteorder != "little" and header.get("descr") == "<f4":
        values.byteswap()
    require(all(math.isfinite(value) for value in values), f"non-finite values in {path}")
    return shape, float(min(values)), float(max(values))


def inspect(output: Path, binary: Path | None) -> dict:
    root_manifest = load_json(output / "nimbus_output.json")
    require(root_manifest.get("publication_state") == "ready", "root publication is not ready")
    require(root_manifest.get("manifest_version") in {6, 7}, "unexpected root manifest version")

    step_dirs = sorted(path for path in output.glob("step_*") if path.is_dir())
    actual_publications = tuple(
        int(path.name.removeprefix("step_")) for path in step_dirs
    )
    require(
        actual_publications == EXPECTED_PUBLICATIONS,
        f"wrong publication sequence: {actual_publications}",
    )

    field_extrema: dict[str, dict[str, float]] = {}
    max_updraft_output_ms = -math.inf
    for publication, step_dir in zip(actual_publications, step_dirs):
        manifest = load_json(step_dir / "manifest.json")
        require(
            manifest.get("publication_state") == "complete",
            f"publication {publication} is incomplete",
        )
        require(
            manifest.get("step_index") == publication,
            f"publication {publication} manifest identity mismatch",
        )
        require(
            manifest.get("simulation_time_s") == publication * 450,
            f"publication {publication} has incorrect simulation time",
        )
        require(
            set(manifest.get("published_fields", [])) == EXPECTED_FIELDS,
            f"publication {publication} has the wrong field set",
        )
        grid = manifest.get("grid", {})
        require(grid.get("dimensions") == {"x": 267, "y": 208, "z": 50}, "wrong reference grid")
        horizontal_frame = grid.get("horizontal_frame", {})
        require(horizontal_frame.get("motion_mode") == "stationary", "reference frame mode changed")
        require(horizontal_frame.get("grid_origin_earth_m") == [0, 0], "stationary origin moved")
        require(horizontal_frame.get("frame_velocity_earth_ms") == [0, 0], "stationary frame velocity changed")

        validation = load_json(step_dir / "validation_report.json")
        require(
            validation.get("failed") is False,
            f"strict validation failed at publication {publication}",
        )
        require(
            validation.get("guard_mode") == "strict",
            f"publication {publication} was not strictly guarded",
        )

        for field in sorted(EXPECTED_FIELDS):
            path = step_dir / f"{field}.npy"
            require(path.is_file(), f"missing {field} at publication {publication}")
            shape, low, high = npy_float32_extrema(path)
            require(
                shape == (267, 208, 50),
                f"wrong {field} shape at publication {publication}",
            )
            extrema = field_extrema.setdefault(field, {"min": low, "max": high})
            extrema["min"] = min(extrema["min"], low)
            extrema["max"] = max(extrema["max"], high)
            if field == "w":
                max_updraft_output_ms = max(max_updraft_output_ms, high)

    records = [load_json(path) for path in sorted((output / "storm_track").glob("track_step_*.json"))]
    require(len(records) == EXPECTED_STORM_SAMPLES, f"expected 150 storm samples, found {len(records)}")
    require(records[-1].get("completed_step") == 3000, "storm history does not reach the final step")
    require(records[-1].get("simulation_time_s") == 4500, "storm history does not reach 4500 s")

    detected = []
    motion = []
    minimum_boundary_m = math.inf
    maximum_tracker_updraft_ms = 0.0
    maximum_absolute_uh_m2s2 = 0.0
    for record in records:
        require(record.get("schema") == "nimbus.storm_feature", "wrong storm record schema")
        require(record.get("schema_version") == 12, "wrong storm record schema version")
        require(record.get("tracking_source") == "root_grid", "wrong storm tracking source")
        require(record.get("prognostic_state_mutation") is False, "tracker mutated prognostic state")
        if record.get("selected_feature_available") is True:
            require(record.get("selected_feature_role") == "continuity_selected_candidate",
                    "selected storm fields have ambiguous provenance")
            require(record.get("boundary_contact") is not True, "storm contacted a physical boundary")
            require(record.get("inner_margin_breached") is not True, "storm breached the planned inner margin")
            detected.append(record)
            distance = record.get("distance_to_physical_boundary_m", {}).get("nearest")
            if distance is not None:
                minimum_boundary_m = min(minimum_boundary_m, float(distance))
            updraft = record.get("maximum_updraft_ms")
            if updraft is not None:
                maximum_tracker_updraft_ms = max(maximum_tracker_updraft_ms, float(updraft))
            uh = record.get("rotation", {}).get("maximum_absolute_updraft_helicity_m2s2")
            if uh is not None:
                maximum_absolute_uh_m2s2 = max(maximum_absolute_uh_m2s2, float(uh))
        else:
            require(record.get("selected_feature_role") is None, "unavailable storm has a selected role")
            require(record.get("distance_to_physical_boundary_m", {}).get("nearest") is None,
                    "unassociated snapshot substituted a boundary envelope")
            require(record.get("boundary_contact") is None and record.get("inner_margin_breached") is None,
                    "unassociated snapshot invented selected-storm boundary state")
        if record.get("motion_available") is True:
            motion.append(record)

    require(len(detected) >= 30, "control did not sustain coherent convection long enough")
    require(len(motion) >= 10, "control did not establish a sustained motion history")
    require(maximum_tracker_updraft_ms >= 10.0, "control never developed a 10 m/s coherent updraft")
    require(minimum_boundary_m >= 10000.0, "storm violated the declared 10 km lifetime buffer")

    checkpoints = sorted((output / "checkpoints").glob("checkpoint_step_*.nimbus"))
    checkpoint_steps = tuple(int(path.stem.removeprefix("checkpoint_step_")) for path in checkpoints)
    require(checkpoint_steps == (1500, 3000), f"wrong checkpoint cadence: {checkpoint_steps}")

    ledgers = {}
    for name in (
        "budget.csv",
        "energy_budget.csv",
        "moisture_budget.csv",
        "momentum_budget.csv",
        "thermodynamic_source_budget.csv",
    ):
        ledgers[name] = finite_csv(output / name)
    inactive_radiation = output / "radiation_budget.csv"
    require(inactive_radiation.is_file(), "inactive radiation ledger was not created")
    ledgers["radiation_budget.csv"] = 0

    return {
        "schema": "nimbus.stationary_storm_lifetime_qualification",
        "schema_version": 1,
        "status": "pass",
        "scientific_scope": "exact_stationary_domain_motion_control_exploratory_kessler",
        "config_sha256": digest(CONFIG),
        "binary_sha256": digest(binary) if binary is not None else None,
        "grid": {"nx": 267, "ny": 208, "nz": 50, "dx_m": 600, "dy_m": 600, "dz_m": 320},
        "duration_s": 4500,
        "completed_steps": 3000,
        "publication_indices": list(actual_publications),
        "publication_times_s": [index * 450 for index in actual_publications],
        "storm_samples": len(records),
        "coherent_samples": len(detected),
        "motion_available_samples": len(motion),
        "minimum_storm_boundary_distance_m": minimum_boundary_m,
        "maximum_tracker_updraft_ms": maximum_tracker_updraft_ms,
        "maximum_output_updraft_ms": max_updraft_output_ms,
        "maximum_absolute_updraft_helicity_m2s2": maximum_absolute_uh_m2s2,
        "field_extrema": field_extrema,
        "checkpoint_steps": list(checkpoint_steps),
        "ledger_rows": ledgers,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin" / "nimbus_sim_cpu")
    parser.add_argument("--output", type=Path, help="inspect an existing completed output instead of running")
    args = parser.parse_args()
    binary = args.binary.resolve()
    require(binary.is_file(), f"missing simulator binary: {binary}")

    if args.output is not None:
        report = inspect(args.output.resolve(), binary)
    else:
        with tempfile.TemporaryDirectory(prefix="nimbus_stationary_lifetime_") as temporary:
            output = Path(temporary)
            run(binary, output)
            report = inspect(output, binary)

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[stationary-storm-lifetime] FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
