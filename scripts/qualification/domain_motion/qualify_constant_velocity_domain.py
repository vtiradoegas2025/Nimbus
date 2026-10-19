#!/usr/bin/env python3
"""Qualify flat CPU constant translation by dry/moist A/C-grid equivalence."""

from __future__ import annotations

import argparse
import array
import ast
import csv
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "tests/fixtures/configs/qualification/constant_velocity_cgrid.yaml"
AGRID_CONFIG = ROOT / "tests/fixtures/configs/qualification/constant_velocity_agrid_ready.yaml"
LONG_CONFIG = ROOT / "tests/fixtures/configs/qualification/constant_velocity_supercell_lifetime.yaml"
FIELDS = ("p", "qr", "qv", "rho", "theta", "u", "v", "w")
LEDGERS = (
    "budget.csv",
    "energy_budget.csv",
    "moisture_budget.csv",
    "momentum_budget.csv",
    "radiation_budget.csv",
    "thermodynamic_source_budget.csv",
)
REQUIRED_COMPLETE_COLUMNS = (
    "completed_step_energy_step_complete",
    "completed_step_energy_cumulative_complete",
    "acoustic_v1_complete",
    "acoustic_v1_cumulative_complete",
    "slow_momentum_energy_step_complete",
    "slow_momentum_energy_cumulative_complete",
    "advection_thermodynamic_boundary_reports_complete",
    "advection_thermodynamic_moist_energy_reports_complete",
)
FRAME_U_MS = 7.0
FRAME_V_MS = -3.0
# Nimbus stores Earth-relative winds in FP32. Adding the frame offset changes
# the ulp of each momentum update even when the grid-relative equation is
# identical, so a bitwise Galilean comparison is neither possible nor honest.
# These absolute 20-s coupled-run envelopes cap both localized error and bulk
# drift; they are intentionally far below meteorologically material scales.
FP32_ENVELOPE = {
    "p": (2.0, 0.25),
    "qr": (2.0e-7, 2.0e-8),
    "qv": (1.2e-5, 1.0e-6),
    "rho": (4.0e-4, 4.0e-5),
    "theta": (0.12, 0.012),
    "u": (0.012, 0.001),
    "v": (0.008, 0.001),
    "w": (0.02, 0.002),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def replace_once(text: str, old: str, new: str) -> str:
    require(text.count(old) == 1, f"qualification fixture must contain exactly one {old!r}")
    return text.replace(old, new, 1)


def stationary_counterpart(text: str) -> str:
    """Remove the coordinate boost while preserving grid-relative initial wind."""
    replacements = (
        ("  mode: constant_velocity\n", "  mode: stationary\n"),
        ("    u_ms: 7\n", "    u_ms: 0\n"),
        ("    v_ms: -3\n", "    v_ms: 0\n"),
        ("    u_sfc_ms: 12\n", "    u_sfc_ms: 5\n"),
        ("    v_sfc_ms: -1\n", "    v_sfc_ms: 2\n"),
        ("    u_1km_ms: 12\n", "    u_1km_ms: 5\n"),
        ("    v_1km_ms: -1\n", "    v_1km_ms: 2\n"),
        ("    u_6km_ms: 12\n", "    u_6km_ms: 5\n"),
        ("    v_6km_ms: -1\n", "    v_6km_ms: 2\n"),
    )
    for old, new in replacements:
        text = replace_once(text, old, new)
    return text


def simple_agrid_variant(text: str) -> str:
    text = replace_once(text, "  staggering: c_grid\n", "  staggering: collocated\n")
    text = replace_once(
        text,
        "  scheme: klemp_wilhelmson_split_cartesian_cgrid\n",
        "  scheme: klemp_wilhelmson_split_cartesian_agrid\n",
    )
    text = replace_once(text, "  scheme: smagorinsky\n", "  scheme: none\n")
    text = replace_once(
        text, "    moisture_qv_kgkg: 0.020\n", "    moisture_qv_kgkg: 0.004\n"
    )
    text = replace_once(
        text,
        "      dtheta_k: 2\n",
        "      dtheta_k: 2\n      cloud_water_kgkg: 0.001\n",
    )
    return replace_once(
        text,
        "  execution:\n    mode: exploratory\n    acknowledge_maturity_limits: false\n",
        "  execution:\n    mode: qualified\n",
    )


def dry_variant(text: str) -> str:
    text = replace_once(
        text,
        "microphysics:\n"
        "  scheme: kessler\n"
        "  shared:\n"
        "    cloud_water_threshold_kgkg: 0.00001\n"
        "    autoconversion_rate_s_inv: 0.02\n",
        "microphysics:\n  scheme: none\n",
    )
    if "    moisture_qv_kgkg: 0.020\n" in text:
        text = replace_once(
            text, "    moisture_qv_kgkg: 0.020\n", "    moisture_qv_kgkg: 0.004\n"
        )
    if "      cloud_water_kgkg: 0.001\n" in text:
        text = replace_once(text, "      cloud_water_kgkg: 0.001\n", "")
    return text


def mixed_phase_variant(text: str, scheme: str) -> str:
    lin = (
        "microphysics:\n"
        "  scheme: lin\n"
        "  shared:\n"
        "    cloud_water_threshold_kgkg: 0.0001\n"
        "    autoconversion_rate_s_inv: 0.01\n"
        "  lin:\n"
        "    accretion_coefficient: 2.2\n"
        "    evaporation_rate_s_inv: 0.001\n"
    )
    if scheme == "thompson":
        replacement = (
            "microphysics:\n"
            "  scheme: thompson\n"
            "  shared:\n"
            "    cloud_water_threshold_kgkg: 0.001\n"
            "    autoconversion_rate_s_inv: 0.001\n"
            "  thompson:\n"
            "    evaporation_rate_s_inv: 0.001\n"
            "    water_friendly_aerosol_surface_m3: 350000000\n"
            "    ice_friendly_aerosol_surface_m3: 2000000\n"
        )
    elif scheme == "milbrandt":
        replacement = (
            "microphysics:\n"
            "  scheme: milbrandt\n"
            "  milbrandt:\n"
            "    hail_processes: true\n"
        )
    else:
        raise ValueError(f"unsupported mixed-phase variant: {scheme}")
    return replace_once(text, lin, replacement)


def tke_variant(text: str) -> str:
    return replace_once(
        text,
        "turbulence:\n"
        "  scheme: smagorinsky\n"
        "  mode: 2d\n"
        "  Cs: 0.18\n"
        "  Pr_t: 0.3333333333333333\n"
        "  Sc_t: 0.3333333333333333\n",
        "turbulence:\n"
        "  scheme: tke\n"
        "  mode: 2d\n"
        "  filter_width: cubic_root\n"
        "  stability_correction: tke_based\n"
        "  Cs: 0.18\n"
        "  Pr_t: 0.3333333333333333\n"
        "  Sc_t: 0.3333333333333333\n",
    )


def pbl_variant(text: str, scheme: str) -> str:
    text = replace_once(
        text,
        "surface:\n  scheme: none\n  apply_fluxes: false\n",
        "surface:\n  scheme: slab\n  apply_fluxes: true\n",
    )
    top = "tke" if scheme == "mynn" else "ri"
    text = replace_once(
        text,
        "boundary_layer:\n  scheme: none\n",
        f"boundary_layer:\n  scheme: {scheme}\n  pbl_top_method: {top}\n",
    )
    if scheme == "slab":
        text = replace_once(
            text,
            "radiation:\n  scheme: rrtmg\n",
            "radiation:\n  scheme: simple_grey\n",
        )
        text = replace_once(
            text,
            "  rrtmg_ozone_vmr: 0.000004\n",
            "  tau_lw_ref: 2.5\n  n_lw: 1\n",
        )
    return text


def initial_chaos_variant(text: str) -> str:
    return replace_once(
        text,
        "  diagnostics:\n",
        "  chaos:\n"
        "    scheme: initial_conditions\n"
        "    seed: 918273\n"
        "    member_id: 4\n"
        "    apply_to_ic: u,v,w,theta\n"
        "  diagnostics:\n",
    )


def evolving_chaos_variant(text: str, scheme: str) -> str:
    text = pbl_variant(text, "mynn")
    targets = (
        "    apply_to_tendencies: pbl\n"
        if scheme == "boundary_layer"
        else "    apply_to_ic: u,v,w,theta\n"
             "    apply_to_tendencies: microphysics,pbl,diffusion\n"
    )
    return replace_once(
        text,
        "  diagnostics:\n",
        "  chaos:\n"
        f"    scheme: {scheme}\n"
        "    seed: 918273\n"
        "    member_id: 4\n"
        + targets
        + "  diagnostics:\n",
    )


def run(
    binary: Path,
    config: Path,
    output: Path,
    threads: int,
    checkpoint_every: int = 0,
    resume: Path | None = None,
    duration_s: int = 20,
) -> None:
    command = [
        str(binary),
        "--headless",
        f"--config={config}",
        f"--outdir={output}",
        "--compute-backend=cpu",
        f"--threads={threads}",
        f"--duration={duration_s}",
        f"--write-every={duration_s}",
        f"--checkpoint-every={checkpoint_every}",
        "--log-profile=quiet",
        "--no-timing",
    ]
    if resume is not None:
        command.append(f"--resume={resume}")
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(
        result.returncode == 0,
        f"{config} run exited {result.returncode}\n{result.stdout[-12000:]}",
    )


def validate_long_lifetime_preflight(binary: Path) -> None:
    result = subprocess.run(
        [
            str(binary),
            "--headless",
            f"--config={LONG_CONFIG}",
            "--compute-backend=cpu",
            "--validate-only",
            "--log-profile=quiet",
            "--no-timing",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(result.returncode == 0, "long translating preflight failed\n" + result.stdout[-8000:])
    require(
        "physical domain=268x246 cells, required=115x115 physical cells" in result.stdout
        and result.stdout.count("; PASS") >= 1,
        "long translating preflight did not use the selected grid-relative frame\n"
        + result.stdout[-8000:],
    )


def validate_agrid_public_scope(binary: Path, root: Path) -> None:
    source = AGRID_CONFIG.read_text(encoding="utf-8")
    parameter_variant = replace_once(
        replace_once(source, "    u_ms: 7\n", "    u_ms: 5\n"),
        "    v_ms: -3\n", "    v_ms: 2\n",
    )
    parameter_path = root / "agrid-parameter-variant.yaml"
    parameter_path.write_text(parameter_variant, encoding="utf-8")
    admitted = subprocess.run(
        [str(binary), "--validate-only", "--config", str(parameter_path)],
        cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    require(
        admitted.returncode == 0 and "[VALIDATE] PASS" in admitted.stdout,
        "finite A-grid frame-velocity variant was rejected\n" + admitted.stdout,
    )

    mutations = (
        ("backend", "    backend: cpu\n", "    backend: metal\n"),
        (
            "terrain",
            "radiation:\n",
            "terrain:\n  scheme: bell\n\nradiation:\n",
        ),
    )
    for index, (label, old, new) in enumerate(mutations):
        config = root / f"agrid-invalid-{index}-{label}.yaml"
        config.write_text(replace_once(source, old, new), encoding="utf-8")
        rejected = subprocess.run(
            [str(binary), "--validate-only", "--config", str(config)],
            cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
        require(
            rejected.returncode != 0,
            f"unsupported A-grid constant-velocity {label} neighbor was admitted",
        )


def load_json(path: Path) -> dict:
    require(path.is_file(), f"missing JSON evidence: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON evidence is not an object: {path}")
    return value


def load_npy_float32(path: Path) -> tuple[tuple[int, ...], array.array]:
    with path.open("rb") as stream:
        require(stream.read(6) == b"\x93NUMPY", f"invalid NPY magic: {path}")
        version = stream.read(2)
        require(len(version) == 2, f"truncated NPY version: {path}")
        length_bytes = 2 if version[0] == 1 else 4
        raw_length = stream.read(length_bytes)
        require(len(raw_length) == length_bytes, f"truncated NPY header length: {path}")
        header_length = struct.unpack("<H" if length_bytes == 2 else "<I", raw_length)[0]
        header = ast.literal_eval(stream.read(header_length).decode("latin1").strip())
        require(header.get("descr") in {"<f4", "=f4", "|f4"}, f"unsupported NPY dtype: {path}")
        require(header.get("fortran_order") is False, f"Fortran-order NPY is unsupported: {path}")
        shape = tuple(int(item) for item in header.get("shape", ()))
        payload = stream.read()
    require(len(payload) == math.prod(shape) * 4, f"wrong NPY payload length: {path}")
    values = array.array("f")
    values.frombytes(payload)
    if sys.byteorder != "little" and header.get("descr") == "<f4":
        values.byteswap()
    require(all(math.isfinite(value) for value in values), f"non-finite value in {path}")
    return shape, values


def compare_field(stationary: Path, moving: Path, field: str) -> dict[str, float]:
    stationary_shape, stationary_values = load_npy_float32(stationary)
    moving_shape, moving_values = load_npy_float32(moving)
    require(stationary_shape == moving_shape, f"{field} shape mismatch")
    offset = FRAME_U_MS if field == "u" else FRAME_V_MS if field == "v" else 0.0
    max_abs = 0.0
    max_index = 0
    sum_squared = 0.0
    for index, (stationary_value, moving_value) in enumerate(zip(stationary_values, moving_values)):
        difference = abs((moving_value - offset) - stationary_value)
        if difference > max_abs:
            max_abs = difference
            max_index = index
        sum_squared += difference * difference
    maximum_tolerance, rms_tolerance = FP32_ENVELOPE[field]
    ny, nz = stationary_shape[1], stationary_shape[2]
    location = [max_index // (ny * nz), (max_index // nz) % ny, max_index % nz]
    rms_error = math.sqrt(sum_squared / len(stationary_values))
    return {
        "maximum_absolute_error": max_abs,
        "root_mean_square_error": rms_error,
        "maximum_error_index": location,
        "maximum_tolerance": maximum_tolerance,
        "rms_tolerance": rms_tolerance,
        "passed": max_abs <= maximum_tolerance and rms_error <= rms_tolerance,
    }


def inspect_pair(stationary: Path, moving: Path, duration_s: int) -> dict:
    result: dict[str, dict] = {}
    for publication in (0, 1):
        step = f"step_{publication:06d}"
        stationary_manifest = load_json(stationary / step / "manifest.json")
        moving_manifest = load_json(moving / step / "manifest.json")
        require(stationary_manifest.get("manifest_version") in {6, 7}, "wrong stationary manifest version")
        require(moving_manifest.get("manifest_version") in {6, 7}, "wrong moving manifest version")
        stationary_frame = stationary_manifest.get("grid", {}).get("horizontal_frame", {})
        moving_frame = moving_manifest.get("grid", {}).get("horizontal_frame", {})
        require(stationary_frame.get("motion_mode") == "stationary", "stationary mode provenance missing")
        require(moving_frame.get("motion_mode") == "constant_velocity", "moving mode provenance missing")
        require(moving_frame.get("frame_velocity_earth_ms") == [7, -3], "moving frame velocity is wrong")
        expected_origin = [
            publication * duration_s * FRAME_U_MS,
            publication * duration_s * FRAME_V_MS,
        ]
        require(moving_frame.get("grid_origin_earth_m") == expected_origin, "moving Earth origin is wrong")
        require(
            moving_frame.get("published_horizontal_wind") == "earth_relative"
            and moving_frame.get("horizontal_transport_velocity") == "grid_relative",
            "wind semantic provenance is incomplete",
        )
        publication_result: dict[str, dict[str, float]] = {}
        for field in FIELDS:
            publication_result[field] = compare_field(
                stationary / step / f"{field}.npy",
                moving / step / f"{field}.npy",
                field,
            )
        result[step] = publication_result
    failures = [
        f"{step}/{field}: max={metrics['maximum_absolute_error']:.9g}/"
        f"{metrics['maximum_tolerance']:.9g} at {metrics['maximum_error_index']}, "
        f"rms={metrics['root_mean_square_error']:.9g}/{metrics['rms_tolerance']:.9g}"
        for step, fields in result.items()
        for field, metrics in fields.items()
        if not metrics["passed"]
    ]
    require(not failures, "Galilean field mismatches:\n  " + "\n  ".join(failures))
    return result


def inspect_thread_identity(one_thread: Path, four_threads: Path) -> dict:
    result: dict[str, dict[str, bool]] = {}
    failures: list[str] = []
    for publication in (0, 1):
        step = f"step_{publication:06d}"
        fields: dict[str, bool] = {}
        for field in FIELDS:
            one_shape, one_values = load_npy_float32(one_thread / step / f"{field}.npy")
            four_shape, four_values = load_npy_float32(four_threads / step / f"{field}.npy")
            identical = one_shape == four_shape and one_values == four_values
            fields[field] = identical
            if not identical:
                failures.append(f"{step}/{field}")
        result[step] = fields
    require(not failures, "one/four-thread field mismatches: " + ", ".join(failures))
    return result


def checkpoint_at(run_dir: Path, simulation_time_s: float) -> Path:
    matches: list[Path] = []
    for sidecar in sorted((run_dir / "checkpoints").glob("*.nimbus.json")):
        record = load_json(sidecar)
        if abs(float(record.get("simulation_time_s", -1.0)) - simulation_time_s) <= 1.0e-9:
            checkpoint = run_dir / "checkpoints" / str(record.get("checkpoint_file", ""))
            require(checkpoint.is_file(), f"missing checkpoint named by {sidecar}")
            matches.append(checkpoint)
    require(len(matches) == 1, f"expected one {simulation_time_s:g}s checkpoint in {run_dir}")
    return matches[0]


def final_frame(run_dir: Path) -> Path:
    frames = sorted(
        path for path in run_dir.iterdir()
        if path.is_dir() and path.name.startswith("step_")
    )
    require(frames, f"missing output frames in {run_dir}")
    return frames[-1]


def inspect_restart_identity(
    continuous: Path, resumed: Path, duration_s: int
) -> dict:
    continuous_final = final_frame(continuous)
    resumed_final = final_frame(resumed)
    fields = {
        field: (continuous_final / f"{field}.npy").read_bytes()
        == (resumed_final / f"{field}.npy").read_bytes()
        for field in FIELDS
    }
    require(all(fields.values()), "restart changed final fields: "
            + ", ".join(field for field, identical in fields.items() if not identical))

    continuous_end = checkpoint_at(continuous, float(duration_s))
    resumed_end = checkpoint_at(resumed, float(duration_s))
    checkpoint_identical = continuous_end.read_bytes() == resumed_end.read_bytes()
    require(checkpoint_identical, "restart changed final checkpoint bytes")

    ledger_suffix: dict[str, bool] = {}
    for filename in LEDGERS:
        continuous_lines = (continuous / filename).read_text(encoding="utf-8").splitlines()
        resumed_lines = (resumed / filename).read_text(encoding="utf-8").splitlines()
        identical = (
            bool(continuous_lines)
            and len(resumed_lines) > 1
            and continuous_lines[0] == resumed_lines[0]
            and continuous_lines[-(len(resumed_lines) - 1):] == resumed_lines[1:]
        )
        ledger_suffix[filename] = identical
    require(all(ledger_suffix.values()), "restart ledger suffix mismatch: "
            + ", ".join(name for name, identical in ledger_suffix.items() if not identical))
    return {
        "fields": fields,
        "checkpoint_bytes": checkpoint_identical,
        "ledger_suffix": ledger_suffix,
    }


def inspect_completed_ledgers(
    run_dir: Path, unsplit_rk: bool = False
) -> dict:
    result: dict[str, dict] = {}
    for filename in LEDGERS:
        path = run_dir / filename
        require(path.is_file(), f"missing ledger: {path}")
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
        require(rows, f"empty ledger: {path}")
        complete_columns = [
            name for name in REQUIRED_COMPLETE_COLUMNS if name in fieldnames
        ]
        if unsplit_rk:
            complete_columns = [
                name for name in complete_columns
                if name not in {
                    "acoustic_v1_complete",
                    "acoustic_v1_cumulative_complete",
                    "slow_momentum_energy_step_complete",
                    "slow_momentum_energy_cumulative_complete",
                    "advection_thermodynamic_boundary_reports_complete",
                    "advection_thermodynamic_moist_energy_reports_complete",
                }
            ]
        for row_index, row in enumerate(rows, start=1):
            for name, raw in row.items():
                if name == "stage" or raw is None or raw == "":
                    continue
                try:
                    value = float(raw)
                except ValueError as exc:
                    raise RuntimeError(
                        f"non-numeric {filename} value at row {row_index}, {name}: {raw!r}"
                    ) from exc
                require(
                    math.isfinite(value),
                    f"non-finite {filename} value at row {row_index}, {name}",
                )
            for name in complete_columns:
                require(
                    float(row[name]) == 1.0,
                    f"incomplete {filename} owner at row {row_index}, {name}",
                )
            if unsplit_rk and filename == "thermodynamic_source_budget.csv":
                owner = float(row[
                    "completed_step_energy_step_unsplit_rk_committed_change_j"
                ])
                committed = float(row[
                    "completed_step_energy_step_committed_owner_sum_j"
                ])
                residual = float(row[
                    "completed_step_energy_step_closure_residual_j"
                ])
                require(
                    owner != 0.0,
                    f"zero unsplit-RK owner at row {row_index}",
                )
                require(
                    abs(residual) / max(1.0, abs(committed)) <= 1.0e-12,
                    f"unclosed unsplit-RK owner at row {row_index}",
                )
        if filename == "radiation_budget.csv":
            maximum_residual = max(
                abs(float(row["relative_booking_residual"])) for row in rows
            )
            require(maximum_residual <= 1.0e-12, "radiation booking residual exceeded 1e-12")
        else:
            maximum_residual = None
        result[filename] = {
            "rows": len(rows),
            "complete_columns": len(complete_columns),
            "maximum_radiation_booking_residual": maximum_residual,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin" / "nimbus_sim_cpu")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    require(binary.is_file(), f"missing CPU binary: {binary}")
    require(AGRID_CONFIG.is_file(), f"missing A-grid fixture: {AGRID_CONFIG}")
    validate_long_lifetime_preflight(binary)

    with tempfile.TemporaryDirectory(prefix="nimbus_constant_velocity_") as temporary:
        root = Path(temporary)
        validate_agrid_public_scope(binary, root)
        base_text = CONFIG.read_text(encoding="utf-8")
        cases: dict[str, dict] = {}
        moving_outputs: dict[str, Path] = {}
        simple_agrid = simple_agrid_variant(base_text)
        maximal_agrid = AGRID_CONFIG.read_text(encoding="utf-8")
        configured_cases = (
            ("c_grid_moist", "c_grid", "moist", 20, False, base_text),
            ("c_grid_dry", "c_grid", "dry", 20, False, dry_variant(base_text)),
            ("a_grid_moist", "a_grid", "moist", 20, False, simple_agrid),
            ("a_grid_dry", "a_grid", "dry", 20, False, dry_variant(simple_agrid)),
            (
                "a_grid_maximal", "a_grid", "maximal", 2, False,
                maximal_agrid,
            ),
            (
                "a_grid_thompson", "a_grid", "thompson", 2, False,
                mixed_phase_variant(maximal_agrid, "thompson"),
            ),
            (
                "a_grid_milbrandt", "a_grid", "milbrandt", 2, False,
                mixed_phase_variant(maximal_agrid, "milbrandt"),
            ),
            (
                "a_grid_tke", "a_grid", "tke", 2, False,
                tke_variant(maximal_agrid),
            ),
            (
                "a_grid_initial_chaos", "a_grid", "initial_chaos", 2, False,
                initial_chaos_variant(maximal_agrid),
            ),
            (
                "a_grid_ssprk3", "a_grid", "ssprk3", 2, True,
                replace_once(
                    maximal_agrid, "time:\n", "time:\n  integrator: ssprk3\n"
                ),
            ),
            (
                "a_grid_rk4", "a_grid", "rk4", 2, True,
                replace_once(
                    maximal_agrid, "time:\n", "time:\n  integrator: rk4\n"
                ),
            ),
        )
        for case_name, grid_name, moisture_name, duration_s, unsplit_rk, moving_text in configured_cases:
                case_root = root / case_name
                case_root.mkdir()
                moving_config = case_root / "moving.yaml"
                stationary_config = case_root / "stationary.yaml"
                moving_config.write_text(moving_text, encoding="utf-8")
                stationary_config.write_text(
                    stationary_counterpart(moving_text), encoding="utf-8"
                )
                thread_outputs: dict[int, tuple[Path, Path]] = {}
                comparisons: dict[str, dict] = {}
                for threads in (1, 4):
                    stationary_output = case_root / f"stationary_t{threads}"
                    moving_output = case_root / f"moving_t{threads}"
                    midpoint_s = duration_s // 2
                    checkpoint_every = midpoint_s if threads == 4 else 0
                    run(binary, stationary_config, stationary_output, threads,
                        checkpoint_every=checkpoint_every, duration_s=duration_s)
                    run(binary, moving_config, moving_output, threads,
                        checkpoint_every=checkpoint_every, duration_s=duration_s)
                    thread_outputs[threads] = (stationary_output, moving_output)
                    comparisons[str(threads)] = inspect_pair(
                        stationary_output, moving_output, duration_s
                    )
                restart_identity: dict[str, dict] = {}
                for mode_name, config, continuous in (
                    ("stationary", stationary_config, thread_outputs[4][0]),
                    ("moving", moving_config, thread_outputs[4][1]),
                ):
                    resumed = case_root / f"{mode_name}_resumed_t4"
                    run(
                        binary,
                        config,
                        resumed,
                        4,
                        checkpoint_every=midpoint_s,
                        resume=checkpoint_at(continuous, float(midpoint_s)),
                        duration_s=duration_s,
                    )
                    restart_identity[mode_name] = inspect_restart_identity(
                        continuous, resumed, duration_s
                    )

                cases[case_name] = {
                    "grid": grid_name,
                    "moisture": moisture_name,
                    "comparisons_by_threads": comparisons,
                    "thread_identity": {
                        "stationary": inspect_thread_identity(
                            thread_outputs[1][0], thread_outputs[4][0]
                        ),
                        "moving": inspect_thread_identity(
                            thread_outputs[1][1], thread_outputs[4][1]
                        ),
                    },
                    "restart_identity": restart_identity,
                    "completed_ledgers": {
                        "stationary": inspect_completed_ledgers(
                            thread_outputs[4][0], unsplit_rk),
                        "moving": inspect_completed_ledgers(
                            thread_outputs[4][1], unsplit_rk),
                    },
                }
                moving_outputs[case_name] = thread_outputs[4][1]

        control_output = moving_outputs["a_grid_maximal"]
        control_final = final_frame(control_output)
        pbl_outputs: dict[str, Path] = {}
        for pbl_scheme in ("slab", "ysu", "mynn"):
            case_name = f"a_grid_pbl_{pbl_scheme}"
            case_root = root / case_name
            case_root.mkdir()
            config = case_root / "moving.yaml"
            config.write_text(
                pbl_variant(maximal_agrid, pbl_scheme), encoding="utf-8"
            )
            one = case_root / "moving_t1"
            four = case_root / "moving_t4"
            resumed = case_root / "moving_resumed_t4"
            run(binary, config, one, 1, duration_s=2)
            run(binary, config, four, 4, checkpoint_every=1, duration_s=2)
            run(
                binary, config, resumed, 4, checkpoint_every=1,
                resume=checkpoint_at(four, 1.0), duration_s=2,
            )
            thread_identity = inspect_thread_identity(one, four)
            restart_identity = inspect_restart_identity(four, resumed, 2)
            ledgers = inspect_completed_ledgers(four)
            active_final = final_frame(four)
            changed = {
                field: (active_final / f"{field}.npy").read_bytes()
                != (control_final / f"{field}.npy").read_bytes()
                for field in ("u", "v", "theta", "qv")
            }
            require(
                any(changed.values()),
                f"moving A-grid {pbl_scheme} surface/PBL branch was inert",
            )
            frame = load_json(active_final / "manifest.json").get(
                "grid", {}
            ).get("horizontal_frame", {})
            require(
                frame.get("motion_mode") == "constant_velocity"
                and frame.get("frame_velocity_earth_ms") == [7, -3],
                f"moving A-grid {pbl_scheme} frame provenance is incomplete",
            )
            cases[case_name] = {
                "grid": "a_grid",
                "moisture": "pbl",
                "scheme": pbl_scheme,
                "thread_identity": thread_identity,
                "restart_identity": restart_identity,
                "completed_ledgers": ledgers,
                "changed_from_no_pbl": changed,
            }
            pbl_outputs[pbl_scheme] = four

        pbl_control_final = final_frame(pbl_outputs["mynn"])
        for chaos_scheme in ("boundary_layer", "full_stochastic"):
            case_name = f"a_grid_chaos_{chaos_scheme}"
            case_root = root / case_name
            case_root.mkdir()
            config = case_root / "moving.yaml"
            config.write_text(
                evolving_chaos_variant(maximal_agrid, chaos_scheme),
                encoding="utf-8",
            )
            one = case_root / "moving_t1"
            four = case_root / "moving_t4"
            resumed = case_root / "moving_resumed_t4"
            run(binary, config, one, 1, duration_s=2)
            run(binary, config, four, 4, checkpoint_every=1, duration_s=2)
            run(
                binary, config, resumed, 4, checkpoint_every=1,
                resume=checkpoint_at(four, 1.0), duration_s=2,
            )
            thread_identity = inspect_thread_identity(one, four)
            restart_identity = inspect_restart_identity(four, resumed, 2)
            ledgers = inspect_completed_ledgers(four)
            active_final = final_frame(four)
            changed = {
                field: (active_final / f"{field}.npy").read_bytes()
                != (pbl_control_final / f"{field}.npy").read_bytes()
                for field in ("u", "v", "theta", "qv", "qr")
            }
            require(
                any(changed.values()),
                f"moving {chaos_scheme} branch was inert",
            )
            frame = load_json(active_final / "manifest.json").get(
                "grid", {}
            ).get("horizontal_frame", {})
            require(
                frame.get("grid_origin_earth_m") == [14, -6],
                f"moving {chaos_scheme} frame provenance is incomplete",
            )
            cases[case_name] = {
                "grid": "a_grid",
                "chaos": chaos_scheme,
                "thread_identity": thread_identity,
                "restart_identity": restart_identity,
                "completed_ledgers": ledgers,
                "changed_from_no_chaos": changed,
            }

        beta_root = root / "a_grid_beta_plane"
        beta_root.mkdir()
        beta_config = beta_root / "moving.yaml"
        beta_config.write_text(
            replace_once(
                maximal_agrid, "  mode: f_plane\n", "  mode: beta_plane\n"
            ),
            encoding="utf-8",
        )
        beta_one = beta_root / "moving_t1"
        beta_four = beta_root / "moving_t4"
        beta_resumed = beta_root / "moving_resumed_t4"
        run(binary, beta_config, beta_one, 1, duration_s=2)
        run(binary, beta_config, beta_four, 4, checkpoint_every=1, duration_s=2)
        run(
            binary, beta_config, beta_resumed, 4, checkpoint_every=1,
            resume=checkpoint_at(beta_four, 1.0), duration_s=2,
        )
        beta_thread_identity = inspect_thread_identity(beta_one, beta_four)
        beta_restart_identity = inspect_restart_identity(
            beta_four, beta_resumed, 2
        )
        beta_ledgers = inspect_completed_ledgers(beta_four)
        beta_final = final_frame(beta_four)
        beta_changed = {
            field: (beta_final / f"{field}.npy").read_bytes()
            != (control_final / f"{field}.npy").read_bytes()
            for field in ("u", "v")
        }
        require(any(beta_changed.values()), "moving beta-plane branch was inert")
        beta_frame = load_json(beta_final / "manifest.json").get(
            "grid", {}
        ).get("horizontal_frame", {})
        require(
            beta_frame.get("motion_mode") == "constant_velocity"
            and beta_frame.get("grid_origin_earth_m") == [14, -6],
            "moving beta-plane Earth-frame provenance is incomplete",
        )
        cases["a_grid_beta_plane"] = {
            "grid": "a_grid",
            "coriolis": "beta_plane",
            "thread_identity": beta_thread_identity,
            "restart_identity": beta_restart_identity,
            "completed_ledgers": beta_ledgers,
            "changed_from_f_plane": beta_changed,
        }

    report = {
        "schema": "nimbus.qualification.constant_velocity_domain",
        "schema_version": 5,
        "status": "pass",
        "frame_velocity_earth_ms": {"x": FRAME_U_MS, "y": FRAME_V_MS},
        "cases": cases,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report_suffix = f"; report={args.report}" if args.report else ""
    print(
        "[constant-velocity-domain] PASS: "
        "11 dry/moist/maximal/RK A/C-grid cases; 22 Galilean thread pairs; "
        "22 exact one/four-thread mode checks; 22 exact restart branches; "
        "3 non-inert moving surface/PBL branches with exact threads/restart; "
        "1 non-inert moving beta-plane branch with exact threads/restart; "
        "2 non-inert moving evolving-chaos branches with exact threads/restart; "
        "168 finite completed ledgers; flexible A-grid velocity; 2 unsupported "
        f"neighbors rejected; long preflight frame-aware{report_suffix}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
