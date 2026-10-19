#!/usr/bin/env python3
"""Qualify strict-YAML stationary nesting through the physical Vulkan runtime."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "bin" / "nimbus_sim"
DEFAULT_VALIDATOR = ROOT / "bin" / "field_validator"
DEFAULT_CONFIG = (
    ROOT / "tests" / "fixtures" / "configs" / "qualification" /
    "stationary_nested_vulkan_dry.yaml"
)
CORE_FIELDS = {"p", "qr", "qv", "rho", "theta", "u", "v", "w"}
FLOAT_PATTERN = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
FLOAT32_EPSILON = 2.0 ** -23


def run(
    command: list[str], environment: dict[str, str] | None = None
) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout}"
        )
    return result.stdout


def require_device_log(output: str) -> None:
    required = (
        "[COMPUTE] backend=vulkan, requested=vulkan, "
        "device_index=-1, fallback=off"
    )
    if required not in output:
        raise RuntimeError(
            "run did not prove fallback-off physical Vulkan execution:\n"
            f"{output}"
        )


def require_hierarchy_memory_plan(output: str, label: str) -> None:
    memory = re.search(
        r"\[MEMORY PLAN\] accelerator_mode=(?:unified|discrete), "
        r"accelerator_fields=([0-9]+), accelerator_levels=([0-9]+), "
        r"host_shared_or_staging_bytes=([0-9]+), "
        r"device_residency_bytes=([0-9]+), "
        r"device_budget_bytes=([0-9]+), "
        r"max_storage_buffer_bytes=([0-9]+)",
        output,
    )
    if memory is None:
        raise RuntimeError(f"{label}: missing hierarchy accelerator memory plan")
    fields, levels, host_bytes, device_bytes, budget_bytes, max_buffer = (
        int(value) for value in memory.groups()
    )
    if (
        fields <= 0 or levels != 2 or host_bytes <= 0 or device_bytes <= 0 or
        budget_bytes <= 0 or max_buffer <= 0 or device_bytes > budget_bytes
    ):
        raise RuntimeError(
            f"{label}: invalid two-level accelerator memory admission: "
            f"{memory.group(0)}"
        )


def require_smagorinsky_log(output: str, label: str) -> None:
    aggregate = re.search(
        r"\[NESTING DEVICE\] "
        r"smagorinsky_coefficients_dispatches=([0-9]+), "
        r"smagorinsky_tendencies_dispatches=([0-9]+), "
        r"sgs_apply_dispatches=([0-9]+)",
        output,
    )
    if aggregate is None or any(int(value) <= 0 for value in aggregate.groups()):
        raise RuntimeError(
            f"{label}: aggregate Smagorinsky device counters are absent or "
            f"vacuous:\n{output}"
        )
    levels = re.search(
        r"\[NESTING DEVICE LEVELS\] "
        r"parent_coefficients=([0-9]+), parent_tendencies=([0-9]+), "
        r"parent_sgs_apply=([0-9]+), child_coefficients=([0-9]+), "
        r"child_tendencies=([0-9]+), child_sgs_apply=([0-9]+)",
        output,
    )
    if levels is None or any(int(value) <= 0 for value in levels.groups()):
        raise RuntimeError(
            f"{label}: all six parent/child Smagorinsky device counters must "
            f"be positive:\n{output}"
        )
    work = re.search(
        r"\[NESTING SGS\] parent_work_steps=([0-9]+), "
        rf"parent_committed_work_j=({FLOAT_PATTERN}), "
        r"parent_stress_steps=([0-9]+), "
        rf"parent_open_boundary_work_j=({FLOAT_PATTERN}), "
        rf"parent_irreversible_dissipation_j=({FLOAT_PATTERN}), "
        rf"parent_stress_residual_j=({FLOAT_PATTERN}), "
        r"child_work_steps=([0-9]+), "
        rf"child_committed_work_j=({FLOAT_PATTERN}), "
        r"child_stress_steps=([0-9]+), "
        rf"child_open_boundary_work_j=({FLOAT_PATTERN}), "
        rf"child_irreversible_dissipation_j=({FLOAT_PATTERN}), "
        rf"child_stress_residual_j=({FLOAT_PATTERN})",
        output,
    )
    if work is None:
        raise RuntimeError(f"{label}: missing parent/child SGS work ledger")
    parent_steps = int(work.group(1))
    parent_work = float(work.group(2))
    parent_stress_steps = int(work.group(3))
    parent_boundary_work = float(work.group(4))
    parent_dissipation = float(work.group(5))
    parent_stress_residual = float(work.group(6))
    child_steps = int(work.group(7))
    child_work = float(work.group(8))
    child_stress_steps = int(work.group(9))
    child_boundary_work = float(work.group(10))
    child_dissipation = float(work.group(11))
    child_stress_residual = float(work.group(12))
    stress_values = (
        parent_boundary_work, parent_dissipation, parent_stress_residual,
        child_boundary_work, child_dissipation, child_stress_residual,
    )
    parent_stress_scale = max(
        1.0, abs(parent_work) + abs(parent_boundary_work) +
        abs(parent_dissipation))
    child_stress_scale = max(
        1.0, abs(child_work) + abs(child_boundary_work) +
        abs(child_dissipation))
    if (
        parent_steps <= 0 or child_steps <= 0 or
        not math.isfinite(parent_work) or not math.isfinite(child_work) or
        any(not math.isfinite(value) for value in stress_values) or
        parent_work == 0.0 or child_work == 0.0 or
        parent_dissipation <= 0.0 or child_dissipation <= 0.0 or
        (parent_stress_steps == 0 and parent_boundary_work != 0.0) or
        (child_stress_steps == 0 and child_boundary_work != 0.0) or
        abs(parent_stress_residual) > 1.0e-10 * parent_stress_scale or
        abs(child_stress_residual) > 1.0e-10 * child_stress_scale
    ):
        raise RuntimeError(
            f"{label}: parent and child SGS work must be finite and nonzero, "
            "irreversible dissipation must be positive, closed profiles may "
            "not invent boundary work, and the stress identity must close; "
            f"row={work.group(0)}"
        )


def require_kessler_log(output: str, label: str) -> None:
    aggregate = re.search(
        r"\[NESTING DEVICE MICROPHYSICS\] "
        r"kessler_pointwise_dispatches=([0-9]+), "
        r"kessler_sedimentation_dispatches=([0-9]+), "
        r"microphysics_apply_dispatches=([0-9]+)",
        output,
    )
    if (
        aggregate is None or
        int(aggregate.group(1)) <= 0 or int(aggregate.group(2)) <= 0 or
        int(aggregate.group(3)) != 0
    ):
        raise RuntimeError(
            f"{label}: Kessler pointwise/sedimentation device counters must "
            "be positive while the qualified FP64 CPU state commit keeps "
            f"MicrophysicsApply at zero:\n{output}"
        )
    require_kessler_level_and_science_logs(output, label)


def read_npy_float32(path: Path) -> tuple[tuple[int, ...], tuple[float, ...]]:
    with path.open("rb") as handle:
        if handle.read(6) != b"\x93NUMPY":
            raise RuntimeError(f"{path}: not an NPY file")
        major, minor = handle.read(2)
        if (major, minor) == (1, 0):
            header_size = struct.unpack("<H", handle.read(2))[0]
        elif major in (2, 3):
            header_size = struct.unpack("<I", handle.read(4))[0]
        else:
            raise RuntimeError(
                f"{path}: unsupported NPY version {major}.{minor}"
            )
        encoding = "utf-8" if major == 3 else "latin1"
        header = ast.literal_eval(handle.read(header_size).decode(encoding))
        if header.get("descr") not in ("<f4", "|f4"):
            raise RuntimeError(f"{path}: expected little-endian float32")
        if header.get("fortran_order") is not False:
            raise RuntimeError(f"{path}: Fortran-order arrays are unsupported")
        shape = header.get("shape")
        if not isinstance(shape, tuple) or not shape:
            raise RuntimeError(f"{path}: invalid shape")
        payload = handle.read()
    count = math.prod(shape)
    if len(payload) != count * 4:
        raise RuntimeError(f"{path}: payload size mismatch")
    values = tuple(value for (value,) in struct.iter_unpack("<f", payload))
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"{path}: non-finite value")
    return shape, values


def final_frame(root: Path) -> Path:
    frames = sorted(
        path for path in root.glob("step_[0-9][0-9][0-9][0-9][0-9][0-9]")
        if path.is_dir()
    )
    if not frames:
        raise RuntimeError(f"{root}: no history frames")
    return frames[-1]


def require_endpoint_parity(cpu_root: Path, vulkan_root: Path,
                            label: str) -> None:
    cpu_frame = final_frame(cpu_root)
    vulkan_frame = final_frame(vulkan_root)
    for field in sorted(CORE_FIELDS):
        cpu_shape, cpu_values = read_npy_float32(cpu_frame / f"{field}.npy")
        vk_shape, vk_values = read_npy_float32(vulkan_frame / f"{field}.npy")
        if cpu_shape != vk_shape or len(cpu_values) != len(vk_values):
            raise RuntimeError(f"{label}: {field} shape mismatch")
        worst = (0.0, 0.0, 0.0, 0)
        for index, (cpu, vk) in enumerate(zip(cpu_values, vk_values)):
            scale = max(1.0, abs(cpu), abs(vk))
            difference = abs(cpu - vk)
            # This is the established parity-registry Stencil tier. The gate
            # deliberately does not multiply it by the twelve-step horizon.
            tolerance = max(5.0e-5, 1.0e-4 * scale)
            if difference > worst[0]:
                worst = (difference, tolerance, scale, index)
            if difference > tolerance:
                raise RuntimeError(
                    f"{label}: {field} endpoint parity failed at {index}: "
                    f"cpu={cpu:.9g} vulkan={vk:.9g} abs={difference:.9g} "
                    f"tolerance={tolerance:.9g}"
                )


def radiation_row(output: str, label: str,
                  expected_last_s: float,
                  expected_next_s: float) -> dict[str, float | str]:
    lines = [
        line for line in output.splitlines()
        if "[NESTING RADIATION] " in line
    ]
    if len(lines) != 1:
        raise RuntimeError(
            f"{label}: expected one nesting radiation row, got {len(lines)}"
        )
    row: dict[str, float | str] = {}
    for key, raw in re.findall(r"([a-z_]+)=([^,]+)", lines[0]):
        value = raw.strip()
        row[key] = value if key.endswith("owner") else float(value)
    required = {
        f"{level}_{name}"
        for level in ("parent", "child")
        for name in (
            "owner", "last_update_time_s", "next_update_time_s",
            "applied_steps", "held_power_w", "held_abs_power_w",
            "nonzero_cells", "nonfinite_cells", "last_expected_energy_j",
            "last_applied_energy_j", "expected_energy_j",
            "applied_energy_j", "energy_residual_j",
        )
    }
    missing = sorted(required - row.keys())
    if missing:
        raise RuntimeError(f"{label}: incomplete radiation row: {missing}")
    for level in ("parent", "child"):
        if row[f"{level}_owner"] != "host":
            raise RuntimeError(f"{label}: radiation owner is not host")
        numeric = [
            float(row[f"{level}_{name}"])
            for name in (
                "last_update_time_s", "next_update_time_s", "applied_steps",
                "held_power_w", "held_abs_power_w", "nonzero_cells",
                "nonfinite_cells", "last_expected_energy_j",
                "last_applied_energy_j", "expected_energy_j",
                "applied_energy_j", "energy_residual_j",
            )
        ]
        if not all(math.isfinite(value) for value in numeric):
            raise RuntimeError(f"{label}: non-finite {level} radiation row")
        maximum_physical_cells = 14 * 14 * 24 if level == "parent" else 8 * 8 * 24
        if (
            float(row[f"{level}_last_update_time_s"]) != expected_last_s or
            float(row[f"{level}_next_update_time_s"]) != expected_next_s or
            float(row[f"{level}_applied_steps"]) <= 0.0 or
            float(row[f"{level}_held_abs_power_w"]) <= 0.0 or
            float(row[f"{level}_nonzero_cells"]) <= 0.0 or
            float(row[f"{level}_nonzero_cells"]) > maximum_physical_cells or
            float(row[f"{level}_nonfinite_cells"]) != 0.0
        ):
            raise RuntimeError(
                f"{label}: vacuous or phase-wrong {level} radiation row: "
                f"{lines[0]}"
            )
        last_expected = float(row[f"{level}_last_expected_energy_j"])
        last_applied = float(row[f"{level}_last_applied_energy_j"])
        cumulative_expected = float(row[f"{level}_expected_energy_j"])
        cumulative_applied = float(row[f"{level}_applied_energy_j"])
        residual = float(row[f"{level}_energy_residual_j"])
        if abs(last_applied - last_expected) > 1.0e-10 * max(
            1.0, abs(last_applied), abs(last_expected)
        ):
            raise RuntimeError(f"{label}: {level} last energy does not close")
        if abs(residual) > 1.0e-12 * max(
            1.0, abs(cumulative_applied), abs(cumulative_expected)
        ):
            raise RuntimeError(
                f"{label}: {level} cumulative energy does not close"
            )
    return row


def require_radiation_ledger_parity(
    cpu: dict[str, float | str],
    vulkan: dict[str, float | str],
    label: str,
) -> None:
    for level in ("parent", "child"):
        for name in (
            "last_update_time_s", "next_update_time_s", "applied_steps",
            "held_power_w", "held_abs_power_w", "nonzero_cells",
            "last_expected_energy_j", "last_applied_energy_j",
            "expected_energy_j", "applied_energy_j", "energy_residual_j",
        ):
            cpu_value = float(cpu[f"{level}_{name}"])
            vulkan_value = float(vulkan[f"{level}_{name}"])
            scale = max(1.0, abs(cpu_value), abs(vulkan_value))
            tolerance = 96.0 * FLOAT32_EPSILON * scale
            if abs(cpu_value - vulkan_value) > tolerance:
                raise RuntimeError(
                    f"{label}: {level}_{name} CPU/Vulkan parity failed: "
                    f"cpu={cpu_value:.17g} vulkan={vulkan_value:.17g} "
                    f"tolerance={tolerance:.9g}"
                )


def require_kessler_level_and_science_logs(output: str, label: str) -> None:
    levels = re.search(
        r"\[NESTING DEVICE MICROPHYSICS LEVELS\] "
        r"parent_pointwise=([0-9]+), parent_sedimentation=([0-9]+), "
        r"parent_apply=([0-9]+), child_pointwise=([0-9]+), "
        r"child_sedimentation=([0-9]+), child_apply=([0-9]+)",
        output,
    )
    if (
        levels is None or
        any(int(levels.group(index)) <= 0 for index in (1, 2, 4, 5)) or
        int(levels.group(3)) != 0 or int(levels.group(6)) != 0
    ):
        raise RuntimeError(
            f"{label}: parent/child Kessler process counters must be positive "
            "and qualified state-apply counters must be zero:\n"
            f"{output}"
        )
    physics = re.search(
        r"\[NESTING MICROPHYSICS\] "
        r"parent_active=(?P<pa>[01]), "
        rf"parent_relative_imbalance=(?P<pr>{FLOAT_PATTERN}), "
        rf"parent_imbalance_kg=(?P<pi>{FLOAT_PATTERN}), "
        rf"parent_baseline_water_kg=(?P<pb>{FLOAT_PATTERN}), "
        rf"parent_water_activity_kg=(?P<pwa>{FLOAT_PATTERN}), "
        rf"parent_material_water_repair_kg=(?P<pwr>{FLOAT_PATTERN}), "
        r"parent_material_water_repair_count=(?P<pwrc>[0-9]+), "
        rf"parent_surface_precipitation_kg=(?P<pp>{FLOAT_PATTERN}), "
        rf"parent_heat_j=(?P<ph>{FLOAT_PATTERN}), "
        rf"parent_theta_repair_j=(?P<ptr>{FLOAT_PATTERN}), "
        r"parent_theta_repair_count=(?P<ptrc>[0-9]+), "
        r"parent_moist_energy_complete=(?P<pec>[01]), "
        rf"parent_moist_energy_residual_j=(?P<per>{FLOAT_PATTERN}), "
        rf"parent_energy_activity_j=(?P<pea>{FLOAT_PATTERN}), "
        r"child_active=(?P<ca>[01]), "
        rf"child_relative_imbalance=(?P<cr>{FLOAT_PATTERN}), "
        rf"child_imbalance_kg=(?P<ci>{FLOAT_PATTERN}), "
        rf"child_baseline_water_kg=(?P<cb>{FLOAT_PATTERN}), "
        rf"child_water_activity_kg=(?P<cwa>{FLOAT_PATTERN}), "
        rf"child_material_water_repair_kg=(?P<cwr>{FLOAT_PATTERN}), "
        r"child_material_water_repair_count=(?P<cwrc>[0-9]+), "
        rf"child_surface_precipitation_kg=(?P<cp>{FLOAT_PATTERN}), "
        rf"child_heat_j=(?P<ch>{FLOAT_PATTERN}), "
        rf"child_theta_repair_j=(?P<ctr>{FLOAT_PATTERN}), "
        r"child_theta_repair_count=(?P<ctrc>[0-9]+), "
        r"child_moist_energy_complete=(?P<cec>[01]), "
        rf"child_moist_energy_residual_j=(?P<cer>{FLOAT_PATTERN}), "
        rf"child_energy_activity_j=(?P<cea>{FLOAT_PATTERN})",
        output,
    )
    if physics is None:
        raise RuntimeError(f"{label}: missing parent/child Kessler science row")

    parent_active = int(physics["pa"])
    parent_relative = float(physics["pr"])
    parent_imbalance = float(physics["pi"])
    parent_baseline = float(physics["pb"])
    parent_water_activity = float(physics["pwa"])
    parent_water_repair = float(physics["pwr"])
    parent_water_repair_count = int(physics["pwrc"])
    parent_precipitation = float(physics["pp"])
    parent_heat = float(physics["ph"])
    parent_theta_repair = float(physics["ptr"])
    parent_theta_repair_count = int(physics["ptrc"])
    parent_energy_complete = int(physics["pec"])
    parent_energy_residual = float(physics["per"])
    parent_energy_activity = float(physics["pea"])
    child_active = int(physics["ca"])
    child_relative = float(physics["cr"])
    child_imbalance = float(physics["ci"])
    child_baseline = float(physics["cb"])
    child_water_activity = float(physics["cwa"])
    child_water_repair = float(physics["cwr"])
    child_water_repair_count = int(physics["cwrc"])
    child_precipitation = float(physics["cp"])
    child_heat = float(physics["ch"])
    child_theta_repair = float(physics["ctr"])
    child_theta_repair_count = int(physics["ctrc"])
    child_energy_complete = int(physics["cec"])
    child_energy_residual = float(physics["cer"])
    child_energy_activity = float(physics["cea"])
    finite_values = (
        parent_relative, parent_imbalance, parent_baseline,
        parent_water_activity, parent_water_repair, parent_precipitation,
        parent_heat, parent_theta_repair, parent_energy_residual,
        parent_energy_activity, child_relative, child_imbalance,
        child_baseline, child_water_activity, child_water_repair,
        child_precipitation, child_heat, child_theta_repair,
        child_energy_residual, child_energy_activity,
    )
    if (
        parent_active != 1 or child_active != 1 or
        parent_energy_complete != 1 or child_energy_complete != 1 or
        any(not math.isfinite(value) for value in finite_values) or
        abs(parent_relative) > 1.0e-5 or abs(child_relative) > 1.0e-5 or
        parent_baseline <= 0.0 or child_baseline <= 0.0 or
        parent_water_activity <= 0.0 or child_water_activity <= 0.0 or
        abs(parent_imbalance) > 1.0e-5 * max(1.0, parent_water_activity) or
        abs(child_imbalance) > 1.0e-5 * max(1.0, child_water_activity) or
        parent_water_repair != 0.0 or child_water_repair != 0.0 or
        parent_water_repair_count != 0 or child_water_repair_count != 0 or
        parent_precipitation <= 1.0e-4 or child_precipitation <= 1.0e-4 or
        parent_heat <= 1.0e8 or child_heat <= 1.0e8 or
        parent_theta_repair != 0.0 or child_theta_repair != 0.0 or
        parent_theta_repair_count != 0 or child_theta_repair_count != 0 or
        parent_energy_activity <= 0.0 or child_energy_activity <= 0.0 or
        abs(parent_energy_residual) >
            1.0e-12 * max(1.0, parent_energy_activity) or
        abs(child_energy_residual) >
            1.0e-12 * max(1.0, child_energy_activity)
    ):
        raise RuntimeError(
            f"{label}: Kessler must be materially active, activity-scaled "
            "water and reservoir-scaled energy attribution must close, and "
            "qualified execution may not require material water/theta repair "
            f"on either level; row={physics.group(0)}"
        )


def load_manifest(step: Path) -> dict[str, object]:
    manifest = step / "manifest.json"
    if not manifest.is_file():
        raise RuntimeError(f"missing completion manifest: {manifest}")
    return json.loads(manifest.read_text(encoding="utf-8"))


def require_history(
    root: Path,
    kind: str,
    dimensions: dict[str, int],
    indices: tuple[int, ...],
) -> None:
    for index in indices:
        step = root / f"step_{index:06d}"
        manifest = load_manifest(step)
        if manifest["data_product"]["kind"] != kind:
            raise RuntimeError(f"{step}: unexpected product identity")
        if manifest["grid"]["dimensions"] != dimensions:
            raise RuntimeError(f"{step}: unexpected dimensions")
        if set(manifest["published_fields"]) != CORE_FIELDS:
            raise RuntimeError(f"{step}: unexpected published field inventory")
        if {path.stem for path in step.glob("*.npy")} != CORE_FIELDS:
            raise RuntimeError(f"{step}: incomplete NPY payload inventory")


def validate_history(validator: Path, history_root: Path) -> None:
    output = run(
        [
            str(validator), "--input", str(history_root), "--mode", "report",
            "--scope", "required",
        ]
    )
    if "nonfinite=0 bounds=0" not in output or "failed=no" not in output:
        raise RuntimeError(
            f"finite/bounds validation failed for {history_root}:\n{output}"
        )


def checkpoints(directory: Path) -> list[Path]:
    return sorted(directory.glob("checkpoint_step_*.nimbus"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--validator", type=Path, default=DEFAULT_VALIDATOR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--require-smagorinsky",
        action="store_true",
        help="require native Smagorinsky coefficient/apply dispatch and work",
    )
    parser.add_argument(
        "--require-kessler",
        action="store_true",
        help="require native Kessler dispatch, apply, heat, and fallout",
    )
    parser.add_argument(
        "--require-radiation",
        action="store_true",
        help=(
            "require host-owned parent/child longwave activity, exact restart, "
            "and paired CPU/Vulkan endpoint and ledger parity"
        ),
    )
    parser.add_argument(
        "--duration-s",
        type=int,
        default=2,
        help="positive even final simulation time; the branch occurs halfway",
    )
    args = parser.parse_args()
    binary = args.binary.resolve()
    validator = args.validator.resolve()
    config = args.config.resolve()
    for required in (binary, validator, config):
        if not required.is_file():
            raise RuntimeError(f"required qualification input is missing: {required}")
    if args.duration_s <= 0 or args.duration_s % 2 != 0:
        raise RuntimeError("--duration-s must be a positive even integer")
    if args.require_radiation and args.duration_s != 6:
        raise RuntimeError("--require-radiation requires --duration-s=6")
    segment_s = args.duration_s // 2

    with tempfile.TemporaryDirectory(
        prefix="nimbus_stationary_nested_vulkan_runtime_"
    ) as temporary:
        root = Path(temporary)
        common = [
            str(binary), "--config", str(config), "--headless",
            "--threads", "1", f"--write-every={segment_s}",
            f"--checkpoint-every={segment_s}", "--log-profile=normal",
        ]

        continuous_output = root / "continuous_output"
        continuous_checkpoints = root / "continuous_checkpoints"
        continuous_log = run(
            common + [
                "--duration", str(args.duration_s),
                f"--outdir={continuous_output}",
                f"--checkpoint-dir={continuous_checkpoints}",
            ],
        )
        require_device_log(continuous_log)
        require_hierarchy_memory_plan(continuous_log, "continuous run")
        if args.require_smagorinsky:
            require_smagorinsky_log(continuous_log, "continuous run")
        if args.require_kessler:
            require_kessler_log(continuous_log, "continuous run")
        continuous_radiation = None
        if args.require_radiation:
            continuous_radiation = radiation_row(
                continuous_log, "continuous Vulkan run", 4.0, 6.0
            )
        if "[NESTING] initialized stationary one-way child" not in continuous_log:
            raise RuntimeError("production driver did not initialize the child level")

        run_contract = json.loads(
            (continuous_output / "nimbus_output.json").read_text(
                encoding="utf-8"
            )
        )
        if run_contract.get("publication_state") != "ready":
            raise RuntimeError("continuous output contract is not ready")
        if run_contract.get("data_products") != [
            "history", "nested_child_history"
        ]:
            raise RuntimeError("continuous output contract lacks child history")

        child_output = continuous_output / "nests" / "child_0000"
        require_history(
            continuous_output, "history", {"x": 16, "y": 16, "z": 24},
            (0, 1, 2),
        )
        require_history(
            child_output, "nested_child_history",
            {"x": 10, "y": 10, "z": 24}, (1, 2),
        )
        if (child_output / "step_000000").exists():
            raise RuntimeError("child history published before a causal parent interval")
        validate_history(validator, continuous_output)
        validate_history(validator, child_output)
        continuous = checkpoints(continuous_checkpoints)
        if len(continuous) != 2:
            raise RuntimeError(
                f"expected two continuous checkpoints, found {continuous}"
            )

        split_output = root / "split_output"
        split_checkpoints = root / "split_checkpoints"
        split_log = run(
            common + [
                "--duration", str(segment_s), f"--outdir={split_output}",
                f"--checkpoint-dir={split_checkpoints}",
            ],
        )
        require_device_log(split_log)
        require_hierarchy_memory_plan(split_log, "split run")
        if args.require_smagorinsky:
            require_smagorinsky_log(split_log, "split run")
        if args.require_kessler:
            require_kessler_log(split_log, "split run")
        if args.require_radiation:
            radiation_row(split_log, "split Vulkan run", 2.0, 4.0)
        first = checkpoints(split_checkpoints)
        if len(first) != 1:
            raise RuntimeError(f"expected one split checkpoint, found {first}")

        resumed_output = root / "resumed_output"
        resumed_checkpoints = root / "resumed_checkpoints"
        resumed_log = run(
            common + [
                "--duration", str(args.duration_s),
                f"--outdir={resumed_output}",
                f"--checkpoint-dir={resumed_checkpoints}",
                f"--resume={first[0]}",
            ],
        )
        require_device_log(resumed_log)
        require_hierarchy_memory_plan(resumed_log, "resumed run")
        if args.require_smagorinsky:
            require_smagorinsky_log(resumed_log, "resumed run")
        if args.require_kessler:
            require_kessler_log(resumed_log, "resumed run")
        resumed_radiation = None
        if args.require_radiation:
            resumed_radiation = radiation_row(
                resumed_log, "resumed Vulkan run", 4.0, 6.0
            )
        resumed = checkpoints(resumed_checkpoints)
        if len(resumed) != 1:
            raise RuntimeError(f"expected one resumed checkpoint, found {resumed}")
        if continuous[-1].read_bytes() != resumed[-1].read_bytes():
            raise RuntimeError(
                "continuous and resumed stationary-nesting checkpoints differ"
            )
        if args.require_radiation:
            if resumed_radiation != continuous_radiation:
                raise RuntimeError(
                    "continuous and resumed Vulkan radiation ledgers differ"
                )

            cpu_common = common + ["--compute-backend", "cpu"]
            cpu_continuous_output = root / "cpu_continuous_output"
            cpu_continuous_checkpoints = root / "cpu_continuous_checkpoints"
            cpu_continuous_log = run(
                cpu_common + [
                    "--duration", str(args.duration_s),
                    f"--outdir={cpu_continuous_output}",
                    f"--checkpoint-dir={cpu_continuous_checkpoints}",
                ]
            )
            cpu_continuous_radiation = radiation_row(
                cpu_continuous_log, "continuous CPU run", 4.0, 6.0
            )
            if "[COMPUTE] backend=cpu" not in cpu_continuous_log:
                raise RuntimeError("paired run did not prove CPU execution")
            cpu_child_output = (
                cpu_continuous_output / "nests" / "child_0000"
            )
            validate_history(validator, cpu_continuous_output)
            validate_history(validator, cpu_child_output)
            cpu_continuous = checkpoints(cpu_continuous_checkpoints)
            if len(cpu_continuous) != 2:
                raise RuntimeError(
                    "expected two continuous CPU checkpoints, found "
                    f"{cpu_continuous}"
                )

            cpu_split_output = root / "cpu_split_output"
            cpu_split_checkpoints = root / "cpu_split_checkpoints"
            cpu_split_log = run(
                cpu_common + [
                    "--duration", str(segment_s),
                    f"--outdir={cpu_split_output}",
                    f"--checkpoint-dir={cpu_split_checkpoints}",
                ]
            )
            radiation_row(cpu_split_log, "split CPU run", 2.0, 4.0)
            cpu_first = checkpoints(cpu_split_checkpoints)
            if len(cpu_first) != 1:
                raise RuntimeError(
                    f"expected one split CPU checkpoint, found {cpu_first}"
                )

            cpu_resumed_output = root / "cpu_resumed_output"
            cpu_resumed_checkpoints = root / "cpu_resumed_checkpoints"
            cpu_resumed_log = run(
                cpu_common + [
                    "--duration", str(args.duration_s),
                    f"--outdir={cpu_resumed_output}",
                    f"--checkpoint-dir={cpu_resumed_checkpoints}",
                    f"--resume={cpu_first[0]}",
                ]
            )
            cpu_resumed_radiation = radiation_row(
                cpu_resumed_log, "resumed CPU run", 4.0, 6.0
            )
            cpu_resumed = checkpoints(cpu_resumed_checkpoints)
            if len(cpu_resumed) != 1:
                raise RuntimeError(
                    f"expected one resumed CPU checkpoint, found {cpu_resumed}"
                )
            if (
                cpu_continuous[-1].read_bytes() !=
                cpu_resumed[-1].read_bytes()
            ):
                raise RuntimeError(
                    "continuous and resumed CPU nesting checkpoints differ"
                )
            if cpu_resumed_radiation != cpu_continuous_radiation:
                raise RuntimeError(
                    "continuous and resumed CPU radiation ledgers differ"
                )

            require_endpoint_parity(
                cpu_continuous_output, continuous_output,
                "root radiation endpoint",
            )
            require_endpoint_parity(
                cpu_child_output, child_output,
                "child radiation endpoint",
            )
            require_radiation_ledger_parity(
                cpu_continuous_radiation,
                continuous_radiation,
                "continuous radiation ledger",
            )

    print(
        "[stationary-nested-vulkan-runtime] PASS: strict YAML reached a "
        "native-capable fallback-off device, published finite root/child "
        "history, and resumed exactly"
        + (" with non-vacuous native Smagorinsky work"
           if args.require_smagorinsky else "")
        + (" with conservative active native Kessler work and fallout"
           if args.require_kessler else "")
        + (" with host-owned longwave activity, paired CPU/Vulkan parity, "
           "and exact CPU/Vulkan restart"
           if args.require_radiation else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
