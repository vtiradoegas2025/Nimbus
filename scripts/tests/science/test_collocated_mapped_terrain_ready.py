#!/usr/bin/env python3

from __future__ import annotations

import argparse
import array
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import math
from pathlib import Path
import re
import subprocess
import struct
import sys
import tempfile

from npy_support import read_float32_npy


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.qualification.checkpointing.qualify_vulkan_restart import (  # noqa: E402
    require_report_owner_identities,
    require_restart_suffix,
)

CONFIG = ROOT / "tests/fixtures/configs/qualification/agrid_stationary_bell_cpu_ready.yaml"
QUALIFICATION = ROOT / "tests/fixtures/configs/qualification"
FULL_STOCHASTIC_CONFIGS = {
    "bell": {
        "none": QUALIFICATION / "agrid_mapped_full_stochastic_none_ready.yaml",
        "kessler": QUALIFICATION / "agrid_mapped_full_stochastic_kessler_ready.yaml",
        "lin": QUALIFICATION / "agrid_mapped_full_stochastic_lin_ready.yaml",
        "thompson": QUALIFICATION / "agrid_mapped_full_stochastic_ready.yaml",
        "milbrandt": QUALIFICATION / "agrid_mapped_full_stochastic_milbrandt_ready.yaml",
    },
    "schar": {
        "none": QUALIFICATION / "agrid_mapped_full_stochastic_schar_none_ready.yaml",
        "kessler": QUALIFICATION / "agrid_mapped_full_stochastic_schar_kessler_ready.yaml",
        "lin": QUALIFICATION / "agrid_mapped_full_stochastic_schar_lin_ready.yaml",
        "thompson": QUALIFICATION / "agrid_mapped_full_stochastic_schar_thompson_ready.yaml",
        "milbrandt": QUALIFICATION / "agrid_mapped_full_stochastic_schar_milbrandt_ready.yaml",
    },
}
CHAOS_MODES = ("none", "initial_conditions", "boundary_layer", "full_stochastic")
MICROPHYSICS_SCHEMES = ("none", "kessler", "lin", "thompson", "milbrandt")
ADVECTION_MODES = ("none", "tvd", "weno5")


def weno_config_path(terrain: str, chaos_mode: str, scheme: str) -> Path:
    if chaos_mode == "full_stochastic":
        return FULL_STOCHASTIC_CONFIGS[terrain][scheme]
    return QUALIFICATION / (
        f"agrid_mapped_{chaos_mode}_{terrain}_{scheme}_ready.yaml"
    )


def active_config_path(
    terrain: str, advection_mode: str, chaos_mode: str, scheme: str,
) -> Path:
    if advection_mode == "weno5":
        return weno_config_path(terrain, chaos_mode, scheme)
    return QUALIFICATION / (
        f"agrid_mapped_advection_{advection_mode}_chaos_{chaos_mode}_"
        f"{terrain}_{scheme}_ready.yaml"
    )


ACTIVE_CONFIGS = {
    terrain: {
        advection_mode: {
            chaos_mode: {
                scheme: active_config_path(
                    terrain, advection_mode, chaos_mode, scheme
                )
                for scheme in MICROPHYSICS_SCHEMES
            }
            for chaos_mode in CHAOS_MODES
        }
        for advection_mode in ADVECTION_MODES
    }
    for terrain in ("bell", "schar")
}
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
SCOPE_ERRORS = (
    "qualified Cartesian A-grid execution is limited to the exact "
    "flat/native complete or mapped CPU reference profiles",
    "qualified mapped terrain is limited to the exact stationary-bell "
    "and constant-velocity Schar CPU reference profiles or the exact dry "
    "stationary Bell fallback-off Vulkan mapped-core profile",
)
FIELDS = ("u", "v", "w", "theta", "p", "rho", "qv", "qr")
ACTIVE_FIELDS = (
    "u", "v", "w", "rho", "p", "theta", "qv", "tracer",
    "qc", "qr", "qi", "qs", "qg", "qh", "tke",
)
LEDGERS = (
    "budget.csv",
    "momentum_budget.csv",
    "energy_budget.csv",
    "radiation_budget.csv",
    "moisture_budget.csv",
    "thermodynamic_source_budget.csv",
)


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def require_success(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed:\n{result.stdout}")
    for marker in ("[SCIENTIFIC INVALIDITY]", "Science profile rejected"):
        if marker in result.stdout:
            raise RuntimeError(f"{label} reported {marker}:\n{result.stdout}")


def require_scope_rejection(
    result: subprocess.CompletedProcess[str], label: str
) -> None:
    if result.returncode == 0 or not any(
        error in result.stdout for error in SCOPE_ERRORS
    ):
        raise RuntimeError(f"{label} did not fail with the exact scope guard:\n{result.stdout}")


def field_payload(output: Path, second: int, field: str):
    path = output / f"step_{second:06d}" / f"{field}.npy"
    if not path.is_file():
        raise RuntimeError(f"missing mapped A-grid output: {path}")
    return read_float32_npy(path)


def active_field_payload(output: Path, second: int, field: str) -> bytes:
    shape, raw, values = field_payload(output, second, field)
    if shape != (18, 18, 18) or any(not math.isfinite(value) for value in values):
        raise RuntimeError(f"invalid active mapped A-grid {field} output")
    if field in ("rho", "p", "theta") and any(value <= 0.0 for value in values):
        raise RuntimeError(f"non-positive active mapped A-grid {field} output")
    if field in (
        "qv", "tracer", "qc", "qr", "qi", "qs", "qg", "qh", "tke"
    ) and any(value < 0.0 for value in values):
        raise RuntimeError(f"negative active mapped A-grid {field} output")
    return raw


def active_geometry_payload(output: Path, second: int, terrain: str) -> bytes:
    manifest_path = output / f"step_{second:06d}" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vertical = manifest.get("grid", {}).get("vertical_geometry", {})
    if vertical.get("storage") != "versioned_npy_geometry":
        raise RuntimeError(
            f"active mapped A-grid {terrain} omitted versioned geometry"
        )
    center_path = vertical.get("center_path")
    if not isinstance(center_path, str) or not center_path:
        raise RuntimeError(
            f"active mapped A-grid {terrain} omitted center geometry path"
        )
    path = (manifest_path.parent / center_path).resolve()
    if not path.is_relative_to(output.resolve()):
        raise RuntimeError(
            f"active mapped A-grid {terrain} geometry escaped its series"
        )
    payload = path.read_bytes()
    if payload[:6] != b"\x93NUMPY" or payload[6] != 1:
        raise RuntimeError(f"invalid active mapped geometry NPY: {path}")
    header_size = struct.unpack_from("<H", payload, 8)[0]
    header_start = 10
    header = ast.literal_eval(
        payload[header_start:header_start + header_size]
        .decode("latin1").strip()
    )
    if (
        header.get("descr") != "<f8" or header.get("fortran_order") or
        tuple(int(value) for value in header["shape"]) != (18, 18, 18)
    ):
        raise RuntimeError(f"active mapped A-grid {terrain} geometry contract changed")
    values = array.array("d")
    values.frombytes(payload[header_start + header_size:])
    if len(values) != 18 * 18 * 18 or any(
        not math.isfinite(value) for value in values
    ):
        raise RuntimeError(f"active mapped A-grid {terrain} geometry is invalid")
    ground = values[0::18]
    if max(ground) - min(ground) <= 0.01:
        raise RuntimeError(f"active mapped A-grid {terrain} geometry is flat")
    return payload


def require_microphysics_activity(output: Path, scheme: str) -> None:
    with (output / "thermodynamic_source_budget.csv").open(
        encoding="utf-8", newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"mapped A-grid {scheme} published no source ledger")
    names = ("qv", "qc", "qr", "qi", "qs", "qg", "qh")
    activity = max(
        abs(float(row[f"microphysics_{name}_requested_kg"]))
        for row in rows for name in names
    )
    if scheme == "none":
        if activity != 0.0:
            raise RuntimeError("mapped A-grid none invented microphysics work")
        return
    if activity <= 0.0:
        raise RuntimeError(f"mapped A-grid {scheme} process owner was inert")
    if any(int(row["microphysics_conservation_active"]) != 1 for row in rows):
        raise RuntimeError(
            f"mapped A-grid {scheme} omitted conservation admission"
        )
    if any(
        abs(float(row["microphysics_conservation_residual_kg"])) /
        max(abs(float(row["microphysics_baseline_water_kg"])), 1.0) > 1.0e-8
        for row in rows
    ):
        raise RuntimeError(f"mapped A-grid {scheme} exceeded water closure")


def require_no_advection_inactivity(output: Path, label: str) -> None:
    with (output / "thermodynamic_source_budget.csv").open(
        encoding="utf-8", newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"active mapped A-grid {label} published no ledger")
    for row in rows:
        for header, raw in row.items():
            if not header.startswith("advection_"):
                continue
            if header.endswith("_complete"):
                if int(raw) != 1:
                    raise RuntimeError(
                        f"active mapped A-grid {label} omitted {header}"
                    )
                continue
            if (
                header.endswith("_step_count") or
                "_schema_version" in header or
                "_basis_version" in header
            ):
                continue
            value = float(raw)
            if not math.isfinite(value) or value != 0.0:
                raise RuntimeError(
                    f"active mapped A-grid {label} authored nonzero "
                    f"no-advection currency {header}={raw}"
                )


def require_dynamics_activity(output: Path, label: str) -> None:
    if all(
        active_field_payload(output, 0, field) ==
        active_field_payload(output, 1, field)
        for field in ("u", "v", "w", "rho", "p")
    ):
        raise RuntimeError(
            f"active mapped A-grid {label} disabled pressure/momentum dynamics"
        )


def require_chaos_targets(output: str, chaos_mode: str, label: str) -> None:
    if f"Initialized chaos scheme: {chaos_mode}" not in output:
        raise RuntimeError(f"active mapped A-grid {label} omitted chaos identity")
    has_ic = "IC variables:" in output
    has_tendencies = "Tendency blocks:" in output
    if chaos_mode == "none":
        if has_ic or has_tendencies:
            raise RuntimeError(f"active mapped A-grid {label} invented chaos targets")
        return
    if chaos_mode in ("initial_conditions", "full_stochastic"):
        for target in ("u(", "v(", "w(", "theta("):
            if target not in output:
                raise RuntimeError(
                    f"active mapped A-grid {label} omitted IC target {target[:-1]}"
                )
    elif has_ic:
        raise RuntimeError(f"active mapped A-grid {label} invented IC targets")
    if chaos_mode == "boundary_layer":
        if "Tendency blocks: pbl(" not in output:
            raise RuntimeError(f"active mapped A-grid {label} omitted PBL target")
        for target in ("microphysics(", "diffusion("):
            if target in output:
                raise RuntimeError(
                    f"active mapped A-grid {label} widened its PBL-only target"
                )
    elif chaos_mode == "full_stochastic":
        for target in ("microphysics(", "pbl(", "diffusion("):
            if target not in output:
                raise RuntimeError(
                    f"active mapped A-grid {label} omitted tendency target {target[:-1]}"
                )
    elif has_tendencies:
        raise RuntimeError(f"active mapped A-grid {label} invented tendency targets")


def render_active_mapped(source: str, qualified: bool) -> str:
    rendered = source
    if not qualified:
        rendered = rendered.replace(
            "  execution:\n    mode: qualified\n",
            "  execution:\n    mode: exploratory\n",
            1,
        )
    return rendered


def render_default_chaos_targets(source: str) -> str:
    target_lines = (
        "    apply_to_ic: u,v,w,theta\n"
        "    apply_to_tendencies: microphysics,pbl,diffusion\n"
    )
    if source.count(target_lines) != 1:
        raise RuntimeError("canonical full-stochastic target block drifted")
    return source.replace(target_lines, "", 1)


def coupled_run(
    binary: Path, config: Path, output: Path, threads: int
):
    result = run([
        str(binary),
        "--headless",
        "--config",
        str(config),
        f"--outdir={output}",
        f"--threads={threads}",
        "--log-profile=normal",
    ])
    require_success(result, f"mapped A-grid run with {threads} worker(s)")
    for marker in (
        "Execution mode: qualified",
        "Dynamics: klemp_wilhelmson_split_cartesian_agrid",
        "Terrain: bell",
        "[COMPUTE] backend=cpu",
        "fallback=off",
        "[TERRAIN_AGRID] owner=cpu, split_explicit=1, retained_flux=1, "
        "scalar_ledgers=1, completed_step=1",
    ):
        if marker not in result.stdout:
            raise RuntimeError(f"mapped A-grid run omitted {marker!r}")
    if result.stdout.count("[TERRAIN_AGRID]") != 20:
        raise RuntimeError("mapped A-grid run did not publish one complete owner row per step")
    return result


def qualify_active_microphysics_profile(
    binary: Path, config: Path, terrain: str, advection_mode: str,
    chaos_mode: str, scheme: str, root: Path, deep: bool,
) -> tuple[dict[str, bytes], dict[str, bytes], bytes] | None:
    label = f"{terrain}/{advection_mode}/{chaos_mode}/{scheme}"
    preflight = run([str(binary), "--validate-only", "--config", str(config)])
    require_success(preflight, f"active mapped A-grid {label} preflight")
    if "[VALIDATE] PASS" not in preflight.stdout:
        raise RuntimeError(
            f"active mapped A-grid {label} preflight omitted PASS"
        )

    serial = root / "serial"
    parallel = root / "parallel"
    resumed = root / "resumed"
    # Deadline rounding can change the final dt; worker comparisons need identical horizons.
    serial_result = run([
        str(binary), "--headless", "--config", str(config),
        f"--duration={2 if deep else 1}", "--write-every=1", "--checkpoint-every=1",
        f"--outdir={serial}", "--threads=1", "--log-profile=normal",
    ])
    require_success(serial_result, f"active mapped A-grid {label} serial")
    for marker in (
        "Execution mode: qualified",
        "Dynamics: klemp_wilhelmson_split_cartesian_agrid",
        f"Numerics-Advection: {advection_mode}",
        f"Terrain: {terrain}",
        f"Initialized microphysics scheme: {scheme}",
        "Initialized turbulence scheme: smagorinsky",
        "Initialized boundary layer scheme: mynn",
        "Initialized radiation scheme: rrtmg",
        f"Initialized chaos scheme: {chaos_mode}",
        "Numerics-Diffusion: implicit",
        "[TERRAIN_AGRID] owner=cpu",
    ):
        if marker not in serial_result.stdout:
            raise RuntimeError(
                f"active mapped A-grid {scheme} omitted {marker!r}"
            )
    require_chaos_targets(serial_result.stdout, chaos_mode, label)
    metric = re.search(
        rf"Terrain fields built: scheme={terrain}, max_height=(\S+) m, "
        rf"min_J=(\S+), max_J=(\S+)",
        serial_result.stdout,
    )
    if metric is None or any(
        not math.isfinite(float(value)) for value in metric.groups()
    ) or float(metric.group(1)) <= 0.0 or float(metric.group(2)) <= 0.0 or \
            float(metric.group(3)) <= 0.0:
        raise RuntimeError(
            f"active mapped A-grid {label} omitted positive metric evidence"
        )
    if serial_result.stdout.count("[TERRAIN_AGRID]") != (20 if deep else 10):
        raise RuntimeError(
            f"active mapped A-grid {scheme} omitted completed-step owners"
        )
    if advection_mode == "none":
        require_no_advection_inactivity(serial, label)
        require_dynamics_activity(serial, label)
    if not deep:
        for field in ACTIVE_FIELDS:
            active_field_payload(serial, 1, field)
        require_microphysics_activity(serial, scheme)
        active_geometry_payload(serial, 1, terrain)
        return None

    parallel_result = run([
        str(binary), "--headless", "--config", str(config),
        "--write-every=1", "--checkpoint-every=1",
        f"--outdir={parallel}", "--threads=4", "--log-profile=normal",
    ])
    require_success(
        parallel_result, f"active mapped A-grid {label} parallel"
    )
    if "threads=4" not in parallel_result.stdout or \
            parallel_result.stdout.count("[TERRAIN_AGRID]") != 20:
        raise RuntimeError(
            f"active mapped A-grid {scheme} parallel ownership is incomplete"
        )
    if chaos_mode in ("initial_conditions", "full_stochastic"):
        for field in ACTIVE_FIELDS:
            if active_field_payload(serial, 0, field) != active_field_payload(
                parallel, 0, field
            ):
                raise RuntimeError(
                    f"active mapped A-grid {label} {field} initial state "
                    "differs between one and four workers"
                )
    for second in (1, 2):
        for field in ACTIVE_FIELDS:
            if active_field_payload(serial, second, field) != active_field_payload(parallel, second, field):
                raise RuntimeError(f"active mapped A-grid {label} {field} differs across workers at {second}s")
    serial_geometry = active_geometry_payload(serial, 1, terrain)
    parallel_geometry = active_geometry_payload(parallel, 1, terrain)
    if serial_geometry != parallel_geometry:
        raise RuntimeError(
            f"active mapped A-grid {label} geometry differs across workers"
        )
    for ledger in LEDGERS:
        if (serial / ledger).read_bytes() != (parallel / ledger).read_bytes():
            raise RuntimeError(f"active mapped A-grid {label} {ledger} differs across workers")
    require_report_owner_identities(serial)
    require_report_owner_identities(parallel)
    require_microphysics_activity(serial, scheme)
    if advection_mode == "none":
        require_no_advection_inactivity(parallel, label)

    midpoint = parallel / "checkpoints/checkpoint_step_000000010.nimbus"
    serial_midpoint = serial / "checkpoints/checkpoint_step_000000010.nimbus"
    for checkpoint in (serial_midpoint, midpoint):
        if not checkpoint.is_file() or not Path(f"{checkpoint}.json").is_file():
            raise RuntimeError(
                f"active mapped A-grid {label} omitted midpoint checkpoint"
            )
    for step in (10, 20):
        name = f"checkpoints/checkpoint_step_{step:09d}.nimbus"
        if (serial / name).read_bytes() != (parallel / name).read_bytes():
            raise RuntimeError(f"active mapped A-grid {label} checkpoint {step} differs across workers")
    continuation = run([
        str(binary), "--headless", "--config", str(config),
        f"--resume={midpoint}", "--duration=2", "--write-every=1",
        "--checkpoint-every=1", f"--outdir={resumed}", "--threads=4",
        "--log-profile=normal",
    ])
    require_success(
        continuation, f"active mapped A-grid {label} continuation"
    )
    if "[CHECKPOINT] resumed step=10, time_s=1" not in continuation.stdout:
        raise RuntimeError(
            f"active mapped A-grid {label} omitted midpoint restore"
        )
    initial_payloads = {
        field: active_field_payload(serial, 0, field)
        for field in ACTIVE_FIELDS
    }
    final_payloads: dict[str, bytes] = {}
    for field in ACTIVE_FIELDS:
        final_payload = active_field_payload(parallel, 2, field)
        if final_payload != active_field_payload(resumed, 1, field):
            raise RuntimeError(
                f"active mapped A-grid {label} {field} continuation mismatch"
            )
        final_payloads[field] = final_payload
    continuous_checkpoint = (
        parallel / "checkpoints/checkpoint_step_000000020.nimbus"
    )
    resumed_checkpoint = (
        resumed / "checkpoints/checkpoint_step_000000020.nimbus"
    )
    if (not continuous_checkpoint.is_file() or
            not resumed_checkpoint.is_file() or
            continuous_checkpoint.read_bytes() != resumed_checkpoint.read_bytes()):
        raise RuntimeError(
            f"active mapped A-grid {label} private checkpoint continuation differs"
        )
    require_report_owner_identities(resumed)
    require_restart_suffix(parallel, resumed)
    if advection_mode == "none":
        require_no_advection_inactivity(resumed, label)
    final_geometry = active_geometry_payload(parallel, 2, terrain)
    if final_geometry != active_geometry_payload(resumed, 1, terrain):
        raise RuntimeError(
            f"active mapped A-grid {label} geometry changed across restart"
        )
    return initial_payloads, final_payloads, final_geometry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    binary = args.binary.resolve()
    active_paths = [
        path for terrain_configs in ACTIVE_CONFIGS.values()
        for advection_configs in terrain_configs.values()
        for chaos_configs in advection_configs.values()
        for path in chaos_configs.values()
    ]
    for path in (binary, CONFIG, *active_paths):
        if not path.is_file():
            raise RuntimeError(f"required mapped A-grid input is missing: {path}")

    preflight = run([str(binary), "--validate-only", "--config", str(CONFIG)])
    require_success(preflight, "mapped A-grid qualified preflight")
    if "[VALIDATE] PASS" not in preflight.stdout:
        raise RuntimeError("mapped A-grid preflight did not publish PASS")

    with tempfile.TemporaryDirectory(prefix="nimbus-collocated-terrain-ready-") as temporary:
        root = Path(temporary)
        source = CONFIG.read_text()
        default_targets_config = root / "full-stochastic-default-targets.yaml"
        default_targets_config.write_text(
            render_default_chaos_targets(
                ACTIVE_CONFIGS["bell"]["weno5"]["full_stochastic"][
                    "thompson"
                ].read_text(encoding="utf-8")
            ),
            encoding="utf-8",
        )
        active_source = render_active_mapped(
            ACTIVE_CONFIGS["bell"]["weno5"]["full_stochastic"][
                "thompson"
            ].read_text(encoding="utf-8"),
            True,
        )
        active_mutations = (
            ("terrain", "    h0: 220\n", "    h0: 221\n"),
            ("diffusion", "  K_h: 4000\n", "  K_h: 3999\n"),
            ("member", "    member_id: 4\n", "    member_id: 5\n"),
            ("ic-targets", "    apply_to_ic: u,v,w,theta\n",
             "    apply_to_ic: u,v,w\n"),
            ("tendency-targets",
             "    apply_to_tendencies: microphysics,pbl,diffusion\n",
             "    apply_to_tendencies: microphysics,pbl\n"),
            ("native-generation", "    native_generation_required: false\n",
             "    native_generation_required: true\n"),
            ("advection-auto", "advection:\n  scheme: weno5\n",
             "advection:\n  scheme: auto\n"),
            ("ownership", "    allow_fallback: false\n", "    allow_fallback: true\n"),
        )
        for name, old, new in active_mutations:
            invalid_source = active_source.replace(old, new, 1)
            if invalid_source == active_source:
                raise RuntimeError(f"active mapped A-grid {name} mutation drifted")
            invalid = root / f"active-invalid-{name}.yaml"
            invalid.write_text(invalid_source, encoding="utf-8")
            require_scope_rejection(
                run([str(binary), "--validate-only", "--config", str(invalid)]),
                f"active mapped A-grid {name} neighbor",
            )
        schar_source = ACTIVE_CONFIGS["schar"]["weno5"][
            "full_stochastic"
        ]["thompson"].read_text(encoding="utf-8")
        invalid_schar_source = schar_source.replace("    H: 220\n", "    H: 221\n", 1)
        if invalid_schar_source == schar_source:
            raise RuntimeError("active mapped A-grid Schär mutation drifted")
        invalid_schar = root / "active-invalid-schar-height.yaml"
        invalid_schar.write_text(invalid_schar_source, encoding="utf-8")
        require_scope_rejection(
            run([str(binary), "--validate-only", "--config", str(invalid_schar)]),
            "active mapped A-grid Schär height neighbor",
        )
        active_geometry: dict[str, bytes] = {}
        tasks = [
            (terrain, advection_mode, chaos_mode, scheme, config)
            for terrain in ("schar", "bell")
            for advection_mode in ADVECTION_MODES
            for chaos_mode in CHAOS_MODES
            for scheme, config in ACTIVE_CONFIGS[terrain][advection_mode][
                chaos_mode
            ].items()
        ]
        deep_results: dict[
            str, dict[
                str, dict[
                    str, dict[
                        str, tuple[dict[str, bytes], dict[str, bytes], bytes]
                    ]
                ]
            ]
        ] = {
            terrain: {
                advection_mode: {
                    chaos_mode: {} for chaos_mode in CHAOS_MODES
                }
                for advection_mode in ADVECTION_MODES
            }
            for terrain in ACTIVE_CONFIGS
        }
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    qualify_active_microphysics_profile,
                    binary, config, terrain, advection_mode, chaos_mode, scheme,
                    root / "active-profiles" / terrain / advection_mode /
                        chaos_mode / scheme,
                    (
                        advection_mode == "weno5" and
                        (chaos_mode == "full_stochastic" or scheme == "thompson")
                    ) or (
                        chaos_mode == "full_stochastic" and
                        scheme == "thompson"
                    ),
                ): (terrain, advection_mode, chaos_mode, scheme)
                for terrain, advection_mode, chaos_mode, scheme, config in tasks
            }
            for future in as_completed(futures):
                terrain, advection_mode, chaos_mode, scheme = futures[future]
                result = future.result()
                if result is None:
                    continue
                initial, final, geometry = result
                deep_results[terrain][advection_mode][chaos_mode][scheme] = (
                    initial, final, geometry
                )
                if advection_mode != "weno5" or chaos_mode != "full_stochastic":
                    continue
                if terrain in active_geometry and active_geometry[terrain] != geometry:
                    raise RuntimeError(
                        f"active mapped A-grid {terrain} geometry changed by microphysics"
                    )
                active_geometry[terrain] = geometry
        microphysics_fields = (
            "theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh",
        )
        for terrain in ACTIVE_CONFIGS:
            full_results = deep_results[terrain]["weno5"]["full_stochastic"]
            for scheme in MICROPHYSICS_SCHEMES:
                if scheme == "none":
                    continue
                if all(
                    full_results[scheme][1][field] ==
                        full_results["none"][1][field]
                    for field in microphysics_fields
                ):
                    raise RuntimeError(
                        f"active mapped A-grid {terrain}/{scheme} matched none"
                    )
        if active_geometry["bell"] == active_geometry["schar"]:
            raise RuntimeError("active Bell and Schär mapped geometry are identical")
        ic_targets = ("u", "v", "w", "theta")
        for terrain in ACTIVE_CONFIGS:
            representatives = {
                chaos_mode: deep_results[terrain]["weno5"][chaos_mode][
                    "thompson"
                ]
                for chaos_mode in CHAOS_MODES
            }
            baseline_initial, baseline_final, baseline_geometry = representatives[
                "none"
            ]
            for chaos_mode, (initial, final, geometry) in representatives.items():
                if geometry != baseline_geometry:
                    raise RuntimeError(
                        f"active mapped A-grid {terrain} geometry changed with "
                        f"chaos={chaos_mode}"
                    )
                if chaos_mode in ("none", "boundary_layer"):
                    if initial != baseline_initial:
                        raise RuntimeError(
                            f"active mapped A-grid {terrain}/{chaos_mode} "
                            "invented initial-condition activity"
                        )
                elif all(initial[field] == baseline_initial[field]
                         for field in ic_targets):
                    raise RuntimeError(
                        f"active mapped A-grid {terrain}/{chaos_mode} IC targets "
                        "were inert"
                    )
                if chaos_mode != "none" and all(
                    final[field] == baseline_final[field]
                    for field in ACTIVE_FIELDS
                ):
                    raise RuntimeError(
                        f"active mapped A-grid {terrain}/{chaos_mode} matched none"
                    )
            for left_mode, right_mode in (
                ("full_stochastic", "initial_conditions"),
                ("full_stochastic", "boundary_layer"),
            ):
                left_final = representatives[left_mode][1]
                right_final = representatives[right_mode][1]
                if all(
                    left_final[field] == right_final[field]
                    for field in ACTIVE_FIELDS
                ):
                    raise RuntimeError(
                        f"active mapped A-grid {terrain} {left_mode} matched "
                        f"{right_mode} in every affected field"
                    )
        transport_fields = (
            "theta", "qv", "tracer", "qc", "qr", "qi", "qs", "qg",
            "qh", "tke",
        )
        for terrain in ACTIVE_CONFIGS:
            representatives = {
                advection_mode: deep_results[terrain][advection_mode][
                    "full_stochastic"
                ]["thompson"]
                for advection_mode in ADVECTION_MODES
            }
            baseline_initial, _, baseline_geometry = representatives["weno5"]
            for advection_mode, (initial, _, geometry) in representatives.items():
                if initial != baseline_initial:
                    raise RuntimeError(
                        f"active mapped A-grid {terrain}/{advection_mode} "
                        "changed initialization before advection"
                    )
                if geometry != baseline_geometry:
                    raise RuntimeError(
                        f"active mapped A-grid {terrain} geometry changed with "
                        f"advection={advection_mode}"
                    )
            for left_mode, right_mode in (
                ("weno5", "tvd"),
                ("none", "weno5"),
                ("none", "tvd"),
            ):
                left_final = representatives[left_mode][1]
                right_final = representatives[right_mode][1]
                if all(
                    left_final[field] == right_final[field]
                    for field in transport_fields
                ):
                    raise RuntimeError(
                        f"active mapped A-grid {terrain} advection={left_mode} "
                        f"matched {right_mode} in every transported field"
                    )
        default_targets_result = qualify_active_microphysics_profile(
            binary, default_targets_config, "bell", "weno5",
            "full_stochastic", "thompson", root / "active-default-targets",
            True,
        )
        if default_targets_result is None:
            raise RuntimeError("default-target full-stochastic profile was not deep")
        if default_targets_result != deep_results["bell"]["weno5"][
            "full_stochastic"
        ]["thompson"]:
            raise RuntimeError(
                "default-target full-stochastic profile did not resolve to the "
                "canonical explicit Bell/Thompson result"
            )
        mutations = (
            ("geometry", "  nx: 18\n", "  nx: 19\n"),
            ("terrain", "    h0: 220\n", "    h0: 221\n"),
            ("boundary",
             "    west: periodic\n    east: periodic\n",
             "    west: rigid\n    east: rigid\n"),
            ("ownership", "    allow_fallback: false\n", "    allow_fallback: true\n"),
        )
        for name, old, new in mutations:
            invalid_source = source.replace(old, new, 1)
            if invalid_source == source:
                raise RuntimeError(f"mapped A-grid {name} mutation did not alter the fixture")
            invalid = root / f"invalid-{name}.yaml"
            invalid.write_text(invalid_source)
            require_scope_rejection(
                run([str(binary), "--validate-only", "--config", str(invalid)]),
                f"mapped A-grid {name} validate-only neighbor",
            )
            require_scope_rejection(
                run([
                    str(binary), "--headless", "--config", str(invalid),
                    "--duration=0", f"--outdir={root / ('invalid-' + name)}",
                ]),
                f"mapped A-grid {name} runtime neighbor",
            )

        serial = root / "serial"
        parallel = root / "parallel"
        resumed = root / "resumed"
        coupled_run(binary, CONFIG, serial, 1)
        coupled_run(binary, CONFIG, parallel, 4)

        evolved = set()
        for field in FIELDS:
            initial_shape, initial_raw, _ = field_payload(parallel, 0, field)
            final_shape, parallel_raw, values = field_payload(parallel, 2, field)
            serial_shape, serial_raw, _ = field_payload(serial, 2, field)
            if initial_shape != (18, 18, 18) or final_shape != initial_shape:
                raise RuntimeError(f"mapped A-grid {field} shape disagrees with the fixture")
            if serial_shape != final_shape or serial_raw != parallel_raw:
                raise RuntimeError(f"mapped A-grid {field} differs between one and four workers")
            if initial_raw != parallel_raw:
                evolved.add(field)
            if any(not math.isfinite(value) for value in values):
                raise RuntimeError(f"mapped A-grid {field} contains non-finite values")
            if field in ("rho", "p") and any(value <= 0.0 for value in values):
                raise RuntimeError(f"mapped A-grid {field} is not strictly positive")
            if field in ("qv", "qr") and any(value < 0.0 for value in values):
                raise RuntimeError(f"mapped A-grid {field} is negative")
        for field in ("w", "theta", "p", "rho"):
            if field not in evolved:
                raise RuntimeError(f"mapped A-grid {field} did not evolve")

        for ledger in LEDGERS:
            serial_path = serial / ledger
            parallel_path = parallel / ledger
            if not serial_path.is_file() or not parallel_path.is_file():
                raise RuntimeError(f"mapped A-grid run omitted {ledger}")
            if serial_path.read_bytes() != parallel_path.read_bytes():
                raise RuntimeError(f"mapped A-grid {ledger} differs between one and four workers")
        serial_checks = require_report_owner_identities(serial)
        parallel_checks = require_report_owner_identities(parallel)
        if serial_checks == 0 or serial_checks != parallel_checks:
            raise RuntimeError("mapped A-grid owner-identity coverage is incomplete")

        checkpoint = parallel / "checkpoints/checkpoint_step_000000010.nimbus"
        if not checkpoint.is_file() or not Path(f"{checkpoint}.json").is_file():
            raise RuntimeError("mapped A-grid run omitted its midpoint checkpoint pair")
        continuation = run([
            str(binary), "--headless", "--config", str(CONFIG),
            f"--resume={checkpoint}", f"--outdir={resumed}",
            "--threads=4", "--log-profile=normal",
        ])
        require_success(continuation, "mapped A-grid midpoint continuation")
        if "[CHECKPOINT] resumed step=10, time_s=1" not in continuation.stdout:
            raise RuntimeError("mapped A-grid continuation did not restore the midpoint")
        for field in FIELDS:
            final_shape, final_raw, _ = field_payload(parallel, 2, field)
            resumed_shape, resumed_raw, _ = field_payload(resumed, 1, field)
            if resumed_shape != final_shape or resumed_raw != final_raw:
                raise RuntimeError(f"mapped A-grid {field} continuation is not bitwise exact")
        require_report_owner_identities(resumed)
        require_restart_suffix(parallel, resumed)

        moving_source = source.replace(
            "  mode: stationary\n",
            "  mode: constant_velocity\n"
            "  frame_velocity:\n"
            "    u_ms: 7\n"
            "    v_ms: -3\n",
            1,
        )
        for old, new in (
            ("    u_sfc_ms: 0\n", "    u_sfc_ms: 7\n"),
            ("    v_sfc_ms: 0\n", "    v_sfc_ms: -3\n"),
            ("    u_1km_ms: 0\n", "    u_1km_ms: 7\n"),
            ("    v_1km_ms: 0\n", "    v_1km_ms: -3\n"),
            ("    u_6km_ms: 0\n", "    u_6km_ms: 7\n"),
            ("    v_6km_ms: 0\n", "    v_6km_ms: -3\n"),
        ):
            moving_source = moving_source.replace(old, new, 1)
        moving_config = root / "moving.yaml"
        moving_config.write_text(moving_source)
        moving_preflight = run([
            str(binary), "--validate-only", "--config", str(moving_config)
        ])
        require_success(moving_preflight, "moving mapped A-grid preflight")

        moving_serial = root / "moving-serial"
        moving_parallel = root / "moving-parallel"
        moving_resumed = root / "moving-resumed"
        coupled_run(binary, moving_config, moving_serial, 1)
        coupled_run(binary, moving_config, moving_parallel, 4)
        envelopes = {
            "u": 0.012, "v": 0.008, "w": 0.02,
            "theta": 0.12, "p": 2.0, "rho": 4.0e-4,
            "qv": 1.2e-5, "qr": 2.0e-7,
        }
        for field in FIELDS:
            stationary_shape, _, stationary_values = field_payload(
                parallel, 2, field
            )
            moving_shape, moving_raw, moving_values = field_payload(
                moving_parallel, 2, field
            )
            moving_serial_shape, moving_serial_raw, _ = field_payload(
                moving_serial, 2, field
            )
            if moving_shape != stationary_shape or moving_serial_shape != moving_shape:
                raise RuntimeError(f"moving mapped A-grid {field} shape mismatch")
            if moving_serial_raw != moving_raw:
                raise RuntimeError(f"moving mapped A-grid {field} differs across workers")
            offset = 7.0 if field == "u" else -3.0 if field == "v" else 0.0
            maximum = max(
                abs((moving - offset) - stationary)
                for stationary, moving in zip(stationary_values, moving_values)
            )
            if maximum > envelopes[field]:
                raise RuntimeError(
                    f"moving mapped A-grid {field} Galilean error {maximum} "
                    f"exceeds {envelopes[field]}"
                )
        for ledger in LEDGERS:
            if (moving_serial / ledger).read_bytes() != (
                moving_parallel / ledger
            ).read_bytes():
                raise RuntimeError(
                    f"moving mapped A-grid {ledger} differs across workers"
                )
        require_report_owner_identities(moving_parallel)
        moving_checkpoint = (
            moving_parallel /
            "checkpoints/checkpoint_step_000000010.nimbus"
        )
        continuation = run([
            str(binary), "--headless", "--config", str(moving_config),
            f"--resume={moving_checkpoint}", f"--outdir={moving_resumed}",
            "--threads=4", "--log-profile=normal",
        ])
        require_success(continuation, "moving mapped A-grid continuation")
        for field in FIELDS:
            _, final_raw, _ = field_payload(moving_parallel, 2, field)
            _, resumed_raw, _ = field_payload(moving_resumed, 1, field)
            if resumed_raw != final_raw:
                raise RuntimeError(
                    f"moving mapped A-grid {field} continuation mismatch"
                )
        require_report_owner_identities(moving_resumed)
        require_restart_suffix(moving_parallel, moving_resumed)

    print(
        "[collocated-mapped-terrain-ready] PASS: stationary and constant-"
        "velocity periodic/rigid Bell CPU profiles and the qualified active "
        "Bell/Schär all-microphysics/Smagorinsky/MYNN/RRTMG/implicit family "
        "validated and evolved all 120 advection/chaos profiles; existing "
        "WENO deep coverage plus both-terrain TVD/WENO/no-advection "
        "representatives retained distinct transport, inactive no-advection "
        "currency with active dynamics/physics, positive invariant geometry, "
        "complete ledgers, bounded threads, private checkpoints, and restart"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
