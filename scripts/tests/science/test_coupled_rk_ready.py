#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import subprocess
import tempfile

from npy_support import read_float32_npy


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
CONFIGS = {
    "rk3": ROOT / "tests/fixtures/configs/qualification/cgrid_ssprk3_coupled_ready.yaml",
    "rk4": ROOT / "tests/fixtures/configs/qualification/cgrid_rk4_coupled_ready.yaml",
}
LIN_CONFIG = ROOT / "tests/fixtures/configs/qualification/cgrid_lin_rrtmg_ready.yaml"
THOMPSON_CONFIG = ROOT / "tests/fixtures/configs/qualification/cgrid_thompson_cpu_ready.yaml"
TKE_CONFIG = ROOT / "tests/fixtures/configs/qualification/cgrid_tke_rrtmg_ready.yaml"
MYNN_CONFIG = ROOT / "tests/fixtures/configs/qualification/cgrid_mynn_tke_top_ready.yaml"
FIELDS = (
    "u", "v", "w", "rho", "p", "theta", "qv", "tracer",
    "qc", "qr", "qi", "qs", "qg", "qh", "tke",
)


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False
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
    expected = (
        "qualified coupled RK integration requires a ready flat CPU",
        "is development-only and unavailable for simulation runs",
        "does not support backend=gpu",
        "qualified execution requires maturity=qualified",
        "qualified mapped terrain is limited",
        "microphysics.scheme=kessler is qualified only",
        "turbulence.scheme=tke is qualified only",
        "turbulence.scheme=tke is outside the qualified flat, RK, A-grid, and mapped CPU families",
        "advection.scheme=weno5 is qualified only",
    )
    if result.returncode == 0 or not any(
        marker in result.stdout for marker in expected
    ):
        raise RuntimeError(f"{label} did not fail with the exact scope guard:\n{result.stdout}")


def field_payload(
    output: Path,
    second: int,
    field: str,
    expected_shape: tuple[int, int, int] = (12, 12, 16),
) -> bytes:
    path = output / f"step_{second:06d}" / f"{field}.npy"
    if not path.is_file():
        raise RuntimeError(f"missing output field: {path}")
    shape, raw, values = read_float32_npy(path)
    if shape != expected_shape:
        raise RuntimeError(f"{field} has unexpected shape {shape}")
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError(f"{field} contains non-finite values")
    if field in ("rho", "p", "theta") and any(value <= 0.0 for value in values):
        raise RuntimeError(f"{field} violates its positive-state contract")
    if field in ("qv", "tracer", "qc", "qr", "qi", "qs", "qg", "qh", "tke") and any(
        value < 0.0 for value in values
    ):
        raise RuntimeError(f"{field} violates its nonnegative-state contract")
    if field == "tracer" and any(value > 1.5 for value in values):
        raise RuntimeError("tracer violates its upper-bound contract")
    if field in ("qv", "qc", "qr", "qi", "qs", "qg", "qh") and any(
        value > 0.05 for value in values
    ):
        raise RuntimeError(f"{field} violates its mixing-ratio upper bound")
    return raw


def require_complete_unsplit_ledger(output: Path, label: str, *, require_every_step_active: bool = True) -> None:
    path = output / "thermodynamic_source_budget.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"{label} unsplit ledger is empty")
    for index, row in enumerate(rows, start=1):
        if float(row["completed_step_energy_step_complete"]) != 1.0 or float(
            row["completed_step_energy_cumulative_complete"]
        ) != 1.0:
            raise RuntimeError(
                f"{label} row {index} has incomplete completed-step ownership"
            )
        owner = float(row[
            "completed_step_energy_step_unsplit_rk_committed_change_j"
        ])
        if require_every_step_active and owner == 0.0:
            raise RuntimeError(f"{label} row {index} has an inert unsplit owner")
        committed = float(row["completed_step_energy_step_committed_owner_sum_j"])
        residual = float(row["completed_step_energy_step_closure_residual_j"])
        if abs(residual) / max(1.0, abs(committed)) > 1.0e-12:
            raise RuntimeError(f"{label} row {index} has unclosed unsplit ownership")


def coupled_run(
    binary: Path, config: Path, output: Path, threads: int, scheme: str
) -> subprocess.CompletedProcess[str]:
    result = run([
        str(binary), "--headless", "--config", str(config),
        f"--outdir={output}", "--write-every=1", f"--threads={threads}",
        "--log-profile=normal",
    ])
    require_success(result, f"{scheme} run with {threads} worker(s)")
    require_complete_unsplit_ledger(output, f"{scheme}/{threads}")
    integrator = scheme.split("-", 1)[0]
    for marker in (
        "Execution mode: qualified",
        f"Time stepping: {integrator}",
        "[COMPUTE] backend=cpu",
        "fallback=off",
    ):
        if marker not in result.stdout:
            raise RuntimeError(f"{scheme} run omitted {marker!r}")
    if scheme.rsplit("-", 1)[-1] in ("kessler", "active", "column") and "Initialized microphysics scheme: kessler" not in result.stdout:
        raise RuntimeError(f"{scheme} did not initialize the Kessler process owner")
    if scheme.endswith("-active"):
        for marker in (
            "Initialized turbulence scheme: smagorinsky",
            "Numerics-Diffusion: explicit",
        ):
            if marker not in result.stdout:
                raise RuntimeError(f"{scheme} did not initialize active owner {marker!r}")
    if scheme.endswith("-column"):
        for marker in (
            "Initialized radiation scheme: simple_grey",
            "Initialized boundary layer scheme: slab",
        ):
            if marker not in result.stdout:
                raise RuntimeError(f"{scheme} did not initialize column owner {marker!r}")
    if "-lin-" in scheme:
        for marker in (
            "Initialized microphysics scheme: lin",
            "Initialized radiation scheme: rrtmg",
            "Initialized boundary layer scheme: ysu",
        ):
            if marker not in result.stdout:
                raise RuntimeError(f"{scheme} did not initialize Lin-stack owner {marker!r}")
    if "-thompson-" in scheme:
        for marker in (
            "Initialized microphysics scheme: thompson",
            "Initialized radiation scheme: rrtmg",
            "Initialized boundary layer scheme: ysu",
        ):
            if marker not in result.stdout:
                raise RuntimeError(f"{scheme} did not initialize Thompson-stack owner {marker!r}")
    if scheme.endswith("-milbrandt") and (
        "Initialized microphysics scheme: milbrandt" not in result.stdout
    ):
        raise RuntimeError(f"{scheme} did not initialize the Milbrandt process owner")
    if "-tke-" in scheme and "-mynn-" not in scheme and (
        "Initialized turbulence scheme: tke" not in result.stdout
    ):
        raise RuntimeError(f"{scheme} did not initialize the TKE process owner")
    if "-mynn-" in scheme:
        for marker in (
            "Initialized boundary layer scheme: mynn",
            "Initialized radiation scheme: rrtmg",
        ):
            if marker not in result.stdout:
                raise RuntimeError(f"{scheme} did not initialize MYNN-stack owner {marker!r}")
    if "-none-none" in scheme and (
        "Numerics-Advection: none" not in result.stdout
    ):
        raise RuntimeError(f"{scheme} did not select the no-advection equation")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        raise RuntimeError(f"missing simulator: {binary}")

    with tempfile.TemporaryDirectory(prefix="nimbus-coupled-rk-ready-") as temporary:
        root = Path(temporary)
        final_payloads: dict[str, dict[str, bytes]] = {}
        for scheme, config in CONFIGS.items():
            if not config.is_file():
                raise RuntimeError(f"missing Ready fixture: {config}")
            preflight = run([str(binary), "--validate-only", "--config", str(config)])
            require_success(preflight, f"{scheme} qualified preflight")
            if "[VALIDATE] PASS" not in preflight.stdout:
                raise RuntimeError(f"{scheme} preflight did not publish PASS")

            source = config.read_text()
            for label, old, new in (
                ("backend", "    backend: cpu\n", "    backend: metal\n"),
                ("physics", "turbulence:\n  scheme: none\n", "turbulence:\n  scheme: tke\n"),
                ("terrain", "terrain:\n  scheme: none\n", "terrain:\n  scheme: bell\n"),
            ):
                mutated = source.replace(old, new, 1)
                if mutated == source:
                    raise RuntimeError(f"{scheme} {label} mutation did not alter fixture")
                path = root / f"{scheme}-invalid-{label}.yaml"
                path.write_text(mutated)
                require_scope_rejection(
                    run([str(binary), "--validate-only", "--config", str(path)]),
                    f"{scheme} {label} scope guard",
                )

            flexible_source = source.replace("  nx: 12\n", "  nx: 16\n", 1)
            flexible_source = flexible_source.replace("  ny: 12\n", "  ny: 14\n", 1)
            flexible_source = flexible_source.replace("  nz: 16\n", "  nz: 18\n", 1)
            flexible_source = flexible_source.replace("  ztop: 4000\n", "  ztop: 4500\n", 1)
            flexible_source = flexible_source.replace("  dt_large: 0.05\n", "  dt_large: 0.04\n", 1)
            flexible = root / f"{scheme}-flexible-domain.yaml"
            flexible.write_text(flexible_source)
            flexible_preflight = run([
                str(binary), "--validate-only", "--config", str(flexible)
            ])
            require_success(flexible_preflight, f"{scheme} flexible-domain preflight")
            flexible_run = run([
                str(binary), "--headless", "--config", str(flexible),
                "--duration=1", "--write-every=0", "--checkpoint-every=0",
                f"--outdir={root / f'{scheme}-flexible-domain'}", "--threads=1",
                "--log-profile=normal",
            ])
            require_success(flexible_run, f"{scheme} flexible-domain run")

            for advection, physics in (
                ("none", "none"),
                ("weno5", "none"),
                ("tvd", "none"),
                ("weno5", "kessler"),
                ("tvd", "kessler"),
                ("tvd", "active"),
                ("weno5", "active"),
                ("tvd", "column"),
                ("weno5", "column"),
                ("tvd", "milbrandt"),
                ("weno5", "milbrandt"),
            ):
                variant_source = source.replace(
                    "advection:\n  scheme: weno5\n",
                    f"advection:\n  scheme: {advection}\n",
                    1,
                ).replace(
                    "output:\n  preset: minimal\n",
                    "output:\n  preset: dynamics_full\n",
                    1,
                )
                variant_source = variant_source.replace(
                    "    u_sfc_ms: 0\n", "    u_sfc_ms: 5\n", 1
                ).replace("    u_1km_ms: 0\n", "    u_1km_ms: 10\n", 1).replace(
                    "    u_6km_ms: 0\n", "    u_6km_ms: 20\n", 1
                )
                if physics in ("kessler", "active", "column"):
                    variant_source = variant_source.replace(
                        "microphysics:\n  scheme: none\n",
                        "microphysics:\n  scheme: kessler\n",
                        1,
                    )
                elif physics == "milbrandt":
                    variant_source = variant_source.replace(
                        "microphysics:\n  scheme: none\n",
                        "microphysics:\n  scheme: milbrandt\n  milbrandt:\n    hail_processes: true\n",
                        1,
                    ).replace(
                        "      dtheta_k: 0.5\n", "      dtheta_k: 2\n", 1
                    )
                if physics == "active":
                    variant_source = variant_source.replace(
                        "turbulence:\n  scheme: none\n",
                        "turbulence:\n  scheme: smagorinsky\n",
                        1,
                    ).replace("  K_h: 0\n", "  K_h: 5\n", 1).replace(
                        "  K_v: 0\n", "  K_v: 2\n", 1
                    )
                if physics == "column":
                    variant_source = variant_source.replace(
                        "surface:\n  scheme: none\n  apply_fluxes: false\n",
                        "surface:\n  scheme: slab\n  apply_fluxes: true\n  tsk_k: 300\n",
                        1,
                    ).replace(
                        "boundary_layer:\n  scheme: none\n",
                        "boundary_layer:\n  scheme: slab\n",
                        1,
                    ).replace(
                        "radiation:\n  scheme: none\n",
                        "radiation:\n  scheme: simple_grey\n  solar_forcing: idealized_diurnal\n  latitude_deg: 35\n  declination_deg: 15\n  local_solar_time_at_start_h: 0\n",
                        1,
                    )
                variant = root / f"{scheme}-{advection}-{physics}.yaml"
                variant.write_text(variant_source)
                preflight = run([str(binary), "--validate-only", "--config", str(variant)])
                require_success(preflight, f"{scheme}/{advection} qualified preflight")

                key = f"{scheme}-{advection}-{physics}"
                serial = root / f"{key}-serial"
                parallel = root / f"{key}-parallel"
                resumed = root / f"{key}-resumed"
                coupled_run(binary, variant, serial, 1, key)
                coupled_run(binary, variant, parallel, 4, key)

                evolved = False
                held_scalars = True
                final_payloads[key] = {}
                for field in FIELDS:
                    initial = field_payload(parallel, 0, field)
                    serial_final = field_payload(serial, 2, field)
                    parallel_final = field_payload(parallel, 2, field)
                    if serial_final != parallel_final:
                        raise RuntimeError(
                            f"{key} {field} differs between one and four workers"
                        )
                    evolved = evolved or initial != parallel_final
                    if advection == "none" and field in (
                        "theta", "qv", "tracer", "qc", "qr", "qi", "qs",
                        "qg", "qh", "tke",
                    ):
                        held_scalars = held_scalars and initial == parallel_final
                    final_payloads[key][field] = parallel_final
                if not evolved:
                    raise RuntimeError(f"{key} qualified profile did not evolve")
                if advection == "none" and not held_scalars:
                    raise RuntimeError(f"{key} changed a held transported scalar")

                serial_checkpoint = serial / "checkpoints/checkpoint_step_000000020.nimbus"
                checkpoint = parallel / "checkpoints/checkpoint_step_000000020.nimbus"
                if not checkpoint.is_file() or not Path(f"{checkpoint}.json").is_file():
                    raise RuntimeError(f"{key} did not publish its midpoint checkpoint")
                if serial_checkpoint.read_bytes() != checkpoint.read_bytes():
                    raise RuntimeError(f"{key} private/public checkpoint differs across workers")
                other_scheme = "rk4" if scheme == "rk3" else "rk3"
                other_source = CONFIGS[other_scheme].read_text().replace(
                    "advection:\n  scheme: weno5\n",
                    f"advection:\n  scheme: {advection}\n",
                    1,
                ).replace(
                    "output:\n  preset: minimal\n",
                    "output:\n  preset: dynamics_full\n",
                    1,
                )
                other_source = other_source.replace(
                    "    u_sfc_ms: 0\n", "    u_sfc_ms: 5\n", 1
                ).replace("    u_1km_ms: 0\n", "    u_1km_ms: 10\n", 1).replace(
                    "    u_6km_ms: 0\n", "    u_6km_ms: 20\n", 1
                )
                if physics in ("kessler", "active", "column"):
                    other_source = other_source.replace(
                        "microphysics:\n  scheme: none\n",
                        "microphysics:\n  scheme: kessler\n",
                        1,
                    )
                elif physics == "milbrandt":
                    other_source = other_source.replace(
                        "microphysics:\n  scheme: none\n",
                        "microphysics:\n  scheme: milbrandt\n  milbrandt:\n    hail_processes: true\n",
                        1,
                    ).replace(
                        "      dtheta_k: 0.5\n", "      dtheta_k: 2\n", 1
                    )
                if physics == "active":
                    other_source = other_source.replace(
                        "turbulence:\n  scheme: none\n",
                        "turbulence:\n  scheme: smagorinsky\n",
                        1,
                    ).replace("  K_h: 0\n", "  K_h: 5\n", 1).replace(
                        "  K_v: 0\n", "  K_v: 2\n", 1
                    )
                if physics == "column":
                    other_source = other_source.replace(
                        "surface:\n  scheme: none\n  apply_fluxes: false\n",
                        "surface:\n  scheme: slab\n  apply_fluxes: true\n  tsk_k: 300\n",
                        1,
                    ).replace(
                        "boundary_layer:\n  scheme: none\n",
                        "boundary_layer:\n  scheme: slab\n",
                        1,
                    ).replace(
                        "radiation:\n  scheme: none\n",
                        "radiation:\n  scheme: simple_grey\n  solar_forcing: idealized_diurnal\n  latitude_deg: 35\n  declination_deg: 15\n  local_solar_time_at_start_h: 0\n",
                        1,
                    )
                mismatch_config = root / f"{key}-mismatch.yaml"
                mismatch_config.write_text(other_source)
                mismatch = run([
                    str(binary), "--headless", "--config", str(mismatch_config),
                    f"--resume={checkpoint}", "--duration=2", "--write-every=0",
                    "--checkpoint-every=0", f"--outdir={root / f'{key}-mismatch'}",
                    "--threads=1", "--log-profile=normal",
                ])
                if mismatch.returncode == 0 or (
                    "checkpoint time-stepping configuration does not exactly match "
                    "the requested config" not in mismatch.stdout
                ):
                    raise RuntimeError(
                        f"{key} checkpoint accepted the {other_scheme} selection:\n"
                        f"{mismatch.stdout}"
                    )
                continuation = run([
                    str(binary), "--headless", "--config", str(variant),
                    f"--resume={checkpoint}", "--duration=2", "--write-every=1",
                    "--checkpoint-every=0", f"--outdir={resumed}", "--threads=4",
                    "--log-profile=normal",
                ])
                require_success(continuation, f"{key} checkpoint continuation")
                if "[CHECKPOINT] resumed step=20, time_s=1" not in continuation.stdout:
                    raise RuntimeError(f"{key} continuation did not restore midpoint")
                for field in FIELDS:
                    if field_payload(resumed, 1, field) != final_payloads[key][field]:
                        raise RuntimeError(f"{key} {field} restart is not bitwise exact")

        if all(
            final_payloads["rk3-weno5-none"][field] ==
                final_payloads["rk4-weno5-none"][field]
            for field in ("u", "v", "w", "rho", "p", "theta")
        ):
            raise RuntimeError("SSPRK3 and RK4 produced identical evolved core states")
        for scheme in CONFIGS:
            if all(
                final_payloads[f"{scheme}-weno5-none"][field] ==
                    final_payloads[f"{scheme}-tvd-none"][field]
                for field in ("theta", "qv", "tracer")
            ):
                raise RuntimeError(f"{scheme} TVD and WENO5 scalar states are identical")
            for advection in ("tvd", "weno5"):
                if all(
                    final_payloads[f"{scheme}-{advection}-active"][field] ==
                        final_payloads[f"{scheme}-{advection}-kessler"][field]
                    for field in ("u", "v", "w", "theta", "qv")
                ):
                    raise RuntimeError(
                        f"{scheme}/{advection} Smagorinsky+diffusion owners were inert"
                    )
            for advection in ("tvd", "weno5"):
                if all(
                    final_payloads[f"{scheme}-{advection}-column"][field] ==
                        final_payloads[f"{scheme}-{advection}-kessler"][field]
                    for field in ("u", "v", "theta", "qv")
                ):
                    raise RuntimeError(
                        f"{scheme}/{advection} slab-PBL/simple-grey owners were inert"
                    )

        for stack, stack_config in (
            ("lin", LIN_CONFIG),
            ("thompson", THOMPSON_CONFIG),
        ):
            if not stack_config.is_file():
                raise RuntimeError(f"missing Ready {stack} fixture: {stack_config}")
            stack_source = stack_config.read_text()
            stack_payloads: dict[str, dict[str, bytes]] = {}
            for scheme, integrator in (("rk3", "ssprk3"), ("rk4", "rk4")):
                for advection in ("tvd", "weno5"):
                    source = stack_source.replace(
                        "time:\n", f"time:\n  integrator: {integrator}\n", 1
                    ).replace("  run_time: 4\n", "  run_time: 2\n", 1).replace(
                        "  dt_large: 0.2\n", "  dt_large: 0.05\n", 1
                    ).replace("  output_interval: 4\n", "  output_interval: 1\n", 1).replace(
                        "output:\n  preset: minimal\n",
                        "output:\n  preset: dynamics_full\n",
                        1,
                    ).replace("    every_s: 2\n", "    every_s: 1\n", 1).replace(
                        "advection:\n  scheme: tvd\n",
                        f"advection:\n  scheme: {advection}\n",
                        1,
                    )
                    config = root / f"{scheme}-{stack}-{advection}.yaml"
                    config.write_text(source)
                    preflight = run([str(binary), "--validate-only", "--config", str(config)])
                    require_success(preflight, f"{scheme}/{stack}/{advection} qualified preflight")

                    key = f"{scheme}-{stack}-{advection}"
                    serial = root / f"{key}-serial"
                    parallel = root / f"{key}-parallel"
                    resumed = root / f"{key}-resumed"
                    coupled_run(binary, config, serial, 1, key)
                    coupled_run(binary, config, parallel, 4, key)
                    stack_payloads[key] = {}
                    evolved = False
                    for field in FIELDS:
                        initial = field_payload(parallel, 0, field, (12, 12, 48))
                        serial_final = field_payload(serial, 2, field, (12, 12, 48))
                        parallel_final = field_payload(parallel, 2, field, (12, 12, 48))
                        if serial_final != parallel_final:
                            raise RuntimeError(f"{key} {field} differs between one and four workers")
                        evolved = evolved or initial != parallel_final
                        stack_payloads[key][field] = parallel_final
                    if not evolved:
                        raise RuntimeError(f"{key} stack did not evolve")

                    serial_checkpoint = serial / "checkpoints/checkpoint_step_000000020.nimbus"
                    checkpoint = parallel / "checkpoints/checkpoint_step_000000020.nimbus"
                    if not checkpoint.is_file() or not Path(f"{checkpoint}.json").is_file():
                        raise RuntimeError(f"{key} did not publish its midpoint checkpoint")
                    if serial_checkpoint.read_bytes() != checkpoint.read_bytes():
                        raise RuntimeError(f"{key} private/public checkpoint differs across workers")
                    continuation = run([
                        str(binary), "--headless", "--config", str(config),
                        f"--resume={checkpoint}", "--duration=2", "--write-every=1",
                        "--checkpoint-every=0", f"--outdir={resumed}", "--threads=4",
                        "--log-profile=normal",
                    ])
                    require_success(continuation, f"{key} checkpoint continuation")
                    for field in FIELDS:
                        if field_payload(resumed, 1, field, (12, 12, 48)) != stack_payloads[key][field]:
                            raise RuntimeError(f"{key} {field} restart is not bitwise exact")

            if all(
                stack_payloads[f"rk3-{stack}-tvd"][field] ==
                    stack_payloads[f"rk4-{stack}-tvd"][field]
                for field in ("u", "v", "w", "theta", "qv")
            ):
                raise RuntimeError(f"SSPRK3 and RK4 produced identical {stack} states")

        for owner, owner_config in (("tke", TKE_CONFIG), ("mynn", MYNN_CONFIG)):
            if not owner_config.is_file():
                raise RuntimeError(f"missing Ready {owner} fixture: {owner_config}")
            owner_source = owner_config.read_text()
            for scheme, integrator in (("rk3", "ssprk3"), ("rk4", "rk4")):
                for advection in ("tvd", "weno5"):
                    source = owner_source.replace(
                        "time:\n", f"time:\n  integrator: {integrator}\n", 1
                    ).replace(
                        "  dt_large: 0.2\n", "  dt_large: 0.05\n", 1
                    ).replace(
                        "advection:\n  scheme: tvd\n",
                        f"advection:\n  scheme: {advection}\n", 1,
                    ).replace(
                        "output:\n  preset: minimal\n",
                        "output:\n  preset: dynamics_full\n", 1,
                    ).replace(
                        "output:\n  preset: presentation\n",
                        "output:\n  preset: dynamics_full\n", 1,
                    ).replace(
                        "  output_interval: 0\n", "  output_interval: 1\n", 1
                    )
                    if owner == "mynn" and "gpu_apply_microphysics:" not in source:
                        source = source.replace(
                            "    allow_fallback: false\n",
                            "    allow_fallback: false\n"
                            "    gpu_apply_microphysics: false\n"
                            "    gpu_apply_slow: false\n"
                            "    gpu_apply_sgs: false\n",
                            1,
                        )
                    config = root / f"{scheme}-{owner}-{advection}.yaml"
                    config.write_text(source)
                    preflight = run([
                        str(binary), "--validate-only", "--config", str(config)
                    ])
                    require_success(
                        preflight, f"{scheme}/{owner}/{advection} qualified preflight"
                    )

                    key = f"{scheme}-{owner}-{advection}"
                    serial = root / f"{key}-serial"
                    parallel = root / f"{key}-parallel"
                    resumed = root / f"{key}-resumed"
                    coupled_run(binary, config, serial, 1, key)
                    coupled_run(binary, config, parallel, 4, key)

                    evolved = False
                    owner_evolved = False
                    final: dict[str, bytes] = {}
                    for field in FIELDS:
                        initial = field_payload(parallel, 0, field)
                        serial_final = field_payload(serial, 2, field)
                        parallel_final = field_payload(parallel, 2, field)
                        if serial_final != parallel_final:
                            raise RuntimeError(
                                f"{key} {field} differs between one and four workers"
                            )
                        evolved = evolved or initial != parallel_final
                        if field in (("tke",) if owner == "tke" else ("u", "v", "theta", "qv", "tke")):
                            owner_evolved = owner_evolved or initial != parallel_final
                        final[field] = parallel_final
                    if not evolved or not owner_evolved:
                        raise RuntimeError(f"{key} owner stack did not evolve")

                    serial_checkpoint = serial / "checkpoints/checkpoint_step_000000020.nimbus"
                    checkpoint = parallel / "checkpoints/checkpoint_step_000000020.nimbus"
                    if not checkpoint.is_file() or not Path(f"{checkpoint}.json").is_file():
                        raise RuntimeError(f"{key} did not publish its midpoint checkpoint")
                    if serial_checkpoint.read_bytes() != checkpoint.read_bytes():
                        raise RuntimeError(f"{key} checkpoint differs across workers")
                    continuation = run([
                        str(binary), "--headless", "--config", str(config),
                        f"--resume={checkpoint}", "--duration=2", "--write-every=1",
                        "--checkpoint-every=0", f"--outdir={resumed}", "--threads=4",
                        "--log-profile=normal",
                    ])
                    require_success(continuation, f"{key} checkpoint continuation")
                    for field in FIELDS:
                        if field_payload(resumed, 1, field) != final[field]:
                            raise RuntimeError(f"{key} {field} restart is not bitwise exact")

    print(
        "[coupled-rk-ready] PASS: bounded SSPRK3/RK4 CPU family advanced "
        "all 15 public prognostic fields and Thompson/Milbrandt private moments "
        "inside no-advection, TVD, and mixed bound-preserving WENO5 stages; "
        "proved the no-advection scalar hold while dynamics evolved; executed Kessler, "
        "Smagorinsky/explicit-diffusion, slab-PBL/simple-grey, Lin/YSU/RRTMG, "
        "Thompson/YSU/RRTMG, Milbrandt, prognostic TKE/RRTMG, and "
        "MYNN/RRTMG compositions; "
        "evolved distinctly, matched 1/4 "
        "workers, resumed bitwise, and rejected near profiles"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
