#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
import sys
import tempfile

from npy_support import read_float32_npy


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.qualification.checkpointing.qualify_vulkan_restart import (  # noqa: E402
    read_csv_rows,
    require_report_owner_identities,
    require_restart_suffix,
)

DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
@dataclass(frozen=True)
class Profile:
    name: str
    config: Path
    terrain_marker: str
    admitted_mutations: tuple[tuple[str, str], ...]


PROFILES = (
    Profile(
        "stationary-bell",
        ROOT / "tests/fixtures/configs/qualification/"
        "cgrid_stationary_bell_pbl_implicit_ready.yaml",
        "Initialized Bell terrain scheme",
        (
            ("  K_v: 4000\n", "  K_v: 3000\n"),
            ("  K_h: 4000\n", "  K_h: 3000\n"),
            ("  tsk_k: 302\n", "  tsk_k: 301\n"),
            ("    h0: 300\n", "    h0: 301\n"),
            ("    u_1km_ms: 2\n", "    u_1km_ms: 3\n"),
        ),
    ),
    Profile(
        "translating-schar",
        ROOT / "tests/fixtures/configs/qualification/"
        "cgrid_translating_schar_pbl_implicit_ready.yaml",
        "Initialized Schär terrain scheme",
        (
            ("  K_v: 4000\n", "  K_v: 3000\n"),
            ("  K_h: 4000\n", "  K_h: 3000\n"),
            ("  tsk_k: 302\n", "  tsk_k: 301\n"),
            ("    H: 300\n", "    H: 301\n"),
            ("    u_ms: 5\n", "    u_ms: 4\n"),
        ),
    ),
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


def field_payload(output: Path, step: int, field: str):
    path = output / f"step_{step:06d}" / f"{field}.npy"
    if not path.is_file():
        raise RuntimeError(f"missing {field} output: {path}")
    return read_float32_npy(path)


def check_family_boundaries(binary: Path, profile: Profile, root: Path) -> None:
    source = profile.config.read_text(encoding="utf-8")
    for index, (old, new) in enumerate(profile.admitted_mutations):
        mutated = source.replace(old, new, 1)
        if mutated == source:
            raise RuntimeError(f"{profile.name} mutation {index} did not alter the fixture")
        config = root / f"{profile.name}-admitted-{index}.yaml"
        config.write_text(mutated, encoding="utf-8")
        result = run([str(binary), "--validate-only", "--config", str(config)])
        require_success(result, f"{profile.name} admitted mutation {index}")
        result = run([
            str(binary), "--headless", "--config", str(config),
            "--duration=1", "--write-every=0", "--checkpoint-every=0",
            f"--outdir={root / f'admitted-{profile.name}-{index}'}",
            "--threads=1", "--log-profile=quiet",
        ])
        require_success(result, f"{profile.name} admitted runtime {index}")

    rejected_mutations = (
        ("microphysics:\n  scheme: none\n", "microphysics:\n  scheme: kessler\n"),
        ("turbulence:\n  scheme: none\n", "turbulence:\n  scheme: tke\n"),
        ("  K_v: 4000\n", "  K_v: 4000\n  coefficient_source: linear_height\n  K_v_surface: 2000\n  K_v_top: 6000\n"),
    )
    for index, (old, new) in enumerate(rejected_mutations):
        mutated = source.replace(old, new, 1)
        if mutated == source:
            raise RuntimeError(f"{profile.name} rejected mutation {index} drifted")
        config = root / f"{profile.name}-rejected-{index}.yaml"
        config.write_text(mutated, encoding="utf-8")
        result = run([str(binary), "--validate-only", "--config", str(config)])
        if result.returncode == 0:
            raise RuntimeError(
                f"{profile.name} unsupported mutation {index} was admitted"
            )


def coupled_run(binary: Path, profile: Profile, output: Path, threads: int) -> None:
    result = run([
        str(binary), "--headless", "--config", str(profile.config),
        f"--outdir={output}", "--write-every=1", f"--threads={threads}",
        "--log-profile=normal",
    ])
    require_success(result, f"{profile.name} run with {threads} worker(s)")
    for marker in (
        "Execution mode: qualified",
        profile.terrain_marker,
        "Initialized boundary layer scheme: slab",
        "Initialized conservative backward-Euler diffusion scheme",
        "Numerics-Diffusion: implicit",
        "[DIFFUSION] Runtime path: src/numerics/diffusion/implicit",
        "[COMPUTE] backend=cpu",
        "fallback=off",
    ):
        if marker not in result.stdout:
            raise RuntimeError(f"{profile.name} run omitted {marker!r}")


def check_outputs(profile: Profile, serial: Path, parallel: Path) -> None:
    evolved = 0
    for field in ("u", "v", "w", "theta", "qv"):
        initial_shape, initial_raw, _ = field_payload(parallel, 0, field)
        final_shape, parallel_raw, values = field_payload(parallel, 2, field)
        serial_shape, serial_raw, _ = field_payload(serial, 2, field)
        if initial_shape != (12, 12, 18) or final_shape != initial_shape:
            raise RuntimeError(f"{profile.name} {field} shape disagrees with the grid")
        if serial_shape != final_shape or serial_raw != parallel_raw:
            raise RuntimeError(
                f"{profile.name} {field} differs between one and four workers"
            )
        evolved += initial_raw != parallel_raw
        if any(not math.isfinite(value) for value in values):
            raise RuntimeError(f"{profile.name} {field} output is non-finite")
        if field == "qv" and any(value < 0.0 for value in values):
            raise RuntimeError(f"{profile.name} qv output is negative")
    if evolved < 3:
        raise RuntimeError(f"{profile.name} did not evolve at least three core fields")


def check_nonlinear_boundary_ledgers(
    profile: Profile, serial: Path, parallel: Path
) -> None:
    serial_checks = require_report_owner_identities(serial)
    parallel_checks = require_report_owner_identities(parallel)
    if serial_checks == 0 or serial_checks != parallel_checks:
        raise RuntimeError(
            f"{profile.name} nonlinear owner-identity coverage is incomplete"
        )

    ledger = "thermodynamic_source_budget.csv"
    if (serial / ledger).read_bytes() != (parallel / ledger).read_bytes():
        raise RuntimeError(
            f"{profile.name} nonlinear boundary ledger differs between workers"
        )

    rows = read_csv_rows(parallel / ledger)
    required_nonzero = (
        "acoustic_v1_boundary_radiative_committed_work_j",
        "acoustic_v1_boundary_committed_work_j",
        "advection_theta_energy_step_outward_boundary_j",
    )
    for column in required_nonzero:
        values = [float(row[column]) for row in rows]
        if not any(math.isfinite(value) and abs(value) > 0.0 for value in values):
            raise RuntimeError(
                f"{profile.name} nonlinear boundary owner remained vacuous: {column}"
            )


def check_continuation(
    binary: Path, profile: Profile, continuous: Path, resumed: Path
) -> None:
    checkpoint = continuous / "checkpoints/checkpoint_step_000000005.nimbus"
    if not checkpoint.is_file() or not Path(f"{checkpoint}.json").is_file():
        raise RuntimeError(f"{profile.name} did not publish its restart pair")
    result = run([
        str(binary), "--headless", "--config", str(profile.config),
        f"--resume={checkpoint}", "--duration=2", "--write-every=1",
        "--checkpoint-every=0", f"--outdir={resumed}", "--threads=4",
        "--log-profile=normal",
    ])
    require_success(result, f"{profile.name} checkpoint continuation")
    if "[CHECKPOINT] resumed step=5, time_s=1" not in result.stdout:
        raise RuntimeError(f"{profile.name} continuation did not restore step 5")
    for field in ("u", "v", "w", "theta", "qv"):
        final_shape, final_raw, _ = field_payload(continuous, 2, field)
        resumed_shape, resumed_raw, _ = field_payload(resumed, 1, field)
        if resumed_shape != final_shape or resumed_raw != final_raw:
            raise RuntimeError(f"{profile.name} {field} continuation is not bitwise exact")


def check_profile(binary: Path, profile: Profile, root: Path) -> None:
    validated = run([str(binary), "--validate-only", "--config", str(profile.config)])
    require_success(validated, f"{profile.name} qualified preflight")
    if "[VALIDATE] PASS" not in validated.stdout:
        raise RuntimeError(f"{profile.name} qualified preflight omitted PASS")

    check_family_boundaries(binary, profile, root)
    serial = root / f"{profile.name}-serial"
    parallel = root / f"{profile.name}-parallel"
    resumed = root / f"{profile.name}-resumed"
    coupled_run(binary, profile, serial, 1)
    coupled_run(binary, profile, parallel, 4)
    check_outputs(profile, serial, parallel)
    check_continuation(binary, profile, parallel, resumed)
    check_nonlinear_boundary_ledgers(profile, serial, parallel)
    require_report_owner_identities(resumed)
    require_restart_suffix(parallel, resumed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    binary = args.binary.resolve()
    for path in (binary, *(profile.config for profile in PROFILES)):
        if not path.is_file():
            raise RuntimeError(f"required ready-state input is missing: {path}")

    with tempfile.TemporaryDirectory(prefix="nimbus-mapped-terrain-ready-") as temporary:
        root = Path(temporary)
        for profile in PROFILES:
            check_profile(binary, profile, root)

    print(
        "[mapped-terrain-ready] PASS: structural stationary/constant-velocity "
        "Bell/Schar slab-PBL/implicit CPU family admitted parameter variants; both "
        "evolved, matched 1/4-worker execution, independently recomposed "
        "nonlinear boundary ledgers, and resumed bitwise"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
