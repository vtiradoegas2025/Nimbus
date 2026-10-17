#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
import tempfile

from npy_support import read_float32_npy


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
CONTROL = ROOT / "tests/fixtures/configs/qualification/agrid_thompson_rrtmg_implicit_weno_ready.yaml"
FIELDS = (
    "u", "v", "w", "rho", "p", "theta", "qv", "tracer",
    "qc", "qr", "qi", "qs", "qg", "qh", "tke",
)
LEDGERS = (
    "budget.csv", "momentum_budget.csv", "energy_budget.csv",
    "radiation_budget.csv", "moisture_budget.csv",
    "thermodynamic_source_budget.csv",
)


@dataclass(frozen=True)
class Profile:
    name: str
    config: Path
    initialization_marker: str


PROFILES = (
    Profile(
        "ssprk3",
        ROOT / "tests/fixtures/configs/qualification/"
        "agrid_thompson_rrtmg_implicit_weno_ssprk3_ready.yaml",
        "Initialized RK3 (SSPRK3) time stepping scheme",
    ),
    Profile(
        "rk4",
        ROOT / "tests/fixtures/configs/qualification/"
        "agrid_thompson_rrtmg_implicit_weno_rk4_ready.yaml",
        "Initialized RK4 time stepping scheme",
    ),
)


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )


def require_success(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed:\n{result.stdout}")
    for marker in ("[SCIENTIFIC INVALIDITY]", "Science profile rejected"):
        if marker in result.stdout:
            raise RuntimeError(f"{label} reported {marker}:\n{result.stdout}")


def field_payload(output: Path, second: int, field: str) -> bytes:
    path = output / f"step_{second:06d}" / f"{field}.npy"
    shape, raw, values = read_float32_npy(path)
    if shape != (18, 18, 18) or any(not math.isfinite(value) for value in values):
        raise RuntimeError(f"invalid A-grid RK output {path}")
    if field in ("rho", "p", "theta") and any(value <= 0.0 for value in values):
        raise RuntimeError(f"non-positive A-grid RK output {path}")
    if field in ("qv", "tracer", "qc", "qr", "qi", "qs", "qg", "qh", "tke") and any(
        value < 0.0 for value in values
    ):
        raise RuntimeError(f"negative A-grid RK output {path}")
    return raw


def execute(binary: Path, profile: Profile, output: Path, threads: int) -> str:
    result = run([
        str(binary), "--headless", "--config", str(profile.config),
        f"--outdir={output}", "--write-every=1", f"--threads={threads}",
        "--log-profile=normal",
    ])
    require_success(result, f"A-grid {profile.name} run with {threads} worker(s)")
    for marker in (
        "Execution mode: qualified",
        "Dynamics: klemp_wilhelmson_split_cartesian_agrid",
        "Numerics-Advection: weno5",
        "Numerics-Diffusion: implicit",
        "Initialized microphysics scheme: thompson",
        "Initialized radiation scheme: rrtmg",
        profile.initialization_marker,
        "[COMPUTE] backend=cpu",
        "fallback=off",
    ):
        if marker not in result.stdout:
            raise RuntimeError(f"A-grid {profile.name} omitted {marker!r}")
    return result.stdout


def require_complete_unsplit_ledger(output: Path, label: str) -> None:
    path = output / "thermodynamic_source_budget.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"{label} unsplit ledger is empty")
    for index, row in enumerate(rows, start=1):
        for key in (
            "completed_step_energy_step_complete",
            "completed_step_energy_cumulative_complete",
        ):
            if float(row[key]) != 1.0:
                raise RuntimeError(f"{label} row {index} omitted {key}")
        owner = float(row[
            "completed_step_energy_step_unsplit_rk_committed_change_j"
        ])
        if owner == 0.0:
            raise RuntimeError(f"{label} row {index} has an inert unsplit owner")
        committed = float(row["completed_step_energy_step_committed_owner_sum_j"])
        residual = float(row["completed_step_energy_step_closure_residual_j"])
        if abs(residual) / max(1.0, abs(committed)) > 1.0e-12:
            raise RuntimeError(f"{label} row {index} has unclosed unsplit ownership")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    binary = args.binary.resolve()
    for path in (binary, CONTROL, *(profile.config for profile in PROFILES)):
        if not path.is_file():
            raise RuntimeError(f"required A-grid RK input is missing: {path}")

    with tempfile.TemporaryDirectory(prefix="nimbus-agrid-rk-") as temporary:
        root = Path(temporary)
        control_output = root / "split-explicit-control"
        control_result = run([
            str(binary), "--headless", "--config", str(CONTROL),
            f"--outdir={control_output}", "--write-every=1", "--threads=1",
            "--log-profile=quiet",
        ])
        require_success(control_result, "A-grid split-explicit control")
        control_fields = {
            field: field_payload(control_output, 2, field) for field in FIELDS
        }

        finals: dict[str, dict[str, bytes]] = {}
        for profile in PROFILES:
            preflight = run([
                str(binary), "--validate-only", "--config", str(profile.config)
            ])
            require_success(preflight, f"A-grid {profile.name} preflight")
            if "[VALIDATE] PASS" not in preflight.stdout:
                raise RuntimeError(f"A-grid {profile.name} preflight omitted PASS")

            source = profile.config.read_text(encoding="utf-8")
            for label, mutated in (
                ("native", source.replace("    backend: cpu\n", "    backend: metal\n", 1)),
                ("terrain", source.replace("terrain:\n  scheme: none\n", "terrain:\n  scheme: bell\n", 1)),
            ):
                if mutated == source:
                    raise RuntimeError(f"A-grid {profile.name} {label} mutation drifted")
                invalid = root / f"invalid-{profile.name}-{label}.yaml"
                invalid.write_text(mutated, encoding="utf-8")
                rejected = run([str(binary), "--validate-only", "--config", str(invalid)])
                if rejected.returncode == 0:
                    raise RuntimeError(f"A-grid {profile.name} {label} neighbor was admitted")

            serial = root / f"{profile.name}-serial"
            parallel = root / f"{profile.name}-parallel"
            resumed = root / f"{profile.name}-resumed"
            execute(binary, profile, serial, 1)
            execute(binary, profile, parallel, 4)
            require_complete_unsplit_ledger(serial, f"A-grid {profile.name} serial")
            require_complete_unsplit_ledger(parallel, f"A-grid {profile.name} parallel")
            finals[profile.name] = {}
            changed_from_split = 0
            for field in FIELDS:
                serial_final = field_payload(serial, 2, field)
                parallel_final = field_payload(parallel, 2, field)
                if serial_final != parallel_final:
                    raise RuntimeError(
                        f"A-grid {profile.name} {field} differs across workers"
                    )
                finals[profile.name][field] = parallel_final
                changed_from_split += parallel_final != control_fields[field]
            if changed_from_split < 3:
                raise RuntimeError(
                    f"A-grid {profile.name} did not differ materially from split-explicit"
                )
            for ledger in LEDGERS:
                if (serial / ledger).read_bytes() != (parallel / ledger).read_bytes():
                    raise RuntimeError(
                        f"A-grid {profile.name} {ledger} differs across workers"
                    )
            serial_checkpoint = serial / "checkpoints/checkpoint_step_000000010.nimbus"
            checkpoint = parallel / "checkpoints/checkpoint_step_000000010.nimbus"
            if serial_checkpoint.read_bytes() != checkpoint.read_bytes():
                raise RuntimeError(
                    f"A-grid {profile.name} checkpoint differs across workers"
                )
            continuation = run([
                str(binary), "--headless", "--config", str(profile.config),
                f"--resume={checkpoint}", "--duration=2", "--write-every=1",
                "--checkpoint-every=0", f"--outdir={resumed}", "--threads=4",
                "--log-profile=normal",
            ])
            require_success(continuation, f"A-grid {profile.name} continuation")
            for field in FIELDS:
                if field_payload(resumed, 1, field) != finals[profile.name][field]:
                    raise RuntimeError(
                        f"A-grid {profile.name} {field} restart mismatch"
                    )
            for ledger in LEDGERS:
                continuous = (parallel / ledger).read_text().splitlines()
                suffix = (resumed / ledger).read_text().splitlines()[1:]
                if suffix != continuous[-len(suffix):]:
                    raise RuntimeError(
                        f"A-grid {profile.name} {ledger} restart suffix mismatch"
                    )

        if all(
            finals["ssprk3"][field] == finals["rk4"][field]
            for field in FIELDS
        ):
            raise RuntimeError("A-grid SSPRK3 and RK4 endpoints are identical")

    print(
        "[agrid-rk-ready] PASS: SSPRK3 and RK4 advanced the maximal "
        "collocated WENO/Thompson/RRTMG/implicit stack distinctly, matched "
        "1/4-worker fields, checkpoints and ledgers, resumed exactly, and "
        "rejected native/terrain neighbors"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
