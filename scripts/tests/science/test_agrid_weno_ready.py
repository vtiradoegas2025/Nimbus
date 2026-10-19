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
CONFIG = ROOT / "tests/fixtures/configs/qualification/agrid_thompson_rrtmg_implicit_weno_ready.yaml"
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
FIELDS = (
    "u", "v", "w", "rho", "p", "theta", "qv", "tracer",
    "qc", "qr", "qi", "qs", "qg", "qh", "tke",
)
LEDGERS = (
    "budget.csv", "momentum_budget.csv", "energy_budget.csv",
    "radiation_budget.csv", "moisture_budget.csv",
    "thermodynamic_source_budget.csv",
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


def replace_once(source: str, old: str, new: str, label: str) -> str:
    rendered = source.replace(old, new, 1)
    if rendered == source:
        raise RuntimeError(f"A-grid WENO {label} mutation drifted")
    return rendered


def field_payload(output: Path, second: int, field: str) -> bytes:
    path = output / f"step_{second:06d}" / f"{field}.npy"
    if not path.is_file():
        raise RuntimeError(f"missing A-grid WENO output {path}")
    shape, raw, values = read_float32_npy(path)
    if shape != (18, 18, 18) or any(not math.isfinite(value) for value in values):
        raise RuntimeError(f"invalid A-grid WENO field {path}")
    if field in ("rho", "p", "theta") and any(value <= 0.0 for value in values):
        raise RuntimeError(f"non-positive A-grid WENO field {path}")
    if field in ("qv", "tracer", "qc", "qr", "qi", "qs", "qg", "qh", "tke") and any(
        value < 0.0 for value in values
    ):
        raise RuntimeError(f"negative A-grid WENO field {path}")
    return raw


def execute(binary: Path, config: Path, output: Path, threads: int) -> str:
    result = run([
        str(binary), "--headless", "--config", str(config),
        f"--outdir={output}", "--write-every=1", f"--threads={threads}",
        "--log-profile=normal",
    ])
    require_success(result, f"A-grid WENO run with {threads} worker(s)")
    for marker in (
        "Execution mode: qualified",
        "Dynamics: klemp_wilhelmson_split_cartesian_agrid",
        "Initialized microphysics scheme: thompson",
        "Initialized radiation scheme: rrtmg",
        "Initialized WENO5 advection scheme",
        "Numerics-Advection: weno5",
        "Initialized conservative backward-Euler diffusion scheme",
        "Numerics-Diffusion: implicit",
        "[COMPUTE] backend=cpu",
        "fallback=off",
    ):
        if marker not in result.stdout:
            raise RuntimeError(f"A-grid WENO run omitted {marker!r}")
    return result.stdout


def require_nonzero_radiation(output: Path) -> None:
    with (output / "radiation_budget.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("A-grid WENO radiation ledger is empty")
    candidates = (
        "net_radiative_energy_j", "atmospheric_radiation_energy_j",
        "cumulative_radiation_energy_j", "radiation_energy_j",
    )
    if not any(
        key in row and math.isfinite(float(row[key])) and abs(float(row[key])) > 0.0
        for row in rows for key in candidates
    ):
        numeric_nonzero = any(
            math.isfinite(float(value)) and abs(float(value)) > 0.0
            for row in rows for key, value in row.items()
            if key not in ("step", "time_s") and value not in (None, "")
        )
        if not numeric_nonzero:
            raise RuntimeError("A-grid WENO RRTMG owner remained inert")


def require_nonzero_diffusion(output: Path) -> None:
    with (output / "thermodynamic_source_budget.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("A-grid WENO source ledger is empty")
    final = rows[-1]
    momentum = float(final["diffusion_cumulative_committed_momentum_work_j"])
    energy = float(final[
        "diffusion_moist_energy_cumulative_committed_energy_change_j"])
    if not all(math.isfinite(value) for value in (momentum, energy)):
        raise RuntimeError("A-grid WENO work is non-finite")
    if abs(momentum) <= 1.0 or abs(energy) <= 1.0:
        raise RuntimeError("A-grid WENO scalar/momentum owners remained inert")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    binary = args.binary.resolve()
    for path in (binary, CONFIG):
        if not path.is_file():
            raise RuntimeError(f"required A-grid WENO input is missing: {path}")

    preflight = run([str(binary), "--validate-only", "--config", str(CONFIG)])
    require_success(preflight, "A-grid WENO qualified preflight")
    if "[VALIDATE] PASS" not in preflight.stdout:
        raise RuntimeError("A-grid WENO preflight omitted PASS")

    source = CONFIG.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="nimbus-agrid-weno-") as temporary:
        root = Path(temporary)
        control_config = root / "agrid-tvd-control.yaml"
        control_config.write_text(replace_once(
            source,
            "advection:\n  scheme: weno5\n",
            "advection:\n  scheme: tvd\n",
            "TVD control",
        ), encoding="utf-8")
        for label, mutated in (
            (
                "native backend",
                replace_once(source, "    backend: cpu\n", "    backend: metal\n", "backend"),
            ),
            (
                "terrain",
                replace_once(
                    source,
                    "terrain:\n  scheme: none\n",
                    "terrain:\n  scheme: bell\n",
                    "terrain",
                ),
            ),
        ):
            invalid = root / f"invalid-{label.replace(' ', '-')}.yaml"
            invalid.write_text(mutated, encoding="utf-8")
            rejected = run([str(binary), "--validate-only", "--config", str(invalid)])
            if rejected.returncode == 0:
                raise RuntimeError(f"A-grid WENO {label} neighbor was admitted")

        serial = root / "serial"
        parallel = root / "parallel"
        control = root / "control"
        resumed = root / "resumed"
        execute(binary, CONFIG, serial, 1)
        execute(binary, CONFIG, parallel, 4)
        control_result = run([
            str(binary), "--headless", "--config", str(control_config),
            f"--outdir={control}", "--write-every=1", "--threads=1",
            "--log-profile=quiet",
        ])
        require_success(control_result, "A-grid TVD control")

        changed = set()
        for field in FIELDS:
            initial = field_payload(parallel, 0, field)
            serial_final = field_payload(serial, 2, field)
            parallel_final = field_payload(parallel, 2, field)
            control_final = field_payload(control, 2, field)
            if serial_final != parallel_final:
                raise RuntimeError(f"A-grid WENO {field} differs across workers")
            if initial != parallel_final:
                changed.add(field)
            if control_final != parallel_final:
                changed.add(f"control:{field}")
        if "control:theta" not in changed or not any(
            f"control:{field}" in changed
            for field in ("qv", "qc", "qr", "qi", "qs", "qg")
        ):
            raise RuntimeError("A-grid WENO did not differ from its TVD control")

        for ledger in LEDGERS:
            if (serial / ledger).read_bytes() != (parallel / ledger).read_bytes():
                raise RuntimeError(f"A-grid WENO {ledger} differs across workers")
        serial_checkpoint = serial / "checkpoints/checkpoint_step_000000010.nimbus"
        checkpoint = parallel / "checkpoints/checkpoint_step_000000010.nimbus"
        if serial_checkpoint.read_bytes() != checkpoint.read_bytes():
            raise RuntimeError("A-grid WENO checkpoint differs across workers")

        continuation = run([
            str(binary), "--headless", "--config", str(CONFIG),
            f"--resume={checkpoint}", "--duration=2", "--write-every=1",
            "--checkpoint-every=0", f"--outdir={resumed}", "--threads=4",
            "--log-profile=normal",
        ])
        require_success(continuation, "A-grid WENO continuation")
        if "[CHECKPOINT] resumed step=10, time_s=1" not in continuation.stdout:
            raise RuntimeError("A-grid WENO continuation restored the wrong boundary")
        for field in FIELDS:
            if field_payload(resumed, 1, field) != field_payload(parallel, 2, field):
                raise RuntimeError(f"A-grid WENO {field} restart mismatch")
        for ledger in LEDGERS:
            parallel_rows = (parallel / ledger).read_text().splitlines()
            resumed_rows = (resumed / ledger).read_text().splitlines()
            if resumed_rows[1:] != parallel_rows[-len(resumed_rows[1:]):]:
                raise RuntimeError(f"A-grid WENO {ledger} restart suffix mismatch")
        require_nonzero_radiation(parallel)
        require_nonzero_diffusion(parallel)

    print(
        "[agrid-weno-ready] PASS: collocated WENO5 Thompson/RRTMG/implicit "
        "differed from TVD, matched 1/4-worker fields, "
        "checkpoints and ledgers, resumed exactly, and rejected native/terrain neighbors"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
