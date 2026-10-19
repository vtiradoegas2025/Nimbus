#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
CONFIG = ROOT / "tests/fixtures/configs/qualification/domain_recycling_kessler_ready.yaml"
FIELDS = ("p", "qr", "qv", "rho", "theta", "u", "v", "w")
LEDGERS = (
    "budget.csv",
    "energy_budget.csv",
    "moisture_budget.csv",
    "momentum_budget.csv",
    "radiation_budget.csv",
    "thermodynamic_source_budget.csv",
)
SCOPE_ERRORS = (
    "microphysics.scheme=kessler is qualified only in an exact nested or domain-recycling reference profile",
    "microphysics.scheme=kessler is qualified only in an exact public reference profile",
    "domain_motion.tracking diagnostic evidence is not reference-qualified",
)
MUTATIONS = (
    ("maximum shift", "    maximum_shift_cells: 6\n", "    maximum_shift_cells: 5\n"),
    ("SGS coefficient", "  Cs: 0.18\n", "  Cs: 0.19\n"),
    ("surface roughness", "  z0m_m: 0.1\n", "  z0m_m: 0.11\n"),
    ("fallback", "    allow_fallback: false\n", "    allow_fallback: true\n"),
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


def coupled_run(binary: Path, output: Path, threads: int) -> None:
    result = run([
        str(binary), "--headless", "--config", str(CONFIG),
        "--duration=30", "--write-every=30", f"--outdir={output}",
        f"--threads={threads}", "--log-profile=normal", "--no-timing",
    ])
    require_success(result, f"domain-recycling run with {threads} worker(s)")
    for marker in (
        "Execution mode: qualified",
        "movement mode live_flat_cgrid_recycling",
        "[COMPUTE] backend=cpu",
        "fallback=off",
    ):
        if marker not in result.stdout:
            raise RuntimeError(
                f"domain-recycling run omitted {marker!r}:\n{result.stdout}"
            )


def final_field(output: Path, field: str) -> bytes:
    path = output / "step_000001" / f"{field}.npy"
    if not path.is_file():
        raise RuntimeError(f"domain-recycling run omitted {path}")
    return path.read_bytes()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file() or not CONFIG.is_file():
        raise RuntimeError("domain-recycling Ready binary or fixture is missing")

    preflight = run([str(binary), "--validate-only", "--config", str(CONFIG)])
    require_success(preflight, "domain-recycling qualified preflight")
    if "[VALIDATE] PASS" not in preflight.stdout:
        raise RuntimeError("domain-recycling preflight omitted its PASS marker")

    with tempfile.TemporaryDirectory(prefix="nimbus-domain-recycling-ready-") as temporary:
        root = Path(temporary)
        startup = run([
            str(binary), "--headless", "--config", str(CONFIG),
            "--duration=0", "--write-every=0", f"--outdir={root / 'valid'}",
            "--threads=4", "--log-profile=normal",
        ])
        require_success(startup, "domain-recycling qualified startup")
        for marker in ("Execution mode: qualified", "[COMPUTE] backend=cpu", "fallback=off"):
            if marker not in startup.stdout:
                raise RuntimeError(f"qualified startup omitted {marker!r}")

        source = CONFIG.read_text(encoding="utf-8")
        for index, (label, original, replacement) in enumerate(MUTATIONS):
            mutated = source.replace(original, replacement, 1)
            if mutated == source:
                raise RuntimeError(f"{label} mutation did not alter the fixture")
            config = root / f"invalid-{index}.yaml"
            config.write_text(mutated, encoding="utf-8")
            commands = (
                ("validate-only", [str(binary), "--validate-only", "--config", str(config)]),
                (
                    "startup",
                    [
                        str(binary), "--headless", "--config", str(config),
                        "--duration=0", "--write-every=0",
                        f"--outdir={root / f'invalid-{index}'}",
                    ],
                ),
            )
            for mode, command in commands:
                result = run(command)
                if result.returncode == 0 or not any(
                    marker in result.stdout for marker in SCOPE_ERRORS
                ):
                    raise RuntimeError(
                        f"{label} {mode} did not fail with the exact-profile scope error:\n"
                        f"{result.stdout}"
                    )

        serial = root / "serial"
        parallel = root / "parallel"
        coupled_run(binary, serial, 1)
        coupled_run(binary, parallel, 4)
        for field in FIELDS:
            if final_field(serial, field) != final_field(parallel, field):
                raise RuntimeError(
                    f"domain-recycling {field} differs between one and four workers"
                )
        for ledger in LEDGERS:
            serial_path = serial / ledger
            parallel_path = parallel / ledger
            if not serial_path.is_file() or not parallel_path.is_file():
                raise RuntimeError(f"domain-recycling run omitted {ledger}")
            if serial_path.read_bytes() != parallel_path.read_bytes():
                raise RuntimeError(
                    f"domain-recycling {ledger} differs between one and four workers"
                )

        qualifier = run([
            "python3",
            "scripts/qualification/domain_motion/qualify_domain_recycling.py",
            "--binary", str(binary), "--config", str(CONFIG),
            "--compute-backend", "cpu", "--threads", "4",
            "--output", str(root / "transaction"),
        ])
        require_success(qualifier, "domain-recycling transaction/restart qualifier")
        if "[domain-recycling] PASS" not in qualifier.stdout:
            raise RuntimeError(
                "domain-recycling transaction qualifier omitted its PASS marker"
            )

    print(
        "[domain-recycling-ready] PASS: exact qualified CPU admission, "
        "four fail-closed scope mutations, one/four-worker field and ledger "
        "identity, live D5 recycling, trajectory/swath ownership, and "
        "checkpoint continuation passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
