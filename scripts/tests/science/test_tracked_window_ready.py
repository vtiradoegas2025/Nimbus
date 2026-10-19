#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "bin" / "nimbus_sim"
CONFIG = (
    ROOT / "tests" / "fixtures" / "configs" / "qualification"
    / "tracked_window_kessler_ready.yaml"
)
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
    "domain_motion.mode=tracked_window is not reference-qualified",
    "domain_motion.tracking diagnostic evidence is not reference-qualified",
    "microphysics.scheme=kessler is qualified only in an exact nested or domain-recycling reference profile",
    "microphysics.scheme=kessler is qualified only in an exact public reference profile",
    "tracked_window requires tracking.enabled=true",
    "tracked_window uses discrete recycling and requires zero continuous frame velocity",
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
        str(binary),
        "--headless",
        "--config",
        str(CONFIG),
        "--duration=30",
        "--write-every=30",
        f"--outdir={output}",
        f"--threads={threads}",
        "--log-profile=normal",
        "--no-timing",
    ])
    require_success(result, f"tracked-window run with {threads} worker(s)")
    for marker in (
        "Execution mode: qualified",
        "movement mode live_flat_cgrid_recycling",
        "[COMPUTE] backend=cpu",
        "fallback=off",
    ):
        if marker not in result.stdout:
            raise RuntimeError(
                f"tracked-window run omitted {marker!r}:\n{result.stdout}"
            )


def final_field(output: Path, field: str) -> bytes:
    path = output / "step_000001" / f"{field}.npy"
    if not path.is_file():
        raise RuntimeError(f"tracked-window run omitted {path}")
    return path.read_bytes()


def require_scope_rejection(
    result: subprocess.CompletedProcess[str], label: str
) -> None:
    if result.returncode == 0 or not any(
        marker in result.stdout for marker in SCOPE_ERRORS
    ):
        raise RuntimeError(
            f"{label} did not fail with the exact-profile scope guard:\n"
            f"{result.stdout}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file() or not CONFIG.is_file():
        raise RuntimeError("tracked-window Ready binary or fixture is missing")

    preflight = run([str(binary), "--validate-only", "--config", str(CONFIG)])
    require_success(preflight, "tracked-window qualified preflight")
    if "[VALIDATE] PASS" not in preflight.stdout:
        raise RuntimeError("tracked-window preflight omitted its PASS marker")

    with tempfile.TemporaryDirectory(prefix="nimbus-tracked-window-ready-") as temporary:
        root = Path(temporary)
        source = CONFIG.read_text(encoding="utf-8")
        mutations = (
            ("tracking", "    enabled: true\n", "    enabled: false\n"),
            ("shift limit", "    maximum_shift_cells: 6\n", "    maximum_shift_cells: 5\n"),
            ("frame velocity", "    u_ms: 0\n", "    u_ms: 1\n"),
            ("fallback", "    allow_fallback: false\n", "    allow_fallback: true\n"),
        )
        for name, old, new in mutations:
            changed = source.replace(old, new, 1)
            if changed == source:
                raise RuntimeError(f"tracked-window {name} mutation was ineffective")
            invalid = root / f"invalid-{name.replace(' ', '-')}.yaml"
            invalid.write_text(changed, encoding="utf-8")
            require_scope_rejection(
                run([str(binary), "--validate-only", "--config", str(invalid)]),
                f"tracked-window {name} validate-only neighbor",
            )
            require_scope_rejection(
                run([
                    str(binary),
                    "--headless",
                    "--config",
                    str(invalid),
                    "--duration=0",
                    f"--outdir={root / ('invalid-' + name.replace(' ', '-'))}",
                ]),
                f"tracked-window {name} runtime neighbor",
            )

        serial = root / "serial"
        parallel = root / "parallel"
        coupled_run(binary, serial, 1)
        coupled_run(binary, parallel, 4)
        for field in FIELDS:
            if final_field(serial, field) != final_field(parallel, field):
                raise RuntimeError(
                    f"tracked-window {field} differs between one and four workers"
                )
        for ledger in LEDGERS:
            serial_path = serial / ledger
            parallel_path = parallel / ledger
            if not serial_path.is_file() or not parallel_path.is_file():
                raise RuntimeError(f"tracked-window run omitted {ledger}")
            if serial_path.read_bytes() != parallel_path.read_bytes():
                raise RuntimeError(
                    f"tracked-window {ledger} differs between one and four workers"
                )

        qualifier = run([
            "python3",
            "scripts/qualification/domain_motion/qualify_domain_recycling.py",
            "--binary",
            str(binary),
            "--config",
            str(CONFIG),
            "--compute-backend",
            "cpu",
            "--threads",
            "4",
        ])
        require_success(qualifier, "tracked-window transaction/restart qualifier")
        if "[domain-recycling] PASS" not in qualifier.stdout:
            raise RuntimeError(
                "tracked-window transaction qualifier omitted its PASS marker"
            )

    print(
        "[tracked-window-ready] PASS: exact public admission, fail-closed "
        "neighbors, 1/4-worker identity, live recycling, trajectory continuity, "
        "and restart passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
