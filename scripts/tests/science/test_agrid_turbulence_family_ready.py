#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import tempfile

from npy_support import read_float32_npy


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "tests/fixtures/configs/qualification/agrid_flat_cpu_core_ready.yaml"
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


def field_payload(output: Path, second: int, field: str) -> bytes:
    path = output / f"step_{second:06d}" / f"{field}.npy"
    if not path.is_file():
        raise RuntimeError(f"missing A-grid turbulence field: {path}")
    shape, raw, values = read_float32_npy(path)
    if shape != (18, 18, 18):
        raise RuntimeError(f"{field} has unexpected shape {shape}")
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError(f"{field} contains non-finite values")
    if field in ("rho", "p", "theta") and any(value <= 0.0 for value in values):
        raise RuntimeError(f"{field} violates its positive-state contract")
    if field in ("qv", "tracer", "qc", "qr", "qi", "qs", "qg", "qh", "tke") and any(
        value < 0.0 for value in values
    ):
        raise RuntimeError(f"{field} violates its nonnegative-state contract")
    return raw


def render(source: str, turbulence: str) -> str:
    block = (
        "turbulence:\n"
        "  scheme: none\n"
        "  mode: 3d\n"
        "  Cs: 0.18\n"
        "  Pr_t: 0.7\n"
        "  Sc_t: 0.7\n"
    )
    stability = "tke_based" if turbulence == "tke" else "none"
    replacement = (
        "turbulence:\n"
        f"  scheme: {turbulence}\n"
        "  mode: 3d\n"
        "  filter_width: cubic_root\n"
        f"  stability_correction: {stability}\n"
        "  dynamic_coefficient: false\n"
        "  Cs: 0.18\n"
        "  Pr_t: 0.3333333333333333\n"
        "  Sc_t: 0.3333333333333333\n"
    )
    if source.count(block) != 1:
        raise RuntimeError("A-grid turbulence fixture block drifted")
    return source.replace(block, replacement, 1).replace(
        "output:\n  preset: minimal\n",
        "output:\n  preset: dynamics_full\n",
        1,
    )


def execute(binary: Path, config: Path, output: Path, threads: int, label: str) -> str:
    result = run([
        str(binary), "--headless", "--config", str(config),
        f"--outdir={output}", "--write-every=1", f"--threads={threads}",
        "--log-profile=normal",
    ])
    require_success(result, label)
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file() or not CONFIG.is_file():
        raise RuntimeError("missing A-grid turbulence gate input")

    source = CONFIG.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="nimbus-agrid-turbulence-") as temporary:
        root = Path(temporary)
        final_payloads: dict[str, dict[str, bytes]] = {}
        for turbulence in ("none", "smagorinsky", "tke"):
            config = root / f"agrid-{turbulence}.yaml"
            config.write_text(render(source, turbulence), encoding="utf-8")
            preflight = run([str(binary), "--validate-only", "--config", str(config)])
            require_success(preflight, f"A-grid {turbulence} preflight")

            serial = root / f"{turbulence}-serial"
            parallel = root / f"{turbulence}-parallel"
            resumed = root / f"{turbulence}-resumed"
            serial_log = execute(binary, config, serial, 1, f"A-grid {turbulence} serial")
            parallel_log = execute(binary, config, parallel, 4, f"A-grid {turbulence} parallel")
            if turbulence != "none":
                marker = f"Initialized turbulence scheme: {turbulence}"
                if marker not in serial_log or marker not in parallel_log:
                    raise RuntimeError(f"A-grid {turbulence} owner was not initialized")

            final_payloads[turbulence] = {}
            evolved = False
            for field in FIELDS:
                initial = field_payload(parallel, 0, field)
                serial_final = field_payload(serial, 2, field)
                parallel_final = field_payload(parallel, 2, field)
                if serial_final != parallel_final:
                    raise RuntimeError(
                        f"A-grid {turbulence} {field} differs across workers"
                    )
                evolved = evolved or initial != parallel_final
                final_payloads[turbulence][field] = parallel_final
            if not evolved:
                raise RuntimeError(f"A-grid {turbulence} state did not evolve")

            for ledger in LEDGERS:
                if (serial / ledger).read_bytes() != (parallel / ledger).read_bytes():
                    raise RuntimeError(
                        f"A-grid {turbulence} {ledger} differs across workers"
                    )
            serial_checkpoint = serial / "checkpoints/checkpoint_step_000000010.nimbus"
            checkpoint = parallel / "checkpoints/checkpoint_step_000000010.nimbus"
            if serial_checkpoint.read_bytes() != checkpoint.read_bytes():
                raise RuntimeError(
                    f"A-grid {turbulence} checkpoint differs across workers"
                )
            continuation = run([
                str(binary), "--headless", "--config", str(config),
                f"--resume={checkpoint}", "--duration=2", "--write-every=1",
                "--checkpoint-every=0", f"--outdir={resumed}", "--threads=4",
                "--log-profile=normal",
            ])
            require_success(continuation, f"A-grid {turbulence} continuation")
            for field in FIELDS:
                if field_payload(resumed, 1, field) != final_payloads[turbulence][field]:
                    raise RuntimeError(
                        f"A-grid {turbulence} {field} restart is not bitwise exact"
                    )

        for turbulence in ("smagorinsky", "tke"):
            compared = ("u", "v", "w", "theta", "qv", "tke")
            if all(
                final_payloads[turbulence][field] == final_payloads["none"][field]
                for field in compared
            ):
                raise RuntimeError(f"A-grid {turbulence} owner was inert")
        if final_payloads["tke"]["tke"] == final_payloads["none"]["tke"]:
            raise RuntimeError("A-grid prognostic TKE did not respond")

        dynamic = root / "invalid-dynamic.yaml"
        dynamic.write_text(
            render(source, "smagorinsky").replace(
                "  dynamic_coefficient: false\n",
                "  dynamic_coefficient: true\n", 1,
            ),
            encoding="utf-8",
        )
        rejected = run([str(binary), "--validate-only", "--config", str(dynamic)])
        if rejected.returncode == 0:
            raise RuntimeError("A-grid dynamic Smagorinsky neighbor was admitted")

    print(
        "[agrid-turbulence-family-ready] PASS: none/Smagorinsky/TKE flat CPU "
        "A-grid owners evolved distinctly, matched 1/4-worker fields, ledgers, "
        "and checkpoints, resumed bitwise, and rejected a noncanonical neighbor"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
