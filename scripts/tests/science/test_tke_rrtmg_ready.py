#!/usr/bin/env python3

from __future__ import annotations

import argparse
import array
import math
from pathlib import Path
import subprocess
import tempfile

from npy_support import read_float32_npy


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "tests/fixtures/configs/qualification/cgrid_tke_rrtmg_ready.yaml"
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
FIELDS = ("u", "v", "w", "rho", "p", "theta", "qv", "tke")


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
    if result.returncode == 0:
        raise RuntimeError(f"{label} unexpectedly succeeded:\n{result.stdout}")
    if not any(marker in result.stdout for marker in (
        "turbulence.scheme=tke requires the maintained flat periodic C-grid reference with its declared explicit or implicit diffusion ownership",
        "turbulence.scheme=tke is outside the qualified flat, RK, A-grid, and mapped CPU families",
    )):
        raise RuntimeError(f"{label} omitted the TKE scope error:\n{result.stdout}")


def field_payload(
    output: Path, step: int, field: str
) -> tuple[tuple[int, ...], bytes, array.array[float]]:
    path = output / f"step_{step:06d}" / f"{field}.npy"
    if not path.is_file():
        raise RuntimeError(f"missing output field: {path}")
    return read_float32_npy(path)


def coupled_run(binary: Path, output: Path, threads: int) -> subprocess.CompletedProcess[str]:
    result = run([
        str(binary), "--headless", "--config", str(CONFIG),
        f"--outdir={output}", "--write-every=1", f"--threads={threads}",
        "--log-profile=normal",
    ])
    require_success(result, f"TKE coupled run with {threads} worker(s)")
    for marker in (
        "Execution mode: qualified",
        "Initialized radiation scheme: rrtmg",
        "Initialized Deardorff-Moeng prognostic TKE closure",
        "Stability correction = tke_based",
        "[COMPUTE] backend=cpu",
        "fallback=off",
    ):
        if marker not in result.stdout:
            raise RuntimeError(f"TKE coupled run omitted {marker!r}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    binary = args.binary.resolve()
    for path in (
        binary,
        CONFIG,
        ROOT / "data/rrtmg/rrtmg_lw.nc",
        ROOT / "data/rrtmg/rrtmg_sw.nc",
    ):
        if not path.is_file():
            raise RuntimeError(f"required ready-state input is missing: {path}")

    preflight = run([str(binary), "--validate-only", "--config", str(CONFIG)])
    require_success(preflight, "TKE qualified preflight")
    if "[VALIDATE] PASS" not in preflight.stdout:
        raise RuntimeError("TKE qualified preflight did not publish PASS")

    with tempfile.TemporaryDirectory(prefix="nimbus-tke-rrtmg-ready-") as temporary:
        root = Path(temporary)
        invalid_config = root / "invalid-tke.yaml"
        source = CONFIG.read_text()
        invalid_source = source.replace(
            "  Pr_t: 0.3333333333333333\n", "  Pr_t: 0.7\n", 1
        )
        if invalid_source == source:
            raise RuntimeError("TKE scope mutation did not alter the fixture")
        invalid_config.write_text(invalid_source)
        require_scope_rejection(
            run([str(binary), "--validate-only", "--config", str(invalid_config)]),
            "TKE validate-only scope guard",
        )
        require_scope_rejection(
            run([
                str(binary), "--headless", "--config", str(invalid_config),
                "--duration=0", "--write-every=0",
                f"--outdir={root / 'invalid-runtime'}",
            ]),
            "TKE runtime scope guard",
        )
        invalid_selector = root / "invalid-tke-selector.yaml"
        invalid_selector.write_text(source.replace(
            "  stability_correction: tke_based\n",
            "  stability_correction: none\n", 1))
        require_scope_rejection(
            run([str(binary), "--validate-only", "--config", str(invalid_selector)]),
            "TKE stability-selector scope guard",
        )
        serial = root / "serial"
        parallel = root / "parallel"
        resumed = root / "resumed"
        coupled_run(binary, serial, 1)
        coupled_run(binary, parallel, 4)

        initial_shape, initial_raw, initial_values = field_payload(
            parallel, 0, "tke"
        )
        final_shape, parallel_raw, final_values = field_payload(
            parallel, 2, "tke"
        )
        if initial_shape != (12, 12, 16) or final_shape != initial_shape:
            raise RuntimeError("TKE output shape does not match the configured grid")
        if initial_raw == parallel_raw:
            raise RuntimeError("coupled TKE state did not evolve from its initial reservoir")
        if any(not math.isfinite(value) or value < 0.0 for value in final_values):
            raise RuntimeError("coupled TKE output is non-finite or negative")
        if any(abs(value - 0.1) > 2.0e-7 for value in initial_values):
            raise RuntimeError("cold-start TKE reservoir does not match initialization truth")
        serial_final: dict[str, bytes] = {}
        for field in FIELDS:
            serial_shape, serial_raw, serial_values = field_payload(
                serial, 2, field
            )
            parallel_shape, parallel_field_raw, parallel_values = field_payload(
                parallel, 2, field
            )
            if serial_shape != final_shape or parallel_shape != final_shape:
                raise RuntimeError(f"{field} output shape does not match the grid")
            if serial_raw != parallel_field_raw:
                raise RuntimeError(
                    f"{field} differs between one and four workers"
                )
            if any(not math.isfinite(value) for value in serial_values) or any(
                not math.isfinite(value) for value in parallel_values
            ):
                raise RuntimeError(f"{field} contains non-finite values")
            serial_final[field] = serial_raw

        checkpoint = parallel / "checkpoints/checkpoint_step_000000005.nimbus"
        if not checkpoint.is_file() or not Path(f"{checkpoint}.json").is_file():
            raise RuntimeError("TKE coupled run did not publish its restart pair")
        continuation = run([
            str(binary), "--headless", "--config", str(CONFIG),
            f"--resume={checkpoint}", "--duration=2", "--write-every=1",
            "--checkpoint-every=0", f"--outdir={resumed}", "--threads=4",
            "--log-profile=normal",
        ])
        require_success(continuation, "TKE checkpoint continuation")
        if "[CHECKPOINT] resumed step=5, time_s=1" not in continuation.stdout:
            raise RuntimeError("TKE checkpoint continuation did not restore step 5")
        for field in FIELDS:
            resumed_shape, resumed_raw, _ = field_payload(resumed, 1, field)
            if resumed_shape != final_shape or resumed_raw != serial_final[field]:
                raise RuntimeError(
                    f"{field} checkpoint continuation is not bitwise exact"
                )

        neutral_config = root / "neutral-control.yaml"
        neutral_config.write_text(
            source.replace("    mode: qualified\n", "    mode: exploratory\n", 1)
                  .replace("    mode: qualified\n",
                           "    mode: qualified\n    acknowledge_maturity_limits: false\n", 1)
                  .replace("  stability_correction: tke_based\n",
                           "  stability_correction: none\n", 1)
        )
        neutral_output = root / "neutral-control"
        neutral = run([
            str(binary), "--headless", "--config", str(neutral_config),
            f"--outdir={neutral_output}", "--write-every=1", "--threads=1",
            "--log-profile=normal",
        ])
        require_success(neutral, "neutral TKE exploratory control")
        _, neutral_raw, _ = field_payload(neutral_output, 2, "tke")
        if neutral_raw == serial_final["tke"]:
            raise RuntimeError("tke_based stability correction had no TKE response")

    print(
        "[tke-rrtmg-ready] PASS: qualified CPU profile applied the explicit "
        "Deardorff stability correction, differed from neutral mixing, matched "
        "1/4-worker execution bitwise, checkpointed, and resumed bitwise"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
