#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
import tempfile

from npy_support import read_float32_npy


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"


@dataclass(frozen=True)
class Profile:
    name: str
    config: Path
    scope_error: str
    mutation_from: str
    mutation_to: str
    markers: tuple[str, ...]


PROFILES = {
    "weno5": Profile(
        "weno5",
        ROOT / "tests/fixtures/configs/qualification/cgrid_weno5_rrtmg_ready.yaml",
        "advection.scheme=weno5 is qualified only for the exact flat periodic C-grid CPU RRTMG reference profile",
        "  lateral_sponge_enabled: false\n",
        "  lateral_sponge_enabled: true\n",
        ("Initialized WENO5 advection scheme", "Numerics-Advection: weno5"),
    ),
    "implicit": Profile(
        "implicit",
        ROOT / "tests/fixtures/configs/qualification/cgrid_implicit_rrtmg_ready.yaml",
        "linear_height is implemented only by explicit diffusion",
        "  K_v: 20\n",
        "  K_v: 20\n  coefficient_source: linear_height\n  K_v_surface: 10\n  K_v_top: 30\n",
        (
            "Initialized conservative backward-Euler diffusion scheme",
            "Numerics-Diffusion: implicit",
        ),
    ),
}


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


def coupled_run(
    binary: Path, profile: Profile, output: Path, threads: int
) -> None:
    result = run([
        str(binary), "--headless", "--config", str(profile.config),
        f"--outdir={output}", "--write-every=1", f"--threads={threads}",
        "--log-profile=normal",
    ])
    require_success(result, f"{profile.name} coupled run with {threads} worker(s)")
    for marker in (
        "Execution mode: qualified",
        "Initialized radiation scheme: rrtmg",
        "[COMPUTE] backend=cpu",
        "fallback=off",
        *profile.markers,
    ):
        if marker not in result.stdout:
            raise RuntimeError(f"{profile.name} coupled run omitted {marker!r}")


def check_profile(binary: Path, profile: Profile) -> None:
    preflight = run([
        str(binary), "--validate-only", "--config", str(profile.config)
    ])
    require_success(preflight, f"{profile.name} qualified preflight")
    if "[VALIDATE] PASS" not in preflight.stdout:
        raise RuntimeError(f"{profile.name} qualified preflight did not publish PASS")

    with tempfile.TemporaryDirectory(
        prefix=f"nimbus-{profile.name}-rrtmg-ready-"
    ) as temporary:
        root = Path(temporary)
        source = profile.config.read_text()
        invalid_source = source.replace(
            profile.mutation_from, profile.mutation_to, 1
        )
        if invalid_source == source:
            raise RuntimeError(f"{profile.name} scope mutation did not alter the fixture")
        invalid_config = root / f"invalid-{profile.name}.yaml"
        invalid_config.write_text(invalid_source)
        for label, command in (
            (
                "validate-only",
                [str(binary), "--validate-only", "--config", str(invalid_config)],
            ),
            (
                "runtime",
                [
                    str(binary), "--headless", "--config", str(invalid_config),
                    "--duration=0", "--write-every=0",
                    f"--outdir={root / 'invalid-runtime'}",
                ],
            ),
        ):
            rejected = run(command)
            if rejected.returncode == 0 or profile.scope_error not in rejected.stdout:
                raise RuntimeError(
                    f"{profile.name} {label} did not fail with its scope error:\n"
                    f"{rejected.stdout}"
                )

        serial = root / "serial"
        parallel = root / "parallel"
        resumed = root / "resumed"
        coupled_run(binary, profile, serial, 1)
        coupled_run(binary, profile, parallel, 4)

        for field in ("theta", "qv"):
            initial_shape, initial_raw, _ = field_payload(parallel, 0, field)
            final_shape, parallel_raw, values = field_payload(parallel, 2, field)
            serial_shape, serial_raw, _ = field_payload(serial, 2, field)
            if initial_shape != (12, 12, 16) or final_shape != initial_shape:
                raise RuntimeError(f"{profile.name} {field} shape disagrees with the grid")
            if serial_shape != final_shape or serial_raw != parallel_raw:
                raise RuntimeError(
                    f"{profile.name} {field} differs between one and four workers"
                )
            if initial_raw == parallel_raw:
                raise RuntimeError(f"{profile.name} {field} did not evolve")
            if any(not math.isfinite(value) for value in values):
                raise RuntimeError(f"{profile.name} {field} output is non-finite")
            if field == "qv" and any(value < 0.0 for value in values):
                raise RuntimeError(f"{profile.name} qv output is negative")

        checkpoint = parallel / "checkpoints/checkpoint_step_000000005.nimbus"
        if not checkpoint.is_file() or not Path(f"{checkpoint}.json").is_file():
            raise RuntimeError(f"{profile.name} did not publish its restart pair")
        continuation = run([
            str(binary), "--headless", "--config", str(profile.config),
            f"--resume={checkpoint}", "--duration=2", "--write-every=1",
            "--checkpoint-every=0", f"--outdir={resumed}", "--threads=4",
            "--log-profile=normal",
        ])
        require_success(continuation, f"{profile.name} checkpoint continuation")
        if "[CHECKPOINT] resumed step=5, time_s=1" not in continuation.stdout:
            raise RuntimeError(f"{profile.name} continuation did not restore step 5")
        for field in ("theta", "qv"):
            final_shape, final_raw, _ = field_payload(parallel, 2, field)
            resumed_shape, resumed_raw, _ = field_payload(resumed, 1, field)
            if resumed_shape != final_shape or resumed_raw != final_raw:
                raise RuntimeError(
                    f"{profile.name} {field} continuation is not bitwise exact"
                )

    print(
        f"[{profile.name}-rrtmg-ready] PASS: qualified CPU profile evolved "
        "theta/qv, matched 1/4-worker execution bitwise, checkpointed, and "
        "resumed bitwise"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument(
        "--profile", choices=("weno5", "implicit", "all"), default="all"
    )
    args = parser.parse_args()
    binary = args.binary.resolve()
    required = (
        binary,
        ROOT / "data/rrtmg/rrtmg_lw.nc",
        ROOT / "data/rrtmg/rrtmg_sw.nc",
    )
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"required ready-state input is missing: {path}")
    selected = PROFILES.values() if args.profile == "all" else (PROFILES[args.profile],)
    for profile in selected:
        if not profile.config.is_file():
            raise RuntimeError(f"missing ready-state config: {profile.config}")
        check_profile(binary, profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
