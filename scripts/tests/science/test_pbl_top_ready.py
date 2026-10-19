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
DEFAULT_BINARY = ROOT / "bin" / "nimbus_sim"
SCOPE_ERROR = (
    "qualified non-Richardson PBL-top diagnosis is limited to the exact "
    "YSU/theta and MYNN/TKE CPU reference profiles"
)
FIELDS = ("p", "qv", "rho", "theta", "u", "v", "w")
LEDGERS = (
    "budget.csv",
    "energy_budget.csv",
    "moisture_budget.csv",
    "momentum_budget.csv",
    "radiation_budget.csv",
    "thermodynamic_source_budget.csv",
)


@dataclass(frozen=True)
class Profile:
    name: str
    scheme: str
    method: str
    config: Path
    mutations: tuple[tuple[str, str], ...]


PROFILES = (
    Profile(
        "ysu-theta",
        "ysu",
        "theta",
        ROOT / "tests/fixtures/configs/qualification/"
        "cgrid_ysu_theta_top_ready.yaml",
        (
            ("  nx: 12\n", "  nx: 13\n"),
            ("  preset: presentation\n", "  preset: minimal\n"),
            ("    allow_fallback: false\n", "    allow_fallback: true\n"),
            ("  scheme: ysu\n", "  scheme: slab\n"),
        ),
    ),
    Profile(
        "mynn-tke",
        "mynn",
        "tke",
        ROOT / "tests/fixtures/configs/qualification/"
        "cgrid_mynn_tke_top_ready.yaml",
        (
            ("  nx: 12\n", "  nx: 13\n"),
            ("  preset: presentation\n", "  preset: minimal\n"),
            ("    allow_fallback: false\n", "    allow_fallback: true\n"),
            ("  pbl_top_method: tke\n", "  pbl_top_method: theta\n"),
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


def require_scope_rejection(
    result: subprocess.CompletedProcess[str], label: str
) -> None:
    if result.returncode == 0 or SCOPE_ERROR not in result.stdout:
        raise RuntimeError(
            f"{label} did not fail with the exact-profile guard:\n"
            f"{result.stdout}"
        )


def reject_near_profiles(binary: Path, profile: Profile, root: Path) -> None:
    source = profile.config.read_text(encoding="utf-8")
    for index, (old, new) in enumerate(profile.mutations):
        mutated = source.replace(old, new, 1)
        if mutated == source:
            raise RuntimeError(
                f"{profile.name} mutation {index} did not alter the fixture"
            )
        config = root / f"{profile.name}-invalid-{index}.yaml"
        config.write_text(mutated, encoding="utf-8")
        require_scope_rejection(
            run([str(binary), "--validate-only", "--config", str(config)]),
            f"{profile.name} validate-only neighbor {index}",
        )
        require_scope_rejection(
            run([
                str(binary), "--headless", "--config", str(config),
                "--duration=0", "--write-every=0",
                f"--outdir={root / f'invalid-{profile.name}-{index}'}",
            ]),
            f"{profile.name} runtime neighbor {index}",
        )


def coupled_run(
    binary: Path, profile: Profile, output: Path, threads: int
) -> str:
    result = run([
        str(binary), "--headless", "--config", str(profile.config),
        "--write-every=1", f"--outdir={output}", f"--threads={threads}",
        "--log-profile=normal", "--no-timing",
    ])
    require_success(result, f"{profile.name} run with {threads} worker(s)")
    for marker in (
        "Execution mode: qualified",
        f"Boundary Layer: {profile.scheme}",
        f"pbl_top: {profile.method}",
        f"Initialized boundary layer scheme: {profile.scheme}",
        "[COMPUTE] backend=cpu",
        "fallback=off",
    ):
        if marker not in result.stdout:
            raise RuntimeError(f"{profile.name} run omitted {marker!r}")
    return result.stdout


def check_pbl_height(profile: Profile, output: Path, step: int):
    shape, raw, values = field_payload(output, step, "pbl_height")
    if shape != (12, 12):
        raise RuntimeError(f"{profile.name} PBL-height shape is {shape}")
    interior = [values[j * 12 + i] for j in range(1, 11) for i in range(1, 11)]
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError(f"{profile.name} PBL-height output is non-finite")
    if any(value < 100.0 or value > 3000.0 for value in interior):
        raise RuntimeError(
            f"{profile.name} interior PBL height left its configured bounds"
        )
    halo = [
        values[j * 12 + i]
        for j in range(12)
        for i in range(12)
        if i in (0, 11) or j in (0, 11)
    ]
    if any(value != 0.0 for value in halo):
        raise RuntimeError(f"{profile.name} PBL-height halo was modified")
    return shape, raw


def check_outputs(profile: Profile, serial: Path, parallel: Path) -> None:
    evolved = 0
    for field in FIELDS:
        initial_shape, initial_raw, _ = field_payload(parallel, 0, field)
        final_shape, parallel_raw, values = field_payload(parallel, 2, field)
        serial_shape, serial_raw, _ = field_payload(serial, 2, field)
        if initial_shape != (12, 12, 16) or final_shape != initial_shape:
            raise RuntimeError(f"{profile.name} {field} shape disagrees with the grid")
        if serial_shape != final_shape or serial_raw != parallel_raw:
            raise RuntimeError(
                f"{profile.name} {field} differs between one and four workers"
            )
        evolved += initial_raw != parallel_raw
        if any(not math.isfinite(value) for value in values):
            raise RuntimeError(f"{profile.name} {field} output is non-finite")
        if field in ("p", "rho") and any(value <= 0.0 for value in values):
            raise RuntimeError(f"{profile.name} {field} output is non-positive")
        if field == "qv" and any(value < 0.0 for value in values):
            raise RuntimeError(f"{profile.name} qv output is negative")
    if evolved < 3:
        raise RuntimeError(f"{profile.name} did not evolve at least three fields")

    serial_shape, serial_raw = check_pbl_height(profile, serial, 2)
    parallel_shape, parallel_raw = check_pbl_height(profile, parallel, 2)
    if serial_shape != parallel_shape or serial_raw != parallel_raw:
        raise RuntimeError(
            f"{profile.name} PBL height differs between one and four workers"
        )

    for ledger in LEDGERS:
        serial_path = serial / ledger
        parallel_path = parallel / ledger
        if not serial_path.is_file() or not parallel_path.is_file():
            raise RuntimeError(f"{profile.name} run omitted {ledger}")
        if serial_path.read_bytes() != parallel_path.read_bytes():
            raise RuntimeError(
                f"{profile.name} {ledger} differs between one and four workers"
            )

    for step in (0, 1, 2):
        if (parallel / f"step_{step:06d}/composite_reflectivity.npy").exists():
            raise RuntimeError(
                f"{profile.name} exported a radar-dependent field with radar disabled"
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
        "--log-profile=normal", "--no-timing",
    ])
    require_success(result, f"{profile.name} checkpoint continuation")
    if "[CHECKPOINT] resumed step=5, time_s=1" not in result.stdout:
        raise RuntimeError(f"{profile.name} continuation did not restore step 5")
    for field in (*FIELDS, "pbl_height"):
        final_shape, final_raw, _ = field_payload(continuous, 2, field)
        resumed_shape, resumed_raw, _ = field_payload(resumed, 1, field)
        if resumed_shape != final_shape or resumed_raw != final_raw:
            raise RuntimeError(
                f"{profile.name} {field} continuation is not bitwise exact"
            )


def check_checkpoint_identity(
    binary: Path, profile: Profile, other: Profile, continuous: Path, root: Path
) -> None:
    checkpoint = continuous / "checkpoints/checkpoint_step_000000005.nimbus"
    result = run([
        str(binary), "--headless", "--config", str(other.config),
        f"--resume={checkpoint}", "--duration=2", "--write-every=0",
        f"--outdir={root / f'{profile.name}-wrong-method'}", "--threads=1",
    ])
    output = result.stdout.lower()
    if result.returncode == 0 or (
        "fingerprint" not in output
        and "checkpoint scientific configuration" not in output
        and "checkpoint boundary-layer configuration" not in output
    ):
        raise RuntimeError(
            f"{profile.name} checkpoint accepted {other.name}:\n{result.stdout}"
        )


def check_profile(
    binary: Path, profile: Profile, other: Profile, root: Path
) -> None:
    validated = run([str(binary), "--validate-only", "--config", str(profile.config)])
    require_success(validated, f"{profile.name} qualified preflight")
    if "[VALIDATE] PASS" not in validated.stdout:
        raise RuntimeError(f"{profile.name} qualified preflight omitted PASS")

    reject_near_profiles(binary, profile, root)
    serial = root / f"{profile.name}-serial"
    parallel = root / f"{profile.name}-parallel"
    resumed = root / f"{profile.name}-resumed"
    coupled_run(binary, profile, serial, 1)
    coupled_run(binary, profile, parallel, 4)
    check_outputs(profile, serial, parallel)
    check_continuation(binary, profile, parallel, resumed)
    check_checkpoint_identity(binary, profile, other, parallel, root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    binary = args.binary.resolve()
    for path in (binary, *(profile.config for profile in PROFILES)):
        if not path.is_file():
            raise RuntimeError(f"required Ready-state input is missing: {path}")

    with tempfile.TemporaryDirectory(prefix="nimbus-pbl-top-ready-") as temporary:
        root = Path(temporary)
        check_profile(binary, PROFILES[0], PROFILES[1], root)
        check_profile(binary, PROFILES[1], PROFILES[0], root)

    print(
        "[pbl-top-ready] PASS: exact public YSU/theta and MYNN/TKE CPU "
        "profiles, threshold routing, fail-closed neighbors, radar-free "
        "presentation output, 1/4-worker identity, and restart identity passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
