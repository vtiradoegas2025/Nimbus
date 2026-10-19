#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "tests/fixtures/configs/qualification/cgrid_lin_rrtmg_cloud_top_ysu_ready.yaml"
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
STATE_FIELDS = ("p", "rho", "theta", "u", "v", "w", "qv", "qc", "qr")
LEDGERS = (
    "budget.csv",
    "energy_budget.csv",
    "moisture_budget.csv",
    "momentum_budget.csv",
    "radiation_budget.csv",
    "thermodynamic_source_budget.csv",
)


def run(command: list[str], threads: int = 1) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["OMP_DYNAMIC"] = "FALSE"
    return subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
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


def final_frame(output: Path) -> Path:
    frames = sorted(path for path in output.glob("step_*") if path.is_dir())
    if not frames:
        raise RuntimeError(f"{output}: no published frames")
    return frames[-1]


def require_state_identity(reference: Path, candidate: Path, label: str) -> None:
    lhs_frame = final_frame(reference)
    rhs_frame = final_frame(candidate)
    for field in STATE_FIELDS:
        if (lhs_frame / f"{field}.npy").read_bytes() != (rhs_frame / f"{field}.npy").read_bytes():
            raise RuntimeError(f"{label}: final {field} bytes differ")


def require_ledger_identity(reference: Path, candidate: Path, label: str) -> None:
    for filename in LEDGERS:
        if (reference / filename).read_bytes() != (candidate / filename).read_bytes():
            raise RuntimeError(f"{label}: {filename} bytes differ")


def require_restart_suffix(reference: Path, resumed: Path) -> None:
    for filename in LEDGERS:
        lhs = (reference / filename).read_text(encoding="utf-8").splitlines()
        rhs = (resumed / filename).read_text(encoding="utf-8").splitlines()
        if not lhs or len(rhs) < 2 or lhs[0] != rhs[0]:
            raise RuntimeError(f"restart: malformed {filename}")
        if lhs[-(len(rhs) - 1):] != rhs[1:]:
            raise RuntimeError(f"restart: {filename} is not an exact suffix")


def require_cloud_transport(output: Path) -> None:
    with (output / "thermodynamic_source_budget.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("cloud-top profile published no thermodynamic ledger")
    for column in (
        "pbl_cloud_liquid_tendency_kg",
        "pbl_cloud_ice_tendency_kg",
    ):
        if max(abs(float(row[column])) for row in rows) <= 0.0:
            raise RuntimeError(f"cloud-top profile did not exercise {column}")
    if any(int(row["microphysics_conservation_active"]) != 1 for row in rows):
        raise RuntimeError("cloud-top profile omitted water conservation admission")
    for row in rows:
        residual = abs(float(row["microphysics_conservation_residual_kg"]))
        baseline = max(abs(float(row["microphysics_baseline_water_kg"])), 1.0)
        if residual / baseline > 1.0e-9:
            raise RuntimeError("cloud-top profile exceeded water closure tolerance")


def run_profile(binary: Path, output: Path, threads: int) -> str:
    result = run([
        str(binary), "--headless", "--config", str(CONFIG),
        f"--outdir={output}", "--write-every=4", "--checkpoint-every=2",
        f"--threads={threads}", "--log-profile=normal",
    ], threads)
    require_success(result, f"cloud-top YSU {threads}-thread evolution")
    for marker in (
        "Execution mode: qualified",
        "Initialized boundary layer scheme: ysu",
        "Cloud-top mixing: surface-coupled Lock-2000 radiative YSU",
        "Initialized radiation scheme: rrtmg",
        "[COMPUTE] backend=cpu",
        "fallback=off",
    ):
        if marker not in result.stdout:
            raise RuntimeError(
                f"cloud-top YSU {threads}-thread evolution omitted {marker!r}")
    return result.stdout


def require_cloud_response(binary: Path, source: str, cloudy: Path, root: Path) -> None:
    anchor = "boundary_layer:\n  scheme: ysu\n  cadence_s: 0.2\n  clear_sky_only: false\n"
    if anchor not in source:
        raise RuntimeError("clear-sky control anchor is missing")
    control = source.replace(anchor, anchor.replace("false", "true"), 1)
    control = control.replace("    mode: qualified\n", "    mode: exploratory\n", 1)
    control = control.replace(
        "    mode: qualified\n",
        "    mode: qualified\n    acknowledge_maturity_limits: false\n",
        1,
    )
    path = root / "clear-sky-control.yaml"
    path.write_text(control, encoding="utf-8")
    output = root / "clear-sky-control"
    result = run([
        str(binary), "--headless", "--config", str(path),
        f"--outdir={output}", "--write-every=4", "--checkpoint-every=0",
        "--threads=1", "--log-profile=quiet",
    ])
    require_success(result, "clear-sky YSU control")
    cloudy_frame = final_frame(cloudy)
    clear_frame = final_frame(output)
    if all(
        (cloudy_frame / f"{field}.npy").read_bytes()
        == (clear_frame / f"{field}.npy").read_bytes()
        for field in ("theta", "qv", "qc", "u", "v")
    ):
        raise RuntimeError("cloud-top closure produced no resolved state response")


def require_scope_rejection(
    binary: Path,
    source: str,
    old: str,
    new: str,
    root: Path,
    label: str,
) -> None:
    if old not in source:
        raise RuntimeError(f"{label}: mutation anchor is missing")
    path = root / f"{label}.yaml"
    path.write_text(source.replace(old, new, 1), encoding="utf-8")
    for mode, arguments in (
        ("validation", ["--validate-only"]),
        ("startup", ["--headless", "--duration=0", "--write-every=0",
                     f"--outdir={root / (label + '-output')}"]),
    ):
        result = run([str(binary), *arguments, "--config", str(path)])
        if result.returncode == 0:
            raise RuntimeError(f"{label} {mode} unexpectedly succeeded")
    if (root / f"{label}-output/nimbus_output.json").exists():
        raise RuntimeError(f"{label}: rejected startup published output")


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
        ROOT / "tests/fixtures/configs/qualification/soundings/ysu_surface_coupled_cloud_profile.txt",
    ):
        if not path.is_file():
            raise RuntimeError(f"required cloud-top YSU input is missing: {path}")

    source = CONFIG.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="nimbus-cloud-top-ysu-ready-") as temporary:
        root = Path(temporary)
        validated = run([str(binary), "--validate-only", "--config", str(CONFIG)])
        require_success(validated, "cloud-top YSU qualified preflight")
        if "[VALIDATE] PASS" not in validated.stdout:
            raise RuntimeError("cloud-top YSU preflight did not publish PASS")

        one_thread = root / "one-thread"
        four_threads = root / "four-threads"
        run_profile(binary, one_thread, 1)
        run_profile(binary, four_threads, 4)
        require_state_identity(one_thread, four_threads, "thread determinism")
        require_ledger_identity(one_thread, four_threads, "thread determinism")
        require_cloud_transport(four_threads)
        require_cloud_response(binary, source, one_thread, root)

        checkpoint = four_threads / "checkpoints/checkpoint_step_000000010.nimbus"
        if not checkpoint.is_file() or not Path(f"{checkpoint}.json").is_file():
            raise RuntimeError("cloud-top YSU run did not publish its midpoint checkpoint")
        resumed = root / "resumed"
        result = run([
            str(binary), "--headless", "--config", str(CONFIG),
            f"--resume={checkpoint}", "--duration=4", "--write-every=4",
            "--checkpoint-every=0", f"--outdir={resumed}",
            "--threads=4", "--log-profile=normal",
        ], 4)
        require_success(result, "cloud-top YSU checkpoint continuation")
        if "[CHECKPOINT] resumed step=10, time_s=2" not in result.stdout:
            raise RuntimeError("cloud-top YSU continuation did not restore the midpoint")
        require_state_identity(four_threads, resumed, "restart continuation")
        require_restart_suffix(four_threads, resumed)

        mutations = (
            ("  cadence_s: 0.2\n", "  cadence_s: 0.4\n", "noncanonical-pbl-cadence"),
            ("  clear_sky_only: false\n", "  clear_sky_only: true\n", "disabled-cloud-mixing"),
            ("  tsk_k: 285\n", "  tsk_k: 284\n", "noncanonical-surface"),
            ("  scheme: ysu\n", "  scheme: slab\n", "wrong-pbl-owner"),
            ("    allow_fallback: false\n", "    allow_fallback: true\n", "fallback-enabled"),
            ("  preset: science_validation\n", "  preset: minimal\n", "wrong-output-contract"),
        )
        for old, new, label in mutations:
            require_scope_rejection(binary, source, old, new, root, label)

    print(
        "[cloud-top-radiative-ysu-ready] PASS: exact CPU profile admitted; "
        "Lock-2000 response, theta_l/qv/qc/qi transport, 1/4-thread identity, "
        "restart, closure, provenance, and negative gates passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
