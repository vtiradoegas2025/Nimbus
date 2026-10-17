#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "tests/fixtures/configs/qualification/cgrid_lin_rrtmg_ready.yaml"
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
STATE_FIELDS = (
    "p", "rho", "theta", "u", "v", "w", "qv", "qr",
)
LEDGERS = (
    "budget.csv",
    "energy_budget.csv",
    "moisture_budget.csv",
    "momentum_budget.csv",
    "radiation_budget.csv",
    "thermodynamic_source_budget.csv",
)


def run(
    command: list[str], threads: int = 1
) -> subprocess.CompletedProcess[str]:
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
    reference_frame = final_frame(reference)
    candidate_frame = final_frame(candidate)
    for field in STATE_FIELDS:
        lhs = reference_frame / f"{field}.npy"
        rhs = candidate_frame / f"{field}.npy"
        if lhs.read_bytes() != rhs.read_bytes():
            raise RuntimeError(f"{label}: final {field} bytes differ")


def require_ledger_identity(reference: Path, candidate: Path, label: str) -> None:
    for filename in LEDGERS:
        if (reference / filename).read_bytes() != (candidate / filename).read_bytes():
            raise RuntimeError(f"{label}: {filename} bytes differ")


def require_restart_suffix(continuous: Path, resumed: Path) -> None:
    for filename in LEDGERS:
        reference = (continuous / filename).read_text(encoding="utf-8").splitlines()
        candidate = (resumed / filename).read_text(encoding="utf-8").splitlines()
        if not reference or len(candidate) < 2 or reference[0] != candidate[0]:
            raise RuntimeError(f"restart: malformed {filename}")
        if reference[-(len(candidate) - 1):] != candidate[1:]:
            raise RuntimeError(f"restart: {filename} is not an exact suffix")


def require_process_activity(output: Path) -> None:
    with (output / "thermodynamic_source_budget.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("Lin bounded run published no source ledger")
    cloud_activity = max(
        abs(float(row[f"microphysics_{name}_requested_kg"]))
        for row in rows for name in ("qv", "qc")
    )
    if cloud_activity <= 0.0:
        raise RuntimeError("Lin bounded run did not form cloud condensate")
    if any(int(row["microphysics_conservation_active"]) != 1 for row in rows):
        raise RuntimeError("Lin bounded run omitted conservation admission")
    relative_residuals = (
        abs(float(row["microphysics_conservation_residual_kg"]))
        / max(abs(float(row["microphysics_baseline_water_kg"])), 1.0e-30)
        for row in rows
    )
    if any(residual > 1.0e-6 for residual in relative_residuals):
        raise RuntimeError("Lin bounded run exceeded its water closure bound")


def require_cloud_optical_response(binary: Path, source: str, cloudy: Path,
                                   root: Path) -> None:
    clear_config = root / "clear-sky-control.yaml"
    clear_source = source.replace(
        "    mode: qualified\n", "    mode: exploratory\n", 1
    ).replace(
        "    mode: qualified\n",
        "    mode: qualified\n    acknowledge_maturity_limits: false\n", 1
    ).replace("  clear_sky_only: false\n", "  clear_sky_only: true\n", 1)
    if clear_source == source:
        raise RuntimeError("clear-sky control mutation did not change the profile")
    clear_config.write_text(clear_source, encoding="utf-8")
    clear_output = root / "clear-sky-control"
    result = run([
        str(binary), "--headless", "--config", str(clear_config),
        f"--outdir={clear_output}", "--write-every=4",
        "--checkpoint-every=0", "--threads=1", "--log-profile=quiet",
    ])
    require_success(result, "Lin clear-sky optical control")
    with (cloudy / "radiation_budget.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        cloudy_rows = list(csv.DictReader(stream))
    with (clear_output / "radiation_budget.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        clear_rows = list(csv.DictReader(stream))
    if len(cloudy_rows) != len(clear_rows) or not cloudy_rows:
        raise RuntimeError("cloudy and clear-sky radiation ledgers do not align")
    maximum_response = max(
        abs(float(cloud["held_atmospheric_power_w"]) -
            float(clear["held_atmospheric_power_w"]))
        for cloud, clear in zip(cloudy_rows, clear_rows)
    )
    if maximum_response <= 1.0:
        raise RuntimeError("cloudy RRTMG produced no resolved optical response")


def run_profile(binary: Path, output: Path, threads: int) -> str:
    result = run([
        str(binary), "--headless", "--config", str(CONFIG),
        f"--outdir={output}", "--write-every=4", "--checkpoint-every=2",
        f"--threads={threads}", "--log-profile=normal",
    ], threads)
    require_success(result, f"Lin {threads}-thread evolution")
    for marker in (
        "Execution mode: qualified",
        "Initialized microphysics scheme: lin",
        "Initialized radiation scheme: rrtmg",
        "Initialized boundary layer scheme: ysu",
        "[COMPUTE] backend=cpu",
        "fallback=off",
    ):
        if marker not in result.stdout:
            raise RuntimeError(
                f"Lin {threads}-thread evolution omitted {marker!r}")
    return result.stdout


def require_scope_rejection(binary: Path, source: str, old: str, new: str,
                            root: Path, label: str) -> None:
    if old not in source:
        raise RuntimeError(f"{label}: mutation anchor is missing")
    config = root / f"{label}.yaml"
    config.write_text(source.replace(old, new, 1), encoding="utf-8")
    for mode, arguments in (
        ("validation", ["--validate-only"]),
        ("startup", ["--headless", "--duration=0", "--write-every=0",
                     f"--outdir={root / (label + '-output')}"]),
    ):
        result = run([str(binary), *arguments, "--config", str(config)])
        if result.returncode == 0:
            raise RuntimeError(f"{label} {mode} unexpectedly succeeded")
        if "microphysics.scheme=lin is qualified only" not in result.stdout:
            raise RuntimeError(
                f"{label} {mode} omitted exact-scope rejection:\n{result.stdout}")
    if (root / f"{label}-output/nimbus_output.json").exists():
        raise RuntimeError(f"{label}: rejected startup published output")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    binary = args.binary.resolve()
    for path in (
        binary, CONFIG,
        ROOT / "data/rrtmg/rrtmg_lw.nc",
        ROOT / "data/rrtmg/rrtmg_sw.nc",
        ROOT / "tests/fixtures/configs/qualification/soundings/lin_cloudy_profile.txt",
    ):
        if not path.is_file():
            raise RuntimeError(f"required Lin ready-state input is missing: {path}")

    source = CONFIG.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="nimbus-lin-rrtmg-ready-") as temporary:
        root = Path(temporary)
        validated = run([str(binary), "--validate-only", "--config", str(CONFIG)])
        require_success(validated, "Lin qualified preflight")
        if "[VALIDATE] PASS" not in validated.stdout:
            raise RuntimeError("Lin qualified preflight did not publish PASS")

        one_thread = root / "one-thread"
        four_threads = root / "four-threads"
        run_profile(binary, one_thread, 1)
        run_profile(binary, four_threads, 4)
        require_state_identity(one_thread, four_threads, "thread determinism")
        require_ledger_identity(one_thread, four_threads, "thread determinism")
        require_process_activity(four_threads)
        require_cloud_optical_response(binary, source, one_thread, root)

        checkpoint = four_threads / "checkpoints/checkpoint_step_000000010.nimbus"
        if not checkpoint.is_file() or not Path(f"{checkpoint}.json").is_file():
            raise RuntimeError("Lin bounded run did not publish the split checkpoint")
        resumed = root / "resumed"
        result = run([
            str(binary), "--headless", "--config", str(CONFIG),
            f"--resume={checkpoint}", "--duration=4", "--write-every=4",
            "--checkpoint-every=0", f"--outdir={resumed}",
            "--threads=4", "--log-profile=normal",
        ], 4)
        require_success(result, "Lin checkpoint continuation")
        if "[CHECKPOINT] resumed step=10, time_s=2" not in result.stdout:
            raise RuntimeError("Lin continuation did not restore the split state")
        require_state_identity(four_threads, resumed, "restart continuation")
        require_restart_suffix(four_threads, resumed)

        require_scope_rejection(
            binary, source,
            "    accretion_coefficient: 2.2\n",
            "    accretion_coefficient: 2.1\n",
            root, "noncanonical-lin-parameter")
        require_scope_rejection(
            binary, source,
            "  scheme: ysu\n",
            "  scheme: slab\n",
            root, "noncanonical-boundary-layer")
        require_scope_rejection(
            binary, source,
            "    allow_fallback: false\n",
            "    allow_fallback: true\n",
            root, "fallback-enabled")

    print(
        "[lin-rrtmg-ready] PASS: exact CPU profile admitted; literal processes, "
        "1/4-thread evolution, conservation, restart, and scope gates passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
