#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.qualification.checkpointing.qualify_vulkan_restart import (  # noqa: E402
    read_csv_rows,
    require_report_owner_identities,
)
from scripts.qualification.domain_motion.qualify_domain_recycling import (  # noqa: E402
    require_selected_backend,
    runtime_environment,
)

DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
QUALIFIER = ROOT / "scripts/qualification/domain_motion/qualify_domain_recycling.py"
CONFIGS = {
    "vulkan": ROOT / "tests/fixtures/configs/qualification/domain_recycling_kessler_native_chaos_vulkan_ready.yaml",
    "metal": ROOT / "tests/fixtures/configs/qualification/domain_recycling_kessler_native_chaos_metal_ready.yaml",
}
FIELDS = ("p", "qr", "qv", "rho", "theta", "u", "v", "w")


def run(
    command: list[str], threads: int = 1, backend: str | None = None
) -> subprocess.CompletedProcess[str]:
    environment = (
        runtime_environment(backend) if backend is not None else os.environ.copy()
    )
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["OMP_DYNAMIC"] = "FALSE"
    return subprocess.run(
        command, cwd=ROOT, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )


def require_success(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed:\n{result.stdout}")
    for marker in ("[SCIENTIFIC INVALIDITY]", "Science profile rejected"):
        if marker in result.stdout:
            raise RuntimeError(f"{label} reported {marker}:\n{result.stdout}")


def require_rejection(
    binary: Path,
    backend: str,
    source: str,
    old: str,
    new: str,
    root: Path,
    label: str,
) -> None:
    changed = source.replace(old, new, 1)
    if changed == source:
        raise RuntimeError(f"{backend} {label}: mutation anchor is missing")
    config = root / f"invalid-{backend}-{label}.yaml"
    config.write_text(changed, encoding="utf-8")
    for mode, arguments in (
        ("validate", ["--validate-only"]),
        ("startup", ["--headless", "--duration=0", "--write-every=0",
                     f"--outdir={root / ('invalid-' + backend + '-' + label)}"]),
    ):
        result = run([str(binary), *arguments, "--config", str(config)])
        rejection_markers = (
            "[VALIDATE]", "[SCIENTIFIC INVALIDITY]", "Science profile rejected",
            "Configuration validation failed", "Configuration load failed",
            "not reference-qualified",
            "extensions.chaos.native_generation_required is admitted only",
        )
        if result.returncode == 0 or not any(
                marker in result.stdout for marker in rejection_markers):
            raise RuntimeError(
                f"{backend} {label} {mode} did not fail closed:\n{result.stdout}"
            )


def run_qualification(
    binary: Path,
    backend: str,
    threads: int,
    output: Path,
) -> None:
    result = run([
        sys.executable, str(QUALIFIER), "--binary", str(binary),
        "--config", str(CONFIGS[backend]), "--compute-backend", backend,
        "--threads", str(threads), f"--require-{backend}",
        "--require-native-chaos", "--output", str(output),
    ], threads)
    require_success(result, f"{backend} {threads}-worker domain-recycling qualification")
    if "native Earth-control dispatch required=True" not in result.stdout:
        raise RuntimeError(f"{backend} qualification omitted native ownership evidence")


def swath_tile(output: Path) -> Path:
    candidates = list((output / "direct" / "domain_recycling_swath").glob(
        "transaction_*/tile_00"))
    if len(candidates) != 1:
        raise RuntimeError(f"{output}: expected one retiring-swath tile")
    return candidates[0]


def require_thread_identity(reference: Path, candidate: Path, backend: str) -> None:
    lhs = swath_tile(reference)
    rhs = swath_tile(candidate)
    for field in FIELDS:
        if (lhs / f"{field}.npy").read_bytes() != (rhs / f"{field}.npy").read_bytes():
            raise RuntimeError(f"{backend} one/four-worker {field} swath bytes differ")
    if (reference / "direct" / "trajectories.npy").read_bytes() != (
            candidate / "direct" / "trajectories.npy").read_bytes():
        raise RuntimeError(f"{backend} one/four-worker trajectory bytes differ")


def require_nonlinear_boundary_ledger(
    binary: Path, backend: str, output: Path
) -> int:
    result = run(
        [
            str(binary), "--headless", "--config", str(CONFIGS[backend]),
            "--duration=2", "--write-every=1", "--threads=1",
            f"--outdir={output}", "--no-timing",
        ],
        backend=backend,
    )
    require_success(result, f"{backend} nonlinear-boundary ledger run")
    require_selected_backend(
        result.stdout, backend, backend, "nonlinear-boundary ledger run"
    )
    for marker in ("Execution mode: qualified", "fallback=off"):
        if marker not in result.stdout:
            raise RuntimeError(
                f"{backend} nonlinear-boundary run omitted {marker!r}"
            )

    checks = require_report_owner_identities(output)
    rows = read_csv_rows(output / "thermodynamic_source_budget.csv")
    required_nonzero = (
        "acoustic_v1_boundary_radiative_committed_work_j",
        "acoustic_v1_boundary_committed_work_j",
        "advection_theta_energy_step_outward_boundary_j",
    )
    for column in required_nonzero:
        if not any(abs(float(row[column])) > 0.0 for row in rows):
            raise RuntimeError(
                f"{backend} nonlinear boundary owner remained vacuous: {column}"
            )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--backend", choices=("vulkan", "metal", "all"), default="all")
    args = parser.parse_args()
    binary = args.binary.resolve()
    backends = tuple(CONFIGS) if args.backend == "all" else (args.backend,)
    for path in (binary, QUALIFIER, *(CONFIGS[name] for name in backends)):
        if not path.is_file():
            raise RuntimeError(f"required native-chaos input is missing: {path}")

    with tempfile.TemporaryDirectory(prefix="nimbus-native-chaos-ready-") as temporary:
        root = Path(temporary)
        for backend in backends:
            source = CONFIGS[backend].read_text(encoding="utf-8")
            preflight = run([str(binary), "--validate-only", "--config", str(CONFIGS[backend])])
            require_success(preflight, f"{backend} preflight")
            if "[VALIDATE] PASS" not in preflight.stdout:
                raise RuntimeError(f"{backend} preflight omitted PASS")

            mutations = (
                ("    allow_fallback: false\n", "    allow_fallback: true\n", "fallback"),
                (f"    backend: {backend}\n", "    backend: cpu\n", "backend"),
                ("  nx: 18\n", "  nx: 19\n", "grid"),
                ("    scheme: full_stochastic\n", "    scheme: none\n", "scheme"),
            )
            for old, new, label in mutations:
                require_rejection(binary, backend, source, old, new, root, label)

            outputs: dict[int, Path] = {}
            for threads in (1, 4):
                output = root / f"{backend}-{threads}"
                run_qualification(binary, backend, threads, output)
                outputs[threads] = output
            require_thread_identity(outputs[1], outputs[4], backend)
            checks = require_nonlinear_boundary_ledger(
                binary, backend, root / f"{backend}-nonlinear-ledger"
            )
            if checks == 0:
                raise RuntimeError(
                    f"{backend} nonlinear-boundary owner gate was vacuous"
                )

    print(
        "[native-chaos-accelerator-ready] PASS: exact fallback-off "
        f"{'/'.join(backends)} "
        "Earth-control evolution retained native ownership through translation "
        "and restart, one/four-worker artifacts were byte-identical, nonlinear "
        "boundary ledgers independently recomposed, and public scope mutations "
        "failed closed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
