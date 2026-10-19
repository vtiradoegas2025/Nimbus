#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    "mynn": ROOT / "tests/fixtures/configs/qualification/cgrid_mynn_rrtmg_ready.yaml",
    "ysu": ROOT / "tests/fixtures/configs/qualification/cgrid_ysu_rrtmg_ready.yaml",
}
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"


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


def exercise_profile(
    binary: Path, scheme: str, config: Path, output_root: Path
) -> None:
    validated = run([str(binary), "--validate-only", "--config", str(config)])
    require_success(validated, f"{scheme} qualified preflight")
    if "[VALIDATE] PASS" not in validated.stdout:
        raise RuntimeError(f"{scheme} qualified preflight did not publish PASS")

    continuous = output_root / f"{scheme}-continuous"
    completed = run([
        str(binary), "--headless", "--config", str(config),
        f"--outdir={continuous}", "--write-every=0", "--threads=4",
        "--log-profile=normal",
    ])
    require_success(completed, f"{scheme} bounded coupled run")
    for marker in (
        "Execution mode: qualified",
        "Initialized radiation scheme: rrtmg",
        f"Initialized boundary layer scheme: {scheme}",
        "[COMPUTE] backend=cpu",
        "fallback=off",
    ):
        if marker not in completed.stdout:
            raise RuntimeError(
                f"{scheme} bounded coupled run omitted {marker!r}")

    checkpoint = continuous / "checkpoints/checkpoint_step_000000005.nimbus"
    if not checkpoint.is_file() or not Path(f"{checkpoint}.json").is_file():
        raise RuntimeError(
            f"{scheme} bounded coupled run did not publish its restart pair")

    resumed = run([
        str(binary), "--headless", "--config", str(config),
        f"--resume={checkpoint}", "--duration=2", "--write-every=0",
        "--checkpoint-every=0", f"--outdir={output_root / f'{scheme}-resumed'}",
        "--threads=4", "--log-profile=normal",
    ])
    require_success(resumed, f"{scheme} bounded resumed run")
    if "[CHECKPOINT] resumed step=5, time_s=1" not in resumed.stdout:
        raise RuntimeError(
            f"{scheme} bounded resumed run did not restore the split state")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    args = parser.parse_args()
    binary = args.binary.resolve()
    required = [
        binary,
        *CONFIGS.values(),
        ROOT / "data/rrtmg/rrtmg_lw.nc",
        ROOT / "data/rrtmg/rrtmg_sw.nc",
    ]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"required ready-state input is missing: {path}")

    mynn_source = CONFIGS["mynn"].read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="nimbus-pbl-rrtmg-ready-") as temporary:
        root = Path(temporary)
        slab_config = root / "slab.yaml"
        slab_config.write_text(
            mynn_source.replace("  scheme: mynn\n", "  scheme: slab\n", 1),
            encoding="utf-8",
        )
        slab = run([
            str(binary), "--headless", "--config", str(slab_config),
            "--duration=0", "--write-every=0", f"--outdir={root / 'slab'}",
            "--threads=4", "--log-profile=normal",
        ])
        require_success(slab, "qualified slab surface profile")
        if "Initialized boundary layer scheme: slab" not in slab.stdout:
            raise RuntimeError(
                "qualified slab surface profile did not reach the implementation")

        for scheme, config in CONFIGS.items():
            exercise_profile(binary, scheme, config, root)

    print(
        "[pbl-rrtmg-ready] PASS: slab admitted; MYNN and YSU admitted, coupled, "
        "checkpointed, and resumed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
