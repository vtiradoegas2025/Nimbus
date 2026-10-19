#!/usr/bin/env python3
"""Bounded native stochastic continuation across an actual noise epoch."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.qualification.checkpointing.qualify_vulkan_restart import require_restart_suffix


def run(binary: Path, backend: str, output: Path, resume: Path | None = None) -> None:
    config = ROOT / f"tests/fixtures/configs/qualification/domain_recycling_kessler_native_chaos_{backend}_ready.yaml"
    command = [str(binary), "--headless", f"--config={config}", "--duration=61",
               "--write-every=30", "--checkpoint-every=30", "--threads=1",
               f"--outdir={output}", "--no-timing"]
    if resume is not None:
        command.append(f"--resume={resume}")
    environment = os.environ.copy()
    environment["OMP_DYNAMIC"] = "FALSE"
    result = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    output.mkdir(parents=True, exist_ok=True)
    (output / "runtime.log").write_text(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"{backend} continuation failed: {result.stdout}")
    if f"[COMPUTE] backend={backend}" not in result.stdout or "fallback=off" not in result.stdout:
        raise RuntimeError(f"{backend} did not retain its requested device owner")
    dispatches = re.findall(r"chaos/earth_control_evolution[^:,}]*:(\d+)", result.stdout)
    if not dispatches or int(dispatches[-1]) < 3:
        raise RuntimeError(f"{backend} omitted native stochastic evolution after initialization/restore")
    if resume is not None and "[CHECKPOINT] resumed step=60, time_s=30" not in result.stdout:
        raise RuntimeError(f"{backend} did not restore the intended pre-epoch checkpoint")


def epoch_frame(output: Path) -> Path:
    for path in output.glob("step_*/manifest.json"):
        manifest = json.loads(path.read_text())
        # The manifest owns the authoritative physical time of each frame.
        if manifest["simulation_time_s"] == 60.0:
            return path.parent
    raise RuntimeError(f"{output} omitted its t=60 epoch frame")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--backend", choices=("metal", "vulkan", "all"), default="all")
    args = parser.parse_args()
    backends = ("metal", "vulkan") if args.backend == "all" else (args.backend,)
    with tempfile.TemporaryDirectory(prefix="nimbus_native_chaos_restart_") as temporary:
        for backend in backends:
            root = Path(temporary) / backend
            continuous, resumed = root / "continuous", root / "resumed"
            run(args.binary.resolve(), backend, continuous)
            checkpoint = continuous / "checkpoints/checkpoint_step_000000060.nimbus"
            if not checkpoint.is_file():
                raise RuntimeError(f"{backend} omitted the pre-epoch checkpoint")
            run(args.binary.resolve(), backend, resumed, checkpoint)
            lhs, rhs = epoch_frame(continuous), epoch_frame(resumed)
            fields = ("p", "qr", "qv", "rho", "theta", "u", "v", "w")
            for field in fields:
                if (lhs / f"{field}.npy").read_bytes() != (rhs / f"{field}.npy").read_bytes():
                    raise RuntimeError(f"{backend} {field} changed across stochastic restart")
            require_restart_suffix(continuous, resumed)
            print(f"[native-chaos-restart] PASS: {backend}, native epoch, fields and budget suffixes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
