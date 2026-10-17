#!/usr/bin/env python3
"""Qualify exact trajectory history across a portable checkpoint branch."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BINARY = ROOT / "bin" / "nimbus_sim"
CONFIG = ROOT / "tests" / "fixtures" / "configs" / "test" / "acceptance" / "t0_cart_calm.yaml"


def run(command: list[str], expected: int = 0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != expected:
        raise RuntimeError(
            f"command returned {completed.returncode}, expected {expected}: "
            f"{' '.join(command)}\n{completed.stdout}"
        )
    return completed


def only_checkpoint(directory: Path) -> Path:
    checkpoints = sorted(directory.glob("checkpoint_step_*.nimbus"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"expected one checkpoint in {directory}, found {checkpoints}")
    return checkpoints[0]


def lifecycle(checkpoint: Path) -> dict[str, object]:
    return json.loads(Path(f"{checkpoint}.json").read_text(encoding="utf-8"))


def main() -> int:
    if not BINARY.is_file():
        raise RuntimeError(f"missing production binary: {BINARY}")

    with tempfile.TemporaryDirectory(prefix="nimbus_trajectory_checkpoint_") as temp:
        root = Path(temp)
        base = [
            str(BINARY),
            "--config", str(CONFIG),
            "--headless",
            "--compute-backend", "cpu",
            "--threads", "1",
            "--dt-large", "1",
            "--write-every=0",
            "--checkpoint-every=1",
            "--log-profile", "quiet",
            "--trajectories",
            "--trajectory-seed-time=0",
            "--trajectory-count=1",
        ]

        first_output = root / "first_output"
        first_checkpoints = root / "first_checkpoints"
        run(base + [
            "--duration", "1",
            f"--outdir={first_output}",
            f"--checkpoint-dir={first_checkpoints}",
        ])
        parent = only_checkpoint(first_checkpoints)
        parent_meta = lifecycle(parent)
        if parent_meta.get("completed_steps") != 1:
            raise RuntimeError(f"unexpected parent lifecycle: {parent_meta}")

        resumed_output = root / "resumed_output"
        resumed_checkpoints = root / "resumed_checkpoints"
        run(base + [
            "--duration", "2",
            f"--outdir={resumed_output}",
            f"--checkpoint-dir={resumed_checkpoints}",
            f"--resume={parent}",
        ])
        resumed = only_checkpoint(resumed_checkpoints)
        resumed_meta = lifecycle(resumed)
        if (resumed_meta.get("completed_steps") != 2 or
                resumed_meta.get("parent_run_id") != parent_meta.get("run_id")):
            raise RuntimeError(f"unexpected resumed lifecycle: {resumed_meta}")

        continuous_output = root / "continuous_output"
        continuous_checkpoints = root / "continuous_checkpoints"
        run(base + [
            "--duration", "2",
            f"--outdir={continuous_output}",
            f"--checkpoint-dir={continuous_checkpoints}",
        ])

        resumed_trajectory = resumed_output / "trajectories.npy"
        continuous_trajectory = continuous_output / "trajectories.npy"
        if resumed_trajectory.read_bytes() != continuous_trajectory.read_bytes():
            raise RuntimeError(
                "resumed trajectory artifact differs from the uninterrupted reference"
            )
        resumed_manifest = json.loads(
            (resumed_output / "trajectories.json").read_text(encoding="utf-8")
        )
        if (resumed_manifest.get("total_samples") != 9 or
                resumed_manifest.get("realized_shape") != [3, 3, 9]):
            raise RuntimeError(f"trajectory history was not continued exactly: {resumed_manifest}")

        # A checkpoint made without trajectory state must fail closed if the new
        # branch requests trajectories; otherwise provenance would start midway.
        plain_output = root / "plain_output"
        plain_checkpoints = root / "plain_checkpoints"
        plain_base = [item for item in base if item not in {
            "--trajectories", "--trajectory-seed-time=0", "--trajectory-count=1"
        }]
        run(plain_base + [
            "--duration", "1",
            f"--outdir={plain_output}",
            f"--checkpoint-dir={plain_checkpoints}",
        ])
        refused = run(base + [
            "--duration", "2",
            f"--outdir={root / 'refused_output'}",
            f"--checkpoint-dir={root / 'refused_checkpoints'}",
            f"--resume={only_checkpoint(plain_checkpoints)}",
        ], expected=1)
        if "requires parcel state" not in refused.stdout:
            raise RuntimeError(f"missing fail-closed trajectory diagnostic:\n{refused.stdout}")

    print(
        "[trajectory-checkpoint] PASS: resumed 1 s -> 2 s with byte-identical "
        "three-sample parcel history; missing state refused"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
