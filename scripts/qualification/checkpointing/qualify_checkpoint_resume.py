#!/usr/bin/env python3
"""Short production-binary checkpoint/resume lifecycle qualification."""

from __future__ import annotations

import json
import hashlib
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BINARY = ROOT / "bin" / "nimbus_sim"
CONFIG = ROOT / "tests" / "fixtures" / "configs" / "test" / "acceptance" / "t0_cart_calm.yaml"


def run(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}"
        )


def load_sidecar(checkpoint: Path) -> dict[str, object]:
    sidecar = Path(f"{checkpoint}.json")
    if not sidecar.is_file():
        raise RuntimeError(f"missing checkpoint sidecar: {sidecar}")
    return json.loads(sidecar.read_text(encoding="utf-8"))


def main() -> int:
    if not BINARY.is_file():
        raise RuntimeError(f"missing production binary: {BINARY}")
    with tempfile.TemporaryDirectory(
        prefix="nimbus_checkpoint_resume_"
    ) as temp:
        root = Path(temp)
        first_checkpoints = root / "first_checkpoints"
        first_output = root / "first_output"
        base = [
            str(BINARY),
            "--config", str(CONFIG),
            "--headless",
            "--compute-backend", "cpu",
            "--threads", "1",
            "--write-every=0",
            "--checkpoint-every=1",
            "--log-profile", "quiet",
        ]
        run(base + [
            "--duration", "1",
            f"--checkpoint-dir={first_checkpoints}",
            f"--outdir={first_output}",
        ])
        first = sorted(first_checkpoints.glob("checkpoint_step_*.nimbus"))
        if len(first) != 1:
            raise RuntimeError(
                f"expected one first-branch checkpoint, found {first}"
            )
        first_meta = load_sidecar(first[0])
        if first_meta.get("checkpoint_sha256") != hashlib.sha256(
                first[0].read_bytes()).hexdigest():
            raise RuntimeError("first checkpoint SHA-256 identity mismatch")
        if (first_meta.get("completed_steps") != 1
                or first_meta.get("simulation_time_s") != 1):
            raise RuntimeError(
                f"unexpected first checkpoint lifecycle: {first_meta}"
            )
        if first_meta.get("domain_recycling", "missing") is not None:
            raise RuntimeError(
                "calm no-recycling checkpoint must publish a null D5 head"
            )

        resumed_checkpoints = root / "resumed_checkpoints"
        resumed_output = root / "resumed_output"
        run(base + [
            "--duration", "2",
            f"--checkpoint-dir={resumed_checkpoints}",
            f"--outdir={resumed_output}",
            f"--resume={first[0]}",
        ])
        resumed = sorted(
            resumed_checkpoints.glob("checkpoint_step_*.nimbus")
        )
        if len(resumed) != 1:
            raise RuntimeError(
                f"expected one resumed checkpoint, found {resumed}"
            )
        resumed_meta = load_sidecar(resumed[0])
        if resumed_meta.get("checkpoint_sha256") != hashlib.sha256(
                resumed[0].read_bytes()).hexdigest():
            raise RuntimeError("resumed checkpoint SHA-256 identity mismatch")
        if (resumed_meta.get("completed_steps") != 2
                or resumed_meta.get("simulation_time_s") != 2):
            raise RuntimeError(
                f"unexpected resumed lifecycle: {resumed_meta}"
            )
        if resumed_meta.get("parent_run_id") != first_meta.get("run_id"):
            raise RuntimeError(
                "resumed checkpoint does not name the first run as parent"
            )
        branch = json.loads(
            (resumed_output / "nimbus_branch.json").read_text(
                encoding="utf-8"
            )
        )
        if branch.get("parent_run_id") != first_meta.get("run_id"):
            raise RuntimeError(
                "branch provenance does not match checkpoint parent"
            )

        tampered = dict(first_meta)
        tampered["checkpoint_sha256"] = "0" * 64
        Path(f"{first[0]}.json").write_text(
            json.dumps(tampered, indent=2) + "\n", encoding="utf-8"
        )
        rejected = subprocess.run(
            base + [
                "--duration", "2",
                f"--checkpoint-dir={root / 'tampered_checkpoints'}",
                f"--outdir={root / 'tampered_output'}",
                f"--resume={first[0]}",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if (rejected.returncode != 1 or
                "sidecar integrity mismatch" not in rejected.stdout):
            raise RuntimeError(
                "tampered SHA-256 sidecar was not rejected:\n" + rejected.stdout
            )

    print(
        "[checkpoint-resume] PASS: runtime v37 restored live dynamics "
        "and advanced 1 s -> 2 s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
