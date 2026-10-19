#!/usr/bin/env python3
"""Qualify recovery from the newest committed boundary after SIGKILL."""

from __future__ import annotations

import hashlib
import json
import signal
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BINARY = ROOT / "bin" / "nimbus_sim"
CONFIG = ROOT / "tests" / "fixtures" / "configs" / "test" / "acceptance" / "t0_cart_calm.yaml"


def complete_checkpoints(directory: Path) -> list[tuple[Path, dict[str, object]]]:
    result: list[tuple[Path, dict[str, object]]] = []
    for checkpoint in directory.glob("checkpoint_step_*.nimbus"):
        sidecar = Path(f"{checkpoint}.json")
        if not sidecar.is_file():
            continue
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (metadata.get("checkpoint_bytes") != checkpoint.stat().st_size or
                metadata.get("checkpoint_sha256") !=
                hashlib.sha256(checkpoint.read_bytes()).hexdigest()):
            continue
        result.append((checkpoint, metadata))
    return sorted(result, key=lambda item: int(item[1]["completed_steps"]))


def run(command: list[str]) -> None:
    completed = subprocess.run(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}"
        )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="nimbus_hard_crash_") as temp:
        root = Path(temp)
        crashed_output = root / "crashed_output"
        crashed_checkpoints = root / "crashed_checkpoints"
        base = [
            str(BINARY), "--config", str(CONFIG), "--headless",
            "--compute-backend", "cpu", "--threads", "1",
            "--dt-large", "1", "--write-every=0",
            "--checkpoint-every=1", "--log-profile", "quiet",
        ]
        process = subprocess.Popen(
            base + [
                "--duration", "10000",
                f"--outdir={crashed_output}",
                f"--checkpoint-dir={crashed_checkpoints}",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 30.0
        committed: list[tuple[Path, dict[str, object]]] = []
        while time.monotonic() < deadline and process.poll() is None:
            committed = complete_checkpoints(crashed_checkpoints)
            if committed:
                break
            time.sleep(0.01)
        if not committed:
            process.kill()
            output, _ = process.communicate()
            raise RuntimeError(f"no checkpoint committed before deadline:\n{output}")

        process.kill()
        output, _ = process.communicate(timeout=10)
        expected_kill_status = -signal.SIGKILL
        if process.returncode != expected_kill_status:
            raise RuntimeError(
                f"hard-crash process returned {process.returncode}, expected "
                f"{expected_kill_status}:\n{output}"
            )

        committed = complete_checkpoints(crashed_checkpoints)
        if not committed:
            raise RuntimeError("SIGKILL damaged every previously committed checkpoint")
        checkpoint, metadata = committed[-1]
        step = int(metadata["completed_steps"])
        simulation_time = float(metadata["simulation_time_s"])
        if not simulation_time.is_integer():
            raise RuntimeError(f"fixture produced non-integral checkpoint time: {simulation_time}")

        resumed_output = root / "resumed_output"
        resumed_checkpoints = root / "resumed_checkpoints"
        run(base + [
            "--duration", str(int(simulation_time) + 1),
            f"--outdir={resumed_output}",
            f"--checkpoint-dir={resumed_checkpoints}",
            f"--resume={checkpoint}",
        ])
        resumed = complete_checkpoints(resumed_checkpoints)
        if len(resumed) != 1:
            raise RuntimeError(f"expected one resumed checkpoint, found {resumed}")
        resumed_meta = resumed[0][1]
        if (resumed_meta.get("completed_steps") != step + 1 or
                resumed_meta.get("simulation_time_s") != simulation_time + 1.0 or
                resumed_meta.get("parent_run_id") != metadata.get("run_id")):
            raise RuntimeError(f"hard-crash resume did not advance exactly: {resumed_meta}")

    print(
        "[hard-crash-checkpoint] PASS: SIGKILL preserved the newest committed "
        "boundary and resume advanced exactly one step"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
