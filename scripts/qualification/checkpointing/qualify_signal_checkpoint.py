#!/usr/bin/env python3
"""Qualify completed-timestep SIGINT/SIGTERM shutdown and checkpoint resume."""

from __future__ import annotations

import json
import signal
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BINARY = ROOT / "bin" / "nimbus_sim"
CONFIG = ROOT / "tests" / "fixtures" / "configs" / "test" / "acceptance" / "t0_cart_calm.yaml"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, object]:
    require(path.is_file(), f"missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_interrupted(root: Path, signal_number: int) -> tuple[Path, dict[str, object]]:
    label = signal.Signals(signal_number).name.lower()
    output = root / f"{label}_output"
    checkpoints = root / f"{label}_checkpoints"
    command = [
        str(BINARY), "--headless", "--config", str(CONFIG),
        "--duration", "1000000000", "--compute-backend", "cpu",
        "--threads", "1", "--write-every=0",
        "--checkpoint-every=1000000000",
        f"--checkpoint-dir={checkpoints}", f"--outdir={output}",
        "--log-profile", "quiet",
    ]
    process = subprocess.Popen(
        command, cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 15.0
    while not output.is_dir() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    require(process.poll() is None, f"process exited before {label} could be delivered")
    require(output.is_dir(), f"runtime did not initialize before {label}")
    time.sleep(0.05)
    process.send_signal(signal_number)
    try:
        log, _ = process.communicate(timeout=30.0)
    except subprocess.TimeoutExpired:
        process.kill()
        log, _ = process.communicate()
        raise RuntimeError(f"graceful {label} shutdown timed out\n{log}")

    expected_status = 128 + signal_number
    require(process.returncode == expected_status,
            f"{label} returned {process.returncode}, expected {expected_status}\n{log}")
    termination = load_json(output / "nimbus_termination.json")
    require(termination.get("schema") == "nimbus.termination", f"bad schema: {termination}")
    require(termination.get("signal") == signal.Signals(signal_number).name,
            f"bad signal record: {termination}")
    require(termination.get("exit_status") == expected_status, f"bad status record: {termination}")
    require(termination.get("boundary") == "completed_timestep", f"unsafe boundary: {termination}")
    require(termination.get("output_drained") is True, f"output was not drained: {termination}")
    require(termination.get("checkpoint_committed") is True,
            f"checkpoint was not committed: {termination}")

    checkpoint_files = sorted(checkpoints.glob("checkpoint_step_*.nimbus"))
    require(len(checkpoint_files) == 1, f"expected one signal checkpoint, found {checkpoint_files}")
    checkpoint = checkpoint_files[0]
    lifecycle = load_json(Path(f"{checkpoint}.json"))
    require(lifecycle.get("boundary") == "completed_timestep",
            f"unsafe checkpoint lifecycle: {lifecycle}")
    require(lifecycle.get("completed_steps") == termination.get("completed_steps")
            and lifecycle.get("simulation_time_s") == termination.get("simulation_time_s"),
            "termination record and checkpoint lifecycle disagree")
    require(termination.get("checkpoint_file") == str(checkpoint),
            "termination record names the wrong checkpoint")
    return checkpoint, lifecycle


def qualify_resume(root: Path, checkpoint: Path, lifecycle: dict[str, object]) -> None:
    source_time = lifecycle.get("simulation_time_s")
    source_step = lifecycle.get("completed_steps")
    require(isinstance(source_time, (int, float)) and source_time >= 0,
            "invalid source simulation time")
    require(isinstance(source_step, int) and source_step >= 0,
            "invalid source completed step")
    target_time = int(source_time) + 1
    resumed_checkpoints = root / "resumed_checkpoints"
    command = [
        str(BINARY), "--headless", "--config", str(CONFIG),
        "--duration", str(target_time), "--compute-backend", "cpu",
        "--threads", "1", "--write-every=0", "--checkpoint-every=1",
        f"--checkpoint-dir={resumed_checkpoints}",
        f"--outdir={root / 'resumed_output'}", f"--resume={checkpoint}",
        "--log-profile", "quiet",
    ]
    completed = subprocess.run(
        command, cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    require(completed.returncode == 0,
            f"signal checkpoint resume failed\n{completed.stdout}")
    resumed = sorted(resumed_checkpoints.glob("checkpoint_step_*.nimbus"))
    require(len(resumed) == 1, f"expected one resumed checkpoint, found {resumed}")
    resumed_lifecycle = load_json(Path(f"{resumed[0]}.json"))
    require(resumed_lifecycle.get("completed_steps") == source_step + 1,
            "resume did not advance one exact step")
    require(resumed_lifecycle.get("simulation_time_s") == target_time,
            "resume ended at the wrong simulation time")
    require(resumed_lifecycle.get("parent_run_id") == lifecycle.get("run_id"),
            "resume lineage lost signal parent")


def main() -> int:
    require(BINARY.is_file(), f"missing production binary: {BINARY}")
    with tempfile.TemporaryDirectory(prefix="nimbus_signal_checkpoint_") as temporary:
        root = Path(temporary)
        run_interrupted(root, signal.SIGINT)
        term_checkpoint, term_lifecycle = run_interrupted(root, signal.SIGTERM)
        qualify_resume(root, term_checkpoint, term_lifecycle)
    print("[signal-checkpoint] PASS: SIGINT=130, SIGTERM=143, checkpoint resume advanced exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
