#!/usr/bin/env python3
"""Qualify bounded live-SHM cadence without durable history duplication."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import tempfile
import os


ROOT = Path(__file__).resolve().parents[3]
BINARY = ROOT / "bin" / "nimbus_sim"
SOURCE_CONFIG = (
    ROOT / "tests" / "fixtures" / "configs" / "test" / "acceptance"
    / "t0_cart_calm.yaml"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def configured_yaml(*, full_rate: bool, fields: tuple[str, ...] = ("w", "theta")) -> str:
    text = SOURCE_CONFIG.read_text(encoding="utf-8")
    replacement = (
        "  live_shm: true\n"
        f"  live_shm_fields: [{', '.join(fields)}]\n"
        "  live_shm_interval_s: 4\n"
        f"  live_shm_full_rate: {'true' if full_rate else 'false'}\n"
    )
    require("  live_shm: false\n" in text, "fixture has no live_shm anchor")
    return text.replace("  live_shm: false\n", replacement, 1)


def endpoint_name(label: str) -> str:
    return f"/nimbus_q_{os.getpid()}_{label}"[:30]


def run_case(root: Path, *, threads: int, full_rate: bool,
             fields: tuple[str, ...] = ("w", "theta")) -> dict[str, object]:
    label = f"{'full' if full_rate else 'cadence'}-t{threads}"
    if fields != ("w", "theta"):
        label += "-diagnostics"
    config = root / f"{label}.yaml"
    output = root / label
    config.write_text(configured_yaml(full_rate=full_rate, fields=fields), encoding="utf-8")
    command = [
        str(BINARY), "--headless", f"--config={config}", "--duration=6",
        "--write-every=0", f"--threads={threads}", f"--outdir={output}",
        f"--live-shm-name={endpoint_name(label)}",
        "--no-timing",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    require(
        result.returncode == 0,
        f"{label} failed ({result.returncode})\n{result.stdout}\n{result.stderr}",
    )
    telemetry_path = output / "nimbus_live_shm.json"
    require(telemetry_path.is_file(), f"{label} omitted SHM telemetry")
    require(
        not list(output.glob("step_*")),
        f"{label} duplicated live frames into durable history",
    )
    require(
        not (output / "nimbus_live_shm.json.tmp").exists(),
        f"{label} left temporary telemetry visible",
    )
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    require(
        telemetry["endpoint_name"] == endpoint_name(label),
        f"{label} endpoint telemetry drift",
    )
    require(telemetry["fields"] == list(fields), f"{label} field contract drift")
    require(telemetry["failed_frames"] == 0, f"{label} reported a failed frame")
    require(
        telemetry["final_sequence"] == 2 * telemetry["frames_published"],
        f"{label} sequence/frame accounting is inconsistent",
    )
    require(
        re.fullmatch(r"[0-9a-fA-F]{32}", telemetry["publication_generation"])
        is not None,
        f"{label} has invalid generation",
    )
    return telemetry


def qualify_restart(root: Path) -> None:
    config = root / "restart.yaml"
    config.write_text(configured_yaml(full_rate=False), encoding="utf-8")
    parent_output = root / "restart-parent"
    checkpoint_dir = root / "restart-checkpoints"
    parent_command = [
        str(BINARY), "--headless", f"--config={config}", "--duration=4",
        "--write-every=0", "--threads=1", "--checkpoint-every=4",
        f"--checkpoint-dir={checkpoint_dir}", f"--outdir={parent_output}",
        f"--live-shm-name={endpoint_name('restart_parent')}",
        "--no-timing",
    ]
    parent = subprocess.run(parent_command, cwd=ROOT, text=True, capture_output=True)
    require(parent.returncode == 0, f"restart parent failed\n{parent.stdout}\n{parent.stderr}")
    checkpoints = list(checkpoint_dir.glob("*.nimbus"))
    require(len(checkpoints) == 1, "restart parent did not publish exactly one checkpoint")
    parent_telemetry = json.loads(
        (parent_output / "nimbus_live_shm.json").read_text(encoding="utf-8")
    )
    require(
        parent_telemetry["endpoint_name"] == endpoint_name("restart_parent"),
        "restart parent endpoint telemetry drift",
    )

    resumed_output = root / "restart-resumed"
    resumed_command = [
        str(BINARY), "--headless", f"--config={config}", "--duration=6",
        "--write-every=0", "--threads=1", "--checkpoint-every=0",
        f"--resume={checkpoints[0]}", f"--outdir={resumed_output}",
        f"--live-shm-name={endpoint_name('restart_resumed')}",
        "--no-timing",
    ]
    resumed = subprocess.run(resumed_command, cwd=ROOT, text=True, capture_output=True)
    require(resumed.returncode == 0, f"restart branch failed\n{resumed.stdout}\n{resumed.stderr}")
    resumed_telemetry = json.loads(
        (resumed_output / "nimbus_live_shm.json").read_text(encoding="utf-8")
    )
    require(
        resumed_telemetry["endpoint_name"] == endpoint_name("restart_resumed"),
        "restart branch endpoint telemetry drift",
    )
    require(
        resumed_telemetry["publication_generation"] !=
        parent_telemetry["publication_generation"],
        "resumed live branch reused its parent publication generation",
    )
    require(
        resumed_telemetry["frames_published"] == 1 and
        resumed_telemetry["final_sequence"] == 2,
        "resumed branch did not publish exactly one coherent restored initial state",
    )
    require(not list(resumed_output.glob("step_*")), "restart duplicated live state into durable history")


def comparable(telemetry: dict[str, object]) -> dict[str, object]:
    return {
        key: value for key, value in telemetry.items()
        if key not in {"publication_generation", "endpoint_name", "publish_time_s"}
    }


def main() -> int:
    require(BINARY.is_file(), "bin/nimbus_sim is missing")
    require(SOURCE_CONFIG.is_file(), "calm strict-v1 fixture is missing")
    with tempfile.TemporaryDirectory(prefix="nimbus_live_shm_") as temporary:
        root = Path(temporary)
        cadence_t1 = run_case(root, threads=1, full_rate=False)
        cadence_t4 = run_case(root, threads=4, full_rate=False)
        require(
            comparable(cadence_t1) == comparable(cadence_t4),
            "live cadence telemetry differs between one and four threads",
        )
        full_t1 = run_case(root, threads=1, full_rate=True)
        require(
            full_t1["frames_published"] > cadence_t1["frames_published"],
            "explicit full-rate mode did not publish more frames",
        )
        require(
            cadence_t1["cadence_skipped_steps"] > 0
            and full_t1["cadence_skipped_steps"] == 0,
            "cadence skip telemetry does not distinguish bounded/full-rate modes",
        )
        qualify_restart(root)
        diagnostics_t1 = run_case(root, threads=1, full_rate=False, fields=("p_prime", "vorticity_z"))
        diagnostics_t4 = run_case(root, threads=4, full_rate=False, fields=("p_prime", "vorticity_z"))
        require(comparable(diagnostics_t1) == comparable(diagnostics_t4),
                "live diagnostic publication differs between one and four threads")
    print(
        "[live-shm] PASS: initial+bounded cadence, explicit full-rate, "
        "1/4-thread identity, restart generation/sequence, telemetry, and no "
        "durable frame duplication; pressure/vorticity publication"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
