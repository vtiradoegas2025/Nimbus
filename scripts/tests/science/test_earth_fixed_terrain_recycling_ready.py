#!/usr/bin/env python3
"""Bounded Earth-fixed terrain/D5 movement and continuation gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.qualification.checkpointing.qualify_vulkan_restart import (  # noqa: E402
    LEDGERS,
    require_report_owner_identities,
    require_restart_suffix,
)

BINARY = ROOT / "bin" / "nimbus_sim"
CONFIG = (
    ROOT / "tests" / "fixtures" / "configs" / "qualification"
    / "earth_fixed_terrain_recycling_ready.yaml"
)
FINAL_FIELDS = (
    "p.npy", "qr.npy", "qv.npy", "rho.npy", "theta.npy", "u.npy",
    "v.npy", "w.npy",
)


def run(arguments: list[str], expect_success: bool = True) -> str:
    completed = subprocess.run(
        [str(BINARY), *arguments], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if expect_success != (completed.returncode == 0):
        raise RuntimeError(
            f"unexpected return code {completed.returncode}:\n{completed.stdout}"
        )
    return completed.stdout


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern!r} below {path}, found {matches}")
    return matches[0]


def base(output: Path, threads: int, config: Path = CONFIG) -> list[str]:
    return [
        "--headless", "--config", str(config), "--compute-backend=cpu",
        f"--threads={threads}", "--log-profile=quiet", "--no-timing",
        f"--outdir={output}",
    ]


def verify_movement(output: Path) -> float:
    event_paths = sorted((output / "domain_recycling_swath").glob(
        "transaction_*/event.json"
    ))
    if not event_paths:
        raise RuntimeError("Earth-fixed D5 run published no movement event")
    event_times: list[float] = []
    cumulative_shift = [0, 0]
    for event_path in event_paths:
        event = json.loads(event_path.read_text(encoding="utf-8"))
        event_time = event.get("simulation_time_s")
        shift = event.get("shift_cells")
        if (not isinstance(shift, list) or len(shift) != 2
                or not all(isinstance(value, int) for value in shift)
                or shift == [0, 0]
                or max(abs(shift[0]), abs(shift[1])) > 8
                or not isinstance(event_time, (int, float))
                or not 1.0 <= float(event_time) <= 65.0):
            raise RuntimeError(f"unexpected Earth-fixed D5 event: {event}")
        if event_times and float(event_time) <= event_times[-1]:
            raise RuntimeError("Earth-fixed D5 event times are not increasing")
        event_times.append(float(event_time))
        cumulative_shift[0] += shift[0]
        cumulative_shift[1] += shift[1]
    manifest = json.loads(
        (output / "step_000001" / "manifest.json").read_text(encoding="utf-8")
    )
    origin = manifest["grid"]["horizontal_frame"]["grid_origin_t0_earth_m"]
    if origin != [500 * cumulative_shift[0], 500 * cumulative_shift[1]]:
        raise RuntimeError(f"terrain movement did not publish the shifted Earth origin: {origin}")
    if manifest["grid"]["vertical_geometry"]["kind"] != "terrain_following":
        raise RuntimeError("post-move history lost terrain-following geometry")
    track = json.loads(
        (output / "storm_track_latest.json").read_text(encoding="utf-8")
    )
    if (track.get("mode")
            != "tracking_with_earth_fixed_terrain_cgrid_recycling"
            or track.get("movement_authority", {}).get("mode")
            != "live_earth_fixed_terrain_cgrid_recycling"):
        raise RuntimeError("storm evidence mislabeled Earth-fixed terrain recycling")
    return event_times[0]


def geometry_digests(output: Path) -> tuple[str, str]:
    step = output / "step_000001"
    manifest = json.loads((step / "manifest.json").read_text(encoding="utf-8"))
    geometry = manifest["grid"]["vertical_geometry"]
    return tuple(
        digest((step / geometry[key]).resolve())
        for key in ("center_path", "interface_path")
    )


def main() -> int:
    global BINARY
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=BINARY)
    arguments = parser.parse_args()
    BINARY = arguments.binary.resolve()
    if not BINARY.is_file() or not CONFIG.is_file():
        raise RuntimeError("Earth-fixed terrain D5 binary or fixture is missing")
    validate = run(["--validate-only", "--config", str(CONFIG)])
    if "[VALIDATE] PASS" not in validate:
        raise RuntimeError("strict Earth-fixed terrain D5 preflight omitted PASS")

    with tempfile.TemporaryDirectory(prefix="nimbus-earth-fixed-d5-") as temp:
        root = Path(temp)
        startup = run([
            "--headless", "--config", str(CONFIG), "--duration=0",
            "--write-every=0", "--threads=1", "--log-profile=normal",
            f"--outdir={root / 'startup'}",
        ])
        for marker in (
            "Execution mode: qualified",
            "[COMPUTE] backend=cpu",
            "movement_mode=live_earth_fixed_terrain_cgrid_recycling",
        ):
            if marker not in startup:
                raise RuntimeError(
                    f"ordinary qualified startup omitted {marker!r}"
                )
        outputs = {threads: root / f"movement-t{threads}" for threads in (1, 4)}
        event_times: dict[int, float] = {}
        for threads, output in outputs.items():
            run(base(output, threads) + ["--duration=65", "--write-every=65"])
            event_times[threads] = verify_movement(output)
        for name in FINAL_FIELDS:
            first = digest(outputs[1] / "step_000001" / name)
            fourth = digest(outputs[4] / "step_000001" / name)
            if first != fourth:
                raise RuntimeError(f"1/4-thread post-move field mismatch: {name}")
        if geometry_digests(outputs[1]) != geometry_digests(outputs[4]):
            raise RuntimeError("1/4-thread post-move terrain geometry mismatch")
        for ledger in LEDGERS:
            first = outputs[1] / ledger
            fourth = outputs[4] / ledger
            if not first.is_file() or not fourth.is_file():
                raise RuntimeError(f"terrain D5 run omitted {ledger}")
            if first.read_bytes() != fourth.read_bytes():
                raise RuntimeError(f"1/4-thread terrain D5 ledger mismatch: {ledger}")
        one_checks = require_report_owner_identities(
            outputs[1], minimum_time_s=event_times[1])
        four_checks = require_report_owner_identities(
            outputs[4], minimum_time_s=event_times[4])
        if one_checks == 0 or one_checks != four_checks:
            raise RuntimeError("terrain D5 report-owner identity coverage is incomplete")

        source_out = root / "checkpoint-source-output"
        source_checkpoints = root / "checkpoint-source"
        run(base(source_out, 1) + [
            "--duration=65", "--write-every=65", "--checkpoint-every=62",
            f"--checkpoint-dir={source_checkpoints}",
        ])
        sources = sorted(source_checkpoints.glob("checkpoint_step_*.nimbus"))
        if len(sources) < 2:
            raise RuntimeError("terrain D5 source run omitted periodic/final checkpoints")
        source = sources[0]
        source_meta = json.loads(
            Path(f"{source}.json").read_text(encoding="utf-8")
        )
        recycling = source_meta.get("domain_recycling")
        if (not isinstance(recycling, dict) or recycling.get("sequence") != 1
                or recycling.get("shift_algorithm")
                != "stagger_aware_destination_indexed_terrain_whole_cell_shift_v1"
                or recycling.get("control_volume_exchange_complete") is not True
                or recycling.get("control_volume_exchange_basis")
                != "instantaneous_fixed_earth_control_volume_exchange_v1"):
            raise RuntimeError("post-move checkpoint omitted D5 provenance")
        expected_owners = {
            "boundary_layer": "held_sample_flat_or_mapped_pbl_shift_v2",
            "radiation": "held_sample_time_flat_or_mapped_radiation_shift_v2",
            "microphysics": "persistent_private_moments_flat_or_mapped_environment_shift_v2",
            "chaos": "earth_control_lattice_translate_or_shift_pbl_cache_v3",
        }
        if recycling.get("owner_algorithms") != expected_owners:
            raise RuntimeError(
                f"post-move checkpoint owner identities disagree: {recycling}"
            )

        resumed: dict[int, Path] = {}
        resumed_outputs: dict[int, Path] = {}
        for threads in (1, 4):
            checkpoint_dir = root / f"resume-t{threads}-checkpoints"
            resumed_outputs[threads] = root / f"resume-t{threads}-output"
            run(base(resumed_outputs[threads], threads) + [
                "--duration=65", "--write-every=65", "--checkpoint-every=65",
                f"--checkpoint-dir={checkpoint_dir}", f"--resume={source}",
            ])
            resumed[threads] = one(checkpoint_dir, "checkpoint_step_*.nimbus")
        for name in FINAL_FIELDS:
            if digest(
                resumed_outputs[1] / "step_000001" / name
            ) != digest(resumed_outputs[4] / "step_000001" / name):
                raise RuntimeError(
                    f"post-move continuation field is not 1/4-thread identical: {name}"
                )
        if geometry_digests(resumed_outputs[1]) != geometry_digests(
            resumed_outputs[4]
        ):
            raise RuntimeError(
                "post-move continuation terrain geometry is not 1/4-thread identical"
            )
        for ledger in LEDGERS:
            if (resumed_outputs[1] / ledger).read_bytes() != (
                resumed_outputs[4] / ledger
            ).read_bytes():
                raise RuntimeError(
                    f"post-move continuation ledger is not 1/4-thread identical: {ledger}"
                )
        require_restart_suffix(outputs[1], resumed_outputs[1])
        require_report_owner_identities(
            resumed_outputs[1], minimum_time_s=62.0)

        invalid = root / "grid-attached-invalid.yaml"
        text = CONFIG.read_text(encoding="utf-8")
        text = text.replace(
            "  horizontal_placement: earth_fixed\n", "  horizontal_placement: grid_attached\n"
        ).replace("  feature_center_earth_x_m: 4500\n", "").replace(
            "  feature_center_earth_y_m: 4500\n", ""
        )
        invalid.write_text(text, encoding="utf-8")
        rejected = run([
            "--headless", "--config", str(invalid), "--duration=0",
            "--write-every=0", f"--outdir={root / 'invalid-output'}",
            "--log-profile=normal",
        ], expect_success=False)
        if "domain_motion.mode=tracked_window is not reference-qualified" not in rejected:
            raise RuntimeError(
                "grid-attached moving terrain did not fail closed:\n" + rejected
            )

        near_profile = root / "near-profile-radiation-cadence.yaml"
        text = CONFIG.read_text(encoding="utf-8").replace(
            "  cadence_s: 2\n", "  cadence_s: 3\n", 1
        )
        near_profile.write_text(text, encoding="utf-8")
        rejected = run(
            ["--validate-only", "--config", str(near_profile)],
            expect_success=False,
        )
        if ("microphysics.scheme=kessler is qualified only"
                not in rejected
                or "qualified mapped terrain is limited" not in rejected):
            raise RuntimeError(
                "one-field near profile bypassed validate maturity/scope rejection:\n"
                + rejected
            )
        rejected = run(
            base(root / "near-profile-output", 1, near_profile)
                + ["--duration=0", "--write-every=0"],
            expect_success=False,
        )
        if "tracked_window is not reference-qualified" not in rejected:
            raise RuntimeError(
                "one-field near profile bypassed ordinary startup scope rejection:\n"
                + rejected
            )

    print(
        "[earth-fixed-terrain-recycling-ready] PASS: physical-height leading "
        "columns, movement, 1/4 identity, restart identity, and negative "
        "admission are bounded"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
