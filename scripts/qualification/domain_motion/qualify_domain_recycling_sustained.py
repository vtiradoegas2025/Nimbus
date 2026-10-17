#!/usr/bin/env python3
"""Qualify a sustained chained flat Cartesian C-grid D5 campaign.

The production run must commit at least four live recycle transactions while
Kessler, slab surface/PBL, horizontal Smagorinsky, full stochastic physics,
simple-grey radiation, radiative lateral boundaries, and trajectories remain
active. The gate independently verifies every durable swath artifact and
transaction hash-chain link, checkpoints after the final commit, restores that
checkpoint, and proves that the restored chain head and Earth window persist.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import tempfile
from pathlib import Path

from qualify_domain_recycling import (
    EXPECTED_FIELDS,
    MAXIMUM_TRAJECTORY_HORIZONTAL_SPEED_MS,
    fnv1a64,
    load_json,
    load_trajectory_npy,
    pair,
    require,
    require_selected_backend,
    require_swath_brick_hierarchy,
    run_command,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "bin" / "nimbus_sim"
CONFIG = ROOT / "tests" / "fixtures" / "configs" / "test" / "domain_recycling_sustained.yaml"
CONTINUOUS_DURATION_S = 120
RESUMED_DURATION_S = 130
DT_S = 0.5
MINIMUM_COMMIT_COUNT = 4
DX_M = 1000.0
DY_M = 1000.0
FRAME_VELOCITY_MS = (-45.0, 0.0)
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
U64_MASK = 0xFFFFFFFFFFFFFFFF


def parse_hex_u64(value: object, context: str) -> int:
    require(isinstance(value, str) and len(value) == 16, f"invalid {context}")
    try:
        parsed = int(value, 16)
    except ValueError as exception:
        raise RuntimeError(f"invalid {context}") from exception
    require(0 <= parsed <= U64_MASK, f"invalid {context}")
    return parsed


def transaction_chain(
    previous: int,
    sequence: int,
    manifest_bytes: int,
    manifest_hash: int,
    transaction_id: str,
    publication_generation: str,
) -> int:
    value = FNV_OFFSET

    def mix_u64(item: int) -> None:
        nonlocal value
        for byte in range(8):
            value ^= (item >> (byte * 8)) & 0xFF
            value = (value * FNV_PRIME) & U64_MASK

    def mix_text(item: str) -> None:
        nonlocal value
        for byte in item.encode("utf-8"):
            value ^= byte
            value = (value * FNV_PRIME) & U64_MASK
        value ^= 0xFF
        value = (value * FNV_PRIME) & U64_MASK

    mix_u64(previous)
    mix_u64(sequence)
    mix_u64(manifest_bytes)
    mix_u64(manifest_hash)
    mix_text(transaction_id)
    mix_text(publication_generation)
    return value


def qualify_trajectories(
    path: Path, commit_times_s: list[float]
) -> dict[str, float | int]:
    (parcel_count, sample_capacity, column_count), values = load_trajectory_npy(path)
    require(parcel_count > 0 and sample_capacity > 0, "trajectory output is empty")
    crossings = [0] * len(commit_times_s)
    maximum_crossing_speed = 0.0
    maximum_all_step_speed = 0.0
    total_samples = 0

    for parcel in range(parcel_count):
        rows: list[array.array] = []
        padding_started = False
        for sample in range(sample_capacity):
            base = (parcel * sample_capacity + sample) * column_count
            row = values[base : base + column_count]
            if math.isnan(row[0]):
                padding_started = True
                require(all(math.isnan(value) for value in row),
                        "partial trajectory padding row")
                continue
            require(not padding_started, "finite trajectory sample follows NaN padding")
            require(all(math.isfinite(value) for value in row),
                    "non-finite populated trajectory sample")
            rows.append(row)
        total_samples += len(rows)

        for previous, current in zip(rows, rows[1:]):
            dt = float(current[0] - previous[0])
            require(dt > 0.0, "trajectory time is not strictly increasing")
            speed = math.hypot(current[1] - previous[1], current[2] - previous[2]) / dt
            maximum_all_step_speed = max(maximum_all_step_speed, speed)
            for index, commit_time_s in enumerate(commit_times_s):
                if previous[0] <= commit_time_s < current[0]:
                    crossings[index] += 1
                    maximum_crossing_speed = max(maximum_crossing_speed, speed)

    require(total_samples > parcel_count, "trajectory paths did not advance")
    require(all(count > 0 for count in crossings),
            f"a D5 transaction has no surviving trajectory crossing: {crossings}")
    require(maximum_crossing_speed <= MAXIMUM_TRAJECTORY_HORIZONTAL_SPEED_MS,
            "Earth-frame trajectory discontinuity across sustained D5 campaign: "
            f"{maximum_crossing_speed:.6g} m/s")
    return {
        "parcel_count": parcel_count,
        "total_samples": total_samples,
        "transaction_crossings": sum(crossings),
        "minimum_crossings_per_transaction": min(crossings),
        "maximum_crossing_speed_ms": maximum_crossing_speed,
        "maximum_all_step_speed_ms": maximum_all_step_speed,
    }


def qualify_artifacts(event_directory: Path, event: dict) -> tuple[int, int]:
    artifacts = event.get("artifacts")
    expected_paths = {"tile_00/manifest.json"}
    expected_paths.update(
        f"tile_00/{field}{suffix}"
        for field in EXPECTED_FIELDS
        for suffix in (".npy", ".bricks.json")
    )
    require(isinstance(artifacts, list),
            "sustained swath artifacts are not a list")
    actual_paths = {
        artifact.get("path") for artifact in artifacts
        if isinstance(artifact, dict)
    }
    require(actual_paths == expected_paths,
            "wrong sustained swath artifact set")
    field_names: set[str] = set()
    brick_fields: set[str] = set()
    total_bytes = 0
    for artifact in artifacts:
        require(isinstance(artifact, dict), "swath artifact entry is not an object")
        relative = artifact.get("path")
        require(isinstance(relative, str) and relative.startswith("tile_00/"),
                "invalid sustained swath artifact path")
        artifact_path = event_directory / relative
        require(artifact_path.is_file(), f"missing swath artifact: {relative}")
        expected_bytes = int(artifact.get("bytes", -1))
        require(artifact_path.stat().st_size == expected_bytes,
                f"swath artifact size mismatch: {relative}")
        require(fnv1a64(artifact_path) == artifact.get("fnv1a64"),
                f"swath artifact hash mismatch: {relative}")
        total_bytes += expected_bytes
        if artifact_path.suffix == ".npy":
            field_names.add(artifact_path.stem)
        elif artifact_path.name.endswith(".bricks.json"):
            hierarchy = load_json(artifact_path)
            field = artifact_path.name.removesuffix(".bricks.json")
            require(field in EXPECTED_FIELDS,
                    f"wrong sustained swath brick field: {field}")
            require_swath_brick_hierarchy(hierarchy, field, relative)
            brick_fields.add(field)
    require(field_names == EXPECTED_FIELDS, f"wrong swath fields: {sorted(field_names)}")
    require(brick_fields == EXPECTED_FIELDS,
            f"wrong sustained swath brick fields: {sorted(brick_fields)}")
    return len(artifacts), total_bytes


def qualify_campaign(output: Path) -> dict:
    tracking_paths = sorted((output / "storm_track").glob("track_step_*.json"))
    require(tracking_paths, "no sustained storm-tracking records were published")
    records = [load_json(path) for path in tracking_paths]
    require(all(record.get("schema") == "nimbus.storm_feature" for record in records),
            "wrong tracker schema")
    require(all(record.get("schema_version") == 12 for record in records),
            "wrong tracker schema version")
    require(all(record.get("tracking_source") == "root_grid" for record in records),
            "wrong tracker source")
    commits = [record for record in records if record.get("state_mutation") is True]
    require(len(commits) >= MINIMUM_COMMIT_COUNT,
            f"expected at least {MINIMUM_COMMIT_COUNT} transactions, found {len(commits)}")

    event_paths = sorted(
        (output / "domain_recycling_swath").glob("transaction_*/event.json"),
        key=lambda path: int(load_json(path).get("transaction_sequence", 0)),
    )
    require(len(event_paths) == len(commits),
            "tracker commits and durable swath events disagree")

    previous_chain = 0
    cumulative_i = 0
    cumulative_j = 0
    publication_generation: str | None = None
    previous_time = -math.inf
    previous_lower_x = -math.inf
    previous_upper_x = -math.inf
    total_artifacts = 0
    total_artifact_bytes = 0
    final_manifest_bytes = 0
    final_manifest_hash = 0

    for expected_sequence, (record, event_path) in enumerate(
        zip(commits, event_paths), start=1
    ):
        event_directory = event_path.parent
        event = load_json(event_path)
        authority = record.get("movement_authority", {})
        transaction_id = authority.get("transaction_id")
        generation = event.get("publication_generation")
        require(event.get("schema") == "nimbus.domain_recycling_swath",
                "wrong swath schema")
        require(event.get("schema_version") == 1, "wrong swath schema version")
        require(event.get("publication_phase") == "captured_precommit",
                "wrong swath capture phase")
        require(event.get("transaction_sequence") == expected_sequence,
                "non-monotone swath transaction sequence")
        require(event.get("transaction_id") == transaction_id,
                "tracker/swath transaction linkage mismatch")
        require(isinstance(transaction_id, str) and transaction_id.startswith("d5_"),
                "invalid sustained transaction id")
        require(isinstance(generation, str) and generation,
                "missing swath publication generation")
        if publication_generation is None:
            publication_generation = generation
        require(generation == publication_generation,
                "continuous campaign changed publication generation")

        commit_time_s = float(record.get("simulation_time_s", -1.0))
        require(commit_time_s > previous_time and commit_time_s < CONTINUOUS_DURATION_S,
                "transaction times are not strictly increasing before checkpoint")
        require(float(event.get("simulation_time_s", -1.0)) == commit_time_s,
                "swath simulation time does not match tracker commit")
        previous_time = commit_time_s

        shift = event.get("shift_cells")
        require(isinstance(shift, list) and len(shift) == 2,
                "invalid swath shift")
        shift_i, shift_j = int(shift[0]), int(shift[1])
        require((shift_i, shift_j) == pair(
            authority.get("committed_shift_cells"), "i", "j", "committed shift"),
            "tracker/swath shift mismatch")
        require(shift_i > 0 and shift_j == 0,
                "sustained fixture did not move along its qualified eastward path")
        cumulative_i += shift_i
        cumulative_j += shift_j
        require(pair(authority.get("cumulative_committed_shift_cells"),
                     "i", "j", "cumulative shift") == (cumulative_i, cumulative_j),
                "wrong cumulative sustained shift")
        require(pair(authority.get("post_shift_grid_origin_t0_earth_m"),
                     "x", "y", "post-shift origin")
                == (cumulative_i * DX_M, cumulative_j * DY_M),
                "wrong post-shift Earth origin")

        pre_origin = event.get("pre_shift_grid_origin_earth_m")
        require(isinstance(pre_origin, list) and len(pre_origin) == 2,
                "missing pre-shift Earth origin")
        expected_pre_t0_x = (cumulative_i - shift_i) * DX_M
        expected_pre_t0_y = (cumulative_j - shift_j) * DY_M
        require(tuple(float(value) for value in pre_origin) == (
            expected_pre_t0_x + FRAME_VELOCITY_MS[0] * commit_time_s,
            expected_pre_t0_y + FRAME_VELOCITY_MS[1] * commit_time_s,
        ), "swath pre-shift Earth origin is inconsistent")

        tiles = event.get("tiles")
        require(isinstance(tiles, list) and len(tiles) == 1,
                "eastward sustained transaction must publish one exact tile")
        tile = tiles[0]
        bounds = tile.get("earth_bounds_m", {})
        lower = bounds.get("lower")
        upper = bounds.get("upper")
        source_bounds = tile.get("source_index_bounds", {})
        source_i = source_bounds.get("i")
        source_j = source_bounds.get("j")
        require(all(isinstance(value, list) and len(value) == 2
                    for value in (lower, upper, source_i, source_j)),
                "incomplete sustained tile geometry")
        require(tuple(float(value) for value in lower) == (
            float(pre_origin[0]) + int(source_i[0]) * DX_M,
            float(pre_origin[1]) + int(source_j[0]) * DY_M,
        ), "swath lower Earth bound is inconsistent")
        require(tuple(float(value) for value in upper) == (
            float(pre_origin[0]) + int(source_i[1]) * DX_M,
            float(pre_origin[1]) + int(source_j[1]) * DY_M,
        ), "swath upper Earth bound is inconsistent")
        require(float(lower[0]) > previous_lower_x and float(upper[0]) > previous_upper_x,
                "successive retiring swaths did not progress in Earth x")
        previous_lower_x = float(lower[0])
        previous_upper_x = float(upper[0])

        artifact_count, artifact_bytes = qualify_artifacts(event_directory, event)
        total_artifacts += artifact_count
        total_artifact_bytes += artifact_bytes

        manifest_bytes = event_path.stat().st_size
        manifest_hash = int(fnv1a64(event_path), 16)
        require(parse_hex_u64(event.get("previous_committed_chain_fnv1a64"),
                              "previous committed chain") == previous_chain,
                "swath does not extend the preceding committed chain")
        previous_chain = transaction_chain(
            previous_chain, expected_sequence, manifest_bytes, manifest_hash,
            transaction_id, generation,
        )
        final_manifest_bytes = manifest_bytes
        final_manifest_hash = manifest_hash

    final = records[-1]
    require(float(final.get("simulation_time_s", -1.0)) == CONTINUOUS_DURATION_S,
            "continuous campaign did not reach its checkpoint time")
    require(pair(final.get("horizontal_frame", {}).get("grid_origin_t0_earth_m"),
                 "x", "y", "final frame origin")
            == (cumulative_i * DX_M, cumulative_j * DY_M),
            "continuous campaign lost its final Earth frame")
    require(pair(final.get("movement_authority", {}).get(
                     "cumulative_committed_shift_cells"),
                 "i", "j", "final cumulative shift")
            == (cumulative_i, cumulative_j),
            "continuous campaign lost its cumulative shift")

    return {
        "commit_count": len(commits),
        "commit_times_s": [float(record["simulation_time_s"]) for record in commits],
        "transaction_ids": [record["movement_authority"]["transaction_id"] for record in commits],
        "publication_generation": publication_generation,
        "cumulative_shift": (cumulative_i, cumulative_j),
        "chain_head": previous_chain,
        "final_manifest_bytes": final_manifest_bytes,
        "final_manifest_hash": final_manifest_hash,
        "artifact_count": total_artifacts,
        "artifact_bytes": total_artifact_bytes,
        "tracking_records": len(records),
    }


def qualify_checkpoint(
    sidecar: dict, campaign: dict, context: str, checkpoint_time_s: float
) -> None:
    require(sidecar.get("completed_steps") == int(checkpoint_time_s / DT_S),
            f"wrong {context} checkpoint step")
    require(float(sidecar.get("simulation_time_s", -1.0)) == checkpoint_time_s,
            f"wrong {context} checkpoint time")
    provenance = sidecar.get("domain_recycling")
    require(isinstance(provenance, dict), f"{context} checkpoint lacks D5 provenance")
    require(provenance.get("algorithm")
            == "checkpoint_bound_d5_transaction_provenance_v1",
            f"wrong {context} provenance algorithm")
    require(provenance.get("sequence") == campaign["commit_count"],
            f"wrong {context} provenance sequence")
    require(provenance.get("transaction_id") == campaign["transaction_ids"][-1],
            f"wrong {context} final transaction id")
    require(provenance.get("publication_generation")
            == campaign["publication_generation"],
            f"wrong {context} publication generation")
    require(provenance.get("event_manifest_bytes")
            == campaign["final_manifest_bytes"],
            f"wrong {context} event-manifest byte count")
    require(parse_hex_u64(provenance.get("event_manifest_fnv1a64"),
                          f"{context} event hash") == campaign["final_manifest_hash"],
            f"wrong {context} event-manifest hash")
    require(parse_hex_u64(provenance.get("committed_chain_fnv1a64"),
                          f"{context} chain head") == campaign["chain_head"],
            f"wrong {context} committed chain head")
    require(provenance.get("control_volume_exchange_complete") is True,
            f"{context} control-volume exchange is incomplete")
    require(provenance.get("control_volume_exchange_basis")
            == "instantaneous_fixed_earth_control_volume_exchange_v1",
            f"wrong {context} exchange basis")


def run_continuous(
    binary: Path,
    root: Path,
    compute_backend: str,
    required_backend: str | None,
) -> tuple[Path, Path]:
    output = root / "continuous"
    checkpoint_directory = root / "checkpoints"
    command = [
        str(binary), "--headless", f"--config={CONFIG}",
        f"--outdir={output}", f"--compute-backend={compute_backend}",
        "--threads=1", f"--duration={CONTINUOUS_DURATION_S}",
        "--write-every=0",
        f"--log-profile={'normal' if required_backend else 'quiet'}",
        "--no-timing", "--trajectories", "--trajectory-seed-time=0",
        "--trajectory-count=3", f"--checkpoint-every={CONTINUOUS_DURATION_S}",
        f"--checkpoint-dir={checkpoint_directory}",
    ]
    run_command(command, "sustained live D5 run", compute_backend, required_backend)
    checkpoints = sorted(checkpoint_directory.glob("checkpoint_step_*.nimbus"))
    require(len(checkpoints) == 1,
            f"expected one sustained D5 checkpoint, found {len(checkpoints)}")
    return output, checkpoints[0]


def run_resumed(
    binary: Path,
    root: Path,
    checkpoint: Path,
    compute_backend: str,
    required_backend: str | None,
) -> tuple[Path, Path]:
    output = root / "resumed"
    checkpoint_directory = root / "resumed_checkpoints"
    command = [
        str(binary), "--headless", f"--config={CONFIG}",
        f"--outdir={output}", f"--compute-backend={compute_backend}",
        "--threads=1", f"--duration={RESUMED_DURATION_S}",
        "--write-every=0",
        f"--log-profile={'normal' if required_backend else 'quiet'}",
        "--no-timing", "--checkpoint-every=10",
        f"--checkpoint-dir={checkpoint_directory}", f"--resume={checkpoint}",
    ]
    run_command(command, "post-final-event sustained D5 restart",
                compute_backend, required_backend)
    checkpoints = sorted(checkpoint_directory.glob("checkpoint_step_*.nimbus"))
    require(len(checkpoints) == 1,
            f"expected one resumed D5 checkpoint, found {len(checkpoints)}")
    return output, checkpoints[0]


def main() -> int:
    global CONTINUOUS_DURATION_S, RESUMED_DURATION_S
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output", type=Path, help="retain evidence in this directory")
    parser.add_argument(
        "--compute-backend", choices=("cpu", "vulkan", "metal"), default="cpu")
    required = parser.add_mutually_exclusive_group()
    required.add_argument("--require-vulkan", action="store_true",
                          help="fail unless every branch retains a physical Vulkan device")
    required.add_argument("--require-metal", action="store_true",
                          help="fail unless every branch retains a physical Metal device")
    parser.add_argument(
        "--continuous-duration", type=int, default=CONTINUOUS_DURATION_S,
        help="continuous campaign and checkpoint time in seconds")
    parser.add_argument(
        "--resumed-duration", type=int, default=RESUMED_DURATION_S,
        help="post-checkpoint target time in seconds")
    args = parser.parse_args()
    require(args.continuous_duration > 0,
            "--continuous-duration must be positive")
    require(args.resumed_duration > args.continuous_duration,
            "--resumed-duration must exceed --continuous-duration")
    CONTINUOUS_DURATION_S = args.continuous_duration
    RESUMED_DURATION_S = args.resumed_duration
    binary = args.binary.resolve()
    require(binary.is_file(), f"missing production binary: {binary}")
    require(CONFIG.is_file(), f"missing qualification config: {CONFIG}")
    required_backend = (
        "vulkan" if args.require_vulkan else "metal" if args.require_metal else None
    )
    require(required_backend is None or args.compute_backend == required_backend,
            f"--require-{required_backend} requires "
            f"--compute-backend={required_backend}")

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.output is None:
        temporary = tempfile.TemporaryDirectory(prefix="nimbus_domain_recycling_sustained_")
        output = Path(temporary.name)
    else:
        output = args.output.resolve()
        require(not output.exists() or not any(output.iterdir()),
                f"output directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)

    try:
        continuous_output, checkpoint = run_continuous(
            binary, output, args.compute_backend, required_backend)
        campaign = qualify_campaign(continuous_output)
        trajectories = qualify_trajectories(
            continuous_output / "trajectories.npy", campaign["commit_times_s"])
        first_sidecar = load_json(Path(f"{checkpoint}.json"))
        qualify_checkpoint(
            first_sidecar, campaign, "continuous", CONTINUOUS_DURATION_S)

        resumed_output, resumed_checkpoint = run_resumed(
            binary, output, checkpoint, args.compute_backend, required_backend)
        branch = load_json(resumed_output / "nimbus_branch.json")
        require(branch.get("parent_run_id") == first_sidecar.get("run_id"),
                "sustained D5 branch parent mismatch")
        require(branch.get("resumed_step") == int(CONTINUOUS_DURATION_S / DT_S),
                "wrong sustained D5 resumed step")
        require(float(branch.get("resumed_simulation_time_s", -1.0))
                == CONTINUOUS_DURATION_S, "wrong sustained D5 resumed time")
        latest = load_json(resumed_output / "storm_track_latest.json")
        require(float(latest.get("simulation_time_s", -1.0)) == RESUMED_DURATION_S,
                "sustained D5 restart did not reach its target time")
        require(pair(latest.get("horizontal_frame", {}).get("grid_origin_t0_earth_m"),
                     "x", "y", "resumed frame origin")
                == tuple(value * spacing for value, spacing in zip(
                    campaign["cumulative_shift"], (DX_M, DY_M))),
                "sustained D5 restart lost its Earth frame")
        require(pair(latest.get("movement_authority", {}).get(
                         "cumulative_committed_shift_cells"),
                     "i", "j", "resumed cumulative shift")
                == campaign["cumulative_shift"],
                "sustained D5 restart lost its cumulative movement history")
        resumed_sidecar = load_json(Path(f"{resumed_checkpoint}.json"))
        qualify_checkpoint(
            resumed_sidecar, campaign, "resumed", RESUMED_DURATION_S)

        print(
            f"[domain-recycling-sustained] PASS ({args.compute_backend}): "
            f"{campaign['commit_count']} commits at {campaign['commit_times_s']} s; "
            f"cumulative shift {campaign['cumulative_shift']}; "
            f"{campaign['artifact_count']} hash-verified artifacts; "
            f"chain head {campaign['chain_head']:016x}; "
            f"minimum {trajectories['minimum_crossings_per_transaction']} "
            f"trajectory crossings/transaction; maximum crossing speed "
            f"{trajectories['maximum_crossing_speed_ms']:.3f} m/s; "
            f"restart {CONTINUOUS_DURATION_S:g}->{RESUMED_DURATION_S:g} s"
        )
    finally:
        if temporary is not None:
            temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
