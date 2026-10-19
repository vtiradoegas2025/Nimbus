#!/usr/bin/env python3
"""Paired exact-byte benchmark for predictive-ZFP history construction."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import time
from typing import Any


OUTPUT_SUMMARY = re.compile(
    r"\[OUTPUT\]\s*(?P<snapshots>\d+) snapshots written,\s*"
    r"(?P<mb>[-+0-9.eE]+) MB total,\s*"
    r"(?P<seconds>[-+0-9.eE]+)s write time"
)
ASSEMBLY_SUMMARY = re.compile(
    r"\[OUTPUT\]\s*Snapshot assembly:\s*"
    r"frames=(?P<frames>\d+),\s*"
    r"total_s=(?P<total>[-+0-9.eE]+),\s*"
    r"max_s=(?P<maximum>[-+0-9.eE]+),\s*"
    r"max_bytes=(?P<bytes>\d+)"
)
MEMORY_MILESTONE = re.compile(
    r"\[MEMORY MILESTONE\].*?peak_rss_bytes=(?P<bytes>\d+)"
)
WORKSPACE_SUMMARY = re.compile(
    r"\[OUTPUT\]\s*ZFP codec workspace:\s*"
    r"initializations=(?P<initializations>\d+),\s*"
    r"bitstream_bindings=(?P<bindings>\d+),\s*"
    r"bitstream_capacity_bytes=(?P<capacity>\d+)"
)
PHASE_SUMMARY = re.compile(
    r"\[OUTPUT\]\s*ZFP instrumented phases:\s*"
    r"control_s=(?P<control>[-+0-9.eE]+),\s*"
    r"codec_s=(?P<codec>[-+0-9.eE]+),\s*"
    r"container_io_s=(?P<container_io>[-+0-9.eE]+)"
)
CODEC_PROVENANCE = re.compile(
    r"\[OUTPUT\]\s*ZFP codec provenance:\s*"
    r"version=(?P<version>[^\r\n]+?),\s*"
    r"execution_policy=(?P<policy>[a-z_]+)"
)
CLEARED_ENV_PREFIXES = ("NIMBUS_", "OMP_", "KMP_", "GOMP_")


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def finite_nonnegative(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return value


def robust_stats(samples: list[float]) -> dict[str, Any]:
    median = statistics.median(samples)
    mad = statistics.median(abs(value - median) for value in samples)
    return {
        "samples": samples,
        "median": median,
        "median_absolute_deviation": mad,
        "three_mad": 3.0 * mad,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_status(
    timing_pass: bool,
    deterministic_memory_bytes: int,
    accept_memory_only: bool,
    lifecycle_pass: bool = False,
    accept_lifecycle_only: bool = False,
) -> tuple[str, bool]:
    if timing_pass:
        return "accepted", True
    if accept_memory_only and deterministic_memory_bytes > 0:
        return "accepted_memory_only", True
    if accept_lifecycle_only and lifecycle_pass:
        return "accepted_lifecycle_only", True
    return "rejected", False


def workspace_lifecycle_pair_pass(pair: dict[str, Any]) -> bool:
    optimized = pair["optimized"]
    comparator = pair["comparator"]
    optimized_workspace = optimized.get("zfp_codec_workspace")
    comparator_workspace = comparator.get("zfp_codec_workspace")
    optimized_phases = optimized.get("zfp_instrumented_phases")
    comparator_phases = comparator.get("zfp_instrumented_phases")
    optimized_provenance = optimized.get("zfp_codec_provenance")
    comparator_provenance = comparator.get("zfp_codec_provenance")
    if not all((optimized_workspace, comparator_workspace,
                optimized_phases, comparator_phases,
                optimized_provenance, comparator_provenance)):
        return False

    frame_types = optimized["history"]["zfp_frame_types"]
    field_writes = sum(frame_types.values())
    if field_writes <= 0:
        return False
    if optimized_workspace["bitstream_capacity_bytes"] <= 0:
        return False
    if (optimized_workspace["bitstream_capacity_bytes"]
            != comparator_workspace["bitstream_capacity_bytes"]):
        return False
    if any(value <= 0.0 for value in optimized_phases.values()):
        return False
    if any(value <= 0.0 for value in comparator_phases.values()):
        return False
    if optimized_provenance != comparator_provenance:
        return False
    if optimized_provenance["execution_policy"] != "serial":
        return False
    return (
        optimized_workspace["initializations"] == 1
        and optimized_workspace["bitstream_bindings"] == 1
        and comparator_workspace["initializations"] == field_writes
        and comparator_workspace["bitstream_bindings"] == 2 * field_writes
    )


def history_digest(root: Path) -> dict[str, Any]:
    normalized_digest = hashlib.sha256()
    payload_digest = hashlib.sha256()
    files = []
    zfp_flags: dict[str, int] = {"keyframe": 0, "simple": 0, "predictive": 0}
    total_bytes = 0
    for path in sorted(root.glob("step_*/*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        payload = path.read_bytes()
        normalized_payload = payload
        if path.suffix == ".json":
            document = json.loads(payload)
            if isinstance(document, dict):
                # This identifier deliberately changes for every atomic run
                # publication.  It is provenance, not field/history content.
                document.pop("publication_generation", None)
            normalized_payload = json.dumps(
                document, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        normalized_digest.update(relative.encode("utf-8"))
        normalized_digest.update(b"\0")
        normalized_digest.update(normalized_payload)
        files.append(relative)
        total_bytes += len(payload)
        if path.suffix == ".zfp3d":
            payload_digest.update(relative.encode("utf-8"))
            payload_digest.update(b"\0")
            payload_digest.update(payload)
            if len(payload) < 40 or payload[:4] != b"ZFP3":
                raise RuntimeError(f"malformed ZFP payload: {relative}")
            flags = payload[36]
            if flags == 0:
                zfp_flags["keyframe"] += 1
            elif flags == 1:
                zfp_flags["simple"] += 1
            elif flags == 3:
                zfp_flags["predictive"] += 1
            else:
                raise RuntimeError(f"unknown ZFP delta flags {flags}: {relative}")
    if not files:
        raise RuntimeError("run published no history files")
    if zfp_flags["keyframe"] == 0 or zfp_flags["simple"] == 0 \
            or zfp_flags["predictive"] == 0:
        raise RuntimeError(f"non-discriminating ZFP sequence: {zfp_flags}")
    return {
        "normalized_tree_sha256": normalized_digest.hexdigest(),
        "zfp_payload_sha256": payload_digest.hexdigest(),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "zfp_frame_types": zfp_flags,
    }


def retain_keyframe_fixture(root: Path, destination: Path) -> str:
    """Retain one real writer keyframe for the independent Rust decoder."""
    for candidate in sorted(root.glob("step_*/*.zfp3d")):
        payload = candidate.read_bytes()
        if len(payload) >= 40 and payload[:4] == b"ZFP3" and payload[36] == 0:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            return hashlib.sha256(payload).hexdigest()
    raise RuntimeError("run published no ZFP keyframe for cross-reader qualification")


def clean_env(threads: int) -> dict[str, str]:
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(CLEARED_ENV_PREFIXES)
    }
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_DYNAMIC"] = "FALSE"
    return env


def run_one(
    label: str,
    binary: Path,
    config: Path,
    duration: int,
    threads: int,
    fixture_output: Path | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"nimbus-zfp-{label}-") as temp:
        output = Path(temp) / "output"
        command = [
            str(binary),
            "--headless",
            f"--config={config}",
            f"--outdir={output}",
            f"--duration={duration}",
            "--write-every=1",
            f"--threads={threads}",
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=clean_env(threads),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            raise RuntimeError(
                f"{label} failed with code {completed.returncode}:\n"
                + "\n".join(completed.stdout.splitlines()[-80:])
            )
        summaries = list(OUTPUT_SUMMARY.finditer(completed.stdout))
        if not summaries:
            raise RuntimeError(f"{label}: missing output summary")
        summary = summaries[-1].groupdict()
        assemblies = list(ASSEMBLY_SUMMARY.finditer(completed.stdout))
        if not assemblies:
            raise RuntimeError(f"{label}: missing snapshot assembly summary")
        assembly_fields = assemblies[-1].groupdict()
        assembly = {
            "frames": int(assembly_fields["frames"]),
            "total_s": float(assembly_fields["total"]),
            "max_s": float(assembly_fields["maximum"]),
            "max_bytes": int(assembly_fields["bytes"]),
        }
        memory = [int(match.group("bytes"))
                  for match in MEMORY_MILESTONE.finditer(completed.stdout)]
        if not memory:
            raise RuntimeError(f"{label}: missing memory milestones")
        history = history_digest(output)
        fixture_sha256 = None
        if fixture_output is not None:
            fixture_sha256 = retain_keyframe_fixture(output, fixture_output)
        workspace_matches = list(WORKSPACE_SUMMARY.finditer(completed.stdout))
        workspace = None
        if workspace_matches:
            workspace_fields = workspace_matches[-1].groupdict()
            workspace = {
                "initializations": int(workspace_fields["initializations"]),
                "bitstream_bindings": int(workspace_fields["bindings"]),
                "bitstream_capacity_bytes": int(workspace_fields["capacity"]),
            }
        phase_matches = list(PHASE_SUMMARY.finditer(completed.stdout))
        phases = None
        if phase_matches:
            phase_fields = phase_matches[-1].groupdict()
            phases = {
                "control_s": float(phase_fields["control"]),
                "codec_s": float(phase_fields["codec"]),
                "container_io_s": float(phase_fields["container_io"]),
            }
        provenance_matches = list(
            CODEC_PROVENANCE.finditer(completed.stdout)
        )
        provenance = None
        if provenance_matches:
            provenance_fields = provenance_matches[-1].groupdict()
            provenance = {
                "version": provenance_fields["version"].strip(),
                "execution_policy": provenance_fields["policy"],
            }
        snapshots = int(summary["snapshots"])
        if snapshots < 3:
            raise RuntimeError(f"{label}: only {snapshots} snapshots")
        if assembly["frames"] != snapshots or assembly["max_bytes"] <= 0:
            raise RuntimeError(f"{label}: invalid snapshot assembly summary")
        return {
            "process_wall_s": elapsed,
            "writer_wall_s": float(summary["seconds"]),
            "snapshots": snapshots,
            "reported_megabytes": float(summary["mb"]),
            "snapshot_assembly": assembly,
            "peak_rss_bytes": max(memory),
            "zfp_codec_workspace": workspace,
            "zfp_instrumented_phases": phases,
            "zfp_codec_provenance": provenance,
            "history": history,
            "cross_reader_fixture_sha256": fixture_sha256,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--comparator", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trials", type=positive_int, default=7)
    parser.add_argument("--duration", type=positive_int, default=8)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument(
        "--experiment", default="predictive_history_working_set"
    )
    parser.add_argument(
        "--comparison-contract",
        default=(
            "same binary objects/config/threads, byte-identical ZFP payloads, "
            "and identical history metadata after excluding the intentionally "
            "run-unique publication_generation; comparator only recompresses "
            "decoded ZFP references and reallocates rotating history buffers"
        ),
    )
    parser.add_argument(
        "--deterministic-memory-reduction-bytes",
        type=int,
        default=0,
    )
    parser.add_argument("--accept-memory-only", action="store_true")
    parser.add_argument("--accept-lifecycle-only", action="store_true")
    parser.add_argument(
        "--minimum-speedup-percent", type=finite_nonnegative, default=2.0
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--zfp-fixture-output",
        type=Path,
        help=(
            "retain one optimized writer keyframe for independent reader "
            "qualification"
        ),
    )
    args = parser.parse_args()

    if args.deterministic_memory_reduction_bytes < 0:
        raise SystemExit("--deterministic-memory-reduction-bytes must be non-negative")

    for path in (args.optimized, args.comparator, args.config):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    variants = {
        "optimized": args.optimized.resolve(),
        "comparator": args.comparator.resolve(),
    }
    pairs: list[dict[str, Any]] = []
    reference_history: dict[str, Any] | None = None
    for pair_index in range(args.trials):
        order = ("optimized", "comparator") if pair_index % 2 == 0 \
            else ("comparator", "optimized")
        pair: dict[str, Any] = {"order": list(order)}
        for label in order:
            fixture_output = None
            if pair_index == 0 and label == "optimized":
                fixture_output = args.zfp_fixture_output
            result = run_one(
                label, variants[label], args.config.resolve(),
                args.duration, args.threads, fixture_output,
            )
            if reference_history is None:
                reference_history = result["history"]
            elif result["history"] != reference_history:
                raise RuntimeError(
                    f"{label}: published history is not byte-identical"
                )
            pair[label] = result
            print(
                f"[zfp-history] pair {pair_index + 1}/{args.trials} {label}: "
                f"writer={result['writer_wall_s']:.6f}s "
                f"process={result['process_wall_s']:.6f}s "
                f"rss={result['peak_rss_bytes']}"
            )
        pairs.append(pair)

    writer_improvements = [
        pair["comparator"]["writer_wall_s"]
        - pair["optimized"]["writer_wall_s"]
        for pair in pairs
    ]
    process_improvements = [
        pair["comparator"]["process_wall_s"]
        - pair["optimized"]["process_wall_s"]
        for pair in pairs
    ]
    peak_rss_reductions = [
        pair["comparator"]["peak_rss_bytes"]
        - pair["optimized"]["peak_rss_bytes"]
        for pair in pairs
    ]
    writer_stats = robust_stats(writer_improvements)
    process_stats = robust_stats(process_improvements)
    peak_rss_stats = robust_stats(peak_rss_reductions)
    comparator_writer_median = statistics.median(
        pair["comparator"]["writer_wall_s"] for pair in pairs
    )
    comparator_process_median = statistics.median(
        pair["comparator"]["process_wall_s"] for pair in pairs
    )
    comparator_peak_rss_median = statistics.median(
        pair["comparator"]["peak_rss_bytes"] for pair in pairs
    )
    minimum_writer = comparator_writer_median \
        * args.minimum_speedup_percent / 100.0
    minimum_process = comparator_process_median \
        * args.minimum_speedup_percent / 100.0
    minimum_peak_rss = comparator_peak_rss_median \
        * args.minimum_speedup_percent / 100.0
    writer_pass = writer_stats["median"] >= max(
        minimum_writer, writer_stats["three_mad"]
    )
    process_pass = process_stats["median"] >= max(
        minimum_process, process_stats["three_mad"]
    )
    peak_rss_observation_pass = peak_rss_stats["median"] >= max(
        minimum_peak_rss, peak_rss_stats["three_mad"]
    )
    lifecycle_pass = all(workspace_lifecycle_pair_pass(pair) for pair in pairs)
    # Process RSS is allocator/page-reuse sensitive at this short duration.
    # Keep it visible, but use the focused allocation-telemetry regression as
    # the deterministic memory gate for the repaired full-field buffers.
    passes_required_gates = writer_pass and process_pass
    status, repair_accepted = classify_status(
        passes_required_gates,
        args.deterministic_memory_reduction_bytes,
        args.accept_memory_only,
        lifecycle_pass,
        args.accept_lifecycle_only,
    )
    report = {
        "schema": "nimbus.zfp-history-working-set.v1",
        "experiment": args.experiment,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": status,
        "comparison_contract": args.comparison_contract,
        "trials": args.trials,
        "duration_s": args.duration,
        "threads": args.threads,
        "config": str(args.config),
        "inputs": {
            "optimized_binary": {
                "path": str(variants["optimized"]),
                "sha256": file_sha256(variants["optimized"]),
            },
            "comparator_binary": {
                "path": str(variants["comparator"]),
                "sha256": file_sha256(variants["comparator"]),
            },
            "config": {
                "path": str(args.config.resolve()),
                "sha256": file_sha256(args.config.resolve()),
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
        },
        "history": reference_history,
        "pairs": pairs,
        "writer_improvement_s": writer_stats,
        "process_improvement_s": process_stats,
        "peak_rss_reduction_bytes": peak_rss_stats,
        "deterministic_persistent_memory_reduction_bytes": (
            args.deterministic_memory_reduction_bytes
        ),
        "gate": {
            "minimum_speedup_percent": args.minimum_speedup_percent,
            "minimum_writer_improvement_s": minimum_writer,
            "minimum_process_improvement_s": minimum_process,
            "minimum_peak_rss_reduction_bytes": minimum_peak_rss,
            "writer_pass": writer_pass,
            "process_pass": process_pass,
            "peak_rss_observation_pass": peak_rss_observation_pass,
            "peak_rss_required": False,
            "passes_required_gates": passes_required_gates,
            "throughput_promoted": passes_required_gates,
            "memory_only_acceptance_enabled": args.accept_memory_only,
            "lifecycle_only_acceptance_enabled": args.accept_lifecycle_only,
            "codec_workspace_lifecycle_pass": lifecycle_pass,
            "repair_accepted": repair_accepted,
        },
    }
    if args.zfp_fixture_output is not None:
        fixture_path = args.zfp_fixture_output.resolve()
        if not fixture_path.is_file():
            raise RuntimeError("cross-reader ZFP fixture was not retained")
        report["inputs"]["cross_reader_fixture"] = {
            "path": str(fixture_path),
            "sha256": file_sha256(fixture_path),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"[zfp-history] status={status} "
        f"writer_gain={writer_stats['median']:.6f}s "
        f"process_gain={process_stats['median']:.6f}s "
        f"history={reference_history['zfp_payload_sha256'] if reference_history else 'none'}"
    )
    return 0 if repair_accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
