#!/usr/bin/env python3
"""Seal draft reversible-radar v4 wire vectors without changing Nimbus readers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess


ROOT = Path(__file__).resolve().parents[3]
FIELDS = ("reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv")
STEPS = ("step_000000", "step_000001")
HEADER = struct.Struct("<4sHHiiidQB3s4s")
MAX_BRICK_BYTES = 4 * 1024 * 1024


class WireError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def write_wire(payloads: list[bytes]) -> bytes:
    payload_size = sum(len(payload) for payload in payloads)
    header = HEADER.pack(b"ZFP3", 4, 3, 12, 12, 48, 0.0,
                         payload_size, 0, b"\0\0\0", b"ZB4D")
    offsets = []
    position = HEADER.size
    for payload in payloads:
        offsets.append(position)
        position += len(payload)
    directory = b"".join(
        struct.pack("<Q", offset) + digest(payload)
        for offset, payload in zip(offsets, payloads)
    )
    return header + b"".join(payloads) + directory


def validate_wire(data: bytes) -> list[dict[str, object]]:
    if len(data) < HEADER.size:
        raise WireError("truncated_header")
    magic, version, mode, d0, d1, d2, parameter, payload_size, flags, reserved, brick_magic = \
        HEADER.unpack_from(data)
    if magic != b"ZFP3":
        raise WireError("bad_magic")
    if version != 4:
        raise WireError("bad_version")
    if mode != 3:
        raise WireError("bad_mode")
    if (d0, d1, d2) != (12, 12, 48):
        raise WireError("bad_exact_fixture_shape")
    if struct.pack("<d", parameter) != b"\0" * 8:
        raise WireError("noncanonical_parameter")
    if flags != 0:
        raise WireError("delta_flag")
    if reserved != b"\0" * 3:
        raise WireError("reserved")
    if brick_magic != b"ZB4D":
        raise WireError("brick_magic")
    directory_start = HEADER.size + payload_size
    expected_size = directory_start + 2 * 40
    if len(data) != expected_size:
        raise WireError("file_size")
    offsets = [struct.unpack_from("<Q", data, directory_start + 40 * index)[0]
               for index in range(2)]
    entries = []
    next_expected = HEADER.size
    for index, offset in enumerate(offsets):
        end = offsets[index + 1] if index + 1 < 2 else directory_start
        if offset != next_expected or end <= offset or end > directory_start:
            raise WireError("directory")
        if end - offset > MAX_BRICK_BYTES:
            raise WireError("oversized_brick")
        payload = data[offset:end]
        expected_digest = data[directory_start + 40 * index + 8:
                               directory_start + 40 * (index + 1)]
        if digest(payload) != expected_digest:
            raise WireError("payload_digest")
        entries.append({"offset": offset, "bytes": len(payload),
                        "payloadSha256": hashlib.sha256(payload).hexdigest()})
        next_expected = end
    if next_expected != directory_start:
        raise WireError("payload_total")
    return entries


def replaced(data: bytes, offset: int, value: bytes) -> bytes:
    return data[:offset] + value + data[offset + len(value):]


def negative_vectors(good: bytes) -> dict[str, bytes]:
    directory_start = len(good) - 80
    first_offset = HEADER.size
    oversized = write_wire([b"\0" * (MAX_BRICK_BYTES + 1), b"\0"])
    return {
        "bad_version": replaced(good, 4, struct.pack("<H", 3)),
        "bad_mode": replaced(good, 6, struct.pack("<H", 0)),
        "noncanonical_parameter": replaced(good, 20, struct.pack("<d", 1.0)),
        "delta_flag": replaced(good, 36, b"\x01"),
        "reserved": replaced(good, 37, b"\x01"),
        "brick_magic": replaced(good, 40, b"ZB3D"),
        "truncated_header": good[:20],
        "truncated_directory": good[:-1],
        "payload_total_mismatch": replaced(
            good, 28, struct.pack("<Q", len(good) - 80 - HEADER.size + 1)),
        "noncanonical_offset": replaced(
            good, directory_start, struct.pack("<Q", first_offset + 1)),
        "duplicate_offset": replaced(
            good, directory_start + 40, struct.pack("<Q", first_offset)),
        "digest_flip": replaced(
            good, directory_start + 8, bytes([good[directory_start + 8] ^ 1])),
        "payload_flip": replaced(good, HEADER.size, bytes([good[HEADER.size] ^ 1])),
        "oversized_brick": oversized,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototype", type=Path, required=True)
    parser.add_argument("--native-rejection-helper", type=Path, required=True)
    parser.add_argument("--npy-run", type=Path, required=True)
    parser.add_argument("--input-provenance-report", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    prototype = args.prototype.resolve(strict=True)
    native_reject = args.native_rejection_helper.resolve(strict=True)
    npy_run = args.npy_run.resolve(strict=True)
    provenance_path = args.input_provenance_report.resolve(strict=True)
    provenance = json.loads(provenance_path.read_text())
    if (provenance["npyRun"] != str(npy_run) or
            provenance["status"] != "fail_decoded_science_contract"):
        raise RuntimeError("draft wire inputs are not the sealed active red case")
    if args.report_output.exists():
        parser.error("use a new report path")
    raw = args.raw_root.resolve()
    raw.mkdir(parents=True, exist_ok=False)
    rows = []
    reference_wire = None
    for step in STEPS:
        for field in FIELDS:
            npy = npy_run / step / f"{field}.npy"
            directory = raw / step / f"{field}_bricks"
            emitted = subprocess.run(
                [str(prototype), "--emit-bricks", field, step,
                 str(npy), str(directory)], cwd=ROOT,
                capture_output=True, text=True, timeout=30, check=True)
            result = json.loads(emitted.stdout)
            if not result["byteIdentical"] or result["brickCount"] != 2:
                raise RuntimeError("draft source payload did not reconstruct exactly")
            payloads = [(directory / f"brick_{index}.bin").read_bytes()
                        for index in range(2)]
            wire = write_wire(payloads)
            entries = validate_wire(wire)
            path = raw / step / f"{field}.zfp3d"
            path.write_bytes(wire)
            subprocess.run([str(native_reject), str(path)], cwd=ROOT,
                           text=True, capture_output=True, timeout=30, check=True)
            if step == STEPS[-1] and field == "rhohv":
                reference_wire = wire
            rows.append({
                "field": field, "step": step,
                "npySha256": sha256(npy), "draftWireSha256": sha256(path),
                "draftWireBytes": len(wire), "payloadBytes": result["payloadBytes"],
                "brickEntries": entries, "currentNativeReaderRejectedVersion4": True,
                "proposedManifestCompression": {
                    "contract_version": 2, "codec": "zfp", "mode": "reversible",
                    "mode_parameter": 0, "mode_parameter_units": "none",
                    "error_contract": "bitwise_float32_reconstruction",
                    "container_version": 4, "frame_role": "keyframe",
                    "parent_step_indices": [], "sparse_threshold": 0,
                    "float16_prefilter": False, "decoded_dtype": "float32",
                    "spatial_layout": "independently_decodable_bricks",
                    "brick_shape": [64, 64, 32],
                },
            })
    if reference_wire is None:
        raise RuntimeError("no active rhohv draft wire reference")
    negative_dir = raw / "corruptions"
    negative_dir.mkdir()
    negatives = []
    for label, variant in negative_vectors(reference_wire).items():
        path = negative_dir / f"{label}.zfp3d"
        path.write_bytes(variant)
        try:
            validate_wire(variant)
        except WireError as error:
            negatives.append({"name": label, "rejection": str(error),
                              "sha256": sha256(path), "bytes": len(variant)})
        else:
            raise RuntimeError(f"draft wire corruption was accepted: {label}")
    report = {
        "schema": "nimbus.r4-radar-zfp-draft-v4-wire.v1",
        "status": "pass_draft_wire_vectors_current_reader_fail_closed",
        "claimBoundary": "Test-only proposed v4 wire vectors for ten selected radar field-frames and fourteen structural/payload corruptions. Detached parser validates proposed bytes/digests; current native reader rejects version 4. No production writer or reader accepts this ABI yet, and Rust reader permission is pending.",
        "inputGeneratorBinarySha256": provenance["binarySha256"],
        "inputProvenanceReportSha256": sha256(provenance_path),
        "currentGuardBinarySha256": sha256(ROOT / "bin/nimbus_sim"),
        "prototypeSha256": sha256(prototype),
        "currentNativeRejectionHelperSha256": sha256(native_reject),
        "wireDriverSha256": sha256(Path(__file__)),
        "rawRoot": str(raw), "positiveVectors": rows,
        "corruptionVectors": negatives,
        "totalDraftWireBytes": sum(row["draftWireBytes"] for row in rows),
    }
    output = args.report_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "positiveVectors": len(rows),
                      "corruptionVectors": len(negatives),
                      "totalDraftWireBytes": report["totalDraftWireBytes"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
