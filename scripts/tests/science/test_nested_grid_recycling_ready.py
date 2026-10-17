#!/usr/bin/env python3
"""Bounded public gate for flat nested-grid D5 recycling."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import re
from pathlib import Path
import subprocess
import struct
import sys
import tempfile

from test_two_way_nested_ready import require_closure, require_water_closure, require_feedback_output_budget, require_private_closure


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.qualification.checkpointing.qualify_vulkan_restart import (  # noqa: E402
    LEDGERS,
    require_restart_suffix,
)

CONFIG = (
    ROOT / "tests" / "fixtures" / "configs" / "qualification"
    / "nested_grid_recycling_ready.yaml"
)
DEFAULT_BINARY = ROOT / "bin" / "nimbus_sim"
FIELDS = ("p.npy", "qr.npy", "qv.npy", "rho.npy", "theta.npy",
          "u.npy", "v.npy", "w.npy")
EXPECTED_NESTED_ORIGIN = [11500, 5500]


def run(command: list[str], expect_success: bool = True) -> str:
    result = subprocess.run(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    if expect_success and any(marker in result.stdout for marker in ("[SCIENTIFIC INVALIDITY]", "Science profile rejected")):
        raise RuntimeError(f"recycling run reported invalid science:\n{result.stdout}")
    if expect_success != (result.returncode == 0):
        raise RuntimeError(
            f"unexpected return code {result.returncode}:\n{result.stdout}"
        )
    return result.stdout


def base(binary: Path, output: Path, threads: int, config: Path = CONFIG) -> list[str]:
    return [
        str(binary), "--headless", "--config", str(config),
        "--duration=65", "--write-every=65", f"--threads={threads}",
        "--log-profile=normal", "--no-timing", f"--outdir={output}",
    ]


def only(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern!r} below {path}, found {matches}")
    return matches[0]


def checkpoint_at(checkpoint_dir: Path, time_s: float) -> Path:
    matches: list[Path] = []
    for checkpoint in checkpoint_dir.glob("checkpoint_step_*.nimbus"):
        sidecar = json.loads(Path(f"{checkpoint}.json").read_text(encoding="utf-8"))
        actual_time = sidecar.get("simulation_time_s")
        if isinstance(actual_time, (int, float)) and math.isclose(actual_time, time_s, rel_tol=0.0, abs_tol=1e-9):
            matches.append(checkpoint)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one checkpoint at {time_s}s below {checkpoint_dir}, found {matches}"
        )
    return matches[0]


def checkpoint_section(checkpoint: Path, section_id: int) -> tuple[int, bytes]:
    data = checkpoint.read_bytes()
    cursor = 28
    count, = struct.unpack_from("<I", data, cursor)
    cursor += 4
    for _ in range(count):
        identity, version, _flags, size, _checksum = struct.unpack_from(
            "<IIIQQ", data, cursor
        )
        cursor += 28
        payload = data[cursor:cursor + size]
        cursor += size
        if identity == section_id:
            return version, payload
    raise RuntimeError(f"checkpoint omitted section {section_id}: {checkpoint}")


def fnv_bytes(data: bytes) -> int:
    value = 0xCBF29CE484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


def scientific_runtime_section(checkpoint: Path, output_roots: list[Path]) -> tuple[int, bytes]:
    version, payload = checkpoint_section(checkpoint, 6)
    metadata = json.loads(Path(f"{checkpoint}.json").read_text())["domain_recycling"]
    transaction = metadata["transaction_id"]
    generation = metadata["publication_generation"]
    manifests = [root / "domain_recycling_swath" / f"transaction_{transaction}" / "event.json"
                 for root in output_roots]
    manifests = [path for path in manifests if path.is_file()]
    if not manifests:
        raise RuntimeError("checkpoint recycling provenance has no durable event")
    event_path = manifests[0]
    event_bytes = event_path.read_bytes()
    event = json.loads(event_bytes)
    if (event["transaction_id"] != transaction or event["publication_generation"] != generation or
            len(event_bytes) != metadata["event_manifest_bytes"] or
            fnv_bytes(event_bytes) != int(metadata["event_manifest_fnv1a64"], 16)):
        raise RuntimeError("checkpoint recycling event identity or digest is invalid")
    for artifact in event["artifacts"]:
        path = (event_path.parent / artifact["path"]).resolve()
        path.relative_to(event_path.parent.resolve())
        data = path.read_bytes()
        if len(data) != artifact["bytes"] or fnv_bytes(data) != int(artifact["fnv1a64"], 16):
            raise RuntimeError(f"recycling artifact integrity failed: {path}")
    previous = int(metadata["previous_chain_fnv1a64"], 16)
    manifest_hash = int(metadata["event_manifest_fnv1a64"], 16)
    committed = int(metadata["committed_chain_fnv1a64"], 16)
    sequence = metadata["sequence"]
    manifest_size = metadata["event_manifest_bytes"]
    chain = struct.pack("<QQQQ", previous, sequence, manifest_size, manifest_hash)
    chain += transaction.encode() + b"\xff" + generation.encode() + b"\xff"
    if fnv_bytes(chain) != committed:
        raise RuntimeError("checkpoint recycling chain does not validate")
    prior_chain = 0
    for prior_sequence in range(1, sequence):
        prior_transaction = f"d5_{generation}_{prior_sequence}"
        candidates = [root / "domain_recycling_swath" / f"transaction_{prior_transaction}" / "event.json"
                      for root in output_roots]
        candidates = [path for path in candidates if path.is_file()]
        if not candidates:
            raise RuntimeError("checkpoint recycling chain is missing an earlier durable event")
        prior_path = candidates[0]
        prior_bytes = prior_path.read_bytes()
        prior_event = json.loads(prior_bytes)
        if (prior_event["transaction_id"] != prior_transaction
                or prior_event["publication_generation"] != generation
                or prior_event["transaction_sequence"] != prior_sequence
                or int(prior_event["previous_committed_chain_fnv1a64"], 16) != prior_chain):
            raise RuntimeError("earlier recycling event breaks the committed chain")
        for artifact in prior_event["artifacts"]:
            path = (prior_path.parent / artifact["path"]).resolve()
            path.relative_to(prior_path.parent.resolve())
            data = path.read_bytes()
            if len(data) != artifact["bytes"] or fnv_bytes(data) != int(artifact["fnv1a64"], 16):
                raise RuntimeError(f"earlier recycling artifact integrity failed: {path}")
        chain = struct.pack("<QQQQ", prior_chain, prior_sequence, len(prior_bytes), fnv_bytes(prior_bytes))
        chain += prior_transaction.encode() + b"\xff" + generation.encode() + b"\xff"
        prior_chain = fnv_bytes(chain)
    if prior_chain != previous:
        raise RuntimeError("checkpoint recycling previous chain does not match its durable history")
    def encoded(text: str) -> bytes:
        data = text.encode()
        return struct.pack("<I", len(data)) + data
    prefix = struct.pack("<BQ", 1, sequence)
    strings = encoded(transaction) + encoded(generation) + encoded(event["storage_format"])
    header = prefix + strings + struct.pack("<QQQQ", manifest_size, manifest_hash, previous, committed)
    if payload.count(header) != 1:
        raise RuntimeError("checkpoint recycling wire identity differs from its sidecar")
    # Fresh publication has a unique nonce; every scientific byte remains exact.
    canonical = prefix + encoded("0" * len(transaction)) + encoded("0" * len(generation))
    canonical += encoded(event["storage_format"]) + struct.pack("<QQQQ", manifest_size, 0, 0, 0)
    return version, payload.replace(header, canonical, 1)


def verify_movement(output: Path, two_way: bool = False) -> None:
    event = json.loads(only(
        output / "domain_recycling_swath", "transaction_*/event.json"
    ).read_text(encoding="utf-8"))
    shift = event.get("shift_cells")
    if two_way:
        if (not isinstance(shift, list) or len(shift) != 2 or shift == [0, 0] or
                any(not isinstance(value, int) or abs(value) > 6 for value in shift) or
                not 0 < event.get("simulation_time_s", 0) < 24):
            raise RuntimeError(f"two-way D5 omitted a bounded pre-checkpoint movement: {event}")
    elif event.get("simulation_time_s") != 12 or shift != [4, 0]:
        raise RuntimeError(f"nested D5 movement identity changed: {event}")
    parent = json.loads(
        (output / "step_000001" / "manifest.json").read_text(encoding="utf-8")
    )
    child = json.loads(
        (output / "nests" / "child_0000" / "step_000001" / "manifest.json")
        .read_text(encoding="utf-8")
    )
    parent_origin = parent["grid"]["horizontal_frame"]["grid_origin_t0_earth_m"]
    child_origin = child["grid"]["horizontal_frame"]["grid_origin_t0_earth_m"]
    if parent_origin != [1000 * value for value in shift] or child_origin != EXPECTED_NESTED_ORIGIN:
        raise RuntimeError(
            f"parent D5 did not preserve the child Earth frame: {parent_origin}, {child_origin}"
        )


def compare_outputs(
    first: Path, second: Path, label: str, compare_ledgers: bool = True, include_cloud: bool = False,
    include_frozen: bool = False,
) -> None:
    extra = (("qc.npy",) if include_cloud else ()) + (("qi.npy", "qs.npy", "qg.npy", "qh.npy") if include_frozen else ())
    for field in FIELDS + extra:
        for relative in (
            Path("step_000001") / field,
            Path("nests/child_0000/step_000001") / field,
        ):
            if (first / relative).read_bytes() != (second / relative).read_bytes():
                raise RuntimeError(f"{label} field mismatch: {relative}")
    if compare_ledgers:
        for ledger in LEDGERS:
            if (first / ledger).read_bytes() != (second / ledger).read_bytes():
                raise RuntimeError(f"{label} ledger mismatch: {ledger}")


def private_recycling_inventory(checkpoint: Path, expected_names: set[str]) -> list[dict]:
    metadata = json.loads(Path(f"{checkpoint}.json").read_text())["domain_recycling"]
    entries = metadata.get("private_inventory", [])
    if (metadata.get("private_inventory_complete") is not True or len(entries) != len(expected_names)
            or {entry["name"] for entry in entries} != expected_names):
        raise RuntimeError("recycling checkpoint omitted private inventory ownership")
    version, payload = checkpoint_section(checkpoint, 6)
    tail = struct.pack("<BH", 1, len(entries))
    for entry in entries:
        name = entry["name"].encode()
        values = (entry["retiring"], entry["admitted"])
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise RuntimeError("recycling checkpoint has invalid private inventories")
        tail += struct.pack("<I", len(name)) + name + struct.pack("<dd", *values)
    if version < 55 or not payload.endswith(tail):
        raise RuntimeError("private recycling sidecar differs from durable checkpoint inventories")
    return entries


def require_rejection(binary: Path, config: Path, output: Path, label: str) -> None:
    for mode, command in (
        ("validate", [str(binary), "--validate-only", "--config", str(config)]),
        ("startup", [str(binary), "--headless", "--config", str(config),
                     "--duration=0", "--write-every=0", f"--outdir={output}"]),
    ):
        text = run(command, expect_success=False)
        if not any(marker in text for marker in (
            "domain_motion.nesting", "domain_motion.mode=tracked_window",
            "domain_motion.tracking diagnostic evidence",
            "tracked_window requires recycle_domain=true",
        )):
            raise RuntimeError(f"{label} {mode} omitted its exact-scope rejection:\n{text}")
        if (output / "nimbus_output.json").exists():
            raise RuntimeError(f"{label} {mode} published output after rejection")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    config_path = args.config.resolve()
    identity = hashlib.sha256(binary.read_bytes()).hexdigest()
    if not binary.is_file() or not config_path.is_file():
        raise RuntimeError("nested-grid recycling gate inputs are missing")

    preflight = run([str(binary), "--validate-only", "--config", str(config_path)])
    if "[VALIDATE] PASS" not in preflight:
        raise RuntimeError("nested-grid recycling preflight omitted PASS")

    source = config_path.read_text(encoding="utf-8")
    two_way = "feedback: two_way" in source
    scheme = re.search(r"(?m)^microphysics:\n  scheme: (\w+)", source)[1]
    private = scheme in ("thompson", "milbrandt")
    water_fields = ("qv", "qc", "qr", "qi", "qs", "qg", "qh") if private else ("qv", "qc", "qr")
    backend = re.search(r"(?m)^    backend: (\w+)", source)[1]
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        if any(args.output.iterdir()):
            raise RuntimeError("nested recycling evidence directory must be empty")
    context = nullcontext(args.output.resolve()) if args.output else tempfile.TemporaryDirectory(prefix="nimbus-nested-grid-d5-")
    with context as temp:
        root = Path(temp)
        startup = run([
            str(binary), "--headless", "--config", str(config_path),
            "--duration=0", "--write-every=0", "--threads=1",
            "--log-profile=normal", f"--outdir={root / 'startup'}",
        ])
        for marker in (
            "Execution mode: qualified", f"[COMPUTE] backend={backend}",
            "movement mode live_flat_cgrid_recycling",
        ):
            if marker not in startup:
                raise RuntimeError(f"ordinary startup omitted {marker!r}")

        mutations = [
            ("origin", "    child_origin_parent_cell_i: 11\n",
             "    child_origin_parent_cell_i: 10\n"),
            ("time-ratio", "    parent_time_step_ratio: 2\n", "    parent_time_step_ratio: 3\n"),
            ("recycling", "    recycle_domain: true\n", "    recycle_domain: false\n"),
        ]
        if two_way:
            mutations += [
                ("backend", f"    backend: {backend}\n", "    backend: metal\n" if backend == "cpu" else "    backend: cpu\n"),
                ("buffer", "    inner_margin_m: 4000\n", "    inner_margin_m: 6000\n"),
            ]
        for label, old, new in mutations:
            mutation = source.replace(old, new, 1)
            if mutation == source:
                raise RuntimeError(f"{label} mutation did not alter the fixture")
            config = root / f"invalid-{label}.yaml"
            config.write_text(mutation, encoding="utf-8")
            require_rejection(binary, config, root / f"invalid-{label}", label)

        outputs = {threads: root / f"movement-t{threads}" for threads in (1, 4)}
        checkpoints = {threads: root / f"checkpoints-t{threads}" for threads in (1, 4)}
        water, momentum, private_closure = {}, {}, {}
        checkpoint_interval = 8 if two_way else 24
        for threads in (1, 4):
            log = run(base(binary, outputs[threads], threads, config_path) + [
                f"--checkpoint-every={checkpoint_interval}", f"--checkpoint-dir={checkpoints[threads]}"
            ])
            (root / f"movement-t{threads}.log").write_text(log)
            if two_way:
                water[threads] = require_water_closure(log)
                momentum[threads] = require_closure(log, f"recycling {threads} workers")
                require_feedback_output_budget(outputs[threads], 1, water_fields)
                if private:
                    private_closure[threads] = require_private_closure(log, scheme)
        if two_way and (water[1] != water[4] or momentum[1] != momentum[4]):
            raise RuntimeError("two-way recycling ledgers changed with worker count")
        if private and private_closure[1] != private_closure[4]:
            raise RuntimeError("private recycling ledgers changed with worker count")
        verify_movement(outputs[1], two_way)
        verify_movement(outputs[4], two_way)
        compare_outputs(outputs[1], outputs[4], "one/four worker", include_cloud=two_way, include_frozen=private)

        midpoint = checkpoint_at(checkpoints[1], 24.0)
        final_one = checkpoint_at(checkpoints[1], 65.0)
        final_four = checkpoint_at(checkpoints[4], 65.0)
        nested_one = checkpoint_section(final_one, 7)
        nested_four = checkpoint_section(final_four, 7)
        if nested_one[0] != 8 or nested_four[0] != 8 or nested_one != nested_four:
            raise RuntimeError("one/four worker v8 nested storage differs")
        private_inventory = []
        if private:
            private_inventory = private_recycling_inventory(final_one, set(private_closure[1]))
            if private_inventory != private_recycling_inventory(final_four, set(private_closure[1])):
                raise RuntimeError("private recycling inventories changed with worker count")
        sidecar = json.loads(Path(f"{midpoint}.json").read_text(encoding="utf-8"))
        recycling = sidecar.get("domain_recycling")
        owners = recycling.get("owner_algorithms", {}) if isinstance(recycling, dict) else {}
        if (recycling is None or recycling.get("sequence") != 1 or
                owners.get("nested_grid") !=
                "preserve_child_earth_frame_rebase_parent_indices_v1"):
            raise RuntimeError(f"midpoint omitted nested-grid D5 owner provenance: {recycling}")

        for start_time in ((8, 24) if two_way else (24,)):
            resumed_outputs: dict[int, Path] = {}
            resumed_checkpoints: dict[int, Path] = {}
            for threads in (1, 4):
                resumed_outputs[threads] = root / f"resumed-{start_time}-output-t{threads}"
                resumed_checkpoints[threads] = root / f"resumed-{start_time}-checkpoints-t{threads}"
                log = run(base(binary, resumed_outputs[threads], threads, config_path) + [
                    f"--resume={checkpoint_at(checkpoints[1], start_time)}", f"--checkpoint-every={checkpoint_interval}",
                    f"--checkpoint-dir={resumed_checkpoints[threads]}",
                ])
                (root / f"resumed-{start_time}-t{threads}.log").write_text(log)
                if two_way:
                    for current, reference in ((require_water_closure(log), water[threads]),
                                               (require_closure(log, "resumed recycling"), momentum[threads])):
                        for name, values in current.items():
                            if values != reference[name][-len(values):]:
                                raise RuntimeError(f"{name} recycling ledger changed across restart")
                    require_feedback_output_budget(resumed_outputs[threads], 1, water_fields)
                    if private:
                        for name, values in require_private_closure(log, scheme, False).items():
                            if values != private_closure[threads][name][-len(values):]:
                                raise RuntimeError(f"{name} private recycling ledger changed across restart")
                if start_time < 11 and two_way:
                    verify_movement(resumed_outputs[threads], two_way)
                resumed_final = checkpoint_at(resumed_checkpoints[threads], 65.0)
                if private and private_recycling_inventory(resumed_final, set(private_closure[1])) != private_inventory:
                    raise RuntimeError("private recycling inventories changed across restart")
                reference = final_one if threads == 1 else final_four
                for section_id in (1, 2, 3, 4, 5, 6, 7):
                    roots = [outputs[1], outputs[4], resumed_outputs[threads]]
                    current_section = scientific_runtime_section(resumed_final, roots) if section_id == 6 else checkpoint_section(resumed_final, section_id)
                    reference_section = scientific_runtime_section(reference, roots) if section_id == 6 else checkpoint_section(reference, section_id)
                    if current_section != reference_section:
                        raise RuntimeError(
                            f"{threads}-worker v8 continuation changed checkpoint "
                            f"section {section_id}"
                        )
            compare_outputs(
                resumed_outputs[1], resumed_outputs[4], "resumed one/four worker", include_cloud=two_way, include_frozen=private
            )
            compare_outputs(
                outputs[1], resumed_outputs[1], "continuous/restarted",
                compare_ledgers=False, include_cloud=two_way, include_frozen=private,
            )
            require_restart_suffix(outputs[1], resumed_outputs[1])
        if hashlib.sha256(binary.read_bytes()).hexdigest() != identity:
            raise RuntimeError("recycling executable changed during the gate")
        (root / "summary.json").write_text(json.dumps({
            "binary_sha256": identity, "configuration_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "two_way": two_way, "backend": backend, "threading": "exact", "restart": "exact", "movement_checked": True,
            "restart_times_s": [8, 24] if two_way else [24],
            "runtime_comparison": "all scientific bytes exact; independently validated publication nonce and derived digests normalized",
            "water_closure": water.get(1, {}), "dry_currency_closure": momentum.get(1, {}),
            "negative_admission": True, "private_closure": private_closure.get(1, {}),
            "private_recycling_inventory": private_inventory,
        }, indent=2) + "\n")

    print(
        "[nested-grid-recycling-ready] PASS: exact flat nested D5 moved the "
        "parent while preserving child Earth ownership; parent/child fields, "
        "v8 nested state, and five ledgers matched across 1/4 workers and restart"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
