#!/usr/bin/env python3
"""Bound ordinary exact mixed-open Metal admission and post-admission CLI scope."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.qualification.qualify_flat_mixed_metal_rebind_v1 import staged_configs, future_cli_contracts
from scripts.qualification.survey_generated_metal import FIELDS, LEDGERS, NATIVE_TVD_OPERATION
from scripts.tests.catalog.qualify_catalog_bounded_runs import dispatch_counts
from scripts.tests.science.npy_support import read_float32_npy

RSS_CAP = 134217728
RAW_CAP = 134217728
TIMEOUT = 60
SOURCES = [
    "src/core/runtime/runtime_config.cpp", "src/core/runtime/nimbus_sim.cpp",
    "include/core/runtime/runtime_config.hpp", "include/core/runtime/flat_mixed_metal_admission.hpp",
    "include/core/runtime/flat_mixed_metal_scope_v1.hpp",
    "include/core/runtime/qualified_generated_metal_cli_scope.hpp",
    "include/core/runtime/generated_metal_unserialized_controls.hpp",
    "src/core/runtime/scheme_catalog.cpp", "tests/core/test_scheme_catalog.cpp",
    "tests/fixtures/reference/catalog/flat_mixed_metal_cadence1_scope_v1.json",
    "ui/src/lib/config/schemeCatalog.ts", "ui/src/lib/config/nativeCatalogSnapshot.json",
    "ui/src-tauri/src/commands/config_preflight.rs",
    "scripts/qualification/qualify_flat_mixed_metal_rebind_v1.py",
    "scripts/qualification/survey_generated_metal.py",
    "scripts/tests/catalog/qualify_catalog_bounded_runs.py",
    "include/compute/vulkan_startup_profile.hpp", "src/core/runtime/configuration_compiler.hpp",
    "src/microphysics/schemes/milbrandt/milbrandt.cpp",
    "src/microphysics/schemes/milbrandt/milbrandt.hpp", "mk/sources.mk", "mk/tests.mk",
    "tests/fixtures/reference/catalog/flat_mixed_boundary_metal_exploratory_v1.yaml",
    "include/init/config/config_v1.hpp", "src/init/config/v1_loader.cpp",
    "src/init/config/config_v1_serialize.cpp", "include/core/output/step_manifest.hpp",
    "src/core/runtime/headless_runtime.cpp", "scripts/tests/science/npy_support.py",
]
SCOPE_ERROR = "qualified Metal execution is limited to exact public"
PERIODIC_ERROR = "periodic lateral boundaries must be configured as opposing pairs"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def identity(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink(), f"not a regular input: {path}")
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def raw_budget(output: Path) -> None:
    files = [path for path in output.rglob("*") if path.is_file()]
    require(all(not path.is_symlink() for path in files), "raw evidence contains a symbolic link")
    require(len(files) < 4096 and sum(path.stat().st_size for path in files) <= RAW_CAP,
            "raw evidence ceiling exceeded")


def requested_runtime(config: Path, flags: list[str]) -> dict:
    source = config.read_text()
    time_blocks = re.findall(r"(?ms)^time:\n(.*?)(?=^\S|\Z)", source)
    require(len(time_blocks) == 1, "requested fixture must have one time section")
    selected = {}
    for field in ("run_time", "output_interval", "dt_large"):
        values = re.findall(r"(?m)^  " + field + r": ([^\n]+)$", time_blocks[0])
        require(len(values) == 1, "requested fixture omits or duplicates " + field)
        selected[field] = float(values[0])
        require(math.isfinite(selected[field]), "requested time value is nonfinite")
    for flag, field in (("--duration", "run_time"), ("--write-every", "output_interval"),
                        ("--dt-large", "dt_large")):
        values = []
        for index, argument in enumerate(flags):
            if argument.startswith(flag + "="):
                values.append(argument.split("=", 1)[1])
            elif argument == flag:
                require(index + 1 < len(flags), "CLI value is missing: " + flag)
                values.append(flags[index + 1])
        require(len(values) <= 1, "CLI time override is duplicated: " + flag)
        if values:
            selected[field] = float(values[0])
            require(math.isfinite(selected[field]), "CLI time override is nonfinite")
    require(selected["run_time"] in (1.0, 2.0), "requested duration leaves the bounded one/two-second gate")
    return {"durationSeconds": int(selected["run_time"]),
            "writeEverySeconds": selected["output_interval"], "dtLargeSeconds": selected["dt_large"]}


def matched_rejection(body: str, markers: list[str]) -> bool:
    headers = {f"[{owner}] Science profile rejected before initialization:" for owner in ("CONFIG", "CLI")}
    diagnostics = []
    keyed_block = False
    for line in body.splitlines():
        stripped = line.strip()
        if stripped in headers:
            keyed_block = True
            continue
        if keyed_block:
            bullet = re.fullmatch(r"[ \t]+- (.+)", line)
            if bullet is not None:
                diagnostics.append(bullet.group(1))
                continue
            keyed_block = False
        if stripped.startswith(("[CONFIG]", "[CLI]")) or any(stripped.startswith(marker) for marker in markers):
            diagnostics.append(stripped)
    return bool(markers) and all(any(marker in line for line in diagnostics) for marker in markers)


def matched_periodic_loader_rejection(body: str, config: Path) -> bool:
    west_lines = [index for index, line in enumerate(config.read_text().splitlines(), 1)
                  if line == "    west: periodic"]
    if len(west_lines) != 1:
        return False
    path = str(config.resolve())
    expected = [
        f"[ERROR] v1 config load failed for: {path}",
        f"[ERROR]   {path}:{west_lines[0]} boundary_conditions.lateral.west: {PERIODIC_ERROR}; "
        "'boundary_conditions.lateral.west' and 'boundary_conditions.lateral.east' must either both be periodic "
        "or both be non-periodic",
        f"[CONFIG] v1 config load failed for '{path}' (1 error(s), listed above)",
    ]
    lines = body.splitlines()
    return any(lines[index:index + len(expected)] == expected for index in range(len(lines) - len(expected) + 1))


def close_time(value, expected: int) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and abs(value - expected) <= 1.0e-9


def output_members(science_output: Path, requested: dict) -> dict:
    duration = requested["durationSeconds"]
    require(requested["writeEverySeconds"] == 1.0 and requested["dtLargeSeconds"] == 0.1,
            "accepted request does not retain cadence1/dt0.1")
    run_manifest = json.loads((science_output / "nimbus_output.json").read_text())
    generation = run_manifest.get("publication_generation")
    require(run_manifest.get("schema") == "nimbus.output" and run_manifest.get("publication_state") == "ready" and
            isinstance(generation, str) and bool(generation), "run publication contract drift")
    manifests = sorted(science_output.glob("step_*/manifest.json"))
    require(len(manifests) == duration + 1, "publication cadence drift")
    members = {}
    for index, manifest_path in enumerate(manifests):
        manifest = json.loads(manifest_path.read_text())
        fields = manifest["published_fields"]
        require(manifest["schema"] == "nimbus.step" and manifest["publication_state"] == "complete" and
                manifest["publication_generation"] == generation and
                type(manifest["step_index"]) is int and manifest["step_index"] == index and
                manifest["step_dir"] == manifest_path.parent.name == f"step_{index:06}" and
                close_time(manifest["simulation_time_s"], index) and
                len(fields) == len(FIELDS) and set(fields) == set(FIELDS) and
                {path.stem for path in manifest_path.parent.glob("*.npy")} == set(FIELDS) and
                manifest["data_product"]["kind"] == "history" and
                manifest["data_product"]["storage_format"] == "npy_3d" and
                manifest["grid"]["dimensions"] == {"x": 18, "y": 18, "z": 18} and
                manifest["grid"]["coordinate_system"] == "cartesian" and
                manifest["grid"]["native_model_staggering"] == "c_grid" and
                manifest["grid"]["spacing_m"] == {"x": 500, "y": 500, "z": 250},
                "publication contract drift")
        for field in FIELDS:
            path = manifest_path.parent / f"{field}.npy"
            record = identity(path)
            shape, _, values = read_float32_npy(path)
            require(shape == (18, 18, 18) and all(math.isfinite(value) for value in values),
                    "published NPY shape/nonfinite contract drift")
            members[str(path.relative_to(science_output))] = record
    for ledger_name in LEDGERS:
        members[ledger_name] = identity(science_output / ledger_name)
    checkpoints = sorted((science_output / "checkpoints").glob("*.nimbus"))
    require(len(checkpoints) == duration, "checkpoint cadence drift")
    for index, path in enumerate(checkpoints, 1):
        record = identity(path)
        sidecar_path = path.with_suffix(".nimbus.json")
        sidecar = json.loads(sidecar_path.read_text())
        require(path.name == f"checkpoint_step_{index * 10:09}.nimbus" and
                sidecar["schema"] == "nimbus.checkpoint.lifecycle" and sidecar["schema_version"] == 1 and
                sidecar["checkpoint_file"] == path.name and sidecar["checkpoint_bytes"] == record["bytes"] and
                sidecar["checkpoint_sha256"] == record["sha256"] and
                type(sidecar["completed_steps"]) is int and sidecar["completed_steps"] == index * 10 and
                close_time(sidecar["simulation_time_s"], index) and sidecar["boundary"] == "completed_timestep",
                "checkpoint cursor/checksum contract drift")
        members[str(path.relative_to(science_output))] = record
        identity(sidecar_path)
    return members


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/nimbus_sim")
    parser.add_argument("--expected-binary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    output = args.output.resolve()
    require(output.is_relative_to(ROOT / "reports") or output.is_relative_to(Path("/private/tmp")),
            "use a new report/private-tmp root")
    require(not output.exists(), "output exists")
    require(identity(binary)["sha256"] == args.expected_binary_sha256, "binary identity drift")
    watched = [binary, Path(__file__), *[ROOT / name for name in SOURCES]]
    before = [identity(path) for path in watched]
    output.mkdir(parents=True)
    configs = staged_configs()
    selected = {name: text for name, text in configs.items()
                if name == "metal_qualified" or name.startswith("negative_")}
    selected["metal_qualified_memory_reject"] = configs["metal_qualified"].replace(
        "    memory_admission: warn\n", "    memory_admission: reject\n", 1)
    require(configs["metal_qualified"].count("    memory_admission: warn\n") == 1,
            "resource-policy fixture anchor is missing or duplicated")
    fixtures = {}
    for name, text in selected.items():
        path = output / "fixtures" / (name + ".yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(text)
        fixtures[name] = path
    watched.extend(fixtures.values())
    before.extend(identity(path) for path in fixtures.values())
    report = {"schema": "nimbus.flat-mixed-metal-public.v1", "status": "running",
              "identityBefore": before, "familyId": "boundary:ed765facac077ef0b331283d/metal",
              "physicsOwners": {"scalarTransport": "native_metal", "lateralDamping": "host",
                                 "initialConditionChaos": "host", "upperDamping": "off", "wOnly": "off"},
              "rssCeilingBytes": RSS_CAP, "rawByteCeiling": RAW_CAP, "timeoutSecondsPerCase": TIMEOUT,
              "measurementKind": "runtime_milestone_lifetime_peak_rss",
              "externalWholeProcessRssMeasured": False,
              "sourceScope": "enumerated public-admission/runtime/publication inputs, not a complete release build seal",
              "fixtures": {name: identity(path) for name, path in fixtures.items()},
              "cases": {}, "errors": [], "baselinePassed": False,
              "boundedChecksPass": False, "I1Complete": False, "wholeBackendReady": False,
              "campaignValidated": False, "externallyReviewed": False, "fullyQualified": False}
    write_new(output / "plan.json", report)
    accepted_reference = None

    def run(name: str, config: Path, flags: list[str], accepted: bool, markers: list[str]) -> None:
        nonlocal accepted_reference
        require([identity(path) for path in watched] == before, "source/binary drift before case")
        require(accepted or report["baselinePassed"], "negative proof before admitted baseline")
        case = output / "cases" / name
        case.mkdir(parents=True)
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(("NIMBUS_", "TMV_", "OMP_", "KMP_", "GOMP_"))}
        environment.update({"OMP_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE"})
        science_output = case / "output"
        command = [str(binary), "--headless", "--config", str(config), "--threads=1",
                   "--checkpoint-every=1", f"--checkpoint-dir={science_output / 'checkpoints'}",
                   f"--outdir={science_output}", "--log-profile=normal", "--no-timing", *flags]
        write_new(case / "request.json", {"command": command, "config": identity(config),
                  "controlledEnvironment": {key: value for key, value in environment.items()
                      if key.startswith(("NIMBUS_", "OMP_", "VK_", "DYLD_"))}})
        requested = requested_runtime(config, flags)
        timed_out = False
        process_error = None
        exit_status = None
        try:
            completed = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       timeout=TIMEOUT, check=False)
            exit_status = completed.returncode
            raw = completed.stdout.encode("utf-8")
        except subprocess.TimeoutExpired as error:
            timed_out = True
            stdout = error.stdout or b""
            stderr = error.stderr or b""
            raw = (stdout if isinstance(stdout, bytes) else stdout.encode("utf-8")) + \
                  (stderr if isinstance(stderr, bytes) else stderr.encode("utf-8"))
        except OSError as error:
            process_error = str(error)
            raw = process_error.encode("utf-8")
        body = raw.decode("utf-8", errors="replace")
        with (case / "runtime.log").open("xb") as stream:
            stream.write(raw)
        write_new(case / "process.json", {"exitStatus": exit_status, "timedOut": timed_out,
                  "error": process_error, "timeoutSeconds": TIMEOUT})
        record = {"acceptedExpected": accepted, "exitStatus": exit_status, "timedOut": timed_out,
                  "requestedRuntime": requested,
                  "request": identity(case / "request.json"), "log": identity(case / "runtime.log"),
                  "process": identity(case / "process.json")}
        report["cases"][name] = record
        raw_budget(output)
        require(not timed_out and process_error is None, f"process timeout/start failure: {name}")
        if accepted:
            require(exit_status == 0 and "backend=metal, requested=metal" in body and
                    "fallback=off" in body and "Execution mode: qualified" in body and
                    "[SCIENTIFIC INVALIDITY]" not in body,
                    f"ordinary native admission/runtime failed: {name}")
            durations = re.findall(r"\[INFO\]\s+Duration: ([0-9.]+)s", body)
            cadences = re.findall(r"\[INFO\]\s+Write every: ([0-9.]+)s", body)
            require(len(durations) == len(cadences) == 1 and
                    float(durations[0]) == requested["durationSeconds"] and
                    float(cadences[0]) == requested["writeEverySeconds"],
                    "requested fixture/CLI differs from reported runtime duration/cadence")
            validation = json.loads((science_output / "validation_summary.json").read_text())
            require(validation["failed"] is False and validation["guard_mode"] == "strict" and
                    validation["guard_fail_on"] == "both" and validation["guard_scope"] == "required_only" and
                    type(validation["report_count"]) is int and
                    validation["report_count"] >= requested["durationSeconds"] + 1 and
                    all(type(validation[key]) is int and validation[key] == 0 for key in
                        ("failed_report_count", "total_violations", "total_nonfinite", "total_bounds",
                         "total_sanitized_nonfinite", "total_sanitized_bounds")),
                    "strict scientific validation failed")
            duration = requested["durationSeconds"]
            members = output_members(science_output, requested)
            hashes = {key: value["sha256"] for key, value in members.items()}
            if duration == 2:
                if accepted_reference is None:
                    accepted_reference = hashes
                require(hashes == accepted_reference, "accepted CLI/resource-policy science bytes changed")
            reports, operations = dispatch_counts(body, "metal")
            require(operations.get(NATIVE_TVD_OPERATION, 0) == 30 * duration,
                    "native conservative TVD owner missing")
            milestones = re.findall(r"\[MEMORY MILESTONE\] phase=([^,]+), peak_rss_bytes=(\d+)", body)
            peaks = [int(value) for _, value in milestones]
            drains = [int(value) for phase, value in milestones if phase == "output_drain_complete"]
            require(peaks and all(value > 0 for value in peaks) and drains and drains[-1] == max(peaks),
                    "terminal positive lifetime RSS evidence is missing")
            require(max(peaks) <= RSS_CAP, "unchanged lifetime RSS ceiling exceeded")
            record.update({"science": members, "dispatchReports": reports, "operations": operations,
                           "peakRssBytes": max(peaks), "measurementKind": report["measurementKind"],
                           "strictValidation": identity(science_output / "validation_summary.json"),
                           "runPublication": identity(science_output / "nimbus_output.json"),
                           "frameManifests": [identity(path) for path in sorted(science_output.glob("step_*/manifest.json"))],
                           "checkpointSidecars": [identity(path) for path in sorted((science_output / "checkpoints").glob("*.nimbus.json"))]})
            if name == "baseline":
                report["baselinePassed"] = True
        else:
            loader_case = name == "negative_unpaired_periodic"
            matched = (exit_status == 1 and matched_periodic_loader_rejection(body, config)) if loader_case else \
                (exit_status in (1, 2) and matched_rejection(body, markers))
            require(matched,
                    f"wrong rejection/error class: {name}")
            require(not science_output.exists(), "rejected selector published an output root")
            record["matchedRejectionMarkers"] = markers
            if loader_case:
                record["rejectionClass"] = "strict_v1_unpaired_periodic"
                record["loaderFixture"] = identity(config)
            record["noPublication"] = True
        require([identity(path) for path in watched] == before, "source/binary drift during case")
        raw_budget(output)
        print(json.dumps({"case": name, "acceptedExpected": accepted, "exitStatus": exit_status}), flush=True)

    try:
        for contract in future_cli_contracts():
            run(contract["name"], fixtures["metal_qualified"], contract["flags"],
                contract["accepted"], contract["rejectionMarkers"])
        run("memory_reject", fixtures["metal_qualified_memory_reject"], [], True, [])
        for name, path in fixtures.items():
            if name.startswith("negative_"):
                markers = (["boundary_conditions.lateral.west", PERIODIC_ERROR]
                           if name == "negative_unpaired_periodic" else [SCOPE_ERROR])
                run(name, path, [], False, markers)
        report["boundedChecksPass"] = True
        report["status"] = "bounded_ordinary_admission_cli_output_resource_pass"
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as error:
        report["status"] = "red_preserved"
        report["errors"].append(str(error))
    report["identityAfter"] = []
    for path in watched:
        try:
            report["identityAfter"].append(identity(path))
        except (OSError, ValueError) as error:
            report["identityAfter"].append({"path": str(path), "unavailable": str(error)})
    if report["identityAfter"] != before:
        report["boundedChecksPass"] = False
        report["status"] = "red_preserved"
        report["errors"].append("final source/binary drift")
    write_new(output / "summary.json", report)
    print(json.dumps({"status": report["status"], "errors": report["errors"],
                      "summary": identity(output / "summary.json")}), flush=True)
    return 0 if report["boundedChecksPass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
