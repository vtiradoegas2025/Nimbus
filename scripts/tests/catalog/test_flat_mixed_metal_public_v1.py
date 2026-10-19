from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest import mock

from scripts.qualification import qualify_flat_mixed_metal_public_v1 as gate


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def npy_bytes(value: float = 0.0) -> bytes:
    header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (18, 18, 18), }"
    header += b" " * ((16 - (10 + len(header) + 1) % 16) % 16) + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + struct.pack("<f", value) * (18 ** 3)


def publish(output: Path, duration: int) -> None:
    generation = "test-generation"
    write_json(output / "nimbus_output.json", {
        "schema": "nimbus.output", "publication_state": "ready", "publication_generation": generation})
    write_json(output / "validation_summary.json", {
        "guard_mode": "strict", "guard_fail_on": "both", "guard_scope": "required_only",
        "failed": False, "report_count": duration + 1, "failed_report_count": 0,
        "total_violations": 0, "total_nonfinite": 0, "total_bounds": 0,
        "total_sanitized_nonfinite": 0, "total_sanitized_bounds": 0})
    for index in range(duration + 1):
        step = output / f"step_{index:06}"
        write_json(step / "manifest.json", {
            "schema": "nimbus.step", "publication_state": "complete",
            "publication_generation": generation, "step_index": index,
            "step_dir": step.name, "simulation_time_s": index,
            "data_product": {"kind": "history", "storage_format": "npy_3d"},
            "published_fields": list(gate.FIELDS),
            "grid": {"dimensions": {"x": 18, "y": 18, "z": 18},
                     "coordinate_system": "cartesian", "native_model_staggering": "c_grid",
                     "spacing_m": {"x": 500, "y": 500, "z": 250}}})
        for field in gate.FIELDS:
            values = {"p": 100000.0, "rho": 1.0, "theta": 300.0, "qv": 0.004}
            (step / f"{field}.npy").write_bytes(npy_bytes(values.get(field, 0.0)))
    for ledger in gate.LEDGERS:
        (output / ledger).write_text("sim_time_s,value\n0,1\n")
    for index in range(1, duration + 1):
        checkpoint = output / "checkpoints" / f"checkpoint_step_{10 * index:09}.nimbus"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        content = f"checkpoint-{index}".encode()
        checkpoint.write_bytes(content)
        write_json(checkpoint.with_suffix(".nimbus.json"), {
            "schema": "nimbus.checkpoint.lifecycle", "schema_version": 1,
            "checkpoint_file": checkpoint.name, "checkpoint_bytes": len(content),
            "checkpoint_sha256": hashlib.sha256(content).hexdigest(),
            "completed_steps": index * 10, "simulation_time_s": index,
            "boundary": "completed_timestep"})


def contract(name: str = "baseline", flags=(), accepted: bool = True, markers=()) -> dict:
    return {"name": name, "flags": list(flags), "accepted": accepted, "rejectionMarkers": list(markers)}


def periodic_loader_body(config: Path) -> str:
    path = str(config.resolve())
    line = next(index for index, value in enumerate(config.read_text().splitlines(), 1)
                if value == "    west: periodic")
    return (f"[ERROR] v1 config load failed for: {path}\n"
            f"[ERROR]   {path}:{line} boundary_conditions.lateral.west: {gate.PERIODIC_ERROR}; "
            "'boundary_conditions.lateral.west' and 'boundary_conditions.lateral.east' must either both be periodic "
            "or both be non-periodic\n"
            f"[CONFIG] v1 config load failed for '{path}' (1 error(s), listed above)\n")


class PublicMixedProducerTests(unittest.TestCase):
    def exercise(self, *, contracts=None, configs=None, change=None, rejected=None, remove=False,
                 raw_cap=None, expected_hash=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            binary = root / "simulator"
            binary.write_text("fixed fake simulator")
            watched = root / "watched.cpp"
            watched.write_text("fixed source")
            output = root / "reports" / "receipt"
            selected = gate.staged_configs() if configs is None else configs
            selected = {key: value for key, value in selected.items()
                        if key == "metal_qualified" or (configs is not None and key.startswith("negative_"))}
            calls = []

            def fake_run(command, **kwargs):
                request_output = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--outdir=")))
                name = request_output.parent.name
                calls.append(name)
                config = Path(command[command.index("--config") + 1])
                self.assertEqual(json.loads((request_output.parent / "request.json").read_text())["config"],
                                 gate.identity(config))
                self.assertEqual(kwargs["env"]["OMP_NUM_THREADS"], "1")
                self.assertEqual(kwargs["env"]["OMP_DYNAMIC"], "FALSE")
                self.assertNotIn("NIMBUS_UNREVIEWED_SWITCH", kwargs["env"])
                if rejected is not None and name == rejected[0]:
                    if isinstance(rejected[1], BaseException):
                        raise rejected[1]
                    rejection_body = rejected[2](config) if callable(rejected[2]) else rejected[2]
                    if change is not None:
                        changed = change(name, config, request_output, watched, binary, rejection_body)
                        if changed is not None:
                            rejection_body = changed
                    return subprocess.CompletedProcess(command, rejected[1], rejection_body)
                duration = 1 if "--duration=1" in command else 2
                publish(request_output, duration)
                body = ("[COMPUTE] backend=metal, requested=metal, fallback=off\n"
                        "[INFO] Execution mode: qualified\n"
                        f"[INFO] Duration: {duration}s\n[INFO] Write every: 1s\n"
                        f"[COMPUTE DISPATCH] backend=metal, kernels={{{gate.NATIVE_TVD_OPERATION}:{30 * duration}}}\n"
                        "[MEMORY MILESTONE] phase=output_drain_complete, peak_rss_bytes=50000000\n")
                if change is not None:
                    result = change(name, config, request_output, watched, binary, body)
                    if result is not None:
                        body = result
                if remove:
                    watched.unlink()
                return subprocess.CompletedProcess(command, 0, body)

            arguments = ["producer", "--binary", str(binary), "--expected-binary-sha256",
                         expected_hash or gate.identity(binary)["sha256"], "--output", str(output)]
            with mock.patch.object(gate, "ROOT", root), \
                    mock.patch.object(gate, "SOURCES", ["watched.cpp"]), \
                    mock.patch.object(gate, "staged_configs", return_value=selected), \
                    mock.patch.object(gate, "future_cli_contracts", return_value=contracts or [contract()]), \
                    mock.patch.object(gate.subprocess, "run", side_effect=fake_run), \
                    mock.patch.dict(gate.os.environ, {"NIMBUS_UNREVIEWED_SWITCH": "1"}), \
                    mock.patch.object(gate.sys, "argv", arguments), \
                    mock.patch.object(gate, "RAW_CAP", gate.RAW_CAP if raw_cap is None else raw_cap), \
                    mock.patch("builtins.print"):
                try:
                    code = gate.main()
                except ValueError:
                    if expected_hash is None:
                        raise
                    code = 1
            summary_path = output / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
            partial = {}
            if output.exists():
                for path in output.rglob("runtime.log"):
                    partial[path.parent.name] = path.read_bytes()
            return code, summary, calls, partial

    def test_baseline_and_resource_policy_preserve_current_epoch_science_bytes(self):
        code, summary, calls, _ = self.exercise(contracts=[contract(), contract("duration_one", ["--duration=1"])])
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["baseline", "duration_one", "memory_reject"])
        self.assertTrue(summary["baselinePassed"])
        self.assertTrue(summary["boundedChecksPass"])
        self.assertEqual(summary["physicsOwners"]["lateralDamping"], "host")
        self.assertEqual(summary["measurementKind"], "runtime_milestone_lifetime_peak_rss")
        self.assertFalse(summary["externalWholeProcessRssMeasured"])
        self.assertFalse(summary["fullyQualified"])

    def test_closed_baseline_cannot_make_negative_checks_green(self):
        marker = "qualified mixed-open Metal NPY scope requires --write-every=1"
        code, summary, calls, _ = self.exercise(
            contracts=[contract(), contract("write_301", ["--write-every=301"], False, [marker])],
            rejected=("baseline", 2, "[CONFIG] qualified Metal execution is limited to exact public profiles\n"))
        self.assertEqual(code, 1)
        self.assertEqual(calls, ["baseline"])
        self.assertFalse(summary["baselinePassed"])

    def test_negative_before_baseline_never_launches_a_process(self):
        code, summary, calls, _ = self.exercise(contracts=[contract("negative_first", [], False, ["expected"])])
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertFalse(summary["baselinePassed"])

    def test_crashes_zero_exit_and_wrong_diagnostics_do_not_prove_cli_rejection(self):
        marker = "qualified mixed-open Metal NPY scope requires --write-every=1"
        for code, body in ((-11, marker), (139, marker), (0, marker), (2, "[CONFIG] unrelated failure"),
                           (2, "[INFO] " + marker)):
            with self.subTest(code=code, body=body):
                result, summary, calls, _ = self.exercise(
                    contracts=[contract(), contract("write_301", ["--write-every=301"], False, [marker])],
                    rejected=("write_301", code, body))
                self.assertEqual(result, 1)
                self.assertEqual(calls, ["baseline", "write_301"])
                self.assertFalse(summary["boundedChecksPass"])

    def test_exact_cli_diagnostic_without_publication_proves_the_negative(self):
        marker = "qualified mixed-open Metal NPY scope requires --write-every=1"
        result, summary, _, _ = self.exercise(
            contracts=[contract(), contract("write_301", ["--write-every=301"], False, [marker])],
            rejected=("write_301", 1, marker + "\nTry 'nimbus_sim --help' for usage.\n"))
        self.assertEqual(result, 0)
        self.assertTrue(summary["cases"]["write_301"]["noPublication"])

    def test_generic_qualified_metal_or_periodic_words_are_not_keyed_fixture_errors(self):
        configs = gate.staged_configs()
        for name, body in (("negative_netcdf", "[CONFIG] unrelated qualified science and Metal issue"),
                           ("negative_unpaired_periodic", "[CONFIG] periodic allocation error")):
            with self.subTest(name=name):
                chosen = {"metal_qualified": configs["metal_qualified"], name: configs[name]}
                code, summary, _, _ = self.exercise(configs=chosen, rejected=(name, 2, body))
                self.assertEqual(code, 1)
                self.assertFalse(summary["boundedChecksPass"])

    def test_timeout_preserves_partial_bytes_process_and_final_summary(self):
        timeout = subprocess.TimeoutExpired(["simulator"], 60, output=b"partial science log\xff\n")
        code, summary, calls, partial = self.exercise(rejected=("baseline", timeout))
        self.assertEqual(code, 1)
        self.assertEqual(calls, ["baseline"])
        self.assertEqual(partial["baseline"], b"partial science log\xff\n")
        self.assertTrue(summary["cases"]["baseline"]["timedOut"])
        self.assertIsNone(summary["cases"]["baseline"]["exitStatus"])

    def test_source_binary_and_fixture_drift_preserve_a_red_summary(self):
        for target in ("source", "binary", "fixture"):
            def mutate(name, config, output, watched, binary, body):
                selected = {"source": watched, "binary": binary, "fixture": config}[target]
                selected.write_text(selected.read_text() + "changed\n")
            with self.subTest(target=target):
                code, summary, calls, _ = self.exercise(change=mutate)
                self.assertEqual(code, 1)
                self.assertEqual(calls, ["baseline"])
                self.assertFalse(summary["boundedChecksPass"])

    def test_deleted_watched_source_cannot_prevent_the_final_receipt(self):
        code, summary, calls, _ = self.exercise(remove=True)
        self.assertEqual(code, 1)
        self.assertEqual(calls, ["baseline"])
        self.assertFalse(summary["boundedChecksPass"])
        self.assertTrue(summary["errors"])

    def test_wrong_expected_binary_hash_prevents_execution(self):
        code, summary, calls, _ = self.exercise(expected_hash="0" * 64)
        self.assertEqual(code, 1)
        self.assertIsNone(summary)
        self.assertEqual(calls, [])

    def test_strict_validation_policy_and_coverage_are_required(self):
        for key, value in (("guard_mode", "warn"), ("guard_fail_on", "nonfinite"),
                           ("guard_scope", "exported"), ("report_count", 0), ("failed_report_count", 1),
                           ("total_bounds", 1), ("total_sanitized_bounds", 1)):
            def mutate(name, config, output, watched, binary, body):
                path = output / "validation_summary.json"
                document = json.loads(path.read_text())
                document[key] = value
                write_json(path, document)
            with self.subTest(key=key):
                code, summary, _, _ = self.exercise(change=mutate)
                self.assertEqual(code, 1)
                self.assertFalse(summary["boundedChecksPass"])

    def test_native_dispatch_missing_or_wrong_backend_cannot_pass(self):
        for token in ("missing", "cpu"):
            def mutate(name, config, output, watched, binary, body):
                if token == "missing":
                    return body.replace(gate.NATIVE_TVD_OPERATION, "unrelated/kernel.comp")
                return body.replace("[COMPUTE DISPATCH] backend=metal", "[COMPUTE DISPATCH] backend=cpu")
            with self.subTest(token=token):
                code, summary, _, _ = self.exercise(change=mutate)
                self.assertEqual(code, 1)
                self.assertFalse(summary["boundedChecksPass"])

    def test_missing_zero_nonterminal_or_over_cap_rss_is_not_resource_evidence(self):
        for value in ("missing", "0", str(gate.RSS_CAP + 1), "nonterminal"):
            def mutate(name, config, output, watched, binary, body):
                if value == "missing":
                    return body.split("[MEMORY MILESTONE]", 1)[0]
                if value == "nonterminal":
                    return body.replace("output_drain_complete", "startup_ready")
                return body.replace("50000000", value)
            with self.subTest(value=value):
                code, summary, _, _ = self.exercise(change=mutate)
                self.assertEqual(code, 1)
                self.assertFalse(summary["boundedChecksPass"])

    def test_run_and_frame_publication_contracts_are_verified(self):
        for kind in ("failed_run", "generation", "wrong_time", "duplicate_field", "extra_npy", "format"):
            def mutate(name, config, output, watched, binary, body):
                run_path = output / "nimbus_output.json"
                manifest_path = output / "step_000001/manifest.json"
                manifest = json.loads(manifest_path.read_text())
                if kind == "failed_run":
                    run = json.loads(run_path.read_text())
                    run["publication_state"] = "failed"
                    write_json(run_path, run)
                elif kind == "extra_npy":
                    (manifest_path.parent / "unselected.npy").write_bytes(npy_bytes())
                else:
                    if kind == "generation":
                        manifest["publication_generation"] = "wrong-generation"
                    elif kind == "wrong_time":
                        manifest["simulation_time_s"] = 301
                    elif kind == "duplicate_field":
                        manifest["published_fields"].append(gate.FIELDS[0])
                    else:
                        manifest["data_product"]["storage_format"] = "netcdf"
                    write_json(manifest_path, manifest)
            with self.subTest(kind=kind):
                code, summary, _, _ = self.exercise(change=mutate)
                self.assertEqual(code, 1)
                self.assertFalse(summary["boundedChecksPass"])

    def test_corrupt_or_nonfinite_npy_payload_cannot_pass_a_zero_validation_summary(self):
        for value in (None, float("nan")):
            def mutate(name, config, output, watched, binary, body):
                path = output / "step_000001/p.npy"
                path.write_bytes(b"not an NPY" if value is None else npy_bytes(value))
            with self.subTest(value=value):
                code, summary, _, _ = self.exercise(change=mutate)
                self.assertEqual(code, 1)
                self.assertFalse(summary["boundedChecksPass"])

    def test_checkpoint_cursor_checksum_and_presence_are_verified(self):
        for kind in ("missing", "sha256", "time", "steps"):
            def mutate(name, config, output, watched, binary, body):
                path = output / "checkpoints/checkpoint_step_000000010.nimbus.json"
                if kind == "missing":
                    path.unlink()
                    return
                document = json.loads(path.read_text())
                if kind == "sha256":
                    document["checkpoint_sha256"] = "0" * 64
                elif kind == "time":
                    document["simulation_time_s"] = 301
                else:
                    document["completed_steps"] = 11
                write_json(path, document)
            with self.subTest(kind=kind):
                code, summary, _, _ = self.exercise(change=mutate)
                self.assertEqual(code, 1)
                self.assertFalse(summary["boundedChecksPass"])

    def test_raw_budget_and_same_epoch_science_bytes_are_enforced(self):
        code, summary, _, _ = self.exercise(raw_cap=1)
        self.assertEqual(code, 1)
        self.assertFalse(summary["boundedChecksPass"])
        def mutate(name, config, output, watched, binary, body):
            if name == "memory_reject":
                (output / "budget.csv").write_text("different science bytes\n")
        code, summary, _, _ = self.exercise(change=mutate)
        self.assertEqual(code, 1)
        self.assertFalse(summary["boundedChecksPass"])

    def test_runtime_reporting_must_match_requested_fixture_and_cli_cadence(self):
        for token, replacement in (("Execution mode: qualified", "Execution mode: exploratory"),
                                   ("Duration: 2s", "Duration: 1s"),
                                   ("Write every: 1s", "Write every: 301s")):
            def mutate(name, config, output, watched, binary, body):
                return body.replace(token, replacement)
            with self.subTest(token=token):
                code, summary, calls, _ = self.exercise(change=mutate)
                self.assertEqual(code, 1)
                self.assertEqual(calls, ["baseline"])
                self.assertFalse(summary["boundedChecksPass"])
        configs = gate.staged_configs()
        configs["metal_qualified"] = configs["metal_qualified"].replace("  run_time: 2\n", "  run_time: 1\n")
        code, summary, calls, _ = self.exercise(configs={"metal_qualified": configs["metal_qualified"]})
        self.assertEqual(code, 1)
        self.assertEqual(calls, ["baseline"])
        self.assertFalse(summary["boundedChecksPass"])

    def test_requested_time_is_derived_and_unbounded_or_duplicate_controls_reject(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.yaml"
            path.write_text("time:\n  run_time: 1\n  output_interval: 1\n  dt_large: 0.1\n")
            self.assertEqual(gate.requested_runtime(path, []),
                             {"durationSeconds": 1, "writeEverySeconds": 1.0, "dtLargeSeconds": 0.1})
            self.assertEqual(gate.requested_runtime(path, ["--duration", "2", "--write-every=301"])["writeEverySeconds"], 301)
            for flags in (["--duration=3"], ["--duration=1", "--duration=2"], ["--duration"]):
                with self.subTest(flags=flags), self.assertRaises(ValueError):
                    gate.requested_runtime(path, flags)
            path.write_text(path.read_text() + "time:\n  run_time: 1\n")
            with self.assertRaises(ValueError):
                gate.requested_runtime(path, [])

    def test_exact_keyed_fixture_error_passes_after_the_admitted_baseline(self):
        configs = gate.staged_configs()
        for name, body, status in (
                ("negative_netcdf", "[CONFIG] " + gate.SCOPE_ERROR + " profiles", 2),
                ("negative_unpaired_periodic", periodic_loader_body, 1)):
            with self.subTest(name=name):
                chosen = {"metal_qualified": configs["metal_qualified"], name: configs[name]}
                code, summary, calls, _ = self.exercise(configs=chosen, rejected=(name, status, body))
                self.assertEqual(code, 0)
                self.assertEqual(calls, ["baseline", "memory_reject", name])
                self.assertTrue(summary["cases"][name]["noPublication"])

    def test_process_start_failure_preserves_raw_and_final_receipt(self):
        code, summary, calls, partial = self.exercise(rejected=("baseline", OSError("fake launch failure")))
        self.assertEqual(code, 1)
        self.assertEqual(calls, ["baseline"])
        self.assertIn(b"fake launch failure", partial["baseline"])
        self.assertIn("baseline", summary["cases"])
        self.assertFalse(summary["boundedChecksPass"])

    def test_declared_source_inventory_includes_the_originating_fixture(self):
        self.assertIn("tests/fixtures/reference/catalog/flat_mixed_boundary_metal_exploratory_v1.yaml", gate.SOURCES)
        self.assertIn("scripts/tests/science/npy_support.py", gate.SOURCES)
    def test_keyed_rejection_with_an_output_root_does_not_prove_no_publication(self):
        marker = "qualified mixed-open Metal NPY scope requires --write-every=1"
        def publish_rejected(name, config, output, watched, binary, body):
            if name == "write_301":
                output.mkdir(parents=True)
        code, summary, _, _ = self.exercise(
            contracts=[contract(), contract("write_301", ["--write-every=301"], False, [marker])],
            rejected=("write_301", 1, marker), change=publish_rejected)
        self.assertEqual(code, 1)
        self.assertFalse(summary["boundedChecksPass"])

    def test_the_unchanged_rss_cap_boundary_is_inclusive(self):
        def exact_cap(name, config, output, watched, binary, body):
            return body.replace("50000000", str(gate.RSS_CAP))
        code, summary, _, _ = self.exercise(change=exact_cap)
        self.assertEqual(code, 0)
        self.assertEqual(summary["cases"]["baseline"]["peakRssBytes"], 134217728)

    def test_keyed_rejection_blocks_accept_only_their_contiguous_bullets(self):
        marker = "qualified mixed-open Metal NPY scope requires --write-every=1"
        for owner in ("CONFIG", "CLI"):
            header = f"[{owner}] Science profile rejected before initialization:"
            with self.subTest(owner=owner):
                self.assertTrue(gate.matched_rejection(header + "\n  - " + marker, [marker]))
                self.assertTrue(gate.matched_rejection(
                    header + "\n  - first keyed science error\n  - " + marker,
                    ["first keyed science error", marker]))

    def test_stray_bullets_wrong_headers_and_interrupted_blocks_do_not_match(self):
        marker = "qualified mixed-open Metal NPY scope requires --write-every=1"
        header = "[CONFIG] Science profile rejected before initialization:"
        wrong_contexts = (
            "  - " + marker,
            "[INFO] Science profile rejected before initialization:\n  - " + marker,
            "[CONFIG] Unrelated failure:\n  - " + marker,
            "[CLI] Unrelated failure:\n  - " + marker,
            header + "\n\n  - " + marker,
            header + "\n[INFO] unrelated output\n  - " + marker,
            header + "\n  - unrelated science error\n[ERROR] another error class\n  - " + marker,
            header + "\n- " + marker,
        )
        for body in wrong_contexts:
            with self.subTest(body=body):
                self.assertFalse(gate.matched_rejection(body, [marker]))

    def test_actual_config_block_rejection_passes_but_crash_and_zero_exit_do_not(self):
        marker = "qualified mixed-open Metal NPY scope requires --write-every=1"
        body = "[CONFIG] Science profile rejected before initialization:\n  - " + marker + "\n"
        for exit_status in (1, -11, 139, 0):
            with self.subTest(exit_status=exit_status):
                code, summary, calls, _ = self.exercise(
                    contracts=[contract(), contract("write_301", ["--write-every=301"], False, [marker])],
                    rejected=("write_301", exit_status, body))
                self.assertEqual(code, 0 if exit_status == 1 else 1)
                self.assertEqual(calls[:2], ["baseline", "write_301"])
                if exit_status == 1:
                    self.assertTrue(summary["cases"]["write_301"]["noPublication"])
                else:
                    self.assertFalse(summary["boundedChecksPass"])

    def test_periodic_loader_diagnostic_is_bound_to_the_fixture_line_key_and_trailer(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "negative_unpaired_periodic.yaml"
            config.write_text("boundary_conditions:\n  lateral:\n    west: periodic\n")
            body = periodic_loader_body(config)
            self.assertTrue(gate.matched_periodic_loader_rejection(body, config))
            mutations = (
                body.replace(str(config.resolve()), str(config.resolve()) + ".wrong", 1),
                body.replace(":3 boundary_conditions", ":4 boundary_conditions"),
                body.replace("boundary_conditions.lateral.west:", "boundary_conditions.lateral.east:"),
                body.replace(gate.PERIODIC_ERROR, "unrelated periodic allocation error"),
                body.replace("[ERROR] v1 config load failed for:", "[INFO] v1 config load failed for:"),
                body.replace("[CONFIG] v1 config load failed", "[ERROR] v1 config load failed"),
                body.replace("(1 error(s), listed above)", "(2 error(s), listed above)"),
                body.replace("[CONFIG]", "[ERROR] unrelated error\n[CONFIG]"),
                "\n".join(body.splitlines()[:-1]),
                body.replace("[ERROR]   ", "[INFO]   "),
            )
            for mutated in mutations:
                with self.subTest(body=mutated):
                    self.assertFalse(gate.matched_periodic_loader_rejection(mutated, config))
            config.write_text(config.read_text().replace("west: periodic", "west: rigid_slip"))
            self.assertFalse(gate.matched_periodic_loader_rejection(body, config))

    def test_periodic_error_lines_are_never_a_general_error_allowlist(self):
        configs = gate.staged_configs()
        chosen = {"metal_qualified": configs["metal_qualified"],
                  "negative_unpaired_periodic": configs["negative_unpaired_periodic"]}
        for status in (1, 2, -11, 139, 0):
            with self.subTest(status=status):
                code, summary, _, _ = self.exercise(
                    configs=chosen, rejected=("negative_unpaired_periodic", status, periodic_loader_body))
                self.assertEqual(code, 0 if status == 1 else 1)
                self.assertEqual(summary["boundedChecksPass"], status == 1)
        body = "[ERROR] " + gate.SCOPE_ERROR
        self.assertFalse(gate.matched_rejection(body, [gate.SCOPE_ERROR]))

    def test_periodic_loader_match_still_requires_no_publication(self):
        configs = gate.staged_configs()
        chosen = {"metal_qualified": configs["metal_qualified"],
                  "negative_unpaired_periodic": configs["negative_unpaired_periodic"]}
        def publish_rejected(name, config, output, watched, binary, body):
            if name == "negative_unpaired_periodic":
                output.mkdir(parents=True)
        code, summary, _, _ = self.exercise(
            configs=chosen, rejected=("negative_unpaired_periodic", 1, periodic_loader_body),
            change=publish_rejected)
        self.assertEqual(code, 1)
        self.assertFalse(summary["boundedChecksPass"])


if __name__ == "__main__":
    unittest.main()
