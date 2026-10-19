"""Bounded native-counter contract mutations and synthetic collector process tests."""

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from scripts.qualification import collect_c6_evidence as evidence
from scripts.qualification import collect_c6_memory_window as memory
from scripts.qualification.provenance import release_manifest


class MemoryWindowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.binary = self.root / "bin/probe"
        self.config = self.root / "configs/fixture.yaml"
        self.fixture = Path(__file__).parent / "fixtures/c6_memory_window_process.py"
        for relative, payload in (("mk/tests.mk", b"fixture roots"), (memory.SOURCE, b"fixture native owner"),
                                  ("configs/fixture.yaml", b"fixture config")):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        self.binary.parent.mkdir()
        self.binary.write_text(f"#!{sys.executable}\n" + self.fixture.read_text())
        self.binary.chmod(0o755)
        self.roots = [Path("mk"), Path("vulkan")]
        self.host = evidence.identity_record({"os": "fixture-os", "architecture": "fixture-arch",
            "cpuModel": "fixture-cpu", "logicalThreads": 4, "ramBytes": 1_000_000,
            "hostname": "fixture-host"}, evidence.HOST_SCHEMA)
        self.device = evidence.identity_record({"name": "synthetic-device", "driver": "synthetic-driver",
            "backendVersion": "synthetic-version"}, evidence.DEVICE_SCHEMA)
        self.index = 0
        self.host_patch = mock.patch.object(evidence, "capture_host", return_value=self.host)
        self.host_patch.start()
        self.addCleanup(self.host_patch.stop)
        self.addCleanup(self.temporary.cleanup)

    def collect(self, extra=None, **options):
        self.index += 1
        output = self.root / f"reports/window{self.index}"
        arguments = ["--headless", "--config", str(self.config), "--compute-backend=vulkan", *(extra or [])]
        return memory.collect(self.root, self.binary, self.config, self.device, [4, 5, 6], 2, 4,
                              arguments, output, _roots=self.roots, **options)

    def read(self, link):
        return memory.read_measurement(self.root, link)

    def rewrite(self, link, mutate):
        path = self.root / link["path"]
        record = json.loads(path.read_text())
        mutate(record)
        path.write_text(json.dumps(record))
        return release_manifest.file_record(path, self.root)

    def events(self):
        link = self.collect()
        measured = self.read(link)
        process = json.loads((self.root / measured["process"]["path"]).read_text())
        text = (self.root / measured["rawLog"]["path"]).read_text()
        records = [json.loads(line[len(memory.PREFIX):]) for line in text.splitlines()]
        return measured, process, records

    def parse(self, measured, process, records):
        return memory.parse_events("\n".join(memory.PREFIX + json.dumps(row) for row in records),
                                  measured["runToken"], process["pid"], 2, 4)

    def test_actual_child_pid_and_raw_archive_bind_measurement(self):
        link = self.collect()
        measured = self.read(link)
        self.assertFalse(measured["releaseAcceptance"])
        self.assertEqual(measured["scienceValidation"], "not_performed")
        self.assertEqual(set(measured["metrics"]), set(memory.METRICS))
        self.assertEqual(measured["metrics"]["device_buffer_extent_bytes"]["highWater"], 128)
        self.assertEqual(measured["window"], memory.window(2, 4))

    def test_native_event_identity_and_types_mutations_reject(self):
        measured, process, records = self.events()
        mutations = {
            "token": lambda row: row.update(runToken="b" * 64),
            "pid": lambda row: row.update(pid=process["pid"] + 1),
            "pid-bool": lambda row: row.update(pid=True),
            "owner": lambda row: row.update(owner="CPUBackend"),
            "source": lambda row: row.update(source="detached.hpp"),
            "scope": lambda row: row.update(scope="simulation_timestep"),
            "aggregation": lambda row: row.update(aggregation="physical_total"),
            "capacity": lambda row: row.update(recordCapacity=3),
            "capacity-bool": lambda row: row.update(recordCapacity=True),
            "first": lambda row: row.update(firstBoundary=1),
            "end": lambda row: row.update(endBoundaryExclusive=5),
            "invalid": lambda row: row.update(valid=False),
            "valid-integer": lambda row: row.update(valid=1),
            "hook-bool": lambda row: row.update(allocationHookEvents=True),
            "hook-overflow": lambda row: row.update(allocationHookEvents=1 << 64),
            "extra": lambda row: row.update(lifetimePeakRssBytes=128),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                changed = copy.deepcopy(records)
                mutate(changed[1])
                with self.assertRaises(memory.MemoryEvidenceError):
                    self.parse(measured, process, changed)

    def test_metric_units_source_bounds_and_compartments_reject(self):
        measured, process, records = self.events()
        for value in (-1, True, 1.0, 1 << 64):
            changed = copy.deepcopy(records)
            changed[1]["metrics"]["device_buffer_extent_bytes"]["highWater"] = value
            with self.subTest(value=value), self.assertRaises(memory.MemoryEvidenceError):
                self.parse(measured, process, changed)
        for field, value in (("unit", "KiB"), ("current", 129), ("begin", 1)):
            changed = copy.deepcopy(records)
            changed[1]["metrics"]["device_buffer_extent_bytes"][field] = value
            with self.subTest(field=field), self.assertRaises(memory.MemoryEvidenceError):
                self.parse(measured, process, changed)
        for action in ("missing", "extra"):
            changed = copy.deepcopy(records)
            if action == "missing":
                changed[1]["metrics"].pop("staging_memory_reserved_bytes")
            else:
                changed[1]["metrics"]["physical_total_bytes"] = {"unit": "bytes", "begin": 0, "current": 0, "highWater": 256}
            with self.subTest(action=action), self.assertRaises(memory.MemoryEvidenceError):
                self.parse(measured, process, changed)

    def test_missing_reversed_duplicate_aborted_events_reject(self):
        measured, process, records = self.events()
        for changed in (records[:1], [], records[::-1], records + records[:1]):
            with self.subTest(length=len(changed)), self.assertRaises(memory.MemoryEvidenceError):
                self.parse(measured, process, changed)
        for key, value in (("completed", False), ("boundary", 3), ("event", "begin")):
            changed = copy.deepcopy(records)
            changed[1][key] = value
            with self.subTest(key=key), self.assertRaises(memory.MemoryEvidenceError):
                self.parse(measured, process, changed)
        text = memory.PREFIX + json.dumps(records[0]).replace('"pid":', '"pid": 1, "pid":', 1)
        with self.assertRaises(memory.MemoryEvidenceError):
            memory.parse_events(text, measured["runToken"], process["pid"], 2, 4)

    def test_invalid_requests_and_resource_bounds_reject(self):
        for first, end in ((0, 2), (2, 2), (3, 2), (True, 3), (1, 1 << 64)):
            with self.subTest(first=first, end=end), self.assertRaises(memory.MemoryEvidenceError):
                memory.window(first, end)
        for options in ({"timeout_s": float("nan")}, {"timeout_s": 0}, {"max_log_bytes": 0}, {"max_log_bytes": True}):
            with self.subTest(options=options), self.assertRaises(memory.MemoryEvidenceError):
                self.collect(**options)

    def test_child_failure_timeout_and_truncation_preserve_negative_receipts(self):
        cases = ((["--exit-failure"], {}, "exitStatus"), (["--hang"], {"timeout_s": .1}, "timedOut"),
                 (["--flood"], {"max_log_bytes": 32}, "truncated"), (["--abort"], {}, None))
        for extra, options, status_key in cases:
            with self.subTest(case=extra), self.assertRaises(memory.MemoryEvidenceError):
                self.collect(extra, **options)
            output = self.root / f"reports/window{self.index}"
            process = json.loads((output / "process.json").read_text())
            if status_key is not None:
                self.assertTrue(process[status_key])
            self.assertFalse((output / "measurement.json").exists())
            if options.get("max_log_bytes"):
                self.assertLessEqual((output / "process.log").stat().st_size, options["max_log_bytes"])

    def test_detached_metric_summary_rejects_after_resealing(self):
        link = self.collect()
        changed = self.rewrite(link, lambda row: row["metrics"]["device_buffer_extent_bytes"].update(highWater=64))
        with self.assertRaises(memory.MemoryEvidenceError):
            self.read(changed)

    def test_process_pid_token_command_and_status_mutations_reject(self):
        link = self.collect()
        original = self.read(link)
        process_link = original["process"]
        process_path = self.root / process_link["path"]
        original_process = process_path.read_text()
        for key, value in (("pid", 1), ("runToken", "a" * 64), ("exitStatus", True),
                           ("timedOut", True), ("truncated", True), ("capturedBytes", 1)):
            process_path.write_text(original_process)
            resealed = self.rewrite(process_link, lambda row, key=key, value=value: row.update({key: value}))
            changed = self.rewrite(link, lambda row: row.update(process=resealed))
            with self.subTest(key=key), self.assertRaises(memory.MemoryEvidenceError):
                self.read(changed)
        process_path.write_text(original_process)
        resealed = self.rewrite(process_link, lambda row: row["command"].extend(["--compute-backend=cpu"]))
        changed = self.rewrite(link, lambda row: row.update(process=resealed))
        with self.assertRaises(memory.MemoryEvidenceError):
            self.read(changed)

    def test_current_source_changes_do_not_rebind_historical_archive(self):
        link = self.collect()
        original = self.read(link)
        (self.root / memory.SOURCE).write_bytes(b"later native producer")
        self.binary.write_text("later binary")
        self.assertEqual(self.read(link), original)
        source_archive = self.root / original["sourceArchive"]["path"]
        with source_archive.open("ab") as stream:
            stream.write(b"changed archive")
        with self.assertRaises(memory.MemoryEvidenceError):
            self.read(link)

    def test_source_drift_during_capture_does_not_publish_measurement(self):
        real_capture = evidence.capture_snapshot
        calls = 0
        def drifting(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                (self.root / memory.SOURCE).write_bytes(b"changed during run")
            return real_capture(*args, **kwargs)
        with mock.patch.object(evidence, "capture_snapshot", side_effect=drifting):
            with self.assertRaises(memory.MemoryEvidenceError):
                self.collect()
        self.assertFalse((self.root / "reports/window1/measurement.json").exists())

    def test_pair_allows_different_binary_without_accepting_science_or_gain(self):
        baseline = self.collect()
        self.binary.write_text(self.binary.read_text() + "\n# separately built candidate\n")
        candidate = self.collect()
        result = memory.compatible_pair(self.root, baseline, candidate)
        self.assertTrue(result["compatibleAllocationWindows"])
        self.assertFalse(result["releaseAcceptance"])
        with self.assertRaises(memory.MemoryEvidenceError):
            memory.compatible_pair(self.root, baseline, baseline)
        self.config.write_bytes(b"different config")
        incompatible = self.collect()
        with self.assertRaises(memory.MemoryEvidenceError):
            memory.compatible_pair(self.root, baseline, incompatible)

    def test_backend_and_config_overrides_reject_before_launch(self):
        for option in ("--compute-backend=cpu", "--config=detached.yaml", "--config", "--compute-backend"):
            with self.subTest(option=option), self.assertRaises(memory.MemoryEvidenceError):
                self.collect([option])

    def test_pair_preserves_execution_window_arguments(self):
        baseline = self.collect(["--write-every=0"])
        candidate = self.collect(["--write-every=1"])
        with self.assertRaises(memory.MemoryEvidenceError):
            memory.compatible_pair(self.root, baseline, candidate)

    def test_historical_readback_uses_producer_directory_after_artifact_move(self):
        link = self.collect()
        original = self.read(link)
        with tempfile.TemporaryDirectory() as copied:
            copied_root = (Path(copied) / "checkout").resolve()
            shutil.copytree(self.root, copied_root)
            self.assertEqual(memory.read_measurement(copied_root, link), original)


if __name__ == "__main__":
    unittest.main()
