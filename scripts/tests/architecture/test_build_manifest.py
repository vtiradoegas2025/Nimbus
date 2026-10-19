#!/usr/bin/env python3
"""Verify explicit source ownership and default build artifacts."""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[3]


def probe_environment() -> dict[str, str]:
    # Matrix probes must not inherit the parent make invocation's profile.
    inherited = {"MAKEFLAGS", "MFLAGS", "MAKEOVERRIDES", "MAKELEVEL", "CPU_ONLY",
                 "PORTABLE", "DISTRIBUTION", "CHECKED", "LTO", "SANITIZE", "BIN", "BUILDDIR"}
    return {key: value for key, value in os.environ.items() if key not in inherited}


def make_lines(target: str) -> list[str]:
    result = subprocess.run(
        [
            "make",
            "--no-print-directory",
            "-s",
            "REQUIRE_FULL_STACK=0",
            "CPU_ONLY=1",
            target,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=probe_environment(),
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def benchmark_profile_failures() -> list[str]:
    profiles = {
        "default": ([], "bin/nimbus_sim_cpu_benchmark", "build/benchmark_cpu"),
        "portable": (["PORTABLE=1"], "bin/nimbus_sim_cpu_portable_benchmark", "build/benchmark_cpu_portable"),
        "cpu_only": (["CPU_ONLY=1"], "bin/nimbus_sim_cpu_only_benchmark", "build/benchmark_cpu_only"),
        "cpu_only_portable": (["CPU_ONLY=1", "PORTABLE=1"],
                              "bin/nimbus_sim_cpu_only_portable_benchmark", "build/benchmark_cpu_only_portable"),
    }
    failures = []
    identities = []
    for name, (profile, artifact, object_root) in profiles.items():
        command = ["make", "--no-print-directory", "-qp", "REQUIRE_FULL_STACK=0", *profile, "help"]
        database = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30,
                                  env=probe_environment())
        if database.returncode not in (0, 1):
            failures.append(f"benchmark profile {name} database failed: {database.stderr}")
            continue
        variables = {}
        for line in database.stdout.splitlines():
            match = re.match(r"^([A-Z][A-Z0-9_]*)\s*:?=\s*(.*)$", line)
            if match:
                variables[match.group(1)] = match.group(2)
        observed = tuple(variables.get(key) for key in
                         ("BENCHMARK_BIN", "BENCHMARK_BUILDDIR", "BENCHMARK_FLAGS_STAMP"))
        expected = (artifact, object_root, object_root + "/.build-flags")
        if observed != expected:
            failures.append(f"benchmark profile {name} artifact/object/stamp differs: {observed}")
        identities.append(observed)
        flags = variables.get("BENCHMARK_CXXFLAGS", "")
        portable = "PORTABLE=1" in profile
        if ("-DNIMBUS_PORTABLE_BUILD" in flags) != portable or (portable and "-march=native" in flags):
            failures.append(f"benchmark profile {name} portable flag identity differs")
        object_path = object_root + "/src/core/infra/coordinate_system.o"
        dry_run = subprocess.run(["make", "--no-print-directory", "-n", "REQUIRE_FULL_STACK=0", *profile, object_path],
                                 cwd=ROOT, capture_output=True, text=True, timeout=30,
                                 env=probe_environment())
        if dry_run.returncode:
            failures.append(f"benchmark profile {name} bounded object dry-run failed: {dry_run.stderr}")
    if len(identities) != 4 or any(len({identity[index] for identity in identities}) != 4 for index in range(3)):
        failures.append("native and portable benchmark artifact/object/stamp identities alias")
    return failures


def main() -> int:
    simulator = make_lines("print-simulator-sources")
    simulator_objects = make_lines("print-simulator-objects")
    validator = make_lines("print-field-validator-sources")
    opt_in = make_lines("print-opt-in-sources")
    artifacts = set(make_lines("print-default-artifacts"))
    failures: list[str] = []

    duplicates = sorted(
        source for source, count in Counter(simulator).items() if count != 1
    )
    if duplicates:
        failures.append("duplicate simulator sources: " + ", ".join(duplicates))

    expected_objects = {
        f"build/cpu_only/{source[:-4]}.o" for source in simulator
    }
    if set(simulator_objects) != expected_objects:
        failures.append("layered object groups do not match simulator sources")

    discovered = {
        path.relative_to(ROOT).as_posix() for path in (ROOT / "src").rglob("*.cpp")
    }
    declared = {
        source
        for source in simulator + validator + opt_in
        if source.startswith("src/")
    }
    missing = sorted(discovered - declared)
    stale = sorted(declared - discovered)
    if missing:
        failures.append("unclassified sources: " + ", ".join(missing))
    if stale:
        failures.append("declared sources do not exist: " + ", ".join(stale))

    expected_overlap = {
        "src/diagnostics/contracts/field_contract.cpp",
        "src/diagnostics/contracts/field_validation.cpp",
    }
    overlap = set(simulator) & set(validator)
    if overlap != expected_overlap:
        failures.append(
            "unexpected simulator/validator overlap: "
            + ", ".join(sorted(overlap ^ expected_overlap))
        )

    if set(opt_in) & (set(simulator) | set(validator)):
        failures.append("opt-in sources leaked into a default artifact")

    required_artifacts = {"bin/nimbus_sim_cpu", "bin/field_validator"}
    if not required_artifacts <= artifacts:
        failures.append(
            "default build omits: "
            + ", ".join(sorted(required_artifacts - artifacts))
        )

    manifest = (ROOT / "mk/sources.mk").read_text(encoding="utf-8")
    for forbidden in ("$(wildcard", "$(shell", "find src"):
        if forbidden in manifest:
            failures.append(f"source manifest uses discovery expression: {forbidden}")

    # The root file is intentionally a small include map. Validate contracts
    # across the complete modular build definition rather than coupling this
    # architecture gate to one physical Makefile.
    build_files = [ROOT / "Makefile", *sorted((ROOT / "mk").glob("*.mk"))]
    makefile = "\n".join(
        path.read_text(encoding="utf-8") for path in build_files
    )
    required_contracts = (
        "SOURCE_MANIFEST := mk/sources.mk",
        "$(BIN): $(OBJS) $(SOURCE_MANIFEST)",
        "$(FIELD_VALIDATOR): $(FIELD_VALIDATOR_OBJS) $(SOURCE_MANIFEST)",
        "TEST_INFRA := tests/test_main.cpp tests/test_harness.cpp src/microphysics/grid_geometry.cpp $(TEST_HDRS)",
        "$(SOURCE_MANIFEST) $(TEST_FLAGS_STAMP)",
        "$(BUILDDIR)/%.o: %.cpp $(MAIN_FLAGS_STAMP)",
        "$(MAIN_FLAGS_STAMP): FORCE",
        "$(TEST_FLAGS_STAMP): FORCE",
        "CORE_LAYER_OBJS :=",
        "MODULE_OBJS :=",
    )
    for contract in required_contracts:
        if contract not in makefile:
            failures.append(f"build graph is not manifest-sensitive: {contract}")

    failures.extend(benchmark_profile_failures())

    if failures:
        for failure in failures:
            print(f"[build-manifest] FAIL: {failure}")
        return 1

    print(
        f"[build-manifest] PASS: {len(discovered)} src files classified; "
        f"{len(simulator_objects)} layered simulator objects; default tools included; "
        "four benchmark profiles isolated"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
