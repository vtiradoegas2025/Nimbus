#!/usr/bin/env python3
"""Static coverage for Windows-only topology and OpenMP authority contracts."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HARDWARE = (ROOT / "src/core/infra/hardware_info.cpp").read_text(encoding="utf-8")
RUNTIME = (ROOT / "src/core/runtime/nimbus_sim.cpp").read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"windows CPU topology contract failed: {message}")


def main() -> int:
    topology_contracts = (
        "GetActiveProcessorCount(ALL_PROCESSOR_GROUPS)",
        "GetActiveProcessorGroupCount()",
        "GetLogicalProcessorInformationEx(RelationAll",
        "record->Relationship == RelationProcessorCore",
        "processor.GroupMask[group].Mask",
        "processor.EfficiencyClass",
        "record->Relationship == RelationGroup",
        "groups.GroupInfo[group].ActiveProcessorCount",
        "classes.performance_processors",
    )
    for contract in topology_contracts:
        require(contract in HARDWARE, f"missing Win32 contract: {contract}")

    start = RUNTIME.index("bool apply_thread_cli_override")
    end = RUNTIME.index("void apply_pre_config_cli_overrides", start)
    policy = RUNTIME[start:end]

    cli_authority = policy.index("if (cli.threads_from_cli)")
    environment_authority = policy.index('std::getenv("OMP_NUM_THREADS")')
    topology_default = policy.index("topology_aware_openmp_default")
    topology_application = policy.index("omp_set_num_threads(topology_default)")
    require(
        cli_authority
        < environment_authority
        < topology_default
        < topology_application,
        "thread authority must remain CLI > OMP_NUM_THREADS > topology default",
    )

    automatic_policy = policy[environment_authority:]
    require(
        "omp_set_dynamic" not in automatic_policy,
        "automatic topology selection must not override OMP_DYNAMIC",
    )

    print("[windows-cpu-topology] PASS: topology APIs and thread authority are intact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
