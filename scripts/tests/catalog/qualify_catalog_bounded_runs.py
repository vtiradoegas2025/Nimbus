#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.qualification.catalog_inventory import INVENTORY, maintained_cases, inventory_counts
from scripts.tests.catalog.test_kernel_ownership_catalog import validate as validate_kernel_catalog
DEFAULT_BINARY = ROOT / "bin/nimbus_sim"
DEFAULT_KERNEL_OWNERSHIP_TOOL = ROOT / "bin/nimbus_kernel_ownership"
DEFAULT_OPERATION_REQUIREMENTS = (
    ROOT / "tests/fixtures/reference/catalog/backend_tuple_operation_requirements_v1.json"
)
QUALIFICATION = ROOT / "tests/fixtures/configs/qualification"
PUBLIC_REPRESENTATIVES = ROOT / "configs/1000meterResolution"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {
        "path": resolved.relative_to(ROOT).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def configured_backend(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("backend:"):
            return stripped.split(":", 1)[1].strip().lower()
    return "cpu"


def requires_moving_reference_window(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    return "preset: moving_reference_v1" in source


def command_for(
    binary: Path,
    config: Path,
    output: Path,
    force_history: bool,
    operation_only: bool,
) -> list[str]:
    duration = 30 if requires_moving_reference_window(config) else 1
    command = [
        str(binary), "--headless", "--config", str(config),
        f"--duration={duration}",
        f"--outdir={output}", "--threads=1", "--log-profile=normal",
    ]
    if operation_only:
        command.extend(["--write-every=0", "--checkpoint-every=0"])
    elif force_history:
        command.extend(["--write-every=1", "--checkpoint-every=0"])
    backend = configured_backend(config)
    if backend != "cpu":
        command.append("--timing")
    if backend == "vulkan":
        return [str(ROOT / "scripts/run_with_vulkan_env.sh"), *command]
    return command


def dispatch_counts(output: str, backend: str) -> tuple[int, dict[str, int]]:
    reports = 0
    counts: dict[str, int] = {}
    for line in output.splitlines():
        if "[COMPUTE DISPATCH]" not in line:
            continue
        match = re.search(r"\[COMPUTE DISPATCH\] backend=([^,]+).*kernels=\{(.*)\}$", line)
        if match is None:
            raise RuntimeError(f"malformed compute dispatch report: {line}")
        if match.group(1) != backend:
            raise RuntimeError(
                f"compute dispatch backend {match.group(1)} differs from configured {backend}"
            )
        reports += 1
        for item in match.group(2).split(","):
            if not item:
                continue
            kernel, separator, count_text = item.rpartition(":")
            if not separator or not kernel or not count_text.isdigit():
                raise RuntimeError(f"malformed kernel dispatch entry: {item}")
            count = int(count_text)
            if count <= 0:
                raise RuntimeError(f"kernel dispatch count is not positive: {item}")
            counts[kernel] = counts.get(kernel, 0) + count
    return reports, counts


def require_native_process_chain(config: Path, counts: dict[str, int]) -> None:
    source = config.read_text(encoding="utf-8")
    selected = re.search(r"(?m)^microphysics:\n  scheme: (\w+)\s*$", source)
    if selected is None:
        raise RuntimeError("maintained native fixture must name its microphysics scheme explicitly")
    scheme = selected[1]
    processes = {
        "none": (),
        "kessler": ("pointwise", "sedimentation"),
        "lin": ("pointwise", "sedimentation"),
        "milbrandt": ("cloud_adjustment", "warm_rain", "cold", "melting", "sedimentation"),
        "thompson": ("cloud_adjustment", "warm_rain", "cloud_phase", "freezing", "deposition",
                     "completion", "moment_sedimentation"),
    }
    if scheme not in processes:
        raise RuntimeError(f"native process ownership policy is missing for {scheme}")
    expected = {f"microphysics/{scheme}_{process}.comp" for process in processes[scheme]}
    observed = {name for name in counts if name.startswith("microphysics/")}
    if expected != observed or any(type(counts[name]) is not int or counts[name] <= 0 for name in expected):
        raise RuntimeError(f"native {scheme} process ownership differs: expected={sorted(expected)}, "
                           f"observed={sorted(observed)}")
    if scheme == "none":
        return
    stages = {name: counts[f"microphysics/{scheme}_{name}.comp"] for name in processes[scheme]}
    timestep = re.search(r"(?m)^  dt_large: ([^\n]+)$", source)
    if timestep is None or "  adaptive_dt: false" not in source.splitlines():
        raise RuntimeError("native process step accounting requires a declared fixed timestep")
    dt = float(timestep[1])
    if not math.isfinite(dt) or dt <= 0:
        raise RuntimeError("native process step accounting has an invalid timestep")
    duration = 30 if requires_moving_reference_window(config) else 1
    parent_steps = math.ceil(math.nextafter(duration / dt, -math.inf))
    nested = re.search(r"(?m)^  nesting:\n((?:    .*\n|\n)*)", source)
    ratio = 0
    if nested and "    enabled: true" in nested[1].splitlines():
        selected_ratio = re.search(r"(?m)^    parent_time_step_ratio: (\d+)\s*$", nested[1])
        if selected_ratio is None or int(selected_ratio[1]) <= 0:
            raise RuntimeError("native process step accounting has no child subcycling ratio")
        ratio = int(selected_ratio[1])
    reference = parent_steps * (1 + ratio)
    for process, count in stages.items():
        if process == "completion":
            if count < reference:
                raise RuntimeError("native Thompson completion omitted a physics step")
        elif "sedimentation" in process and scheme in ("thompson", "milbrandt"):
            if count != 5 * reference:
                raise RuntimeError(f"native {scheme} sedimentation omitted a species or physics step")
        elif count != reference:
            raise RuntimeError(f"native {scheme} pointwise process counts disagree")


def require_bounded_run(
    binary: Path,
    config: Path,
    output: Path,
    active_kernels: dict[str, str],
    operation_only: bool,
    environment: dict[str, str] | None = None,
) -> dict[str, object] | None:
    force_history = not requires_moving_reference_window(config)
    result = subprocess.run(
        command_for(binary, config, output, force_history, operation_only), cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{config.relative_to(ROOT)} failed:\n{result.stdout}")
    for marker in (
        "[SCIENTIFIC INVALIDITY]", "Science profile rejected",
        "[CONFIG ERROR]", "fallback=active",
    ):
        if marker in result.stdout:
            raise RuntimeError(
                f"{config.relative_to(ROOT)} reported {marker}:\n{result.stdout}"
            )
    backend = configured_backend(config)
    if backend in ("vulkan", "metal"):
        expected = f"[COMPUTE] backend={backend}"
        if expected not in result.stdout or "fallback=off" not in result.stdout:
            raise RuntimeError(
                f"{config.relative_to(ROOT)} omitted native backend proof:\n"
                f"{result.stdout}"
            )
        reports, kernels = dispatch_counts(result.stdout, backend)
        if reports == 0 or not kernels:
            raise RuntimeError(
                f"{config.relative_to(ROOT)} published no non-vacuous kernel dispatch set"
            )
        unknown = set(kernels) - set(active_kernels)
        if unknown:
            raise RuntimeError(
                f"{config.relative_to(ROOT)} dispatched unknown or inactive kernels: "
                f"{sorted(unknown)}"
            )
        require_native_process_chain(config, kernels)
        operation_ids = sorted(active_kernels[kernel] for kernel in kernels)
        observation: dict[str, object] | None = {
            "config": str(config.relative_to(ROOT)),
            "configSha256": sha256_file(config),
            "backend": backend,
            "durationSeconds": 30 if requires_moving_reference_window(config) else 1,
            "dispatchReports": reports,
            "kernelCounts": dict(sorted(kernels.items())),
            "operationIds": operation_ids,
        }
    else:
        observation = None
    if force_history and not operation_only:
        manifest = output / "step_000001/manifest.json"
        if not manifest.is_file() or '"publication_state": "complete"' not in manifest.read_text(
            encoding="utf-8"
        ):
            raise RuntimeError(
                f"{config.relative_to(ROOT)} did not publish a completed one-second frame"
            )
    return observation


def require_operation_counts(observed: dict[str, int], requirement: dict[str, object]) -> None:
    exact = requirement["kernelCounts"]
    solver = requirement.get("implicitSolver")
    if solver is None:
        if observed != exact:
            raise RuntimeError(f"operation dispatch counts differ: expected={exact}, observed={observed}")
        return
    stencil = "diffusion/implicit_diffusion_stencil.comp"
    vector = "diffusion/implicit_diffusion_vector.comp"
    if (not isinstance(solver, dict) or set(solver) != {"solves", "maximumIterations"}
            or type(solver["solves"]) is not int or solver["solves"] <= 0
            or solver["maximumIterations"] != 4096
            or set(observed) != set(exact) | {stencil, vector}
            or any(observed[name] != count for name, count in exact.items())):
        raise RuntimeError("implicit operation ownership or fixed dispatch counts differ")
    solves = solver["solves"]
    updates = observed[vector] - 2 * solves
    iterations = updates // 2
    verifications = observed[stencil] - solves - iterations
    if (updates < 0 or updates % 2 or iterations > solves * solver["maximumIterations"]
            or verifications < 0 or verifications > iterations
            or (iterations > 0 and verifications == 0)):
        raise RuntimeError("implicit solver dispatches violate the iteration and residual-check contract")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument(
        "--case-kind", choices=("all", "cpu", "accelerator"), default="all",
        help="run the complete inventory or one exact backend partition",
    )
    parser.add_argument(
        "--kernel-ownership-tool", type=Path,
        default=DEFAULT_KERNEL_OWNERSHIP_TOOL,
    )
    parser.add_argument("--operation-output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--operation-only", action="store_true")
    parser.add_argument(
        "--operation-requirements", type=Path,
        default=DEFAULT_OPERATION_REQUIREMENTS,
    )
    args = parser.parse_args()
    if args.operation_output is not None and not args.operation_only:
        parser.error("--operation-output requires --operation-only")
    if args.operation_only and (
        args.case_kind != "accelerator" or args.operation_output is None
    ):
        raise RuntimeError(
            "--operation-only requires --case-kind=accelerator and --operation-output"
        )
    binary = args.binary.resolve()
    if not binary.is_file():
        raise RuntimeError(f"missing simulator: {binary}")
    binary_sha256 = sha256_file(binary)
    script_path = Path(__file__).resolve()
    script_sha256 = sha256_file(script_path)
    started = time.monotonic()
    kernel_tool = args.kernel_ownership_tool.resolve()
    kernel_tool_sha256 = ""
    active_kernels: dict[str, str] = {}
    operation_requirements: dict[str, dict[str, object]] = {}
    requirements_path = args.operation_requirements.resolve()
    requirements_sha256 = ""
    if args.case_kind in ("all", "accelerator"):
        if not kernel_tool.is_file():
            raise RuntimeError(f"missing kernel ownership tool: {kernel_tool}")
        kernel_tool_sha256 = sha256_file(kernel_tool)
        catalog_result = subprocess.run(
            [str(kernel_tool)], cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
        if catalog_result.returncode != 0:
            raise RuntimeError(
                "kernel ownership tool failed:\n" +
                catalog_result.stderr.decode("utf-8", errors="replace")
            )
        catalog = json.loads(catalog_result.stdout)
        validate_kernel_catalog(catalog)
        active_kernels = {
            record["kernelName"]: record["operationId"]
            for record in catalog["records"] if record["availability"] == "active"
        }
    if args.operation_only:
        if not requirements_path.is_file():
            raise RuntimeError(f"missing operation requirements: {requirements_path}")
        requirements_sha256 = sha256_file(requirements_path)
        requirement_document = json.loads(requirements_path.read_text(encoding="utf-8"))
        requirement_records = requirement_document.get("records")
        run_contract = requirement_document.get("runContract")
        if (
            requirement_document.get("schema") != "nimbus.backend-tuple-operation-requirements.v1"
            or requirement_document.get("version") != 1
            or not isinstance(requirement_records, list)
            or run_contract != {
                "ordinaryDurationSeconds": 1,
                "movingReferenceDurationSeconds": 30,
                "writeEverySeconds": 0,
                "checkpointEverySeconds": 0,
                "threads": 1,
                "timingTelemetry": True,
            }
        ):
            raise RuntimeError("operation requirement document is missing or malformed")
        for record in requirement_records:
            if not isinstance(record, dict):
                raise RuntimeError("operation requirement record is malformed")
            config_name = record.get("config")
            backend = record.get("backend")
            counts = record.get("kernelCounts")
            if (
                not isinstance(config_name, str)
                or config_name in operation_requirements
                or backend not in ("vulkan", "metal")
                or not isinstance(counts, dict)
                or not counts
                or any(
                    kernel not in active_kernels or not isinstance(count, int) or count <= 0
                    for kernel, count in counts.items()
                )
            ):
                raise RuntimeError(f"invalid operation requirement record: {record}")
            config_path = ROOT / config_name
            if not config_path.is_file() or record.get("configSha256") != sha256_file(config_path):
                raise RuntimeError(f"operation requirement config identity drifted: {config_name}")
            if configured_backend(config_path) != backend:
                raise RuntimeError(f"operation requirement backend drifted: {config_name}")
            operation_requirements[config_name] = record
        expected_accelerators = inventory_counts()["acceleratorCases"]
        if len(operation_requirements) != expected_accelerators:
            raise RuntimeError(
                f"operation requirements expected {expected_accelerators} accelerator fixtures, got {len(operation_requirements)}"
            )

    qualified, representatives = maintained_cases()
    config_identities = {path: sha256_file(path) for path in qualified + representatives}
    inventory_identity = sha256_file(INVENTORY)

    shader_identities = {}
    with tempfile.TemporaryDirectory(prefix="nimbus-catalog-bounded-") as temporary:
        root = Path(temporary)
        run_binary = root / "nimbus_sim"
        shutil.copy2(binary, run_binary)
        if sha256_file(run_binary) != binary_sha256:
            raise RuntimeError("failed to create an exact private simulator copy")
        environment = os.environ.copy()
        if args.case_kind in ("all", "accelerator"):
            source_shaders = ROOT / "vulkan/shaders/compute"
            for source in sorted(source_shaders.rglob("*.spv")):
                relative = source.relative_to(source_shaders)
                shader_identities[relative.as_posix()] = sha256_file(source)
                destination = root / "shaders" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                if sha256_file(destination) != shader_identities[relative.as_posix()]:
                    raise RuntimeError("shader changed while creating its private execution copy")
            if not shader_identities:
                raise RuntimeError("native catalog gate has no shader artifacts")
            environment["TMV_SHADER_DIR"] = str(root / "shaders")
            environment["NIMBUS_GPU_KERNEL_TIMING"] = "1"

        def qualify(index: int, config: Path) -> dict[str, object] | None:
            output = root / f"case-{index:03d}"
            observation = require_bounded_run(
                run_binary, config, output, active_kernels, args.operation_only, environment,
            )
            shutil.rmtree(output, ignore_errors=True)
            return observation

        cases = list(enumerate(qualified + representatives))
        cpu_cases = [
            case for case in cases if configured_backend(case[1]) == "cpu"
        ]
        accelerator_cases = [
            case for case in cases if configured_backend(case[1]) != "cpu"
        ]
        selected_cpu_cases = cpu_cases if args.case_kind in ("all", "cpu") else []
        selected_accelerator_cases = (
            accelerator_cases
            if args.case_kind in ("all", "accelerator") else []
        )
        completed_cpu = 0
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(qualify, index, config): config
                for index, config in selected_cpu_cases
            }
            for future in as_completed(futures):
                try:
                    future.result()
                    completed_cpu += 1
                    if completed_cpu % 20 == 0:
                        print(f"[catalog-progress] CPU {completed_cpu}/{len(selected_cpu_cases)}", flush=True)
                except Exception as error:
                    raise RuntimeError(
                        f"bounded catalog case failed for "
                        f"{futures[future].relative_to(ROOT)}"
                    ) from error
        observations: list[dict[str, object]] = []
        for index, config in selected_accelerator_cases:
            observation = qualify(index, config)
            if observation is not None:
                if args.operation_only:
                    config_name = observation["config"]
                    requirement = operation_requirements.get(config_name)
                    if requirement is None:
                        raise RuntimeError(f"no operation requirement for {config_name}")
                    if observation["backend"] != requirement["backend"]:
                        raise RuntimeError(f"operation backend differs for {config_name}")
                    if observation["configSha256"] != requirement["configSha256"]:
                        raise RuntimeError(f"operation config identity differs for {config_name}")
                    require_operation_counts(observation["kernelCounts"], requirement)
                observations.append(observation)
                if len(observations) % 20 == 0:
                    print(f"[catalog-progress] accelerator {len(observations)}/{len(selected_accelerator_cases)}",
                          flush=True)
        if sha256_file(run_binary) != binary_sha256 or any(
            sha256_file(root / "shaders" / name) != identity
            for name, identity in shader_identities.items()
        ):
            raise RuntimeError("private catalog execution inputs changed during the gate")

    if sha256_file(INVENTORY) != inventory_identity or any(
        sha256_file(path) != digest for path, digest in config_identities.items()
    ):
        raise RuntimeError("catalog configuration inventory changed during execution")
    if sha256_file(binary) != binary_sha256:
        raise RuntimeError("simulator changed while the bounded catalog gate ran")
    if kernel_tool_sha256 and sha256_file(kernel_tool) != kernel_tool_sha256:
        raise RuntimeError("kernel ownership tool changed while the bounded catalog gate ran")
    if requirements_sha256 and sha256_file(requirements_path) != requirements_sha256:
        raise RuntimeError("operation requirements changed while the bounded catalog gate ran")
    if sha256_file(script_path) != script_sha256:
        raise RuntimeError("bounded catalog runner changed while the gate ran")

    if args.operation_output is not None:
        if len(observations) != len(accelerator_cases):
            raise RuntimeError(
                "operation evidence requires the complete accelerator partition: "
                f"observed={len(observations)}, expected={len(accelerator_cases)}"
            )
        if {record["config"] for record in observations} != set(operation_requirements):
            raise RuntimeError("observed accelerator fixtures differ from operation requirements")
        operation_output = args.operation_output.resolve()
        operation_output.parent.mkdir(parents=True, exist_ok=True)
        evidence = {
            "schema": "nimbus.backend-tuple-operation-observation.v1",
            "version": 1,
            "binary": {
                "path": str(binary.relative_to(ROOT)),
                "sha256": binary_sha256,
            },
            "kernelOwnershipTool": {
                "path": str(kernel_tool.relative_to(ROOT)),
                "sha256": kernel_tool_sha256,
            },
            "operationRequirements": {
                "path": str(requirements_path.relative_to(ROOT)),
                "sha256": requirements_sha256,
            },
            "records": observations,
            "shaderSha256": shader_identities,
            "summary": {
                "acceleratorFixtures": len(observations),
                "vulkanFixtures": sum(record["backend"] == "vulkan" for record in observations),
                "metalFixtures": sum(record["backend"] == "metal" for record in observations),
                "observedOperationIds": sorted({
                    operation
                    for record in observations
                    for operation in record["operationIds"]
                }),
            },
            "claimBoundary": (
                "Exact required and observed bounded-window dispatch sets for maintained "
                "accelerator fixtures; this is not universal accepted-tuple execution or a long campaign."
            ),
        }
        write_json_atomic(operation_output, evidence)

    if args.report is not None:
        selected_cases = selected_cpu_cases + selected_accelerator_cases
        report = {
            "schema": "nimbus.catalog-bounded-partition-evidence.v1",
            "version": 1,
            "status": "pass",
            "completedUtc": datetime.now(timezone.utc).isoformat(),
            "elapsedWallSeconds": time.monotonic() - started,
            "binary": file_record(binary),
            "runner": file_record(script_path),
            "shaderSha256": shader_identities,
            "partition": args.case_kind,
            "runContract": {
                "cpuWorkers": 6,
                "threadsPerCase": 1,
                "ordinaryDurationSeconds": 1,
                "movingReferenceDurationSeconds": 30,
                "ordinaryPublicationRequired": not args.operation_only,
            },
            "summary": {
                "qualifiedFixtures": len(qualified),
                "publicRepresentatives": len(representatives),
                "catalogCases": len(cases),
                "cpuCases": len(cpu_cases),
                "acceleratorCases": len(accelerator_cases),
                "completed": len(selected_cases),
            },
            "cases": [
                {
                    "index": index,
                    "path": config.relative_to(ROOT).as_posix(),
                    "sha256": sha256_file(config),
                    "backend": configured_backend(config),
                    "durationSeconds": 30 if requires_moving_reference_window(config) else 1,
                }
                for index, config in selected_cases
            ],
            "claimBoundary": (
                "Genuine bounded execution for the selected maintained catalog partition; "
                "not universal generated-tuple execution, a long campaign, or external review."
            ),
        }
        write_json_atomic(args.report.resolve(), report)

    print(
        "[qualify-catalog-bounded-runs] PASS: completed the selected genuine "
        f"bounded-window partition from {len(qualified)} qualified fixtures "
        f"and {len(representatives)} shipped categorical preset families with "
        "six isolated CPU workers "
        "and serialized accelerator ownership (one-second ordinary cases; "
        "30-second exact moving-reference cases); exact binary sha256="
        f"{binary_sha256}; case_kind={args.case_kind}; "
        f"completed={len(selected_cpu_cases) + len(selected_accelerator_cases)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
