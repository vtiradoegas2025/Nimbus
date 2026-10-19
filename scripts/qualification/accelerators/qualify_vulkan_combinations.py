#!/usr/bin/env python3
"""Fail-closed physical Vulkan matrix for admitted flat C-grid ownership tuples."""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
from typing import Iterable


SLOW_TENDENCY_STAGE = "dynamics/cartesian_cgrid_slow_tendencies.comp.spv"
SLOW_APPLY_STAGE = "dynamics/slow_apply.comp.spv"
SGS_TENDENCY_STAGE = "smagorinsky_fused"
SGS_APPLY_STAGE = "turbulence/sgs_apply.comp.spv"
MICROPHYSICS_APPLY_STAGE = "microphysics_apply"
LIN_STAGE = "lin_pointwise_sedimentation_batch"
KESSLER_STAGES = (
    "microphysics/kessler_pointwise.comp.spv",
    "microphysics/kessler_sedimentation.comp.spv",
)
DEVICE_PATTERN = re.compile(
    r'\[COMPUTE\]\[VULKAN\] selected device\[([0-9]+)\] "([^"]+)"'
)


@dataclass(frozen=True)
class Case:
    microphysics: str
    turbulence: str
    guard_mode: str
    apply_microphysics: bool
    apply_slow: bool
    apply_sgs: bool

    @property
    def name(self) -> str:
        return (
            f"mp-{self.microphysics}_sgs-{self.turbulence}"
            f"_guard-{self.guard_mode}"
            f"_apply-{int(self.apply_microphysics)}"
            f"{int(self.apply_slow)}{int(self.apply_sgs)}"
        )

    @property
    def admitted(self) -> bool:
        return (
            self.microphysics in ("none", "kessler", "lin")
            and self.turbulence in ("none", "smagorinsky")
        )


def cases() -> tuple[Case, ...]:
    values: list[Case] = []
    for microphysics in (
        "none", "kessler", "lin", "thompson", "milbrandt",
    ):
        for turbulence in ("none", "smagorinsky", "tke"):
            microphysics_controls: Iterable[bool] = (
                (False, True)
                if microphysics in ("kessler", "lin") else (False,)
            )
            sgs_controls: Iterable[bool] = (
                (False, True) if turbulence == "smagorinsky" else (False,)
            )
            for guard_mode in ("off", "warn", "error"):
                for apply_microphysics in microphysics_controls:
                    for apply_slow in (False, True):
                        for apply_sgs in sgs_controls:
                            values.append(Case(
                                microphysics=microphysics,
                                turbulence=turbulence,
                                guard_mode=guard_mode,
                                apply_microphysics=apply_microphysics,
                                apply_slow=apply_slow,
                                apply_sgs=apply_sgs,
                            ))
    return tuple(values)


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(
            f"base config drift: expected one occurrence of {old!r}, "
            f"found {text.count(old)}"
        )
    return text.replace(old, new, 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def render_config(base: str, case: Case) -> str:
    text = replace_once(
        base, "microphysics:\n  scheme: lin",
        f"microphysics:\n  scheme: {case.microphysics}",
    )
    text = replace_once(
        text, "turbulence:\n  scheme: smagorinsky",
        f"turbulence:\n  scheme: {case.turbulence}",
    )
    text = replace_once(
        text, "    backend: gpu\n",
        "    backend: gpu\n    allow_fallback: false\n",
    )
    text = replace_once(
        text, "    gpu_apply_microphysics: true",
        "    gpu_apply_microphysics: "
        + ("true" if case.apply_microphysics else "false"),
    )
    text = replace_once(
        text, "    gpu_apply_slow: true",
        "    gpu_apply_slow: " + ("true" if case.apply_slow else "false"),
    )
    text = replace_once(
        text, "    gpu_apply_sgs: true",
        "    gpu_apply_sgs: " + ("true" if case.apply_sgs else "false"),
    )
    return text


def parse_stage_dispatches(output: str) -> dict[str, int]:
    stages: dict[str, int] = {}
    pattern = re.compile(
        r"^\[GPU STAGE TELEMETRY\] (.+?) command_buffers=[0-9]+ "
        r"dispatches=([0-9]+)\b"
    )
    for line in output.splitlines():
        match = pattern.match(line)
        if match:
            stages[match.group(1)] = int(match.group(2))
    return stages


def require_stage(stages: dict[str, int], stage: str, case: Case) -> None:
    if stages.get(stage, 0) <= 0:
        observed = ", ".join(sorted(stages)) or "<none>"
        raise RuntimeError(
            f"{case.name}: required stage did not execute: {stage}; "
            f"observed stages: {observed}"
        )


def forbid_stage(stages: dict[str, int], stage: str, case: Case) -> None:
    if stages.get(stage, 0) != 0:
        raise RuntimeError(
            f"{case.name}: disabled/inactive stage unexpectedly executed: {stage}"
        )


def verify_stages(stages: dict[str, int], case: Case) -> None:
    require_stage(stages, SLOW_TENDENCY_STAGE, case)
    slow_dispatch = case.apply_slow and case.guard_mode == "off"
    (require_stage if slow_dispatch else forbid_stage)(
        stages, SLOW_APPLY_STAGE, case
    )

    if case.microphysics == "lin":
        require_stage(stages, LIN_STAGE, case)
        for stage in KESSLER_STAGES:
            forbid_stage(stages, stage, case)
    elif case.microphysics == "kessler":
        forbid_stage(stages, LIN_STAGE, case)
        for stage in KESSLER_STAGES:
            require_stage(stages, stage, case)
    else:
        forbid_stage(stages, LIN_STAGE, case)
        for stage in KESSLER_STAGES:
            forbid_stage(stages, stage, case)
    microphysics_dispatch = (
        case.apply_microphysics and case.guard_mode == "off"
    )
    (require_stage if microphysics_dispatch else forbid_stage)(
        stages, MICROPHYSICS_APPLY_STAGE, case
    )

    if case.turbulence == "smagorinsky":
        require_stage(stages, SGS_TENDENCY_STAGE, case)
    else:
        forbid_stage(stages, SGS_TENDENCY_STAGE, case)
    (require_stage if case.apply_sgs else forbid_stage)(
        stages, SGS_APPLY_STAGE, case
    )


def parse_total_telemetry(output: str, case: Case) -> dict[str, int]:
    lines = [line for line in output.splitlines() if line.startswith("[GPU TELEMETRY]")]
    if not lines:
        raise RuntimeError(f"{case.name}: missing final GPU telemetry")
    values: dict[str, int] = {}
    for key in ("model_steps", "command_buffers", "dispatches"):
        match = re.search(rf"\b{key}=([0-9]+)\b", lines[-1])
        if match is None or int(match.group(1)) <= 0:
            raise RuntimeError(f"{case.name}: telemetry has no positive {key}")
        values[key] = int(match.group(1))
    return values


def parse_device_selection(output: str, case: Case) -> dict[str, object]:
    matches = DEVICE_PATTERN.findall(output)
    if len(matches) != 1:
        raise RuntimeError(
            f"{case.name}: expected one physical device selection, "
            f"observed {len(matches)}"
        )
    index, name = matches[0]
    return {"index": int(index), "name": name}


def validate_final_output(output_dir: Path, case: Case) -> int:
    frames = sorted(path for path in output_dir.glob("step_*") if path.is_dir())
    if not frames:
        raise RuntimeError(f"{case.name}: no final output frame was written")
    fields = sorted(frames[-1].glob("*.npy"))
    if not fields:
        raise RuntimeError(f"{case.name}: final output contains no NPY fields")
    validated_values = 0
    for path in fields:
        with path.open("rb") as handle:
            if handle.read(6) != b"\x93NUMPY":
                raise RuntimeError(f"{case.name}: invalid NPY magic in {path.name}")
            major, minor = handle.read(2)
            if (major, minor) == (1, 0):
                header_size = struct.unpack("<H", handle.read(2))[0]
            elif major in (2, 3):
                header_size = struct.unpack("<I", handle.read(4))[0]
            else:
                raise RuntimeError(
                    f"{case.name}: unsupported NPY version {major}.{minor}"
                )
            encoding = "utf-8" if major == 3 else "latin1"
            header = ast.literal_eval(handle.read(header_size).decode(encoding))
            if header.get("descr") not in ("<f4", "|f4"):
                raise RuntimeError(f"{case.name}: {path.name} is not float32")
            shape = header.get("shape")
            if not isinstance(shape, tuple) or not shape:
                raise RuntimeError(f"{case.name}: invalid shape in {path.name}")
            payload = handle.read()
        expected_bytes = math.prod(shape) * 4
        if len(payload) != expected_bytes:
            raise RuntimeError(f"{case.name}: truncated payload in {path.name}")
        for (value,) in struct.iter_unpack("<f", payload):
            if not math.isfinite(value):
                raise RuntimeError(
                    f"{case.name}: non-finite final value in {path.name}"
                )
            validated_values += 1
    return validated_values


def run_case(
    binary: Path, config: Path, duration: int, case: Case, output_dir: Path
) -> dict[str, object]:
    environment = os.environ.copy()
    environment["NIMBUS_GPU_KERNEL_TIMING"] = "1"
    environment["NIMBUS_GPU_RESIDENT"] = "1"
    environment["NIMBUS_GPU_LIN_BATCH"] = "1"
    result = subprocess.run(
        [
            str(binary), "--config", str(config), "--headless",
            "--compute-backend", "vulkan", "--guard-mode", case.guard_mode,
            f"--duration={duration}", f"--write-every={duration}",
            f"--outdir={output_dir}",
        ],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if not case.admitted:
        if result.returncode == 0:
            raise RuntimeError(
                f"{case.name}: unsupported GPU science tuple was admitted"
            )
        rejection = "[CONFIG] Science profile rejected before initialization:"
        if rejection not in result.stdout:
            raise RuntimeError(
                f"{case.name}: unsupported tuple did not fail in preflight\n"
                + result.stdout[-5000:]
            )
        expected_errors: list[str] = []
        if case.microphysics not in ("none", "kessler", "lin"):
            expected_errors.append(
                f"microphysics.scheme={case.microphysics} does not support "
                "backend=gpu"
            )
        if case.turbulence not in ("none", "smagorinsky"):
            expected_errors.append(
                f"turbulence.scheme={case.turbulence} does not support "
                "backend=gpu"
            )
        missing = [error for error in expected_errors if error not in result.stdout]
        if missing:
            raise RuntimeError(
                f"{case.name}: missing expected preflight errors: {missing}\n"
                + result.stdout[-5000:]
            )
        if "selected device" in result.stdout:
            raise RuntimeError(
                f"{case.name}: unsupported tuple reached device selection"
            )
        return {
            "case": asdict(case),
            "name": case.name,
            "status": "expected_preflight_rejection",
            "preflight_errors": expected_errors,
        }
    if result.returncode != 0:
        raise RuntimeError(
            f"{case.name}: simulator exited {result.returncode}\n"
            + result.stdout[-5000:]
        )
    if "selected device" not in result.stdout:
        raise RuntimeError(f"{case.name}: no physical Vulkan device was selected")
    if "Falling back to CPU backend" in result.stdout:
        raise RuntimeError(f"{case.name}: Vulkan request fell back to CPU")
    if "[PHYSICS GUARD FIELD]" in result.stdout:
        raise RuntimeError(f"{case.name}: strict guard observed a state violation")
    stages = parse_stage_dispatches(result.stdout)
    verify_stages(stages, case)
    totals = parse_total_telemetry(result.stdout, case)
    device = parse_device_selection(result.stdout, case)
    validated_values = validate_final_output(output_dir, case)
    return {
        "case": asdict(case),
        "name": case.name,
        "status": "physical_pass",
        "device": device,
        "stages": stages,
        "telemetry": totals,
        "validated_finite_output_values": validated_values,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("bin/nimbus_sim"))
    parser.add_argument(
        "--base-config", type=Path,
        default=Path("tests/fixtures/configs/qualification/gpu_residency.yaml"),
    )
    parser.add_argument("--duration", type=int, default=4)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    matrix = cases()
    if len(matrix) != 168:
        raise RuntimeError(
            f"generated matrix drift: expected 168 cases, got {len(matrix)}"
        )
    if args.list:
        for case in matrix:
            print(case.name)
        return 0
    if args.duration <= 0:
        parser.error("--duration must be positive")
    binary = args.binary.resolve()
    base_config = args.base_config.resolve()
    if not binary.is_file():
        parser.error(f"binary not found: {binary}")
    if not base_config.is_file():
        parser.error(f"base config not found: {base_config}")
    base = base_config.read_text(encoding="utf-8")
    catalog_result = subprocess.run(
        [str(binary), "--scheme-catalog=json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if catalog_result.returncode != 0:
        raise RuntimeError(
            "failed to read compiled scheme catalog:\n"
            + catalog_result.stderr[-5000:]
        )
    catalog = json.loads(catalog_result.stdout)
    interaction_policy_version = catalog["runtimeCapabilities"][
        "interactionPolicy"
    ]["version"]
    if interaction_policy_version < 6:
        raise RuntimeError(
            "compiled interaction policy predates guard-aware GPU ownership"
        )

    records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="nimbus_vulkan_combinations_") as raw:
        root = Path(raw)
        for index, case in enumerate(matrix, start=1):
            config = root / f"{index:02d}_{case.name}.yaml"
            config.write_text(render_config(base, case), encoding="utf-8")
            output_dir = root / f"{index:02d}_output"
            record = run_case(
                binary, config, args.duration, case, output_dir
            )
            records.append(record)
            if case.admitted:
                totals = record["telemetry"]
                print(
                    f"[vulkan-combinations] PASS {index:02d}/{len(matrix)} "
                    f"{case.name} dispatches={totals['dispatches']}"
                )
            else:
                print(
                    f"[vulkan-combinations] REJECT {index:02d}/{len(matrix)} "
                    f"{case.name} preflight=expected"
                )

    physical_passes = sum(
        record["status"] == "physical_pass" for record in records
    )
    expected_rejections = len(records) - physical_passes
    devices = {
        json.dumps(record["device"], sort_keys=True)
        for record in records if record["status"] == "physical_pass"
    }
    if len(devices) != 1:
        raise RuntimeError(
            f"physical matrix selected inconsistent devices: {sorted(devices)}"
        )
    physical_device = json.loads(next(iter(devices)))
    report = {
        "schema": "nimbus.vulkan-combination-matrix.v2",
        "status": "pass",
        "binary": str(args.binary),
        "base_config": str(args.base_config),
        "duration_s": args.duration,
        "case_count": len(records),
        "physical_pass_count": physical_passes,
        "expected_preflight_rejection_count": expected_rejections,
        "guard_modes": ["off", "warn", "error"],
        "physical_device": physical_device,
        "provenance": {
            "binary_sha256": sha256_file(binary),
            "base_config_sha256": sha256_file(base_config),
            "qualification_script_sha256": sha256_file(
                Path(__file__).resolve()
            ),
            "shader_manifest_sha256": sha256_file(
                Path(__file__).resolve().parents[3]
                / "vulkan" / "shaders" / "manifest.json"
            ),
            "scheme_catalog_sha256": sha256_text(catalog_result.stdout),
            "interaction_policy_version": interaction_policy_version,
        },
        "cases": records,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        f"[vulkan-combinations] RESULT: PASS cases={len(records)} "
        f"physical={physical_passes} rejected={expected_rejections}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
