#!/usr/bin/env python3
"""Exercise exact internal and public live-storm admission boundaries."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
INTERNAL_BINARY = ROOT / "bin" / "nimbus_sim_internal_qualification"
PUBLIC_BINARY = ROOT / "bin" / "nimbus_sim"
CPU_CONFIG = (
    ROOT / "tests" / "qualification" / "fixtures" /
    "moving_nested_supercell_frontier_v5.yaml"
)
VULKAN_CONFIG = (
    ROOT / "tests" / "qualification" / "fixtures" /
    "moving_nested_supercell_frontier_vulkan_v1.yaml"
)
PUBLIC_CONFIG = (
    ROOT / "tests" / "fixtures" / "configs" / "qualification" /
    "moving_nested_supercell_vulkan_v1.yaml"
)
PUBLIC_CPU_CONFIG = (
    ROOT / "tests" / "fixtures" / "configs" / "qualification" /
    "moving_nested_supercell_cpu_v1.yaml"
)
PUBLIC_METAL_CONFIG = (
    ROOT / "tests" / "fixtures" / "configs" / "qualification" /
    "moving_nested_supercell_metal_v1.yaml"
)
PROFILE_ERRORS = (
    "the internal moving-nest campaign requires its exact versioned CPU or "
    "fallback-off Vulkan/Metal exploratory",
    "the internal moving-nest campaign accepts only its exact versioned CPU "
    "or fallback-off Vulkan/Metal exploratory",
)
PUBLIC_PROFILE_ERROR = (
    "public moving_nest requires the exact versioned CPU or fallback-off Vulkan/Metal "
    "qualified supercell profile"
)
PUBLIC_METAL_SCOPE_ERROR = (
    "qualified Metal execution is limited to exact public moving-nest"
)


def load_extractor():
    path = ROOT / "scripts" / "qualification" / "nested_domains" / "extract_moving_nested_reference_series.py"
    spec = importlib.util.spec_from_file_location(
        "moving_nested_reference_extractor", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load configuration extractor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(
            f"fixture mutation expected one occurrence of {old!r}, "
            f"found {source.count(old)}"
        )
    return source.replace(old, new, 1)


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one_match(pattern: str, output: str, label: str) -> re.Match[str]:
    matches = list(re.finditer(pattern, output, re.MULTILINE))
    if len(matches) != 1:
        raise RuntimeError(f"moving Vulkan positive requires exactly one {label}")
    return matches[0]


def _dispatch_counters(output: str, label: str) -> dict[str, int]:
    line = _one_match(
        rf"^\[INFO\]\s+\[{re.escape(label)}\] ([^\n]+)$",
        output, label,
    ).group(1)
    counters: dict[str, int] = {}
    for token in line.split(", "):
        match = re.fullmatch(r"([a-z_]+)=(\d+)", token)
        if match is None or match.group(1) in counters:
            raise RuntimeError(f"moving Vulkan positive has malformed {label} counters")
        counters[match.group(1)] = int(match.group(2))
    return counters


def verify_vulkan_positive_log(output: str, requested_policy: str = "auto") -> dict[str, object]:
    if requested_policy not in {"auto", "direct", "staged"}:
        raise RuntimeError("moving Vulkan positive has invalid requested memory policy")
    if re.search(
        r"\[PHYSICS GUARD FIELD\]|\[ERROR\]|validation error|VUID-|non-finite|invalid science",
        output, re.IGNORECASE,
    ):
        raise RuntimeError("moving Vulkan positive contains validation or science errors")
    if "Falling back to CPU backend" in output:
        raise RuntimeError("moving Vulkan positive used CPU fallback")

    selected = _one_match(
        r'^\[COMPUTE\]\[VULKAN\] selected device\[(\d+)\] "([^"\n]+)"$',
        output, "selected Vulkan device",
    )
    selected_index = int(selected.group(1))
    selected_name = selected.group(2)
    candidates = list(re.finditer(
        r'^\[COMPUTE\]\[VULKAN\] device\[(\d+)\] name="([^"\n]+)", '
        r'type=([^,\n]+), [^\n]*suitable=yes(?:,|$)',
        output, re.MULTILINE,
    ))
    selected_candidates = [match for match in candidates
                           if int(match.group(1)) == selected_index and
                           match.group(2) == selected_name and
                           match.group(3) in {"discrete", "integrated"}]
    if len(selected_candidates) != 1:
        raise RuntimeError("moving Vulkan positive did not select one suitable hardware GPU")

    active = _one_match(
        r'^\[INFO\]\s+\[COMPUTE\] backend=([^,]+), requested=([^,]+), '
        r'device_index=(-?\d+), fallback=(on|off), fallback_allowed=(on|off), '
        r'parity=(on|off)$', output, "active backend report",
    )
    if (active.group(1), active.group(2), active.group(4), active.group(5)) != (
        "vulkan", "vulkan", "off", "off"
    ):
        raise RuntimeError("moving Vulkan positive did not retain fallback-off Vulkan")
    requested_index = int(active.group(3))
    if requested_index >= 0 and requested_index != selected_index:
        raise RuntimeError("moving Vulkan positive selected the wrong device index")

    plan = _one_match(
        r'^\[COMPUTE\]\[VULKAN\] memory plan=([^\n]+)$',
        output, "Vulkan memory plan",
    ).group(1)
    direct = re.fullmatch(
        r'direct_shared \(compatible host-visible primary device heap, policy=(auto|direct)\)', plan,
    )
    staged = re.fullmatch(r'staged_discrete', plan)
    aperture = re.fullmatch(
        r'host_visible_device_local_aperture; retaining staged transfers through the primary device heap',
        plan,
    )
    forced = re.fullmatch(
        r'staged_forced \(physical_topology=(direct_shared|staged_discrete|'
        r'host_visible_device_local_aperture)\)', plan,
    )
    if requested_policy == "auto":
        if direct and direct.group(1) == "auto":
            topology = "direct_shared"
        elif staged:
            topology = "staged_discrete"
        elif aperture:
            topology = "host_visible_device_local_aperture"
        else:
            raise RuntimeError("moving Vulkan positive has an incorrect auto memory plan")
    elif requested_policy == "direct":
        if not direct or direct.group(1) != "direct":
            raise RuntimeError("moving Vulkan positive did not honor direct memory policy")
        topology = "direct_shared"
    else:
        if not forced:
            raise RuntimeError("moving Vulkan positive did not honor staged memory policy")
        topology = forced.group(1)

    required = {
        "NESTING DEVICE": (
            "smagorinsky_coefficients_dispatches",
            "smagorinsky_tendencies_dispatches", "sgs_apply_dispatches",
        ),
        "NESTING DEVICE LEVELS": (
            "parent_coefficients", "parent_tendencies", "parent_sgs_apply",
            "child_coefficients", "child_tendencies", "child_sgs_apply",
        ),
        "NESTING DEVICE MICROPHYSICS": (
            "kessler_pointwise_dispatches", "kessler_sedimentation_dispatches",
        ),
    }
    observed: dict[str, dict[str, int]] = {}
    for label, keys in required.items():
        counters = _dispatch_counters(output, label)
        if any(counters.get(key, 0) <= 0 for key in keys):
            raise RuntimeError(f"moving Vulkan positive has no required {label} dispatch")
        observed[label] = counters
    return {"selected_device_index": selected_index, "selected_device": selected_name,
            "memory_plan": plan, "physical_topology": topology,
            "requested_memory_policy": requested_policy,
            "required_dispatches": observed}


def validate_semantic_delta() -> None:
    extractor = load_extractor()
    cpu = extractor.flat_yaml_scalars(CPU_CONFIG)
    vulkan = extractor.flat_yaml_scalars(VULKAN_CONFIG)
    cpu.setdefault("advection.scheme", "tvd")
    vulkan.setdefault("advection.scheme", "tvd")
    allowed = {
        "extensions.compute.backend",
        "extensions.compute.allow_fallback",
        "extensions.compute.gpu_apply_microphysics",
        "extensions.compute.gpu_apply_slow",
        "extensions.compute.gpu_apply_sgs",
        "extensions.checkpoint.every_s",
    }
    cpu_common = {key: value for key, value in cpu.items() if key not in allowed}
    vulkan_common = {
        key: value for key, value in vulkan.items() if key not in allowed
    }
    if cpu_common != vulkan_common:
        missing = sorted(set(cpu_common) - set(vulkan_common))
        extra = sorted(set(vulkan_common) - set(cpu_common))
        changed = sorted(
            key for key in set(cpu_common) & set(vulkan_common)
            if cpu_common[key] != vulkan_common[key]
        )
        raise RuntimeError(
            "moving CPU/Vulkan fixtures differ outside backend ownership: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    expected = {
        "extensions.compute.backend": "vulkan",
        "extensions.compute.allow_fallback": "false",
        "extensions.compute.gpu_apply_microphysics": "false",
        "extensions.compute.gpu_apply_slow": "false",
        "extensions.compute.gpu_apply_sgs": "true",
        "extensions.checkpoint.every_s": "450",
    }
    observed = {key: vulkan.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(
            f"moving Vulkan ownership tuple changed: {observed}"
        )

    public = extractor.flat_yaml_scalars(PUBLIC_CONFIG)
    execution_delta = {
        "extensions.execution.mode",
        "extensions.execution.acknowledge_maturity_limits",
    }
    vulkan_common = {
        key: value for key, value in vulkan.items()
        if key not in execution_delta
    }
    public_common = {
        key: value for key, value in public.items()
        if key not in execution_delta
    }
    if public_common != vulkan_common:
        raise RuntimeError(
            "public moving Vulkan profile differs from the qualified evidence "
            "tuple outside execution admission"
        )
    expected_execution = {
        "extensions.execution.mode": "qualified",
    }
    observed_execution = {
        key: public.get(key) for key in expected_execution
    }
    if observed_execution != expected_execution:
        raise RuntimeError(
            f"public execution tuple changed: {observed_execution}"
        )

    public_cpu = extractor.flat_yaml_scalars(PUBLIC_CPU_CONFIG)
    cpu_allowed = {
        "extensions.execution.mode",
        "extensions.execution.acknowledge_maturity_limits",
        "extensions.checkpoint.every_s",
    }
    cpu_evidence_common = {
        key: value for key, value in cpu.items() if key not in cpu_allowed
    }
    public_cpu_common = {
        key: value for key, value in public_cpu.items()
        if key not in cpu_allowed
    }
    if public_cpu_common != cpu_evidence_common:
        raise RuntimeError(
            "public moving CPU profile differs from the ratified CPU evidence "
            "tuple outside execution and checkpoint cadence"
        )
    expected_cpu_public = {
        "extensions.execution.mode": "qualified",
        "extensions.checkpoint.every_s": "450",
    }
    observed_cpu_public = {
        key: public_cpu.get(key) for key in expected_cpu_public
    }
    if observed_cpu_public != expected_cpu_public:
        raise RuntimeError(
            f"public CPU execution tuple changed: {observed_cpu_public}"
        )

    public_metal = extractor.flat_yaml_scalars(PUBLIC_METAL_CONFIG)
    metal_common = {
        key: value for key, value in public_metal.items()
        if key != "extensions.compute.backend"
    }
    public_common = {
        key: value for key, value in public.items()
        if key != "extensions.compute.backend"
    }
    if metal_common != public_common or (
        public_metal.get("extensions.compute.backend") != "metal"
    ):
        raise RuntimeError(
            "public moving Metal profile differs from the public Vulkan tuple "
            "outside backend identity"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=INTERNAL_BINARY)
    parser.add_argument("--public-binary", type=Path, default=PUBLIC_BINARY)
    parser.add_argument(
        "--require-vulkan-positive",
        action="store_true",
        help="require the exact profile to reach fallback-off physical startup",
    )
    parser.add_argument(
        "--require-metal-positive",
        action="store_true",
        help="require the exact Metal profile to reach fallback-off physical startup",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="write a hash-bound JSON admission report after all checks pass",
    )
    args = parser.parse_args()
    binary = args.binary.resolve()
    public_binary = args.public_binary.resolve()
    for required in (
        binary, public_binary, CPU_CONFIG, VULKAN_CONFIG, PUBLIC_CONFIG,
        PUBLIC_CPU_CONFIG, PUBLIC_METAL_CONFIG,
    ):
        if not required.is_file():
            raise RuntimeError(f"required admission input is missing: {required}")

    validate_semantic_delta()
    base = VULKAN_CONFIG.read_text(encoding="utf-8")
    internal_mutations = {
        "fallback enabled": (
            replace_once(
                base, "    allow_fallback: false\n", "    allow_fallback: true\n"
            ),
            PROFILE_ERRORS,
        ),
        "device microphysics commit": (
            replace_once(
                base,
                "    gpu_apply_microphysics: false\n",
                "    gpu_apply_microphysics: true\n",
            ),
            PROFILE_ERRORS,
        ),
        "device slow commit": (
            replace_once(
                base, "    gpu_apply_slow: false\n", "    gpu_apply_slow: true\n"
            ),
            PROFILE_ERRORS,
        ),
        "host SGS commit": (
            replace_once(
                base, "    gpu_apply_sgs: true\n", "    gpu_apply_sgs: false\n"
            ),
            PROFILE_ERRORS,
        ),
        "changed child width": (
            replace_once(
                base, "    child_cells_x: 156\n", "    child_cells_x: 154\n"
            ),
            PROFILE_ERRORS,
        ),
        "changed movement cap": (
            replace_once(
                base,
                "    maximum_shift_cells: 14\n",
                "    maximum_shift_cells: 13\n",
            ),
            PROFILE_ERRORS,
        ),
        "qualified execution": (
            replace_once(
                replace_once(
                    base, "    mode: exploratory\n", "    mode: qualified\n"
                ),
                "    acknowledge_maturity_limits: false\n",
                "    acknowledge_maturity_limits: true\n",
            ),
            PROFILE_ERRORS + (
                "domain_motion.tracking diagnostic evidence is not reference-qualified",
            ),
        ),
    }

    physical_markers_verified = False
    physical_vulkan_evidence: dict[str, object] | None = None
    with tempfile.TemporaryDirectory(
        prefix="nimbus_moving_nested_vulkan_admission_"
    ) as temporary:
        temporary_root = Path(temporary)

        metal_base = replace_once(
            base, "    backend: vulkan\n", "    backend: metal\n")
        metal_config = temporary_root / "internal_metal_positive.yaml"
        metal_config.write_text(metal_base, encoding="utf-8")
        metal_positive = run([
            str(binary), "--qualification-moving-nest", "--validate-only",
            "--config", str(metal_config),
        ])
        if metal_positive.returncode != 0:
            raise RuntimeError(
                "exact internal moving Metal profile failed validation:\n"
                f"{metal_positive.stdout}"
            )
        metal_mutations = {
            "Metal fallback enabled": replace_once(
                metal_base,
                "    allow_fallback: false\n",
                "    allow_fallback: true\n",
            ),
            "Metal device microphysics commit": replace_once(
                metal_base,
                "    gpu_apply_microphysics: false\n",
                "    gpu_apply_microphysics: true\n",
            ),
            "Metal changed child width": replace_once(
                metal_base, "    child_cells_x: 156\n",
                "    child_cells_x: 154\n",
            ),
        }
        for index, (label, contents) in enumerate(metal_mutations.items()):
            candidate = temporary_root / f"metal_negative_{index:02d}.yaml"
            candidate.write_text(contents, encoding="utf-8")
            result = run([
                str(binary), "--qualification-moving-nest", "--validate-only",
                "--config", str(candidate),
            ])
            if result.returncode == 0 or not any(
                marker in result.stdout for marker in PROFILE_ERRORS
            ):
                raise RuntimeError(
                    f"{label} did not fail at the internal Metal boundary:\n"
                    f"{result.stdout}"
                )
        internal_public_metal = run([
            str(public_binary), "--validate-only", "--config",
            str(metal_config),
        ])
        if internal_public_metal.returncode == 0 or (
            PUBLIC_PROFILE_ERROR not in internal_public_metal.stdout
        ):
            raise RuntimeError(
                "public binary did not reject exploratory moving Metal:\n"
                f"{internal_public_metal.stdout}"
            )

        for index, (label, mutation) in enumerate(internal_mutations.items()):
            contents, expected_errors = mutation
            candidate = temporary_root / f"negative_{index:02d}.yaml"
            candidate.write_text(contents, encoding="utf-8")
            result = run([
                str(binary), "--qualification-moving-nest", "--validate-only",
                "--config", str(candidate),
            ])
            if result.returncode == 0 or not any(
                marker in result.stdout for marker in expected_errors
            ):
                raise RuntimeError(
                    f"{label} did not fail at the moving Vulkan profile "
                    f"boundary:\n{result.stdout}"
                )

        public_config = run([
            str(public_binary), "--validate-only", "--config",
            str(VULKAN_CONFIG),
        ])
        if public_config.returncode == 0 or (
            PUBLIC_PROFILE_ERROR not in public_config.stdout
        ):
            raise RuntimeError(
                "public binary did not keep the exploratory moving Vulkan "
                f"fixture internal-only:\n{public_config.stdout}"
            )

        public_positive = run([
            str(public_binary), "--validate-only", "--config",
            str(PUBLIC_CONFIG),
        ])
        if public_positive.returncode != 0:
            raise RuntimeError(
                "exact public moving Vulkan profile failed validation:\n"
                f"{public_positive.stdout}"
            )

        public_cpu_positive = run([
            str(public_binary), "--validate-only", "--config",
            str(PUBLIC_CPU_CONFIG),
        ])
        if public_cpu_positive.returncode != 0:
            raise RuntimeError(
                "exact public moving CPU profile failed validation:\n"
                f"{public_cpu_positive.stdout}"
            )

        public_metal_positive = run([
            str(public_binary), "--validate-only", "--config",
            str(PUBLIC_METAL_CONFIG),
        ])
        if public_metal_positive.returncode != 0:
            raise RuntimeError(
                "exact public moving Metal profile failed validation:\n"
                f"{public_metal_positive.stdout}"
            )

        cpu_startup = run([
            str(public_binary), "--headless", "--config",
            str(PUBLIC_CPU_CONFIG), "--duration", "30", "--threads", "4",
            f"--outdir={temporary_root / 'cpu_startup'}",
            "--log-profile=normal",
        ])
        if cpu_startup.returncode != 0 or (
            "[COMPUTE] backend=cpu, requested=cpu" not in cpu_startup.stdout
        ):
            raise RuntimeError(
                "exact public moving CPU profile failed ordinary startup:\n"
                f"{cpu_startup.stdout}"
            )

        public_base = PUBLIC_CONFIG.read_text(encoding="utf-8")
        public_mutations = {
            "public fallback enabled": replace_once(
                public_base,
                "    allow_fallback: false\n",
                "    allow_fallback: true\n",
            ),
            "public device microphysics commit": replace_once(
                public_base,
                "    gpu_apply_microphysics: false\n",
                "    gpu_apply_microphysics: true\n",
            ),
            "public changed child width": replace_once(
                public_base,
                "    child_cells_x: 156\n",
                "    child_cells_x: 154\n",
            ),
            "public changed movement cap": replace_once(
                public_base,
                "    maximum_shift_cells: 14\n",
                "    maximum_shift_cells: 13\n",
            ),
            "public changed SGS coefficient": replace_once(
                public_base, "  Cs: 0.18\n", "  Cs: 0.19\n"
            ),
            "public changed history cadence": replace_once(
                public_base,
                "  output_interval: 450\n",
                "  output_interval: 900\n",
            ),
            "public exploratory execution": replace_once(
                public_base,
                "    mode: qualified\n",
                "    mode: exploratory\n",
            ),
        }
        for index, (label, contents) in enumerate(public_mutations.items()):
            candidate = temporary_root / f"public_negative_{index:02d}.yaml"
            candidate.write_text(contents, encoding="utf-8")
            result = run([
                str(public_binary), "--validate-only", "--config",
                str(candidate),
            ])
            if result.returncode == 0 or PUBLIC_PROFILE_ERROR not in result.stdout:
                raise RuntimeError(
                    f"{label} did not fail at the public moving Vulkan "
                    f"profile boundary:\n{result.stdout}"
                )

        public_metal_base = PUBLIC_METAL_CONFIG.read_text(encoding="utf-8")
        public_metal_mutations = {
            "public Metal fallback enabled": replace_once(
                public_metal_base,
                "    allow_fallback: false\n",
                "    allow_fallback: true\n",
            ),
            "public Metal device microphysics commit": replace_once(
                public_metal_base,
                "    gpu_apply_microphysics: false\n",
                "    gpu_apply_microphysics: true\n",
            ),
            "public Metal changed child width": replace_once(
                public_metal_base,
                "    child_cells_x: 156\n",
                "    child_cells_x: 154\n",
            ),
        }
        for index, (label, contents) in enumerate(public_metal_mutations.items()):
            candidate = temporary_root / f"public_metal_negative_{index:02d}.yaml"
            candidate.write_text(contents, encoding="utf-8")
            validation = run([
                str(public_binary), "--validate-only", "--config", str(candidate),
            ])
            startup = run([
                str(public_binary), "--headless", "--config", str(candidate),
                "--duration", "0", "--write-every=0",
                f"--outdir={temporary_root / f'public_metal_negative_{index:02d}'}",
            ])
            if any(
                result.returncode == 0 or
                not any(marker in result.stdout for marker in (
                    PUBLIC_PROFILE_ERROR, PUBLIC_METAL_SCOPE_ERROR,
                ))
                for result in (validation, startup)
            ):
                raise RuntimeError(
                    f"{label} did not fail at the public moving Metal "
                    "profile boundary in both admission paths:\n"
                    f"validate-only:\n{validation.stdout}\n"
                    f"startup:\n{startup.stdout}"
                )

        public_cpu_base = PUBLIC_CPU_CONFIG.read_text(encoding="utf-8")
        public_cpu_mutations = {
            "public CPU changed child width": replace_once(
                public_cpu_base,
                "    child_cells_x: 156\n",
                "    child_cells_x: 154\n",
            ),
            "public CPU changed movement cap": replace_once(
                public_cpu_base,
                "    maximum_shift_cells: 14\n",
                "    maximum_shift_cells: 13\n",
            ),
            "public CPU changed SGS coefficient": replace_once(
                public_cpu_base, "  Cs: 0.18\n", "  Cs: 0.19\n"
            ),
            "public CPU exploratory execution": replace_once(
                public_cpu_base,
                "    mode: qualified\n",
                "    mode: exploratory\n",
            ),
        }
        for index, (label, contents) in enumerate(public_cpu_mutations.items()):
            candidate = temporary_root / f"public_cpu_negative_{index:02d}.yaml"
            candidate.write_text(contents, encoding="utf-8")
            validation = run([
                str(public_binary), "--validate-only", "--config",
                str(candidate),
            ])
            startup = run([
                str(public_binary), "--headless", "--config", str(candidate),
                "--duration", "30", "--threads", "1",
                f"--outdir={temporary_root / f'public_cpu_negative_{index:02d}'}",
            ])
            if any(
                result.returncode == 0 or
                PUBLIC_PROFILE_ERROR not in result.stdout
                for result in (validation, startup)
            ):
                raise RuntimeError(
                    f"{label} did not fail at the public moving CPU "
                    "profile boundary in both admission paths:\n"
                    f"validate-only:\n{validation.stdout}\n"
                    f"startup:\n{startup.stdout}"
                )
        public_flag = run([
            str(public_binary), "--qualification-moving-nest",
            "--validate-only", "--config", str(VULKAN_CONFIG),
        ])
        if public_flag.returncode == 0 or (
            "internal qualification binary" not in public_flag.stdout
        ):
            raise RuntimeError(
                "public binary did not reject the internal moving flag:\n"
                f"{public_flag.stdout}"
            )

        if args.require_vulkan_positive:
            positive = run([
                str(public_binary), "--headless",
                "--config", str(PUBLIC_CONFIG), "--duration", "30",
                "--write-every=30", "--threads", "12",
                f"--outdir={temporary_root / 'positive'}",
                "--log-profile=normal",
            ])
            if positive.returncode != 0:
                raise RuntimeError(
                    "exact moving Vulkan profile failed physical startup "
                    f"({positive.returncode}):\n{positive.stdout}"
                )
            physical_vulkan_evidence = verify_vulkan_positive_log(
                positive.stdout,
                os.environ.get("NIMBUS_VULKAN_MEMORY_POLICY", "auto"),
            )
            physical_markers_verified = True

        metal_markers_verified = False
        if args.require_metal_positive:
            positive = run([
                str(public_binary), "--headless",
                "--config", str(PUBLIC_METAL_CONFIG), "--duration", "30",
                "--write-every=30", "--threads", "12",
                f"--outdir={temporary_root / 'metal_positive'}",
                "--log-profile=normal",
            ])
            if positive.returncode != 0:
                raise RuntimeError(
                    "exact moving Metal profile failed physical startup "
                    f"({positive.returncode}):\n{positive.stdout}"
                )
            markers = (
                "[COMPUTE] backend=metal, requested=metal",
                "[MEMORY PLAN] accelerator_mode=unified",
                "fallback=off",
            )
            if not all(marker in positive.stdout for marker in markers):
                raise RuntimeError(
                    "moving Metal positive did not prove fallback-off device "
                    f"startup:\n{positive.stdout}"
                )
            metal_markers_verified = True

    if args.report is not None:
        report_path = args.report.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "schema": "nimbus.moving_nested.public_vulkan_admission",
            "schema_version": 2,
            "status": "pass",
            "public_binary": {
                "path": str(public_binary.relative_to(ROOT)),
                "sha256": sha256_file(public_binary),
            },
            "internal_binary": {
                "path": str(binary.relative_to(ROOT)),
                "sha256": sha256_file(binary),
            },
            "public_config": {
                "path": str(PUBLIC_CONFIG.relative_to(ROOT)),
                "sha256": sha256_file(PUBLIC_CONFIG),
            },
            "public_metal_config": {
                "path": str(PUBLIC_METAL_CONFIG.relative_to(ROOT)),
                "sha256": sha256_file(PUBLIC_METAL_CONFIG),
            },
            "evidence_config": {
                "path": str(VULKAN_CONFIG.relative_to(ROOT)),
                "sha256": sha256_file(VULKAN_CONFIG),
            },
            "checks": {
                "exact_public_profile_validates_without_internal_flag": True,
                "exploratory_profile_remains_internal_only": True,
                "public_internal_cli_flag_rejected": True,
                "internal_negative_profiles_rejected": len(internal_mutations),
                "internal_metal_exact_profile_validates": True,
                "internal_metal_negative_profiles_rejected": len(metal_mutations),
                "public_metal_profile_validates": True,
                "public_metal_negative_profiles_rejected": len(public_metal_mutations),
                "public_negative_profiles_rejected": len(public_mutations),
                "physical_vulkan_required": args.require_vulkan_positive,
                "physical_vulkan_fallback_off_markers_verified": (
                    physical_markers_verified
                ),
                "physical_memory_plan_direct_shared_verified": (
                    physical_vulkan_evidence is not None and
                    physical_vulkan_evidence["memory_plan"].startswith("direct_shared")
                ),
                "physical_vulkan_required_dispatches_verified": physical_markers_verified,
                "physical_vulkan_evidence": physical_vulkan_evidence,
                "physical_metal_required": args.require_metal_positive,
                "physical_metal_fallback_off_markers_verified": (
                    metal_markers_verified
                ),
            },
            "claim_limits": {
                "exact_profile_only": True,
                "device_resident_relocation": False,
                "broader_physics_or_terrain": False,
                "throughput_promotion": False,
            },
        }
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(
        "[moving-nested-admission] PASS: exact CPU/Vulkan/Metal ownership and "
        "public qualification/neighbor rejection are fail-closed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
