from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/tests/gpu/test_moving_nested_vulkan_admission.py"
SPEC = importlib.util.spec_from_file_location("moving_nested_vulkan_admission", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load moving Vulkan admission script")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def successful_log(plan: str = "direct_shared (compatible host-visible primary device heap, policy=auto)") -> str:
    return "\n".join((
        '[COMPUTE][VULKAN] device[0] name="Example GPU", type=discrete, api=1.3.0, '
        'compute_queue=yes(family=0, queues=1), max_invocations=1024, '
        'shared_mem_bytes=65536, portability_subset=no, suitable=yes, score=1200',
        '[COMPUTE][VULKAN] selected device[0] "Example GPU"',
        f'[COMPUTE][VULKAN] memory plan={plan}',
        '[INFO]  [COMPUTE] backend=vulkan, requested=vulkan, device_index=-1, '
        'fallback=off, fallback_allowed=off, parity=off',
        '[INFO]  [NESTING DEVICE] smagorinsky_coefficients_dispatches=80, '
        'smagorinsky_tendencies_dispatches=80, sgs_apply_dispatches=80',
        '[INFO]  [NESTING DEVICE LEVELS] parent_coefficients=40, '
        'parent_tendencies=40, parent_sgs_apply=40, child_coefficients=40, '
        'child_tendencies=40, child_sgs_apply=40',
        '[INFO]  [NESTING DEVICE MICROPHYSICS] kessler_pointwise_dispatches=60, '
        'kessler_sedimentation_dispatches=60, microphysics_apply_dispatches=0',
    ))


class MovingVulkanPositiveLogContractTests(unittest.TestCase):
    def test_direct_shared_m4_policy_remains_accepted(self) -> None:
        result = MODULE.verify_vulkan_positive_log(successful_log())
        self.assertEqual(result["physical_topology"], "direct_shared")
        self.assertEqual(result["selected_device"], "Example GPU")

    def test_discrete_and_aperture_auto_topologies_are_accepted(self) -> None:
        for plan in (
            "staged_discrete",
            "host_visible_device_local_aperture; retaining staged transfers "
            "through the primary device heap",
        ):
            with self.subTest(plan=plan):
                result = MODULE.verify_vulkan_positive_log(successful_log(plan))
                self.assertEqual(result["physical_topology"], plan.split(";")[0])

    def test_explicit_staged_and_direct_policies_require_matching_logs(self) -> None:
        staged = successful_log("staged_forced (physical_topology=staged_discrete)")
        self.assertEqual(
            MODULE.verify_vulkan_positive_log(staged, "staged")["physical_topology"],
            "staged_discrete",
        )
        direct = successful_log(
            "direct_shared (compatible host-visible primary device heap, policy=direct)"
        )
        self.assertEqual(
            MODULE.verify_vulkan_positive_log(direct, "direct")["physical_topology"],
            "direct_shared",
        )

    def test_cpu_fallback_or_wrong_backend_rejects(self) -> None:
        base = successful_log()
        variants = (
            base + "\n[WARN] Falling back to CPU backend.",
            base.replace("backend=vulkan, requested=vulkan", "backend=cpu, requested=vulkan"),
            base.replace("fallback=off, fallback_allowed=off", "fallback=on, fallback_allowed=on"),
        )
        for variant in variants:
            with self.subTest(variant=variant[-80:]):
                with self.assertRaisesRegex(RuntimeError, "fallback|Vulkan"):
                    MODULE.verify_vulkan_positive_log(variant)

    def test_missing_wrong_or_software_device_rejects(self) -> None:
        base = successful_log()
        variants = (
            base.replace('[COMPUTE][VULKAN] selected device[0] "Example GPU"\n', ""),
            base.replace('selected device[0] "Example GPU"', 'selected device[1] "Example GPU"'),
            base.replace("type=discrete", "type=cpu"),
            base.replace("device_index=-1", "device_index=1"),
        )
        for variant in variants:
            with self.subTest(variant=variant[:130]):
                with self.assertRaisesRegex(RuntimeError, "device|GPU"):
                    MODULE.verify_vulkan_positive_log(variant)

    def test_zero_or_missing_required_dispatch_rejects(self) -> None:
        base = successful_log()
        variants = (
            base.replace("smagorinsky_coefficients_dispatches=80", "smagorinsky_coefficients_dispatches=0"),
            base.replace("child_sgs_apply=40", "child_sgs_apply=0"),
            base.replace("kessler_sedimentation_dispatches=60", "kessler_sedimentation_dispatches=0"),
            "\n".join(line for line in base.splitlines() if "[NESTING DEVICE LEVELS]" not in line),
        )
        for variant in variants:
            with self.subTest(variant=variant[-130:]):
                with self.assertRaisesRegex(RuntimeError, "dispatch|NESTING DEVICE"):
                    MODULE.verify_vulkan_positive_log(variant)

    def test_incorrect_memory_policy_rejects(self) -> None:
        auto = successful_log()
        cases = (
            (auto.replace("policy=auto", "policy=direct"), "auto"),
            (successful_log("staged_discrete"), "direct"),
            (auto, "staged"),
            (successful_log("staged_forced (physical_topology=staged_discrete)"), "auto"),
        )
        for output, policy in cases:
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(RuntimeError, "memory plan|memory policy"):
                    MODULE.verify_vulkan_positive_log(output, policy)

    def test_validation_and_science_errors_reject(self) -> None:
        base = successful_log()
        for error in (
            "[ERROR] Vulkan validation failed",
            "Validation Error: VUID-vkCmdDispatch-None-00000",
            "[PHYSICS GUARD FIELD] theta non-finite",
        ):
            with self.subTest(error=error):
                with self.assertRaisesRegex(RuntimeError, "validation or science"):
                    MODULE.verify_vulkan_positive_log(base + "\n" + error)


if __name__ == "__main__":
    unittest.main()
