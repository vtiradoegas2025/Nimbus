"""Production-runtime smoke tests for bounded native Metal ownership.

These fixtures are deliberately hybrid: they prove the named kernels execute
on Metal while unsupported or guard-incompatible operators remain explicit
host work. ``allow_fallback: false`` forbids a selected native operation from
silently substituting CPU work; it does not claim every model operator has a
Metal implementation.

Ordinary test discovery skips when macOS exposes no Metal device (including
managed sandboxes). Physical qualification sets ``NIMBUS_GPU_REQUIRED=1`` and
therefore remains fail-closed.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
GPU_REQUIRED = os.environ.get("NIMBUS_GPU_REQUIRED") == "1"


def require_native_metal(test: unittest.TestCase, output: str) -> None:
    if "[COMPUTE] backend=metal, requested=metal" in output:
        return
    unavailable = "Metal system default device is unavailable" in output
    if unavailable and not GPU_REQUIRED:
        test.skipTest(
            "native Metal device is unavailable; rerun physical qualification "
            "with NIMBUS_GPU_REQUIRED=1 outside the managed sandbox"
        )
    test.fail("native Metal backend was not selected:\n" + output)


def parse_dispatch_evidence(output: str) -> tuple[int, int, str] | None:
    line = re.search(r"^.*\[COMPUTE DISPATCH\] backend=metal, .*?$", output, re.MULTILINE)
    if line is None:
        return None
    text = line.group(0)
    submissions = re.search(r"\bcompleted_submissions=(\d+)\b", text)
    dispatches = re.search(r"\bkernel_dispatches=(\d+)\b", text)
    kernels = re.search(r"\bkernels=\{([^}]*)\}", text)
    if submissions is None or dispatches is None or kernels is None:
        return None
    return int(submissions.group(1)), int(dispatches.group(1)), kernels.group(1)


class MetalProductionRuntimeTest(unittest.TestCase):
    def test_one_cgrid_step_reports_native_metal_dispatch_provenance(self) -> None:
        binary = ROOT / "bin" / "nimbus_sim"
        self.assertTrue(binary.is_file(), f"missing production binary: {binary}")

        with tempfile.TemporaryDirectory(prefix="nimbus-metal-runtime-") as outdir:
            result = subprocess.run(
                [
                    str(binary),
                    "--config",
                    str(
                        ROOT
                        / "tests"
                        / "fixtures"
                        / "configs"
                        / "qualification"
                        / "gpu_report_free.yaml"
                    ),
                    "--compute-backend=metal",
                    "--duration=2",
                    "--write-every=0",
                    f"--outdir={outdir}",
                    "--headless",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )

        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        require_native_metal(self, output)
        self.assertIn("fallback=off", output)
        self.assertIn("[MEMORY PLAN] accelerator_mode=unified", output)
        evidence = parse_dispatch_evidence(output)
        self.assertIsNotNone(evidence, output)
        assert evidence is not None
        self.assertGreater(evidence[0], 0, output)
        self.assertGreater(evidence[1], 0, output)
        self.assertIn(
            "acoustic/acoustic_momentum_cartesian_cgrid.comp:",
            evidence[2],
            output,
        )
        self.assertIn(
            "advection/advect_conservative_mass_flux_cartesian_cgrid.comp:",
            evidence[2],
            output,
        )
        self.assertIn(
            "advection/advect_conservative_update_cartesian_cgrid.comp:",
            evidence[2],
            output,
        )
        self.assertIn(
            "advection/advect_pd_limiter_cartesian_cgrid.comp:",
            evidence[2],
            output,
        )
        self.assertNotIn(
            "advection/advect_conservative_scalar_flux_cartesian_cgrid.comp:",
            evidence[2],
            output,
        )

    def test_constant_velocity_cgrid_admits_native_metal(self) -> None:
        binary = ROOT / "bin" / "nimbus_sim"
        self.assertTrue(binary.is_file(), f"missing production binary: {binary}")

        with tempfile.TemporaryDirectory(
            prefix="nimbus-metal-constant-velocity-"
        ) as outdir:
            result = subprocess.run(
                [
                    str(binary),
                    "--config",
                    str(
                        ROOT
                        / "tests"
                        / "fixtures"
                        / "configs"
                        / "qualification"
                        / "constant_velocity_cgrid.yaml"
                    ),
                    "--compute-backend=metal",
                    "--duration=1",
                    "--write-every=0",
                    f"--outdir={outdir}",
                    "--headless",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )

        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        require_native_metal(self, output)
        self.assertIn("fallback_allowed=off", output)
        self.assertIn("Config file: ", output)
        self.assertIn("constant_velocity_cgrid.yaml", output)
        self.assertNotIn(
            "constant_velocity currently requires an explicit supported",
            output,
        )
        evidence = parse_dispatch_evidence(output)
        self.assertIsNotNone(evidence, output)
        assert evidence is not None
        self.assertGreater(evidence[0], 0, output)
        self.assertGreater(evidence[1], 0, output)
        self.assertIn("diffusion/diffusion.comp:", evidence[2], output)
        self.assertIn("microphysics/kessler_pointwise.comp:", evidence[2], output)

    def test_live_recycling_preflight_admits_metal_shift_pipeline(self) -> None:
        binary = ROOT / "bin" / "nimbus_sim"
        self.assertTrue(binary.is_file(), f"missing production binary: {binary}")
        base = (
            ROOT
            / "tests"
            / "fixtures"
            / "configs"
            / "qualification"
            / "constant_velocity_cgrid.yaml"
        ).read_text(encoding="utf-8")
        marker = "    v_ms: -3\n\n"
        self.assertIn(marker, base)
        recycling = base.replace(
            marker,
            marker.rstrip()
            + "\n"
            + "  tracking:\n"
            + "    enabled: true\n"
            + "    recycle_domain: true\n"
            + "    interval_s: 30\n"
            + "    inner_margin_m: 2000\n"
            + "    maximum_shift_cells: 2\n"
            + "    updraft_helicity_bottom_m_agl: 1000\n"
            + "    updraft_helicity_top_m_agl: 3500\n\n",
            1,
        ).replace(": periodic", ": radiative")

        with tempfile.TemporaryDirectory(
            prefix="nimbus-metal-recycling-admission-"
        ) as tempdir:
            temp_path = pathlib.Path(tempdir)
            config = temp_path / "constant_velocity_recycling.yaml"
            config.write_text(recycling, encoding="utf-8")
            result = subprocess.run(
                [
                    str(binary),
                    "--config",
                    str(config),
                    "--compute-backend=metal",
                    "--duration=1",
                    "--write-every=0",
                    f"--outdir={temp_path / 'output'}",
                    "--headless",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )

        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        require_native_metal(self, output)
        self.assertNotIn(
            "[DOMAIN RECYCLING] live homogeneous recycling currently requires",
            output,
        )
        self.assertRegex(
            output,
            r"\[COMPUTE DISPATCH\] backend=metal, completed_submissions=[1-9]",
        )


if __name__ == "__main__":
    unittest.main()
