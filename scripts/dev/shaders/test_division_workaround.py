import pathlib
import re
import subprocess
import tempfile
import unittest
from unittest import mock

import division_workaround as d
import verify_spirv as v


class DivisionVariants(unittest.TestCase):
    def test_refresh_publishes_metadata_only_for_unchanged_inputs(self):
        for mutation in (None, "source", "include", "policy", "new-source"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory)
                shader_root = root / "vulkan/shaders"
                shader_root.mkdir(parents=True)
                source = shader_root / "fixture.comp"
                source.write_text("source-before")
                include = shader_root / "helper.glsl"
                include.write_text("include-before")
                policy = root / "compiler-policy.py"
                policy.write_text("policy-before")
                artifact = source.with_name(source.name + ".spv")
                artifact.write_bytes(b"old-artifact")
                manifest = shader_root / "manifest.json"
                manifest.write_text("manifest-before")
                header = root / "vulkan/include/compute/generated_division_variants.hpp"
                header.parent.mkdir(parents=True)
                header.write_text("header-before")

                def compile_fixture(_source, output):
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(b"new-artifact")
                    if mutation is not None:
                        target = {"source": source, "include": include, "policy": policy,
                                  "new-source": shader_root / "added.comp"}[mutation]
                        target.write_text("changed-during-compilation")

                with mock.patch.multiple(v, ROOT=root, SHADER_ROOT=shader_root, MANIFEST=manifest,
                                         COMPILER_POLICY_FILES=(policy,)), \
                     mock.patch("sys.argv", ["verify_spirv.py", "--refresh"]), \
                     mock.patch.object(v, "compile_shader", side_effect=compile_fixture), \
                     mock.patch.object(v, "validate"), \
                     mock.patch.object(v, "canonical_bytes", side_effect=lambda path, _: path.read_bytes()), \
                     mock.patch.object(v, "reflection", return_value={}), \
                     mock.patch.object(v, "entry", return_value={"source": "fixture.comp"}), \
                     mock.patch.object(v, "manifest_payload", side_effect=lambda entries: {"shaders": entries}), \
                     mock.patch.object(d, "build_library", return_value=root / "library.spv"), \
                     mock.patch.object(d, "lower", return_value=0), \
                     mock.patch("builtins.print"):
                    if mutation is None:
                        self.assertEqual(v.main(), 0)
                        self.assertNotEqual(header.read_text(), "header-before")
                        self.assertNotEqual(manifest.read_text(), "manifest-before")
                    else:
                        with self.assertRaisesRegex(RuntimeError, "metadata publication refused"):
                            v.main()
                        self.assertEqual(header.read_text(), "header-before")
                        self.assertEqual(manifest.read_text(), "manifest-before")

    def test_portable_fma_preserves_rounding_without_vendor_extensions(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = pathlib.Path(directory)
            source = v.SHADER_ROOT / "compute/chaos/earth_control_evolution.comp"
            artifact = temp / "earth.spv"
            v.compile_shader(source, artifact)
            v.validate(artifact)
            asm = d.run("spirv-dis", str(artifact))
            fmas = re.findall(r"(%\S+) = OpExtInst %\S+ %\S+ Fma\b", asm)
            self.assertGreater(len(fmas), 0)
            self.assertEqual(v.compile_flags(source), ["-V"])
            helpers = set(re.findall(r'OpName (%\S+) "df_[^"]+"', asm))
            self.assertGreater(len(helpers), 0)
            for helper in helpers:
                self.assertRegex(asm, re.escape(helper) + r" = OpFunction %\S+ DontInline ")
            self.assertNotIn("OpCapability FMAKHR", asm)
            self.assertNotIn("OpCapability Float64", asm)
            for result in fmas:
                self.assertEqual(asm.count(f"OpDecorate {result} NoContraction\n"), 1)

    def test_compensated_series_remain_bounded_loops(self):
        self.assertEqual(v.compile_flags(v.SHADER_ROOT / "compute/microphysics/thompson_completion.comp"), ["-V"])
        with tempfile.TemporaryDirectory() as directory:
            artifact = pathlib.Path(directory) / "pressure.spv"
            v.compile_shader(v.SHADER_ROOT / "compute/acoustic/acoustic_pressure_cartesian_cgrid.comp", artifact)
            v.validate(artifact)
            asm = d.run("spirv-dis", str(artifact))
            for name in ("df_exp", "df_log"):
                identity = re.search(r'OpName (%\S+) "' + name + r'\(', asm)[1]
                body = re.search(re.escape(identity) + r" = OpFunction .*?OpFunctionEnd", asm, re.S)[0]
                self.assertIn("DontInline", body)
                self.assertRegex(body, r"OpLoopMerge %\S+ %\S+ DontUnroll")

    def test_nonportable_cold_keeps_original_function_and_loop_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            source = v.SHADER_ROOT / "compute/microphysics/milbrandt_cold.comp"
            artifact = pathlib.Path(directory) / "cold.spv"
            self.assertEqual(v.compile_flags(source), ["-V"])
            v.compile_shader(source, artifact)
            v.validate(artifact)
            asm = d.run("spirv-dis", str(artifact))
            helpers = set(re.findall(r'OpName (%\S+) "df_[^"]+"', asm))
            self.assertGreater(len(helpers), 0)
            self.assertNotRegex(asm, r'OpName %\S+ "df_product(?:_error)?\(')
            for helper in helpers:
                body = re.search(re.escape(helper) + r" = OpFunction .*?OpFunctionEnd", asm, re.S)[0]
                self.assertNotIn("DontInline", body)
                self.assertNotRegex(body, r"OpLoopMerge [^\n]*DontUnroll")

    def test_linkage_preserves_interfaces_workgroups_and_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = pathlib.Path(directory)
            library = d.build_library(v.SHADER_ROOT / "support/exact_division.glsl", temp)
            source = temp / "fixture.comp"
            source.write_text((v.ROOT / "tests/fixtures/vulkan/rte_division.comp").read_text().replace(
                "void main()", "layout(set=0,binding=8) readonly buffer Reserved { float reserved[]; };\nvoid main()"))
            original, lowered = temp / "original.spv", temp / "lowered.spv"
            subprocess.run(["glslangValidator", "-V", str(source), "-o", str(original)], check=True, stdout=subprocess.DEVNULL)
            self.assertEqual(d.lower(original, library, lowered, temp), 1)
            self.assertEqual(v.reflection(original), v.reflection(lowered))
            asm = d.run("spirv-dis", str(lowered))
            self.assertNotIn("OpFDiv", asm)
            self.assertIn("RoundingModeRTE 32", asm)
            self.assertIn("SignedZeroInfNanPreserve 32", asm)
            self.assertIn("LocalSize 64 1 1", asm)
            self.assertEqual(asm.count("BuiltIn WorkgroupSize"), 1)
            self.assertNotIn("LinkageAttributes", asm)
            self.assertIn("OpCapability Int64", asm)
            source.write_text(source.read_text().replace('capabilities = [4467], 4462, 32', 'capabilities = [4466], 4461, 32'))
            subprocess.run(["glslangValidator", "-V", str(source), "-o", str(original)], check=True, stdout=subprocess.DEVNULL)
            self.assertEqual(d.lower(original, library, lowered, temp), 0)

    def test_variant_identity_changes_with_helper_or_artifact(self):
        row = dict(source_artifact="a.spv", source_hash="source", artifact="variant.spv", artifact_hash="hash", helper_sha256="helper")
        original = d.render_header([row])
        self.assertNotEqual(original, d.render_header([{**row, "artifact_hash": "changed"}]))
        self.assertNotEqual(original, d.render_header([{**row, "helper_sha256": "changed"}]))

    def test_compensated_fma_is_fused_and_cannot_be_reassociated(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = pathlib.Path(directory)
            library = d.build_library(v.SHADER_ROOT / "support/exact_division.glsl", temp)
            original = v.SHADER_ROOT / "compute/time_stepping/rk_combination.comp.spv"
            lowered = temp / "rk.spv"
            self.assertGreater(d.lower(original, library, lowered, temp), 4)
            asm = d.run("spirv-dis", str(lowered))
            self.assertNotIn(" Fma ", asm)
            self.assertIn("OpFmaKHR", asm)
            self.assertIn("NoContraction", asm)
            self.assertEqual(v.reflection(original), v.reflection(lowered))


if __name__ == "__main__":
    unittest.main()
