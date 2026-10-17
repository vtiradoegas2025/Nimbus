import pathlib
import subprocess
import tempfile
import unittest

import division_workaround as d
import verify_spirv as v


class DivisionVariants(unittest.TestCase):
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
