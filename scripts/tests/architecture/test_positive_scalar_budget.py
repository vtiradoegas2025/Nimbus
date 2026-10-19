"""Checks the shared scalar-budget arithmetic and selector boundaries."""

import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vulkan/shaders/compute/advection/positive_scalar_budget.glsl"
GENERATOR = ROOT / "scripts/dev/shaders/generate_positive_scalar_budget_msl.py"

HOST = r"""#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cfenv>
#include <cstdio>
#include <random>
#include <limits>
#pragma STDC FENV_ACCESS ON
using uint = uint32_t;
using std::max;
float uintBitsToFloat(uint u) { float f; std::memcpy(&f,&u,4); return f; }
uint floatBitsToUint(float f) { uint u; std::memcpy(&u,&f,4); return u; }
int findMSB(uint u) { return 31-__builtin_clz(u); }
"""
DRIVER = r"""int main() {
static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559);
if(std::fesetenv(FE_DFL_ENV)!=0 || std::fesetround(FE_TONEAREST)!=0) return 2;
std::mt19937 rng(0x31415926u); unsigned errors=0; uint64_t checks=0;
for (unsigned i=0;i<1048576;i++) {
 uint a=rng(), b=rng();
 if (i%4==0) { a &= 0x807fffffu; }
 if (i%4==1) { a=(a&0x807fffffu)|((1u+(rng()%5u))<<23); b=(b&0x807fffffu)|((1u+(rng()%5u))<<23); }
 if (i%4==2) { b=(a&0x7fffffffu) ^ (rng()%512u); b |= ((a^0x80000000u)&0x80000000u); }
 if ((a&0x7fffffffu)>=0x7f800000u || (b&0x7fffffffu)>=0x7f800000u) continue;
 volatile float x=uintBitsToFloat(a),y=uintBitsToFloat(b);
 volatile float sum=x+y,product=x*y;
 uint expected_sum=floatBitsToUint(sum),expected_product=floatBitsToUint(product);
 uint actual_sum=pd_add_bits(a,b),actual_product=pd_product_bits(a,b);
 auto equivalent=[](uint a,uint b){return a==b || ((a|b)&0x7fffffffu)==0;};
 if(!equivalent(actual_sum,expected_sum)||!equivalent(actual_product,expected_product)) {
  if(errors<10) std::printf("bad i=%u a=%08x b=%08x add=%08x/%08x product=%08x/%08x\n",i,a,b,actual_sum,expected_sum,actual_product,expected_product); ++errors;
 }
 checks+=2;
}
std::fesetround(FE_TONEAREST);
std::printf("{\"checks\":%llu,\"failures\":%u}\n",static_cast<unsigned long long>(checks),errors);
return errors?1:0;
}
"""

class PositiveScalarBudgetTests(unittest.TestCase):
    def test_generated_metal_matches_the_canonical_helper(self):
        spec = importlib.util.spec_from_file_location("positive_scalar_budget_generator", GENERATOR)
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        self.assertEqual(generator.DESTINATION.read_text(), generator.generated_header())

    def test_mapped_dispatch_does_not_enable_the_cartesian_budget(self):
        vulkan = (ROOT / "vulkan/src/compute/compute_backend_vulkan.cpp").read_text()
        metal = (ROOT / "metal/src/compute/backend/terrain.inc").read_text()
        for text, name in ((vulkan, "flux_pc"), (metal, "flux_constants")):
            assignments = re.findall(rf"{name}\.pad0\s*=\s*([^;]+);", text)
            self.assertEqual(len(assignments), 1)
            self.assertNotRegex(assignments[0], r"\b4\b|\bpositive_definite\b")
        flat = [value for value in re.findall(r"pc\.pad0\s*=\s*([^;]+);", vulkan)
                if "dispatch.weno" in value]
        self.assertEqual(len(flat), 1)
        self.assertIn("positive_definite", flat[0])
        self.assertRegex(flat[0], r"\b4\b")

    def test_integer_packing_matches_independent_binary32_operations(self):
        selected = os.environ.get("NIMBUS_TEST_CXX") or os.environ.get("CXX")
        compiler = shlex.split(selected) if selected else [shutil.which("c++") or shutil.which("clang++") or shutil.which("g++") or ""]
        self.assertTrue(compiler[0], "A C++ compiler is required for the binary32 reference")
        helper = SOURCE.read_text().replace("precise float", "volatile float")
        helper = re.sub(r"\bout\s+int\s+(\w+)", r"int& \1", helper)
        with tempfile.TemporaryDirectory(prefix="nimbus-pd-budget-") as directory:
            directory = Path(directory)
            source = directory / "reference.cpp"
            executable = directory / ("reference.exe" if os.name == "nt" else "reference")
            source.write_text(HOST + helper + DRIVER)
            compiled = subprocess.run(compiler + ["-std=c++17", "-O0", "-frounding-math", "-ffp-contract=off",
                                      str(source), "-o", str(executable)],
                                      text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            self.assertEqual(compiled.returncode, 0, compiled.stdout)
            result = subprocess.run([str(executable)], text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout)
            receipt = json.loads(result.stdout)
            self.assertEqual(receipt["checks"], 2088962)
            self.assertEqual(receipt["failures"], 0)


if __name__ == "__main__":
    unittest.main()
