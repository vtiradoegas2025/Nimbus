#!/usr/bin/env python3
"""Exercise compiler probing without a platform-specific compiler install."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
PROBE = ROOT / "scripts/dependencies/probe_openmp.sh"
BASH = shutil.which("bash")


class OpenMpProbeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(BASH, "Bash is required for the OpenMP probe contract")
        self.temporary = tempfile.TemporaryDirectory(prefix="nimbus probe test ")
        self.directory = Path(self.temporary.name)
        self.compiler = self.directory / "compiler with spaces"
        self.log = self.directory / "compiler.json"
        self.compiler.write_text("""#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
source = pathlib.Path(args[args.index('-fopenmp') + 1])
output = pathlib.Path(args[args.index('-o') + 1])
assert '#include <omp.h>' in source.read_text()
assert 'omp_get_max_threads()' in source.read_text()
assert output.name == 'probe.exe' and output.parent == source.parent
pathlib.Path(os.environ['NIMBUS_PROBE_LOG']).write_text(json.dumps({'output': str(output)}))
output.write_bytes(b'compile fixture, never executable')
if os.environ.get('NIMBUS_PROBE_FAIL') == '1': sys.exit(1)
print('warning: a successful compiler may emit diagnostics', file=sys.stderr)
""", newline="\n")
        self.compiler.chmod(0o755)
        self.environment = dict(os.environ, NIMBUS_PROBE_LOG=str(self.log))

    def tearDown(self):
        self.temporary.cleanup()

    def test_success_with_diagnostics_and_path_spaces_is_accepted_and_cleaned(self):
        result = subprocess.run([BASH, PROBE.as_posix(), self.compiler.as_posix()], env=self.environment, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, b"")
        self.assertFalse(Path(json.loads(self.log.read_text())["output"]).parent.exists())

    def test_silent_failure_is_rejected_and_cleaned(self):
        result = subprocess.run([BASH, PROBE.as_posix(), self.compiler.as_posix()],
                                env={**self.environment, "NIMBUS_PROBE_FAIL": "1"}, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(Path(json.loads(self.log.read_text())["output"]).parent.exists())

    def test_make_admission_uses_exit_status(self):
        makefile = self.directory / "probe.mk"
        makefile.write_text("""UNAME_S := MINGW64_NT
WINDOWS_HOST := 1
CPU_ONLY := 1
ALLOW_SERIAL := 1
ZFP := 0
NETCDF := 0
include mk/dependencies.mk
all:
\t@printf '%s\\n' '$(OPENMP_FLAG)'
""", newline="\n")
        command = ["make", "--no-print-directory", "-s", "-f", makefile.as_posix(),
                   "CXX=" + shlex.quote(self.compiler.as_posix()), "LIBOMP_PATH="]
        for fails, expected in [(False, "-fopenmp"), (True, "")]:
            result = subprocess.run(command, cwd=ROOT,
                env={**self.environment, "NIMBUS_PROBE_FAIL": str(int(fails))}, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), expected)


if __name__ == "__main__":
    unittest.main()
