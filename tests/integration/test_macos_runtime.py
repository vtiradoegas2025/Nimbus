import importlib.util
import json
from pathlib import Path
import plistlib
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('macos_runtime', ROOT / 'ui/scripts/macos_runtime.py')
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def image(minimum=0x000b0000, cpu=0x0100000c):
    command = struct.pack('<6I', 0x32, 24, 1, minimum, 0x001a0000, 0)
    return struct.pack('<8I', 0xfeedfacf, cpu, 0 if cpu == 0x0100000c else 3, 2, 1, 24, 0, 0) + command


class MachOInspectionTests(unittest.TestCase):
    def test_minimum_and_architecture_are_per_slice(self):
        a, b = image(), image(0x000d0000, 0x01000007)
        fat = struct.pack('>II', 0xcafebabe, 2)
        fat += struct.pack('>5I', 0x0100000c, 0, 48, len(a), 0)
        fat += struct.pack('>5I', 0x01000007, 3, 48 + len(a), len(b), 0)
        parsed = runtime.inspect_bytes(fat + a + b)
        runtime.compatible(parsed, {'arm64'}, (11, 0, 0), 'test')
        with self.assertRaisesRegex(ValueError, 'requires macOS'):
            runtime.compatible(parsed, {'arm64', 'x86_64'}, (11, 0, 0), 'test')

    def test_malformed_and_missing_architectures_reject(self):
        for data in (b'', image()[:40], image()[:-1], b'not a Mach-O'):
            with self.assertRaises(ValueError): runtime.inspect_bytes(data)
        with self.assertRaisesRegex(ValueError, 'missing architectures'):
            runtime.compatible(runtime.inspect_bytes(image()), {'x86_64'}, (11, 0, 0), 'test')
        bad = bytearray(image());struct.pack_into('<I', bad, 36, 1000)
        with self.assertRaises(ValueError): runtime.inspect_bytes(bad)

    def test_path_offset_must_stay_inside_command(self):
        data = bytearray(image());struct.pack_into('<I', data, 32, 0xc);struct.pack_into('<I', data, 40, 100)
        with self.assertRaisesRegex(ValueError, 'path offset'): runtime.inspect_bytes(data)


@unittest.skipUnless(sys.platform == 'darwin', 'Mach-O relocation requires Apple build tools')
class MacOSBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='nimbus runtime ')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prefix = self.root / 'source prefix'
        (self.prefix / 'lib').mkdir(parents=True)
        self.app = self.root / 'Relocated App.app'
        for path in ('Contents/MacOS', 'Contents/Resources/bin'):
            (self.app / path).mkdir(parents=True)
        (self.app / 'Contents/Info.plist').write_bytes(plistlib.dumps(
            dict(CFBundleExecutable='studio', LSMinimumSystemVersion='11.0')))
        for name, body, link in [
            ('hdf5', 'int hdf(void){return 7;}', []),
            ('hdf5_hl', 'int hdf(void);int hl(void){return hdf();}', ['-lhdf5']),
            ('netcdf', 'int hl(void);int nc(void){return hl();}', ['-lhdf5_hl']),
            ('MoltenVK', 'int driver(void){return 0;}', []),
        ]:
            source = self.prefix / (name + '.c');source.write_text(body)
            self.run_tool(['xcrun', 'clang', '-mmacosx-version-min=11.0', '-dynamiclib', str(source),
                '-L' + str(self.prefix / 'lib'), *link, '-Wl,-rpath,' + str(self.prefix / 'lib'),
                '-Wl,-install_name,@rpath/lib' + name + '.dylib', '-Wl,-headerpad_max_install_names',
                '-o', str(self.prefix / 'lib' / ('lib' + name + '.dylib'))])
        source = self.prefix / 'main.c';source.write_text('int nc(void);int main(void){return nc()==7?0:1;}')
        self.simulator = self.prefix / 'nimbus_sim'
        self.run_tool(['xcrun', 'clang', '-mmacosx-version-min=11.0', str(source),
            '-L' + str(self.prefix / 'lib'), '-lnetcdf', '-Wl,-rpath,' + str(self.prefix / 'lib'),
            '-Wl,-headerpad_max_install_names', '-o', str(self.simulator)])
        shutil.copy2(self.simulator, self.app / 'Contents/MacOS/studio')
        shutil.copy2(self.simulator, self.app / 'Contents/Resources/bin/nimbus_sim')

    def run_tool(self, command):
        subprocess.run(command, check=True, capture_output=True)

    def prepare(self):
        runtime.prepare(self.app, self.simulator, self.prefix,
                        [self.prefix / 'lib/libMoltenVK.dylib'], (11, 0, 0))

    def test_transitive_runtime_survives_removing_build_prefix(self):
        self.prepare()
        manifest = json.loads((self.app / runtime.MANIFEST).read_text())
        for rel in manifest['images']:
            self.run_tool(['codesign', '--force', '--sign', '-', str(self.app / rel)])
        self.prefix.rename(self.root / 'hidden source prefix')
        runtime.verify(self.app, None, (11, 0, 0))
        self.run_tool([str(self.app / 'Contents/Resources/bin/nimbus_sim')])
        self.run_tool([str(self.app / 'Contents/MacOS/studio')])

    def test_newer_dependency_rejects_before_rewriting(self):
        source = self.prefix / 'MoltenVK.c'
        self.run_tool(['xcrun', 'clang', '-mmacosx-version-min=13.0', '-dynamiclib', str(source),
            '-o', str(self.prefix / 'lib/libMoltenVK.dylib')])
        before = (self.app / 'Contents/Resources/bin/nimbus_sim').read_bytes()
        with self.assertRaisesRegex(ValueError, 'requires macOS'): self.prepare()
        self.assertEqual(before, (self.app / 'Contents/Resources/bin/nimbus_sim').read_bytes())

    def test_changed_code_is_detected(self):
        self.prepare()
        path = self.app / 'Contents/Resources/bin/nimbus_sim'
        data = bytearray(path.read_bytes())
        at = data.index(b'__text')
        offset = struct.unpack_from('<I', data, at + 48)[0]
        data[offset] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'metadata or sections changed'):
            runtime.verify(self.app, None, (11, 0, 0))

    def test_packaged_lookup_keeps_missing_libraries_inside_the_bundle(self):
        source = self.prefix / 'lookup.cpp'
        source.write_text('#include "util/packaged_runtime.hpp"\n#include <iostream>\n'
                          'int main(){std::cout << tmv::packaged_runtime::library("missing.dylib");}')
        self.run_tool(['xcrun', 'clang++', '-std=c++17', '-mmacosx-version-min=11.0',
                       '-I' + str(ROOT / 'include'), str(source), '-o', str(self.simulator)])
        self.assertEqual(subprocess.check_output([str(self.simulator)]), b'')
        for path in runtime.roots(self.app):
            shutil.copy2(self.simulator, path)
            self.assertEqual(subprocess.check_output([str(path)], text=True),
                             str(self.app.resolve() / 'Contents/Frameworks/missing.dylib'))


if __name__ == '__main__': unittest.main()
