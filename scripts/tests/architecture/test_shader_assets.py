#!/usr/bin/env python3
"""Exercise ordinary-build shader identity checks without a shader toolchain."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/dev/shaders'))
import check_artifacts as assets
import division_workaround


class ShaderAssets(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='nimbus-shader-assets-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.shader = self.root / 'vulkan/shaders/compute/fixture.comp'
        self.shader.parent.mkdir(parents=True)
        self.shader.write_text('#version 450\nvoid main() {}\n')
        self.binary = self.shader.with_suffix('.comp.spv')
        self.binary.write_bytes(bytes.fromhex('0302230700000100000000000100000000000000'))
        self.helper = self.root / 'vulkan/shaders/support/exact_division.glsl'
        self.helper.parent.mkdir()
        self.helper.write_text('// identity fixture\n')
        self.include = self.shader.with_name('included.glsl')
        self.include.write_text('// another include\n')
        self.header = self.root / 'vulkan/include/compute/generated_division_variants.hpp'
        self.header.parent.mkdir(parents=True)
        self.header.write_text(division_workaround.render_header([]))
        self.manifest = {
            'schema_version': 1,
            'include_sha256': {p.relative_to(self.root).as_posix(): assets.sha256(p)
                               for p in (self.helper, self.include)},
            'shaders': [{'source': self.shader.relative_to(self.root).as_posix(),
                         'artifact': self.binary.relative_to(self.root).as_posix(),
                         'stage': 'comp', 'source_sha256': assets.sha256(self.shader),
                         'spirv_sha256': assets.sha256(self.binary),
                         'spirv_fnv1a64': assets.fnv1a64(self.binary)}],
            'division_variants': [],
        }
        self.save_manifest()

    def save_manifest(self):
        (self.root / 'vulkan/shaders/manifest.json').write_text(json.dumps(self.manifest))

    def test_current_assets_need_no_external_tools(self):
        with patch.dict(os.environ, {'PATH': ''}):
            self.assertEqual(assets.verify(self.root), (1, 0))

    def test_changed_shader_is_rejected(self):
        self.shader.write_text(self.shader.read_text() + '// changed source\n')
        with self.assertRaisesRegex(ValueError, 'shader source changed'):
            assets.verify(self.root)

    def test_changed_included_science_is_rejected(self):
        self.include.write_text('// changed included arithmetic\n')
        with self.assertRaisesRegex(ValueError, 'GLSL include inventory/identity'):
            assets.verify(self.root)

    def test_changed_artifact_is_rejected(self):
        self.binary.write_bytes(self.binary.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'shader artifact changed'):
            assets.verify(self.root)

    def test_new_source_requires_a_manifest_entry(self):
        self.shader.with_name('new.comp').write_text('void main() {}')
        with self.assertRaisesRegex(ValueError, 'source inventory'):
            assets.verify(self.root)

    def test_variant_source_and_compiled_header_are_bound(self):
        name = f'variants/{division_workaround.STRATEGY}/fixture.comp.spv'
        binary = self.shader.parent / name
        binary.parent.mkdir(parents=True)
        binary.write_bytes(self.binary.read_bytes() + b'variant')
        variant = {'source_artifact': 'fixture.comp.spv',
                   'source_hash': assets.fnv1a64(self.binary),
                   'artifact': name, 'artifact_hash': assets.fnv1a64(binary),
                   'helper_sha256': assets.sha256(self.helper),
                   'arithmetic_rewrites': 1, 'requires_fma': False}
        self.manifest['division_variants'] = [variant]
        self.header.write_text(division_workaround.render_header([variant]))
        self.save_manifest()
        self.assertEqual(assets.verify(self.root), (1, 1))
        variant['source_hash'] = '0000000000000000'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'variant source identity'):
            assets.verify(self.root)
        variant['source_hash'] = assets.fnv1a64(self.binary)
        self.save_manifest()
        self.header.write_text(division_workaround.render_header([]))
        with self.assertRaisesRegex(ValueError, 'compiled division-variant header'):
            assets.verify(self.root)


if __name__ == '__main__':
    unittest.main()
