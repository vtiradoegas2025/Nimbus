#!/usr/bin/env python3
"""Verify the committed shader assets without rebuilding them with a host toolchain."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import division_workaround

ROOT = Path(__file__).resolve().parents[3]
SOURCE_SUFFIXES = {'.comp': 'comp', '.vert': 'vert', '.frag': 'frag'}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fnv1a64(path: Path) -> str:
    value = 0xCBF29CE484222325
    for byte in path.read_bytes():
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f'{value:016x}'


def verify(root: Path) -> tuple[int, int]:
    shader_root = root / 'vulkan/shaders'
    manifest = json.loads((shader_root / 'manifest.json').read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('unknown shader manifest version')
    includes = {p.relative_to(root).as_posix(): sha256(p)
                for p in sorted(shader_root.rglob('*.glsl'))}
    if includes != manifest['include_sha256']:
        raise ValueError('GLSL include inventory/identity changed; review and rebuild the shaders')
    sources = {p.relative_to(root).as_posix() for p in shader_root.rglob('*')
               if p.suffix in SOURCE_SUFFIXES and p.is_file()}
    entries = manifest['shaders']
    declared = [entry['source'] for entry in entries]
    if not sources or len(declared) != len(set(declared)) or set(declared) != sources:
        raise ValueError('shader source inventory differs from the committed manifest')
    artifacts = set()
    hashes = {}
    for entry in entries:
        source = root / entry['source']
        expected_artifact = entry['source'] + '.spv'
        if entry['artifact'] != expected_artifact or entry['stage'] != SOURCE_SUFFIXES[source.suffix]:
            raise ValueError(f'incorrect shader artifact/stage: {entry["source"]}')
        artifact = root / expected_artifact
        if sha256(source) != entry['source_sha256']:
            raise ValueError(f'shader source changed: {entry["source"]}; review and rebuild the shaders')
        if sha256(artifact) != entry['spirv_sha256'] or fnv1a64(artifact) != entry['spirv_fnv1a64']:
            raise ValueError(f'shader artifact changed: {expected_artifact}')
        artifacts.add(expected_artifact)
        if 'vulkan/shaders/compute/' in expected_artifact:
            hashes[expected_artifact.removeprefix('vulkan/shaders/compute/')] = entry['spirv_fnv1a64']
    variants = manifest['division_variants']
    helper_digest = sha256(shader_root / 'support/exact_division.glsl')
    for variant in variants:
        relative = Path(variant['artifact'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[:2] != ('variants', division_workaround.STRATEGY):
            raise ValueError(f'invalid division variant path: {relative}')
        artifact = shader_root / 'compute' / relative
        name = artifact.relative_to(root).as_posix()
        if name in artifacts:
            raise ValueError(f'duplicate shader variant: {name}')
        if hashes.get(variant['source_artifact']) != variant['source_hash']:
            raise ValueError(f'variant source identity changed: {name}')
        if variant['helper_sha256'] != helper_digest or fnv1a64(artifact) != variant['artifact_hash']:
            raise ValueError(f'variant/helper identity changed: {name}')
        artifacts.add(name)
    present = {p.relative_to(root).as_posix() for p in shader_root.rglob('*.spv')}
    if artifacts != present:
        raise ValueError('shader artifact inventory differs from the committed manifest')
    header = root / 'vulkan/include/compute/generated_division_variants.hpp'
    if header.read_text() != division_workaround.render_header(variants):
        raise ValueError('compiled division-variant header differs from its manifest')
    return len(entries), len(variants)


def main() -> int:
    try:
        shaders, variants = verify(ROOT)
    except (OSError, ValueError, KeyError) as error:
        print(f'Shader asset verification failed: {error}', file=sys.stderr)
        return 1
    print(f'Committed shader assets verified: {shaders} shaders, {variants} division variants.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
