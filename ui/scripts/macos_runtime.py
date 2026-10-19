#!/usr/bin/env python3
"""Bundle and verify the Mach-O dependency graph used by Nimbus Studio."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import plistlib
import shutil
import struct
import subprocess

SYSTEM_PREFIXES = ('/System/Library/', '/usr/lib/')
ARCHITECTURES = {0x01000007: 'x86_64', 0x0100000c: 'arm64'}
LOAD_DYLIB = {0xc, 0x80000018, 0x8000001f, 0x80000023, 0x20}
MANIFEST = 'Contents/Resources/macos-runtime.json'


def version(value):
    parts = [int(part) for part in value.split('.')]
    if not 1 <= len(parts) <= 3 or parts[0] < 0 or parts[0] > 65535 or any(part < 0 or part > 255 for part in parts[1:]):
        raise ValueError('invalid macOS version: ' + value)
    return tuple((parts + [0, 0])[:3])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_bytes(data):
    if len(data) < 8:
        raise ValueError('truncated Mach-O header')
    magic = data[:4]
    if magic in (b'\xca\xfe\xba\xbe', b'\xca\xfe\xba\xbf'):
        count = struct.unpack_from('>I', data, 4)[0]
        size = 32 if magic[-1] == 0xbf else 20
        if count < 1 or count > 64 or 8 + count * size > len(data):
            raise ValueError('invalid universal Mach-O table')
        result = {}
        for i in range(count):
            fmt = '>IIQQII' if size == 32 else '>IIIII'
            entry = struct.unpack_from(fmt, data, 8 + i * size)
            start, length = entry[2:4]
            if start < 8 + count * size or length < 32 or start + length > len(data):
                raise ValueError('invalid universal Mach-O slice')
            slices = inspect_bytes(data[start:start + length])
            if len(slices) != 1 or set(result).intersection(slices):
                raise ValueError('duplicate or nested universal Mach-O slice')
            if ARCHITECTURES.get(entry[0]) not in slices:
                raise ValueError('universal architecture table disagrees with slice')
            result.update(slices)
        return result
    if magic != b'\xcf\xfa\xed\xfe' or len(data) < 32:
        raise ValueError('expected a supported 64-bit Mach-O image')
    _, cpu, subtype, filetype, count, size, _, _ = struct.unpack_from('<8I', data)
    arch = ARCHITECTURES.get(cpu)
    if arch is None or (arch == 'x86_64' and subtype & 0xffffff != 3) or (arch == 'arm64' and subtype & 0xffffff not in (0, 1)):
        raise ValueError('unsupported or host-specialized Mach-O architecture')
    if count > 4096 or 32 + size > len(data):
        raise ValueError('invalid Mach-O load-command table')
    deps, rpaths, sections, minimum = [], [], {}, None
    cursor = 32
    for _ in range(count):
        if cursor + 8 > 32 + size:
            raise ValueError('truncated Mach-O load command')
        cmd, length = struct.unpack_from('<II', data, cursor)
        if length < 8 or cursor + length > 32 + size:
            raise ValueError('invalid Mach-O load-command size')
        payload = data[cursor:cursor + length]
        if cmd in LOAD_DYLIB or cmd == 0x8000001c:
            if length < 12:
                raise ValueError('truncated Mach-O path command')
            offset = struct.unpack_from('<I', payload, 8)[0]
            if offset < 12 or offset >= length or b'\0' not in payload[offset:]:
                raise ValueError('invalid Mach-O path offset')
            value = payload[offset:].split(b'\0', 1)[0].decode('utf-8')
            (rpaths if cmd == 0x8000001c else deps).append(value)
        elif cmd in (0x32, 0x24):
            if length < (24 if cmd == 0x32 else 16):
                raise ValueError('truncated Mach-O deployment command')
            if cmd == 0x32 and struct.unpack_from('<I', payload, 8)[0] != 1:
                raise ValueError('non-macOS image in a macOS bundle')
            packed = struct.unpack_from('<I', payload, 12 if cmd == 0x32 else 8)[0]
            found = (packed >> 16, (packed >> 8) & 255, packed & 255)
            minimum = max(minimum or found, found)
        elif cmd == 0x19:
            if length < 72:
                raise ValueError('truncated Mach-O segment')
            nsects = struct.unpack_from('<I', payload, 64)[0]
            if 72 + nsects * 80 > length:
                raise ValueError('invalid Mach-O section table')
            for i in range(nsects):
                at = 72 + i * 80
                name = payload[at:at + 16].split(b'\0', 1)[0].decode()
                segment = payload[at + 16:at + 32].split(b'\0', 1)[0].decode()
                extent = struct.unpack_from('<Q', payload, at + 40)[0]
                offset = struct.unpack_from('<I', payload, at + 48)[0]
                flags = struct.unpack_from('<I', payload, at + 64)[0]
                if not extent or flags & 255 in (1, 12, 18):
                    continue
                if offset < 32 + size or offset + extent > len(data):
                    raise ValueError('invalid Mach-O section extent')
                key = segment + '/' + name
                if key in sections:
                    raise ValueError('duplicate Mach-O section')
                sections[key] = hashlib.sha256(data[offset:offset + extent]).hexdigest()
        cursor += length
    if cursor != 32 + size or minimum is None:
        raise ValueError('missing deployment target or malformed Mach-O commands')
    return {arch: dict(minimum=minimum, dependencies=deps, rpaths=rpaths,
                       sections=sections, filetype=filetype)}


def inspect(path):
    return inspect_bytes(path.read_bytes())


def is_system(name):
    return name.startswith(SYSTEM_PREFIXES)


def compatible(info, architectures, target, label):
    if not architectures.issubset(info):
        raise ValueError(f'{label}: missing architectures {sorted(architectures - set(info))}')
    for arch in architectures:
        if tuple(info[arch]['minimum']) > target:
            raise ValueError(f'{label}: {arch} requires macOS {info[arch]["minimum"]}, target is {target}')


def roots(app):
    metadata = plistlib.loads((app / 'Contents/Info.plist').read_bytes())
    return [app / 'Contents/MacOS' / metadata['CFBundleExecutable'],
            app / 'Contents/Resources/bin/nimbus_sim']


def expand(name, owner, executable):
    for prefix, parent in (('@loader_path/', owner.parent), ('@executable_path/', executable.parent)):
        if name.startswith(prefix):
            return parent / name[len(prefix):]
    return Path(name)


def prepare(app, simulator, prefix, extras, target):
    app = app.resolve(strict=True)
    simulator = simulator.resolve(strict=True)
    prefix = prefix.resolve(strict=True)
    extras = [p.resolve(strict=True) for p in extras]
    binaries = roots(app)
    if sha(binaries[1]) != sha(simulator):
        raise ValueError('bundled simulator differs from the selected source binary')
    architectures = set(inspect(binaries[0]))
    allowed = [prefix.resolve()]
    extra_paths = {p.resolve() for p in extras}
    graph, queue = {}, [(p.resolve(), p.resolve(), ()) for p in binaries + extras]
    while queue:
        path, executable, inherited = queue.pop(0)
        if path in graph:
            continue
        info = inspect(path)
        compatible(info, architectures, target, str(path))
        dependencies = {}
        for arch in architectures:
            search = tuple(expand(p, path, executable) for p in info[arch]['rpaths']) + inherited
            for name in info[arch]['dependencies']:
                if is_system(name):
                    continue
                candidates = [p / name[7:] for p in search] if name.startswith('@rpath/') else [expand(name, path, executable)]
                found = next((p.resolve() for p in candidates if p.is_file()), None)
                if found is None:
                    raise ValueError(f'{path}: unresolved dependency {name}')
                if found not in extra_paths and not any(found.is_relative_to(p) for p in allowed) and not found.is_relative_to(app):
                    raise ValueError(f'{path}: dependency escaped declared runtime prefix: {found}')
                if name in dependencies and dependencies[name] != found:
                    raise ValueError('architecture-dependent dependency requires a universal library: ' + name)
                dependencies[name] = found
                queue.append((found, executable, search))
        graph[path] = dict(info=info, dependencies=dependencies, source_sha256=sha(path))
    destination = {p.resolve(): p for p in binaries}
    frameworks = app / 'Contents/Frameworks'
    names = {}
    for source in graph:
        if source in destination:
            continue
        name = source.name
        if name in names and names[name] != source:
            raise ValueError('runtime library basename collision: ' + name)
        names[name] = source
        destination[source] = frameworks / name
    frameworks.mkdir(parents=True, exist_ok=True)
    for source, dest in destination.items():
        if source != dest:
            if dest.exists() or dest.is_symlink():
                raise ValueError('refusing to overwrite existing runtime: ' + str(dest))
            shutil.copy2(source, dest)
            dest.chmod(dest.stat().st_mode | 0o200)
    records = {}
    for source, entry in graph.items():
        dest = destination[source]
        args = ['install_name_tool']
        for old, dependency in entry['dependencies'].items():
            relative = os.path.relpath(destination[dependency], dest.parent)
            args += ['-change', old, '@loader_path/' + relative]
        for rpath in dict.fromkeys(p for a in entry['info'].values() for p in a['rpaths']):
            args += ['-delete_rpath', rpath]
        if all(a['filetype'] == 6 for a in entry['info'].values()):
            args += ['-id', '@rpath/' + dest.name]
        if len(args) > 1:
            subprocess.run(args + [str(dest)], check=True, capture_output=True)
        actual = inspect(dest)
        for arch in architectures:
            if actual[arch]['sections'] != entry['info'][arch]['sections']:
                raise ValueError('library relocation changed a code/data section: ' + str(dest))
        records[str(dest.relative_to(app))] = dict(source=str(source), source_sha256=entry['source_sha256'], image=actual)
    aliases = {}
    for stem in ('libnetcdf', 'libhdf5', 'libhdf5_hl', 'libMoltenVK'):
        candidates = [p for p in destination.values() if p.parent == frameworks and
                      (p.name == stem + '.dylib' or p.name.startswith(stem + '.') and p.name.endswith('.dylib'))]
        if len(candidates) != 1:
            raise ValueError('one dynamic runtime entry is required for ' + stem)
        alias = frameworks / (stem + '.dylib')
        if alias != candidates[0]:
            alias.symlink_to(candidates[0].name)
        aliases[str(alias.relative_to(app))] = str(candidates[0].relative_to(app))
    manifest = dict(schema='nimbus.macos-runtime.v1', minimum_macos=list(target), architectures=sorted(architectures),
                    source_simulator_sha256=sha(simulator), images=records, aliases=aliases)
    (app / MANIFEST).write_text(json.dumps(manifest, indent=2) + '\n')
    verify(app, simulator, target)


def verify(app, simulator, target):
    app = app.resolve(strict=True)
    manifest = json.loads((app / MANIFEST).read_text())
    architectures = set(manifest['architectures'])
    if tuple(manifest['minimum_macos']) != target:
        raise ValueError('runtime manifest deployment target mismatch')
    metadata = plistlib.loads((app / 'Contents/Info.plist').read_bytes())
    if version(metadata.get('LSMinimumSystemVersion', '0')) != target:
        raise ValueError('Info.plist must declare the selected deployment target')
    if simulator and sha(simulator) != manifest['source_simulator_sha256']:
        raise ValueError('source simulator changed since packaging')
    for root in roots(app):
        if str(root.relative_to(app)) not in manifest['images']:
            raise ValueError('unverified bundle executable: ' + str(root))
    for alias, target_path in manifest['aliases'].items():
        if (app / alias).resolve() != app / target_path:
            raise ValueError('runtime alias changed: ' + alias)
    for rel, expected in manifest['images'].items():
        path = app / rel
        if not path.resolve().is_relative_to(app):
            raise ValueError('runtime image escaped bundle')
        info = inspect(path)
        compatible(info, architectures, target, rel)
        for arch in architectures:
            if info[arch] != {**expected['image'][arch], 'minimum': tuple(expected['image'][arch]['minimum'])}:
                raise ValueError('runtime image metadata or sections changed: ' + rel)
            if info[arch]['rpaths']:
                raise ValueError('runtime image retains search paths: ' + rel)
            for name in info[arch]['dependencies']:
                if is_system(name):
                    continue
                if not name.startswith('@loader_path/'):
                    raise ValueError('external runtime dependency: ' + name)
                resolved = expand(name, path, path).resolve()
                if not resolved.is_relative_to(app) or str(resolved.relative_to(app)) not in manifest['images']:
                    raise ValueError('runtime dependency is not a verified bundled image: ' + name)
    print(f"macOS runtime verified: {len(manifest['images'])} images, {', '.join(sorted(architectures))}, minimum {'.'.join(map(str, target))}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'verify'))
    p.add_argument('--app', type=Path, required=True)
    p.add_argument('--simulator', type=Path)
    p.add_argument('--runtime-prefix', type=Path)
    p.add_argument('--extra-library', type=Path, action='append', default=[])
    p.add_argument('--minimum-macos', default='12.0')
    a = p.parse_args()
    app = a.app.resolve(strict=True)
    target = version(a.minimum_macos)
    if a.mode == 'prepare':
        if not a.simulator or not a.runtime_prefix:
            p.error('prepare requires source simulator and runtime prefix')
        prepare(app, a.simulator.resolve(strict=True), a.runtime_prefix, a.extra_library, target)
    else:
        verify(app, a.simulator, target)


if __name__ == '__main__':
    main()
