#!/usr/bin/env python3
"""Copy saved storm frames unchanged into a portable, hash-verified visual reference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from generate_visual_load_fixture import atomic_json, history_index


def sha256(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024): sha.update(block)
    return sha.hexdigest()


def copy_unchanged(source, destination):
    cloned = sys.platform == 'darwin' and subprocess.run(
        ['cp', '-c', str(source), str(destination)], capture_output=True).returncode == 0
    if not cloned: shutil.copyfile(source, destination)
    digest = sha256(source)
    if sha256(destination) != digest: raise ValueError(f'Copy changed: {source}')
    return digest


def prepare(source, output, steps):
    source = source.resolve()
    output = output.resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError('Choose a new output directory outside the saved source run')
    if len(steps) < 3 or len(steps) > 16 or steps != sorted(set(steps)):
        raise ValueError('Choose 3–16 unique, increasing saved frame indices')
    sources = [source / f'step_{index:06}' for index in steps]
    field_sets = [sorted(path.name for path in directory.glob('*.npy')) for directory in sources]
    if not field_sets[0] or any(fields != field_sets[0] for fields in field_sets):
        raise ValueError('Selected frames must publish the same NPY field set')
    if not {'cloud_condensate.npy', 'qr.npy', 'w.npy'}.issubset(field_sets[0]):
        raise ValueError('The multifield reference requires saved cloud condensate, rain, and vertical velocity')
    total = sum((directory / name).stat().st_size for directory in sources for name in field_sets[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < total + 4 * 1024**3:
        raise ValueError('Insufficient space for a full independent copy')
    manifests = [json.loads((directory / 'manifest.json').read_text()) for directory in sources]
    if any(manifest['step_index'] != index for manifest, index in zip(manifests, steps)):
        raise ValueError('Saved frame identity does not match its directory')
    shape = list(np.load(sources[0] / 'cloud_condensate.npy', mmap_mode='r', allow_pickle=False).shape)
    specification = dict(schema='nimbus.saved-visual-reference.v1', source_kind='saved-history-reference',
        synthetic=False, values_modified=False, source_name=source.name, frames=len(steps), frame_indices=steps,
        shape=shape, source_grid=manifests[0]['grid'], fields=[Path(name).stem for name in field_sets[0]],
        bytes_per_frame=total // len(steps), purpose='Historical storm appearance and renderer behavior; not current solver validation')
    output.mkdir()
    atomic_json(output / 'visual_load_fixture.json', specification)
    (output / 'README_VISUAL_REFERENCE.txt').write_text(
        'SAVED STORM VISUAL REFERENCE\nAll field files and step manifests are unchanged copies. '
        'Original resolution, times, numerical artifacts, and provenance limitations are retained. '
        'This reference does not claim 50 m detail or validate the current solver.\n')
    entries = []
    receipt = []
    for index, directory in zip(steps, sources):
        target = output / directory.name
        target.mkdir()
        fields = {}
        for name in field_sets[0]:
            digest = copy_unchanged(directory / name, target / name)
            fields[Path(name).stem] = dict(bytes=(target / name).stat().st_size, sha256=digest)
        digest = copy_unchanged(directory / 'manifest.json', target / 'manifest.json')
        entries.append(dict(step_index=index, step_dir=directory.name, manifest_sha256=digest))
        receipt.append(dict(step=index, fields=fields, manifest_sha256=digest))
        print(f'Copied and verified frame {index}', flush=True)
    if (source / 'nimbus_output.json').exists():
        root = json.loads((source / 'nimbus_output.json').read_text())
        generation = manifests[0].get('publication_generation')
        if not generation or any(manifest.get('publication_generation') != generation for manifest in manifests):
            raise ValueError('Versioned frames must share one publication generation')
        root['publication_state'] = 'ready'
        root['visual_reference'] = specification
        atomic_json(output / 'nimbus_output.json', root)
        history_index(output, generation, entries)
    atomic_json(output / 'fixture_receipt.json', dict(specification=specification, frames=receipt,
        total_field_bytes=total, source_directory=str(source), values_modified=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, nargs='+', required=True)
    args = parser.parse_args()
    prepare(args.source, args.output, args.steps)
