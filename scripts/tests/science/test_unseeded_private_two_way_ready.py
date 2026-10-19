#!/usr/bin/env python3
"""Bounded public unseeded private-moment nesting across maintained geometries."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from test_native_two_way_nested_ready import ROOT, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, default=ROOT / 'bin/nimbus_sim')
    parser.add_argument('--private-probe', type=Path, default=ROOT / 'bin/nimbus_private_checkpoint_probe')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args()
    context = nullcontext(args.output.resolve()) if args.output else tempfile.TemporaryDirectory(prefix='nimbus-unseeded-private-')
    with context as directory:
        run_matrix(args, Path(directory))
    print('[unseeded-private-two-way-ready] PASS: 120 backend cases; native ownership, conservation, threading and exact restart')


def run_matrix(args, output):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError('unseeded private evidence directory must be empty')
    binary, probe = args.binary.resolve(strict=True), args.private_probe.resolve(strict=True)
    identity, probe_identity = sha(binary), sha(probe)
    snapshot, probe_snapshot = output / 'nimbus_sim', output / 'private_probe'
    shutil.copy2(binary, snapshot)
    shutil.copy2(probe, probe_snapshot)
    if sha(snapshot) != identity or sha(probe_snapshot) != probe_identity:
        raise RuntimeError('unseeded private executable copy identities differ')
    source_shaders = Path(os.environ.get('TMV_SHADER_DIR') or ROOT / 'vulkan/shaders/compute').resolve(strict=True)
    shader_files = sorted(source_shaders.rglob('*.spv'))
    if not shader_files:
        raise RuntimeError('unseeded private matrix has no Vulkan artifacts')
    shader_hashes = {str(path.relative_to(source_shaders)): sha(path) for path in shader_files}
    shaders = output / 'shaders'
    for path in shader_files:
        target = shaders / path.relative_to(source_shaders)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if sha(target) != shader_hashes[str(path.relative_to(source_shaders))]:
            raise RuntimeError('unseeded private shader copy identities differ')
    environment = os.environ.copy()
    environment['TMV_SHADER_DIR'] = str(shaders)
    groups = [(family, terrain, advection)
              for family in ('thompson', 'milbrandt', 'milbrandt_no_hail')
              for terrain in ('flat', 'bell', 'schar', 'bell3', 'schar3')
              for advection in ('tvd', 'weno')]

    def run(group):
        family, terrain, advection = group
        label = '-'.join(group)
        command = [sys.executable, str(ROOT / 'scripts/tests/science/test_native_two_way_nested_ready.py'),
                   '--binary', str(snapshot), '--private-probe', str(probe_snapshot),
                   '--family', family, '--terrain', terrain, '--advection', advection,
                   '--initialization', 'unseeded', '--output', str(output / label)]
        with (output / f'{label}.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            receipt = {'status': 'failed', 'returncode': result.returncode}
        else:
            receipt = json.loads((output / label / 'summary.json').read_text())
            expected = {f'{family}-{backend}' for backend in ('cpu', 'metal', 'vulkan', 'vulkan-staged')}
            if (set(receipt['results']) != expected or receipt['binary_sha256'] != identity
                    or receipt['private_probe_sha256'] != probe_identity):
                raise RuntimeError(f'unseeded private evidence identity or coverage differs: {label}')
        print(label, receipt['status'], flush=True)
        return label, receipt

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = dict(pool.map(run, groups))
    if (sha(snapshot) != identity or sha(probe_snapshot) != probe_identity
            or any(sha(shaders / path) != value for path, value in shader_hashes.items())):
        raise RuntimeError('executed unseeded private inputs changed')
    passed = all(receipt['status'] == 'Ready' for receipt in results.values())
    (output / 'summary.json').write_text(json.dumps({
        'status': 'bounded_passed' if passed else 'incomplete',
        'binary_sha256': identity, 'private_probe_sha256': probe_identity,
        'shader_sha256': shader_hashes, 'expected_groups': 30, 'expected_backend_cases': 120,
        'groups': results, 'campaign_validated': False, 'externally_reviewed': False,
    }, indent=2) + '\n')
    if not passed:
        raise RuntimeError(f'unseeded private matrix failed; inspect {output}')


if __name__ == '__main__':
    main()
