#!/usr/bin/env python3
"""Bounded stationary Thompson two-way nesting through public admission."""
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
    parser.add_argument('--initialization', choices=('seeded', 'unseeded'), default='seeded')
    args = parser.parse_args()
    binary, probe = args.binary.resolve(strict=True), args.private_probe.resolve(strict=True)
    identity, probe_identity = sha(binary), sha(probe)
    context = nullcontext(args.output.resolve()) if args.output else tempfile.TemporaryDirectory(prefix='nimbus-thompson-two-way-')
    with context as directory:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise RuntimeError('Thompson two-way evidence directory must be empty')
        shutil.copy2(binary, output / 'nimbus_sim')
        shutil.copy2(probe, output / 'private_probe')
        if sha(output / 'nimbus_sim') != identity or sha(output / 'private_probe') != probe_identity:
            raise RuntimeError('Thompson two-way executable copy identities differ')
        source_shaders = Path(os.environ.get('TMV_SHADER_DIR') or ROOT / 'vulkan/shaders/compute').resolve(strict=True)
        shader_hashes = {}
        for source in sorted(source_shaders.rglob('*.spv')):
            relative = source.relative_to(source_shaders)
            shader_hashes[relative.as_posix()] = sha(source)
            target = output / 'shaders' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if sha(target) != shader_hashes[relative.as_posix()]:
                raise RuntimeError('Thompson two-way shader copy identities differ')
        if not shader_hashes:
            raise RuntimeError('Thompson two-way matrix has no native shader artifacts')
        environment = os.environ.copy()
        environment['TMV_SHADER_DIR'] = str(output / 'shaders')
        groups = [(terrain, advection) for terrain in ('flat', 'bell', 'schar', 'bell3', 'schar3')
                  for advection in ('tvd', 'weno')]

        def run(group):
            terrain, advection = group
            label = f'{terrain}-{advection}'
            command = [sys.executable, str(ROOT / 'scripts/tests/science/test_native_two_way_nested_ready.py'),
                       '--binary', str(output / 'nimbus_sim'), '--private-probe', str(output / 'private_probe'),
                       '--family', 'thompson', '--terrain', terrain, '--advection', advection,
                       '--initialization', args.initialization,
                       '--output', str(output / label)]
            with (output / f'{label}.log').open('w') as log:
                result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                return label, {'status': 'failed', 'returncode': result.returncode}
            receipt = json.loads((output / label / 'summary.json').read_text())
            expected_backends = {'thompson-cpu', 'thompson-metal', 'thompson-vulkan', 'thompson-vulkan-staged'}
            if set(receipt.get('results', {})) != expected_backends:
                raise RuntimeError(f'incomplete Thompson backend coverage: {label}')
            if receipt['binary_sha256'] != identity or receipt['private_probe_sha256'] != probe_identity:
                raise RuntimeError('Thompson two-way executable identities changed')
            return label, receipt

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = dict(pool.map(run, groups))
        passed = len(results) == 10 and all(value['status'] == 'Ready' for value in results.values())
        if sha(output / 'nimbus_sim') != identity or sha(output / 'private_probe') != probe_identity:
            raise RuntimeError('Thompson two-way executed copies changed')
        if any(sha(output / 'shaders' / name) != value for name, value in shader_hashes.items()):
            raise RuntimeError('Thompson two-way executed shaders changed')
        (output / 'summary.json').write_text(json.dumps({
            'status': 'bounded_passed' if passed else 'incomplete',
            'binary_sha256': identity, 'private_probe_sha256': probe_identity,
            'initialization': args.initialization, 'shader_sha256': shader_hashes,
            'expected_groups': 10, 'expected_backend_cases': 40, 'groups': results,
            'campaign_validated': False, 'externally_reviewed': False,
        }, indent=2) + '\n')
        if not passed:
            raise RuntimeError(f'Thompson two-way matrix failed; inspect {output}')
    print(f'[thompson-two-way-ready] PASS: all ten {args.initialization} stationary geometries/reconstructions on CPU, Metal, Vulkan and forced staging; closure, private state, threading and restart')


if __name__ == '__main__':
    main()
