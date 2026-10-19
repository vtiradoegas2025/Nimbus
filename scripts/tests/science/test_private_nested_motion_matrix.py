#!/usr/bin/env python3
"""Bounded matrices for the maintained private moving and recycling families."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, default=ROOT / 'bin/nimbus_sim')
    parser.add_argument('--private-probe', type=Path, default=ROOT / 'bin/nimbus_private_checkpoint_probe')
    parser.add_argument('--family', choices=('moving', 'recycling'), required=True)
    parser.add_argument('--microphysics', choices=('thompson', 'milbrandt', 'milbrandt_no_hail'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=2)
    parser.add_argument('--restart-coverage', choices=('full', 'synchronized'), default='full')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError('private motion matrix evidence directory must be empty')
    binary, probe = args.binary.resolve(strict=True), args.private_probe.resolve(strict=True)
    identity, probe_identity = sha(binary), sha(probe)
    snapshot, probe_snapshot = output / 'nimbus_sim', output / 'private_probe'
    shutil.copy2(binary, snapshot)
    shutil.copy2(probe, probe_snapshot)
    if sha(snapshot) != identity or sha(probe_snapshot) != probe_identity:
        raise RuntimeError('private motion executable copy identities differ')
    source_shader_root = Path(os.environ.get('TMV_SHADER_DIR') or ROOT / 'vulkan/shaders/compute').resolve(strict=True)
    shader_files = sorted(source_shader_root.rglob('*.spv'))
    if not shader_files:
        raise RuntimeError('private motion matrix has no Vulkan shader artifacts')
    shader_hashes = {str(path.relative_to(source_shader_root)): sha(path) for path in shader_files}
    shader_root = output / 'shaders'
    for path in shader_files:
        target = shader_root / path.relative_to(source_shader_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if sha(target) != shader_hashes[str(path.relative_to(source_shader_root))]:
            raise RuntimeError('private motion shader copy identities differ')
    environment = os.environ.copy()
    environment['TMV_SHADER_DIR'] = str(shader_root)
    terrains = ('flat', 'bell2', 'schar2', 'bell3', 'schar3') if args.family == 'moving' else ('flat',)
    advections = ('tvd', 'weno')
    groups = [(terrain, advection) for terrain in terrains for advection in advections]

    def run(group):
        terrain, advection = group
        label = f'{terrain}-{advection}'
        coverage = args.restart_coverage
        if (terrain, advection) in (('schar2', 'tvd'), ('bell3', 'weno')):
            coverage = 'full'
        command = [sys.executable, str(ROOT / 'scripts/tests/science/test_native_nested_motion_ready.py'),
                   '--binary', str(snapshot), '--private-probe', str(probe_snapshot),
                   '--family', args.family, '--microphysics', args.microphysics,
                   '--terrain', terrain, '--advection', advection, '--restart-coverage', coverage,
                   '--output', str(output / label)]
        with (output / f'{label}.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            receipt = {'status': 'failed', 'returncode': result.returncode}
        else:
            receipt = json.loads((output / label / 'summary.json').read_text())
            expected = {f'{args.family}-{backend}' for backend in ('cpu', 'metal', 'vulkan', 'vulkan-staged')}
            if (set(receipt['results']) != expected or receipt['binary_sha256'] != identity
                    or receipt['private_probe_sha256'] != probe_identity):
                raise RuntimeError(f'incomplete private motion evidence: {label}')
        print(label, receipt['status'], flush=True)
        return label, receipt

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = dict(pool.map(run, groups))
    if (sha(snapshot) != identity or sha(probe_snapshot) != probe_identity
            or any(sha(shader_root / name) != value for name, value in shader_hashes.items())):
        raise RuntimeError('executed private motion inputs changed')
    passed = all(receipt['status'] == 'Ready' for receipt in results.values())
    (output / 'summary.json').write_text(json.dumps({
        'status': 'bounded_passed' if passed else 'incomplete',
        'family': args.family, 'microphysics': args.microphysics,
        'binary_sha256': identity, 'private_probe_sha256': probe_identity,
        'source_shader_root': str(source_shader_root), 'shader_root': str(shader_root),
        'shader_sha256': shader_hashes, 'expected_groups': len(groups),
        'expected_backend_cases': 4 * len(groups), 'groups': results,
        'campaign_validated': False, 'externally_reviewed': False,
    }, indent=2) + '\n')
    if not passed:
        raise RuntimeError(f'private motion matrix failed; inspect {output}')


if __name__ == '__main__':
    main()
