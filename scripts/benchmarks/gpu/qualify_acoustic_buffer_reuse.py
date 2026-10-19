#!/usr/bin/env python3
"""Bounded Linux allocation and exact-science gate for acoustic buffer reuse."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).with_name('field_allocation_probe.c')
SPEC = importlib.util.spec_from_file_location('flat_lin_gate', ROOT / 'scripts/qualification/accelerators/qualify_flat_lin_storm.py')
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, required=True)
    parser.add_argument('--seconds', type=int, default=1)
    args = parser.parse_args()
    if platform.system() != 'Linux' or args.threads < 1 or not 1 <= args.seconds <= 4:
        parser.error('use a Linux Vulkan host, positive workers and 1–4 simulated seconds')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    binary = {name: path.resolve(strict=True) for name, path in [('baseline', args.baseline), ('candidate', args.candidate)]}
    config = ROOT / 'configs/custom/rtx5090_lifecycle_50m/smoke/01_classic_supercell.yaml'
    text = config.read_text()
    shape = [int(re.search(rf'^  {axis}: (\d+)$', text, re.M)[1]) for axis in ('nx', 'ny', 'nz')]
    substeps = int(re.search(r'^  acoustic_substeps: (\d+)$', text, re.M)[1])
    size = 4 * shape[0] * shape[1] * shape[2]
    aligned_size = (size + 16383) // 16384 * 16384
    library = args.output / 'field_allocation_probe.so'
    subprocess.run(['cc', '-shared', '-fPIC', '-O2', str(SOURCE), '-ldl', '-o', str(library)], check=True)
    evidence = dict(schema='nimbus.acoustic-buffer-reuse.v1', passed=False, performance_claim=False,
                    probe_source_sha256=gate.sha(SOURCE), config_sha256=gate.sha(config), shape=shape,
                    binary_sha256={name: gate.sha(path) for name, path in binary.items()}, runs={})
    try:
        for name, executable in binary.items():
            output = args.output / name
            env = {k: v for k, v in os.environ.items() if not k.startswith(('NIMBUS_', 'OMP_', 'KMP_', 'GOMP_'))}
            env.update(LD_PRELOAD=str(library), NIMBUS_PROBE_FIELD_BYTES=str(aligned_size))
            command = [str(executable), '--headless', '--config=' + str(config), '--compute-backend=vulkan',
                       '--threads=' + str(args.threads), '--duration=' + str(args.seconds),
                       '--write-every=' + str(args.seconds), '--checkpoint-every=' + str(args.seconds),
                       '--outdir=' + str(output)]
            with (args.output / (name + '.log')).open('w') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=180)
            log = (args.output / (name + '.log')).read_text()
            if 'Execution mode: qualified' not in log or 'fallback=on' in log or '[COMPUTE] backend=vulkan,' not in log:
                raise RuntimeError('ordinary native Vulkan admission was not used')
            if not re.search(r'\[RUN SETTINGS\].*guard_mode=strict', log):
                raise RuntimeError('strict science guards were not used')
            matches = re.findall(r'\[FIELD ALLOCATION PROBE\] bytes=(\d+) allocations=(\d+) requested_bytes=(\d+)', log)
            if len(matches) != 1 or int(matches[0][0]) != aligned_size:
                raise RuntimeError('one matching allocation receipt is required')
            perf = re.search(r'\[PERF SUMMARY\] measured_steps=(\d+), warmup_steps=(\d+)', log)
            if not perf or int(perf[2]) != 0 or 'promoting to' in log:
                raise RuntimeError('the fixed acoustic-step accounting contract changed')
            final = gate.final_state(output)
            if not final['fields']:
                raise RuntimeError('empty final field inventory')
            evidence['runs'][name] = dict(command=command, model_steps=int(perf[1]),
                allocations=int(matches[0][1]), requested_bytes=int(matches[0][2]), final=final,
                ledgers={label: gate.sha(output / label) for label in gate.LEDGERS},
                science=gate.science(output, False, True))
            if gate.sha(executable) != evidence['binary_sha256'][name] or gate.sha(config) != evidence['config_sha256']:
                raise RuntimeError('binary or configuration changed during validation')
        before, after = (evidence['runs'][name] for name in ('baseline', 'candidate'))
        for key in ('final', 'ledgers', 'model_steps'):
            if before[key] != after[key]:
                raise RuntimeError('buffer reuse changed ' + key)
        expected = 4 * (substeps - 1) * before['model_steps']
        saved = before['allocations'] - after['allocations']
        if saved != expected or before['requested_bytes'] - after['requested_bytes'] != saved * aligned_size:
            raise RuntimeError('field allocation reduction differs from the substep reuse contract')
        evidence.update(passed=True, field_allocations_saved=saved, allocation_bytes_avoided=saved * aligned_size)
        print(f'PASS: {saved} field allocations removed; exact fields, checkpoint and six ledgers')
    finally:
        (args.output / 'results.json').write_text(json.dumps(evidence, indent=2) + '\n')


if __name__ == '__main__':
    main()
