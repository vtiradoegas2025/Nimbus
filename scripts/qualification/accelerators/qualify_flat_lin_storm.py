#!/usr/bin/env python3
"""Bounded public-admission, restart, threading, and closure evidence for flat Lin storms."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
CASES = ('01_classic_supercell', '02_pulse_decay', '03_moist_supercell')
LEDGERS = ('budget.csv', 'energy_budget.csv', 'moisture_budget.csv', 'momentum_budget.csv',
           'radiation_budget.csv', 'thermodynamic_source_budget.csv')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def ledger_lines(path):
    return [line for line in path.read_text().splitlines() if line and not line.startswith('#')]


def science(output, require_activity, require_complete):
    validation = json.loads((output / 'validation_summary.json').read_text())
    if validation['failed'] or validation['total_violations']:
        raise RuntimeError('runtime scientific validation failed')
    rows = list(csv.DictReader(ledger_lines(output / 'thermodynamic_source_budget.csv')))
    if not rows: raise RuntimeError('missing scientific source ledger')
    maximum_water_error = maximum_energy_error = activity = 0.0
    complete_rows = 0
    for row in rows:
        activity = max(activity, *(abs(float(row[f'microphysics_{name}_requested_kg'])) for name in ('qv', 'qc', 'qi', 'qr')))
        if int(row['microphysics_conservation_active']) != 1:
            raise RuntimeError('Lin conservation admission was not active')
        water = abs(float(row['microphysics_conservation_residual_kg'])) / max(abs(float(row['microphysics_baseline_water_kg'])), 1e-30)
        if not math.isfinite(water) or water > 1e-6:
            raise RuntimeError('Lin water closure exceeded the maintained reference bound')
        maximum_water_error = max(maximum_water_error, water)
        complete = int(row['completed_step_energy_step_complete']) == 1
        if require_complete and float(row['time_s']) > 0 and not complete:
            raise RuntimeError('incomplete completed-step energy ownership')
        if complete:
            complete_rows += 1
            error = abs(float(row['completed_step_energy_step_closure_residual_j'])) / max(1.0, abs(float(row['completed_step_energy_step_committed_owner_sum_j'])))
            if not math.isfinite(error) or error > 1e-12:
                raise RuntimeError('completed-step energy closure exceeded the maintained reference bound')
            maximum_energy_error = max(maximum_energy_error, error)
    if require_activity and activity <= 0:
        raise RuntimeError('seeded case did not exercise Lin cloud processes')
    if require_complete and not complete_rows: raise RuntimeError('no complete energy records')
    return dict(validation=validation, maximum_relative_water_closure=maximum_water_error,
                maximum_relative_energy_closure=maximum_energy_error, complete_energy_rows=complete_rows,
                cloud_process_activity_kg=activity)


def final_state(output):
    frame = sorted(output.glob('step_*'))[-1]
    checkpoint = sorted((output / 'checkpoints').glob('*.nimbus'))[-1]
    return dict(checkpoint=sha(checkpoint), fields={p.name: sha(p) for p in frame.glob('*.npy')})


def checkpoint_at_time(output, target):
    matches = []
    for checkpoint in (output / 'checkpoints').glob('*.nimbus'):
        value = json.loads(Path(str(checkpoint) + '.json').read_text()).get('simulation_time_s')
        # Summed decimal timesteps can place a scheduled checkpoint a few ULPs away.
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and abs(value - target) <= 4 * math.ulp(target):
            matches.append(checkpoint)
    if len(matches) != 1:
        raise RuntimeError(f'missing unique continuous midpoint checkpoint at t={target:g}')
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu', 'metal', 'vulkan'), required=True)
    parser.add_argument('--threads', type=int, required=True)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--cadence', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', default=CASES)
    parser.add_argument('--skip-seeded', action='store_true')
    parser.add_argument('--config-dir', type=Path,
                        default=ROOT / 'tests/fixtures/configs/lin_storm')
    args = parser.parse_args()
    if args.cuda and args.backend != 'vulkan': parser.error('CUDA assist requires Vulkan')
    if args.threads < 1 or args.cadence < 1: parser.error('threads and cadence must be positive')
    args.binary = args.binary.resolve(strict=True)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    environment = {k: v for k, v in os.environ.items() if not k.startswith(('NIMBUS_', 'OMP_', 'KMP_', 'GOMP_'))}
    evidence = dict(binary_sha256=sha(args.binary), backend=args.backend, cuda_assist=args.cuda,
                    workers=args.threads, cadence=args.cadence, runs=[], passed=False)

    def execute(config, label, threads, *, resume=None, activity=False):
        output = args.output / label
        command = [str(args.binary), '--headless', '--config=' + str(config), '--threads=' + str(threads),
                   '--outdir=' + str(output)]
        if resume: command += ['--resume=' + str(resume)]
        log = args.output / (label + '.log')
        with log.open('w') as stream:
            completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
        if completed.returncode: raise RuntimeError(f'{label} failed: {log}')
        text = log.read_text()
        if 'Execution mode: qualified' not in text or 'fallback=on' in text:
            raise RuntimeError(f'{label} did not use ordinary qualified admission')
        if args.cuda and 'CUDA assist active:' not in text: raise RuntimeError('CUDA selection was not honored')
        if args.backend != 'cpu' and f'backend={args.backend}' not in text: raise RuntimeError('wrong compute backend')
        if args.backend == 'vulkan' and args.cadence > 1 and not resume:
            match = re.search(r'acoustic/acoustic_retained_flux_fp64\.comp:(\d+)', text)
            if not match or int(match.group(1)) <= 0: raise RuntimeError('resident acoustic loop did not execute')
        row = dict(label=label, command=command, config_sha256=sha(config),
                   science=science(output, activity, args.cadence == 1), final=final_state(output))
        evidence['runs'].append(row)
        (args.output / 'results.json').write_text(json.dumps(evidence, indent=2) + '\n')
        print(label, 'PASS', flush=True)
        return output, row['final']

    try:
        for case in args.cases:
            source = args.config_dir.resolve(strict=True) / (case + '.yaml')
            text = source.read_text().replace('mode: exploratory', 'mode: qualified')
            text = text.replace('backend: vulkan', 'backend: ' + args.backend)
            text = text.replace('  run_time: 2\n', '  run_time: 3\n')
            text = text.replace('    scientific_every_steps: 1\n', f'    scientific_every_steps: {args.cadence}\n')
            if args.cuda: text = text.replace('    backend: vulkan\n', '    backend: vulkan\n    cuda_assist: true\n')
            config = args.output / (case + '.yaml')
            config.write_text(text)
            continuous, reference = execute(config, case + '-continuous', args.threads)
            checkpoint = checkpoint_at_time(continuous, 2.0)
            resumed, actual = execute(config, case + '-resumed', args.threads, resume=checkpoint)
            if actual != reference: raise RuntimeError(case + ': restart changed final state')
            for name in LEDGERS:
                before, after = ledger_lines(continuous / name), ledger_lines(resumed / name)
                if before[0] != after[0] or before[-(len(after)-1):] != after[1:]:
                    raise RuntimeError(case + ': restart ledger suffix differs: ' + name)
            serial, actual = execute(config, case + '-one-worker', 1)
            if actual != reference: raise RuntimeError(case + ': threading changed final state')
            for name in LEDGERS:
                if sha(continuous / name) != sha(serial / name): raise RuntimeError(case + ': threaded ledger changed: ' + name)
            if not args.skip_seeded:
                for label, height in [('warm', 1500), ('cold', 8000)]:
                    seeded = args.output / (case + '-' + label + '.yaml')
                    seeded.write_text(text.replace('cloud_water_kgkg: 0\n', 'cloud_water_kgkg: 0.0015\n')
                                      .replace('center_z_m: 1500\n', f'center_z_m: {height}\n'))
                    execute(seeded, case + '-' + label, args.threads, activity=True)
        evidence['passed'] = True
    finally:
        (args.output / 'results.json').write_text(json.dumps(evidence, indent=2) + '\n')


if __name__ == '__main__': main()
