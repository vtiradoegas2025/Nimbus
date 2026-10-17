#!/usr/bin/env python3
"""Inventory retained paths and collect bounded, correctness-checked architecture profiles."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
LEDGERS = ('budget.csv', 'energy_budget.csv', 'moisture_budget.csv',
           'momentum_budget.csv', 'radiation_budget.csv', 'thermodynamic_source_budget.csv')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def numeric_metrics(text):
    values = {}
    for key, value in re.findall(r'\b([a-z][a-z_0-9]*)=([^,\s]+)', text):
        try:
            number = float(value)
        except ValueError:
            continue
        if not math.isfinite(number):
            raise ValueError('nonfinite telemetry: ' + key)
        values[key] = int(number) if number.is_integer() else number
    return values


def telemetry(text, backend):
    perf = re.findall(r'\[PERF SUMMARY\] (.*)', text)
    dynamics = re.findall(r'\[DYNAMICS PERF\] (.*)', text)
    if len(perf) != 1 or len(dynamics) != 1:
        raise ValueError('one complete model and dynamics timing summary required')
    stages = {}
    for label, values in re.findall(r'\[GPU STAGE TELEMETRY\] (\S+) (.*)', text):
        stages[label] = numeric_metrics(values)
    for label, gpu, fence, count in re.findall(
            r'\[GPU SUBMIT TIMING\] (\S+) ([\d.eE+-]+) ([\d.eE+-]+) (\d+) ', text):
        stages.setdefault(label, {}).update(gpu_total_ms=float(gpu),
                                            fence_wait_ms=float(fence), submissions=int(count))
    metal = [numeric_metrics(line) for line in re.findall(r'\[METAL TELEMETRY\] (.*)', text)]
    timing_rows = metal if backend == 'metal' else list(stages.values())
    complete = bool(timing_rows) and all(
        row.get('command_buffers', 0) > 0 and
        row.get('timestamped_command_buffers') == row['command_buffers'] for row in timing_rows)
    if any(row.get('gpu_total_ms', 0) < 0 for row in timing_rows):
        raise ValueError('negative device duration')
    totals = re.findall(r'\[GPU TELEMETRY\] (.*)', text)
    dispatch = re.findall(r'\[COMPUTE DISPATCH\] (.*)', text)
    return dict(model=numeric_metrics(perf[0]), dynamics=numeric_metrics(dynamics[0]),
                gpu_timing_scope='complete backend generation including initialization and warmup',
                gpu_timing_complete=complete, gpu_stages=stages, metal_generations=metal,
                gpu=numeric_metrics(totals[-1]) if totals else None,
                dispatch=numeric_metrics(dispatch[-1]) if dispatch else None,
                dispatched_kernels={name: int(count) for name, count in
                    re.findall(r'([^,{\s]+\.comp):(\d+)', dispatch[-1])} if dispatch else {},
                peak_rss_bytes=max(map(int, re.findall(r'peak_rss_bytes=(\d+)', text)), default=None))


def inventory(binary):
    catalog = json.loads(subprocess.check_output([str(binary), '--scheme-catalog=json'],
                                                 cwd=ROOT, text=True, timeout=30))
    source = (ROOT / 'src/compute/kernel_manifest.cpp').read_text()
    pattern = r'\{KernelId::(\w+),\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)"'
    matches = list(re.finditer(pattern, source))
    shaders = {entry['source']: entry for entry in
               json.loads((ROOT / 'vulkan/shaders/manifest.json').read_text())['shaders']
               if '/compute/' in entry['source']}
    kernels = []
    for i, match in enumerate(matches):
        identifier, operation, path, artifact, digest = match.groups()
        block = source[match.end():matches[i + 1].start() if i + 1 < len(matches) else source.index('\n}};', match.end())]
        canonical = 'vulkan/shaders/compute/' + path
        if canonical not in shaders or shaders[canonical]['spirv_fnv1a64'] != digest:
            raise ValueError('kernel/artifact inventory mismatch: ' + path)
        references = re.findall(r'"([^"]+)"', block)
        availability = re.search(r'A::(Orphan|Deferred)', block)
        kernels.append(dict(id=identifier, operation=operation, source=canonical,
                            artifact='vulkan/shaders/compute/' + artifact,
                            artifact_sha256=shaders[canonical]['spirv_sha256'],
                            cpu_reference=references[0] if references else None,
                            availability=availability[1].lower() if availability else 'active',
                            arithmetic='fp64_fma_fp32_storage' if 'Float32StorageFloat64Arithmetic' in block else 'fp32',
                            optimization_status='requires_review'))
    if len(kernels) != len(shaders) or len({row['source'] for row in kernels}) != len(shaders):
        raise ValueError('incomplete or duplicate compute inventory')
    return dict(catalog_version=catalog['catalogVersion'], modules=catalog['modules'],
                kernels=kernels, readiness=catalog['readiness'],
                engineering_blockers=[row for row in catalog['readiness']['items']
                                      if row['state'] == 'engineering_incomplete'],
                other_owners=['boundary conditions', 'surface fluxes', 'thermodynamic closure',
                              'nested feedback and domain motion', 'diagnostics and scientific reports',
                              'checkpoint and output', 'remote transfer', 'Studio rendering and playback'],
                universal_optimization_verified=False, long_three_sounding_runs_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--backend', choices=('cpu', 'metal', 'vulkan'), default='cpu')
    parser.add_argument('--threads', type=int, nargs='+', default=[1])
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--seconds', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--omp-proc-bind', choices=('false', 'close', 'spread'))
    parser.add_argument('--omp-places', choices=('threads', 'cores', 'sockets'))
    parser.add_argument('--vulkan-memory-policy', choices=('auto', 'direct', 'staged'))
    args = parser.parse_args()
    if not 1 <= args.seconds <= 10 or not 1 <= args.trials <= 5 or args.warmup < 1:
        parser.error('bounded audit requires 1–10 seconds, 1–5 trials, and positive warmup')
    if any(n < 1 or n > (os.cpu_count() or 1) for n in args.threads) or len(set(args.threads)) != len(args.threads):
        parser.error('thread counts must be distinct and within this host logical-CPU count')
    if args.vulkan_memory_policy is not None and args.backend != 'vulkan':
        parser.error('--vulkan-memory-policy requires --backend=vulkan')
    binary = args.binary.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binaries = {'candidate': binary}
    if args.baseline:
        binaries['baseline'] = args.baseline.resolve(strict=True)
    binary_hashes = {name: sha(path) for name, path in binaries.items()}
    if args.baseline and (binary_hashes['baseline'] == binary_hashes['candidate'] or args.trials < 3):
        parser.error('paired comparisons require distinct binaries and at least three trials')
    result = dict(schema='nimbus.architecture-optimization-audit.v2', host=platform.platform(),
                  binary_hashes=binary_hashes, driver_sha256=sha(Path(__file__)),
                  binary_sha256=sha(binary), inventory=inventory(binary), runs=[],
                  profiling_completed=False, optimization_complete=False)
    try:
        if not args.config:
            print('Inventory:', len(result['inventory']['modules']), 'modules,',
                  len(result['inventory']['kernels']), 'kernels,',
                  len(result['inventory']['engineering_blockers']), 'engineering blockers')
            return
        original = args.config.resolve(strict=True)
        config_text = original.read_text()
        config_text, count = re.subn(r'(?m)^    backend: (cpu|metal|vulkan)$',
                                    '    backend: ' + args.backend, config_text)
        if count != 1 or '    deterministic_reductions: true' not in config_text:
            raise ValueError('audit requires one backend selection and deterministic scientific reductions')
        if '    warmup_steps:' in config_text:
            config_text = re.sub(r'(?m)^    warmup_steps: \d+$', f'    warmup_steps: {args.warmup}', config_text)
        else:
            config_text = config_text.replace('    timing: true\n', f'    timing: true\n    warmup_steps: {args.warmup}\n')
        config = output / 'config.yaml'
        config.write_text(config_text)
        result.update(original_config=str(original), original_config_sha256=sha(original),
                      config_sha256=sha(config), backend=args.backend, seconds=args.seconds)
        environment = {k: v for k, v in os.environ.items()
                       if not k.startswith(('NIMBUS_', 'OMP_', 'KMP_', 'GOMP_'))}
        environment['NIMBUS_GPU_KERNEL_TIMING'] = '1'
        overrides = {name: value for name, value in (
            ('OMP_PROC_BIND', args.omp_proc_bind),
            ('OMP_PLACES', args.omp_places),
            ('NIMBUS_VULKAN_MEMORY_POLICY', args.vulkan_memory_policy)) if value is not None}
        environment.update(overrides)
        result['environment_overrides'] = overrides
        reference = None
        for trial in range(args.trials):
            for threads in (args.threads if trial % 2 == 0 else list(reversed(args.threads))):
                builds = (['baseline', 'candidate'] if trial % 2 == 0 else ['candidate', 'baseline']) if args.baseline else ['candidate']
                for build in builds:
                    if sha(binaries[build]) != binary_hashes[build] or sha(config) != result['config_sha256']:
                        raise RuntimeError('binary or configuration changed during profiling')
                    directory = output / f'trial-{trial + 1}-threads-{threads}-{build}'
                    command = [str(binaries[build]), '--headless', '--config=' + str(config),
                               '--threads=' + str(threads), '--duration=' + str(args.seconds),
                               '--write-every=' + str(args.seconds), '--checkpoint-every=' + str(args.seconds),
                               '--outdir=' + str(directory)]
                    log = output / (directory.name + '.log')
                    started = time.monotonic()
                    with log.open('w') as stream:
                        run = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                                             stderr=subprocess.STDOUT, timeout=600)
                    if run.returncode:
                        raise RuntimeError('bounded run failed: ' + str(log))
                    text = log.read_text()
                    validation = json.loads((directory / 'validation_summary.json').read_text())
                    if (validation['failed'] or validation['total_violations'] or not validation['report_count'] or
                            'Execution mode: qualified' not in text or 'fallback=on' in text or
                            not re.search(rf'\[COMPUTE\] backend={args.backend},', text) or
                            not re.search(rf'OpenMP threads:\s*{threads}\b', text)):
                        raise RuntimeError('science, backend, or worker contract failed: ' + str(log))
                    frame = sorted(directory.glob('step_*'))[-1]
                    checkpoint = sorted((directory / 'checkpoints').glob('*.nimbus'))[-1]
                    identity = dict(checkpoint=sha(checkpoint), fields={p.name: sha(p) for p in frame.glob('*.npy')},
                                    ledgers={name: sha(directory / name) for name in LEDGERS})
                    if not identity['fields']:
                        raise RuntimeError('empty final frame')
                    if reference is not None and identity != reference:
                        raise RuntimeError('thread or repeat changed scientific state/ledgers')
                    reference = identity
                    measurements = telemetry(text, args.backend)
                    openmp = re.findall(r'\[HARDWARE\]\s+OpenMP: (.*)', text)
                    memory_plans = re.findall(r'\[COMPUTE\]\[VULKAN\] memory plan=(.*)', text)
                    if args.omp_proc_bind is not None and not any(
                            f'proc_bind={args.omp_proc_bind}' in line for line in openmp):
                        raise RuntimeError('requested OpenMP binding was not observed: ' + str(log))
                    if args.vulkan_memory_policy == 'staged' and not any(
                            line.startswith('staged_forced') for line in memory_plans):
                        raise RuntimeError('requested staged Vulkan policy was not observed: ' + str(log))
                    if args.vulkan_memory_policy == 'direct' and not any(
                            line.startswith('direct_shared') for line in memory_plans):
                        raise RuntimeError('requested direct Vulkan policy was not observed: ' + str(log))
                    if measurements['model'].get('measured_steps', 0) < 10:
                        raise RuntimeError('insufficient post-warmup model steps')
                    result['runs'].append(dict(build=build, trial=trial + 1, threads=threads, command=command,
                        wall_seconds=time.monotonic() - started, validation=validation,
                        final_state=identity, measurements=measurements,
                        openmp_configuration=openmp, vulkan_memory_plans=memory_plans,
                        saved_bytes=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())))
                    print(args.backend, build, threads, trial + 1, measurements['model']['avg_step_ms'],
                          'ms/step; exact state and ledgers', flush=True)
                    (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
        result['summary'] = {}
        for build in binaries:
            for threads in args.threads:
                times = [row['measurements']['model']['avg_step_ms'] for row in result['runs']
                         if row['threads'] == threads and row['build'] == build]
                result['summary'][f'{build}:{threads}'] = dict(median_step_ms=statistics.median(times),
                    minimum_step_ms=min(times), maximum_step_ms=max(times))
        if args.baseline:
            result['paired_comparison'] = {}
            for threads in args.threads:
                pairs = [{row['build']: row['measurements']['model']['avg_step_ms']
                          for row in result['runs'] if row['threads'] == threads and row['trial'] == trial + 1}
                         for trial in range(args.trials)]
                improvements = [pair['baseline'] - pair['candidate'] for pair in pairs]
                delta = statistics.median(improvements)
                noise = 3 * 1.4826 * statistics.median(abs(value - delta) for value in improvements)
                percent = 100 * delta / statistics.median(pair['baseline'] for pair in pairs)
                result['paired_comparison'][str(threads)] = dict(paired_improvement_percent=percent,
                    three_mad_ms=noise, median_delta_ms=delta, speedup_and_noise_passed=delta > noise and percent >= 5)
        result['profiling_completed'] = True
    except Exception as error:
        result['error'] = str(error)
        raise
    finally:
        (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
