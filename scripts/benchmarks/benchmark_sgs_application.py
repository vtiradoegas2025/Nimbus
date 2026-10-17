#!/usr/bin/env python3
"""Compare SGS application costs while checking exact field and ledger fingerprints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--threads', type=int, nargs='+', required=True)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--nx', type=int, default=128)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 32 <= args.nx <= 640 or not 3 <= args.trials <= 5 or any(not 1 <= n <= 64 for n in args.threads):
        parser.error('Use 32–640 horizontal cells, 3–5 pairs, and 1–64 workers')
    args.output.mkdir(parents=True, exist_ok=False)
    binaries = {name: getattr(args, name).resolve(strict=True) for name in ['baseline', 'candidate']}
    hashes = {name: sha(path) for name, path in binaries.items()}
    result = dict(schema='nimbus.sgs-application-benchmark.v1', host=platform.platform(),
                  binary_sha256=hashes, rows=[], comparisons={}, completed=False)
    reference = None
    try:
        for trial in range(args.trials):
            for threads in args.threads:
                order = ['baseline', 'candidate'] if trial % 2 == 0 else ['candidate', 'baseline']
                for name in order:
                    assert sha(binaries[name]) == hashes[name]
                    env = {k: v for k, v in os.environ.items() if not k.startswith(('NIMBUS_', 'OMP_', 'KMP_', 'GOMP_'))}
                    env['NIMBUS_SGS_BENCHMARK_THREADS'] = str(threads)
                    env['NIMBUS_SGS_BENCHMARK_NX'] = str(args.nx)
                    env['OMP_DYNAMIC'] = 'FALSE'
                    log = args.output / f'{trial+1}-{threads}-{name}.log'
                    with log.open('w') as stream:
                        subprocess.run([str(binaries[name]), '[.sgs-application-benchmark]'], env=env,
                                       stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=900)
                    match = re.search(r'^NIMBUS_SGS_BENCHMARK (.+)$', log.read_text(), re.MULTILINE)
                    if not match:
                        raise RuntimeError('Missing benchmark receipt: ' + str(log))
                    measured = json.loads(match[1])
                    identity = (measured['fingerprint'], measured['shape'])
                    if reference is not None and identity != reference:
                        raise RuntimeError('State or ledger fingerprint changed')
                    reference = identity
                    assert measured['threads'] == threads
                    assert all(len(measured[key]) == 6 for key in ['apply_ms', 'stability_ms', 'commit_ms'])
                    total = [sum(parts) for parts in zip(measured['apply_ms'], measured['stability_ms'], measured['commit_ms'])]
                    result['rows'].append(dict(build=name, trial=trial+1, **measured,
                                              apply_median_ms=statistics.median(measured['apply_ms']),
                                              operator_median_ms=statistics.median(total), log_sha256=sha(log)))
                    print(name, threads, trial+1, round(result['rows'][-1]['apply_median_ms'], 3), 'ms', flush=True)
        for threads in args.threads:
            result['comparisons'][str(threads)] = {}
            for metric in ['apply_median_ms', 'operator_median_ms']:
                pairs = [{row['build']: row[metric] for row in result['rows']
                          if row['threads'] == threads and row['trial'] == trial+1} for trial in range(args.trials)]
                deltas = [pair['baseline'] - pair['candidate'] for pair in pairs]
                delta = statistics.median(deltas)
                noise = 3 * 1.4826 * statistics.median(abs(value-delta) for value in deltas)
                baseline = statistics.median(pair['baseline'] for pair in pairs)
                percent = 100 * delta / baseline
                result['comparisons'][str(threads)][metric] = dict(baseline_median_ms=baseline,
                    candidate_median_ms=statistics.median(pair['candidate'] for pair in pairs),
                    paired_delta_ms=delta, paired_improvement_percent=percent, three_mad_ms=noise,
                    speedup_and_noise_passed=percent >= 5 and delta > noise)
        result['completed'] = True
    finally:
        (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
