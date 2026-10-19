#!/usr/bin/env python3
"""Paired bounded checkpoint read/validation timing with exact roundtrip checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--inputs',type=Path,nargs='+',required=True)
    p.add_argument('--threads',type=int,required=True)
    p.add_argument('--pairs',type=int,default=3)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.threads<1 or not 3<=a.pairs<=9:p.error('positive workers and 3–9 pairs required')
    binaries={k:v.resolve(strict=True) for k,v in [('baseline',a.baseline),('candidate',a.candidate)]}
    identities={k:sha(v) for k,v in binaries.items()}
    if identities['baseline']==identities['candidate']:p.error('distinct binaries required')
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    env={k:v for k,v in os.environ.items() if not k.startswith(('NIMBUS_','OMP_','KMP_','GOMP_'))}
    env['OMP_NUM_THREADS']=str(a.threads)
    report=dict(schema='nimbus.checkpoint-read-benchmark.v1',binary_sha256=identities,pairs=a.pairs,threads=a.threads,
                driver_sha256=sha(Path(__file__)),runs=[],passed=False,simulation_speedup_claim=False,
                cache_scope='repeated reads; OS file cache not flushed')
    try:
        for number,source in enumerate(a.inputs):
            source=source.resolve(strict=True)
            if source.stat().st_size>4<<30:raise RuntimeError('checkpoint exceeds bounded 4 GiB limit')
            source_sha=sha(source)
            sidecar=json.loads(source.with_suffix(source.suffix+'.json').read_text())
            if sidecar['checkpoint_sha256']!=source_sha:raise RuntimeError('source sidecar does not match file')
            for pair in range(a.pairs):
                for label in (('baseline','candidate') if pair%2==0 else ('candidate','baseline')):
                    assert sha(binaries[label])==identities[label]
                    command=[str(binaries[label]),str(source)]
                    roundtrip=a.output/f'{number}-{label}-roundtrip.nimbus'
                    if pair==0:command.append(str(roundtrip))
                    run=subprocess.run(command,env=env,capture_output=True,text=True,timeout=300)
                    log=a.output/f'{number}-{pair+1}-{label}.log';log.write_text(run.stdout+run.stderr)
                    if run.returncode:raise RuntimeError('read/roundtrip failed: '+str(log))
                    rows=re.findall(r'\[CHECKPOINT READ BENCHMARK\] (.*)',run.stdout)
                    if len(rows)!=1:raise RuntimeError('missing reader measurements')
                    row=json.loads(rows[0])
                    if row['sha256']!=source_sha or row['bytes']!=sidecar['checkpoint_bytes'] or row['fnv1a64']!=sidecar['checkpoint_fnv1a64']:
                        raise RuntimeError('checkpoint integrity changed')
                    if pair==0 and sha(roundtrip)!=source_sha:raise RuntimeError('restored scientific state changed')
                    report['runs'].append(dict(build=label,input_index=number,input_path=str(source),pair=pair+1,
                                              roundtrip_verified=pair==0,command=command,**row))
                    print(number,label,pair+1,round(row['total_s'],4),'s; exact integrity',flush=True)
                    (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
            if sha(source)!=source_sha:raise RuntimeError('input checkpoint changed during profiling')
        report['comparisons']={}
        for number in range(len(a.inputs)):
            report['comparisons'][str(number)]={}
            for key in ('io_s','fnv_s','sha_s','validate_decode_s','total_s'):
                pairs=[{r['build']:r[key] for r in report['runs'] if r['input_index']==number and r['pair']==n+1} for n in range(a.pairs)]
                delta=statistics.median(v['baseline']-v['candidate'] for v in pairs)
                noise=3*1.4826*statistics.median(abs(v['baseline']-v['candidate']-delta) for v in pairs)
                percent=100*delta/statistics.median(v['baseline'] for v in pairs)
                report['comparisons'][str(number)][key]=dict(median_paired_delta_s=delta,three_mad_s=noise,
                    paired_improvement_percent=percent,speedup_and_noise_passed=percent>=5 and delta>noise)
        report['passed']=True
    finally:
        (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
