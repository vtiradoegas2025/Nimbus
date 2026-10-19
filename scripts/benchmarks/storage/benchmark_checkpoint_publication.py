#!/usr/bin/env python3
"""Bounded, alternating checkpoint publication timing with independent byte identity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess


def sha(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): result.update(block)
    return result.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--elements',type=int,nargs='+',default=[128*128*180,256*256*180])
    parser.add_argument('--fields',type=int,default=12)
    parser.add_argument('--pairs',type=int,default=3)
    a=parser.parse_args()
    if not 3<=a.pairs<=9 or not 1<=a.fields<=256 or any(n<1 or n*a.fields>1<<29 for n in a.elements):
        parser.error('requires 3–9 pairs and 1–256 fields, at most 2 GiB per checkpoint')
    binaries={name:p.resolve(strict=True) for name,p in [('baseline',a.baseline),('candidate',a.candidate)]}
    hashes={name:sha(p) for name,p in binaries.items()}
    if hashes['baseline']==hashes['candidate']: parser.error('distinct binaries required')
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='nimbus.checkpoint-publication-benchmark.v1',host=platform.platform(),binary_sha256=hashes,
                driver_sha256=sha(Path(__file__)),pairs=a.pairs,fields=a.fields,runs=[],passed=False,
                timing_scope='three-pass encoding, checksums, writing, durable sync and atomic publication',
                synthetic_payload=True,simulation_speedup_claim=False)
    environment={k:v for k,v in os.environ.items() if not k.startswith(('NIMBUS_','OMP_','KMP_','GOMP_'))}
    try:
        for elements in a.elements:
            identity=None
            for pair in range(a.pairs):
                for name in (('baseline','candidate') if pair%2==0 else ('candidate','baseline')):
                    assert sha(binaries[name])==hashes[name]
                    artifact=a.output/f'{elements}-{name}.nimbus'
                    command=[str(binaries[name]),str(artifact),str(elements),str(a.fields)]
                    result=subprocess.run(command,env=environment,text=True,capture_output=True,timeout=300,check=True)
                    measured=json.loads(result.stdout)
                    actual=sha(artifact)
                    if measured['bytes']!=artifact.stat().st_size or measured['sha256']!=actual:
                        raise RuntimeError('published integrity differs from independently hashed file')
                    if identity is not None and identity!=actual: raise RuntimeError('checkpoint bytes changed')
                    identity=actual
                    report['runs'].append(dict(build=name,elements=elements,pair=pair+1,command=command,**measured))
                    print(elements,name,pair+1,round(measured['seconds'],4),'s; exact bytes',flush=True)
                    (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        report['comparisons']={}
        for elements in a.elements:
            pairs=[{r['build']:r['seconds'] for r in report['runs'] if r['elements']==elements and r['pair']==p+1} for p in range(a.pairs)]
            delta=statistics.median(p['baseline']-p['candidate'] for p in pairs)
            noise=3*1.4826*statistics.median(abs(p['baseline']-p['candidate']-delta) for p in pairs)
            percent=100*delta/statistics.median(p['baseline'] for p in pairs)
            report['comparisons'][str(elements)]=dict(median_paired_delta_s=delta,three_mad_s=noise,
                paired_improvement_percent=percent,speedup_and_noise_passed=delta>noise and percent>=5)
        report['passed']=True
    finally:
        (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
