#!/usr/bin/env python3
"""Bounded native output/checkpoint scaling with exact installed-baseline science."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
LEDGERS=('budget.csv','energy_budget.csv','moisture_budget.csv','momentum_budget.csv','radiation_budget.csv','thermodynamic_source_budget.csv')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--backend',choices=('metal','vulkan','cpu'),required=True)
    p.add_argument('--threads',type=int,required=True)
    p.add_argument('--sizes',type=int,nargs='+',default=[128,256])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.threads<1 or any(n<48 or n>256 for n in a.sizes): p.error('bounded grid range is 48–256 with positive worker count')
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    binaries={name:x.resolve(strict=True) for name,x in [('baseline',a.baseline),('candidate',a.candidate)]}
    identities={name:sha(x) for name,x in binaries.items()}
    report=dict(schema='nimbus.checkpoint-output-scaling.v1',binary_sha256=identities,backend=a.backend,threads=a.threads,
                simulated_seconds=1,runs=[],passed=False,throughput_speedup_claim=False)
    env={k:v for k,v in os.environ.items() if not k.startswith(('NIMBUS_','OMP_','KMP_','GOMP_'))}
    try:
        for size in a.sizes:
            config=a.config.resolve(strict=True).read_text()
            for key in ('nx','ny'): config,n=re.subn(rf'(?m)^  {key}: \d+$',f'  {key}: {size}',config);assert n==1
            config,n=re.subn(r'(?m)^    backend: (cpu|metal|vulkan)$','    backend: '+a.backend,config);assert n==1
            if '    deterministic_reductions: true' not in config: raise RuntimeError('deterministic reductions required')
            nz=int(re.search(r'(?m)^  nz: (\d+)$',config)[1])
            if nz<2 or size*size*nz>256*256*180: raise RuntimeError('grid exceeds bounded cell budget')
            config_path=a.output/f'{size}.yaml';config_path.write_text(config)
            reference=None
            for name in ('baseline','candidate'):
                assert sha(binaries[name])==identities[name]
                output=a.output/f'{size}-{name}';log=a.output/f'{size}-{name}.log'
                command=[str(binaries[name]),'--headless','--config='+str(config_path),'--threads='+str(a.threads),
                         '--duration=1','--write-every=1','--checkpoint-every=1','--outdir='+str(output)]
                start=time.monotonic()
                with log.open('w') as f: subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600,check=True)
                elapsed=time.monotonic()-start; text=log.read_text()
                validation=json.loads((output/'validation_summary.json').read_text())
                if validation['failed'] or validation['total_violations'] or not validation['report_count'] or \
                    'Execution mode: qualified' not in text or 'fallback=on' in text or f'backend={a.backend},' not in text:
                    raise RuntimeError('native/scientific admission failed')
                frames=sorted(output.glob('step_*'));checkpoint=sorted((output/'checkpoints').glob('*.nimbus'))[-1]
                identity=dict(fields={str(p.relative_to(output)):sha(p) for d in frames for p in d.glob('*.npy')},
                              checkpoint=sha(checkpoint),ledgers={n:sha(output/n) for n in LEDGERS})
                if not identity['fields']: raise RuntimeError('missing output fields')
                if reference is not None and identity!=reference: raise RuntimeError('saved scientific bytes differ')
                reference=identity
                checkpoint_rows=[dict(completed_step=int(step),bytes=int(b),artifact_publication_s=float(s),mib_per_s=float(m)) for step,b,s,m in re.findall(
                    r'\[CHECKPOINT PERF\] completed_step=(\d+), bytes=(\d+), artifact_publication_s=([\d.eE+-]+), mib_per_s=([\d.eE+-]+)',text)]
                if name=='candidate' and (len(checkpoint_rows)!=1 or checkpoint_rows[0]['bytes']!=checkpoint.stat().st_size or checkpoint_rows[0]['artifact_publication_s']<=0):
                    raise RuntimeError('missing/inconsistent checkpoint publication timing')
                output_time=re.findall(r'output_preparation_submission_s=([\d.eE+-]+)',text)
                if len(output_time)!=1: raise RuntimeError('missing periodic output timing')
                periodic_bytes=sum(x.stat().st_size for x in frames[-1].rglob('*') if x.is_file())
                output_seconds=float(output_time[0])
                report['runs'].append(dict(build=name,grid=[size,size,nz],command=command,config_sha256=sha(config_path),
                    elapsed_seconds=elapsed,validation=validation,identity=identity,checkpoint_bytes=checkpoint.stat().st_size,
                    checkpoint_publication=checkpoint_rows,periodic_output_bytes=periodic_bytes,periodic_output_seconds=output_seconds,
                    periodic_output_mib_per_s=periodic_bytes/(1024*1024*output_seconds),
                    peak_rss_bytes=max(map(int,re.findall(r'peak_rss_bytes=(\d+)',text))),
                    saved_bytes=sum(x.stat().st_size for x in output.rglob('*') if x.is_file())))
                print(size,name,'exact science; checkpoint',checkpoint.stat().st_size,'bytes',flush=True)
                (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        report['passed']=True
    finally:
        (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
