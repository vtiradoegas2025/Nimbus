#!/usr/bin/env python3
"""Compare automatic and portable SHA paths with independent hashlib oracles."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--auto',type=Path,required=True)
p.add_argument('--portable',type=Path,required=True)
p.add_argument('--baseline',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
rng=random.Random(150927)
cases=[]
for length in list(range(0,130))+[191,192,193,255,256,257,511,512,513,4095,4096,4097,65535,65536,65537,1048576]:
    payload=rng.randbytes(length)
    for chunk in (1,7,55,56,63,64,65,127,4096,1048576) if length<8192 else (63,64,4096,1048576):
        cases.append((payload,chunk,rng.randrange(0,64)))
source=''.join(f'{chunk} {offset} {payload.hex() or "-"}\n' for payload,chunk,offset in cases)
expected=[(hashlib.sha256(data).hexdigest(),hashlib.sha256(data+b'x').hexdigest(),hashlib.sha256(data+b'y').hexdigest()) for data,_,_ in cases]
report=dict(cases=len(cases),digests_per_case=3,passed=False,implementations={})
try:
    for label,path in [('auto',a.auto),('portable',a.portable)]+([('baseline',a.baseline)] if a.baseline else []):
        path=path.resolve(strict=True)
        run=subprocess.run([str(path)],input=source,text=True,capture_output=True,check=True,timeout=120)
        rows=[row.split() for row in run.stdout.splitlines()]
        if len(rows)!=len(expected):raise RuntimeError(label+': missing digest rows')
        for i,(row,oracle) in enumerate(zip(rows,expected)):
            if len(row)!=4 or tuple(row[:3])!=oracle:raise RuntimeError(f'{label}: digest/copy/continuation mismatch at case {i}')
        engines={r[3] for r in rows}
        if len(engines)!=1 or (label=='portable' and engines!={'portable'}):raise RuntimeError('implementation selection changed')
        report['implementations'][label]=dict(engine=engines.pop(),binary_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    report['passed']=True
finally:
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
