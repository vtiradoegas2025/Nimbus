#!/usr/bin/env python3
"""Verify every saved field against a manufactured renderer fixture receipt."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    receipt_bytes = (root / 'fixture_receipt.json').read_bytes()
    receipt = json.loads(receipt_bytes)
    spec = receipt['specification']
    if json.loads((root / 'visual_load_fixture.json').read_text()) != spec:
        raise ValueError('Fixture specification differs from its receipt')
    saved = spec.get('schema') == 'nimbus.saved-visual-reference.v1' and spec.get('values_modified') is False
    if not (spec.get('not_a_simulation') is True or saved):
        raise ValueError('Not a labelled visual fixture')
    total = 0
    files = 0
    for frame in receipt['frames']:
        if 'manifest_sha256' in frame:
            manifest = root / f'step_{frame["step"]:06}' / 'manifest.json'
            if hashlib.sha256(manifest.read_bytes()).hexdigest() != frame['manifest_sha256']:
                raise ValueError(f'Saved manifest changed: {manifest}')
        for field, expected in frame['fields'].items():
            path = root / f'step_{frame["step"]:06}' / f'{field}.npy'
            sha = hashlib.sha256()
            with path.open('rb') as stream:
                while block := stream.read(4 * 1024 * 1024):
                    sha.update(block)
            if path.stat().st_size != expected['bytes'] or sha.hexdigest() != expected['sha256']:
                raise ValueError(f'Field payload differs: {path}')
            total += path.stat().st_size
            files += 1
    if total != receipt['total_field_bytes']:
        raise ValueError('Total payload differs')
    return dict(status='passed', directory=str(root), fields=files, bytes=total,
                receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory.resolve()), indent=2))
