#!/usr/bin/env python3
"""Build the maintained LaTeX guides; keep intermediate files out of docs/."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
GUIDES = {'mathematics': 'MATHEMATICS.pdf', 'architecture': 'ARCHITECTURE.pdf'}


def identity(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_inputs(name: str) -> dict[str, str]:
    source_dir = ROOT / 'scripts/dev/docs/latex'
    pending = [source_dir / (name + '.tex'), source_dir / 'solver-coverage.json']
    paths = set()
    while pending:
        path = pending.pop().resolve()
        if path in paths:
            continue
        if not path.is_relative_to(source_dir):
            raise RuntimeError(f'guide input escapes scripts/dev/docs/latex: {path}')
        paths.add(path)
        for value in re.findall(r'\\(?:input|include)\{([^{}]+)\}', path.read_text()):
            dependency = source_dir / value
            if not dependency.suffix:
                dependency = dependency.with_suffix('.tex')
            pending.append(dependency)
    return {str(path.relative_to(ROOT)): identity(path) for path in sorted(paths)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('guides', nargs='*', choices=tuple(GUIDES), default=list(GUIDES))
    parser.add_argument('--check', action='store_true', help='verify checked-in PDFs match their recorded sources')
    args = parser.parse_args()
    coverage = subprocess.run([sys.executable, str(ROOT / 'scripts/dev/docs/check_solver_coverage.py')])
    if coverage.returncode:
        return coverage.returncode
    manifest_path = ROOT / 'scripts/dev/docs/latex/manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    selected = args.guides or GUIDES
    if args.check:
        for name in selected:
            pdf = ROOT / 'docs' / GUIDES[name]
            record = manifest.get(name, {})
            if record.get('sources') != source_inputs(name) or not pdf.exists() or record.get('pdf_sha256') != identity(pdf):
                parser.error(f'{name} PDF/source identity changed; rebuild with make docs-pdf')
        print('PDF/source identities verified: ' + ', '.join(selected))
        return 0
    engine = shutil.which('tectonic')
    if not engine:
        parser.error('Tectonic is required; see docs/BUILDING.md for document builds.')
    source_dir = ROOT / 'scripts/dev/docs/latex'
    output = ROOT / 'build/docs'
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.setdefault('SOURCE_DATE_EPOCH', '0')
    for name in selected:
        source = source_dir / (name + '.tex')
        if not source.is_file():
            parser.error(f'missing maintained guide source: {source.relative_to(ROOT)}')
        log = output / (name + '-build.log')
        with log.open('w') as stream:
            run = subprocess.run([engine, '--untrusted', '--keep-logs', '--outdir', str(output), str(source)],
                                 cwd=source_dir, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        if run.returncode:
            print(log.read_text(), file=sys.stderr)
            return run.returncode
        transcript = (output / (name + '.log')).read_text(errors='replace')
        problems = [line for line in transcript.splitlines() if any(marker in line for marker in
                    ('Missing character:', 'Overfull \\hbox', 'Undefined control sequence', 'There were undefined references'))]
        if problems:
            raise RuntimeError(f'{name} has typesetting errors:\n' + '\n'.join(problems))
        produced = output / (name + '.pdf')
        if not produced.is_file() or produced.stat().st_size == 0:
            raise RuntimeError(f'LaTeX produced no PDF for {name}')
        destination = ROOT / 'docs' / GUIDES[name]
        shutil.copyfile(produced, destination)
        manifest[name] = {'sources': source_inputs(name), 'pdf_sha256': identity(destination),
                          'engine': subprocess.check_output([engine, '--version'], text=True).strip()}
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        print(f'Built {destination.relative_to(ROOT)} ({produced.stat().st_size:,} bytes)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
