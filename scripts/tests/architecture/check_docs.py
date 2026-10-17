#!/usr/bin/env python3
"""Check maintained Markdown links and repository paths without build artifacts."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[3]
GENERATED = {'.git', '.venv', 'venv', '.cache', '.claude', 'node_modules', 'target',
             'bin', 'reports', 'artifacts', 'devDocs', '__pycache__', 'screenshots', 'gen'}
PREFIXES = ('src/', 'include/', 'docs/', 'tests/', 'scripts/', 'configs/', 'ui/',
            'metal/', 'vulkan/', 'tools/', 'examples/', 'mk/')
ARCHIVED = ('docs/devDocs/', 'reports/backend_repairs/', 'threadHandOff.md',
            'CM1Soundings/', 'BenchMarkTesting/')


def markdown_files(root: Path):
    for directory, children, files in os.walk(root):
        relative = Path(directory).relative_to(root)
        children[:] = [name for name in children if name not in GENERATED
                       and not name.startswith(('nimbus-gate-result-', 'nimbus-release-manifest-'))
                       and not (name == 'build' and relative != Path('scripts/dev'))
                       and not (name == 'custom' and relative == Path('configs'))
                       and not (name == 'public' and relative == Path('ui/integration'))]
        for name in files:
            if name.endswith('.md') and name != 'threadHandOff.md':
                yield Path(directory) / name


def local_target(raw: str) -> str | None:
    raw = raw.strip()
    if raw.startswith('<'):
        raw = raw[1:raw.find('>')]
    else:
        raw = re.split(r'\s+["\']', raw, maxsplit=1)[0]
    parsed = urlsplit(raw)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    return unquote(parsed.path)


def check(path: Path, root: Path) -> list[str]:
    errors = []
    text = path.read_text(encoding='utf-8')
    prose = re.sub(r'(?ms)^\s*(`{3,}|~{3,}).*?^\s*\1\s*$', '', text)
    prose = re.sub(r'\$\$[\s\S]*?\$\$', '', prose)
    prose = re.sub(r'(?<!\\)\$[^$\n]+\$', '', prose)
    targets = re.findall(r'!?\[[^\]\n]*\]\(([^\n]+?)\)', prose)
    targets += re.findall(r'(?m)^\s*\[[^\]]+\]:\s*(\S+)', prose)
    for raw in targets:
        target = local_target(raw)
        if target is None:
            continue
        destination = (path.parent / target).resolve()
        if not destination.is_relative_to(root) or not destination.exists():
            errors.append(f'{path.relative_to(root)}: missing local link {raw}')
    for old in ARCHIVED:
        if old in text:
            errors.append(f'{path.relative_to(root)}: archived reference {old}')
    for span in re.findall(r'`([^`\n]+)`', prose):
        value = span.strip().removeprefix('./').split(' ', 1)[0].rstrip('.,:;')
        if not value.startswith(PREFIXES) or any(token in value for token in ('*', '...', '$', '<', '>')):
            continue
        parts = Path(value).parts
        if any(part in GENERATED or part == 'build' for part in parts) or value.startswith('configs/custom/'):
            continue
        if not (root / value).exists() and not (path.parent / value).exists():
            errors.append(f'{path.relative_to(root)}: missing repository path `{value}`')
    return sorted(set(errors))


def main() -> int:
    namelist = subprocess.run([sys.executable, str(ROOT / 'scripts/dev/docs/generate_namelist.py'), '--check'])
    if namelist.returncode:
        return namelist.returncode
    files = sorted(markdown_files(ROOT))
    failures = [error for path in files for error in check(path, ROOT)]
    for source in (ROOT / 'scripts/dev/docs/latex').rglob('*.tex'):
        for target in re.findall(r'\\href\{([^{}\n]+)\}', source.read_text()):
            target = target.replace('\\_', '_').replace('\\%', '%').replace('\\#', '#')
            local = local_target(target)
            if local is not None and not (ROOT / 'docs' / local).resolve().exists():
                failures.append(f'{source.relative_to(ROOT)}: missing PDF link {target}')
    for error in failures:
        print(error, file=sys.stderr)
    if failures:
        print(f'Documentation check failed: {len(failures)} references in {len(files)} files.', file=sys.stderr)
        return 1
    print(f'Documentation check passed: {len(files)} maintained Markdown files.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
