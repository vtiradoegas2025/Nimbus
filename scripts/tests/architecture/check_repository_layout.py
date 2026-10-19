#!/usr/bin/env python3
"""Reject generated or retired public-tree entries in the Git index."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
RETIRED = (
    'benchmarks/', 'tools/', 'examples/', 'data/', 'docs/latex/',
    'ui/integration/', 'ui/testdata/', 'docs/devDocs/', 'scripts/dev/docs/',
)
GENERATED = ('build/', 'bin/', 'reports/', 'runs/', 'tests/artifacts/', 'ui/build/')
CACHE_PARTS = {'node_modules', 'target', '__pycache__', '.venv', '.cache'}
RETIRED_FILES = {'docs/CM1Mapping.md', 'docs/domain.md', 'docs/WINDOWS_ACCEPTANCE.md'}
PRIVATE_FILES = {'docs/DEVELOPMENT.md'}


def main() -> int:
    tracked = subprocess.check_output(
        ['git', 'ls-files', '-z'], cwd=ROOT, text=True).split('\0')
    failures = []
    for name in filter(None, tracked):
        path = Path(name)
        if (name.startswith(RETIRED + GENERATED)
                or name in RETIRED_FILES
                or name in PRIVATE_FILES
                or CACHE_PARTS.intersection(path.parts)
                or (name.startswith('docs/') and path.suffix == '.tex')
                or (name.startswith('ui/') and path.name.startswith('test_')
                    and path.suffix in {'.ts', '.tsx'})):
            failures.append(name)
    if failures:
        print('Retired/generated entries must not be tracked:\n' + '\n'.join(failures),
              file=sys.stderr)
        return 1
    print('Repository layout passed: publication, application and test sources are separated.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
