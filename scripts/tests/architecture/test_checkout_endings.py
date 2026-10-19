#!/usr/bin/env python3
"""Keep source hashes portable without rewriting reviewed reference payloads."""
from pathlib import Path
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]


def checkout(attributes: bool) -> dict[str, tuple[bytes, bytes]]:
    with tempfile.TemporaryDirectory(prefix='nimbus-checkout-endings-') as directory:
        root = Path(directory)
        source = root / 'source'
        source.mkdir()
        files = {
            name: (ROOT / name).read_bytes() for name in (
                'src/init/config/v1_loader.cpp',
                'vulkan/shaders/compute/microphysics/thompson_completion.comp',
                'docs/STUDIO.pdf',
            )
        }
        files['tests/fixtures/reference/line-ending-fixture.txt'] = b'first\r\nsecond\r\n'
        if attributes:
            files['.gitattributes'] = (ROOT / '.gitattributes').read_bytes()
        for name, content in files.items():
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

        def git(*arguments: str) -> None:
            environment = {key: value for key, value in os.environ.items()
                           if key not in {'GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE'}}
            environment.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1', GIT_ATTR_NOSYSTEM='1')
            subprocess.run(['git', '-c', f'core.attributesFile={os.devnull}', '-C', str(source), *arguments],
                           env=environment, check=True, capture_output=True)

        git('init', '-q')
        git('-c', 'core.autocrlf=false', 'add', '--all')
        destination = root / 'checkout'
        git('-c', 'core.autocrlf=true', 'checkout-index', '--all', f'--prefix={destination}/')
        return {name: (content, (destination / name).read_bytes()) for name, content in files.items()}


def main() -> None:
    baseline = checkout(False)
    assert baseline['src/init/config/v1_loader.cpp'][0] != baseline['src/init/config/v1_loader.cpp'][1]
    for name, (before, after) in checkout(True).items():
        assert before == after, f'Windows-style checkout changed reviewed bytes: {name}'
    print('Checkout line endings: source hashes and reference bytes survive core.autocrlf=true.')


if __name__ == '__main__':
    main()
