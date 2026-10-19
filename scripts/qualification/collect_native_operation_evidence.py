#!/usr/bin/env python3
"""Collect bounded native dispatch observations for review, without blessing a baseline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.qualification.catalog_inventory import INVENTORY, configured_backend, maintained_cases
from scripts.tests.catalog.qualify_catalog_bounded_runs import (
    require_bounded_run, sha256_file, write_json_atomic,
)
from scripts.tests.catalog.test_kernel_ownership_catalog import validate as validate_kernel_catalog


def collect(binary: Path, kernel_tool: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError('native observation directory must be empty')
    inputs = output / 'inputs'
    inputs.mkdir()
    identities = {}

    def freeze(source: Path, relative: Path) -> Path:
        destination = inputs / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        identity = sha256_file(source)
        shutil.copy2(source, destination)
        if sha256_file(destination) != identity:
            raise RuntimeError(f'input changed while copying: {source}')
        identities[relative.as_posix()] = identity
        return destination

    executable = freeze(binary, Path('nimbus_sim'))
    tool = freeze(kernel_tool, Path('kernel_ownership'))
    freeze(INVENTORY, Path('inventory.json'))
    freeze(Path(__file__), Path('collector.py'))
    freeze(ROOT / 'scripts/tests/catalog/qualify_catalog_bounded_runs.py', Path('validator.py'))
    freeze(ROOT / 'scripts/qualification/catalog_inventory.py', Path('catalog_inventory.py'))
    freeze(ROOT / 'scripts/tests/catalog/test_kernel_ownership_catalog.py', Path('kernel_validator.py'))
    shaders = sorted((ROOT / 'vulkan/shaders/compute').rglob('*.spv'))
    if not shaders:
        raise RuntimeError('native observations require shader artifacts')
    for shader in shaders:
        freeze(shader, Path('shaders') / shader.relative_to(ROOT / 'vulkan/shaders/compute'))
    result = subprocess.run([str(tool)], cwd=ROOT, capture_output=True, text=True, check=True)
    catalog = json.loads(result.stdout)
    validate_kernel_catalog(catalog)
    write_json_atomic(output / 'kernel_catalog.json', catalog)
    active = {row['kernelName']: row['operationId'] for row in catalog['records']
              if row['availability'] == 'active'}
    cases = [path for group in maintained_cases() for path in group if configured_backend(path) != 'cpu']
    frozen_configs = {path: freeze(path, Path('configs') / path.relative_to(ROOT)) for path in cases}
    evidence = {
        'schema': 'nimbus.native-operation-observations.v1', 'version': 1,
        'status': 'collecting', 'expectedCases': len(cases), 'inputs': identities,
        'runContract': {'ordinaryDurationSeconds': 1, 'movingReferenceDurationSeconds': 30,
                        'writeEverySeconds': 0, 'checkpointEverySeconds': 0, 'threads': 1,
                        'timingTelemetry': True},
        'records': [], 'failures': [],
        'claimBoundary': 'Unreviewed observations, not operation requirements or readiness evidence. '
                         'The strict catalog gate must independently verify a reviewed requirements document.',
    }
    environment = {'TMV_SHADER_DIR': str(inputs / 'shaders'), 'NIMBUS_GPU_KERNEL_TIMING': '1'}
    previous = {name: os.environ.get(name) for name in environment}
    scratch = ROOT / 'build/qualification'
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        os.environ.update(environment)
        for index, config in enumerate(cases, 1):
            name = config.relative_to(ROOT).as_posix()
            with tempfile.TemporaryDirectory(prefix='native-observation-', dir=scratch) as directory:
                temporary = Path(directory)
                selected = temporary / config.name
                shutil.copy2(frozen_configs[config], selected)
                try:
                    record = require_bounded_run(executable, selected, temporary / 'run', active, True)
                    if record is None or record['configSha256'] != identities[f'configs/{name}']:
                        raise RuntimeError('native observation has inconsistent configuration identity')
                    record['config'] = name
                    evidence['records'].append(record)
                    outcome = 'observed'
                except Exception as error:
                    evidence['failures'].append({'config': name, 'error': str(error)})
                    outcome = 'failed'
            write_json_atomic(output / 'observations.json', evidence)
            print(f'[{index}/{len(cases)}] {outcome}: {config.name}', flush=True)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if any(sha256_file(inputs / name) != identity for name, identity in identities.items()):
        raise RuntimeError('frozen native observation inputs changed during collection')
    evidence['status'] = 'incomplete' if evidence['failures'] else 'observed_unreviewed'
    evidence['completedUtc'] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(output / 'observations.json', evidence)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'bin/nimbus_sim')
    parser.add_argument('--kernel-ownership-tool', type=Path, default=ROOT / 'bin/nimbus_kernel_ownership')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.binary.resolve(strict=True), args.kernel_ownership_tool.resolve(strict=True),
                     args.output.resolve())
    return 1 if result['failures'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
