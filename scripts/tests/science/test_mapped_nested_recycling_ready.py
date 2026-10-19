#!/usr/bin/env python3
"""Bounded public Bell/Schär coupled parent recycling and native field ownership."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from npy_support import read_float32_npy
from test_nested_grid_recycling_ready import checkpoint_at, checkpoint_section, scientific_runtime_section, private_recycling_inventory
from test_two_way_nested_ready import require_closure, require_water_closure, require_private_closure, require_feedback_output_budget
from scripts.qualification.checkpointing.qualify_vulkan_restart import LEDGERS, require_restart_suffix
from scripts.qualification.domain_motion.qualify_domain_recycling import runtime_environment

LEVELS = (Path(), Path('nests/child_0000'))
SHIFTS = {
    'bell2': [[5, 0], [-3, -3]],
    'bell3': [[1, 0]],
    'schar2': [[5, -3]],
    'schar3': [[0, 4], [6, 0]],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frames(root):
    result = {}
    for level in LEVELS:
        for manifest in sorted((root / level).glob('step_*/manifest.json')):
            document = json.loads(manifest.read_text())
            result[level, round(document['simulation_time_s'], 8)] = (manifest.parent, document)
    if not result or any(not any(key[0] == level for key in result) for level in LEVELS):
        raise RuntimeError(f'missing parent or child history: {root}')
    return result


def compare(reference, actual, exact, refinement_ratio=2, *, restart=False, fields=None, water_species=()):
    before, after = frames(reference), frames(actual)
    for level in LEVELS:
        actual_times = {time for item_level, time in after if item_level == level}
        expected_times = {time for item_level, time in before if item_level == level}
        if restart:
            expected_times = {time for time in expected_times if time >= min(actual_times)}
        if actual_times != expected_times:
            raise RuntimeError(f'incomplete field history for {level}: {actual}')
    maximum_ratio = 0.0
    compared = 0
    for key, (directory, manifest) in after.items():
        if key not in before:
            raise RuntimeError(f'unknown continuation time {key}: {directory}')
        expected, expected_manifest = before[key]
        for name in ('horizontal_frame', 'vertical_geometry'):
            if name == 'horizontal_frame':
                if manifest['grid'][name] != expected_manifest['grid'][name]:
                    raise RuntimeError(f'Earth frame changed: {directory}')
            else:
                for field in ('center_path', 'interface_path'):
                    if sha(directory / manifest['grid'][name][field]) != sha(expected / expected_manifest['grid'][name][field]):
                        raise RuntimeError(f'canonical geometry changed: {directory}')
        paths = sorted(expected.glob('*.npy'))
        if {p.name for p in paths} != {p.name for p in directory.glob('*.npy')}:
            raise RuntimeError(f'published field set changed: {directory}')
        water = [read_float32_npy(expected / f'{name}.npy')[2] for name in water_species
                 if (expected / f'{name}.npy').exists()]
        water_scale = max((math.fsum(values) for values in zip(*water)), default=0)
        for path in paths:
            shape, raw, values = read_float32_npy(path)
            other_shape, other_raw, other = read_float32_npy(directory / path.name)
            if shape != other_shape or not all(math.isfinite(v) for v in other):
                raise RuntimeError(f'invalid field: {directory / path.name}')
            if path.stem in ('qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'qh') and min(other) < 0:
                raise RuntimeError(f'negative water: {directory / path.name}')
            if fields is not None and path.stem not in fields:
                continue
            if exact:
                if raw != other_raw:
                    raise RuntimeError(f'field identity changed: {directory / path.name}')
            else:
                scale = water_scale if path.stem in water_species else max(map(abs, values))
                ulp = math.ldexp(1.0, math.frexp(scale)[1] - 24) if scale else math.ldexp(1.0, -149)
                steps = max(1, round(key[1] / .2) * (refinement_ratio if key[0].parts else 1))
                limit = 32 * steps * ulp
                if path.stem in ('u', 'v', 'w'):
                    limit = max(limit, steps * 5e-5)
                error = max(abs(a - b) for a, b in zip(values, other))
                if error > limit:
                    raise RuntimeError(f'native parity failed: {directory / path.name}: {error} > {limit}')
                maximum_ratio = max(maximum_ratio, error / limit)
            compared += 1
    return {'fields_compared': compared, 'maximum_error_to_limit': maximum_ratio}


def checkpoint_identity(reference, actual):
    first = checkpoint_at(reference / 'checkpoints', 65)
    second = checkpoint_at(actual / 'checkpoints', 65)
    for section in range(1, 8):
        a = scientific_runtime_section(first, [reference, actual]) if section == 6 else checkpoint_section(first, section)
        b = scientific_runtime_section(second, [reference, actual]) if section == 6 else checkpoint_section(second, section)
        if a != b:
            raise RuntimeError(f'checkpoint section {section} differs: {actual}')


def movement(root, log, geometry="bell2"):
    ratio = int(geometry[-1])
    events = [json.loads(path.read_text()) for path in sorted((root / 'domain_recycling_swath').glob('transaction_*/event.json'))]
    events.sort(key=lambda event: event['transaction_sequence'])
    if [event['shift_cells'] for event in events] != SHIFTS[geometry]:
        raise RuntimeError(f'unexpected {geometry} mapped shifts: {root}')
    for event in events:
        if f"committed transaction={event['transaction_id']}" not in log:
            raise RuntimeError('retiring swath lacks a successful runtime commit')
    history = frames(root)
    child_geometry = set()
    for (level, _), (directory, manifest) in history.items():
        if level.parts:
            if manifest['grid']['horizontal_frame']['grid_origin_t0_earth_m'] != [6000 - 500 / ratio, 3000 - 500 / ratio]:
                raise RuntimeError('parent recycling moved the child Earth frame')
            geometry = manifest['grid']['vertical_geometry']
            child_geometry.add(tuple(sha(directory / geometry[key]) for key in ('center_path', 'interface_path')))
    if len(child_geometry) != 1:
        raise RuntimeError('parent recycling changed fixed child geometry')
    shifts = [sum(event['shift_cells'][axis] for event in events) for axis in (0, 1)]
    if history[Path(), 65][1]['grid']['horizontal_frame']['grid_origin_t0_earth_m'] != [500 * value for value in shifts]:
        raise RuntimeError('parent cumulative Earth origin did not publish')
    bounds = json.loads((root / 'storm_track_latest.json').read_text())['movement_authority']['legal_parent_shift_bounds_cells']
    span = 4
    expected = {'available': True, 'minimum_i': 11 - shifts[0] + span - 15, 'maximum_i': 10 - shifts[0],
                'minimum_j': 5 - shifts[1] + span - 15, 'maximum_j': 4 - shifts[1]}
    if bounds != expected:
        raise RuntimeError('movement evidence omitted the fixed-child placement bounds')
    return events


def qualify(binary, config, backend, root, environment, geometry="bell2", *, private_scheme=None):
    first_restart = 5 if geometry.startswith("schar") or private_scheme else 20
    root.mkdir()
    def run(label, threads, resume=None, expect=True, selected=config, duration=65):
        output = root / label
        command = [str(binary), '--headless', '--config', str(selected), f'--duration={duration}',
                   '--write-every=5', f'--checkpoint-every={first_restart}', f'--threads={threads}', f'--outdir={output}']
        if resume:
            command.append(f'--resume={resume}')
        result = subprocess.run(command, cwd=ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (root / f'{label}.log').write_text(result.stdout)
        if (result.returncode == 0) != expect:
            raise RuntimeError(f'{label}: unexpected exit {result.returncode}; see {root / (label + ".log")}')
        return output, result.stdout

    source = config.read_text()
    ratio = int(geometry[-1])
    mutations = [
        ('height', 'h0: 300' if geometry.startswith('bell') else 'H: 300',
         'h0: 301' if geometry.startswith('bell') else 'H: 301'),
        ('feature', 'feature_center_earth_x_m: 4500', 'feature_center_earth_x_m: 4501'),
        ('placement', 'child_origin_parent_cell_i: 11', 'child_origin_parent_cell_i: 10'),
        ('feedback', 'feedback: two_way', 'feedback: one_way'),
        ('sgs-owner', 'gpu_apply_sgs: false', 'gpu_apply_sgs: true'),
        ('footprint', f'child_cells_x: {4 * ratio}\n    child_cells_y: {4 * ratio}',
         f'child_cells_x: {3 * ratio}\n    child_cells_y: {3 * ratio}'),
    ]
    if private_scheme == 'thompson':
        mutations.append(('aerosol', 'water_friendly_aerosol_surface_m3: 350000000',
                          'water_friendly_aerosol_surface_m3: 350000001'))
    elif private_scheme:
        mutations.append(('evaporation', 'scheme: milbrandt\n',
                          'scheme: milbrandt\n  shared:\n    autoconversion_rate_s_inv: 0.002\n'))
    for label, old, new in mutations:
        if old not in source:
            raise RuntimeError(f'negative case missing its source token: {label}')
        invalid = root / f'{label}.yaml'
        invalid.write_text(source.replace(old, new))
        output, _ = run(label, 1, expect=False, selected=invalid, duration=0)
        if list(output.glob('step_*')):
            raise RuntimeError(f'rejected selection published fields: {label}')

    continuous, log = run('continuous-t1', 1)
    if 'Execution mode: qualified' not in log or f'[COMPUTE] backend={backend}' not in log or 'fallback=off' not in log:
        raise RuntimeError('ordinary fallback-off science admission was not exercised')
    require_closure(log, backend)
    require_water_closure(log, True)
    private_inventory = None
    if private_scheme:
        private = require_private_closure(log, private_scheme)
        private_inventory = private_recycling_inventory(checkpoint_at(continuous / 'checkpoints', 65), set(private))
        for entry in private_inventory:
            aerosol = '.Nwfa_' in entry['name'] or '.Nifa_' in entry['name']
            if (aerosol and entry['admitted'] <= 0) or (not aerosol and entry['admitted'] != 0):
                raise RuntimeError('entering private inventory is not a fresh clear-air environment')
        require_feedback_output_budget(continuous, 13, ('qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'qh'))
    events = movement(continuous, log, geometry)
    if backend != 'cpu':
        for kernel in ('domain/domain_recycling_homogeneous_field.comp', 'terrain/mapped_scalar_update.comp',
                       'terrain/mapped_slow_tendencies.comp', 'terrain/mapped_fast_momentum.comp',
                       'dynamics/conservative_momentum.comp', 'chaos/earth_control_evolution.comp',
                       'radiation/simple_grey.comp', 'boundary_layer/slab_column.comp'):
            if not re.search(re.escape(kernel) + r':[1-9][0-9]*', log):
                raise RuntimeError(f'missing native owner {kernel}')
        family = private_scheme or 'kessler'
        kernels = {'kessler': ('kessler_pointwise', 'kessler_sedimentation'),
                   'thompson': ('thompson_cloud_adjustment', 'thompson_warm_rain', 'thompson_cloud_phase',
                                'thompson_freezing', 'thompson_deposition', 'thompson_completion',
                                'thompson_moment_sedimentation'),
                   'milbrandt': ('milbrandt_warm_rain', 'milbrandt_cloud_adjustment', 'milbrandt_melting',
                                 'milbrandt_cold', 'milbrandt_sedimentation')}[family]
        for kernel in kernels:
            if not re.search(re.escape(f'microphysics/{kernel}.comp') + r':[1-9][0-9]*', log):
                raise RuntimeError(f'missing native {family} owner {kernel}')
        if 'gpu_apply_slow: true' in source and not re.search(r'dynamics/slow_apply\.comp:[1-9][0-9]*', log):
            raise RuntimeError('native slow-application profile omitted its device writer')
    threaded, threaded_log = run('continuous-t4', 4)
    movement(threaded, threaded_log, geometry)
    compare(continuous, threaded, True)
    checkpoint_identity(continuous, threaded)
    if private_scheme:
        if require_private_closure(threaded_log, private_scheme) != private:
            raise RuntimeError('threading changed private feedback closure')
        if private_recycling_inventory(checkpoint_at(threaded / 'checkpoints', 65), set(private)) != private_inventory:
            raise RuntimeError('threading changed private inflow inventory')
    for level in LEVELS:
        for ledger in LEDGERS:
            if (continuous / level / ledger).read_bytes() != (threaded / level / ledger).read_bytes():
                raise RuntimeError(f'threading changed {level / ledger}')
    for time in (first_restart, 60):
        resumed, resumed_log = run(f'resumed-{time}-t4', 4, checkpoint_at(continuous / 'checkpoints', time))
        compare(continuous, resumed, True, restart=True)
        checkpoint_identity(continuous, resumed)
        for level in LEVELS:
            require_restart_suffix(continuous / level, resumed / level)
        require_closure(resumed_log, f'{backend} restart')
        require_water_closure(resumed_log)
        if private_scheme:
            for name, values in require_private_closure(resumed_log, private_scheme, False).items():
                if values != private[name][-len(values):]:
                    raise RuntimeError(f'restart changed private feedback {name}')
            if private_recycling_inventory(checkpoint_at(resumed / 'checkpoints', 65), set(private)) != private_inventory:
                raise RuntimeError('restart changed private inflow inventory')
        if time == first_restart:
            movement(resumed, resumed_log, geometry)
    return {'geometry': geometry, 'backend': backend, 'private_scheme': private_scheme,
            'private_recycling_inventory': private_inventory, 'native_slow_apply': 'gpu_apply_slow: true' in source,
            'status': 'Ready', 'events': [event['shift_cells'] for event in events]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, default=ROOT / 'bin/nimbus_sim')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--geometry', choices=(*SHIFTS, 'all'), default='all')
    args = parser.parse_args()
    root = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix='nimbus-mapped-recycling-')) / 'evidence'
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / 'inputs'
    inputs.mkdir()
    binary = inputs / 'nimbus_sim'
    shutil.copy2(args.binary.resolve(), binary)
    shaders = inputs / 'shaders'
    source_shaders = Path(os.environ.get('TMV_SHADER_DIR') or ROOT / 'vulkan/shaders/compute')
    for path in source_shaders.rglob('*.spv'):
        target = shaders / path.relative_to(source_shaders)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    configs = {}
    geometries = SHIFTS if args.geometry == 'all' else (args.geometry,)
    for geometry in geometries:
        for backend in ('cpu', 'metal', 'vulkan'):
            variant = f'{geometry}-{backend}'
            configs[variant] = (inputs / f'{variant}.yaml', geometry, backend)
            shutil.copy2(ROOT / f'tests/fixtures/configs/qualification/mapped_nested_recycling_kessler_{geometry}_tvd_{backend}_ready.yaml', configs[variant][0])
        for backend in ('metal', 'vulkan'):
            variant = f'{geometry}-{backend}-native-slow'
            configs[variant] = (inputs / f'{variant}.yaml', geometry, backend)
            shutil.copy2(ROOT / f'tests/fixtures/configs/qualification/mapped_nested_recycling_kessler_{geometry}_tvd_native_slow_{backend}_ready.yaml', configs[variant][0])
    hashes = {str(path.relative_to(inputs)): sha(path) for path in inputs.rglob('*') if path.is_file()}
    (root / 'inputs.json').write_text(json.dumps(hashes, indent=2))
    def one(variant):
        config, geometry, backend = configs[variant]
        environment = runtime_environment(backend)
        environment['TMV_SHADER_DIR'] = str(shaders)
        return qualify(binary, config, backend, root / variant, environment, geometry)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(one, configs))
    parity = {variant: compare(root / f'{geometry}-cpu/continuous-t1', root / variant / 'continuous-t1', False, int(geometry[-1]))
              for variant, (_, geometry, backend) in configs.items() if backend != 'cpu'}
    if hashes != {str(path.relative_to(inputs)): sha(path) for path in inputs.rglob('*') if path.is_file()}:
        raise RuntimeError('qualification inputs changed during execution')
    (root / 'summary.json').write_text(json.dumps({'status': 'Ready', 'cases': results, 'parity': parity,
                                                   'long_campaigns': 'pending', 'external_review': 'pending'}, indent=2))
    print(f'mapped nested recycling bounded gate PASS: {root}')


if __name__ == '__main__':
    main()
