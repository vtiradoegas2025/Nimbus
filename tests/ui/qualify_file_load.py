#!/usr/bin/env python3
"""Run the production Studio components against an on-disk manufactured fixture."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def validate(report, renderer=None, native_gpu_samples=()):
    required = {'native-history', 'single-volume', 'single-supercell', 'quad-volume',
                'quad-supercell', 'playback-sync', 'steady-export'}
    if report.get('profile') != 'file-load' or report.get('status') != 'passed':
        raise ValueError('Renderer test failed: ' + json.dumps(report.get('error')))
    checks = report.get('checks', [])
    if not required.issubset({item['name'] for item in checks}) or not all(item['passed'] for item in checks):
        raise ValueError('Missing or failed renderer checks')
    fixture = report['fixture']['specification']
    saved = fixture.get('schema') == 'nimbus.saved-visual-reference.v1' and fixture.get('values_modified') is False
    if not (fixture.get('not_a_simulation') is True or saved) or len(report['steps']) != fixture['frames']:
        raise ValueError('Incomplete visual fixture evidence')
    for name, panes in [('single-volume', 1), ('single-supercell', 1), ('quad-volume', 4), ('quad-supercell', 4)]:
        scenes = [scene for scene in report['scenes'] if scene['name'] == name]
        if len(scenes) != 1 or len(scenes[0]['captures']) != panes:
            raise ValueError('Missing scene captures: ' + name)
        for capture in scenes[0]['captures']:
            evidence = capture['evidence']
            if evidence['descriptor']['fidelityMode'] != 'scientific' or not capture['pngDataUri'].startswith('data:image/png;base64,'):
                raise ValueError('Capture is missing or uses presentation mode')
            native_match = any(renderer and renderer.lower() in sample.get('gpu', '').lower()
                               and sample.get('ownedWebKitProcesses') for sample in native_gpu_samples)
            if renderer and renderer.lower() not in capture['gpu']['renderer'].lower() and not native_match:
                raise ValueError('Unexpected GPU: ' + json.dumps(capture['gpu']))
    return {'status': 'passed', 'checks': len(checks), 'scenes': len(report['scenes']),
            'source_shape': fixture['shape'], 'field_bytes_per_frame': fixture['bytes_per_frame'],
            'note': 'Source workload size; inspect captured geometry for rendered LOD and regions.'}


def nvidia_sample(owner_pid):
    descendants = {owner_pid}
    pending = [owner_pid]
    while pending:
        pid = pending.pop()
        try:
            for child in Path(f'/proc/{pid}/task/{pid}/children').read_text().split():
                if int(child) not in descendants:
                    descendants.add(int(child)); pending.append(int(child))
        except OSError: pass
    try:
        xml = subprocess.check_output(['nvidia-smi', '-q', '-x'], timeout=3, stderr=subprocess.DEVNULL)
        samples = []
        for gpu in ET.fromstring(xml).findall('gpu'):
            processes = []
            for process in gpu.findall('processes/process_info'):
                pid = int(process.findtext('pid', '0'))
                name = process.findtext('process_name', '')
                if pid in descendants and 'WebKit' in name and 'G' in process.findtext('type', ''):
                    processes.append(dict(pid=pid, name=name, memory=process.findtext('used_memory')))
            samples.append(dict(time=time.time(), gpu=gpu.findtext('product_name'),
                uuid=gpu.findtext('uuid'), ownerPid=owner_pid, descendants=sorted(descendants), ownedWebKitProcesses=processes))
        return samples
    except (OSError, ValueError, subprocess.SubprocessError, ET.ParseError): return []


def process_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--application', type=Path, required=True, help='macOS .app or Linux native executable')
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--renderer', help='Require this substring in the actual WebGL renderer')
    parser.add_argument('--wayland-display')
    args = parser.parse_args()
    output = args.output.resolve()
    progress = Path(str(output) + '.progress.ndjson')
    launch_path = Path(str(output) + '.launch.json')
    if output.exists() or progress.exists() or launch_path.exists():
        parser.error('Use a new output path; prior evidence is preserved')
    output.parent.mkdir(parents=True, exist_ok=True)
    application = args.application.resolve()
    executable = application / 'Contents/MacOS/nimbus-studio' if sys.platform == 'darwin' else application
    executable_hash = hashlib.sha256(executable.read_bytes()).hexdigest()
    launch = dict(schema='nimbus.studio.visual-load-launch.v1', executable=str(executable),
        executable_sha256=executable_hash, dataset=str(args.directory.resolve()),
        fixture_receipt_sha256=hashlib.sha256((args.directory / 'fixture_receipt.json').read_bytes()).hexdigest(),
        status='running', started_at=time.time())
    launch_path.write_text(json.dumps(launch, indent=2) + '\n')
    env = os.environ.copy()
    settings = {'NIMBUS_STUDIO_VIS_QUALIFICATION_OUTPUT': str(output),
                'NIMBUS_STUDIO_VIS_QUALIFICATION_PROFILE': 'file-load',
                'NIMBUS_STUDIO_VIS_LOAD_DIRECTORY': str(args.directory.resolve())}
    env.update(settings)
    gpu_samples = []
    if sys.platform == 'darwin':
        command = ['open', '-n', '-W']
        for key, value in settings.items(): command.extend(['--env', key + '=' + value])
        command.append(str(args.application.resolve()))
    else:
        command = [str(args.application.resolve())]
        if args.wayland_display:
            env['XDG_RUNTIME_DIR'] = '/run/user/' + str(os.getuid())
            env['WAYLAND_DISPLAY'] = args.wayland_display
            env['DBUS_SESSION_BUS_ADDRESS'] = 'unix:path=' + env['XDG_RUNTIME_DIR'] + '/bus'
    with Path(str(output) + '.log').open('w') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + args.timeout
            last = ''
            while process.poll() is None and not output.exists():
                if time.monotonic() >= deadline: raise TimeoutError('Native renderer test timed out')
                if progress.exists():
                    lines = progress.read_text().splitlines()
                    if not lines:
                        time.sleep(.1)
                        continue
                    current = lines[-1]
                    if current != last: print(current, flush=True); last = current
                    if not process_alive(json.loads(lines[0])['pid']) and not output.exists():
                        raise RuntimeError('The native test window exited before completing its report; it may have been closed')
                    if sys.platform.startswith('linux') and args.renderer:
                        samples = nvidia_sample(process.pid)
                        gpu_samples.extend(samples)
                        with Path(str(output) + '.gpu.ndjson').open('a') as evidence:
                            for sample in samples: evidence.write(json.dumps(sample) + '\n')
                time.sleep(1)
            if not output.exists(): raise RuntimeError(f'Native viewer exited without evidence ({process.poll()})')
            result = validate(json.loads(output.read_text()), args.renderer, gpu_samples)
            if hashlib.sha256(executable.read_bytes()).hexdigest() != executable_hash:
                raise RuntimeError('Viewer executable changed during the test')
            launch['status'] = 'passed'
            launch['report_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
            print(json.dumps(result, indent=2))
        finally:
            launch['ended_at'] = time.time()
            if launch['status'] == 'running': launch['status'] = 'incomplete'
            launch_path.write_text(json.dumps(launch, indent=2) + '\n')
            if process.poll() is None:
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if sys.platform != 'darwin': os.killpg(process.pid, signal.SIGTERM)
                    else:
                        if progress.exists():
                            pid = json.loads(progress.read_text().splitlines()[0])['pid']
                            actual = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], text=True, capture_output=True).stdout.strip()
                            if actual == str(args.application.resolve() / 'Contents/MacOS/nimbus-studio'):
                                os.kill(pid, signal.SIGTERM)
                        process.terminate()


if __name__ == '__main__':
    main()
