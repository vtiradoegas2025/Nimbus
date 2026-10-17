import hashlib
import json
import os
from pathlib import Path
import signal
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time

MAX_REPLY = 8 * 1024 * 1024
ACTION = sys.argv[1]
ARGS = json.loads(sys.argv[2])
WORK = Path(ARGS['workDirectory']).resolve(strict=True)
if sys.platform != 'linux' or sys.version_info < (3, 8) or not WORK.is_dir():
    raise RuntimeError('Remote computation requires Linux, Python 3.8+ and an existing work directory')


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def owned_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = WORK / path
    resolved = path.resolve()
    if resolved == WORK or WORK not in resolved.parents:
        raise RuntimeError('Remote path escapes the configured work directory')
    return resolved


def bounded(args, timeout=60):
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        child = subprocess.Popen(args, cwd=Path(ARGS.get('jobDirectory', str(WORK))), stdin=subprocess.DEVNULL,
                                 stdout=stdout, stderr=stderr, start_new_session=True)
        deadline = time.monotonic() + timeout
        while child.poll() is None:
            if time.monotonic() >= deadline or os.fstat(stdout.fileno()).st_size > MAX_REPLY or os.fstat(stderr.fileno()).st_size > MAX_REPLY:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                raise RuntimeError('Remote command exceeded its time or output bound')
            time.sleep(0.02)
        stdout.seek(0)
        stderr.seek(0)
        out = stdout.read(MAX_REPLY + 1)
        err = stderr.read(MAX_REPLY + 1)
        if len(out) > MAX_REPLY or len(err) > MAX_REPLY:
            raise RuntimeError('Remote command exceeded its output bound')
        return child.returncode, out, err


def identity():
    binary = Path(ARGS['binaryPath']).resolve(strict=True)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError('Remote Nimbus executable is unavailable')
    before = digest(binary)
    code, out, err = bounded([str(binary), '--scheme-catalog=json'])
    if code != 0:
        raise RuntimeError('Remote catalog rejected: ' + err.decode('utf-8', 'replace')[:2048])
    if digest(binary) != before:
        raise RuntimeError('Remote executable changed while querying its catalog')
    catalog = json.loads(out)
    if not isinstance(catalog, dict):
        raise RuntimeError('Remote catalog is not an object')
    fingerprint = 0xcbf29ce484222325
    for byte in out:
        fingerprint = ((fingerprint ^ byte) * 0x100000001b3) & 0xffffffffffffffff
    metadata = binary.stat()
    return dict(available=True, platform='linux', binaryPath=str(binary),
                binaryFingerprint='sha256:' + before, catalogFingerprint='fnv1a64:%016x' % fingerprint,
                byteLength=metadata.st_size, modifiedUnixMs=metadata.st_mtime_ns // 1000000,
                catalog=catalog, catalogText=out.decode('utf-8'), workDirectory=str(WORK))


def verify():
    current = identity()
    if current['binaryFingerprint'] != ARGS['expectedBinaryFingerprint'] or current['catalogFingerprint'] != ARGS['expectedCatalogFingerprint']:
        raise RuntimeError('Remote executable or catalog changed after preflight')
    if digest(owned_path(ARGS['configPath'])) != ARGS['sourceSha256']:
        raise RuntimeError('Remote staged YAML changed after validation')
    for asset in ARGS.get('assets', []):
        if digest(owned_path(asset['remotePath'])) != asset['sha256']:
            raise RuntimeError('Remote configuration asset changed after staging')
    return current


def birth(pid):
    stat = Path('/proc/%d/stat' % pid)
    if stat.exists():
        return stat.read_text().rsplit(')', 1)[1].split()[19]
    value = subprocess.run(['ps', '-p', str(pid), '-o', 'lstart='], capture_output=True, text=True, timeout=5)
    return value.stdout.strip() if value.returncode == 0 else None


def controller_path():
    token = ARGS['token']
    if not token or any(c not in '0123456789abcdef-' for c in token):
        raise RuntimeError('Invalid remote run token')
    return owned_path('.nimbus-studio/' + token + '/process.json')


def publish_controller(path, receipt):
    temporary = path.with_name('process.json.pending')
    with open(temporary, 'w') as output:
        json.dump(receipt, output)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def active(receipt):
    pid = receipt['pid']
    try:
        return birth(pid) == receipt['birth'] and os.getpgid(pid) == pid
    except ProcessLookupError:
        return False


def group_alive(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def stop(receipt, owned_child=None):
    if owned_child is None and not active(receipt):
        return
    pid = receipt['pid']
    if not group_alive(pid):
        return
    os.killpg(pid, signal.SIGINT)
    deadline = time.monotonic() + 5
    while group_alive(pid) and time.monotonic() < deadline:
        if owned_child is not None:
            owned_child.poll()
        time.sleep(0.05)
    if group_alive(pid):
        os.killpg(pid, signal.SIGKILL)
        deadline = time.monotonic() + 5
        while group_alive(pid) and time.monotonic() < deadline:
            if owned_child is not None:
                owned_child.poll()
            time.sleep(0.05)
    if group_alive(pid):
        raise RuntimeError('Remote process group has not stopped')


if ACTION == 'probe':
    print(json.dumps(identity()))
elif ACTION == 'stage':
    destination = owned_path(ARGS['destination'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if ARGS['byteLength'] > shutil.disk_usage(destination.parent).free - 16 * 1024 * 1024:
        raise RuntimeError('Remote input staging exceeds available workspace disk space')
    temporary = destination.parent / ('.' + destination.name + '.upload-' + ARGS['token'])
    try:
        with open(temporary, 'xb') as output:
            os.chmod(temporary, 0o600)
            count = 0
            while True:
                chunk = sys.stdin.buffer.read(1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                if count > ARGS['byteLength']:
                    raise RuntimeError('Staged asset exceeds its declared length')
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if count != ARGS['byteLength'] or digest(temporary) != ARGS['sha256']:
            raise RuntimeError('Staged asset length or fingerprint mismatch')
        os.replace(temporary, destination)
        print(json.dumps(dict(path=str(destination), sha256=ARGS['sha256'])))
    finally:
        temporary.unlink(missing_ok=True)
elif ACTION == 'verify-asset':
    path = owned_path(ARGS['path'])
    if not path.is_file() or digest(path) != ARGS['sha256']:
        raise RuntimeError('Absolute mapped assets must already exist with the exact requested bytes')
    print(json.dumps(dict(sha256=ARGS['sha256'])))
elif ACTION == 'validate':
    current = verify()
    argv = [current['binaryPath'], '--config', str(owned_path(ARGS['configPath'])), '--validate-only', '--headless']
    if ARGS.get('requireContext'):
        argv.append('--preflight-context=json')
    code, stdout, stderr = bounded(argv, 90)
    verify()
    print(json.dumps(dict(exitCode=code, stdout=stdout.decode('utf-8', 'replace'), stderr=stderr.decode('utf-8', 'replace'))))
elif ACTION == 'migrate':
    current = verify()
    code, stdout, stderr = bounded([current['binaryPath'], '--migrate-config', str(owned_path(ARGS['configPath']))], 90)
    verify()
    print(json.dumps(dict(exitCode=code, stdout=stdout.decode('utf-8', 'replace'), stderr=stderr.decode('utf-8', 'replace'))))
elif ACTION == 'start':
    child = None
    lost_connection = threading.Event()
    finished = threading.Event()
    signals = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    def disconnected(signum, frame):
        lost_connection.set()
    for sig in signals:
        signal.signal(sig, disconnected)
    def connection_owner():
        while select.select([sys.stdin.fileno()], [], [], 3)[0]:
            if not os.read(sys.stdin.fileno(), 1):
                break
        lost_connection.set()
    threading.Thread(target=connection_owner, daemon=True).start()
    def retire_disconnected_process():
        lost_connection.wait()
        while child is None and not finished.is_set():
            time.sleep(0.02)
        if child is not None:
            stop(dict(pid=child.pid), child)
    threading.Thread(target=retire_disconnected_process, daemon=True).start()
    try:
        current = verify()
        path = controller_path()
        cancellation = path.with_name('cancelled')
        if lost_connection.is_set() or cancellation.exists():
            raise RuntimeError('Remote launch was cancelled before process creation')
        outdir = owned_path(ARGS['outputPath'])
        outdir.mkdir(parents=True, exist_ok=False)
        child = subprocess.Popen([current['binaryPath'], '--config', str(owned_path(ARGS['configPath'])),
                                  '--headless', '--log-profile=normal', '--outdir=' + str(outdir)],
                                 cwd=Path(ARGS['jobDirectory']), stdin=subprocess.DEVNULL, start_new_session=True)
        receipt = dict(pid=child.pid, birth=birth(child.pid), token=ARGS['token'])
        if receipt['birth'] is None:
            receipt['exitCode'] = child.wait()
        publish_controller(path, receipt)
        if lost_connection.is_set() or cancellation.exists():
            stop(receipt, child)
        code = child.wait()
        if group_alive(child.pid):
            stop(receipt, child)
        receipt['exitCode'] = code
        publish_controller(path, receipt)
        sys.exit(code if code >= 0 else 128 - code)
    finally:
        finished.set()
        if child is not None:
            if group_alive(child.pid):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
elif ACTION == 'cancel':
    path = controller_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_name('cancelled').touch(exist_ok=True)
    deadline = time.monotonic() + 5
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    if path.exists():
        receipt = json.loads(path.read_text())
        if receipt['token'] != ARGS['token']:
            raise RuntimeError('Remote process receipt token mismatch')
        if 'exitCode' not in receipt:
            stop(receipt)
    print(json.dumps(dict(stopped=True)))
elif ACTION == 'status':
    path = controller_path()
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16384:
        raise RuntimeError('A bounded remote process receipt is required to confirm termination')
    receipt = json.loads(path.read_text())
    if receipt.get('token') != ARGS['token'] or type(receipt.get('pid')) is not int or receipt['pid'] <= 0:
        raise RuntimeError('Invalid remote process receipt identity')
    stopped = not active(receipt)
    print(json.dumps(dict(stopped=stopped, exitCode=receipt.get('exitCode'))))
elif ACTION == 'files':
    outdir = owned_path(ARGS['outputPath'])
    files = []
    for path in sorted(outdir.rglob('*')):
        if path.is_symlink():
            raise RuntimeError('Remote output contains a symbolic link')
        if path.is_file():
            relative = path.relative_to(outdir).as_posix()
            if any(part.startswith('.') for part in Path(relative).parts):
                continue
            files.append(dict(path=relative, byteLength=path.stat().st_size, sha256=digest(path)))
            if len(files) > 100000:
                raise RuntimeError('Remote history exceeds the output file-count bound')
    payload = json.dumps(files)
    if len(payload.encode('utf-8')) > MAX_REPLY:
        raise RuntimeError('Remote history manifest exceeds its reply bound')
    print(payload)
elif ACTION == 'checkpoint-files':
    checkpoint = owned_path(ARGS['path'])
    files = []
    for path in (checkpoint, checkpoint.with_name(checkpoint.name + '.json')):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError('Remote checkpoint and lifecycle sidecar must be regular files')
        files.append(dict(path=path.relative_to(WORK).as_posix(), byteLength=path.stat().st_size, sha256=digest(path)))
    print(json.dumps(files))
elif ACTION == 'fetch':
    outdir = WORK if Path(ARGS['outputPath']).resolve() == WORK else owned_path(ARGS['outputPath'])
    path = (outdir / ARGS['path']).resolve(strict=True)
    if outdir not in path.parents or path.is_symlink() or not path.is_file():
        raise RuntimeError('Invalid remote output path')
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            sys.stdout.buffer.write(chunk)
else:
    raise RuntimeError('Unknown remote compute action')
