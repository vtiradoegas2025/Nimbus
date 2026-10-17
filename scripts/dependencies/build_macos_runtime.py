#!/usr/bin/env python3
"""Build pinned scientific runtime libraries for the supported macOS minimum."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
SPECIFICATION = Path(__file__).with_name('macos_runtime_sources.json')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=ROOT / 'build/macos-runtime')
    parser.add_argument('--architecture', choices=('arm64', 'x86_64'), default=platform.machine())
    parser.add_argument('--jobs', type=int, default=min(os.cpu_count() or 1, 8))
    args = parser.parse_args()
    if platform.system() != 'Darwin' or args.jobs < 1:
        parser.error('a macOS build host and positive worker count are required')
    if not hasattr(tarfile, 'data_filter'):
        parser.error('Python with safe tar extraction is required; update Homebrew python3')
    spec = json.loads(SPECIFICATION.read_text())
    minimum = spec['minimum_macos']
    work = args.work.resolve()
    prefix = work / 'prefix'
    for name in ('downloads', 'sources', 'logs', 'prefix'):
        (work / name).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['MACOSX_DEPLOYMENT_TARGET'] = minimum
    env['PKG_CONFIG_LIBDIR'] = str(prefix / 'lib/pkgconfig')
    sdk = subprocess.check_output(['xcrun', '--show-sdk-path'], text=True).strip()
    cc = subprocess.check_output(['xcrun', '--find', 'clang'], text=True).strip()
    cxx = subprocess.check_output(['xcrun', '--find', 'clang++'], text=True).strip()
    common = ['-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_INSTALL_PREFIX=' + str(prefix),
        '-DCMAKE_OSX_DEPLOYMENT_TARGET=' + minimum, '-DCMAKE_OSX_ARCHITECTURES=' + args.architecture,
        '-DCMAKE_OSX_SYSROOT=' + sdk, '-DCMAKE_C_COMPILER=' + cc, '-DCMAKE_CXX_COMPILER=' + cxx,
        '-DCMAKE_PREFIX_PATH=' + str(prefix), '-DCMAKE_IGNORE_PREFIX_PATH=/opt/homebrew;/usr/local',
        '-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF', '-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF',
        '-DCMAKE_INSTALL_LIBDIR=lib', '-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-headerpad_max_install_names',
        '-DCMAKE_EXE_LINKER_FLAGS=-Wl,-headerpad_max_install_names', '-DBUILD_TESTING=OFF', '-DBUILD_SHARED_LIBS=ON']
    options = {
        'libomp': ['-DLLVM_ENABLE_RUNTIMES=openmp', '-DLLVM_INCLUDE_TESTS=OFF', '-DOPENMP_INCLUDE_TESTS=OFF',
                   '-DLIBOMP_INCLUDE_TESTS=OFF', '-DOPENMP_ENABLE_LIBOMPTARGET=OFF',
                   '-DOPENMP_ENABLE_OMPT_TOOLS=OFF', '-DLIBOMP_INSTALL_ALIASES=OFF'],
        'libaec': [],
        'zstd': ['-DZSTD_BUILD_PROGRAMS=OFF', '-DZSTD_BUILD_TESTS=OFF', '-DZSTD_BUILD_STATIC=OFF'],
        'hdf5': ['-DHDF5_BUILD_FORTRAN=OFF', '-DHDF5_BUILD_CPP_LIB=OFF', '-DHDF5_BUILD_TOOLS=OFF',
                 '-DHDF5_ENABLE_NONSTANDARD_FEATURE_FLOAT16=OFF', '-DHDF5_ENABLE_SZIP_SUPPORT=ON',
                 '-DHDF5_ENABLE_ZLIB_SUPPORT=ON', '-DHDF5_USE_GNU_DIRS=ON',
                 '-DHDF5_INSTALL_CMAKE_DIR=lib/cmake/hdf5', '-DHDF5_USE_LIBAEC=ON'],
        'netcdf': ['-DNETCDF_ENABLE_TESTS=OFF', '-DENABLE_TESTS=OFF', '-DNETCDF_ENABLE_NETCDF_4=ON',
                   '-DNETCDF_ENABLE_DAP=ON', '-DNETCDF_ENABLE_NCZARR=ON'],
        'zfp': ['-DZFP_WITH_OPENMP=OFF', '-DZFP_WITH_CUDA=OFF', '-DBUILD_UTILITIES=OFF', '-DBUILD_EXAMPLES=OFF'],
    }
    receipts = {}
    for name in ('libomp', 'libaec', 'zstd', 'hdf5', 'netcdf', 'zfp'):
        source = spec['components'][name]
        archive = work / 'downloads' / source['url'].rsplit('/', 1)[-1]
        if not archive.exists():
            temporary = archive.with_name(archive.name + '.partial')
            subprocess.run(['curl', '--fail', '--location', '--retry', '2', '--connect-timeout', '15',
                            '--output', str(temporary), source['url']], check=True)
            if digest(temporary) != source['sha256']:
                raise RuntimeError(name + ': source checksum mismatch')
            temporary.replace(archive)
        if digest(archive) != source['sha256']:
            raise RuntimeError(name + ': cached source checksum mismatch')
        directory = work / 'sources' / name
        extracted = directory / '.extracted.json'
        if not extracted.exists():
            directory.mkdir(exist_ok=True)
            with tarfile.open(archive) as tar:
                members = tar.getmembers()
                if name == 'libomp':
                    members = [m for m in members if len(Path(m.name).parts) < 2 or
                        Path(m.name).parts[1] in ('openmp', 'cmake', 'runtimes', 'third-party') or
                        Path(m.name).parts[1] == 'llvm' and
                        (len(Path(m.name).parts) < 3 or Path(m.name).parts[2] in ('cmake', 'include', 'utils'))]
                tar.extractall(directory, members=members, filter='data')
            extracted.write_text(json.dumps(source) + '\n')
        elif json.loads(extracted.read_text()) != source:
            raise RuntimeError(name + ': source extraction belongs to another archive')
        tree = next(p for p in directory.iterdir() if p.is_dir())
        src = tree / {'libomp': 'runtimes', 'zstd': 'build/cmake'}.get(name, '')
        build = work / 'build' / name
        receipt = build / 'nimbus-build.json'
        request = dict(source=source, minimum_macos=minimum, architecture=args.architecture,
                       arguments=common + options[name], compiler=subprocess.check_output([cc, '--version'], text=True))
        if not receipt.exists() or json.loads(receipt.read_text()) != request:
            with (work / 'logs' / (name + '.log')).open('w') as log:
                for command in [['cmake', '-S', str(src), '-B', str(build), *common, *options[name]],
                                ['cmake', '--build', str(build), '-j', str(args.jobs)], ['cmake', '--install', str(build)]]:
                    log.write('COMMAND ' + json.dumps(command) + '\n')
                    log.flush()
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            receipt.write_text(json.dumps(request, indent=2) + '\n')
        license_dir = prefix / 'share/nimbus-runtime-licenses' / name
        license_dir.mkdir(parents=True, exist_ok=True)
        licenses = [p for p in tree.iterdir() if p.is_file() and p.name.upper().startswith(('LICENSE', 'COPYING', 'COPYRIGHT'))]
        if name == 'libomp':
            licenses += list((tree / 'openmp').glob('LICENSE*'))
        if not licenses:
            raise RuntimeError(name + ': distribution license missing')
        for license in licenses:
            shutil.copy2(license, license_dir / license.name)
        receipts[name] = request
        print(name + ': macOS ' + minimum + ' runtime built', flush=True)
    (prefix / 'nimbus-runtime-build.json').write_text(json.dumps(receipts, indent=2) + '\n')


if __name__ == '__main__':
    main()
