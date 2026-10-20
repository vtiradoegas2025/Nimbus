#!/usr/bin/env python3
"""Validate the native MSYS netCDF/HDF5 ABI and repair the one reviewed package mismatch."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile

PROFILES = {
    "UCRT64": ("ucrt64", "mingw-w64-ucrt-x86_64", 0x8664,
               "ec34643003bdc6ae7e8416ab5549afb90c960bdc601fab7fdcfec1ee04f8e8ba"),
    "CLANGARM64": ("clangarm64", "mingw-w64-clang-aarch64", 0xAA64,
                   "372c366d0889875e2e60927a67330d6893f071664a52ade461971ffb673c727e"),
}
COMPATIBLE_HDF5 = "2.2.0-2"


def pe_symbols(path, machine):
    data = Path(path).read_bytes()

    def unpack(fmt, offset):
        size = struct.calcsize(fmt)
        if offset < 0 or offset + size > len(data):
            raise ValueError(f"Truncated PE table: {path}")
        return struct.unpack_from(fmt, data, offset)

    if data[:2] != b"MZ":
        raise ValueError(f"Not a native PE binary: {path}")
    pe = unpack("<I", 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0" or unpack("<H", pe + 4)[0] != machine:
        raise ValueError(f"Wrong native PE architecture: {path}")
    optional = pe + 24
    if unpack("<H", optional)[0] != 0x20B or unpack("<I", optional + 108)[0] < 2:
        raise ValueError(f"Expected a PE32+ import/export directory: {path}")
    sections = optional + unpack("<H", pe + 20)[0]
    ranges = [unpack("<IIII", sections + i * 40 + 8)
              for i in range(unpack("<H", pe + 6)[0])]

    def offset(rva, size=1):
        for virtual_size, address, raw_size, raw in ranges:
            if address <= rva < address + max(virtual_size, raw_size):
                relative = rva - address
                if relative + size <= raw_size and raw + relative + size <= len(data):
                    return raw + relative
        raise ValueError(f"Unbacked PE address in {path}: {rva:#x}")

    def string(rva):
        start = offset(rva)
        end = data.find(b"\0", start, min(start + 4096, len(data)))
        if end < 0:
            raise ValueError(f"Unterminated PE symbol: {path}")
        return data[start:end].decode("ascii")

    imports = {}
    import_rva, import_size = unpack("<II", optional + 120)
    if import_rva:
        for i in range(import_size // 20):
            descriptor = unpack("<IIIII", offset(import_rva + i * 20, 20))
            if not any(descriptor):
                break
            lookup, _, _, name, first = descriptor
            symbols = set()
            for index in range(len(data) // 8):
                entry = unpack("<Q", offset((lookup or first) + index * 8, 8))[0]
                if entry == 0:
                    break
                symbols.add(f"#{entry & 0xFFFF}" if entry >> 63 else string(entry + 2))
            else:
                raise ValueError(f"Unterminated PE import table: {path}")
            imports[string(name).lower()] = symbols
        else:
            raise ValueError(f"Unterminated PE import directory: {path}")
    exports = set()
    export_rva, _ = unpack("<II", optional + 112)
    if export_rva:
        header = offset(export_rva, 40)
        count, names = unpack("<I", header + 24)[0], unpack("<I", header + 32)[0]
        if count > len(data) // 4:
            raise ValueError(f"Invalid PE export count: {path}")
        exports = {string(unpack("<I", offset(names + i * 4, 4))[0]) for i in range(count)}
    return imports, exports


def repair_allowed(netcdf_version, hdf5_version, missing):
    # The newer HDF5 package removed ROS3 exports without rebuilding this netCDF DLL.
    return (netcdf_version, hdf5_version, set(missing)) == (
        "4.10.1-2", "2.2.0-4", {"H5Pset_fapl_ros3"})


def required_signature_config(text):
    pattern = re.compile(r"(^\[options\][^\n]*\n)(.*?)(?=^\[|\Z)", re.M | re.S)
    if len(list(pattern.finditer(text))) != 1:
        raise ValueError("Cannot safely identify pacman's single [options] section")

    def replace(match):
        body = re.sub(r"^\s*LocalFileSigLevel\s*=.*$", "", match[2], flags=re.M)
        return match[1] + body.rstrip() + "\nLocalFileSigLevel = Required TrustedOnly\n\n"
    return pattern.sub(replace, text)


def output(*command):
    return subprocess.check_output(command, text=True, encoding="utf-8",
                                   env={**os.environ, "LC_ALL": "C.UTF-8"}).strip()


def package_version(package):
    fields = output("pacman", "-Q", package).split()
    if len(fields) != 2 or fields[0] != package:
        raise ValueError(f"Unexpected package identity for {package}")
    return fields[1]


def runtime_state(directory, package, machine):
    netcdf = directory / "libnetcdf.dll"
    imports, _ = pe_symbols(netcdf, machine)
    providers = [name for name in imports if re.fullmatch(r"libhdf5-[0-9]+\.dll", name)]
    if len(providers) != 1:
        raise ValueError("Expected one native HDF5 provider in the netCDF import table")
    provider = directory / providers[0]
    _, exports = pe_symbols(provider, machine)
    return {"netcdf": str(netcdf), "hdf5": str(provider),
            "netcdf_version": package_version(package + "-netcdf"),
            "hdf5_version": package_version(package + "-hdf5"),
            "missing": sorted(imports[providers[0]] - exports)}


def provision(profile, state, cache):
    prefix, package, _, expected = profile
    if not repair_allowed(state["netcdf_version"], state["hdf5_version"], state["missing"]):
        raise ValueError(f"Unreviewed netCDF/HDF5 ABI mismatch; no packages changed: {state}")
    filename = f"{package}-hdf5-{COMPATIBLE_HDF5}-any.pkg.tar.zst"
    url = f"https://repo.msys2.org/mingw/{prefix}/{filename}"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / filename
    if not archive.exists():
        partial = archive.with_name(filename + ".partial")
        subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--output", str(partial), url], check=True)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != expected:
            raise ValueError("Compatible HDF5 package digest mismatch; nothing installed")
        partial.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise ValueError("Cached compatible HDF5 package digest mismatch; nothing installed")
    signature = archive.with_name(filename + ".sig")
    subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--output", str(signature), url + ".sig"], check=True)
    original = Path(output("cygpath", "-w", "/etc/pacman.conf")).read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="nimbus-netcdf-signature-") as temporary:
        config = Path(temporary) / "pacman.conf"
        config.write_text(required_signature_config(original), encoding="utf-8")
        subprocess.run(["pacman", "--config", output("cygpath", "-u", str(config)),
                        "-U", "--noconfirm", output("cygpath", "-u", str(archive))], check=True)
    if package_version(package + "-hdf5") != COMPATIBLE_HDF5:
        raise ValueError("The compatible HDF5 package was not installed")
    return {"from_hdf5": state["hdf5_version"], "to_hdf5": COMPATIBLE_HDF5,
            "netcdf": state["netcdf_version"], "missing_imports": state["missing"],
            "package_url": url, "sha256": expected, "signature_policy": "Required TrustedOnly"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate only; never download or modify packages")
    args = parser.parse_args()
    profile = PROFILES.get(os.environ.get("MSYSTEM"))
    if os.name != "nt" or profile is None:
        raise ValueError("Run this check in native MSYS2 UCRT64 or CLANGARM64")
    prefix, package, machine, _ = profile
    directory = Path(output("cygpath", "-w", f"/{prefix}/bin"))
    state = runtime_state(directory, package, machine)
    repair = None
    if state["missing"]:
        if args.check:
            raise ValueError(f"netCDF/HDF5 DLL imports do not close; run scripts/setup.ps1 to provision the reviewed pair: {state}")
        repair = provision(profile, state, Path(__file__).resolve().parents[2] / "build/windows-deps/netcdf-runtime")
        state = runtime_state(directory, package, machine)
        if state["missing"]:
            raise ValueError(f"netCDF/HDF5 ABI remains incompatible after provisioning: {state}")
    library = ctypes.CDLL(state["netcdf"], winmode=0)
    version = library.nc_inq_libvers
    version.restype = ctypes.c_char_p
    state["runtime_version"] = version().decode("utf-8", errors="replace")
    state["repair"] = repair
    print(json.dumps(state, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Windows netCDF runtime check failed: {error}")
