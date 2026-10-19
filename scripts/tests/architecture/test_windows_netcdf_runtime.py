#!/usr/bin/env python3
import importlib.util
import hashlib
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("netcdf_runtime", ROOT / "scripts/windows/ensure_netcdf_runtime.py")
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)


def fixture(machine, imported=("H5open", "H5Pset_fapl_ros3"), exported=("H5open",)):
    data = bytearray(0x2200)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<HH", data, 0x84, machine, 1)
    struct.pack_into("<H", data, 0x94, 240)
    optional = 0x98
    struct.pack_into("<H", data, optional, 0x20B)
    struct.pack_into("<I", data, optional + 108, 16)
    struct.pack_into("<II", data, optional + 112, 0x1500, 40)
    struct.pack_into("<II", data, optional + 120, 0x1000, 40)
    struct.pack_into("<IIII", data, optional + 240 + 8, 0x2000, 0x1000, 0x2000, 0x200)
    address = lambda rva: rva - 0x1000 + 0x200
    struct.pack_into("<IIIII", data, address(0x1000), 0x1100, 0, 0, 0x1200, 0x1100)
    name = b"libhdf5-320.dll\0"
    data[address(0x1200):address(0x1200) + len(name)] = name
    for index, symbol in enumerate(imported):
        rva = 0x1300 + index * 64
        struct.pack_into("<Q", data, address(0x1100) + index * 8, rva)
        encoded = symbol.encode() + b"\0"
        data[address(rva) + 2:address(rva) + 2 + len(encoded)] = encoded
    struct.pack_into("<I", data, address(0x1500) + 24, len(exported))
    struct.pack_into("<I", data, address(0x1500) + 32, 0x1600)
    for index, symbol in enumerate(exported):
        rva = 0x1700 + index * 64
        struct.pack_into("<I", data, address(0x1600) + index * 4, rva)
        encoded = symbol.encode() + b"\0"
        data[address(rva):address(rva) + len(encoded)] = encoded
    return bytes(data)


class WindowsNetcdfRuntime(unittest.TestCase):
    def test_native_command_output_preserves_unicode_paths(self):
        value = "C:/Météo/云"
        command = f"import sys; sys.stdout.buffer.write({value!r}.encode('utf-8'))"
        self.assertEqual(RUNTIME.output(sys.executable, "-c", command), value)

    def test_native_architectures_expose_the_missing_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provider.dll"
            for machine in [0x8664, 0xAA64]:
                path.write_bytes(fixture(machine))
                imports, exports = RUNTIME.pe_symbols(path, machine)
                self.assertEqual(imports, {"libhdf5-320.dll": {"H5open", "H5Pset_fapl_ros3"}})
                self.assertEqual(imports["libhdf5-320.dll"] - exports, {"H5Pset_fapl_ros3"})
                with self.assertRaisesRegex(ValueError, "architecture"):
                    RUNTIME.pe_symbols(path, 0xAA64 if machine == 0x8664 else 0x8664)
                path.write_bytes(fixture(machine, exported=("H5open", "H5Pset_fapl_ros3")))
                imports, exports = RUNTIME.pe_symbols(path, machine)
                self.assertFalse(imports["libhdf5-320.dll"] - exports)

    def test_malformed_pe_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.dll"
            for data in [b"", b"MZ", fixture(0x8664)[:0x200]]:
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    RUNTIME.pe_symbols(path, 0x8664)

    def test_repair_is_confined_to_the_reviewed_pair_and_symbol(self):
        self.assertTrue(RUNTIME.repair_allowed("4.10.1-2", "2.2.0-4", ["H5Pset_fapl_ros3"]))
        for versions, symbols in [(("4.10.1-3", "2.2.0-4"), ["H5Pset_fapl_ros3"]),
                                  (("4.10.1-2", "2.2.1-1"), ["H5Pset_fapl_ros3"]),
                                  (("4.10.1-2", "2.2.0-4"), ["H5open"]),
                                  (("4.10.1-2", "2.2.0-4"), ["H5Pset_fapl_ros3", "H5open"]),
                                  (("4.10.1-2", "2.2.0-4"), [])]:
            self.assertFalse(RUNTIME.repair_allowed(*versions, symbols))

    def test_local_package_signatures_remain_required_and_trusted(self):
        original = "[options]\nRootDir = /\nSigLevel = Required DatabaseOptional\nLocalFileSigLevel = Never\n[core]\nInclude = /etc/pacman.d/mirrorlist\n"
        selected = RUNTIME.required_signature_config(original)
        self.assertNotIn("LocalFileSigLevel = Never", selected)
        self.assertIn("LocalFileSigLevel = Required TrustedOnly\n\n[core]", selected)
        self.assertIn("SigLevel = Required DatabaseOptional", selected)
        self.assertIn("Include = /etc/pacman.d/mirrorlist", selected)
        with self.assertRaises(ValueError):
            RUNTIME.required_signature_config("[core]\nServer = x\n")
        with self.assertRaises(ValueError):
            RUNTIME.required_signature_config("[options]\n[options]\n")

    def test_cached_package_tampering_cannot_reach_pacman(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            profile = RUNTIME.PROFILES["UCRT64"]
            (cache / f"{profile[1]}-hdf5-{RUNTIME.COMPATIBLE_HDF5}-any.pkg.tar.zst").write_bytes(b"wrong")
            state = {"netcdf_version": "4.10.1-2", "hdf5_version": "2.2.0-4", "missing": ["H5Pset_fapl_ros3"]}
            with patch.object(RUNTIME.subprocess, "run") as command:
                with self.assertRaisesRegex(ValueError, "digest mismatch"):
                    RUNTIME.provision(profile, state, cache)
                command.assert_not_called()

    def test_reviewed_install_enforces_signatures_without_bypassing_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            payload = b"deterministic signed-package fixture"
            package = "mingw-w64-ucrt-x86_64"
            profile = ("ucrt64", package, 0x8664, hashlib.sha256(payload).hexdigest())
            archive = cache / f"{package}-hdf5-{RUNTIME.COMPATIBLE_HDF5}-any.pkg.tar.zst"
            archive.write_bytes(payload)
            config = cache / "original.conf"
            original = "# Météo / 云\n[options]\nLocalFileSigLevel = Never\n[repo]\nServer = trusted-mirror\n"
            config.write_text(original, encoding="utf-8")
            commands = []

            def execute(command, **_):
                commands.append(command)
                if command[0] == "curl":
                    self.assertTrue(command[-1].endswith(".pkg.tar.zst.sig"))
                    Path(command[command.index("--output") + 1]).write_bytes(b"signature fixture")
                elif command[0] == "pacman":
                    selected = Path(command[command.index("--config") + 1]).read_text(encoding="utf-8")
                    self.assertIn("LocalFileSigLevel = Required TrustedOnly", selected)
                    self.assertNotIn("LocalFileSigLevel = Never", selected)
                    self.assertEqual(command[3:], ["-U", "--noconfirm", str(archive)])
                else:
                    self.fail(f"Unexpected installer command: {command}")

            def convert(*command):
                return str(config) if command == ("cygpath", "-w", "/etc/pacman.conf") else command[-1]

            state = {"netcdf_version": "4.10.1-2", "hdf5_version": "2.2.0-4", "missing": ["H5Pset_fapl_ros3"]}
            with patch.object(RUNTIME.subprocess, "run", side_effect=execute), \
                 patch.object(RUNTIME, "output", side_effect=convert), \
                 patch.object(RUNTIME, "package_version", return_value=RUNTIME.COMPATIBLE_HDF5):
                receipt = RUNTIME.provision(profile, state, cache)
            self.assertEqual([command[0] for command in commands], ["curl", "pacman"])
            self.assertEqual(config.read_text(encoding="utf-8"), original)
            self.assertEqual(receipt["from_hdf5"], "2.2.0-4")
            self.assertEqual(receipt["to_hdf5"], "2.2.0-2")
            self.assertEqual(receipt["signature_policy"], "Required TrustedOnly")

    def test_unknown_pair_never_reaches_package_installation(self):
        state = {"netcdf_version": "4.10.2-1", "hdf5_version": "2.2.0-4", "missing": ["H5Pset_fapl_ros3"]}
        with patch.object(RUNTIME.subprocess, "run") as command:
            with self.assertRaisesRegex(ValueError, "Unreviewed"):
                RUNTIME.provision(RUNTIME.PROFILES["UCRT64"], state, Path("unused"))
            command.assert_not_called()

    def test_check_only_never_provisions_a_broken_pair(self):
        state = {"netcdf_version": "4.10.1-2", "hdf5_version": "2.2.0-4", "missing": ["H5Pset_fapl_ros3"]}
        with patch.object(RUNTIME, "os", types.SimpleNamespace(name="nt", environ={"MSYSTEM": "UCRT64"})), \
             patch.object(RUNTIME, "output", return_value="unused"), \
             patch.object(RUNTIME, "runtime_state", return_value=state), \
             patch.object(RUNTIME, "provision") as install, patch("sys.argv", ["ensure", "--check"]):
            with self.assertRaisesRegex(ValueError, "imports do not close"):
                RUNTIME.main()
            install.assert_not_called()

    def test_setup_and_direct_packaging_validate_before_make(self):
        setup = (ROOT / "scripts/setup.ps1").read_text(encoding="utf-8")
        package = (ROOT / "ui/scripts/build_native_app.ps1").read_text(encoding="utf-8")
        self.assertIn("ensure_netcdf_runtime.py --check", setup)
        self.assertLess(setup.index("python3 scripts/windows/ensure_netcdf_runtime.py'"), setup.index('make -j"${NIMBUS_BUILD_JOBS'))
        self.assertLess(package.index("python3 scripts/windows/ensure_netcdf_runtime.py'"), package.index('make -j"${NIMBUS_BUILD_JOBS'))
        build = (ROOT / "mk/build.mk").read_text(encoding="utf-8")
        self.assertIn("ifeq ($(WINDOWS_HOST),1)\nifneq ($(strip $(NETCDF_LDLIBS)),)", build)
        self.assertIn("$(BIN) $(FIELD_VALIDATOR): | check-windows-netcdf-runtime", build)
        self.assertIn("python3 scripts/windows/ensure_netcdf_runtime.py --check", build)


if __name__ == "__main__":
    unittest.main()
