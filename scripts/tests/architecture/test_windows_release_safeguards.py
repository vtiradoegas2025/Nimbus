#!/usr/bin/env python3
"""Static and make-evaluated contract for distributable Windows Studio builds."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OPTIONS = (ROOT / "mk/options.mk").read_text(encoding="utf-8")
INFRASTRUCTURE = (ROOT / "mk/infrastructure.mk").read_text(encoding="utf-8")
BUILDER = (ROOT / "ui/scripts/build_native_app.ps1").read_text(encoding="utf-8")
WINDOWS_CONFIG = json.loads(
    (ROOT / "ui/src-tauri/tauri.windows.conf.json").read_text(encoding="utf-8")
)
BUILDING = (ROOT / "docs/BUILDING.md").read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"Windows release safeguard failed: {message}")


def evaluate_flags(*assignments: str) -> subprocess.CompletedProcess[str]:
    makefile = "include mk/options.mk\nprint-flags:\n\t@printf '%s' '$(CXXFLAGS)'\n"
    return subprocess.run(
        ["make", "--no-print-directory", "-f", "-", *assignments, "print-flags"],
        cwd=ROOT,
        input=makefile,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def evaluate_compiler(msystem: str, override: str | None = None, *, command_line: bool = False) -> str:
    environment = dict(os.environ)
    for key in ("CXX", "MAKEFLAGS", "MFLAGS", "MAKEOVERRIDES"):
        environment.pop(key, None)
    assignments = ["UNAME_S=MINGW64_NT", f"MSYSTEM={msystem}", "COMPILER_CACHE=none"]
    if override is not None:
        if command_line:
            assignments.append(f"CXX={override}")
        else:
            environment["CXX"] = override
    result = subprocess.run(
        ["make", "--no-print-directory", "-f", "-", *assignments, "print-compiler"],
        cwd=ROOT, env=environment, text=True, capture_output=True,
        input="include mk/options.mk\nprint-compiler:\n\t@printf '%s' '$(CXX)'\n",
    )
    require(result.returncode == 0, f"compiler selection failed: {result.stderr}")
    return result.stdout


for system, compiler in (("CLANGARM64", "/clangarm64/bin/clang++"), ("UCRT64", "/ucrt64/bin/g++")):
    require(evaluate_compiler(system) == compiler,
            f"{system} default must select its compiler even when another g++ exists on PATH")
    for command_line in (False, True):
        require(evaluate_compiler(system, "custom-cxx", command_line=command_line) == "custom-cxx",
                "explicit compiler selection must remain available")

if os.name == "nt" and os.environ.get("MSYSTEM") in ("CLANGARM64", "UCRT64"):
    system = os.environ["MSYSTEM"]
    compiler = evaluate_compiler(system)
    native_path = subprocess.check_output(["cygpath", "-w", compiler], text=True).strip()
    target = subprocess.check_output([native_path, "-dumpmachine"], text=True).strip()
    expected = "aarch64" if system == "CLANGARM64" else "x86_64"
    require(target.startswith(expected + "-"), f"{system} compiler targets {target}, expected {expected}")


native = evaluate_flags("PORTABLE=0", "DISTRIBUTION=0", "COMPILER_CACHE=none")
require(native.returncode == 0, f"local native profile did not evaluate: {native.stderr}")
require("-march=native" in native.stdout, "local source builds must retain host-native tuning")
require("NIMBUS_PORTABLE_BUILD" not in native.stdout, "native profile must not claim portability")

portable = evaluate_flags("PORTABLE=1", "DISTRIBUTION=1", "COMPILER_CACHE=none")
require(portable.returncode == 0, f"distribution profile did not evaluate: {portable.stderr}")
require("-march=native" not in portable.stdout, "distribution profile must not encode the build host ISA")
require("-mtune=native" not in portable.stdout, "distribution profile must not tune for the build host")
require("-DNIMBUS_PORTABLE_BUILD" in portable.stdout, "portable binary must carry runtime profile evidence")

unsafe = evaluate_flags("PORTABLE=0", "DISTRIBUTION=1", "COMPILER_CACHE=none")
require(unsafe.returncode != 0, "DISTRIBUTION=1 must reject PORTABLE=0")
require(
    "DISTRIBUTION=1 requires PORTABLE=1" in unsafe.stderr,
    "unsafe distribution rejection must explain the repair",
)

overridden = evaluate_flags(
    "PORTABLE=1",
    "DISTRIBUTION=1",
    "COMPILER_CACHE=none",
    "CXXFLAGS=-O3 -march=native",
)
require(overridden.returncode != 0, "distribution must reject command-line host targeting")
require(
    "distribution CXXFLAGS must not contain -march or -mtune" in overridden.stderr,
    "host-targeting rejection must identify the unsafe flags",
)

for contract in (
    "PORTABLE=1 DISTRIBUTION=1 REQUIRE_FULL_STACK=1",
    "bin\\nimbus_sim_portable.exe",
    '"profile portable"',
    "if (-not $BundleOnly)",
    '"-BundleOnly requires setup to prebuild PORTABLE=1 DISTRIBUTION=1."',
    "NIMBUS_WINDOWS_SIGNING_REQUIRED",
    "NIMBUS_WINDOWS_CERTIFICATE_THUMBPRINT",
    "NIMBUS_WINDOWS_TIMESTAMP_URL",
    "Resolve-SignTool",
    "Invoke-AuthenticodeSign",
    "Assert-AuthenticodeSignature $StagedSimulator",
    "Assert-AuthenticodeSignature $StudioExecutable",
    "Assert-AuthenticodeSignature $Installer.FullName",
    "Refusing to clear unexpected NSIS output path",
    "Get-PeMachine",
    "Assert-NativePe $PortableSimulator",
    "Assert-NativePe $StudioExecutable",
    "0x8664",
    "0xAA64",
    '"windows-prefix-arm64"',
):
    require(contract in BUILDER, f"packager is missing contract: {contract}")

require(
    "Copy-Item -LiteralPath $PortableSimulator" in BUILDER,
    "NSIS staging must copy the portable artifact",
)
require(
    BUILDER.index("if (-not $BundleOnly)")
    < BUILDER.index("if (-not (Test-Path -LiteralPath $PortableSimulator -PathType Leaf))"),
    "BundleOnly may skip compilation but must retain the portable-artifact check",
)
require(
    "Copy-Item -LiteralPath $Simulator" not in BUILDER,
    "NSIS staging must not fall back to the native artifact",
)
require(
    "Signing is required, but NIMBUS_WINDOWS_CERTIFICATE_THUMBPRINT is not set" in BUILDER,
    "required signing must fail before bundling when certificate selection is absent",
)

windows = WINDOWS_CONFIG.get("bundle", {}).get("windows", {})
require(windows.get("digestAlgorithm") == "sha256", "Tauri Windows signing must use SHA-256")
require(windows.get("allowDowngrades") is False, "NSIS must reject installer downgrades")
require(windows.get("certificateThumbprint") is None, "the repository must not pin a certificate")
require(windows.get("timestampUrl") is None, "the repository must not pin a signing service")

for documentation in (
    "DISTRIBUTION=1",
    "NIMBUS_WINDOWS_SIGNING_REQUIRED",
    "NIMBUS_WINDOWS_CERTIFICATE_THUMBPRINT",
    "Get-AuthenticodeSignature",
    "No certificate or password belongs in the repository",
):
    require(documentation in BUILDING, f"release documentation is missing: {documentation}")

require(
    "DISTRIBUTION ?= 0" in OPTIONS,
    "distribution mode must remain an explicit opt-in distinct from local builds",
)
require(
    "--setting DISTRIBUTION=$(DISTRIBUTION)" in INFRASTRUCTURE,
    "build fingerprints must record distribution intent",
)

print("Windows release safeguards: PASS")
