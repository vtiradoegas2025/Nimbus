#!/usr/bin/env python3
"""Fail-closed source contract for native Windows x64/ARM64 parity."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SETUP = (ROOT / "scripts/setup.ps1").read_text(encoding="utf-8-sig")
BUILDER = (ROOT / "ui/scripts/build_native_app.ps1").read_text(encoding="utf-8")
CI = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
PHYSICAL = (ROOT / ".github/workflows/windows-hardware.yml").read_text(encoding="utf-8")
SUPPORTED = (ROOT / "docs/SUPPORTED_PLATFORMS.md").read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"Windows architecture matrix failed: {message}")


def require_all(source: str, values: tuple[str, ...], label: str) -> None:
    for value in values:
        require(value in source, f"{label} is missing {value!r}")


require_all(
    SETUP,
    (
        '"AMD64", "X86_64"',
        '"ARM64", "AARCH64"',
        '"ucrt64"',
        '"clangarm64"',
        '"mingw-w64-ucrt-x86_64"',
        '"mingw-w64-clang-aarch64"',
        '"x86_64-pc-windows-msvc"',
        '"aarch64-pc-windows-msvc"',
        '"windows-prefix-arm64"',
        "Version.Build -lt 22000",
    ),
    "setup",
)
require("ARM64 is not yet a qualified target" not in SETUP, "stale ARM64 exclusion remains")

require_all(
    BUILDER,
    (
        '"UCRT64"',
        '"CLANGARM64"',
        "0x8664",
        "0xAA64",
        "Get-PeMachine",
        "Assert-NativePe $PortableSimulator",
        "Assert-NativePe $StudioExecutable",
        "(ucrt64|clangarm64)",
    ),
    "packager",
)

require_all(CI, ("UCRT64", "CLANGARM64"), "hosted CI")
require_all(PHYSICAL, ("-ucrt64", "-clangarm64"), "physical CI")

require_all(
    CI,
    (
        "macos-15",
        "macos-15-intel",
        "ubuntu-24.04-arm",
        "windows-latest",
        "windows-11-arm",
        "mingw-w64-ucrt-x86_64",
        "mingw-w64-clang-aarch64",
        "nimbus-studio-windows-${{ matrix.arch }}-nsis",
    ),
    "hosted CI",
)
require_all(
    SUPPORTED,
    ("Apple Silicon", "Intel", "Windows 11 ARM64", "Linux", "x64", "ARM64", "does not target 32-bit"),
    "supported-platform matrix",
)

print("Windows native x64/ARM64 architecture matrix: PASS")
