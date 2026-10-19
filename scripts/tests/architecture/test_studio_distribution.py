#!/usr/bin/env python3
"""Static distribution contract for the UI-first Nimbus setup path."""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SETUP = (ROOT / "scripts/setup.sh").read_text(encoding="utf-8")
WINDOWS_SETUP = (ROOT / "scripts/setup.ps1").read_text(encoding="utf-8")
WINDOWS_SETUP_BYTES = (ROOT / "scripts/setup.ps1").read_bytes()
BUILDER = (ROOT / "ui/scripts/build_native_app.sh").read_text(encoding="utf-8")
WINDOWS_BUILDER = (ROOT / "ui/scripts/build_native_app.ps1").read_text(encoding="utf-8")
WINDOWS_ZFP = (ROOT / "scripts/windows/install_zfp.sh").read_text(encoding="utf-8")
WINDOWS_RUST_ENV = (ROOT / "scripts/windows/rust_build_environment.ps1").read_text(encoding="utf-8")
CONFIG = json.loads((ROOT / "ui/src-tauri/tauri.conf.json").read_text(encoding="utf-8"))
MACOS_CONFIG = json.loads(
    (ROOT / "ui/src-tauri/tauri.macos.conf.json").read_text(encoding="utf-8")
)
LINUX_CONFIG = json.loads(
    (ROOT / "ui/src-tauri/tauri.linux.conf.json").read_text(encoding="utf-8")
)
WINDOWS_CONFIG = json.loads(
    (ROOT / "ui/src-tauri/tauri.windows.conf.json").read_text(encoding="utf-8")
)
README = (ROOT / "README.md").read_text(encoding="utf-8")
OUTPUTS_RS = (ROOT / "ui/src-tauri/src/commands/outputs.rs").read_text(encoding="utf-8")
DEPENDENCIES_MK = (ROOT / "mk/dependencies.mk").read_text(encoding="utf-8")
CI_WORKFLOW = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
WINDOWS_HARDWARE_WORKFLOW = (ROOT / ".github/workflows/windows-hardware.yml").read_text(encoding="utf-8")
PLAYWRIGHT_CONFIG = (ROOT / "ui/playwright.config.ts").read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"studio distribution contract failed: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


BASH = shutil.which("bash")
require(BASH is not None, "Bash must be available for setup contract checks")


def run_bash(script: str, *arguments: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    # Resolve the MSYS executable explicitly and send exact LF bytes on stdin;
    # Windows command-line quoting must not reinterpret shell function bodies.
    result = subprocess.run(
        [BASH, "--noprofile", "--norc", "-s", "--", *arguments],
        input=script.encode("utf-8"), capture_output=True, env=env,
    )
    return subprocess.CompletedProcess(result.args, result.returncode,
                                       result.stdout.decode("utf-8", errors="replace"),
                                       result.stderr.decode("utf-8", errors="replace"))


node_range = ">=22.13.0 <23 || >=24"
package = json.loads((ROOT / "ui/package.json").read_text())
package_lock = json.loads((ROOT / "ui/package-lock.json").read_text())
require(package["engines"]["node"] == node_range, "Studio must declare the supported Node range")
require(
    package_lock["packages"][""]["engines"]["node"] == node_range,
    "the locked package metadata must preserve the same Node range",
)

node_minimum = re.search(r"^NODE_22_MIN=.+$", SETUP, re.MULTILINE)
require(node_minimum is not None, "setup must declare its Node 22 minimum")
node_check = node_minimum.group(0) + "\n"
for function in ("version_ge", "check_node"):
    match = re.search(rf"^{function}\(\)\n\{{.*?^\}}", SETUP, re.MULTILINE | re.DOTALL)
    require(match is not None, f"setup must retain its {function} check")
    node_check += match.group(0) + "\n"
node_check += """
node() { printf '%s\\n' "$TEST_NODE_VERSION"; }
npm() { printf '%s\\n' '10.0.0'; }
check_cmd() { return 0; }
ok() { :; }
warn() { :; }
note() { :; }
required_impact() { :; }
require_missing() { printf 'missing:%s\\n' "$1"; }
TEST_NODE_VERSION="$1"
check_node
"""
for version, accepted in (
    ("v18.20.8", False), ("v20.19.0", False), ("v22.12.0", False),
    ("v22.13.0", True), ("v22.13.1", True), ("v23.11.0", False),
    ("v24.0.0", True), ("v25.0.0", True),
):
    checked = run_bash(node_check, version)
    require(checked.returncode == 0, f"Node fixture failed for {version}: {checked.stderr}")
    require(
        ("missing:node" not in checked.stdout) == accepted,
        f"setup Node admission is incorrect for {version}",
    )
require(
    'return ($Version.Major -eq 22 -and $Version -ge [version]"22.13.0") -or $Version.Major -ge 24'
    in WINDOWS_SETUP,
    "Windows setup must enforce the same Node version boundaries",
)

for name, source, variable, default in (
    ("setup", SETUP, "jobs", "7"), ("packager", BUILDER, "BUILD_JOBS", "8"),
):
    validation = re.search(r'^if \[\[ -n "\$\{NIMBUS_BUILD_JOBS\+x\}".*?^fi', source, re.MULTILINE | re.DOTALL)
    assignment = re.search(rf'^\s*{variable}="\$\{{NIMBUS_BUILD_JOBS:.*$', source, re.MULTILINE)
    require(validation is not None and assignment is not None, f"{name} must validate and apply the job override")
    script = validation.group(0) + "\nprocessor_count() { echo 7; }\n"
    script += assignment.group(0) + f'\nprintf "%s" "${variable}"\n'
    for value in (None, "1", "2", "128", "", "0", "-1", "abc", "1.5", "02", "2\n"):
        environment = dict(os.environ)
        environment.pop("NIMBUS_BUILD_JOBS", None)
        if value is not None:
            environment["NIMBUS_BUILD_JOBS"] = value
        checked = run_bash(script, env=environment)
        accepted = value in (None, "1", "2", "128")
        require((checked.returncode == 0) == accepted, f"{name} job admission is incorrect for {value!r}")
        if accepted:
            require(checked.stdout == (default if value is None else value), f"{name} changed the requested/default worker count")
        else:
            require("positive integer" in checked.stderr, f"{name} must explain invalid worker counts")

for name, source, first_operation in (
    ("Windows setup", WINDOWS_SETUP, "function Install-WingetPackage"),
    ("Windows packager", WINDOWS_BUILDER, "$UiDir ="),
):
    require(
        r"$env:NIMBUS_BUILD_JOBS -cnotmatch '\A[1-9][0-9]*\z'" in source,
        f"{name} must reject malformed worker counts",
    )
    require(source.index("NIMBUS_BUILD_JOBS must be a positive integer") < source.index(first_operation),
            f"{name} must validate before installation or build work")
    make_lines = [line for line in source.splitlines() if "make -j" in line]
    require(bool(make_lines) and all('make -j"${NIMBUS_BUILD_JOBS:-$(nproc)}"' in line for line in make_lines),
            f"{name} must forward the override into every MSYS make invocation")

require('cmake --build "$WORK_DIR/build-nimbus" "${PARALLEL_ARGS[@]}"' in WINDOWS_ZFP,
        "Windows ZFP must pass its selected worker count explicitly to CMake")
zfp_worker_check = WINDOWS_ZFP.split("\nROOT_DIR=", 1)[0] + '\nprintf "<%s>" "${PARALLEL_ARGS[@]}"\n'
for nimbus_jobs, cmake_jobs, expected in (
    (None, None, "<--parallel>"), ("2", None, "<--parallel><2>"),
    (None, "3", "<--parallel><3>"), ("2", "3", "<--parallel><2>"),
    ("0", None, None), (None, "0", None), ("", None, None),
    ("bad", "2", None), (None, "1.5", None),
):
    environment = dict(os.environ)
    for name, value in (("NIMBUS_BUILD_JOBS", nimbus_jobs), ("CMAKE_BUILD_PARALLEL_LEVEL", cmake_jobs)):
        environment.pop(name, None)
        if value is not None:
            environment[name] = value
    checked = run_bash(zfp_worker_check, env=environment)
    require((checked.returncode == 0 and checked.stdout == expected) if expected is not None
            else (checked.returncode != 0 and "positive integer" in checked.stderr),
            f"Windows ZFP worker selection is incorrect for {nimbus_jobs!r}, {cmake_jobs!r}")


require(
    not CONFIG.get("bundle", {}).get("resources"),
    "base Tauri config must not leak an extensionless Unix simulator into Windows",
)
for platform_name, platform_config in (("macOS", MACOS_CONFIG), ("Linux", LINUX_CONFIG)):
    resources = platform_config.get("bundle", {}).get("resources", {})
    require(
        resources.get("../../bin/nimbus_sim") == "bin/nimbus_sim",
        f"the {platform_name} desktop bundle must carry the simulator built by setup",
    )

runtime_sources = json.loads((ROOT / "scripts/dependencies/macos_runtime_sources.json").read_text())
require(
    MACOS_CONFIG["bundle"]["macOS"]["minimumSystemVersion"] == runtime_sources["minimum_macos"] == "12.0",
    "the macOS bundle and pinned runtime builds must agree on macOS 12",
)
for contract in ("build_macos_runtime.py", '"$SCRIPT_DIR/macos_runtime.py" prepare',
                 '"$SCRIPT_DIR/macos_runtime.py" verify', "PORTABLE=1 DISTRIBUTION=1 REQUIRE_FULL_STACK=1"):
    require(contract in BUILDER, f"macOS distribution must enforce its runtime contract: {contract}")

icons = CONFIG.get("bundle", {}).get("icon", [])
require("icons/icon.icns" in icons, "macOS bundle must declare the Nimbus ICNS artwork")
icon = ROOT / "ui/src-tauri/icons/icon.icns"
require(icon.stat().st_size > 100_000, "the release ICNS must contain a real multi-size asset")
require(icon.read_bytes()[:4] == b"icns", "the declared release icon must be a valid ICNS container")
windows_icons = CONFIG.get("bundle", {}).get("icon", [])
require("icons/icon.ico" in windows_icons, "Windows bundle must declare the Nimbus ICO artwork")
windows_icon = ROOT / "ui/src-tauri/icons/icon.ico"
require(windows_icon.stat().st_size > 20_000, "the release ICO must contain real multi-size artwork")
require(windows_icon.read_bytes()[:4] == b"\x00\x00\x01\x00", "the release icon must be a valid ICO container")
artwork_manifest = ROOT / "ui/src-tauri/icons/nimbus-artwork.sha256"
for line in artwork_manifest.read_text(encoding="utf-8").splitlines():
    digest, name = line.split(maxsplit=1)
    artwork = artwork_manifest.parent / name.strip()
    require(artwork.is_file(), f"declared icon artwork is missing: {name}")
    require(sha256(artwork) == digest, f"icon artwork changed without an intentional manifest update: {name}")

windows_bundle = WINDOWS_CONFIG.get("bundle", {})
require(windows_bundle.get("targets") == ["nsis"], "native Windows must produce an NSIS installer")
require(
    windows_bundle.get("resources", {}).get("windows-runtime/*") == "bin/",
    "Windows installer must bundle its simulator and runtime DLLs",
)
require(
    windows_bundle.get("windows", {}).get("webviewInstallMode", {}).get("type")
    == "embedBootstrapper",
    "Windows installer must carry its WebView2 bootstrapper",
)

require('BUNDLES="${NIMBUS_STUDIO_BUNDLES:-app}"' in BUILDER, "macOS must build an app bundle")
require(
    'BUNDLES="${NIMBUS_STUDIO_BUNDLES:-appimage}"' in BUILDER,
    "Linux must build a portable AppImage rather than a macOS-only target",
)
require("xattr -cr" not in BUILDER, "installer must not recursively erase Finder icon metadata")
for evidence in (
    'cmp -s "$SOURCE_ICON" "$bundled_icon"',
    "Set :CFBundleVersion",
    '"$LSREGISTER" -f "$INSTALL_APP"',
    'Contents/Resources/bin/nimbus_sim',
    'Refusing to package a reduced simulator',
):
    require(evidence in BUILDER, f"missing installer evidence: {evidence}")

for option in ("--headless-only", "--dependencies-only"):
    require(option in SETUP, f"setup help must expose {option}")
for option in ("$HeadlessOnly", "$DependenciesOnly", "$Check", "$NoColor", "$NoArt", "$Art"):
    require(option in WINDOWS_SETUP, f"Windows setup must expose {option[1:]}")

for presentation_contract in (
    "Show-NimbusBanner",
    "Write-NimbusStormFull",
    "Write-NimbusStormCompact",
    "A modern approach to supercell modeling.",
    "[Console]::IsOutputRedirected",
    "$env:NO_COLOR",
    "Launch Nimbus Studio (default workflow)",
    "if ($Columns -lt 38)",
    "if ($Columns -ge 56)",
):
    require(
        presentation_contract in WINDOWS_SETUP,
        f"Windows setup must preserve presentation contract: {presentation_contract}",
    )

require(
    WINDOWS_SETUP_BYTES.startswith(b"\xef\xbb\xbf"),
    "Windows setup artwork must retain its UTF-8 BOM for Windows PowerShell 5.1",
)

windows_package_checks = re.search(r'\$MsysCheckPackages = @\((.*?)\n\)', WINDOWS_SETUP, re.DOTALL)
require(windows_package_checks is not None, "Windows setup must check concrete installed packages")
require("-vulkan-devel" not in windows_package_checks.group(1),
        "pacman -Q must not query the Vulkan development package group")
for package_name in ("vulkan-headers", "vulkan-loader", "spirv-tools"):
    require(f'"$MsysPackagePrefix-{package_name}"' in windows_package_checks.group(1),
            f"Windows Vulkan package checks must include {package_name}")
require("$MsysPackagePrefix-vulkan-tools" not in WINDOWS_SETUP,
        "Windows setup must not request the nonexistent MSYS2 vulkan-tools package")
require("${{ matrix.package_prefix }}-vulkan-tools" not in CI_WORKFLOW,
        "Windows CI must not request the nonexistent MSYS2 vulkan-tools package")
require('"$MsysPackagePrefix-clang"' in WINDOWS_SETUP
        and '"$MsysPackagePrefix-clang-libs"' in windows_package_checks.group(1)
        and "${{ matrix.package_prefix }}-clang" in CI_WORKFLOW,
        "fresh Windows builds must provision native clang/libclang for Rust bindgen")
for owner, content in (("setup", WINDOWS_SETUP), ("packager", WINDOWS_BUILDER), ("CI", CI_WORKFLOW)):
    require("Initialize-NimbusRustBuildEnvironment" in content,
            f"Windows {owner} must resolve the native Rust build dependencies")
for evidence in ("0xAA64", "0x8664", "x86_64-pc-windows-msvc", "aarch64-pc-windows-msvc",
                 "Assert-NimbusNativePe $CmakePath", "Assert-NimbusNativePe $LibclangLibrary",
                 "$env:LIBCLANG_PATH", "$env:CMAKE", "$env:CARGO_BUILD_TARGET"):
    require(evidence in WINDOWS_RUST_ENV, f"Windows Rust environment is missing {evidence}")
require(WINDOWS_RUST_ENV.index("Assert-NimbusNativePe $LibclangLibrary") < WINDOWS_RUST_ENV.index("$env:CMAKE ="),
        "Rust dependency guards must complete before publishing process environment")
require("SetEnvironmentVariable" not in WINDOWS_RUST_ENV and "$env:Path +=" in WINDOWS_RUST_ENV,
        "Rust dependency discovery must preserve persistent PATH and existing command precedence")
require("test_windows_rust_environment.ps1" in CI_WORKFLOW,
        "Windows CI must execute the native dependency architecture guard tests")
for native_guard in (
    "[Environment]::Is64BitProcess",
    "PROCESSOR_ARCHITECTURE",
    "core.longpaths",
    '"ARM64", "AARCH64"',
    '"clangarm64"',
    '"mingw-w64-clang-aarch64"',
    '"aarch64-pc-windows-msvc"',
):
    require(native_guard in WINDOWS_SETUP, f"Windows setup must guard native prerequisite: {native_guard}")
require(
    '-I"$(ZFP_PREFIX)/include"' in DEPENDENCIES_MK
    and '-L"$(ZFP_PREFIX)/lib"' in DEPENDENCIES_MK,
    "external Windows dependency paths must remain valid when the checkout contains spaces",
)
for shared_artwork in (
    "▄▄▄███████████████████▄▄▄▄▄▄▄▄▄",
    "▁▂▃▄▅▆▇█",
    "███╗   ██╗██╗███╗   ███╗",
):
    require(shared_artwork in SETUP, f"macOS setup is missing shared artwork: {shared_artwork}")
    require(shared_artwork in WINDOWS_SETUP, f"Windows setup is missing shared artwork: {shared_artwork}")

for capability in ("OpenMP", "Vulkan", "ZFP", "netCDF", "live shared memory"):
    require(capability in WINDOWS_SETUP, f"Windows setup must advertise required {capability} parity")

for runtime in ("nimbus_sim.exe", "ffmpeg.exe", "netCDF DLL"):
    require(runtime in WINDOWS_BUILDER, f"Windows package must require {runtime}")
require("Remove-Item -LiteralPath $Icon" not in WINDOWS_BUILDER, "Windows packaging must not delete icon artwork")
require("Copy-Item -LiteralPath $Icon" not in WINDOWS_BUILDER, "Windows packaging must not rewrite icon artwork")
require("npx tauri build --bundles nsis" in WINDOWS_BUILDER, "Windows package must use the native NSIS route")
require(
    "Refusing to package a reduced simulator" in WINDOWS_BUILDER,
    "Windows packaging must reject capability-reduced simulator binaries",
)
for architecture_contract in (
    "Get-PeMachine",
    "Assert-NativePe $PortableSimulator",
    "Assert-NativePe $StudioExecutable",
    "0x8664",
    "0xAA64",
    '"CLANGARM64"',
):
    require(
        architecture_contract in WINDOWS_BUILDER,
        f"Windows packaging must verify native architecture: {architecture_contract}",
    )

engine_build = SETUP.index('make -j"$jobs"')
dependency_install = SETUP.index("(cd ui && npm ci)")
studio_build = SETUP.index("(cd ui && npm run tauri:build)")
require(
    engine_build < dependency_install < studio_build,
    "setup must build the engine before packaging the locked desktop app",
)
require(
    'make -j"$jobs" REQUIRE_FULL_STACK=1' in SETUP,
    "macOS/Linux setup must fail closed on a reduced engine build",
)
require(
    "REQUIRE_FULL_STACK=1" in WINDOWS_SETUP,
    "Windows setup must fail closed on a reduced engine build",
)
require(
    "PORTABLE=1 DISTRIBUTION=1 REQUIRE_FULL_STACK=1" in WINDOWS_SETUP
    and '"scripts\\build_native_app.ps1" -BundleOnly' in WINDOWS_SETUP,
    "UI-first Windows setup must compile the portable simulator once and revalidate it while bundling",
)
require(
    "Launch Nimbus Studio (default workflow)" in SETUP,
    "setup handoff must lead with the desktop UI",
)
require(
    "Secondary terminal-only setup and launch" in README,
    "top-level quick start must present headless operation as secondary",
)
require(".\\scripts\\setup.ps1" in README, "top-level quick start must expose native Windows setup")
require(
    "cfg(not(unix))" not in OUTPUTS_RS,
    "Studio output/netCDF readers must not disable the native Windows adapter",
)

for hosted_gate in (
    ".\\scripts\\setup.ps1 -Check -NoArt",
    "tests.integration.test_cpu_benchmark_runner",
    "production-shell.spec.ts",
    "Silently install, launch and uninstall the produced NSIS package",
    "profile portable",
    "windows-11-arm",
    "CLANGARM64",
    "mingw-w64-clang-aarch64",
    "macos-15-intel",
    "ubuntu-24.04-arm",
    "Nimbus Studio package (macOS ${{ matrix.arch }})",
    "Nimbus Studio package (Linux ${{ matrix.arch }} AppImage)",
):
    require(hosted_gate in CI_WORKFLOW, f"hosted Windows CI is missing gate: {hosted_gate}")

for physical_gate in (
    "NIMBUS_WINDOWS_UI_PERFORMANCE",
    "chromium-windows-hardware-performance",
    '"${{ inputs.architecture }}"',
    "-clangarm64",
):
    require(
        physical_gate in WINDOWS_HARDWARE_WORKFLOW,
        f"physical Windows qualification is missing gate: {physical_gate}",
    )
require(
    "chromium-windows-hardware-performance" in PLAYWRIGHT_CONFIG
    and "supercell-responsiveness\\.spec\\.ts" in PLAYWRIGHT_CONFIG,
    "Playwright must expose a real-hardware Windows 3D responsiveness project",
)
print("studio distribution contract: PASS")
