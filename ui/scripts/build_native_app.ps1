[CmdletBinding()]
param(
    [switch]$BundleOnly,
    [switch]$PrintPlan
)

$ErrorActionPreference = "Stop"
if ($null -ne $env:NIMBUS_BUILD_JOBS -and $env:NIMBUS_BUILD_JOBS -cnotmatch '\A[1-9][0-9]*\z')
{
    throw "NIMBUS_BUILD_JOBS must be a positive integer."
}

$UiDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RootDir = (Resolve-Path (Join-Path $UiDir "..")).Path
. (Join-Path $RootDir "scripts\windows\invoke_bash.ps1")
$PortableSimulator = Join-Path $RootDir "bin\nimbus_sim_portable.exe"
$RuntimeDir = Join-Path $UiDir "src-tauri\windows-runtime"
$Icon = Join-Path $UiDir "src-tauri\icons\icon.ico"

$NativeArchitecture = if ($env:PROCESSOR_ARCHITEW6432) {
    $env:PROCESSOR_ARCHITEW6432
} else {
    $env:PROCESSOR_ARCHITECTURE
}
switch ($NativeArchitecture.ToUpperInvariant()) {
    { $_ -in @("AMD64", "X86_64") } {
        $NimbusArchitecture = "x64"
        $MsysSystem = "UCRT64"
        $MsysPrefix = "ucrt64"
        $WindowsPrefixName = "windows-prefix"
        $ExpectedPeMachine = 0x8664
        $SignToolArchitecture = "x64"
        break
    }
    { $_ -in @("ARM64", "AARCH64") } {
        $NimbusArchitecture = "arm64"
        $MsysSystem = "CLANGARM64"
        $MsysPrefix = "clangarm64"
        $WindowsPrefixName = "windows-prefix-arm64"
        $ExpectedPeMachine = 0xAA64
        $SignToolArchitecture = "arm64"
        break
    }
    default {
        throw "Nimbus Studio packaging supports native Windows x64 and ARM64; detected '$NativeArchitecture'."
    }
}

function Get-PeMachine([string]$Path) {
    $Stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
    try {
        $Reader = [IO.BinaryReader]::new($Stream)
        if ($Reader.ReadUInt16() -ne 0x5A4D) { throw "Not a PE executable: $Path" }
        $Stream.Position = 0x3C
        $PeOffset = $Reader.ReadUInt32()
        $Stream.Position = $PeOffset
        if ($Reader.ReadUInt32() -ne 0x00004550) { throw "Invalid PE signature: $Path" }
        return $Reader.ReadUInt16()
    } finally {
        $Stream.Dispose()
    }
}

function Assert-NativePe([string]$Path) {
    $Machine = Get-PeMachine $Path
    if ($Machine -ne $ExpectedPeMachine) {
        throw ("Architecture mismatch for {0}: expected {1} PE machine 0x{2:X4}, found 0x{3:X4}." -f `
            $Path, $NimbusArchitecture, $ExpectedPeMachine, $Machine)
    }
}

function Read-BinarySwitch([string]$Name, [string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value) -or $Value -eq "0") { return $false }
    if ($Value -eq "1") { return $true }
    throw "$Name must be 0 or 1."
}

function Resolve-SignTool {
    if ($env:NIMBUS_WINDOWS_SIGNTOOL) {
        if (-not (Test-Path -LiteralPath $env:NIMBUS_WINDOWS_SIGNTOOL -PathType Leaf)) {
            throw "NIMBUS_WINDOWS_SIGNTOOL does not name a file: $($env:NIMBUS_WINDOWS_SIGNTOOL)"
        }
        return (Resolve-Path -LiteralPath $env:NIMBUS_WINDOWS_SIGNTOOL).Path
    }
    $Command = Get-Command "signtool.exe" -ErrorAction SilentlyContinue
    if ($Command) { return $Command.Source }

    $ProgramFilesX86 = [Environment]::GetFolderPath(
        [Environment+SpecialFolder]::ProgramFilesX86
    )
    $WindowsKitsBin = Join-Path $ProgramFilesX86 "Windows Kits\10\bin"
    if (Test-Path -LiteralPath $WindowsKitsBin -PathType Container) {
        $SignToolPattern = '[\\/]' + [Regex]::Escape($SignToolArchitecture) + '[\\/]signtool\.exe$'
        $Candidate = Get-ChildItem -LiteralPath $WindowsKitsBin -Filter "signtool.exe" -File -Recurse |
            Where-Object { $_.FullName -match $SignToolPattern } |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($Candidate) { return $Candidate.FullName }
    }
    throw "Authenticode signing was requested, but signtool.exe was not found. Set NIMBUS_WINDOWS_SIGNTOOL."
}

function Invoke-AuthenticodeSign(
    [string]$SignTool,
    [string]$Path,
    [string]$Thumbprint,
    [string]$TimestampUrl,
    [bool]$UseTsp
) {
    $Arguments = @("sign", "/sha1", $Thumbprint, "/fd", "SHA256")
    if ($UseTsp) {
        $Arguments += @("/tr", $TimestampUrl, "/td", "SHA256")
    } else {
        $Arguments += @("/t", $TimestampUrl)
    }
    $Arguments += $Path
    & $SignTool @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Authenticode signing failed for $Path" }
}

function Assert-AuthenticodeSignature([string]$Path, [string]$Thumbprint) {
    $Signature = Get-AuthenticodeSignature -LiteralPath $Path
    if ($Signature.Status -ne [System.Management.Automation.SignatureStatus]::Valid) {
        throw "Required Authenticode signature is not valid for $Path ($($Signature.Status): $($Signature.StatusMessage))"
    }
    $ActualThumbprint = $Signature.SignerCertificate.Thumbprint -replace '\s', ''
    if ($ActualThumbprint.ToUpperInvariant() -ne $Thumbprint) {
        throw "Authenticode signer mismatch for $Path."
    }
}

$SigningRequired = Read-BinarySwitch "NIMBUS_WINDOWS_SIGNING_REQUIRED" $env:NIMBUS_WINDOWS_SIGNING_REQUIRED
$SigningThumbprint = [string]$env:NIMBUS_WINDOWS_CERTIFICATE_THUMBPRINT
$SigningThumbprint = ($SigningThumbprint -replace '\s', '').ToUpperInvariant()
$SigningConfigured = -not [string]::IsNullOrWhiteSpace($SigningThumbprint)
$TimestampUrl = $env:NIMBUS_WINDOWS_TIMESTAMP_URL
$TimestampTsp = Read-BinarySwitch "NIMBUS_WINDOWS_TIMESTAMP_TSP" $env:NIMBUS_WINDOWS_TIMESTAMP_TSP
if ($SigningRequired -and -not $SigningConfigured) {
    throw "Signing is required, but NIMBUS_WINDOWS_CERTIFICATE_THUMBPRINT is not set."
}
if ($SigningConfigured) {
    if ($SigningThumbprint -notmatch '^[0-9A-F]{40}$') {
        throw "NIMBUS_WINDOWS_CERTIFICATE_THUMBPRINT must be a 40-digit SHA-1 certificate thumbprint."
    }
    [Uri]$ParsedTimestamp = $null
    if ([string]::IsNullOrWhiteSpace($TimestampUrl) -or
        -not [Uri]::TryCreate($TimestampUrl, [UriKind]::Absolute, [ref]$ParsedTimestamp) -or
        $ParsedTimestamp.Scheme -notin @("http", "https")) {
        throw "Signing requires an absolute HTTP(S) NIMBUS_WINDOWS_TIMESTAMP_URL."
    }
}

if (-not (Test-Path -LiteralPath $Icon -PathType Leaf)) {
    throw "Windows application icon is missing: $Icon"
}

$MsysRoot = if ($env:NIMBUS_MSYS2_ROOT) { $env:NIMBUS_MSYS2_ROOT } else { "C:\msys64" }
$MsysBash = Join-Path $MsysRoot "usr\bin\bash.exe"
if (-not (Test-Path -LiteralPath $MsysBash -PathType Leaf)) {
    throw "MSYS2 bash was not found at $MsysBash"
}
$env:MSYSTEM = $MsysSystem
$env:MSYS2_PATH_TYPE = "inherit"
$env:CHERE_INVOKING = "1"

if ($PrintPlan) {
    Write-Output "platform=windows"
    Write-Output "architecture=$NimbusArchitecture"
    Write-Output "msystem=$MsysSystem"
    Write-Output "bundles=nsis"
    Write-Output "simulator=$PortableSimulator"
    Write-Output "portable=yes"
    Write-Output "simulator-build=$(if ($BundleOnly) { 'skip' } else { 'rebuild' })"
    Write-Output "signing-required=$($SigningRequired.ToString().ToLowerInvariant())"
    Write-Output "signing-configured=$($SigningConfigured.ToString().ToLowerInvariant())"
    Write-Output "runtime=$RuntimeDir"
    exit 0
}

$env:NIMBUS_WINDOWS_RELEASE_ROOT = $RootDir
Invoke-NimbusBash -Bash $MsysBash -Script 'set -euo pipefail; root=$(cygpath -u "$NIMBUS_WINDOWS_RELEASE_ROOT"); cd "$root"; python3 scripts/windows/ensure_netcdf_runtime.py'

# Packaging is a distribution boundary, not a reuse of the locally tuned
# development binary. The paired make switches deliberately fail closed if a
# future edit drops PORTABLE=1, and the profile has its own object directory.
# Setup can prebuild that exact profile and pass -BundleOnly to avoid compiling
# it twice; the binary and runtime profile checks below remain mandatory.
if (-not $BundleOnly) {
    $env:NIMBUS_WINDOWS_RELEASE_ROOT = $RootDir
    $env:NIMBUS_WINDOWS_RELEASE_PREFIX = Join-Path $RootDir "build\$WindowsPrefixName"
    Invoke-NimbusBash -Bash $MsysBash -Script 'set -euo pipefail; root=$(cygpath -u "$NIMBUS_WINDOWS_RELEASE_ROOT"); prefix=$(cygpath -u "$NIMBUS_WINDOWS_RELEASE_PREFIX"); cd "$root"; make -j"${NIMBUS_BUILD_JOBS:-$(nproc)}" PORTABLE=1 DISTRIBUTION=1 REQUIRE_FULL_STACK=1 ZFP_PREFIX="$prefix" NETCDF=1'
}
if (-not (Test-Path -LiteralPath $PortableSimulator -PathType Leaf)) {
    $Hint = if ($BundleOnly) {
        "-BundleOnly requires setup to prebuild PORTABLE=1 DISTRIBUTION=1."
    } else {
        "The portable simulator build did not produce its declared output."
    }
    throw "Portable simulator is missing: $PortableSimulator. $Hint"
}
Assert-NativePe $PortableSimulator

# The directory is a build-owned staging area and is never used for user data.
if (Test-Path -LiteralPath $RuntimeDir) {
    $ExpectedParent = (Resolve-Path (Join-Path $UiDir "src-tauri")).Path
    $ActualParent = (Resolve-Path (Split-Path -Parent $RuntimeDir)).Path
    if ($ActualParent -ne $ExpectedParent -or (Split-Path -Leaf $RuntimeDir) -ne "windows-runtime") {
        throw "Refusing to clear unexpected runtime staging path: $RuntimeDir"
    }
    Remove-Item -LiteralPath $RuntimeDir -Recurse -Force
}
New-Item -ItemType Directory -Path $RuntimeDir | Out-Null
Copy-Item -LiteralPath $PortableSimulator -Destination (Join-Path $RuntimeDir "nimbus_sim.exe")
$Ffmpeg = Join-Path $MsysRoot "$MsysPrefix\bin\ffmpeg.exe"
if (-not (Test-Path -LiteralPath $Ffmpeg -PathType Leaf)) {
    throw "The full Windows product requires ffmpeg: $Ffmpeg"
}
Copy-Item -LiteralPath $Ffmpeg -Destination (Join-Path $RuntimeDir "ffmpeg.exe")

# Bundle every non-system DLL the MinGW simulator resolves, including the
# transitive netCDF/HDF5/OpenMP runtimes. Keeping them beside nimbus_sim.exe
# preserves Windows' ordinary application-local DLL search semantics.
$Dlls = @()
foreach ($Payload in @($PortableSimulator, $Ffmpeg)) {
    $env:NIMBUS_WINDOWS_PAYLOAD = $Payload
    $Resolved = Invoke-NimbusBash -Bash $MsysBash -CaptureOutput -Script 'set -o pipefail; payload=$(cygpath -u "$NIMBUS_WINDOWS_PAYLOAD"); ldd "$payload" | awk ''/=> \/(ucrt64|clangarm64)\// {print $3} /^\/(ucrt64|clangarm64)\// {print $1}'''
    $Dlls += $Resolved
}
foreach ($Dll in ($Dlls | Sort-Object -Unique)) {
    if ([string]::IsNullOrWhiteSpace($Dll)) { continue }
    $env:NIMBUS_WINDOWS_DLL = $Dll
    $WindowsDll = (Invoke-NimbusBash -Bash $MsysBash -CaptureOutput -Script 'cygpath -w "$NIMBUS_WINDOWS_DLL"').Trim()
    if (-not (Test-Path -LiteralPath $WindowsDll -PathType Leaf)) {
        throw "Could not stage Windows runtime dependency: $Dll"
    }
    Copy-Item -LiteralPath $WindowsDll -Destination $RuntimeDir
}

# The simulator's SHARPpy sounding reader intentionally loads the optional
# HDF5 high-level API at runtime, so it does not appear in nimbus_sim's static
# `ldd` graph. Stage that sibling explicitly to keep HDF5 sounding support
# identical to macOS/Linux.
$RuntimeBin = Join-Path $MsysRoot "$MsysPrefix\bin"
$Hdf5HighLevel = Get-ChildItem -LiteralPath $RuntimeBin -Filter "*hdf5*hl*.dll" -File
if (-not $Hdf5HighLevel) {
    throw "The full Windows product requires the HDF5 high-level runtime."
}
foreach ($Library in $Hdf5HighLevel) {
    Copy-Item -LiteralPath $Library.FullName -Destination $RuntimeDir
}
$NetcdfPluginDir = Join-Path $MsysRoot "$MsysPrefix\lib\netcdf"
if (Test-Path -LiteralPath $NetcdfPluginDir -PathType Container) {
    foreach ($Plugin in (Get-ChildItem -LiteralPath $NetcdfPluginDir -Filter "*.dll" -File)) {
        Copy-Item -LiteralPath $Plugin.FullName -Destination $RuntimeDir
    }
}

$StagedSimulator = Join-Path $RuntimeDir "nimbus_sim.exe"
$PriorPath = $env:Path
try {
    $env:Path = "$RuntimeDir;$env:Path"
    $Capabilities = (& $StagedSimulator --version 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) { throw "The staged simulator could not start with its bundled DLLs." }
    foreach ($Required in @(
        "profile portable",
        "OpenMP runtime: available",
        "ZFP compression: available",
        "netCDF4 history output: available",
        "Vulkan compute: available",
        "live shared memory: available"
    )) {
        if (-not $Capabilities.Contains($Required)) {
            throw "Refusing to package a reduced simulator: missing '$Required'."
        }
    }
} finally {
    $env:Path = $PriorPath
}

$NetcdfRuntime = Get-ChildItem -LiteralPath $RuntimeDir -Filter "*netcdf*.dll" -File
if (-not $NetcdfRuntime) {
    throw "The full Windows product requires libnetcdf, but no netCDF DLL was staged."
}
foreach ($NativePayload in (Get-ChildItem -LiteralPath $RuntimeDir -File | Where-Object { $_.Extension -in @(".exe", ".dll") })) {
    Assert-NativePe $NativePayload.FullName
}

$SignTool = $null
if ($SigningConfigured) {
    $SignTool = Resolve-SignTool
    Invoke-AuthenticodeSign $SignTool $StagedSimulator $SigningThumbprint $TimestampUrl $TimestampTsp
    Assert-AuthenticodeSignature $StagedSimulator $SigningThumbprint
}

. (Join-Path $RootDir "scripts\windows\rust_build_environment.ps1")
Initialize-NimbusRustBuildEnvironment -MsysRoot $MsysRoot -Architecture $NimbusArchitecture

$BundleRoot = Join-Path $UiDir "src-tauri\target\release\bundle\nsis"
if (Test-Path -LiteralPath $BundleRoot -PathType Container) {
    $ExpectedBundleParent = Join-Path $UiDir "src-tauri\target\release\bundle"
    $ActualBundleParent = (Resolve-Path (Split-Path -Parent $BundleRoot)).Path
    if ($ActualBundleParent -ne $ExpectedBundleParent -or (Split-Path -Leaf $BundleRoot) -ne "nsis") {
        throw "Refusing to clear unexpected NSIS output path: $BundleRoot"
    }
    Remove-Item -LiteralPath $BundleRoot -Recurse -Force
}

$SigningConfigPath = $null
Push-Location $UiDir
try {
    if ($SigningConfigured) {
        $SigningConfigPath = Join-Path ([IO.Path]::GetTempPath()) "nimbus-tauri-signing-$([Guid]::NewGuid().ToString('N')).json"
        $SigningOverride = @{
            bundle = @{
                windows = @{
                    certificateThumbprint = $SigningThumbprint
                    digestAlgorithm = "sha256"
                    timestampUrl = $TimestampUrl
                    tsp = $TimestampTsp
                }
            }
        } | ConvertTo-Json -Depth 5
        [IO.File]::WriteAllText(
            $SigningConfigPath,
            $SigningOverride,
            [Text.UTF8Encoding]::new($false)
        )
        & npx tauri build --bundles nsis --config $SigningConfigPath
    } else {
        Write-Output "Authenticode signing: disabled (local/development package)"
        & npx tauri build --bundles nsis
    }
    if ($LASTEXITCODE -ne 0) { throw "Tauri Windows build failed." }
} finally {
    Pop-Location
    if ($SigningConfigPath -and (Test-Path -LiteralPath $SigningConfigPath -PathType Leaf)) {
        Remove-Item -LiteralPath $SigningConfigPath -Force
    }
}

$Installer = Get-ChildItem -LiteralPath $BundleRoot -Filter "*-setup.exe" -File |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1
if (-not $Installer) {
    throw "Tauri completed without producing the expected NSIS installer."
}
$StudioExecutable = Join-Path $UiDir "src-tauri\target\release\nimbus-studio.exe"
if (-not (Test-Path -LiteralPath $StudioExecutable -PathType Leaf)) {
    throw "Tauri completed without producing the expected Studio executable."
}
Assert-NativePe $StudioExecutable
if ($SigningConfigured) {
    Assert-AuthenticodeSignature $StudioExecutable $SigningThumbprint
    Assert-AuthenticodeSignature $Installer.FullName $SigningThumbprint
    Write-Output "Authenticode signing: verified ($SigningThumbprint)"
} elseif ($SigningRequired) {
    # Defensive invariant: the required branch is rejected before the build,
    # so reaching this line would mean a future edit bypassed the guard.
    throw "Signing was required, but the produced installer is unsigned."
}
Write-Output "Nimbus Studio Windows installer: $($Installer.FullName)"
