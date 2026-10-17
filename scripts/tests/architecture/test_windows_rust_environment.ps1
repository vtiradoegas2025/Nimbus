$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
. (Join-Path $Root "scripts\windows\rust_build_environment.ps1")
$FixtureRoot = Join-Path ([IO.Path]::GetTempPath()) ("nimbus-pe-guards-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $FixtureRoot | Out-Null

function Write-PeFixture([string]$Path, [int]$Machine) {
    $Stream = [IO.File]::Create($Path)
    try {
        $Stream.SetLength(128)
        $Writer = [IO.BinaryWriter]::new($Stream)
        $Writer.Write([uint16]0x5A4D)
        $Stream.Position = 0x3C
        $Writer.Write([uint32]64)
        $Stream.Position = 64
        $Writer.Write([uint32]0x00004550)
        $Writer.Write([uint16]$Machine)
    } finally { $Stream.Dispose() }
}

function Require-Rejected([string]$Path, [int]$Machine) {
    $Rejected = $false
    try { Assert-NimbusNativePe $Path $Machine } catch { $Rejected = $true }
    if (-not $Rejected) { throw "Native dependency guard accepted invalid input: $Path" }
}

try {
    $X64 = Join-Path $FixtureRoot "x64.dll"
    $Arm64 = Join-Path $FixtureRoot "arm64.dll"
    Write-PeFixture $X64 0x8664
    Write-PeFixture $Arm64 0xAA64
    Assert-NimbusNativePe $X64 0x8664
    Assert-NimbusNativePe $Arm64 0xAA64
    Require-Rejected $X64 0xAA64
    Require-Rejected $Arm64 0x8664
    Require-Rejected (Join-Path $FixtureRoot "missing.dll") 0x8664
    $Invalid = Join-Path $FixtureRoot "invalid.dll"
    [IO.File]::WriteAllBytes($Invalid, [byte[]](1, 2, 3, 4))
    Require-Rejected $Invalid 0x8664
    $Truncated = [IO.File]::OpenWrite($X64)
    $Truncated.SetLength(65)
    $Truncated.Dispose()
    Require-Rejected $X64 0x8664
    Write-Output "Windows Rust dependency PE guards: 7 checks passed; fixtures were never executed."
} finally {
    Remove-Item -LiteralPath $FixtureRoot -Recurse -Force
}
