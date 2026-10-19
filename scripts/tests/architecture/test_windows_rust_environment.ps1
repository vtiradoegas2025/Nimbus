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
    foreach ($RustHost in @("x86_64-pc-windows-msvc", "aarch64-pc-windows-msvc")) {
        $Extra = '-I"C:\SDK with spaces" -DMY_FLAG=1'
        $Expected = "$Extra --target=$RustHost"
        if ((Get-NimbusBindgenArguments $Extra $RustHost) -ne $Expected -or
            (Get-NimbusBindgenArguments $Expected $RustHost) -ne $Expected) {
            throw "Bindgen arguments lost a user flag or duplicated the native target."
        }
        foreach ($Conflict in @('--target=x86_64-w64-windows-gnu', '-target "aarch64-w64-windows-gnu"')) {
            $Rejected = $false
            try { Get-NimbusBindgenArguments $Conflict $RustHost | Out-Null } catch { $Rejected = $true }
            if (-not $Rejected) { throw "Bindgen accepted a conflicting ABI: $Conflict" }
        }
    }
    foreach ($Version in @("1.79.0", "1.90.0")) {
        Assert-NimbusRustVersion @("host: x86_64-pc-windows-msvc", "release: $Version")
    }
    foreach ($Version in @("1.77.2", "1.78.0", "unknown")) {
        $Rejected = $false
        try { Assert-NimbusRustVersion @("release: $Version") } catch { $Rejected = $true }
        if (-not $Rejected) { throw "Rust minimum guard accepted unsupported release '$Version'." }
    }
    $MsvcHeaders = Join-Path $FixtureRoot "MSVC include"
    $UcrtHeaders = Join-Path $FixtureRoot "SDK ucrt"
    New-Item -ItemType Directory -Path $MsvcHeaders, $UcrtHeaders | Out-Null
    [IO.File]::WriteAllText((Join-Path $MsvcHeaders "vcruntime.h"), "")
    foreach ($Header in @("corecrt.h", "inttypes.h")) {
        [IO.File]::WriteAllText((Join-Path $UcrtHeaders $Header), "")
    }
    if (-not (Test-NimbusMsvcHeaders "$MsvcHeaders;$UcrtHeaders;") -or
        (Test-NimbusMsvcHeaders $MsvcHeaders) -or (Test-NimbusMsvcHeaders $UcrtHeaders) -or
        (Test-NimbusMsvcHeaders "")) {
        throw "MSVC header discovery accepted an incomplete SDK or rejected complete include directories."
    }
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
    Write-Output "Windows Rust dependency PE guards: 7 checks passed; synthetic PE fixtures were never executed."

    Initialize-NimbusNativeLibrarySearch
    $PreviousDirectory = [Text.StringBuilder]::new(32768)
    $PreviousLength = [NimbusBuild.NativeLibrarySearch]::GetDllDirectory(32768, $PreviousDirectory)
    if ($PreviousLength -ge 32768) { throw "Previous DLL directory exceeds the test buffer." }
    $WitnessName = "nimbus-dll-search-$([guid]::NewGuid()).dll"
    $WitnessPath = Join-Path $FixtureRoot $WitnessName
    Copy-Item -LiteralPath (Join-Path $env:SystemRoot "System32\version.dll") -Destination $WitnessPath
    try {
        Set-NimbusNativeDllDirectory $FixtureRoot
        $Child = @'
$ErrorActionPreference = "Stop"
Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; using System.Text;
public static class NimbusDllWitness {
 [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)] public static extern IntPtr LoadLibrary(string name);
 [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)] public static extern uint GetModuleFileName(IntPtr module, StringBuilder name, int length);
 [DllImport("kernel32.dll")] public static extern bool FreeLibrary(IntPtr module);
}'
$Library = [NimbusDllWitness]::LoadLibrary("__NIMBUS_WITNESS_NAME__")
if ($Library -eq [IntPtr]::Zero) {
 throw [ComponentModel.Win32Exception]::new([Runtime.InteropServices.Marshal]::GetLastWin32Error())
}
try {
 $LoadedPath = [Text.StringBuilder]::new(32768)
 if ([NimbusDllWitness]::GetModuleFileName($Library, $LoadedPath, 32768) -eq 0) { throw "Loaded DLL has no path." }
 Write-Output $LoadedPath.ToString()
} finally { [NimbusDllWitness]::FreeLibrary($Library) | Out-Null }
'@
        $Child = $Child.Replace("__NIMBUS_WITNESS_NAME__", $WitnessName)
        $Encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Child))
        $HostExecutable = [Diagnostics.Process]::GetCurrentProcess().MainModule.FileName
        $LoadedPath = & $HostExecutable -NoProfile -NonInteractive -EncodedCommand $Encoded
        if ($LASTEXITCODE -ne 0 -or @($LoadedPath).Count -ne 1 -or
            -not [string]::Equals([string]$LoadedPath, $WitnessPath, [StringComparison]::OrdinalIgnoreCase)) {
            throw "A build child did not inherit the selected native DLL directory."
        }
        Write-Output "Windows child DLL search: inherited directory resolved the exact isolated system-library copy."
    } finally {
        $RestoreDirectory = if ($PreviousLength -gt 0) { $PreviousDirectory.ToString() } else { $null }
        if (-not [NimbusBuild.NativeLibrarySearch]::SetDllDirectory($RestoreDirectory)) {
            throw "Could not restore the previous DLL search directory."
        }
    }
} finally {
    Remove-Item -LiteralPath $FixtureRoot -Recurse -Force
}
