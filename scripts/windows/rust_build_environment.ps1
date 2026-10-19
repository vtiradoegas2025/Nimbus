function Assert-NimbusNativePe([string]$Path, [int]$ExpectedMachine) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required native build dependency is missing: $Path"
    }
    $Stream = [IO.File]::OpenRead($Path)
    try {
        $Reader = [IO.BinaryReader]::new($Stream)
        if ($Stream.Length -lt 64 -or $Reader.ReadUInt16() -ne 0x5A4D) {
            throw "Native build dependency is not a PE binary: $Path"
        }
        $Stream.Position = 0x3C
        $Offset = $Reader.ReadUInt32()
        if ($Offset -gt $Stream.Length - 6) { throw "Invalid PE header: $Path" }
        $Stream.Position = $Offset
        if ($Reader.ReadUInt32() -ne 0x00004550) { throw "Invalid PE signature: $Path" }
        $Machine = $Reader.ReadUInt16()
        if ($Machine -ne $ExpectedMachine) {
            throw ("Native build dependency architecture mismatch: {0}; expected 0x{1:X4}, found 0x{2:X4}." -f $Path, $ExpectedMachine, $Machine)
        }
    } finally {
        $Stream.Dispose()
    }
}

function Initialize-NimbusNativeLibrarySearch {
    if ("NimbusBuild.NativeLibrarySearch" -as [type]) { return }
    Add-Type -TypeDefinition @'
using System.Runtime.InteropServices;
using System.Text;
namespace NimbusBuild {
    public static class NativeLibrarySearch {
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool SetDllDirectory(string path);
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        public static extern uint GetDllDirectory(uint length, StringBuilder path);
    }
}
'@
}

function Set-NimbusNativeDllDirectory([string]$Directory) {
    Initialize-NimbusNativeLibrarySearch
    if (-not [NimbusBuild.NativeLibrarySearch]::SetDllDirectory($Directory)) {
        $Failure = [ComponentModel.Win32Exception]::new([Runtime.InteropServices.Marshal]::GetLastWin32Error())
        throw "Cannot select native DLL directory '$Directory': $($Failure.Message)"
    }
}

function Assert-NimbusRustVersion([string[]]$VersionOutput) {
    $Release = @($VersionOutput | Where-Object { $_ -match '^release: ' })
    if ($Release.Count -ne 1 -or $Release[0] -notmatch '^release: (\d+\.\d+\.\d+)(?:[-+].*)?$') {
        throw "Could not determine the Rust compiler release from rustc -vV."
    }
    if ([version]$Matches[1] -lt [version]"1.79.0") {
        throw "Nimbus Studio's locked dependencies require Rust 1.79.0 or newer; found $($Matches[1]). Update the selected MSVC toolchain."
    }
}

function Get-NimbusBindgenArguments([string]$Existing, [string]$RustHost) {
    $Pattern = '(?:^|\s)["'']?--?target(?:=|\s+)["'']?([^\s"'']+)["'']?(?=\s|$)'
    foreach ($Target in [regex]::Matches($Existing, $Pattern)) {
        if ($Target.Groups[1].Value -ne $RustHost) {
            throw "Bindgen target '$($Target.Groups[1].Value)' conflicts with the native Rust host $RustHost."
        }
    }
    $Arguments = $Existing.Trim()
    if ($Arguments -match ('(?:^|\s)--target=' + [regex]::Escape($RustHost) + '$')) { return $Arguments }
    return "$Arguments --target=$RustHost".Trim()
}

function Test-NimbusMsvcHeaders([string]$IncludePath) {
    $Directories = @($IncludePath -split ';' | ForEach-Object { $_.Trim().Trim('"') } | Where-Object { $_ })
    foreach ($Header in @("vcruntime.h", "corecrt.h", "inttypes.h")) {
        if (-not ($Directories | Where-Object { Test-Path -LiteralPath (Join-Path $_ $Header) -PathType Leaf })) {
            return $false
        }
    }
    return $true
}

function Initialize-NimbusMsvcHeaders([string]$Architecture) {
    if (Test-NimbusMsvcHeaders $env:INCLUDE) { return }
    if ($env:INCLUDE) {
        throw "INCLUDE does not contain the MSVC and Universal CRT headers. Start a native Visual Studio Developer PowerShell or remove the incomplete override."
    }
    $VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $VsWhere -PathType Leaf)) {
        throw "Visual Studio discovery is unavailable. Install the native C++ Build Tools through Nimbus setup."
    }
    $Component = if ($Architecture -eq "arm64") { "Microsoft.VisualStudio.Component.VC.Tools.ARM64" } else { "Microsoft.VisualStudio.Component.VC.Tools.x86.x64" }
    $Installation = & $VsWhere -latest -prerelease -products * -requires $Component -property installationPath
    if ($LASTEXITCODE -ne 0 -or @($Installation).Count -ne 1) {
        throw "Cannot locate Visual Studio C++ tools for $Architecture."
    }
    $DeveloperCommand = Join-Path $Installation "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path -LiteralPath $DeveloperCommand -PathType Leaf)) {
        throw "Visual Studio developer environment is missing: $DeveloperCommand"
    }
    $CommandLine = '""{0}" -no_logo -arch={1} -host_arch={1} >nul && set INCLUDE"' -f $DeveloperCommand, $Architecture
    $Output = & $env:ComSpec /d /s /c $CommandLine
    $Includes = @($Output | Where-Object { $_ -match '^INCLUDE=' })
    if ($LASTEXITCODE -ne 0 -or $Includes.Count -ne 1 -or
        -not (Test-NimbusMsvcHeaders $Includes[0].Substring(8))) {
        throw "Visual Studio did not provide the MSVC and Universal CRT headers for $Architecture."
    }
    # MSYS Clang can lack Visual Studio discovery; bindgen's driver consumes INCLUDE.
    $env:INCLUDE = $Includes[0].Substring(8)
}

function Initialize-NimbusRustBuildEnvironment {
    param(
        [Parameter(Mandatory=$true)][string]$MsysRoot,
        [Parameter(Mandatory=$true)][ValidateSet("x64", "arm64")][string]$Architecture
    )
    $ExpectedMachine = if ($Architecture -eq "arm64") { 0xAA64 } else { 0x8664 }
    $RustHost = if ($Architecture -eq "arm64") { "aarch64-pc-windows-msvc" } else { "x86_64-pc-windows-msvc" }
    $Prefix = if ($Architecture -eq "arm64") { "clangarm64" } else { "ucrt64" }
    $NativeBin = (Resolve-Path -LiteralPath (Join-Path $MsysRoot "$Prefix\bin")).Path

    $RustCompiler = Get-Command rustc -CommandType Application -ErrorAction Stop
    $RustVersion = & $RustCompiler.Source -vV
    if ($LASTEXITCODE -ne 0 -or -not ($RustVersion -contains "host: $RustHost")) {
        throw "Nimbus Studio requires the native Rust host $RustHost; inspect rustup show and select that MSVC toolchain."
    }
    Assert-NimbusRustVersion $RustVersion
    if ($env:CARGO_BUILD_TARGET -and $env:CARGO_BUILD_TARGET -ne $RustHost) {
        throw "CARGO_BUILD_TARGET must be $RustHost for this native Studio build."
    }

    $CmakePath = Join-Path $NativeBin "cmake.exe"
    if ($env:CMAKE) {
        $CmakePath = (Get-Command $env:CMAKE -CommandType Application -ErrorAction Stop).Source
    }
    Assert-NimbusNativePe $CmakePath $ExpectedMachine

    $LibclangPath = if ($env:LIBCLANG_PATH) {
        (Resolve-Path -LiteralPath $env:LIBCLANG_PATH).Path
    } else { $NativeBin }
    $LibclangLibrary = if (Test-Path -LiteralPath $LibclangPath -PathType Container) {
        Join-Path $LibclangPath "libclang.dll"
    } else {
        $LibclangPath
    }
    Assert-NimbusNativePe $LibclangLibrary $ExpectedMachine
    $ClangPath = if ($env:CLANG_PATH) {
        (Get-Command $env:CLANG_PATH -CommandType Application -ErrorAction Stop).Source
    } else { Join-Path (Split-Path -Parent $LibclangLibrary) "clang.exe" }
    Assert-NimbusNativePe $ClangPath $ExpectedMachine
    Initialize-NimbusMsvcHeaders $Architecture

    $BindgenKey = "BINDGEN_EXTRA_CLANG_ARGS_$RustHost"
    $ExistingArguments = ""
    foreach ($Key in @($BindgenKey, "BINDGEN_EXTRA_CLANG_ARGS_$($RustHost.Replace('-', '_'))", "BINDGEN_EXTRA_CLANG_ARGS")) {
        $Value = [Environment]::GetEnvironmentVariable($Key, "Process")
        if ($null -ne $Value) { $ExistingArguments = $Value; break }
    }
    $BindgenArguments = Get-NimbusBindgenArguments $ExistingArguments $RustHost

    $env:CMAKE = $CmakePath
    $env:LIBCLANG_PATH = $LibclangPath
    $env:CLANG_PATH = $ClangPath
    # Bindgen omits --target on native builds, but MSYS libclang defaults to GNU.
    Set-Item -LiteralPath "Env:$BindgenKey" -Value $BindgenArguments
    # DLL dependencies must be discoverable by bindgen's child process. Append
    # directories so existing Rust, Node and compiler command precedence stays intact.
    foreach ($Directory in @($NativeBin, (Split-Path -Parent $LibclangLibrary), (Split-Path -Parent $CmakePath))) {
        if (($env:Path -split ';') -notcontains $Directory) { $env:Path += ";$Directory" }
    }
    # An absolute libclang path does not select its dependent DLLs. This search
    # directory is inherited by build children without reordering executables.
    Set-NimbusNativeDllDirectory (Split-Path -Parent $LibclangLibrary)
    "#include <stdint.h>`n#include <inttypes.h>`n#include <stddef.h>`nint nimbus_pointer_width[(sizeof(void*) == 8) ? 1 : -1];" |
        & $ClangPath "--target=$RustHost" -x c -fsyntax-only -
    if ($LASTEXITCODE -ne 0) {
        throw "Native Clang cannot parse the selected MSVC/Universal CRT headers for $RustHost."
    }
}
