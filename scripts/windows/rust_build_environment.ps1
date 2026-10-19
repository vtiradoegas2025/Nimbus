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

    $env:CMAKE = $CmakePath
    $env:LIBCLANG_PATH = $LibclangPath
    # DLL dependencies must be discoverable by bindgen's child process. Append
    # directories so existing Rust, Node and compiler command precedence stays intact.
    foreach ($Directory in @($NativeBin, (Split-Path -Parent $LibclangLibrary), (Split-Path -Parent $CmakePath))) {
        if (($env:Path -split ';') -notcontains $Directory) { $env:Path += ";$Directory" }
    }
}
