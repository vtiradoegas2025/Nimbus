[CmdletBinding()]
param(
    [switch]$Check,
    [switch]$Yes,
    [switch]$HeadlessOnly,
    [switch]$DependenciesOnly,
    [switch]$NoColor,
    [switch]$NoArt,
    [switch]$Art
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($null -ne $env:NIMBUS_BUILD_JOBS -and $env:NIMBUS_BUILD_JOBS -cnotmatch '\A[1-9][0-9]*\z')
{
    throw "NIMBUS_BUILD_JOBS must be a positive integer."
}

# Keep the Windows installer visually and semantically aligned with setup.sh.
# Artwork is shown only in an interactive terminal by default; automation gets
# concise plain output. -Art can explicitly retain it in captured build logs.
$OutputRedirected = [Console]::IsOutputRedirected
$UseColor = -not $NoColor -and -not $env:NO_COLOR -and -not $OutputRedirected
if ($UseColor) {
    $VirtualTerminalProperty = $Host.UI.PSObject.Properties["SupportsVirtualTerminal"]
    if ($null -eq $VirtualTerminalProperty -or -not [bool]$VirtualTerminalProperty.Value) {
        $UseColor = $false
    }
}
$ShowArt = -not $NoArt -and ($Art -or -not $OutputRedirected)

$Escape = [char]27
$Reset = if ($UseColor) { "${Escape}[0m" } else { "" }
$Bold = if ($UseColor) { "${Escape}[1m" } else { "" }
$Anvil = if ($UseColor) { "${Escape}[38;5;253m" } else { "" }
$Cloud = if ($UseColor) { "${Escape}[38;5;249m" } else { "" }
$Storm = if ($UseColor) { "${Escape}[38;5;246m" } else { "" }
$StormMid = if ($UseColor) { "${Escape}[38;5;243m" } else { "" }
$StormDark = if ($UseColor) { "${Escape}[38;5;240m" } else { "" }
$Bolt = if ($UseColor) { "${Escape}[38;5;227m" } else { "" }
$Funnel = if ($UseColor) { "${Escape}[38;5;244m" } else { "" }
$Ground = if ($UseColor) { "${Escape}[38;5;238m" } else { "" }
$Mark = if ($UseColor) { "${Escape}[38;5;250m" } else { "" }
$Mark2 = if ($UseColor) { "${Escape}[38;5;244m" } else { "" }
$Green = if ($UseColor) { "${Escape}[38;5;77m" } else { "" }
$Yellow = if ($UseColor) { "${Escape}[38;5;221m" } else { "" }
$Red = if ($UseColor) { "${Escape}[38;5;203m" } else { "" }
$Dim = if ($UseColor) { "${Escape}[38;5;244m" } else { "" }

function Write-NimbusLine([AllowEmptyString()][string]$Text = "") 
{
    Write-Host $Text
}

function Write-NimbusInfo([string]$Text) 
{
    Write-NimbusLine "${Bold}${Text}${Reset}"
}

function Write-NimbusOk([string]$Text) 
{
    Write-NimbusLine "  ${Green}ok${Reset}     $Text"
}

function Write-NimbusWarning([string]$Text) 
{
    Write-NimbusLine "  ${Yellow}miss${Reset}   $Text"
}

function Write-NimbusFailure([string]$Text) 
{
    Write-NimbusLine "  ${Red}FAIL${Reset}   $Text"
}

function Write-NimbusNote([string]$Text) 
{
    Write-NimbusLine "         ${Dim}${Text}${Reset}"
}

function Get-TerminalColumns 
{
    try 
    {
        $Columns = [int]$Host.UI.RawUI.WindowSize.Width
        if ($Columns -gt 0) { return $Columns }
    } 
    catch 
    {
        # Headless hosts do not expose RawUI. The art is already disabled for
        # redirected output unless the caller explicitly supplied -Art.
    }
    return 80
}

function Write-NimbusStormFull 
{
    Write-NimbusLine "                         ${Anvil}▄▄▄▄${Reset}"
    Write-NimbusLine "                    ${Anvil}▄▄▄████████▄▄▄${Reset}"
    Write-NimbusLine "              ${Anvil}▄▄▄███████████████████▄▄▄▄▄▄▄▄▄${Reset}"
    Write-NimbusLine "       ${Cloud}▄▄▄███████████████████████████████████████▄▄${Reset}"
    Write-NimbusLine "     ${Cloud}▄██████████████████████████████████████████████▄${Reset}"
    Write-NimbusLine "    ${Storm}████████████████████████████████${Cloud}▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀${Reset}"
    Write-NimbusLine "    ${Storm}███████████████████████████${StormMid}▀▀${Reset}      ${Bolt}▟▛${Reset}"
    Write-NimbusLine "     ${StormMid}██████████████████████${StormDark}▀▀${Reset}         ${Bolt}▟▛${Reset}"
    Write-NimbusLine "      ${StormMid}█████████████████${StormDark}▀${Reset}            ${Bolt}▀▜▛▀${Reset}"
    Write-NimbusLine "       ${StormDark}▀███████████▀${Reset}               ${Bolt}▟▛${Reset}"
    Write-NimbusLine "         ${StormDark}▀███████▀${Reset}                ${Bolt}▟▘${Reset}"
    Write-NimbusLine "          ${StormDark}▐█████▌${Reset}"
    Write-NimbusLine "           ${Funnel}▝███▘${Reset}"
    Write-NimbusLine "            ${Funnel}▝█▘${Reset}"
    Write-NimbusLine "    ${Ground}▁▂▃▄▅▆▇█${Funnel}▒▓▒${Ground}█▇▆▅▄▃▂▁▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▁▂▃▄▅▆▇█▇▆▅▄▃▂▁${Reset}"
}

function Write-NimbusStormCompact 
{
    Write-NimbusLine "             ${Anvil}▄▄▄▄${Reset}"
    Write-NimbusLine "         ${Anvil}▄▄▄████████▄▄▄${Reset}"
    Write-NimbusLine "    ${Cloud}▄▄███████████████▄▄▄▄▄▄▄${Reset}"
    Write-NimbusLine "  ${Cloud}▄███████████████████████████▄${Reset}"
    Write-NimbusLine "  ${Storm}██████████████████${Cloud}▀▀▀▀▀▀▀▀▀▀${Reset}"
    Write-NimbusLine "  ${Storm}███████████████${StormMid}▀▀${Reset}  ${Bolt}▟▛${Reset}"
    Write-NimbusLine "   ${StormMid}███████████${StormDark}▀▀${Reset}    ${Bolt}▀▜▛${Reset}"
    Write-NimbusLine "    ${StormDark}▀███████▀${Reset}        ${Bolt}▟▘${Reset}"
    Write-NimbusLine "      ${StormDark}▀███▀${Reset}"
    Write-NimbusLine "      ${StormDark}▐███▌${Reset}"
    Write-NimbusLine "       ${Funnel}▝█▘${Reset}"
    Write-NimbusLine "  ${Ground}▁▂▃▄▅${Funnel}▒▓▒${Ground}▇▆▅▄▃▂▁▁▂▃▄▅▆▇█▇▆▅▄▃▂▁${Reset}"
}

function Write-NimbusWordmark 
{
    Write-NimbusLine "   ${Mark} ███╗   ██╗██╗███╗   ███╗██████╗ ██╗   ██╗███████╗${Reset}"
    Write-NimbusLine "   ${Mark} ████╗  ██║██║████╗ ████║██╔══██╗██║   ██║██╔════╝${Reset}"
    Write-NimbusLine "   ${Mark} ██╔██╗ ██║██║██╔████╔██║██████╔╝██║   ██║███████╗${Reset}"
    Write-NimbusLine "   ${Mark2} ██║╚██╗██║██║██║╚██╔╝██║██╔══██╗██║   ██║╚════██║${Reset}"
    Write-NimbusLine "   ${Mark2} ██║ ╚████║██║██║ ╚═╝ ██║██████╔╝╚██████╔╝███████║${Reset}"
    Write-NimbusLine "   ${Mark2} ╚═╝  ╚═══╝╚═╝╚═╝     ╚═╝╚═════╝  ╚═════╝ ╚══════╝${Reset}"
}

function Write-NimbusAsciiWordmark 
{
    Write-NimbusLine "   ${Mark}  _  _ ___ __  __ ___ _   _ ___${Reset}"
    Write-NimbusLine "   ${Mark} | \| |_ _|  \/  | _ ) | | / __|${Reset}"
    Write-NimbusLine "   ${Mark2} | .`` || || |\/| | _ \ |_| \__ \${Reset}"
    Write-NimbusLine "   ${Mark2} |_|\_|___|_|  |_|___/\___/|___/${Reset}"
}

function Show-NimbusBanner 
{
    if (-not $ShowArt) { return }
    $Columns = Get-TerminalColumns
    if ($Columns -lt 38) { return }

    Write-NimbusLine
    if ($Columns -ge 56) 
    {
        Write-NimbusStormFull
        Write-NimbusLine
        Write-NimbusWordmark
        Write-NimbusLine
        Write-NimbusLine "   ${Dim}A modern approach to supercell modeling.${Reset}"
        Write-NimbusLine "   ${Dim}Convective-scale atmospheric simulation in C++17.${Reset}"
    } 
    else 
    {
        Write-NimbusStormCompact
        Write-NimbusLine
        Write-NimbusAsciiWordmark
        Write-NimbusLine
        Write-NimbusLine "   ${Dim}Supercell modeling in C++17.${Reset}"
    }
    Write-NimbusLine
}

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) 
{
    throw "scripts/setup.ps1 is the native Windows installer. Use scripts/setup.sh on macOS or Linux."
}
if (-not [Environment]::Is64BitOperatingSystem)
{
    throw "Nimbus requires 64-bit Windows."
}
if (-not [Environment]::Is64BitProcess) 
{
    throw "Nimbus setup requires 64-bit PowerShell. Open the 64-bit Windows PowerShell or PowerShell 7 terminal and rerun it."
}
$NativeArchitecture = if ($env:PROCESSOR_ARCHITEW6432) 
{
    $env:PROCESSOR_ARCHITEW6432
} 
else 
{
    $env:PROCESSOR_ARCHITECTURE
}
$ArchitectureKey = $NativeArchitecture.ToUpperInvariant()
switch ($ArchitectureKey) 
{
    { $_ -in @("AMD64", "X86_64") } 
    {
        $NimbusArchitecture = "x64"
        $MsysEnvironment = "ucrt64"
        $MsysSystem = "UCRT64"
        $MsysPrefix = "ucrt64"
        $MsysPackagePrefix = "mingw-w64-ucrt-x86_64"
        $RustTarget = "x86_64-pc-windows-msvc"
        $VsToolsComponent = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
        $WindowsPrefixName = "windows-prefix"
        break
    }
    { $_ -in @("ARM64", "AARCH64") } 
    {
        if ([Environment]::OSVersion.Version.Build -lt 22000) 
        {
            throw "Native Windows ARM64 setup requires Windows 11 (build 22000 or newer)."
        }
        $NimbusArchitecture = "arm64"
        $MsysEnvironment = "clangarm64"
        $MsysSystem = "CLANGARM64"
        $MsysPrefix = "clangarm64"
        $MsysPackagePrefix = "mingw-w64-clang-aarch64"
        $RustTarget = "aarch64-pc-windows-msvc"
        $VsToolsComponent = "Microsoft.VisualStudio.Component.VC.Tools.ARM64"
        $WindowsPrefixName = "windows-prefix-arm64"
        break
    }
    default 
    {
        throw "Nimbus supports native Windows x64 and ARM64; detected unsupported architecture '$NativeArchitecture'."
    }
}

Show-NimbusBanner

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
. (Join-Path $RootDir "scripts\windows\invoke_bash.ps1")
$MsysRoot = if ($env:NIMBUS_MSYS2_ROOT) { $env:NIMBUS_MSYS2_ROOT } else { "C:\msys64" }
$MsysShell = Join-Path $MsysRoot "msys2_shell.cmd"
$MsysBash = Join-Path $MsysRoot "usr\bin\bash.exe"
$WindowsPrefix = Join-Path $RootDir "build\$WindowsPrefixName"

# Git for Windows otherwise remains vulnerable to legacy MAX_PATH behavior in
# deeply nested Rust/npm build trees. Repository-local scope avoids changing a
# tester's global Git policy.
if ((Test-Path -LiteralPath (Join-Path $RootDir ".git")) -and
    ($null -ne (Get-Command "git" -ErrorAction SilentlyContinue))) 
    {
    & git -C $RootDir config core.longpaths true
    if ($LASTEXITCODE -ne 0) 
    {
        Write-NimbusWarning "Git long-path support could not be enabled for this checkout."
    }
}
$MsysPackages = @(
    "make", "git", "curl", "tar", "base-devel",
    "$MsysPackagePrefix-toolchain",
    "$MsysPackagePrefix-clang",
    "$MsysPackagePrefix-cmake",
    "$MsysPackagePrefix-ninja",
    "$MsysPackagePrefix-python",
    "$MsysPackagePrefix-pkgconf",
    "$MsysPackagePrefix-vulkan-devel",
    "$MsysPackagePrefix-glslang",
    "$MsysPackagePrefix-glfw",
    "$MsysPackagePrefix-netcdf",
    "$MsysPackagePrefix-ffmpeg"
)
$MsysCompilerPackage = if ($NimbusArchitecture -eq "arm64") 
{
    "$MsysPackagePrefix-clang"
} 
else 
{
    "$MsysPackagePrefix-gcc"
}
$MsysCheckPackages = @(
    "make", "git", "curl", "tar",
    $MsysCompilerPackage,
    "$MsysPackagePrefix-clang",
    "$MsysPackagePrefix-clang-libs",
    "$MsysPackagePrefix-cmake",
    "$MsysPackagePrefix-ninja",
    "$MsysPackagePrefix-python",
    "$MsysPackagePrefix-pkgconf",
    "$MsysPackagePrefix-vulkan-headers",
    "$MsysPackagePrefix-vulkan-loader",
    "$MsysPackagePrefix-spirv-tools",
    "$MsysPackagePrefix-glslang",
    "$MsysPackagePrefix-glfw",
    "$MsysPackagePrefix-netcdf",
    "$MsysPackagePrefix-ffmpeg"
)

function Test-Command([string]$Name) 
{
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Refresh-ProcessPath 
{
    $Machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $User = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$Machine;$User"
    $CargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
    if (Test-Path -LiteralPath $CargoBin) { $env:Path = "$CargoBin;$env:Path" }
}

function Install-WingetPackage([string]$Id, [string]$Label, [string]$Override = "") 
{
    Write-NimbusInfo "Installing $Label"
    $Arguments = @("install", "--id", $Id, "--exact", "--accept-package-agreements", "--accept-source-agreements", "--silent")
    if ($Override) { $Arguments += @("--override", $Override) }
    & winget @Arguments
    if ($LASTEXITCODE -ne 0) { throw "winget could not install $Label ($Id)." }
}

function Invoke-Msys([string]$Command) 
{
    $env:MSYSTEM = $MsysSystem
    $env:MSYS2_PATH_TYPE = "inherit"
    $env:CHERE_INVOKING = "1"
    Invoke-NimbusBash -Bash $MsysBash -Script $Command
}

function Test-MsysPackageSet 
{
    if (-not (Test-Path -LiteralPath $MsysShell -PathType Leaf)) { return $false }
    # base-devel and toolchain are pacman groups, not queryable package names;
    # verify the concrete runtime/compiler members the product actually uses.
    $PackageNames = $MsysCheckPackages -join " "
    try { Invoke-Msys "pacman -Q $PackageNames >/dev/null 2>&1"; return $true }
    catch { return $false }
}

function Test-NodeVersion 
{
    if (-not (Test-Command "node")) { return $false }
    try 
    {
        $Version = [version]((& node --version).TrimStart("v"))
        return ($Version.Major -eq 22 -and $Version -ge [version]"22.13.0") -or $Version.Major -ge 24
    } 
    catch 
    {
        return $false
    }
}

$NeedsMsys = -not (Test-Path -LiteralPath $MsysShell -PathType Leaf)
$NeedsNode = -not (Test-NodeVersion)
$NeedsRust = -not (Test-Command "cargo")
$VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$HasBuildTools = $false
if (Test-Path -LiteralPath $VsWhere -PathType Leaf) 
{
    $Installations = & $VsWhere -latest -products * -requires $VsToolsComponent -property installationPath
    $HasBuildTools = -not [string]::IsNullOrWhiteSpace(($Installations -join ""))
}
$NeedsBuildTools = -not $HeadlessOnly -and -not $HasBuildTools
$NeedsMsysPackages = -not $NeedsMsys -and -not (Test-MsysPackageSet)
$NeedsZfp = -not (
    (Test-Path -LiteralPath (Join-Path $WindowsPrefix "include\zfp.h") -PathType Leaf) -and
    (Test-Path -LiteralPath (Join-Path $WindowsPrefix "lib\libzfp.a") -PathType Leaf)
)
$Missing = @()
if ($NeedsMsys) { $Missing += "MSYS2 $MsysSystem scientific toolchain" }
elseif ($NeedsMsysPackages) { $Missing += "MSYS2 $MsysSystem scientific packages" }
if ($NeedsZfp) { $Missing += "digest-pinned ZFP 1.0.1" }
if (-not $HeadlessOnly -and $NeedsNode) { $Missing += "Node.js 22.13+ within 22, or 24+" }
if (-not $HeadlessOnly -and $NeedsRust) { $Missing += "Rust" }
if ($NeedsBuildTools) { $Missing += "Visual Studio C++ Build Tools" }

Write-NimbusInfo "Platform: Native Windows $NimbusArchitecture"
Write-NimbusLine "  Product:       $(if ($HeadlessOnly) { 'headless simulator (secondary)' } else { 'Nimbus Studio + bundled simulator (default)' })"
Write-NimbusLine "  Simulator:     CPU + OpenMP + Vulkan + ZFP + netCDF + live shared memory"
Write-NimbusLine "  Desktop GPU:   WebView2 WebGL/WebGPU plus native Vulkan compute"
Write-NimbusLine "  Toolchain:     MSYS2 $MsysSystem for C++; $RustTarget for Tauri"
if ($Missing.Count -eq 0) 
{
    Write-NimbusOk "Complete build stack is ready."
} 
else 
{
    Write-NimbusWarning ($Missing -join ", ")
}

if ($Check) 
{
    if ($Missing.Count -gt 0) { exit 1 }
    if (-not (Test-Path -LiteralPath $MsysBash -PathType Leaf)) { exit 1 }
    if (-not $HeadlessOnly -and -not (Test-Path -LiteralPath (Join-Path $RootDir "ui\src-tauri\icons\icon.ico") -PathType Leaf)) { exit 1 }
    if (-not $HeadlessOnly) {
        . (Join-Path $RootDir "scripts\windows\rust_build_environment.ps1")
        Initialize-NimbusRustBuildEnvironment -MsysRoot $MsysRoot -Architecture $NimbusArchitecture
    }
    exit 0
}

if (-not $Yes) 
{
    $Answer = Read-Host "Continue with the native Windows build? [y/N]"
    if ($Answer -notmatch '^[Yy]') { Write-NimbusNote "Cancelled; no changes were made."; exit 0 }
}

if ($Missing.Count -gt 0 -and -not (Test-Command "winget")) 
{
    throw "Windows Package Manager (winget) is required to bootstrap missing dependencies. Install App Installer, then run this script again."
}
if ($NeedsMsys) { Install-WingetPackage "MSYS2.MSYS2" "MSYS2" }
if (-not $HeadlessOnly -and $NeedsNode) { Install-WingetPackage "OpenJS.NodeJS.LTS" "Node.js LTS" }
if (-not $HeadlessOnly -and $NeedsRust) { Install-WingetPackage "Rustlang.Rustup" "Rust" }
if ($NeedsBuildTools) 
{
    Install-WingetPackage "Microsoft.VisualStudio.2022.BuildTools" "Visual Studio C++ Build Tools" "--wait --quiet --norestart --add Microsoft.VisualStudio.Workload.VCTools --add $VsToolsComponent --includeRecommended"
}
Refresh-ProcessPath

if (-not $HeadlessOnly -and -not (Test-NodeVersion))
{
    throw "Nimbus Studio requires Node.js 22.13+ within 22, or 24+. Open a new terminal after installing a supported Node.js LTS release."
}

if (-not (Test-Path -LiteralPath $MsysShell -PathType Leaf)) 
{
    throw "MSYS2 was installed but $MsysShell is unavailable. Open MSYS2 once, then rerun setup."
}

Invoke-Msys "pacman -Syu --noconfirm"
Invoke-Msys "pacman -S --needed --noconfirm $($MsysPackages -join ' ')"

$env:NIMBUS_WINDOWS_ROOT = $RootDir
$env:NIMBUS_WINDOWS_PREFIX = $WindowsPrefix
Invoke-Msys 'cd "$(cygpath -u "$NIMBUS_WINDOWS_ROOT")" && NIMBUS_WINDOWS_PREFIX="$(cygpath -u "$NIMBUS_WINDOWS_PREFIX")" ./scripts/windows/install_zfp.sh'

if (-not $HeadlessOnly) {
    . (Join-Path $RootDir "scripts\windows\rust_build_environment.ps1")
    Initialize-NimbusRustBuildEnvironment -MsysRoot $MsysRoot -Architecture $NimbusArchitecture
}

if ($DependenciesOnly) 
{
    Write-NimbusLine
    Write-NimbusOk "Nimbus Windows dependencies are installed."
    exit 0
}

if ($HeadlessOnly) 
{
    Invoke-Msys 'cd "$(cygpath -u "$NIMBUS_WINDOWS_ROOT")" && prefix="$(cygpath -u "$NIMBUS_WINDOWS_PREFIX")" && make -j"${NIMBUS_BUILD_JOBS:-$(nproc)}" REQUIRE_FULL_STACK=1 ZFP_PREFIX="$prefix" NETCDF=1'
    $BuiltSimulator = Join-Path $RootDir "bin\nimbus_sim.exe"
} 
else 
{
    # Studio distributes this binary to other machines of the same architecture, so its ISA must
    # not inherit the setup host's -march=native profile. Build it once here;
    # the packager's -BundleOnly mode revalidates rather than recompiling it.
    Invoke-Msys 'cd "$(cygpath -u "$NIMBUS_WINDOWS_ROOT")" && prefix="$(cygpath -u "$NIMBUS_WINDOWS_PREFIX")" && make -j"${NIMBUS_BUILD_JOBS:-$(nproc)}" PORTABLE=1 DISTRIBUTION=1 REQUIRE_FULL_STACK=1 ZFP_PREFIX="$prefix" NETCDF=1'
    $BuiltSimulator = Join-Path $RootDir "bin\nimbus_sim_portable.exe"
}
$env:NIMBUS_WINDOWS_BUILT_SIMULATOR = $BuiltSimulator
Invoke-Msys 'simulator="$(cygpath -u "$NIMBUS_WINDOWS_BUILT_SIMULATOR")" && capabilities="$("$simulator" --version)" && for required in "OpenMP runtime: available" "ZFP compression: available" "netCDF4 history output: available" "Vulkan compute: available" "live shared memory: available"; do grep -Fq "$required" <<<"$capabilities" || { printf "Missing product capability: %s\n" "$required" >&2; exit 1; }; done && "$simulator" --scheme-catalog=json >/dev/null'

if ($HeadlessOnly) 
{
    Write-NimbusLine
    Write-NimbusInfo "Nimbus headless simulator is ready (secondary workflow)"
    Write-NimbusLine "  $RootDir\bin\nimbus_sim.exe"
    Write-NimbusNote "Default desktop workflow: .\scripts\setup.ps1"
    exit 0
}

if (-not (Test-Command "node") -or -not (Test-Command "npm") -or -not (Test-Command "cargo")) 
{
    throw "Node/npm/Rust are not visible in this PowerShell session. Open a new terminal and rerun scripts\setup.ps1."
}
if (Test-Command "rustup") 
{
    & rustup target add $RustTarget
    if ($LASTEXITCODE -ne 0) { throw "Could not install the native Windows Rust target." }
}

Push-Location (Join-Path $RootDir "ui")
try 
{
    & npm ci
    if ($LASTEXITCODE -ne 0) { throw "npm ci failed." }
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\build_native_app.ps1" -BundleOnly
    if ($LASTEXITCODE -ne 0) { throw "Nimbus Studio Windows packaging failed." }
} 
finally 
{
    Pop-Location
}

$InstallerRoot = Join-Path $RootDir "ui\src-tauri\target\release\bundle\nsis"
$Installer = Get-ChildItem -LiteralPath $InstallerRoot -Filter "*.exe" -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1
Write-NimbusLine
Write-NimbusInfo "Launch Nimbus Studio (default workflow)"
if ($null -ne $Installer) 
{
    Write-NimbusOk "Windows installer: $($Installer.FullName)"
} 
else 
{
    Write-NimbusOk "Nimbus Studio packaging completed."
    Write-NimbusNote "Installer output: $InstallerRoot"
}
Write-NimbusNote "Terminal-only alternative: .\scripts\setup.ps1 -HeadlessOnly"
