param([Parameter(Mandatory=$true)][string]$Bash)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
. (Join-Path $Root "scripts\windows\invoke_bash.ps1")

$PreviousJobs = $env:NIMBUS_BUILD_JOBS
$PreviousValue = $env:NIMBUS_BASH_TEST_VALUE
try {
    $env:NIMBUS_BUILD_JOBS = "2"
    $env:NIMBUS_BASH_TEST_VALUE = 'C:\a path\literal $(not-a-command) & file'
    $Script = @'
set -euo pipefail
make() { printf '<%s>\n' "$@"; }
prefix='/a path/prefix'
make -j"${NIMBUS_BUILD_JOBS:-$(nproc)}" ZFP_PREFIX="$prefix"
printf '%s\n' "$NIMBUS_BASH_TEST_VALUE"
printf '\316\251\n'
'@
    $Script += "`nprintf '%s\n' '" + [char]0x03A9 + "'`n"
    $Expected = @('-j2', 'ZFP_PREFIX=/a path/prefix') | ForEach-Object { "<$_>" }
    $Expected += @($env:NIMBUS_BASH_TEST_VALUE, [string][char]0x03A9, [string][char]0x03A9)
    foreach ($Source in @($Script, ($Script -replace "`r?`n", "`r`n"))) {
        $Actual = @(Invoke-NimbusBash -Bash $Bash -Script $Source -NoProfile -CaptureOutput)
        if (($Actual -join "`n") -cne ($Expected -join "`n")) {
            throw "Bash source, quoted arguments, line endings or UTF-8 output changed across the process boundary."
        }
    }
    $Rejected = $false
    try { Invoke-NimbusBash -Bash $Bash -Script 'exit 37' -NoProfile -CaptureOutput }
    catch { $Rejected = $_.Exception.Message -match 'exit code 37' }
    if (-not $Rejected) { throw "Bash failure was not propagated with its exit code." }
    Invoke-NimbusBash -Bash $Bash -Script 'printf "Bash inherited output works\n"' -NoProfile
    Write-Output "PowerShell/Bash process contract passed: worker count, spaces, literal shell text, LF/CRLF, UTF-8 and failures."
} finally {
    $env:NIMBUS_BUILD_JOBS = $PreviousJobs
    $env:NIMBUS_BASH_TEST_VALUE = $PreviousValue
}
