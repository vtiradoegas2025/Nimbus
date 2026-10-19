param([Parameter(Mandatory = $true)][string]$Fixture)

$ErrorActionPreference = "Stop"
$Fixture = [System.IO.Path]::GetFullPath($Fixture)
if (-not (Test-Path -LiteralPath $Fixture -PathType Leaf)) {
    throw "Native process fixture is missing: $Fixture"
}
Add-Type -TypeDefinition (Get-Content -LiteralPath (Join-Path $PSScriptRoot "qualification_job.cs") -Raw) -Language CSharp
$PriorExit = $env:NIMBUS_TEST_PROCESS_EXIT
$PriorWait = $env:NIMBUS_TEST_PROCESS_WAIT
$Process = $null
try {
    $env:NIMBUS_TEST_PROCESS_WAIT = $null
    foreach ($Expected in @(0, 37, 0, 91, 0)) {
        $env:NIMBUS_TEST_PROCESS_EXIT = "$Expected"
        try {
            $Process = [NimbusQualificationJob]::Launch($Fixture, (Split-Path -Parent $Fixture), "NimbusJobTest-$([Guid]::NewGuid().ToString('N'))")
            if (-not $Process.WaitForExit(10000)) { throw "Fast-exit fixture did not exit." }
            # Delay the query until the child has exited and the launch call has closed its original handle.
            Start-Sleep -Milliseconds 100
            if ($null -eq $Process.ExitCode -or $Process.ExitCode -ne $Expected) {
                throw "Expected retained status $Expected, received '$($Process.ExitCode)'."
            }
        } finally {
            if ($null -ne $Process) { $Process.Dispose(); $Process = $null }
            [NimbusQualificationJob]::Terminate(1)
            [NimbusQualificationJob]::Close()
        }
    }
    $env:NIMBUS_TEST_PROCESS_WAIT = "1"
    $Process = [NimbusQualificationJob]::Launch($Fixture, (Split-Path -Parent $Fixture), "NimbusJobTest-$([Guid]::NewGuid().ToString('N'))")
    if ($Process.WaitForExit(50)) { throw "Waiting fixture unexpectedly exited." }
    [NimbusQualificationJob]::Terminate(73)
    if (-not $Process.WaitForExit(10000)) { throw "Job termination failed to stop the fixture." }
    if ($null -eq $Process.ExitCode -or $Process.ExitCode -ne 73) {
        throw "Job termination lost its exit status: '$($Process.ExitCode)'."
    }
    $Process.Dispose(); $Process = $null
    [NimbusQualificationJob]::Close()
    $Rejected = $false
    try {
        [NimbusQualificationJob]::Launch("$Fixture.missing", (Split-Path -Parent $Fixture), "NimbusJobTest-$([Guid]::NewGuid().ToString('N'))") | Out-Null
    } catch { $Rejected = $true }
    if (-not $Rejected) { throw "Missing executable was accepted." }
    $env:NIMBUS_TEST_PROCESS_WAIT = $null
    $env:NIMBUS_TEST_PROCESS_EXIT = "42"
    $Process = [NimbusQualificationJob]::Launch($Fixture, (Split-Path -Parent $Fixture), "NimbusJobTest-$([Guid]::NewGuid().ToString('N'))")
    if (-not $Process.WaitForExit(10000) -or $null -eq $Process.ExitCode -or $Process.ExitCode -ne 42) {
        throw "Launch did not recover after a missing executable."
    }
    Write-Output "Qualification launcher retained all exit codes, enforced termination and recovered after launch failure."
} finally {
    [NimbusQualificationJob]::Terminate(1)
    if ($null -ne $Process) { $Process.Dispose() }
    [NimbusQualificationJob]::Close()
    $env:NIMBUS_TEST_PROCESS_EXIT = $PriorExit
    $env:NIMBUS_TEST_PROCESS_WAIT = $PriorWait
}
