param(
    [ValidateSet("smoke", "production")]
    [string]$Profile = "smoke",
    [ValidateSet("debug", "release")]
    [string]$Mode = "debug",
    [string]$Output = "",
    [switch]$SkipBuild,
    [ValidateRange(1, 3600)]
    [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = "Stop"
$JobLauncherSource = Get-Content -LiteralPath (Join-Path $PSScriptRoot "release/qualification_job.cs") -Raw
$UiDir = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "../../ui"))
if (-not $Output) {
    $Output = Join-Path $UiDir "build\qualification\packaged-visualization.json"
}
$Output = [System.IO.Path]::GetFullPath($Output)
$Progress = "$Output.progress.ndjson"
$OutputDirectory = Split-Path -Parent $Output
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
if (Test-Path -LiteralPath $Output) {
    throw "Refusing to overwrite existing evidence: $Output"
}
if (Test-Path -LiteralPath $Progress) {
    throw "Refusing to reuse existing progress evidence: $Progress"
}

Push-Location $UiDir
try {
    if (-not $SkipBuild) {
        if ($Mode -eq "debug") {
            & npx tauri build --debug --no-bundle --ci
        } else {
            & npx tauri build --no-bundle --ci
        }
        if ($LASTEXITCODE -ne 0) { throw "Tauri build failed with $LASTEXITCODE" }
    }

    $Binary = Join-Path $UiDir "src-tauri\target\$Mode\nimbus-studio.exe"
    if (-not (Test-Path -LiteralPath $Binary -PathType Leaf)) {
        throw "Packaged qualification binary is missing: $Binary"
    }

    $PriorOutput = $env:NIMBUS_STUDIO_VIS_QUALIFICATION_OUTPUT
    $PriorProfile = $env:NIMBUS_STUDIO_VIS_QUALIFICATION_PROFILE
    $PriorJob = $env:NIMBUS_STUDIO_WHOLE_APP_JOB
    $JobName = "NimbusStudioQualification-$PID-$([Guid]::NewGuid().ToString('N'))"
    $JobLoaded = $false
    $Process = $null
    try {
        $env:NIMBUS_STUDIO_VIS_QUALIFICATION_OUTPUT = $Output
        $env:NIMBUS_STUDIO_VIS_QUALIFICATION_PROFILE = $Profile
        $env:NIMBUS_STUDIO_WHOLE_APP_JOB = $JobName
        Add-Type -TypeDefinition $JobLauncherSource -Language CSharp
        $JobLoaded = $true
        $Process = [NimbusQualificationJob]::Launch($Binary, $UiDir, $JobName)
        if (-not $Process.WaitForExit($TimeoutSeconds * 1000)) {
            [NimbusQualificationJob]::Terminate(1)
            if (Test-Path -LiteralPath $Progress -PathType Leaf) {
                Write-Warning "Last packaged visualization progress events:`n$((Get-Content -LiteralPath $Progress -Tail 20) -join "`n")"
            }
            throw "Packaged visualization qualification timed out after ${TimeoutSeconds}s."
        }
        $AppStatus = $Process.ExitCode
        if ($null -eq $AppStatus) { throw "Packaged application exit status is unavailable." }
    } finally {
        if ($null -ne $Process) { $Process.Dispose() }
        if ($JobLoaded) { [NimbusQualificationJob]::Close() }
        $env:NIMBUS_STUDIO_VIS_QUALIFICATION_OUTPUT = $PriorOutput
        $env:NIMBUS_STUDIO_VIS_QUALIFICATION_PROFILE = $PriorProfile
        $env:NIMBUS_STUDIO_WHOLE_APP_JOB = $PriorJob
    }

    if (-not (Test-Path -LiteralPath $Output -PathType Leaf)) {
        if (Test-Path -LiteralPath $Progress -PathType Leaf) {
            Write-Warning "Last packaged visualization progress events:`n$((Get-Content -LiteralPath $Progress -Tail 20) -join "`n")"
        }
        throw "Packaged application exited without writing qualification evidence (status $AppStatus)."
    }
    $Architecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
    $TargetArch = switch ($Architecture) {
        "X64" { "x86_64" }
        "Arm64" { "aarch64" }
        default { throw "Unsupported packaged qualification architecture: $Architecture" }
    }
    & node --experimental-strip-types --import ../tests/ui/register-ts-resolve.mjs `
        ../tests/ui/release/validate_packaged_visualization_evidence.ts `
        $Output --profile $Profile --mode $Mode --target-os windows --target-arch $TargetArch
    if ($LASTEXITCODE -ne 0) { throw "Packaged visualization evidence validation failed." }
    if ($AppStatus -ne 0) { throw "Packaged application reported failure status $AppStatus." }
    Remove-Item -LiteralPath $Progress -Force -ErrorAction SilentlyContinue
} finally {
    Pop-Location
}
