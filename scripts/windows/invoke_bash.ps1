function Invoke-NimbusBash {
    param(
        [Parameter(Mandatory=$true)][string]$Bash,
        [Parameter(Mandatory=$true)][string]$Script,
        [switch]$CaptureOutput,
        [switch]$NoProfile
    )
    $SourceFile = [IO.Path]::GetTempFileName()
    $Start = [Diagnostics.ProcessStartInfo]::new()
    $Start.FileName = $Bash
    $ProfileArguments = if ($NoProfile) { "--noprofile --norc" } else { "--login" }
    $Start.Arguments = $ProfileArguments + ' -- "' + $SourceFile.Replace('\', '/') + '"'
    $Start.UseShellExecute = $false
    $Start.RedirectStandardOutput = $CaptureOutput.IsPresent
    if ($CaptureOutput) { $Start.StandardOutputEncoding = [Text.UTF8Encoding]::new($false) }
    $Process = [Diagnostics.Process]::new()
    $Process.StartInfo = $Start
    try {
        # A file avoids both command-line reinterpretation and .NET Framework's stdin BOM.
        $Bytes = [Text.UTF8Encoding]::new($false).GetBytes($Script.Replace("`r`n", "`n") + "`n")
        [IO.File]::WriteAllBytes($SourceFile, $Bytes)
        if (-not $Process.Start()) { throw "Could not start MSYS2 Bash: $Bash" }
        $Read = if ($CaptureOutput) { $Process.StandardOutput.ReadToEndAsync() } else { $null }
        $Output = if ($Read) { $Read.GetAwaiter().GetResult() } else { $null }
        $Process.WaitForExit()
        if ($Process.ExitCode -ne 0) { throw "MSYS2 Bash script failed with exit code $($Process.ExitCode)." }
        if ($CaptureOutput) { $Output -split '\r?\n' | Where-Object { $_.Length -gt 0 } }
    } finally {
        $Process.Dispose()
        [IO.File]::Delete($SourceFile)
    }
}
