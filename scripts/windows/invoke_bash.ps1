function Invoke-NimbusBash {
    param(
        [Parameter(Mandatory=$true)][string]$Bash,
        [Parameter(Mandatory=$true)][string]$Script,
        [switch]$CaptureOutput,
        [switch]$NoProfile
    )
    $Start = [Diagnostics.ProcessStartInfo]::new()
    $Start.FileName = $Bash
    $Start.Arguments = if ($NoProfile) { "--noprofile --norc -s --" } else { "--login -s --" }
    $Start.UseShellExecute = $false
    $Start.RedirectStandardInput = $true
    $Start.RedirectStandardOutput = $CaptureOutput.IsPresent
    if ($CaptureOutput) { $Start.StandardOutputEncoding = [Text.UTF8Encoding]::new($false) }
    $Process = [Diagnostics.Process]::new()
    $Process.StartInfo = $Start
    try {
        if (-not $Process.Start()) { throw "Could not start MSYS2 Bash: $Bash" }
        $Read = if ($CaptureOutput) { $Process.StandardOutput.ReadToEndAsync() } else { $null }
        # Windows PowerShell's command-line quoting and CRLF conversion must not alter Bash source.
        $Bytes = [Text.UTF8Encoding]::new($false).GetBytes($Script.Replace("`r`n", "`n") + "`n")
        $Process.StandardInput.BaseStream.Write($Bytes, 0, $Bytes.Length)
        $Process.StandardInput.Close()
        $Output = if ($Read) { $Read.GetAwaiter().GetResult() } else { $null }
        $Process.WaitForExit()
        if ($Process.ExitCode -ne 0) { throw "MSYS2 Bash script failed with exit code $($Process.ExitCode)." }
        if ($CaptureOutput) { $Output -split '\r?\n' | Where-Object { $_.Length -gt 0 } }
    } finally {
        $Process.Dispose()
    }
}
