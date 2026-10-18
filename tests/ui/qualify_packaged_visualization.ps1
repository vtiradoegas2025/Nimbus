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
$JobLauncherSource = @'
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

public static class NimbusQualificationJob {
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct STARTUPINFO {
        public UInt32 cb;
        public string lpReserved;
        public string lpDesktop;
        public string lpTitle;
        public UInt32 dwX;
        public UInt32 dwY;
        public UInt32 dwXSize;
        public UInt32 dwYSize;
        public UInt32 dwXCountChars;
        public UInt32 dwYCountChars;
        public UInt32 dwFillAttribute;
        public UInt32 dwFlags;
        public UInt16 wShowWindow;
        public UInt16 cbReserved2;
        public IntPtr lpReserved2;
        public IntPtr hStdInput;
        public IntPtr hStdOutput;
        public IntPtr hStdError;
    }
    [StructLayout(LayoutKind.Sequential)]
    private struct PROCESS_INFORMATION {
        public IntPtr hProcess;
        public IntPtr hThread;
        public UInt32 dwProcessId;
        public UInt32 dwThreadId;
    }
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateJobObject(IntPtr attributes, string name);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool CreateProcess(
        string applicationName, StringBuilder commandLine, IntPtr processAttributes,
        IntPtr threadAttributes, bool inheritHandles, UInt32 creationFlags,
        IntPtr environment, string currentDirectory, ref STARTUPINFO startup,
        out PROCESS_INFORMATION processInformation);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern UInt32 ResumeThread(IntPtr thread);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool TerminateJobObject(IntPtr job, UInt32 exitCode);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool TerminateProcess(IntPtr process, UInt32 exitCode);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr handle);

    private const UInt32 CREATE_SUSPENDED = 0x00000004;
    private static IntPtr job = IntPtr.Zero;

    public static Process Launch(string executable, string workingDirectory, string jobName) {
        job = CreateJobObject(IntPtr.Zero, jobName);
        if (job == IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error());
        var startup = new STARTUPINFO();
        startup.cb = (UInt32)Marshal.SizeOf(startup);
        PROCESS_INFORMATION info;
        var command = new StringBuilder("\"" + executable + "\"");
        if (!CreateProcess(executable, command, IntPtr.Zero, IntPtr.Zero, false,
            CREATE_SUSPENDED, IntPtr.Zero, workingDirectory, ref startup, out info)) {
            throw new Win32Exception(Marshal.GetLastWin32Error());
        }
        try {
            if (!AssignProcessToJobObject(job, info.hProcess))
                throw new Win32Exception(Marshal.GetLastWin32Error());
            if (ResumeThread(info.hThread) == UInt32.MaxValue)
                throw new Win32Exception(Marshal.GetLastWin32Error());
            return Process.GetProcessById((int)info.dwProcessId);
        } catch {
            TerminateProcess(info.hProcess, 1);
            TerminateJobObject(job, 1);
            throw;
        } finally {
            CloseHandle(info.hThread);
            CloseHandle(info.hProcess);
        }
    }

    public static void Terminate(UInt32 exitCode) {
        if (job != IntPtr.Zero) TerminateJobObject(job, exitCode);
    }
    public static void Close() {
        if (job != IntPtr.Zero) {
            CloseHandle(job);
            job = IntPtr.Zero;
        }
    }
}
'@
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
    } finally {
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
