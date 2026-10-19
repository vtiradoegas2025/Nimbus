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
        if (job != IntPtr.Zero) throw new InvalidOperationException("A qualification job is already active.");
        job = CreateJobObject(IntPtr.Zero, jobName);
        if (job == IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error());
        var startup = new STARTUPINFO();
        startup.cb = (UInt32)Marshal.SizeOf(startup);
        PROCESS_INFORMATION info;
        var command = new StringBuilder("\"" + executable + "\"");
        if (!CreateProcess(executable, command, IntPtr.Zero, IntPtr.Zero, false,
            CREATE_SUSPENDED, IntPtr.Zero, workingDirectory, ref startup, out info)) {
            var error = new Win32Exception(Marshal.GetLastWin32Error());
            Close();
            throw error;
        }
        Process process = null;
        try {
            if (!AssignProcessToJobObject(job, info.hProcess))
                throw new Win32Exception(Marshal.GetLastWin32Error());
            process = Process.GetProcessById((int)info.dwProcessId);
            // Pin the query handle before the child can exit; a PID cannot retain its exit status.
            if (process.Handle == IntPtr.Zero)
                throw new InvalidOperationException("Cannot retain the qualification process handle.");
            if (ResumeThread(info.hThread) == UInt32.MaxValue)
                throw new Win32Exception(Marshal.GetLastWin32Error());
            return process;
        } catch {
            TerminateProcess(info.hProcess, 1);
            TerminateJobObject(job, 1);
            if (process != null) process.Dispose();
            Close();
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
