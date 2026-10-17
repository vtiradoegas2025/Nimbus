use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::AppResult;

/*
 * Detected machine specs surfaced in the Settings page + Queue Panel
 * footer so the user can size their simulation budget at a glance.
 *
 * Probe is best-effort: missing fields come back as null on the JS side
 * rather than failing the command. GPU detection is deferred — the
 * Vulkan loader is the eventual source of truth there.
 */

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareInfo {
    pub cpu_brand: Option<String>,
    pub core_count: Option<u32>,
    pub ram_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProcessMemoryInfo {
    pub schema: &'static str,
    pub schema_version: u32,
    pub current_rss_bytes: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
    pub process_tree_rss_bytes: Option<u64>,
    pub process_tree_peak_rss_bytes: Option<u64>,
    pub process_tree_process_count: Option<u32>,
    pub process_tree_coverage: &'static str,
    pub measurement: &'static str,
}

static PROCESS_TREE_PEAK_RSS_BYTES: AtomicU64 = AtomicU64::new(0);

#[tauri::command]
pub fn hardware_info() -> AppResult<HardwareInfo> {
    Ok(HardwareInfo {
        cpu_brand: get_cpu_brand(),
        core_count: get_core_count(),
        ram_bytes: get_ram_bytes(),
    })
}

#[tauri::command]
pub fn process_memory_info() -> AppResult<ProcessMemoryInfo> {
    let (current_rss_bytes, peak_rss_bytes, measurement) = process_memory_values();
    let (process_tree_rss_bytes, process_tree_process_count, process_tree_coverage) =
        process_tree_memory_values();
    let process_tree_peak_rss_bytes = sampled_tree_peak(process_tree_rss_bytes);
    Ok(ProcessMemoryInfo {
        schema: "nimbus.studio.process-memory",
        schema_version: 3,
        current_rss_bytes,
        peak_rss_bytes,
        process_tree_rss_bytes,
        process_tree_peak_rss_bytes,
        process_tree_process_count,
        process_tree_coverage,
        measurement,
    })
}

fn sampled_tree_peak(current: Option<u64>) -> Option<u64> {
    let current = current?;
    PROCESS_TREE_PEAK_RSS_BYTES.fetch_max(current, Ordering::Relaxed);
    Some(PROCESS_TREE_PEAK_RSS_BYTES.load(Ordering::Relaxed))
}

#[cfg(target_os = "macos")]
fn process_tree_memory_values() -> (Option<u64>, Option<u32>, &'static str) {
    macos_resource_coalition_memory_values()
        .map(|(bytes, count)| (Some(bytes), Some(count), "whole_application"))
        .unwrap_or_else(unix_descendant_process_tree_memory_values)
}

#[cfg(all(unix, not(target_os = "macos")))]
fn process_tree_memory_values() -> (Option<u64>, Option<u32>, &'static str) {
    #[cfg(target_os = "linux")]
    if std::env::var_os("NIMBUS_STUDIO_WHOLE_APP_SESSION").as_deref()
        == Some(std::ffi::OsStr::new("1"))
    {
        if let Some((bytes, count)) = linux_session_memory_values() {
            return (Some(bytes), Some(count), "whole_application");
        }
    }
    unix_descendant_process_tree_memory_values()
}

#[cfg(target_os = "linux")]
fn linux_session_memory_values() -> Option<(u64, u32)> {
    let session = unsafe { libc::getsid(0) };
    if session <= 0 {
        return None;
    }
    let sampler = std::process::Command::new("ps")
        .args(["-axo", "pid=,sid=,rss="])
        .stdout(std::process::Stdio::piped())
        .spawn()
        .ok()?;
    let sampler_pid = sampler.id();
    let output = sampler.wait_with_output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let mut rss_kib = 0_u64;
    let mut members = 0_u32;
    for line in text.lines() {
        let mut columns = line.split_whitespace();
        let pid: u32 = columns.next()?.parse().ok()?;
        let sid: i32 = columns.next()?.parse().ok()?;
        let rss: u64 = columns.next()?.parse().ok()?;
        if pid == sampler_pid || sid != session {
            continue;
        }
        rss_kib = rss_kib.checked_add(rss)?;
        members = members.checked_add(1)?;
    }
    (members > 0).then(|| (rss_kib.saturating_mul(1024), members))
}

#[cfg(unix)]
fn unix_descendant_process_tree_memory_values() -> (Option<u64>, Option<u32>, &'static str) {
    let Some(rows) = unix_process_rows() else {
        return (None, None, "host_only");
    };
    descendant_rss(std::process::id(), &rows)
        .map(|(bytes, count)| (Some(bytes), Some(count), "host_plus_descendants"))
        .unwrap_or((None, None, "host_only"))
}

#[cfg(unix)]
fn unix_process_rows() -> Option<Vec<(u32, u32, u64)>> {
    let sampler = match std::process::Command::new("ps")
        .args(["-axo", "pid=,ppid=,rss="])
        .stdout(std::process::Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        _ => return None,
    };
    let sampler_pid = sampler.id();
    let output = match sampler.wait_with_output() {
        Ok(output) if output.status.success() => output,
        _ => return None,
    };
    let text = match String::from_utf8(output.stdout) {
        Ok(text) => text,
        Err(_) => return None,
    };
    Some(
        text.lines()
            .filter_map(|line| {
                let mut columns = line.split_whitespace();
                let pid = columns.next()?.parse().ok()?;
                if pid == sampler_pid {
                    return None;
                }
                Some((
                    pid,
                    columns.next()?.parse().ok()?,
                    columns.next()?.parse().ok()?,
                ))
            })
            .collect(),
    )
}

#[cfg(unix)]
fn descendant_rss(root_pid: u32, rows: &[(u32, u32, u64)]) -> Option<(u64, u32)> {
    let mut included = std::collections::HashSet::from([root_pid]);
    loop {
        let before = included.len();
        for &(pid, parent, _) in rows {
            if included.contains(&parent) {
                included.insert(pid);
            }
        }
        if included.len() == before {
            break;
        }
    }
    let mut kib = 0_u64;
    let mut count = 0_u32;
    for &(pid, _, rss_kib) in rows {
        if included.contains(&pid) {
            kib = kib.saturating_add(rss_kib);
            count = count.saturating_add(1);
        }
    }
    (count > 0).then(|| (kib.saturating_mul(1024), count))
}

#[cfg(target_os = "macos")]
#[repr(C)]
struct ProcPidCoalitionInfo {
    coalition_id: [u64; 2],
    reserved: [u64; 3],
}

#[cfg(target_os = "macos")]
fn macos_process_resource_coalition(pid: i32) -> Option<u64> {
    const PROC_PIDCOALITIONINFO: i32 = 20;
    let mut info = ProcPidCoalitionInfo {
        coalition_id: [0; 2],
        reserved: [0; 3],
    };
    let size = std::mem::size_of::<ProcPidCoalitionInfo>();
    let read = unsafe {
        libc::proc_pidinfo(
            pid,
            PROC_PIDCOALITIONINFO,
            0,
            &mut info as *mut ProcPidCoalitionInfo as *mut libc::c_void,
            size as i32,
        )
    };
    (read == size as i32 && info.coalition_id[0] > 0).then_some(info.coalition_id[0])
}

#[cfg(target_os = "macos")]
fn macos_process_resident_bytes(pid: i32) -> Option<u64> {
    let mut usage = unsafe { std::mem::zeroed::<libc::rusage_info_v0>() };
    let ok = unsafe {
        libc::proc_pid_rusage(
            pid,
            libc::RUSAGE_INFO_V0,
            &mut usage as *mut libc::rusage_info_v0 as *mut libc::rusage_info_t,
        )
    } == 0;
    if ok && usage.ri_resident_size > 0 {
        return Some(usage.ri_resident_size);
    }
    let mut task = unsafe { std::mem::zeroed::<libc::proc_taskinfo>() };
    let size = std::mem::size_of::<libc::proc_taskinfo>();
    let read = unsafe {
        libc::proc_pidinfo(
            pid,
            libc::PROC_PIDTASKINFO,
            0,
            &mut task as *mut libc::proc_taskinfo as *mut libc::c_void,
            size as i32,
        )
    };
    (read == size as i32 && task.pti_resident_size > 0).then_some(task.pti_resident_size)
}

#[cfg(target_os = "macos")]
fn macos_resource_coalition_memory_values() -> Option<(u64, u32)> {
    let coalition = macos_process_resource_coalition(std::process::id() as i32)?;
    let rows = unix_process_rows()?;
    coalition_rss(coalition, &rows, |pid| {
        macos_process_resource_coalition(pid as i32)
    })
}

#[cfg(any(test, target_os = "macos"))]
fn coalition_rss<F>(
    coalition: u64,
    rows: &[(u32, u32, u64)],
    mut coalition_for_pid: F,
) -> Option<(u64, u32)>
where
    F: FnMut(u32) -> Option<u64>,
{
    if coalition == 0 {
        return None;
    }
    let mut resident = 0_u64;
    let mut members = 0_u32;
    for &(pid, _, rss_kib) in rows {
        if coalition_for_pid(pid) != Some(coalition) {
            continue;
        }
        resident = resident.checked_add(rss_kib.checked_mul(1024)?)?;
        members = members.checked_add(1)?;
    }
    (members > 0 && resident > 0).then_some((resident, members))
}

#[cfg(target_os = "windows")]
fn process_tree_memory_values() -> (Option<u64>, Option<u32>, &'static str) {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W,
        TH32CS_SNAPPROCESS,
    };
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::JobObjects::IsProcessInJob;
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, GetCurrentProcessId, OpenProcess,
        PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_VM_READ,
    };

    let mut current_in_job = 0;
    let isolated_job = std::env::var_os("NIMBUS_STUDIO_WHOLE_APP_JOB").is_some()
        && unsafe { IsProcessInJob(GetCurrentProcess(), std::ptr::null_mut(), &mut current_in_job) }
            != 0
        && current_in_job != 0;

    let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) };
    if snapshot == INVALID_HANDLE_VALUE {
        return (None, None, "host_only");
    }
    let mut entry = unsafe { std::mem::zeroed::<PROCESSENTRY32W>() };
    entry.dwSize = std::mem::size_of::<PROCESSENTRY32W>() as u32;
    let mut rows = Vec::new();
    let mut ok = unsafe { Process32FirstW(snapshot, &mut entry) };
    while ok != 0 {
        rows.push((entry.th32ProcessID, entry.th32ParentProcessID));
        ok = unsafe { Process32NextW(snapshot, &mut entry) };
    }
    unsafe { CloseHandle(snapshot) };

    let root = unsafe { GetCurrentProcessId() };
    let mut included = std::collections::HashSet::from([root]);
    loop {
        let before = included.len();
        for &(pid, parent) in &rows {
            if included.contains(&parent) {
                included.insert(pid);
            }
        }
        if included.len() == before {
            break;
        }
    }
    let mut bytes = 0_u64;
    let mut count = 0_u32;
    for pid in included {
        let process =
            unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, 0, pid) };
        if process.is_null() {
            continue;
        }
        let mut counters = unsafe { std::mem::zeroed::<PROCESS_MEMORY_COUNTERS>() };
        counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        let read = unsafe { K32GetProcessMemoryInfo(process, &mut counters, counters.cb) };
        unsafe { CloseHandle(process) };
        if read != 0 {
            bytes = bytes.saturating_add(counters.WorkingSetSize as u64);
            count = count.saturating_add(1);
        }
    }
    if count == 0 {
        (None, None, "host_only")
    } else {
        (
            Some(bytes),
            Some(count),
            if isolated_job {
                "whole_application"
            } else {
                "host_plus_descendants"
            },
        )
    }
}

#[cfg(not(any(unix, target_os = "windows")))]
fn process_tree_memory_values() -> (Option<u64>, Option<u32>, &'static str) {
    (None, None, "host_only")
}

#[cfg(target_os = "linux")]
fn process_memory_values() -> (Option<u64>, Option<u64>, &'static str) {
    let current = std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|text| text.split_whitespace().nth(1)?.parse::<u64>().ok())
        .and_then(|pages| {
            let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            (page_size > 0).then(|| pages.saturating_mul(page_size as u64))
        });
    (
        current,
        unix_peak_rss_bytes(true),
        "proc_self_statm+getrusage",
    )
}

#[cfg(target_os = "macos")]
fn process_memory_values() -> (Option<u64>, Option<u64>, &'static str) {
    let current = macos_process_resident_bytes(std::process::id() as i32);
    (
        current,
        unix_peak_rss_bytes(false),
        "proc_pid_rusage_resident+getrusage_peak",
    )
}

#[cfg(unix)]
fn unix_peak_rss_bytes(kibibytes: bool) -> Option<u64> {
    let mut usage = unsafe { std::mem::zeroed::<libc::rusage>() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 || usage.ru_maxrss < 0 {
        return None;
    }
    let value = usage.ru_maxrss as u64;
    Some(if kibibytes {
        value.saturating_mul(1024)
    } else {
        value
    })
}

#[cfg(target_os = "windows")]
fn process_memory_values() -> (Option<u64>, Option<u64>, &'static str) {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut counters = unsafe { std::mem::zeroed::<PROCESS_MEMORY_COUNTERS>() };
    counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    if ok == 0 {
        return (None, None, "GetProcessMemoryInfo_unavailable");
    }
    (
        Some(counters.WorkingSetSize as u64),
        Some(counters.PeakWorkingSetSize as u64),
        "GetProcessMemoryInfo",
    )
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn process_memory_values() -> (Option<u64>, Option<u64>, &'static str) {
    (None, None, "unavailable")
}

/* ---------- Per-platform probes ---------- */

#[cfg(target_os = "macos")]
fn get_cpu_brand() -> Option<String> {
    sysctl_string("machdep.cpu.brand_string")
}

#[cfg(target_os = "linux")]
fn get_cpu_brand() -> Option<String> {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in cpuinfo.lines() {
        if line.starts_with("model name") {
            return line.split(':').nth(1).map(|s| s.trim().to_string());
        }
    }
    None
}

#[cfg(target_os = "windows")]
fn get_cpu_brand() -> Option<String> {
    powershell_value(
        "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
    )
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
fn get_cpu_brand() -> Option<String> {
    None
}

#[cfg(target_os = "macos")]
fn get_core_count() -> Option<u32> {
    sysctl_u64("hw.ncpu").and_then(|n| u32::try_from(n).ok())
}

#[cfg(not(target_os = "macos"))]
fn get_core_count() -> Option<u32> {
    std::thread::available_parallelism()
        .ok()
        .map(|n| n.get() as u32)
}

#[cfg(target_os = "macos")]
pub(crate) fn get_ram_bytes() -> Option<u64> {
    sysctl_u64("hw.memsize")
}

#[cfg(target_os = "linux")]
pub(crate) fn get_ram_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        if line.starts_with("MemTotal:") {
            let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(target_os = "windows")]
pub(crate) fn get_ram_bytes() -> Option<u64> {
    powershell_value("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")?
        .parse()
        .ok()
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
pub(crate) fn get_ram_bytes() -> Option<u64> {
    None
}

#[cfg(target_os = "windows")]
fn powershell_value(expression: &str) -> Option<String> {
    let output = std::process::Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", expression])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[cfg(target_os = "macos")]
fn sysctl_string(key: &str) -> Option<String> {
    std::process::Command::new("sysctl")
        .arg("-n")
        .arg(key)
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

#[cfg(target_os = "macos")]
fn sysctl_u64(key: &str) -> Option<u64> {
    std::process::Command::new("sysctl")
        .arg("-n")
        .arg(key)
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_memory_report_is_versioned_and_monotonic_when_available() {
        let report = process_memory_info().unwrap();
        assert_eq!(report.schema, "nimbus.studio.process-memory");
        assert_eq!(report.schema_version, 3);
        assert!(!report.measurement.is_empty());
        if let Some(peak) = report.peak_rss_bytes {
            assert!(peak > 0);
            if let Some(current) = report.current_rss_bytes {
                assert!(current > 0);
                assert!(peak >= current);
            }
        }
        #[cfg(target_os = "macos")]
        {
            assert!(report.current_rss_bytes.is_some_and(|bytes| bytes > 0));
            assert!(matches!(
                report.process_tree_coverage,
                "whole_application" | "host_plus_descendants" | "host_only"
            ));
        }
        if let Some(tree) = report.process_tree_rss_bytes {
            assert!(tree > 0);
            assert!(report.process_tree_process_count.unwrap_or(0) >= 1);
            if let Some(current) = report.current_rss_bytes {
                assert!(tree >= current);
            }
            assert!(report.process_tree_peak_rss_bytes.unwrap_or(0) >= tree);
        }
    }

    #[cfg(unix)]
    #[test]
    fn descendant_rss_includes_transitive_children_once() {
        let rows = [(10, 1, 100), (11, 10, 20), (12, 11, 3), (20, 1, 999)];
        assert_eq!(descendant_rss(10, &rows), Some((123 * 1024, 3)));
    }

    #[test]
    fn sampled_tree_peak_is_monotonic() {
        let first = sampled_tree_peak(Some(100)).unwrap();
        let second = sampled_tree_peak(Some(50)).unwrap();
        assert!(second >= first);
        assert!(sampled_tree_peak(Some(first.saturating_add(1))).unwrap() > first);
    }

    #[test]
    fn coalition_rss_includes_only_exact_resource_members() {
        let rows = [(10, 1, 100), (11, 1, 20), (12, 1, 3), (13, 1, 999)];
        let coalitions = std::collections::HashMap::from([
            (10, Some(42)),
            (11, Some(42)),
            (12, None),
            (13, Some(7)),
        ]);
        assert_eq!(
            coalition_rss(42, &rows, |pid| coalitions[&pid]),
            Some((120 * 1024, 2))
        );
        assert_eq!(coalition_rss(0, &rows, |_| Some(0)), None);
    }

    #[test]
    fn isolated_session_arithmetic_excludes_the_launcher() {
        let rows = [(10, 42, 100), (11, 42, 20), (12, 7, 999)];
        let selected = rows.iter().filter(|(_, sid, _)| *sid == 42);
        let (kib, count) = selected.fold((0_u64, 0_u32), |(bytes, count), row| {
            (bytes + row.2, count + 1)
        });
        assert_eq!((kib * 1024, count), (120 * 1024, 2));
    }
}
