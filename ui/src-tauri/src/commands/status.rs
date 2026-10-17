use std::env;
use std::path::PathBuf;

use serde::Serialize;
use tauri::State;

use crate::error::AppResult;
use crate::state::AppState;

/*
 * Probe + report the simulator binary's status. A non-empty Settings
 * override is authoritative. Without one, the default lookup order is:
 *   1. simulator bundled into a packaged Nimbus Studio build
 *   2. NIMBUS_SIM_BIN env var
 *   3. bin/nimbus_sim below cwd or one of its ancestors
 *
 * The `gpu` field is the *physical* device name (e.g. "Apple M4 Max"),
 * detected via system_profiler (macOS) or lspci (Linux). Whether the
 * simulator actually uses the GPU at runtime is a per-config decision
 * (`numerics.compute.backend: vulkan` enables it). The UI surfaces the
 * device here so the user knows what hardware is available; the backend
 * choice lives in the config editor.
 */

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SimulatorStatus {
    pub binary_path: Option<String>,
    pub state: SimState,
    pub gpu: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SimState {
    Ready,
    Missing,
    /// Reserved for the path-set-but-not-yet-probed state used by the
    /// Settings panel handshake in Phase 1.E.
    #[allow(dead_code)]
    Unknown,
}

#[tauri::command]
pub fn simulator_status(
    state: State<'_, AppState>,
    simulator_path: Option<String>,
) -> AppResult<SimulatorStatus> {
    let explicit_override = simulator_path
        .as_deref()
        .map(str::trim)
        .filter(|path| !path.is_empty());
    let cached = state
        .simulator_binary
        .lock()
        .ok()
        .and_then(|cached| cached.clone());
    let path = resolve_simulator_status_path_with_cached(explicit_override, cached);
    // Settings overrides are request-scoped. In particular, a typo must not
    // erase the packaged fallback cached during application setup; clearing
    // the field should immediately return to the bundled simulator.
    if explicit_override.is_none() {
        if let Ok(mut cache) = state.simulator_binary.lock() {
            *cache = path.clone();
        }
    }

    let gpu = cached_gpu(&state);

    let resp = match &path {
        Some(p) => SimulatorStatus {
            binary_path: Some(p.display().to_string()),
            state: SimState::Ready,
            gpu,
        },
        None => SimulatorStatus {
            binary_path: None,
            state: SimState::Missing,
            gpu,
        },
    };
    Ok(resp)
}

fn resolve_simulator_status_path(override_path: Option<&str>) -> Option<PathBuf> {
    if let Some(path) = override_path.map(str::trim).filter(|path| !path.is_empty()) {
        let candidate = crate::platform::expand_tilde(path);
        return candidate.is_file().then_some(candidate);
    }
    locate_simulator()
}

fn resolve_simulator_status_path_with_cached(
    override_path: Option<&str>,
    cached: Option<PathBuf>,
) -> Option<PathBuf> {
    if override_path.is_some() {
        return resolve_simulator_status_path(override_path);
    }
    cached
        .filter(|candidate| candidate.is_file())
        .or_else(locate_simulator)
}

pub(crate) fn locate_simulator() -> Option<PathBuf> {
    if let Ok(env_path) = env::var("NIMBUS_SIM_BIN") {
        let p = PathBuf::from(env_path);
        if p.exists() {
            return Some(p);
        }
    }
    // Walk up from cwd looking for the native simulator executable.
    let cwd = env::current_dir().ok()?;
    for ancestor in cwd.ancestors() {
        let candidate = crate::platform::simulator_below(ancestor);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

/// Memoized GPU device probe. First call shells out to the platform tool;
/// subsequent calls return the cached value so polling stays cheap.
fn cached_gpu(state: &State<'_, AppState>) -> Option<String> {
    if let Ok(cache) = state.gpu_probe.lock() {
        if let Some(cached) = cache.as_ref() {
            return cached.clone();
        }
    }
    let probed = detect_gpu_device();
    if let Ok(mut cache) = state.gpu_probe.lock() {
        *cache = Some(probed.clone());
    }
    probed
}

#[cfg(target_os = "macos")]
fn detect_gpu_device() -> Option<String> {
    let output = std::process::Command::new("system_profiler")
        .args(["-detailLevel", "mini", "SPDisplaysDataType"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("Chipset Model:") {
            let name = rest.trim().to_string();
            if !name.is_empty() {
                return Some(name);
            }
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn detect_gpu_device() -> Option<String> {
    // lspci -mm prints quoted fields: BDF "Class" "Vendor" "Device" ["SVendor" "SDevice"]
    let output = std::process::Command::new("lspci")
        .arg("-mm")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    for line in text.lines() {
        let lower = line.to_ascii_lowercase();
        let is_gpu_class = lower.contains("\"vga ")
            || lower.contains("\"3d ")
            || lower.contains("vga compatible")
            || lower.contains("\"display ");
        if !is_gpu_class {
            continue;
        }
        let fields: Vec<&str> = line.split('"').collect();
        // Indices: 0=BDF+ws, 1="Class", 2=ws, 3="Vendor", 4=ws, 5="Device"
        if fields.len() >= 6 {
            let vendor = fields[3].trim();
            let device = fields[5].trim();
            if !device.is_empty() {
                return Some(if vendor.is_empty() {
                    device.to_string()
                } else {
                    format!("{} {}", vendor, device)
                });
            }
        }
    }
    None
}

#[cfg(target_os = "windows")]
fn detect_gpu_device() -> Option<String> {
    let output = std::process::Command::new("powershell.exe")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "(Get-CimInstance Win32_VideoController | Where-Object {$_.Name} | Select-Object -First 1 -ExpandProperty Name)",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty())
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
fn detect_gpu_device() -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::{resolve_simulator_status_path, resolve_simulator_status_path_with_cached};

    #[test]
    fn explicit_status_override_must_name_an_existing_file() {
        let path =
            std::env::temp_dir().join(format!("nimbus-status-override-{}", std::process::id()));
        std::fs::write(&path, b"fixture").expect("create override fixture");
        assert_eq!(
            resolve_simulator_status_path(path.to_str()),
            Some(path.clone())
        );
        std::fs::remove_file(&path).expect("remove override fixture");
        assert_eq!(resolve_simulator_status_path(path.to_str()), None);
    }

    #[test]
    fn clearing_a_settings_override_returns_to_the_packaged_simulator() {
        let path = std::env::temp_dir().join(format!(
            "nimbus-status-packaged-fallback-{}",
            std::process::id()
        ));
        std::fs::write(&path, b"fixture").expect("create packaged simulator fixture");
        assert_eq!(
            resolve_simulator_status_path_with_cached(None, Some(path.clone())),
            Some(path.clone())
        );
        assert_eq!(
            resolve_simulator_status_path_with_cached(
                Some("/missing/nimbus_sim"),
                Some(path.clone())
            ),
            None,
            "an explicit invalid override remains authoritative for that request"
        );
        std::fs::remove_file(path).expect("remove packaged simulator fixture");
    }
}
