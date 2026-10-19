use std::fs;
use std::path::PathBuf;
use std::time::UNIX_EPOCH;

use serde::Serialize;

use crate::error::{AppError, AppResult};

/*
 * Plugin discovery — lists Python snippet files under ~/.nimbus/plugins/.
 *
 * The directory is auto-created on first call so a fresh install doesn't
 * surface a "directory not found" error; the user just sees an empty
 * gallery and can save a new snippet to populate it.
 *
 * Recursion depth: top-level only. Subdirectories are skipped on
 * purpose — the gallery is a flat list and trees would complicate the
 * UI without a clear win for v1.
 *
 * Native user-directory resolution uses HOME on Unix and USERPROFILE on
 * Windows so the same ~/.nimbus/plugins contract is preserved.
 */

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PluginEntry {
    /// Display name (filename including .py extension).
    pub name: String,
    /// Absolute path on disk — handed back to read_config / write_config
    /// for load + save.
    pub path: String,
    /// File size in bytes, for "1.2 KB" style display.
    pub size_bytes: u64,
    /// Last-modified unix ms. UI shows "edited 3 min ago" style relative
    /// time. None when the FS doesn't expose mtime (rare on macOS/Linux).
    pub modified_ms: Option<u128>,
}

/// Resolve the plugins directory (~/.nimbus/plugins/), creating it if
/// it doesn't yet exist. Returns the absolute path so the UI can pass
/// it to openPath() for the "Open dir" button.
#[tauri::command]
pub fn plugins_dir() -> AppResult<String> {
    let dir = resolve_dir()?;
    fs::create_dir_all(&dir)?;
    Ok(dir.display().to_string())
}

/// List `.py` files in the plugins directory. Sorted by mtime descending
/// (newest first) so the most recently edited plugin appears at the top.
#[tauri::command]
pub fn list_plugins() -> AppResult<Vec<PluginEntry>> {
    let dir = resolve_dir()?;
    fs::create_dir_all(&dir)?;

    let mut entries: Vec<PluginEntry> = fs::read_dir(&dir)?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let path = e.path();
            if !path.is_file() {
                return None;
            }
            if path.extension().and_then(|x| x.to_str()) != Some("py") {
                return None;
            }
            let name = path.file_name()?.to_string_lossy().to_string();
            let meta = e.metadata().ok()?;
            let modified_ms = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_millis());
            Some(PluginEntry {
                name,
                path: path.display().to_string(),
                size_bytes: meta.len(),
                modified_ms,
            })
        })
        .collect();

    // mtime descending; ties broken by name ascending for stable order.
    entries.sort_by(|a, b| {
        b.modified_ms
            .unwrap_or(0)
            .cmp(&a.modified_ms.unwrap_or(0))
            .then_with(|| a.name.cmp(&b.name))
    });
    Ok(entries)
}

fn resolve_dir() -> AppResult<PathBuf> {
    let home = crate::platform::home_dir().ok_or_else(|| {
        AppError::NotFound("user home directory is unavailable; cannot resolve plugins dir".into())
    })?;
    Ok(home.join(".nimbus").join("plugins"))
}

/*
 * Open a file or directory in the OS's default handler — Finder on macOS,
 * the file manager on Linux, Explorer on Windows. We do this in Rust
 * instead of going through @tauri-apps/plugin-shell's open() because that
 * plugin's default scope only accepts URLs matching a mailto/tel/http
 * regex; arbitrary filesystem paths get rejected at the bridge.
 *
 * Each branch shells out to the platform's canonical "open this" command:
 *   - macOS:  `open <path>` — Apple's official desktop services helper
 *   - Linux:  `xdg-open <path>` — freedesktop.org convention
 *   - Windows: `explorer <path>` — relies on Explorer's quirky CLI but is
 *              the simplest approach without taking a winapi dep
 *
 * The spawned process is detached: we don't await it, just verify the
 * spawn succeeded. The user closing the spawned file manager later does
 * not affect this process.
 */
#[tauri::command]
pub fn reveal_path(path: String) -> AppResult<()> {
    let p = std::path::Path::new(&path);
    if !p.exists() {
        return Err(AppError::NotFound(path));
    }

    let result;
    #[cfg(target_os = "macos")]
    {
        result = std::process::Command::new("open").arg(&path).spawn();
    }
    #[cfg(target_os = "linux")]
    {
        result = std::process::Command::new("xdg-open").arg(&path).spawn();
    }
    #[cfg(target_os = "windows")]
    {
        result = std::process::Command::new("explorer").arg(&path).spawn();
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        return Err(AppError::Subprocess(
            "reveal_path not implemented on this platform".into(),
        ));
    }

    result.map_err(|e| AppError::Subprocess(format!("reveal {path}: {e}")))?;
    Ok(())
}
