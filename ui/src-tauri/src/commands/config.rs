use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::error::{AppError, AppResult};
use once_cell::sync::Lazy;
use tauri::{AppHandle, Manager};

/*
 * Config file I/O commands: general UTF-8 text read/write over IPC. Schema
 * validation is the React side's job; the Rust side keeps the I/O dumb.
 *
 * Trust model (local single-user desktop app)
 * -------------------------------------------
 * An earlier version of this comment claimed the path was confined to an
 * "allowed root." That was never implemented, and — importantly — it should
 * NOT be, given how these commands are used. `read_config` / `write_config`
 * back three flows (see ui/src/lib/tauri.ts):
 *   1. config templates under the project root (frontend-constructed paths), and
 *   2. plot specs (`.json`) + 3. Python snippets (`.py`) saved/opened via the
 *      OS-native file dialog, where the user explicitly chooses an arbitrary
 *      location (e.g. emailing a shared `.nimbus-plot.json` to ~/Desktop).
 * Confining to the project root would break (2) and (3) — the dialog IS the
 * user-consent boundary for those arbitrary paths, and a path returned by the
 * OS dialog is by definition user-authorized.
 *
 * The real security boundary for this app is therefore: (a) Tauri's capability
 * allowlist + the bundled, trusted frontend (no remote navigation; CSP set in
 * tauri.conf.json), and (b) the OS file dialogs for arbitrary-location access.
 * Anything the frontend can write here, the single user could already write
 * from a shell on their own machine.
 *
 * Defense-in-depth we DO enforce here: these commands only touch the text file
 * types they exist to handle (see ALLOWED_CONFIG_EXTENSIONS). That keeps the
 * IPC surface from being repurposed to read secrets (e.g. ~/.ssh/id_rsa) or
 * overwrite shell rc files / binaries, without breaking any legitimate flow.
 *
 * Stronger confinement (rejecting arbitrary paths outright) would require
 * moving the dialog-mediated saves behind backend-owned dialog commands so the
 * chosen path never round-trips through (potentially compromised) frontend JS;
 * that is a larger refactor of the save/open flows, flagged for follow-up.
 */

/// File extensions these commands are permitted to read/write. Every caller in
/// ui/src/lib/tauri.ts uses one of these (`.yaml`/`.yml` configs, `.json` plot
/// specs, `.py` snippets); anything else is rejected as defense-in-depth.
const ALLOWED_CONFIG_EXTENSIONS: [&str; 4] = ["yaml", "yml", "json", "py"];
/// Text IPC is intentionally bounded so an accidental multi-gigabyte selection
/// cannot allocate and synchronously parse the whole file in the webview.
const MAX_CONFIG_TEXT_BYTES: u64 = 8 * 1024 * 1024;
static WRITE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static CONFIG_WRITE_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

/// Validate an incoming path: non-empty and one of ALLOWED_CONFIG_EXTENSIONS
/// (case-insensitive). Returns the owned PathBuf on success.
fn validate_config_path(path: &str) -> AppResult<PathBuf> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(AppError::Invalid("empty path".into()));
    }
    let pb = PathBuf::from(trimmed);
    let allowed = pb
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| ALLOWED_CONFIG_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false);
    if !allowed {
        return Err(AppError::Invalid(format!(
            "refusing to access '{}': only {} files are permitted through this command",
            trimmed,
            ALLOWED_CONFIG_EXTENSIONS.join(", ")
        )));
    }
    Ok(pb)
}

#[tauri::command]
pub fn read_config(path: String) -> AppResult<String> {
    let p = validate_config_path(&path)?;
    if !p.exists() {
        return Err(AppError::NotFound(path));
    }
    let metadata = fs::metadata(&p)?;
    if !metadata.is_file() {
        return Err(AppError::Invalid(format!(
            "'{}' is not a regular file",
            p.display()
        )));
    }
    if metadata.len() > MAX_CONFIG_TEXT_BYTES {
        return Err(AppError::Invalid(format!(
            "refusing to read '{}': text files are limited to {} MiB",
            p.display(),
            MAX_CONFIG_TEXT_BYTES / (1024 * 1024)
        )));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    fs::File::open(&p)?
        .take(MAX_CONFIG_TEXT_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_CONFIG_TEXT_BYTES {
        return Err(AppError::Invalid(format!(
            "refusing to read '{}': file grew beyond the {} MiB limit",
            p.display(),
            MAX_CONFIG_TEXT_BYTES / (1024 * 1024)
        )));
    }
    String::from_utf8(bytes).map_err(|error| {
        AppError::Invalid(format!("'{}' is not valid UTF-8: {error}", p.display()))
    })
}

#[tauri::command]
pub fn write_config(path: String, contents: String) -> AppResult<()> {
    let p = validate_config_path(&path)?;
    if contents.len() as u64 > MAX_CONFIG_TEXT_BYTES {
        return Err(AppError::Invalid(format!(
            "refusing to write '{}': text files are limited to {} MiB",
            p.display(),
            MAX_CONFIG_TEXT_BYTES / (1024 * 1024)
        )));
    }
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent)?;
    }
    let _guard = CONFIG_WRITE_LOCK.lock().map_err(|error| {
        AppError::Invalid(format!("configuration write lock poisoned: {error}"))
    })?;
    write_text_atomically(&p, contents.as_bytes())
}

fn write_text_atomically(destination: &Path, bytes: &[u8]) -> AppResult<()> {
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    let name = destination
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("config");
    let temporary = parent.join(format!(
        ".{name}.nimbus-write-{}-{}.tmp",
        std::process::id(),
        WRITE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| -> AppResult<()> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        crate::platform::atomic_replace_file(&temporary, destination)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

/// Absolute root of the read-only configuration template catalog. Packaged
/// apps resolve this from Tauri resources; development resolves the checkout.
///
/// Why this matters: during `tauri dev`, the app's working directory is
/// `ui/src-tauri/`, so a JS-side write to e.g.
/// `configs/500meterResolution/kessler/baseline.yaml` lands inside
/// `src-tauri/configs/500meterResolution/kessler/baseline.yaml`. Tauri's dev watcher
/// then sees a file change in src-tauri/ and rebuilds the app —
/// killing any spawned simulator child process. Resolving template
/// paths against the real project root keeps template reads outside
/// src-tauri/. User writes use `user_config_dir` instead.
///
/// Resolution strategy: walk up from `CARGO_MANIFEST_DIR` (the
/// src-tauri/ directory at compile time) until we find a directory
/// that contains both `configs/` and `bin/`. Falls back to two-levels-up
/// if that probe fails (which mirrors the dev layout).
#[tauri::command]
pub fn project_root(app: AppHandle) -> AppResult<String> {
    if let Ok(resources) = app.path().resource_dir() {
        if resources.join("configs").is_dir() {
            return Ok(resources.display().to_string());
        }
    }
    let start = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut current: Option<&Path> = Some(start.as_path());
    while let Some(dir) = current {
        if dir.join("configs").is_dir() && dir.join("bin").is_dir() {
            return Ok(dir.display().to_string());
        }
        current = dir.parent();
    }
    // Dev-layout fallback: src-tauri/ → ui/ → project root.
    let fallback = start
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .ok_or_else(|| {
            AppError::NotFound(format!(
                "could not derive project root from {}",
                start.display()
            ))
        })?;
    if fallback.join("configs").is_dir() {
        Ok(fallback.display().to_string())
    } else {
        Err(AppError::NotFound(
            "Nimbus configuration templates are missing from the application resources".into(),
        ))
    }
}

/// Writable, application-owned directory for cloned templates and new configs.
/// It is deliberately separate from the read-only bundled template catalog.
#[tauri::command]
pub fn user_config_dir(app: AppHandle) -> AppResult<String> {
    let directory = app
        .path()
        .app_config_dir()
        .map_err(|error| {
            AppError::Invalid(format!(
                "could not resolve application config directory: {error}"
            ))
        })?
        .join("configs");
    fs::create_dir_all(&directory)?;
    Ok(directory.display().to_string())
}

#[tauri::command]
pub fn list_configs(directory: String) -> AppResult<Vec<String>> {
    let root = Path::new(&directory);
    if !root.is_dir() {
        return Err(AppError::NotFound(directory));
    }
    let mut entries = Vec::new();
    collect_config_paths(root, root, &mut entries)?;
    entries.sort();
    Ok(entries)
}

/// Recursively discover YAML while keeping the public root out of returned
/// paths. Symlinks are never followed and hidden directories are skipped, so
/// template discovery cannot escape its tree or expose internal fixture trees.
fn collect_config_paths(root: &Path, directory: &Path, entries: &mut Vec<String>) -> AppResult<()> {
    for item in fs::read_dir(directory)? {
        let item = item?;
        let file_type = item.file_type()?;
        if file_type.is_symlink() {
            continue;
        }

        let path = item.path();
        if file_type.is_dir() {
            let hidden = item
                .file_name()
                .to_str()
                .map(|name| name.starts_with('.'))
                .unwrap_or(true);
            if !hidden {
                collect_config_paths(root, &path, entries)?;
            }
            continue;
        }

        let is_yaml = path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| {
                extension.eq_ignore_ascii_case("yaml") || extension.eq_ignore_ascii_case("yml")
            })
            .unwrap_or(false);
        if !file_type.is_file() || !is_yaml {
            continue;
        }

        let relative = path.strip_prefix(root).map_err(|_| {
            AppError::Invalid(format!(
                "config path '{}' escaped discovery root '{}'",
                path.display(),
                root.display()
            ))
        })?;
        entries.push(relative.to_string_lossy().replace('\\', "/"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn accepts_supported_text_extensions() {
        // Every caller in tauri.ts: yaml/yml configs, json specs, py snippets.
        for ok in [
            "configs/500meterResolution/kessler/baseline.yaml",
            "/abs/foo.yml",
            "spec.json",
            "snippet.py",
            "shared.nimbus-plot.json", // extension() is "json"
            "MIXED.YAML",              // case-insensitive
        ] {
            assert!(
                validate_config_path(ok).is_ok(),
                "expected {ok} to be allowed"
            );
        }
    }

    #[test]
    fn rejects_non_text_and_extensionless_paths() {
        // Blast-radius cases: secrets, shell rc files, binaries, no extension.
        for bad in [
            "/Users/me/.ssh/id_rsa",
            "/Users/me/.zshrc",
            "/usr/local/bin/nimbus_sim",
            "evil.sh",
            "noextension",
            "",
            "   ",
        ] {
            assert!(
                validate_config_path(bad).is_err(),
                "expected {bad:?} to be rejected"
            );
        }
    }

    #[test]
    fn config_listing_is_recursive_relative_sorted_and_hidden_safe() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "nimbus_config_listing_{}_{}",
            std::process::id(),
            nonce
        ));
        fs::create_dir_all(root.join("500meterResolution/nested")).expect("nested tree");
        fs::create_dir_all(root.join(".internal")).expect("hidden tree");
        fs::write(root.join("top.yml"), "schema: nimbus.v1\n").expect("top config");
        fs::write(
            root.join("500meterResolution/nested/custom.YAML"),
            "schema: nimbus.v1\n",
        )
        .expect("nested config");
        fs::write(root.join("500meterResolution/notes.txt"), "not yaml\n").expect("non-config");
        fs::write(root.join(".internal/secret.yaml"), "schema: nimbus.v1\n")
            .expect("hidden config");

        let listed = list_configs(root.display().to_string()).expect("list configs");
        fs::remove_dir_all(&root).expect("remove test tree");

        assert_eq!(
            listed,
            vec![
                "500meterResolution/nested/custom.YAML".to_string(),
                "top.yml".to_string(),
            ]
        );
    }

    #[test]
    fn config_writes_replace_atomically_without_leaving_a_temporary_file() {
        let root =
            std::env::temp_dir().join(format!("nimbus_atomic_config_write_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let target = root.join("experiment.yaml");
        fs::write(&target, "old\n").unwrap();
        write_config(target.display().to_string(), "schema: nimbus.v1\n".into()).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "schema: nimbus.v1\n");
        assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn oversized_reads_and_writes_fail_closed() {
        let root =
            std::env::temp_dir().join(format!("nimbus_bounded_config_io_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let target = root.join("oversized.yaml");
        let file = fs::File::create(&target).unwrap();
        file.set_len(MAX_CONFIG_TEXT_BYTES + 1).unwrap();
        assert!(read_config(target.display().to_string()).is_err());
        assert!(write_config(
            target.display().to_string(),
            "x".repeat(MAX_CONFIG_TEXT_BYTES as usize + 1)
        )
        .is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
