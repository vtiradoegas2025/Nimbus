#[cfg(windows)]
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::io::{BufReader, Read};

use sha2::{Digest, Sha256};

#[cfg(windows)]
use std::os::windows::ffi::OsStrExt;

pub(crate) fn home_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var_os("USERPROFILE")
            .or_else(|| {
                let drive = std::env::var_os("HOMEDRIVE")?;
                let path = std::env::var_os("HOMEPATH")?;
                let mut combined = OsString::from(drive);
                combined.push(path);
                Some(combined)
            })
            .map(PathBuf::from)
    }
    #[cfg(not(windows))]
    {
        std::env::var_os("HOME").map(PathBuf::from)
    }
}

pub(crate) fn expand_tilde(path: &str) -> PathBuf {
    if path == "~" {
        return home_dir().unwrap_or_else(|| PathBuf::from(path));
    }
    if let Some(rest) = path.strip_prefix("~/").or_else(|| path.strip_prefix("~\\")) {
        if let Some(home) = home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(path)
}

pub(crate) const fn simulator_executable_name() -> &'static str {
    if cfg!(windows) {
        "nimbus_sim.exe"
    } else {
        "nimbus_sim"
    }
}

pub(crate) fn simulator_below(root: &Path) -> PathBuf {
    root.join("bin").join(simulator_executable_name())
}

pub(crate) fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut reader = BufReader::new(std::fs::File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

/// Replace an existing publication in one atomic filesystem operation.
///
/// Rust's `rename` maps to replacement semantics on POSIX, while Windows
/// requires `MoveFileExW(MOVEFILE_REPLACE_EXISTING)`.  The desktop bridge uses
/// this for files that readers may already have discovered, so delete+rename
/// is intentionally forbidden: it creates a visible gap.
pub(crate) fn atomic_replace_file(source: &Path, destination: &Path) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::Storage::FileSystem::{
            MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
        };

        let source_wide: Vec<u16> = source
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        let destination_wide: Vec<u16> = destination
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        let result = unsafe {
            MoveFileExW(
                source_wide.as_ptr(),
                destination_wide.as_ptr(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
            )
        };
        if result == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
    #[cfg(not(windows))]
    {
        std::fs::rename(source, destination)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulator_name_matches_the_native_executable_convention() {
        assert_eq!(
            simulator_executable_name(),
            if cfg!(windows) {
                "nimbus_sim.exe"
            } else {
                "nimbus_sim"
            }
        );
    }

    #[test]
    fn ordinary_paths_are_not_rewritten() {
        assert_eq!(expand_tilde("relative/run"), PathBuf::from("relative/run"));
    }

    #[test]
    fn atomic_replace_overwrites_an_existing_publication() {
        let root =
            std::env::temp_dir().join(format!("nimbus-platform-replace-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("create replacement test directory");
        let source = root.join("manifest.tmp");
        let destination = root.join("manifest.json");
        std::fs::write(&destination, b"old").expect("write old publication");
        std::fs::write(&source, b"new").expect("write new publication");
        atomic_replace_file(&source, &destination).expect("atomically replace publication");
        assert_eq!(
            std::fs::read(&destination).expect("read publication"),
            b"new"
        );
        assert!(!source.exists());
        std::fs::remove_dir_all(root).expect("remove replacement test directory");
    }

    #[test]
    fn file_sha256_binds_exact_bytes() {
        let path = std::env::temp_dir().join(format!(
            "nimbus-sha256-{}.bin",
            std::process::id()
        ));
        std::fs::write(&path, b"abc").unwrap();
        assert_eq!(
            sha256_file(&path).unwrap(),
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        std::fs::remove_file(path).unwrap();
    }
}
