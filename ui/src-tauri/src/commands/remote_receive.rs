use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter, Manager};

use super::provenance::{read_validated_run_provenance, RunProvenance};
use super::remote_compute::{RemoteFile, RemoteRun};
use crate::error::{AppError, AppResult};
use crate::state::{AppState, RunStatus};

const RECEIVE_FILE: &str = "remote-receive.json";
const MAX_RECEIPT_BYTES: u64 = 16 * 1024;
const MAX_RECEIVES: usize = 16;
const INVENTORY_FILE: &str = "remote-receive-inventory.json";
const MAX_INVENTORY_BYTES: u64 = 20 * 1024 * 1024;
static RECEIVES: OnceLock<Mutex<HashMap<PathBuf, RemoteRun>>> = OnceLock::new();
static VALIDATED_INVENTORIES: OnceLock<Mutex<ValidatedInventoryCache>> = OnceLock::new();

struct ValidatedInventoryEntry {
    inventory_digest: String,
    file_identity: String,
    accessed: u64,
}

#[derive(Default)]
struct ValidatedInventoryCache {
    entries: HashMap<PathBuf, ValidatedInventoryEntry>,
    clock: u64,
}

impl ValidatedInventoryCache {
    fn matches(&mut self, path: &Path, inventory_digest: &str, file_identity: &str) -> bool {
        let Some(entry) = self.entries.get_mut(path) else {
            return false;
        };
        if entry.inventory_digest != inventory_digest || entry.file_identity != file_identity {
            return false;
        }
        self.clock = self.clock.saturating_add(1);
        entry.accessed = self.clock;
        true
    }

    fn insert(&mut self, path: PathBuf, inventory_digest: String, file_identity: String) {
        if !self.entries.contains_key(&path)
            && self.entries.len() >= super::run_library::MAX_RUN_DIRECTORIES
        {
            if let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.accessed)
                .map(|(path, _)| path.clone())
            {
                self.entries.remove(&oldest);
            }
        }
        self.clock = self.clock.saturating_add(1);
        self.entries.insert(
            path,
            ValidatedInventoryEntry {
                inventory_digest,
                file_identity,
                accessed: self.clock,
            },
        );
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RemoteReceiveSummary {
    pub state: String,
    pub error: Option<String>,
    pub retry_available: bool,
    pub updated_unix_ms: u128,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ReceiveReceipt {
    schema: String,
    schema_version: u32,
    run_id: String,
    configuration_digest: String,
    execution_target_fingerprint: String,
    remote_output_directory: String,
    summary: RemoteReceiveSummary,
    #[serde(default)]
    inventory_sha256: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ReceiveInventory {
    schema: String,
    schema_version: u32,
    run_id: String,
    configuration_digest: String,
    execution_target_fingerprint: String,
    remote_output_directory: String,
    launch_sha256: String,
    original_runtime_sha256: Option<String>,
    files: Vec<RemoteFile>,
}

fn receives() -> &'static Mutex<HashMap<PathBuf, RemoteRun>> {
    RECEIVES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}
fn configuration_digest(launch: &RunProvenance) -> String {
    format!(
        "{}:{}",
        launch.configuration.digest.algorithm, launch.configuration.digest.value
    )
}

fn summary(state: &str, error: Option<String>) -> RemoteReceiveSummary {
    RemoteReceiveSummary {
        state: state.into(),
        error,
        retry_available: state != "receiving" && state != "invalid",
        updated_unix_ms: now_ms(),
    }
}

pub fn remote_receive_summary_with_provenance(
    run_dir: &Path,
    launch: &RunProvenance,
) -> Option<RemoteReceiveSummary> {
    let remote = launch.remote_execution.as_ref()?;
    if let Err(error) = RemoteRun::from_execution(remote) {
        return Some(summary("invalid", Some(error.to_string())));
    }
    let normalized = fs::canonicalize(run_dir).unwrap_or_else(|_| run_dir.to_path_buf());
    let active = receives()
        .lock()
        .map(|active| active.contains_key(&normalized))
        .unwrap_or(false);
    let path = run_dir.join(RECEIVE_FILE);
    if !path.exists() {
        let mut value = summary(if active { "receiving" } else { "available" }, None);
        value.updated_unix_ms = launch.created_unix_ms;
        return Some(value);
    }
    let read = (|| -> AppResult<ReceiveReceipt> {
        let metadata = fs::symlink_metadata(&path)?;
        if !metadata.file_type().is_file() || metadata.len() > MAX_RECEIPT_BYTES {
            return Err(AppError::Invalid(
                "remote receive receipt must be a bounded regular file".into(),
            ));
        }
        let receipt: ReceiveReceipt = serde_json::from_slice(&fs::read(&path)?)
            .map_err(|error| AppError::Invalid(format!("invalid receive receipt: {error}")))?;
        if receipt.schema != "nimbus.studio.remote-receive"
            || receipt.schema_version != 1
            || receipt.run_id != launch.run_id
            || receipt.configuration_digest != configuration_digest(launch)
            || receipt.execution_target_fingerprint != remote.execution_target_fingerprint
            || receipt.remote_output_directory != remote.remote_output_directory
            || !matches!(
                receipt.summary.state.as_str(),
                "receiving" | "complete" | "failed" | "cancelled"
            )
            || receipt
                .summary
                .error
                .as_ref()
                .is_some_and(|error| error.len() > 4096)
            || receipt.summary.updated_unix_ms < launch.created_unix_ms
        {
            return Err(AppError::Invalid(
                "remote receive receipt disagrees with launch provenance".into(),
            ));
        }
        Ok(receipt)
    })();
    match read {
        Ok(mut receipt) => {
            if active {
                receipt.summary.state = "receiving".into();
                receipt.summary.retry_available = false;
            } else if receipt.summary.state == "receiving" {
                receipt.summary.state = "cancelled".into();
                receipt.summary.retry_available = true;
                receipt.summary.error = Some("The prior Studio session ended while receiving results. Retry receiving the retained remote output.".into());
            } else {
                receipt.summary.retry_available = true;
            }
            Some(receipt.summary)
        }
        Err(error) => Some(summary("invalid", Some(error.to_string()))),
    }
}

pub fn remote_receive_summary(run_dir: &Path) -> Option<RemoteReceiveSummary> {
    let launch = read_validated_run_provenance(run_dir).ok()?;
    remote_receive_summary_with_provenance(run_dir, &launch)
}

pub(crate) fn record_receive_state(
    run_dir: &Path,
    launch: &RunProvenance,
    state: &str,
    error: Option<String>,
) -> AppResult<RemoteReceiveSummary> {
    let remote = launch
        .remote_execution
        .as_ref()
        .ok_or_else(|| AppError::Invalid("run has no remote launch provenance".into()))?;
    let summary = summary(state, error.map(|error| error.chars().take(1024).collect()));
    let receipt = ReceiveReceipt {
        schema: "nimbus.studio.remote-receive".into(),
        schema_version: 1,
        run_id: launch.run_id.clone(),
        configuration_digest: configuration_digest(launch),
        execution_target_fingerprint: remote.execution_target_fingerprint.clone(),
        remote_output_directory: remote.remote_output_directory.clone(),
        summary: summary.clone(),
        inventory_sha256: if state == "complete" {
            Some(crate::platform::sha256_file(&run_dir.join(INVENTORY_FILE))?)
        } else {
            None
        },
    };
    let bytes = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| AppError::Invalid(error.to_string()))?;
    if bytes.len() as u64 > MAX_RECEIPT_BYTES {
        return Err(AppError::Invalid(
            "remote receive receipt exceeds its byte bound".into(),
        ));
    }
    let temporary = run_dir.join(format!(
        ".remote-receive-{}-{}.pending",
        std::process::id(),
        now_ms()
    ));
    let result = (|| -> AppResult<()> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, run_dir.join(RECEIVE_FILE))?;
        Ok(())
    })();
    let _ = fs::remove_file(&temporary);
    result?;
    Ok(summary)
}

struct ReceiveLock {
    _file: File,
}
fn lock_receive(run_dir: &Path) -> AppResult<ReceiveLock> {
    let path = run_dir.join(".remote-receive.lock");
    if fs::symlink_metadata(&path).is_ok_and(|metadata| !metadata.file_type().is_file()) {
        return Err(AppError::Invalid(
            "remote receive lock must be a regular file".into(),
        ));
    }
    let mut options = OpenOptions::new();
    options.create(true).read(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.share_mode(0);
    }
    let file = options.open(path).map_err(|error| {
        AppError::Invalid(format!("another receive may already own this run: {error}"))
    })?;
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err(AppError::Invalid(
                "another receive already owns this run".into(),
            ));
        }
    }
    Ok(ReceiveLock { _file: file })
}

struct ActiveReceive(PathBuf);
impl Drop for ActiveReceive {
    fn drop(&mut self) {
        if let Ok(mut active) = receives().lock() {
            active.remove(&self.0);
        }
    }
}

pub(crate) fn receive_with_run(
    run_dir: &Path,
    launch: &RunProvenance,
    remote: RemoteRun,
) -> AppResult<RemoteReceiveSummary> {
    receive_with_run_progress(run_dir, launch, remote, || {})
}

fn receive_with_run_progress(
    run_dir: &Path,
    launch: &RunProvenance,
    remote: RemoteRun,
    progress: impl Fn(),
) -> AppResult<RemoteReceiveSummary> {
    let run_dir = fs::canonicalize(run_dir)?;
    let _lock = lock_receive(&run_dir)?;
    {
        let mut active = receives()
            .lock()
            .map_err(|error| AppError::Task(error.to_string()))?;
        if active.contains_key(&run_dir) || active.len() >= MAX_RECEIVES {
            return Err(AppError::Invalid(
                "remote receiving is already active or its concurrency bound is reached".into(),
            ));
        }
        active.insert(run_dir.clone(), remote.clone());
    }
    let _active = ActiveReceive(run_dir.clone());
    record_receive_state(&run_dir, launch, "receiving", None)?;
    progress();
    let launch_sha256 =
        crate::platform::sha256_file(&run_dir.join(super::provenance::PROVENANCE_FILE_NAME))?;
    let runtime_path = run_dir.join(super::provenance::RUNTIME_PROVENANCE_FILE_NAME);
    let original_runtime_sha256 = if runtime_path.exists() {
        Some(crate::platform::sha256_file(&runtime_path)?)
    } else {
        None
    };
    let result = remote
        .confirm_stopped()
        .and_then(|()| remote.mirror_outputs_checked(&run_dir));
    match result {
        Ok(files) => {
            let execution = launch.remote_execution.as_ref().unwrap();
            let inventory = ReceiveInventory {
                schema: "nimbus.studio.remote-receive-inventory".into(),
                schema_version: 1,
                run_id: launch.run_id.clone(),
                configuration_digest: configuration_digest(launch),
                execution_target_fingerprint: execution.execution_target_fingerprint.clone(),
                remote_output_directory: execution.remote_output_directory.clone(),
                launch_sha256,
                original_runtime_sha256,
                files,
            };
            let bytes = serde_json::to_vec(&inventory)
                .map_err(|error| AppError::Invalid(error.to_string()))?;
            if bytes.len() as u64 > MAX_INVENTORY_BYTES {
                return Err(AppError::Invalid(
                    "received inventory exceeds its byte bound".into(),
                ));
            }
            let temporary = run_dir.join(".remote-receive-inventory.pending");
            let mut output = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&temporary)?;
            output.write_all(&bytes)?;
            output.sync_all()?;
            drop(output);
            fs::rename(&temporary, run_dir.join(INVENTORY_FILE))?;
            record_receive_state(&run_dir, launch, "complete", None)
        }
        Err(error) => {
            let state = if remote.receive_cancelled() {
                "cancelled"
            } else {
                "failed"
            };
            record_receive_state(&run_dir, launch, state, Some(error.to_string()))
        }
    }
}

pub fn validate_recovered_remote_outputs(
    run_dir: &Path,
    launch: &RunProvenance,
    original_runtime_digest: &str,
) -> AppResult<bool> {
    let execution = match &launch.remote_execution {
        Some(execution) => execution,
        None => return Ok(false),
    };
    if remote_receive_summary_with_provenance(run_dir, launch)
        .map_or(true, |summary| summary.state != "complete")
    {
        return Ok(false);
    }
    let receipt: ReceiveReceipt = serde_json::from_slice(&fs::read(run_dir.join(RECEIVE_FILE))?)
        .map_err(|error| AppError::Invalid(error.to_string()))?;
    let inventory_path = run_dir.join(INVENTORY_FILE);
    let metadata = fs::symlink_metadata(&inventory_path)?;
    if !metadata.file_type().is_file() || metadata.len() > MAX_INVENTORY_BYTES {
        return Err(AppError::Invalid(
            "recovery inventory must be a bounded regular file".into(),
        ));
    }
    let bytes = fs::read(&inventory_path)?;
    let inventory_digest = format!("sha256:{:x}", Sha256::digest(&bytes));
    if receipt.inventory_sha256.as_deref() != Some(&inventory_digest) {
        return Err(AppError::Invalid(
            "recovery inventory digest disagrees with its receipt".into(),
        ));
    }
    let inventory: ReceiveInventory =
        serde_json::from_slice(&bytes).map_err(|error| AppError::Invalid(error.to_string()))?;
    if inventory.schema != "nimbus.studio.remote-receive-inventory"
        || inventory.schema_version != 1
        || inventory.run_id != launch.run_id
        || inventory.configuration_digest != configuration_digest(launch)
        || inventory.execution_target_fingerprint != execution.execution_target_fingerprint
        || inventory.remote_output_directory != execution.remote_output_directory
        || inventory.original_runtime_sha256.as_deref() != Some(original_runtime_digest)
        || inventory.launch_sha256
            != crate::platform::sha256_file(&run_dir.join(super::provenance::PROVENANCE_FILE_NAME))?
        || inventory.files.len() > 100_000
    {
        return Err(AppError::Invalid(
            "recovery inventory is not bound to the original launch and runtime evidence".into(),
        ));
    }
    let mut identity = Sha256::new();
    let mut expected = std::collections::HashSet::new();
    let mut roots = std::collections::HashSet::new();
    for file in &inventory.files {
        let path = Path::new(&file.path);
        if file.path.is_empty()
            || path
                .components()
                .any(|part| !matches!(part, std::path::Component::Normal(_)))
            || path
                .components()
                .any(|part| part.as_os_str().to_string_lossy().starts_with('.'))
            || !expected.insert(file.path.clone())
            || file.sha256.len() != 64
            || !file.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(AppError::Invalid(
                "invalid recovery inventory artifact".into(),
            ));
        }
        roots.insert(file.path.split('/').next().unwrap().to_string());
        let metadata = fs::symlink_metadata(run_dir.join(path))?;
        if !metadata.file_type().is_file() || metadata.len() != file.byte_length {
            return Err(AppError::Invalid(
                "received artifact shape or length changed".into(),
            ));
        }
        identity.update(file.path.as_bytes());
        identity.update(metadata.len().to_le_bytes());
        identity.update(
            metadata
                .modified()?
                .duration_since(UNIX_EPOCH)
                .map_err(|error| AppError::Invalid(error.to_string()))?
                .as_nanos()
                .to_le_bytes(),
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            identity.update(metadata.dev().to_le_bytes());
            identity.update(metadata.ino().to_le_bytes());
            identity.update(metadata.ctime().to_le_bytes());
            identity.update(metadata.ctime_nsec().to_le_bytes());
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            identity.update(metadata.creation_time().to_le_bytes());
        }
    }
    let mut pending = Vec::new();
    for entry in fs::read_dir(run_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if roots.contains(&name) || name.starts_with("step_") || name == "checkpoints" {
            pending.push((entry.path(), name));
        }
    }
    let mut checked = 0;
    while let Some((path, relative)) = pending.pop() {
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_dir() {
            for entry in fs::read_dir(&path)? {
                let entry = entry?;
                pending.push((
                    entry.path(),
                    format!("{relative}/{}", entry.file_name().to_string_lossy()),
                ));
            }
        } else if metadata.file_type().is_file() && expected.contains(&relative) {
            checked += 1;
        } else {
            return Err(AppError::Invalid(
                "current received output/checkpoint inventory differs from recovery evidence"
                    .into(),
            ));
        }
        if checked + pending.len() > 100_000 {
            return Err(AppError::Invalid(
                "current recovery inventory exceeds its file bound".into(),
            ));
        }
    }
    if checked != expected.len() {
        return Err(AppError::Invalid(
            "received artifacts are missing from recovery evidence".into(),
        ));
    }
    let identity = format!("{:x}", identity.finalize());
    let cache =
        VALIDATED_INVENTORIES.get_or_init(|| Mutex::new(ValidatedInventoryCache::default()));
    let key = fs::canonicalize(run_dir)?;
    if cache
        .lock()
        .map_err(|error| AppError::Task(error.to_string()))?
        .matches(&key, &inventory_digest, &identity)
    {
        return Ok(true);
    }
    for file in &inventory.files {
        if crate::platform::sha256_file(&run_dir.join(&file.path))?
            != format!("sha256:{}", file.sha256)
        {
            return Err(AppError::Invalid(
                "received artifact SHA256 differs from recovery evidence".into(),
            ));
        }
    }
    let mut cache = cache
        .lock()
        .map_err(|error| AppError::Task(error.to_string()))?;
    cache.insert(key, inventory_digest, identity);
    Ok(true)
}

pub(crate) fn receive_initial_results(remote: &RemoteRun, run_dir: &Path) -> AppResult<()> {
    let launch = read_validated_run_provenance(run_dir)?;
    let result = receive_with_run(run_dir, &launch, remote.clone())?;
    if result.state != "complete" {
        return Err(AppError::Invalid(
            result
                .error
                .unwrap_or_else(|| "Remote receiving did not complete".into()),
        ));
    }
    Ok(())
}

#[tauri::command]
pub async fn receive_remote_results(
    app: AppHandle,
    run_dir: String,
) -> AppResult<RemoteReceiveSummary> {
    tauri::async_runtime::spawn_blocking(move || {
        let path = fs::canonicalize(&run_dir)?;
        let launch = read_validated_run_provenance(&path)?;
        let state = app.state::<AppState>();
        if state.runs.lock().map_err(|error| AppError::Task(error.to_string()))?.values().any(|record| record.lock().map(|record|
            fs::canonicalize(&record.output_dir).ok().as_ref() == Some(&path)
                && (matches!(record.status, RunStatus::Running) || record.remote_transfer_in_progress || record.remote_termination_unconfirmed)).unwrap_or(true)) {
            return Err(AppError::Invalid("The simulation or its initial result transfer has not finished. Stop and confirm it before receiving again.".into()));
        }
        let execution = launch.remote_execution.as_ref().ok_or_else(|| AppError::Invalid("run was not launched on remote compute".into()))?;
        let remote = RemoteRun::from_execution(execution)?;
        let result = receive_with_run_progress(&path, &launch, remote, || {
            let _ = app.emit("run:receive", serde_json::json!({"runDir":path.display().to_string()}));
        });
        let _ = app.emit("run:receive", serde_json::json!({"runDir":path.display().to_string()}));
        result
    }).await.map_err(|error| AppError::Task(error.to_string()))?
}

#[tauri::command]
pub fn cancel_remote_receive(run_dir: String) -> AppResult<()> {
    let path = fs::canonicalize(run_dir)?;
    let active = receives()
        .lock()
        .map_err(|error| AppError::Task(error.to_string()))?;
    if let Some(remote) = active.get(&path) {
        remote.cancel_receive();
    }
    Ok(())
}

pub fn cancel_all_receives() {
    if let Ok(active) = receives().lock() {
        for remote in active.values() {
            remote.cancel_receive();
        }
    }
    let deadline = std::time::Instant::now() + Duration::from_secs(1);
    while std::time::Instant::now() < deadline {
        if receives()
            .lock()
            .map(|active| active.is_empty())
            .unwrap_or(true)
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovered_archive_scan_keeps_all_unchanged_inventory_digests_warm() {
        let mut cache = ValidatedInventoryCache::default();
        let width = super::super::run_library::MAX_RUN_DIRECTORIES;
        assert!(width > 16);
        for index in 0..width {
            cache.insert(
                PathBuf::from(format!("/archive/{index}")),
                format!("inventory:{index}"),
                format!("identity:{index}"),
            );
        }
        for _scan in 0..3 {
            for index in 0..width {
                assert!(
                    cache.matches(
                        Path::new(&format!("/archive/{index}")),
                        &format!("inventory:{index}"),
                        &format!("identity:{index}")
                    ),
                    "unchanged archive {index} would require a redundant full-output hash"
                );
            }
        }
        assert_eq!(cache.entries.len(), width);
        assert!(!cache.matches(
            Path::new("/archive/17"),
            "inventory:17",
            "changed-file-identity"
        ));
        assert!(!cache.matches(
            Path::new("/archive/18"),
            "changed-receipt-digest",
            "identity:18"
        ));
        cache.insert(
            PathBuf::from("/archive/17"),
            "inventory:17".into(),
            "changed-file-identity".into(),
        );
        assert!(cache.matches(
            Path::new("/archive/17"),
            "inventory:17",
            "changed-file-identity"
        ));
        assert!(cache.matches(Path::new("/archive/0"), "inventory:0", "identity:0"));
        cache.insert(
            PathBuf::from("/archive/new"),
            "inventory:new".into(),
            "identity:new".into(),
        );
        assert_eq!(cache.entries.len(), width);
        assert!(cache.matches(Path::new("/archive/0"), "inventory:0", "identity:0"));
        assert!(!cache.entries.contains_key(Path::new("/archive/1")));
        assert!(cache.matches(Path::new("/archive/511"), "inventory:511", "identity:511"));
    }
}
