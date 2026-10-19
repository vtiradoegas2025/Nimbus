use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::error::{AppError, AppResult};

const AGENT: &str = include_str!("remote_compute_agent.py");
const MAX_REPLY: usize = 16 * 1024 * 1024;
static SEQUENCE: AtomicU64 = AtomicU64::new(0);
static RESERVED_DOWNLOAD_BYTES: Mutex<u64> = Mutex::new(0);

struct DownloadReservation(u64);
impl Drop for DownloadReservation {
    fn drop(&mut self) {
        if let Ok(mut total) = RESERVED_DOWNLOAD_BYTES.lock() {
            *total = total.saturating_sub(self.0);
        }
    }
}

fn reserve_download(path: &Path, bytes: u64) -> AppResult<DownloadReservation> {
    #[cfg(unix)]
    let free = {
        use std::os::unix::ffi::OsStrExt;
        let path = std::ffi::CString::new(path.as_os_str().as_bytes())
            .map_err(|_| invalid("invalid download directory"))?;
        let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
        if unsafe { libc::statvfs(path.as_ptr(), stats.as_mut_ptr()) } != 0 {
            return Err(AppError::Io(std::io::Error::last_os_error()));
        }
        let stats = unsafe { stats.assume_init() };
        (stats.f_bavail as u64).saturating_mul(stats.f_frsize as u64)
    };
    #[cfg(windows)]
    let free = {
        use std::os::windows::ffi::OsStrExt;
        let path = path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect::<Vec<_>>();
        let mut available = 0_u64;
        if unsafe {
            windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW(
                path.as_ptr(),
                &mut available,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        } == 0
        {
            return Err(AppError::Io(std::io::Error::last_os_error()));
        }
        available
    };
    #[cfg(not(any(unix, windows)))]
    let free = 0_u64;
    let mut reserved = RESERVED_DOWNLOAD_BYTES
        .lock()
        .map_err(|_| AppError::Task("remote download budget lock poisoned".into()))?;
    let next = reserved
        .checked_add(bytes)
        .ok_or_else(|| invalid("remote download size overflow"))?;
    if next > free.saturating_sub(16 * 1024 * 1024) {
        return Err(invalid(
            "remote history exceeds the available local download space",
        ));
    }
    *reserved = next;
    Ok(DownloadReservation(bytes))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RemoteComputeTarget {
    pub id: String,
    pub name: String,
    pub host: String,
    pub binary_path: String,
    pub work_directory: String,
    #[serde(default)]
    pub asset_mappings: Vec<RemoteAssetMapping>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RemoteAssetMapping {
    pub local_path: String,
    pub remote_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RemoteComputeProbe {
    pub available: bool,
    pub platform: String,
    pub binary_path: String,
    pub binary_fingerprint: String,
    pub catalog_fingerprint: String,
    pub byte_length: u64,
    pub modified_unix_ms: Option<u128>,
    pub catalog: Value,
    pub catalog_text: String,
    pub work_directory: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RemoteAssetReceipt {
    pub remote_path: String,
    pub requested_path: String,
    pub sha256: String,
}

#[derive(Debug, Clone)]
pub struct RemoteRun {
    pub target: RemoteComputeTarget,
    pub token: String,
    pub output_path: String,
    transport: RemoteTransport,
    mirror_cancelled: Arc<AtomicBool>,
    mirroring: Arc<AtomicBool>,
    connection_owner: Arc<Mutex<Option<ChildStdin>>>,
    ownership_revoked: Arc<AtomicBool>,
    shutdown_requested: Arc<AtomicBool>,
}

#[derive(Debug, Clone)]
pub struct RemoteStage {
    pub run: RemoteRun,
    pub config_path: String,
    pub source_sha256: String,
    pub assets: Vec<RemoteAssetReceipt>,
    pub probe: RemoteComputeProbe,
    pub execution_target_fingerprint: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RemoteValidation {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Debug, Clone)]
struct RemoteTransport {
    ssh: PathBuf,
}

impl Default for RemoteTransport {
    fn default() -> Self {
        Self {
            ssh: PathBuf::from("ssh"),
        }
    }
}

fn token() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!(
        "{now:x}-{:x}-{:x}",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    )
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn invalid(message: &str) -> AppError {
    AppError::Invalid(message.into())
}

fn safe_absolute(value: &str) -> bool {
    value.starts_with('/')
        && value.len() <= 4096
        && !value.chars().any(char::is_control)
        && !value.split('/').any(|part| part == "..")
}

fn remote_normalized(value: &str) -> String {
    let parts = value
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .collect::<Vec<_>>()
        .join("/");
    if value.starts_with('/') {
        format!("/{parts}")
    } else {
        parts
    }
}

impl RemoteComputeTarget {
    pub fn validate(&self) -> AppResult<()> {
        if self.id.trim().is_empty()
            || self.id.len() > 128
            || self.name.trim().is_empty()
            || self.name.len() > 256
        {
            return Err(invalid("remote compute target needs a bounded ID and name"));
        }
        if self.host.is_empty()
            || self.host.len() > 255
            || !self.host.as_bytes()[0].is_ascii_alphanumeric()
            || !self
                .host
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"._@:[]-".contains(&byte))
        {
            return Err(invalid("remote host must be an SSH alias or user@host"));
        }
        if !safe_absolute(&self.binary_path)
            || !safe_absolute(&self.work_directory)
            || self.work_directory == "/"
        {
            return Err(invalid(
                "remote binary and work directory must be absolute Linux paths",
            ));
        }
        if self.asset_mappings.len() > 32 {
            return Err(invalid("remote target exceeds 32 asset mappings"));
        }
        let mut destinations = std::collections::HashSet::new();
        for asset in &self.asset_mappings {
            if !Path::new(&asset.local_path).is_absolute()
                || asset.remote_path.trim().is_empty()
                || asset.remote_path.len() > 4096
                || asset.remote_path.chars().any(char::is_control)
                || asset.remote_path.split('/').any(|part| part == "..")
            {
                return Err(invalid(
                    "remote asset mapping has an invalid local or remote path",
                ));
            }
            let work = remote_normalized(&self.work_directory);
            let destination = remote_normalized(&if asset.remote_path.starts_with('/') {
                asset.remote_path.clone()
            } else {
                format!("{work}/{}", asset.remote_path)
            });
            if !destination.starts_with(&format!("{work}/"))
                || destination == work
                || !destinations.insert(destination)
            {
                return Err(invalid(
                    "remote assets must have unique destinations inside the work directory",
                ));
            }
        }
        Ok(())
    }
}

impl RemoteTransport {
    fn command(
        &self,
        target: &RemoteComputeTarget,
        action: &str,
        mut args: Value,
    ) -> AppResult<Command> {
        target.validate()?;
        args["workDirectory"] = json!(target.work_directory);
        args["binaryPath"] = json!(target.binary_path);
        let remote = format!(
            "exec python3 -u -c {} {} {}",
            shell_quote(AGENT),
            shell_quote(action),
            shell_quote(
                &serde_json::to_string(&args).map_err(|error| AppError::Invalid(format!(
                    "serialize remote request: {error}"
                )))?
            )
        );
        let mut command = Command::new(&self.ssh);
        command
            .args([
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                "ConnectTimeout=10",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=2",
                "--",
            ])
            .arg(&target.host)
            .arg(remote)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        Ok(command)
    }

    fn run(
        &self,
        target: &RemoteComputeTarget,
        action: &str,
        args: Value,
        input: Option<File>,
        timeout: Duration,
    ) -> AppResult<Vec<u8>> {
        self.run_cancelable(target, action, args, input, timeout, None)
    }

    fn run_cancelable(
        &self,
        target: &RemoteComputeTarget,
        action: &str,
        args: Value,
        input: Option<File>,
        timeout: Duration,
        cancelled: Option<&AtomicBool>,
    ) -> AppResult<Vec<u8>> {
        let mut command = self.command(target, action, args)?;
        command.stdin(input.map(Stdio::from).unwrap_or_else(Stdio::null));
        let mut child = command
            .spawn()
            .map_err(|error| AppError::Subprocess(format!("remote SSH {action}: {error}")))?;
        let overflow = Arc::new(AtomicBool::new(false));
        let read_pipe = |mut pipe: Box<dyn Read + Send>, overflow: Arc<AtomicBool>| {
            thread::spawn(move || {
                let mut bytes = Vec::new();
                let mut buffer = [0_u8; 64 * 1024];
                loop {
                    let count = pipe.read(&mut buffer)?;
                    if count == 0 {
                        break;
                    }
                    if bytes.len() + count > MAX_REPLY {
                        overflow.store(true, Ordering::Relaxed);
                    } else {
                        bytes.extend_from_slice(&buffer[..count]);
                    }
                }
                Ok::<_, std::io::Error>(bytes)
            })
        };
        let stdout = read_pipe(Box::new(child.stdout.take().unwrap()), overflow.clone());
        let stderr = read_pipe(Box::new(child.stderr.take().unwrap()), overflow.clone());
        let deadline = Instant::now() + timeout;
        let status = loop {
            if cancelled.is_some_and(|cancelled| cancelled.load(Ordering::Acquire)) {
                terminate_transport(&mut child);
                return Err(invalid("remote history download was cancelled"));
            }
            if let Some(status) = child.try_wait()? {
                break status;
            }
            if overflow.load(Ordering::Relaxed) || Instant::now() >= deadline {
                terminate_transport(&mut child);
                return Err(AppError::Subprocess(format!(
                    "remote {action} exceeded its time or output bound"
                )));
            }
            thread::sleep(Duration::from_millis(20));
        };
        while !stdout.is_finished() || !stderr.is_finished() {
            if overflow.load(Ordering::Relaxed)
                || Instant::now() >= deadline
                || cancelled.is_some_and(|cancelled| cancelled.load(Ordering::Acquire))
            {
                terminate_transport(&mut child);
                return Err(AppError::Subprocess(format!(
                    "remote {action} pipes did not drain within their bound"
                )));
            }
            thread::sleep(Duration::from_millis(20));
        }
        let out = stdout
            .join()
            .map_err(|_| AppError::Task("remote stdout reader panicked".into()))??;
        let err = stderr
            .join()
            .map_err(|_| AppError::Task("remote stderr reader panicked".into()))??;
        if overflow.load(Ordering::Relaxed) {
            return Err(invalid("remote reply exceeded its output bound"));
        }
        if !status.success() {
            return Err(AppError::Subprocess(format!(
                "remote {action} failed (exit {:?}): {}",
                status.code(),
                String::from_utf8_lossy(&err)
                    .chars()
                    .take(4096)
                    .collect::<String>()
            )));
        }
        Ok(out)
    }

    fn probe(&self, target: &RemoteComputeTarget) -> AppResult<RemoteComputeProbe> {
        let bytes = self.run(target, "probe", json!({}), None, Duration::from_secs(90))?;
        let probe: RemoteComputeProbe = serde_json::from_slice(&bytes).map_err(|error| {
            AppError::Invalid(format!("invalid remote compute receipt: {error}"))
        })?;
        if !probe.available
            || probe.platform != "linux"
            || !safe_absolute(&probe.binary_path)
            || probe.binary_fingerprint.len() != 71
            || !probe.binary_fingerprint.starts_with("sha256:")
            || !probe.binary_fingerprint[7..]
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            || probe.catalog_fingerprint != fingerprint(probe.catalog_text.as_bytes())
            || serde_json::from_str::<Value>(&probe.catalog_text)
                .ok()
                .as_ref()
                != Some(&probe.catalog)
        {
            return Err(invalid(
                "remote executable/catalog identity receipt is inconsistent",
            ));
        }
        super::runner::parse_scheme_catalog_json(probe.catalog_text.as_bytes())?;
        Ok(probe)
    }

    fn upload(
        &self,
        target: &RemoteComputeTarget,
        token: &str,
        path: &Path,
        destination: &str,
    ) -> AppResult<RemoteAssetReceipt> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(invalid("remote source assets must be regular files"));
        }
        let sha256 = crate::platform::sha256_file(path)?
            .trim_start_matches("sha256:")
            .to_string();
        self.run(target, "stage", json!({"token":token,"destination":destination,"byteLength":metadata.len(),"sha256":sha256}),
                 Some(file), Duration::from_secs(300))?;
        Ok(RemoteAssetReceipt {
            remote_path: destination.into(),
            requested_path: destination.into(),
            sha256,
        })
    }
}

fn terminate_transport(child: &mut Child) {
    #[cfg(unix)]
    unsafe {
        libc::kill(-(child.id() as i32), libc::SIGKILL);
    }
    let _ = child.kill();
    let _ = child.wait();
}

pub(crate) fn fingerprint(bytes: &[u8]) -> String {
    let hash = bytes.iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    });
    format!("fnv1a64:{hash:016x}")
}

pub fn query_remote_compute(
    execution_target: RemoteComputeTarget,
) -> AppResult<RemoteComputeProbe> {
    RemoteTransport::default().probe(&execution_target)
}

#[tauri::command]
pub async fn probe_remote_compute(
    execution_target: RemoteComputeTarget,
) -> AppResult<RemoteComputeProbe> {
    tauri::async_runtime::spawn_blocking(move || query_remote_compute(execution_target))
        .await
        .map_err(|error| AppError::Task(error.to_string()))?
}

pub fn stage_source(target: RemoteComputeTarget, contents: &str) -> AppResult<RemoteStage> {
    stage_source_with_transport(target, contents, RemoteTransport::default())
}

fn stage_source_with_transport(
    target: RemoteComputeTarget,
    contents: &str,
    transport: RemoteTransport,
) -> AppResult<RemoteStage> {
    target.validate()?;
    if contents.trim().is_empty() || contents.len() > 8 * 1024 * 1024 {
        return Err(invalid("remote YAML source is empty or exceeds 8 MiB"));
    }
    let probe = transport.probe(&target)?;
    let token = token();
    let relative = format!(".nimbus-studio/{token}/source.yaml");
    let config_path = format!(
        "{}/{}",
        target.work_directory.trim_end_matches('/'),
        relative
    );
    let temporary = std::env::temp_dir().join(format!("nimbus-remote-source-{token}.yaml"));
    let result = (|| {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        file.write_all(contents.as_bytes())?;
        file.sync_all()?;
        drop(file);
        let config = transport.upload(&target, &token, &temporary, &config_path)?;
        let mut assets = Vec::new();
        let mut identity_assets = Vec::new();
        for mapping in &target.asset_mappings {
            let sha256 = crate::platform::sha256_file(Path::new(&mapping.local_path))?
                .trim_start_matches("sha256:")
                .to_string();
            identity_assets.push(json!({"mapping":mapping,"sha256":sha256}));
            let remote_path = if mapping.remote_path.starts_with('/') {
                transport.run(
                    &target,
                    "verify-asset",
                    json!({"path":mapping.remote_path,"sha256":sha256}),
                    None,
                    Duration::from_secs(300),
                )?;
                mapping.remote_path.clone()
            } else {
                let destination = format!(
                    "{}/.nimbus-studio/{token}/{}",
                    target.work_directory.trim_end_matches('/'),
                    mapping.remote_path
                );
                transport.upload(
                    &target,
                    &token,
                    Path::new(&mapping.local_path),
                    &destination,
                )?;
                destination
            };
            assets.push(RemoteAssetReceipt {
                remote_path,
                requested_path: mapping.remote_path.clone(),
                sha256,
            });
        }
        let context = serde_json::to_vec(&json!({"target":target,"assets":identity_assets}))
            .map_err(|error| {
                AppError::Invalid(format!("serialize remote input identity: {error}"))
            })?;
        let execution_target_fingerprint = format!("sha256:{:x}", Sha256::digest(context));
        Ok(RemoteStage {
            run: RemoteRun {
                output_path: format!(
                    "{}/.nimbus-studio/{token}/output",
                    target.work_directory.trim_end_matches('/')
                ),
                target,
                token,
                transport,
                mirror_cancelled: Arc::new(AtomicBool::new(false)),
                mirroring: Arc::new(AtomicBool::new(false)),
                connection_owner: Arc::new(Mutex::new(None)),
                ownership_revoked: Arc::new(AtomicBool::new(false)),
                shutdown_requested: Arc::new(AtomicBool::new(false)),
            },
            config_path,
            source_sha256: config.sha256,
            assets,
            probe,
            execution_target_fingerprint,
        })
    })();
    let _ = fs::remove_file(temporary);
    result
}

impl RemoteStage {
    pub fn verify_launch_identity(
        &self,
        binary: Option<&str>,
        catalog: Option<&str>,
        context: Option<&str>,
    ) -> AppResult<()> {
        if binary != Some(self.probe.binary_fingerprint.as_str())
            || catalog != Some(self.probe.catalog_fingerprint.as_str())
            || context != Some(self.execution_target_fingerprint.as_str())
        {
            return Err(invalid(
                "remote target, input assets, executable or catalog changed after preflight",
            ));
        }
        Ok(())
    }
    fn arguments(&self) -> Value {
        json!({"token":self.run.token,"configPath":self.config_path,"sourceSha256":self.source_sha256,
               "jobDirectory":self.config_path.rsplit_once('/').unwrap().0,
               "assets":self.assets,"outputPath":self.run.output_path,
               "expectedBinaryFingerprint":self.probe.binary_fingerprint,"expectedCatalogFingerprint":self.probe.catalog_fingerprint})
    }

    pub fn validate_source(&self, require_context: bool) -> AppResult<RemoteValidation> {
        let mut args = self.arguments();
        args["requireContext"] = json!(require_context);
        serde_json::from_slice(&self.run.transport.run(
            &self.run.target,
            "validate",
            args,
            None,
            Duration::from_secs(120),
        )?)
        .map_err(|error| AppError::Invalid(format!("invalid remote validation reply: {error}")))
    }

    pub fn migrate_source(&self) -> AppResult<RemoteValidation> {
        serde_json::from_slice(&self.run.transport.run(
            &self.run.target,
            "migrate",
            self.arguments(),
            None,
            Duration::from_secs(120),
        )?)
        .map_err(|error| AppError::Invalid(format!("invalid remote migration reply: {error}")))
    }

    pub fn spawn(&self) -> AppResult<Child> {
        let mut child = self
            .run
            .transport
            .command(&self.run.target, "start", self.arguments())?
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|error| AppError::Subprocess(format!("launch remote Nimbus: {error}")))?;
        if let Some(input) = child.stdin.take() {
            self.run.attach_connection(input);
        }
        Ok(child)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RemoteFile {
    pub path: String,
    pub byte_length: u64,
    pub sha256: String,
}

fn verify_existing_output(destination: &Path, top: &str, files: &[RemoteFile]) -> AppResult<()> {
    let expected = files.iter().filter(|file| file.path == top || file.path.starts_with(&format!("{top}/")))
        .map(|file| (file.path.clone(), file)).collect::<std::collections::HashMap<_, _>>();
    let mut directories = std::collections::HashSet::new();
    for path in expected.keys() {
        let parts = path.split('/').collect::<Vec<_>>();
        for end in 1..parts.len() { directories.insert(parts[..end].join("/")); }
    }
    let mut pending = vec![(destination.to_path_buf(), top.to_string())];
    let mut checked = 0;
    while let Some((path, relative)) = pending.pop() {
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_dir() {
            if !directories.contains(&relative) { return Err(invalid("remote output conflicts with a local directory")); }
            for entry in fs::read_dir(&path)? {
                let entry = entry?;
                let name = entry.file_name().into_string().map_err(|_| invalid("local output path is not UTF-8"))?;
                pending.push((entry.path(), format!("{relative}/{name}")));
                if pending.len() > files.len() { return Err(invalid("local output inventory exceeds the remote manifest")); }
            }
        } else if metadata.file_type().is_file() {
            let expected = expected.get(&relative).ok_or_else(|| invalid("remote output conflicts with a local artifact"))?;
            if metadata.len() != expected.byte_length || crate::platform::sha256_file(&path)? != format!("sha256:{}", expected.sha256) {
                return Err(invalid("remote output would overwrite differing local artifact bytes"));
            }
            checked += 1;
        } else { return Err(invalid("local output contains a non-regular artifact")); }
    }
    if checked != expected.len() { return Err(invalid("local output is incomplete and cannot be treated as an identical published artifact")); }
    Ok(())
}

fn publish_remote_artifact(source: &Path, destination: &Path) -> AppResult<()> {
    #[cfg(unix)] {
        use std::os::unix::ffi::OsStrExt;
        let source = std::ffi::CString::new(source.as_os_str().as_bytes()).map_err(|_| invalid("invalid staged path"))?;
        let destination = std::ffi::CString::new(destination.as_os_str().as_bytes()).map_err(|_| invalid("invalid output path"))?;
        #[cfg(target_os = "macos")]
        let result = unsafe { libc::renamex_np(source.as_ptr(), destination.as_ptr(), libc::RENAME_EXCL) };
        #[cfg(target_os = "linux")]
        let result = unsafe { libc::syscall(libc::SYS_renameat2, libc::AT_FDCWD, source.as_ptr(), libc::AT_FDCWD, destination.as_ptr(), libc::RENAME_NOREPLACE) as i32 };
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        return Err(invalid("atomic no-replace remote publication is unavailable on this viewer"));
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        if result != 0 { return Err(std::io::Error::last_os_error().into()); }
    }
    #[cfg(windows)] {
        use std::os::windows::ffi::OsStrExt;
        let source = source.as_os_str().encode_wide().chain(std::iter::once(0)).collect::<Vec<_>>();
        let destination = destination.as_os_str().encode_wide().chain(std::iter::once(0)).collect::<Vec<_>>();
        if unsafe { windows_sys::Win32::Storage::FileSystem::MoveFileW(source.as_ptr(), destination.as_ptr()) } == 0 {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    Ok(())
}

impl RemoteRun {
    pub fn from_execution(execution: &super::provenance::RemoteExecutionProvenance) -> AppResult<Self> {
        Self::from_execution_with_transport(execution, RemoteTransport::default())
    }

    fn from_execution_with_transport(execution: &super::provenance::RemoteExecutionProvenance, transport: RemoteTransport) -> AppResult<Self> {
        execution.target.validate()?;
        let prefix = format!("{}/.nimbus-studio/", execution.target.work_directory.trim_end_matches('/'));
        let token = execution.remote_output_directory.strip_prefix(&prefix).and_then(|path| path.strip_suffix("/output"))
            .filter(|token| !token.is_empty() && token.len() <= 128 && token.bytes().all(|byte| byte.is_ascii_hexdigit() || byte == b'-'))
            .ok_or_else(|| invalid("remote output must have the exact .nimbus-studio/<token>/output identity"))?;
        let assets = execution.target.asset_mappings.iter().map(|mapping| {
            let asset = execution.assets.iter().find(|asset| asset.requested_path == mapping.remote_path)
                .ok_or_else(|| invalid("remote asset identity is missing from launch provenance"))?;
            Ok(json!({"mapping":mapping,"sha256":asset.sha256}))
        }).collect::<AppResult<Vec<_>>>()?;
        let identity = serde_json::to_vec(&json!({"target":execution.target,"assets":assets})).map_err(|error| invalid(&error.to_string()))?;
        if execution.execution_target_fingerprint != format!("sha256:{:x}", Sha256::digest(identity)) {
            return Err(invalid("remote target or asset identity disagrees with launch provenance"));
        }
        Ok(Self { target: execution.target.clone(), token: token.into(), output_path: execution.remote_output_directory.clone(), transport,
            mirror_cancelled: Arc::new(AtomicBool::new(false)), mirroring: Arc::new(AtomicBool::new(false)),
            connection_owner: Arc::new(Mutex::new(None)), ownership_revoked: Arc::new(AtomicBool::new(false)), shutdown_requested: Arc::new(AtomicBool::new(false)) })
    }

    pub fn confirm_stopped(&self) -> AppResult<()> {
        let bytes = self.transport.run_cancelable(&self.target, "status", json!({"token":self.token}), None,
            Duration::from_secs(30), Some(&self.mirror_cancelled))?;
        let reply: Value = serde_json::from_slice(&bytes).map_err(|error| invalid(&format!("invalid remote completion receipt: {error}")))?;
        if reply["stopped"] != true { return Err(invalid("Remote computation is still active; receiving cannot infer a completed state or stop it.")); }
        Ok(())
    }

    pub fn cancel_receive(&self) { self.mirror_cancelled.store(true, Ordering::Release); }
    pub fn receive_cancelled(&self) -> bool { self.mirror_cancelled.load(Ordering::Acquire) }

    fn attach_connection(&self, input: ChildStdin) {
        if let Ok(mut owner) = self.connection_owner.lock() {
            *owner = Some(input);
        }
        self.ownership_revoked.store(false, Ordering::Release);
        let run = self.clone();
        thread::spawn(move || loop {
            let mut owner = match run.connection_owner.lock() {
                Ok(owner) => owner,
                Err(_) => return,
            };
            if run.ownership_revoked.load(Ordering::Acquire) {
                owner.take();
                return;
            }
            if !owner
                .as_mut()
                .is_some_and(|input| input.write_all(&[1]).is_ok())
            {
                owner.take();
                return;
            }
            drop(owner);
            thread::sleep(Duration::from_millis(250));
        });
    }

    pub fn revoke_transport_ownership(&self) {
        self.ownership_revoked.store(true, Ordering::Release);
        if let Ok(mut owner) = self.connection_owner.try_lock() {
            owner.take();
        }
    }

    pub fn begin_shutdown(&self) {
        self.shutdown_requested.store(true, Ordering::Release);
        self.mirror_cancelled.store(true, Ordering::Release);
        self.revoke_transport_ownership();
    }

    pub fn cancel_for_shutdown(&self) -> AppResult<()> {
        self.cancel_with_timeout(Duration::from_secs(1))
    }

    pub fn cancel(&self) -> AppResult<()> {
        self.cancel_with_timeout(Duration::from_secs(30))
    }

    fn cancel_with_timeout(&self, timeout: Duration) -> AppResult<()> {
        if self.mirroring.load(Ordering::Acquire) {
            self.mirror_cancelled.store(true, Ordering::Release);
        }
        let bytes = self.transport.run(
            &self.target,
            "cancel",
            json!({"token":self.token}),
            None,
            timeout,
        )?;
        let reply: Value = serde_json::from_slice(&bytes)
            .map_err(|error| AppError::Invalid(format!("invalid remote stop receipt: {error}")))?;
        if reply["stopped"] != true {
            return Err(invalid("remote process termination was not confirmed"));
        }
        Ok(())
    }

    fn fetch(&self, source: &RemoteFile, destination: &Path) -> AppResult<()> {
        if source.sha256.len() != 64 || !source.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(invalid("invalid remote output fingerprint"));
        }
        let mut command = self.transport.command(
            &self.target,
            "fetch",
            json!({"outputPath":self.output_path,"path":source.path}),
        )?;
        let mut child = command.stdin(Stdio::null()).spawn()?;
        let finished = Arc::new(AtomicBool::new(false));
        let watcher_finished = finished.clone();
        let download_cancelled = self.mirror_cancelled.clone();
        let pid = child.id();
        let watchdog = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(900);
            while !watcher_finished.load(Ordering::Acquire) {
                if Instant::now() >= deadline || download_cancelled.load(Ordering::Acquire) {
                    #[cfg(unix)]
                    unsafe {
                        libc::kill(-(pid as i32), libc::SIGKILL);
                    }
                    #[cfg(windows)]
                    let _ = Command::new("taskkill")
                        .args(["/PID", &pid.to_string(), "/T", "/F"])
                        .status();
                    break;
                }
                thread::sleep(Duration::from_millis(50));
            }
        });
        let mut pipe = child.stdout.take().unwrap();
        let mut stderr = child.stderr.take().unwrap();
        let errors = thread::spawn(move || {
            let mut bytes = Vec::new();
            let mut buffer = [0_u8; 8192];
            loop {
                let count = stderr.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                if bytes.len() < 8192 {
                    bytes.extend_from_slice(&buffer[..count.min(8192 - bytes.len())]);
                }
            }
            Ok::<_, std::io::Error>(bytes)
        });
        let mut hash = Sha256::new();
        let mut count = 0_u64;
        let mut buffer = [0_u8; 1024 * 1024];
        let result = (|| -> AppResult<()> {
            let mut output = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(destination)?;
            loop {
                if self.mirror_cancelled.load(Ordering::Acquire) {
                    return Err(invalid("remote history download was cancelled"));
                }
                let read = pipe.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                count = count
                    .checked_add(read as u64)
                    .ok_or_else(|| invalid("remote output length overflow"))?;
                if count > source.byte_length {
                    return Err(invalid("remote output grew during transfer"));
                }
                hash.update(&buffer[..read]);
                output.write_all(&buffer[..read])?;
            }
            output.sync_all()?;
            let status = child.wait()?;
            if !status.success()
                || count != source.byte_length
                || format!("{:x}", hash.finalize()) != source.sha256
            {
                return Err(invalid(
                    "remote output transfer length, fingerprint or SSH exit mismatch",
                ));
            }
            Ok(())
        })();
        if result.is_err() {
            terminate_transport(&mut child);
        }
        finished.store(true, Ordering::Release);
        let _ = watchdog.join();
        let _ = errors.join();
        result
    }

    #[cfg(test)]
    pub fn mirror_outputs(&self, local: &Path) -> AppResult<()> {
        self.mirror_outputs_checked(local).map(|_| ())
    }

    pub(crate) fn mirror_outputs_checked(&self, local: &Path) -> AppResult<Vec<RemoteFile>> {
        if self.shutdown_requested.load(Ordering::Acquire) {
            return Err(invalid("remote history download was cancelled by shutdown"));
        }
        if self.mirror_cancelled.load(Ordering::Acquire) { return Err(invalid("remote history download was cancelled")); }
        if self.mirroring.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() { return Err(invalid("remote history is already being received")); }
        struct MirrorGuard(Arc<AtomicBool>);
        impl Drop for MirrorGuard {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
            }
        }
        let _guard = MirrorGuard(self.mirroring.clone());
        let bytes = self.transport.run_cancelable(
            &self.target,
            "files",
            json!({"outputPath":self.output_path}),
            None,
            Duration::from_secs(300),
            Some(&self.mirror_cancelled),
        )?;
        let files: Vec<RemoteFile> = serde_json::from_slice(&bytes).map_err(|error| {
            AppError::Invalid(format!("invalid remote history manifest: {error}"))
        })?;
        if files.len() > 100_000 {
            return Err(invalid("remote history exceeds 100000 files"));
        }
        let mut paths = std::collections::HashSet::new();
        for file in &files {
            let path = Path::new(&file.path);
            let normalized = path.components().map(|part| part.as_os_str().to_string_lossy()).collect::<Vec<_>>().join("/");
            if file.path.is_empty() || normalized != file.path || path.components().any(|part| !matches!(part, Component::Normal(_)))
                || path.components().any(|part| part.as_os_str().to_string_lossy().starts_with('.')) || !paths.insert(file.path.clone()) {
                return Err(invalid("remote history contains an unsafe or duplicate output path"));
            }
        }
        let mut reused = std::collections::HashSet::new();
        for file in &files {
            let top = file.path.split('/').next().unwrap();
            if reused.contains(top) { continue; }
            let destination = local.join(top);
            if fs::symlink_metadata(&destination).is_ok() {
                verify_existing_output(&destination, top, &files)?;
                reused.insert(top.to_string());
            }
        }
        let total_bytes = files.iter().filter(|file| !reused.contains(file.path.split('/').next().unwrap()))
            .try_fold(0_u64, |sum, file| sum.checked_add(file.byte_length))
            .ok_or_else(|| invalid("remote history byte count overflow"))?;
        let _space = reserve_download(local, total_bytes)?;
        let staging = local.join(format!(".remote-download-{}", self.token));
        if fs::symlink_metadata(&staging).is_ok() {
            if !fs::symlink_metadata(&staging)?.file_type().is_dir() { return Err(invalid("remote staging path must be an owned directory")); }
            fs::remove_dir_all(&staging)?;
        }
        fs::create_dir(&staging)?;
        let result = (|| -> AppResult<()> {
            for file in &files {
                if reused.contains(file.path.split('/').next().unwrap()) { continue; }
                if self.mirror_cancelled.load(Ordering::Acquire) {
                    return Err(invalid("remote history download was cancelled"));
                }
                let path = Path::new(&file.path);
                if path.as_os_str().is_empty()
                    || path
                        .components()
                        .any(|part| !matches!(part, Component::Normal(_)))
                    || path
                        .components()
                        .any(|part| part.as_os_str().to_string_lossy().starts_with('.'))
                {
                    return Err(invalid("remote history contains an unsafe output path"));
                }
                let destination = staging.join(path);
                fs::create_dir_all(destination.parent().unwrap())?;
                self.fetch(file, &destination)?;
            }
            // A complete frame directory becomes discoverable only after all
            // of its files have passed their remote byte-count and SHA checks.
            let entries = fs::read_dir(&staging)?.collect::<Result<Vec<_>, _>>()?;
            for entry in &entries {
                let destination = local.join(entry.file_name());
                if destination.exists() {
                    return Err(invalid(
                        "remote output would overwrite a local run artifact",
                    ));
                }
            }
            for entry in entries {
                if self.mirror_cancelled.load(Ordering::Acquire) { return Err(invalid("remote history download was cancelled")); }
                let destination = local.join(entry.file_name());
                publish_remote_artifact(&entry.path(), &destination)?;
            }
            Ok(())
        })();
        let _ = fs::remove_dir_all(staging);
        result?;
        Ok(files)
    }
}

pub fn copy_remote_checkpoint(
    target: RemoteComputeTarget,
    path: &str,
) -> AppResult<(PathBuf, PathBuf, String)> {
    let transport = RemoteTransport::default();
    let bytes = transport.run(
        &target,
        "checkpoint-files",
        json!({"path":path}),
        None,
        Duration::from_secs(300),
    )?;
    let files: Vec<RemoteFile> = serde_json::from_slice(&bytes).map_err(|error| {
        AppError::Invalid(format!("invalid remote checkpoint receipt: {error}"))
    })?;
    if files.len() != 2
        || files[0].byte_length > 16 * 1024 * 1024 * 1024 * 1024
        || files[1].byte_length > 1024 * 1024
    {
        return Err(invalid("remote checkpoint exceeds its inspection bounds"));
    }
    let temporary = std::env::temp_dir().join(format!("nimbus-remote-checkpoint-{}", token()));
    let total_bytes = files
        .iter()
        .try_fold(0_u64, |sum, file| sum.checked_add(file.byte_length))
        .ok_or_else(|| invalid("checkpoint size overflow"))?;
    let _space = reserve_download(&std::env::temp_dir(), total_bytes)?;
    fs::create_dir(&temporary)?;
    let result = (|| {
        let run = RemoteRun {
            output_path: target.work_directory.clone(),
            target,
            token: token(),
            transport,
            mirror_cancelled: Arc::new(AtomicBool::new(false)),
            mirroring: Arc::new(AtomicBool::new(false)),
            connection_owner: Arc::new(Mutex::new(None)),
            ownership_revoked: Arc::new(AtomicBool::new(false)),
            shutdown_requested: Arc::new(AtomicBool::new(false)),
        };
        for source in &files {
            if Path::new(&source.path)
                .components()
                .any(|part| !matches!(part, Component::Normal(_)))
            {
                return Err(invalid("unsafe remote checkpoint path"));
            }
            let filename = Path::new(&source.path)
                .file_name()
                .ok_or_else(|| invalid("remote checkpoint filename missing"))?;
            run.fetch(source, &temporary.join(filename))?;
        }
        let checkpoint = temporary.join(Path::new(&files[0].path).file_name().unwrap());
        let remote_path = format!(
            "{}/{}",
            run.target.work_directory.trim_end_matches('/'),
            files[0].path
        );
        Ok((temporary.clone(), checkpoint, remote_path))
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&temporary);
    }
    result
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    struct Fixture {
        root: PathBuf,
        target: RemoteComputeTarget,
        transport: RemoteTransport,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    impl Fixture {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!("nimbus remote ' fixture-{}", token()));
            fs::create_dir(&root).unwrap();
            fs::write(
                root.join("catalog.json"),
                include_str!("../../../src/lib/config/nativeCatalogSnapshot.json"),
            )
            .unwrap();
            let binary = root.join("native sim ' script");
            fs::write(&binary, r#"#!/usr/bin/env python3
import json, pathlib, signal, sys, time
root = pathlib.Path(__file__).parent
if '--scheme-catalog=json' in sys.argv:
    if (root / 'catalog-delay').exists(): time.sleep(0.5)
    sys.stdout.write((root / 'catalog.json').read_text())
    sys.exit(0)
source = pathlib.Path(sys.argv[sys.argv.index('--config') + 1]) if '--config' in sys.argv else pathlib.Path(sys.argv[-1])
data = source.read_text()
if '--migrate-config' in sys.argv:
    print('schema: nimbus.v1\n' + data)
    sys.exit(0)
if '--validate-only' in sys.argv:
    if 'reject: true' in data:
        print('strict science rejected', file=sys.stderr)
        sys.exit(2)
    print('[VALIDATE] PASS')
    sys.exit(0)
print('[VEL-MAX] t=1.25s step=3 vmax=8', flush=True)
if 'loop: true' in data:
    signal.signal(signal.SIGINT, lambda *_: sys.exit(130))
    while True: time.sleep(0.05)
out = pathlib.Path(next(value[9:] for value in sys.argv if value.startswith('--outdir=')))
(out / 'step_000000').mkdir()
(out / 'step_000000' / 'manifest.json').write_text('{"publication_state":"complete"}')
(out / 'step_000000' / 'theta_prime.npy').write_bytes(b'finite source bytes')
(out / 'z.txt').write_text('completed')
"#).unwrap();
            fs::set_permissions(&binary, fs::Permissions::from_mode(0o755)).unwrap();
            let ssh = root.join("fake ssh");
            fs::write(&ssh, r#"#!/usr/bin/env python3
import json, os, pathlib, shlex, sys
assert '-T' in sys.argv and 'BatchMode=yes' in sys.argv and 'StrictHostKeyChecking=yes' in sys.argv
assert sys.argv[-2] == 'fixture-node'
argv = shlex.split(sys.argv[-1])
assert argv[:4] == ['exec', 'python3', '-u', '-c']
args = json.loads(argv[6])
root = pathlib.Path(args['workDirectory'])
if argv[5] == 'fetch':
    with open(root / 'fetch-count', 'a') as counter: counter.write('fetch\n')
if argv[5] == 'files' and (root / 'receive-delay').exists():
    import time
    (root / 'receive-started').touch()
    time.sleep(10)
if argv[5] == 'fetch' and (root / 'corrupt-download').exists():
    argv[4] += '\n'
    argv[4] = argv[4].replace('sys.stdout.buffer.write(chunk)', 'sys.stdout.buffer.write(bytes([chunk[0] ^ 1]) + chunk[1:])')
argv[4] = 'import sys\nsys.platform = "linux"\n' + argv[4]
os.execvp('python3', argv[1:])
"#).unwrap();
            fs::set_permissions(&ssh, fs::Permissions::from_mode(0o755)).unwrap();
            let target = RemoteComputeTarget {
                id: "fixture".into(),
                name: "Fixture node".into(),
                host: "fixture-node".into(),
                binary_path: binary.display().to_string(),
                work_directory: root.display().to_string(),
                asset_mappings: Vec::new(),
            };
            Self {
                root,
                target,
                transport: RemoteTransport { ssh },
            }
        }

        fn stage(&self, yaml: &str) -> RemoteStage {
            stage_source_with_transport(self.target.clone(), yaml, self.transport.clone()).unwrap()
        }

        fn restore(&self, stage: &RemoteStage) -> RemoteRun {
            RemoteRun::from_execution_with_transport(&self.execution(stage), self.transport.clone()).unwrap()
        }

        fn execution(&self, stage: &RemoteStage) -> super::super::provenance::RemoteExecutionProvenance {
            super::super::provenance::RemoteExecutionProvenance { target: stage.run.target.clone(), remote_output_directory: stage.run.output_path.clone(),
                staged_config_sha256: stage.source_sha256.clone(), execution_target_fingerprint: stage.execution_target_fingerprint.clone(),
                assets: stage.assets.iter().map(|asset| super::super::provenance::RemoteSourceAssetProvenance {
                    remote_path: asset.remote_path.clone(), requested_path: asset.requested_path.clone(), sha256: asset.sha256.clone() }).collect() }
        }

        fn launch(&self, stage: &RemoteStage, local: &Path) -> super::super::provenance::RunProvenance {
            use super::super::provenance::{create_observed_run_provenance, ExecutableProvenance, ProvenanceDigest, RequestedRunValues};
            fs::create_dir(local).unwrap();
            let config = self.root.join("recovery-source.yaml"); fs::write(&config, "schema: nimbus.v1\n").unwrap();
            create_observed_run_provenance(local, "recovery-run", 1, ExecutableProvenance { canonical_path: stage.probe.binary_path.clone(),
                digest: ProvenanceDigest { algorithm: "sha256".into(), value: stage.probe.binary_fingerprint[7..].into() } },
                &config, Some(stage.probe.catalog_fingerprint.clone()), RequestedRunValues::default(), Some(self.execution(stage))).unwrap()
        }

        fn complete(&self, stage: &RemoteStage) -> String {
            let mut child = stage.spawn().unwrap();
            let owner = child.stdin.take();
            let mut stdout = child.stdout.take().unwrap();
            let mut stderr = child.stderr.take().unwrap();
            let out = thread::spawn(move || {
                let mut bytes = Vec::new();
                stdout.read_to_end(&mut bytes).unwrap();
                bytes
            });
            let err = thread::spawn(move || {
                let mut bytes = Vec::new();
                stderr.read_to_end(&mut bytes).unwrap();
                bytes
            });
            let status = child.wait().unwrap();
            drop(owner);
            let log = String::from_utf8(out.join().unwrap()).unwrap();
            let error = String::from_utf8(err.join().unwrap()).unwrap();
            assert!(status.success(), "{error}");
            log
        }
    }

    #[test]
    fn remote_aliases_paths_and_shell_arguments_are_validated() {
        let fixture = Fixture::new();
        assert!(fixture.target.validate().is_ok());
        for host in ["user@host", "user@[::1]", "ssh-alias"] {
            let mut target = fixture.target.clone();
            target.host = host.into();
            assert!(target.validate().is_ok());
        }
        for host in ["-oProxyCommand=evil", "host; echo bad", "host\nother", ""] {
            let mut target = fixture.target.clone();
            target.host = host.into();
            assert!(target.validate().is_err());
        }
        let mut target = fixture.target.clone();
        target.work_directory = "/".into();
        assert!(target.validate().is_err());
        target = fixture.target.clone();
        target.asset_mappings = vec![RemoteAssetMapping {
            local_path: "/tmp/a".into(),
            remote_path: "../escape".into(),
        }];
        assert!(target.validate().is_err());
        assert_eq!(shell_quote("a ' $(literal)"), "'a '\\'' $(literal)'");
    }

    #[test]
    fn remote_probe_stage_validate_launch_logs_and_atomic_history_mirror() {
        let mut fixture = Fixture::new();
        let asset = fixture.root.join("local sounding");
        fs::write(&asset, b"exact sounding bytes\n").unwrap();
        fixture.target.asset_mappings.push(RemoteAssetMapping {
            local_path: asset.display().to_string(),
            remote_path: "assets/sounding.txt".into(),
        });
        let stage = fixture.stage("schema: nimbus.v1\nloop: false\n");
        assert_eq!(
            stage.source_sha256,
            format!("{:x}", Sha256::digest(b"schema: nimbus.v1\nloop: false\n"))
        );
        assert_eq!(
            fs::read(&stage.config_path).unwrap(),
            b"schema: nimbus.v1\nloop: false\n"
        );
        assert_eq!(
            fs::read(&stage.assets[0].remote_path).unwrap(),
            b"exact sounding bytes\n"
        );
        assert_eq!(stage.validate_source(false).unwrap().exit_code, 0);
        let log = fixture.complete(&stage);
        assert!(log.contains("t=1.25s step=3"));
        let local = fixture.root.join("local run");
        fs::create_dir(&local).unwrap();
        stage.run.mirror_outputs(&local).unwrap();
        assert_eq!(
            fs::read(local.join("step_000000/theta_prime.npy")).unwrap(),
            b"finite source bytes"
        );
        assert_eq!(
            fs::read_to_string(local.join("z.txt")).unwrap(),
            "completed"
        );
        assert!(!fs::read_dir(&local).unwrap().any(|entry| entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with('.')));
    }

    #[test]
    fn remote_relative_assets_are_job_private_and_changed_inputs_invalidate_launch_identity() {
        let mut fixture = Fixture::new();
        let local = fixture.root.join("local sounding");
        fs::write(&local, b"original source").unwrap();
        fixture.target.asset_mappings.push(RemoteAssetMapping {
            local_path: local.display().to_string(),
            remote_path: "sounding.txt".into(),
        });
        let first = fixture.stage("schema: nimbus.v1\n");
        fs::write(&local, b"different source").unwrap();
        let second = fixture.stage("schema: nimbus.v1\n");
        assert_ne!(first.assets[0].remote_path, second.assets[0].remote_path);
        assert_eq!(
            fs::read(&first.assets[0].remote_path).unwrap(),
            b"original source"
        );
        assert_eq!(
            fs::read(&second.assets[0].remote_path).unwrap(),
            b"different source"
        );
        assert_eq!(first.validate_source(false).unwrap().exit_code, 0);
        assert_ne!(
            first.execution_target_fingerprint,
            second.execution_target_fingerprint
        );
        assert!(second
            .verify_launch_identity(
                Some(&first.probe.binary_fingerprint),
                Some(&first.probe.catalog_fingerprint),
                Some(&first.execution_target_fingerprint)
            )
            .is_err());
        assert!(first
            .verify_launch_identity(
                Some(&first.probe.binary_fingerprint),
                Some(&first.probe.catalog_fingerprint),
                Some(&first.execution_target_fingerprint)
            )
            .is_ok());
    }

    #[test]
    fn remote_absolute_mapped_assets_are_verified_without_overwriting_shared_inputs() {
        let mut fixture = Fixture::new();
        let local = fixture.root.join("local sounding");
        fs::write(&local, b"immutable source").unwrap();
        let existing = fixture.root.join("shared sounding");
        fs::write(&existing, b"immutable source").unwrap();
        fixture.target.asset_mappings.push(RemoteAssetMapping {
            local_path: local.display().to_string(),
            remote_path: existing.display().to_string(),
        });
        let first = fixture.stage("schema: nimbus.v1\n");
        assert_eq!(first.assets[0].remote_path, existing.display().to_string());
        fs::write(&local, b"changed local source").unwrap();
        assert!(stage_source_with_transport(
            fixture.target.clone(),
            "schema: nimbus.v1\n",
            fixture.transport.clone()
        )
        .is_err());
        assert_eq!(fs::read(&existing).unwrap(), b"immutable source");
    }

    #[test]
    fn remote_launch_provenance_records_the_remote_executable_and_staged_asset_bytes() {
        use super::super::provenance::{
            create_observed_run_provenance, read_validated_run_provenance, ExecutableProvenance,
            ProvenanceDigest, RemoteExecutionProvenance, RemoteSourceAssetProvenance,
            RequestedRunValues, PROVENANCE_FILE_NAME,
        };
        let mut fixture = Fixture::new();
        let local = fixture.root.join("local sounding");
        fs::write(&local, b"bound sounding bytes").unwrap();
        fixture.target.asset_mappings.push(RemoteAssetMapping {
            local_path: local.display().to_string(),
            remote_path: "sounding.txt".into(),
        });
        let source = "schema: nimbus.v1\n";
        let stage = fixture.stage(source);
        let config = fixture.root.join("launch.yaml");
        fs::write(&config, source).unwrap();
        let output = fixture.root.join("local run");
        fs::create_dir(&output).unwrap();
        create_observed_run_provenance(
            &output,
            "remote-test",
            1,
            ExecutableProvenance {
                canonical_path: stage.probe.binary_path.clone(),
                digest: ProvenanceDigest {
                    algorithm: "sha256".into(),
                    value: stage.probe.binary_fingerprint[7..].into(),
                },
            },
            &config,
            Some(stage.probe.catalog_fingerprint.clone()),
            RequestedRunValues {
                sounding_path: Some("sounding.txt".into()),
                ..RequestedRunValues::default()
            },
            Some(RemoteExecutionProvenance {
                target: stage.run.target.clone(),
                remote_output_directory: stage.run.output_path.clone(),
                staged_config_sha256: stage.source_sha256.clone(),
                execution_target_fingerprint: stage.execution_target_fingerprint.clone(),
                assets: stage
                    .assets
                    .iter()
                    .map(|asset| RemoteSourceAssetProvenance {
                        remote_path: asset.remote_path.clone(),
                        requested_path: asset.requested_path.clone(),
                        sha256: asset.sha256.clone(),
                    })
                    .collect(),
            }),
        )
        .unwrap();
        fs::remove_file(local).unwrap();
        let provenance = read_validated_run_provenance(&output).unwrap();
        assert_eq!(provenance.executable.digest.algorithm, "sha256");
        let sounding = &provenance.source_artifacts["initialConditions.soundingFile"];
        assert_eq!(
            sounding.canonical_path.as_deref(),
            Some(stage.assets[0].remote_path.as_str())
        );
        assert_eq!(
            sounding.digest.as_ref().unwrap().value,
            stage.assets[0].sha256
        );
        let mut changed = serde_json::to_value(&provenance).unwrap();
        changed["remoteExecution"]["stagedConfigSha256"] = json!("0".repeat(64));
        fs::write(
            output.join(PROVENANCE_FILE_NAME),
            serde_json::to_vec(&changed).unwrap(),
        )
        .unwrap();
        assert!(read_validated_run_provenance(&output).is_err());
    }

    #[test]
    fn remote_mismatches_and_strict_admission_reject_without_local_fallback() {
        let fixture = Fixture::new();
        let rejected = fixture.stage("schema: nimbus.v1\nreject: true\n");
        assert_eq!(rejected.validate_source(false).unwrap().exit_code, 2);
        let source_changed = fixture.stage("schema: nimbus.v1\n");
        fs::write(&source_changed.config_path, b"changed").unwrap();
        assert!(source_changed.validate_source(false).is_err());
        let binary_changed = fixture.stage("schema: nimbus.v1\n");
        let mut file = OpenOptions::new()
            .append(true)
            .open(&fixture.target.binary_path)
            .unwrap();
        writeln!(file, "# changed").unwrap();
        assert!(binary_changed.validate_source(false).is_err());
    }

    #[test]
    fn remote_abort_terminates_the_real_remote_process_and_is_idempotent() {
        let fixture = Fixture::new();
        let stage = fixture.stage("schema: nimbus.v1\nloop: true\n");
        let mut child = stage.spawn().unwrap();
        let owner = child.stdin.take();
        let receipt = fixture
            .root
            .join(format!(".nimbus-studio/{}/process.json", stage.run.token));
        let deadline = Instant::now() + Duration::from_secs(10);
        while !receipt.exists() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(20));
        }
        assert!(receipt.exists());
        stage.run.cancel().unwrap();
        assert!(!child.wait().unwrap().success());
        drop(owner);
        stage.run.cancel().unwrap();
        let final_receipt: Value = serde_json::from_slice(&fs::read(receipt).unwrap()).unwrap();
        assert!(final_receipt["exitCode"].is_number());
    }

    #[test]
    fn remote_early_abort_cannot_be_followed_by_late_process_creation() {
        let fixture = Fixture::new();
        let stage = fixture.stage("schema: nimbus.v1\nloop: true\n");
        fs::write(fixture.root.join("catalog-delay"), b"slow").unwrap();
        let mut child = stage.spawn().unwrap();
        let owner = child.stdin.take();
        stage.run.cancel().unwrap();
        assert!(!child.wait().unwrap().success());
        drop(owner);
        assert!(!Path::new(&stage.run.output_path).exists());
    }

    #[test]
    fn remote_disconnect_closes_the_remote_process_ownership_channel() {
        let fixture = Fixture::new();
        let stage = fixture.stage("schema: nimbus.v1\nloop: true\n");
        let mut child = stage.spawn().unwrap();
        stage.run.revoke_transport_ownership();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(!status.success());
                break;
            }
            assert!(
                Instant::now() < deadline,
                "remote simulator survived disconnect"
            );
            thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn remote_disconnect_after_receipt_retires_the_started_process() {
        let fixture = Fixture::new();
        let stage = fixture.stage("schema: nimbus.v1\nloop: true\n");
        let mut child = stage.spawn().unwrap();
        let receipt = fixture
            .root
            .join(format!(".nimbus-studio/{}/process.json", stage.run.token));
        let deadline = Instant::now() + Duration::from_secs(10);
        while !receipt.exists() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(20));
        }
        assert!(receipt.exists());
        stage.run.revoke_transport_ownership();
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(!status.success());
                break;
            }
            assert!(
                Instant::now() < deadline,
                "remote simulator survived post-receipt disconnect"
            );
            thread::sleep(Duration::from_millis(20));
        }
        let value: Value = serde_json::from_slice(&fs::read(receipt).unwrap()).unwrap();
        assert!(value["exitCode"].is_number());
    }

    #[test]
    fn cancelled_manifest_work_retires_without_publishing_local_frames() {
        let fixture = Fixture::new();
        let stage = fixture.stage("schema: nimbus.v1\n");
        fixture.complete(&stage);
        let cancelled = AtomicBool::new(true);
        assert!(fixture
            .transport
            .run_cancelable(
                &fixture.target,
                "files",
                json!({"outputPath":stage.run.output_path}),
                None,
                Duration::from_secs(30),
                Some(&cancelled)
            )
            .is_err());
        assert_eq!(
            fs::read(Path::new(&stage.run.output_path).join("step_000000/theta_prime.npy"))
                .unwrap(),
            b"finite source bytes"
        );
    }

    #[test]
    fn receive_recovery_supplement_keeps_original_runtime_valid_after_new_publications_arrive() {
        use super::super::{remote_receive, provenance, outputs};
        let fixture = Fixture::new(); let stage = fixture.stage("schema: nimbus.v1\n"); fixture.complete(&stage);
        let local = fixture.root.join("qualified recovery local"); let launch = fixture.launch(&stage, &local);
        let generation = "11111111111111111111111111111111";
        let marker = json!({"schema":"nimbus.output","manifest_version":2,"frame_completion":"atomic_step_manifest", "publication_generation":generation,"publication_state":"ready"});
        fs::write(local.join("nimbus_output.json"), serde_json::to_vec(&marker).unwrap()).unwrap();
        let original = provenance::create_runtime_completion_provenance(&local, provenance::RuntimeCompletionInput { run_id: launch.run_id.clone(),
            process_started_unix_ms: 1, process_ended_unix_ms: 2, final_status: "failed".into(), exit_code: Some(1), log_evidence: Default::default() }).unwrap();
        assert_eq!(original.output.complete_frame_count, Some(0));
        let runtime_bytes = fs::read(local.join(provenance::RUNTIME_PROVENANCE_FILE_NAME)).unwrap();
        let remote_output = Path::new(&stage.run.output_path);
        fs::write(remote_output.join("nimbus_output.json"), serde_json::to_vec(&marker).unwrap()).unwrap();
        fs::write(remote_output.join("step_000000/manifest.json"), serde_json::to_vec(&json!({"schema":"nimbus.step", "manifest_version":2,
            "publication_state":"complete", "publication_generation":generation, "step_index":0, "step_dir":"step_000000", "simulation_time_s":10.0,
            "grid":{"nr":2,"nth":2,"nz":2,"coordinate_system":"cartesian","staggering":"c_grid","dr_m":1.0,"dy_m":1.0,"dz_m":1.0,
                "scalar_origin_m":[0.5,0.5,0.5],"physical_bounds_m":{"lower":[0.0,0.0,0.0],"upper":[2.0,2.0,2.0]},
                "valid_index_range":{"lower":[0,0,0],"upper":[1,1,1]},"halo_width":{"lower":[0,0,0],"upper":[0,0,0]}},
            "published_fields":["theta_prime"],"fields":[{"field_id":"theta_prime","units":"K"}]})).unwrap()).unwrap();
        let mut header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2, 2), }".to_string();
        while (10 + header.len() + 1) % 64 != 0 { header.push(' '); } header.push('\n');
        let mut npy = b"\x93NUMPY\x01\x00".to_vec(); npy.extend_from_slice(&(header.len() as u16).to_le_bytes()); npy.extend_from_slice(header.as_bytes());
        for value in [0.0f32, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0] { npy.extend_from_slice(&value.to_le_bytes()); }
        fs::write(remote_output.join("step_000000/theta_prime.npy"), npy).unwrap();
        remote_receive::receive_with_run(&local, &launch, fixture.restore(&stage)).unwrap();
        assert_eq!(outputs::inspect_qualified_output_summary(&local).unwrap().complete_frame_count, 1);
        let recovered = provenance::read_validated_runtime_completion(&local, &launch).unwrap();
        assert_eq!(recovered.output.complete_frame_count, Some(0));
        assert_eq!(fs::read(local.join(provenance::RUNTIME_PROVENANCE_FILE_NAME)).unwrap(), runtime_bytes);
        let field = local.join("step_000000/theta_prime.npy"); let mut changed = fs::read(&field).unwrap(); let end = changed.len()-1; changed[end] ^= 1; fs::write(field, changed).unwrap();
        assert!(provenance::read_validated_runtime_completion(&local, &launch).is_err());
    }

    #[test]
    fn receive_recovery_restores_terminal_jobs_reuses_identical_files_and_preserves_runtime_evidence() {
        use super::super::{remote_receive, provenance};
        let fixture = Fixture::new(); let stage = fixture.stage("schema: nimbus.v1\n"); fixture.complete(&stage);
        let local = fixture.root.join("recovery local"); let launch = fixture.launch(&stage, &local);
        provenance::create_runtime_completion_provenance(&local, provenance::RuntimeCompletionInput { run_id: launch.run_id.clone(),
            process_started_unix_ms: 1, process_ended_unix_ms: 2, final_status: "failed".into(), exit_code: Some(1), log_evidence: Default::default() }).unwrap();
        let original = fs::read(local.join(provenance::RUNTIME_PROVENANCE_FILE_NAME)).unwrap();
        let digest = crate::platform::sha256_file(&local.join(provenance::RUNTIME_PROVENANCE_FILE_NAME)).unwrap();
        remote_receive::record_receive_state(&local, &launch, "failed", Some("prior transport failed".into())).unwrap();
        assert!(remote_receive::remote_receive_summary(&local).unwrap().retry_available);
        let received = remote_receive::receive_with_run(&local, &launch, fixture.restore(&stage)).unwrap(); assert_eq!(received.state, "complete");
        assert!(remote_receive::validate_recovered_remote_outputs(&local, &launch, &digest).unwrap());
        let fetches = fs::read_to_string(fixture.root.join("fetch-count")).unwrap();
        assert_eq!(remote_receive::receive_with_run(&local, &launch, fixture.restore(&stage)).unwrap().state, "complete");
        assert_eq!(fs::read_to_string(fixture.root.join("fetch-count")).unwrap(), fetches);
        assert_eq!(fs::read(local.join(provenance::RUNTIME_PROVENANCE_FILE_NAME)).unwrap(), original);
        assert_eq!(fs::read(local.join("step_000000/theta_prime.npy")).unwrap(), b"finite source bytes");
        assert_eq!(fs::read(Path::new(&stage.run.output_path).join("step_000000/theta_prime.npy")).unwrap(), b"finite source bytes");
        assert!(remote_receive::validate_recovered_remote_outputs(&local, &launch, &digest).unwrap());
        let field = local.join("step_000000/theta_prime.npy"); let mut changed = fs::read(&field).unwrap(); changed[0] ^= 1; fs::write(field, changed).unwrap();
        assert!(remote_receive::validate_recovered_remote_outputs(&local, &launch, &digest).is_err());
    }

    #[test]
    fn receive_recovery_rejects_receipt_inventory_and_target_tampering() {
        use super::super::{remote_receive, provenance};
        let fixture = Fixture::new(); let stage = fixture.stage("schema: nimbus.v1\n"); fixture.complete(&stage);
        let local = fixture.root.join("tamper local"); let launch = fixture.launch(&stage, &local);
        provenance::create_runtime_completion_provenance(&local, provenance::RuntimeCompletionInput { run_id: launch.run_id.clone(),
            process_started_unix_ms: 1, process_ended_unix_ms: 2, final_status: "failed".into(), exit_code: Some(1), log_evidence: Default::default() }).unwrap();
        let digest = crate::platform::sha256_file(&local.join(provenance::RUNTIME_PROVENANCE_FILE_NAME)).unwrap();
        remote_receive::receive_with_run(&local, &launch, fixture.restore(&stage)).unwrap();
        assert!(!remote_receive::validate_recovered_remote_outputs(&local, &launch, &format!("sha256:{}", "0".repeat(64))).unwrap_or(false));
        let inventory = local.join("remote-receive-inventory.json"); let mut changed: Value = serde_json::from_slice(&fs::read(&inventory).unwrap()).unwrap();
        changed["files"][0]["sha256"] = json!("0".repeat(64)); fs::write(inventory, serde_json::to_vec(&changed).unwrap()).unwrap();
        assert!(remote_receive::validate_recovered_remote_outputs(&local, &launch, &digest).is_err());
        let receipt = local.join("remote-receive.json"); let mut changed: Value = serde_json::from_slice(&fs::read(&receipt).unwrap()).unwrap();
        changed["runId"] = json!("different run"); fs::write(receipt, serde_json::to_vec(&changed).unwrap()).unwrap();
        assert!(!remote_receive::remote_receive_summary(&local).unwrap().retry_available);
        let mut execution = fixture.execution(&stage); execution.remote_output_directory.push_str("/nested"); assert!(RemoteRun::from_execution(&execution).is_err());
        execution = fixture.execution(&stage); execution.target.host = "another-host".into(); assert!(RemoteRun::from_execution(&execution).is_err());
    }

    #[test]
    fn receive_recovery_cancels_only_transfer_rejects_duplicates_and_recovers_interrupted_staging() {
        use super::super::remote_receive;
        let fixture = Fixture::new(); let stage = fixture.stage("schema: nimbus.v1\n"); fixture.complete(&stage);
        let local = fixture.root.join("cancel local"); let launch = fixture.launch(&stage, &local);
        let remote = fixture.restore(&stage); fs::write(fixture.root.join("receive-delay"), b"hold").unwrap();
        let worker_remote = remote.clone(); let worker_local = local.clone(); let worker_launch = launch.clone();
        let worker = thread::spawn(move || remote_receive::receive_with_run(&worker_local, &worker_launch, worker_remote).unwrap());
        let deadline = Instant::now() + Duration::from_secs(5);
        while !fixture.root.join("receive-started").exists() { assert!(Instant::now() < deadline); thread::sleep(Duration::from_millis(20)); }
        assert_eq!(remote_receive::remote_receive_summary(&local).unwrap().state, "receiving");
        assert!(remote_receive::receive_with_run(&local, &launch, fixture.restore(&stage)).is_err());
        remote.cancel_receive(); assert_eq!(worker.join().unwrap().state, "cancelled");
        assert!(!local.join("step_000000").exists());
        assert!(!Path::new(&stage.run.output_path).parent().unwrap().join("cancelled").exists());
        fs::remove_file(fixture.root.join("receive-delay")).unwrap();
        let staging = local.join(format!(".remote-download-{}", remote.token)); fs::create_dir(&staging).unwrap(); fs::write(staging.join("leftover"), b"partial").unwrap();
        remote_receive::record_receive_state(&local, &launch, "receiving", None).unwrap();
        assert_eq!(remote_receive::remote_receive_summary(&local).unwrap().state, "cancelled");
        assert_eq!(remote_receive::receive_with_run(&local, &launch, fixture.restore(&stage)).unwrap().state, "complete");
        assert!(!staging.exists());
    }

    #[test]
    fn receive_recovery_rejects_active_jobs_without_stopping_compute() {
        use super::super::remote_receive;
        let fixture = Fixture::new(); let stage = fixture.stage("schema: nimbus.v1\nloop: true\n");
        let local = fixture.root.join("active local");
        let mut child = stage.spawn().unwrap(); let receipt = fixture.root.join(format!(".nimbus-studio/{}/process.json", stage.run.token));
        let deadline = Instant::now() + Duration::from_secs(5);
        while !receipt.exists() { assert!(Instant::now() < deadline); thread::sleep(Duration::from_millis(20)); }
        assert!(fixture.restore(&stage).confirm_stopped().is_err());
        assert!(child.try_wait().unwrap().is_none());
        stage.run.cancel().unwrap(); let _ = child.wait();
        fs::create_dir(local).unwrap();
        let _ = remote_receive::remote_receive_summary(&fixture.root);
    }

    #[test]
    fn remote_transfer_failure_and_destination_conflict_publish_no_partial_frames() {
        let fixture = Fixture::new();
        let stage = fixture.stage("schema: nimbus.v1\n");
        fixture.complete(&stage);
        let local = fixture.root.join("local run");
        fs::create_dir(&local).unwrap();
        fs::write(local.join("z.txt"), b"preserved").unwrap();
        assert!(stage.run.mirror_outputs(&local).is_err());
        assert!(!local.join("step_000000").exists());
        assert_eq!(fs::read(local.join("z.txt")).unwrap(), b"preserved");
        fs::remove_file(local.join("z.txt")).unwrap();
        fs::write(fixture.root.join("corrupt-download"), b"corrupt").unwrap();
        assert!(stage.run.mirror_outputs(&local).is_err());
        assert_eq!(fs::read_dir(&local).unwrap().count(), 0);
        assert_eq!(
            fs::read(Path::new(&stage.run.output_path).join("step_000000/theta_prime.npy"))
                .unwrap(),
            b"finite source bytes"
        );
    }

    #[test]
    fn shutdown_ignores_remote_stale_pids_and_does_not_wait_for_unreachable_ssh() {
        use crate::state::{AppState, RunRecord, RunStatus};
        use std::os::unix::process::CommandExt;
        let fixture = Fixture::new();
        fs::write(&fixture.transport.ssh, "#!/bin/sh\nsleep 20\n").unwrap();
        let mut unrelated = Command::new("sleep")
            .arg("30")
            .process_group(0)
            .spawn()
            .unwrap();
        for (transfer, unconfirmed, exited) in [
            (true, false, true),
            (false, true, true),
            (false, false, true),
            (false, false, false),
        ] {
            let state = AppState::default();
            let remote = RemoteRun {
                target: fixture.target.clone(),
                token: token(),
                output_path: fixture.root.join("output").display().to_string(),
                transport: fixture.transport.clone(),
                mirror_cancelled: Arc::new(AtomicBool::new(false)),
                mirroring: Arc::new(AtomicBool::new(transfer)),
                connection_owner: Arc::new(Mutex::new(None)),
                ownership_revoked: Arc::new(AtomicBool::new(false)),
                shutdown_requested: Arc::new(AtomicBool::new(false)),
            };
            state.runs.lock().unwrap().insert(
                "ended-remote".into(),
                Arc::new(Mutex::new(RunRecord {
                    id: "ended-remote".into(),
                    case_name: "fixture".into(),
                    scheme: String::new(),
                    duration_s: 0.0,
                    config_path: fixture.root.join("source.yaml"),
                    pid: unrelated.id(),
                    status: RunStatus::Running,
                    abort_requested: false,
                    progress: None,
                    started_at_ms: 0,
                    ended_at_ms: None,
                    output_dir: fixture.root.clone(),
                    remote: Some(remote),
                    remote_transfer_in_progress: transfer,
                    remote_termination_unconfirmed: unconfirmed,
                    transport_exited: exited,
                    runtime_log_evidence: Arc::new(Mutex::new(
                        super::super::provenance::RuntimeLogEvidence::default(),
                    )),
                })),
            );
            let began = Instant::now();
            super::super::runner::terminate_all_runs(&state);
            assert!(
                began.elapsed() < Duration::from_millis(750),
                "unreachable SSH delayed desktop shutdown"
            );
            assert!(
                unrelated.try_wait().unwrap().is_none(),
                "an ended remote run signalled a recycled local PID"
            );
        }
        unrelated.kill().unwrap();
        unrelated.wait().unwrap();
        thread::sleep(Duration::from_millis(1200));
    }
}
