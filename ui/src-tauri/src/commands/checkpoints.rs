use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{AppError, AppResult};

const LIFECYCLE_SCHEMA: &str = "nimbus.checkpoint.lifecycle";
const LIFECYCLE_SCHEMA_VERSION: u32 = 1;
const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME: u64 = 1_099_511_628_211;
const WIRE_CHECKSUM_OFFSET: u64 = 1_469_598_103_934_665_603;
const SIMULATION_MAGIC: &[u8; 8] = b"NIMBSIM1";
const SIMULATION_CHECKPOINT_VERSION: u32 = 2;
const WIRE_HEADER_BYTES: u64 = 8 + 4 + 8 + 8;
const MAX_CHECKPOINT_BYTES: u64 = 16 * 1024 * 1024 * 1024 * 1024;
const MAX_CHECKPOINT_SIDECAR_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Deserialize)]
struct CheckpointSidecar {
    schema: String,
    schema_version: u32,
    checkpoint_file: String,
    checkpoint_bytes: u64,
    checkpoint_fnv1a64: String,
    run_id: String,
    #[serde(default)]
    parent_run_id: String,
    #[serde(default)]
    parent_checkpoint: String,
    completed_steps: u64,
    simulation_time_s: f64,
    boundary: String,
    portable_across_compute_memory_topology: bool,
}

/// Evidence returned only after Studio has read both files and reproduced the
/// simulator's byte-count + FNV-1a integrity check. Scientific-configuration
/// equality is intentionally not claimed here: the simulator compares the
/// decoded checkpoint state to the requested config before replacing live
/// state.
#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CheckpointInspection {
    pub(crate) path: String,
    sidecar_path: String,
    schema: String,
    schema_version: u32,
    pub(crate) checkpoint_bytes: u64,
    pub(crate) checkpoint_fnv1a64: String,
    pub(crate) run_id: String,
    parent_run_id: Option<String>,
    parent_checkpoint: Option<String>,
    pub(crate) completed_steps: u64,
    simulation_time_s: f64,
    boundary: String,
    portable_across_compute_memory_topology: bool,
    integrity_verified: bool,
    format_compatible: bool,
    scientific_compatibility: &'static str,
}

fn validate_checkpoint_path(path: &str) -> AppResult<PathBuf> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(AppError::Invalid("checkpoint path is empty".into()));
    }
    let candidate = PathBuf::from(trimmed);
    let extension_ok = candidate
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.eq_ignore_ascii_case("nimbus"))
        .unwrap_or(false);
    if !extension_ok {
        return Err(AppError::Invalid(
            "checkpoint needs a .nimbus extension".into(),
        ));
    }
    let canonical = candidate
        .canonicalize()
        .map_err(|_| AppError::NotFound(trimmed.into()))?;
    let metadata = fs::symlink_metadata(&canonical)?;
    if !metadata.file_type().is_file() {
        return Err(AppError::Invalid(
            "checkpoint must be a regular file".into(),
        ));
    }
    if metadata.len() > MAX_CHECKPOINT_BYTES {
        return Err(AppError::Invalid(format!(
            "checkpoint exceeds the {} GiB inspection limit",
            MAX_CHECKPOINT_BYTES / (1024 * 1024 * 1024)
        )));
    }
    Ok(canonical)
}

fn sidecar_path(checkpoint: &Path) -> PathBuf {
    PathBuf::from(format!("{}.json", checkpoint.display()))
}

#[cfg(test)]
fn fnv1a64(path: &Path) -> AppResult<(u64, String)> {
    let mut file = File::open(path)?;
    let mut buffer = [0_u8; 64 * 1024];
    let mut length = 0_u64;
    let mut digest = FNV_OFFSET_BASIS;
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        length = length
            .checked_add(count as u64)
            .ok_or_else(|| AppError::Invalid("checkpoint byte count overflowed".into()))?;
        for byte in &buffer[..count] {
            digest ^= u64::from(*byte);
            digest = digest.wrapping_mul(FNV_PRIME);
        }
    }
    Ok((length, format!("{digest:016x}")))
}

fn inspect_simulation_checkpoint_stream(path: &Path, file_bytes: u64) -> AppResult<(u64, String)> {
    if file_bytes < WIRE_HEADER_BYTES {
        return Err(AppError::Invalid(
            "selected file is too short to be a Nimbus simulation checkpoint".into(),
        ));
    }
    let mut file = File::open(path)?;
    let mut header = [0_u8; WIRE_HEADER_BYTES as usize];
    file.read_exact(&mut header)?;
    if &header[..8] != SIMULATION_MAGIC {
        return Err(AppError::Invalid(
            "selected file does not have Nimbus simulation checkpoint magic".into(),
        ));
    }
    let version = u32::from_le_bytes(header[8..12].try_into().expect("fixed header"));
    if version != SIMULATION_CHECKPOINT_VERSION {
        return Err(AppError::Invalid(format!(
            "simulation checkpoint format v{version} is not supported (expected v{SIMULATION_CHECKPOINT_VERSION})"
        )));
    }
    let payload_bytes = u64::from_le_bytes(header[12..20].try_into().expect("fixed header"));
    let expected_checksum = u64::from_le_bytes(header[20..28].try_into().expect("fixed header"));
    if payload_bytes != file_bytes - WIRE_HEADER_BYTES {
        return Err(AppError::Invalid(
            "simulation checkpoint payload length does not match its envelope".into(),
        ));
    }

    let mut checkpoint_digest = FNV_OFFSET_BASIS;
    for byte in header {
        checkpoint_digest ^= u64::from(byte);
        checkpoint_digest = checkpoint_digest.wrapping_mul(FNV_PRIME);
    }
    let mut payload_checksum = WIRE_CHECKSUM_OFFSET;
    let mut buffer = [0_u8; 64 * 1024];
    let mut consumed = WIRE_HEADER_BYTES;
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        consumed += count as u64;
        for byte in &buffer[..count] {
            payload_checksum ^= u64::from(*byte);
            payload_checksum = payload_checksum.wrapping_mul(FNV_PRIME);
            checkpoint_digest ^= u64::from(*byte);
            checkpoint_digest = checkpoint_digest.wrapping_mul(FNV_PRIME);
        }
    }
    if consumed != file_bytes || payload_checksum != expected_checksum {
        return Err(AppError::Invalid(
            "simulation checkpoint envelope checksum does not match its payload".into(),
        ));
    }
    Ok((consumed, format!("{checkpoint_digest:016x}")))
}

pub(crate) fn inspect_checkpoint_blocking(path: &str) -> AppResult<CheckpointInspection> {
    let checkpoint = validate_checkpoint_path(path)?;
    let sidecar = sidecar_path(&checkpoint);
    let sidecar_metadata = fs::symlink_metadata(&sidecar).map_err(|_| {
        AppError::NotFound(format!("matching checkpoint sidecar {}", sidecar.display()))
    })?;
    if !sidecar_metadata.file_type().is_file()
        || sidecar_metadata.len() > MAX_CHECKPOINT_SIDECAR_BYTES
    {
        return Err(AppError::Invalid(
            "checkpoint sidecar must be a regular file no larger than 1 MiB".into(),
        ));
    }

    let sidecar_text = fs::read_to_string(&sidecar)?;
    let lifecycle: CheckpointSidecar = serde_json::from_str(&sidecar_text).map_err(|error| {
        AppError::Invalid(format!(
            "checkpoint sidecar is not valid lifecycle JSON: {error}"
        ))
    })?;
    let filename = checkpoint
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| AppError::Invalid("checkpoint filename is not valid UTF-8".into()))?;

    if lifecycle.schema != LIFECYCLE_SCHEMA || lifecycle.schema_version != LIFECYCLE_SCHEMA_VERSION
    {
        return Err(AppError::Invalid(format!(
            "checkpoint sidecar schema {} v{} is not supported by this Studio (expected {} v{})",
            lifecycle.schema, lifecycle.schema_version, LIFECYCLE_SCHEMA, LIFECYCLE_SCHEMA_VERSION
        )));
    }
    if lifecycle.checkpoint_file != filename {
        return Err(AppError::Invalid(format!(
            "checkpoint sidecar names '{}' but the selected file is '{filename}'",
            lifecycle.checkpoint_file
        )));
    }
    if lifecycle.run_id.trim().is_empty() {
        return Err(AppError::Invalid(
            "checkpoint sidecar does not identify its source run".into(),
        ));
    }
    if lifecycle.boundary != "completed_timestep" {
        return Err(AppError::Invalid(format!(
            "checkpoint boundary '{}' is not restart-safe; completed_timestep is required",
            lifecycle.boundary
        )));
    }
    if !lifecycle.portable_across_compute_memory_topology {
        return Err(AppError::Invalid(
            "checkpoint does not declare portability across compute-memory topologies".into(),
        ));
    }
    if !lifecycle.simulation_time_s.is_finite() || lifecycle.simulation_time_s < 0.0 {
        return Err(AppError::Invalid(
            "checkpoint sidecar has an invalid simulation time".into(),
        ));
    }
    if lifecycle.checkpoint_fnv1a64.len() != 16
        || !lifecycle
            .checkpoint_fnv1a64
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(AppError::Invalid(
            "checkpoint sidecar has an invalid FNV-1a digest".into(),
        ));
    }

    let checkpoint_bytes = fs::metadata(&checkpoint)?.len();
    if checkpoint_bytes != lifecycle.checkpoint_bytes {
        return Err(AppError::Invalid(format!(
            "checkpoint integrity does not match its sidecar (expected {} bytes / {}, found {} bytes before streaming)",
            lifecycle.checkpoint_bytes, lifecycle.checkpoint_fnv1a64, checkpoint_bytes
        )));
    }
    let (actual_bytes, actual_digest) =
        inspect_simulation_checkpoint_stream(&checkpoint, checkpoint_bytes)?;
    if actual_bytes != lifecycle.checkpoint_bytes || actual_digest != lifecycle.checkpoint_fnv1a64 {
        return Err(AppError::Invalid(format!(
            "checkpoint integrity does not match its sidecar (expected {} bytes / {}, read {} bytes / {})",
            lifecycle.checkpoint_bytes,
            lifecycle.checkpoint_fnv1a64,
            actual_bytes,
            actual_digest
        )));
    }
    Ok(CheckpointInspection {
        path: checkpoint.display().to_string(),
        sidecar_path: sidecar.display().to_string(),
        schema: lifecycle.schema,
        schema_version: lifecycle.schema_version,
        checkpoint_bytes: actual_bytes,
        checkpoint_fnv1a64: actual_digest,
        run_id: lifecycle.run_id,
        parent_run_id: (!lifecycle.parent_run_id.is_empty()).then_some(lifecycle.parent_run_id),
        parent_checkpoint: (!lifecycle.parent_checkpoint.is_empty())
            .then_some(lifecycle.parent_checkpoint),
        completed_steps: lifecycle.completed_steps,
        simulation_time_s: lifecycle.simulation_time_s,
        boundary: lifecycle.boundary,
        portable_across_compute_memory_topology: lifecycle.portable_across_compute_memory_topology,
        integrity_verified: true,
        format_compatible: true,
        scientific_compatibility: "verified_before_state_restore",
    })
}

#[tauri::command]
pub async fn inspect_checkpoint(path: String, execution_target: Option<super::remote_compute::RemoteComputeTarget>) -> AppResult<CheckpointInspection> {
    tauri::async_runtime::spawn_blocking(move || {
        if let Some(target) = execution_target {
            if let Some(mapping) = target.asset_mappings.iter().find(|mapping| mapping.remote_path == path) {
                let sidecar = target.asset_mappings.iter().find(|mapping| mapping.remote_path == format!("{path}.json"))
                    .ok_or_else(|| AppError::Invalid("mapped checkpoint requires a mapped lifecycle sidecar".into()))?;
                if crate::platform::sha256_file(Path::new(&sidecar.local_path))?
                    != crate::platform::sha256_file(&PathBuf::from(format!("{}.json", mapping.local_path)))? {
                    return Err(AppError::Invalid("mapped checkpoint lifecycle sidecar differs from inspected source".into()));
                }
                let mut inspection = inspect_checkpoint_blocking(&mapping.local_path)?;
                inspection.path = path.clone(); inspection.sidecar_path = format!("{path}.json");
                return Ok(inspection);
            }
            let (temporary, checkpoint, remote_path) = super::remote_compute::copy_remote_checkpoint(target, &path)?;
            let result = inspect_checkpoint_blocking(&checkpoint.display().to_string()).map(|mut inspection| {
                inspection.path = remote_path.clone(); inspection.sidecar_path = format!("{remote_path}.json"); inspection
            });
            let _ = fs::remove_dir_all(temporary);
            result
        } else { inspect_checkpoint_blocking(&path) }
    })
        .await
        .map_err(|error| AppError::Task(error.to_string()))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static NEXT_FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

    struct Fixture {
        directory: PathBuf,
        checkpoint: PathBuf,
        sidecar: PathBuf,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.directory);
        }
    }

    fn fixture() -> Fixture {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let sequence = NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "nimbus-checkpoint-inspection-{}-{unique}-{sequence}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).expect("fixture directory");
        let checkpoint = directory.join("checkpoint_step_000000042.nimbus");
        let payload = b"portable checkpoint fixture";
        let mut payload_checksum = WIRE_CHECKSUM_OFFSET;
        for byte in payload {
            payload_checksum ^= u64::from(*byte);
            payload_checksum = payload_checksum.wrapping_mul(FNV_PRIME);
        }
        let mut checkpoint_bytes = Vec::from(*SIMULATION_MAGIC);
        checkpoint_bytes.extend_from_slice(&SIMULATION_CHECKPOINT_VERSION.to_le_bytes());
        checkpoint_bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        checkpoint_bytes.extend_from_slice(&payload_checksum.to_le_bytes());
        checkpoint_bytes.extend_from_slice(payload);
        fs::write(&checkpoint, checkpoint_bytes).expect("checkpoint");
        let (bytes, digest) = fnv1a64(&checkpoint).expect("digest");
        let sidecar = sidecar_path(&checkpoint);
        fs::write(
            &sidecar,
            serde_json::json!({
                "schema": LIFECYCLE_SCHEMA,
                "schema_version": 1,
                "checkpoint_file": "checkpoint_step_000000042.nimbus",
                "checkpoint_bytes": bytes,
                "checkpoint_fnv1a64": digest,
                "run_id": "run-parent",
                "parent_run_id": "run-grandparent",
                "parent_checkpoint": "/archive/grandparent.nimbus",
                "completed_steps": 42,
                "simulation_time_s": 21.0,
                "boundary": "completed_timestep",
                "portable_across_compute_memory_topology": true
            })
            .to_string(),
        )
        .expect("sidecar");
        Fixture {
            directory,
            checkpoint,
            sidecar,
        }
    }

    #[test]
    fn accepts_complete_integrity_evidence() {
        let fixture = fixture();
        let inspection =
            inspect_checkpoint_blocking(fixture.checkpoint.to_str().unwrap()).expect("inspection");
        assert!(inspection.integrity_verified);
        assert!(inspection.format_compatible);
        assert_eq!(inspection.run_id, "run-parent");
        assert_eq!(inspection.parent_run_id.as_deref(), Some("run-grandparent"));
        assert_eq!(inspection.completed_steps, 42);
        assert_eq!(inspection.simulation_time_s, 21.0);
    }

    #[test]
    fn rejects_tampered_checkpoint() {
        let fixture = fixture();
        fs::write(&fixture.checkpoint, b"tampered checkpoint fixture").expect("tamper");
        let error = inspect_checkpoint_blocking(fixture.checkpoint.to_str().unwrap())
            .expect_err("tamper must fail");
        assert!(error.to_string().contains("integrity does not match"));
    }

    #[test]
    fn rejects_non_simulation_checkpoint_envelope() {
        let fixture = fixture();
        fs::write(&fixture.checkpoint, b"not a simulation checkpoint envelope")
            .expect("replace fixture");
        let bytes = fs::metadata(&fixture.checkpoint).unwrap().len();
        let error = inspect_simulation_checkpoint_stream(&fixture.checkpoint, bytes)
            .expect_err("invalid magic must fail");
        assert!(error.to_string().contains("simulation checkpoint magic"));
    }

    #[test]
    fn rejects_incompatible_or_missing_sidecar() {
        let fixture = fixture();
        fs::write(
            &fixture.sidecar,
            serde_json::json!({
                "schema": LIFECYCLE_SCHEMA,
                "schema_version": 99,
                "checkpoint_file": "checkpoint_step_000000042.nimbus",
                "checkpoint_bytes": 0,
                "checkpoint_fnv1a64": "0000000000000000",
                "run_id": "run-parent",
                "completed_steps": 42,
                "simulation_time_s": 21.0,
                "boundary": "completed_timestep",
                "portable_across_compute_memory_topology": true
            })
            .to_string(),
        )
        .expect("incompatible sidecar");
        let error = inspect_checkpoint_blocking(fixture.checkpoint.to_str().unwrap())
            .expect_err("unsupported schema must fail");
        assert!(error.to_string().contains("is not supported"));

        fs::remove_file(&fixture.sidecar).expect("remove fixture sidecar");
        let error = inspect_checkpoint_blocking(fixture.checkpoint.to_str().unwrap())
            .expect_err("missing sidecar must fail");
        assert!(error.to_string().contains("matching checkpoint sidecar"));
    }
}
