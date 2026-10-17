use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde::Serialize;

use crate::commands::outputs::{run_generation_gate, RunGenerationGate};
use crate::commands::provenance::{
    read_validated_run_provenance, read_validated_runtime_completion, PROVENANCE_FILE_NAME,
    RUNTIME_PROVENANCE_FILE_NAME,
};
use crate::error::{AppError, AppResult};
use super::remote_compute::RemoteComputeTarget;
use super::remote_receive::{remote_receive_summary_with_provenance, RemoteReceiveSummary};

pub(super) const MAX_RUN_DIRECTORIES: usize = 512;
const MAX_ENTRIES_PER_RUN: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LibraryHealth {
    Ready,
    Partial,
    Invalid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ProvenanceHealth {
    Verified,
    Missing,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunProvenanceSummary {
    pub health: ProvenanceHealth,
    pub issue: Option<String>,
    pub runtime_health: ProvenanceHealth,
    pub runtime_issue: Option<String>,
    pub run_id: Option<String>,
    pub created_unix_ms: Option<u128>,
    pub executable_path: Option<String>,
    pub executable_digest: Option<String>,
    pub configuration_digest: Option<String>,
    pub catalog_fingerprint: Option<String>,
    pub requested_backend: Option<String>,
    pub requested_fallback_allowed: Option<bool>,
    pub requested_output_format: Option<String>,
    pub requested_execution_mode: Option<String>,
    pub requested_validation_mode: Option<String>,
    pub requested_schemes: BTreeMap<String, String>,
    pub sounding_requested_path: Option<String>,
    pub sounding_canonical_path: Option<String>,
    pub sounding_digest: Option<String>,
    pub sounding_issue: Option<String>,
    pub runtime_actuals_available: bool,
    pub actual_backend: Option<String>,
    pub fallback_active: Option<bool>,
    pub actual_schemes: BTreeMap<String, String>,
    pub final_status: Option<String>,
    pub exit_code: Option<i32>,
    pub process_started_unix_ms: Option<u128>,
    pub process_ended_unix_ms: Option<u128>,
    pub complete_frame_count: Option<u64>,
    pub actual_output_format: Option<String>,
    pub verified_checkpoint_count: Option<u64>,
    pub latest_checkpoint_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunLibraryEntry {
    pub id: String,
    pub case_name: String,
    pub output_dir: String,
    pub health: LibraryHealth,
    pub issue: Option<String>,
    pub started_at_ms: Option<u128>,
    pub modified_at_ms: Option<u128>,
    pub frame_count: usize,
    pub latest_step: Option<u64>,
    pub simulation_time_s: Option<f64>,
    pub field_count: usize,
    pub output_format: String,
    pub scan_limited: bool,
    pub scan_limit: usize,
    pub provenance: RunProvenanceSummary,
    pub execution_target: Option<RemoteComputeTarget>,
    pub remote_receive: Option<RemoteReceiveSummary>,
}

#[derive(Default)]
struct RemoteLibraryMetadata {
    execution_target: Option<RemoteComputeTarget>,
    remote_receive: Option<RemoteReceiveSummary>,
}

/// Rebuild the local Nimbus run index from disk. The scan is intentionally
/// shallow, sorted, bounded, and read-only: only direct children of the
/// configured runs root can become runs, and symlinked directories are not
/// followed.
#[tauri::command]
pub async fn discover_run_library(runs_root: String) -> AppResult<Vec<RunLibraryEntry>> {
    let root = effective_runs_root(&runs_root);
    tauri::async_runtime::spawn_blocking(move || discover_runs(&root))
        .await
        .map_err(|error| AppError::Task(format!("run-library scan failed: {error}")))?
}

/// Keep discovery aligned with runner::default_output_dir. A blank setting is
/// the native Nimbus default, never PathBuf("") / the process working directory.
fn effective_runs_root(runs_root: &str) -> PathBuf {
    let trimmed = runs_root.trim();
    if trimmed.is_empty() {
        return crate::platform::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".nimbus")
            .join("runs");
    }
    expand_tilde(trimmed)
}

fn discover_runs(root: &Path) -> AppResult<Vec<RunLibraryEntry>> {
    if !root.exists() {
        return Ok(Vec::new());
    }
    if !root.is_dir() {
        return Err(AppError::Invalid(format!(
            "runs directory is not a folder: {}",
            root.display()
        )));
    }

    let mut directories = fs::read_dir(root)
        .map_err(|e| {
            AppError::Invalid(format!(
                "cannot read runs directory {}: {e}",
                root.display()
            ))
        })?
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .collect::<Vec<_>>();
    directories.sort_by_key(|entry| entry.file_name());

    let truncated = directories.len() > MAX_RUN_DIRECTORIES;
    let start = directories.len().saturating_sub(MAX_RUN_DIRECTORIES);
    let mut runs = directories[start..]
        .iter()
        .map(|entry| inspect_run(&entry.path()))
        .collect::<Vec<_>>();
    for run in &mut runs {
        run.scan_limited = truncated;
    }
    runs.sort_by(|a, b| {
        b.started_at_ms
            .or(b.modified_at_ms)
            .cmp(&a.started_at_ms.or(a.modified_at_ms))
            .then_with(|| a.output_dir.cmp(&b.output_dir))
    });

    if truncated {
        log::warn!(
            "run-library scan of {} was bounded to the newest {} lexicographic directories",
            root.display(),
            MAX_RUN_DIRECTORIES
        );
    }
    Ok(runs)
}

fn inspect_run(path: &Path) -> RunLibraryEntry {
    let name = path
        .file_name()
        .and_then(|part| part.to_str())
        .unwrap_or("unknown");
    let (started_at_ms, case_name) = run_identity_from_name(name);
    let normalized = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let output_dir = normalized.display().to_string();
    let modified_at_ms = fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis());
    let mut remote = RemoteLibraryMetadata::default();
    let provenance = inspect_provenance(path, &mut remote);

    let mut step_dirs = match fs::read_dir(path) {
        Ok(entries) => entries
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
            .filter_map(|entry| parse_step_dir(&entry.path()).map(|step| (step, entry.path())))
            .collect::<Vec<_>>(),
        Err(error) => {
            return invalid_entry(
                output_dir,
                case_name,
                started_at_ms,
                modified_at_ms,
                format!("Nimbus Studio cannot read this run directory: {error}"),
                provenance,
                remote,
            );
        }
    };
    step_dirs.sort_by_key(|(step, _)| *step);

    if step_dirs.is_empty() {
        return invalid_entry(
            output_dir,
            case_name,
            started_at_ms,
            modified_at_ms,
            "No step directories were found. Keep the folder if the simulation may still be publishing outputs."
                .into(),
            provenance,
            remote,
        );
    }

    let was_bounded = step_dirs.len() > MAX_ENTRIES_PER_RUN;
    if was_bounded {
        step_dirs = step_dirs.split_off(step_dirs.len() - MAX_ENTRIES_PER_RUN);
    }

    // Same gate list_step_dirs applies. Without it a run interrupted between
    // publishing a fresh generation and clearing its old step directories
    // reports the *previous* run's frame count and simulation time as Ready,
    // while Plot Studio correctly shows zero frames.
    let generation_gate = run_generation_gate(path);
    if generation_gate == RunGenerationGate::Invalid {
        return invalid_entry(
            output_dir,
            case_name,
            started_at_ms,
            modified_at_ms,
            "This run's output marker is unreadable, so its frames cannot be trusted.".into(),
            provenance,
            remote,
        );
    }
    let legacy_run = generation_gate == RunGenerationGate::Legacy;

    let mut complete = Vec::new();
    let mut unreadable = 0usize;
    let mut incomplete = 0usize;
    let mut superseded = 0usize;
    let mut has_netcdf = false;
    let mut has_npy = false;
    let mut has_zfp = false;
    let mut legacy_without_publication_state = false;
    for (step, directory) in &step_dirs {
        let manifest_path = directory.join("manifest.json");
        let value = match fs::read_to_string(&manifest_path)
            .ok()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        {
            Some(value) => value,
            None => {
                unreadable += 1;
                continue;
            }
        };
        let publication_state = value
            .get("publication_state")
            .and_then(|value| value.as_str());
        if publication_state != Some("complete") {
            if legacy_run && publication_state.is_none() {
                legacy_without_publication_state = true;
            } else {
                incomplete += 1;
                continue;
            }
        }
        if let RunGenerationGate::Require(expected) = &generation_gate {
            let generation = value
                .get("publication_generation")
                .and_then(|value| value.as_str());
            if generation != Some(expected.as_str()) {
                superseded += 1;
                continue;
            }
        }
        let simulation_time_s = value
            .get("simulation_time_s")
            .and_then(|value| value.as_f64());
        let published_fields = value
            .get("published_fields")
            .and_then(|value| value.as_array())
            .map(|fields| {
                fields
                    .iter()
                    .filter_map(|field| field.as_str())
                    .collect::<HashSet<_>>()
            });
        let field_count = published_fields.as_ref().map_or_else(
            || {
                value
                    .get("fields")
                    .and_then(|value| value.as_array())
                    .map_or(0, Vec::len)
            },
            HashSet::len,
        );
        if directory.join("history.nc").is_file() {
            has_netcdf = true;
        }
        if let Some(fields) = value.get("fields").and_then(|value| value.as_array()) {
            for field in fields {
                let field_id = field.get("field_id").and_then(|value| value.as_str());
                if published_fields
                    .as_ref()
                    .is_some_and(|published| !field_id.is_some_and(|id| published.contains(id)))
                {
                    continue;
                }
                if let Some(pattern) = field.get("file_pattern").and_then(|value| value.as_str()) {
                    has_npy |= pattern.ends_with(".npy");
                    has_zfp |= pattern.ends_with(".zfp3d");
                }
            }
        }
        complete.push((*step, simulation_time_s, field_count));
    }

    let mut formats = Vec::new();
    if has_netcdf {
        formats.push("netCDF");
    }
    if has_zfp {
        formats.push("ZFP");
    }
    if has_npy {
        formats.push("NPY");
    }
    let output_format = if formats.is_empty() {
        "Nimbus output".to_string()
    } else {
        formats.join(" + ")
    };
    let latest = complete.last().copied();
    let unverified = unreadable + incomplete + superseded;
    let health = if complete.is_empty() {
        LibraryHealth::Invalid
    } else if legacy_without_publication_state || unverified > 0 || was_bounded {
        LibraryHealth::Partial
    } else {
        LibraryHealth::Ready
    };
    let issue = match health {
        LibraryHealth::Ready => None,
        LibraryHealth::Partial if legacy_without_publication_state => Some(
            "Markerless legacy output is available read-only for plotting, but it predates atomic publication and generation evidence."
                .into(),
        ),
        LibraryHealth::Partial if superseded > 0 && complete.is_empty() => Some(format!(
            "{superseded} frame{} belong to a previous run of this directory and were ignored.",
            plural(superseded)
        )),
        LibraryHealth::Partial => Some(format!(
            "{} complete frame{} found; {} frame{} could not be verified. Plot Studio will use only complete publications.",
            complete.len(),
            plural(complete.len()),
            unverified,
            plural(unverified)
        )),
        // Distinguish "nothing readable" from "everything here is stale" — the
        // second is the interrupted-rerun case and is not a corrupt run.
        LibraryHealth::Invalid if superseded > 0 => Some(format!(
            "No frames from the current run were found; {} step director{} belong to a previous run of this directory.",
            superseded,
            if superseded == 1 { "y" } else { "ies" }
        )),
        LibraryHealth::Invalid => Some(format!(
            "No complete publications were found across {} step director{}; {} manifest{} could not be read.",
            step_dirs.len(),
            if step_dirs.len() == 1 { "y" } else { "ies" },
            unreadable,
            plural(unreadable)
        )),
    };

    RunLibraryEntry {
        id: format!("disk-{:016x}", fnv1a64(output_dir.as_bytes())),
        case_name,
        output_dir,
        health,
        issue,
        started_at_ms,
        modified_at_ms,
        frame_count: complete.len(),
        latest_step: latest.map(|value| value.0),
        simulation_time_s: latest.and_then(|value| value.1),
        field_count: latest.map_or(0, |value| value.2),
        output_format,
        scan_limited: false,
        scan_limit: MAX_RUN_DIRECTORIES,
        provenance,
        execution_target: remote.execution_target,
        remote_receive: remote.remote_receive,
    }
}

fn invalid_entry(
    output_dir: String,
    case_name: String,
    started_at_ms: Option<u128>,
    modified_at_ms: Option<u128>,
    issue: String,
    provenance: RunProvenanceSummary,
    remote: RemoteLibraryMetadata,
) -> RunLibraryEntry {
    let issue = if remote.remote_receive.as_ref().is_some_and(|receipt| receipt.retry_available) {
        format!("Remote results can be received again. {issue}")
    } else { issue };
    RunLibraryEntry {
        id: format!("disk-{:016x}", fnv1a64(output_dir.as_bytes())),
        case_name,
        output_dir,
        health: LibraryHealth::Invalid,
        issue: Some(issue),
        started_at_ms,
        modified_at_ms,
        frame_count: 0,
        latest_step: None,
        simulation_time_s: None,
        field_count: 0,
        output_format: "Unknown".into(),
        scan_limited: false,
        scan_limit: MAX_RUN_DIRECTORIES,
        provenance,
        execution_target: remote.execution_target,
        remote_receive: remote.remote_receive,
    }
}

fn inspect_provenance(path: &Path, remote: &mut RemoteLibraryMetadata) -> RunProvenanceSummary {
    if !path.join(PROVENANCE_FILE_NAME).exists() {
        return RunProvenanceSummary {
            health: ProvenanceHealth::Missing,
            issue: Some(
                "This run predates Studio provenance or was not launched by Studio.".into(),
            ),
            runtime_health: if path.join(RUNTIME_PROVENANCE_FILE_NAME).exists() {
                ProvenanceHealth::Invalid
            } else {
                ProvenanceHealth::Missing
            },
            runtime_issue: Some(if path.join(RUNTIME_PROVENANCE_FILE_NAME).exists() {
                "Runtime provenance cannot be trusted without its launch record.".into()
            } else {
                "No Studio runtime completion record was found.".into()
            }),
            run_id: None,
            created_unix_ms: None,
            executable_path: None,
            executable_digest: None,
            configuration_digest: None,
            catalog_fingerprint: None,
            requested_backend: None,
            requested_fallback_allowed: None,
            requested_output_format: None,
            requested_execution_mode: None,
            requested_validation_mode: None,
            requested_schemes: BTreeMap::new(),
            sounding_requested_path: None,
            sounding_canonical_path: None,
            sounding_digest: None,
            sounding_issue: None,
            runtime_actuals_available: false,
            actual_backend: None,
            fallback_active: None,
            actual_schemes: BTreeMap::new(),
            final_status: None,
            exit_code: None,
            process_started_unix_ms: None,
            process_ended_unix_ms: None,
            complete_frame_count: None,
            actual_output_format: None,
            verified_checkpoint_count: None,
            latest_checkpoint_path: None,
        };
    }
    match read_validated_run_provenance(path) {
        Ok(provenance) => {
            remote.execution_target = provenance.remote_execution.as_ref().map(|value| value.target.clone());
            remote.remote_receive = remote_receive_summary_with_provenance(path, &provenance);
            let sounding = provenance
                .source_artifacts
                .get("initialConditions.soundingFile")
                .cloned();
            let runtime_result = if path.join(RUNTIME_PROVENANCE_FILE_NAME).exists() {
                read_validated_runtime_completion(path, &provenance)
            } else {
                Err(AppError::NotFound(RUNTIME_PROVENANCE_FILE_NAME.into()))
            };
            let runtime_exists = path.join(RUNTIME_PROVENANCE_FILE_NAME).exists();
            let (runtime_health, runtime_issue) = match &runtime_result {
                Ok(_) => (ProvenanceHealth::Verified, None),
                Err(error) if runtime_exists => {
                    (ProvenanceHealth::Invalid, Some(error.to_string()))
                }
                Err(_) => (
                    ProvenanceHealth::Missing,
                    Some(
                        "The run is still active or predates Studio runtime completion provenance."
                            .into(),
                    ),
                ),
            };
            let runtime = runtime_result.ok();
            RunProvenanceSummary {
                health: ProvenanceHealth::Verified,
                issue: None,
                runtime_health,
                runtime_issue,
                run_id: Some(provenance.run_id),
                created_unix_ms: Some(provenance.created_unix_ms),
                executable_path: Some(provenance.executable.canonical_path),
                executable_digest: Some(format!(
                    "{}:{}",
                    provenance.executable.digest.algorithm, provenance.executable.digest.value
                )),
                configuration_digest: Some(format!(
                    "{}:{}",
                    provenance.configuration.digest.algorithm,
                    provenance.configuration.digest.value
                )),
                catalog_fingerprint: provenance
                    .catalog_fingerprint
                    .observed_value()
                    .map(str::to_string),
                requested_backend: provenance
                    .requested
                    .backend
                    .observed_value()
                    .map(str::to_string),
                requested_fallback_allowed: provenance
                    .requested
                    .allow_fallback
                    .observed_value()
                    .copied(),
                requested_output_format: provenance
                    .requested
                    .output_format
                    .observed_value()
                    .map(str::to_string),
                requested_execution_mode: provenance
                    .requested
                    .execution_mode
                    .observed_value()
                    .map(str::to_string),
                requested_validation_mode: provenance
                    .requested
                    .validation_mode
                    .observed_value()
                    .map(str::to_string),
                requested_schemes: provenance.requested.schemes,
                sounding_requested_path: sounding
                    .as_ref()
                    .and_then(|artifact| artifact.requested_path.clone()),
                sounding_canonical_path: sounding
                    .as_ref()
                    .and_then(|artifact| artifact.canonical_path.clone()),
                sounding_digest: sounding
                    .as_ref()
                    .and_then(|artifact| artifact.digest.as_ref())
                    .map(|digest| format!("{}:{}", digest.algorithm, digest.value)),
                sounding_issue: sounding
                    .as_ref()
                    .and_then(|artifact| artifact.status.unavailable_reason().map(str::to_string)),
                runtime_actuals_available: runtime.is_some(),
                actual_backend: runtime
                    .as_ref()
                    .and_then(|value| value.actual.backend.observed_value().map(str::to_string)),
                fallback_active: runtime
                    .as_ref()
                    .and_then(|value| value.actual.fallback_active.observed_value().copied()),
                actual_schemes: runtime
                    .as_ref()
                    .map(|value| value.actual.initialized_schemes.clone())
                    .unwrap_or_default(),
                final_status: runtime.as_ref().and_then(|value| {
                    value
                        .lifecycle
                        .final_status
                        .observed_value()
                        .map(str::to_string)
                }),
                exit_code: runtime
                    .as_ref()
                    .and_then(|value| value.lifecycle.exit_code.observed_value().copied()),
                process_started_unix_ms: runtime.as_ref().and_then(|value| {
                    value
                        .lifecycle
                        .process_started_unix_ms
                        .observed_value()
                        .copied()
                }),
                process_ended_unix_ms: runtime.as_ref().and_then(|value| {
                    value
                        .lifecycle
                        .process_ended_unix_ms
                        .observed_value()
                        .copied()
                }),
                complete_frame_count: runtime
                    .as_ref()
                    .and_then(|value| value.output.complete_frame_count),
                actual_output_format: runtime
                    .as_ref()
                    .and_then(|value| value.output.output_format.clone()),
                verified_checkpoint_count: runtime
                    .as_ref()
                    .and_then(|value| value.checkpoints.verified_checkpoint_count),
                latest_checkpoint_path: runtime
                    .as_ref()
                    .and_then(|value| value.checkpoints.latest_checkpoint_path.clone()),
            }
        }
        Err(error) => RunProvenanceSummary {
            health: ProvenanceHealth::Invalid,
            issue: Some(error.to_string()),
            runtime_health: if path.join(RUNTIME_PROVENANCE_FILE_NAME).exists() {
                ProvenanceHealth::Invalid
            } else {
                ProvenanceHealth::Missing
            },
            runtime_issue: Some(
                "Runtime provenance cannot be validated because launch provenance is invalid."
                    .into(),
            ),
            run_id: None,
            created_unix_ms: None,
            executable_path: None,
            executable_digest: None,
            configuration_digest: None,
            catalog_fingerprint: None,
            requested_backend: None,
            requested_fallback_allowed: None,
            requested_output_format: None,
            requested_execution_mode: None,
            requested_validation_mode: None,
            requested_schemes: BTreeMap::new(),
            sounding_requested_path: None,
            sounding_canonical_path: None,
            sounding_digest: None,
            sounding_issue: None,
            runtime_actuals_available: false,
            actual_backend: None,
            fallback_active: None,
            actual_schemes: BTreeMap::new(),
            final_status: None,
            exit_code: None,
            process_started_unix_ms: None,
            process_ended_unix_ms: None,
            complete_frame_count: None,
            actual_output_format: None,
            verified_checkpoint_count: None,
            latest_checkpoint_path: None,
        },
    }
}

fn parse_step_dir(path: &Path) -> Option<u64> {
    path.file_name()?
        .to_str()?
        .strip_prefix("step_")?
        .parse()
        .ok()
}

fn run_identity_from_name(name: &str) -> (Option<u128>, String) {
    match name.split_once('-') {
        Some((timestamp, case_name))
            if timestamp.parse::<u128>().is_ok() && !case_name.is_empty() =>
        {
            (timestamp.parse().ok(), case_name.to_string())
        }
        _ => (None, name.to_string()),
    }
}

fn expand_tilde(path: &str) -> PathBuf {
    crate::platform::expand_tilde(path)
}

fn plural(count: usize) -> &'static str {
    if count == 1 {
        ""
    } else {
        "s"
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::process::{Command, Stdio};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "nimbus-run-library-{label}-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn write_manifest(step: &Path, state: &str, time: f64) {
        fs::create_dir_all(step).unwrap();
        let mut file = fs::File::create(step.join("manifest.json")).unwrap();
        write!(file, "{{\"publication_state\":\"{state}\",\"simulation_time_s\":{time},\"fields\":[{{\"file_pattern\":\"theta.npy\"}}]}}").unwrap();
    }

    #[test]
    fn blank_runs_root_uses_the_same_native_default_as_launch() {
        let resolved = effective_runs_root(" \n ");
        assert!(resolved.ends_with(Path::new(".nimbus").join("runs")));
        assert_ne!(resolved, PathBuf::new());

        let explicit = std::env::temp_dir().join("nimbus-explicit-runs-root");
        assert_eq!(
            effective_runs_root(&format!("  {}  ", explicit.display())),
            explicit,
        );
    }

    #[test]
    fn remote_archives_remain_discoverable_before_receiving_any_frames() {
        use crate::commands::provenance::{create_observed_run_provenance, ExecutableProvenance,
            ProvenanceDigest, RemoteExecutionProvenance, RequestedRunValues};
        use sha2::Digest;
        let root = fixture_root("remote-receive");
        let run_dir = root.join("1700000000000-remote");
        fs::create_dir(&run_dir).unwrap();
        let config = root.join("source.yaml");
        let yaml = "schema: nimbus.v1\n";
        fs::write(&config, yaml).unwrap();
        let target = RemoteComputeTarget { id: "node".into(), name: "Linux compute".into(),
            host: "compute.example".into(), binary_path: "/opt/nimbus/bin/nimbus_sim".into(),
            work_directory: "/srv/nimbus".into(), asset_mappings: vec![] };
        let identity = serde_json::to_vec(&serde_json::json!({"target":target,"assets":[]})).unwrap();
        let launch = create_observed_run_provenance(&run_dir, "remote-run", 1_700_000_000_000,
            ExecutableProvenance { canonical_path: target.binary_path.clone(),
                digest: ProvenanceDigest { algorithm: "sha256".into(), value: "1".repeat(64) } },
            &config, Some("catalog:fixture".into()), RequestedRunValues::default(),
            Some(RemoteExecutionProvenance { target: target.clone(),
                remote_output_directory: "/srv/nimbus/.nimbus-studio/abc-123/output".into(),
                staged_config_sha256: format!("{:x}", sha2::Sha256::digest(yaml.as_bytes())),
                assets: vec![], execution_target_fingerprint: format!("sha256:{:x}", sha2::Sha256::digest(identity)) }),
        ).unwrap();
        let first = inspect_run(&run_dir);
        assert_eq!(first.frame_count, 0);
        assert_eq!(first.health, LibraryHealth::Invalid);
        assert_eq!(first.execution_target.as_ref(), Some(&target));
        assert!(first.remote_receive.unwrap().retry_available);
        super::super::remote_receive::record_receive_state(&run_dir, &launch, "failed", Some("connection interrupted".into())).unwrap();
        let failed = inspect_run(&run_dir);
        assert_eq!(failed.remote_receive.as_ref().unwrap().state, "failed");
        assert!(failed.issue.as_ref().unwrap().contains("received again"));
        let mut receipt: serde_json::Value = serde_json::from_slice(&fs::read(run_dir.join("remote-receive.json")).unwrap()).unwrap();
        receipt["runId"] = "different-run".into();
        fs::write(run_dir.join("remote-receive.json"), serde_json::to_vec(&receipt).unwrap()).unwrap();
        let unbound = inspect_run(&run_dir);
        assert_eq!(unbound.remote_receive.as_ref().unwrap().state, "invalid");
        assert!(!unbound.remote_receive.unwrap().retry_available);
        let mut invalid_launch = serde_json::to_value(&launch).unwrap();
        invalid_launch["remoteExecution"]["stagedConfigSha256"] = "0".repeat(64).into();
        fs::write(run_dir.join(PROVENANCE_FILE_NAME), serde_json::to_vec(&invalid_launch).unwrap()).unwrap();
        let invalid = inspect_run(&run_dir);
        assert_eq!(invalid.provenance.health, ProvenanceHealth::Invalid);
        assert!(invalid.execution_target.is_none());
        assert!(invalid.remote_receive.is_none());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn discovers_ready_partial_and_invalid_runs_deterministically() {
        let root = fixture_root("health");
        let binary = root.join("nimbus_sim");
        let config = root.join("case.yaml");
        let sounding = root.join("wk.sounding");
        fs::write(&binary, b"simulator").unwrap();
        fs::write(&config, "schema: nimbus.v1\n").unwrap();
        fs::write(&sounding, "1000 300 0.014\n").unwrap();
        let ready = root.join("1700000000000-supercell");
        write_manifest(&ready.join("step_000001"), "complete", 10.0);
        crate::commands::provenance::create_run_provenance(
            &ready,
            "run-provenance-test",
            1_700_000_000_000,
            &binary,
            &config,
            Some("fnv1a64:catalog".into()),
            crate::commands::provenance::RequestedRunValues {
                backend: Some("cpu".into()),
                allow_fallback: Some(false),
                output_format: Some("npy_3d".into()),
                execution_mode: Some("qualified".into()),
                validation_mode: Some("error".into()),
                sounding_path: Some(sounding.display().to_string()),
                schemes: BTreeMap::from([("microphysics".into(), "kessler".into())]),
            },
        )
        .unwrap();
        let mut runtime_evidence = crate::commands::provenance::RuntimeLogEvidence::default();
        crate::commands::provenance::record_runtime_log_line(
            &mut runtime_evidence,
            "[INFO]  [COMPUTE] backend=cpu, requested=cpu, device_index=0, fallback=off, parity=off",
        );
        crate::commands::provenance::record_runtime_log_line(
            &mut runtime_evidence,
            "[INFO]  Initialized microphysics scheme: kessler",
        );
        crate::commands::provenance::create_runtime_completion_provenance(
            &ready,
            crate::commands::provenance::RuntimeCompletionInput {
                run_id: "run-provenance-test".into(),
                process_started_unix_ms: 1_700_000_000_001,
                process_ended_unix_ms: 1_700_000_000_100,
                final_status: "completed".into(),
                exit_code: Some(0),
                log_evidence: runtime_evidence,
            },
        )
        .unwrap();
        let partial = root.join("1700000000001-bubble");
        write_manifest(&partial.join("step_000001"), "complete", 5.0);
        write_manifest(&partial.join("step_000002"), "publishing", 10.0);
        let invalid = root.join("1700000000002-broken");
        fs::create_dir_all(invalid.join("step_000001")).unwrap();
        fs::write(invalid.join("step_000001/manifest.json"), "{broken").unwrap();

        let runs = discover_runs(&root).unwrap();
        assert_eq!(runs.len(), 3);
        assert_eq!(runs[0].case_name, "broken");
        assert_eq!(runs[0].health, LibraryHealth::Invalid);
        assert_eq!(runs[1].health, LibraryHealth::Partial);
        assert_eq!(runs[1].frame_count, 1);
        assert_eq!(runs[2].health, LibraryHealth::Ready);
        assert_eq!(runs[2].latest_step, Some(1));
        assert_eq!(runs[2].simulation_time_s, Some(10.0));
        assert_eq!(runs[2].field_count, 1);
        assert_eq!(runs[2].output_format, "NPY");
        assert_eq!(runs[2].provenance.health, ProvenanceHealth::Verified);
        assert_eq!(
            runs[2].provenance.runtime_health,
            ProvenanceHealth::Verified
        );
        assert_eq!(runs[2].provenance.actual_backend.as_deref(), Some("cpu"));
        assert_eq!(runs[2].provenance.fallback_active, Some(false));
        assert_eq!(runs[2].provenance.requested_fallback_allowed, Some(false));
        assert_eq!(
            runs[2].provenance.sounding_canonical_path.as_deref(),
            Some(fs::canonicalize(&sounding).unwrap().to_str().unwrap())
        );
        assert!(runs[2].provenance.sounding_digest.is_some());
        assert_eq!(runs[2].provenance.requested_backend.as_deref(), Some("cpu"));
        assert_eq!(
            runs[2].provenance.requested_schemes["microphysics"],
            "kessler"
        );
        assert_eq!(runs[1].provenance.health, ProvenanceHealth::Missing);
        assert_eq!(runs[2].id, inspect_run(&ready).id);
        let runtime_path = ready.join(RUNTIME_PROVENANCE_FILE_NAME);
        let mut tampered: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&runtime_path).unwrap()).unwrap();
        tampered["launchProvenance"]["digest"]["value"] = "0000000000000000".into();
        fs::write(&runtime_path, serde_json::to_vec_pretty(&tampered).unwrap()).unwrap();
        assert_eq!(
            inspect_run(&ready).provenance.runtime_health,
            ProvenanceHealth::Invalid
        );
        fs::remove_dir_all(root).unwrap();
    }

    const GENERATION_CURRENT: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const GENERATION_PREVIOUS: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn write_run_marker(run: &Path, generation: &str) {
        fs::create_dir_all(run).unwrap();
        fs::write(
            run.join("nimbus_output.json"),
            format!(
                r#"{{"schema":"nimbus.output","manifest_version":5,
                     "frame_completion":"atomic_step_manifest",
                     "publication_generation":"{generation}",
                     "publication_state":"ready"}}"#
            ),
        )
        .unwrap();
    }

    fn write_generation_manifest(step: &Path, generation: &str, time: f64) {
        fs::create_dir_all(step).unwrap();
        fs::write(
            step.join("manifest.json"),
            format!(
                r#"{{"publication_state":"complete","simulation_time_s":{time},
                     "publication_generation":"{generation}",
                     "fields":[{{"file_pattern":"theta.npy"}}]}}"#
            ),
        )
        .unwrap();
    }

    /// Regression: the library checked only `publication_state`, so a run
    /// interrupted between publishing a new generation and clearing its old
    /// step dirs reported the *previous* run's frames as Ready while Plot
    /// Studio correctly showed none.
    #[test]
    fn stale_generation_frames_are_not_counted_as_current_output() {
        let root = fixture_root("generation-gate");
        let run = root.join("1700000000000-supercell");
        write_run_marker(&run, GENERATION_CURRENT);
        write_generation_manifest(&run.join("step_000001"), GENERATION_CURRENT, 10.0);
        write_generation_manifest(&run.join("step_000002"), GENERATION_PREVIOUS, 999.0);

        let entry = inspect_run(&run);
        assert_eq!(entry.frame_count, 1, "only the current generation counts");
        assert_eq!(
            entry.simulation_time_s,
            Some(10.0),
            "a superseded frame must not set the run's latest time"
        );
        assert_eq!(entry.health, LibraryHealth::Partial);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_fully_superseded_run_reports_no_current_frames() {
        let root = fixture_root("generation-all-stale");
        let run = root.join("1700000000000-supercell");
        write_run_marker(&run, GENERATION_CURRENT);
        write_generation_manifest(&run.join("step_000001"), GENERATION_PREVIOUS, 5.0);

        let entry = inspect_run(&run);
        assert_eq!(entry.frame_count, 0);
        assert_eq!(entry.health, LibraryHealth::Invalid);
        assert!(
            entry
                .issue
                .as_deref()
                .is_some_and(|issue| issue.contains("previous run")),
            "the stale-rerun case must not read as a corrupt run: {:?}",
            entry.issue
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn matching_generation_frames_stay_ready() {
        let root = fixture_root("generation-match");
        let run = root.join("1700000000000-supercell");
        write_run_marker(&run, GENERATION_CURRENT);
        write_generation_manifest(&run.join("step_000001"), GENERATION_CURRENT, 10.0);
        write_generation_manifest(&run.join("step_000002"), GENERATION_CURRENT, 20.0);

        let entry = inspect_run(&run);
        assert_eq!(entry.frame_count, 2);
        assert_eq!(entry.health, LibraryHealth::Ready);
        assert_eq!(entry.simulation_time_s, Some(20.0));
        fs::remove_dir_all(root).unwrap();
    }

    /// Pre-contract runs have no marker at all; they must keep working.
    #[test]
    fn legacy_runs_without_a_marker_are_still_readable() {
        let root = fixture_root("generation-legacy");
        let run = root.join("1700000000000-supercell");
        write_manifest(&run.join("step_000001"), "complete", 10.0);

        let entry = inspect_run(&run);
        assert_eq!(entry.frame_count, 1);
        assert_eq!(entry.health, LibraryHealth::Ready);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn pre_publication_state_legacy_manifests_remain_plot_readable() {
        let root = fixture_root("generation-pre-publication-state");
        let run = root.join("1700000000000-beast-xl");
        let step = run.join("step_000001");
        fs::create_dir_all(&step).unwrap();
        fs::write(step.join("theta.npy"), b"fixture").unwrap();
        fs::write(
            step.join("manifest.json"),
            r#"{"step_index":1,"simulation_time_s":10.0,"fields":[{"field_id":"theta","file_pattern":"theta.npy"}]}"#,
        )
        .unwrap();

        let entry = inspect_run(&run);
        assert_eq!(entry.frame_count, 1);
        assert_eq!(entry.latest_step, Some(1));
        assert_eq!(entry.simulation_time_s, Some(10.0));
        assert_eq!(entry.output_format, "NPY");
        assert_eq!(entry.health, LibraryHealth::Partial);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn an_unreadable_marker_invalidates_the_whole_run() {
        let root = fixture_root("generation-broken-marker");
        let run = root.join("1700000000000-supercell");
        fs::create_dir_all(&run).unwrap();
        fs::write(run.join("nimbus_output.json"), "{not json").unwrap();
        write_generation_manifest(&run.join("step_000001"), GENERATION_CURRENT, 10.0);

        let entry = inspect_run(&run);
        assert_eq!(entry.health, LibraryHealth::Invalid);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn current_zfp_run_counts_only_published_fields_and_reports_codec() {
        let root = fixture_root("zfp-format");
        let run = root.join("1700000000000-zfp");
        let step = run.join("step_000000");
        fs::create_dir_all(&step).unwrap();
        fs::write(
            step.join("manifest.json"),
            r#"{
              "publication_state":"complete",
              "simulation_time_s":0,
              "published_fields":["theta","qv"],
              "fields":[
                {"field_id":"theta","file_pattern":"theta.zfp3d"},
                {"field_id":"qv","file_pattern":"qv.zfp3d"},
                {"field_id":"cape","file_pattern":"cape.npy"}
              ]
            }"#,
        )
        .unwrap();

        let entry = inspect_run(&run);
        assert_eq!(entry.health, LibraryHealth::Ready);
        assert_eq!(entry.field_count, 2);
        assert_eq!(entry.output_format, "ZFP");

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn absent_root_is_an_empty_library_and_file_root_is_rejected() {
        let root = fixture_root("root");
        let absent = root.join("absent");
        assert!(discover_runs(&absent).unwrap().is_empty());
        let file = root.join("not-a-directory");
        fs::write(&file, "x").unwrap();
        assert!(discover_runs(&file).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn symlinked_run_directories_are_not_followed() {
        let root = fixture_root("symlink");
        let outside = fixture_root("outside");
        write_manifest(&outside.join("step_000001"), "complete", 1.0);
        #[cfg(unix)]
        std::os::unix::fs::symlink(&outside, root.join("linked-run")).unwrap();
        assert!(discover_runs(&root).unwrap().is_empty());
        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(outside).unwrap();
    }

    #[test]
    fn root_scan_is_bounded_and_reports_the_limit() {
        let root = fixture_root("bounded");
        for index in 0..(MAX_RUN_DIRECTORIES + 2) {
            fs::create_dir(root.join(format!("{index:013}-case-{index:04}"))).unwrap();
        }
        let runs = discover_runs(&root).unwrap();
        assert_eq!(runs.len(), MAX_RUN_DIRECTORIES);
        assert!(runs.iter().all(|run| run.scan_limited));
        assert!(runs.iter().all(|run| run.scan_limit == MAX_RUN_DIRECTORIES));
        assert_eq!(
            runs[0].case_name,
            format!("case-{:04}", MAX_RUN_DIRECTORIES + 1)
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[ignore = "release qualification: requires bin/nimbus_sim, ncgen, and libnetcdf"]
    fn maintained_cm1_guided_smoke_qualifies_native_studio_evidence() {
        let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("repository root should resolve");
        let fixture = repository.join("tests/workflows/cm1_guided_smoke");
        let binary = crate::platform::simulator_below(&repository)
            .canonicalize()
            .expect("production simulator must be built before qualification");
        let config = fixture
            .join("nimbus-smoke.yaml")
            .canonicalize()
            .expect("maintained Nimbus smoke configuration should exist");
        let sounding = fixture
            .join("nimbus-sounding.txt")
            .canonicalize()
            .expect("reviewed sounding projection should exist");
        let cdl = fixture
            .join("cm1-control.cdl")
            .canonicalize()
            .expect("maintained CM1 control should exist");

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "nimbus-cm1-guided-native-{}-{nonce}",
            std::process::id()
        ));
        let runs_root = root.join("runs");
        fs::create_dir_all(&runs_root).expect("qualification root should be creatable");

        let cm1_path = root.join("cm1out_000001.nc");
        let ncgen = Command::new("ncgen")
            .arg("-4")
            .arg("-o")
            .arg(&cm1_path)
            .arg(&cdl)
            .stdin(Stdio::null())
            .output()
            .expect("ncgen must be installed for the native CM1 qualification");
        assert!(
            ncgen.status.success(),
            "ncgen failed: {}",
            String::from_utf8_lossy(&ncgen.stderr)
        );
        let cm1 = crate::commands::netcdf_dynamic::inspect(
            cm1_path
                .to_str()
                .expect("temporary CM1 path should be UTF-8"),
        )
        .expect("Studio's native netCDF4 adapter should inspect the maintained control");
        assert_eq!(cm1.source_classification, "cm1_likely");
        assert!(cm1.plotting_supported);
        for field in ["winterp", "cref", "th", "prs", "qv", "zvort"] {
            assert!(
                cm1.variables.iter().any(|variable| variable.name == field),
                "CM1 control is missing {field}"
            );
        }
        let (cref_shape, cref) =
            crate::commands::netcdf_dynamic::read_field(cm1_path.to_str().unwrap(), "cref", 0)
                .expect("Studio should read a familiar CM1 reflectivity product");
        assert_eq!(cref_shape, vec![2, 2]);
        // Native reads normalize CM1's (y, x) storage into Studio's
        // x-major field order.
        assert_eq!(cref, vec![5.0, 35.0, 20.0, 55.0]);

        let catalog = Command::new(&binary)
            .arg("--scheme-catalog=json")
            .stdin(Stdio::null())
            .output()
            .expect("exact simulator catalog should be queryable");
        assert!(catalog.status.success());
        let catalog_hash = fnv1a64(&catalog.stdout);
        let catalog_fingerprint = format!("fnv1a64:{catalog_hash:016x}");
        crate::commands::runner::verify_catalog_fingerprint(&binary, &catalog_fingerprint)
            .expect("exact-binary evidence should still match immediately before launch");

        let created_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let run_id = format!("run-{created_unix_ms}-cm1-guided-native");
        let run_dir = runs_root.join(format!("{created_unix_ms}-cm1-sounding-smoke"));
        fs::create_dir_all(&run_dir).unwrap();
        crate::commands::provenance::create_run_provenance(
            &run_dir,
            &run_id,
            created_unix_ms,
            &binary,
            &config,
            Some(catalog_fingerprint),
            crate::commands::provenance::RequestedRunValues {
                backend: Some("cpu".into()),
                allow_fallback: Some(false),
                output_format: Some("npy_3d".into()),
                execution_mode: Some("exploratory".into()),
                validation_mode: Some("warn".into()),
                sounding_path: Some(sounding.display().to_string()),
                schemes: BTreeMap::from([
                    ("boundary_layer".into(), "slab".into()),
                    ("chaos".into(), "none".into()),
                    (
                        "dynamics".into(),
                        "klemp_wilhelmson_split_cartesian_cgrid".into(),
                    ),
                    ("microphysics".into(), "none".into()),
                    ("radiation".into(), "none".into()),
                    ("terrain".into(), "none".into()),
                    ("turbulence".into(), "smagorinsky".into()),
                ]),
            },
        )
        .expect("Studio launch provenance should publish before process creation");

        let process_started_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .max(created_unix_ms);
        let simulation = Command::new(&binary)
            .current_dir(&repository)
            .arg("--headless")
            .arg("--config")
            .arg(&config)
            .arg("--log-profile=normal")
            .arg(format!("--outdir={}", run_dir.display()))
            .stdin(Stdio::null())
            .output()
            .expect("preflighted 30-second simulator process should start");
        let process_ended_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .max(process_started_unix_ms);
        let stdout = String::from_utf8_lossy(&simulation.stdout);
        let stderr = String::from_utf8_lossy(&simulation.stderr);
        assert!(
            simulation.status.success(),
            "maintained 30-second smoke failed:\n{stdout}\n{stderr}"
        );
        assert!(stdout.contains("Duration: 30s"));
        assert!(stdout.contains("Write every: 10s"));
        assert!(stdout.contains("[OUTPUT] 4 snapshots written"));

        let mut runtime_evidence = crate::commands::provenance::RuntimeLogEvidence::default();
        for line in stdout.lines().chain(stderr.lines()) {
            crate::commands::provenance::record_runtime_log_line(&mut runtime_evidence, line);
        }
        crate::commands::provenance::create_runtime_completion_provenance(
            &run_dir,
            crate::commands::provenance::RuntimeCompletionInput {
                run_id,
                process_started_unix_ms,
                process_ended_unix_ms,
                final_status: "completed".into(),
                exit_code: simulation.status.code(),
                log_evidence: runtime_evidence,
            },
        )
        .expect("Studio runtime evidence should qualify only after output drain");

        let runs = discover_runs(&runs_root).expect("Run Library should rescan the native output");
        assert_eq!(runs.len(), 1);
        let run = &runs[0];
        assert_eq!(run.case_name, "cm1-sounding-smoke");
        assert_eq!(run.health, LibraryHealth::Ready);
        assert_eq!(run.frame_count, 4);
        assert_eq!(run.latest_step, Some(3));
        assert_eq!(run.simulation_time_s, Some(30.0));
        assert_eq!(run.output_format, "NPY");
        assert_eq!(run.provenance.health, ProvenanceHealth::Verified);
        assert_eq!(run.provenance.runtime_health, ProvenanceHealth::Verified);
        assert_eq!(run.provenance.final_status.as_deref(), Some("completed"));
        assert_eq!(run.provenance.exit_code, Some(0));
        assert_eq!(run.provenance.actual_backend.as_deref(), Some("cpu"));
        assert_eq!(run.provenance.fallback_active, Some(false));
        assert_eq!(run.provenance.complete_frame_count, Some(4));
        assert_eq!(run.provenance.actual_output_format.as_deref(), Some("NPY"));
        assert_eq!(
            run.provenance.sounding_canonical_path.as_deref(),
            Some(sounding.to_str().unwrap())
        );

        fs::remove_dir_all(&root)
            .expect("successful qualification should clean its temporary data");
    }
}
