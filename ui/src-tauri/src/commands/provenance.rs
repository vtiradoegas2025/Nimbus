use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::Digest;

use crate::commands::checkpoints::inspect_checkpoint_blocking;
use crate::commands::outputs::inspect_qualified_output_summary;
use crate::error::{AppError, AppResult};

pub const PROVENANCE_FILE_NAME: &str = "provenance.json";
pub const RUNTIME_PROVENANCE_FILE_NAME: &str = "runtime-provenance.json";
const PROVENANCE_SCHEMA: &str = "nimbus.studio.run-provenance";
const RUNTIME_PROVENANCE_SCHEMA: &str = "nimbus.studio.runtime-provenance";
const PROVENANCE_BUNDLE_SCHEMA: &str = "nimbus.studio.run-provenance-bundle";
const MAX_PROVENANCE_BYTES: u64 = 8 * 1024 * 1024;
const MAX_RUNTIME_SCHEMES: usize = 32;
const MAX_CHECKPOINT_COMMITS: usize = 128;
const MAX_RETAINED_CHECKPOINT_INSPECTIONS: usize = 4;

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestedRunValues {
    pub backend: Option<String>,
    pub allow_fallback: Option<bool>,
    pub output_format: Option<String>,
    pub execution_mode: Option<String>,
    pub validation_mode: Option<String>,
    pub sounding_path: Option<String>,
    #[serde(default)]
    pub schemes: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ObservationStatus {
    Observed,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StringObservation {
    status: ObservationStatus,
    value: Option<String>,
    reason: Option<String>,
}

impl StringObservation {
    fn requested(value: Option<String>, label: &str) -> Self {
        match value
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
        {
            Some(value) => Self {
                status: ObservationStatus::Observed,
                value: Some(value),
                reason: None,
            },
            None => Self {
                status: ObservationStatus::Unavailable,
                value: None,
                reason: Some(format!(
                    "{label} was not supplied at the Studio launch boundary"
                )),
            },
        }
    }

    pub fn observed_value(&self) -> Option<&str> {
        (self.status == ObservationStatus::Observed)
            .then_some(self.value.as_deref())
            .flatten()
    }

    pub fn unavailable_reason(&self) -> Option<&str> {
        (self.status == ObservationStatus::Unavailable)
            .then_some(self.reason.as_deref())
            .flatten()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ValueObservation<T> {
    status: ObservationStatus,
    value: Option<T>,
    reason: Option<String>,
}

impl<T> ValueObservation<T> {
    fn observed(value: T) -> Self {
        Self {
            status: ObservationStatus::Observed,
            value: Some(value),
            reason: None,
        }
    }

    fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            status: ObservationStatus::Unavailable,
            value: None,
            reason: Some(reason.into()),
        }
    }

    pub fn observed_value(&self) -> Option<&T> {
        (self.status == ObservationStatus::Observed)
            .then_some(self.value.as_ref())
            .flatten()
    }

    #[cfg(test)]
    fn unavailable_reason(&self) -> Option<&str> {
        (self.status == ObservationStatus::Unavailable)
            .then_some(self.reason.as_deref())
            .flatten()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProvenanceDigest {
    pub algorithm: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecutableProvenance {
    pub canonical_path: String,
    pub digest: ProvenanceDigest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfigurationProvenance {
    pub canonical_source_path: String,
    pub exact_yaml: String,
    pub digest: ProvenanceDigest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestedProvenance {
    pub backend: StringObservation,
    #[serde(default = "legacy_unavailable_fallback_policy")]
    pub allow_fallback: ValueObservation<bool>,
    pub output_format: StringObservation,
    pub execution_mode: StringObservation,
    pub validation_mode: StringObservation,
    pub schemes: BTreeMap<String, String>,
    pub schemes_status: StringObservation,
}

fn legacy_unavailable_fallback_policy() -> ValueObservation<bool> {
    ValueObservation::unavailable("fallback policy was not recorded by this older v1 launch record")
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceArtifactProvenance {
    pub status: StringObservation,
    pub requested_path: Option<String>,
    pub canonical_path: Option<String>,
    pub digest: Option<ProvenanceDigest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeObservationProvenance {
    pub actual_backend: StringObservation,
    pub backend_fallback_used: StringObservation,
    pub runtime_status: StringObservation,
    pub process_started_unix_ms: StringObservation,
    pub process_ended_unix_ms: StringObservation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunProvenance {
    pub schema: String,
    pub schema_version: u32,
    pub run_id: String,
    pub created_unix_ms: u128,
    pub executable: ExecutableProvenance,
    pub configuration: ConfigurationProvenance,
    pub catalog_fingerprint: StringObservation,
    pub requested: RequestedProvenance,
    pub source_artifacts: BTreeMap<String, SourceArtifactProvenance>,
    pub runtime_observed: RuntimeObservationProvenance,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remote_execution: Option<RemoteExecutionProvenance>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RemoteExecutionProvenance {
    pub target: super::remote_compute::RemoteComputeTarget,
    pub remote_output_directory: String,
    pub staged_config_sha256: String,
    pub assets: Vec<RemoteSourceAssetProvenance>,
    pub execution_target_fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RemoteSourceAssetProvenance {
    pub remote_path: String,
    pub requested_path: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeLogEvidence {
    pub actual_backend: Option<String>,
    pub fallback_active: Option<bool>,
    pub initialized_schemes: BTreeMap<String, String>,
    pub checkpoint_paths: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LaunchProvenanceLink {
    pub file_name: String,
    pub digest: ProvenanceDigest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeLifecycleProvenance {
    pub process_started_unix_ms: ValueObservation<u128>,
    pub process_ended_unix_ms: ValueObservation<u128>,
    pub final_status: StringObservation,
    pub exit_code: ValueObservation<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeActualProvenance {
    pub backend: StringObservation,
    pub fallback_active: ValueObservation<bool>,
    pub initialized_schemes: BTreeMap<String, String>,
    pub unavailable_initialized_schemes: BTreeMap<String, String>,
    pub initialized_schemes_status: StringObservation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeOutputSummary {
    pub status: StringObservation,
    pub complete_frame_count: Option<u64>,
    pub latest_step: Option<u64>,
    pub output_format: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeCheckpointSummary {
    pub status: StringObservation,
    pub verified_checkpoint_count: Option<u64>,
    pub latest_checkpoint_path: Option<String>,
    pub latest_completed_step: Option<u64>,
    pub latest_checkpoint_digest: Option<ProvenanceDigest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeCompletionProvenance {
    pub schema: String,
    pub schema_version: u32,
    pub run_id: String,
    pub launch_provenance: LaunchProvenanceLink,
    pub lifecycle: RuntimeLifecycleProvenance,
    pub actual: RuntimeActualProvenance,
    pub output: RuntimeOutputSummary,
    pub checkpoints: RuntimeCheckpointSummary,
}

#[derive(Debug, Clone)]
pub struct RuntimeCompletionInput {
    pub run_id: String,
    pub process_started_unix_ms: u128,
    pub process_ended_unix_ms: u128,
    pub final_status: String,
    pub exit_code: Option<i32>,
    pub log_evidence: RuntimeLogEvidence,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProvenanceBundle {
    schema: &'static str,
    schema_version: u32,
    launch: RunProvenance,
    runtime_completion: RuntimeCompletionProvenance,
}

fn fnv1a64_bytes(bytes: &[u8]) -> ProvenanceDigest {
    let hash = bytes.iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    });
    ProvenanceDigest {
        algorithm: "fnv1a64".into(),
        value: format!("{hash:016x}"),
    }
}

fn fnv1a64_file(path: &Path) -> AppResult<ProvenanceDigest> {
    let mut file = fs::File::open(path)?;
    let mut hash = 0xcbf29ce484222325_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        for byte in &buffer[..read] {
            hash = (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
        }
    }
    Ok(ProvenanceDigest {
        algorithm: "fnv1a64".into(),
        value: format!("{hash:016x}"),
    })
}

fn valid_digest(digest: &ProvenanceDigest) -> bool {
    ((digest.algorithm == "fnv1a64" && digest.value.len() == 16)
        || (digest.algorithm == "sha256" && digest.value.len() == 64))
        && digest
            .value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn observed_string(value: impl Into<String>) -> StringObservation {
    StringObservation {
        status: ObservationStatus::Observed,
        value: Some(value.into()),
        reason: None,
    }
}

fn clean_observed_value(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte)))
    .then(|| value.to_string())
}

/// Capture only stable, exact runtime log contracts. Unrecognized prose is
/// ignored rather than interpreted. The bounded evidence is finalized only
/// after both reader threads have drained at process exit.
pub fn record_runtime_log_line(evidence: &mut RuntimeLogEvidence, line: &str) {
    let message = line
        .strip_prefix("[INFO]  ")
        .or_else(|| line.strip_prefix("[WARN]  "))
        .unwrap_or(line)
        .trim();

    if let Some(fields) = message.strip_prefix("[COMPUTE] backend=") {
        let mut pieces = fields.split(',').map(str::trim);
        if let Some(backend) = pieces.next().and_then(clean_observed_value) {
            let fallback = pieces.find_map(|piece| {
                piece
                    .strip_prefix("fallback=")
                    .and_then(|value| match value {
                        "active" => Some(true),
                        "off" => Some(false),
                        _ => None,
                    })
            });
            if let Some(fallback) = fallback {
                evidence.actual_backend = Some(backend);
                evidence.fallback_active = Some(fallback);
            }
        }
        return;
    }

    const SCHEME_PREFIXES: [(&str, &str); 9] = [
        ("Initialized dynamics scheme: ", "dynamics"),
        ("Initialized microphysics scheme: ", "microphysics"),
        ("Initialized radiation scheme: ", "radiation"),
        ("Initialized boundary layer scheme: ", "boundary_layer"),
        ("Initialized turbulence scheme: ", "turbulence"),
        ("Initialized radar scheme: ", "radar"),
        ("Initialized terrain scheme: ", "terrain"),
        ("Initialized chaos scheme: ", "chaos"),
        ("Initialized sounding scheme: ", "sounding"),
    ];
    if evidence.initialized_schemes.len() < MAX_RUNTIME_SCHEMES {
        for (prefix, module) in SCHEME_PREFIXES {
            if let Some(value) = message.strip_prefix(prefix).and_then(clean_observed_value) {
                evidence
                    .initialized_schemes
                    .insert(module.to_string(), value);
                return;
            }
        }
    }

    if let Some(rest) = message.strip_prefix("[CHECKPOINT] committed ") {
        if let Some(path) = rest.split_once(" at step=").map(|value| value.0) {
            let path = path.trim();
            if !path.is_empty() && path.len() <= 4096 {
                if evidence.checkpoint_paths.len() == MAX_CHECKPOINT_COMMITS {
                    evidence.checkpoint_paths.remove(0);
                }
                evidence.checkpoint_paths.push(path.to_string());
            }
        }
    }
}

fn unavailable(reason: &str) -> StringObservation {
    StringObservation {
        status: ObservationStatus::Unavailable,
        value: None,
        reason: Some(reason.into()),
    }
}

fn sounding_artifact(requested_path: Option<String>) -> SourceArtifactProvenance {
    let Some(requested_path) = requested_path
        .map(|path| path.trim().to_string())
        .filter(|path| !path.is_empty())
    else {
        return SourceArtifactProvenance {
            status: unavailable("no file-sounding path was supplied at the Studio launch boundary"),
            requested_path: None,
            canonical_path: None,
            digest: None,
        };
    };
    let path = PathBuf::from(&requested_path);
    match fs::canonicalize(&path).and_then(|canonical| {
        let digest = fnv1a64_file(&canonical).map_err(|error| match error {
            AppError::Io(error) => error,
            other => std::io::Error::other(other.to_string()),
        })?;
        Ok((canonical, digest))
    }) {
        Ok((canonical, digest)) => SourceArtifactProvenance {
            status: StringObservation::requested(
                Some("path and bytes observed before spawn".into()),
                "file sounding",
            ),
            requested_path: Some(requested_path),
            canonical_path: Some(canonical.display().to_string()),
            digest: Some(digest),
        },
        Err(error) => SourceArtifactProvenance {
            status: unavailable(&format!(
                "requested file sounding could not be observed before spawn: {error}"
            )),
            requested_path: Some(requested_path),
            canonical_path: None,
            digest: None,
        },
    }
}

pub fn create_run_provenance(
    output_dir: &Path,
    run_id: &str,
    created_unix_ms: u128,
    binary: &Path,
    config_path: &Path,
    catalog_fingerprint: Option<String>,
    requested: RequestedRunValues,
) -> AppResult<RunProvenance> {
    let canonical_binary = fs::canonicalize(binary).map_err(AppError::Io)?;
    let executable = ExecutableProvenance {
        canonical_path: canonical_binary.display().to_string(), digest: fnv1a64_file(&canonical_binary)?,
    };
    create_observed_run_provenance(output_dir, run_id, created_unix_ms, executable, config_path,
                                  catalog_fingerprint, requested, None)
}

#[allow(clippy::too_many_arguments)]
pub fn create_observed_run_provenance(
    output_dir: &Path,
    run_id: &str,
    created_unix_ms: u128,
    executable: ExecutableProvenance,
    config_path: &Path,
    catalog_fingerprint: Option<String>,
    requested: RequestedRunValues,
    remote_execution: Option<RemoteExecutionProvenance>,
) -> AppResult<RunProvenance> {
    if !valid_digest(&executable.digest) { return Err(AppError::Invalid("invalid observed executable digest".into())); }
    let canonical_config = fs::canonicalize(config_path).map_err(AppError::Io)?;
    let exact_yaml = fs::read_to_string(&canonical_config).map_err(|error| {
        AppError::Invalid(format!(
            "cannot read launch YAML {} as UTF-8 for provenance: {error}",
            canonical_config.display()
        ))
    })?;
    let schemes_status = if requested.schemes.is_empty() {
        StringObservation::requested(None, "scheme selections")
    } else {
        StringObservation::requested(
            Some("observed at exact-catalog preflight".into()),
            "scheme selections",
        )
    };
    let sounding = if let Some(remote) = &remote_execution {
        let requested_path = requested.sounding_path.clone();
        match requested_path.as_deref().and_then(|path| remote.assets.iter().find(|asset| asset.requested_path == path)) {
            Some(asset) => SourceArtifactProvenance { status: observed_string("exact bytes staged and verified on the remote host"),
                requested_path, canonical_path: Some(asset.remote_path.clone()),
                digest: Some(ProvenanceDigest { algorithm: "sha256".into(), value: asset.sha256.clone() }) },
            None => SourceArtifactProvenance { status: unavailable("remote sounding artifact was not among the hash-bound asset mappings"),
                requested_path, canonical_path: None, digest: None },
        }
    } else { sounding_artifact(requested.sounding_path.clone()) };
    let provenance = RunProvenance {
        schema: PROVENANCE_SCHEMA.into(),
        schema_version: 1,
        run_id: run_id.to_string(),
        created_unix_ms,
        executable,
        configuration: ConfigurationProvenance {
            canonical_source_path: canonical_config.display().to_string(),
            digest: fnv1a64_bytes(exact_yaml.as_bytes()),
            exact_yaml,
        },
        catalog_fingerprint: StringObservation::requested(
            catalog_fingerprint,
            "catalog fingerprint",
        ),
        requested: RequestedProvenance {
            backend: StringObservation::requested(requested.backend, "backend"),
            allow_fallback: requested.allow_fallback.map_or_else(
                || {
                    ValueObservation::unavailable(
                        "compute fallback policy was not supplied at the Studio launch boundary",
                    )
                },
                ValueObservation::observed,
            ),
            output_format: StringObservation::requested(requested.output_format, "output format"),
            execution_mode: StringObservation::requested(
                requested.execution_mode,
                "execution mode",
            ),
            validation_mode: StringObservation::requested(
                requested.validation_mode,
                "validation mode",
            ),
            schemes: requested.schemes,
            schemes_status,
        },
        source_artifacts: BTreeMap::from([("initialConditions.soundingFile".into(), sounding)]),
        runtime_observed: RuntimeObservationProvenance {
            actual_backend: unavailable(
                "runtime had not started when immutable provenance was published",
            ),
            backend_fallback_used: unavailable(
                "runtime had not started when immutable provenance was published",
            ),
            runtime_status: unavailable(
                "runtime had not started when immutable provenance was published",
            ),
            process_started_unix_ms: unavailable(
                "process creation occurs only after immutable provenance publication",
            ),
            process_ended_unix_ms: unavailable(
                "process had not started when immutable provenance was published",
            ),
        },
        remote_execution,
    };
    write_provenance_once(output_dir, &provenance)?;
    Ok(provenance)
}

fn write_provenance_once(output_dir: &Path, provenance: &RunProvenance) -> AppResult<()> {
    let target = output_dir.join(PROVENANCE_FILE_NAME);
    if target.exists() {
        return Err(AppError::Invalid(format!(
            "refusing to overwrite existing run provenance: {}",
            target.display()
        )));
    }
    let bytes = serde_json::to_vec_pretty(provenance)
        .map_err(|error| AppError::Invalid(format!("serialize run provenance: {error}")))?;
    let temporary = output_dir.join(format!(".provenance.{}.tmp", provenance.run_id));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(AppError::Io)?;
    let result = (|| -> AppResult<()> {
        file.write_all(&bytes)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        fs::hard_link(&temporary, &target).map_err(|error| {
            AppError::Invalid(format!(
                "cannot atomically publish never-overwritten provenance {}: {error}",
                target.display()
            ))
        })?;
        Ok(())
    })();
    let _ = fs::remove_file(&temporary);
    result
}

fn observe_output_summary(output_dir: &Path) -> RuntimeOutputSummary {
    match inspect_qualified_output_summary(output_dir) {
        Ok(summary) => RuntimeOutputSummary {
            status: observed_string("current publication contract revalidated after process exit"),
            complete_frame_count: Some(summary.complete_frame_count),
            latest_step: summary.latest_step,
            output_format: summary.output_format,
        },
        Err(error) => RuntimeOutputSummary {
            status: unavailable(&format!(
                "qualified output could not be established after exit: {error}"
            )),
            complete_frame_count: None,
            latest_step: None,
            output_format: None,
        },
    }
}

fn observe_checkpoint_summary(
    output_dir: &Path,
    run_id: &str,
    paths: &[String],
) -> RuntimeCheckpointSummary {
    if paths.len() > MAX_CHECKPOINT_COMMITS {
        return RuntimeCheckpointSummary {
            status: unavailable("checkpoint observation exceeded the 128-commit provenance bound"),
            verified_checkpoint_count: None,
            latest_checkpoint_path: None,
            latest_completed_step: None,
            latest_checkpoint_digest: None,
        };
    }
    let canonical_run = match fs::canonicalize(output_dir) {
        Ok(path) => path,
        Err(error) => {
            return RuntimeCheckpointSummary {
                status: unavailable(&format!(
                    "run directory could not be canonicalized: {error}"
                )),
                verified_checkpoint_count: None,
                latest_checkpoint_path: None,
                latest_completed_step: None,
                latest_checkpoint_digest: None,
            }
        }
    };
    let mut retained = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for raw_path in paths.iter().rev() {
        if retained.len() == MAX_RETAINED_CHECKPOINT_INSPECTIONS {
            break;
        }
        if Path::new(raw_path).is_file() && seen.insert(raw_path.as_str()) {
            retained.push(raw_path);
        }
    }
    let mut verified = BTreeMap::<String, (u64, ProvenanceDigest)>::new();
    for raw_path in retained {
        let inspection = match inspect_checkpoint_blocking(raw_path) {
            Ok(inspection) => inspection,
            Err(error) => {
                return RuntimeCheckpointSummary {
                    status: unavailable(&format!(
                        "an observed checkpoint did not pass canonical inspection: {error}"
                    )),
                    verified_checkpoint_count: None,
                    latest_checkpoint_path: None,
                    latest_completed_step: None,
                    latest_checkpoint_digest: None,
                }
            }
        };
        let canonical = PathBuf::from(&inspection.path);
        if canonical.strip_prefix(&canonical_run).is_err() || inspection.run_id != run_id {
            return RuntimeCheckpointSummary {
                status: unavailable(
                    "an observed checkpoint was not owned by this run directory and run identity",
                ),
                verified_checkpoint_count: None,
                latest_checkpoint_path: None,
                latest_completed_step: None,
                latest_checkpoint_digest: None,
            };
        }
        verified.insert(
            inspection.path,
            (
                inspection.completed_steps,
                ProvenanceDigest {
                    algorithm: "fnv1a64".into(),
                    value: inspection.checkpoint_fnv1a64,
                },
            ),
        );
    }
    let latest = verified
        .iter()
        .max_by(|left, right| left.1 .0.cmp(&right.1 .0).then_with(|| left.0.cmp(right.0)));
    RuntimeCheckpointSummary {
        status: observed_string("retained checkpoint integrity sidecars were observed after exit"),
        verified_checkpoint_count: Some(verified.len() as u64),
        latest_checkpoint_path: latest.map(|value| value.0.clone()),
        latest_completed_step: latest.map(|value| value.1 .0),
        latest_checkpoint_digest: latest.map(|value| value.1 .1.clone()),
    }
}

fn write_runtime_completion_once(
    output_dir: &Path,
    completion: &RuntimeCompletionProvenance,
) -> AppResult<()> {
    let target = output_dir.join(RUNTIME_PROVENANCE_FILE_NAME);
    if target.exists() {
        return Err(AppError::Invalid(format!(
            "refusing to overwrite existing runtime provenance: {}",
            target.display()
        )));
    }
    let bytes = serde_json::to_vec_pretty(completion)
        .map_err(|error| AppError::Invalid(format!("serialize runtime provenance: {error}")))?;
    let temporary = output_dir.join(format!(".runtime-provenance.{}.tmp", completion.run_id));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    let result = (|| -> AppResult<()> {
        file.write_all(&bytes)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        fs::hard_link(&temporary, &target).map_err(|error| {
            AppError::Invalid(format!(
                "cannot atomically publish never-overwritten runtime provenance {}: {error}",
                target.display()
            ))
        })?;
        Ok(())
    })();
    let _ = fs::remove_file(&temporary);
    result
}

pub fn create_runtime_completion_provenance(
    output_dir: &Path,
    input: RuntimeCompletionInput,
) -> AppResult<RuntimeCompletionProvenance> {
    let launch = read_validated_run_provenance(output_dir)?;
    if launch.run_id != input.run_id {
        return Err(AppError::Invalid(
            "runtime completion run ID does not match launch provenance".into(),
        ));
    }
    if input.process_ended_unix_ms < input.process_started_unix_ms
        || input.process_started_unix_ms < launch.created_unix_ms
    {
        return Err(AppError::Invalid(
            "runtime lifecycle timestamps are not monotonic".into(),
        ));
    }
    if !matches!(
        input.final_status.as_str(),
        "completed" | "failed" | "aborted"
    ) {
        return Err(AppError::Invalid(
            "runtime completion has an invalid final status".into(),
        ));
    }
    let launch_digest = fnv1a64_file(&output_dir.join(PROVENANCE_FILE_NAME))?;
    let schemes_status = if input.log_evidence.initialized_schemes.is_empty() {
        unavailable("no exact initialized-scheme log contract was observed")
    } else {
        observed_string("exact initialized-scheme log contracts were observed")
    };
    let unavailable_initialized_schemes = launch
        .requested
        .schemes
        .keys()
        .filter(|module| !input.log_evidence.initialized_schemes.contains_key(*module))
        .map(|module| {
            (
                module.clone(),
                "no exact initialized-scheme log contract was observed for this requested module"
                    .into(),
            )
        })
        .collect();
    let completion = RuntimeCompletionProvenance {
        schema: RUNTIME_PROVENANCE_SCHEMA.into(),
        schema_version: 1,
        run_id: input.run_id,
        launch_provenance: LaunchProvenanceLink {
            file_name: PROVENANCE_FILE_NAME.into(),
            digest: launch_digest,
        },
        lifecycle: RuntimeLifecycleProvenance {
            process_started_unix_ms: ValueObservation::observed(input.process_started_unix_ms),
            process_ended_unix_ms: ValueObservation::observed(input.process_ended_unix_ms),
            final_status: observed_string(input.final_status),
            exit_code: input.exit_code.map_or_else(
                || {
                    ValueObservation::unavailable(
                        "process terminated without a portable numeric exit code",
                    )
                },
                ValueObservation::observed,
            ),
        },
        actual: RuntimeActualProvenance {
            backend: input.log_evidence.actual_backend.map_or_else(
                || unavailable("no exact compute-backend completion line was observed"),
                observed_string,
            ),
            fallback_active: input.log_evidence.fallback_active.map_or_else(
                || {
                    ValueObservation::unavailable(
                        "no exact compute fallback completion field was observed",
                    )
                },
                ValueObservation::observed,
            ),
            initialized_schemes: input.log_evidence.initialized_schemes,
            unavailable_initialized_schemes,
            initialized_schemes_status: schemes_status,
        },
        output: observe_output_summary(output_dir),
        checkpoints: observe_checkpoint_summary(
            output_dir,
            &launch.run_id,
            &input.log_evidence.checkpoint_paths,
        ),
    };
    write_runtime_completion_once(output_dir, &completion)?;
    Ok(completion)
}

pub fn read_validated_run_provenance(run_dir: &Path) -> AppResult<RunProvenance> {
    let path = run_dir.join(PROVENANCE_FILE_NAME);
    let metadata = fs::symlink_metadata(&path)?;
    if !metadata.file_type().is_file() || metadata.len() > MAX_PROVENANCE_BYTES {
        return Err(AppError::Invalid(
            "provenance.json must be a regular file no larger than 8 MiB".into(),
        ));
    }
    let raw = fs::read_to_string(&path)?;
    let provenance: RunProvenance = serde_json::from_str(&raw)
        .map_err(|error| AppError::Invalid(format!("invalid provenance.json: {error}")))?;
    if provenance.schema != PROVENANCE_SCHEMA || provenance.schema_version != 1 {
        return Err(AppError::Invalid(
            "unsupported run provenance schema; expected nimbus.studio.run-provenance v1".into(),
        ));
    }
    let requested_observations_valid = valid_string_observation(&provenance.requested.backend)
        && valid_value_observation(&provenance.requested.allow_fallback)
        && valid_string_observation(&provenance.requested.output_format)
        && valid_string_observation(&provenance.requested.execution_mode)
        && valid_string_observation(&provenance.requested.validation_mode)
        && valid_string_observation(&provenance.requested.schemes_status)
        && ((provenance.requested.schemes_status.status == ObservationStatus::Observed)
            != provenance.requested.schemes.is_empty())
        && provenance.requested.schemes.len() <= MAX_RUNTIME_SCHEMES
        && provenance.requested.schemes.iter().all(|(module, scheme)| {
            clean_observed_value(module).as_deref() == Some(module.as_str())
                && clean_observed_value(scheme).as_deref() == Some(scheme.as_str())
        });
    let source_artifacts_valid = provenance.source_artifacts.len() <= 16
        && provenance.source_artifacts.iter().all(|(key, artifact)| {
            !key.trim().is_empty() && key.len() <= 256 && valid_source_artifact(artifact)
        });
    let remote_execution_valid = provenance.remote_execution.as_ref().map_or(true, |remote| {
        let sha = |value: &str| value.len() == 64
            && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
        remote.target.validate().is_ok()
            && remote.assets.len() == remote.target.asset_mappings.len()
            && sha(&remote.staged_config_sha256)
            && remote.staged_config_sha256 == format!("{:x}", sha2::Sha256::digest(provenance.configuration.exact_yaml.as_bytes()))
            && remote.execution_target_fingerprint.starts_with("sha256:")
            && sha(remote.execution_target_fingerprint.trim_start_matches("sha256:"))
            && remote.remote_output_directory.starts_with(&format!("{}/.nimbus-studio/", remote.target.work_directory.trim_end_matches('/')))
            && remote.assets.iter().all(|asset| sha(&asset.sha256)
                && remote.target.asset_mappings.iter().any(|mapping| mapping.remote_path == asset.requested_path)
                && asset.remote_path.starts_with(&format!("{}/", remote.target.work_directory.trim_end_matches('/'))))
    });
    if provenance.run_id.trim().is_empty()
        || provenance.executable.canonical_path.trim().is_empty()
        || provenance
            .configuration
            .canonical_source_path
            .trim()
            .is_empty()
        || !valid_digest(&provenance.executable.digest)
        || !valid_digest(&provenance.configuration.digest)
        || provenance.configuration.digest
            != fnv1a64_bytes(provenance.configuration.exact_yaml.as_bytes())
        || !valid_string_observation(&provenance.catalog_fingerprint)
        || !requested_observations_valid
        || !source_artifacts_valid
        || !remote_execution_valid
    {
        return Err(AppError::Invalid(
            "run provenance identity or embedded YAML digest is invalid".into(),
        ));
    }
    Ok(provenance)
}

fn valid_string_observation(observation: &StringObservation) -> bool {
    match observation.status {
        ObservationStatus::Observed => {
            observation
                .value
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty() && value.len() <= 4096)
                && observation.reason.is_none()
        }
        ObservationStatus::Unavailable => {
            observation.value.is_none()
                && observation
                    .reason
                    .as_deref()
                    .is_some_and(|reason| !reason.trim().is_empty() && reason.len() <= 4096)
        }
    }
}

fn valid_value_observation<T>(observation: &ValueObservation<T>) -> bool {
    match observation.status {
        ObservationStatus::Observed => observation.value.is_some() && observation.reason.is_none(),
        ObservationStatus::Unavailable => {
            observation.value.is_none()
                && observation
                    .reason
                    .as_deref()
                    .is_some_and(|reason| !reason.trim().is_empty() && reason.len() <= 4096)
        }
    }
}

fn valid_source_artifact(artifact: &SourceArtifactProvenance) -> bool {
    if !valid_string_observation(&artifact.status) {
        return false;
    }
    let requested_path_valid = artifact
        .requested_path
        .as_deref()
        .map(|path| !path.trim().is_empty() && path.len() <= 4096)
        .unwrap_or(true);
    match artifact.status.status {
        ObservationStatus::Observed => {
            requested_path_valid
                && artifact.requested_path.is_some()
                && artifact
                    .canonical_path
                    .as_deref()
                    .is_some_and(|path| !path.trim().is_empty() && path.len() <= 4096)
                && artifact.digest.as_ref().is_some_and(valid_digest)
        }
        ObservationStatus::Unavailable => {
            requested_path_valid && artifact.canonical_path.is_none() && artifact.digest.is_none()
        }
    }
}

fn recorded_output_matches_live(run_dir: &Path, recorded: &RuntimeOutputSummary) -> bool {
    if recorded.status.status == ObservationStatus::Unavailable {
        return true;
    }
    let live = observe_output_summary(run_dir);
    live.status.status == ObservationStatus::Observed
        && live.complete_frame_count == recorded.complete_frame_count
        && live.latest_step == recorded.latest_step
        && live.output_format == recorded.output_format
}

fn recorded_checkpoint_matches_live(
    run_dir: &Path,
    run_id: &str,
    recorded: &RuntimeCheckpointSummary,
) -> bool {
    if recorded.status.status == ObservationStatus::Unavailable {
        return true;
    }
    let Some(count) = recorded.verified_checkpoint_count else {
        return false;
    };
    if count > MAX_CHECKPOINT_COMMITS as u64 {
        return false;
    }
    if count == 0 {
        return true;
    }
    let (Some(path), Some(step), Some(digest)) = (
        recorded.latest_checkpoint_path.as_deref(),
        recorded.latest_completed_step,
        recorded.latest_checkpoint_digest.as_ref(),
    ) else {
        return false;
    };
    let Ok(canonical_run) = fs::canonicalize(run_dir) else {
        return false;
    };
    let Ok(inspection) = inspect_checkpoint_blocking(path) else {
        return false;
    };
    Path::new(&inspection.path)
        .strip_prefix(canonical_run)
        .is_ok()
        && inspection.run_id == run_id
        && inspection.completed_steps == step
        && digest.algorithm == "fnv1a64"
        && digest.value == inspection.checkpoint_fnv1a64
}

pub fn read_validated_runtime_completion(
    run_dir: &Path,
    launch: &RunProvenance,
) -> AppResult<RuntimeCompletionProvenance> {
    let path = run_dir.join(RUNTIME_PROVENANCE_FILE_NAME);
    let metadata = fs::symlink_metadata(&path)?;
    if !metadata.file_type().is_file() || metadata.len() > MAX_PROVENANCE_BYTES {
        return Err(AppError::Invalid(
            "runtime-provenance.json must be a regular file no larger than 8 MiB".into(),
        ));
    }
    let raw = fs::read_to_string(&path)?;
    let completion: RuntimeCompletionProvenance = serde_json::from_str(&raw)
        .map_err(|error| AppError::Invalid(format!("invalid runtime-provenance.json: {error}")))?;
    let actual_launch_digest = fnv1a64_file(&run_dir.join(PROVENANCE_FILE_NAME))?;
    let timestamps_valid = completion
        .lifecycle
        .process_started_unix_ms
        .observed_value()
        .zip(completion.lifecycle.process_ended_unix_ms.observed_value())
        .is_some_and(|(started, ended)| *started >= launch.created_unix_ms && ended >= started);
    let final_status_valid = completion
        .lifecycle
        .final_status
        .observed_value()
        .is_some_and(|status| matches!(status, "completed" | "failed" | "aborted"));
    let schemes_valid = completion.actual.initialized_schemes.len() <= MAX_RUNTIME_SCHEMES
        && completion
            .actual
            .initialized_schemes
            .iter()
            .all(|(module, scheme)| {
                clean_observed_value(module).as_deref() == Some(module.as_str())
                    && clean_observed_value(scheme).as_deref() == Some(scheme.as_str())
            });
    let unavailable_schemes_valid = completion
        .actual
        .unavailable_initialized_schemes
        .iter()
        .all(|(module, reason)| {
            !completion.actual.initialized_schemes.contains_key(module)
                && clean_observed_value(module).as_deref() == Some(module.as_str())
                && !reason.trim().is_empty()
        })
        && launch.requested.schemes.keys().all(|module| {
            completion.actual.initialized_schemes.contains_key(module)
                || completion
                    .actual
                    .unavailable_initialized_schemes
                    .contains_key(module)
        });
    let scheme_status_matches = (completion.actual.initialized_schemes_status.status
        == ObservationStatus::Observed)
        != completion.actual.initialized_schemes.is_empty();
    let backend_fallback_pair_matches = completion.actual.backend.observed_value().is_some()
        == completion.actual.fallback_active.observed_value().is_some();
    let output_valid = valid_string_observation(&completion.output.status)
        && match completion.output.status.status {
            ObservationStatus::Observed => completion.output.complete_frame_count.is_some(),
            ObservationStatus::Unavailable => {
                completion.output.complete_frame_count.is_none()
                    && completion.output.latest_step.is_none()
                    && completion.output.output_format.is_none()
            }
        };
    let checkpoint_latest_fields_match = match completion.checkpoints.verified_checkpoint_count {
        Some(0) => {
            completion.checkpoints.latest_checkpoint_path.is_none()
                && completion.checkpoints.latest_completed_step.is_none()
                && completion.checkpoints.latest_checkpoint_digest.is_none()
        }
        Some(_) => {
            completion.checkpoints.latest_checkpoint_path.is_some()
                && completion.checkpoints.latest_completed_step.is_some()
                && completion.checkpoints.latest_checkpoint_digest.is_some()
        }
        None => true,
    };
    let checkpoints_valid = valid_string_observation(&completion.checkpoints.status)
        && match completion.checkpoints.status.status {
            ObservationStatus::Observed => {
                completion.checkpoints.verified_checkpoint_count.is_some()
            }
            ObservationStatus::Unavailable => {
                completion.checkpoints.verified_checkpoint_count.is_none()
                    && completion.checkpoints.latest_checkpoint_path.is_none()
                    && completion.checkpoints.latest_completed_step.is_none()
                    && completion.checkpoints.latest_checkpoint_digest.is_none()
            }
        }
        && completion
            .checkpoints
            .latest_checkpoint_digest
            .as_ref()
            .map(valid_digest)
            .unwrap_or(true)
        && checkpoint_latest_fields_match;
    if completion.schema != RUNTIME_PROVENANCE_SCHEMA
        || completion.schema_version != 1
        || completion.run_id != launch.run_id
        || completion.launch_provenance.file_name != PROVENANCE_FILE_NAME
        || !valid_digest(&completion.launch_provenance.digest)
        || completion.launch_provenance.digest != actual_launch_digest
        || !timestamps_valid
        || !final_status_valid
        || !valid_value_observation(&completion.lifecycle.exit_code)
        || !valid_string_observation(&completion.actual.backend)
        || !valid_value_observation(&completion.actual.fallback_active)
        || !backend_fallback_pair_matches
        || !valid_string_observation(&completion.actual.initialized_schemes_status)
        || !schemes_valid
        || !unavailable_schemes_valid
        || !scheme_status_matches
        || !output_valid
        || !checkpoints_valid
    {
        return Err(AppError::Invalid(
            "runtime provenance identity, launch link, or observed evidence is invalid".into(),
        ));
    }
    let original_inventory_matches = recorded_output_matches_live(run_dir, &completion.output)
        && recorded_checkpoint_matches_live(run_dir, &completion.run_id, &completion.checkpoints);
    if !original_inventory_matches {
        let original_runtime_digest = format!("sha256:{:x}", sha2::Sha256::digest(raw.as_bytes()));
        let recovered = launch.remote_execution.is_some()
            && super::remote_receive::validate_recovered_remote_outputs(
                run_dir, launch, &original_runtime_digest,
            )?;
        if !recovered {
            return Err(AppError::Invalid(
                "runtime provenance observed output changed without a validated remote recovery receipt".into(),
            ));
        }
    }
    Ok(completion)
}

fn validated_bundle_bytes(run_dir: &Path) -> AppResult<Vec<u8>> {
    let launch = read_validated_run_provenance(run_dir)?;
    let runtime_completion = read_validated_runtime_completion(run_dir, &launch)?;
    let mut bytes = serde_json::to_vec_pretty(&ProvenanceBundle {
        schema: PROVENANCE_BUNDLE_SCHEMA,
        schema_version: 1,
        launch,
        runtime_completion,
    })
    .map_err(|error| AppError::Invalid(format!("serialize provenance bundle: {error}")))?;
    bytes.push(b'\n');
    Ok(bytes)
}

#[tauri::command]
pub async fn copy_run_provenance(run_dir: String, destination: String) -> AppResult<String> {
    tauri::async_runtime::spawn_blocking(move || {
        let run_dir = PathBuf::from(run_dir.trim());
        let destination = PathBuf::from(destination.trim());
        if destination
            .extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| !extension.eq_ignore_ascii_case("json"))
            .unwrap_or(true)
        {
            return Err(AppError::Invalid(
                "provenance exports must use a .json destination".into(),
            ));
        }
        let bytes = validated_bundle_bytes(&run_dir)?;
        let source = run_dir.join(PROVENANCE_FILE_NAME);
        let runtime_source = run_dir.join(RUNTIME_PROVENANCE_FILE_NAME);
        let destination_identity = fs::canonicalize(&destination).ok();
        if destination.exists()
            && (destination_identity == fs::canonicalize(&source).ok()
                || destination_identity == fs::canonicalize(&runtime_source).ok())
        {
            return Err(AppError::Invalid(
                "export destination must not replace either immutable source provenance record"
                    .into(),
            ));
        }
        fs::write(&destination, bytes).map_err(|error| {
            AppError::Io(std::io::Error::new(
                error.kind(),
                format!(
                    "write validated provenance bundle to {}: {error}",
                    destination.display()
                ),
            ))
        })?;
        Ok(destination.display().to_string())
    })
    .await
    .map_err(|error| AppError::Task(error.to_string()))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn fixture_root(label: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("nimbus-provenance-{label}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn checkpoint_log_keeps_the_latest_bounded_paths() {
        let mut evidence = RuntimeLogEvidence::default();
        for step in 0..(MAX_CHECKPOINT_COMMITS + 3) {
            record_runtime_log_line(
                &mut evidence,
                &format!("[CHECKPOINT] committed /run/checkpoint_{step}.nimbus at step={step}"),
            );
        }
        assert_eq!(evidence.checkpoint_paths.len(), MAX_CHECKPOINT_COMMITS);
        assert!(evidence.checkpoint_paths[0].ends_with("checkpoint_3.nimbus"));
        assert!(evidence
            .checkpoint_paths
            .last()
            .unwrap()
            .ends_with(&format!("checkpoint_{}.nimbus", MAX_CHECKPOINT_COMMITS + 2)));
    }

    #[test]
    fn creates_validated_never_overwritten_provenance() {
        let root = fixture_root("create");
        let binary = root.join("nimbus_sim");
        let config = root.join("case.yaml");
        fs::write(&binary, b"binary fixture").unwrap();
        fs::write(&config, "schema: nimbus.v1\n").unwrap();
        let output = root.join("run");
        fs::create_dir(&output).unwrap();
        let requested = RequestedRunValues {
            backend: Some("cpu".into()),
            allow_fallback: Some(false),
            output_format: Some("netcdf".into()),
            execution_mode: Some("qualified".into()),
            validation_mode: Some("error".into()),
            sounding_path: None,
            schemes: BTreeMap::from([("microphysics".into(), "kessler".into())]),
        };
        create_run_provenance(
            &output,
            "run-1",
            123,
            &binary,
            &config,
            Some("fnv1a64:catalog".into()),
            requested,
        )
        .unwrap();
        let read = read_validated_run_provenance(&output).unwrap();
        assert_eq!(read.run_id, "run-1");
        assert_eq!(read.configuration.exact_yaml, "schema: nimbus.v1\n");
        assert_eq!(read.requested.schemes["microphysics"], "kessler");
        assert_eq!(read.requested.allow_fallback.observed_value(), Some(&false));

        let launch_path = output.join(PROVENANCE_FILE_NAME);
        let original = fs::read(&launch_path).unwrap();
        let mut incoherent: Value = serde_json::from_slice(&original).unwrap();
        incoherent["sourceArtifacts"]["initialConditions.soundingFile"]["status"] = serde_json::json!({
            "status": "observed", "value": "tampered", "reason": null
        });
        fs::write(
            &launch_path,
            serde_json::to_vec_pretty(&incoherent).unwrap(),
        )
        .unwrap();
        assert!(read_validated_run_provenance(&output).is_err());
        fs::write(&launch_path, &original).unwrap();

        assert!(create_run_provenance(
            &output,
            "run-2",
            124,
            &binary,
            &config,
            None,
            RequestedRunValues {
                schemes: BTreeMap::from([
                    ("dynamics".into(), "cartesian".into()),
                    ("microphysics".into(), "kessler".into()),
                ]),
                ..RequestedRunValues::default()
            },
        )
        .unwrap_err()
        .to_string()
        .contains("refusing to overwrite"));
        let mut legacy: Value = serde_json::from_slice(&fs::read(&launch_path).unwrap()).unwrap();
        legacy["requested"]
            .as_object_mut()
            .unwrap()
            .remove("allowFallback");
        fs::write(&launch_path, serde_json::to_vec_pretty(&legacy).unwrap()).unwrap();
        let legacy = read_validated_run_provenance(&output).unwrap();
        assert!(legacy.requested.allow_fallback.observed_value().is_none());
        assert!(legacy
            .requested
            .allow_fallback
            .unavailable_reason()
            .is_some_and(|reason| reason.contains("older v1")));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_tampered_embedded_yaml_digest() {
        let root = fixture_root("tamper");
        let path = root.join(PROVENANCE_FILE_NAME);
        fs::write(
            &path,
            r#"{"schema":"nimbus.studio.run-provenance","schemaVersion":1,"runId":"r","createdUnixMs":1,"executable":{"canonicalPath":"/bin/nimbus","digest":{"algorithm":"fnv1a64","value":"0"}},"configuration":{"canonicalSourcePath":"/case.yaml","exactYaml":"changed","digest":{"algorithm":"fnv1a64","value":"0"}},"catalogFingerprint":{"status":"unavailable","value":null,"reason":"missing"},"requested":{"backend":{"status":"unavailable","value":null,"reason":"missing"},"outputFormat":{"status":"unavailable","value":null,"reason":"missing"},"executionMode":{"status":"unavailable","value":null,"reason":"missing"},"validationMode":{"status":"unavailable","value":null,"reason":"missing"},"schemes":{},"schemesStatus":{"status":"unavailable","value":null,"reason":"missing"}},"sourceArtifacts":{},"runtimeObserved":{"actualBackend":{"status":"unavailable","value":null,"reason":"missing"},"backendFallbackUsed":{"status":"unavailable","value":null,"reason":"missing"},"runtimeStatus":{"status":"unavailable","value":null,"reason":"missing"},"processStartedUnixMs":{"status":"unavailable","value":null,"reason":"missing"},"processEndedUnixMs":{"status":"unavailable","value":null,"reason":"missing"}}}"#,
        )
        .unwrap();
        assert!(read_validated_run_provenance(&root).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn completion_is_linked_create_once_and_records_only_exact_runtime_evidence() {
        let root = fixture_root("runtime");
        let binary = root.join("nimbus_sim");
        let config = root.join("case.yaml");
        fs::write(&binary, b"binary fixture").unwrap();
        fs::write(&config, "schema: nimbus.v1\n").unwrap();
        create_run_provenance(
            &root,
            "run-runtime",
            100,
            &binary,
            &config,
            None,
            RequestedRunValues {
                schemes: BTreeMap::from([
                    ("dynamics".into(), "cartesian".into()),
                    ("microphysics".into(), "kessler".into()),
                ]),
                ..RequestedRunValues::default()
            },
        )
        .unwrap();
        let mut evidence = RuntimeLogEvidence::default();
        record_runtime_log_line(
            &mut evidence,
            "[INFO]  [COMPUTE] backend=cpu, requested=vulkan, device_index=0, fallback=active, parity=off",
        );
        record_runtime_log_line(
            &mut evidence,
            "[INFO]  Initialized microphysics scheme: kessler",
        );
        record_runtime_log_line(
            &mut evidence,
            "Initialized microphysics scheme: kessler (guessed)",
        );
        let false_complete_step = root.join("step_000001");
        fs::create_dir(&false_complete_step).unwrap();
        fs::write(
            false_complete_step.join("manifest.json"),
            r#"{"publication_state":"complete","step":1}"#,
        )
        .unwrap();
        let completion = create_runtime_completion_provenance(
            &root,
            RuntimeCompletionInput {
                run_id: "run-runtime".into(),
                process_started_unix_ms: 101,
                process_ended_unix_ms: 200,
                final_status: "completed".into(),
                exit_code: Some(0),
                log_evidence: evidence,
            },
        )
        .unwrap();
        assert_eq!(completion.actual.backend.observed_value(), Some("cpu"));
        assert_eq!(
            completion.actual.fallback_active.observed_value(),
            Some(&true)
        );
        assert_eq!(
            completion.actual.initialized_schemes["microphysics"],
            "kessler"
        );
        assert!(completion
            .actual
            .unavailable_initialized_schemes
            .contains_key("dynamics"));
        assert!(completion.output.status.unavailable_reason().is_some());
        assert_eq!(completion.output.complete_frame_count, None);
        let launch = read_validated_run_provenance(&root).unwrap();
        assert_eq!(
            read_validated_runtime_completion(&root, &launch)
                .unwrap()
                .lifecycle
                .final_status
                .observed_value(),
            Some("completed")
        );
        let bundle: Value =
            serde_json::from_slice(&validated_bundle_bytes(&root).unwrap()).unwrap();
        assert_eq!(
            bundle.get("schema").and_then(Value::as_str),
            Some(PROVENANCE_BUNDLE_SCHEMA)
        );
        assert_eq!(bundle["launch"]["runId"], "run-runtime");
        assert_eq!(bundle["runtimeCompletion"]["runId"], "run-runtime");
        assert!(create_runtime_completion_provenance(
            &root,
            RuntimeCompletionInput {
                run_id: "run-runtime".into(),
                process_started_unix_ms: 101,
                process_ended_unix_ms: 201,
                final_status: "completed".into(),
                exit_code: Some(0),
                log_evidence: RuntimeLogEvidence::default(),
            },
        )
        .unwrap_err()
        .to_string()
        .contains("refusing to overwrite"));
        let runtime_path = root.join(RUNTIME_PROVENANCE_FILE_NAME);
        let original = fs::read(&runtime_path).unwrap();
        let mut tampered: Value = serde_json::from_slice(&original).unwrap();
        tampered["actual"]["initializedSchemesStatus"] = serde_json::json!({
            "status": "unavailable", "value": null, "reason": "tampered"
        });
        fs::write(&runtime_path, serde_json::to_vec_pretty(&tampered).unwrap()).unwrap();
        assert!(read_validated_runtime_completion(&root, &launch).is_err());

        let mut tampered: Value = serde_json::from_slice(&original).unwrap();
        tampered["actual"]["unavailableInitializedSchemes"] = serde_json::json!({});
        fs::write(&runtime_path, serde_json::to_vec_pretty(&tampered).unwrap()).unwrap();
        assert!(read_validated_runtime_completion(&root, &launch).is_err());

        let mut tampered: Value = serde_json::from_slice(&original).unwrap();
        tampered["actual"]["backend"] = serde_json::json!({
            "status": "unavailable", "value": null, "reason": "tampered"
        });
        fs::write(&runtime_path, serde_json::to_vec_pretty(&tampered).unwrap()).unwrap();
        assert!(read_validated_runtime_completion(&root, &launch).is_err());

        let mut tampered: Value = serde_json::from_slice(&original).unwrap();
        tampered["checkpoints"]["verifiedCheckpointCount"] = serde_json::json!(1);
        fs::write(&runtime_path, serde_json::to_vec_pretty(&tampered).unwrap()).unwrap();
        assert!(read_validated_runtime_completion(&root, &launch).is_err());

        let mut tampered: Value = serde_json::from_slice(&original).unwrap();
        tampered["output"] = serde_json::json!({
            "status": {"status": "observed", "value": "tampered", "reason": null},
            "completeFrameCount": 1,
            "latestStep": 1,
            "outputFormat": "netcdf"
        });
        fs::write(&runtime_path, serde_json::to_vec_pretty(&tampered).unwrap()).unwrap();
        assert!(read_validated_runtime_completion(&root, &launch).is_err());
        fs::write(&runtime_path, original).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn completion_does_not_verify_a_checkpoint_outside_the_run_tree() {
        let root = fixture_root("foreign-checkpoint");
        let run = root.join("run");
        let foreign = root.join("foreign");
        fs::create_dir(&run).unwrap();
        fs::create_dir(&foreign).unwrap();
        let binary = root.join("nimbus_sim");
        let config = root.join("case.yaml");
        fs::write(&binary, b"binary fixture").unwrap();
        fs::write(&config, "schema: nimbus.v1\n").unwrap();
        create_run_provenance(
            &run,
            "run-owned",
            100,
            &binary,
            &config,
            None,
            RequestedRunValues::default(),
        )
        .unwrap();

        let checkpoint = foreign.join("checkpoint_step_000000007.nimbus");
        let mut checkpoint_bytes = Vec::from(*b"NIMBSIM1");
        checkpoint_bytes.extend_from_slice(&1_u32.to_le_bytes());
        checkpoint_bytes.extend_from_slice(&0_u64.to_le_bytes());
        checkpoint_bytes.extend_from_slice(&1_469_598_103_934_665_603_u64.to_le_bytes());
        fs::write(&checkpoint, &checkpoint_bytes).unwrap();
        let digest = fnv1a64_file(&checkpoint).unwrap();
        fs::write(
            format!("{}.json", checkpoint.display()),
            serde_json::json!({
                "schema": "nimbus.checkpoint.lifecycle",
                "schema_version": 1,
                "checkpoint_file": "checkpoint_step_000000007.nimbus",
                "checkpoint_bytes": checkpoint_bytes.len(),
                "checkpoint_fnv1a64": digest.value,
                "run_id": "run-owned",
                "completed_steps": 7,
                "simulation_time_s": 3.5,
                "boundary": "completed_timestep",
                "portable_across_compute_memory_topology": true
            })
            .to_string(),
        )
        .unwrap();

        let completion = create_runtime_completion_provenance(
            &run,
            RuntimeCompletionInput {
                run_id: "run-owned".into(),
                process_started_unix_ms: 101,
                process_ended_unix_ms: 102,
                final_status: "completed".into(),
                exit_code: Some(0),
                log_evidence: RuntimeLogEvidence {
                    checkpoint_paths: vec![checkpoint.display().to_string()],
                    ..RuntimeLogEvidence::default()
                },
            },
        )
        .unwrap();
        assert!(completion.checkpoints.status.unavailable_reason().is_some());
        assert_eq!(completion.checkpoints.verified_checkpoint_count, None);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn completion_link_fails_closed_when_launch_bytes_change() {
        let root = fixture_root("runtime-link");
        let binary = root.join("nimbus_sim");
        let config = root.join("case.yaml");
        fs::write(&binary, b"binary fixture").unwrap();
        fs::write(&config, "schema: nimbus.v1\n").unwrap();
        create_run_provenance(
            &root,
            "run-link",
            100,
            &binary,
            &config,
            None,
            RequestedRunValues::default(),
        )
        .unwrap();
        create_runtime_completion_provenance(
            &root,
            RuntimeCompletionInput {
                run_id: "run-link".into(),
                process_started_unix_ms: 100,
                process_ended_unix_ms: 101,
                final_status: "failed".into(),
                exit_code: Some(2),
                log_evidence: RuntimeLogEvidence::default(),
            },
        )
        .unwrap();
        let launch_path = root.join(PROVENANCE_FILE_NAME);
        let mut launch_bytes = fs::read(&launch_path).unwrap();
        launch_bytes.push(b'\n');
        fs::write(&launch_path, launch_bytes).unwrap();
        let launch = read_validated_run_provenance(&root).unwrap();
        assert!(read_validated_runtime_completion(&root, &launch).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
