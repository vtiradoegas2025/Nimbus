use std::collections::HashSet;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

use serde::Serialize;
use serde_json::Value;
use tauri::{AppHandle, Emitter, Manager, State};

use super::scheme_catalog_contract::{validate_readiness_contract, validate_scheme_readiness};
use crate::commands::provenance::{
    create_run_provenance, create_runtime_completion_provenance, record_runtime_log_line,
    RequestedRunValues, RuntimeCompletionInput, RuntimeLogEvidence,
};
use crate::error::{AppError, AppResult};
use crate::state::{AppState, RunProgress, RunRecord, RunStatus, SharedRun};

static CONFIG_VALIDATION_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/*
 * Runner — manages `bin/nimbus_sim` subprocesses + their lifecycle.
 *
 * For each spawn we kick off three threads:
 *   - stdout reader: emits `run:line` per line, parses progress signals
 *   - stderr reader: emits `run:line` (stream="stderr")
 *   - waiter: blocks on Child::wait(), emits `run:exit` when it returns
 *
 * State is shared via Arc<Mutex<RunRecord>> in AppState.runs. abort_sim
 * requests graceful termination (SIGINT, like a CLI Ctrl-C) and escalates to
 * SIGKILL after a grace period if the child ignores it; the run is recorded
 * as Aborted only once the waiter observes the child actually exit.
 *
 * Tauri events emitted:
 *   run:line     { runId, stream: "stdout"|"stderr", line: String, ts: u128 }
 *   run:progress { runId, progress: RunProgress }
 *   run:exit     { runId, status: RunStatus, exitCode: Option<i32> }
 */

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunInfo {
    pub id: String,
    pub case_name: String,
    /// Dynamics scheme captured at spawn from the config. Empty when unknown.
    pub scheme: String,
    /// Configured run length in seconds (v1 `time.run_time`). 0 when unknown.
    pub duration_s: f64,
    pub status: RunStatus,
    pub pid: u32,
    pub started_at_ms: u128,
    /// None while running; set when the waiter sees the child exit. Lets
    /// the UI freeze the wall clock on completed/failed/aborted runs.
    pub ended_at_ms: Option<u128>,
    pub progress: Option<RunProgress>,
    /// The simulator's output dir for this run. UI uses it to auto-load
    /// plots when the user picks the run from the queue panel.
    pub output_dir: String,
    pub execution_target: Option<super::remote_compute::RemoteComputeTarget>,
    pub remote_termination_unconfirmed: bool,
    pub remote_transfer_in_progress: bool,
    pub remote_receive: Option<super::remote_receive::RemoteReceiveSummary>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct LineEvent {
    run_id: String,
    stream: &'static str,
    line: String,
    ts: u128,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProgressEvent {
    run_id: String,
    progress: RunProgress,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ExitEvent {
    run_id: String,
    status: RunStatus,
    exit_code: Option<i32>,
}

// Tauri maps these launch controls directly from the invoke payload.
#[allow(clippy::too_many_arguments)]
#[tauri::command]
pub async fn spawn_sim(app: AppHandle, config_path: String, simulator_path: Option<String>,
    expected_catalog_fingerprint: Option<String>, expected_binary_fingerprint: Option<String>,
    expected_config_fingerprint: Option<String>, expected_execution_target_fingerprint: Option<String>,
    runs_root: Option<String>, output_dir: Option<String>, scheme: Option<String>, duration_s: Option<f64>,
    requested_values: Option<RequestedRunValues>, execution_target: Option<super::remote_compute::RemoteComputeTarget>) -> AppResult<String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        spawn_sim_blocking(app.clone(), state, config_path, simulator_path, expected_catalog_fingerprint, expected_binary_fingerprint,
        expected_config_fingerprint, expected_execution_target_fingerprint, runs_root, output_dir, scheme, duration_s,
        requested_values, execution_target)
    }).await.map_err(|error| AppError::Task(error.to_string()))?
}

fn spawn_sim_blocking(
    app: AppHandle,
    state: State<'_, AppState>,
    config_path: String,
    simulator_path: Option<String>,
    expected_catalog_fingerprint: Option<String>,
    expected_binary_fingerprint: Option<String>,
    expected_config_fingerprint: Option<String>,
    expected_execution_target_fingerprint: Option<String>,
    runs_root: Option<String>,
    output_dir: Option<String>,
    scheme: Option<String>,
    duration_s: Option<f64>,
    requested_values: Option<RequestedRunValues>,
    execution_target: Option<super::remote_compute::RemoteComputeTarget>,
) -> AppResult<String> {
    let binary = if execution_target.is_none() { Some(resolve_binary(&state, simulator_path)?) } else { None };
    let cfg = PathBuf::from(&config_path);
    if !cfg.exists() {
        return Err(AppError::NotFound(config_path));
    }
    let remote_stage = if let Some(target) = execution_target {
        let contents = std::fs::read_to_string(&cfg)?;
        let expected_config = expected_config_fingerprint.as_deref().filter(|value| !value.is_empty())
            .ok_or_else(|| AppError::Invalid("remote launch requires exact-source preflight identity".into()))?;
        if super::remote_compute::fingerprint(contents.as_bytes()) != expected_config {
            return Err(AppError::Invalid("configuration changed after remote exact validation".into()));
        }
        let stage = super::remote_compute::stage_source(target, &contents)?;
        stage.verify_launch_identity(expected_binary_fingerprint.as_deref(), expected_catalog_fingerprint.as_deref(),
            expected_execution_target_fingerprint.as_deref())?;
        let admission = stage.validate_source(false)?;
        if admission.exit_code != 0 || !admission.stdout.contains("[VALIDATE] PASS") {
            return Err(AppError::Invalid(format!("remote source rejected at launch: {}", admission.stderr.chars().take(2048).collect::<String>())));
        }
        Some(stage)
    } else { None };

    let case_name = cfg
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let created_unix_ms = now_unix_ms();
    let outdir = match output_dir
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        Some(p) => PathBuf::from(expand_tilde(p)),
        None => default_output_dir(runs_root.as_deref(), &case_name, created_unix_ms),
    };
    // mkdir -p so the simulator finds the directory ready. The simulator
    // creates the per-step subdirs on its own; we just guarantee the root.
    std::fs::create_dir_all(&outdir).map_err(|e| {
        AppError::Subprocess(format!("create output dir {}: {e}", outdir.display()))
    })?;

    if let Some(expected) = expected_catalog_fingerprint
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        if let Some(binary) = &binary { verify_catalog_fingerprint(binary, expected)?; }
    }
    if let Some(expected) = expected_binary_fingerprint
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let actual = if let Some(binary) = &binary { crate::platform::sha256_file(binary)? }
            else { remote_stage.as_ref().unwrap().probe.binary_fingerprint.clone() };
        if actual != expected {
            return Err(AppError::Invalid(format!(
                "simulator executable changed after preflight: expected {expected}, found {actual}"
            )));
        }
    }
    if let Some(expected) = expected_config_fingerprint
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let bytes = std::fs::read(&cfg)?;
        let actual = format!("fnv1a64:{:016x}", fnv1a64(&bytes));
        if actual != expected {
            return Err(AppError::Invalid(format!(
                "configuration changed after exact validation: expected {expected}, found {actual}"
            )));
        }
    }

    // Establish the durable identity before process creation. A run that
    // cannot publish its immutable provenance does not spawn. If spawning
    // later fails, the directory intentionally retains this launch-attempt
    // evidence rather than silently erasing what Studio observed.
    let run_id = format!(
        "run-{created_unix_ms}-{:016x}",
        fnv1a64(outdir.to_string_lossy().as_bytes())
    );
    if let Some(stage) = &remote_stage {
        use super::provenance::{create_observed_run_provenance, ExecutableProvenance, ProvenanceDigest,
                               RemoteExecutionProvenance, RemoteSourceAssetProvenance};
        create_observed_run_provenance(&outdir, &run_id, created_unix_ms,
            ExecutableProvenance { canonical_path: stage.probe.binary_path.clone(),
                digest: ProvenanceDigest { algorithm: "sha256".into(),
                    value: stage.probe.binary_fingerprint.trim_start_matches("sha256:").into() } },
            &cfg, expected_catalog_fingerprint.clone(), requested_values.unwrap_or_default(),
            Some(RemoteExecutionProvenance { target: stage.run.target.clone(), remote_output_directory: stage.run.output_path.clone(),
                execution_target_fingerprint: stage.execution_target_fingerprint.clone(),
                staged_config_sha256: stage.source_sha256.clone(), assets: stage.assets.iter().map(|asset| RemoteSourceAssetProvenance {
                    remote_path: asset.remote_path.clone(), requested_path: asset.requested_path.clone(), sha256: asset.sha256.clone() }).collect() }))?;
    } else { create_run_provenance(
        &outdir,
        &run_id,
        created_unix_ms,
        binary.as_ref().unwrap(),
        &cfg,
        expected_catalog_fingerprint.clone(),
        requested_values.unwrap_or_default(),
    )?; }

    // nimbus_sim accepts log-profile = quiet | normal | debug. "normal"
    // matches what the CLI users invoke by default and gives the UI the
    // [VEL-MAX] / [PHYSICS BUDGET] lines the progress parser depends on.
    let mut child = if let Some(stage) = &remote_stage { stage.spawn()? } else {
    let binary = binary.as_ref().unwrap();
    let mut command = Command::new(binary);
    command
        .arg("--config")
        .arg(&cfg)
        .arg("--headless")
        .arg("--log-profile=normal")
        .arg(format!("--outdir={}", outdir.display()))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    configure_process_group(&mut command);
    command
        .spawn()
        .map_err(|e| AppError::Subprocess(format!("spawn {}: {e}", binary.display())))?
    };

    let pid = child.id();
    let process_started_unix_ms = now_unix_ms();
    let stdout = child.stdout.take().ok_or_else(|| {
        AppError::Subprocess("child stdout pipe missing immediately after spawn".into())
    })?;
    let stderr = child.stderr.take().ok_or_else(|| {
        AppError::Subprocess("child stderr pipe missing immediately after spawn".into())
    })?;

    let runtime_evidence = Arc::new(Mutex::new(RuntimeLogEvidence::default()));
    let record = Arc::new(Mutex::new(RunRecord {
        id: run_id.clone(),
        case_name,
        scheme: scheme.unwrap_or_default(),
        duration_s: duration_s.unwrap_or(0.0),
        config_path: cfg,
        pid,
        status: RunStatus::Running,
        abort_requested: false,
        progress: None,
        started_at_ms: process_started_unix_ms,
        ended_at_ms: None,
        output_dir: outdir,
        remote: remote_stage.map(|stage| stage.run),
        remote_termination_unconfirmed: false,
        remote_transfer_in_progress: false,
        transport_exited: false,
        runtime_log_evidence: runtime_evidence.clone(),
    }));

    {
        let mut runs = state
            .runs
            .lock()
            .map_err(|e| AppError::Subprocess(format!("runs lock poisoned: {e}")))?;
        runs.insert(run_id.clone(), record.clone());
    }

    let (drained_tx, reader_drained) = mpsc::channel();
    spawn_reader_thread(
        app.clone(),
        run_id.clone(),
        record.clone(),
        runtime_evidence.clone(),
        stdout,
        true,
        drained_tx.clone(),
    );
    spawn_reader_thread(
        app.clone(),
        run_id.clone(),
        record.clone(),
        runtime_evidence.clone(),
        stderr,
        false,
        drained_tx.clone(),
    );
    // The waiter's drain check keys on every sender being dropped, so ours
    // must go before it starts waiting.
    drop(drained_tx);
    spawn_waiter_thread(
        app,
        run_id.clone(),
        record,
        runtime_evidence,
        child,
        reader_drained,
    );

    Ok(run_id)
}

/* ---------- exact-binary scheme catalog ---------- */

/// Ask the configured simulator binary which schemes it actually contains.
/// Keeping discovery in the executable makes the runtime—not Studio—a single
/// source of truth for canonical ids, aliases, maturity, and capabilities.
#[tauri::command]
pub async fn scheme_catalog(app: AppHandle, simulator_path: Option<String>, execution_target: Option<super::remote_compute::RemoteComputeTarget>) -> AppResult<Value> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        scheme_catalog_blocking(state, simulator_path, execution_target)
    }).await.map_err(|error| AppError::Task(error.to_string()))?
}

fn scheme_catalog_blocking(
    state: State<'_, AppState>,
    simulator_path: Option<String>,
    execution_target: Option<super::remote_compute::RemoteComputeTarget>,
) -> AppResult<Value> {
    if let Some(target) = execution_target {
        return super::remote_compute::query_remote_compute(target).map(|probe| probe.catalog);
    }
    let binary = resolve_binary(&state, simulator_path)?;
    let stdout = query_scheme_catalog_bytes(&binary)?;
    parse_scheme_catalog_json(&stdout)
}

fn query_scheme_catalog_bytes(binary: &std::path::Path) -> AppResult<Vec<u8>> {
    let output = Command::new(binary)
        .arg("--scheme-catalog=json")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| AppError::Subprocess(format!("spawn {}: {e}", binary.display())))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(AppError::Subprocess(format!(
            "{} rejected --scheme-catalog=json (exit {:?}): {}",
            binary.display(),
            process_exit_code(&output.status),
            stderr.trim()
        )));
    }

    Ok(output.stdout)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

pub(crate) fn verify_catalog_fingerprint(
    binary: &std::path::Path,
    expected: &str,
) -> AppResult<()> {
    let catalog_bytes = query_scheme_catalog_bytes(binary)?;
    parse_scheme_catalog_json(&catalog_bytes)?;
    let actual = format!("fnv1a64:{:016x}", fnv1a64(&catalog_bytes));
    if actual != expected {
        return Err(AppError::Subprocess(format!(
            "simulator binary changed after configuration preflight: expected catalog {expected}, found {actual}"
        )));
    }
    Ok(())
}

pub(crate) fn parse_scheme_catalog_json(stdout: &[u8]) -> AppResult<Value> {
    let catalog: Value = serde_json::from_slice(stdout).map_err(|e| {
        AppError::Subprocess(format!("invalid JSON from simulator scheme catalog: {e}"))
    })?;
    let object = catalog.as_object().ok_or_else(|| {
        AppError::Subprocess("simulator scheme catalog must be a JSON object".into())
    })?;
    if object.get("schema").and_then(Value::as_str) != Some("nimbus.scheme-catalog.v9") {
        return Err(AppError::Subprocess(
            "unsupported simulator scheme catalog schema".into(),
        ));
    }
    if object.get("catalogVersion").and_then(Value::as_u64) != Some(9) {
        return Err(AppError::Subprocess(
            "unsupported simulator scheme catalog version".into(),
        ));
    }
    let runtime_state_fields = object
        .get("runtimeStateFields")
        .and_then(Value::as_array)
        .filter(|fields| !fields.is_empty())
        .ok_or_else(|| {
            AppError::Subprocess(
                "simulator scheme catalog contains no runtime state inventory".into(),
            )
        })?;
    let mut unique_runtime_fields = HashSet::new();
    for field in runtime_state_fields {
        let field = field
            .as_str()
            .filter(|field| !field.trim().is_empty())
            .ok_or_else(|| {
                AppError::Subprocess(
                    "simulator scheme catalog contains a malformed runtime state field".into(),
                )
            })?;
        if field != field.trim() || !unique_runtime_fields.insert(field) {
            return Err(AppError::Subprocess(
                "simulator scheme catalog contains a duplicate runtime state field".into(),
            ));
        }
    }
    validate_runtime_output_capabilities(object)?;
    validate_readiness_contract(&catalog)?;
    validate_scheme_readiness(&catalog)?;
    let modules = object
        .get("modules")
        .and_then(Value::as_array)
        .filter(|modules| !modules.is_empty())
        .ok_or_else(|| {
            AppError::Subprocess("simulator scheme catalog contains no modules".into())
        })?;
    for module in modules {
        let studio_selection = module.get("studioSelection").and_then(Value::as_object);
        let valid = module.get("id").and_then(Value::as_str).is_some()
            && module
                .get("selectionPath")
                .and_then(Value::as_str)
                .is_some()
            && module.get("schemes").and_then(Value::as_array).is_some()
            && studio_selection.is_some();
        if !valid {
            return Err(AppError::Subprocess(
                "simulator scheme catalog contains a malformed module".into(),
            ));
        }
        let studio_selection = studio_selection.expect("validated above");
        let mode = studio_selection.get("mode").and_then(Value::as_str);
        let fixed = studio_selection
            .get("fixedSchemeId")
            .and_then(Value::as_str);
        let reason = studio_selection.get("reason").and_then(Value::as_str);
        let policy_valid = match (mode, fixed, reason) {
            (Some("form"), Some(""), Some("")) => true,
            (Some("fixed_v1"), Some(fixed), Some(reason))
                if !fixed.is_empty() && !reason.trim().is_empty() =>
            {
                module["schemes"]
                    .as_array()
                    .expect("validated above")
                    .iter()
                    .any(|scheme| {
                        scheme.get("id").and_then(Value::as_str) == Some(fixed)
                            && scheme.get("constructible").and_then(Value::as_bool) == Some(true)
                            && scheme.get("runtimeSelectable").and_then(Value::as_bool)
                                == Some(true)
                    })
            }
            (Some("product_set_v1"), Some(""), Some(reason))
                if !reason.trim().is_empty() => true,
            (Some("unavailable_v1"), Some(""), Some(reason)) if !reason.trim().is_empty() => true,
            _ => false,
        };
        if !policy_valid {
            return Err(AppError::Subprocess(
                "simulator scheme catalog contains an invalid Studio selection policy".into(),
            ));
        }
        for scheme in module["schemes"].as_array().expect("validated above") {
            let parameters = scheme.get("parameters").and_then(Value::as_array);
            if scheme.get("id").and_then(Value::as_str).is_none()
                || scheme
                    .get("runtimeSelectable")
                    .and_then(Value::as_bool)
                    .is_none()
                || parameters.is_none()
            {
                return Err(AppError::Subprocess(
                    "simulator scheme catalog contains a malformed scheme".into(),
                ));
            }
            for parameter in parameters.expect("validated above") {
                let kind = parameter.get("kind").and_then(Value::as_str);
                let default_matches_kind = match kind {
                    Some("float") => parameter.get("default").and_then(Value::as_f64).is_some(),
                    Some("boolean") => parameter.get("default").and_then(Value::as_bool).is_some(),
                    _ => false,
                };
                if parameter.get("path").and_then(Value::as_str).is_none()
                    || parameter
                        .get("displayName")
                        .and_then(Value::as_str)
                        .is_none()
                    || !default_matches_kind
                {
                    return Err(AppError::Subprocess(
                        "simulator scheme catalog contains a malformed parameter".into(),
                    ));
                }
            }
        }
    }
    Ok(catalog)
}

fn validate_runtime_output_capabilities(object: &serde_json::Map<String, Value>) -> AppResult<()> {
    let formats = object
        .get("runtimeCapabilities")
        .and_then(Value::as_object)
        .and_then(|capabilities| capabilities.get("outputFormats"))
        .and_then(Value::as_array)
        .filter(|formats| !formats.is_empty())
        .ok_or_else(|| {
            AppError::Subprocess(
                "simulator scheme catalog contains no output-format capabilities".into(),
            )
        })?;
    let mut ids = HashSet::new();
    for format in formats {
        let id = format.get("id").and_then(Value::as_str).unwrap_or_default();
        let available = format.get("available").and_then(Value::as_bool);
        let reason = format.get("reason").and_then(Value::as_str);
        let implementation = format.get("implementation").and_then(Value::as_str);
        if id.is_empty()
            || id != id.trim().to_lowercase()
            || !ids.insert(id)
            || available.is_none()
            || reason.is_none()
            || implementation.is_none()
            || (available == Some(false) && reason.is_some_and(|value| value.trim().is_empty()))
            || (available == Some(true)
                && implementation.is_some_and(|value| value.trim().is_empty()))
        {
            return Err(AppError::Subprocess(
                "simulator scheme catalog contains a malformed output-format capability".into(),
            ));
        }
    }
    if !ids.contains("npy_3d") || !ids.contains("netcdf") {
        return Err(AppError::Subprocess(
            "simulator scheme catalog omits a required output-format capability".into(),
        ));
    }
    Ok(())
}

/* ---------- migrate_config (D4) ---------- */

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MigrateConfigResult {
    /// v1-shaped YAML on stdout. Empty when the simulator failed before
    /// writing (check exit_code + stderr in that case).
    pub migrated_yaml: String,
    /// Free-text warnings the migrator emitted (one per `[migrate]` line
    /// in `v0_to_v1.cpp` — synthesized defaults, dropped unknown keys,
    /// ambiguous best-guesses).
    pub stderr: String,
    pub exit_code: Option<i32>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ValidateConfigResult {
    pub valid: bool,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
}

/// Ask the exact selected simulator to load and project the candidate YAML.
/// The temporary file is never published and is removed before returning;
/// `--validate-only` allocates no simulation fields and runs no timestep.
#[tauri::command]
pub async fn validate_config_contents(app: AppHandle, contents: String, config_path: Option<String>, simulator_path: Option<String>, execution_target: Option<super::remote_compute::RemoteComputeTarget>) -> AppResult<ValidateConfigResult> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        validate_config_contents_blocking(state, contents, config_path, simulator_path, execution_target)
    }).await.map_err(|error| AppError::Task(error.to_string()))?
}

fn validate_config_contents_blocking(
    state: State<'_, AppState>,
    contents: String,
    config_path: Option<String>,
    simulator_path: Option<String>,
    execution_target: Option<super::remote_compute::RemoteComputeTarget>,
) -> AppResult<ValidateConfigResult> {
    const MAX_CANDIDATE_BYTES: usize = 8 * 1024 * 1024;
    if contents.len() > MAX_CANDIDATE_BYTES {
        return Err(AppError::Invalid(
            "configuration candidate exceeds the 8 MiB validation limit".into(),
        ));
    }
    if let Some(target) = execution_target {
        let stage = super::remote_compute::stage_source(target, &contents)?;
        let result = stage.validate_source(false)?;
        return Ok(ValidateConfigResult { valid: result.exit_code == 0 && result.stdout.contains("[VALIDATE] PASS"),
            stdout: result.stdout, stderr: result.stderr, exit_code: Some(result.exit_code) });
    }
    let binary = resolve_binary(&state, simulator_path)?;
    let serial = CONFIG_VALIDATION_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temporary = std::env::temp_dir().join(format!(
        "nimbus-config-validation-{}-{}-{serial}.yaml",
        std::process::id(),
        now_unix_ms()
    ));
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    if let Err(error) = file
        .write_all(contents.as_bytes())
        .and_then(|_| file.sync_all())
    {
        let _ = std::fs::remove_file(&temporary);
        return Err(AppError::Io(error));
    }
    drop(file);

    let mut command = Command::new(&binary);
    command
        .arg("--config")
        .arg(&temporary)
        .arg("--validate-only")
        .arg("--headless")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(parent) = config_path
        .as_deref()
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .and_then(|path| std::path::Path::new(path).parent())
        .filter(|parent| parent.is_dir())
    {
        command.current_dir(parent);
    }
    let output = command.output();
    let _ = std::fs::remove_file(&temporary);
    let output = output.map_err(|error| {
        AppError::Subprocess(format!(
            "spawn {} for validation: {error}",
            binary.display()
        ))
    })?;
    Ok(ValidateConfigResult {
        valid: output.status.success(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        exit_code: process_exit_code(&output.status),
    })
}

/// Run `bin/nimbus_sim --migrate-config PATH` and capture its stdout
/// (v1 YAML) + stderr (migrator warnings). This is a synchronous,
/// short-lived invocation — the simulator parses + reserializes, no
/// time stepping, no fields allocated. Used by Studio's D4 migration
/// banner when an uploaded YAML trips D1's `v0_legacy` status.
///
/// The caller is responsible for showing the user a diff / accept-or-
/// cancel UI; this command just gives them the migrated text.
#[tauri::command]
pub async fn migrate_config(app: AppHandle, config_path: String, simulator_path: Option<String>, execution_target: Option<super::remote_compute::RemoteComputeTarget>) -> AppResult<MigrateConfigResult> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        migrate_config_blocking(state, config_path, simulator_path, execution_target)
    }).await.map_err(|error| AppError::Task(error.to_string()))?
}

fn migrate_config_blocking(
    state: State<'_, AppState>,
    config_path: String,
    simulator_path: Option<String>,
    execution_target: Option<super::remote_compute::RemoteComputeTarget>,
) -> AppResult<MigrateConfigResult> {
    if let Some(target) = execution_target {
        let source = std::fs::read_to_string(&config_path)?;
        let stage = super::remote_compute::stage_source(target, &source)?;
        let result = stage.migrate_source()?;
        return Ok(MigrateConfigResult { migrated_yaml: result.stdout, stderr: result.stderr, exit_code: Some(result.exit_code) });
    }
    let binary = resolve_binary(&state, simulator_path)?;
    let cfg = PathBuf::from(&config_path);
    if !cfg.exists() {
        return Err(AppError::NotFound(config_path));
    }

    let output = Command::new(&binary)
        .arg("--migrate-config")
        .arg(&cfg)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| AppError::Subprocess(format!("spawn {}: {e}", binary.display())))?;

    let migrated_yaml = String::from_utf8(output.stdout)
        .map_err(|e| AppError::Subprocess(format!("invalid UTF-8 in migrated YAML stdout: {e}")))?;
    let stderr = String::from_utf8(output.stderr).unwrap_or_else(|_| "<non-UTF-8 stderr>".into());

    Ok(MigrateConfigResult {
        migrated_yaml,
        stderr,
        exit_code: process_exit_code(&output.status),
    })
}

/// Grace period between the SIGINT request and the SIGKILL escalation. Long
/// enough for the simulator to catch the signal, flush output, and exit
/// cleanly; short enough that a stuck child is reaped promptly.
const ABORT_GRACE: Duration = Duration::from_secs(5);

#[tauri::command]
pub async fn abort_sim(app: AppHandle, run_id: String) -> AppResult<()> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        abort_sim_blocking(app.clone(), state, run_id)
    }).await.map_err(|error| AppError::Task(error.to_string()))?
}

fn abort_sim_blocking(app: AppHandle, state: State<'_, AppState>, run_id: String) -> AppResult<()> {
    let record = lookup_run(&state, &run_id)?;

    let (pid, already_requested, remote, unconfirmed) = {
        let mut r = record
            .lock()
            .map_err(|e| AppError::Subprocess(format!("record lock poisoned: {e}")))?;
        // Only a live child can be aborted. A run the waiter has already
        // reaped (completed/failed/aborted) is a no-op rather than an error,
        // so a stale UI click can't fail or signal a recycled PID.
        if !matches!(r.status, RunStatus::Running) {
            return Ok(());
        }
        if r.remote_transfer_in_progress {
            if let Some(remote) = &r.remote { remote.cancel_receive(); }
            return Ok(());
        }
        let already = r.abort_requested;
        r.abort_requested = true;
        (r.pid, already, r.remote.clone(), r.remote_termination_unconfirmed)
    };

    if let Some(remote) = remote {
        remote.cancel()?;
        if unconfirmed {
            let (output_dir, started_at_ms) = {
                let mut run = record.lock().map_err(|error| AppError::Task(error.to_string()))?;
                run.remote_termination_unconfirmed = false;
                run.remote_transfer_in_progress = true;
                (run.output_dir.clone(), run.started_at_ms)
            };
            if let Err(error) = super::remote_receive::receive_initial_results(&remote, &output_dir) {
                let _ = app.emit("run:line", LineEvent { run_id: run_id.clone(), stream: "stderr",
                    line: format!("[REMOTE] termination confirmed; retained output mirror failed: {error}"), ts: now_unix_ms() });
            }
            let ended_at_ms = now_unix_ms();
            {
                let mut run = record.lock().map_err(|error| AppError::Task(error.to_string()))?;
                run.status = RunStatus::Aborted; run.ended_at_ms = Some(ended_at_ms); run.remote_transfer_in_progress = false;
            }
            create_runtime_completion_provenance(&output_dir, RuntimeCompletionInput { run_id: run_id.clone(),
                process_started_unix_ms: started_at_ms, process_ended_unix_ms: ended_at_ms,
                final_status: "aborted".into(), exit_code: None, log_evidence: retained_runtime_evidence(&record) })?;
            let _ = app.emit("run:exit", ExitEvent { run_id, status: RunStatus::Aborted, exit_code: None });
        }
        return Ok(());
    }

    // Graceful first: SIGINT mirrors a CLI Ctrl-C so the simulator's handlers
    // flush and exit. We deliberately do NOT set status = Aborted here; the
    // waiter records that once the child actually exits, so the gallery never
    // shows "Aborted" while the process is still running.
    let graceful_result = request_terminate(pid);

    // Escalate to SIGKILL after the grace period if the child ignored SIGINT.
    // Only arm one timer per run, even if the user clicks Abort repeatedly.
    // Arm it even if the graceful transport itself failed: a launch/handler
    // race must never leave an un-abortable process behind.
    if !already_requested {
        spawn_abort_escalation_thread(record, pid);
    }
    graceful_result
}

/// How long to keep re-issuing SIGKILL, and how often. A single kill is not
/// enough: it can fail transiently (EPERM, or the child sitting in
/// uninterruptible IO), and `abort_sim` arms this thread only once per run, so
/// giving up after one attempt leaves a run that can never be aborted or
/// deleted.
const ABORT_KILL_WINDOW: Duration = Duration::from_secs(30);
const ABORT_KILL_RETRY: Duration = Duration::from_secs(2);

/// After ABORT_GRACE, force-kill the child if it's still running, retrying
/// until it dies or the window closes. Re-reads the shared record each pass:
/// once the waiter flips the status away from Running the child has exited and
/// we must not signal again (the PID may have been recycled by the OS).
fn spawn_abort_escalation_thread(record: SharedRun, pid: u32) {
    thread::spawn(move || {
        thread::sleep(ABORT_GRACE);
        let deadline = Instant::now() + ABORT_KILL_WINDOW;
        loop {
            let still_running = match record.lock() {
                Ok(r) => matches!(r.status, RunStatus::Running) && r.abort_requested,
                Err(_) => false,
            };
            if !still_running || !process_alive(pid) {
                return;
            }
            let _ = force_kill(pid);
            if Instant::now() >= deadline {
                log::error!(
                    "pid {pid} survived SIGKILL for {}s after abort; run stays Running",
                    ABORT_KILL_WINDOW.as_secs()
                );
                return;
            }
            thread::sleep(ABORT_KILL_RETRY);
        }
    });
}

/// Shutdown budget. Shorter than ABORT_GRACE: the user asked to quit, so we
/// can't stall the exit, but a SIGINT still gets a chance to flush first.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(2);
const SHUTDOWN_POLL: Duration = Duration::from_millis(50);

/// Terminate every live child at app exit. `Child` has no kill-on-drop and the
/// handle lives in the waiter thread, so without this the simulator outlives
/// the window that spawned it and keeps writing to the run directory.
///
/// Best-effort by design: a poisoned lock is recovered rather than propagated,
/// because refusing to kill is worse than reading a torn record.
pub(crate) fn terminate_all_runs(state: &AppState) {
    let remotes: Vec<_> = state.runs.lock().unwrap_or_else(|error| error.into_inner()).values()
        .filter_map(|record| { let run = record.lock().unwrap_or_else(|error| error.into_inner());
            matches!(run.status, RunStatus::Running).then(|| run.remote.clone()).flatten() }).collect();
    for remote in remotes {
        remote.begin_shutdown();
        thread::spawn(move || { if let Err(error) = remote.cancel_for_shutdown() {
            log::error!("remote shutdown could not confirm termination: {error}");
        } });
    }
    let pids: Vec<u32> = {
        let runs = state.runs.lock().unwrap_or_else(|e| e.into_inner());
        runs.values()
            .filter_map(|record| {
                let r = record.lock().unwrap_or_else(|e| e.into_inner());
                (matches!(r.status, RunStatus::Running) && r.remote.is_none() && !r.transport_exited).then_some(r.pid)
            })
            .collect()
    };
    if pids.is_empty() {
        return;
    }

    for pid in &pids {
        let _ = request_terminate(*pid);
    }

    let deadline = Instant::now() + SHUTDOWN_GRACE;
    while Instant::now() < deadline {
        if !pids.iter().any(|pid| process_alive(*pid)) {
            return;
        }
        thread::sleep(SHUTDOWN_POLL);
    }

    for pid in pids.iter().filter(|pid| process_alive(**pid)) {
        let _ = force_kill(*pid);
    }
}

#[tauri::command]
pub fn list_runs(state: State<'_, AppState>) -> AppResult<Vec<RunInfo>> {
    let runs = state
        .runs
        .lock()
        .map_err(|e| AppError::Subprocess(format!("runs lock poisoned: {e}")))?;

    let mut info: Vec<RunInfo> = runs
        .values()
        .filter_map(|r| {
            let r = r.lock().ok()?;
            Some(RunInfo {
                id: r.id.clone(),
                case_name: r.case_name.clone(),
                scheme: r.scheme.clone(),
                duration_s: r.duration_s,
                status: r.status,
                pid: r.pid,
                started_at_ms: r.started_at_ms,
                ended_at_ms: r.ended_at_ms,
                progress: r.progress,
                output_dir: r.output_dir.display().to_string(),
                execution_target: r.remote.as_ref().map(|remote| remote.target.clone()),
                remote_termination_unconfirmed: r.remote_termination_unconfirmed,
                remote_transfer_in_progress: r.remote_transfer_in_progress,
                remote_receive: super::remote_receive::remote_receive_summary(&r.output_dir),
            })
        })
        .collect();
    info.sort_by_key(|r| std::cmp::Reverse(r.started_at_ms));
    Ok(info)
}

#[tauri::command]
pub fn poll_run(state: State<'_, AppState>, run_id: String) -> AppResult<RunInfo> {
    let record = lookup_run(&state, &run_id)?;
    let r = record
        .lock()
        .map_err(|e| AppError::Subprocess(format!("record lock poisoned: {e}")))?;
    Ok(RunInfo {
        id: r.id.clone(),
        case_name: r.case_name.clone(),
        scheme: r.scheme.clone(),
        duration_s: r.duration_s,
        status: r.status,
        pid: r.pid,
        started_at_ms: r.started_at_ms,
        ended_at_ms: r.ended_at_ms,
        progress: r.progress,
        output_dir: r.output_dir.display().to_string(),
        execution_target: r.remote.as_ref().map(|remote| remote.target.clone()),
        remote_termination_unconfirmed: r.remote_termination_unconfirmed,
        remote_transfer_in_progress: r.remote_transfer_in_progress,
        remote_receive: super::remote_receive::remote_receive_summary(&r.output_dir),
    })
}

#[tauri::command]
pub fn delete_run(state: State<'_, AppState>, run_id: String) -> AppResult<()> {
    // Defense-in-depth: the UI hides X while running, but if a stale snapshot
    // ever asks us to delete a still-running record we refuse rather than
    // orphan the child + reader threads.
    let mut runs = state
        .runs
        .lock()
        .map_err(|e| AppError::Subprocess(format!("runs lock poisoned: {e}")))?;
    let record = runs
        .get(&run_id)
        .cloned()
        .ok_or_else(|| AppError::UnknownRun(run_id.clone()))?;
    {
        let r = record
            .lock()
            .map_err(|e| AppError::Subprocess(format!("record lock poisoned: {e}")))?;
        if matches!(r.status, RunStatus::Running) {
            return Err(AppError::Invalid(
                "cannot delete a running run; abort it first".into(),
            ));
        }
    }
    runs.remove(&run_id);
    Ok(())
}

/* ---------- internals ---------- */

fn lookup_run(state: &State<'_, AppState>, run_id: &str) -> AppResult<SharedRun> {
    let runs = state
        .runs
        .lock()
        .map_err(|e| AppError::Subprocess(format!("runs lock poisoned: {e}")))?;
    runs.get(run_id)
        .cloned()
        .ok_or_else(|| AppError::UnknownRun(run_id.to_string()))
}

pub(crate) fn resolve_binary(
    state: &State<'_, AppState>,
    override_path: Option<String>,
) -> AppResult<PathBuf> {
    // Empty / whitespace-only override means "Settings field cleared, fall
    // back to auto-resolve" — distinct from a real path that happens to be
    // missing, which should error out so the user sees their typo.
    let trimmed = override_path
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    if let Some(p) = trimmed {
        let pb = PathBuf::from(expand_tilde(p));
        if !pb.exists() {
            return Err(AppError::NotFound(pb.display().to_string()));
        }
        return Ok(pb);
    }
    let cached = state
        .simulator_binary
        .lock()
        .map_err(|e| AppError::Subprocess(format!("sim lock poisoned: {e}")))?
        .clone();
    if let Some(binary) = cached {
        return Ok(binary);
    }

    let discovered = crate::commands::status::locate_simulator().ok_or_else(|| {
        AppError::Invalid("simulator binary not configured; set the path in Settings".into())
    })?;
    if let Ok(mut cache) = state.simulator_binary.lock() {
        *cache = Some(discovered.clone());
    }
    Ok(discovered)
}

/// How long the waiter waits for both log readers to hit EOF after the child
/// exits. Normally instant; it is unbounded only when a surviving grandchild
/// still holds the pipe's write end.
const READER_DRAIN_GRACE: Duration = Duration::from_secs(5);

/// True when every reader dropped its sender (all EOF). The channel is never
/// sent on, so `Disconnected` is the success signal and `Timeout` means at
/// least one reader is still blocked. `grace` is a parameter so the timeout
/// path is testable without a five-second test.
fn wait_for_reader_drain(drained: &mpsc::Receiver<()>, grace: Duration) -> bool {
    matches!(
        drained.recv_timeout(grace),
        Err(mpsc::RecvTimeoutError::Disconnected)
    )
}

fn spawn_reader_thread<R>(
    app: AppHandle,
    run_id: String,
    record: SharedRun,
    runtime_evidence: Arc<Mutex<RuntimeLogEvidence>>,
    stream: R,
    is_stdout: bool,
    // Never sent on. Dropping it when this thread returns is the EOF signal
    // the waiter waits for; a stuck reader simply never drops it.
    drained: mpsc::Sender<()>,
) where
    R: Read + Send + 'static,
{
    let stream_label = if is_stdout { "stdout" } else { "stderr" };
    thread::spawn(move || {
        // Bind it inside so the closure actually captures it. A `move` closure
        // only captures what it names, and letting this drop when the spawning
        // call returns would signal EOF before a single line was read.
        let _drained = drained;
        let mut reader = BufReader::new(stream);
        loop {
            let mut bytes = Vec::new();
            match reader.by_ref().take(64 * 1024).read_until(b'\n', &mut bytes) {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
            let line = String::from_utf8_lossy(&bytes).trim_end_matches(['\r', '\n']).to_string();
            if let Ok(mut evidence) = runtime_evidence.lock() {
                record_runtime_log_line(&mut evidence, &line);
            }
            let _ = app.emit(
                "run:line",
                LineEvent {
                    run_id: run_id.clone(),
                    stream: stream_label,
                    line: line.clone(),
                    ts: now_unix_ms(),
                },
            );

            // Only stdout carries progress signals; stderr is warnings/errors.
            if is_stdout {
                if let Some(progress) = parse_progress(&line) {
                    let merged = merge_progress(&record, progress);
                    let _ = app.emit(
                        "run:progress",
                        ProgressEvent {
                            run_id: run_id.clone(),
                            progress: merged,
                        },
                    );
                }
            }
        }
    });
}

fn spawn_waiter_thread(
    app: AppHandle,
    run_id: String,
    record: SharedRun,
    runtime_evidence: Arc<Mutex<RuntimeLogEvidence>>,
    mut child: std::process::Child,
    reader_drained: mpsc::Receiver<()>,
) {
    thread::spawn(move || {
        // Child::wait closes stdin. The SSH channel owns the remote process
        // until exit, so keep that channel open throughout the wait.
        let connection_owner = child.stdin.take();
        let exit = child.wait();
        if let Ok(mut run) = record.lock() { run.transport_exited = true; }
        drop(connection_owner);
        let process_ended_unix_ms = now_unix_ms();
        let remote = record.lock().ok().and_then(|run| run.remote.clone());
        if let Some(remote) = &remote { remote.revoke_transport_ownership(); }
        let mut mirror_failed = false;
        if let Some(remote) = remote {
            if !exit.as_ref().ok().is_some_and(|status| status.success()) {
                let mut confirmed = false;
                for _ in 0..3 {
                    match remote.cancel() {
                        Ok(()) => { confirmed = true; break; },
                        Err(error) => {
                            let _ = app.emit("run:line", LineEvent { run_id: run_id.clone(), stream: "stderr",
                                line: format!("[REMOTE] connection ended; remote termination remains unconfirmed: {error}"), ts: now_unix_ms() });
                            thread::sleep(Duration::from_secs(5));
                        }
                    }
                }
                if !confirmed {
                    let _ = wait_for_reader_drain(&reader_drained, READER_DRAIN_GRACE);
                    if let Ok(mut run) = record.lock() { run.remote_termination_unconfirmed = true; }
                    return;
                }
            }
            {
                let output_dir = record.lock().ok().map(|run| run.output_dir.clone());
                if let Some(output_dir) = output_dir {
                    if let Ok(mut run) = record.lock() { run.remote_transfer_in_progress = true; }
                    let _ = app.emit("run:line", LineEvent { run_id: run_id.clone(), stream: "stdout",
                        line: "[REMOTE] computation finished; downloading completed history for local plots".into(), ts: now_unix_ms() });
                    if let Err(error) = super::remote_receive::receive_initial_results(&remote, &output_dir) {
                        mirror_failed = true;
                        let _ = app.emit("run:line", LineEvent { run_id: run_id.clone(), stream: "stderr",
                            line: format!("[REMOTE] completed history could not be mirrored: {error}"), ts: now_unix_ms() });
                    }
                }
            }
        }
        // Prefer the full captured stream for the completion record, but bound
        // the wait: EOF only follows child exit if nothing else inherited the
        // pipe. A surviving grandchild would otherwise block here forever,
        // leaving the run stuck in `Running` and undeletable.
        if !wait_for_reader_drain(&reader_drained, READER_DRAIN_GRACE) {
            log::warn!(
                "{run_id}: log readers did not reach EOF within {}s; \
                 completion evidence may be truncated",
                READER_DRAIN_GRACE.as_secs()
            );
        }

        // Decide final status now that the child has actually exited. An
        // abort request (flag set by abort_sim) wins over the exit code — a
        // SIGINT/SIGKILL'd child often exits non-zero, but the user asked for
        // it, so it's Aborted, not Failed.
        let (final_status, exit_code, started_at_ms, ended_at_ms, output_dir) = {
            let mut r = match record.lock() {
                Ok(r) => r,
                Err(_) => return,
            };
            let code = exit.as_ref().ok().and_then(process_exit_code);
            let success = exit.as_ref().ok().map(|s| s.success() && !mirror_failed);
            let resolved = resolve_final_status(r.abort_requested, success);
            r.status = resolved;
            r.remote_transfer_in_progress = false;
            r.remote_termination_unconfirmed = false;
            let ended_at_ms = process_ended_unix_ms;
            r.ended_at_ms = Some(ended_at_ms);
            (
                resolved,
                code,
                r.started_at_ms,
                ended_at_ms,
                r.output_dir.clone(),
            )
        };

        let evidence = runtime_evidence
            .lock()
            .map(|value| value.clone())
            .unwrap_or_default();
        if let Err(error) = create_runtime_completion_provenance(
            &output_dir,
            RuntimeCompletionInput {
                run_id: run_id.clone(),
                process_started_unix_ms: started_at_ms,
                process_ended_unix_ms: ended_at_ms,
                final_status: match final_status {
                    RunStatus::Completed => "completed",
                    RunStatus::Failed => "failed",
                    RunStatus::Aborted => "aborted",
                    RunStatus::Running => "running",
                }
                .into(),
                exit_code,
                log_evidence: evidence,
            },
        ) {
            log::error!("failed to publish runtime provenance for {run_id}: {error}");
        }

        let _ = app.emit(
            "run:exit",
            ExitEvent {
                run_id,
                status: final_status,
                exit_code,
            },
        );

        // AppState is reachable via the handle so the gallery list_runs stays
        // accurate after exit. We don't remove the record — it lingers for
        // the user to inspect afterward.
        let _ = app.state::<AppState>();
    });
}

fn merge_progress(record: &SharedRun, partial: ParsedProgress) -> RunProgress {
    let mut r = match record.lock() {
        Ok(r) => r,
        // If the lock is poisoned we still want to emit something useful.
        Err(_) => {
            return RunProgress {
                sim_time_s: partial.sim_time_s.unwrap_or(0.0),
                step: partial.step,
                vel_max: partial.vel_max,
            };
        }
    };
    let mut current = r.progress.unwrap_or_default();
    if let Some(t) = partial.sim_time_s {
        current.sim_time_s = t;
    }
    if let Some(s) = partial.step {
        current.step = Some(s);
    }
    if let Some(v) = partial.vel_max {
        current.vel_max = Some(v);
    }
    r.progress = Some(current);
    current
}

fn retained_runtime_evidence(record: &SharedRun) -> RuntimeLogEvidence {
    record.lock().ok().and_then(|run| run.runtime_log_evidence.lock().ok().map(|evidence| evidence.clone()))
        .unwrap_or_default()
}

#[derive(Debug, Default, Clone, Copy)]
struct ParsedProgress {
    sim_time_s: Option<f64>,
    step: Option<u64>,
    vel_max: Option<f64>,
}

/// Match either `t=12.34s` (typical) or `t=12.34` (some lines drop the
/// unit). Robust to a couple of formats the simulator uses.
fn parse_progress(line: &str) -> Option<ParsedProgress> {
    let mut parsed = ParsedProgress::default();
    let mut hit = false;

    if let Some(t) = extract_keyed_number(line, "t=") {
        parsed.sim_time_s = Some(t);
        hit = true;
    }
    if let Some(s) = extract_keyed_number(line, "step=") {
        parsed.step = Some(s as u64);
        hit = true;
    } else if let Some(s) = extract_keyed_number(line, "Step ") {
        // "[TIME STEP DEBUG] Step 1234 (t=12.5s)" form.
        parsed.step = Some(s as u64);
        hit = true;
    }
    if let Some(v) =
        extract_keyed_number(line, "max_u=").or_else(|| extract_keyed_number(line, "vmax="))
    {
        parsed.vel_max = Some(v);
        hit = true;
    }

    if hit {
        Some(parsed)
    } else {
        None
    }
}

/// Find `<key><number>` in `line` and parse the number. Returns None if
/// the key isn't present or what follows isn't a number.
fn extract_keyed_number(line: &str, key: &str) -> Option<f64> {
    let i = line.find(key)?;
    let tail = &line[i + key.len()..];
    let mut end = 0;
    let mut saw_digit = false;
    for (idx, c) in tail.char_indices() {
        if c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E' {
            end = idx + c.len_utf8();
            if c.is_ascii_digit() {
                saw_digit = true;
            }
        } else {
            break;
        }
    }
    if !saw_digit {
        return None;
    }
    tail[..end].parse::<f64>().ok()
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// Per-run output directory. `runs_root` overrides the default native user
/// directory (`~/.nimbus/runs`) when supplied non-empty from Settings; the
/// timestamped per-run subdir is always built underneath.
fn default_output_dir(runs_root: Option<&str>, case_name: &str, created_unix_ms: u128) -> PathBuf {
    let root = match runs_root.map(str::trim).filter(|s| !s.is_empty()) {
        Some(r) => PathBuf::from(expand_tilde(r)),
        None => crate::platform::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".nimbus")
            .join("runs"),
    };
    root.join(format!("{created_unix_ms}-{case_name}"))
}

/// Expand a leading tilde using HOME on Unix or USERPROFILE on Windows.
fn expand_tilde(p: &str) -> String {
    crate::platform::expand_tilde(p).display().to_string()
}

/// Map an exited child's outcome to a terminal RunStatus. Pulled out of the
/// waiter so the (load-bearing) precedence rule — an abort request wins over
/// the exit code — is unit-testable without spawning a process.
///
/// `success` mirrors `ExitStatus::success()`: Some(true) clean, Some(false)
/// non-zero, None when `wait()` itself errored.
fn resolve_final_status(abort_requested: bool, success: Option<bool>) -> RunStatus {
    if abort_requested {
        RunStatus::Aborted
    } else {
        match success {
            Some(true) => RunStatus::Completed,
            _ => RunStatus::Failed,
        }
    }
}

/* ---------- process signalling (platform-specific) ---------- */

/// Own process group (pgid == child pid) so abort and shutdown reach any helper
/// the simulator forks, not just the direct child. Windows gets the same reach
/// from `taskkill /T`. Shared with the tests so they exercise the real config.
#[cfg(unix)]
fn configure_process_group(command: &mut Command) {
    command.process_group(0);
}

/// Negated pid = the child's process group. `spawn_sim` puts every child in
/// its own group, so this reaches forked helpers too.
#[cfg(unix)]
fn process_group_of(pid: u32) -> AppResult<libc::pid_t> {
    libc::pid_t::try_from(pid)
        .map(|value| -value)
        .map_err(|_| AppError::Subprocess(format!("pid {pid} exceeds pid_t")))
}

#[cfg(unix)]
fn signal_pid(pid: u32, sig: i32) -> AppResult<()> {
    let result = unsafe { libc::kill(process_group_of(pid)?, sig) };
    if result == 0 {
        return Ok(());
    }
    let err = std::io::Error::last_os_error();
    // ESRCH = empty process group: the child already exited (e.g. it caught the
    // earlier SIGINT) between our liveness check and this signal. That's the
    // desired end state, not a failure.
    if err.raw_os_error() == Some(libc::ESRCH) {
        return Ok(());
    }
    Err(AppError::Subprocess(format!(
        "killpg({pid}, {sig}) failed: {err}"
    )))
}

/// Graceful termination request — SIGINT, equivalent to a CLI Ctrl-C, so the
/// simulator's signal handlers can flush and exit cleanly.
#[cfg(unix)]
fn request_terminate(pid: u32) -> AppResult<()> {
    signal_pid(pid, libc::SIGINT)
}

/// Forceful termination — SIGKILL, used to escalate when the child ignores the
/// graceful request.
#[cfg(unix)]
fn force_kill(pid: u32) -> AppResult<()> {
    signal_pid(pid, libc::SIGKILL)
}

/// Liveness probe: signal 0 performs the kernel's permission/existence check
/// without delivering a signal. Probing the group answers "is anything from
/// this run still alive", which is what escalation needs.
#[cfg(unix)]
fn process_alive(pid: u32) -> bool {
    match process_group_of(pid) {
        Ok(group) => unsafe { libc::kill(group, 0) == 0 },
        Err(_) => false,
    }
}

#[cfg(windows)]
fn run_taskkill(pid: u32, force: bool) -> AppResult<()> {
    // taskkill is the built-in Windows process terminator. Without /F it asks
    // the process (tree, via /T) to close; with /F it terminates forcefully.
    let mut cmd = Command::new("taskkill");
    cmd.arg("/PID").arg(pid.to_string()).arg("/T");
    if force {
        cmd.arg("/F");
    }
    // A non-zero exit means the PID was already gone — the goal state — so we
    // only surface a spawn failure (taskkill missing / not permitted).
    cmd.status()
        .map(|_| ())
        .map_err(|e| AppError::Subprocess(format!("taskkill {pid}: {e}")))
}

#[cfg(windows)]
fn request_terminate(pid: u32) -> AppResult<()> {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{OpenEventW, SetEvent, EVENT_MODIFY_STATE};

    let mut name: Vec<u16> = format!("Local\\Nimbus_Terminate_{pid}")
        .encode_utf16()
        .collect();
    name.push(0);
    // A user can click Abort immediately after CreateProcess succeeds, a few
    // milliseconds before the simulator installs its runtime handlers.  Keep
    // the operation deterministic by giving that named event a short bounded
    // publication window; otherwise the first click errors and, critically,
    // no force-escalation timer is armed.
    let mut event = std::ptr::null_mut();
    for _ in 0..20 {
        event = unsafe { OpenEventW(EVENT_MODIFY_STATE, 0, name.as_ptr()) };
        if !event.is_null() {
            break;
        }
        if !process_alive(pid) {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(25));
    }
    if event.is_null() {
        return Err(AppError::Subprocess(format!(
            "simulator {pid} did not publish its graceful-stop event within 500ms"
        )));
    }
    let signalled = unsafe { SetEvent(event) } != 0;
    unsafe { CloseHandle(event) };
    if signalled {
        Ok(())
    } else {
        Err(AppError::Subprocess(format!(
            "could not request a graceful stop from simulator {pid}"
        )))
    }
}

#[cfg(windows)]
fn force_kill(pid: u32) -> AppResult<()> {
    run_taskkill(pid, true)
}

#[cfg(windows)]
fn process_alive(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::{CloseHandle, STILL_ACTIVE};
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };

    let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
    if process.is_null() {
        return false;
    }
    let mut exit_code = 0;
    let queried = unsafe { GetExitCodeProcess(process, &mut exit_code) } != 0;
    unsafe { CloseHandle(process) };
    queried && exit_code == STILL_ACTIVE
}

#[cfg(not(any(unix, windows)))]
fn request_terminate(_pid: u32) -> AppResult<()> {
    Err(AppError::Subprocess(
        "abort not supported on this platform".into(),
    ))
}

#[cfg(not(any(unix, windows)))]
fn force_kill(_pid: u32) -> AppResult<()> {
    Err(AppError::Subprocess(
        "abort not supported on this platform".into(),
    ))
}

#[cfg(not(any(unix, windows)))]
fn process_alive(_pid: u32) -> bool {
    false
}

fn process_exit_code(status: &std::process::ExitStatus) -> Option<i32> {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        status.code().or_else(|| status.signal().map(|signal| -signal))
    }
    #[cfg(not(unix))]
    { status.code() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn process_exit_preserves_signals_and_normal_codes() {
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(process_exit_code(&std::process::ExitStatus::from_raw(4)), Some(-4));
        assert_eq!(process_exit_code(&std::process::ExitStatus::from_raw(11)), Some(-11));
        assert_eq!(process_exit_code(&std::process::ExitStatus::from_raw(132 << 8)), Some(132));
        assert_eq!(process_exit_code(&std::process::ExitStatus::from_raw(0)), Some(0));
    }

    #[test]
    fn abort_request_wins_over_exit_code() {
        // A SIGINT/SIGKILL'd child usually exits non-zero, but the user asked
        // for it: Aborted, not Failed — regardless of how it exited.
        assert_eq!(resolve_final_status(true, Some(true)), RunStatus::Aborted);
        assert_eq!(resolve_final_status(true, Some(false)), RunStatus::Aborted);
        assert_eq!(resolve_final_status(true, None), RunStatus::Aborted);
    }

    #[test]
    fn unaborted_runs_follow_the_exit_code() {
        assert_eq!(
            resolve_final_status(false, Some(true)),
            RunStatus::Completed
        );
        assert_eq!(resolve_final_status(false, Some(false)), RunStatus::Failed);
        // wait() error (None) is treated as a failure, not a clean exit.
        assert_eq!(resolve_final_status(false, None), RunStatus::Failed);
    }

    #[test]
    fn accepts_versioned_scheme_catalog() {
        let json = br#"{
            "schema":"nimbus.scheme-catalog.v9",
            "catalogVersion":9,
            "runtimeStateFields":["rho","u","v","w"],
            "readiness":{"version":1,"items":[{
                "id":"catalog.runtime_cli_studio",
                "area":"capability_truth",
                "scope":"Exact-binary catalog agreement",
                "state":"ready",
                "boundedEvidence":"tests",
                "campaignEvidence":"",
                "externalReviewEvidence":"",
                "blocker":""
            }]},
            "runtimeCapabilities":{"outputFormats":[
                {"id":"npy_3d","available":true,"reason":"","implementation":"builtin_npy"},
                {"id":"netcdf","available":false,"reason":"no libnetcdf","implementation":""}
            ]},
            "modules":[{"id":"microphysics","selectionPath":"microphysics.scheme",
                "studioSelection":{"mode":"form","fixedSchemeId":"","reason":""},"schemes":[{
                "id":"kessler",
                "maturity":"qualified",
                "constructible":true,
                "runtimeSelectable":true,
                "readinessState":"ready",
                "parameters":[{"path":"microphysics.shared.autoconversion_rate_s_inv",
                    "displayName":"Autoconversion rate","kind":"float","default":0.001}]
            }]}]
        }"#;
        let mut parsed = parse_scheme_catalog_json(json).expect("catalog should parse");
        assert_eq!(parsed["catalogVersion"], 9);

        parsed["modules"][0]
            .as_object_mut()
            .expect("fixture module")
            .remove("studioSelection");
        let missing_policy = serde_json::to_vec(&parsed).expect("serialize fixture");
        assert!(parse_scheme_catalog_json(&missing_policy).is_err());

        let mut malformed_readiness: Value = serde_json::from_slice(json).expect("fixture JSON");
        malformed_readiness["readiness"]["items"][0]["boundedEvidence"] =
            Value::String(String::new());
        let malformed_readiness =
            serde_json::to_vec(&malformed_readiness).expect("serialize fixture");
        assert!(parse_scheme_catalog_json(&malformed_readiness).is_err());

        let mut contradictory_scheme: Value = serde_json::from_slice(json).expect("fixture JSON");
        contradictory_scheme["modules"][0]["schemes"][0]["constructible"] = Value::Bool(false);
        let contradictory_scheme =
            serde_json::to_vec(&contradictory_scheme).expect("serialize fixture");
        assert!(parse_scheme_catalog_json(&contradictory_scheme).is_err());
    }

    #[test]
    fn rejects_unknown_or_malformed_scheme_catalog() {
        let wrong_schema =
            br#"{"schema":"other","catalogVersion":9,"runtimeStateFields":["u"],"modules":[{}]}"#;
        assert!(parse_scheme_catalog_json(wrong_schema).is_err());

        let old_version =
            br#"{"schema":"nimbus.scheme-catalog.v1","catalogVersion":1,"modules":[{}]}"#;
        assert!(parse_scheme_catalog_json(old_version).is_err());

        let missing_modules = br#"{"schema":"nimbus.scheme-catalog.v9","catalogVersion":9,"runtimeStateFields":["u"]}"#;
        assert!(parse_scheme_catalog_json(missing_modules).is_err());

        let missing_state_inventory =
            br#"{"schema":"nimbus.scheme-catalog.v9","catalogVersion":9,"modules":[{}]}"#;
        assert!(parse_scheme_catalog_json(missing_state_inventory).is_err());

        let duplicate_state = br#"{"schema":"nimbus.scheme-catalog.v9","catalogVersion":9,"runtimeStateFields":["u","u"],"modules":[{}]}"#;
        assert!(parse_scheme_catalog_json(duplicate_state).is_err());

        let missing_parameters = br#"{
            "schema":"nimbus.scheme-catalog.v9","catalogVersion":9,
            "runtimeStateFields":["u"],
            "modules":[{"id":"microphysics","selectionPath":"microphysics.scheme",
                "studioSelection":{"mode":"form","fixedSchemeId":"","reason":""},
                "schemes":[{"id":"kessler"}]}]
        }"#;
        assert!(parse_scheme_catalog_json(missing_parameters).is_err());

        let wrong_parameter_type = br#"{
            "schema":"nimbus.scheme-catalog.v9","catalogVersion":9,
            "runtimeStateFields":["u"],
            "modules":[{"id":"microphysics","selectionPath":"microphysics.scheme",
                "studioSelection":{"mode":"form","fixedSchemeId":"","reason":""},
                "schemes":[{"id":"kessler","parameters":[{
                    "path":"microphysics.shared.autoconversion_rate_s_inv",
                    "displayName":"Autoconversion rate","kind":"float","default":true
                }]}]}]
        }"#;
        assert!(parse_scheme_catalog_json(wrong_parameter_type).is_err());
    }

    /// Spawn a throwaway child the same way `spawn_sim` does, so the group and
    /// shutdown assertions exercise the real spawn configuration.
    #[cfg(unix)]
    fn spawn_grouped_child() -> std::process::Child {
        let mut command = Command::new("sleep");
        command
            .arg("30")
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        configure_process_group(&mut command);
        command.spawn().expect("sleep should spawn")
    }

    #[cfg(unix)]
    fn running_record(pid: u32) -> SharedRun {
        Arc::new(Mutex::new(RunRecord {
            id: "run-test".into(),
            case_name: "test".into(),
            scheme: String::new(),
            duration_s: 0.0,
            config_path: PathBuf::from("test.yaml"),
            pid,
            status: RunStatus::Running,
            abort_requested: false,
            progress: None,
            started_at_ms: 0,
            ended_at_ms: None,
            output_dir: PathBuf::from("/tmp"),
            remote: None,
            remote_termination_unconfirmed: false,
            remote_transfer_in_progress: false,
            transport_exited: false,
            runtime_log_evidence: Arc::new(Mutex::new(RuntimeLogEvidence::default())),
        }))
    }

    #[cfg(unix)]
    #[test]
    fn recovery_completion_retains_observed_backend_and_leaves_missing_values_unavailable() {
        let root = std::env::temp_dir().join(format!("nimbus-recovery-evidence-{}", now_unix_ms()));
        std::fs::create_dir(&root).unwrap();
        let binary = root.join("simulator"); std::fs::write(&binary, b"identity fixture").unwrap();
        let config = root.join("source.yaml"); std::fs::write(&config, b"schema: nimbus.v1\n").unwrap();
        create_run_provenance(&root, "run-test", 1, &binary, &config, None, RequestedRunValues::default()).unwrap();
        let record = running_record(0);
        let evidence = record.lock().unwrap().runtime_log_evidence.clone();
        evidence.lock().unwrap().actual_backend = Some("vulkan".into());
        let completion = create_runtime_completion_provenance(&root, RuntimeCompletionInput { run_id: "run-test".into(),
            process_started_unix_ms: 2, process_ended_unix_ms: 3, final_status: "aborted".into(), exit_code: None,
            log_evidence: retained_runtime_evidence(&record) }).unwrap();
        let value = serde_json::to_value(completion).unwrap();
        assert_eq!(value["actual"]["backend"]["status"], "observed");
        assert_eq!(value["actual"]["backend"]["value"], "vulkan");
        assert_eq!(value["actual"]["fallbackActive"]["status"], "unavailable");
        assert!(value["actual"]["fallbackActive"]["value"].is_null());
        std::fs::remove_dir_all(root).unwrap();
    }

    /// Abort and shutdown signal the negated pid, which only reaches forked
    /// helpers if the child leads its own group.
    #[cfg(unix)]
    #[test]
    fn spawned_child_leads_its_own_process_group() {
        let mut child = spawn_grouped_child();
        let pid = child.id();
        let pgid = unsafe { libc::getpgid(pid as libc::pid_t) };
        assert_eq!(pgid, pid as libc::pid_t, "child should lead its own group");
        let _ = force_kill(pid);
        let _ = child.wait();
    }

    /// Regression: quitting Studio left nimbus_sim running headless, because
    /// nothing signalled the child on exit.
    #[cfg(unix)]
    #[test]
    fn terminate_all_runs_kills_live_children() {
        let state = AppState::default();
        let mut child = spawn_grouped_child();
        let pid = child.id();
        state
            .runs
            .lock()
            .expect("runs lock")
            .insert("run-test".into(), running_record(pid));

        // Mirror production: the waiter thread reaps, so a killed child does
        // not linger as a zombie and keep liveness probes true.
        let reaper = thread::spawn(move || child.wait());
        terminate_all_runs(&state);

        let status = reaper.join().expect("reaper thread").expect("wait");
        assert!(!status.success(), "child should die by signal, not exit 0");
        assert!(!process_alive(pid), "no group member should survive");
    }

    /// A run the waiter already reaped must not be signalled — the pid may
    /// have been recycled.
    #[cfg(unix)]
    #[test]
    fn terminate_all_runs_ignores_finished_records() {
        let state = AppState::default();
        let mut child = spawn_grouped_child();
        let pid = child.id();
        let record = running_record(pid);
        record.lock().expect("record lock").status = RunStatus::Completed;
        state
            .runs
            .lock()
            .expect("runs lock")
            .insert("run-test".into(), record);

        terminate_all_runs(&state);
        assert!(
            process_alive(pid),
            "completed records must not be signalled"
        );

        let _ = force_kill(pid);
        let _ = child.wait();
    }

    const TEST_GRACE: Duration = Duration::from_millis(80);

    #[test]
    fn drain_completes_once_every_reader_sender_drops() {
        let (tx, rx) = mpsc::channel::<()>();
        let a = tx.clone();
        let b = tx;
        drop(a);
        drop(b);
        assert!(wait_for_reader_drain(&rx, TEST_GRACE));
    }

    /// Regression: the waiter joined the readers unconditionally, so a
    /// grandchild holding the pipe open stranded the run in `Running` forever.
    #[test]
    fn drain_times_out_while_a_reader_still_holds_the_pipe() {
        let (tx, rx) = mpsc::channel::<()>();
        let stuck = tx.clone();
        drop(tx);
        assert!(
            !wait_for_reader_drain(&rx, TEST_GRACE),
            "one live sender must not report a completed drain"
        );
        drop(stuck);
        assert!(
            wait_for_reader_drain(&rx, TEST_GRACE),
            "drain completes after it releases"
        );
    }

    #[test]
    fn catalog_fingerprint_is_stable_and_byte_exact() {
        assert_eq!(fnv1a64(b"catalog\n"), fnv1a64(b"catalog\n"));
        assert_ne!(fnv1a64(b"catalog\n"), fnv1a64(b"catalog"));
        assert_ne!(fnv1a64(b"catalog\n"), fnv1a64(b"catalog!\n"));
    }

    #[test]
    fn exact_binary_evidence_reaches_initialized_microphysics_identity() {
        let repository = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("repository root should resolve");
        let binary_candidate = crate::platform::simulator_below(&repository);
        if !binary_candidate.is_file() {
            eprintln!("native Nimbus simulator is not built; skipping native launch regression");
            return;
        }
        let binary = binary_candidate
            .canonicalize()
            .expect("simulator binary should have a canonical identity");

        // Exercise the same byte-level catalog query, v2 parser, and
        // fingerprint verifier used immediately before spawn_sim.
        let catalog_bytes = query_scheme_catalog_bytes(&binary)
            .expect("exact simulator catalog should be queryable");
        let catalog = parse_scheme_catalog_json(&catalog_bytes)
            .expect("exact simulator catalog should satisfy the native contract");
        let microphysics = catalog["modules"]
            .as_array()
            .and_then(|modules| modules.iter().find(|module| module["id"] == "microphysics"))
            .expect("catalog should publish microphysics");
        let kessler = microphysics["schemes"]
            .as_array()
            .and_then(|schemes| schemes.iter().find(|scheme| scheme["id"] == "kessler"))
            .expect("catalog should publish kessler");
        assert_eq!(kessler["constructible"], Value::Bool(true));

        let fingerprint = format!("fnv1a64:{:016x}", fnv1a64(&catalog_bytes));
        verify_catalog_fingerprint(&binary, &fingerprint)
            .expect("unchanged exact binary should retain its preflight evidence");
        let stale = verify_catalog_fingerprint(&binary, "fnv1a64:0000000000000000")
            .expect_err("stale catalog evidence must fail before launch");
        assert!(stale
            .to_string()
            .contains("simulator binary changed after configuration preflight"));

        let output_dir = std::env::temp_dir().join(format!(
            "nimbus-native-preflight-launch-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&output_dir).expect("temporary output directory should exist");
        let config = repository.join("tests/fixtures/configs/test/acceptance/t5_cart_kessler.yaml");
        let output = Command::new(&binary)
            .current_dir(&repository)
            .arg("--headless")
            .arg("--config")
            .arg(&config)
            .arg("--duration=0")
            .arg("--write-every=0")
            .arg("--log-profile=normal")
            .arg(format!("--outdir={}", output_dir.display()))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .expect("preflighted simulator should spawn");
        let mut log = String::from_utf8_lossy(&output.stdout).into_owned();
        log.push_str(&String::from_utf8_lossy(&output.stderr));
        let _ = std::fs::remove_dir_all(&output_dir);

        assert!(
            output.status.success(),
            "tiny preflighted launch failed:\n{log}"
        );
        let configured = log
            .find("Microphysics: kessler")
            .expect("configuration summary should retain selected kessler identity");
        let initialized = log
            .find("Initialized microphysics scheme: kessler")
            .expect("initialized runtime should report selected kessler identity");
        assert!(initialized > configured);
        assert!(!log.contains("Falling back to Kessler"));
    }
}
