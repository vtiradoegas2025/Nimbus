use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tauri::State;

use super::scheme_catalog_contract::{validate_readiness_contract, validate_scheme_readiness};
use crate::error::{AppError, AppResult};
use crate::state::AppState;

const REQUEST_SCHEMA: &str = "nimbus.config-preflight.v10";
const REPORT_SCHEMA: &str = "nimbus.config-preflight-report.v10";
static PREFLIGHT_SOURCE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A complete, versioned description of the configuration choices that can
/// affect scheme compatibility. Studio builds this from its parsed config;
/// the Rust command deliberately does not maintain a second YAML schema.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ConfigPreflightRequest {
    pub schema: String,
    pub request_version: u32,
    pub source: PreflightSource,
    pub execution_mode: String,
    pub maturity_limits_acknowledged: bool,
    pub backend: String,
    pub allow_backend_fallback: bool,
    pub validation_guard_mode: String,
    pub gpu_apply_microphysics: bool,
    pub gpu_apply_slow: bool,
    pub gpu_apply_sgs: bool,
    pub output_format: String,
    pub grid_metrics: GridMetrics,
    pub time_control: TimeControl,
    pub output: OutputInteractionValues,
    pub coordinate_system: String,
    pub staggering: String,
    pub boundary: BoundarySelection,
    pub diagnostics: DiagnosticsSelection,
    pub initialization: InitializationSelection,
    pub interactions: ScienceInteractionValues,
    pub available_state_fields: Vec<String>,
    pub selections: Vec<SchemeSelection>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PreflightSource {
    pub kind: String,
    pub yaml_text: Option<String>,
    pub yaml_path: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GridMetrics {
    pub nx: i64,
    pub ny: i64,
    pub nz: i64,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    pub ztop: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TimeControl {
    pub dt_large: f64,
    pub acoustic_substeps: i64,
    pub output_interval: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OutputInteractionValues {
    pub preset: String,
    pub async_io: bool,
    pub zfp_mode: String,
    pub zfp_per_field_tolerances: bool,
    pub zfp_sparse_threshold: f64,
    pub zfp_float16_prefilter: bool,
    pub zfp_field_execution: String,
    pub zfp_field_workers: i64,
    pub zfp_field_scratch_mib: i64,
    pub zfp_field_tolerances: BTreeMap<String, f64>,
    pub keep_last_frames: i64,
    pub minimum_free_space_mib: i64,
    pub maximum_retained_history_files: i64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct InitializationSelection {
    pub sounding_source: String,
    pub hodograph_source: String,
    pub trigger_source: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BoundarySelection {
    pub west: String,
    pub east: String,
    pub south: String,
    pub north: String,
    pub bottom: String,
    pub top: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DiagnosticsSelection {
    pub scientific_every_steps: i64,
    pub deterministic_reductions: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SchemeSelection {
    pub module_id: String,
    pub scheme_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ScienceInteractionValues {
    pub radiation_shortwave_enabled: bool,
    pub turbulence_mode: String,
    pub turbulence_dynamic_coefficient: bool,
    pub surface_scheme: String,
    pub surface_apply_fluxes: bool,
    pub diffusion_target: String,
    pub diffusion_active: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ConfigPreflightReport {
    pub schema: &'static str,
    pub report_version: u32,
    pub compatible: bool,
    pub native_admission: NativeAdmissionEvidence,
    pub executable: ExecutableIdentity,
    pub execution_target_fingerprint: Option<String>,
    pub context: PreflightContext,
    pub resolved_selections: Vec<ResolvedSelection>,
    pub hard_errors: Vec<PreflightIssue>,
    pub warnings: Vec<PreflightIssue>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NativeAdmissionEvidence {
    pub state: &'static str,
    pub descriptor_projection: &'static str,
    pub source_sha256: Option<String>,
    pub source_fingerprint: Option<String>,
    pub source_path_fingerprint: Option<String>,
    pub validated_binary_sha256: Option<String>,
    pub validation_exit_code: Option<i32>,
}

struct NativeValidationOutcome {
    accepted: bool,
    source_sha256: String,
    source_fingerprint: String,
    source_path_fingerprint: String,
    binary_sha256: String,
    exit_code: Option<i32>,
    diagnostic: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct NativeValidateOnlyContext {
    schema: String,
    version: u32,
    source_kind: String,
    source_sha256: String,
    generated_metal_scope_kind: String,
    context: PreflightContext,
    selected_schemes: BTreeMap<String, String>,
    radar_products: Vec<String>,
    unserialized_controls: NativeUnserializedControls,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct NativeUnserializedControls {
    lateral_sponge_enabled: bool,
    lateral_width: i64,
    lateral_alpha: f64,
    w_only_top_sponge: bool,
    w_only_depth_cells: i64,
    w_only_alpha_max: f64,
    slow_rk2: bool,
    lid_bc_gamma: f64,
    axis_u_m1_filter: bool,
}

struct UniqueJson(Value);

impl<'de> Deserialize<'de> for UniqueJson {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct UniqueVisitor;
        impl<'de> serde::de::Visitor<'de> for UniqueVisitor {
            type Value = UniqueJson;
            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("JSON with unique object keys")
            }
            fn visit_bool<E: serde::de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(UniqueJson(Value::Bool(value)))
            }
            fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(UniqueJson(Value::from(value)))
            }
            fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(UniqueJson(Value::from(value)))
            }
            fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(value).map(|value| UniqueJson(Value::Number(value)))
                    .ok_or_else(|| E::custom("nonfinite context number"))
            }
            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(UniqueJson(Value::String(value.into())))
            }
            fn visit_string<E: serde::de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(UniqueJson(Value::String(value)))
            }
            fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(UniqueJson(Value::Null))
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(value) = sequence.next_element::<UniqueJson>()? {
                    values.push(value.0);
                }
                Ok(UniqueJson(Value::Array(values)))
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut values = serde_json::Map::new();
                while let Some((key, value)) = map.next_entry::<String, UniqueJson>()? {
                    if values.insert(key, value.0).is_some() {
                        return Err(serde::de::Error::custom("duplicate context JSON key"));
                    }
                }
                Ok(UniqueJson(Value::Object(values)))
            }
        }
        deserializer.deserialize_any(UniqueVisitor)
    }
}

fn parse_native_validate_only_context(stdout: &str, expected_sha256: &str) -> AppResult<NativeValidateOnlyContext> {
    let records = stdout.lines().filter(|line| line.starts_with("[VALIDATE_CONTEXT]")).collect::<Vec<_>>();
    if records.len() != 1 {
        return Err(AppError::Invalid("native validation must emit exactly one context receipt".into()));
    }
    let record = records[0].strip_prefix("[VALIDATE_CONTEXT] ")
        .ok_or_else(|| AppError::Invalid("native context proof marker is malformed".into()))?;
    if record.len() > 64 * 1024 {
        return Err(AppError::Invalid("native context receipt exceeds 64 KiB".into()));
    }
    let unique: UniqueJson = serde_json::from_str(record)
        .map_err(|error| AppError::Invalid(format!("invalid native context JSON: {error}")))?;
    let receipt: NativeValidateOnlyContext = serde_json::from_value(unique.0)
        .map_err(|error| AppError::Invalid(format!("invalid native context fields: {error}")))?;
    if receipt.schema != "nimbus.validate-only-context.v1" || receipt.version != 1 ||
        receipt.source_kind != "exact_document" || receipt.source_sha256 != expected_sha256 ||
        !["none", "canonical", "east_npy", "mixed_npy"].contains(&receipt.generated_metal_scope_kind.as_str()) {
        return Err(AppError::Invalid("native context identity, schema or scope kind mismatched".into()));
    }
    let facts = &receipt.unserialized_controls;
    let known_selections = ["dynamics", "microphysics", "radiation", "boundary_layer", "turbulence",
                            "advection", "diffusion", "time_stepping", "chaos", "terrain", "soundings"];
    if !facts.lateral_alpha.is_finite() || !facts.w_only_alpha_max.is_finite() ||
        !facts.lid_bc_gamma.is_finite() || facts.lateral_width < 0 || facts.w_only_depth_cells < 0 ||
        receipt.context.available_state_fields.len() > 512 ||
        receipt.context.output.zfp_field_tolerances.len() > 256 ||
        receipt.selected_schemes.len() < 10 || receipt.selected_schemes.len() > 11 ||
        receipt.selected_schemes.keys().any(|key| !known_selections.contains(&key.as_str())) ||
        known_selections[..10].iter().any(|key| !receipt.selected_schemes.contains_key(*key)) ||
        receipt.radar_products.len() > 32 ||
        receipt.context.available_state_fields.iter().collect::<BTreeSet<_>>().len() !=
            receipt.context.available_state_fields.len() {
        return Err(AppError::Invalid("native context fact bounds or inventory mismatched".into()));
    }
    Ok(receipt)
}

fn apply_native_context_validation(report: &mut ConfigPreflightReport, native: NativeValidationOutcome,
                                   receipt: Option<NativeValidateOnlyContext>, catalog: &Value) -> AppResult<()> {
    if !native.accepted {
        apply_native_validation(report, native);
        return Ok(());
    }
    let receipt = receipt.ok_or_else(|| AppError::Invalid("accepted source omitted native context receipt".into()))?;
    let mut context = receipt.context;
    context.backend = normalize_backend(&context.backend);
    context.coordinate_system = normalize(&context.coordinate_system);
    context.staggering = normalize(&context.staggering);
    context.initialization.sounding_source = normalize_sounding(&context.initialization.sounding_source);
    context.initialization.hodograph_source = normalize_hodograph(&context.initialization.hodograph_source);
    context.initialization.trigger_source = normalize_trigger(&context.initialization.trigger_source);
    context.available_state_fields.sort();
    let mut requested_context = report.context.clone();
    requested_context.available_state_fields.sort();
    let mismatched_context = context != requested_context;
    let mismatched_selection = report.resolved_selections.iter().any(|selection| {
        if selection.module_id == "radar" {
            return !receipt.radar_products.contains(&selection.requested_scheme_id);
        }
        let Some(actual) = receipt.selected_schemes.get(&selection.module_id) else {
            return true;
        };
        let Some(module) = catalog["modules"].as_array().and_then(|modules|
            modules.iter().find(|module| string(module, "id") == Some(selection.module_id.as_str()))) else {
            return true;
        };
        let canonical = module["schemes"].as_array().and_then(|schemes| find_scheme(schemes, actual))
            .and_then(|scheme| string(scheme, "id"));
        canonical != Some(selection.canonical_scheme_id.as_str())
    });
    let has_generated_pending = report.hard_errors.iter().any(|issue|
        issue.code == "qualified_generated_metal_native_admission_required");
    let facts = &receipt.unserialized_controls;
    let expected_generated_scope = if requested_context.boundary.east == "radiative" {
        "east_npy"
    } else if [requested_context.boundary.west.as_str(), requested_context.boundary.east.as_str(),
               requested_context.boundary.south.as_str(), requested_context.boundary.north.as_str()] == ["rigid_slip"; 4] {
        "mixed_npy"
    } else {
        "canonical"
    };
    let mismatched_generated_facts = has_generated_pending &&
        (receipt.generated_metal_scope_kind != expected_generated_scope ||
         !facts.lateral_sponge_enabled || facts.lateral_width != 20 || facts.lateral_alpha != 0.05 ||
         !facts.w_only_top_sponge || facts.w_only_depth_cells != 8 || facts.w_only_alpha_max != 0.2 ||
         !facts.slow_rk2 || facts.lid_bc_gamma != -1.0 || facts.axis_u_m1_filter);
    if mismatched_context || mismatched_selection || mismatched_generated_facts {
        report.hard_errors.push(issue(
            "native_source_context_mismatch", None, None,
            "native validated source context differs from the declared preflight request".into(),
            vec!["matching native-owned source context and selected schemes".into()],
            Some(format!("context={mismatched_context},selection={mismatched_selection},scope={mismatched_generated_facts}")),
        ));
        let pending = report.hard_errors.iter().filter(|issue|
            ["qualified_generated_metal_native_admission_required", "qualified_zfp_radar_native_admission_required",
             "qualified_vertical_only_sgs_native_admission_required"].contains(&issue.code))
            .cloned().collect::<Vec<_>>();
        apply_native_validation(report, native);
        report.hard_errors.extend(pending);
        report.compatible = false;
    } else {
        apply_native_validation(report, native);
    }
    Ok(())
}

struct TemporaryPreflightSource(PathBuf);

impl Drop for TemporaryPreflightSource {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn validate_exact_document(
    binary: &Path,
    expected_binary_sha256: &str,
    source: &PreflightSource,
) -> AppResult<NativeValidationOutcome> {
    validate_exact_document_with_context(binary, expected_binary_sha256, source, false).map(|result| result.0)
}

fn validate_exact_document_with_context(
    binary: &Path,
    expected_binary_sha256: &str,
    source: &PreflightSource,
    require_context: bool,
) -> AppResult<(NativeValidationOutcome, Option<NativeValidateOnlyContext>)> {
    let text = source.yaml_text.as_deref().ok_or_else(||
        AppError::Invalid("exact preflight source has no YAML text".into()))?;
    let source_path = source.yaml_path.as_deref().ok_or_else(||
        AppError::Invalid("exact preflight source has no YAML path".into()))?;
    if text.trim().is_empty() || text.len() > 8 * 1024 * 1024 {
        return Err(AppError::Invalid("exact preflight YAML is empty or exceeds 8 MiB".into()));
    }
    let parent = Path::new(source_path).parent()
        .filter(|path| path.is_dir())
        .ok_or_else(|| AppError::Invalid("exact preflight YAML path has no readable parent".into()))?;
    if crate::platform::sha256_file(binary)? != expected_binary_sha256 {
        return Err(AppError::Invalid("simulator changed before exact source validation".into()));
    }
    let sequence = PREFLIGHT_SOURCE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let now = SystemTime::now().duration_since(UNIX_EPOCH)
        .map_err(|_| AppError::Invalid("system clock is before Unix epoch".into()))?
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "nimbus-exact-preflight-{}-{now}-{sequence}.yaml", std::process::id()));
    let mut file = fs::OpenOptions::new().create_new(true).write(true).open(&path)?;
    let temporary = TemporaryPreflightSource(path);
    file.write_all(text.as_bytes())?;
    file.sync_all()?;
    drop(file);
    let mut command = Command::new(binary);
    command.arg("--config").arg(&temporary.0).arg("--validate-only").arg("--headless");
    if require_context {
        command.arg("--preflight-context=json");
    }
    let output = command.current_dir(parent)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|error| AppError::Subprocess(format!(
            "spawn {} for exact preflight: {error}", binary.display())))?;
    if fs::read(&temporary.0)? != text.as_bytes() {
        return Err(AppError::Invalid("native validation changed its staged YAML source".into()));
    }
    let binary_sha256 = crate::platform::sha256_file(binary)?;
    if binary_sha256 != expected_binary_sha256 {
        return Err(AppError::Invalid("simulator changed during exact source validation".into()));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let accepted = output.status.success() && stdout.contains("[VALIDATE] PASS");
    let source_sha256 = format!("{:x}", Sha256::digest(text.as_bytes()));
    let context = if require_context && accepted {
        Some(parse_native_validate_only_context(&stdout, &source_sha256)?)
    } else {
        None
    };
    let diagnostic = if accepted { String::new() } else {
        format!("{stderr}\n{stdout}").trim().chars().take(2048).collect()
    };
    Ok((NativeValidationOutcome {
        accepted,
        source_sha256,
        source_fingerprint: format!("fnv1a64:{:016x}", fnv1a64(text.as_bytes())),
        source_path_fingerprint: format!("fnv1a64:{:016x}", fnv1a64(source_path.as_bytes())),
        binary_sha256,
        exit_code: output.status.code(),
        diagnostic,
    }, context))
}

fn apply_native_validation(report: &mut ConfigPreflightReport, native: NativeValidationOutcome) {
    if native.accepted {
        report.hard_errors.retain(|issue|
            issue.code != "qualified_zfp_radar_native_admission_required"
                && issue.code != "qualified_vertical_only_sgs_native_admission_required"
                && issue.code != "qualified_generated_metal_native_admission_required");
        report.compatible = report.hard_errors.is_empty();
    } else {
        report.hard_errors.push(issue(
            "native_strict_admission_rejected", None, None,
            format!("exact simulator validation rejected the source YAML: {}", native.diagnostic),
            vec!["exact native --validate-only acceptance".into()],
            native.exit_code.map(|code| format!("exit {code}")),
        ));
        report.compatible = false;
    }
    report.native_admission = NativeAdmissionEvidence {
        state: if native.accepted { "accepted" } else { "rejected" },
        descriptor_projection: "not_checked",
        source_sha256: Some(native.source_sha256),
        source_fingerprint: Some(native.source_fingerprint),
        source_path_fingerprint: Some(native.source_path_fingerprint),
        validated_binary_sha256: Some(native.binary_sha256),
        validation_exit_code: native.exit_code,
    };
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ExecutableIdentity {
    pub canonical_path: String,
    pub byte_length: u64,
    pub modified_unix_ms: Option<u128>,
    /// Stable, non-cryptographic fingerprint of the exact catalog bytes
    /// returned by this executable invocation.
    pub catalog_fingerprint: String,
    /// Stable fingerprint of the executable bytes themselves.
    pub binary_fingerprint: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PreflightContext {
    pub execution_mode: String,
    pub maturity_limits_acknowledged: bool,
    pub backend: String,
    pub allow_backend_fallback: bool,
    pub validation_guard_mode: String,
    pub gpu_apply_microphysics: bool,
    pub gpu_apply_slow: bool,
    pub gpu_apply_sgs: bool,
    pub output_format: String,
    pub grid_metrics: GridMetrics,
    pub time_control: TimeControl,
    pub output: OutputInteractionValues,
    pub coordinate_system: String,
    pub staggering: String,
    pub boundary: BoundarySelection,
    pub diagnostics: DiagnosticsSelection,
    pub initialization: InitializationSelection,
    pub interactions: ScienceInteractionValues,
    pub available_state_fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedSelection {
    pub module_id: String,
    pub requested_scheme_id: String,
    pub canonical_scheme_id: String,
    pub maturity: String,
    pub constructible: bool,
    pub runtime_selectable: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PreflightIssue {
    pub code: &'static str,
    pub module_id: Option<String>,
    pub scheme_id: Option<String>,
    pub message: String,
    pub expected: Vec<String>,
    pub actual: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CatalogInteractionPolicy {
    version: u32,
    generated_metal_canonical_scope: CatalogGeneratedMetalScope,
    qualified_flat_mixed_metal_npy_scope: Option<CatalogQualifiedFlatMixedMetalNpyScope>,
    inactive_scheme_id: String,
    horizontal_only_turbulence_modes: Vec<String>,
    known_surface_schemes: Vec<String>,
    active_surface_scheme_id: String,
    terrain_radiation_schemes: Vec<String>,
    terrain_longwave_radiation_scheme_id: String,
    terrain_shortwave_allowed: bool,
    terrain_boundary_layer_schemes: Vec<String>,
    terrain_turbulence_schemes: Vec<String>,
    terrain_chaos_schemes: Vec<String>,
    chaos_canonical_targets: BTreeMap<String, CatalogChaosCanonicalTargets>,
    terrain_active_diffusion_schemes: Vec<String>,
    terrain_active_diffusion_targets: Vec<String>,
    qualified_zfp_terrain_schemes: Vec<String>,
    qualified_zfp_preset_ids: Vec<String>,
    qualified_zfp_preset_fields: BTreeMap<String, Vec<String>>,
    qualified_zfp_mode: String,
    qualified_zfp_field_execution_policies: Vec<String>,
    qualified_zfp_max_field_workers: i64,
    qualified_zfp_min_field_scratch_mib: i64,
    qualified_zfp_max_field_scratch_mib: i64,
    qualified_zfp_requires_per_field_tolerances: bool,
    qualified_zfp_minimum_keep_last_frames: i64,
    qualified_zfp_minimum_free_space_mib: i64,
    qualified_zfp_requires_retained_file_ceiling: bool,
    qualified_execution_requires_maturity_acknowledgment: bool,
    qualified_zfp_max_sparse_threshold: f64,
    qualified_zfp_allows_float16_prefilter: bool,
    qualified_zfp_field_budgets: BTreeMap<String, f64>,
    qualified_zfp_field_units: BTreeMap<String, String>,
    qualified_zfp_reversible_radar_v4: Option<CatalogQualifiedRadarZfpV4Scope>,
    qualified_vertical_only_sgs: Option<CatalogQualifiedVerticalOnlySgsScope>,
    turbulence_dynamic_coefficient_default: Option<bool>,
    milbrandt_triple_moment: Option<CatalogMilbrandtTripleMomentScope>,
    lateral_boundary_modes: Vec<String>,
    radiative_lateral_staggerings: Vec<String>,
    legacy_open_cgrid_resolution: String,
    legacy_open_collocated_resolution: String,
    periodic_lateral_pairs_required: bool,
    bottom_boundary_modes: Vec<String>,
    top_boundary_modes: Vec<String>,
    scientific_diagnostics_omitted_sentinel: i64,
    scientific_diagnostics_minimum_cadence: i64,
    gpu_apply_controls_no_op_on_cpu: bool,
    gpu_apply_microphysics_schemes: Vec<String>,
    gpu_apply_sgs_schemes: Vec<String>,
    gpu_apply_slow_dynamics_schemes: Vec<String>,
    qualified_gpu_apply_slow_dynamics_schemes: Vec<String>,
    gpu_apply_microphysics_dispatch_guard_modes: Vec<String>,
    gpu_apply_microphysics_host_commit_guard_modes: Vec<String>,
    gpu_apply_slow_dispatch_guard_modes: Vec<String>,
    gpu_apply_slow_host_commit_guard_modes: Vec<String>,
    gpu_apply_sgs_dispatch_guard_modes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CatalogQualifiedFlatMixedMetalNpyScope {
    schema: String,
    version: u32,
    profile_count: i64,
    backend: String,
    execution_mode: String,
    family_id: String,
    strict_canonical_yaml_sha256: String,
    output_formats: Vec<String>,
    preset: String,
    write_cadence_seconds: Vec<i64>,
    async_io: bool,
    live_shm: bool,
    trajectories: bool,
    allow_fallback: bool,
    memory_admission_policies: Vec<String>,
    memory_policy_canonical_value: String,
    grid_shape: Vec<i64>,
    grid_spacing_meters: Vec<f64>,
    top_height_meters: f64,
    dt_large_seconds: f64,
    acoustic_substeps: i64,
    adaptive_dt: bool,
    lateral_boundaries: Vec<String>,
    bottom_boundary: String,
    top_boundary: String,
    scalar_transport_owner: String,
    initial_condition_chaos_owner: String,
    lateral_damping_owner: String,
    lateral_sponge_enabled: bool,
    lateral_sponge_width: i64,
    lateral_sponge_alpha: f64,
    w_only_top_sponge: bool,
    w_only_depth: i64,
    w_only_alpha: f64,
    effective_w_only_sponge: bool,
    source_profile_owner_symbol: String,
    manifest_sha256: String,
}

fn expected_flat_mixed_metal_scope() -> Option<CatalogQualifiedFlatMixedMetalNpyScope> {
    let source = include_str!("../../../../tests/fixtures/reference/catalog/flat_mixed_metal_cadence1_scope_v1.json");
    let digest = format!("{:x}", Sha256::digest(source.as_bytes()));
    if digest != "ea8921dabe62cceac5f95a023bdb92f3f5063dc9cd7aea6e8368925712014402" {
        return None;
    }
    let mut value: Value = serde_json::from_str(source).ok()?;
    value["manifestSha256"] = Value::String(digest);
    serde_json::from_value(value).ok()
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CatalogGeneratedMetalScope {
    version: u32,
    backend: String,
    execution_mode: String,
    tuple_count: u64,
    distinct_config_count: u64,
    output_formats: Vec<String>,
    scalar_transport_owner: String,
    base_manifest_sha256: String,
    manifest_sha256: String,
    npy_netcdf_manifest_sha256: Option<String>,
    zfp_typed_output_manifest_sha256: Option<String>,
    write_cadence_seconds_by_format: Option<BTreeMap<String, Vec<i64>>>,
    additional_boundary_npy_scope: Option<CatalogAdditionalBoundaryNpyScope>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CatalogAdditionalBoundaryNpyScope {
    version: u32,
    profile_count: u64,
    family_id: String,
    strict_canonical_yaml_sha256: String,
    output_formats: Vec<String>,
    write_cadence_seconds: Vec<i64>,
    manifest_sha256: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CatalogQualifiedRadarZfpV4Scope {
    version: u32,
    admitted: bool,
    product_ids: Vec<String>,
    nonempty_product_subsets: u32,
    core_fields: Vec<String>,
    presets: Vec<String>,
    named_preset_product_sets: BTreeMap<String, Vec<String>>,
    polarimetric_extra_fields: Vec<String>,
    output_format_tiers: Vec<String>,
    backend: String,
    coordinate_system: String,
    staggering: String,
    terrain: String,
    domain_motion: String,
    nesting: bool,
    reviewed_source_profile_contract: String,
    reviewed_source_profile_owner_symbol: String,
    reviewed_source_profile_families: Vec<String>,
    async_io_modes: Vec<bool>,
    grid_shape: Vec<i64>,
    grid_spacing_meters: Vec<f64>,
    dt_large_seconds: f64,
    acoustic_substeps: i64,
    write_cadence_seconds: Vec<i64>,
    non_radar_mode: String,
    non_radar_container_version: u32,
    non_radar_budget_owner: String,
    radar_mode: String,
    radar_container_version: u32,
    radar_keyframe_every_export: bool,
    radar_tolerance_policy: String,
    manifest_contract: String,
    source_field_bounds_contract: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CatalogQualifiedVerticalOnlySgsScope {
    version: u32,
    admitted: bool,
    scheme: String,
    mode: String,
    dynamic_coefficient: bool,
    execution_mode: String,
    backend: String,
    coordinate_system: String,
    staggering: String,
    terrain: String,
    domain_motion: String,
    nesting: bool,
    lateral_boundary_mode: String,
    active_boundary_layer_allowed: bool,
    grid_shape: Vec<i64>,
    grid_spacing_meters: Vec<f64>,
    dt_large_seconds: f64,
    acoustic_substeps: i64,
    output_format: String,
    write_cadence_seconds: Vec<i64>,
    source_profile_contract: String,
    source_profile_owner_symbol: String,
    whole_step_rollback: String,
    dynamic_non_three_d_allowed: bool,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CatalogMilbrandtTripleMomentScope {
    version: u32,
    retained: bool,
    admitted: bool,
    implementation_state: String,
    selector: String,
    strict_v1_explicit_key_admitted: bool,
    scheme: String,
    baseline_state: Vec<String>,
    additional_state: Vec<String>,
    additional_moment_order: u32,
    additional_moment_species: Vec<String>,
    private_storage_units: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CatalogChaosCanonicalTargets {
    initial_conditions: Vec<String>,
    tendencies: Vec<String>,
    native_generation_required: bool,
}

/// Query the exact configured Nimbus executable. An exact_document request
/// also validates its original YAML with that binary before reporting admission.
#[tauri::command]
pub async fn preflight_config(app: tauri::AppHandle, simulator_path: Option<String>, request: ConfigPreflightRequest,
    execution_target: Option<super::remote_compute::RemoteComputeTarget>) -> AppResult<ConfigPreflightReport> {
    use tauri::Manager;
    tauri::async_runtime::spawn_blocking(move || {
        preflight_config_blocking(app.state::<AppState>(), simulator_path, request, execution_target)
    }).await.map_err(|error| AppError::Task(error.to_string()))?
}

fn preflight_config_blocking(
    state: State<'_, AppState>,
    simulator_path: Option<String>,
    request: ConfigPreflightRequest,
    execution_target: Option<super::remote_compute::RemoteComputeTarget>,
) -> AppResult<ConfigPreflightReport> {
    validate_request(&request)?;
    if let Some(target) = execution_target {
        return preflight_remote_config(target, request);
    }
    let binary = resolve_binary(&state, simulator_path)?;
    let output = Command::new(&binary)
        .arg("--scheme-catalog=json")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|error| AppError::Subprocess(format!("spawn {}: {error}", binary.display())))?;

    if !output.status.success() {
        return Err(AppError::Subprocess(format!(
            "{} rejected --scheme-catalog=json (exit {:?}): {}",
            binary.display(),
            output.status.code(),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let catalog = parse_catalog(&output.stdout)?;
    let catalog_state = catalog_runtime_state_fields(&catalog)?;
    let requested_state: BTreeSet<String> = request
        .available_state_fields
        .iter()
        .map(|field| field.trim().to_string())
        .filter(|field| !field.is_empty())
        .collect();
    if requested_state != catalog_state {
        return Err(AppError::Invalid(
            "preflight state inventory does not match the exact simulator catalog".into(),
        ));
    }
    let identity = executable_identity(&binary, &output.stdout)?;
    let mut report = evaluate_preflight(&request, &catalog, identity)?;
    if request.source.kind == "exact_document" {
        let native_binary = PathBuf::from(&report.executable.canonical_path);
        let require_context = catalog["runtimeCapabilities"]["interactionPolicy"]["version"] == 14;
        let (native, context) = validate_exact_document_with_context(
            &native_binary, &report.executable.binary_fingerprint, &request.source, require_context)?;
        let catalog_after = Command::new(&native_binary)
            .arg("--scheme-catalog=json")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|error| AppError::Subprocess(format!(
                "requery {} after exact preflight: {error}", native_binary.display())))?;
        if !catalog_after.status.success() || catalog_after.stdout != output.stdout ||
            crate::platform::sha256_file(&native_binary)? != report.executable.binary_fingerprint {
            return Err(AppError::Invalid(
                "simulator catalog or executable changed during exact preflight".into()));
        }
        if require_context {
            apply_native_context_validation(&mut report, native, context, &catalog)?;
        } else {
            apply_native_validation(&mut report, native);
        }
    }
    Ok(report)
}

fn preflight_remote_config(
    target: super::remote_compute::RemoteComputeTarget,
    request: ConfigPreflightRequest,
) -> AppResult<ConfigPreflightReport> {
    let probe = super::remote_compute::query_remote_compute(target.clone())?;
    let catalog = parse_catalog(probe.catalog_text.as_bytes())?;
    let requested_state: BTreeSet<String> = request.available_state_fields.iter()
        .map(|field| field.trim().to_string()).filter(|field| !field.is_empty()).collect();
    if requested_state != catalog_runtime_state_fields(&catalog)? {
        return Err(AppError::Invalid("preflight state inventory does not match the exact remote simulator catalog".into()));
    }
    let identity = ExecutableIdentity {
        canonical_path: probe.binary_path.clone(), byte_length: probe.byte_length,
        modified_unix_ms: probe.modified_unix_ms, catalog_fingerprint: probe.catalog_fingerprint.clone(),
        binary_fingerprint: probe.binary_fingerprint.clone(),
    };
    let mut report = evaluate_preflight(&request, &catalog, identity)?;
    if request.source.kind == "exact_document" {
        let text = request.source.yaml_text.as_deref().ok_or_else(|| AppError::Invalid("remote exact source has no YAML".into()))?;
        let stage = super::remote_compute::stage_source(target, text)?;
        if stage.probe.binary_fingerprint != probe.binary_fingerprint || stage.probe.catalog_fingerprint != probe.catalog_fingerprint {
            return Err(AppError::Invalid("remote executable/catalog changed during exact preflight staging".into()));
        }
        let require_context = catalog["runtimeCapabilities"]["interactionPolicy"]["version"] == 14;
        let validation = stage.validate_source(require_context)?;
        report.execution_target_fingerprint = Some(stage.execution_target_fingerprint.clone());
        let accepted = validation.exit_code == 0 && validation.stdout.contains("[VALIDATE] PASS");
        let context = if require_context && accepted {
            Some(parse_native_validate_only_context(&validation.stdout, &stage.source_sha256)?)
        } else { None };
        let source_path = request.source.yaml_path.as_deref().unwrap_or_default();
        let native = NativeValidationOutcome {
            accepted, source_sha256: stage.source_sha256,
            source_fingerprint: format!("fnv1a64:{:016x}", fnv1a64(text.as_bytes())),
            source_path_fingerprint: format!("fnv1a64:{:016x}", fnv1a64(source_path.as_bytes())),
            binary_sha256: probe.binary_fingerprint, exit_code: Some(validation.exit_code),
            diagnostic: if accepted { String::new() } else { format!("{}\n{}", validation.stderr, validation.stdout).chars().take(2048).collect() },
        };
        if require_context { apply_native_context_validation(&mut report, native, context, &catalog)?; }
        else { apply_native_validation(&mut report, native); }
    }
    Ok(report)
}

fn validate_request(request: &ConfigPreflightRequest) -> AppResult<()> {
    if request.schema != REQUEST_SCHEMA || request.request_version != 10 {
        return Err(AppError::Invalid(format!(
            "unsupported configuration preflight request '{}'/{}",
            request.schema, request.request_version
        )));
    }
    match request.source.kind.as_str() {
        "descriptor_only" if request.source.yaml_text.is_none()
            && request.source.yaml_path.is_none() => {}
        "exact_document" if request.source.yaml_text.as_ref().is_some_and(|text| {
                !text.trim().is_empty() && text.len() <= 8 * 1024 * 1024
            }) && request.source.yaml_path.as_ref().is_some_and(|path| {
                !path.trim().is_empty() && Path::new(path).parent().is_some_and(Path::is_dir)
            }) => {}
        _ => return Err(AppError::Invalid(
            "preflight source must be descriptor_only or a bounded exact_document with a valid YAML path".into(),
        )),
    }
    for (name, value) in [
        ("backend", request.backend.as_str()),
        ("executionMode", request.execution_mode.as_str()),
        ("outputFormat", request.output_format.as_str()),
        ("coordinateSystem", request.coordinate_system.as_str()),
        ("staggering", request.staggering.as_str()),
        (
            "validationGuardMode",
            request.validation_guard_mode.as_str(),
        ),
        (
            "initialization.soundingSource",
            request.initialization.sounding_source.as_str(),
        ),
        (
            "initialization.hodographSource",
            request.initialization.hodograph_source.as_str(),
        ),
        (
            "initialization.triggerSource",
            request.initialization.trigger_source.as_str(),
        ),
        ("boundary.west", request.boundary.west.as_str()),
        ("boundary.east", request.boundary.east.as_str()),
        ("boundary.south", request.boundary.south.as_str()),
        ("boundary.north", request.boundary.north.as_str()),
        ("boundary.bottom", request.boundary.bottom.as_str()),
        ("boundary.top", request.boundary.top.as_str()),
        (
            "interactions.turbulenceMode",
            request.interactions.turbulence_mode.as_str(),
        ),
        (
            "interactions.surfaceScheme",
            request.interactions.surface_scheme.as_str(),
        ),
        (
            "interactions.diffusionTarget",
            request.interactions.diffusion_target.as_str(),
        ),
    ] {
        if normalize(value).is_empty() {
            return Err(AppError::Invalid(format!(
                "preflight {name} must not be empty"
            )));
        }
    }
    if !matches!(
        normalize(&request.execution_mode).as_str(),
        "exploratory" | "qualified"
    ) {
        return Err(AppError::Invalid(
            "preflight executionMode must be exploratory or qualified".into(),
        ));
    }
    if request.selections.is_empty() {
        return Err(AppError::Invalid(
            "preflight must contain at least one scheme selection".into(),
        ));
    }
    if request.grid_metrics.nx < 1
        || request.grid_metrics.ny < 1
        || request.grid_metrics.nz < 1
        || !request.grid_metrics.dx.is_finite()
        || request.grid_metrics.dx <= 0.0
        || !request.grid_metrics.dy.is_finite()
        || request.grid_metrics.dy <= 0.0
        || !request.grid_metrics.dz.is_finite()
        || request.grid_metrics.dz <= 0.0
        || !request.grid_metrics.ztop.is_finite()
        || request.grid_metrics.ztop <= 0.0
        || !request.time_control.dt_large.is_finite()
        || request.time_control.dt_large <= 0.0
        || request.time_control.acoustic_substeps < 0
        || !request.time_control.output_interval.is_finite()
        || request.time_control.output_interval < 0.0
    {
        return Err(AppError::Invalid(
            "preflight grid/time envelope must be finite and physically valid".into(),
        ));
    }
    if request.selections.iter().any(|selection| {
        normalize(&selection.module_id).is_empty() || normalize(&selection.scheme_id).is_empty()
    }) {
        return Err(AppError::Invalid(
            "preflight module and scheme ids must not be empty".into(),
        ));
    }
    Ok(())
}

fn resolve_binary(
    state: &State<'_, AppState>,
    override_path: Option<String>,
) -> AppResult<PathBuf> {
    if let Some(path) = override_path
        .as_deref()
        .map(str::trim)
        .filter(|path| !path.is_empty())
    {
        let path = PathBuf::from(expand_tilde(path));
        if !path.is_file() {
            return Err(AppError::NotFound(path.display().to_string()));
        }
        return Ok(path);
    }
    if let Some(path) = state
        .simulator_binary
        .lock()
        .map_err(|error| AppError::Subprocess(format!("sim lock poisoned: {error}")))?
        .clone()
    {
        if path.is_file() {
            return Ok(path);
        }
    }
    crate::commands::status::locate_simulator().ok_or_else(|| {
        AppError::Invalid("simulator binary not configured; set the path in Settings".into())
    })
}

fn expand_tilde(path: &str) -> String {
    crate::platform::expand_tilde(path).display().to_string()
}

fn executable_identity(binary: &Path, catalog_bytes: &[u8]) -> AppResult<ExecutableIdentity> {
    let canonical = fs::canonicalize(binary).map_err(AppError::Io)?;
    let metadata = fs::metadata(&canonical).map_err(AppError::Io)?;
    let modified_unix_ms = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis());
    Ok(ExecutableIdentity {
        canonical_path: canonical.display().to_string(),
        byte_length: metadata.len(),
        modified_unix_ms,
        catalog_fingerprint: format!("fnv1a64:{:016x}", fnv1a64(catalog_bytes)),
        binary_fingerprint: crate::platform::sha256_file(&canonical)?,
    })
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn parse_catalog(bytes: &[u8]) -> AppResult<Value> {
    let catalog: Value = serde_json::from_slice(bytes).map_err(|error| {
        AppError::Subprocess(format!(
            "invalid JSON from simulator scheme catalog: {error}"
        ))
    })?;
    if catalog.get("schema").and_then(Value::as_str) != Some("nimbus.scheme-catalog.v9")
        || catalog.get("catalogVersion").and_then(Value::as_u64) != Some(9)
        || !catalog.get("modules").is_some_and(Value::is_array)
    {
        return Err(AppError::Subprocess(
            "unsupported or malformed simulator scheme catalog".into(),
        ));
    }
    catalog_runtime_state_fields(&catalog)?;
    catalog_output_format_capabilities(&catalog)?;
    catalog_interaction_policy(&catalog)?;
    validate_readiness_contract(&catalog)?;
    validate_scheme_readiness(&catalog)?;
    validate_studio_selection_policies(&catalog)?;
    Ok(catalog)
}

fn validate_studio_selection_policies(catalog: &Value) -> AppResult<()> {
    let modules = catalog
        .get("modules")
        .and_then(Value::as_array)
        .ok_or_else(|| AppError::Subprocess("scheme catalog modules are malformed".into()))?;
    for module in modules {
        let policy = module
            .get("studioSelection")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                AppError::Subprocess("scheme catalog module has no Studio selection policy".into())
            })?;
        let mode = policy.get("mode").and_then(Value::as_str);
        let fixed = policy.get("fixedSchemeId").and_then(Value::as_str);
        let reason = policy.get("reason").and_then(Value::as_str);
        let valid = match (mode, fixed, reason) {
            (Some("form"), Some(""), Some("")) => true,
            (Some("product_set_v1"), Some(""), Some(reason)) if !reason.trim().is_empty() => true,
            (Some("fixed_v1"), Some(fixed), Some(reason))
                if !fixed.is_empty() && !reason.trim().is_empty() =>
            {
                module
                    .get("schemes")
                    .and_then(Value::as_array)
                    .is_some_and(|schemes| {
                        schemes.iter().any(|scheme| {
                            scheme.get("id").and_then(Value::as_str) == Some(fixed)
                                && scheme.get("constructible").and_then(Value::as_bool)
                                    == Some(true)
                        })
                    })
            }
            (Some("unavailable_v1"), Some(""), Some(reason)) if !reason.trim().is_empty() => true,
            _ => false,
        };
        if !valid {
            return Err(AppError::Subprocess(
                "scheme catalog module has an invalid Studio selection policy".into(),
            ));
        }
    }
    Ok(())
}

fn catalog_output_format_capabilities(
    catalog: &Value,
) -> AppResult<BTreeMap<String, (bool, String)>> {
    let values = catalog
        .get("runtimeCapabilities")
        .and_then(Value::as_object)
        .and_then(|capabilities| capabilities.get("outputFormats"))
        .and_then(Value::as_array)
        .ok_or_else(|| {
            AppError::Subprocess(
                "simulator scheme catalog has no output-format capability inventory".into(),
            )
        })?;
    let mut formats = BTreeMap::new();
    for value in values {
        let id = value
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty() && *id == id.trim() && *id == id.to_ascii_lowercase())
            .ok_or_else(|| {
                AppError::Subprocess(
                    "simulator scheme catalog has a malformed output-format capability".into(),
                )
            })?;
        let available = value.get("available").and_then(Value::as_bool);
        let reason = value.get("reason").and_then(Value::as_str);
        let implementation = value.get("implementation").and_then(Value::as_str);
        if available.is_none()
            || reason.is_none()
            || implementation.is_none()
            || (available == Some(false) && reason.is_some_and(|entry| entry.trim().is_empty()))
            || (available == Some(true)
                && implementation.is_some_and(|entry| entry.trim().is_empty()))
            || formats
                .insert(
                    id.to_string(),
                    (
                        available.unwrap_or(false),
                        reason.unwrap_or_default().to_string(),
                    ),
                )
                .is_some()
        {
            return Err(AppError::Subprocess(
                "simulator scheme catalog has a malformed output-format capability".into(),
            ));
        }
    }
    if !formats.contains_key("npy_3d") || !formats.contains_key("netcdf") {
        return Err(AppError::Subprocess(
            "simulator scheme catalog omits a required output-format capability".into(),
        ));
    }
    Ok(formats)
}

fn catalog_runtime_state_fields(catalog: &Value) -> AppResult<BTreeSet<String>> {
    let values = catalog
        .get("runtimeStateFields")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            AppError::Subprocess(
                "simulator scheme catalog has no authoritative runtime state inventory".into(),
            )
        })?;
    let mut fields = BTreeSet::new();
    for value in values {
        let field = value
            .as_str()
            .map(str::trim)
            .filter(|field| !field.is_empty())
            .ok_or_else(|| {
                AppError::Subprocess(
                    "simulator scheme catalog has a malformed runtime state inventory".into(),
                )
            })?;
        if !fields.insert(field.to_string()) {
            return Err(AppError::Subprocess(
                "simulator scheme catalog has duplicate runtime state fields".into(),
            ));
        }
    }
    if fields.is_empty() {
        return Err(AppError::Subprocess(
            "simulator scheme catalog has an empty runtime state inventory".into(),
        ));
    }
    Ok(fields)
}

fn catalog_interaction_policy(catalog: &Value) -> AppResult<CatalogInteractionPolicy> {
    let raw = catalog
        .get("runtimeCapabilities")
        .and_then(|capabilities| capabilities.get("interactionPolicy"))
        .cloned()
        .ok_or_else(|| {
            AppError::Subprocess("simulator scheme catalog has no interaction policy".into())
        })?;
    let radar_scope_present = raw.as_object().is_some_and(|object|
        object.contains_key("qualifiedZfpReversibleRadarV4"));
    let vertical_scope_present = raw.as_object().is_some_and(|object|
        object.contains_key("qualifiedVerticalOnlySgs"));
    let dynamic_default_present = raw.as_object().is_some_and(|object|
        object.contains_key("turbulenceDynamicCoefficientDefault"));
    let milbrandt_triple_present = raw.as_object().is_some_and(|object|
        object.contains_key("milbrandtTripleMoment"));
    let policy: CatalogInteractionPolicy = serde_json::from_value(raw).map_err(|error| {
        AppError::Subprocess(format!(
            "simulator scheme catalog has a malformed interaction policy: {error}"
        ))
    })?;
    let valid_list = |values: &[String]| {
        !values.is_empty()
            && values
                .iter()
                .all(|entry| !entry.is_empty() && entry == &normalize(entry))
            && values.iter().collect::<BTreeSet<_>>().len() == values.len()
    };
    let lists = [
        policy.horizontal_only_turbulence_modes.as_slice(),
        policy.known_surface_schemes.as_slice(),
        policy.terrain_radiation_schemes.as_slice(),
        policy.terrain_boundary_layer_schemes.as_slice(),
        policy.terrain_turbulence_schemes.as_slice(),
        policy.terrain_chaos_schemes.as_slice(),
        policy.terrain_active_diffusion_schemes.as_slice(),
        policy.terrain_active_diffusion_targets.as_slice(),
        policy.qualified_zfp_terrain_schemes.as_slice(),
        policy.qualified_zfp_preset_ids.as_slice(),
        policy.lateral_boundary_modes.as_slice(),
        policy.radiative_lateral_staggerings.as_slice(),
        policy.bottom_boundary_modes.as_slice(),
        policy.top_boundary_modes.as_slice(),
        policy.gpu_apply_microphysics_schemes.as_slice(),
        policy.gpu_apply_sgs_schemes.as_slice(),
        policy.gpu_apply_slow_dynamics_schemes.as_slice(),
        policy
            .qualified_gpu_apply_slow_dynamics_schemes
            .as_slice(),
        policy
            .gpu_apply_microphysics_dispatch_guard_modes
            .as_slice(),
        policy
            .gpu_apply_microphysics_host_commit_guard_modes
            .as_slice(),
        policy.gpu_apply_slow_dispatch_guard_modes.as_slice(),
        policy.gpu_apply_slow_host_commit_guard_modes.as_slice(),
        policy.gpu_apply_sgs_dispatch_guard_modes.as_slice(),
    ];
    let canonical_chaos_ids = policy
        .chaos_canonical_targets
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    let terrain_chaos_ids = policy
        .terrain_chaos_schemes
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let metal_scope = &policy.generated_metal_canonical_scope;
    let valid_sha = |value: &str| {
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    };
    let metal_formats = metal_scope
        .output_formats
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let east_extension = metal_scope.additional_boundary_npy_scope.as_ref();
    let valid_east_extension = east_extension.is_some_and(|scope| {
        scope.version == 1
            && scope.profile_count == 1
            && scope.family_id == "boundary:2b5bfbe6dfa466c6baf991e3"
            && scope.strict_canonical_yaml_sha256
                == "a244b58833f7e3cd7060d941e8453d4e5face3dff9e18e87ebf61dd581a24d55"
            && scope.output_formats == ["npy_3d"]
            && scope.write_cadence_seconds == [1]
            && scope.manifest_sha256
                == "26712e5d5714b313cc1e17f348d67b9db9eeea60e5b888d56b0b7cf3cc5673a9"
    });
    let radar_v4_scope = policy.qualified_zfp_reversible_radar_v4.as_ref();
    let valid_radar_v4_scope = radar_v4_scope.is_some_and(|scope| {
        scope.version == 1
            && scope.product_ids == ["reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv"]
            && scope.nonempty_product_subsets == 31
            && scope.core_fields == ["u", "v", "w", "theta", "p", "rho", "qv", "qr"]
            && scope.presets == ["minimal", "radar_ready", "radar_polarimetric_ready"]
            && scope.named_preset_product_sets.len() == 2
            && scope.named_preset_product_sets.get("radar_ready")
                == Some(&vec!["reflectivity_dbz".into(), "radial_velocity".into()])
            && scope.named_preset_product_sets.get("radar_polarimetric_ready")
                == Some(&vec!["zdr".into(), "kdp".into(), "rhohv".into()])
            && scope.polarimetric_extra_fields == ["qc"]
            && scope.output_format_tiers == ["zfp_low", "zfp_medium", "zfp_high"]
            && scope.backend == "cpu"
            && scope.coordinate_system == "cartesian"
            && scope.staggering == "c_grid"
            && scope.terrain == "none"
            && scope.domain_motion == "stationary"
            && !scope.nesting
            && scope.reviewed_source_profile_contract == "nimbus.qualified-radar-source-profile.v2"
            && scope.reviewed_source_profile_owner_symbol == "is_public_qualified_radar_profile"
            && scope.reviewed_source_profile_families
                == ["lin_cloudy_radar_ready_v1", "kessler_warm_cloud_radar_v1"]
            && scope.async_io_modes == [false, true]
            && scope.grid_shape == [12, 12, 48]
            && scope.grid_spacing_meters == [1000.0, 1000.0, 250.0]
            && scope.dt_large_seconds == 0.2
            && scope.acoustic_substeps == 6
            && scope.write_cadence_seconds == [1, 4]
            && scope.non_radar_mode == "accuracy"
            && scope.non_radar_container_version == 3
            && scope.non_radar_budget_owner == "qualifiedZfpFieldBudgets"
            && scope.radar_mode == "reversible"
            && scope.radar_container_version == 4
            && scope.radar_keyframe_every_export
            && scope.radar_tolerance_policy == "forbidden"
            && scope.manifest_contract == "nimbus.zfp.per_field.v2"
            && scope.source_field_bounds_contract == "registered_field_contract_v1"
            && scope.core_fields.iter().all(|field|
                policy.qualified_zfp_field_budgets.contains_key(field))
            && scope.polarimetric_extra_fields.iter().all(|field|
                policy.qualified_zfp_field_budgets.contains_key(field))
            && scope.product_ids.iter().all(|field|
                !policy.qualified_zfp_field_budgets.contains_key(field))
    });
    let valid_radar_policy = if policy.version == 11 || policy.version == 12 || policy.version == 13 || policy.version == 14 {
        radar_scope_present && valid_radar_v4_scope
    } else {
        !radar_scope_present && radar_v4_scope.is_none()
    };
    let vertical_scope = policy.qualified_vertical_only_sgs.as_ref();
    let valid_vertical_scope = vertical_scope.is_some_and(|scope| {
        scope.version == 1
            && scope.scheme == "smagorinsky"
            && scope.mode == "vertical_only"
            && !scope.dynamic_coefficient
            && scope.execution_mode == "qualified"
            && scope.backend == "cpu"
            && scope.coordinate_system == "cartesian"
            && scope.staggering == "c_grid"
            && scope.terrain == "none"
            && scope.domain_motion == "stationary"
            && !scope.nesting
            && scope.lateral_boundary_mode == "periodic"
            && !scope.active_boundary_layer_allowed
            && scope.grid_shape == [12, 12, 16]
            && scope.grid_spacing_meters == [1000.0, 1000.0, 250.0]
            && scope.dt_large_seconds == 0.2
            && scope.acoustic_substeps == 6
            && scope.output_format == "npy_3d"
            && scope.write_cadence_seconds == [1]
            && scope.source_profile_contract == "nimbus.static-vertical-sgs-cpu.v1"
            && scope.source_profile_owner_symbol
                == "is_transactional_static_vertical_only_cpu_candidate"
            && scope.whole_step_rollback == "state_and_six_science_ledgers"
            && !scope.dynamic_non_three_d_allowed
    });
    let valid_vertical_policy = if policy.version == 12 || policy.version == 13 || policy.version == 14 {
        vertical_scope_present && valid_vertical_scope && dynamic_default_present
            && policy.turbulence_dynamic_coefficient_default == Some(false)
    } else {
        !vertical_scope_present && vertical_scope.is_none()
            && !dynamic_default_present
            && policy.turbulence_dynamic_coefficient_default.is_none()
    };
    let valid_milbrandt_triple = if policy.version == 13 || policy.version == 14 {
        milbrandt_triple_present && policy.milbrandt_triple_moment.as_ref().is_some_and(|scope| {
            scope.version == 1
                && scope.retained
                && !scope.admitted
                && scope.implementation_state == "engineering_incomplete"
                && scope.selector == "microphysics.milbrandt.triple_moment"
                && !scope.strict_v1_explicit_key_admitted
                && scope.scheme == "milbrandt"
                && scope.baseline_state == ["q", "N"]
                && scope.additional_state == ["Z"]
                && scope.additional_moment_order == 6
                && scope.additional_moment_species == ["rain", "ice", "snow", "graupel", "hail"]
                && scope.private_storage_units == "m^6/kg_dry"
        })
    } else {
        !milbrandt_triple_present && policy.milbrandt_triple_moment.is_none()
    };
    let triple_readiness = catalog.get("readiness")
        .and_then(|readiness| readiness.get("items"))
        .and_then(Value::as_array)
        .map(|items| items.iter().filter(|item|
            item.get("id").and_then(Value::as_str)
                == Some("microphysics.milbrandt_triple_moment")).collect::<Vec<_>>())
        .unwrap_or_default();
    let valid_milbrandt_readiness = if policy.version == 13 || policy.version == 14 {
        triple_readiness.len() == 1
            && triple_readiness[0].get("state").and_then(Value::as_str)
                == Some("engineering_incomplete")
            && triple_readiness[0].get("blocker").and_then(Value::as_str)
                .is_some_and(|blocker| !blocker.trim().is_empty())
    } else {
        triple_readiness.is_empty()
    };
    let metal_scope_v2 = metal_scope.version == 2
        && policy.version == 9
        && metal_scope.tuple_count == 4032
        && metal_scope.distinct_config_count == 3024
        && metal_formats == ["npy_3d", "netcdf"]
        && metal_scope.npy_netcdf_manifest_sha256.is_none()
        && metal_scope.zfp_typed_output_manifest_sha256.is_none()
        && metal_scope.write_cadence_seconds_by_format.is_none()
        && east_extension.is_none();
    let metal_scope_v3 = metal_scope.version == 3
        && policy.version == 9
        && metal_scope.tuple_count == 6048
        && metal_scope.distinct_config_count == 4536
        && metal_formats == ["npy_3d", "netcdf", "zfp_high"]
        && metal_scope
            .npy_netcdf_manifest_sha256
            .as_deref()
            .is_some_and(valid_sha)
        && metal_scope
            .zfp_typed_output_manifest_sha256
            .as_deref()
            .is_some_and(valid_sha)
        && metal_scope
            .write_cadence_seconds_by_format
            .as_ref()
            .is_some_and(|cadences| {
                cadences.len() == 3
                    && cadences.get("npy_3d") == Some(&vec![1, 301])
                    && cadences.get("netcdf") == Some(&vec![1, 301])
                    && cadences.get("zfp_high") == Some(&vec![1])
            })
        && east_extension.is_none();
    let base_v4_hashes = metal_scope.base_manifest_sha256
        == "85642f094c418fe95eb4c2fa0ba5a2a947aec95f8f1c43b17abcd924b0f068a0"
        && metal_scope.npy_netcdf_manifest_sha256.as_deref()
            == Some("30d5fcc9494016894733ef973909849f8c58fbd04b3e0d6bb2e5e497c0c9ca99");
    let base_cadences = metal_scope.write_cadence_seconds_by_format.as_ref()
        .is_some_and(|cadences| cadences.get("npy_3d") == Some(&vec![1, 301])
            && cadences.get("netcdf") == Some(&vec![1, 301]));
    let metal_scope_no_zfp_east = metal_scope.version == 3
        && (policy.version == 10 || policy.version == 11 || policy.version == 12 || policy.version == 13 || policy.version == 14)
        && metal_scope.tuple_count == 4033
        && metal_scope.distinct_config_count == 3025
        && metal_formats == ["npy_3d", "netcdf"]
        && base_v4_hashes
        && base_cadences
        && metal_scope.zfp_typed_output_manifest_sha256.is_none()
        && metal_scope.write_cadence_seconds_by_format.as_ref()
            .is_some_and(|cadences| cadences.len() == 2)
        && metal_scope.manifest_sha256
            == "4604b059d425fb86690207f69000fd708f153054d739091e0ec2ba00c7d9a4e3"
        && valid_east_extension;
    let metal_scope_v4 = metal_scope.version == 4
        && (policy.version == 10 || policy.version == 11 || policy.version == 12 || policy.version == 13 || policy.version == 14)
        && metal_scope.tuple_count == 6049
        && metal_scope.distinct_config_count == 4537
        && metal_formats == ["npy_3d", "netcdf", "zfp_high"]
        && base_v4_hashes
        && base_cadences
        && metal_scope.zfp_typed_output_manifest_sha256.as_deref()
            == Some("89ef136c9637c6e23d17cd996af20631973286447b783f1978d1471d4982c9fc")
        && metal_scope.write_cadence_seconds_by_format.as_ref()
            .is_some_and(|cadences| cadences.len() == 3
                && cadences.get("zfp_high") == Some(&vec![1]))
        && metal_scope.manifest_sha256
            == "062b609452e090b0b6e5ff1f7bb15cab896700f193796ff4d87a600994f28967"
        && valid_east_extension;
    let valid_mixed_policy = if policy.version == 14 {
        expected_flat_mixed_metal_scope().as_ref().is_some_and(|expected|
            policy.qualified_flat_mixed_metal_npy_scope.as_ref() == Some(expected))
    } else {
        policy.qualified_flat_mixed_metal_npy_scope.is_none()
    };
    if !valid_radar_policy || !valid_vertical_policy
        || !valid_milbrandt_triple || !valid_milbrandt_readiness || !valid_mixed_policy
        || !(metal_scope_v2 || metal_scope_v3 || metal_scope_no_zfp_east || metal_scope_v4)
        || metal_scope.backend != "metal"
        || metal_scope.execution_mode != "qualified"
        || metal_scope.tuple_count == 0
        || metal_scope.distinct_config_count == 0
        || metal_scope.distinct_config_count > metal_scope.tuple_count
        || !valid_list(&metal_scope.output_formats)
        || !metal_scope.output_formats.contains(&"npy_3d".to_string())
        || !metal_scope.output_formats.contains(&"netcdf".to_string())
        || metal_scope.scalar_transport_owner != "native_conservative_mc"
        || !valid_sha(&metal_scope.base_manifest_sha256)
        || !valid_sha(&metal_scope.manifest_sha256)
        || policy.inactive_scheme_id.is_empty()
        || policy.inactive_scheme_id != normalize(&policy.inactive_scheme_id)
        || policy.active_surface_scheme_id.is_empty()
        || policy.active_surface_scheme_id != normalize(&policy.active_surface_scheme_id)
        || policy.terrain_longwave_radiation_scheme_id.is_empty()
        || policy.terrain_longwave_radiation_scheme_id
            != normalize(&policy.terrain_longwave_radiation_scheme_id)
        || policy.qualified_zfp_mode.is_empty()
        || policy.qualified_zfp_mode != normalize(&policy.qualified_zfp_mode)
        || canonical_chaos_ids != terrain_chaos_ids
        || policy.chaos_canonical_targets.values().any(|targets| {
            targets.native_generation_required
                || !targets.initial_conditions.is_empty()
                    && !valid_list(&targets.initial_conditions)
                || !targets.tendencies.is_empty() && !valid_list(&targets.tendencies)
        })
        || !policy.qualified_zfp_max_sparse_threshold.is_finite()
        || policy.qualified_zfp_max_sparse_threshold < 0.0
        || policy.qualified_zfp_field_budgets.is_empty()
        || policy.qualified_zfp_field_units.len() != policy.qualified_zfp_field_budgets.len()
        || policy.qualified_zfp_preset_fields.len() != policy.qualified_zfp_preset_ids.len()
        || policy.qualified_zfp_field_execution_policies
            != vec!["serial".to_string(), "field_parallel".to_string()]
        || policy.qualified_zfp_max_field_workers < 1
        || policy.qualified_zfp_min_field_scratch_mib < 1
        || policy.qualified_zfp_max_field_scratch_mib < policy.qualified_zfp_min_field_scratch_mib
        || policy.qualified_zfp_minimum_keep_last_frames <= 0
        || policy.qualified_zfp_minimum_free_space_mib <= 0
        || !policy.qualified_zfp_requires_retained_file_ceiling
        || policy.qualified_execution_requires_maturity_acknowledgment
        || policy
            .qualified_zfp_field_budgets
            .iter()
            .any(|(field, budget)| {
                field.is_empty()
                    || field != &normalize(field)
                    || !budget.is_finite()
                    || *budget <= 0.0
            })
        || policy.qualified_zfp_field_units.iter().any(|(field, units)| {
            !policy.qualified_zfp_field_budgets.contains_key(field) || units.trim().is_empty()
        })
        || policy.qualified_zfp_preset_ids.iter().any(|preset| {
            match policy.qualified_zfp_preset_fields.get(preset) {
                None => true,
                Some(fields) => {
                    fields.is_empty()
                        || !valid_list(fields)
                        || fields.iter().any(|field| {
                            !policy.qualified_zfp_field_budgets.contains_key(field)
                        })
                }
            }
        })
        || policy.qualified_zfp_preset_fields.keys().any(|preset| {
            !policy.qualified_zfp_preset_ids.contains(preset)
        })
        || policy.legacy_open_cgrid_resolution.is_empty()
        || policy.legacy_open_cgrid_resolution != normalize(&policy.legacy_open_cgrid_resolution)
        || policy.legacy_open_collocated_resolution.is_empty()
        || policy.legacy_open_collocated_resolution
            != normalize(&policy.legacy_open_collocated_resolution)
        || !policy
            .lateral_boundary_modes
            .contains(&policy.legacy_open_cgrid_resolution)
        || !policy
            .lateral_boundary_modes
            .contains(&policy.legacy_open_collocated_resolution)
        || !policy
            .radiative_lateral_staggerings
            .contains(&"c_grid".to_string())
        || !policy.periodic_lateral_pairs_required
        || policy.scientific_diagnostics_omitted_sentinel != -1
        || policy.scientific_diagnostics_minimum_cadence != 0
        || !policy.gpu_apply_controls_no_op_on_cpu
        || lists.iter().any(|values| !valid_list(values))
        || !policy
            .known_surface_schemes
            .contains(&policy.inactive_scheme_id)
        || !policy
            .known_surface_schemes
            .contains(&policy.active_surface_scheme_id)
        || !policy
            .terrain_radiation_schemes
            .contains(&policy.terrain_longwave_radiation_scheme_id)
    {
        return Err(AppError::Subprocess(
            "simulator scheme catalog has an inconsistent interaction policy".into(),
        ));
    }
    Ok(policy)
}

fn evaluate_preflight(
    request: &ConfigPreflightRequest,
    catalog: &Value,
    executable: ExecutableIdentity,
) -> AppResult<ConfigPreflightReport> {
    let backend = normalize_backend(&request.backend);
    let execution_mode = normalize(&request.execution_mode);
    let validation_guard_mode = normalize(&request.validation_guard_mode);
    let output_format = normalize(&request.output_format);
    let mut zfp_field_tolerances = BTreeMap::new();
    for (field, tolerance) in &request.output.zfp_field_tolerances {
        let canonical = normalize(field);
        if canonical.is_empty() || zfp_field_tolerances.insert(canonical, *tolerance).is_some() {
            return Err(AppError::Subprocess(
                "configuration preflight contains an empty or duplicate ZFP field tolerance".into(),
            ));
        }
    }
    let output = OutputInteractionValues {
        preset: normalize(&request.output.preset),
        async_io: request.output.async_io,
        zfp_mode: normalize(&request.output.zfp_mode),
        zfp_per_field_tolerances: request.output.zfp_per_field_tolerances,
        zfp_sparse_threshold: request.output.zfp_sparse_threshold,
        zfp_float16_prefilter: request.output.zfp_float16_prefilter,
        zfp_field_execution: normalize(&request.output.zfp_field_execution),
        zfp_field_workers: request.output.zfp_field_workers,
        zfp_field_scratch_mib: request.output.zfp_field_scratch_mib,
        zfp_field_tolerances,
        keep_last_frames: request.output.keep_last_frames,
        minimum_free_space_mib: request.output.minimum_free_space_mib,
        maximum_retained_history_files: request.output.maximum_retained_history_files,
    };
    let coordinate_system = normalize(&request.coordinate_system);
    let staggering = normalize(&request.staggering);
    let boundary = BoundarySelection {
        west: normalize(&request.boundary.west),
        east: normalize(&request.boundary.east),
        south: normalize(&request.boundary.south),
        north: normalize(&request.boundary.north),
        bottom: normalize(&request.boundary.bottom),
        top: normalize(&request.boundary.top),
    };
    let diagnostics = request.diagnostics.clone();
    let initialization = InitializationSelection {
        sounding_source: normalize_sounding(&request.initialization.sounding_source),
        hodograph_source: normalize_hodograph(&request.initialization.hodograph_source),
        trigger_source: normalize_trigger(&request.initialization.trigger_source),
    };
    let interactions = ScienceInteractionValues {
        radiation_shortwave_enabled: request.interactions.radiation_shortwave_enabled,
        turbulence_mode: normalize(&request.interactions.turbulence_mode),
        turbulence_dynamic_coefficient: request.interactions.turbulence_dynamic_coefficient,
        surface_scheme: normalize(&request.interactions.surface_scheme),
        surface_apply_fluxes: request.interactions.surface_apply_fluxes,
        diffusion_target: normalize(&request.interactions.diffusion_target),
        diffusion_active: request.interactions.diffusion_active,
    };
    let interaction_policy = catalog_interaction_policy(catalog)?;
    let available_state: BTreeSet<String> = request
        .available_state_fields
        .iter()
        .map(|field| field.trim().to_string())
        .filter(|field| !field.is_empty())
        .collect();
    let modules = catalog
        .get("modules")
        .and_then(Value::as_array)
        .ok_or_else(|| AppError::Subprocess("scheme catalog modules are malformed".into()))?;
    let module_map: BTreeMap<&str, &Value> = modules
        .iter()
        .filter_map(|module| {
            module
                .get("id")
                .and_then(Value::as_str)
                .map(|id| (id, module))
        })
        .collect();

    let mut hard_errors = Vec::new();
    let mut warnings = Vec::new();
    let mut resolved_selections = Vec::new();
    let mut selected_modules = BTreeSet::new();

    if execution_mode == "qualified" && !request.maturity_limits_acknowledged {
        hard_errors.push(issue(
            "explicit_false_maturity_acknowledgment",
            None,
            None,
            "Qualified execution rejects an explicit false legacy maturity acknowledgment; omit the obsolete key".into(),
            vec!["omitted".into()],
            Some("false".into()),
        ));
    } else if execution_mode == "exploratory" && request.maturity_limits_acknowledged {
        hard_errors.push(issue(
            "inactive_maturity_acknowledgment",
            None,
            None,
            "Scheme maturity acknowledgment is active only for qualified execution".into(),
            vec!["false".into()],
            Some("true".into()),
        ));
    }
    if execution_mode == "qualified" && request.allow_backend_fallback {
        hard_errors.push(issue(
            "qualified_backend_fallback_forbidden",
            None,
            None,
            "Qualified execution cannot permit backend fallback".into(),
            vec!["false".into()],
            Some("true".into()),
        ));
    }
    if execution_mode == "qualified" && validation_guard_mode != "error" {
        hard_errors.push(issue(
            "qualified_validation_guard_required",
            None,
            None,
            "Qualified execution requires validation.guard_mode=error".into(),
            vec!["error".into()],
            Some(validation_guard_mode.clone()),
        ));
    }

    let output_capabilities = catalog_output_format_capabilities(catalog)?;
    match output_capabilities.get(&output_format) {
        None => hard_errors.push(issue(
            "unknown_output_format",
            None,
            None,
            format!("the exact simulator does not report output format '{output_format}'"),
            output_capabilities.keys().cloned().collect(),
            Some(output_format.clone()),
        )),
        Some((false, reason)) => hard_errors.push(issue(
            "output_format_unavailable",
            None,
            None,
            format!("output format '{output_format}' is unavailable: {reason}"),
            output_capabilities
                .iter()
                .filter_map(|(id, (available, _))| available.then_some(id.clone()))
                .collect(),
            Some(output_format.clone()),
        )),
        Some((true, _)) => {}
    }
    let qualified_zfp = execution_mode == "qualified"
        && matches!(output_format.as_str(), "zfp" | "zfp_low" | "zfp_medium" | "zfp_high");
    let radar_products: Vec<String> = request.selections.iter()
        .filter(|selection| normalize(&selection.module_id) == "radar")
        .map(|selection| normalize(&selection.scheme_id))
        .collect();
    let radar_v4_candidate = qualified_zfp && !radar_products.is_empty()
        && interaction_policy.qualified_zfp_reversible_radar_v4.as_ref()
            .is_some_and(|scope| scope.admitted);
    evaluate_output_interactions(
        &output_format,
        &output,
        &interaction_policy,
        execution_mode == "qualified",
        radar_v4_candidate,
        &mut hard_errors,
    );
    if qualified_zfp && !radar_products.is_empty() {
        if radar_v4_candidate {
            let scope = interaction_policy.qualified_zfp_reversible_radar_v4.as_ref().unwrap();
            let selected: BTreeSet<String> = radar_products.iter().cloned().collect();
            let preset_matches = if output.preset == "minimal" {
                true
            } else {
                scope.named_preset_product_sets.get(&output.preset)
                    .is_some_and(|products| products.iter().cloned().collect::<BTreeSet<_>>() == selected)
            };
            let exact_envelope = radar_products.len() == selected.len()
                && selected.iter().all(|product| scope.product_ids.contains(product))
                && preset_matches
                && scope.output_format_tiers.contains(&output_format)
                && backend == scope.backend
                && !request.allow_backend_fallback
                && coordinate_system == scope.coordinate_system
                && staggering == scope.staggering
                && [request.grid_metrics.nx, request.grid_metrics.ny, request.grid_metrics.nz]
                    == scope.grid_shape.as_slice()
                && [request.grid_metrics.dx, request.grid_metrics.dy, request.grid_metrics.dz]
                    == scope.grid_spacing_meters.as_slice()
                && request.grid_metrics.ztop == 12000.0
                && request.time_control.dt_large == scope.dt_large_seconds
                && request.time_control.acoustic_substeps == scope.acoustic_substeps
                && scope.write_cadence_seconds.iter().any(|cadence|
                    request.time_control.output_interval == *cadence as f64)
                && diagnostics.deterministic_reductions
                && output.zfp_mode == scope.non_radar_mode
                && output.zfp_per_field_tolerances
                && output.zfp_sparse_threshold == 0.0
                && !output.zfp_float16_prefilter
                && scope.async_io_modes.contains(&output.async_io)
                && output.zfp_field_tolerances.keys().all(|field|
                    !scope.product_ids.contains(field));
            if !exact_envelope {
                hard_errors.push(issue(
                    "qualified_zfp_radar_scope_incompatible",
                    Some("radar"),
                    None,
                    "selected radar ZFP descriptor is outside the exact reversible-v4 catalog envelope".into(),
                    vec!["reviewed flat CPU radar-v4 source and output profile".into()],
                    Some(format!("{} / {}", output_format, output.preset)),
                ));
            }
            let (code, message) = if request.source.kind == "exact_document" {
                ("qualified_zfp_radar_native_admission_required",
                 "radar-v4 selection requires acceptance of this exact YAML by the selected simulator")
            } else {
                ("qualified_zfp_radar_exact_document_required",
                 "descriptor-only preflight cannot establish radar-v4 executable admission")
            };
            hard_errors.push(issue(
                code, Some("radar"), None, message.into(),
                vec!["exact_document with native --validate-only acceptance".into()],
                Some(request.source.kind.clone()),
            ));
        } else {
            for product in &radar_products {
                hard_errors.push(issue(
                    "qualified_zfp_radar_product_not_ready",
                    Some("radar"), Some(product),
                    format!(
                        "qualified radar ZFP product '{}' remains closed by this simulator's public gate",
                        product
                    ),
                    vec!["npy_3d".into()], Some(output_format.clone()),
                ));
            }
        }
    }
    evaluate_boundary_and_diagnostic_interactions(
        &staggering,
        &boundary,
        &diagnostics,
        &interaction_policy,
        &mut hard_errors,
    );
    for selection in &request.selections {
        let module_id = normalize(&selection.module_id);
        let requested_scheme_id = normalize(&selection.scheme_id);
        let Some(module) = module_map.get(module_id.as_str()) else {
            hard_errors.push(issue(
                "unknown_module",
                Some(&module_id),
                Some(&requested_scheme_id),
                format!("the exact simulator does not report module '{module_id}'"),
                module_map.keys().map(|id| (*id).to_string()).collect(),
                Some(module_id.clone()),
            ));
            continue;
        };
        let product_set = module
            .get("studioSelection")
            .and_then(|policy| policy.get("mode"))
            .and_then(Value::as_str)
            == Some("product_set_v1");
        if !selected_modules.insert(module_id.clone()) && !product_set {
            hard_errors.push(issue(
                "duplicate_module_selection",
                Some(&module_id),
                Some(&requested_scheme_id),
                format!("module '{module_id}' was selected more than once"),
                Vec::new(),
                None,
            ));
            continue;
        }
        let schemes = module
            .get("schemes")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                AppError::Subprocess(format!(
                    "scheme catalog module '{module_id}' has malformed schemes"
                ))
            })?;
        let Some(scheme) = find_scheme(schemes, &requested_scheme_id) else {
            hard_errors.push(issue(
                "unknown_scheme",
                Some(&module_id),
                Some(&requested_scheme_id),
                format!(
                    "the exact simulator does not report scheme '{requested_scheme_id}' in module '{module_id}'"
                ),
                schemes.iter().filter_map(|value| string(value, "id")).map(str::to_owned).collect(),
                Some(requested_scheme_id.clone()),
            ));
            continue;
        };
        evaluate_scheme(
            &module_id,
            &selection.scheme_id,
            scheme,
            execution_mode == "qualified",
            &backend,
            request.allow_backend_fallback,
            &coordinate_system,
            &staggering,
            &initialization,
            &available_state,
            &mut resolved_selections,
            &mut hard_errors,
            &mut warnings,
        )?;
    }

    let selected_schemes: BTreeMap<String, String> = resolved_selections
        .iter()
        .map(|selection| {
            (
                selection.module_id.clone(),
                selection.canonical_scheme_id.clone(),
            )
        })
        .collect();
    let generated_scope = &interaction_policy.generated_metal_canonical_scope;
    let generated_metal_common = matches!(generated_scope.version, 3 | 4)
        && backend == "metal"
        && execution_mode == "qualified"
        && coordinate_system == "cartesian"
        && staggering == "c_grid"
        && request.grid_metrics.nx == 18
        && request.grid_metrics.ny == 18
        && request.grid_metrics.nz == 18
        && selected_schemes.get("dynamics").map(String::as_str)
            == Some("klemp_wilhelmson_split_cartesian_cgrid")
        && selected_schemes.get("advection").map(String::as_str) == Some("tvd")
        && selected_schemes.get("microphysics").map(String::as_str) == Some("none")
        && selected_schemes.get("terrain").map(String::as_str) == Some("none")
        && selected_schemes.get("time_stepping").map(String::as_str)
            == Some("split_explicit");
    let generated_metal_base = generated_metal_common
        && [boundary.west.as_str(), boundary.east.as_str(),
            boundary.south.as_str(), boundary.north.as_str()] == ["rigid"; 4]
        && boundary.bottom == "rigid_slip"
        && boundary.top == "rigid"
        && generated_scope.output_formats.contains(&output_format);
    let generated_metal_east = generated_metal_common
        && boundary.west == "rigid_slip"
        && boundary.east == "radiative"
        && boundary.south == "rigid_slip"
        && boundary.north == "rigid_slip"
        && boundary.bottom == "rigid_noslip"
        && boundary.top == "open"
        && generated_scope.additional_boundary_npy_scope.is_some();
    let generated_metal_mixed = generated_metal_common
        && interaction_policy.version == 14
        && [boundary.west.as_str(), boundary.east.as_str(),
            boundary.south.as_str(), boundary.north.as_str()] == ["rigid_slip"; 4]
        && boundary.bottom == "rigid_noslip"
        && boundary.top == "open"
        && interaction_policy.qualified_flat_mixed_metal_npy_scope.is_some();
    let generated_metal_candidate = generated_metal_base || generated_metal_east || generated_metal_mixed;
    if generated_metal_candidate {
        if interaction_policy.version == 14 {
            let (code, message) = if request.source.kind == "exact_document" {
                ("qualified_generated_metal_native_admission_required",
                 "the exact generated Metal profile requires native validation of the original YAML controls")
            } else {
                ("qualified_generated_metal_exact_document_required",
                 "the exact generated Metal profile requires original YAML bytes and a source path")
            };
            hard_errors.push(issue(
                code, None, None, message.into(),
                vec!["exact_document with native --validate-only acceptance".into()],
                Some(request.source.kind.clone()),
            ));
        }
        let grid = &request.grid_metrics;
        if grid.dx != 500.0 || grid.dy != 500.0 || grid.dz != 250.0 || grid.ztop != 4500.0 {
            hard_errors.push(issue(
                "generated_metal_grid_envelope",
                None,
                None,
                "generated Metal canonical grid spacing/top height is outside the exact reviewed envelope".into(),
                vec!["dx=500,dy=500,dz=250,ztop=4500".into()],
                Some(format!("dx={},dy={},dz={},ztop={}", grid.dx, grid.dy, grid.dz, grid.ztop)),
            ));
        }
        let time = &request.time_control;
        if time.dt_large != 0.1 || time.acoustic_substeps != 6 {
            hard_errors.push(issue(
                "generated_metal_time_envelope",
                None,
                None,
                "generated Metal canonical timestep/acoustic substeps are outside the exact reviewed envelope".into(),
                vec!["dt_large=0.1,acoustic_substeps=6".into()],
                Some(format!("dt_large={},acoustic_substeps={}", time.dt_large, time.acoustic_substeps)),
            ));
        }
        if generated_metal_east {
            let east = generated_scope.additional_boundary_npy_scope.as_ref().unwrap();
            if !east.output_formats.contains(&output_format) {
                hard_errors.push(issue(
                    "generated_metal_east_output_format",
                    None, None,
                    "the east-radiative Metal extension admits only uncompressed NPY".into(),
                    east.output_formats.clone(), Some(output_format.clone()),
                ));
            }
            if output.preset != "minimal" {
                hard_errors.push(issue(
                    "generated_metal_east_output_preset",
                    None, None,
                    "the east-radiative Metal extension requires the minimal output preset".into(),
                    vec!["minimal".into()], Some(output.preset.clone()),
                ));
            }
            if output.async_io {
                hard_errors.push(issue(
                    "generated_metal_east_async_io",
                    None, None,
                    "the east-radiative Metal extension requires synchronous NPY publication".into(),
                    vec!["false".into()], Some("true".into()),
                ));
            }
        }
        if generated_metal_mixed {
            let mixed = interaction_policy.qualified_flat_mixed_metal_npy_scope.as_ref().unwrap();
            if !mixed.output_formats.contains(&output_format) {
                hard_errors.push(issue(
                    "generated_metal_mixed_output_format", None, None,
                    "the mixed-open Metal profile requires uncompressed NPY".into(),
                    mixed.output_formats.clone(), Some(output_format.clone()),
                ));
            }
            if output.preset != mixed.preset {
                hard_errors.push(issue(
                    "generated_metal_mixed_output_preset", None, None,
                    "the mixed-open Metal profile requires the minimal output preset".into(),
                    vec![mixed.preset.clone()], Some(output.preset.clone()),
                ));
            }
            if output.async_io != mixed.async_io {
                hard_errors.push(issue(
                    "generated_metal_mixed_async_io", None, None,
                    "the mixed-open Metal profile requires synchronous NPY publication".into(),
                    vec!["false".into()], Some(output.async_io.to_string()),
                ));
            }
            if request.allow_backend_fallback {
                hard_errors.push(issue(
                    "generated_metal_mixed_fallback", None, None,
                    "the mixed-open Metal profile requires fallback disabled".into(),
                    vec!["false".into()], Some("true".into()),
                ));
            }
            if output.keep_last_frames != 0 || output.minimum_free_space_mib != 0 ||
                output.maximum_retained_history_files != 0 {
                hard_errors.push(issue(
                    "generated_metal_mixed_history", None, None,
                    "the mixed-open Metal profile retains the exact default history policy".into(),
                    vec!["keep_last_frames=0,minimum_free_space_mib=0,maximum_retained_history_files=0".into()],
                    Some(format!("{},{},{}", output.keep_last_frames,
                                 output.minimum_free_space_mib, output.maximum_retained_history_files)),
                ));
            }
            if request.diagnostics.scientific_every_steps != 1 ||
                !request.diagnostics.deterministic_reductions {
                hard_errors.push(issue(
                    "generated_metal_mixed_diagnostics", None, None,
                    "the mixed-open Metal profile requires deterministic scientific diagnostics each step".into(),
                    vec!["scientific_every_steps=1,deterministic_reductions=true".into()],
                    Some(format!("{},{}", request.diagnostics.scientific_every_steps,
                                 request.diagnostics.deterministic_reductions)),
                ));
            }
        }
        let cadences = if generated_metal_mixed {
            interaction_policy.qualified_flat_mixed_metal_npy_scope.as_ref()
                .map(|mixed| &mixed.write_cadence_seconds)
        } else if generated_metal_east {
            generated_scope.additional_boundary_npy_scope.as_ref()
                .map(|east| &east.write_cadence_seconds)
        } else {
            generated_scope.write_cadence_seconds_by_format.as_ref()
                .and_then(|by_format| by_format.get(&output_format))
        };
        if let Some(values) = cadences {
            if !values.iter().any(|value| *value as f64 == time.output_interval) {
                hard_errors.push(issue(
                    "generated_metal_output_cadence",
                    None,
                    None,
                    "generated Metal output cadence is outside the catalog's reviewed format envelope".into(),
                    values.iter().map(ToString::to_string).collect(),
                    Some(time.output_interval.to_string()),
                ));
            }
        }
    }
    evaluate_science_interactions(
        &interactions,
        &interaction_policy,
        &selected_schemes,
        &output_format,
        execution_mode == "qualified",
        &mut hard_errors,
    );
    if execution_mode == "qualified"
        && selected_schemes.get("turbulence").map(String::as_str) == Some("smagorinsky")
        && interactions.turbulence_mode == "vertical_only"
        && !interactions.turbulence_dynamic_coefficient
    {
        if let Some(scope) = interaction_policy.qualified_vertical_only_sgs.as_ref()
            .filter(|scope| scope.admitted)
        {
            let exact_envelope = backend == scope.backend
                && !request.allow_backend_fallback
                && coordinate_system == scope.coordinate_system
                && staggering == scope.staggering
                && output_format == scope.output_format
                && [request.grid_metrics.nx, request.grid_metrics.ny, request.grid_metrics.nz]
                    == scope.grid_shape.as_slice()
                && [request.grid_metrics.dx, request.grid_metrics.dy, request.grid_metrics.dz]
                    == scope.grid_spacing_meters.as_slice()
                && request.grid_metrics.ztop == 4000.0
                && request.time_control.dt_large == scope.dt_large_seconds
                && request.time_control.acoustic_substeps == scope.acoustic_substeps
                && scope.write_cadence_seconds.iter().any(|cadence|
                    request.time_control.output_interval == *cadence as f64)
                && [boundary.west.as_str(), boundary.east.as_str(),
                    boundary.south.as_str(), boundary.north.as_str()]
                    == [scope.lateral_boundary_mode.as_str(); 4]
                && selected_schemes.get("boundary_layer").map_or(true, |scheme|
                    scheme == &interaction_policy.inactive_scheme_id)
                && selected_schemes.get("terrain").map_or(true, |scheme|
                    scheme == &interaction_policy.inactive_scheme_id)
                && !request.gpu_apply_microphysics
                && !request.gpu_apply_slow
                && !request.gpu_apply_sgs;
            if !exact_envelope {
                hard_errors.push(issue(
                    "qualified_vertical_only_sgs_scope_incompatible",
                    Some("turbulence"), Some("smagorinsky"),
                    "static vertical-only SGS descriptor is outside the exact reviewed CPU scope".into(),
                    vec!["flat periodic 12x12x16 CPU NPY, no PBL/terrain/GPU apply".into()],
                    Some(format!("{} / {}", backend, output_format)),
                ));
            }
            let (code, message) = if request.source.kind == "exact_document" {
                ("qualified_vertical_only_sgs_native_admission_required",
                 "static vertical-only SGS requires exact-source native admission")
            } else {
                ("qualified_vertical_only_sgs_exact_document_required",
                 "descriptor-only preflight cannot establish executable static vertical SGS admission")
            };
            hard_errors.push(issue(
                code, Some("turbulence"), Some("smagorinsky"), message.into(),
                vec!["exact_document with native --validate-only acceptance".into()],
                Some(request.source.kind.clone()),
            ));
        }
    }
    evaluate_gpu_apply_interactions(
        &backend,
        &validation_guard_mode,
        request.gpu_apply_microphysics,
        request.gpu_apply_slow,
        request.gpu_apply_sgs,
        execution_mode == "qualified",
        &selected_schemes,
        &interaction_policy,
        &mut hard_errors,
        &mut warnings,
    );

    let compatible = hard_errors.is_empty();
    Ok(ConfigPreflightReport {
        schema: REPORT_SCHEMA,
        report_version: 10,
        compatible,
        native_admission: NativeAdmissionEvidence {
            state: "not_checked",
            descriptor_projection: "not_checked",
            source_sha256: None,
            source_fingerprint: None,
            source_path_fingerprint: None,
            validated_binary_sha256: None,
            validation_exit_code: None,
        },
        executable,
        execution_target_fingerprint: None,
        context: PreflightContext {
            execution_mode,
            maturity_limits_acknowledged: request.maturity_limits_acknowledged,
            backend,
            allow_backend_fallback: request.allow_backend_fallback,
            validation_guard_mode,
            gpu_apply_microphysics: request.gpu_apply_microphysics,
            gpu_apply_slow: request.gpu_apply_slow,
            gpu_apply_sgs: request.gpu_apply_sgs,
            output_format,
            grid_metrics: request.grid_metrics.clone(),
            time_control: request.time_control.clone(),
            output,
            coordinate_system,
            staggering,
            boundary,
            diagnostics,
            initialization,
            interactions,
            available_state_fields: available_state.into_iter().collect(),
        },
        resolved_selections,
        hard_errors,
        warnings,
    })
}

#[allow(clippy::too_many_arguments)]
fn evaluate_scheme(
    module_id: &str,
    requested_scheme_id: &str,
    scheme: &Value,
    require_qualified: bool,
    backend: &str,
    allow_backend_fallback: bool,
    coordinate_system: &str,
    staggering: &str,
    initialization: &InitializationSelection,
    available_state: &BTreeSet<String>,
    resolved: &mut Vec<ResolvedSelection>,
    errors: &mut Vec<PreflightIssue>,
    warnings: &mut Vec<PreflightIssue>,
) -> AppResult<()> {
    let canonical = required_string(scheme, "id", module_id)?;
    let maturity = required_string(scheme, "maturity", module_id)?;
    let constructible = scheme
        .get("constructible")
        .and_then(Value::as_bool)
        .ok_or_else(|| {
            AppError::Subprocess(format!(
                "scheme '{module_id}.{canonical}' lacks constructible state"
            ))
        })?;
    let runtime_selectable = scheme
        .get("runtimeSelectable")
        .and_then(Value::as_bool)
        .ok_or_else(|| {
            AppError::Subprocess(format!(
                "scheme '{module_id}.{canonical}' lacks runtime-selectable state"
            ))
        })?;
    resolved.push(ResolvedSelection {
        module_id: module_id.into(),
        requested_scheme_id: requested_scheme_id.into(),
        canonical_scheme_id: canonical.clone(),
        maturity: maturity.clone(),
        constructible,
        runtime_selectable,
    });
    if !constructible {
        errors.push(issue(
            "scheme_not_constructible",
            Some(module_id),
            Some(&canonical),
            format!("scheme '{module_id}.{canonical}' is known but not constructible in this executable"),
            vec!["constructible".into()],
            Some("not_constructible".into()),
        ));
    }
    if !runtime_selectable {
        errors.push(issue(
            "scheme_not_runtime_selectable",
            Some(module_id),
            Some(&canonical),
            format!("scheme '{module_id}.{canonical}' is development-only and unavailable for simulation runs"),
            vec!["runtimeSelectable=true".into()],
            Some("runtimeSelectable=false".into()),
        ));
    }
    if maturity != "qualified" {
        let target = if require_qualified {
            &mut *errors
        } else {
            &mut *warnings
        };
        target.push(issue(
            "scheme_maturity",
            Some(module_id),
            Some(&canonical),
            format!("scheme '{module_id}.{canonical}' has {maturity} maturity"),
            vec!["qualified".into()],
            Some(maturity.clone()),
        ));
    }
    if let Some(caveat) = string(scheme, "caveat").filter(|value| !value.trim().is_empty()) {
        warnings.push(issue(
            "scheme_caveat",
            Some(module_id),
            Some(&canonical),
            caveat.to_string(),
            Vec::new(),
            None,
        ));
    }
    let capabilities = scheme
        .get("capabilities")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            AppError::Subprocess(format!(
                "scheme '{module_id}.{canonical}' lacks capabilities"
            ))
        })?;
    if capabilities.get("declared").and_then(Value::as_bool) != Some(true) {
        errors.push(issue(
            "capabilities_undeclared",
            Some(module_id),
            Some(&canonical),
            "the executable cannot certify this scheme's capabilities".into(),
            vec!["declared".into()],
            Some("undeclared".into()),
        ));
        return Ok(());
    }
    if capabilities
        .get("configValidationContract")
        .and_then(Value::as_str)
        .map_or(true, |value| value.trim().is_empty())
    {
        errors.push(issue(
            "validation_contract_missing",
            Some(module_id),
            Some(&canonical),
            "the executable reports no configuration validation contract for this scheme".into(),
            Vec::new(),
            None,
        ));
    }

    check_axis(
        "coordinate_system_incompatible",
        "coordinate system",
        capabilities.get("coordinateSystems"),
        coordinate_system,
        module_id,
        &canonical,
        errors,
    )?;
    check_axis(
        "staggering_incompatible",
        "staggering",
        capabilities.get("staggerings"),
        staggering,
        module_id,
        &canonical,
        errors,
    )?;
    check_backend(
        capabilities.get("backends"),
        backend,
        allow_backend_fallback,
        module_id,
        &canonical,
        errors,
        warnings,
    )?;
    check_initialization(
        capabilities.get("acceptedSoundingSources"),
        "sounding",
        &initialization.sounding_source,
        module_id,
        &canonical,
        errors,
    )?;
    check_initialization(
        capabilities.get("acceptedHodographSources"),
        "hodograph",
        &initialization.hodograph_source,
        module_id,
        &canonical,
        errors,
    )?;
    check_initialization(
        capabilities.get("acceptedTriggerSources"),
        "trigger",
        &initialization.trigger_source,
        module_id,
        &canonical,
        errors,
    )?;

    let required_fields = string_array(
        capabilities.get("requiredStateFields"),
        "requiredStateFields",
        module_id,
        &canonical,
    )?;
    let missing_fields: Vec<String> = required_fields
        .iter()
        .filter(|field| !available_state.contains(field.as_str()))
        .cloned()
        .collect();
    if !missing_fields.is_empty() {
        errors.push(issue(
            "required_state_missing",
            Some(module_id),
            Some(&canonical),
            format!(
                "required model state is unavailable: {}",
                missing_fields.join(", ")
            ),
            required_fields,
            Some(format!("missing: {}", missing_fields.join(", "))),
        ));
    }
    if backend == "gpu" || backend == "metal" {
        let parity = capabilities
            .get("cpuGpuParityCoverage")
            .and_then(Value::as_str)
            .unwrap_or("unspecified");
        if parity != "covered" && parity != "not_applicable" {
            warnings.push(issue(
                "gpu_parity_maturity",
                Some(module_id),
                Some(&canonical),
                format!("CPU/GPU parity coverage is '{parity}'"),
                vec!["covered".into()],
                Some(parity.into()),
            ));
        }
    }
    Ok(())
}

fn check_axis(
    code: &'static str,
    label: &str,
    raw_supported: Option<&Value>,
    actual: &str,
    module_id: &str,
    scheme_id: &str,
    errors: &mut Vec<PreflightIssue>,
) -> AppResult<()> {
    let supported = string_array(raw_supported, label, module_id, scheme_id)?;
    if capability_not_applicable(&supported) {
        return Ok(());
    }
    if !supported
        .iter()
        .any(|value| value == "any" || value == actual)
    {
        errors.push(issue(
            code,
            Some(module_id),
            Some(scheme_id),
            format!("{label} '{actual}' is not supported by '{module_id}.{scheme_id}'"),
            supported,
            Some(actual.into()),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn check_backend(
    raw_supported: Option<&Value>,
    backend: &str,
    allow_fallback: bool,
    module_id: &str,
    scheme_id: &str,
    errors: &mut Vec<PreflightIssue>,
    warnings: &mut Vec<PreflightIssue>,
) -> AppResult<()> {
    let supported = string_array(raw_supported, "backends", module_id, scheme_id)?;
    let capability_backend = if backend == "metal" { "gpu" } else { backend };
    if capability_not_applicable(&supported)
        || supported
            .iter()
            .any(|value| value == "any" || value == capability_backend)
    {
        return Ok(());
    }
    let falls_back_to_cpu = allow_fallback
        && (backend == "gpu" || backend == "metal")
        && supported.iter().any(|value| value == "cpu");
    let (code, message) = if falls_back_to_cpu {
        (
            "backend_cpu_fallback",
            format!("'{module_id}.{scheme_id}' will use the permitted CPU fallback for a GPU run"),
        )
    } else {
        (
            "backend_incompatible",
            format!("backend '{backend}' is not supported by '{module_id}.{scheme_id}'"),
        )
    };
    let finding = issue(
        code,
        Some(module_id),
        Some(scheme_id),
        message,
        supported,
        Some(backend.into()),
    );
    if falls_back_to_cpu {
        warnings.push(finding);
    } else {
        errors.push(finding);
    }
    Ok(())
}

fn check_initialization(
    raw_accepted: Option<&Value>,
    kind: &str,
    actual: &str,
    module_id: &str,
    scheme_id: &str,
    errors: &mut Vec<PreflightIssue>,
) -> AppResult<()> {
    let accepted = string_array(raw_accepted, kind, module_id, scheme_id)?;
    if capability_not_applicable(&accepted)
        || accepted
            .iter()
            .any(|value| value == "any" || value == actual)
    {
        return Ok(());
    }
    errors.push(issue(
        "initialization_source_incompatible",
        Some(module_id),
        Some(scheme_id),
        format!("{kind} source '{actual}' is not accepted by '{module_id}.{scheme_id}'"),
        accepted,
        Some(actual.into()),
    ));
    Ok(())
}

fn string_array(
    value: Option<&Value>,
    field: &str,
    module_id: &str,
    scheme_id: &str,
) -> AppResult<Vec<String>> {
    value
        .and_then(Value::as_array)
        .filter(|values| values.iter().all(Value::is_string))
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(normalize)
                .collect()
        })
        .ok_or_else(|| {
            AppError::Subprocess(format!(
                "scheme '{module_id}.{scheme_id}' has malformed capability '{field}'"
            ))
        })
}

fn capability_not_applicable(values: &[String]) -> bool {
    values.len() == 1 && values[0] == "not_applicable"
}

fn find_scheme<'a>(schemes: &'a [Value], requested: &str) -> Option<&'a Value> {
    schemes.iter().find(|scheme| {
        string(scheme, "id").is_some_and(|id| normalize(id) == requested)
            || scheme
                .get("aliases")
                .and_then(Value::as_array)
                .is_some_and(|aliases| {
                    aliases.iter().any(|alias| {
                        alias
                            .as_str()
                            .is_some_and(|alias| normalize(alias) == requested)
                    })
                })
    })
}

fn required_string(value: &Value, key: &str, owner: &str) -> AppResult<String> {
    string(value, key)
        .map(str::to_string)
        .ok_or_else(|| AppError::Subprocess(format!("catalog entry '{owner}' lacks '{key}'")))
}

fn evaluate_output_interactions(
    output_format: &str,
    output: &OutputInteractionValues,
    policy: &CatalogInteractionPolicy,
    require_qualified: bool,
    radar_v4_candidate: bool,
    errors: &mut Vec<PreflightIssue>,
) {
    let zfp_output = matches!(output_format, "zfp" | "zfp_low" | "zfp_medium" | "zfp_high");
    if !zfp_output {
        return;
    }

    if !output.zfp_sparse_threshold.is_finite() || output.zfp_sparse_threshold < 0.0 {
        errors.push(issue(
            "zfp_sparse_threshold_invalid",
            None,
            None,
            "ZFP sparse threshold must be finite and nonnegative".into(),
            vec!["finite value >= 0".into()],
            Some(output.zfp_sparse_threshold.to_string()),
        ));
    }
    if !policy
        .qualified_zfp_field_execution_policies
        .contains(&output.zfp_field_execution)
    {
        errors.push(issue(
            "zfp_field_execution_invalid",
            None,
            None,
            "ZFP field execution policy is unsupported by the exact binary".into(),
            policy.qualified_zfp_field_execution_policies.clone(),
            Some(output.zfp_field_execution.clone()),
        ));
    }
    if output.zfp_field_workers < 0
        || output.zfp_field_workers > policy.qualified_zfp_max_field_workers
    {
        errors.push(issue(
            "zfp_field_workers_invalid",
            None,
            None,
            "ZFP field worker request is outside the exact-binary bound".into(),
            vec![format!("0..={}", policy.qualified_zfp_max_field_workers)],
            Some(output.zfp_field_workers.to_string()),
        ));
    }
    if output.zfp_field_scratch_mib < policy.qualified_zfp_min_field_scratch_mib
        || output.zfp_field_scratch_mib > policy.qualified_zfp_max_field_scratch_mib
    {
        errors.push(issue(
            "zfp_field_scratch_invalid",
            None,
            None,
            "ZFP field scratch budget is outside the exact-binary bound".into(),
            vec![format!(
                "{}..={} MiB",
                policy.qualified_zfp_min_field_scratch_mib,
                policy.qualified_zfp_max_field_scratch_mib
            )],
            Some(output.zfp_field_scratch_mib.to_string()),
        ));
    }
    for (field, tolerance) in &output.zfp_field_tolerances {
        if !tolerance.is_finite() || *tolerance <= 0.0 {
            errors.push(issue(
                "zfp_field_tolerance_invalid",
                None,
                None,
                format!(
                    "ZFP field '{}' tolerance must be finite and positive",
                    field
                ),
                vec!["finite value > 0".into()],
                Some(tolerance.to_string()),
            ));
        } else if let Some(maximum) = policy.qualified_zfp_field_budgets.get(field) {
            if *tolerance > *maximum {
                errors.push(issue(
                    "qualified_zfp_field_tolerance_exceeded",
                    None,
                    None,
                    format!(
                        "qualified ZFP field '{}' tolerance={} exceeds maximum={}",
                        field, tolerance, maximum
                    ),
                    vec![format!("0 < tolerance <= {maximum}")],
                    Some(tolerance.to_string()),
                ));
            }
        }
    }
    if !require_qualified {
        return;
    }

    let radar_preset = radar_v4_candidate
        && policy.qualified_zfp_reversible_radar_v4.as_ref()
            .is_some_and(|scope| scope.presets.contains(&output.preset));
    if !policy.qualified_zfp_preset_ids.contains(&output.preset) && !radar_preset {
        errors.push(issue(
            "qualified_zfp_preset_incompatible",
            None,
            None,
            format!(
                "qualified output.format={} does not admit output.preset={}",
                output_format, output.preset
            ),
            policy.qualified_zfp_preset_ids.clone(),
            Some(output.preset.clone()),
        ));
    }
    if output.zfp_mode != policy.qualified_zfp_mode {
        errors.push(issue(
            "qualified_zfp_mode_incompatible",
            None,
            None,
            format!("qualified ZFP requires mode={}", policy.qualified_zfp_mode),
            vec![policy.qualified_zfp_mode.clone()],
            Some(output.zfp_mode.clone()),
        ));
    }
    if policy.qualified_zfp_requires_per_field_tolerances && !output.zfp_per_field_tolerances {
        errors.push(issue(
            "qualified_zfp_field_tolerances_required",
            None,
            None,
            "qualified ZFP requires explicit per-field tolerances".into(),
            vec!["true".into()],
            Some("false".into()),
        ));
    }
    if output.keep_last_frames < policy.qualified_zfp_minimum_keep_last_frames {
        errors.push(issue(
            "qualified_zfp_retention_required",
            None,
            None,
            "qualified ZFP requires positive finite frame retention".into(),
            vec![format!(
                ">={}",
                policy.qualified_zfp_minimum_keep_last_frames
            )],
            Some(output.keep_last_frames.to_string()),
        ));
    }
    if output.minimum_free_space_mib < policy.qualified_zfp_minimum_free_space_mib {
        errors.push(issue(
            "qualified_zfp_free_space_required",
            None,
            None,
            "qualified ZFP requires a positive retained free-space reserve".into(),
            vec![format!(">={}", policy.qualified_zfp_minimum_free_space_mib)],
            Some(output.minimum_free_space_mib.to_string()),
        ));
    }
    if policy.qualified_zfp_requires_retained_file_ceiling
        && output.maximum_retained_history_files <= 0
    {
        errors.push(issue(
            "qualified_zfp_file_ceiling_required",
            None,
            None,
            "qualified ZFP requires a positive retained history file ceiling".into(),
            vec![">0".into()],
            Some(output.maximum_retained_history_files.to_string()),
        ));
    }
    if output.zfp_sparse_threshold > policy.qualified_zfp_max_sparse_threshold {
        errors.push(issue(
            "qualified_zfp_sparse_threshold_incompatible",
            None,
            None,
            "qualified ZFP sparse threshold exceeds the exact-binary limit".into(),
            vec![policy.qualified_zfp_max_sparse_threshold.to_string()],
            Some(output.zfp_sparse_threshold.to_string()),
        ));
    }
    if output.zfp_float16_prefilter && !policy.qualified_zfp_allows_float16_prefilter {
        errors.push(issue(
            "qualified_zfp_float16_prefilter_incompatible",
            None,
            None,
            "qualified ZFP forbids the float16 prefilter".into(),
            vec!["false".into()],
            Some("true".into()),
        ));
    }
    let required_fields = if radar_v4_candidate {
        let Some(scope) = policy.qualified_zfp_reversible_radar_v4.as_ref() else {
            return;
        };
        let mut fields = scope.core_fields.clone();
        if output.preset == "radar_polarimetric_ready" {
            fields.extend(scope.polarimetric_extra_fields.iter().cloned());
        }
        fields
    } else {
        let Some(fields) = policy.qualified_zfp_preset_fields.get(&output.preset) else {
            return;
        };
        fields.clone()
    };
    for field in &required_fields {
        let maximum = policy.qualified_zfp_field_budgets[field];
        match output.zfp_field_tolerances.get(field) {
            None => errors.push(issue(
                "qualified_zfp_field_tolerance_missing",
                None,
                None,
                format!(
                    "qualified ZFP field '{}' is missing its exact-binary tolerance",
                    field
                ),
                vec![format!("0 < tolerance <= {maximum}")],
                None,
            )),
            Some(_) => {}
        }
    }
}

fn evaluate_boundary_and_diagnostic_interactions(
    staggering: &str,
    boundary: &BoundarySelection,
    diagnostics: &DiagnosticsSelection,
    policy: &CatalogInteractionPolicy,
    errors: &mut Vec<PreflightIssue>,
) {
    for (face, mode) in [
        ("west", boundary.west.as_str()),
        ("east", boundary.east.as_str()),
        ("south", boundary.south.as_str()),
        ("north", boundary.north.as_str()),
    ] {
        if !policy
            .lateral_boundary_modes
            .iter()
            .any(|allowed| allowed == mode)
        {
            errors.push(issue(
                "unknown_lateral_boundary",
                None,
                None,
                format!("boundary_conditions.lateral.{face} selects unknown closure '{mode}'"),
                policy.lateral_boundary_modes.clone(),
                Some(mode.to_string()),
            ));
            continue;
        }
        let resolved = if mode == "open" {
            if staggering == "c_grid" {
                policy.legacy_open_cgrid_resolution.as_str()
            } else {
                policy.legacy_open_collocated_resolution.as_str()
            }
        } else {
            mode
        };
        if resolved == "radiative"
            && !policy
                .radiative_lateral_staggerings
                .iter()
                .any(|allowed| allowed == staggering)
        {
            errors.push(issue(
                "radiative_boundary_staggering_unsupported",
                None,
                None,
                format!(
                    "boundary_conditions.lateral.{face}=radiative requires staggering=c_grid; use zero_gradient for a collocated grid"
                ),
                policy.radiative_lateral_staggerings.clone(),
                Some(staggering.to_string()),
            ));
        }
    }

    if policy.periodic_lateral_pairs_required {
        for (axis, first, second) in [
            ("west/east", boundary.west.as_str(), boundary.east.as_str()),
            (
                "south/north",
                boundary.south.as_str(),
                boundary.north.as_str(),
            ),
        ] {
            if (first == "periodic") != (second == "periodic") {
                errors.push(issue(
                    "periodic_boundary_pair_required",
                    None,
                    None,
                    format!(
                        "periodic {axis} lateral boundaries must be configured as an opposing pair"
                    ),
                    vec!["both periodic".into(), "both non-periodic".into()],
                    Some(format!("{first}/{second}")),
                ));
            }
        }
    }
    if !policy.bottom_boundary_modes.contains(&boundary.bottom) {
        errors.push(issue(
            "unknown_bottom_boundary",
            None,
            None,
            format!(
                "boundary_conditions.bottom selects unknown closure '{}'",
                boundary.bottom
            ),
            policy.bottom_boundary_modes.clone(),
            Some(boundary.bottom.clone()),
        ));
    }
    if !policy.top_boundary_modes.contains(&boundary.top) {
        errors.push(issue(
            "unknown_top_boundary",
            None,
            None,
            format!(
                "boundary_conditions.top selects unknown closure '{}'",
                boundary.top
            ),
            policy.top_boundary_modes.clone(),
            Some(boundary.top.clone()),
        ));
    }
    if diagnostics.scientific_every_steps != policy.scientific_diagnostics_omitted_sentinel
        && diagnostics.scientific_every_steps < policy.scientific_diagnostics_minimum_cadence
    {
        errors.push(issue(
            "invalid_scientific_diagnostic_cadence",
            None,
            None,
            "extensions.diagnostics.scientific_every_steps must be omitted or non-negative".into(),
            vec![
                format!(
                    "{} (omitted compatibility cadence)",
                    policy.scientific_diagnostics_omitted_sentinel
                ),
                format!(">= {}", policy.scientific_diagnostics_minimum_cadence),
            ],
            Some(diagnostics.scientific_every_steps.to_string()),
        ));
    }
}

fn evaluate_science_interactions(
    values: &ScienceInteractionValues,
    policy: &CatalogInteractionPolicy,
    selections: &BTreeMap<String, String>,
    output_format: &str,
    require_qualified: bool,
    errors: &mut Vec<PreflightIssue>,
) {
    let inactive = policy.inactive_scheme_id.as_str();
    let boundary_layer = selections.get("boundary_layer").map(String::as_str);
    let turbulence = selections.get("turbulence").map(String::as_str);

    if values.turbulence_dynamic_coefficient {
        if turbulence != Some("smagorinsky") {
            errors.push(issue(
                "dynamic_smagorinsky_scheme_required",
                Some("turbulence"), turbulence,
                "dynamic Smagorinsky requires turbulence.scheme=smagorinsky".into(),
                vec!["smagorinsky".into()], turbulence.map(str::to_string),
            ));
        }
        if values.turbulence_mode != "3d" {
            errors.push(issue(
                "dynamic_smagorinsky_requires_3d",
                Some("turbulence"), turbulence,
                "dynamic Smagorinsky requires turbulence.mode=3d in every execution mode".into(),
                vec!["3d".into()], Some(values.turbulence_mode.clone()),
            ));
        }
    } else if require_qualified && turbulence == Some("smagorinsky")
        && values.turbulence_mode == "vertical_only"
        && policy.qualified_vertical_only_sgs.as_ref().is_some_and(|scope| !scope.admitted)
    {
        errors.push(issue(
            "qualified_vertical_only_sgs_not_ready",
            Some("turbulence"), turbulence,
            "qualified static vertical-only SGS remains fail-closed pending whole-step validation".into(),
            vec!["engineering-complete native vertical-only scope".into()],
            Some(values.turbulence_mode.clone()),
        ));
    }

    if boundary_layer.is_some_and(|scheme| scheme != inactive) {
        match turbulence {
            None => errors.push(issue(
                "interaction_selection_missing",
                Some("turbulence"),
                None,
                "an active boundary-layer closure requires the turbulence selection for vertical-mixing ownership preflight".into(),
                vec!["turbulence scheme selection".into()],
                None,
            )),
            Some(scheme)
                if scheme != inactive
                    && !policy
                        .horizontal_only_turbulence_modes
                        .contains(&values.turbulence_mode) =>
            {
                errors.push(issue(
                    "vertical_mixing_overlap",
                    Some("turbulence"),
                    Some(scheme),
                    format!(
                        "boundary_layer={} and turbulence={} mode={} both own vertical momentum/heat/moisture mixing",
                        boundary_layer.unwrap_or(inactive),
                        scheme,
                        values.turbulence_mode
                    ),
                    policy.horizontal_only_turbulence_modes.clone(),
                    Some(values.turbulence_mode.clone()),
                ));
            }
            Some(_) => {}
        }
    }

    if !policy
        .known_surface_schemes
        .contains(&values.surface_scheme)
    {
        errors.push(issue(
            "unknown_surface_scheme",
            Some("surface"),
            Some(&values.surface_scheme),
            format!(
                "surface scheme '{}' is not recognized by the exact-binary interaction policy",
                values.surface_scheme
            ),
            policy.known_surface_schemes.clone(),
            Some(values.surface_scheme.clone()),
        ));
    } else {
        if values.surface_scheme == inactive && values.surface_apply_fluxes {
            errors.push(issue(
                "surface_fluxes_without_surface",
                Some("surface"),
                Some(inactive),
                "surface.apply_fluxes cannot be true when surface.scheme=none".into(),
                vec!["false".into()],
                Some("true".into()),
            ));
        }
        if values.surface_scheme == policy.active_surface_scheme_id
            && boundary_layer.map_or(true, |scheme| scheme == inactive)
        {
            errors.push(issue(
                "surface_requires_boundary_layer",
                Some("surface"),
                Some(&values.surface_scheme),
                "an active slab surface requires an explicitly active boundary-layer closure"
                    .into(),
                vec!["boundary_layer scheme other than none".into()],
                boundary_layer.map(str::to_string),
            ));
        }
    }

    let terrain = selections.get("terrain").map(String::as_str);
    let terrain_id = terrain.unwrap_or(inactive);
    let zfp_output = matches!(output_format, "zfp" | "zfp_low" | "zfp_medium" | "zfp_high");
    if require_qualified
        && zfp_output
        && !policy
            .qualified_zfp_terrain_schemes
            .iter()
            .any(|scheme| scheme == terrain_id)
    {
        errors.push(issue(
            "qualified_zfp_terrain_incompatible",
            Some("terrain"),
            Some(terrain_id),
            format!(
                "qualified output.format={} does not support terrain={}",
                output_format, terrain_id
            ),
            policy.qualified_zfp_terrain_schemes.clone(),
            Some(terrain_id.into()),
        ));
    }
    if terrain.map_or(true, |scheme| scheme == inactive) {
        return;
    }
    let required = [
        "radiation",
        "boundary_layer",
        "turbulence",
        "chaos",
        "diffusion",
    ];
    for module in required {
        if !selections.contains_key(module) {
            errors.push(issue(
                "interaction_selection_missing",
                Some(module),
                None,
                format!(
                    "active terrain '{}' requires the '{}' selection for exact-binary interaction preflight",
                    terrain_id, module
                ),
                vec![format!("{module} scheme selection")],
                None,
            ));
        }
    }

    if let Some(radiation) = selections.get("radiation") {
        if !policy.terrain_radiation_schemes.contains(radiation)
            || (radiation == &policy.terrain_longwave_radiation_scheme_id
                && values.radiation_shortwave_enabled
                && !policy.terrain_shortwave_allowed)
        {
            errors.push(issue(
                "terrain_radiation_incompatible",
                Some("radiation"),
                Some(radiation),
                format!(
                    "terrain={} does not support radiation={} with shortwave={}",
                    terrain_id, radiation, values.radiation_shortwave_enabled
                ),
                policy.terrain_radiation_schemes.clone(),
                Some(radiation.clone()),
            ));
        }
    }
    if let Some(boundary_layer) = boundary_layer {
        if !policy
            .terrain_boundary_layer_schemes
            .iter()
            .any(|scheme| scheme == boundary_layer)
        {
            errors.push(issue(
                "terrain_boundary_layer_incompatible",
                Some("boundary_layer"),
                Some(boundary_layer),
                format!(
                    "terrain={} does not support boundary_layer={}",
                    terrain_id, boundary_layer
                ),
                policy.terrain_boundary_layer_schemes.clone(),
                Some(boundary_layer.into()),
            ));
        }
    }
    if let Some(turbulence) = turbulence {
        if !policy
            .terrain_turbulence_schemes
            .iter()
            .any(|scheme| scheme == turbulence)
        {
            errors.push(issue(
                "terrain_turbulence_incompatible",
                Some("turbulence"),
                Some(turbulence),
                format!(
                    "terrain={} does not support turbulence={}",
                    terrain_id, turbulence
                ),
                policy.terrain_turbulence_schemes.clone(),
                Some(turbulence.into()),
            ));
        }
    }
    if let Some(chaos) = selections.get("chaos") {
        if !policy.terrain_chaos_schemes.contains(chaos) {
            errors.push(issue(
                "terrain_chaos_incompatible",
                Some("chaos"),
                Some(chaos),
                format!("terrain={} does not support chaos={}", terrain_id, chaos),
                policy.terrain_chaos_schemes.clone(),
                Some(chaos.clone()),
            ));
        }
    }
    if values.diffusion_active {
        if let Some(diffusion) = selections.get("diffusion") {
            if !policy.terrain_active_diffusion_schemes.contains(diffusion)
                || !policy
                    .terrain_active_diffusion_targets
                    .contains(&values.diffusion_target)
            {
                errors.push(issue(
                    "terrain_diffusion_incompatible",
                    Some("diffusion"),
                    Some(diffusion),
                    format!(
                        "terrain={} does not support active diffusion={} target={}",
                        terrain_id, diffusion, values.diffusion_target
                    ),
                    policy.terrain_active_diffusion_targets.clone(),
                    Some(values.diffusion_target.clone()),
                ));
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_gpu_apply_interactions(
    backend: &str,
    guard_mode: &str,
    apply_microphysics: bool,
    apply_slow: bool,
    apply_sgs: bool,
    require_qualified: bool,
    selections: &BTreeMap<String, String>,
    policy: &CatalogInteractionPolicy,
    errors: &mut Vec<PreflightIssue>,
    warnings: &mut Vec<PreflightIssue>,
) {
    if backend == "cpu" {
        if !policy.gpu_apply_controls_no_op_on_cpu
            && (apply_microphysics || apply_slow || apply_sgs)
        {
            errors.push(issue(
                "gpu_apply_cpu_policy_inconsistent",
                None,
                None,
                "The exact binary does not define accelerator-apply controls as CPU no-ops".into(),
                vec!["all accelerator apply controls disabled".into()],
                None,
            ));
        }
        return;
    }

    let active_microphysics = selections
        .get("microphysics")
        .is_some_and(|scheme| scheme != &policy.inactive_scheme_id);
    evaluate_gpu_apply_component(
        apply_microphysics && active_microphysics,
        "microphysics",
        guard_mode,
        &policy.gpu_apply_microphysics_schemes,
        &policy.gpu_apply_microphysics_dispatch_guard_modes,
        &policy.gpu_apply_microphysics_host_commit_guard_modes,
        selections,
        errors,
        warnings,
    );
    let qualified_slow_supported = selections
        .get("dynamics")
        .is_some_and(|scheme| {
            policy
                .qualified_gpu_apply_slow_dynamics_schemes
                .contains(scheme)
        });
    if apply_slow && require_qualified && !qualified_slow_supported {
        errors.push(issue(
            "qualified_gpu_apply_slow_forbidden",
            Some("dynamics"),
            selections.get("dynamics").map(String::as_str),
            "Qualified execution does not admit the accelerator slow-state commit".into(),
            policy.qualified_gpu_apply_slow_dynamics_schemes.clone(),
            selections.get("dynamics").cloned(),
        ));
    } else {
        evaluate_gpu_apply_component(
            apply_slow,
            "dynamics",
            guard_mode,
            &policy.gpu_apply_slow_dynamics_schemes,
            &policy.gpu_apply_slow_dispatch_guard_modes,
            &policy.gpu_apply_slow_host_commit_guard_modes,
            selections,
            errors,
            warnings,
        );
    }
    let active_turbulence = selections
        .get("turbulence")
        .is_some_and(|scheme| scheme != &policy.inactive_scheme_id);
    evaluate_gpu_apply_component(
        apply_sgs && active_turbulence,
        "turbulence",
        guard_mode,
        &policy.gpu_apply_sgs_schemes,
        &policy.gpu_apply_sgs_dispatch_guard_modes,
        &[],
        selections,
        errors,
        warnings,
    );
}

#[allow(clippy::too_many_arguments)]
fn evaluate_gpu_apply_component(
    enabled: bool,
    module: &str,
    guard_mode: &str,
    supported_schemes: &[String],
    dispatch_guards: &[String],
    host_commit_guards: &[String],
    selections: &BTreeMap<String, String>,
    errors: &mut Vec<PreflightIssue>,
    warnings: &mut Vec<PreflightIssue>,
) {
    if !enabled {
        return;
    }
    let selected = selections.get(module).map(String::as_str).unwrap_or("");
    if !supported_schemes.iter().any(|scheme| scheme == selected) {
        errors.push(issue(
            "gpu_apply_scheme_incompatible",
            Some(module),
            Some(selected),
            format!("accelerator apply is unavailable for {module}={selected}"),
            supported_schemes.to_vec(),
            Some(selected.into()),
        ));
        return;
    }
    if dispatch_guards.iter().any(|guard| guard == guard_mode) {
        return;
    }
    if host_commit_guards.iter().any(|guard| guard == guard_mode) {
        warnings.push(issue(
            "gpu_apply_host_commit",
            Some(module),
            Some(selected),
            format!(
                "accelerator apply for {module}={selected} retains the policy-required host state commit under guard_mode={guard_mode}"
            ),
            dispatch_guards.to_vec(),
            Some(guard_mode.into()),
        ));
        return;
    }
    errors.push(issue(
        "gpu_apply_guard_incompatible",
        Some(module),
        Some(selected),
        format!(
            "accelerator apply for {module}={selected} is unavailable under guard_mode={guard_mode}"
        ),
        dispatch_guards
            .iter()
            .chain(host_commit_guards.iter())
            .cloned()
            .collect(),
        Some(guard_mode.into()),
    ));
}

fn string<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn normalize(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('-', "_")
}

fn normalize_backend(value: &str) -> String {
    match normalize(value).as_str() {
        "vulkan" | "vk" => "gpu".into(),
        other => other.into(),
    }
}

fn normalize_sounding(value: &str) -> String {
    match normalize(value).as_str() {
        "cape" | "parametriccape" => "parametric_cape".into(),
        "sharpy" | "raob" | "sounding_file" => "file".into(),
        "targets" | "diagnostics" | "from_table" | "parametrictargets" => {
            "parametric_targets".into()
        }
        other => other.into(),
    }
}

fn normalize_hodograph(value: &str) -> String {
    match normalize(value).as_str() {
        "wk" | "wkparam" | "custom" => "wk_param".into(),
        "static" | "calm" => "zero".into(),
        other => other.into(),
    }
}

fn normalize_trigger(value: &str) -> String {
    match normalize(value).as_str() {
        "bubble" | "warmbubble" => "warm_bubble".into(),
        "off" | "disabled" => "none".into(),
        "vortex" | "vortexseed" | "rankine" => "vortex_seed".into(),
        other => other.into(),
    }
}

fn issue(
    code: &'static str,
    module_id: Option<&str>,
    scheme_id: Option<&str>,
    message: String,
    expected: Vec<String>,
    actual: Option<String>,
) -> PreflightIssue {
    PreflightIssue {
        code,
        module_id: module_id.map(str::to_string),
        scheme_id: scheme_id.map(str::to_string),
        message,
        expected,
        actual,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalog() -> Value {
        let mut interaction_policy = serde_json::json!({
            "version": 9,
            "generatedMetalCanonicalScope": {
                "version": 2,
                "backend": "metal",
                "executionMode": "qualified",
                "tupleCount": 4032,
                "distinctConfigCount": 3024,
                "outputFormats": ["npy_3d", "netcdf"],
                "scalarTransportOwner": "native_conservative_mc",
                "baseManifestSha256": "85642f094c418fe95eb4c2fa0ba5a2a947aec95f8f1c43b17abcd924b0f068a0",
                "manifestSha256": "30d5fcc9494016894733ef973909849f8c58fbd04b3e0d6bb2e5e497c0c9ca99"
            },
            "inactiveSchemeId": "none",
            "horizontalOnlyTurbulenceModes": ["2d", "horizontal_only"],
            "knownSurfaceSchemes": ["none", "slab"],
            "activeSurfaceSchemeId": "slab",
            "terrainRadiationSchemes": ["none", "simple_grey", "rrtmg"],
            "terrainLongwaveRadiationSchemeId": "rrtmg",
            "terrainShortwaveAllowed": true,
            "terrainBoundaryLayerSchemes": ["none", "slab", "ysu", "mynn"],
            "terrainTurbulenceSchemes": ["none", "smagorinsky", "tke"],
            "terrainChaosSchemes": ["none", "initial_conditions", "boundary_layer", "full_stochastic"],
            "chaosCanonicalTargets": {
                "none": {"initialConditions": [], "tendencies": [], "nativeGenerationRequired": false},
                "initial_conditions": {
                    "initialConditions": ["u", "v", "w", "theta"],
                    "tendencies": [], "nativeGenerationRequired": false
                },
                "boundary_layer": {
                    "initialConditions": [], "tendencies": ["pbl"],
                    "nativeGenerationRequired": false
                },
                "full_stochastic": {
                    "initialConditions": ["u", "v", "w", "theta"],
                    "tendencies": ["microphysics", "pbl", "diffusion"],
                    "nativeGenerationRequired": false
                }
            },
            "terrainActiveDiffusionSchemes": ["explicit", "implicit"],
            "terrainActiveDiffusionTargets": ["scalars", "momentum", "scalars_and_momentum", "all"],
            "qualifiedZfpTerrainSchemes": ["none", "bell", "schar"],
            "qualifiedZfpPresetIds": ["minimal"],
            "qualifiedZfpPresetFields": {
                "minimal": ["p", "qr", "qv", "rho", "theta", "u", "v", "w"]
            },
            "qualifiedZfpMode": "accuracy",
            "qualifiedZfpRequiresPerFieldTolerances": true,
            "qualifiedZfpMinimumKeepLastFrames": 1,
            "qualifiedZfpMinimumFreeSpaceMib": 1,
            "qualifiedZfpRequiresRetainedFileCeiling": true,
            "qualifiedExecutionRequiresMaturityAcknowledgment": false,
            "qualifiedZfpMaxSparseThreshold": 0.0,
            "qualifiedZfpAllowsFloat16Prefilter": false,
            "qualifiedZfpFieldBudgets": {
                "p": 5.0e-2, "qr": 1.0e-8, "qv": 1.0e-8,
                "rho": 1.0e-6, "theta": 1.0e-3,
                "u": 1.0e-4, "v": 1.0e-4, "w": 1.0e-4
            },
            "qualifiedZfpFieldUnits": {
                "p": "Pa", "qr": "kg/kg", "qv": "kg/kg", "rho": "kg/m^3",
                "theta": "K", "u": "m/s", "v": "m/s", "w": "m/s"
            },
            "lateralBoundaryModes": [
                "periodic", "radiative", "zero_gradient", "open",
                "rigid", "rigid_slip"
            ],
            "radiativeLateralStaggerings": ["c_grid"],
            "legacyOpenCgridResolution": "radiative",
            "legacyOpenCollocatedResolution": "zero_gradient",
            "periodicLateralPairsRequired": true,
            "bottomBoundaryModes": ["rigid_noslip", "rigid_slip"],
            "topBoundaryModes": ["rigid", "rigid_with_sponge", "open"],
            "scientificDiagnosticsOmittedSentinel": -1,
            "scientificDiagnosticsMinimumCadence": 0,
            "gpuApplyControlsNoOpOnCpu": true,
            "gpuApplyMicrophysicsSchemes": ["kessler", "lin"],
            "gpuApplySgsSchemes": ["smagorinsky"],
            "gpuApplySlowDynamicsSchemes": ["klemp_wilhelmson_split_cartesian_agrid", "klemp_wilhelmson_split_cartesian_cgrid"],
            "qualifiedGpuApplySlowDynamicsSchemes": [
                "klemp_wilhelmson_split_cartesian_agrid",
                "klemp_wilhelmson_split_cartesian_cgrid"
            ],
            "gpuApplyMicrophysicsDispatchGuardModes": ["off"],
            "gpuApplyMicrophysicsHostCommitGuardModes": ["warn", "error"],
            "gpuApplySlowDispatchGuardModes": ["off"],
            "gpuApplySlowHostCommitGuardModes": ["warn", "error"],
            "gpuApplySgsDispatchGuardModes": ["off", "warn", "error"]
        });
        interaction_policy["qualifiedZfpFieldExecutionPolicies"] =
            serde_json::json!(["serial", "field_parallel"]);
        interaction_policy["qualifiedZfpMaxFieldWorkers"] = serde_json::json!(16);
        interaction_policy["qualifiedZfpMinFieldScratchMib"] = serde_json::json!(1);
        interaction_policy["qualifiedZfpMaxFieldScratchMib"] = serde_json::json!(4096);
        serde_json::json!({
            "schema": "nimbus.scheme-catalog.v9",
            "catalogVersion": 9,
            "runtimeStateFields": ["u", "v", "w", "theta", "qv"],
            "readiness": {"version": 1, "items": [{
                "id": "catalog.runtime_cli_studio",
                "area": "capability_truth",
                "scope": "Exact-binary catalog agreement",
                "state": "ready",
                "boundedEvidence": "tests",
                "campaignEvidence": "",
                "externalReviewEvidence": "",
                "blocker": ""
            }]},
            "runtimeCapabilities": {"outputFormats": [
                {"id": "npy_3d", "available": true, "reason": "", "implementation": "builtin_npy"},
                {"id": "zfp_high", "available": true, "reason": "", "implementation": "libzfp"},
                {"id": "netcdf", "available": false, "reason": "no libnetcdf", "implementation": ""}
            ], "interactionPolicy": interaction_policy},
            "modules": [
                {"id": "dynamics", "selectionPath": "dynamics.scheme",
                 "studioSelection": {"mode": "form", "fixedSchemeId": "", "reason": ""}, "schemes": [
                    scheme("cgrid", vec!["cart"], "qualified", true,
                        vec!["cartesian"], vec!["c_grid"], vec!["cpu", "gpu"],
                        vec!["u", "v", "w"],
                        vec!["parametric_cape"], vec!["wk_param"], vec!["warm_bubble"],
                        "covered")
                ]},
                {"id": "boundary_layer", "selectionPath": "boundary_layer.scheme",
                 "studioSelection": {"mode": "form", "fixedSchemeId": "", "reason": ""}, "schemes": [
                    scheme("slab", vec![], "experimental", true,
                        vec!["cartesian"], vec!["collocated", "c_grid"], vec!["cpu"],
                        vec!["theta", "qv"],
                        vec!["not_applicable"], vec!["not_applicable"], vec!["not_applicable"],
                        "none")
                ]},
                {"id": "turbulence", "selectionPath": "turbulence.scheme",
                 "studioSelection": {"mode": "form", "fixedSchemeId": "", "reason": ""}, "schemes": [
                    scheme("none", vec![], "qualified", true,
                        vec!["not_applicable"], vec!["not_applicable"], vec!["not_applicable"],
                        vec![], vec!["not_applicable"], vec!["not_applicable"], vec!["not_applicable"],
                        "not_applicable")
                ]},
                {"id": "radiation", "selectionPath": "radiation.scheme",
                 "studioSelection": {"mode": "form", "fixedSchemeId": "", "reason": ""}, "schemes": [
                    scheme("rrtmg", vec!["rrtmgp"], "deferred", false,
                        vec!["not_applicable"], vec!["not_applicable"], vec!["not_implemented"],
                        vec![], vec!["not_applicable"], vec!["not_applicable"], vec!["not_applicable"],
                        "not_applicable")
                ]}
            ]
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn scheme(
        id: &str,
        aliases: Vec<&str>,
        maturity: &str,
        constructible: bool,
        coordinate_systems: Vec<&str>,
        staggerings: Vec<&str>,
        backends: Vec<&str>,
        required_state_fields: Vec<&str>,
        sounding: Vec<&str>,
        hodograph: Vec<&str>,
        trigger: Vec<&str>,
        parity: &str,
    ) -> Value {
        serde_json::json!({
            "id": id,
            "aliases": aliases,
            "displayName": id,
            "maturity": maturity,
            "constructible": constructible,
            "runtimeSelectable": constructible,
            "readinessState": if maturity == "qualified" && constructible {
                "ready"
            } else {
                "engineering_incomplete"
            },
            "caveat": "",
            "parameters": [],
            "capabilities": {
                "declared": true,
                "coordinateSystems": coordinate_systems,
                "staggerings": staggerings,
                "backends": backends,
                "requiredStateFields": required_state_fields,
                "producedStateFields": [],
                "optionalProducts": [],
                "acceptedSoundingSources": sounding,
                "acceptedHodographSources": hodograph,
                "acceptedTriggerSources": trigger,
                "recommendedTriggerSource": "warm_bubble",
                "configValidationContract": "test.contract",
                "cpuReferenceCoverage": "covered",
                "gpuImplementationCoverage": "covered",
                "cpuGpuParityCoverage": parity,
                "storage": {"coverage": "covered"}
            }
        })
    }

    fn request() -> ConfigPreflightRequest {
        ConfigPreflightRequest {
            schema: REQUEST_SCHEMA.into(),
            request_version: 10,
            source: PreflightSource {
                kind: "descriptor_only".into(),
                yaml_text: None,
                yaml_path: None,
            },
            execution_mode: "exploratory".into(),
            maturity_limits_acknowledged: false,
            backend: "vulkan".into(),
            allow_backend_fallback: true,
            validation_guard_mode: "warn".into(),
            gpu_apply_microphysics: false,
            gpu_apply_slow: false,
            gpu_apply_sgs: false,
            output_format: "npy_3d".into(),
            grid_metrics: GridMetrics {
                nx: 18,
                ny: 18,
                nz: 18,
                dx: 500.0,
                dy: 500.0,
                dz: 250.0,
                ztop: 4500.0,
            },
            time_control: TimeControl {
                dt_large: 0.1,
                acoustic_substeps: 6,
                output_interval: 1.0,
            },
            output: OutputInteractionValues {
                preset: "minimal".into(),
                async_io: false,
                zfp_mode: "accuracy".into(),
                zfp_per_field_tolerances: true,
                zfp_sparse_threshold: 0.0,
                zfp_float16_prefilter: false,
                zfp_field_execution: "serial".into(),
                zfp_field_workers: 0,
                zfp_field_scratch_mib: 64,
                zfp_field_tolerances: BTreeMap::from([
                    ("p".into(), 5.0e-2),
                    ("qr".into(), 1.0e-8),
                    ("qv".into(), 1.0e-8),
                    ("rho".into(), 1.0e-6),
                    ("theta".into(), 1.0e-3),
                    ("u".into(), 1.0e-4),
                    ("v".into(), 1.0e-4),
                    ("w".into(), 1.0e-4),
                ]),
                keep_last_frames: 3,
                minimum_free_space_mib: 4096,
                maximum_retained_history_files: 256,
            },
            coordinate_system: "cartesian".into(),
            staggering: "c-grid".into(),
            boundary: BoundarySelection {
                west: "open".into(),
                east: "open".into(),
                south: "open".into(),
                north: "open".into(),
                bottom: "rigid_noslip".into(),
                top: "rigid_with_sponge".into(),
            },
            diagnostics: DiagnosticsSelection {
                scientific_every_steps: -1,
                deterministic_reductions: false,
            },
            initialization: InitializationSelection {
                sounding_source: "cape".into(),
                hodograph_source: "custom".into(),
                trigger_source: "bubble".into(),
            },
            interactions: ScienceInteractionValues {
                radiation_shortwave_enabled: true,
                turbulence_mode: "3d".into(),
                turbulence_dynamic_coefficient: false,
                surface_scheme: "slab".into(),
                surface_apply_fluxes: true,
                diffusion_target: "scalars".into(),
                diffusion_active: false,
            },
            available_state_fields: vec![
                "u".into(),
                "v".into(),
                "w".into(),
                "theta".into(),
                "qv".into(),
            ],
            selections: vec![
                SchemeSelection {
                    module_id: "dynamics".into(),
                    scheme_id: "cart".into(),
                },
                SchemeSelection {
                    module_id: "boundary-layer".into(),
                    scheme_id: "slab".into(),
                },
                SchemeSelection {
                    module_id: "turbulence".into(),
                    scheme_id: "none".into(),
                },
            ],
        }
    }

    fn identity() -> ExecutableIdentity {
        ExecutableIdentity {
            canonical_path: "/test/nimbus_sim".into(),
            byte_length: 42,
            modified_unix_ms: Some(1),
            catalog_fingerprint: "fnv1a64:test".into(),
            binary_fingerprint: "fnv1a64:test-binary".into(),
        }
    }

    #[test]
    fn compatible_gpu_configuration_resolves_aliases_and_warns_on_cpu_fallback() {
        let report = evaluate_preflight(&request(), &catalog(), identity()).unwrap();
        assert!(report.compatible);
        assert_eq!(report.context.backend, "gpu");
        assert_eq!(report.context.staggering, "c_grid");
        assert_eq!(report.context.initialization.trigger_source, "warm_bubble");
        assert_eq!(report.resolved_selections[0].canonical_scheme_id, "cgrid");
        assert!(report
            .warnings
            .iter()
            .any(|issue| issue.code == "backend_cpu_fallback"));
        assert!(report
            .warnings
            .iter()
            .any(|issue| issue.code == "scheme_maturity"));
    }

    #[test]
    fn cpu_only_scheme_is_hard_error_when_fallback_is_disabled() {
        let mut request = request();
        request.allow_backend_fallback = false;
        let report = evaluate_preflight(&request, &catalog(), identity()).unwrap();
        assert!(!report.compatible);
        assert!(report
            .hard_errors
            .iter()
            .any(|issue| issue.code == "backend_incompatible"));
        assert!(!report
            .warnings
            .iter()
            .any(|issue| issue.code == "backend_cpu_fallback"));
    }

    #[test]
    fn metal_uses_accelerator_capabilities_without_losing_backend_identity() {
        let mut request = request();
        request.backend = "metal".into();
        let report = evaluate_preflight(&request, &catalog(), identity()).unwrap();
        assert!(report.compatible);
        assert_eq!(report.context.backend, "metal");
        assert!(report
            .warnings
            .iter()
            .any(|issue| issue.code == "backend_cpu_fallback"));
    }

    #[test]
    fn accelerator_apply_rejects_an_unsupported_selected_scheme() {
        let mut request = request();
        request.gpu_apply_slow = true;
        let report = evaluate_preflight(&request, &catalog(), identity()).unwrap();
        assert!(!report.compatible);
        assert!(report
            .hard_errors
            .iter()
            .any(|issue| issue.code == "gpu_apply_scheme_incompatible"));
    }

    #[test]
    fn accelerator_apply_controls_are_inert_on_cpu() {
        let mut request = request();
        request.backend = "cpu".into();
        request.gpu_apply_microphysics = true;
        request.gpu_apply_slow = true;
        request.gpu_apply_sgs = true;
        let report = evaluate_preflight(&request, &catalog(), identity()).unwrap();
        assert!(report
            .hard_errors
            .iter()
            .all(|issue| !issue.code.starts_with("gpu_apply_")));
    }

    #[test]
    fn qualified_execution_rejects_explicit_false_legacy_acknowledgment() {
        let mut request = request();
        request.execution_mode = "qualified".into();
        request.allow_backend_fallback = false;
        let blocked = evaluate_preflight(&request, &catalog(), identity()).unwrap();
        assert!(!blocked.compatible);
        assert!(blocked
            .hard_errors
            .iter()
            .any(|issue| issue.code == "explicit_false_maturity_acknowledgment"));

        request.maturity_limits_acknowledged = true;
        let acknowledged = evaluate_preflight(&request, &catalog(), identity()).unwrap();
        assert!(acknowledged
            .hard_errors
            .iter()
            .all(|issue| issue.code != "explicit_false_maturity_acknowledgment"));
        assert_eq!(acknowledged.context.execution_mode, "qualified");
        assert!(acknowledged.context.maturity_limits_acknowledged);
        assert!(acknowledged
            .hard_errors
            .iter()
            .any(|issue| issue.code == "scheme_maturity"));
    }

    #[test]
    fn development_only_scheme_is_a_hard_error_in_every_execution_mode() {
        let request = request();
        let mut catalog = catalog();
        catalog["modules"][0]["schemes"][0]["runtimeSelectable"] = Value::Bool(false);
        let report = evaluate_preflight(&request, &catalog, identity()).unwrap();
        assert!(!report.compatible);
        assert!(report
            .hard_errors
            .iter()
            .any(|issue| issue.code == "scheme_not_runtime_selectable"));
        assert!(!report.resolved_selections[0].runtime_selectable);
    }

    #[test]
    fn qualified_execution_allows_exact_binary_zfp_capability() {
        let mut request = request();
        request.execution_mode = "qualified".into();
        request.maturity_limits_acknowledged = true;
        request.allow_backend_fallback = false;
        request.validation_guard_mode = "error".into();
        request.backend = "cpu".into();
        request.output_format = "zfp_high".into();
        request.interactions.surface_scheme = "none".into();
        request.interactions.surface_apply_fluxes = false;
        request.selections.truncate(1);
        let report = evaluate_preflight(&request, &catalog(), identity()).unwrap();
        assert!(report.compatible);
        assert_eq!(report.schema, REPORT_SCHEMA);
        assert_eq!(report.report_version, 10);
        assert!(report.hard_errors.is_empty());

        let mut missing_theta = request.clone();
        missing_theta.output.zfp_field_tolerances.remove("theta");
        let missing_report = evaluate_preflight(&missing_theta, &catalog(), identity()).unwrap();
        assert!(missing_report
            .hard_errors
            .iter()
            .any(|issue| issue.code == "qualified_zfp_field_tolerance_missing"));

        request.output.zfp_field_execution = "field_parallel".into();
        request.output.zfp_field_workers = 4;
        request.output.zfp_field_scratch_mib = 128;
        let parallel = evaluate_preflight(&request, &catalog(), identity()).unwrap();
        assert!(parallel.compatible);
        assert_eq!(
            parallel.context.output.zfp_field_execution,
            "field_parallel"
        );
        assert_eq!(parallel.context.output.zfp_field_workers, 4);
    }

    #[test]
    fn qualified_zfp_preflight_binds_exact_compression_values() {
        let mut request = request();
        request.execution_mode = "qualified".into();
        request.maturity_limits_acknowledged = true;
        request.allow_backend_fallback = false;
        request.backend = "cpu".into();
        request.output_format = "zfp_high".into();
        request.interactions.surface_scheme = "none".into();
        request.interactions.surface_apply_fluxes = false;
        request.selections.truncate(1);
        request.output.preset = "presentation".into();
        request.output.zfp_mode = "rate".into();
        request.output.zfp_per_field_tolerances = false;
        request.output.zfp_sparse_threshold = 1.0e-8;
        request.output.zfp_float16_prefilter = true;
        request.output.zfp_field_execution = "shared_stream".into();
        request.output.zfp_field_workers = 17;
        request.output.zfp_field_scratch_mib = 0;
        request.output.keep_last_frames = 0;
        request.output.minimum_free_space_mib = 0;
        request.output.maximum_retained_history_files = 0;
        request
            .output
            .zfp_field_tolerances
            .insert("rho".into(), 1.0e-3);

        let report = evaluate_preflight(&request, &catalog(), identity()).unwrap();
        assert!(!report.compatible);
        let codes: BTreeSet<&str> = report.hard_errors.iter().map(|issue| issue.code).collect();
        assert!(codes.contains("qualified_zfp_preset_incompatible"));
        assert!(codes.contains("qualified_zfp_mode_incompatible"));
        assert!(codes.contains("qualified_zfp_field_tolerances_required"));
        assert!(codes.contains("qualified_zfp_sparse_threshold_incompatible"));
        assert!(codes.contains("qualified_zfp_float16_prefilter_incompatible"));
        assert!(codes.contains("zfp_field_execution_invalid"));
        assert!(codes.contains("zfp_field_workers_invalid"));
        assert!(codes.contains("zfp_field_scratch_invalid"));
        assert!(codes.contains("qualified_zfp_retention_required"));
        assert!(codes.contains("qualified_zfp_free_space_required"));
        assert!(codes.contains("qualified_zfp_file_ceiling_required"));

        request.output.preset = "minimal".into();
        let admitted_preset = evaluate_preflight(&request, &catalog(), identity()).unwrap();
        assert!(admitted_preset
            .hard_errors
            .iter()
            .any(|issue| issue.code == "qualified_zfp_field_tolerance_exceeded"));
    }

    #[test]
    fn exact_binary_output_capability_blocks_unavailable_format() {
        let mut request = request();
        request.output_format = "netcdf".into();
        let report = evaluate_preflight(&request, &catalog(), identity()).unwrap();
        assert!(!report.compatible);
        let issue = report
            .hard_errors
            .iter()
            .find(|issue| issue.code == "output_format_unavailable")
            .expect("unavailable output format must be a hard error");
        assert!(issue.message.contains("no libnetcdf"));
        assert_eq!(issue.actual.as_deref(), Some("netcdf"));
        assert_eq!(issue.expected, vec!["npy_3d", "zfp_high"]);
    }

    #[test]
    fn generated_boundary_preflight_matches_native_topology_and_staggering_policy() {
        let policy = catalog_interaction_policy(&catalog()).unwrap();
        let modes = policy.lateral_boundary_modes.clone();
        let diagnostics = DiagnosticsSelection {
            scientific_every_steps: -1,
            deterministic_reductions: true,
        };
        let mut cases = 0_usize;
        for staggering in ["c_grid", "collocated"] {
            for west in &modes {
                for east in &modes {
                    for south in &modes {
                        for north in &modes {
                            let boundary = BoundarySelection {
                                west: west.clone(),
                                east: east.clone(),
                                south: south.clone(),
                                north: north.clone(),
                                bottom: "rigid_noslip".into(),
                                top: "rigid_with_sponge".into(),
                            };
                            let mut errors = Vec::new();
                            evaluate_boundary_and_diagnostic_interactions(
                                staggering,
                                &boundary,
                                &diagnostics,
                                &policy,
                                &mut errors,
                            );
                            let radiative_on_collocated = staggering == "collocated"
                                && [west, east, south, north]
                                    .iter()
                                    .any(|mode| mode.as_str() == "radiative");
                            let unpaired_periodic = (west.as_str() == "periodic")
                                != (east.as_str() == "periodic")
                                || (south.as_str() == "periodic") != (north.as_str() == "periodic");
                            assert_eq!(
                                errors.is_empty(),
                                !radiative_on_collocated && !unpaired_periodic,
                                "staggering={staggering} west={west} east={east} south={south} north={north}: {errors:?}"
                            );
                            cases += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(cases, 2 * 6_usize.pow(4));

        let mut errors = Vec::new();
        evaluate_boundary_and_diagnostic_interactions(
            "c_grid",
            &BoundarySelection {
                west: "zero_gradient".into(),
                east: "zero_gradient".into(),
                south: "rigid_slip".into(),
                north: "rigid_slip".into(),
                bottom: "open".into(),
                top: "periodic".into(),
            },
            &DiagnosticsSelection {
                scientific_every_steps: -2,
                deterministic_reductions: false,
            },
            &policy,
            &mut errors,
        );
        let codes: BTreeSet<&str> = errors.iter().map(|issue| issue.code).collect();
        assert!(codes.contains("unknown_bottom_boundary"));
        assert!(codes.contains("unknown_top_boundary"));
        assert!(codes.contains("invalid_scientific_diagnostic_cadence"));
    }

    #[test]
    fn exact_binary_interaction_policy_admits_only_supported_terrain_values() {
        let policy = catalog_interaction_policy(&catalog()).unwrap();
        let mut values = ScienceInteractionValues {
            radiation_shortwave_enabled: false,
            turbulence_mode: "3d".into(),
            turbulence_dynamic_coefficient: false,
            surface_scheme: "none".into(),
            surface_apply_fluxes: false,
            diffusion_target: "scalars".into(),
            diffusion_active: true,
        };
        let mut selections = BTreeMap::from([
            ("terrain".into(), "bell".into()),
            ("radiation".into(), "simple_grey".into()),
            ("boundary_layer".into(), "none".into()),
            ("turbulence".into(), "smagorinsky".into()),
            ("chaos".into(), "none".into()),
            ("diffusion".into(), "explicit".into()),
        ]);
        let mut errors = Vec::new();
        evaluate_science_interactions(&values, &policy, &selections, "npy_3d", false, &mut errors);
        assert!(errors.is_empty());

        selections.insert("radiation".into(), "rrtmg".into());
        evaluate_science_interactions(&values, &policy, &selections, "npy_3d", false, &mut errors);
        assert!(!errors
            .iter()
            .any(|issue| issue.code == "terrain_radiation_incompatible"));

        errors.clear();
        values.radiation_shortwave_enabled = true;
        evaluate_science_interactions(&values, &policy, &selections, "npy_3d", false, &mut errors);
        assert!(!errors
            .iter()
            .any(|issue| issue.code == "terrain_radiation_incompatible"));

        selections.insert("radiation".into(), "simple_grey".into());
        errors.clear();
        values.diffusion_target = "unsupported_target".into();
        selections.insert("boundary_layer".into(), "unsupported_pbl".into());
        evaluate_science_interactions(&values, &policy, &selections, "npy_3d", false, &mut errors);
        let codes: BTreeSet<&str> = errors.iter().map(|issue| issue.code).collect();
        assert!(!codes.contains("terrain_radiation_incompatible"));
        assert!(codes.contains("terrain_boundary_layer_incompatible"));
        assert!(codes.contains("terrain_diffusion_incompatible"));
    }

    #[test]
    fn qualified_zfp_admits_mapped_terrain_geometry() {
        let policy = catalog_interaction_policy(&catalog()).unwrap();
        let values = ScienceInteractionValues {
            radiation_shortwave_enabled: false,
            turbulence_mode: "3d".into(),
            turbulence_dynamic_coefficient: false,
            surface_scheme: "none".into(),
            surface_apply_fluxes: false,
            diffusion_target: "scalars".into(),
            diffusion_active: false,
        };
        let selections = BTreeMap::from([
            ("terrain".into(), "bell".into()),
            ("radiation".into(), "none".into()),
            ("boundary_layer".into(), "none".into()),
            ("turbulence".into(), "none".into()),
            ("chaos".into(), "none".into()),
            ("diffusion".into(), "explicit".into()),
        ]);
        let mut errors = Vec::new();
        evaluate_science_interactions(&values, &policy, &selections, "zfp_high", true, &mut errors);
        assert!(!errors
            .iter()
            .any(|issue| issue.code == "qualified_zfp_terrain_incompatible"));

        errors.clear();
        evaluate_science_interactions(
            &values,
            &policy,
            &selections,
            "zfp_high",
            false,
            &mut errors,
        );
        assert!(errors.is_empty());
    }

    #[test]
    fn reports_geometry_initialization_and_state_failures_independently() {
        let mut request = request();
        request.coordinate_system = "cylindrical".into();
        request.initialization.trigger_source = "none".into();
        request.available_state_fields.retain(|field| field != "w");
        let report = evaluate_preflight(&request, &catalog(), identity()).unwrap();
        assert!(!report.compatible);
        let codes: BTreeSet<&str> = report.hard_errors.iter().map(|issue| issue.code).collect();
        assert!(codes.contains("coordinate_system_incompatible"));
        assert!(codes.contains("initialization_source_incompatible"));
        assert!(codes.contains("required_state_missing"));
    }

    #[test]
    fn unknown_and_duplicate_selections_fail_closed() {
        let mut request = request();
        request.selections.push(SchemeSelection {
            module_id: "dynamics".into(),
            scheme_id: "missing".into(),
        });
        request.selections.push(SchemeSelection {
            module_id: "microphysics".into(),
            scheme_id: "kessler".into(),
        });
        let report = evaluate_preflight(&request, &catalog(), identity()).unwrap();
        assert!(!report.compatible);
        assert!(report
            .hard_errors
            .iter()
            .any(|issue| issue.code == "duplicate_module_selection"));
        assert!(report
            .hard_errors
            .iter()
            .any(|issue| issue.code == "unknown_module"));
    }

    #[test]
    fn product_set_module_accepts_multiple_descriptor_backed_products() {
        let mut catalog = catalog();
        catalog["modules"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "id": "radar",
                "selectionPath": "radar.products",
                "studioSelection": {
                    "mode": "product_set_v1",
                    "fixedSchemeId": "",
                    "reason": "Strict v1 ordered radar product set."
                },
                "schemes": [
                    scheme("reflectivity", vec!["reflectivity_dbz"], "qualified", true,
                        vec!["cartesian"], vec!["c_grid"], vec!["cpu"], vec!["rho"],
                        vec!["not_applicable"], vec!["not_applicable"], vec!["not_applicable"],
                        "covered"),
                    scheme("velocity", vec!["radial_velocity"], "qualified", true,
                        vec!["cartesian"], vec!["c_grid"], vec!["cpu"], vec!["u", "v", "w"],
                        vec!["not_applicable"], vec!["not_applicable"], vec!["not_applicable"],
                        "covered")
                ]
            }));
        let mut request = request();
        request.available_state_fields.push("rho".into());
        request.selections.extend([
            SchemeSelection {
                module_id: "radar".into(),
                scheme_id: "reflectivity_dbz".into(),
            },
            SchemeSelection {
                module_id: "radar".into(),
                scheme_id: "radial_velocity".into(),
            },
        ]);
        let report = evaluate_preflight(&request, &catalog, identity()).unwrap();
        assert!(report.compatible);
        assert_eq!(
            report
                .resolved_selections
                .iter()
                .filter(|selection| selection.module_id == "radar")
                .count(),
            2
        );
        assert!(!report
            .hard_errors
            .iter()
            .any(|issue| issue.code == "duplicate_module_selection"));

        let mut qualified_zfp = request.clone();
        qualified_zfp.execution_mode = "qualified".into();
        qualified_zfp.maturity_limits_acknowledged = true;
        qualified_zfp.allow_backend_fallback = false;
        qualified_zfp.validation_guard_mode = "error".into();
        qualified_zfp.backend = "cpu".into();
        qualified_zfp.output_format = "zfp_high".into();
        qualified_zfp.interactions.surface_scheme = "none".into();
        qualified_zfp.interactions.surface_apply_fluxes = false;
        qualified_zfp.selections.retain(|selection| {
            selection.module_id == "radar" || selection.module_id == "dynamics"
        });
        let rejected = evaluate_preflight(&qualified_zfp, &catalog, identity()).unwrap();
        assert!(!rejected.compatible);
        let rejected_products: BTreeSet<&str> = rejected.hard_errors.iter()
            .filter(|issue| issue.code == "qualified_zfp_radar_product_not_ready")
            .filter_map(|issue| issue.scheme_id.as_deref())
            .collect();
        assert_eq!(rejected_products, BTreeSet::from(["reflectivity_dbz", "radial_velocity"]));

        qualified_zfp.selections.retain(|selection| selection.module_id != "radar");
        let non_radar = evaluate_preflight(&qualified_zfp, &catalog, identity()).unwrap();
        assert!(!non_radar.hard_errors.iter().any(|issue| {
            issue.code == "qualified_zfp_radar_product_not_ready"
        }));
    }

    #[test]
    fn known_but_unconstructible_scheme_is_not_selectable() {
        let mut request = request();
        request.selections = vec![SchemeSelection {
            module_id: "radiation".into(),
            scheme_id: "rrtmgp".into(),
        }];
        let report = evaluate_preflight(&request, &catalog(), identity()).unwrap();
        assert!(!report.compatible);
        assert!(report
            .hard_errors
            .iter()
            .any(|issue| issue.code == "scheme_not_constructible"));
        assert!(report
            .warnings
            .iter()
            .any(|issue| issue.code == "scheme_maturity"));
    }

    #[test]
    fn rejects_unknown_request_versions_before_process_launch() {
        let mut request = request();
        request.request_version = 1;
        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn rejects_invalid_grid_time_envelopes_before_process_launch() {
        let mut invalid_grid = request();
        invalid_grid.grid_metrics.dx = 0.0;
        assert!(validate_request(&invalid_grid).is_err());

        let mut invalid_substeps = request();
        invalid_substeps.time_control.acoustic_substeps = -1;
        assert!(validate_request(&invalid_substeps).is_err());

        let mut invalid_cadence = request();
        invalid_cadence.time_control.output_interval = f64::NAN;
        assert!(validate_request(&invalid_cadence).is_err());
    }

    #[test]
    fn generated_metal_v3_advisory_rejects_reviewed_numeric_neighbors() {
        let mut exact_catalog = catalog();
        let scope = &mut exact_catalog["runtimeCapabilities"]["interactionPolicy"]
            ["generatedMetalCanonicalScope"];
        scope["version"] = Value::from(3);
        scope["tupleCount"] = Value::from(6048);
        scope["distinctConfigCount"] = Value::from(4536);
        scope["outputFormats"] = serde_json::json!(["npy_3d", "netcdf", "zfp_high"]);
        scope["npyNetcdfManifestSha256"] =
            Value::String("30d5fcc9494016894733ef973909849f8c58fbd04b3e0d6bb2e5e497c0c9ca99".into());
        scope["zfpTypedOutputManifestSha256"] =
            Value::String("66de3a477f83cddd3fc921835900ada6d39eef070c4d43131402f0cd4898b2ab".into());
        scope["writeCadenceSecondsByFormat"] =
            serde_json::json!({"npy_3d": [1, 301], "netcdf": [1, 301], "zfp_high": [1]});
        let modules = exact_catalog["modules"].as_array_mut().unwrap();
        modules[0]["schemes"].as_array_mut().unwrap().push(scheme(
            "klemp_wilhelmson_split_cartesian_cgrid", vec![], "qualified", true,
            vec!["cartesian"], vec!["c_grid"], vec!["cpu", "gpu"],
            vec!["u", "v", "w"], vec!["parametric_cape"], vec!["wk_param"],
            vec!["warm_bubble"], "covered",
        ));
        for (module_id, scheme_id) in [
            ("advection", "tvd"),
            ("microphysics", "none"),
            ("terrain", "none"),
            ("time_stepping", "split_explicit"),
        ] {
            modules.push(serde_json::json!({
                "id": module_id,
                "selectionPath": format!("{module_id}.scheme"),
                "studioSelection": {"mode": "form", "fixedSchemeId": "", "reason": ""},
                "schemes": [scheme(
                    scheme_id, vec![], "qualified", true,
                    vec!["cartesian"], vec!["c_grid"], vec!["cpu", "gpu"],
                    vec![], vec!["not_applicable"], vec!["not_applicable"],
                    vec!["not_applicable"], "covered",
                )],
            }));
        }
        let mut exact_request = request();
        exact_request.execution_mode = "qualified".into();
        exact_request.maturity_limits_acknowledged = true;
        exact_request.backend = "metal".into();
        exact_request.allow_backend_fallback = false;
        exact_request.validation_guard_mode = "error".into();
        exact_request.boundary = BoundarySelection {
            west: "rigid".into(), east: "rigid".into(),
            south: "rigid".into(), north: "rigid".into(),
            bottom: "rigid_slip".into(), top: "rigid".into(),
        };
        exact_request.interactions.radiation_shortwave_enabled = false;
        exact_request.interactions.surface_scheme = "none".into();
        exact_request.interactions.surface_apply_fluxes = false;
        exact_request.selections = [
            ("dynamics", "klemp_wilhelmson_split_cartesian_cgrid"),
            ("advection", "tvd"),
            ("microphysics", "none"),
            ("terrain", "none"),
            ("time_stepping", "split_explicit"),
        ]
        .into_iter()
        .map(|(module_id, scheme_id)| SchemeSelection {
            module_id: module_id.into(), scheme_id: scheme_id.into(),
        })
        .collect();
        let generated_errors = |request: &ConfigPreflightRequest| {
            evaluate_preflight(request, &exact_catalog, identity())
                .unwrap()
                .hard_errors
                .into_iter()
                .filter(|issue| issue.code.starts_with("generated_metal_"))
                .map(|issue| issue.code)
                .collect::<Vec<_>>()
        };
        assert!(generated_errors(&exact_request).is_empty());

        let mut spacing = exact_request.clone();
        spacing.grid_metrics.dx = 501.0;
        assert_eq!(generated_errors(&spacing), ["generated_metal_grid_envelope"]);

        let mut timestep = exact_request.clone();
        timestep.time_control.dt_large = 0.2;
        assert_eq!(generated_errors(&timestep), ["generated_metal_time_envelope"]);

        let mut cadence = exact_request.clone();
        cadence.time_control.output_interval = 2.0;
        assert_eq!(generated_errors(&cadence), ["generated_metal_output_cadence"]);

        let mut zfp_cadence = exact_request;
        zfp_cadence.output_format = "zfp_high".into();
        zfp_cadence.time_control.output_interval = 301.0;
        assert_eq!(generated_errors(&zfp_cadence), ["generated_metal_output_cadence"]);
    }

    fn east_v4_catalog() -> Value {
        let mut exact = catalog();
        let policy = &mut exact["runtimeCapabilities"]["interactionPolicy"];
        policy["version"] = Value::from(10);
        let scope = &mut policy["generatedMetalCanonicalScope"];
        scope["version"] = Value::from(4);
        scope["tupleCount"] = Value::from(6049);
        scope["distinctConfigCount"] = Value::from(4537);
        scope["outputFormats"] = serde_json::json!(["npy_3d", "netcdf", "zfp_high"]);
        scope["npyNetcdfManifestSha256"] = Value::String(
            "30d5fcc9494016894733ef973909849f8c58fbd04b3e0d6bb2e5e497c0c9ca99".into());
        scope["zfpTypedOutputManifestSha256"] = Value::String(
            "89ef136c9637c6e23d17cd996af20631973286447b783f1978d1471d4982c9fc".into());
        scope["manifestSha256"] = Value::String(
            "062b609452e090b0b6e5ff1f7bb15cab896700f193796ff4d87a600994f28967".into());
        scope["writeCadenceSecondsByFormat"] = serde_json::json!({
            "npy_3d": [1, 301], "netcdf": [1, 301], "zfp_high": [1]});
        scope["additionalBoundaryNpyScope"] = serde_json::json!({
            "version": 1,
            "profileCount": 1,
            "familyId": "boundary:2b5bfbe6dfa466c6baf991e3",
            "strictCanonicalYamlSha256":
                "a244b58833f7e3cd7060d941e8453d4e5face3dff9e18e87ebf61dd581a24d55",
            "outputFormats": ["npy_3d"],
            "writeCadenceSeconds": [1],
            "manifestSha256":
                "26712e5d5714b313cc1e17f348d67b9db9eeea60e5b888d56b0b7cf3cc5673a9"
        });
        exact
    }

    fn radar_policy11_catalog() -> Value {
        let mut exact = east_v4_catalog();
        let policy = &mut exact["runtimeCapabilities"]["interactionPolicy"];
        policy["version"] = Value::from(11);
        policy["qualifiedZfpFieldBudgets"]["qc"] = serde_json::json!(1e-8);
        policy["qualifiedZfpFieldUnits"]["qc"] = serde_json::json!("kg/kg");
        policy["qualifiedZfpReversibleRadarV4"] = serde_json::json!({
            "version": 1,
            "admitted": false,
            "productIds": ["reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv"],
            "nonemptyProductSubsets": 31,
            "coreFields": ["u", "v", "w", "theta", "p", "rho", "qv", "qr"],
            "presets": ["minimal", "radar_ready", "radar_polarimetric_ready"],
            "namedPresetProductSets": {
                "radar_ready": ["reflectivity_dbz", "radial_velocity"],
                "radar_polarimetric_ready": ["zdr", "kdp", "rhohv"]
            },
            "polarimetricExtraFields": ["qc"],
            "outputFormatTiers": ["zfp_low", "zfp_medium", "zfp_high"],
            "backend": "cpu",
            "coordinateSystem": "cartesian",
            "staggering": "c_grid",
            "terrain": "none",
            "domainMotion": "stationary",
            "nesting": false,
            "reviewedSourceProfileContract": "nimbus.qualified-radar-source-profile.v2",
            "reviewedSourceProfileOwnerSymbol": "is_public_qualified_radar_profile",
            "reviewedSourceProfileFamilies": [
                "lin_cloudy_radar_ready_v1", "kessler_warm_cloud_radar_v1"],
            "asyncIoModes": [false, true],
            "gridShape": [12, 12, 48],
            "gridSpacingMeters": [1000, 1000, 250],
            "dtLargeSeconds": 0.2,
            "acousticSubsteps": 6,
            "writeCadenceSeconds": [1, 4],
            "nonRadarMode": "accuracy",
            "nonRadarContainerVersion": 3,
            "nonRadarBudgetOwner": "qualifiedZfpFieldBudgets",
            "radarMode": "reversible",
            "radarContainerVersion": 4,
            "radarKeyframeEveryExport": true,
            "radarTolerancePolicy": "forbidden",
            "manifestContract": "nimbus.zfp.per_field.v2",
            "sourceFieldBoundsContract": "registered_field_contract_v1"
        });
        exact
    }

    fn vertical_policy12_catalog() -> Value {
        let mut exact = radar_policy11_catalog();
        let policy = &mut exact["runtimeCapabilities"]["interactionPolicy"];
        policy["version"] = Value::from(12);
        policy["turbulenceDynamicCoefficientDefault"] = Value::Bool(false);
        policy["qualifiedVerticalOnlySgs"] = serde_json::json!({
            "version": 1,
            "admitted": false,
            "scheme": "smagorinsky",
            "mode": "vertical_only",
            "dynamicCoefficient": false,
            "executionMode": "qualified",
            "backend": "cpu",
            "coordinateSystem": "cartesian",
            "staggering": "c_grid",
            "terrain": "none",
            "domainMotion": "stationary",
            "nesting": false,
            "lateralBoundaryMode": "periodic",
            "activeBoundaryLayerAllowed": false,
            "gridShape": [12, 12, 16],
            "gridSpacingMeters": [1000, 1000, 250],
            "dtLargeSeconds": 0.2,
            "acousticSubsteps": 6,
            "outputFormat": "npy_3d",
            "writeCadenceSeconds": [1],
            "sourceProfileContract": "nimbus.static-vertical-sgs-cpu.v1",
            "sourceProfileOwnerSymbol": "is_transactional_static_vertical_only_cpu_candidate",
            "wholeStepRollback": "state_and_six_science_ledgers",
            "dynamicNonThreeDAllowed": false
        });
        exact
    }

    fn milbrandt_policy13_catalog() -> Value {
        let mut exact = vertical_policy12_catalog();
        let policy = &mut exact["runtimeCapabilities"]["interactionPolicy"];
        policy["version"] = Value::from(13);
        policy["milbrandtTripleMoment"] = serde_json::json!({
            "version": 1,
            "retained": true,
            "admitted": false,
            "implementationState": "engineering_incomplete",
            "selector": "microphysics.milbrandt.triple_moment",
            "strictV1ExplicitKeyAdmitted": false,
            "scheme": "milbrandt",
            "baselineState": ["q", "N"],
            "additionalState": ["Z"],
            "additionalMomentOrder": 6,
            "additionalMomentSpecies": ["rain", "ice", "snow", "graupel", "hail"],
            "privateStorageUnits": "m^6/kg_dry"
        });
        exact["readiness"]["items"].as_array_mut().unwrap().push(serde_json::json!({
            "id": "microphysics.milbrandt_triple_moment",
            "area": "microphysics",
            "scope": "Retained Milbrandt-Yau triple-moment selection",
            "state": "engineering_incomplete",
            "boundedEvidence": "focused implementation evidence",
            "campaignEvidence": "",
            "externalReviewEvidence": "",
            "blocker": "Five reflectivity moment process, transport, and backend owners remain."
        }));
        exact
    }

    fn policy14_catalog() -> Value {
        serde_json::from_str(include_str!("../../../src/lib/config/nativeCatalogSnapshot.json")).unwrap()
    }

    fn policy14_mixed_request() -> ConfigPreflightRequest {
        let mut selected = request();
        selected.execution_mode = "qualified".into();
        selected.backend = "metal".into();
        selected.allow_backend_fallback = false;
        selected.validation_guard_mode = "error".into();
        selected.boundary = BoundarySelection {
            west: "rigid_slip".into(), east: "rigid_slip".into(),
            south: "rigid_slip".into(), north: "rigid_slip".into(),
            bottom: "rigid_noslip".into(), top: "open".into(),
        };
        selected.output.keep_last_frames = 0;
        selected.output.minimum_free_space_mib = 0;
        selected.output.maximum_retained_history_files = 0;
        selected.diagnostics.scientific_every_steps = 1;
        selected.diagnostics.deterministic_reductions = true;
        selected.interactions.surface_scheme = "none".into();
        selected.interactions.surface_apply_fluxes = false;
        selected.available_state_fields = policy14_catalog()["runtimeStateFields"].as_array().unwrap()
            .iter().map(|field| field.as_str().unwrap().to_owned()).collect();
        selected.selections = [
            ("dynamics", "klemp_wilhelmson_split_cartesian_cgrid"),
            ("advection", "tvd"), ("microphysics", "none"),
            ("terrain", "none"), ("time_stepping", "split_explicit"),
        ].into_iter().map(|(module_id, scheme_id)| SchemeSelection {
            module_id: module_id.into(), scheme_id: scheme_id.into(),
        }).collect();
        selected
    }

    #[test]
    fn policy14_mixed_scope_requires_exact_keys_hashes_counts_and_bounds() {
        let valid = policy14_catalog();
        assert!(parse_catalog(&serde_json::to_vec(&valid).unwrap()).is_ok());
        let mut historical = valid.clone();
        historical["runtimeCapabilities"]["interactionPolicy"]["version"] = Value::from(13);
        assert!(parse_catalog(&serde_json::to_vec(&historical).unwrap()).is_err());
        historical["runtimeCapabilities"]["interactionPolicy"].as_object_mut().unwrap()
            .remove("qualifiedFlatMixedMetalNpyScope");
        assert!(parse_catalog(&serde_json::to_vec(&historical).unwrap()).is_ok());
        let mut missing = valid.clone();
        missing["runtimeCapabilities"]["interactionPolicy"].as_object_mut().unwrap()
            .remove("qualifiedFlatMixedMetalNpyScope");
        assert!(parse_catalog(&serde_json::to_vec(&missing).unwrap()).is_err());
        for (field, value) in [
            ("profileCount", serde_json::json!(2)),
            ("familyId", serde_json::json!("boundary:orphan")),
            ("strictCanonicalYamlSha256", serde_json::json!("0".repeat(64))),
            ("manifestSha256", serde_json::json!("0".repeat(64))),
            ("writeCadenceSeconds", serde_json::json!([1, 301])),
            ("outputFormats", serde_json::json!(["npy_3d", "netcdf"])),
            ("gridShape", serde_json::json!([18, 18, 20])),
            ("dtLargeSeconds", serde_json::json!(0.2)),
            ("acousticSubsteps", serde_json::json!(5)),
            ("lateralSpongeEnabled", serde_json::json!(false)),
            ("lateralSpongeWidth", serde_json::json!(0)),
            ("lateralSpongeAlpha", serde_json::json!(0)),
            ("scalarTransportOwner", serde_json::json!("cpu")),
            ("initialConditionChaosOwner", serde_json::json!("native")),
            ("lateralDampingOwner", serde_json::json!("native")),
            ("effectiveWOnlySponge", serde_json::json!(true)),
            ("unreviewed", serde_json::json!(true)),
        ] {
            let mut invalid = valid.clone();
            invalid["runtimeCapabilities"]["interactionPolicy"]["qualifiedFlatMixedMetalNpyScope"][field] = value;
            assert!(parse_catalog(&serde_json::to_vec(&invalid).unwrap()).is_err(), "{field}");
        }
    }

    #[test]
    fn policy14_mixed_advisory_does_not_inherit_301_or_other_output_owners() {
        let catalog = policy14_catalog();
        let baseline = policy14_mixed_request();
        let errors = |selected: &ConfigPreflightRequest| {
            evaluate_preflight(selected, &catalog, identity()).unwrap().hard_errors.into_iter()
                .filter(|issue| issue.code.starts_with("generated_metal_"))
                .map(|issue| issue.code).collect::<Vec<_>>()
        };
        assert!(errors(&baseline).is_empty());
        for cadence in [301.0, 2.0, 0.0] {
            let mut selected = baseline.clone();
            selected.time_control.output_interval = cadence;
            assert_eq!(errors(&selected), ["generated_metal_output_cadence"]);
        }
        let mut selected = baseline.clone();
        selected.output_format = "netcdf".into();
        assert_eq!(errors(&selected), ["generated_metal_mixed_output_format"]);
        selected = baseline.clone();
        selected.output.async_io = true;
        assert_eq!(errors(&selected), ["generated_metal_mixed_async_io"]);
        selected = baseline.clone();
        selected.output.keep_last_frames = 1;
        assert_eq!(errors(&selected), ["generated_metal_mixed_history"]);
    }

    #[test]
    fn policy14_metal_original_source_authority_keeps_other_errors_on_native_acceptance() {
        let catalog = policy14_catalog();
        let mut selected = policy14_mixed_request();
        let descriptor = evaluate_preflight(&selected, &catalog, identity()).unwrap();
        assert!(!descriptor.compatible);
        assert!(descriptor.hard_errors.iter().any(|issue|
            issue.code == "qualified_generated_metal_exact_document_required"));
        selected.source = PreflightSource {
            kind: "exact_document".into(), yaml_text: Some("schema: nimbus.v1\n".into()),
            yaml_path: Some("/tmp/mixed-original.yaml".into()),
        };
        let mut report = evaluate_preflight(&selected, &catalog, identity()).unwrap();
        assert!(report.hard_errors.iter().any(|issue|
            issue.code == "qualified_generated_metal_native_admission_required"));
        let before: Vec<_> = report.hard_errors.iter().filter(|issue|
            issue.code != "qualified_generated_metal_native_admission_required").cloned().collect();
        apply_native_validation(&mut report, NativeValidationOutcome {
            accepted: true, source_sha256: "a".repeat(64), source_fingerprint: "fnv1a64:source".into(),
            source_path_fingerprint: "fnv1a64:path".into(), binary_sha256: "sha256:binary".into(),
            exit_code: Some(0), diagnostic: String::new(),
        });
        assert_eq!(report.hard_errors, before);
        assert_eq!(report.native_admission.state, "accepted");
        selected.time_control.output_interval = 301.0;
        let mut rejected = evaluate_preflight(&selected, &catalog, identity()).unwrap();
        apply_native_validation(&mut rejected, NativeValidationOutcome {
            accepted: true, source_sha256: "a".repeat(64), source_fingerprint: "fnv1a64:source".into(),
            source_path_fingerprint: "fnv1a64:path".into(), binary_sha256: "sha256:binary".into(),
            exit_code: Some(0), diagnostic: String::new(),
        });
        assert!(!rejected.compatible);
        assert!(rejected.hard_errors.iter().any(|issue| issue.code == "generated_metal_output_cadence"));
        for (key, value) in [("lateral_sponge_enabled", "false"), ("lateral_width", "0"),
                             ("lateral_alpha", "0"), ("slow_rk2", "false")] {
            selected.source.yaml_text = Some(format!("schema: nimbus.v1\n# {key}: {value}\n"));
            let mut rejected = evaluate_preflight(&selected, &catalog, identity()).unwrap();
            apply_native_validation(&mut rejected, NativeValidationOutcome {
                accepted: false, source_sha256: "b".repeat(64), source_fingerprint: "fnv1a64:mutated".into(),
                source_path_fingerprint: "fnv1a64:path".into(), binary_sha256: "sha256:binary".into(),
                exit_code: Some(2), diagnostic: format!("typed source control {key} rejected"),
            });
            assert!(!rejected.compatible);
            assert_eq!(rejected.native_admission.state, "rejected");
        }
    }

    fn native_context_test_value(report: &ConfigPreflightReport) -> Value {
        serde_json::json!({
            "schema": "nimbus.validate-only-context.v1", "version": 1,
            "sourceKind": "exact_document", "sourceSha256": "a".repeat(64),
            "generatedMetalScopeKind": "mixed_npy", "context": report.context,
            "selectedSchemes": {
                "dynamics": "klemp_wilhelmson_split_cartesian_cgrid", "microphysics": "none",
                "radiation": "none", "boundary_layer": "none", "turbulence": "none",
                "advection": "tvd", "diffusion": "explicit", "time_stepping": "split_explicit",
                "chaos": "initial_conditions", "terrain": "none"
            },
            "radarProducts": [],
            "unserializedControls": {
                "lateralSpongeEnabled": true, "lateralWidth": 20, "lateralAlpha": 0.05,
                "wOnlyTopSponge": true, "wOnlyDepthCells": 8, "wOnlyAlphaMax": 0.2,
                "slowRk2": true, "lidBcGamma": -1.0, "axisUM1Filter": false
            }
        })
    }

    #[test]
    fn native_context_parser_rejects_missing_duplicate_unknown_hash_and_bounds() {
        let catalog = policy14_catalog();
        let report = evaluate_preflight(&policy14_mixed_request(), &catalog, identity()).unwrap();
        let value = native_context_test_value(&report);
        let line = format!("[VALIDATE_CONTEXT] {value}");
        assert!(parse_native_validate_only_context(&line, &"a".repeat(64)).is_ok());
        assert!(parse_native_validate_only_context("[VALIDATE] PASS", &"a".repeat(64)).is_err());
        assert!(parse_native_validate_only_context(&format!("{line}\n{line}"), &"a".repeat(64)).is_err());
        assert!(parse_native_validate_only_context(&line, &"b".repeat(64)).is_err());
        let duplicate = line.replacen("\"schema\":", "\"version\":1,\"schema\":", 1);
        assert!(parse_native_validate_only_context(&duplicate, &"a".repeat(64)).is_err());
        let duplicate_map = line.replacen("\"selectedSchemes\":{", "\"selectedSchemes\":{\"terrain\":\"none\",", 1);
        assert!(parse_native_validate_only_context(&duplicate_map, &"a".repeat(64)).is_err());
        for pointer in ["/unexpected", "/context/unexpected", "/context/gridMetrics/unexpected",
                        "/unserializedControls/unexpected", "/selectedSchemes/unexpected"] {
            let mut invalid = value.clone();
            let (parent, field) = pointer.rsplit_once('/').unwrap();
            invalid.pointer_mut(parent).unwrap().as_object_mut().unwrap().insert(field.into(), Value::Bool(true));
            assert!(parse_native_validate_only_context(&format!("[VALIDATE_CONTEXT] {invalid}"), &"a".repeat(64)).is_err(), "{pointer}");
        }
        let mut missing = value.clone();
        missing["context"].as_object_mut().unwrap().remove("backend");
        assert!(parse_native_validate_only_context(&format!("[VALIDATE_CONTEXT] {missing}"), &"a".repeat(64)).is_err());
        let oversized = format!("[VALIDATE_CONTEXT] {}", " ".repeat(64 * 1024 + 1));
        assert!(parse_native_validate_only_context(&oversized, &"a".repeat(64)).is_err());
        let exact = line.replacen("\"rho\":1e-6", "\"rho\":9.9999999999999995e-07", 1);
        assert_ne!(exact, line);
        let receipt = parse_native_validate_only_context(&exact, &"a".repeat(64)).unwrap();
        assert_eq!(receipt.context.output.zfp_field_tolerances["rho"].to_bits(), (1e-6_f64).to_bits());
        let neighboring = exact.replacen("9.9999999999999995e-07", "1.0000000000000002e-06", 1);
        let receipt = parse_native_validate_only_context(&neighboring, &"a".repeat(64)).unwrap();
        assert_eq!(receipt.context.output.zfp_field_tolerances["rho"].to_bits(), (1e-6_f64).to_bits() + 1);
        assert_eq!(serde_json::from_str::<f64>("0.10000000000000001").unwrap(), 0.1);
        assert_eq!(serde_json::from_str::<f64>("0.050000000000000003").unwrap(), 0.05);
    }

    #[test]
    fn native_context_mismatches_never_clear_pending_authority_or_other_errors() {
        let catalog = policy14_catalog();
        let mut selected = policy14_mixed_request();
        selected.source = PreflightSource {
            kind: "exact_document".into(), yaml_text: Some("schema: nimbus.v1\n".into()),
            yaml_path: Some("/tmp/original.yaml".into()),
        };
        let baseline = evaluate_preflight(&selected, &catalog, identity()).unwrap();
        let value = native_context_test_value(&baseline);
        let accepted = || NativeValidationOutcome {
            accepted: true, source_sha256: "a".repeat(64), source_fingerprint: "fnv1a64:source".into(),
            source_path_fingerprint: "fnv1a64:path".into(), binary_sha256: "sha256:binary".into(),
            exit_code: Some(0), diagnostic: String::new(),
        };
        for (pointer, replacement) in [
            ("/context/executionMode", serde_json::json!("exploratory")),
            ("/context/backend", serde_json::json!("cpu")),
            ("/context/timeControl/dtLarge", serde_json::json!(0.2)),
            ("/context/gridMetrics/nx", serde_json::json!(20)),
            ("/context/output/asyncIo", serde_json::json!(true)),
            ("/context/output/zfpFieldTolerances/rho", serde_json::json!(f64::from_bits((1e-6_f64).to_bits() + 1))),
            ("/selectedSchemes/advection", serde_json::json!("weno5")),
            ("/generatedMetalScopeKind", serde_json::json!("east_npy")),
            ("/unserializedControls/slowRk2", serde_json::json!(false)),
        ] {
            let mut changed = value.clone();
            *changed.pointer_mut(pointer).unwrap() = replacement;
            let receipt = parse_native_validate_only_context(&format!("[VALIDATE_CONTEXT] {changed}"), &"a".repeat(64)).unwrap();
            let mut report = baseline.clone();
            apply_native_context_validation(&mut report, accepted(), Some(receipt), &catalog).unwrap();
            assert!(!report.compatible, "{pointer}");
            assert!(report.hard_errors.iter().any(|issue| issue.code == "native_source_context_mismatch"));
            assert!(report.hard_errors.iter().any(|issue| issue.code == "qualified_generated_metal_native_admission_required"));
            assert_eq!(report.native_admission.state, "accepted");
        }
        let mut report = baseline.clone();
        assert!(apply_native_context_validation(&mut report, accepted(), None, &catalog).is_err());
        let before: Vec<_> = report.hard_errors.iter().filter(|issue|
            issue.code != "qualified_generated_metal_native_admission_required").cloned().collect();
        let receipt = parse_native_validate_only_context(&format!("[VALIDATE_CONTEXT] {value}"), &"a".repeat(64)).unwrap();
        apply_native_context_validation(&mut report, accepted(), Some(receipt), &catalog).unwrap();
        assert_eq!(report.hard_errors, before);
    }

    #[test]
    #[ignore = "requires hash-bound context-capable policy14 simulator; native validation only, no GPU"]
    fn policy14_native_context_binds_actual_source_and_rejects_exploratory_request_mismatch() {
        let binary = PathBuf::from(std::env::var("NIMBUS_CONTEXT_PREFLIGHT_BINARY").expect("context binary"));
        let expected = format!("sha256:{}", std::env::var("NIMBUS_CONTEXT_PREFLIGHT_BINARY_SHA256")
            .expect("context binary hash"));
        let output = Command::new(&binary).arg("--scheme-catalog=json").output().unwrap();
        assert!(output.status.success());
        let catalog = parse_catalog(&output.stdout).unwrap();
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap();
        let path = root.join("tests/fixtures/reference/catalog/flat_mixed_boundary_metal_exploratory_v1.yaml");
        let exploratory = fs::read_to_string(&path).unwrap();
        let qualified = exploratory.replacen("    mode: exploratory\n", "    mode: qualified\n", 1);
        let source = |text: String| PreflightSource {
            kind: "exact_document".into(), yaml_text: Some(text), yaml_path: Some(path.display().to_string()),
        };
        let (positive, receipt) = validate_exact_document_with_context(&binary, &expected, &source(qualified.clone()), true).unwrap();
        assert!(positive.accepted);
        let native_context = receipt.as_ref().unwrap().context.clone();
        let mut selected = policy14_mixed_request();
        selected.execution_mode = native_context.execution_mode;
        selected.maturity_limits_acknowledged = native_context.maturity_limits_acknowledged;
        selected.backend = native_context.backend;
        selected.allow_backend_fallback = native_context.allow_backend_fallback;
        selected.validation_guard_mode = native_context.validation_guard_mode;
        selected.gpu_apply_microphysics = native_context.gpu_apply_microphysics;
        selected.gpu_apply_slow = native_context.gpu_apply_slow;
        selected.gpu_apply_sgs = native_context.gpu_apply_sgs;
        selected.output_format = native_context.output_format;
        selected.grid_metrics = native_context.grid_metrics;
        selected.time_control = native_context.time_control;
        selected.output = native_context.output;
        selected.coordinate_system = native_context.coordinate_system;
        selected.staggering = native_context.staggering;
        selected.boundary = native_context.boundary;
        selected.diagnostics = native_context.diagnostics;
        selected.initialization = native_context.initialization;
        selected.interactions = native_context.interactions;
        selected.available_state_fields = native_context.available_state_fields;
        selected.source = source(qualified);
        let mut report = evaluate_preflight(&selected, &catalog, identity()).unwrap();
        apply_native_context_validation(&mut report, positive, receipt, &catalog).unwrap();
        assert!(report.compatible, "{:?}", report.hard_errors);
        selected.source = source(exploratory);
        let (native, receipt) = validate_exact_document_with_context(&binary, &expected, &selected.source, true).unwrap();
        assert!(native.accepted);
        let mut mismatched = evaluate_preflight(&selected, &catalog, identity()).unwrap();
        apply_native_context_validation(&mut mismatched, native, receipt, &catalog).unwrap();
        assert!(!mismatched.compatible);
        assert!(mismatched.hard_errors.iter().any(|issue| issue.code == "native_source_context_mismatch"));
        assert!(mismatched.hard_errors.iter().any(|issue| issue.code == "qualified_generated_metal_native_admission_required"));
        assert_eq!(crate::platform::sha256_file(&binary).unwrap(), expected);
    }

    #[test]
    #[ignore = "requires hash-bound context-capable policy14 simulator; native validation only, no GPU"]
    fn policy14_actual_native_context_profiles_aliases_and_declared_mismatches() {
        let binary = PathBuf::from(std::env::var("NIMBUS_CONTEXT_PREFLIGHT_BINARY").expect("context binary"));
        let expected = format!("sha256:{}", std::env::var("NIMBUS_CONTEXT_PREFLIGHT_BINARY_SHA256")
            .expect("context binary hash"));
        assert_eq!(crate::platform::sha256_file(&binary).unwrap(), expected);
        let catalog_output = Command::new(&binary).arg("--scheme-catalog=json").output().unwrap();
        assert!(catalog_output.status.success());
        let catalog = parse_catalog(&catalog_output.stdout).unwrap();
        assert_eq!(catalog["runtimeCapabilities"]["interactionPolicy"]["version"], 14);
        let executable = executable_identity(&binary, &catalog_output.stdout).unwrap();
        assert_eq!(executable.binary_fingerprint, expected);
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap();
        let request_for = |path: &Path, text: String, scope: &str, format: &str| {
            let mut selected = policy14_mixed_request();
            selected.maturity_limits_acknowledged = true;
            selected.interactions.turbulence_mode = "2d".into();
            selected.output_format = format.into();
            selected.output.zfp_per_field_tolerances = format == "zfp_high";
            if format == "zfp_high" {
                selected.output.async_io = true;
                selected.output.keep_last_frames = 3;
                selected.output.minimum_free_space_mib = 4096;
                selected.output.maximum_retained_history_files = 256;
            } else {
                selected.output.zfp_field_tolerances.clear();
            }
            if scope == "canonical" {
                selected.boundary = BoundarySelection {
                    west: "rigid".into(), east: "rigid".into(),
                    south: "rigid".into(), north: "rigid".into(),
                    bottom: "rigid_slip".into(), top: "rigid".into(),
                };
            } else if scope == "east_npy" {
                selected.boundary.east = "radiative".into();
            }
            selected.available_state_fields = catalog_runtime_state_fields(&catalog).unwrap()
                .into_iter().collect();
            selected.selections = [
                ("dynamics", "klemp_wilhelmson_split_cartesian_cgrid"),
                ("microphysics", "none"), ("radiation", "none"),
                ("boundary_layer", "none"), ("turbulence", "none"),
                ("advection", "tvd"), ("diffusion", "explicit"),
                ("time_stepping", "split_explicit"), ("chaos", "initial_conditions"),
                ("terrain", "none"),
            ].into_iter().map(|(module_id, scheme_id)| SchemeSelection {
                module_id: module_id.into(), scheme_id: scheme_id.into(),
            }).collect();
            selected.source = PreflightSource {
                kind: "exact_document".into(), yaml_text: Some(text),
                yaml_path: Some(path.display().to_string()),
            };
            selected
        };
        let profiles = [
            ("generated_metal_canonical_v1.yaml", "canonical", "npy_3d"),
            ("generated_metal_netcdf_v1.yaml", "canonical", "netcdf"),
            ("generated_metal_zfp_candidate_v1.yaml", "canonical", "zfp_high"),
            ("flat_east_radiative_metal_npy_candidate_v1.yaml", "east_npy", "npy_3d"),
            ("flat_mixed_boundary_metal_exploratory_v1.yaml", "mixed_npy", "npy_3d"),
        ];
        let mut original_sources = Vec::new();
        for (name, scope, format) in profiles {
            let path = root.join("tests/fixtures/reference/catalog").join(name);
            let original = fs::read_to_string(&path).unwrap();
            let qualified = original.replacen("    mode: exploratory\n", "    mode: qualified\n", 1);
            assert!(qualified.contains("    mode: qualified\n"), "{name}");
            let selected = request_for(&path, qualified, scope, format);
            validate_request(&selected).unwrap();
            let (native, receipt) = validate_exact_document_with_context(
                &binary, &expected, &selected.source, true).unwrap();
            assert!(native.accepted, "{name}: {}", native.diagnostic);
            assert_eq!(receipt.as_ref().unwrap().generated_metal_scope_kind, scope, "{name}");
            let native_output = receipt.as_ref().unwrap().context.output.clone();
            let source_sha256 = format!("{:x}", Sha256::digest(selected.source.yaml_text.as_ref().unwrap().as_bytes()));
            let mut report = evaluate_preflight(&selected, &catalog, executable.clone()).unwrap();
            assert!(report.hard_errors.iter().any(|issue|
                issue.code == "qualified_generated_metal_native_admission_required"), "{name}");
            apply_native_context_validation(&mut report, native, receipt, &catalog).unwrap();
            assert!(report.compatible, "{name}: native output={native_output:?}; requested output={:?}; errors={:?}",
                report.context.output, report.hard_errors);
            assert_eq!(report.native_admission.state, "accepted", "{name}");
            assert_eq!(report.native_admission.source_sha256.as_deref(), Some(source_sha256.as_str()), "{name}");
            assert_eq!(report.native_admission.validated_binary_sha256.as_deref(), Some(expected.as_str()), "{name}");
            let fixture_sha256 = format!("{:x}", Sha256::digest(original.as_bytes()));
            eprintln!("[ACTUAL_CONTEXT] profile={name} scope={scope} format={format} sourceSha256={source_sha256} fixtureSha256={fixture_sha256}");
            original_sources.push((path, original));
        }
        let (path, original) = &original_sources[0];
        let canonical = request_for(path, original.clone(), "canonical", "npy_3d");
        let mut aliases = canonical.clone();
        aliases.execution_mode = " QUALIFIED ".into();
        aliases.backend = " METAL ".into();
        aliases.validation_guard_mode = " ERROR ".into();
        aliases.output_format = " NPY-3D ".into();
        aliases.coordinate_system = " CARTESIAN ".into();
        aliases.staggering = " C-GRID ".into();
        aliases.output.preset = " MINIMAL ".into();
        aliases.output.zfp_mode = " ACCURACY ".into();
        aliases.output.zfp_field_execution = " SERIAL ".into();
        aliases.boundary = BoundarySelection {
            west: " RIGID ".into(), east: " RIGID ".into(),
            south: " RIGID ".into(), north: " RIGID ".into(),
            bottom: " RIGID-SLIP ".into(), top: " RIGID ".into(),
        };
        aliases.initialization = InitializationSelection {
            sounding_source: " CAPE ".into(), hodograph_source: " WK ".into(),
            trigger_source: " WARM-BUBBLE ".into(),
        };
        aliases.interactions.turbulence_mode = " 2D ".into();
        aliases.interactions.surface_scheme = " NONE ".into();
        aliases.interactions.diffusion_target = " SCALARS ".into();
        aliases.available_state_fields.reverse();
        for selection in &mut aliases.selections {
            selection.scheme_id = match selection.module_id.as_str() {
                "time_stepping" => " Klemp-Wilhelmson ".into(),
                "chaos" => " IC ".into(),
                "terrain" => " FLAT ".into(),
                "radiation" | "boundary_layer" | "turbulence" => " OFF ".into(),
                _ => selection.scheme_id.to_ascii_uppercase().replace('_', "-"),
            };
            selection.module_id = selection.module_id.to_ascii_uppercase().replace('_', "-");
        }
        validate_request(&aliases).unwrap();
        let (native, receipt) = validate_exact_document_with_context(
            &binary, &expected, &aliases.source, true).unwrap();
        assert!(native.accepted, "alias source: {}", native.diagnostic);
        let mut report = evaluate_preflight(&aliases, &catalog, executable.clone()).unwrap();
        apply_native_context_validation(&mut report, native, receipt, &catalog).unwrap();
        assert!(report.compatible, "supported aliases: {:?}", report.hard_errors);
        eprintln!("[ACTUAL_CONTEXT] aliases=accepted sourceSha256={}", report.native_admission.source_sha256.as_deref().unwrap());
        for field in ["backend", "grid", "dt", "output", "chaos"] {
            let mut selected = canonical.clone();
            match field {
                "backend" => selected.backend = "cpu".into(),
                "grid" => selected.grid_metrics.dx = 501.0,
                "dt" => selected.time_control.dt_large = 0.2,
                "output" => selected.output_format = "netcdf".into(),
                "chaos" => selected.selections.iter_mut().find(|selection|
                    selection.module_id == "chaos").unwrap().scheme_id = "none".into(),
                _ => unreachable!(),
            }
            validate_request(&selected).unwrap();
            let (native, receipt) = validate_exact_document_with_context(
                &binary, &expected, &selected.source, true).unwrap();
            assert!(native.accepted, "{field}: {}", native.diagnostic);
            let mut report = evaluate_preflight(&selected, &catalog, executable.clone()).unwrap();
            let existing = report.hard_errors.clone();
            let pending_before = existing.iter().filter(|issue|
                issue.code == "qualified_generated_metal_native_admission_required").count();
            if field != "backend" {
                assert_eq!(pending_before, 1, "{field}");
            }
            apply_native_context_validation(&mut report, native, receipt, &catalog).unwrap();
            assert!(!report.compatible, "{field}");
            assert_eq!(report.native_admission.state, "accepted", "{field}");
            assert_eq!(report.hard_errors.len(), existing.len() + 1, "{field}: {:?}", report.hard_errors);
            assert!(existing.iter().all(|issue| report.hard_errors.contains(issue)), "{field}");
            assert_eq!(report.hard_errors.iter().filter(|issue|
                issue.code == "qualified_generated_metal_native_admission_required").count(), pending_before, "{field}");
            let mismatch = report.hard_errors.iter().find(|issue|
                issue.code == "native_source_context_mismatch").unwrap();
            assert_eq!(mismatch.actual.as_deref(), Some(if field == "chaos" {
                "context=false,selection=true,scope=false"
            } else {
                "context=true,selection=false,scope=false"
            }), "{field}");
            eprintln!("[ACTUAL_CONTEXT] declaredMismatch={field} sourceSha256={} pendingPreserved={pending_before} existingErrorsPreserved={} mismatch={}",
                report.native_admission.source_sha256.as_deref().unwrap(), existing.len(), mismatch.actual.as_deref().unwrap());
        }
        for (path, original) in original_sources {
            assert_eq!(fs::read_to_string(&path).unwrap(), original, "{}", path.display());
        }
        let catalog_after = Command::new(&binary).arg("--scheme-catalog=json").output().unwrap();
        assert!(catalog_after.status.success());
        assert_eq!(catalog_after.stdout, catalog_output.stdout);
        assert_eq!(crate::platform::sha256_file(&binary).unwrap(), expected);
        eprintln!("actual native context: 5 qualified profiles, 1 alias profile, 5 declared mismatches passed; binarySha256={expected}");
    }

    #[test]
    #[ignore = "requires hash-bound post-repair policy14 simulator; native validation only, no GPU"]
    fn policy14_original_nine_controls_require_native_rejection() {
        let binary = PathBuf::from(std::env::var("NIMBUS_POLICY14_PREFLIGHT_BINARY").expect("policy14 binary"));
        let expected = format!("sha256:{}", std::env::var("NIMBUS_POLICY14_PREFLIGHT_BINARY_SHA256")
            .expect("policy14 binary hash"));
        assert_eq!(crate::platform::sha256_file(&binary).unwrap(), expected);
        let catalog_bytes = Command::new(&binary).arg("--scheme-catalog=json").output().unwrap();
        assert!(catalog_bytes.status.success());
        let catalog = parse_catalog(&catalog_bytes.stdout).unwrap();
        assert_eq!(catalog["runtimeCapabilities"]["interactionPolicy"]["version"], 14);
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap();
        for name in ["generated_metal_canonical_v1.yaml", "generated_metal_netcdf_v1.yaml",
                     "generated_metal_zfp_candidate_v1.yaml", "flat_east_radiative_metal_npy_candidate_v1.yaml"] {
            let path = root.join("tests/fixtures/reference/catalog").join(name);
            let original = fs::read_to_string(&path).unwrap()
                .replacen("    mode: exploratory\n", "    mode: qualified\n", 1);
            assert!(original.contains("    mode: qualified\n"), "{name}");
            let source = |text: String| PreflightSource {
                kind: "exact_document".into(), yaml_text: Some(text),
                yaml_path: Some(path.display().to_string()),
            };
            assert!(validate_exact_document(&binary, &expected, &source(original.clone())).unwrap().accepted, "{name}");
            for (section, field, value) in [
                ("damping", "lateral_sponge_enabled", "false"), ("damping", "lateral_width", "0"),
                ("damping", "lateral_alpha", "0"), ("damping", "w_only_top_sponge", "false"),
                ("damping", "w_only_depth_cells", "0"), ("damping", "w_only_alpha_max", "0"),
                ("solver", "slow_rk2", "false"), ("solver", "lid_bc_gamma", "0.5"),
                ("solver", "axis_u_m1_filter", "true"),
            ] {
                let changed = if section == "damping" {
                    original.replacen("damping:\n", &format!("damping:\n  {field}: {value}\n"), 1)
                } else {
                    original.replacen("  execution:\n", &format!("  solver:\n    {field}: {value}\n  execution:\n"), 1)
                };
                assert_ne!(changed, original);
                let native = validate_exact_document(&binary, &expected, &source(changed)).unwrap();
                assert!(!native.accepted && native.exit_code == Some(2), "{name}/{field}: {}", native.diagnostic);
            }
        }
        assert_eq!(crate::platform::sha256_file(&binary).unwrap(), expected);
    }

    #[test]
    fn generated_metal_v4_east_npy_advisory_keeps_codec_and_async_neighbors_closed() {
        let mut exact_catalog = east_v4_catalog();
        assert!(parse_catalog(&serde_json::to_vec(&exact_catalog).unwrap()).is_ok());
        let modules = exact_catalog["modules"].as_array_mut().unwrap();
        modules[0]["schemes"].as_array_mut().unwrap().push(scheme(
            "klemp_wilhelmson_split_cartesian_cgrid", vec![], "qualified", true,
            vec!["cartesian"], vec!["c_grid"], vec!["cpu", "gpu"],
            vec!["u", "v", "w"], vec!["parametric_cape"], vec!["wk_param"],
            vec!["warm_bubble"], "covered",
        ));
        for (module_id, scheme_id) in [
            ("advection", "tvd"),
            ("microphysics", "none"),
            ("terrain", "none"),
            ("time_stepping", "split_explicit"),
        ] {
            modules.push(serde_json::json!({
                "id": module_id,
                "selectionPath": format!("{module_id}.scheme"),
                "studioSelection": {"mode": "form", "fixedSchemeId": "", "reason": ""},
                "schemes": [scheme(
                    scheme_id, vec![], "qualified", true,
                    vec!["cartesian"], vec!["c_grid"], vec!["cpu", "gpu"],
                    vec![], vec!["not_applicable"], vec!["not_applicable"],
                    vec!["not_applicable"], "covered",
                )],
            }));
        }
        let mut exact_request = request();
        exact_request.execution_mode = "qualified".into();
        exact_request.maturity_limits_acknowledged = true;
        exact_request.backend = "metal".into();
        exact_request.allow_backend_fallback = false;
        exact_request.validation_guard_mode = "error".into();
        exact_request.boundary = BoundarySelection {
            west: "rigid_slip".into(), east: "radiative".into(),
            south: "rigid_slip".into(), north: "rigid_slip".into(),
            bottom: "rigid_noslip".into(), top: "open".into(),
        };
        exact_request.interactions.radiation_shortwave_enabled = false;
        exact_request.interactions.surface_scheme = "none".into();
        exact_request.interactions.surface_apply_fluxes = false;
        exact_request.output.preset = "minimal".into();
        exact_request.output.async_io = false;
        exact_request.selections = [
            ("dynamics", "klemp_wilhelmson_split_cartesian_cgrid"),
            ("advection", "tvd"),
            ("microphysics", "none"),
            ("terrain", "none"),
            ("time_stepping", "split_explicit"),
        ].into_iter().map(|(module_id, scheme_id)| SchemeSelection {
            module_id: module_id.into(), scheme_id: scheme_id.into(),
        }).collect();
        let generated_errors = |request: &ConfigPreflightRequest| {
            evaluate_preflight(request, &exact_catalog, identity()).unwrap()
                .hard_errors.into_iter()
                .filter(|issue| issue.code.starts_with("generated_metal_"))
                .map(|issue| issue.code).collect::<Vec<_>>()
        };
        assert!(generated_errors(&exact_request).is_empty());
        let mut netcdf = exact_request.clone();
        netcdf.output_format = "netcdf".into();
        assert_eq!(generated_errors(&netcdf), ["generated_metal_east_output_format"]);
        let mut zfp = exact_request.clone();
        zfp.output_format = "zfp_high".into();
        assert_eq!(generated_errors(&zfp), ["generated_metal_east_output_format"]);
        let mut preset = exact_request.clone();
        preset.output.preset = "presentation".into();
        assert_eq!(generated_errors(&preset), ["generated_metal_east_output_preset"]);
        let mut asynchronous = exact_request.clone();
        asynchronous.output.async_io = true;
        assert_eq!(generated_errors(&asynchronous), ["generated_metal_east_async_io"]);
        let mut cadence = exact_request.clone();
        cadence.time_control.output_interval = 301.0;
        assert_eq!(generated_errors(&cadence), ["generated_metal_output_cadence"]);
        let mut base = exact_request.clone();
        base.boundary.west = "rigid".into();
        base.boundary.east = "rigid".into();
        base.boundary.south = "rigid".into();
        base.boundary.north = "rigid".into();
        base.boundary.bottom = "rigid_slip".into();
        base.boundary.top = "rigid".into();
        base.time_control.output_interval = 301.0;
        assert!(generated_errors(&base).is_empty());
        let source = fs::read(PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/reference/catalog/flat_east_radiative_metal_npy_candidate_v1.yaml"))
            .expect("read exact east YAML fixture");
        assert_eq!(format!("{:x}", Sha256::digest(&source)),
                   "a244b58833f7e3cd7060d941e8453d4e5face3dff9e18e87ebf61dd581a24d55");
        let source_text = String::from_utf8(source).unwrap();
        assert!(source_text.contains("    west: rigid_slip\n    east: radiative\n    south: rigid_slip\n    north: rigid_slip"));
        assert!(source_text.contains("  bottom: rigid_noslip\n  top: open"));
        let mut missing_async = serde_json::to_value(&exact_request.output).unwrap();
        missing_async.as_object_mut().unwrap().remove("asyncIo");
        assert!(serde_json::from_value::<OutputInteractionValues>(missing_async).is_err());
    }

    #[test]
    fn catalog_fingerprint_is_stable() {
        assert_eq!(fnv1a64(b"nimbus"), 0x2834f89f7b381b5f);
        assert_ne!(fnv1a64(b"nimbus"), fnv1a64(b"Nimbus"));
    }

    #[test]
    #[ignore = "driven by make test-catalog-studio-native-parity"]
    fn external_catalog_request_batch_matches_native() {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct BatchEntry {
            config_path: String,
            request: ConfigPreflightRequest,
        }

        let batch_path = std::env::var("NIMBUS_PREFLIGHT_BATCH")
            .expect("NIMBUS_PREFLIGHT_BATCH is required");
        let binary = std::env::var("NIMBUS_PREFLIGHT_BINARY")
            .expect("NIMBUS_PREFLIGHT_BINARY is required");
        let catalog_output = std::process::Command::new(&binary)
            .arg("--scheme-catalog=json")
            .output()
            .expect("query exact simulator catalog");
        assert!(catalog_output.status.success());
        let exact_catalog = parse_catalog(&catalog_output.stdout)
            .expect("parse exact simulator catalog");
        let batch: Vec<BatchEntry> = serde_json::from_slice(
            &std::fs::read(batch_path).expect("read preflight batch"),
        )
        .expect("parse preflight batch");
        assert!(!batch.is_empty());
        for entry in batch {
            validate_request(&entry.request)
                .unwrap_or_else(|error| panic!("{}: {error}", entry.config_path));
            let report = evaluate_preflight(
                &entry.request,
                &exact_catalog,
                ExecutableIdentity {
                    canonical_path: binary.clone(),
                    byte_length: 1,
                    modified_unix_ms: None,
                    catalog_fingerprint: "fnv1a64:0".into(),
                    binary_fingerprint: format!("sha256:{}", "0".repeat(64)),
                },
            )
            .unwrap_or_else(|error| panic!("{}: {error}", entry.config_path));
            assert!(
                report.compatible,
                "{}: Studio/Rust preflight rejected native-admitted config: {:?}",
                entry.config_path,
                report.hard_errors,
            );
        }
    }

    #[test]
    fn catalog_parser_requires_v9_runtime_readiness_output_and_studio_evidence() {
        let valid = serde_json::to_vec(&catalog()).unwrap();
        assert!(parse_catalog(&valid).is_ok());

        let mut missing = catalog();
        missing
            .as_object_mut()
            .unwrap()
            .remove("runtimeStateFields");
        assert!(parse_catalog(&serde_json::to_vec(&missing).unwrap()).is_err());

        let mut missing_output_capabilities = catalog();
        missing_output_capabilities
            .as_object_mut()
            .unwrap()
            .remove("runtimeCapabilities");
        assert!(parse_catalog(&serde_json::to_vec(&missing_output_capabilities).unwrap()).is_err());

        let mut stale_metal_scope = catalog();
        stale_metal_scope["runtimeCapabilities"]["interactionPolicy"]
            ["generatedMetalCanonicalScope"]["manifestSha256"] = Value::String("stale".into());
        assert!(parse_catalog(&serde_json::to_vec(&stale_metal_scope).unwrap()).is_err());

        let mut typed_zfp_scope = catalog();
        let zfp = &mut typed_zfp_scope["runtimeCapabilities"]["interactionPolicy"]
            ["generatedMetalCanonicalScope"];
        zfp["version"] = Value::from(3);
        zfp["tupleCount"] = Value::from(6048);
        zfp["distinctConfigCount"] = Value::from(4536);
        zfp["outputFormats"] = serde_json::json!(["npy_3d", "netcdf", "zfp_high"]);
        zfp["npyNetcdfManifestSha256"] =
            Value::String("30d5fcc9494016894733ef973909849f8c58fbd04b3e0d6bb2e5e497c0c9ca99".into());
        zfp["zfpTypedOutputManifestSha256"] =
            Value::String("66de3a477f83cddd3fc921835900ada6d39eef070c4d43131402f0cd4898b2ab".into());
        zfp["writeCadenceSecondsByFormat"] =
            serde_json::json!({"npy_3d": [1, 301], "netcdf": [1, 301], "zfp_high": [1]});
        assert!(parse_catalog(&serde_json::to_vec(&typed_zfp_scope).unwrap()).is_ok());

        let mut uppercase_manifest = typed_zfp_scope.clone();
        uppercase_manifest["runtimeCapabilities"]["interactionPolicy"]
            ["generatedMetalCanonicalScope"]["zfpTypedOutputManifestSha256"] = Value::String(
            "66DE3A477F83CDDD3FC921835900ADA6D39EEF070C4D43131402F0CD4898B2AB".into(),
        );
        assert!(parse_catalog(&serde_json::to_vec(&uppercase_manifest).unwrap()).is_err());

        let mut stale_zfp_cadence = typed_zfp_scope.clone();
        stale_zfp_cadence["runtimeCapabilities"]["interactionPolicy"]
            ["generatedMetalCanonicalScope"]["writeCadenceSecondsByFormat"]["zfp_high"] =
            serde_json::json!([1, 301]);
        assert!(parse_catalog(&serde_json::to_vec(&stale_zfp_cadence).unwrap()).is_err());

        let mut missing_typed_manifest = typed_zfp_scope;
        missing_typed_manifest["runtimeCapabilities"]["interactionPolicy"]
            ["generatedMetalCanonicalScope"]
            .as_object_mut()
            .unwrap()
            .remove("zfpTypedOutputManifestSha256");
        assert!(parse_catalog(&serde_json::to_vec(&missing_typed_manifest).unwrap()).is_err());

        let east_v4 = east_v4_catalog();
        assert!(parse_catalog(&serde_json::to_vec(&east_v4).unwrap()).is_ok());
        let mut no_zfp = east_v4.clone();
        let no_zfp_scope = &mut no_zfp["runtimeCapabilities"]["interactionPolicy"]
            ["generatedMetalCanonicalScope"];
        no_zfp_scope["version"] = Value::from(3);
        no_zfp_scope["tupleCount"] = Value::from(4033);
        no_zfp_scope["distinctConfigCount"] = Value::from(3025);
        no_zfp_scope["outputFormats"] = serde_json::json!(["npy_3d", "netcdf"]);
        no_zfp_scope["manifestSha256"] = Value::String(
            "4604b059d425fb86690207f69000fd708f153054d739091e0ec2ba00c7d9a4e3".into());
        no_zfp_scope.as_object_mut().unwrap().remove("zfpTypedOutputManifestSha256");
        no_zfp_scope["writeCadenceSecondsByFormat"].as_object_mut().unwrap().remove("zfp_high");
        assert!(parse_catalog(&serde_json::to_vec(&no_zfp).unwrap()).is_ok());

        let mut east_negatives = Vec::new();
        for (path, replacement) in [
            ("/runtimeCapabilities/interactionPolicy/generatedMetalCanonicalScope/additionalBoundaryNpyScope/outputFormats",
             serde_json::json!(["netcdf"])),
            ("/runtimeCapabilities/interactionPolicy/generatedMetalCanonicalScope/additionalBoundaryNpyScope/outputFormats",
             serde_json::json!(["zfp_high"])),
            ("/runtimeCapabilities/interactionPolicy/generatedMetalCanonicalScope/additionalBoundaryNpyScope/writeCadenceSeconds",
             serde_json::json!([301])),
            ("/runtimeCapabilities/interactionPolicy/generatedMetalCanonicalScope/additionalBoundaryNpyScope/strictCanonicalYamlSha256",
             serde_json::json!("0".repeat(64))),
            ("/runtimeCapabilities/interactionPolicy/generatedMetalCanonicalScope/additionalBoundaryNpyScope/manifestSha256",
             serde_json::json!("0".repeat(64))),
            ("/runtimeCapabilities/interactionPolicy/generatedMetalCanonicalScope/tupleCount",
             serde_json::json!(6048)),
            ("/runtimeCapabilities/interactionPolicy/generatedMetalCanonicalScope/manifestSha256",
             serde_json::json!("0".repeat(64))),
            ("/runtimeCapabilities/interactionPolicy/version", serde_json::json!(9)),
        ] {
            let mut invalid = east_v4.clone();
            *invalid.pointer_mut(path).unwrap() = replacement;
            east_negatives.push(invalid);
        }
        let mut missing_extension = east_v4.clone();
        missing_extension["runtimeCapabilities"]["interactionPolicy"]
            ["generatedMetalCanonicalScope"].as_object_mut().unwrap()
            .remove("additionalBoundaryNpyScope");
        east_negatives.push(missing_extension);
        let mut extra_extension_key = east_v4.clone();
        extra_extension_key["runtimeCapabilities"]["interactionPolicy"]
            ["generatedMetalCanonicalScope"]["additionalBoundaryNpyScope"]["normalizedCodecsAdmitted"] =
            serde_json::json!(["zfp_high"]);
        east_negatives.push(extra_extension_key);
        for invalid in east_negatives {
            assert!(parse_catalog(&serde_json::to_vec(&invalid).unwrap()).is_err());
        }

        let mut missing_metal_scope = catalog();
        missing_metal_scope["runtimeCapabilities"]["interactionPolicy"]
            .as_object_mut()
            .unwrap()
            .remove("generatedMetalCanonicalScope");
        assert!(parse_catalog(&serde_json::to_vec(&missing_metal_scope).unwrap()).is_err());

        let mut missing_studio_policy = catalog();
        missing_studio_policy["modules"][0]
            .as_object_mut()
            .expect("fixture module")
            .remove("studioSelection");
        assert!(parse_catalog(&serde_json::to_vec(&missing_studio_policy).unwrap()).is_err());

        let mut missing_readiness = catalog();
        missing_readiness
            .as_object_mut()
            .unwrap()
            .remove("readiness");
        assert!(parse_catalog(&serde_json::to_vec(&missing_readiness).unwrap()).is_err());

        let mut missing_blocker = catalog();
        missing_blocker["readiness"]["items"][0]["state"] =
            Value::String("engineering_incomplete".into());
        missing_blocker["readiness"]["items"][0]["blocker"] = Value::String(String::new());
        assert!(parse_catalog(&serde_json::to_vec(&missing_blocker).unwrap()).is_err());

        let mut contradictory_scheme = catalog();
        contradictory_scheme["modules"][3]["schemes"][0]["readinessState"] =
            Value::String("ready".into());
        assert!(parse_catalog(&serde_json::to_vec(&contradictory_scheme).unwrap()).is_err());

        let mut legacy = catalog();
        legacy["schema"] = Value::String("nimbus.scheme-catalog.v1".into());
        legacy["catalogVersion"] = Value::from(1);
        assert!(parse_catalog(&serde_json::to_vec(&legacy).unwrap()).is_err());
    }

    #[test]
    fn policy11_reversible_radar_scope_is_exact_and_remains_guarded() {
        let valid = radar_policy11_catalog();
        assert!(parse_catalog(&serde_json::to_vec(&valid).unwrap()).is_ok());
        let mut admitted = valid.clone();
        admitted["runtimeCapabilities"]["interactionPolicy"]
            ["qualifiedZfpReversibleRadarV4"]["admitted"] = Value::Bool(true);
        assert!(parse_catalog(&serde_json::to_vec(&admitted).unwrap()).is_ok());

        let mut missing = valid.clone();
        missing["runtimeCapabilities"]["interactionPolicy"]
            .as_object_mut().unwrap().remove("qualifiedZfpReversibleRadarV4");
        assert!(parse_catalog(&serde_json::to_vec(&missing).unwrap()).is_err());
        let mut historical_with_scope = valid.clone();
        historical_with_scope["runtimeCapabilities"]["interactionPolicy"]["version"] =
            Value::from(10);
        assert!(parse_catalog(&serde_json::to_vec(&historical_with_scope).unwrap()).is_err());
        historical_with_scope["runtimeCapabilities"]["interactionPolicy"]
            ["qualifiedZfpReversibleRadarV4"] = Value::Null;
        assert!(parse_catalog(&serde_json::to_vec(&historical_with_scope).unwrap()).is_err());

        for (path, replacement) in [
            ("/admitted", serde_json::json!("yes")),
            ("/nonemptyProductSubsets", serde_json::json!(30)),
            ("/productIds", serde_json::json!(["reflectivity_dbz"])),
            ("/coreFields", serde_json::json!(["u", "v"])),
            ("/presets", serde_json::json!(["minimal"])),
            ("/namedPresetProductSets/radar_ready", serde_json::json!(["zdr"])),
            ("/outputFormatTiers", serde_json::json!(["zfp_high"])),
            ("/reviewedSourceProfileContract", serde_json::json!("unknown")),
            ("/reviewedSourceProfileOwnerSymbol", serde_json::json!("unknown")),
            ("/reviewedSourceProfileFamilies", serde_json::json!([])),
            ("/asyncIoModes", serde_json::json!([true])),
            ("/gridShape", serde_json::json!([18, 18, 18])),
            ("/gridSpacingMeters", serde_json::json!([500, 500, 250])),
            ("/dtLargeSeconds", serde_json::json!(0.1)),
            ("/writeCadenceSeconds", serde_json::json!([1])),
            ("/nonRadarBudgetOwner", serde_json::json!("radar")),
            ("/radarMode", serde_json::json!("accuracy")),
            ("/radarTolerancePolicy", serde_json::json!("allowed")),
            ("/manifestContract", serde_json::json!("unknown")),
            ("/sourceFieldBoundsContract", serde_json::json!("unknown")),
        ] {
            let mut invalid = valid.clone();
            let scope = &mut invalid["runtimeCapabilities"]["interactionPolicy"]
                ["qualifiedZfpReversibleRadarV4"];
            *scope.pointer_mut(path).unwrap() = replacement;
            assert!(parse_catalog(&serde_json::to_vec(&invalid).unwrap()).is_err(), "{path}");
        }
        let mut extra = valid.clone();
        extra["runtimeCapabilities"]["interactionPolicy"]
            ["qualifiedZfpReversibleRadarV4"]["unreviewed"] = Value::Bool(true);
        assert!(parse_catalog(&serde_json::to_vec(&extra).unwrap()).is_err());

        let mut qualified = request();
        qualified.execution_mode = "qualified".into();
        qualified.backend = "cpu".into();
        qualified.output_format = "zfp_high".into();
        qualified.selections.push(SchemeSelection {
            module_id: "radar".into(), scheme_id: "reflectivity_dbz".into(),
        });
        let report = evaluate_preflight(&qualified, &valid, identity()).unwrap();
        assert!(!report.compatible);
        assert!(report.hard_errors.iter().any(|issue|
            issue.code == "qualified_zfp_radar_product_not_ready"));
    }

    #[test]
    fn policy12_static_vertical_sgs_scope_and_dynamic_axis_fail_closed() {
        let valid = vertical_policy12_catalog();
        assert!(parse_catalog(&serde_json::to_vec(&valid).unwrap()).is_ok());
        let mut admitted = valid.clone();
        admitted["runtimeCapabilities"]["interactionPolicy"]
            ["qualifiedVerticalOnlySgs"]["admitted"] = Value::Bool(true);
        assert!(parse_catalog(&serde_json::to_vec(&admitted).unwrap()).is_ok());
        let mut historical = valid.clone();
        historical["runtimeCapabilities"]["interactionPolicy"]["version"] = Value::from(11);
        assert!(parse_catalog(&serde_json::to_vec(&historical).unwrap()).is_err());
        let mut missing = valid.clone();
        missing["runtimeCapabilities"]["interactionPolicy"]
            .as_object_mut().unwrap().remove("qualifiedVerticalOnlySgs");
        assert!(parse_catalog(&serde_json::to_vec(&missing).unwrap()).is_err());
        missing = valid.clone();
        missing["runtimeCapabilities"]["interactionPolicy"]
            .as_object_mut().unwrap().remove("turbulenceDynamicCoefficientDefault");
        assert!(parse_catalog(&serde_json::to_vec(&missing).unwrap()).is_err());
        let mut wrong_default = valid.clone();
        wrong_default["runtimeCapabilities"]["interactionPolicy"]
            ["turbulenceDynamicCoefficientDefault"] = Value::Bool(true);
        assert!(parse_catalog(&serde_json::to_vec(&wrong_default).unwrap()).is_err());
        for (path, replacement) in [
            ("/admitted", serde_json::json!("yes")),
            ("/dynamicCoefficient", serde_json::json!(true)),
            ("/mode", serde_json::json!("3d")),
            ("/backend", serde_json::json!("metal")),
            ("/gridShape", serde_json::json!([18, 18, 18])),
            ("/writeCadenceSeconds", serde_json::json!([2])),
            ("/sourceProfileOwnerSymbol", serde_json::json!("unknown")),
            ("/wholeStepRollback", serde_json::json!("partial")),
            ("/dynamicNonThreeDAllowed", serde_json::json!(true)),
        ] {
            let mut invalid = valid.clone();
            *invalid["runtimeCapabilities"]["interactionPolicy"]
                ["qualifiedVerticalOnlySgs"].pointer_mut(path).unwrap() = replacement;
            assert!(parse_catalog(&serde_json::to_vec(&invalid).unwrap()).is_err(), "{path}");
        }
        let mut extra = valid.clone();
        extra["runtimeCapabilities"]["interactionPolicy"]
            ["qualifiedVerticalOnlySgs"]["unreviewed"] = Value::Bool(true);
        assert!(parse_catalog(&serde_json::to_vec(&extra).unwrap()).is_err());

        let policy = catalog_interaction_policy(&valid).unwrap();
        let mut values = ScienceInteractionValues {
            radiation_shortwave_enabled: false,
            turbulence_mode: "vertical_only".into(),
            turbulence_dynamic_coefficient: false,
            surface_scheme: "none".into(),
            surface_apply_fluxes: false,
            diffusion_target: "scalars".into(),
            diffusion_active: false,
        };
        let selections = BTreeMap::from([
            ("turbulence".into(), "smagorinsky".into()),
            ("boundary_layer".into(), "none".into()),
        ]);
        let codes = |values: &ScienceInteractionValues, qualified: bool| {
            let mut errors = Vec::new();
            evaluate_science_interactions(values, &policy, &selections, "npy_3d", qualified,
                                          &mut errors);
            errors.into_iter().map(|item| item.code).collect::<BTreeSet<_>>()
        };
        assert!(codes(&values, true).contains("qualified_vertical_only_sgs_not_ready"));
        assert!(!codes(&values, false).contains("qualified_vertical_only_sgs_not_ready"));
        values.turbulence_dynamic_coefficient = true;
        assert!(codes(&values, false).contains("dynamic_smagorinsky_requires_3d"));
        assert!(codes(&values, true).contains("dynamic_smagorinsky_requires_3d"));
        values.turbulence_mode = "2d".into();
        assert!(codes(&values, false).contains("dynamic_smagorinsky_requires_3d"));
        values.turbulence_mode = "3d".into();
        assert!(!codes(&values, false).contains("dynamic_smagorinsky_requires_3d"));
    }

    #[test]
    fn policy13_retained_milbrandt_triple_scope_is_exact_and_unadmitted() {
        let valid = milbrandt_policy13_catalog();
        assert!(parse_catalog(&serde_json::to_vec(&valid).unwrap()).is_ok());
        let scope = &valid["runtimeCapabilities"]["interactionPolicy"]["milbrandtTripleMoment"];
        assert_eq!(scope["retained"], true);
        assert_eq!(scope["admitted"], false);
        assert_eq!(scope["strictV1ExplicitKeyAdmitted"], false);
        let mut missing = valid.clone();
        missing["runtimeCapabilities"]["interactionPolicy"]
            .as_object_mut().unwrap().remove("milbrandtTripleMoment");
        assert!(parse_catalog(&serde_json::to_vec(&missing).unwrap()).is_err());
        let mut missing_readiness = valid.clone();
        missing_readiness["readiness"]["items"].as_array_mut().unwrap().retain(|item|
            item.get("id").and_then(Value::as_str) != Some("microphysics.milbrandt_triple_moment"));
        assert!(parse_catalog(&serde_json::to_vec(&missing_readiness).unwrap()).is_err());
        let mut wrong_readiness = valid.clone();
        let row = wrong_readiness["readiness"]["items"].as_array_mut().unwrap().iter_mut()
            .find(|item| item.get("id").and_then(Value::as_str)
                == Some("microphysics.milbrandt_triple_moment")).unwrap();
        row["state"] = Value::String("ready".into());
        row["blocker"] = Value::String(String::new());
        assert!(parse_catalog(&serde_json::to_vec(&wrong_readiness).unwrap()).is_err());
        let mut empty_blocker = valid.clone();
        let row = empty_blocker["readiness"]["items"].as_array_mut().unwrap().iter_mut()
            .find(|item| item.get("id").and_then(Value::as_str)
                == Some("microphysics.milbrandt_triple_moment")).unwrap();
        row["blocker"] = Value::String(String::new());
        assert!(parse_catalog(&serde_json::to_vec(&empty_blocker).unwrap()).is_err());
        let mut duplicate = valid.clone();
        let row = duplicate["readiness"]["items"].as_array().unwrap().last().unwrap().clone();
        duplicate["readiness"]["items"].as_array_mut().unwrap().push(row);
        assert!(parse_catalog(&serde_json::to_vec(&duplicate).unwrap()).is_err());
        let mut historical = valid.clone();
        historical["runtimeCapabilities"]["interactionPolicy"]["version"] = Value::from(12);
        assert!(parse_catalog(&serde_json::to_vec(&historical).unwrap()).is_err());
        for (path, replacement) in [
            ("/retained", serde_json::json!(false)),
            ("/admitted", serde_json::json!(true)),
            ("/implementationState", serde_json::json!("ready")),
            ("/selector", serde_json::json!("microphysics.milbrandt")),
            ("/strictV1ExplicitKeyAdmitted", serde_json::json!(true)),
            ("/baselineState", serde_json::json!(["q"])),
            ("/additionalState", serde_json::json!(["N"])),
            ("/additionalMomentOrder", serde_json::json!(3)),
            ("/additionalMomentSpecies", serde_json::json!(["rain"])),
            ("/privateStorageUnits", serde_json::json!("kg/kg")),
        ] {
            let mut invalid = valid.clone();
            *invalid["runtimeCapabilities"]["interactionPolicy"]["milbrandtTripleMoment"]
                .pointer_mut(path).unwrap() = replacement;
            assert!(parse_catalog(&serde_json::to_vec(&invalid).unwrap()).is_err(), "{path}");
        }
        let mut extra = valid;
        extra["runtimeCapabilities"]["interactionPolicy"]
            ["milbrandtTripleMoment"]["unreviewed"] = Value::Bool(true);
        assert!(parse_catalog(&serde_json::to_vec(&extra).unwrap()).is_err());
    }

    #[test]
    fn admitted_static_vertical_sgs_needs_exact_native_source_and_scope() {
        let mut catalog = vertical_policy12_catalog();
        catalog["runtimeCapabilities"]["interactionPolicy"]
            ["qualifiedVerticalOnlySgs"]["admitted"] = Value::Bool(true);
        let turbulence = catalog["modules"].as_array_mut().unwrap().iter_mut()
            .find(|module| module["id"].as_str() == Some("turbulence")).unwrap();
        turbulence["schemes"].as_array_mut().unwrap().push(scheme(
            "smagorinsky", vec![], "qualified", true,
            vec!["cartesian"], vec!["c_grid"], vec!["cpu"], vec!["u", "v", "w"],
            vec!["not_applicable"], vec!["not_applicable"], vec!["not_applicable"],
            "covered"));
        let mut request = request();
        request.source = PreflightSource {
            kind: "exact_document".into(),
            yaml_text: Some("schema: nimbus.v1\n".into()),
            yaml_path: Some(std::env::temp_dir().join("vertical-sgs.yaml").display().to_string()),
        };
        request.execution_mode = "qualified".into();
        request.maturity_limits_acknowledged = true;
        request.backend = "cpu".into();
        request.allow_backend_fallback = false;
        request.validation_guard_mode = "error".into();
        request.grid_metrics = GridMetrics {
            nx: 12, ny: 12, nz: 16, dx: 1000.0, dy: 1000.0,
            dz: 250.0, ztop: 4000.0,
        };
        request.time_control = TimeControl {
            dt_large: 0.2, acoustic_substeps: 6, output_interval: 1.0,
        };
        request.boundary = BoundarySelection {
            west: "periodic".into(), east: "periodic".into(),
            south: "periodic".into(), north: "periodic".into(),
            bottom: "rigid_slip".into(), top: "rigid".into(),
        };
        request.interactions.turbulence_mode = "vertical_only".into();
        request.interactions.turbulence_dynamic_coefficient = false;
        request.interactions.radiation_shortwave_enabled = false;
        request.interactions.surface_scheme = "none".into();
        request.interactions.surface_apply_fluxes = false;
        request.selections = vec![
            SchemeSelection { module_id: "dynamics".into(), scheme_id: "cart".into() },
            SchemeSelection { module_id: "turbulence".into(), scheme_id: "smagorinsky".into() },
        ];
        assert!(validate_request(&request).is_ok());
        let exact = evaluate_preflight(&request, &catalog, identity()).unwrap();
        assert!(!exact.compatible);
        assert!(exact.hard_errors.iter().any(|issue|
            issue.code == "qualified_vertical_only_sgs_native_admission_required"));
        assert!(!exact.hard_errors.iter().any(|issue|
            issue.code == "qualified_vertical_only_sgs_scope_incompatible"));

        let mut descriptor = request.clone();
        descriptor.source = PreflightSource {
            kind: "descriptor_only".into(), yaml_text: None, yaml_path: None,
        };
        let report = evaluate_preflight(&descriptor, &catalog, identity()).unwrap();
        assert!(!report.compatible);
        assert!(report.hard_errors.iter().any(|issue|
            issue.code == "qualified_vertical_only_sgs_exact_document_required"));

        let mut neighbor = request.clone();
        neighbor.grid_metrics.nx = 13;
        let report = evaluate_preflight(&neighbor, &catalog, identity()).unwrap();
        assert!(report.hard_errors.iter().any(|issue|
            issue.code == "qualified_vertical_only_sgs_scope_incompatible"));
        let mut overlap = request.clone();
        overlap.selections.push(SchemeSelection {
            module_id: "boundary_layer".into(), scheme_id: "slab".into(),
        });
        let report = evaluate_preflight(&overlap, &catalog, identity()).unwrap();
        assert!(report.hard_errors.iter().any(|issue|
            issue.code == "vertical_mixing_overlap"));

        let mut accepted = exact.clone();
        apply_native_validation(&mut accepted, NativeValidationOutcome {
            accepted: true,
            source_sha256: "1".repeat(64),
            source_fingerprint: "fnv1a64:1234567890abcdef".into(),
            source_path_fingerprint: "fnv1a64:1234567890abcdef".into(),
            binary_sha256: "sha256:".to_string() + &"2".repeat(64),
            exit_code: Some(0), diagnostic: String::new(),
        });
        assert!(accepted.compatible, "{:?}", accepted.hard_errors);
        let mut rejected = exact;
        apply_native_validation(&mut rejected, NativeValidationOutcome {
            accepted: false,
            source_sha256: "1".repeat(64),
            source_fingerprint: "fnv1a64:1234567890abcdef".into(),
            source_path_fingerprint: "fnv1a64:1234567890abcdef".into(),
            binary_sha256: "sha256:".to_string() + &"2".repeat(64),
            exit_code: Some(2), diagnostic: "off-scope source".into(),
        });
        assert!(!rejected.compatible);
        assert!(rejected.hard_errors.iter().any(|issue|
            issue.code == "native_strict_admission_rejected"));
    }

    #[test]
    fn admitted_radar_v4_requires_exact_document_native_acceptance() {
        let mut catalog = radar_policy11_catalog();
        catalog["runtimeCapabilities"]["interactionPolicy"]
            ["qualifiedZfpReversibleRadarV4"]["admitted"] = Value::Bool(true);
        catalog["modules"].as_array_mut().unwrap().push(serde_json::json!({
            "id": "radar", "selectionPath": "radar.products",
            "studioSelection": {"mode": "product_set_v1", "fixedSchemeId": "",
                                "reason": "Strict v1 ordered radar product set."},
            "schemes": [
                scheme("reflectivity_dbz", vec![], "qualified", true,
                    vec!["cartesian"], vec!["c_grid"], vec!["cpu"], vec!["rho"],
                    vec!["not_applicable"], vec!["not_applicable"], vec!["not_applicable"],
                    "covered"),
                scheme("radial_velocity", vec![], "qualified", true,
                    vec!["cartesian"], vec!["c_grid"], vec!["cpu"], vec!["u", "v", "w"],
                    vec!["not_applicable"], vec!["not_applicable"], vec!["not_applicable"],
                    "covered"),
                scheme("zdr", vec![], "qualified", true,
                    vec!["cartesian"], vec!["c_grid"], vec!["cpu"], vec!["qr"],
                    vec!["not_applicable"], vec!["not_applicable"], vec!["not_applicable"],
                    "covered"),
                scheme("kdp", vec![], "qualified", true,
                    vec!["cartesian"], vec!["c_grid"], vec!["cpu"], vec!["qr"],
                    vec!["not_applicable"], vec!["not_applicable"], vec!["not_applicable"],
                    "covered"),
                scheme("rhohv", vec![], "qualified", true,
                    vec!["cartesian"], vec!["c_grid"], vec!["cpu"], vec!["qr"],
                    vec!["not_applicable"], vec!["not_applicable"], vec!["not_applicable"],
                    "covered")]
        }));
        catalog["runtimeCapabilities"]["outputFormats"].as_array_mut().unwrap().extend([
            serde_json::json!({"id": "zfp_low", "available": true,
                               "reason": "", "implementation": "libzfp"}),
            serde_json::json!({"id": "zfp_medium", "available": true,
                               "reason": "", "implementation": "libzfp"}),
        ]);
        let mut request = request();
        request.source = PreflightSource {
            kind: "exact_document".into(),
            yaml_text: Some("schema: nimbus.v1\n".into()),
            yaml_path: Some(std::env::temp_dir().join("radar-v4.yaml").display().to_string()),
        };
        request.execution_mode = "qualified".into();
        request.maturity_limits_acknowledged = true;
        request.backend = "cpu".into();
        request.allow_backend_fallback = false;
        request.validation_guard_mode = "error".into();
        request.output_format = "zfp_high".into();
        request.grid_metrics = GridMetrics {
            nx: 12, ny: 12, nz: 48, dx: 1000.0, dy: 1000.0,
            dz: 250.0, ztop: 12000.0,
        };
        request.time_control = TimeControl {
            dt_large: 0.2, acoustic_substeps: 6, output_interval: 4.0,
        };
        request.diagnostics.deterministic_reductions = true;
        request.interactions.radiation_shortwave_enabled = false;
        request.interactions.surface_scheme = "none".into();
        request.interactions.surface_apply_fluxes = false;
        request.selections = vec![
            SchemeSelection { module_id: "dynamics".into(), scheme_id: "cart".into() },
            SchemeSelection { module_id: "radar".into(), scheme_id: "reflectivity_dbz".into() },
        ];
        request.available_state_fields.push("rho".into());
        assert!(validate_request(&request).is_ok());
        let exact = evaluate_preflight(&request, &catalog, identity()).unwrap();
        let exact_codes: BTreeSet<&str> = exact.hard_errors.iter().map(|issue| issue.code).collect();
        assert!(exact_codes.contains("qualified_zfp_radar_native_admission_required"));
        assert!(!exact_codes.contains("qualified_zfp_radar_scope_incompatible"));
        assert!(!exact.compatible);

        let mut descriptor = request.clone();
        descriptor.source = PreflightSource {
            kind: "descriptor_only".into(), yaml_text: None, yaml_path: None,
        };
        let descriptor_report = evaluate_preflight(&descriptor, &catalog, identity()).unwrap();
        assert!(!descriptor_report.compatible);
        assert!(descriptor_report.hard_errors.iter().any(|issue|
            issue.code == "qualified_zfp_radar_exact_document_required"));
        assert_eq!(descriptor_report.native_admission.state, "not_checked");
        let mut descriptor_with_forged_acceptance = descriptor_report.clone();
        apply_native_validation(&mut descriptor_with_forged_acceptance, NativeValidationOutcome {
            accepted: true,
            source_sha256: "1".repeat(64),
            source_fingerprint: "fnv1a64:1234567890abcdef".into(),
            source_path_fingerprint: "fnv1a64:1234567890abcdef".into(),
            binary_sha256: "sha256:".to_string() + &"2".repeat(64),
            exit_code: Some(0), diagnostic: String::new(),
        });
        assert!(!descriptor_with_forged_acceptance.compatible);

        let mut wrong_grid = request.clone();
        wrong_grid.grid_metrics.nx = 13;
        let mut wrong_preset = request.clone();
        wrong_preset.output.preset = "radar_ready".into();
        let mut wrong_cadence = request.clone();
        wrong_cadence.time_control.output_interval = 2.0;
        let mut wrong_tier = request.clone();
        wrong_tier.output_format = "zfp".into();
        for neighbor in [wrong_grid, wrong_preset, wrong_cadence, wrong_tier] {
            let report = evaluate_preflight(&neighbor, &catalog, identity()).unwrap();
            assert!(report.hard_errors.iter().any(|issue|
                issue.code == "qualified_zfp_radar_scope_incompatible"));
        }
        let mut asynchronous = request.clone();
        asynchronous.output.async_io = true;
        let report = evaluate_preflight(&asynchronous, &catalog, identity()).unwrap();
        assert!(!report.hard_errors.iter().any(|issue|
            issue.code == "qualified_zfp_radar_scope_incompatible"));
        let mut pair = request.clone();
        pair.output.preset = "radar_ready".into();
        pair.selections.push(SchemeSelection {
            module_id: "radar".into(), scheme_id: "radial_velocity".into(),
        });
        for tier in ["zfp_low", "zfp_medium", "zfp_high"] {
            pair.output_format = tier.into();
            let report = evaluate_preflight(&pair, &catalog, identity()).unwrap();
            assert!(!report.hard_errors.iter().any(|issue| matches!(issue.code,
                "qualified_zfp_radar_scope_incompatible" | "qualified_zfp_preset_incompatible")));
        }
        let mut polarimetric = request.clone();
        polarimetric.output.preset = "radar_polarimetric_ready".into();
        polarimetric.selections.retain(|selection| selection.module_id != "radar");
        for product in ["zdr", "kdp", "rhohv"] {
            polarimetric.selections.push(SchemeSelection {
                module_id: "radar".into(), scheme_id: product.into(),
            });
        }
        polarimetric.output.zfp_field_tolerances.insert("qc".into(), 1e-8);
        let report = evaluate_preflight(&polarimetric, &catalog, identity()).unwrap();
        assert!(!report.hard_errors.iter().any(|issue| matches!(issue.code,
            "qualified_zfp_radar_scope_incompatible" | "qualified_zfp_preset_incompatible" |
            "qualified_zfp_field_tolerance_missing")));
        let mut radar_tolerance = request.clone();
        radar_tolerance.output.zfp_field_tolerances.insert("reflectivity_dbz".into(), 0.01);
        let report = evaluate_preflight(&radar_tolerance, &catalog, identity()).unwrap();
        assert!(report.hard_errors.iter().any(|issue|
            issue.code == "qualified_zfp_radar_scope_incompatible"));
        let mut off_scope_after_native_acceptance = report;
        apply_native_validation(&mut off_scope_after_native_acceptance, NativeValidationOutcome {
            accepted: true,
            source_sha256: "1".repeat(64),
            source_fingerprint: "fnv1a64:1234567890abcdef".into(),
            source_path_fingerprint: "fnv1a64:1234567890abcdef".into(),
            binary_sha256: "sha256:".to_string() + &"2".repeat(64),
            exit_code: Some(0), diagnostic: String::new(),
        });
        assert!(!off_scope_after_native_acceptance.compatible);

        let mut accepted = exact.clone();
        apply_native_validation(&mut accepted, NativeValidationOutcome {
            accepted: true,
            source_sha256: "1".repeat(64),
            source_fingerprint: "fnv1a64:1234567890abcdef".into(),
            source_path_fingerprint: "fnv1a64:1234567890abcdef".into(),
            binary_sha256: "sha256:".to_string() + &"2".repeat(64),
            exit_code: Some(0), diagnostic: String::new(),
        });
        assert!(!accepted.hard_errors.iter().any(|issue|
            issue.code == "qualified_zfp_radar_native_admission_required"));
        assert_eq!(accepted.native_admission.state, "accepted");
        assert!(accepted.compatible, "{:?}", accepted.hard_errors);

        let mut rejected = exact;
        apply_native_validation(&mut rejected, NativeValidationOutcome {
            accepted: false,
            source_sha256: "1".repeat(64),
            source_fingerprint: "fnv1a64:1234567890abcdef".into(),
            source_path_fingerprint: "fnv1a64:1234567890abcdef".into(),
            binary_sha256: "sha256:".to_string() + &"2".repeat(64),
            exit_code: Some(2), diagnostic: "off-scope source".into(),
        });
        assert!(!rejected.compatible);
        assert!(rejected.hard_errors.iter().any(|issue|
            issue.code == "native_strict_admission_rejected"));
    }

    #[test]
    fn v10_exact_document_requires_source_text_path_and_dynamic_axis() {
        let mut request = request();
        assert!(validate_request(&request).is_ok());
        request.source.kind = "exact_document".into();
        assert!(validate_request(&request).is_err());
        request.source.yaml_text = Some("schema: nimbus.v1\n".into());
        assert!(validate_request(&request).is_err());
        request.source.yaml_path = Some("/no/such/preflight-parent/source.yaml".into());
        assert!(validate_request(&request).is_err());
        request.source.yaml_path = Some(std::env::temp_dir()
            .join("nimbus-v9-exact-source.yaml").display().to_string());
        assert!(validate_request(&request).is_ok());
        let mut missing_dynamic = serde_json::to_value(&request).unwrap();
        missing_dynamic["interactions"].as_object_mut().unwrap()
            .remove("turbulenceDynamicCoefficient");
        assert!(serde_json::from_value::<ConfigPreflightRequest>(missing_dynamic).is_err());
        request.request_version = 9;
        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn v9_descriptor_source_mismatch_fails_closed_in_both_directions() {
        let mut report = evaluate_preflight(&request(), &catalog(), identity()).unwrap();
        assert!(report.compatible);
        apply_native_validation(&mut report, NativeValidationOutcome {
            accepted: false,
            source_sha256: "1".repeat(64),
            source_fingerprint: "fnv1a64:1234567890abcdef".into(),
            source_path_fingerprint: "fnv1a64:1234567890abcdef".into(),
            binary_sha256: "sha256:".to_string() + &"2".repeat(64),
            exit_code: Some(2),
            diagnostic: "strict source rejected".into(),
        });
        assert!(!report.compatible);
        assert_eq!(report.native_admission.state, "rejected");
        assert_eq!(report.native_admission.descriptor_projection, "not_checked");
        assert!(report.hard_errors.iter().any(|item|
            item.code == "native_strict_admission_rejected"));
        assert_eq!(report.native_admission.validation_exit_code, Some(2));

        let mut incompatible = request();
        incompatible.selections[0].scheme_id = "unknown_dynamics".into();
        let mut report = evaluate_preflight(&incompatible, &catalog(), identity()).unwrap();
        assert!(!report.compatible);
        apply_native_validation(&mut report, NativeValidationOutcome {
            accepted: true,
            source_sha256: "1".repeat(64),
            source_fingerprint: "fnv1a64:1234567890abcdef".into(),
            source_path_fingerprint: "fnv1a64:1234567890abcdef".into(),
            binary_sha256: "sha256:".to_string() + &"2".repeat(64),
            exit_code: Some(0),
            diagnostic: String::new(),
        });
        assert!(!report.compatible);
        assert_eq!(report.native_admission.state, "accepted");
        assert_eq!(report.native_admission.descriptor_projection, "not_checked");
    }

    #[cfg(unix)]
    #[test]
    fn v9_native_source_validation_binds_relative_path_and_binary_changes() {
        use std::os::unix::fs::PermissionsExt;
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "nimbus-v9-native-source-{}-{id}", std::process::id()));
        let with_marker = root.join("with-marker");
        let without_marker = root.join("without-marker");
        fs::create_dir_all(&with_marker).unwrap();
        fs::create_dir_all(&without_marker).unwrap();
        fs::write(with_marker.join("relative.marker"), b"present").unwrap();
        fs::set_permissions(&with_marker, fs::Permissions::from_mode(0o555)).unwrap();
        let binary = root.join("native-validator.sh");
        fs::write(&binary, "#!/bin/sh\nif [ -f relative.marker ]; then printf '[VALIDATE] PASS\\n'; else printf 'missing relative marker\\n' >&2; exit 2; fi\n").unwrap();
        fs::set_permissions(&binary, fs::Permissions::from_mode(0o755)).unwrap();
        let expected = crate::platform::sha256_file(&binary).unwrap();
        let mut source = PreflightSource {
            kind: "exact_document".into(),
            yaml_text: Some("schema: nimbus.v1\n".into()),
            yaml_path: Some(with_marker.join("source.yaml").display().to_string()),
        };
        let accepted = validate_exact_document(&binary, &expected, &source).unwrap();
        assert!(accepted.accepted);
        assert_eq!(accepted.binary_sha256, expected);
        source.yaml_path = Some(without_marker.join("source.yaml").display().to_string());
        let rejected = validate_exact_document(&binary, &expected, &source).unwrap();
        assert!(!rejected.accepted);
        assert_eq!(rejected.exit_code, Some(2));
        assert_ne!(accepted.source_path_fingerprint, rejected.source_path_fingerprint);
        assert!(validate_exact_document(&binary, &format!("sha256:{}", "0".repeat(64)), &source).is_err());

        fs::write(&binary, "#!/bin/sh\nprintf '# changed\\n' > \"$0.replacement\"\nmv \"$0.replacement\" \"$0\"\nprintf '[VALIDATE] PASS\\n'\n").unwrap();
        fs::set_permissions(&binary, fs::Permissions::from_mode(0o755)).unwrap();
        let before_swap = crate::platform::sha256_file(&binary).unwrap();
        assert!(validate_exact_document(&binary, &before_swap, &source).is_err());
        fs::set_permissions(&with_marker, fs::Permissions::from_mode(0o755)).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[ignore = "requires the exact built simulator and retained qualification fixtures"]
    fn v9_exact_native_source_distinguishes_east_and_radar_hidden_neighbors() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap();
        let binary = root.join("bin/nimbus_sim");
        let binary_sha = crate::platform::sha256_file(&binary).unwrap();
        let east = fs::read_to_string(root.join(
            "tests/fixtures/reference/catalog/flat_east_radiative_metal_npy_candidate_v1.yaml")).unwrap();
        let radar = fs::read_to_string(root.join(
            "tests/fixtures/configs/qualification/cgrid_radar_polarimetric_ready.yaml")).unwrap();
        let source_path = root.join("v9-exact-source-context.yaml").display().to_string();
        let outcome = |text: String| {
            let source = PreflightSource {
                kind: "exact_document".into(), yaml_text: Some(text),
                yaml_path: Some(source_path.clone()),
            };
            validate_exact_document(&binary, &binary_sha, &source).unwrap()
        };
        assert!(outcome(east.clone()).accepted);
        for changed in [
            east.replace("      center_x_m: 8000\n      center_y_m: 1000",
                         "      center_x_m: 4500\n      center_y_m: 4500"),
            east.replace("  adaptive_dt: false", "  adaptive_dt: true"),
            east.replace("  run_time: 2", "  run_time: 3"),
            east.replace("damping:\n  enabled: false", "damping:\n  enabled: true"),
            east.replace("  live_shm: false", "  live_shm: true"),
            east.replace("    east: radiative", "    east: zero_gradient"),
            east.replace("    west: rigid_slip", "    west: radiative"),
        ] {
            assert_ne!(changed, east);
            assert!(!outcome(changed).accepted);
        }
        assert!(outcome(radar.clone()).accepted);
        for changed in [
            radar.replace("  zdr_tier: zhang2021_rain_mp", "  zdr_tier: simple"),
            radar.replace(
                "tests/fixtures/configs/qualification/soundings/radar_warm_cloud_profile.txt",
                "/private/tmp/nimbus-v9-missing-sounding.txt"),
        ] {
            assert_ne!(changed, radar);
            assert!(!outcome(changed).accepted);
        }
        assert_eq!(crate::platform::sha256_file(&binary).unwrap(), binary_sha);
    }

    #[test]
    #[ignore = "requires hash-bound policy13 Milbrandt source fixtures and exact simulator"]
    fn policy13_real_exact_source_keeps_triple_guard_and_double_control() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap();
        let binary = root.join("bin/nimbus_sim");
        let binary_sha = crate::platform::sha256_file(&binary).unwrap();
        let expected_sha = std::env::var("NIMBUS_MILBRANDT_TRIPLE_BINARY_SHA256")
            .expect("exact policy13 simulator SHA");
        assert_eq!(binary_sha, format!("sha256:{expected_sha}"));
        let catalog_output = Command::new(&binary).arg("--scheme-catalog=json").output().unwrap();
        assert!(catalog_output.status.success());
        let catalog = parse_catalog(&catalog_output.stdout).unwrap();
        let policy = &catalog["runtimeCapabilities"]["interactionPolicy"];
        assert_eq!(policy["version"], 13);
        assert_eq!(policy["milbrandtTripleMoment"]["admitted"], false);
        assert_eq!(policy["milbrandtTripleMoment"]["strictV1ExplicitKeyAdmitted"], false);
        let fixtures = root.join("tests/fixtures/configs/milbrandt_admission");
        let context = root.join("milbrandt-policy13-studio-source.yaml").display().to_string();
        for (name, source_sha, accepted) in [
            ("double_control.yaml", "808116bee4f03d636fe61e3687656a6f0c6d07b08f9f6759cd5dfb322bb0e3f1", true),
            ("strict_true.yaml", "e00e636e182dd99c78ba6b8fdb79f30164a4966b2d3a7dad7602fef8dd2bdf41", false),
            ("strict_false.yaml", "a5e19856876319b7469f8dcf21538ebea6275fb40660becaa423f5951cdcb4d7", false),
            ("strict_wrong_scheme_true.yaml", "43418a0ed15ad17300cdc77bb0155db121f1a9e801c64e82a7b7b6241b0879d3", false),
            ("legacy_true.yaml", "96e4d824426fb803a768d9fbc0999f2c13902575255a19baae02fdfa4fe0d5d4", false),
            ("legacy_wrong_scheme_true.yaml", "3f3a734372820420b0624d243f5eb0dcd2897d41a7b6d12ce215f515b740c722", false),
        ] {
            let path = fixtures.join(name);
            assert_eq!(crate::platform::sha256_file(&path).unwrap(), format!("sha256:{source_sha}"));
            let source = PreflightSource {
                kind: "exact_document".into(),
                yaml_text: Some(fs::read_to_string(&path).unwrap()),
                yaml_path: Some(context.clone()),
            };
            let outcome = validate_exact_document(&binary, &binary_sha, &source).unwrap();
            assert_eq!(outcome.accepted, accepted, "{name}: {}", outcome.diagnostic);
            assert_eq!(outcome.source_sha256, source_sha);
            assert_eq!(outcome.binary_sha256, binary_sha);
            println!("[milbrandt-policy13-exact-source] {name}: accepted={accepted}");
        }
        let after = Command::new(&binary).arg("--scheme-catalog=json").output().unwrap();
        assert!(after.status.success() && after.stdout == catalog_output.stdout);
        assert_eq!(crate::platform::sha256_file(&binary).unwrap(), binary_sha);
    }

    #[test]
    #[ignore = "requires hash-bound current public radar-v4 YAML and exact simulator"]
    fn admitted_radar_v4_real_exact_documents_require_native_and_descriptor_agreement() {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Case {
            config_path: String,
            yaml_path_context: String,
            request: ConfigPreflightRequest,
            expected_native_acceptance: bool,
        }
        let binary = PathBuf::from(std::env::var("NIMBUS_RADAR_V4_PREFLIGHT_BINARY")
            .expect("exact radar preflight binary"));
        let expected_sha = std::env::var("NIMBUS_RADAR_V4_PREFLIGHT_BINARY_SHA256")
            .expect("exact radar preflight binary SHA");
        assert_eq!(crate::platform::sha256_file(&binary).unwrap(),
                   format!("sha256:{expected_sha}"));
        let batch = PathBuf::from(std::env::var("NIMBUS_RADAR_V4_PREFLIGHT_BATCH")
            .expect("source-bound radar preflight batch"));
        let cases: Vec<Case> = serde_json::from_slice(&fs::read(batch).unwrap()).unwrap();
        assert!((2..=16).contains(&cases.len()));
        let catalog_bytes = Command::new(&binary).arg("--scheme-catalog=json")
            .output().unwrap();
        assert!(catalog_bytes.status.success());
        let catalog = parse_catalog(&catalog_bytes.stdout).unwrap();
        assert!(catalog["runtimeCapabilities"]["interactionPolicy"]
            ["qualifiedZfpReversibleRadarV4"]["admitted"].as_bool() == Some(true));
        for case in cases {
            validate_request(&case.request).unwrap();
            assert_eq!(case.request.source.kind, "exact_document");
            assert_eq!(case.request.source.yaml_path.as_deref(),
                       Some(case.yaml_path_context.as_str()));
            assert_eq!(case.request.source.yaml_text.as_deref(),
                       Some(fs::read_to_string(&case.config_path).unwrap().as_str()));
            let identity = executable_identity(&binary, &catalog_bytes.stdout).unwrap();
            if case.expected_native_acceptance {
                let mut descriptor = case.request.clone();
                descriptor.source = PreflightSource {
                    kind: "descriptor_only".into(), yaml_text: None, yaml_path: None,
                };
                let advisory = evaluate_preflight(&descriptor, &catalog, identity.clone()).unwrap();
                assert!(!advisory.compatible);
                assert!(advisory.hard_errors.iter().any(|issue|
                    issue.code == "qualified_zfp_radar_exact_document_required"));
            }
            let mut report = evaluate_preflight(&case.request, &catalog, identity).unwrap();
            let native = validate_exact_document(
                &binary, &report.executable.binary_fingerprint, &case.request.source).unwrap();
            assert_eq!(native.accepted, case.expected_native_acceptance, "{}", case.config_path);
            apply_native_validation(&mut report, native);
            assert_eq!(report.compatible, case.expected_native_acceptance,
                       "{}: {:?}", case.config_path, report.hard_errors);
            assert_eq!(report.native_admission.state,
                       if case.expected_native_acceptance { "accepted" } else { "rejected" });
            let after = Command::new(&binary).arg("--scheme-catalog=json").output().unwrap();
            assert!(after.status.success() && after.stdout == catalog_bytes.stdout);
            assert_eq!(crate::platform::sha256_file(&binary).unwrap(),
                       format!("sha256:{expected_sha}"));
            println!("[radar-v4-exact-preflight] {}: native={} compatible={}",
                     case.config_path, case.expected_native_acceptance, report.compatible);
        }
    }

    #[test]
    #[ignore = "requires hash-bound current-policy static/dynamic SGS YAML and exact simulator"]
    fn admitted_vertical_sgs_real_exact_documents_match_native() {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Case {
            config_path: String,
            request: ConfigPreflightRequest,
            expected_native_acceptance: bool,
        }
        let binary = PathBuf::from(std::env::var("NIMBUS_VERTICAL_SGS_PREFLIGHT_BINARY")
            .expect("exact vertical SGS preflight binary"));
        let expected_sha = std::env::var("NIMBUS_VERTICAL_SGS_PREFLIGHT_BINARY_SHA256")
            .expect("exact vertical SGS preflight binary SHA");
        assert_eq!(crate::platform::sha256_file(&binary).unwrap(),
                   format!("sha256:{expected_sha}"));
        let batch = PathBuf::from(std::env::var("NIMBUS_VERTICAL_SGS_PREFLIGHT_BATCH")
            .expect("source-bound vertical SGS preflight batch"));
        let cases: Vec<Case> = serde_json::from_slice(&fs::read(batch).unwrap()).unwrap();
        assert!((4..=16).contains(&cases.len()));
        let catalog_output = Command::new(&binary).arg("--scheme-catalog=json")
            .output().unwrap();
        assert!(catalog_output.status.success());
        let catalog = parse_catalog(&catalog_output.stdout).unwrap();
        assert_eq!(catalog["runtimeCapabilities"]["interactionPolicy"]["version"], 14);
        assert_eq!(catalog["runtimeCapabilities"]["interactionPolicy"]
            ["qualifiedVerticalOnlySgs"]["admitted"], true);
        for case in cases {
            validate_request(&case.request).unwrap();
            assert_eq!(case.request.source.kind, "exact_document");
            assert_eq!(case.request.source.yaml_path.as_deref(), Some(case.config_path.as_str()));
            assert_eq!(case.request.source.yaml_text.as_deref(),
                       Some(fs::read_to_string(&case.config_path).unwrap().as_str()));
            let identity = executable_identity(&binary, &catalog_output.stdout).unwrap();
            if case.request.execution_mode == "qualified"
                && case.request.interactions.turbulence_mode == "vertical_only"
                && !case.request.interactions.turbulence_dynamic_coefficient
            {
                let mut descriptor = case.request.clone();
                descriptor.source = PreflightSource {
                    kind: "descriptor_only".into(), yaml_text: None, yaml_path: None,
                };
                let advisory = evaluate_preflight(&descriptor, &catalog, identity.clone()).unwrap();
                assert!(!advisory.compatible);
                assert!(advisory.hard_errors.iter().any(|issue|
                    issue.code == "qualified_vertical_only_sgs_exact_document_required"));
            }
            let mut report = evaluate_preflight(&case.request, &catalog, identity).unwrap();
            let native = validate_exact_document(
                &binary, &report.executable.binary_fingerprint, &case.request.source).unwrap();
            assert_eq!(native.accepted, case.expected_native_acceptance, "{}", case.config_path);
            apply_native_validation(&mut report, native);
            assert_eq!(report.compatible, case.expected_native_acceptance,
                       "{}: {:?}", case.config_path, report.hard_errors);
            let after = Command::new(&binary).arg("--scheme-catalog=json").output().unwrap();
            assert!(after.status.success() && after.stdout == catalog_output.stdout);
            assert_eq!(crate::platform::sha256_file(&binary).unwrap(),
                       format!("sha256:{expected_sha}"));
            println!("[vertical-sgs-exact-preflight] {}: native={} compatible={}",
                     case.config_path, case.expected_native_acceptance, report.compatible);
        }
    }
    #[test]
    #[ignore = "Requires an explicit SSH target and exact-document batch"]
    fn remote_exact_document_batch_matches_native() {
        let path = std::env::var("NIMBUS_REMOTE_PREFLIGHT_BATCH").expect("remote batch path");
        let batch: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let target: super::super::remote_compute::RemoteComputeTarget =
            serde_json::from_value(batch["executionTarget"].clone()).unwrap();
        let cases = batch["cases"].as_array().unwrap();
        assert!(!cases.is_empty());
        for case in cases {
            let request = serde_json::from_value(case["request"].clone()).unwrap();
            let report = preflight_remote_config(target.clone(), request).unwrap();
            println!("[remote-exact-preflight] {}", serde_json::json!({
                "name": case["name"], "expectedAcceptance": case["expectedAcceptance"], "report": report
            }));
            assert_eq!(report.compatible, case["expectedAcceptance"].as_bool().unwrap());
        }
    }

}
