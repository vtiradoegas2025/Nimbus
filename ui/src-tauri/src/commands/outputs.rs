use std::collections::HashMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::ipc::Response;

use crate::error::{AppError, AppResult};

/*
 * Run-output reader commands. The simulator writes one field file per step
 * inside <run_dir>/step_NNNNNN/, plus a manifest.json with simulation_time_s
 * and grid metadata. Fields are either uncompressed .npy or ZFP-compressed
 * .zfp3d (output.format: zfp_*); both are surfaced and read transparently —
 * .zfp3d is decoded in `crate::zfp3d` and re-serialized to the same <f4
 * C-order .npy byte stream, so the TS side (lib/npy.ts) is unchanged. See
 * src/io/npy_writer.cpp, src/core/output/output_writer.cpp, and
 * src/core/runtime/headless_runtime.cpp.
 *
 * Two commands:
 *   - list_step_dirs(run_dir)            → frame table + per-frame field list
 *   - read_field_bytes(run_dir, step,    → .npy bytes (decoded from .zfp3d when
 *                      field_name)         needed), parsed in TS via lib/npy.ts
 *
 * Bytes return uses `tauri::ipc::Response` so the body crosses IPC as
 * binary, not a JSON array of numbers (which would inflate ~3x).
 */

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StepInfo {
    pub index: u32,
    pub simulation_time_s: Option<f64>,
    pub fields: Vec<String>,
    pub grid: Option<GridInfo>,
    /// `field_id` -> physical units, read from the manifest's `fields` catalog,
    /// which the simulator emits straight from `field_contract.cpp`. The UI
    /// labels colorbars and value axes from this map so displayed units never
    /// drift from the C++ field contract. Empty for manifests that predate the
    /// catalog (the UI then shows the field name with no unit suffix).
    pub field_units: std::collections::HashMap<String, String>,
    /// Exact per-field lossy storage contract. Empty for lossless products and
    /// legacy manifests. Python/plot consumers use this to distinguish decoded
    /// dtype from the accuracy/precision/rate contract that produced it.
    pub field_compression: std::collections::HashMap<String, FieldCompressionInfo>,
    pub field_brick_hierarchies: std::collections::HashMap<String, FieldBrickHierarchyInfo>,
    pub compression_execution: Option<CompressionExecutionInfo>,
    /// Whether this frame passed the current atomic-publication contract. A
    /// legacy frame is surfaced deliberately (not inferred from a missing first
    /// manifest); incomplete/invalid current frames are returned for status
    /// accounting but are never selectable in Studio.
    pub publication_state: StepPublicationState,
    /// Unique simulator publication generation. Current frames always carry
    /// one; legacy frames use `None` and remain outside the strong replacement
    /// guarantee.
    pub publication_generation: Option<String>,
    /// Frame provenance for this step. `None` only for manifests that predate
    /// the horizontal-frame contract; current manifests always carry it.
    pub motion: Option<FrameMotion>,
}

/// One simulator-published refinement level below the selected run root.
///
/// Child histories deliberately share the root publication marker and
/// generation. Returning the validated frame table here avoids presenting the
/// child as a legacy run merely because it has no duplicate
/// `nimbus_output.json` of its own.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NestedHistoryLevel {
    /// Stable path identity relative to the root hierarchy, for example
    /// `child_0000/child_0002`.
    pub level_id: String,
    /// `root` for a direct child, otherwise the parent's full level identity.
    pub parent_level_id: String,
    /// One for a direct child. Bounded during discovery before allocation.
    pub depth: usize,
    pub run_dir: String,
    pub steps: Vec<StepInfo>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FieldCompressionInfo {
    pub codec: String,
    pub mode: String,
    pub mode_parameter: f64,
    pub mode_parameter_units: String,
    pub error_contract: String,
    pub sparse_threshold: f64,
    pub float16_prefilter: bool,
    pub decoded_dtype: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FieldBrickHierarchyInfo {
    pub path: String,
    pub content_sha256: String,
    pub schema_version: u32,
    pub native_brick_shape: [u32; 3],
    pub extrema_semantics: String,
    pub extrema_absolute_padding: f64,
    pub culling_safe: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CompressionExecutionInfo {
    pub field_execution_policy: String,
    pub field_workers: u32,
    pub field_scratch_budget_mib: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StepPublicationState {
    Complete,
    Incomplete,
    InvalidManifest,
    Legacy,
}

/// Domain-motion frame provenance, read from the manifest's
/// `grid.horizontal_frame` block (see `include/core/output/step_manifest.hpp`,
/// which refuses to publish a step without complete finite provenance).
///
/// Surfaced at STEP level rather than inside `grid` because it describes the
/// frame the step was computed in, not the lattice's shape, and because the UI
/// contract (`ui/src/lib/run/frameMotion.ts`) reads it there. Every guard in
/// that module — swath accumulation, cross-run comparison, wind frame, axis
/// labelling — is fail-closed on this data, so dropping it here would silently
/// turn all of them fail-open.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FrameMotion {
    pub mode: Option<String>,
    pub horizontal_frame_contract: Option<String>,
    pub frame_velocity_ms: Option<[f64; 2]>,
    pub earth_origin_m: Option<[f64; 2]>,
    pub grid_origin_t0_earth_m: Option<[f64; 2]>,
    pub winds_earth_relative: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GridInfo {
    pub nr: Option<u32>,
    pub nth: Option<u32>,
    pub nz: Option<u32>,
    pub coordinate_system: Option<String>,
    pub staggering: Option<String>,
    pub dr_m: Option<f64>,
    pub dy_m: Option<f64>,
    pub dtheta_rad: Option<f64>,
    pub dz_m: Option<f64>,
    pub terrain_following: bool,
    pub terrain_geometry_identity: Option<String>,
    pub horizontal_scalar_origin_m: Option<[f64; 2]>,
    pub scalar_origin_m: Option<[f64; 3]>,
    pub physical_bounds_m: Option<AxisBoundsF64>,
    pub valid_index_range: Option<AxisBoundsU32>,
    pub halo_width: Option<AxisBoundsU32>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AxisBoundsF64 {
    pub lower: [f64; 3],
    pub upper: [f64; 3],
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AxisBoundsU32 {
    pub lower: [u32; 3],
    pub upper: [u32; 3],
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RunOutputContract {
    Current {
        generation: String,
        ready: bool,
        manifest_version: u64,
        history_index_required: bool,
    },
    Invalid,
    Legacy,
}

const RUN_OUTPUT_CONTRACT_FILE: &str = "nimbus_output.json";
const LEGACY_OUTPUT_MANIFEST_VERSION: u64 = 2;
const PREVIOUS_OUTPUT_MANIFEST_VERSION: u64 = 3;
const PHYSICAL_OWNERSHIP_MANIFEST_VERSION: u64 = 4;
const TERRAIN_GEOMETRY_MANIFEST_VERSION: u64 = 5;
const FRAME_POLICY_MANIFEST_VERSION: u64 = 6;
const OUTPUT_MANIFEST_VERSION: u64 = 7;
const MAX_STEP_DIRECTORIES: usize = 1_000_000;
const MAX_STEP_ENTRIES: usize = 4096;
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
const MAX_CACHED_RUN_STEP_LISTS: usize = 64;

#[derive(Clone)]
struct CachedRunSteps {
    contract: RunOutputContract,
    index_sequence: Option<u64>,
    index_tip: Option<(String, String)>,
    steps: std::collections::BTreeMap<u32, CachedStep>,
}

#[derive(Clone)]
struct CachedStep {
    fingerprint: StepCacheFingerprint,
    manifest_sha256: Option<String>,
    terrain_geometry: Option<Arc<CachedTerrainGeometry>>,
    info: StepInfo,
}

struct CachedTerrainGeometry {
    manifest_version: u64,
    validation_manifest: serde_json::Value,
    validation_identity: Arc<str>,
}

#[derive(Clone, PartialEq, Eq)]
struct StepCacheFingerprint {
    directory_modified: Option<std::time::SystemTime>,
    manifest: Option<(u64, std::time::SystemTime)>,
}

static RUN_STEP_CACHE: Lazy<Mutex<HashMap<PathBuf, Arc<CachedRunSteps>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Deserialize)]
struct HistoryIndexHead {
    schema: String,
    version: u32,
    publication_generation: String,
    sequence: u64,
    reset_sequence: u64,
    segment: String,
    segment_sha256: String,
    checksum: String,
}

#[derive(Deserialize)]
struct HistoryIndexEntry {
    step_index: u32,
    step_dir: String,
    manifest_sha256: String,
}

#[derive(Deserialize)]
struct HistoryIndexSegment {
    schema: String,
    version: u32,
    publication_generation: String,
    sequence: u64,
    operation: String,
    previous_segment: Option<String>,
    previous_segment_sha256: Option<String>,
    entries: Vec<HistoryIndexEntry>,
    checksum: String,
}

fn history_head_payload(head: &HistoryIndexHead) -> String {
    format!(
        "{{\n  \"schema\": \"nimbus.history.index.head\",\n  \"version\": 1,\n  \"publication_generation\": \"{}\",\n  \"sequence\": {},\n  \"reset_sequence\": {},\n  \"segment\": \"{}\",\n  \"segment_sha256\": \"{}\"\n}}",
        head.publication_generation,
        head.sequence,
        head.reset_sequence,
        head.segment,
        head.segment_sha256
    )
}

fn history_segment_payload(segment: &HistoryIndexSegment) -> String {
    let previous = match (&segment.previous_segment, &segment.previous_segment_sha256) {
        (Some(path), Some(hash)) => {
            format!("\"{path}\",\n  \"previous_segment_sha256\": \"{hash}\"")
        }
        (None, None) => "null,\n  \"previous_segment_sha256\": null".into(),
        _ => return String::new(),
    };
    let mut output = format!(
        "{{\n  \"schema\": \"nimbus.history.index.segment\",\n  \"version\": 1,\n  \"publication_generation\": \"{}\",\n  \"sequence\": {},\n  \"operation\": \"{}\",\n  \"previous_segment\": {},\n  \"entries\": [",
        segment.publication_generation, segment.sequence, segment.operation, previous
    );
    for (position, entry) in segment.entries.iter().enumerate() {
        if position != 0 {
            output.push(',');
        }
        output.push_str(&format!(
            "\n    {{\"step_index\": {}, \"step_dir\": \"{}\", \"manifest_sha256\": \"{}\"}}",
            entry.step_index, entry.step_dir, entry.manifest_sha256
        ));
    }
    if !segment.entries.is_empty() {
        output.push_str("\n  ");
    }
    output.push_str("]\n}");
    output
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn sha256_file_hex(path: &Path) -> AppResult<String> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(digest.finalize().iter().map(|byte| format!("{byte:02x}")).collect())
}

fn read_history_index(
    dir: &Path,
    generation: &str,
    cached: Option<&CachedRunSteps>,
) -> AppResult<(
    u64,
    (String, String),
    std::collections::BTreeMap<u32, (PathBuf, String)>,
)> {
    if dir.join("history_index.pending.json").exists() {
        return Err(AppError::Invalid(
            "history index recovery is pending after an interrupted publication".into(),
        ));
    }
    let path = dir.join("history_index.json");
    let metadata = fs::symlink_metadata(&path).map_err(|error| {
        AppError::Invalid(format!(
            "required current history index is unavailable: {error}"
        ))
    })?;
    if !metadata.file_type().is_file() || metadata.len() > 1024 * 1024 * 1024 {
        return Err(AppError::Invalid(
            "required current history index is not a bounded regular file".into(),
        ));
    }
    let head: HistoryIndexHead = serde_json::from_str(&fs::read_to_string(&path)?)
        .map_err(|error| AppError::Invalid(format!("current history index is corrupt: {error}")))?;
    if head.schema != "nimbus.history.index.head"
        || head.version != 1
        || head.publication_generation != generation
        || head.checksum.len() != 64
        || sha256_hex(history_head_payload(&head).as_bytes()) != head.checksum
    {
        return Err(AppError::Invalid(
            "current history index identity or checksum is invalid".into(),
        ));
    }
    let valid_segment_path = |value: &str| {
        value == "history_index/active_a.json"
            || value == "history_index/active_b.json"
            || (value.starts_with("history_index/segment_")
                && value.ends_with(".json")
                && !value["history_index/".len()..].contains('/'))
    };
    if !valid_segment_path(&head.segment) || head.reset_sequence > head.sequence {
        return Err(AppError::Invalid(
            "current history index head path is invalid".into(),
        ));
    }
    let cached_tip = cached.and_then(|value| value.index_tip.as_ref());
    let fast_cached = cached.filter(|value| {
        value
            .index_sequence
            .is_some_and(|sequence| sequence >= head.reset_sequence)
    });
    let mut path = head.segment.clone();
    let mut expected_hash = head.segment_sha256.clone();
    let mut segments = Vec::new();
    let mut reached_cache = false;
    let mut completed_chain = false;
    let mut visited = std::collections::HashSet::new();
    for _ in 0..MAX_STEP_DIRECTORIES {
        if !visited.insert((path.clone(), expected_hash.clone())) {
            return Err(AppError::Invalid(
                "history index segment cycle detected".into(),
            ));
        }
        if cached_tip == Some(&(path.clone(), expected_hash.clone())) {
            reached_cache = true;
            completed_chain = true;
            break;
        }
        let bytes = fs::read(dir.join(&path))?;
        if sha256_hex(&bytes) != expected_hash {
            return Err(AppError::Invalid(
                "committed history index segment is corrupt".into(),
            ));
        }
        let segment: HistoryIndexSegment = serde_json::from_slice(&bytes).map_err(|error| {
            AppError::Invalid(format!("history index segment is corrupt: {error}"))
        })?;
        if segment.schema != "nimbus.history.index.segment"
            || segment.version != 1
            || segment.publication_generation != generation
            || segment.sequence > head.sequence
            || sha256_hex(history_segment_payload(&segment).as_bytes()) != segment.checksum
            || !matches!(segment.operation.as_str(), "append" | "reset")
            || segment.entries.len() > 256
        {
            return Err(AppError::Invalid(
                "history index segment contract is invalid".into(),
            ));
        }
        let reset = segment.operation == "reset";
        path = segment.previous_segment.clone().unwrap_or_default();
        expected_hash = segment.previous_segment_sha256.clone().unwrap_or_default();
        if !path.is_empty() && !valid_segment_path(&path) {
            return Err(AppError::Invalid(
                "history index previous-segment path is invalid".into(),
            ));
        }
        segments.push(segment);
        if fast_cached.is_some() {
            reached_cache = true;
            completed_chain = true;
            break;
        }
        if reset || path.is_empty() {
            completed_chain = true;
            break;
        }
    }
    if !completed_chain {
        return Err(AppError::Invalid(
            "history index segment chain exceeds its bound".into(),
        ));
    }
    let mut entries = if reached_cache {
        fast_cached
            .or(cached)
            .unwrap()
            .steps
            .iter()
            .filter_map(|(index, step)| {
                Some((
                    *index,
                    (
                        dir.join(format!("step_{index:06}")),
                        step.manifest_sha256.clone()?,
                    ),
                ))
            })
            .collect()
    } else {
        std::collections::BTreeMap::new()
    };
    for segment in segments.into_iter().rev() {
        if segment.operation == "reset" {
            entries.clear();
        }
        for entry in segment.entries {
            let expected = format!("step_{:06}", entry.step_index);
            if entry.step_dir != expected
                || entry.manifest_sha256.len() != 64
                || !entry
                    .manifest_sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit())
            {
                return Err(AppError::Invalid("history index entry is invalid".into()));
            }
            entries.insert(
                entry.step_index,
                (dir.join(entry.step_dir), entry.manifest_sha256),
            );
        }
    }
    Ok((head.sequence, (head.segment, head.segment_sha256), entries))
}

fn step_cache_fingerprint(path: &Path) -> StepCacheFingerprint {
    let directory_modified = path
        .metadata()
        .and_then(|metadata| metadata.modified())
        .ok();
    let manifest = path
        .join("manifest.json")
        .metadata()
        .ok()
        .and_then(|metadata| Some((metadata.len(), metadata.modified().ok()?)));
    StepCacheFingerprint {
        directory_modified,
        manifest,
    }
}

fn cache_terrain_geometry(
    manifest: &serde_json::Value,
    manifest_version: u64,
) -> Option<Arc<CachedTerrainGeometry>> {
    let grid = manifest.get("grid")?;
    let vertical = grid.get("vertical_geometry")?.as_object()?;
    let keys = [
        "kind", "center_coordinate", "interface_coordinate", "dtype", "layout", "storage",
        "contract", "content_sha256", "publication_generation", "geometry_manifest",
        "center_path", "interface_path",
    ];
    let vertical: serde_json::Map<String, serde_json::Value> = keys
        .into_iter()
        .filter_map(|key| vertical.get(key).map(|value| (key.to_owned(), value.clone())))
        .collect();
    let validation_manifest = serde_json::json!({
        "publication_generation": manifest.get("publication_generation"),
        "data_product": { "storage_format": manifest_storage_format(manifest) },
        "grid": {
            "dimensions": {
                "x": grid.pointer("/dimensions/x"),
                "y": grid.pointer("/dimensions/y"),
                "z": grid.pointer("/dimensions/z"),
            },
            "vertical_geometry": vertical,
        },
    });
    let validation_identity = Arc::from(format!("{manifest_version}:{validation_manifest}"));
    Some(Arc::new(CachedTerrainGeometry {
        manifest_version,
        validation_manifest,
        validation_identity,
    }))
}

type TerrainValidationMemo = HashMap<(PathBuf, Arc<str>), bool>;

fn cached_terrain_geometry_is_complete(
    entry: &CachedStep,
    step_dir: &Path,
    memo: &mut TerrainValidationMemo,
) -> bool {
    let Some(geometry) = &entry.terrain_geometry else { return true; };
    // Shared versioned paths are relative to the run; sidecars and netCDF remain
    // step-local. The complete contract keeps distinct shapes/generations apart.
    let base = if geometry.manifest_version >= OUTPUT_MANIFEST_VERSION
        && geometry.validation_manifest.pointer("/grid/vertical_geometry/storage")
            .and_then(|value| value.as_str()) == Some("versioned_npy_geometry")
    {
        step_dir.parent().unwrap_or(step_dir)
    } else {
        step_dir
    };
    let key = (base.to_path_buf(), Arc::clone(&geometry.validation_identity));
    *memo.entry(key).or_insert_with(|| {
        terrain_vertical_geometry_is_complete(
            &geometry.validation_manifest,
            step_dir,
            geometry.manifest_version,
        )
    })
}

const FIELD_WINDOW_SCHEMA_VERSION: u32 = 2;
const FIELD_WINDOW_RESPONSE_SCHEMA_VERSION: u32 = 5;
const FIELD_WINDOW_MAGIC: [u8; 8] = *b"NMBLOD1\0";
const MAX_FIELD_WINDOW_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_FIELD_WINDOW_BYTES: usize = 32 * 1024 * 1024;
const MAX_FIELD_WINDOW_SOURCE_BYTES: usize = 256 * 1024 * 1024;
const DEFAULT_FIELD_WINDOW_SOURCE_BYTES: usize = 64 * 1024 * 1024;
const MAX_ACTIVE_FIELD_READS: usize = 64;
const MAX_REQUEST_ID_BYTES: usize = 128;

/// Cancellation is explicit and request-scoped. This differs from the UI's
/// stale-result suppression: setting this flag makes cooperative native loops
/// stop reading/allocating. A blocking third-party decoder can only observe the
/// flag immediately before and after that call, which is surfaced as
/// `boundary_only` rather than being described as interruptible.
struct FieldReadControl {
    cancelled: AtomicBool,
    phase: AtomicU8,
}

impl FieldReadControl {
    fn new() -> Self {
        Self {
            cancelled: AtomicBool::new(false),
            phase: AtomicU8::new(ReadPhase::Registered as u8),
        }
    }

    fn set_phase(&self, phase: ReadPhase) {
        self.phase.store(phase as u8, Ordering::Release);
    }

    fn check(&self, request_id: &str) -> AppResult<()> {
        if self.cancelled.load(Ordering::Acquire) {
            Err(AppError::Task(format!(
                "field read cancelled: request '{request_id}'"
            )))
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Copy)]
#[repr(u8)]
enum ReadPhase {
    Registered = 0,
    NpyHeader = 1,
    NpyRows = 2,
    ZfpDecode = 3,
    Sampling = 4,
    NetcdfNative = 5,
}

fn phase_name(phase: u8) -> &'static str {
    match phase {
        1 => "npy_header",
        2 => "npy_rows",
        3 => "zfp_decode",
        4 => "sampling",
        5 => "netcdf_native",
        _ => "registered",
    }
}

fn phase_interruptibility(phase: u8) -> &'static str {
    if phase == ReadPhase::ZfpDecode as u8 || phase == ReadPhase::NetcdfNative as u8 {
        "boundary_only"
    } else {
        "cooperative"
    }
}

static ACTIVE_FIELD_READS: Lazy<Mutex<HashMap<String, Arc<FieldReadControl>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static FIELD_REDUCTION_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

#[derive(Clone)]
struct ValidatedFieldLodComponent {
    path: PathBuf,
    bytes: u64,
    modified: std::time::SystemTime,
}

#[derive(Clone)]
struct ValidatedFieldLod {
    path: PathBuf,
    bytes: u64,
    modified: std::time::SystemTime,
    shape: [usize; 3],
    source_shape: [usize; 3],
    source_offset: [usize; 3],
    stride: [usize; 3],
    geometry: Option<[ValidatedFieldLodComponent; 4]>,
}

static VALIDATED_FIELD_LODS: Lazy<Mutex<HashMap<String, ValidatedFieldLod>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
const MAX_VALIDATED_FIELD_LODS: usize = 4096;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FieldWindowRequest {
    pub schema_version: u32,
    pub request_id: String,
    pub aggregation: String,
    pub origin: Vec<usize>,
    pub extent: Vec<usize>,
    pub stride: Vec<usize>,
    pub max_output_bytes: Option<usize>,
    pub max_source_bytes: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct FieldWindowHeader {
    schema: &'static str,
    schema_version: u32,
    request_id: String,
    aggregation: String,
    weighting: &'static str,
    missing_data_policy: &'static str,
    sample_location: &'static str,
    geometry_kind: &'static str,
    geometry_dtype: Option<&'static str>,
    geometry_layout: Option<&'static str>,
    geometry_components: Option<Vec<&'static str>>,
    source_shape: Vec<usize>,
    origin: Vec<usize>,
    extent: Vec<usize>,
    stride: Vec<usize>,
    output_shape: Vec<usize>,
    element_count: usize,
    decoded_bytes: usize,
    geometry_decoded_bytes: usize,
    geometry_padding_bytes: usize,
    source_working_bytes: usize,
    source_working_accounting: &'static str,
    dtype: &'static str,
    order: &'static str,
    source_format: &'static str,
    cancellation_mode: &'static str,
    non_interruptible_phase: Option<&'static str>,
}

#[derive(Debug)]
struct FieldWindowPayload {
    source_shape: Vec<usize>,
    output_shape: Vec<usize>,
    field_body: Vec<u8>,
    geometry_body: Vec<u8>,
    weighting: &'static str,
    sample_location: &'static str,
    geometry_kind: &'static str,
    source_working_bytes: usize,
    source_format: &'static str,
    cancellation_mode: &'static str,
    non_interruptible_phase: Option<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelFieldReadResult {
    pub request_id: String,
    pub found: bool,
    /// `true` means the cancellation flag was set. Completion is confirmed by
    /// the read rejecting with the stable `field read cancelled:` prefix.
    pub cancellation_requested: bool,
    pub phase: Option<&'static str>,
    pub interruptibility: Option<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeFieldReadCapabilities {
    pub schema: &'static str,
    pub schema_version: u32,
    pub max_window_bytes: usize,
    pub max_source_bytes: usize,
    pub aggregations: [&'static str; 6],
    pub weighting: &'static str,
    pub terrain_following_window_read: bool,
    pub terrain_following_reductions: bool,
    pub geometry_limitation: &'static str,
    pub max_active_requests: usize,
    pub npy: NativeFormatCapability,
    pub zfp3d: NativeFormatCapability,
    pub netcdf_classic: NativeFormatCapability,
    pub netcdf4: NativeFormatCapability,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeFormatCapability {
    pub window_read: bool,
    pub cancellation: &'static str,
    pub limitation: Option<&'static str>,
}

#[tauri::command]
pub async fn list_step_dirs(run_dir: String) -> AppResult<Vec<StepInfo>> {
    // Scanning a run dir stats every step_*/ and reads each manifest.json (~90
    // files for a beast run); run it on the blocking pool so opening a run never
    // janks the UI thread.
    tauri::async_runtime::spawn_blocking(move || list_step_dirs_blocking(run_dir))
        .await
        .map_err(|e| AppError::Task(e.to_string()))?
}

/// Discover the canonical recursive `nests/child_NNNN` history graph below a
/// selected Nimbus run. The directory relationship is only a locator: every
/// frame still passes generation, product, geometry and time checks before the
/// TypeScript resolver admits it into a displayed branch.
#[tauri::command]
pub async fn list_nested_history_levels(
    run_dir: String,
    step_index: Option<u32>,
) -> AppResult<Vec<NestedHistoryLevel>> {
    tauri::async_runtime::spawn_blocking(move || {
        list_nested_history_levels_for_step_blocking(run_dir, step_index)
    })
    .await
    .map_err(|e| AppError::Task(e.to_string()))?
}

#[cfg(test)]
fn list_nested_history_levels_blocking(run_dir: String) -> AppResult<Vec<NestedHistoryLevel>> {
    list_nested_history_levels_for_step_blocking(run_dir, None)
}

fn list_nested_history_levels_for_step_blocking(
    run_dir: String,
    step_index: Option<u32>,
) -> AppResult<Vec<NestedHistoryLevel>> {
    let requested_dir = PathBuf::from(&run_dir);
    if !requested_dir.is_dir() {
        // Imported netCDF/CM1 datasets use a source identity rather than a
        // native history directory. They are valid plot sources but cannot own
        // Nimbus nested-history folders.
        return Ok(Vec::new());
    }
    let root = requested_dir.canonicalize()?;
    if !matches!(
        read_run_output_contract(&root),
        RunOutputContract::Current { ready: true, .. }
    ) {
        return Ok(Vec::new());
    }
    let mut levels = Vec::new();
    discover_nested_history_children(&root, "root", "", 1, step_index, &mut levels)?;
    Ok(levels)
}

const MAX_NESTED_HISTORY_DEPTH: usize = 8;
const MAX_NESTED_HISTORY_LEVELS: usize = 64;

fn discover_nested_history_children(
    parent_dir: &Path,
    parent_level_id: &str,
    parent_path_id: &str,
    depth: usize,
    step_index: Option<u32>,
    levels: &mut Vec<NestedHistoryLevel>,
) -> AppResult<()> {
    let nests = parent_dir.join("nests");
    let metadata = match fs::symlink_metadata(&nests) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    if depth > MAX_NESTED_HISTORY_DEPTH {
        return Err(AppError::Invalid(format!(
            "nested history exceeds the supported depth of {MAX_NESTED_HISTORY_DEPTH}"
        )));
    }
    if !metadata.file_type().is_dir() {
        return Err(AppError::Invalid(
            "nested history owner is not a regular directory".into(),
        ));
    }

    let mut children = std::collections::BTreeMap::<String, PathBuf>::new();
    for entry in fs::read_dir(&nests)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(index) = name.strip_prefix("child_") else {
            continue;
        };
        if index.len() != 4 || !index.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(AppError::Invalid(format!(
                "non-canonical nested history directory '{name}'; expected child_NNNN"
            )));
        }
        let canonical = entry.path().canonicalize()?;
        if canonical.parent() != Some(nests.as_path()) {
            return Err(AppError::Invalid(format!(
                "nested history escaped the selected run: {}",
                canonical.display()
            )));
        }
        if children.insert(name.to_owned(), canonical).is_some() {
            return Err(AppError::Invalid(format!(
                "duplicate nested history identity '{name}'"
            )));
        }
        if children.len() > 32 {
            return Err(AppError::Invalid(
                "run contains more than 32 direct nested histories".into(),
            ));
        }
    }

    for (local_id, path) in children {
        if levels.len() >= MAX_NESTED_HISTORY_LEVELS {
            return Err(AppError::Invalid(format!(
                "nested history exceeds the supported level count of {MAX_NESTED_HISTORY_LEVELS}"
            )));
        }
        let level_id = if parent_path_id.is_empty() {
            local_id
        } else {
            format!("{parent_path_id}/{local_id}")
        };
        let run_dir = path.display().to_string();
        let steps = list_step_dirs_filtered_blocking(run_dir.clone(), step_index)?;
        levels.push(NestedHistoryLevel {
            level_id: level_id.clone(),
            parent_level_id: parent_level_id.to_owned(),
            depth,
            run_dir,
            steps,
        });
        discover_nested_history_children(
            &path,
            &level_id,
            &level_id,
            depth + 1,
            step_index,
            levels,
        )?;
    }
    Ok(())
}

/// Synchronous worker behind `list_step_dirs`, run on the blocking pool.
pub(crate) fn list_step_dirs_blocking(run_dir: String) -> AppResult<Vec<StepInfo>> {
    list_step_dirs_filtered_blocking(run_dir, None)
}

fn list_step_dirs_filtered_blocking(
    run_dir: String,
    only_step: Option<u32>,
) -> AppResult<Vec<StepInfo>> {
    let requested_dir = PathBuf::from(&run_dir);
    if !requested_dir.is_dir() {
        return Err(AppError::NotFound(run_dir));
    }
    let dir = requested_dir.canonicalize()?;

    let run_contract = read_run_output_contract(&dir);
    let cached = RUN_STEP_CACHE
        .lock()
        .ok()
        .and_then(|cache| cache.get(&dir).cloned())
        .filter(|entry| entry.contract == run_contract);
    let indexed = if only_step.is_none() {
        match &run_contract {
            RunOutputContract::Current {
                generation,
                ready: true,
                history_index_required: true,
                ..
            } => Some(read_history_index(&dir, generation, cached.as_deref())?),
            _ => None,
        }
    } else {
        None
    };
    let mut terrain_validation = TerrainValidationMemo::new();
    if let (Some(cache), Some((sequence, _, _))) = (&cached, &indexed) {
        if cache.index_sequence == Some(*sequence) {
            if cache.steps.iter().all(|(index, entry)| {
                entry.info.publication_state == StepPublicationState::Complete
                    && cached_terrain_geometry_is_complete(
                        entry,
                        &dir.join(format!("step_{index:06}")),
                        &mut terrain_validation,
                    )
            }) {
                return Ok(cache
                    .steps
                    .values()
                    .map(|entry| entry.info.clone())
                    .collect());
            }
        } else if cache
            .index_sequence
            .is_some_and(|cached_sequence| *sequence <= cached_sequence)
        {
            return Err(AppError::Invalid(
                "current history index cursor is not monotonic".into(),
            ));
        }
    }
    let mut steps: Vec<StepInfo> = Vec::new();
    let mut next_cache = std::collections::BTreeMap::new();
    let mut step_paths = std::collections::BTreeMap::<u32, PathBuf>::new();
    let mut indexed_hashes = std::collections::BTreeMap::<u32, String>::new();
    if let Some(index) = only_step {
        let path = dir.join(format!("step_{index:06}"));
        match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type().is_dir() => {
                let canonical = path.canonicalize()?;
                if canonical.parent() != Some(dir.as_path()) {
                    return Err(AppError::Invalid(format!(
                        "step directory escaped the selected run: {}",
                        canonical.display()
                    )));
                }
                step_paths.insert(index, canonical);
            }
            Ok(_) => {
                return Err(AppError::Invalid(format!(
                    "{} is not a regular step directory",
                    path.display()
                )))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    } else if let Some((_, _, entries)) = &indexed {
        for (index, (path, checksum)) in entries {
            step_paths.insert(*index, path.clone());
            indexed_hashes.insert(*index, checksum.clone());
        }
    } else {
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            let Some(index_str) = name.strip_prefix("step_") else {
                continue;
            };
            let Ok(index) = index_str.parse::<u32>() else {
                continue;
            };
            if name != format!("step_{index:06}") {
                return Err(AppError::Invalid(format!(
                    "non-canonical Nimbus step directory '{name}'; expected step_<zero-padded index>"
                )));
            }
            let path = entry.path().canonicalize()?;
            if path.parent() != Some(dir.as_path()) {
                return Err(AppError::Invalid(format!(
                    "step directory escaped the selected run: {}",
                    path.display()
                )));
            }
            if step_paths.insert(index, path).is_some() {
                return Err(AppError::Invalid(format!(
                    "duplicate Nimbus step identity {index}"
                )));
            }
            if step_paths.len() > MAX_STEP_DIRECTORIES {
                return Err(AppError::Invalid(format!(
                    "run contains more than {MAX_STEP_DIRECTORIES} step directories"
                )));
            }
        }
    }

    for (index, path) in step_paths {
        let name = format!("step_{index:06}");
        let indexed_sha256 = indexed_hashes.get(&index);

        if let Some(entry) = cached
            .as_ref()
            .and_then(|entry| entry.steps.get(&index))
            .filter(|entry| {
                indexed_sha256.is_some()
                    && entry.info.publication_state == StepPublicationState::Complete
                    && entry.manifest_sha256.as_ref() == indexed_sha256
                    && cached_terrain_geometry_is_complete(entry, &path, &mut terrain_validation)
            })
        {
            next_cache.insert(index, entry.clone());
            steps.push(entry.info.clone());
            continue;
        }

        let fingerprint = step_cache_fingerprint(&path);

        if let Some(entry) = cached
            .as_ref()
            .and_then(|entry| entry.steps.get(&index))
            .filter(|entry| {
                entry.info.publication_state == StepPublicationState::Complete
                    && entry.fingerprint == fingerprint
                    && cached_terrain_geometry_is_complete(entry, &path, &mut terrain_validation)
            })
        {
            let info = entry.info.clone();
            next_cache.insert(index, entry.clone());
            steps.push(info);
            continue;
        }

        // Surface standalone NPY/ZFP fields first. A current v3 netCDF frame
        // adds fields from its exact shared-file manifest mapping below.
        let mut field_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let mut step_entry_count = 0_usize;
        for e in fs::read_dir(&path)?.filter_map(|e| e.ok()) {
            step_entry_count += 1;
            if step_entry_count > MAX_STEP_ENTRIES {
                return Err(AppError::Invalid(format!(
                    "{} contains more than {MAX_STEP_ENTRIES} entries",
                    path.display()
                )));
            }
            if !e.file_type().is_ok_and(|kind| kind.is_file()) {
                continue;
            }
            let p = e.path();
            let Some(ext) = p.extension().and_then(|x| x.to_str()) else {
                continue;
            };
            if ext == "npy" || ext == "zfp3d" {
                if let Some(stem) = p.file_stem().and_then(|x| x.to_str()) {
                    // v5 terrain coordinates are lossless grid metadata, not
                    // selectable atmospheric fields and therefore are not in
                    // published_fields or the canonical field catalog.
                    if stem != "physical_z_center_m" && stem != "physical_z_interface_m"
                        && !is_field_lod_payload_stem(stem) {
                        field_set.insert(stem.to_string());
                    }
                }
            }
        }
        let manifest_path = path.join("manifest.json");
        let manifest_read = match fs::symlink_metadata(&manifest_path) {
            Ok(metadata)
                if metadata.file_type().is_file() && metadata.len() <= MAX_MANIFEST_BYTES =>
            {
                fs::read_to_string(&manifest_path)
            }
            Ok(_) => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "manifest is non-regular or exceeds 1 MiB",
            )),
            Err(error) => Err(error),
        };
        let manifest_missing = manifest_read
            .as_ref()
            .err()
            .is_some_and(|error| error.kind() == std::io::ErrorKind::NotFound);
        let manifest_value = manifest_read
            .as_ref()
            .ok()
            .map(String::as_str)
            .and_then(|t| serde_json::from_str::<serde_json::Value>(t).ok());
        if let Some(expected_sha256) = indexed_sha256 {
            let actual = manifest_read
                .as_ref()
                .ok()
                .map(|text| sha256_hex(text.as_bytes()));
            if actual.as_deref() != Some(expected_sha256.as_str()) {
                return Err(AppError::Invalid(format!(
                    "history index checksum does not match {name}/manifest.json"
                )));
            }
        }

        if let Some(netcdf_fields) = manifest_value
            .as_ref()
            .and_then(|manifest| netcdf_manifest_fields(manifest, &path))
        {
            field_set.extend(netcdf_fields);
        }
        let fields: Vec<String> = field_set.into_iter().collect();

        let (publication_state, validated_manifest) = match &run_contract {
            RunOutputContract::Current {
                generation,
                ready,
                manifest_version,
                ..
            } if *ready => match manifest_value.as_ref() {
                None if manifest_missing => (StepPublicationState::Incomplete, None),
                None => (StepPublicationState::InvalidManifest, None),
                Some(manifest)
                    if validate_current_step_manifest(
                        manifest,
                        generation,
                        index,
                        &name,
                        &fields,
                        &path,
                        *manifest_version,
                        expected_history_product_kind(&dir),
                    ) =>
                {
                    (StepPublicationState::Complete, Some(manifest))
                }
                Some(_) => (StepPublicationState::InvalidManifest, None),
            },
            RunOutputContract::Current { .. } => (StepPublicationState::Incomplete, None),
            RunOutputContract::Invalid => (StepPublicationState::InvalidManifest, None),
            RunOutputContract::Legacy => (StepPublicationState::Legacy, manifest_value.as_ref()),
        };

        let publication_generation = match &run_contract {
            RunOutputContract::Current { generation, .. } => Some(generation.clone()),
            RunOutputContract::Invalid | RunOutputContract::Legacy => None,
        };

        let simulation_time_s = validated_manifest
            .and_then(|v| v.get("simulation_time_s"))
            .and_then(|t| t.as_f64());

        let grid_value = validated_manifest.and_then(|v| v.get("grid"));
        let grid = grid_value.map(parse_grid_info);
        let motion = grid_value.and_then(parse_frame_motion);

        let field_units = validated_manifest
            .map(parse_field_units)
            .unwrap_or_default();
        let field_compression = validated_manifest
            .map(parse_field_compression)
            .unwrap_or_default();
        let field_brick_hierarchies = validated_manifest
            .map(parse_field_brick_hierarchies)
            .unwrap_or_default();
        let compression_execution = validated_manifest
            .and_then(parse_compression_execution);

        let info = StepInfo {
            index,
            simulation_time_s,
            fields,
            grid,
            field_units,
            field_compression,
            field_brick_hierarchies,
            compression_execution,
            publication_state,
            publication_generation,
            motion,
        };
        next_cache.insert(
            index,
            CachedStep {
                fingerprint,
                manifest_sha256: indexed_sha256.cloned(),
                terrain_geometry: validated_manifest.and_then(|manifest| match &run_contract {
                    RunOutputContract::Current { manifest_version, .. } =>
                        cache_terrain_geometry(manifest, *manifest_version),
                    RunOutputContract::Invalid | RunOutputContract::Legacy => None,
                }),
                info: info.clone(),
            },
        );
        steps.push(info);
    }

    steps.sort_by_key(|s| s.index);
    if only_step.is_none() {
        if let Ok(mut cache) = RUN_STEP_CACHE.lock() {
            if !cache.contains_key(&dir) && cache.len() >= MAX_CACHED_RUN_STEP_LISTS {
                if let Some(key) = cache.keys().next().cloned() {
                    cache.remove(&key);
                }
            }
            cache.insert(
                dir,
                Arc::new(CachedRunSteps {
                    contract: run_contract,
                    index_sequence: indexed.as_ref().map(|(sequence, _, _)| *sequence),
                    index_tip: indexed.as_ref().map(|(_, tip, _)| tip.clone()),
                    steps: next_cache,
                }),
            );
        }
    }
    Ok(steps)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct QualifiedOutputSummary {
    pub complete_frame_count: u64,
    pub latest_step: Option<u64>,
    pub output_format: Option<String>,
}

/// Reuse Plot Studio's current-publication validator for runtime evidence.
/// Legacy, incomplete, duplicate, non-canonical, or malformed steps never
/// become a "complete frame" provenance claim.
pub(crate) fn inspect_qualified_output_summary(
    run_dir: &Path,
) -> AppResult<QualifiedOutputSummary> {
    let canonical_run = run_dir.canonicalize()?;
    if !matches!(
        read_run_output_contract(&canonical_run),
        RunOutputContract::Current { ready: true, .. }
    ) {
        return Err(AppError::Invalid(
            "run does not have a ready current Nimbus publication contract".into(),
        ));
    }
    let steps = list_step_dirs_blocking(canonical_run.display().to_string())?;
    if let Some(step) = steps
        .iter()
        .find(|step| step.publication_state != StepPublicationState::Complete)
    {
        return Err(AppError::Invalid(format!(
            "step_{:06} is not a qualified complete publication ({:?})",
            step.index, step.publication_state
        )));
    }
    let latest_step = steps.last().map(|step| u64::from(step.index));
    let output_format = if let Some(index) = latest_step {
        let manifest_path = canonical_run.join(format!("step_{index:06}/manifest.json"));
        let metadata = fs::symlink_metadata(&manifest_path)?;
        if !metadata.file_type().is_file() || metadata.len() > MAX_MANIFEST_BYTES {
            return Err(AppError::Invalid(
                "qualified output manifest became non-regular or oversized".into(),
            ));
        }
        let manifest: serde_json::Value = serde_json::from_str(&fs::read_to_string(manifest_path)?)
            .map_err(|error| {
                AppError::Invalid(format!("qualified output manifest changed: {error}"))
            })?;
        manifest_storage_format(&manifest).map(|format| match format {
            "npy_3d" | "npy_2d_slices" => "NPY".to_string(),
            "zfp" => "ZFP".to_string(),
            "netcdf" => "netCDF".to_string(),
            other => other.to_string(),
        })
    } else {
        None
    };
    Ok(QualifiedOutputSummary {
        complete_frame_count: steps.len() as u64,
        latest_step,
        output_format,
    })
}

/// Read the run-level publication contract. Absence alone means legacy; a
/// present-but-invalid marker fails closed so corruption can never turn a
/// current run into the permissive legacy path.
/// What a reader must require of each step manifest in a run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RunGenerationGate {
    /// Every complete manifest must carry exactly this generation. A manifest
    /// with any other value belongs to a superseded run that was interrupted
    /// before its step directories were cleared.
    Require(String),
    /// Pre-contract run: there is no marker, so generation cannot be checked.
    Legacy,
    /// Marker present but unusable — nothing in the run can be trusted.
    Invalid,
}

/// Exposed so every reader applies the same identity gate as `list_step_dirs`.
/// Checking `publication_state` alone accepts stale-generation manifests left
/// by an interrupted rerun.
pub(crate) fn run_generation_gate(run_dir: &Path) -> RunGenerationGate {
    match read_run_output_contract(run_dir) {
        RunOutputContract::Current { generation, .. } => RunGenerationGate::Require(generation),
        RunOutputContract::Legacy => RunGenerationGate::Legacy,
        RunOutputContract::Invalid => RunGenerationGate::Invalid,
    }
}

fn publication_contract_root(run_dir: &Path) -> &Path {
    let mut current = run_dir;
    loop {
        let child_name_is_canonical = current
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.strip_prefix("child_"))
            .is_some_and(|index| {
                index.len() == 4 && index.bytes().all(|byte| byte.is_ascii_digit())
            });
        let Some(nests) = current.parent() else { break };
        let Some(parent) = nests.parent() else { break };
        if !child_name_is_canonical
            || nests.file_name().and_then(|name| name.to_str()) != Some("nests")
        {
            break;
        }
        current = parent;
    }
    if current.join(RUN_OUTPUT_CONTRACT_FILE).is_file() {
        current
    } else {
        run_dir
    }
}

fn expected_history_product_kind(run_dir: &Path) -> &'static str {
    if publication_contract_root(run_dir) == run_dir {
        "history"
    } else {
        "nested_child_history"
    }
}

fn read_run_output_contract(run_dir: &Path) -> RunOutputContract {
    let path = publication_contract_root(run_dir).join(RUN_OUTPUT_CONTRACT_FILE);
    match fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.file_type().is_file() && metadata.len() <= MAX_MANIFEST_BYTES => {}
        Ok(_) => return RunOutputContract::Invalid,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return RunOutputContract::Legacy,
        Err(_) => return RunOutputContract::Invalid,
    }
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return RunOutputContract::Legacy,
        Err(_) => return RunOutputContract::Invalid,
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return RunOutputContract::Invalid;
    };
    let manifest_version = value.get("manifest_version").and_then(|v| v.as_u64());
    let valid = value.get("schema").and_then(|v| v.as_str()) == Some("nimbus.output")
        && manifest_version.is_some_and(|version| {
            version == LEGACY_OUTPUT_MANIFEST_VERSION
                || version == PREVIOUS_OUTPUT_MANIFEST_VERSION
                || version == PHYSICAL_OWNERSHIP_MANIFEST_VERSION
                || version == TERRAIN_GEOMETRY_MANIFEST_VERSION
                || version == FRAME_POLICY_MANIFEST_VERSION
                || version == OUTPUT_MANIFEST_VERSION
        })
        && value.get("frame_completion").and_then(|v| v.as_str()) == Some("atomic_step_manifest");
    let generation = value.get("publication_generation").and_then(|v| v.as_str());
    let state = value.get("publication_state").and_then(|v| v.as_str());
    let history_index_required = match value.get("history_index") {
        None => false,
        Some(index)
            if index.get("schema").and_then(|v| v.as_str()) == Some("nimbus.history.index")
                && index.get("version").and_then(|v| v.as_u64()) == Some(1)
                && index.get("checksum").and_then(|v| v.as_str()) == Some("sha256") =>
        {
            true
        }
        Some(_) => return RunOutputContract::Invalid,
    };
    if !valid || !generation.is_some_and(valid_publication_generation) {
        return RunOutputContract::Invalid;
    }
    match state {
        Some("initializing") => RunOutputContract::Current {
            generation: generation.unwrap().to_owned(),
            ready: false,
            manifest_version: manifest_version.unwrap(),
            history_index_required,
        },
        Some("ready") => RunOutputContract::Current {
            generation: generation.unwrap().to_owned(),
            ready: true,
            manifest_version: manifest_version.unwrap(),
            history_index_required,
        },
        _ => RunOutputContract::Invalid,
    }
}

fn valid_publication_generation(generation: &str) -> bool {
    generation.len() == 32 && generation.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn manifest_storage_format(manifest: &serde_json::Value) -> Option<&str> {
    manifest
        .get("data_product")?
        .get("storage_format")?
        .as_str()
}

/// Resolve the field inventory of a v3 shared-file netCDF frame. Returning
/// `None` is deliberately fail-closed: a missing/empty history file or any
/// incomplete field-to-variable mapping keeps the frame out of Studio.
fn netcdf_manifest_fields(
    manifest: &serde_json::Value,
    step_dir: &Path,
) -> Option<std::collections::BTreeSet<String>> {
    if manifest_storage_format(manifest) != Some("netcdf") {
        return None;
    }
    let history = step_dir.join("history.nc");
    if !fs::metadata(history)
        .ok()
        .is_some_and(|metadata| metadata.is_file() && metadata.len() > 0)
    {
        return None;
    }
    let published = manifest.get("published_fields")?.as_array()?;
    let catalog = manifest.get("fields")?.as_array()?;
    if published.is_empty() {
        return None;
    }
    let mut fields = std::collections::BTreeSet::new();
    for value in published {
        let field = value.as_str().filter(|field| !field.is_empty())?;
        let mapped = catalog.iter().any(|entry| {
            entry.get("field_id").and_then(|value| value.as_str()) == Some(field)
                && entry.get("file_pattern").and_then(|value| value.as_str()) == Some("history.nc")
                && entry.get("storage_layout").and_then(|value| value.as_str())
                    == Some("shared_file")
                && entry.get("variable_name").and_then(|value| value.as_str()) == Some(field)
        });
        if !mapped || !fields.insert(field.to_owned()) {
            return None;
        }
    }
    Some(fields)
}

/// Validate the completion marker before exposing a current-format frame. The
/// final manifest is useful only when it identifies this exact directory,
/// carries a complete grid, declares every visible field, and every visible
/// payload is non-empty. Atomic publication guarantees these checks never see
/// a half-written final JSON file from the simulator.
// These values jointly identify the exact atomic publication being validated.
#[allow(clippy::too_many_arguments)]
fn validate_current_step_manifest(
    manifest: &serde_json::Value,
    expected_generation: &str,
    step_index: u32,
    step_dir_name: &str,
    fields: &[String],
    step_dir: &Path,
    expected_manifest_version: u64,
    expected_product_kind: &str,
) -> bool {
    if manifest.get("schema").and_then(|v| v.as_str()) != Some("nimbus.step")
        || manifest.get("manifest_version").and_then(|v| v.as_u64())
            != Some(expected_manifest_version)
        || manifest.get("publication_state").and_then(|v| v.as_str()) != Some("complete")
        || manifest
            .get("publication_generation")
            .and_then(|v| v.as_str())
            != Some(expected_generation)
        || manifest.get("step_index").and_then(|v| v.as_u64()) != Some(u64::from(step_index))
        || manifest.get("step_dir").and_then(|v| v.as_str()) != Some(step_dir_name)
        || !manifest
            .get("simulation_time_s")
            .and_then(|v| v.as_f64())
            .is_some_and(f64::is_finite)
    {
        return false;
    }

    let Some(grid) = manifest.get("grid") else {
        return false;
    };
    if !grid_is_complete_for_version(grid, expected_manifest_version) {
        return false;
    }
    if expected_manifest_version >= PHYSICAL_OWNERSHIP_MANIFEST_VERSION
        && !canonical_history_contract_is_complete(manifest, expected_product_kind)
    {
        return false;
    }

    let Some(published_values) = manifest
        .get("published_fields")
        .and_then(|value| value.as_array())
    else {
        return false;
    };
    let published: std::collections::BTreeSet<&str> = published_values
        .iter()
        .filter_map(|value| value.as_str())
        .filter(|field| !field.is_empty())
        .collect();
    let visible: std::collections::BTreeSet<&str> = fields.iter().map(String::as_str).collect();
    if published.is_empty() || published.len() != published_values.len() || published != visible {
        return false;
    }

    let Some(catalog) = manifest.get("fields").and_then(|v| v.as_array()) else {
        return false;
    };
    if fields.is_empty()
        || fields
            .iter()
            .any(|field| !manifest_declares_field(catalog, field))
        || (expected_manifest_version >= PHYSICAL_OWNERSHIP_MANIFEST_VERSION
            && fields
                .iter()
                .any(|field| !canonical_field_metadata_is_complete(catalog, field)))
    {
        return false;
    }

    let has_v2_field = catalog.iter().any(|entry| {
        entry.get("compression").is_some_and(|compression| {
            compression.get("contract_version").and_then(|value| value.as_u64()) == Some(2)
                || compression.get("container_version").and_then(|value| value.as_u64()) == Some(4)
                || compression.get("mode").and_then(|value| value.as_str()) == Some("reversible")
        })
    });
    if (manifest.get("data_product").and_then(|product| product.get("compression_contract")).is_some()
        || has_v2_field)
        && !mixed_zfp_v2_is_complete(manifest, fields, step_index, step_dir)
    {
        return false;
    }
    if expected_manifest_version >= OUTPUT_MANIFEST_VERSION
        && manifest_storage_format(manifest) == Some("zfp")
        && manifest.get("data_product").and_then(|product| product.get("compression_contract")).is_none()
        && !zfp_v1_field_storage_is_complete(manifest, fields, step_dir)
    {
        return false;
    }

    if !terrain_vertical_geometry_is_complete(manifest, step_dir, expected_manifest_version) {
        return false;
    }
    if !field_brick_hierarchy_dependencies_are_complete(manifest, step_dir) {
        return false;
    }

    if manifest_storage_format(manifest) == Some("netcdf") {
        return netcdf_manifest_fields(manifest, step_dir)
            .is_some_and(|mapped| mapped.iter().eq(fields.iter()));
    }

    fields.iter().all(|field| {
        let npy = step_dir.join(format!("{field}.npy"));
        let preferred = if npy.exists() {
            npy
        } else {
            step_dir.join(format!("{field}.zfp3d"))
        };
        fs::metadata(preferred)
            .map(|metadata| metadata.is_file() && metadata.len() > 0)
            .unwrap_or(false)
    })
}

fn is_field_lod_payload_stem(stem: &str) -> bool {
    let Some((_, suffix)) = stem.rsplit_once(".lod") else {
        return false;
    };
    !suffix.is_empty()
        && suffix.bytes().all(|byte| byte.is_ascii_digit())
        && suffix.parse::<u32>().is_ok_and(|level| level > 0)
}

fn field_brick_hierarchy_dependencies_are_complete(
    manifest: &serde_json::Value,
    step_dir: &Path,
) -> bool {
    let declared = parse_field_brick_hierarchies(manifest);
    let catalog = match manifest.get("fields").and_then(|value| value.as_array()) {
        Some(catalog) => catalog,
        None => return false,
    };
    let hierarchy_count = catalog.iter()
        .filter(|entry| entry.get("brick_hierarchy").is_some())
        .count();
    if hierarchy_count != declared.len() {
        return false;
    }
    for (field, info) in declared {
        let path = step_dir.join(&info.path);
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type().is_file()
                && metadata.len() > 0 && metadata.len() <= 64 * 1024 * 1024 => metadata,
            _ => return false,
        };
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(_) => return false,
        };
        if bytes.len() as u64 != metadata.len() || sha256_hex(&bytes) != info.content_sha256 {
            return false;
        }
        let hierarchy: serde_json::Value = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(_) => return false,
        };
        if hierarchy.get("schema").and_then(|value| value.as_str())
                != Some("nimbus.field-brick-hierarchy")
            || hierarchy.get("schemaVersion").and_then(|value| value.as_u64())
                != Some(u64::from(info.schema_version))
            || hierarchy.get("field").and_then(|value| value.as_str()) != Some(field.as_str())
            || hierarchy.get("publicationGeneration").and_then(|value| value.as_str())
                != manifest.get("publication_generation").and_then(|value| value.as_str()) {
            return false;
        }
        if info.schema_version < 2 {
            continue;
        }
        let levels = match hierarchy.get("levels").and_then(|value| value.as_array()) {
            Some(levels) if !levels.is_empty() => levels,
            _ => return false,
        };
        for (position, level) in levels.iter().enumerate() {
            let source = match level.get("source").and_then(|value| value.as_object()) {
                Some(source) => source,
                None => return false,
            };
            if position == 0 {
                if source.get("kind").and_then(|value| value.as_str()) != Some("primary_field") {
                    return false;
                }
                continue;
            }
            let expected = format!("{field}.lod{position}.npy");
            if level.get("level").and_then(|value| value.as_u64()) != Some(position as u64)
                || source.get("kind").and_then(|value| value.as_str()) != Some("npy_lod")
                || source.get("path").and_then(|value| value.as_str()) != Some(expected.as_str())
                || source.get("aggregation").and_then(|value| value.as_str())
                    != Some("recursive_cell_mean_2x2x2") {
                return false;
            }
            if !fs::symlink_metadata(step_dir.join(expected)).is_ok_and(|metadata|
                metadata.file_type().is_file() && metadata.len() > 0) {
                return false;
            }
            if info.schema_version == 3 {
                let geometry = match source.get("geometry").and_then(|value| value.as_object()) {
                    Some(geometry) if geometry.get("kind").and_then(|value| value.as_str())
                        == Some("physical_volume_centroids") => geometry,
                    _ => return false,
                };
                for (component, prefix) in [
                    ("xM", "physical_x_center_m"),
                    ("yM", "physical_y_center_m"),
                    ("zM", "physical_z_center_m"),
                    ("volumeM3", "physical_volume_m3"),
                ] {
                    let expected_geometry = format!("{prefix}.lod{position}.npy");
                    if geometry.get(component)
                            .and_then(|value| value.get("path"))
                            .and_then(|value| value.as_str())
                            != Some(expected_geometry.as_str())
                        || !fs::symlink_metadata(step_dir.join(expected_geometry))
                            .is_ok_and(|metadata| metadata.file_type().is_file()
                                && metadata.len() > 0) {
                        return false;
                    }
                }
            }
        }
    }
    true
}

#[cfg(test)]
thread_local! {
    static TERRAIN_GEOMETRY_VALIDATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn terrain_vertical_geometry_is_complete(
    manifest: &serde_json::Value,
    step_dir: &Path,
    manifest_version: u64,
) -> bool {
    #[cfg(test)]
    TERRAIN_GEOMETRY_VALIDATIONS.with(|count| count.set(count.get() + 1));
    let Some(vertical) = manifest
        .get("grid")
        .and_then(|grid| grid.get("vertical_geometry"))
    else {
        return true;
    };
    if manifest_version < TERRAIN_GEOMETRY_MANIFEST_VERSION
        || vertical.get("kind").and_then(|value| value.as_str()) != Some("terrain_following")
        || vertical
            .get("center_coordinate")
            .and_then(|value| value.as_str())
            != Some("physical_z_center_m")
        || vertical
            .get("interface_coordinate")
            .and_then(|value| value.as_str())
            != Some("physical_z_interface_m")
        || vertical.get("dtype").and_then(|value| value.as_str()) != Some("float64")
        || vertical.get("layout").and_then(|value| value.as_str()) != Some("x_y_z")
    {
        return false;
    }
    match manifest_storage_format(manifest) {
        Some("netcdf") => {
            vertical.get("storage").and_then(|value| value.as_str()) == Some("netcdf_variable")
                && step_dir
                    .join("history.nc")
                    .metadata()
                    .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
        }
        Some("npy_3d" | "zfp") => {
            let dimensions = manifest.get("grid").and_then(|grid| grid.get("dimensions"));
            let shape = dimensions.and_then(|dimensions| {
                Some([
                    usize::try_from(dimensions.get("x")?.as_u64()?).ok()?,
                    usize::try_from(dimensions.get("y")?.as_u64()?).ok()?,
                    usize::try_from(dimensions.get("z")?.as_u64()?).ok()?,
                ])
            });
            shape.is_some_and(|[nx, ny, nz]| {
                let directory = match vertical.get("storage").and_then(|value| value.as_str()) {
                    Some("npy_sidecar") if manifest_version <= FRAME_POLICY_MANIFEST_VERSION => {
                        Some(step_dir.to_path_buf())
                    }
                    Some("versioned_npy_geometry")
                        if manifest_version >= OUTPUT_MANIFEST_VERSION =>
                    {
                        versioned_terrain_geometry_directory(manifest, step_dir, [nx, ny, nz])
                    }
                    _ => None,
                };
                directory.is_some_and(|directory| {
                    validate_f64_npy_geometry(
                        &directory.join("physical_z_center_m.npy"),
                        [nx, ny, nz],
                    ) && nz.checked_add(1).is_some_and(|interfaces| {
                        validate_f64_npy_geometry(
                            &directory.join("physical_z_interface_m.npy"),
                            [nx, ny, interfaces],
                        )
                    })
                })
            })
        }
        _ => false,
    }
}

fn versioned_terrain_geometry_directory(
    step_manifest: &serde_json::Value,
    step_dir: &Path,
    shape: [usize; 3],
) -> Option<std::path::PathBuf> {
    let vertical = step_manifest.get("grid")?.get("vertical_geometry")?;
    let generation = step_manifest.get("publication_generation")?.as_str()?;
    let reference_generation = vertical.get("publication_generation")?.as_str()?;
    let digest = vertical.get("content_sha256")?.as_str()?;
    if reference_generation != generation
        || digest.len() != 64
        || !digest.bytes().all(|byte| byte.is_ascii_hexdigit())
        || vertical.get("contract")?.as_str()? != "nimbus.terrain-geometry.v1"
    {
        return None;
    }
    let relative_root = format!("../geometry/{generation}/{digest}");
    if vertical.get("geometry_manifest")?.as_str()? != format!("{relative_root}/manifest.json")
        || vertical.get("center_path")?.as_str()?
            != format!("{relative_root}/physical_z_center_m.npy")
        || vertical.get("interface_path")?.as_str()?
            != format!("{relative_root}/physical_z_interface_m.npy")
    {
        return None;
    }
    let directory = step_dir
        .parent()?
        .join("geometry")
        .join(generation)
        .join(digest);
    let metadata = fs::symlink_metadata(directory.join("manifest.json")).ok()?;
    if !metadata.file_type().is_file() || metadata.len() > MAX_MANIFEST_BYTES {
        return None;
    }
    let geometry: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(directory.join("manifest.json")).ok()?).ok()?;
    let expected_center = serde_json::json!([shape[0], shape[1], shape[2]]);
    let expected_interface = serde_json::json!([shape[0], shape[1], shape[2].checked_add(1)?]);
    if geometry.get("schema")?.as_str()? != "nimbus.terrain-geometry"
        || geometry.get("version")?.as_u64()? != 1
        || geometry.get("publication_generation")?.as_str()? != generation
        || geometry.get("content_sha256")?.as_str()? != digest
        || geometry.get("identity_encoding")?.as_str()? != "shape_u64_le_then_ieee754_f64_le"
        || geometry.get("dtype")?.as_str()? != "float64"
        || geometry.get("layout")?.as_str()? != "x_y_z"
        || geometry.get("center_shape")? != &expected_center
        || geometry.get("interface_shape")? != &expected_interface
    {
        return None;
    }
    Some(directory)
}

fn validate_f64_npy_geometry(path: &Path, expected_shape: [usize; 3]) -> bool {
    let Ok(metadata) = path.metadata() else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    let Ok(mut file) = fs::File::open(path) else {
        return false;
    };
    let mut prefix = [0_u8; 10];
    if file.read_exact(&mut prefix).is_err()
        || &prefix[..6] != b"\x93NUMPY"
        || prefix[6] != 1
        || prefix[7] != 0
    {
        return false;
    }
    let header_length = usize::from(u16::from_le_bytes([prefix[8], prefix[9]]));
    if header_length == 0 || header_length > crate::npy::MAX_NPY_HEADER_BYTES {
        return false;
    }
    let mut header = vec![0_u8; header_length];
    if file.read_exact(&mut header).is_err() {
        return false;
    }
    let Ok(header) = std::str::from_utf8(&header) else {
        return false;
    };
    let compact: String = header
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    let expected = format!(
        "'shape':({},{},{}),",
        expected_shape[0], expected_shape[1], expected_shape[2]
    );
    if compact.matches("'descr':").count() != 1
        || !compact.contains("'descr':'<f8'")
        || compact.matches("'fortran_order':").count() != 1
        || !compact.contains("'fortran_order':False")
        || compact.matches("'shape':").count() != 1
        || !compact.contains(&expected)
    {
        return false;
    }
    let Some(elements) = expected_shape
        .into_iter()
        .try_fold(1_usize, usize::checked_mul)
    else {
        return false;
    };
    let Some(expected_bytes) = 10_usize.checked_add(header_length).and_then(|header| {
        elements
            .checked_mul(size_of::<f64>())
            .and_then(|body| header.checked_add(body))
    }) else {
        return false;
    };
    u64::try_from(expected_bytes).ok() == Some(metadata.len())
}

fn canonical_history_contract_is_complete(
    manifest: &serde_json::Value,
    expected_product_kind: &str,
) -> bool {
    let product = manifest.get("data_product");
    let identity = manifest.get("field_identity");
    product
        .and_then(|value| value.get("kind"))
        .and_then(|value| value.as_str())
        == Some(expected_product_kind)
        && product
            .and_then(|value| value.get("restart_complete"))
            .and_then(|value| value.as_bool())
            == Some(false)
        && product
            .and_then(|value| value.get("storage_format"))
            .and_then(|value| value.as_str())
            .is_some_and(|format| matches!(format, "npy_3d" | "npy_2d_slices" | "zfp" | "netcdf"))
        && identity
            .and_then(|value| value.get("canonical_schema"))
            .and_then(|value| value.as_str())
            == Some("nimbus.fields")
        && identity
            .and_then(|value| value.get("canonical_version"))
            .and_then(|value| value.as_u64())
            == Some(1)
        && identity
            .and_then(|value| value.get("cm1_alias_schema"))
            .and_then(|value| value.as_str())
            == Some("cm1.history")
        && identity
            .and_then(|value| value.get("cm1_alias_version"))
            .and_then(|value| value.as_u64())
            == Some(1)
}

fn manifest_declares_field(catalog: &[serde_json::Value], visible_stem: &str) -> bool {
    catalog
        .iter()
        .any(|entry| manifest_entry_matches(entry, visible_stem))
}

fn manifest_entry_matches(entry: &serde_json::Value, visible_stem: &str) -> bool {
    if entry.get("field_id").and_then(|v| v.as_str()) == Some(visible_stem) {
        return true;
    }
    let Some(suffix) = entry.get("file_suffix").and_then(|v| v.as_str()) else {
        return false;
    };
    let Some(rest) = visible_stem.strip_prefix("th") else {
        return false;
    };
    let Some((theta, field_suffix)) = rest.split_once('_') else {
        return false;
    };
    !theta.is_empty() && theta.bytes().all(|byte| byte.is_ascii_digit()) && field_suffix == suffix
}

fn canonical_field_metadata_is_complete(catalog: &[serde_json::Value], visible_stem: &str) -> bool {
    catalog
        .iter()
        .find(|entry| manifest_entry_matches(entry, visible_stem))
        .is_some_and(|entry| {
            entry
                .get("grid_location")
                .and_then(|value| value.as_str())
                .is_some_and(|location| !location.is_empty())
                && entry
                    .get("state_semantic")
                    .and_then(|value| value.as_str())
                    .is_some_and(|semantic| {
                        matches!(
                            semantic,
                            "total_state" | "perturbation" | "diagnostic" | "tendency"
                        )
                    })
                && entry
                    .get("cm1_aliases")
                    .and_then(|value| value.as_array())
                    .is_some_and(|aliases| aliases.iter().all(|alias| alias.as_str().is_some()))
        })
}

fn grid_dimension(grid: &serde_json::Value, axis: &str, legacy_key: &str) -> Option<u64> {
    grid.get("dimensions")
        .and_then(|dimensions| dimensions.get(axis))
        .and_then(|value| value.as_u64())
        .or_else(|| grid.get(legacy_key).and_then(|value| value.as_u64()))
}

fn grid_spacing(grid: &serde_json::Value, axis: &str, legacy_key: &str) -> Option<f64> {
    grid.get("spacing_m")
        .and_then(|spacing| spacing.get(axis))
        .and_then(|value| value.as_f64())
        .or_else(|| grid.get(legacy_key).and_then(|value| value.as_f64()))
}

fn grid_is_complete_for_version(grid: &serde_json::Value, manifest_version: u64) -> bool {
    let canonical = manifest_version >= PREVIOUS_OUTPUT_MANIFEST_VERSION;
    if canonical {
        if !grid
            .get("dimensions")
            .is_some_and(serde_json::Value::is_object)
            || !grid
                .get("spacing_m")
                .is_some_and(serde_json::Value::is_object)
        {
            return false;
        }
        let axis_order = grid.get("axis_order").and_then(|value| value.as_array());
        if !axis_order.is_some_and(|axes| {
            axes.len() == 3
                && axes[0].as_str() == Some("x")
                && axes[1].as_str() == Some("y")
                && axes[2].as_str() == Some("z")
        }) || grid
            .get("publication_staggering")
            .and_then(|value| value.as_str())
            != Some("cell_centered")
        {
            return false;
        }
    }

    let dims = [
        grid_dimension(grid, "x", "nr"),
        grid_dimension(grid, "y", "nth"),
        grid_dimension(grid, "z", "nz"),
    ];
    let spacings = [
        grid_spacing(grid, "x", "dr_m"),
        grid_spacing(grid, "y", "dy_m"),
        grid_spacing(grid, "z", "dz_m"),
    ];
    if dims.iter().any(|value| {
        !value.is_some_and(|dimension| dimension > 0 && dimension <= u64::from(u32::MAX))
    }) || spacings
        .iter()
        .any(|value| !value.is_some_and(|spacing| spacing.is_finite() && spacing > 0.0))
        || grid
            .get("coordinate_system")
            .and_then(|value| value.as_str())
            != Some("cartesian")
    {
        return false;
    }

    let staggering_key = if canonical {
        "native_model_staggering"
    } else {
        "staggering"
    };
    let staggering_valid = grid
        .get(staggering_key)
        .and_then(|v| v.as_str())
        .is_some_and(|v| v == "collocated" || v == "c_grid");
    let terrain_following = manifest_version >= TERRAIN_GEOMETRY_MANIFEST_VERSION
        && grid
            .get("vertical_geometry")
            .and_then(|geometry| geometry.get("kind"))
            .and_then(|kind| kind.as_str())
            == Some("terrain_following");
    let origin = if terrain_following {
        let Some(horizontal) = parse_f64_pair(grid.get("horizontal_scalar_origin_m")) else {
            return false;
        };
        [horizontal[0], horizontal[1], f64::NAN]
    } else {
        let Some(origin) = parse_f64_triplet(grid.get("scalar_origin_m")) else {
            return false;
        };
        origin
    };
    let Some(bounds) = parse_f64_bounds(grid.get("physical_bounds_m")) else {
        return false;
    };
    let Some(valid) = parse_u32_bounds(grid.get("valid_index_range")) else {
        return false;
    };
    let Some(halo) = parse_u32_bounds(grid.get("halo_width")) else {
        return false;
    };
    let dims = dims.map(Option::unwrap);
    let spacings = spacings.map(Option::unwrap);
    let staggering = grid.get(staggering_key).and_then(|v| v.as_str());
    let v4_ownership = manifest_version >= PHYSICAL_OWNERSHIP_MANIFEST_VERSION;
    staggering_valid
        && (0..3).all(|axis| {
            let expected_halo = if axis < 2 || staggering == Some("collocated") {
                if dims[axis] >= 3 {
                    1_u32
                } else {
                    0_u32
                }
            } else {
                0
            };
            let expected_lower = if v4_ownership {
                f64::from(expected_halo) * spacings[axis]
            } else {
                0.0
            };
            let expected_upper = if v4_ownership {
                (dims[axis] - u64::from(expected_halo)) as f64 * spacings[axis]
            } else {
                dims[axis] as f64 * spacings[axis]
            };
            let coordinates_valid = if terrain_following && axis == 2 {
                bounds.lower[axis].is_finite()
                    && bounds.upper[axis].is_finite()
                    && bounds.upper[axis] > bounds.lower[axis]
            } else {
                approximately(bounds.lower[axis], expected_lower)
                    && approximately(origin[axis], 0.5 * spacings[axis])
                    && approximately(bounds.upper[axis], expected_upper)
            };
            coordinates_valid
                && valid.lower[axis] <= valid.upper[axis]
                && u64::from(valid.upper[axis]) < dims[axis]
                && u64::from(halo.lower[axis]) + u64::from(halo.upper[axis]) < dims[axis]
                && valid.lower[axis] == halo.lower[axis]
                && u64::from(valid.upper[axis]) + u64::from(halo.upper[axis]) + 1 == dims[axis]
                && (!v4_ownership
                    || (halo.lower[axis] == expected_halo && halo.upper[axis] == expected_halo))
        })
}

fn parse_f64_pair(value: Option<&serde_json::Value>) -> Option<[f64; 2]> {
    let values = value?.as_array()?;
    if values.len() != 2 {
        return None;
    }
    let parsed = [values[0].as_f64()?, values[1].as_f64()?];
    parsed.iter().all(|v| v.is_finite()).then_some(parsed)
}

fn approximately(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() <= 1.0e-9 * expected.abs().max(1.0)
}

fn parse_f64_triplet(value: Option<&serde_json::Value>) -> Option<[f64; 3]> {
    let values = value?.as_array()?;
    if values.len() != 3 {
        return None;
    }
    let parsed = [
        values[0].as_f64()?,
        values[1].as_f64()?,
        values[2].as_f64()?,
    ];
    parsed.iter().all(|v| v.is_finite()).then_some(parsed)
}

fn parse_u32_triplet(value: Option<&serde_json::Value>) -> Option<[u32; 3]> {
    let values = value?.as_array()?;
    if values.len() != 3 {
        return None;
    }
    let raw = [
        values[0].as_u64()?,
        values[1].as_u64()?,
        values[2].as_u64()?,
    ];
    if raw.iter().any(|v| *v > u64::from(u32::MAX)) {
        return None;
    }
    Some([raw[0] as u32, raw[1] as u32, raw[2] as u32])
}

fn parse_f64_bounds(value: Option<&serde_json::Value>) -> Option<AxisBoundsF64> {
    let value = value?;
    Some(AxisBoundsF64 {
        lower: parse_f64_triplet(value.get("lower"))?,
        upper: parse_f64_triplet(value.get("upper"))?,
    })
}

fn parse_u32_bounds(value: Option<&serde_json::Value>) -> Option<AxisBoundsU32> {
    let value = value?;
    Some(AxisBoundsU32 {
        lower: parse_u32_triplet(value.get("lower"))?,
        upper: parse_u32_triplet(value.get("upper"))?,
    })
}

/// Resolve `<run_dir>/step_NNNNNN/<field_name>.{npy,zfp3d}`, preferring an
/// uncompressed `.npy` when both exist, refusing any field name that could
/// escape the step directory, and confirming the file exists. Shared by every
/// per-field command so the path + safety contract is defined once. Callers
/// branch on the extension (`is_zfp3d`) to decode ZFP vs read raw.
fn field_path(run_dir: &str, step_index: u32, field_name: &str) -> AppResult<PathBuf> {
    // Defense in depth: refuse any field name that could escape the step dir.
    if field_name.is_empty()
        || field_name.contains('/')
        || field_name.contains('\\')
        || field_name.contains("..")
    {
        return Err(AppError::Invalid(format!(
            "invalid field name: {field_name}"
        )));
    }

    let step_dir = Path::new(run_dir).join(format!("step_{:06}", step_index));
    let npy = step_dir.join(format!("{field_name}.npy"));
    if npy.exists() {
        return Ok(npy);
    }
    let zfp = step_dir.join(format!("{field_name}.zfp3d"));
    if zfp.exists() {
        return Ok(zfp);
    }
    Err(AppError::NotFound(npy.display().to_string()))
}

fn netcdf_field_mapping(
    run_dir: &str,
    step_index: u32,
    field_name: &str,
) -> AppResult<Option<(PathBuf, String)>> {
    if field_name.is_empty()
        || field_name.contains('/')
        || field_name.contains('\\')
        || field_name.contains("..")
    {
        return Err(AppError::Invalid(format!(
            "invalid field name: {field_name}"
        )));
    }
    let step_dir = Path::new(run_dir).join(format!("step_{step_index:06}"));
    let manifest_path = step_dir.join("manifest.json");
    let manifest: serde_json::Value = match fs::read_to_string(&manifest_path) {
        Ok(text) => serde_json::from_str(&text).map_err(|error| {
            AppError::Invalid(format!("invalid step manifest for netCDF field: {error}"))
        })?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    if manifest_storage_format(&manifest) != Some("netcdf") {
        return Ok(None);
    }
    let mapped_fields = netcdf_manifest_fields(&manifest, &step_dir).ok_or_else(|| {
        AppError::Invalid("netCDF step manifest has an incomplete shared-file contract".into())
    })?;
    if !mapped_fields.contains(field_name) {
        return Err(AppError::Invalid(format!(
            "field '{field_name}' is not published by the netCDF step manifest"
        )));
    }
    let catalog = manifest
        .get("fields")
        .and_then(|value| value.as_array())
        .ok_or_else(|| AppError::Invalid("netCDF step manifest has no field catalog".into()))?;
    let mapping = catalog
        .iter()
        .find(|entry| entry.get("field_id").and_then(|value| value.as_str()) == Some(field_name));
    let Some(mapping) = mapping else {
        return Err(AppError::Invalid(format!(
            "netCDF step manifest does not map field '{field_name}'"
        )));
    };
    if mapping.get("file_pattern").and_then(|value| value.as_str()) != Some("history.nc")
        || mapping
            .get("storage_layout")
            .and_then(|value| value.as_str())
            != Some("shared_file")
    {
        return Err(AppError::Invalid(format!(
            "netCDF step manifest has an invalid shared-file mapping for '{field_name}'"
        )));
    }
    let variable = mapping
        .get("variable_name")
        .and_then(|value| value.as_str())
        .filter(|value| *value == field_name)
        .ok_or_else(|| {
            AppError::Invalid(format!(
                "netCDF step manifest has an invalid variable mapping for '{field_name}'"
            ))
        })?;
    let path = step_dir.join("history.nc");
    if !path.is_file() {
        return Err(AppError::NotFound(path.display().to_string()));
    }
    Ok(Some((path, variable.to_owned())))
}

fn read_netcdf_field(path: &Path, variable: &str) -> AppResult<(Vec<usize>, Vec<f32>)> {
    crate::commands::netcdf_dynamic::read_field(&path.to_string_lossy(), variable, 0)
}

/// Whether a resolved field path is a ZFP-compressed volume (needs decoding)
/// rather than a raw `.npy`.
fn is_zfp3d(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()) == Some("zfp3d")
}

fn ensure_expected_publication_generation(
    run_dir: &Path,
    step_index: u32,
    expected_generation: Option<&str>,
) -> AppResult<()> {
    match read_run_output_contract(run_dir) {
        RunOutputContract::Legacy if expected_generation.is_none() => Ok(()),
        RunOutputContract::Current {
            generation,
            ready: true,
            manifest_version,
            ..
        } if expected_generation == Some(generation.as_str()) => {
            let manifest_path = run_dir
                .join(format!("step_{step_index:06}"))
                .join("manifest.json");
            let manifest: serde_json::Value =
                serde_json::from_str(&fs::read_to_string(&manifest_path)?).map_err(|error| {
                    AppError::Invalid(format!(
                        "invalid step manifest for generation check: {error}"
                    ))
                })?;
            let matches = manifest
                .get("publication_state")
                .and_then(|value| value.as_str())
                == Some("complete")
                && manifest
                    .get("manifest_version")
                    .and_then(|value| value.as_u64())
                    == Some(manifest_version)
                && manifest.get("step_index").and_then(|value| value.as_u64())
                    == Some(u64::from(step_index))
                && manifest
                    .get("publication_generation")
                    .and_then(|value| value.as_str())
                    == Some(generation.as_str());
            if matches {
                Ok(())
            } else {
                Err(AppError::Invalid(
                    "step publication generation no longer matches the requested frame".to_string(),
                ))
            }
        }
        _ => Err(AppError::Invalid(
            "run publication generation no longer matches the requested frame".to_string(),
        )),
    }
}

/// Return the exact current-manifest grid shape after validating publication
/// identity. Legacy runs deliberately return no shape expectation because
/// their metadata was optional and cannot support the strong agreement check.
fn expected_manifest_grid_shape(
    run_dir: &Path,
    step_index: u32,
    field_name: &str,
    expected_generation: Option<&str>,
) -> AppResult<Option<[usize; 3]>> {
    ensure_expected_publication_generation(run_dir, step_index, expected_generation)?;
    if expected_generation.is_none() {
        return Ok(None);
    }
    let manifest_path = run_dir
        .join(format!("step_{step_index:06}"))
        .join("manifest.json");
    let manifest: serde_json::Value = serde_json::from_str(&fs::read_to_string(manifest_path)?)
        .map_err(|error| {
            AppError::Invalid(format!("invalid step manifest for shape check: {error}"))
        })?;
    let published = manifest
        .get("published_fields")
        .and_then(|value| value.as_array())
        .is_some_and(|fields| {
            fields
                .iter()
                .any(|field| field.as_str() == Some(field_name))
        });
    let declared = manifest
        .get("fields")
        .and_then(|value| value.as_array())
        .is_some_and(|catalog| manifest_declares_field(catalog, field_name));
    if !published || !declared {
        return Err(AppError::Invalid(format!(
            "field '{field_name}' is not published by the selected step manifest"
        )));
    }
    let manifest_version = manifest
        .get("manifest_version")
        .and_then(|value| value.as_u64())
        .ok_or_else(|| AppError::Invalid("step manifest has no version".into()))?;
    let grid = manifest
        .get("grid")
        .filter(|grid| grid_is_complete_for_version(grid, manifest_version))
        .ok_or_else(|| AppError::Invalid("step manifest has no valid grid descriptor".into()))?;
    let read_dim = |axis: &str, legacy_key: &str| -> AppResult<usize> {
        let raw = grid_dimension(grid, axis, legacy_key)
            .ok_or_else(|| AppError::Invalid(format!("step manifest grid missing {axis}")))?;
        usize::try_from(raw)
            .map_err(|_| AppError::Invalid(format!("step manifest {axis} does not fit usize")))
    };
    let shape = [
        read_dim("x", "nr")?,
        read_dim("y", "nth")?,
        read_dim("z", "nz")?,
    ];
    crate::npy::validate_logical_field_shape(&shape)?;
    Ok(Some(shape))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TerrainGeometryStorage {
    NpySidecar,
    NetcdfVariable,
}

#[derive(Debug, Clone)]
struct TerrainWindowDescriptor {
    source_shape: [usize; 3],
    horizontal_origin_m: [f64; 2],
    horizontal_spacing_m: [f64; 2],
    storage: TerrainGeometryStorage,
    geometry_directory: Option<std::path::PathBuf>,
}

/// Resolve the complete physical geometry needed by a terrain window.  A
/// terrain declaration that is not the exact manifest-v5 contract is an error,
/// never a downgrade to uniform weighting.
fn terrain_window_descriptor(
    run_dir: &Path,
    step_index: u32,
    expected_generation: Option<&str>,
) -> AppResult<Option<TerrainWindowDescriptor>> {
    ensure_expected_publication_generation(run_dir, step_index, expected_generation)?;
    let manifest_path = run_dir
        .join(format!("step_{step_index:06}"))
        .join("manifest.json");
    if !manifest_path.is_file() {
        return Ok(None);
    }
    let manifest: serde_json::Value = serde_json::from_str(&fs::read_to_string(&manifest_path)?)
        .map_err(|error| {
            AppError::Invalid(format!("invalid step manifest for terrain window: {error}"))
        })?;
    let Some(vertical) = manifest
        .get("grid")
        .and_then(|grid| grid.get("vertical_geometry"))
    else {
        return Ok(None);
    };
    if vertical.get("kind").and_then(|value| value.as_str()) != Some("terrain_following") {
        return Ok(None);
    }
    let version = manifest
        .get("manifest_version")
        .and_then(|value| value.as_u64())
        .ok_or_else(|| AppError::Invalid("terrain step manifest has no version".into()))?;
    let grid = manifest
        .get("grid")
        .filter(|grid| grid_is_complete_for_version(grid, version))
        .ok_or_else(|| AppError::Invalid("terrain step manifest has no valid grid".into()))?;
    if version < TERRAIN_GEOMETRY_MANIFEST_VERSION
        || vertical
            .get("center_coordinate")
            .and_then(|value| value.as_str())
            != Some("physical_z_center_m")
        || vertical
            .get("interface_coordinate")
            .and_then(|value| value.as_str())
            != Some("physical_z_interface_m")
        || vertical.get("dtype").and_then(|value| value.as_str()) != Some("float64")
        || vertical.get("layout").and_then(|value| value.as_str()) != Some("x_y_z")
    {
        return Err(AppError::Invalid(
            "terrain window requires the complete manifest-v5 float64 center/interface contract"
                .into(),
        ));
    }
    let dimensions = grid
        .get("dimensions")
        .ok_or_else(|| AppError::Invalid("terrain grid has no dimensions".into()))?;
    let dimension = |axis: &str| {
        dimensions
            .get(axis)
            .and_then(|value| value.as_u64())
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value > 0)
            .ok_or_else(|| AppError::Invalid(format!("terrain grid has invalid {axis} dimension")))
    };
    let source_shape = [dimension("x")?, dimension("y")?, dimension("z")?];
    let spacing = grid
        .get("spacing_m")
        .ok_or_else(|| AppError::Invalid("terrain grid has no spacing".into()))?;
    let horizontal_spacing = |axis: &str| {
        spacing
            .get(axis)
            .and_then(|value| value.as_f64())
            .filter(|value| value.is_finite() && *value > 0.0)
            .ok_or_else(|| AppError::Invalid(format!("terrain grid has invalid {axis} spacing")))
    };
    let horizontal_spacing_m = [horizontal_spacing("x")?, horizontal_spacing("y")?];
    let horizontal_origin_m = parse_f64_pair(grid.get("horizontal_scalar_origin_m"))
        .ok_or_else(|| AppError::Invalid("terrain grid has no horizontal scalar origin".into()))?;
    let (storage, geometry_directory) = match vertical
        .get("storage")
        .and_then(|value| value.as_str())
    {
        Some("npy_sidecar") if version <= FRAME_POLICY_MANIFEST_VERSION => {
            (TerrainGeometryStorage::NpySidecar, None)
        }
        Some("versioned_npy_geometry") if version >= OUTPUT_MANIFEST_VERSION => {
            let step_dir = run_dir.join(format!("step_{step_index:06}"));
            let directory = versioned_terrain_geometry_directory(
                &manifest, &step_dir, source_shape,
            ).ok_or_else(|| AppError::Invalid(
                "terrain geometry version reference is incomplete, corrupt, or from another generation".into(),
            ))?;
            (TerrainGeometryStorage::NpySidecar, Some(directory))
        }
        Some("netcdf_variable") => (TerrainGeometryStorage::NetcdfVariable, None),
        _ => {
            return Err(AppError::Invalid(
                "terrain window has unsupported geometry storage".into(),
            ));
        }
    };
    Ok(Some(TerrainWindowDescriptor {
        source_shape,
        horizontal_origin_m,
        horizontal_spacing_m,
        storage,
        geometry_directory,
    }))
}

// Window identity, geometry and cancellation are intentionally explicit here.
#[allow(clippy::too_many_arguments)]
fn read_terrain_geometry_window(
    descriptor: &TerrainWindowDescriptor,
    run_dir: &Path,
    step_index: u32,
    netcdf_path: Option<&Path>,
    origin: [usize; 3],
    extent: [usize; 3],
    control: &FieldReadControl,
    request_id: &str,
) -> AppResult<super::terrain_window::TerrainGeometryWindow> {
    match descriptor.storage {
        TerrainGeometryStorage::NpySidecar => {
            let step_dir = run_dir.join(format!("step_{step_index:06}"));
            let geometry_dir = descriptor
                .geometry_directory
                .as_deref()
                .unwrap_or(&step_dir);
            let mut cancel_check = || control.check(request_id);
            super::terrain_window::read_npy_geometry_window(
                geometry_dir,
                descriptor.source_shape,
                origin,
                extent,
                &mut cancel_check,
            )
        }
        TerrainGeometryStorage::NetcdfVariable => {
            let path = netcdf_path.ok_or_else(|| {
                AppError::Invalid(
                    "terrain manifest stores geometry in netCDF but the field has no netCDF mapping"
                        .into(),
                )
            })?;
            let path = path.to_string_lossy();
            control.check(request_id)?;
            let (center_shape, centers_m, center_bytes) = super::netcdf_dynamic::read_f64_window(
                &path,
                "physical_z_center_m",
                origin,
                extent,
            )?;
            control.check(request_id)?;
            let interface_extent_z = extent[2]
                .checked_add(1)
                .ok_or_else(|| AppError::Invalid("terrain interface window overflow".into()))?;
            let interface_extent = [extent[0], extent[1], interface_extent_z];
            let (interface_shape, interfaces_m, interface_bytes) =
                super::netcdf_dynamic::read_f64_window(
                    &path,
                    "physical_z_interface_m",
                    origin,
                    interface_extent,
                )?;
            control.check(request_id)?;
            let source_interface_nz = descriptor.source_shape[2]
                .checked_add(1)
                .ok_or_else(|| AppError::Invalid("terrain interface shape overflow".into()))?;
            if center_shape != descriptor.source_shape
                || interface_shape
                    != vec![
                        descriptor.source_shape[0],
                        descriptor.source_shape[1],
                        source_interface_nz,
                    ]
            {
                return Err(AppError::Invalid(
                    "terrain netCDF coordinate shapes disagree with the step manifest".into(),
                ));
            }
            Ok(super::terrain_window::TerrainGeometryWindow {
                centers_m,
                interfaces_m,
                source_bytes: center_bytes.checked_add(interface_bytes).ok_or_else(|| {
                    AppError::Invalid("terrain netCDF geometry byte count overflow".into())
                })?,
            })
        }
    }
}

fn encode_f64_le(values: &[f64]) -> AppResult<Vec<u8>> {
    let bytes = values
        .len()
        .checked_mul(size_of::<f64>())
        .ok_or_else(|| AppError::Invalid("terrain geometry output size overflow".into()))?;
    let mut encoded = Vec::with_capacity(bytes);
    for value in values {
        encoded.extend_from_slice(&value.to_le_bytes());
    }
    Ok(encoded)
}

fn expected_manifest_valid_index_range(
    run_dir: &Path,
    step_index: u32,
    expected_generation: Option<&str>,
) -> AppResult<Option<AxisBoundsU32>> {
    if expected_generation.is_none() {
        return Ok(None);
    }
    ensure_expected_publication_generation(run_dir, step_index, expected_generation)?;
    let manifest_path = run_dir
        .join(format!("step_{step_index:06}"))
        .join("manifest.json");
    let manifest: serde_json::Value = serde_json::from_str(&fs::read_to_string(manifest_path)?)
        .map_err(|error| {
            AppError::Invalid(format!("invalid step manifest for valid range: {error}"))
        })?;
    let manifest_version = manifest
        .get("manifest_version")
        .and_then(|value| value.as_u64())
        .ok_or_else(|| AppError::Invalid("step manifest has no version".into()))?;
    let grid = manifest
        .get("grid")
        .filter(|grid| grid_is_complete_for_version(grid, manifest_version))
        .ok_or_else(|| AppError::Invalid("step manifest has no valid grid descriptor".into()))?;
    parse_u32_bounds(grid.get("valid_index_range"))
        .map(Some)
        .ok_or_else(|| AppError::Invalid("step manifest has no valid index range".into()))
}

/// Current Nimbus field payloads are either full 3-D grid fields or 2-D
/// horizontal diagnostics. Every command checks this before returning or
/// reducing data so fields in one published frame cannot silently disagree.
fn validate_manifest_shape(shape: &[usize], grid: Option<[usize; 3]>) -> AppResult<()> {
    let Some([nr, nth, nz]) = grid else {
        return Ok(());
    };
    if shape == [nr, nth, nz] || shape == [nr, nth] {
        Ok(())
    } else {
        Err(AppError::Invalid(format!(
            "field shape {} does not match manifest grid {nr}x{nth}x{nz}",
            shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join("x")
        )))
    }
}

/// Validate the identity of a run-scoped artifact (for example
/// `trajectories.npy`). Unlike a field read there is no step manifest to
/// inspect, but the run marker still provides the replacement barrier. Current
/// artifacts are accepted only while the marker is ready and exactly matches
/// the generation selected by Studio; markerless legacy runs remain available
/// only through the explicit `None` compatibility path.
fn ensure_expected_run_generation(
    run_dir: &Path,
    expected_generation: Option<&str>,
) -> AppResult<()> {
    match read_run_output_contract(run_dir) {
        RunOutputContract::Legacy if expected_generation.is_none() => Ok(()),
        RunOutputContract::Current {
            generation,
            ready: true,
            ..
        } if expected_generation == Some(generation.as_str()) => Ok(()),
        _ => Err(AppError::Invalid(
            "run publication generation no longer matches the requested artifact".to_string(),
        )),
    }
}

fn read_run_artifact_bytes(
    run_dir: &Path,
    artifact_name: &str,
    expected_generation: Option<&str>,
) -> AppResult<Vec<u8>> {
    ensure_expected_run_generation(run_dir, expected_generation)?;
    let path = run_dir.join(artifact_name);
    if !path.is_file() {
        return Err(AppError::NotFound(path.display().to_string()));
    }
    let bytes = if path.extension().and_then(|extension| extension.to_str()) == Some("npy") {
        crate::npy::read_f32_file(&path)?
    } else {
        fs::read(&path)?
    };
    // A same-directory replacement may begin while the file is being read.
    // Reject those bytes rather than committing an old artifact under the new
    // generation selected by the UI.
    ensure_expected_run_generation(run_dir, expected_generation)?;
    Ok(bytes)
}

fn run_artifact_exists(
    run_dir: &Path,
    artifact_name: &str,
    expected_generation: Option<&str>,
) -> bool {
    if ensure_expected_run_generation(run_dir, expected_generation).is_err() {
        return false;
    }
    let exists = run_dir.join(artifact_name).is_file();
    exists && ensure_expected_run_generation(run_dir, expected_generation).is_ok()
}

fn validate_request_id(request_id: &str) -> AppResult<()> {
    if request_id.is_empty()
        || request_id.len() > MAX_REQUEST_ID_BYTES
        || !request_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        return Err(AppError::Invalid(format!(
            "request id must be 1..={MAX_REQUEST_ID_BYTES} ASCII letters, digits, '-', '_', '.', or ':'"
        )));
    }
    Ok(())
}

/// Registry slot released on drop. The command future can be dropped mid-await
/// (webview navigation, window close, IPC teardown); releasing in `Drop` rather
/// than after the await is what stops those from leaking slots until the
/// MAX_ACTIVE_FIELD_READS cap wedges every subsequent read.
struct FieldReadGuard {
    request_id: String,
    control: Arc<FieldReadControl>,
}

impl FieldReadGuard {
    fn control(&self) -> Arc<FieldReadControl> {
        Arc::clone(&self.control)
    }
}

impl Drop for FieldReadGuard {
    fn drop(&mut self) {
        self.control.cancelled.store(true, Ordering::Release);
        super::io_budget::wake_waiters();
        unregister_field_read(&self.request_id);
    }
}

fn register_field_read(request: &FieldWindowRequest) -> AppResult<FieldReadGuard> {
    validate_request_id(&request.request_id)?;
    super::io_budget::reservation_bytes(
        request
            .max_source_bytes
            .unwrap_or(DEFAULT_FIELD_WINDOW_SOURCE_BYTES),
        request
            .max_output_bytes
            .unwrap_or(DEFAULT_FIELD_WINDOW_BYTES),
    )?;
    let mut active = ACTIVE_FIELD_READS
        .lock()
        .map_err(|_| AppError::Task("field-read registry lock poisoned".into()))?;
    if active.contains_key(&request.request_id) {
        return Err(AppError::Invalid(format!(
            "field read request '{}' is already active",
            request.request_id
        )));
    }
    if active.len() >= MAX_ACTIVE_FIELD_READS {
        return Err(AppError::Task(format!(
            "active field-read limit reached ({MAX_ACTIVE_FIELD_READS})"
        )));
    }
    let control = Arc::new(FieldReadControl::new());
    active.insert(request.request_id.clone(), Arc::clone(&control));
    Ok(FieldReadGuard {
        request_id: request.request_id.clone(),
        control,
    })
}

fn unregister_field_read(request_id: &str) {
    if let Ok(mut active) = ACTIVE_FIELD_READS.lock() {
        active.remove(request_id);
    }
}

/// Request cancellation of a native window read. The receipt deliberately
/// says `cancellationRequested`, not `cancelled`: cooperative loops confirm
/// cancellation by rejecting their read. During a ZFP decode the flag is
/// observed at the next decoder boundary because libzfp exposes no abort hook.
#[tauri::command]
pub fn cancel_field_read(request_id: String) -> AppResult<CancelFieldReadResult> {
    validate_request_id(&request_id)?;
    let active = ACTIVE_FIELD_READS
        .lock()
        .map_err(|_| AppError::Task("field-read registry lock poisoned".into()))?;
    let Some(control) = active.get(&request_id) else {
        return Ok(CancelFieldReadResult {
            request_id,
            found: false,
            cancellation_requested: false,
            phase: None,
            interruptibility: None,
        });
    };
    let phase = control.phase.load(Ordering::Acquire);
    control.cancelled.store(true, Ordering::Release);
    super::io_budget::wake_waiters();
    Ok(CancelFieldReadResult {
        request_id,
        found: true,
        cancellation_requested: true,
        phase: Some(phase_name(phase)),
        interruptibility: Some(phase_interruptibility(phase)),
    })
}

/// Machine-readable truth table for callers deciding whether a source can be
/// genuinely interrupted. NPY stops between bounded source/geometry rows;
/// runtime-libnetcdf bounds a hyperslab but its C call has no abort hook.
#[tauri::command]
pub fn native_field_read_capabilities() -> NativeFieldReadCapabilities {
    NativeFieldReadCapabilities {
        schema: "nimbus.studio.native-field-read-capabilities",
        schema_version: 5,
        max_window_bytes: MAX_FIELD_WINDOW_BYTES,
        max_source_bytes: MAX_FIELD_WINDOW_SOURCE_BYTES,
        aggregations: [
            "cell_mean",
            "linear_reflectivity_mean",
            "block_max",
            "block_min",
            "magnitude_extreme",
            "point_sample",
        ],
        weighting: "uniform_or_physical_cell_volume",
        terrain_following_window_read: true,
        terrain_following_reductions: true,
        geometry_limitation: "terrain-following native windows use exact float64 interface-volume weights \
             and return physical-volume centroids; scalar/axis reductions stream bounded \
             interface slabs with the same weights. Plan views retain honest model levels; \
             vertical, volume, vector, Python, comparison, and radar-floor consumers use the \
             returned geometry for conservative physical-coordinate remapping or draping",
        max_active_requests: MAX_ACTIVE_FIELD_READS,
        npy: NativeFormatCapability {
            window_read: true,
            cancellation: "cooperative",
            limitation: None,
        },
        zfp3d: NativeFormatCapability {
            window_read: true,
            cancellation: "boundary_only",
            limitation: Some(
                "v3 point samples and native-resolution windows stream selected 64x64x32 bricks and check cancellation between decodes; each libzfp call is non-interruptible. strided aggregates reduce one reconstructed window after bounded brick replay; legacy v1/v2 artifacts retain full-volume reconstruction",
            ),
        },
        netcdf_classic: NativeFormatCapability {
            window_read: true,
            cancellation: "cooperative",
            limitation: Some(
                "bounded classic-netCDF source rows stop cooperatively between row reads",
            ),
        },
        netcdf4: NativeFormatCapability {
            window_read: true,
            cancellation: "boundary_only",
            limitation: Some(
                "nc_get_vars float/double calls read bounded hyperslabs without an abort hook; cancellation is checked at call boundaries and a cancelled result is discarded",
            ),
        },
    }
}

fn parse_npy_shape_from_header(header: &[u8]) -> AppResult<Vec<usize>> {
    let header =
        std::str::from_utf8(header).map_err(|_| AppError::Invalid("non-UTF8 NPY header".into()))?;
    let compact: String = header
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    if compact.matches("'descr':").count() != 1 || !compact.contains("'descr':'<f4'") {
        return Err(AppError::Invalid(
            "unsupported NPY dtype (only <f4 supported)".into(),
        ));
    }
    if compact.matches("'fortran_order':").count() != 1
        || !compact.contains("'fortran_order':False")
    {
        return Err(AppError::Invalid(
            "unsupported NPY order (only C order supported)".into(),
        ));
    }
    if compact.matches("'shape':").count() != 1 {
        return Err(AppError::Invalid(
            "NPY header must contain exactly one shape".into(),
        ));
    }
    let key = "'shape':(";
    let start = compact
        .find(key)
        .ok_or_else(|| AppError::Invalid("NPY header missing shape".into()))?
        + key.len();
    let end = compact[start..]
        .find(')')
        .ok_or_else(|| AppError::Invalid("NPY header malformed shape".into()))?
        + start;
    let raw = &compact[start..end];
    let mut shape = Vec::new();
    let mut parts = raw.split(',').peekable();
    while let Some(part) = parts.next() {
        if part.is_empty() {
            if !shape.is_empty() && parts.peek().is_none() && raw.ends_with(',') {
                continue;
            }
            return Err(AppError::Invalid("empty NPY shape dimension".into()));
        }
        if !part.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(AppError::Invalid(format!(
                "bad NPY shape dimension '{part}'"
            )));
        }
        shape.push(
            part.parse::<usize>()
                .map_err(|_| AppError::Invalid(format!("bad NPY shape dimension '{part}'")))?,
        );
    }
    crate::npy::validate_logical_field_shape(&shape)?;
    Ok(shape)
}

fn read_npy_header(file: &mut fs::File) -> AppResult<(Vec<usize>, usize)> {
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix)?;
    if &prefix[..6] != b"\x93NUMPY" || prefix[7] != 0 {
        return Err(AppError::Invalid("unsupported NPY magic or version".into()));
    }
    let (header_len, prefix_len) = match prefix[6] {
        1 => {
            let mut length = [0_u8; 2];
            file.read_exact(&mut length)?;
            (u16::from_le_bytes(length) as usize, 10_usize)
        }
        2 => {
            let mut length = [0_u8; 4];
            file.read_exact(&mut length)?;
            (
                usize::try_from(u32::from_le_bytes(length)).map_err(|_| {
                    AppError::Invalid("NPY header length does not fit usize".into())
                })?,
                12_usize,
            )
        }
        major => {
            return Err(AppError::Invalid(format!(
                "unsupported NPY version {major}.0"
            )));
        }
    };
    if header_len == 0 || header_len > crate::npy::MAX_NPY_HEADER_BYTES {
        return Err(AppError::Invalid(format!(
            "NPY header is {header_len} bytes; limit is {}",
            crate::npy::MAX_NPY_HEADER_BYTES
        )));
    }
    let mut header = vec![0_u8; header_len];
    file.read_exact(&mut header)?;
    let data_offset = prefix_len
        .checked_add(header_len)
        .ok_or_else(|| AppError::Invalid("NPY header offset overflow".into()))?;
    Ok((parse_npy_shape_from_header(&header)?, data_offset))
}

fn validate_window(
    request: &FieldWindowRequest,
    source_shape: &[usize],
) -> AppResult<(Vec<usize>, usize, usize, usize)> {
    validate_window_source_policy(request, source_shape, true)
}

fn validate_window_source_policy(
    request: &FieldWindowRequest,
    source_shape: &[usize],
    enforce_dense_source: bool,
) -> AppResult<(Vec<usize>, usize, usize, usize)> {
    if request.schema_version != FIELD_WINDOW_SCHEMA_VERSION {
        return Err(AppError::Invalid(format!(
            "unsupported field-window schema version {} (expected {FIELD_WINDOW_SCHEMA_VERSION})",
            request.schema_version
        )));
    }
    if !matches!(
        request.aggregation.as_str(),
        "cell_mean"
            | "linear_reflectivity_mean"
            | "block_max"
            | "block_min"
            | "magnitude_extreme"
            | "point_sample"
    ) {
        return Err(AppError::Invalid(
            "unsupported field-window aggregation".into(),
        ));
    }
    let rank = source_shape.len();
    if rank == 0 {
        return Err(AppError::Invalid(
            "field windows require at least one axis".into(),
        ));
    }
    if request.origin.len() != rank || request.extent.len() != rank || request.stride.len() != rank
    {
        return Err(AppError::Invalid(format!(
            "origin, extent, and stride must each contain exactly {rank} axes"
        )));
    }
    let mut output_shape = Vec::with_capacity(rank);
    for (axis, &source_len) in source_shape.iter().enumerate() {
        let origin = request.origin[axis];
        let extent = request.extent[axis];
        let stride = request.stride[axis];
        if extent == 0 || stride == 0 || origin >= source_len {
            return Err(AppError::Invalid(format!(
                "invalid field window on axis {axis}: origin={origin}, extent={extent}, stride={stride}, source={}",
                source_len
            )));
        }
        let end = origin
            .checked_add(extent)
            .ok_or_else(|| AppError::Invalid("field-window bound overflow".into()))?;
        if end > source_len {
            return Err(AppError::Invalid(format!(
                "field window exceeds axis {axis}: end={end}, source={}",
                source_len
            )));
        }
        output_shape.push((extent - 1) / stride + 1);
    }
    let elements = output_shape
        .iter()
        .try_fold(1_usize, |product, dimension| {
            product
                .checked_mul(*dimension)
                .ok_or_else(|| AppError::Invalid("field-window shape overflow".into()))
        })?;
    let decoded_bytes = elements
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| AppError::Invalid("field-window byte size overflow".into()))?;
    let budget = request
        .max_output_bytes
        .unwrap_or(DEFAULT_FIELD_WINDOW_BYTES);
    if budget == 0 || budget > MAX_FIELD_WINDOW_BYTES {
        return Err(AppError::Invalid(format!(
            "maxOutputBytes must be within 1..={MAX_FIELD_WINDOW_BYTES}"
        )));
    }
    if decoded_bytes > budget {
        return Err(AppError::Invalid(format!(
            "field window needs {decoded_bytes} bytes; request budget is {budget}"
        )));
    }
    let source_bytes = if !enforce_dense_source {
        0
    } else if request.aggregation == "point_sample" {
        let last_axis = rank - 1;
        let prefix_count =
            output_shape[..last_axis]
                .iter()
                .try_fold(1_usize, |product, dimension| {
                    product.checked_mul(*dimension).ok_or_else(|| {
                        AppError::Invalid("point-sample prefix count overflow".into())
                    })
                })?;
        prefix_count
            .checked_mul(request.extent[last_axis])
            .and_then(|count| count.checked_mul(size_of::<f32>()))
            .ok_or_else(|| AppError::Invalid("point-sample row byte size overflow".into()))?
    } else {
        request
            .extent
            .iter()
            .try_fold(1_usize, |product, dimension| {
                product
                    .checked_mul(*dimension)
                    .ok_or_else(|| AppError::Invalid("field-window source shape overflow".into()))
            })?
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| AppError::Invalid("field-window source byte size overflow".into()))?
    };
    let source_budget = request
        .max_source_bytes
        .unwrap_or(DEFAULT_FIELD_WINDOW_SOURCE_BYTES);
    if source_budget == 0 || source_budget > MAX_FIELD_WINDOW_SOURCE_BYTES {
        return Err(AppError::Invalid(format!(
            "maxSourceBytes must be within 1..={MAX_FIELD_WINDOW_SOURCE_BYTES}"
        )));
    }
    if source_bytes > source_budget {
        return Err(AppError::Invalid(format!(
            "field window must inspect {source_bytes} source bytes; request budget is {source_budget}"
        )));
    }
    Ok((output_shape, elements, decoded_bytes, source_bytes))
}

#[derive(Clone, Copy)]
enum FieldWindowAggregationKind {
    CellMean,
    PointSample,
    LinearReflectivityMean,
    BlockMax,
    BlockMin,
    MagnitudeExtreme,
}

fn field_window_aggregation_kind(aggregation: &str) -> AppResult<FieldWindowAggregationKind> {
    match aggregation {
        "cell_mean" => Ok(FieldWindowAggregationKind::CellMean),
        "point_sample" => Ok(FieldWindowAggregationKind::PointSample),
        "linear_reflectivity_mean" => Ok(FieldWindowAggregationKind::LinearReflectivityMean),
        "block_max" => Ok(FieldWindowAggregationKind::BlockMax),
        "block_min" => Ok(FieldWindowAggregationKind::BlockMin),
        "magnitude_extreme" => Ok(FieldWindowAggregationKind::MagnitudeExtreme),
        _ => Err(AppError::Invalid(
            "unsupported field-window aggregation".into(),
        )),
    }
}

fn aggregation_input(value: f32, aggregation: FieldWindowAggregationKind) -> f64 {
    if matches!(
        aggregation,
        FieldWindowAggregationKind::LinearReflectivityMean
    ) {
        10_f64.powf(f64::from(value) / 10.0)
    } else {
        f64::from(value)
    }
}

fn accumulate_aggregation(
    sum: &mut f64,
    count: &mut usize,
    value: f32,
    aggregation: FieldWindowAggregationKind,
) {
    let numeric = f64::from(value);
    match aggregation {
        FieldWindowAggregationKind::BlockMax if *count == 0 || numeric > *sum => *sum = numeric,
        FieldWindowAggregationKind::BlockMin if *count == 0 || numeric < *sum => *sum = numeric,
        FieldWindowAggregationKind::MagnitudeExtreme
            if *count == 0 || numeric.abs() > (*sum).abs() =>
        {
            *sum = numeric
        }
        FieldWindowAggregationKind::CellMean
        | FieldWindowAggregationKind::PointSample
        | FieldWindowAggregationKind::LinearReflectivityMean => {
            *sum += aggregation_input(value, aggregation);
        }
        _ => {}
    }
    *count += 1;
}

fn aggregation_output(sum: f64, count: usize, aggregation: FieldWindowAggregationKind) -> f32 {
    if matches!(
        aggregation,
        FieldWindowAggregationKind::BlockMax
            | FieldWindowAggregationKind::BlockMin
            | FieldWindowAggregationKind::MagnitudeExtreme
    ) {
        return sum as f32;
    }
    let mean = sum / count as f64;
    if matches!(
        aggregation,
        FieldWindowAggregationKind::LinearReflectivityMean
    ) {
        (10.0 * mean.log10()) as f32
    } else {
        mean as f32
    }
}

fn source_linear_index(indices: &[usize], shape: &[usize]) -> AppResult<usize> {
    indices
        .iter()
        .zip(shape)
        .try_fold(0_usize, |linear, (index, dimension)| {
            linear
                .checked_mul(*dimension)
                .and_then(|value| value.checked_add(*index))
                .ok_or_else(|| AppError::Invalid("field index overflow".into()))
        })
}

fn source_prefix_indices(
    prefix_linear: usize,
    source_extent: &[usize],
    origin: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let prefix_rank = source_extent.len().saturating_sub(1);
    let mut remainder = prefix_linear;
    let mut relative = vec![0_usize; source_extent.len()];
    let mut indices = origin.to_vec();
    for axis in (0..prefix_rank).rev() {
        relative[axis] = remainder % source_extent[axis];
        remainder /= source_extent[axis];
        indices[axis] += relative[axis];
    }
    (indices, relative)
}

fn read_npy_window(
    path: &Path,
    request: &FieldWindowRequest,
    control: &FieldReadControl,
) -> AppResult<(Vec<usize>, Vec<usize>, Vec<u8>)> {
    control.set_phase(ReadPhase::NpyHeader);
    control.check(&request.request_id)?;
    let mut file = fs::File::open(path)?;
    let file_len = usize::try_from(file.metadata()?.len())
        .map_err(|_| AppError::Invalid("NPY file length does not fit usize".into()))?;
    let (source_shape, data_offset) = read_npy_header(&mut file)?;
    let validated = crate::npy::validate_logical_field_shape(&source_shape)?;
    let expected_len = data_offset
        .checked_add(validated.decoded_bytes)
        .ok_or_else(|| AppError::Invalid("NPY file length overflow".into()))?;
    if file_len != expected_len {
        return Err(AppError::Invalid(format!(
            "NPY body size mismatch: need {expected_len} bytes, have {file_len}"
        )));
    }
    let (output_shape, elements, decoded_bytes, _) = validate_window(request, &source_shape)?;
    if request.aggregation == "point_sample" {
        let rank = source_shape.len();
        let last_axis = rank - 1;
        let row_samples = output_shape[last_axis];
        let prefix_count = output_shape[..last_axis]
            .iter()
            .copied()
            .product::<usize>()
            .max(1);
        let row_bytes = request.extent[last_axis]
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| AppError::Invalid("point-sample row byte size overflow".into()))?;
        // One row read per selected prefix avoids a seek/read syscall for every
        // returned cell. The planner budgets this exact inspected span.
        let mut row = vec![0_u8; row_bytes];
        let mut output = Vec::with_capacity(decoded_bytes);
        control.set_phase(ReadPhase::NpyRows);
        for prefix in 0..prefix_count {
            control.check(&request.request_id)?;
            let mut remainder = prefix;
            let mut source_indices = request.origin.clone();
            for axis in (0..last_axis).rev() {
                let output_index = remainder % output_shape[axis];
                remainder /= output_shape[axis];
                source_indices[axis] += output_index * request.stride[axis];
            }
            source_indices[last_axis] = request.origin[last_axis];
            let linear = source_linear_index(&source_indices, &source_shape)?;
            let offset = data_offset
                .checked_add(
                    linear
                        .checked_mul(size_of::<f32>())
                        .ok_or_else(|| AppError::Invalid("field byte offset overflow".into()))?,
                )
                .ok_or_else(|| AppError::Invalid("field byte offset overflow".into()))?;
            file.seek(SeekFrom::Start(offset as u64))?;
            file.read_exact(&mut row)?;
            for last in 0..row_samples {
                let sample_start = last
                    .checked_mul(request.stride[last_axis])
                    .and_then(|index| index.checked_mul(size_of::<f32>()))
                    .ok_or_else(|| AppError::Invalid("point-sample row offset overflow".into()))?;
                let sample_end = sample_start + size_of::<f32>();
                output.extend_from_slice(&row[sample_start..sample_end]);
            }
        }
        debug_assert_eq!(output.len(), decoded_bytes);
        debug_assert_eq!(output.len() / size_of::<f32>(), elements);
        return Ok((source_shape, output_shape, output));
    }
    let aggregation = field_window_aggregation_kind(&request.aggregation)?;
    let last_axis = source_shape.len() - 1;
    let source_last = request.extent[last_axis];
    let row_bytes = source_last
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| AppError::Invalid("field-window row byte size overflow".into()))?;
    let prefix_count = request.extent[..last_axis]
        .iter()
        .copied()
        .product::<usize>()
        .max(1);
    let mut scratch = vec![0_u8; row_bytes];
    let mut sums = vec![0_f64; elements];
    let mut counts = vec![0_usize; elements];
    let mut invalid = vec![false; elements];
    control.set_phase(ReadPhase::NpyRows);
    for prefix in 0..prefix_count {
        control.check(&request.request_id)?;
        let (mut source_indices, relative) =
            source_prefix_indices(prefix, &request.extent, &request.origin);
        source_indices[last_axis] = request.origin[last_axis];
        let linear = source_linear_index(&source_indices, &source_shape)?;
        let offset = data_offset
            .checked_add(
                linear
                    .checked_mul(size_of::<f32>())
                    .ok_or_else(|| AppError::Invalid("field byte offset overflow".into()))?,
            )
            .ok_or_else(|| AppError::Invalid("field byte offset overflow".into()))?;
        file.seek(SeekFrom::Start(offset as u64))?;
        file.read_exact(&mut scratch)?;
        let target_prefix = relative[..last_axis]
            .iter()
            .zip(&request.stride[..last_axis])
            .zip(&output_shape[..last_axis])
            .try_fold(0_usize, |linear, ((relative_index, stride), dimension)| {
                linear
                    .checked_mul(*dimension)
                    .and_then(|value| value.checked_add(relative_index / stride))
                    .ok_or_else(|| AppError::Invalid("field target index overflow".into()))
            })?;
        let target_row_base = target_prefix
            .checked_mul(output_shape[last_axis])
            .ok_or_else(|| AppError::Invalid("field target row overflow".into()))?;
        for last in 0..source_last {
            // The final axis is contiguous in C order. Computing the reduced
            // row base once avoids a heap allocation plus a generic rank walk
            // for every source voxel (tens of millions of allocations on a
            // large isosurface window in debug builds).
            let target = target_row_base + last / request.stride[last_axis];
            let source_byte = last * size_of::<f32>();
            let value = f32::from_le_bytes(
                scratch[source_byte..source_byte + size_of::<f32>()]
                    .try_into()
                    .map_err(|_| AppError::Invalid("truncated NPY field row".into()))?,
            );
            if value.is_finite() {
                accumulate_aggregation(&mut sums[target], &mut counts[target], value, aggregation);
            } else {
                invalid[target] = true;
            }
        }
    }
    control.check(&request.request_id)?;
    let mut output = Vec::with_capacity(decoded_bytes);
    for index in 0..elements {
        let value = if invalid[index] || counts[index] == 0 {
            f32::NAN
        } else {
            aggregation_output(sums[index], counts[index], aggregation)
        };
        output.extend_from_slice(&value.to_le_bytes());
    }
    debug_assert_eq!(output.len(), decoded_bytes);
    Ok((source_shape, output_shape, output))
}

/// Read the exact unaggregated source extent for terrain weighting.  The
/// terrain aggregator must see every source value beside its float64 cell
/// geometry; applying the uniform `cell_mean` first would be irreversible.
fn read_npy_source_window(
    path: &Path,
    request: &FieldWindowRequest,
    control: &FieldReadControl,
) -> AppResult<(Vec<usize>, Vec<f32>)> {
    control.set_phase(ReadPhase::NpyHeader);
    control.check(&request.request_id)?;
    let mut file = fs::File::open(path)?;
    let file_len = usize::try_from(file.metadata()?.len())
        .map_err(|_| AppError::Invalid("NPY file length does not fit usize".into()))?;
    let (source_shape, data_offset) = read_npy_header(&mut file)?;
    let validated = crate::npy::validate_logical_field_shape(&source_shape)?;
    let expected_len = data_offset
        .checked_add(validated.decoded_bytes)
        .ok_or_else(|| AppError::Invalid("NPY file length overflow".into()))?;
    if file_len != expected_len {
        return Err(AppError::Invalid(format!(
            "NPY body size mismatch: need {expected_len} bytes, have {file_len}"
        )));
    }
    validate_window(request, &source_shape)?;
    if source_shape.len() != 3 {
        return Err(AppError::Invalid(
            "terrain field windows require a three-dimensional source".into(),
        ));
    }
    let extent = [request.extent[0], request.extent[1], request.extent[2]];
    let elements = extent.into_iter().try_fold(1_usize, |product, value| {
        product
            .checked_mul(value)
            .ok_or_else(|| AppError::Invalid("terrain NPY source shape overflow".into()))
    })?;
    let row_bytes = request.extent[2]
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| AppError::Invalid("terrain NPY row size overflow".into()))?;
    let mut row = vec![0_u8; row_bytes];
    let mut values = Vec::with_capacity(elements);
    control.set_phase(ReadPhase::NpyRows);
    for i in 0..request.extent[0] {
        for j in 0..request.extent[1] {
            control.check(&request.request_id)?;
            let linear = (request.origin[0] + i)
                .checked_mul(source_shape[1])
                .and_then(|value| value.checked_add(request.origin[1] + j))
                .and_then(|value| value.checked_mul(source_shape[2]))
                .and_then(|value| value.checked_add(request.origin[2]))
                .ok_or_else(|| AppError::Invalid("terrain NPY source index overflow".into()))?;
            let offset = data_offset
                .checked_add(linear.checked_mul(size_of::<f32>()).ok_or_else(|| {
                    AppError::Invalid("terrain NPY source offset overflow".into())
                })?)
                .ok_or_else(|| AppError::Invalid("terrain NPY source offset overflow".into()))?;
            file.seek(SeekFrom::Start(u64::try_from(offset).map_err(|_| {
                AppError::Invalid("terrain NPY source offset does not fit u64".into())
            })?))?;
            file.read_exact(&mut row)?;
            for bytes in row.chunks_exact(size_of::<f32>()) {
                values.push(f32::from_le_bytes(bytes.try_into().map_err(|_| {
                    AppError::Invalid("truncated terrain NPY source value".into())
                })?));
            }
        }
    }
    debug_assert_eq!(values.len(), elements);
    Ok((source_shape, values))
}

fn sample_decoded_window(
    source_shape: &[usize],
    source: &[f32],
    request: &FieldWindowRequest,
    control: &FieldReadControl,
) -> AppResult<(Vec<usize>, Vec<u8>)> {
    sample_decoded_values(
        source_shape,
        |index| source.get(index).copied(),
        request,
        control,
    )
}

fn sample_decoded_values<F>(
    source_shape: &[usize],
    mut source: F,
    request: &FieldWindowRequest,
    control: &FieldReadControl,
) -> AppResult<(Vec<usize>, Vec<u8>)>
where
    F: FnMut(usize) -> Option<f32>,
{
    let (output_shape, elements, decoded_bytes, _) = validate_window(request, source_shape)?;
    let aggregation = field_window_aggregation_kind(&request.aggregation)?;
    if request.aggregation == "point_sample" || request.stride.iter().all(|stride| *stride == 1) {
        let mut output = Vec::with_capacity(decoded_bytes);
        control.set_phase(ReadPhase::Sampling);
        for linear in 0..elements {
            if linear % 16_384 == 0 {
                control.check(&request.request_id)?;
            }
            let mut remainder = linear;
            let mut index = 0usize;
            let mut multiplier = 1usize;
            for axis in (0..source_shape.len()).rev() {
                let relative = remainder % output_shape[axis] * request.stride[axis];
                remainder /= output_shape[axis];
                index = index
                    .checked_add(
                        (request.origin[axis] + relative)
                            .checked_mul(multiplier)
                            .ok_or_else(|| {
                                AppError::Invalid("decoded source index overflow".into())
                            })?,
                    )
                    .ok_or_else(|| AppError::Invalid("decoded source index overflow".into()))?;
                multiplier = multiplier
                    .checked_mul(source_shape[axis])
                    .ok_or_else(|| AppError::Invalid("decoded source stride overflow".into()))?;
            }
            let value = source(index).ok_or_else(|| {
                AppError::Invalid("decoded field is shorter than its shape".into())
            })?;
            let value = if value.is_finite() {
                let mut sum = 0.0;
                let mut count = 0;
                accumulate_aggregation(&mut sum, &mut count, value, aggregation);
                aggregation_output(sum, count, aggregation)
            } else {
                f32::NAN
            };
            output.extend_from_slice(&value.to_le_bytes());
        }
        return Ok((output_shape, output));
    }

    let mut sums = vec![0_f64; elements];
    let mut counts = vec![0_usize; elements];
    let mut invalid = vec![false; elements];
    control.set_phase(ReadPhase::Sampling);
    let source_window_elements = request.extent.iter().product::<usize>();
    for window_linear in 0..source_window_elements {
        if window_linear % 16_384 == 0 {
            control.check(&request.request_id)?;
        }
        let mut remainder = window_linear;
        let mut source_index = 0_usize;
        let mut source_multiplier = 1_usize;
        let mut target_index = 0_usize;
        let mut target_multiplier = 1_usize;
        let mut point_selected = true;
        for axis in (0..source_shape.len()).rev() {
            let relative = remainder % request.extent[axis];
            remainder /= request.extent[axis];
            if matches!(aggregation, FieldWindowAggregationKind::PointSample)
                && relative % request.stride[axis] != 0
            {
                point_selected = false;
            }
            source_index = source_index
                .checked_add(
                    (request.origin[axis] + relative)
                        .checked_mul(source_multiplier)
                        .ok_or_else(|| AppError::Invalid("decoded source index overflow".into()))?,
                )
                .ok_or_else(|| AppError::Invalid("decoded source index overflow".into()))?;
            source_multiplier = source_multiplier
                .checked_mul(source_shape[axis])
                .ok_or_else(|| AppError::Invalid("decoded source stride overflow".into()))?;
            target_index = target_index
                .checked_add(
                    (relative / request.stride[axis])
                        .checked_mul(target_multiplier)
                        .ok_or_else(|| AppError::Invalid("decoded target index overflow".into()))?,
                )
                .ok_or_else(|| AppError::Invalid("decoded target index overflow".into()))?;
            target_multiplier = target_multiplier
                .checked_mul(output_shape[axis])
                .ok_or_else(|| AppError::Invalid("decoded target stride overflow".into()))?;
        }
        if !point_selected {
            continue;
        }
        let value = source(source_index)
            .ok_or_else(|| AppError::Invalid("decoded field is shorter than its shape".into()))?;
        if value.is_finite() {
            accumulate_aggregation(
                &mut sums[target_index],
                &mut counts[target_index],
                value,
                aggregation,
            );
        } else {
            invalid[target_index] = true;
        }
    }
    control.check(&request.request_id)?;
    let mut output = Vec::with_capacity(decoded_bytes);
    for index in 0..elements {
        let value = if invalid[index] || counts[index] == 0 {
            f32::NAN
        } else {
            aggregation_output(sums[index], counts[index], aggregation)
        };
        output.extend_from_slice(&value.to_le_bytes());
    }
    Ok((output_shape, output))
}

fn encode_field_window(
    header: &FieldWindowHeader,
    body: &[u8],
    geometry_body: &[u8],
) -> AppResult<Vec<u8>> {
    let expected_geometry_padding = if geometry_body.is_empty() {
        0
    } else {
        (8 - (body.len() % 8)) % 8
    };
    if header.schema_version != FIELD_WINDOW_RESPONSE_SCHEMA_VERSION
        || header.geometry_padding_bytes != expected_geometry_padding
    {
        return Err(AppError::Task(
            "field-window geometry padding metadata is inconsistent".into(),
        ));
    }
    let mut json = serde_json::to_vec(header)
        .map_err(|error| AppError::Task(format!("field-window metadata encode failed: {error}")))?;
    let padding = (8 - ((FIELD_WINDOW_MAGIC.len() + size_of::<u32>() + json.len()) % 8)) % 8;
    json.resize(json.len() + padding, b' ');
    let json_len = u32::try_from(json.len())
        .map_err(|_| AppError::Task("field-window metadata is too large".into()))?;
    let capacity = FIELD_WINDOW_MAGIC
        .len()
        .checked_add(size_of::<u32>())
        .and_then(|value| value.checked_add(json.len()))
        .and_then(|value| value.checked_add(body.len()))
        .and_then(|value| value.checked_add(header.geometry_padding_bytes))
        .and_then(|value| value.checked_add(geometry_body.len()))
        .ok_or_else(|| AppError::Task("field-window response size overflow".into()))?;
    let mut encoded = Vec::with_capacity(capacity);
    encoded.extend_from_slice(&FIELD_WINDOW_MAGIC);
    encoded.extend_from_slice(&json_len.to_le_bytes());
    encoded.extend_from_slice(&json);
    encoded.extend_from_slice(body);
    encoded.resize(encoded.len() + header.geometry_padding_bytes, 0);
    encoded.extend_from_slice(geometry_body);
    Ok(encoded)
}

fn native_netcdf_source_format(format: &str) -> &'static str {
    match format {
        "netcdf4" => "netcdf4",
        "netcdf4_classic" => "netcdf4_classic",
        "netcdf_classic" => "netcdf_classic",
        _ => "netcdf",
    }
}

// The dispatcher keeps source variants explicit so each path can enforce its own I/O budget.
#[allow(clippy::too_many_arguments)]
fn read_uniform_window_payload(
    run_dir: &str,
    step_index: u32,
    field_name: &str,
    request: &FieldWindowRequest,
    expected_grid: Option<[usize; 3]>,
    netcdf: Option<&(PathBuf, String)>,
    path: Option<&Path>,
    control: &FieldReadControl,
) -> AppResult<FieldWindowPayload> {
    if let Some((netcdf_path, variable)) = netcdf {
        control.set_phase(ReadPhase::NetcdfNative);
        control.check(&request.request_id)?;
        let payload = {
            let native_request = crate::commands::datasets::NetcdfWindowRequest {
                schema_version: request.schema_version,
                request_id: request.request_id.clone(),
                aggregation: request.aggregation.clone(),
                origin: request.origin.clone(),
                extent: request.extent.clone(),
                stride: request.stride.clone(),
                max_output_bytes: request.max_output_bytes,
                max_output_samples: request.max_output_bytes.map(|bytes| bytes / 4),
                max_source_bytes: request.max_source_bytes.map(|bytes| bytes as u64),
            };
            let native_control = crate::commands::datasets::NetcdfReadControl::new();
            super::netcdf_dynamic::read_field_window(
                &netcdf_path.to_string_lossy(),
                variable,
                0,
                &native_request,
                &native_control,
            )?
        };
        control.check(&request.request_id)?;
        validate_manifest_shape(&payload.source_shape, expected_grid)?;
        return Ok(FieldWindowPayload {
            source_shape: payload.source_shape,
            output_shape: payload.output_shape,
            field_body: bytemuck::cast_slice(&payload.values).to_vec(),
            geometry_body: Vec::new(),
            weighting: "uniform_cell_volume",
            sample_location: if request.aggregation == "point_sample" {
                "source_cell_center"
            } else {
                "block_center"
            },
            geometry_kind: "separable_uniform",
            source_working_bytes: usize::try_from(payload.source_read_bytes)
                .map_err(|_| AppError::Invalid("netCDF source byte count exceeds usize".into()))?,
            source_format: native_netcdf_source_format(&payload.source_format),
            cancellation_mode: payload.cancellation_mode,
            non_interruptible_phase: payload.non_interruptible_phase,
        });
    }

    let path = path.ok_or_else(|| AppError::NotFound(field_name.to_owned()))?;
    if is_zfp3d(path) {
        control.set_phase(ReadPhase::ZfpDecode);
        control.check(&request.request_id)?;
        if request.origin.len() != 3 || request.extent.len() != 3 {
            return Err(AppError::Invalid(
                "ZFP field windows require a three-dimensional request".into(),
            ));
        }
        let origin = [request.origin[0], request.origin[1], request.origin[2]];
        let extent = [request.extent[0], request.extent[1], request.extent[2]];
        if request.stride.len() != 3 {
            return Err(AppError::Invalid(
                "ZFP field windows require three stride axes".into(),
            ));
        }
        let source_budget = request
            .max_source_bytes
            .unwrap_or(DEFAULT_FIELD_WINDOW_SOURCE_BYTES);
        let output_budget = request
            .max_output_bytes
            .unwrap_or(DEFAULT_FIELD_WINDOW_BYTES);
        if source_budget == 0
            || source_budget > MAX_FIELD_WINDOW_SOURCE_BYTES
            || output_budget == 0
            || output_budget > MAX_FIELD_WINDOW_BYTES
        {
            return Err(AppError::Invalid(
                "Invalid field-window source or output budget".into(),
            ));
        }
        {
            let bounding_shape = (0..3)
                .map(|axis| {
                    origin[axis]
                        .checked_add(extent[axis])
                        .ok_or_else(|| AppError::Invalid("field-window bounds overflow".into()))
                })
                .collect::<AppResult<Vec<_>>>()?;
            let (output_shape, elements, _, _) =
                validate_window_source_policy(request, &bounding_shape, false)?;
            let aggregate = request.aggregation != "point_sample"
                && request.stride.iter().any(|stride| *stride != 1);
            let source_bytes = extent
                .iter()
                .try_fold(4usize, |bytes, size| bytes.checked_mul(*size))
                .ok_or_else(|| AppError::Invalid("ZFP source-window size overflow".into()))?;
            let aggregation_working = if aggregate {
                elements
                    .checked_mul(
                        size_of::<f64>()
                            + size_of::<usize>()
                            + size_of::<bool>()
                            + size_of::<f32>(),
                    )
                    .and_then(|bytes| bytes.checked_add(source_bytes))
                    .ok_or_else(|| {
                        AppError::Invalid("ZFP aggregation working-set overflow".into())
                    })?
            } else {
                0
            };
            if aggregation_working > source_budget {
                return Err(AppError::Invalid(format!("ZFP aggregation needs {aggregation_working} working bytes; budget is {source_budget}")));
            }
            if let Some(mut window) = crate::zfp3d::read_sampled_field_window(
                run_dir,
                step_index,
                field_name,
                origin,
                extent,
                if aggregate {
                    [1, 1, 1]
                } else {
                    [request.stride[0], request.stride[1], request.stride[2]]
                },
                source_budget,
                if aggregate {
                    source_bytes
                } else {
                    output_budget
                },
                expected_grid,
                |decoding| {
                    control.set_phase(if decoding {
                        ReadPhase::ZfpDecode
                    } else {
                        ReadPhase::Sampling
                    });
                    control.check(&request.request_id)
                },
            )? {
                validate_manifest_shape(&window.source_shape, expected_grid)?;
                validate_window_source_policy(request, &window.source_shape, false)?;
                if !aggregate && window.output_shape != output_shape {
                    return Err(AppError::Invalid("Sampled ZFP output shape changed".into()));
                }
                let body = if aggregate {
                    let mut local = request.clone();
                    local.origin = vec![0, 0, 0];
                    sample_decoded_values(
                        &request.extent,
                        |index| {
                            window
                                .bytes
                                .get(index * 4..index * 4 + 4)
                                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                        },
                        &local,
                        control,
                    )?
                    .1
                } else {
                    let aggregation = field_window_aggregation_kind(&request.aggregation)?;
                    for (index, bytes) in window.bytes.chunks_exact_mut(4).enumerate() {
                        if index % 16_384 == 0 {
                            control.check(&request.request_id)?;
                        }
                        let value = f32::from_le_bytes(bytes.try_into().unwrap());
                        let value = if value.is_finite() {
                            let mut sum = 0.0;
                            let mut count = 0;
                            accumulate_aggregation(&mut sum, &mut count, value, aggregation);
                            aggregation_output(sum, count, aggregation)
                        } else {
                            f32::NAN
                        };
                        bytes.copy_from_slice(&value.to_le_bytes());
                    }
                    window.bytes
                };
                return Ok(FieldWindowPayload {
                    source_shape: window.source_shape,
                    output_shape,
                    field_body: body,
                    geometry_body: Vec::new(),
                    weighting: "uniform_cell_volume",
                    sample_location: if request.aggregation == "point_sample" {
                        "source_cell_center"
                    } else {
                        "block_center"
                    },
                    geometry_kind: "separable_uniform",
                    source_working_bytes: window.source_working_bytes.max(aggregation_working),
                    source_format: "zfp3d",
                    cancellation_mode: "cooperative",
                    non_interruptible_phase: Some("zfp_decode"),
                });
            }
        }
        let (declared_shape, reconstruction_bytes) =
            crate::zfp3d::field_shape_and_window_reconstruction_bytes(
                run_dir, step_index, field_name, origin, extent,
            )?;
        validate_manifest_shape(&declared_shape, expected_grid)?;
        let (_, output_elements, _, _) = validate_window(request, &declared_shape)?;
        let sampling_working = if request.aggregation != "point_sample"
            && request.stride.iter().any(|stride| *stride != 1)
        {
            output_elements
                .checked_mul(
                    size_of::<f64>() + size_of::<usize>() + size_of::<bool>() + size_of::<f32>(),
                )
                .and_then(|bytes| {
                    request
                        .extent
                        .iter()
                        .try_fold(4usize, |count, size| count.checked_mul(*size))
                        .and_then(|source| bytes.checked_add(source))
                })
                .ok_or_else(|| AppError::Invalid("ZFP aggregation working-set overflow".into()))?
        } else {
            0
        };
        let reconstruction_bytes = reconstruction_bytes.max(sampling_working);
        let source_budget = request
            .max_source_bytes
            .unwrap_or(DEFAULT_FIELD_WINDOW_SOURCE_BYTES);
        if reconstruction_bytes > source_budget {
            return Err(AppError::Invalid(format!(
                "ZFP field-window reconstruction needs at most {reconstruction_bytes} inspected/working bytes; request budget is {source_budget}"
            )));
        }
        let (shape, data) = crate::zfp3d::reconstruct_field_window(
            run_dir, step_index, field_name, origin, extent,
        )?;
        control.check(&request.request_id)?;
        if shape != declared_shape {
            return Err(AppError::Invalid(
                "ZFP decoded shape changed after header validation".into(),
            ));
        }
        validate_manifest_shape(&shape, expected_grid)?;
        let mut local_request = request.clone();
        local_request.origin = vec![0, 0, 0];
        let (output_shape, field_body) =
            sample_decoded_window(&request.extent, &data, &local_request, control)?;
        return Ok(FieldWindowPayload {
            source_shape: shape,
            output_shape,
            field_body,
            geometry_body: Vec::new(),
            weighting: "uniform_cell_volume",
            sample_location: if request.aggregation == "point_sample" {
                "source_cell_center"
            } else {
                "block_center"
            },
            geometry_kind: "separable_uniform",
            source_working_bytes: reconstruction_bytes,
            source_format: "zfp3d",
            cancellation_mode: "boundary_only",
            non_interruptible_phase: Some("zfp_decode"),
        });
    }

    let (source_shape, output_shape, field_body) = read_npy_window(path, request, control)?;
    validate_manifest_shape(&source_shape, expected_grid)?;
    let (_, _, _, source_working_bytes) = validate_window(request, &source_shape)?;
    Ok(FieldWindowPayload {
        source_shape,
        output_shape,
        field_body,
        geometry_body: Vec::new(),
        weighting: "uniform_cell_volume",
        sample_location: if request.aggregation == "point_sample" {
            "source_cell_center"
        } else {
            "block_center"
        },
        geometry_kind: "separable_uniform",
        source_working_bytes,
        source_format: "npy",
        cancellation_mode: "cooperative",
        non_interruptible_phase: None,
    })
}

// Terrain windows additionally carry the validated physical-geometry descriptor.
#[allow(clippy::too_many_arguments)]
fn read_terrain_point_payload(
    run_dir: &str,
    step_index: u32,
    field_name: &str,
    request: &FieldWindowRequest,
    expected_grid: Option<[usize; 3]>,
    descriptor: &TerrainWindowDescriptor,
    netcdf: Option<&(PathBuf, String)>,
    path: Option<&Path>,
    control: &FieldReadControl,
) -> AppResult<FieldWindowPayload> {
    let (output_shape, elements, field_bytes, _) =
        validate_window_source_policy(request, &descriptor.source_shape, false)?;
    validate_manifest_shape(&descriptor.source_shape, expected_grid)?;
    let origin: [usize; 3] = request
        .origin
        .clone()
        .try_into()
        .map_err(|_| AppError::Invalid("terrain point origin rank".into()))?;
    let stride: [usize; 3] = request
        .stride
        .clone()
        .try_into()
        .map_err(|_| AppError::Invalid("terrain point stride rank".into()))?;
    let count: [usize; 3] = output_shape.clone().try_into().unwrap();
    let extent: [usize; 3] = std::array::from_fn(|axis| (count[axis] - 1) * stride[axis] + 1);
    let output_bytes = elements
        .checked_mul(36)
        .and_then(|n| n.checked_add((8 - field_bytes % 8) % 8))
        .ok_or_else(|| AppError::Invalid("terrain point output overflow".into()))?;
    if output_bytes
        > request
            .max_output_bytes
            .unwrap_or(DEFAULT_FIELD_WINDOW_BYTES)
    {
        return Err(AppError::Invalid(
            "terrain point output exceeds field/geometry budget".into(),
        ));
    }
    let geometry_io = match descriptor.storage {
        TerrainGeometryStorage::NpySidecar => {
            super::terrain_window::sampled_npy_bytes(count, stride)?.checked_mul(3)
        }
        TerrainGeometryStorage::NetcdfVariable => elements.checked_mul(24),
    }
    .ok_or_else(|| AppError::Invalid("terrain point geometry budget overflow".into()))?;
    let geometry_budget = elements
        .checked_mul(96)
        .and_then(|n| n.checked_add(geometry_io))
        .ok_or_else(|| AppError::Invalid("terrain point working budget overflow".into()))?;
    let source_budget = request
        .max_source_bytes
        .unwrap_or(DEFAULT_FIELD_WINDOW_SOURCE_BYTES);
    let field_budget = source_budget
        .checked_sub(geometry_budget)
        .filter(|n| *n >= field_bytes)
        .ok_or_else(|| AppError::Invalid("terrain point geometry exceeds source budget".into()))?;
    control.check(&request.request_id)?;
    let geometry_paths = match descriptor.storage {
        TerrainGeometryStorage::NpySidecar => {
            let step_dir = Path::new(run_dir).join(format!("step_{step_index:06}"));
            let directory = descriptor
                .geometry_directory
                .as_deref()
                .unwrap_or(&step_dir);
            vec![
                directory.join("physical_z_center_m.npy"),
                directory.join("physical_z_interface_m.npy"),
            ]
        }
        TerrainGeometryStorage::NetcdfVariable => vec![netcdf
            .ok_or_else(|| AppError::Invalid("terrain netCDF mapping missing".into()))?
            .0
            .clone()],
    };
    let versions = || -> AppResult<Vec<_>> {
        geometry_paths
            .iter()
            .map(|path| {
                let metadata = fs::metadata(path)?;
                Ok((metadata.len(), metadata.modified()?))
            })
            .collect()
    };
    let original_versions = versions()?;
    let mut field_request = request.clone();
    field_request.max_source_bytes = Some(field_budget);
    field_request.max_output_bytes = Some(field_bytes);
    let mut payload = read_uniform_window_payload(
        run_dir,
        step_index,
        field_name,
        &field_request,
        expected_grid,
        netcdf,
        path,
        control,
    )?;
    if payload.output_shape != output_shape || payload.source_shape != descriptor.source_shape {
        return Err(AppError::Invalid(
            "terrain point field shape mismatch".into(),
        ));
    }
    let read = |interfaces: bool, upper: bool| -> AppResult<Vec<f64>> {
        control.check(&request.request_id)?;
        let mut shape = descriptor.source_shape;
        let mut start = origin;
        if interfaces {
            shape[2] += 1;
        }
        if upper {
            start[2] += 1;
        }
        let variable = if interfaces {
            "physical_z_interface_m"
        } else {
            "physical_z_center_m"
        };
        match descriptor.storage {
            TerrainGeometryStorage::NpySidecar => {
                control.set_phase(ReadPhase::NpyRows);
                let step_dir = Path::new(run_dir).join(format!("step_{step_index:06}"));
                let directory = descriptor
                    .geometry_directory
                    .as_deref()
                    .unwrap_or(&step_dir);
                super::terrain_window::read_npy_samples(
                    &directory.join(format!("{variable}.npy")),
                    shape,
                    start,
                    count,
                    stride,
                    &mut || control.check(&request.request_id),
                )
            }
            TerrainGeometryStorage::NetcdfVariable => {
                control.set_phase(ReadPhase::NetcdfNative);
                let path = netcdf.ok_or_else(|| {
                    AppError::Invalid("terrain points require netCDF geometry mapping".into())
                })?;
                let (actual, values, _) = super::netcdf_dynamic::read_f64_sampled_window(
                    &path.0.to_string_lossy(),
                    variable,
                    start,
                    extent,
                    stride,
                )?;
                control.check(&request.request_id)?;
                if actual != shape {
                    return Err(AppError::Invalid(
                        "terrain point coordinate shape mismatch".into(),
                    ));
                }
                Ok(values)
            }
        }
    };
    let centers = read(false, false)?;
    let lower = read(true, false)?;
    let upper = read(true, true)?;
    let mut geometry_body = Vec::with_capacity(elements * 32);
    for axis in 0..4 {
        for index in 0..elements {
            if index % 1024 == 0 {
                control.check(&request.request_id)?;
            }
            let volume = descriptor.horizontal_spacing_m[0]
                * descriptor.horizontal_spacing_m[1]
                * (upper[index] - lower[index]);
            if !centers[index].is_finite()
                || !lower[index].is_finite()
                || !upper[index].is_finite()
                || centers[index] <= lower[index]
                || centers[index] >= upper[index]
                || !volume.is_finite()
                || volume <= 0.0
            {
                return Err(AppError::Invalid(
                    "invalid sampled terrain cell geometry".into(),
                ));
            }
            let value = match axis {
                0 => {
                    descriptor.horizontal_origin_m[0]
                        + (origin[0] + index / (count[1] * count[2]) * stride[0]) as f64
                            * descriptor.horizontal_spacing_m[0]
                }
                1 => {
                    descriptor.horizontal_origin_m[1]
                        + (origin[1] + index / count[2] % count[1] * stride[1]) as f64
                            * descriptor.horizontal_spacing_m[1]
                }
                2 => centers[index],
                _ => volume,
            };
            if !value.is_finite() {
                return Err(AppError::Invalid("invalid terrain point coordinate".into()));
            }
            geometry_body.extend_from_slice(&value.to_le_bytes());
        }
    }
    control.check(&request.request_id)?;
    if versions()? != original_versions {
        return Err(AppError::Invalid(
            "terrain geometry changed during point read".into(),
        ));
    }
    payload.geometry_body = geometry_body;
    payload.weighting = "physical_cell_volume";
    payload.geometry_kind = "terrain_following";
    payload.source_working_bytes = payload
        .source_working_bytes
        .checked_add(geometry_budget)
        .filter(|n| *n <= source_budget)
        .ok_or_else(|| AppError::Invalid("terrain point source budget exceeded".into()))?;
    if matches!(descriptor.storage, TerrainGeometryStorage::NetcdfVariable) {
        payload.non_interruptible_phase = Some("nc_get_vars_float_and_double");
    }
    Ok(payload)
}

#[allow(clippy::too_many_arguments)]
fn read_terrain_window_payload(
    run_dir: &str,
    step_index: u32,
    field_name: &str,
    request: &FieldWindowRequest,
    expected_grid: Option<[usize; 3]>,
    descriptor: &TerrainWindowDescriptor,
    netcdf: Option<&(PathBuf, String)>,
    path: Option<&Path>,
    control: &FieldReadControl,
) -> AppResult<FieldWindowPayload> {
    if request.origin.len() != 3 || request.extent.len() != 3 || request.stride.len() != 3 {
        return Err(AppError::Invalid(
            "terrain field windows require a three-dimensional request".into(),
        ));
    }
    if request.aggregation == "point_sample" {
        return read_terrain_point_payload(run_dir, step_index, field_name, request, expected_grid,
            descriptor, netcdf, path, control);
    }
    let origin = [request.origin[0], request.origin[1], request.origin[2]];
    let extent = [request.extent[0], request.extent[1], request.extent[2]];
    let stride = [request.stride[0], request.stride[1], request.stride[2]];
    let source_shape = descriptor.source_shape.to_vec();
    let (expected_output_shape, output_elements, field_output_bytes, _) =
        validate_window(request, &source_shape)?;
    validate_manifest_shape(&source_shape, expected_grid)?;
    let interface_elements = extent[0]
        .checked_mul(extent[1])
        .and_then(|value| value.checked_mul(extent[2].checked_add(1)?))
        .ok_or_else(|| AppError::Invalid("terrain interface window size overflow".into()))?;
    let center_elements = extent
        .into_iter()
        .try_fold(1_usize, |product, value| product.checked_mul(value))
        .ok_or_else(|| AppError::Invalid("terrain center window size overflow".into()))?;
    let dense_field_source_bytes = center_elements
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| AppError::Invalid("terrain field source byte size overflow".into()))?;
    let geometry_source_bytes = center_elements
        .checked_add(interface_elements)
        .and_then(|value| value.checked_mul(size_of::<f64>()))
        .ok_or_else(|| AppError::Invalid("terrain geometry byte size overflow".into()))?;
    let aggregation_bytes = super::terrain_window::aggregation_working_bytes(output_elements)?;
    let geometry_output_bytes = output_elements
        .checked_mul(4 * size_of::<f64>())
        .ok_or_else(|| AppError::Invalid("terrain geometry output size overflow".into()))?;
    let geometry_padding_bytes = if geometry_output_bytes == 0 {
        0
    } else {
        (8 - (field_output_bytes % 8)) % 8
    };
    let total_output_bytes = field_output_bytes
        .checked_add(geometry_padding_bytes)
        .and_then(|value| value.checked_add(geometry_output_bytes))
        .ok_or_else(|| AppError::Invalid("terrain window output size overflow".into()))?;
    let output_budget = request
        .max_output_bytes
        .unwrap_or(DEFAULT_FIELD_WINDOW_BYTES);
    if total_output_bytes > output_budget {
        return Err(AppError::Invalid(format!(
            "terrain field window returns {total_output_bytes} field/geometry/padding bytes; request budget is {output_budget}"
        )));
    }
    let source_budget = request
        .max_source_bytes
        .unwrap_or(DEFAULT_FIELD_WINDOW_SOURCE_BYTES);
    let admit_working_set = |field_working_bytes: usize| -> AppResult<usize> {
        let total = field_working_bytes
            .checked_add(geometry_source_bytes)
            .and_then(|value| value.checked_add(aggregation_bytes))
            .ok_or_else(|| AppError::Invalid("terrain source/working budget overflow".into()))?;
        if total > source_budget {
            return Err(AppError::Invalid(format!(
                "terrain field window needs {total} inspected/working bytes (field {field_working_bytes}, geometry {geometry_source_bytes}, aggregation {aggregation_bytes}); request budget is {source_budget}"
            )));
        }
        Ok(total)
    };

    let (
        field_values,
        field_working_bytes,
        source_format,
        cancellation_mode,
        blocked_phase,
        netcdf_path,
    ) = if let Some((netcdf_path, variable)) = netcdf {
        control.set_phase(ReadPhase::NetcdfNative);
        control.check(&request.request_id)?;
        let raw_bytes = dense_field_source_bytes;
        admit_working_set(raw_bytes)?;
        let payload = {
            let native_request = crate::commands::datasets::NetcdfWindowRequest {
                schema_version: request.schema_version,
                request_id: request.request_id.clone(),
                aggregation: if request.aggregation == "point_sample" {
                    "cell_mean".into()
                } else {
                    request.aggregation.clone()
                },
                origin: request.origin.clone(),
                extent: request.extent.clone(),
                stride: vec![1, 1, 1],
                max_output_bytes: Some(raw_bytes),
                max_output_samples: Some(center_elements),
                max_source_bytes: request.max_source_bytes.map(|bytes| bytes as u64),
            };
            let native_control = crate::commands::datasets::NetcdfReadControl::new();
            super::netcdf_dynamic::read_field_window(
                &netcdf_path.to_string_lossy(),
                variable,
                0,
                &native_request,
                &native_control,
            )?
        };
        control.check(&request.request_id)?;
        if payload.source_shape != source_shape || payload.output_shape != request.extent {
            return Err(AppError::Invalid(
                "terrain netCDF source window shape disagrees with the manifest/request".into(),
            ));
        }
        (
            payload.values,
            usize::try_from(payload.source_read_bytes)
                .map_err(|_| AppError::Invalid("terrain netCDF byte count exceeds usize".into()))?,
            native_netcdf_source_format(&payload.source_format),
            payload.cancellation_mode,
            Some("nc_get_vars_float_and_double"),
            Some(netcdf_path.as_path()),
        )
    } else {
        let path = path.ok_or_else(|| AppError::NotFound(field_name.to_owned()))?;
        if is_zfp3d(path) {
            control.set_phase(ReadPhase::ZfpDecode);
            let (declared_shape, reconstruction_bytes) =
                crate::zfp3d::field_shape_and_window_reconstruction_bytes(
                    run_dir, step_index, field_name, origin, extent,
                )?;
            if declared_shape != source_shape {
                return Err(AppError::Invalid(
                    "terrain ZFP shape disagrees with the step manifest".into(),
                ));
            }
            admit_working_set(reconstruction_bytes)?;
            let (shape, values) = crate::zfp3d::reconstruct_field_window(
                run_dir, step_index, field_name, origin, extent,
            )?;
            control.check(&request.request_id)?;
            if shape != source_shape || values.len() != center_elements {
                return Err(AppError::Invalid(
                    "terrain ZFP reconstruction shape changed during decode".into(),
                ));
            }
            (
                values,
                reconstruction_bytes,
                "zfp3d",
                "boundary_only",
                Some("zfp_decode"),
                None,
            )
        } else {
            admit_working_set(dense_field_source_bytes)?;
            let mut dense_request = request.clone();
            dense_request.aggregation = "cell_mean".into();
            dense_request.stride = vec![1, 1, 1];
            dense_request.max_output_bytes = Some(dense_field_source_bytes);
            let (shape, values) = read_npy_source_window(path, &dense_request, control)?;
            if shape != source_shape || values.len() != center_elements {
                return Err(AppError::Invalid(
                    "terrain NPY shape disagrees with the step manifest".into(),
                ));
            }
            (
                values,
                dense_field_source_bytes,
                "npy",
                "cooperative",
                None,
                None,
            )
        }
    };

    let source_working_bytes = admit_working_set(field_working_bytes)?;
    let geometry = read_terrain_geometry_window(
        descriptor,
        Path::new(run_dir),
        step_index,
        netcdf_path,
        origin,
        extent,
        control,
        &request.request_id,
    )?;
    if geometry.source_bytes != geometry_source_bytes {
        return Err(AppError::Invalid(
            "terrain geometry reader byte count disagrees with the admitted budget".into(),
        ));
    }
    let aggregate = super::terrain_window::aggregate_physical_volume(
        &field_values,
        origin,
        extent,
        stride,
        descriptor.horizontal_origin_m,
        descriptor.horizontal_spacing_m,
        &geometry,
        &request.aggregation,
    )?;
    if aggregate.output_shape != expected_output_shape
        || aggregate.working_bytes != aggregation_bytes
    {
        return Err(AppError::Invalid(
            "terrain aggregation result disagrees with its admitted shape/budget".into(),
        ));
    }
    Ok(FieldWindowPayload {
        source_shape,
        output_shape: aggregate.output_shape,
        field_body: bytemuck::cast_slice(&aggregate.values).to_vec(),
        geometry_body: encode_f64_le(
            &aggregate
                .centroids_m
                .iter()
                .copied()
                .chain(aggregate.volumes_m3.iter().copied())
                .collect::<Vec<_>>(),
        )?,
        weighting: "physical_cell_volume",
        sample_location: if request.aggregation == "point_sample" {
            "source_cell_center"
        } else {
            "physical_volume_centroid"
        },
        geometry_kind: "terrain_following",
        source_working_bytes,
        source_format,
        cancellation_mode,
        non_interruptible_phase: blocked_phase,
    })
}

/// Read a bounded spatial subset and/or LOD from a native Nimbus field. The
/// response is binary: `NMBLOD1\0`, a little-endian u32 JSON-header length,
/// UTF-8 header, then little-endian float32 C-order data. The output budget is
/// checked before allocating the response body.
#[tauri::command]
pub async fn read_field_window(
    run_dir: String,
    step_index: u32,
    field_name: String,
    expected_generation: Option<String>,
    request: FieldWindowRequest,
) -> AppResult<Response> {
    // Held across the await so the slot is released on drop, not only on the
    // success path.
    let guard = register_field_read(&request)?;
    let worker_control = guard.control();
    let joined = tauri::async_runtime::spawn_blocking(
        move || -> AppResult<super::io_budget::NativeWindowReadResult> {
            let cancellation_request_id = request.request_id.clone();
            super::io_budget::read_with_budget(
                request
                    .max_source_bytes
                    .unwrap_or(DEFAULT_FIELD_WINDOW_SOURCE_BYTES),
                request
                    .max_output_bytes
                    .unwrap_or(DEFAULT_FIELD_WINDOW_BYTES),
                || worker_control.check(&cancellation_request_id),
                || {
                    worker_control.check(&request.request_id)?;
                    let expected_grid = expected_manifest_grid_shape(
                        Path::new(&run_dir),
                        step_index,
                        &field_name,
                        expected_generation.as_deref(),
                    )?;
                    let terrain = terrain_window_descriptor(
                        Path::new(&run_dir),
                        step_index,
                        expected_generation.as_deref(),
                    )?;
                    let netcdf = netcdf_field_mapping(&run_dir, step_index, &field_name)?;
                    let path = if netcdf.is_none() {
                        Some(field_path(&run_dir, step_index, &field_name)?)
                    } else {
                        None
                    };
                    // Terrain geometry applies to rank-3 atmospheric volumes. Surface and
                    // column products remain ordinary separable x/y arrays even when the
                    // parent run uses a terrain-following vertical coordinate.
                    let payload = if let Some(descriptor) =
                        terrain.as_ref().filter(|_| request.origin.len() == 3)
                    {
                        read_terrain_window_payload(
                            &run_dir,
                            step_index,
                            &field_name,
                            &request,
                            expected_grid,
                            descriptor,
                            netcdf.as_ref(),
                            path.as_deref(),
                            &worker_control,
                        )?
                    } else {
                        read_uniform_window_payload(
                            &run_dir,
                            step_index,
                            &field_name,
                            &request,
                            expected_grid,
                            netcdf.as_ref(),
                            path.as_deref(),
                            &worker_control,
                        )?
                    };
                    ensure_expected_publication_generation(
                        Path::new(&run_dir),
                        step_index,
                        expected_generation.as_deref(),
                    )?;
                    worker_control.check(&request.request_id)?;
                    let header = FieldWindowHeader {
                        schema: "nimbus.studio.field-window",
                        schema_version: FIELD_WINDOW_RESPONSE_SCHEMA_VERSION,
                        request_id: request.request_id.clone(),
                        aggregation: request.aggregation.clone(),
                        weighting: payload.weighting,
                        missing_data_policy: "propagate_non_finite",
                        sample_location: payload.sample_location,
                        geometry_kind: payload.geometry_kind,
                        geometry_dtype: (payload.geometry_kind == "terrain_following")
                            .then_some("<f8"),
                        geometry_layout: (payload.geometry_kind == "terrain_following")
                            .then_some("component_major_x_y_z_volume"),
                        geometry_components: (payload.geometry_kind == "terrain_following")
                            .then_some(vec!["x_m", "y_m", "z_m", "volume_m3"]),
                        source_shape: payload.source_shape,
                        origin: request.origin.clone(),
                        extent: request.extent.clone(),
                        stride: request.stride.clone(),
                        output_shape: payload.output_shape.clone(),
                        element_count: payload.output_shape.iter().product(),
                        decoded_bytes: payload.field_body.len(),
                        geometry_decoded_bytes: payload.geometry_body.len(),
                        geometry_padding_bytes: if payload.geometry_body.is_empty() {
                            0
                        } else {
                            (8 - (payload.field_body.len() % 8)) % 8
                        },
                        source_working_bytes: payload.source_working_bytes,
                        source_working_accounting: "inspected_payload_plus_explicit_working_set",
                        dtype: "<f4",
                        order: "C",
                        source_format: payload.source_format,
                        cancellation_mode: payload.cancellation_mode,
                        non_interruptible_phase: payload.non_interruptible_phase,
                    };
                    encode_field_window(&header, &payload.field_body, &payload.geometry_body)
                },
            )
        },
    )
    .await;
    let result = joined.map_err(|error| AppError::Task(error.to_string()))?;
    let response = result.map(super::io_budget::NativeWindowReadResult::into_response);
    drop(guard);
    response
}

#[tauri::command]
pub async fn read_field_bytes(
    run_dir: String,
    step_index: u32,
    field_name: String,
    expected_generation: Option<String>,
) -> AppResult<Response> {
    // Read (and ZFP-decode, when needed) the field off the Tauri main thread.
    // A synchronous command runs ON the main thread, so a ~168 MB beast-scale
    // field froze the whole UI for the entire read + decode. spawn_blocking moves
    // the blocking I/O onto the blocking pool; only the result bytes cross back.
    let bytes = tauri::async_runtime::spawn_blocking(move || -> AppResult<Vec<u8>> {
        let expected_grid = expected_manifest_grid_shape(
            Path::new(&run_dir),
            step_index,
            &field_name,
            expected_generation.as_deref(),
        )?;
        let netcdf = netcdf_field_mapping(&run_dir, step_index, &field_name)?;
        let path = if netcdf.is_none() {
            Some(field_path(&run_dir, step_index, &field_name)?)
        } else {
            None
        };
        let bytes = if let Some((netcdf_path, variable)) = netcdf {
            let (shape, data) = read_netcdf_field(&netcdf_path, &variable)?;
            validate_manifest_shape(&shape, expected_grid)?;
            crate::zfp3d::encode_npy_f32(&shape, &data)?
        } else if is_zfp3d(path.as_ref().unwrap()) {
            // Decode ZFP and re-serialize as .npy so the TS reader is unchanged.
            let (shape, data) = crate::zfp3d::reconstruct_field(&run_dir, step_index, &field_name)?;
            validate_manifest_shape(&shape, expected_grid)?;
            crate::zfp3d::encode_npy_f32(&shape, &data)?
        } else {
            let bytes = crate::npy::read_f32_file(path.as_ref().unwrap())?;
            let layout = crate::npy::parse_f32_layout(&bytes)?;
            validate_manifest_shape(&layout.shape, expected_grid)?;
            bytes
        };
        // Re-check after all I/O/decode so replacing the directory while a read
        // is in flight cannot commit bytes under the newer run generation.
        ensure_expected_publication_generation(
            Path::new(&run_dir),
            step_index,
            expected_generation.as_deref(),
        )?;
        Ok(bytes)
    })
    .await
    .map_err(|e| AppError::Task(e.to_string()))??;
    Ok(Response::new(bytes))
}

fn json_usize_triplet(value: Option<&serde_json::Value>) -> Option<[usize; 3]> {
    let values = value?.as_array()?;
    if values.len() != 3 {
        return None;
    }
    Some([
        usize::try_from(values[0].as_u64()?).ok()?,
        usize::try_from(values[1].as_u64()?).ok()?,
        usize::try_from(values[2].as_u64()?).ok()?,
    ])
}

fn field_lod_registry_key(
    step_dir: &Path,
    field_name: &str,
    generation: &str,
    level: usize,
) -> String {
    format!("{}\n{field_name}\n{generation}\n{level}", step_dir.to_string_lossy())
}

fn validated_field_brick_hierarchy(
    run_dir: &str,
    step_index: u32,
    field_name: &str,
    expected_generation: Option<&str>,
) -> AppResult<serde_json::Value> {
        ensure_expected_publication_generation(Path::new(run_dir), step_index, expected_generation)?;
        if field_name.is_empty() || field_name.contains('/') || field_name.contains('\\')
            || field_name.contains("..") {
            return Err(AppError::Invalid(format!("invalid field name: {field_name}")));
        }
        let step_dir = Path::new(run_dir).join(format!("step_{step_index:06}"));
        let manifest: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(step_dir.join("manifest.json"))?)
            .map_err(|error| AppError::Invalid(format!("invalid step manifest: {error}")))?;
        let hierarchy = parse_field_brick_hierarchies(&manifest)
            .remove(field_name)
            .ok_or_else(|| AppError::NotFound(format!("{field_name}.bricks.json")))?;
        let path = step_dir.join(&hierarchy.path);
        const MAX_HIERARCHY_BYTES: u64 = 64 * 1024 * 1024;
        let metadata = fs::metadata(&path)?;
        if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_HIERARCHY_BYTES {
            return Err(AppError::Invalid("field brick hierarchy size is invalid".into()));
        }
        let bytes = fs::read(&path)?;
        if sha256_hex(&bytes) != hierarchy.content_sha256 {
            return Err(AppError::Invalid("field brick hierarchy checksum mismatch".into()));
        }
        let value: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|error| AppError::Invalid(format!("invalid field brick hierarchy: {error}")))?;
        let generation = value.get("publicationGeneration").and_then(|item| item.as_str());
        if value.get("schema").and_then(|item| item.as_str())
                != Some("nimbus.field-brick-hierarchy")
            || value.get("schemaVersion").and_then(|item| item.as_u64())
                != Some(u64::from(hierarchy.schema_version))
            || value.get("field").and_then(|item| item.as_str()) != Some(field_name)
            || generation != expected_generation {
            return Err(AppError::Invalid("field brick hierarchy identity mismatch".into()));
        }
        if hierarchy.schema_version >= 2 {
            let source_shape = json_usize_triplet(value.get("sourceShape"))
                .ok_or_else(|| AppError::Invalid("field brick hierarchy source shape is invalid".into()))?;
            let source_offset = json_usize_triplet(value.get("sourceIndexOffset"))
                .ok_or_else(|| AppError::Invalid("field brick hierarchy source offset is invalid".into()))?;
            let levels = value.get("levels").and_then(|item| item.as_array())
                .ok_or_else(|| AppError::Invalid("field brick hierarchy levels are invalid".into()))?;
            let generation = generation.ok_or_else(|| AppError::Invalid(
                "field brick hierarchy generation is absent".into()))?;
            for (position, level) in levels.iter().enumerate().skip(1) {
                if level.get("level").and_then(|item| item.as_u64()) != Some(position as u64) {
                    return Err(AppError::Invalid("field brick hierarchy LOD order is invalid".into()));
                }
                let shape = json_usize_triplet(level.get("shape"))
                    .ok_or_else(|| AppError::Invalid("field brick LOD shape is invalid".into()))?;
                let stride = json_usize_triplet(level.get("stride"))
                    .ok_or_else(|| AppError::Invalid("field brick LOD stride is invalid".into()))?;
                let source = level.get("source").and_then(|item| item.as_object())
                    .ok_or_else(|| AppError::Invalid("field brick LOD source is invalid".into()))?;
                let expected_path = format!("{field_name}.lod{position}.npy");
                let path_name = source.get("path").and_then(|item| item.as_str());
                let expected_sha = source.get("contentSha256").and_then(|item| item.as_str());
                if source.get("kind").and_then(|item| item.as_str()) != Some("npy_lod")
                    || source.get("aggregation").and_then(|item| item.as_str())
                        != Some("recursive_cell_mean_2x2x2")
                    || path_name != Some(expected_path.as_str())
                    || !expected_sha.is_some_and(|sha| sha.len() == 64
                        && sha.bytes().all(|byte| byte.is_ascii_hexdigit())) {
                    return Err(AppError::Invalid("field brick LOD identity is invalid".into()));
                }
                let lod_path = step_dir.join(&expected_path);
                let metadata = fs::metadata(&lod_path)?;
                if !metadata.is_file() || metadata.len() == 0 {
                    return Err(AppError::Invalid("field brick LOD payload is absent".into()));
                }
                let mut file = fs::File::open(&lod_path)?;
                let (npy_shape, data_offset) = read_npy_header(&mut file)?;
                let expected_bytes = shape.into_iter().try_fold(1_u64, |count, extent|
                    count.checked_mul(extent as u64))
                    .and_then(|count| count.checked_mul(4))
                    .and_then(|body| body.checked_add(data_offset as u64));
                if npy_shape != shape || expected_bytes != Some(metadata.len()) {
                    return Err(AppError::Invalid(
                        "field brick LOD payload shape/size is invalid".into()));
                }
                if sha256_file_hex(&lod_path)? != expected_sha.unwrap().to_ascii_lowercase() {
                    return Err(AppError::Invalid("field brick LOD payload checksum mismatch".into()));
                }
                let geometry = if hierarchy.schema_version == 3 {
                    let geometry = source.get("geometry").and_then(|item| item.as_object())
                        .filter(|geometry| geometry.get("kind").and_then(|item| item.as_str())
                            == Some("physical_volume_centroids"))
                        .ok_or_else(|| AppError::Invalid(
                            "terrain field brick LOD geometry is invalid".into()))?;
                    let mut components = Vec::with_capacity(4);
                    for (component, prefix) in [
                        ("xM", "physical_x_center_m"),
                        ("yM", "physical_y_center_m"),
                        ("zM", "physical_z_center_m"),
                        ("volumeM3", "physical_volume_m3"),
                    ] {
                        let descriptor = geometry.get(component).and_then(|item| item.as_object())
                            .ok_or_else(|| AppError::Invalid(
                                "terrain field brick LOD geometry component is invalid".into()))?;
                        let expected_geometry_path = format!("{prefix}.lod{position}.npy");
                        let geometry_path_name = descriptor.get("path").and_then(|item| item.as_str());
                        let geometry_sha = descriptor.get("contentSha256")
                            .and_then(|item| item.as_str());
                        if geometry_path_name != Some(expected_geometry_path.as_str())
                            || !geometry_sha.is_some_and(|sha| sha.len() == 64
                                && sha.bytes().all(|byte| byte.is_ascii_hexdigit())) {
                            return Err(AppError::Invalid(
                                "terrain field brick LOD geometry identity is invalid".into()));
                        }
                        let geometry_path = step_dir.join(expected_geometry_path);
                        let geometry_metadata = fs::metadata(&geometry_path)?;
                        if !geometry_metadata.is_file() || geometry_metadata.len() == 0 {
                            return Err(AppError::Invalid(
                                "terrain field brick LOD geometry payload is absent".into()));
                        }
                        let mut geometry_file = fs::File::open(&geometry_path)?;
                        super::terrain_window::parse_f64_npy_header(&mut geometry_file, shape)?;
                        if sha256_file_hex(&geometry_path)?
                            != geometry_sha.unwrap().to_ascii_lowercase() {
                            return Err(AppError::Invalid(
                                "terrain field brick LOD geometry checksum mismatch".into()));
                        }
                        components.push(ValidatedFieldLodComponent {
                            path: geometry_path,
                            bytes: geometry_metadata.len(),
                            modified: geometry_metadata.modified()?,
                        });
                    }
                    Some(components.try_into().map_err(|_| AppError::Task(
                        "terrain field brick LOD geometry component count changed".into()))?)
                } else {
                    None
                };
                let validated = ValidatedFieldLod {
                    path: lod_path,
                    bytes: metadata.len(),
                    modified: metadata.modified()?,
                    shape,
                    source_shape,
                    source_offset,
                    stride,
                    geometry,
                };
                let current_key = field_lod_registry_key(
                    &step_dir, field_name, generation, position);
                let current_prefix = format!(
                    "{}\n{field_name}\n{generation}\n", step_dir.to_string_lossy());
                let mut registry = VALIDATED_FIELD_LODS.lock().map_err(|_| AppError::Task(
                    "field brick LOD registry is poisoned".into()))?;
                registry.insert(current_key, validated);
                while registry.len() > MAX_VALIDATED_FIELD_LODS {
                    let stale = registry.keys()
                        .find(|key| !key.starts_with(&current_prefix))
                        .cloned();
                    let Some(stale) = stale else {
                        break;
                    };
                    registry.remove(&stale);
                }
            }
        }
        ensure_expected_publication_generation(Path::new(run_dir), step_index, expected_generation)?;
        Ok(value)
}

#[tauri::command]
pub async fn read_field_brick_hierarchy(
    run_dir: String,
    step_index: u32,
    field_name: String,
    expected_generation: Option<String>,
) -> AppResult<serde_json::Value> {
    tauri::async_runtime::spawn_blocking(move || {
        validated_field_brick_hierarchy(
            &run_dir, step_index, &field_name, expected_generation.as_deref())
    })
    .await
    .map_err(|error| AppError::Task(error.to_string()))?
}

#[tauri::command]
pub async fn read_field_brick_window(
    run_dir: String,
    step_index: u32,
    field_name: String,
    expected_generation: Option<String>,
    level: u32,
    request: FieldWindowRequest,
) -> AppResult<Response> {
    if level == 0 {
        return Err(AppError::Invalid(
            "level zero must use the primary field-window command".into(),
        ));
    }
    let generation = expected_generation.clone().ok_or_else(|| {
        AppError::Invalid("durable brick LOD reads require a publication generation".into())
    })?;
    let guard = register_field_read(&request)?;
    let worker_control = guard.control();
    let joined = tauri::async_runtime::spawn_blocking(move || -> AppResult<super::io_budget::NativeWindowReadResult> {
        let cancellation_request_id = request.request_id.clone();
        super::io_budget::read_with_budget(
            request.max_source_bytes.unwrap_or(DEFAULT_FIELD_WINDOW_SOURCE_BYTES),
            request.max_output_bytes.unwrap_or(DEFAULT_FIELD_WINDOW_BYTES),
            || worker_control.check(&cancellation_request_id),
            || {
                worker_control.check(&request.request_id)?;
                ensure_expected_publication_generation(
                    Path::new(&run_dir), step_index, Some(&generation))?;
                let step_dir = Path::new(&run_dir).join(format!("step_{step_index:06}"));
                let key = field_lod_registry_key(&step_dir, &field_name, &generation, level as usize);
                let lod = VALIDATED_FIELD_LODS.lock().map_err(|_| AppError::Task(
                    "field brick LOD registry is poisoned".into()))?.get(&key).cloned()
                    .ok_or_else(|| AppError::Invalid(
                        "field brick LOD was not validated through its hierarchy".into()))?;
                let metadata = fs::metadata(&lod.path)?;
                if !metadata.is_file() || metadata.len() != lod.bytes
                    || metadata.modified()? != lod.modified {
                    return Err(AppError::Invalid(
                        "field brick LOD payload changed after hierarchy validation".into()));
                }
                if let Some(geometry) = &lod.geometry {
                    for component in geometry {
                        let metadata = fs::metadata(&component.path)?;
                        if !metadata.is_file() || metadata.len() != component.bytes
                            || metadata.modified()? != component.modified {
                            return Err(AppError::Invalid(
                                "terrain field brick LOD geometry changed after hierarchy validation".into()));
                        }
                    }
                }
                if request.origin.len() != 3 || request.extent.len() != 3
                    || request.stride.len() != 3 || request.aggregation != "cell_mean" {
                    return Err(AppError::Invalid(
                        "field brick LOD request geometry is invalid".into()));
                }
                let mut lod_origin = vec![0_usize; 3];
                let mut lod_extent = vec![0_usize; 3];
                for axis in 0..3 {
                    if request.stride[axis] != lod.stride[axis]
                        || request.origin[axis] < lod.source_offset[axis] {
                        return Err(AppError::Invalid(
                            "field brick LOD request disagrees with its hierarchy stride".into()));
                    }
                    let local_origin = request.origin[axis] - lod.source_offset[axis];
                    if local_origin % lod.stride[axis] != 0 {
                        return Err(AppError::Invalid(
                            "field brick LOD request origin is not level-aligned".into()));
                    }
                    lod_origin[axis] = local_origin / lod.stride[axis];
                    lod_extent[axis] = (request.extent[axis] - 1) / lod.stride[axis] + 1;
                    if lod_origin[axis].checked_add(lod_extent[axis])
                            .map_or(true, |upper| upper > lod.shape[axis])
                        || request.extent[axis] != std::cmp::min(
                            lod.source_shape[axis] - local_origin,
                            lod_extent[axis] * lod.stride[axis]) {
                        return Err(AppError::Invalid(
                            "field brick LOD request extent is not level-aligned".into()));
                    }
                }
                const FLOAT_BYTES: usize = std::mem::size_of::<f32>();
                let output_bytes = lod_extent.iter().try_fold(FLOAT_BYTES, |bytes, extent|
                    bytes.checked_mul(*extent))
                    .ok_or_else(|| AppError::Invalid("field brick LOD request size overflow".into()))?;
                let mut internal = request.clone();
                internal.aggregation = "point_sample".into();
                internal.origin = lod_origin;
                internal.extent = lod_extent;
                internal.stride = vec![1, 1, 1];
                internal.max_output_bytes = Some(output_bytes);
                internal.max_source_bytes = Some(output_bytes);
                let (lod_shape, output_shape, field_body) =
                    read_npy_window(&lod.path, &internal, &worker_control)?;
                if lod_shape != lod.shape || field_body.len() != output_bytes {
                    return Err(AppError::Invalid(
                        "field brick LOD payload changed during its bounded read".into()));
                }
                let (geometry_body, geometry_source_bytes) = if let Some(geometry) = &lod.geometry {
                    let mut check = || worker_control.check(&request.request_id);
                    let physical = super::terrain_window::read_npy_lod_geometry_window(
                        [
                            geometry[0].path.as_path(), geometry[1].path.as_path(),
                            geometry[2].path.as_path(), geometry[3].path.as_path(),
                        ],
                        lod.shape,
                        [internal.origin[0], internal.origin[1], internal.origin[2]],
                        [internal.extent[0], internal.extent[1], internal.extent[2]],
                        &mut check,
                    )?;
                    let values = physical.x_m.into_iter()
                        .chain(physical.y_m)
                        .chain(physical.z_m)
                        .chain(physical.volumes_m3)
                        .collect::<Vec<_>>();
                    (encode_f64_le(&values)?, physical.source_bytes)
                } else {
                    (Vec::new(), 0)
                };
                let geometry_padding_bytes = if geometry_body.is_empty() {
                    0
                } else {
                    (8 - (field_body.len() % 8)) % 8
                };
                let total_output_bytes = field_body.len()
                    .checked_add(geometry_padding_bytes)
                    .and_then(|bytes| bytes.checked_add(geometry_body.len()))
                    .ok_or_else(|| AppError::Invalid("field brick LOD response size overflow".into()))?;
                if total_output_bytes > request.max_output_bytes.unwrap_or(DEFAULT_FIELD_WINDOW_BYTES) {
                    return Err(AppError::Invalid(
                        "field brick LOD response exceeds its admitted output budget".into()));
                }
                let source_working_bytes = output_bytes.checked_add(geometry_source_bytes)
                    .ok_or_else(|| AppError::Invalid("field brick LOD source size overflow".into()))?;
                if source_working_bytes > request.max_source_bytes
                        .unwrap_or(DEFAULT_FIELD_WINDOW_SOURCE_BYTES) {
                    return Err(AppError::Invalid(
                        "field brick LOD source exceeds its admitted working budget".into()));
                }
                let storage_shape = expected_manifest_grid_shape(
                    Path::new(&run_dir), step_index, &field_name, Some(&generation))?
                    .map(|shape| shape.to_vec())
                    .unwrap_or_else(|| lod.source_shape.iter().zip(lod.source_offset)
                        .map(|(shape, offset)| shape + offset).collect());
                ensure_expected_publication_generation(
                    Path::new(&run_dir), step_index, Some(&generation))?;
                worker_control.check(&request.request_id)?;
                let header = FieldWindowHeader {
                    schema: "nimbus.studio.field-window",
                    schema_version: FIELD_WINDOW_RESPONSE_SCHEMA_VERSION,
                    request_id: request.request_id.clone(),
                    aggregation: "cell_mean".into(),
                    weighting: if geometry_body.is_empty() {
                        "uniform_cell_volume"
                    } else {
                        "physical_cell_volume"
                    },
                    missing_data_policy: "propagate_non_finite",
                    sample_location: if geometry_body.is_empty() {
                        "block_center"
                    } else {
                        "physical_volume_centroid"
                    },
                    geometry_kind: if geometry_body.is_empty() {
                        "separable_uniform"
                    } else {
                        "terrain_following"
                    },
                    geometry_dtype: (!geometry_body.is_empty()).then_some("<f8"),
                    geometry_layout: (!geometry_body.is_empty())
                        .then_some("component_major_x_y_z_volume"),
                    geometry_components: (!geometry_body.is_empty())
                        .then_some(vec!["x_m", "y_m", "z_m", "volume_m3"]),
                    source_shape: storage_shape,
                    origin: request.origin,
                    extent: request.extent,
                    stride: request.stride,
                    output_shape: output_shape.clone(),
                    element_count: output_shape.iter().product(),
                    decoded_bytes: field_body.len(),
                    geometry_decoded_bytes: geometry_body.len(),
                    geometry_padding_bytes,
                    source_working_bytes,
                    source_working_accounting: "inspected_payload_plus_explicit_working_set",
                    dtype: "<f4",
                    order: "C",
                    source_format: "npy_lod",
                    cancellation_mode: "cooperative",
                    non_interruptible_phase: None,
                };
                encode_field_window(&header, &field_body, &geometry_body)
            },
        )
    }).await;
    let result = joined.map_err(|error| AppError::Task(error.to_string()))??;
    let response = result.into_response();
    drop(guard);
    Ok(response)
}

/// Read the run-level `trajectories.npy` (Lagrangian parcel paths) written once
/// per run at the output-dir root — shape (n_parcels, n_samples, 9) where the
/// columns are [t, x, y, z, w, zeta, tilting, stretching, baroclinic].
/// `NotFound` when the run was produced without `--trajectories`.
#[tauri::command]
pub async fn read_trajectories(
    run_dir: String,
    expected_generation: Option<String>,
) -> AppResult<Response> {
    // Off-main-thread like read_field_bytes: trajectories.npy can be multi-MB.
    let bytes = tauri::async_runtime::spawn_blocking(move || -> AppResult<Vec<u8>> {
        let bytes = read_run_artifact_bytes(
            Path::new(&run_dir),
            "trajectories.npy",
            expected_generation.as_deref(),
        )?;
        let layout = crate::npy::parse_f32_layout(&bytes)?;
        if layout.shape.len() != 3 || layout.shape.last() != Some(&9) {
            return Err(AppError::Invalid(format!(
                "trajectories.npy must have shape (parcels, samples, 9), got {}",
                layout
                    .shape
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join("x")
            )));
        }
        Ok(bytes)
    })
    .await
    .map_err(|e| AppError::Task(e.to_string()))??;
    Ok(Response::new(bytes))
}

/// Upper bound on the storm-track summary. The tracker writes one small JSON
/// object per publication, so anything larger means a different file was
/// published under that name — refusing is safer than streaming an unbounded
/// blob into the UI thread.
const MAX_STORM_TRACK_BYTES: usize = 1 << 20;

/// Read the run-level `storm_track_latest.json` written by the simulator's
/// storm tracker (mesocyclone centroid, rotating-core extent, peak updraft
/// helicity, rotation class).
///
/// Returns `Ok(None)` when the file is absent. Tracking is gated on
/// `storm_tracking.enabled`, so most runs simply do not have one — that is an
/// ordinary configuration, not a failure, and surfacing it as an error would
/// put a red banner on every untracked run.
#[tauri::command]
pub fn read_storm_track(
    run_dir: String,
    expected_generation: Option<String>,
) -> AppResult<Option<String>> {
    const ARTIFACT: &str = "storm_track_latest.json";
    let path = Path::new(&run_dir);
    if !run_artifact_exists(path, ARTIFACT, expected_generation.as_deref()) {
        return Ok(None);
    }
    let bytes = read_run_artifact_bytes(path, ARTIFACT, expected_generation.as_deref())?;
    if bytes.len() > MAX_STORM_TRACK_BYTES {
        return Err(AppError::Invalid(format!(
            "{ARTIFACT} is {} bytes, above the {MAX_STORM_TRACK_BYTES}-byte summary bound",
            bytes.len()
        )));
    }
    // The tracker writes the file continuously, so a read can land mid-write and
    // catch a partial object. Invalid UTF-8 is that race, not corruption: report
    // absence so the caller retries on the next publication instead of latching
    // an error the run will resolve on its own.
    match String::from_utf8(bytes) {
        Ok(contents) => Ok(Some(contents)),
        Err(_) => Ok(None),
    }
}

/// Cheap existence check for the run's trajectories.npy (no read/parse). Drives
/// data-aware gating of the Trajectories plot category.
#[tauri::command]
pub fn run_has_trajectories(run_dir: String, expected_generation: Option<String>) -> bool {
    run_artifact_exists(
        Path::new(&run_dir),
        "trajectories.npy",
        expected_generation.as_deref(),
    )
}

/// Scalar reduction of one field, computed on the Rust side so the multi-MB
/// array never crosses the IPC boundary. `None` for the four statistics means
/// the field had no finite values (the TS side maps that to NaN). `finite` /
/// `total` expose the non-finite ratio for diagnostics.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FieldReduction {
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub mean: Option<f64>,
    pub rms: Option<f64>,
    pub finite: u64,
    pub total: u64,
    /// Counts of finite cells passing a threshold test, 0 when no threshold was
    /// requested. `count_above`: v > t; `count_below`: v < t; `count_abs_above`:
    /// |v| > t. Back area-style time-series metrics (mesocyclone area = count of
    /// |ζ| > t cells; cold-pool extent = count of θ' < t cells).
    pub count_above: u64,
    pub count_below: u64,
    pub count_abs_above: u64,
}

struct FieldReductionAccumulator {
    min: f64,
    max: f64,
    sum: f64,
    sum_sq: f64,
    finite: u64,
    total: u64,
    count_above: u64,
    count_below: u64,
    count_abs_above: u64,
}

impl FieldReductionAccumulator {
    fn new() -> Self {
        Self {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            sum: 0.0,
            sum_sq: 0.0,
            finite: 0,
            total: 0,
            count_above: 0,
            count_below: 0,
            count_abs_above: 0,
        }
    }

    fn push(&mut self, value: f32, threshold: Option<f64>) {
        self.total += 1;
        let value = f64::from(value);
        if !value.is_finite() {
            return;
        }
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        self.sum += value;
        self.sum_sq += value * value;
        self.finite += 1;
        if let Some(limit) = threshold {
            self.count_above += u64::from(value > limit);
            self.count_below += u64::from(value < limit);
            self.count_abs_above += u64::from(value.abs() > limit);
        }
    }

    fn finish(self) -> FieldReduction {
        let divisor = self.finite as f64;
        FieldReduction {
            min: (self.finite != 0).then_some(self.min),
            max: (self.finite != 0).then_some(self.max),
            mean: (self.finite != 0).then_some(self.sum / divisor),
            rms: (self.finite != 0).then_some((self.sum_sq / divisor).sqrt()),
            finite: self.finite,
            total: self.total,
            count_above: self.count_above,
            count_below: self.count_below,
            count_abs_above: self.count_abs_above,
        }
    }
}

fn visit_npy_f32<F>(path: &Path, mut visitor: F) -> AppResult<Vec<usize>>
where
    F: FnMut(&[usize], usize, f32),
{
    let mut file = fs::File::open(path)?;
    let file_len = usize::try_from(file.metadata()?.len())
        .map_err(|_| AppError::Invalid("NPY file length does not fit usize".into()))?;
    let (shape, data_offset) = read_npy_header(&mut file)?;
    let logical = crate::npy::validate_logical_field_shape(&shape)?;
    let expected_len = data_offset
        .checked_add(logical.decoded_bytes)
        .ok_or_else(|| AppError::Invalid("NPY file length overflow".into()))?;
    if file_len != expected_len {
        return Err(AppError::Invalid(format!(
            "NPY body size mismatch: need {expected_len} bytes, have {file_len}"
        )));
    }
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut linear = 0_usize;
    while linear < logical.element_count {
        let elements = (logical.element_count - linear).min(buffer.len() / size_of::<f32>());
        let bytes = elements * size_of::<f32>();
        file.read_exact(&mut buffer[..bytes])?;
        for chunk in buffer[..bytes].chunks_exact(size_of::<f32>()) {
            visitor(
                &shape,
                linear,
                f32::from_le_bytes(chunk.try_into().expect("four-byte f32 chunk")),
            );
            linear += 1;
        }
    }
    Ok(shape)
}

fn reduce_npy_stream(
    path: &Path,
    threshold: Option<f64>,
    valid: Option<&AxisBoundsU32>,
) -> AppResult<(Vec<usize>, FieldReduction)> {
    let mut accumulator = FieldReductionAccumulator::new();
    let shape = visit_npy_f32(path, |shape, linear, value| {
        if linear_index_is_valid(linear, Some(shape), valid) {
            accumulator.push(value, threshold);
        }
    })?;
    Ok((shape, accumulator.finish()))
}

fn reduce_zfp_bricks(
    run_dir: &str,
    step_index: u32,
    field_name: &str,
    threshold: Option<f64>,
    valid: Option<&AxisBoundsU32>,
) -> AppResult<(Vec<usize>, FieldReduction)> {
    let mut accumulator = FieldReductionAccumulator::new();
    let shape = crate::zfp3d::visit_reconstructed_field_bricks(
        run_dir,
        step_index,
        field_name,
        |origin, extent, values| {
            for i in 0..extent[0] {
                for j in 0..extent[1] {
                    for k in 0..extent[2] {
                        let global = [origin[0] + i, origin[1] + j, origin[2] + k];
                        if valid.is_some_and(|range| {
                            (0..3).any(|axis| {
                                global[axis] < range.lower[axis] as usize
                                    || global[axis] > range.upper[axis] as usize
                            })
                        }) {
                            continue;
                        }
                        let linear = (i * extent[1] + j) * extent[2] + k;
                        accumulator.push(values[linear], threshold);
                    }
                }
            }
            Ok(())
        },
    )?;
    Ok((shape, accumulator.finish()))
}

/// Reduce a little-endian `f32` body to min/max/mean/rms over finite values,
/// NaN-skipping. Matches `reduceScalar` / `fieldRange` in `ui/src/lib/npy.ts`:
/// both read each `f32` widened to `f64` and accumulate in `f64`.
fn reduce_f32_le(
    body: &[u8],
    threshold: Option<f64>,
    shape: Option<&[usize]>,
    valid: Option<&AxisBoundsU32>,
) -> FieldReduction {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut finite: u64 = 0;
    let mut count_above: u64 = 0;
    let mut count_below: u64 = 0;
    let mut count_abs_above: u64 = 0;

    let mut total: u64 = 0;
    for (linear, chunk) in body.chunks_exact(4).enumerate() {
        if !linear_index_is_valid(linear, shape, valid) {
            continue;
        }
        total += 1;
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64;
        if v.is_finite() {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            sum += v;
            sum_sq += v * v;
            finite += 1;
            if let Some(t) = threshold {
                if v > t {
                    count_above += 1;
                }
                if v < t {
                    count_below += 1;
                }
                if v.abs() > t {
                    count_abs_above += 1;
                }
            }
        }
    }

    if finite == 0 {
        return FieldReduction {
            min: None,
            max: None,
            mean: None,
            rms: None,
            finite: 0,
            total,
            count_above: 0,
            count_below: 0,
            count_abs_above: 0,
        };
    }
    let n = finite as f64;
    FieldReduction {
        min: Some(min),
        max: Some(max),
        mean: Some(sum / n),
        rms: Some((sum_sq / n).sqrt()),
        finite,
        total,
        count_above,
        count_below,
        count_abs_above,
    }
}

fn linear_index_is_valid(
    linear: usize,
    shape: Option<&[usize]>,
    valid: Option<&AxisBoundsU32>,
) -> bool {
    let (Some(shape), Some(valid)) = (shape, valid) else {
        return true;
    };
    if shape.len() == 3 {
        let d1 = shape[1];
        let d2 = shape[2];
        let i = linear / (d1 * d2);
        let remainder = linear % (d1 * d2);
        let j = remainder / d2;
        let k = remainder % d2;
        return i >= valid.lower[0] as usize
            && i <= valid.upper[0] as usize
            && j >= valid.lower[1] as usize
            && j <= valid.upper[1] as usize
            && k >= valid.lower[2] as usize
            && k <= valid.upper[2] as usize;
    }
    if shape.len() == 2 {
        let i = linear / shape[1];
        let j = linear % shape[1];
        return i >= valid.lower[0] as usize
            && i <= valid.upper[0] as usize
            && j >= valid.lower[1] as usize
            && j <= valid.upper[1] as usize;
    }
    true
}

const TERRAIN_REDUCTION_INTERFACE_SLAB_BYTES: usize = 8 * 1024 * 1024;

fn read_terrain_interface_slab(
    descriptor: &TerrainWindowDescriptor,
    run_dir: &Path,
    step_index: u32,
    netcdf_path: Option<&Path>,
    i_begin: usize,
    i_count: usize,
) -> AppResult<Vec<f64>> {
    let extent = [
        i_count,
        descriptor.source_shape[1],
        descriptor.source_shape[2]
            .checked_add(1)
            .ok_or_else(|| AppError::Invalid("terrain interface shape overflow".into()))?,
    ];
    let origin = [i_begin, 0, 0];
    match descriptor.storage {
        TerrainGeometryStorage::NpySidecar => {
            let step_dir = run_dir.join(format!("step_{step_index:06}"));
            super::terrain_window::read_npy_interface_window(
                &step_dir,
                descriptor.source_shape,
                origin,
                extent,
            )
            .map(|(values, _)| values)
        }
        TerrainGeometryStorage::NetcdfVariable => {
            let path = netcdf_path.ok_or_else(|| {
                AppError::Invalid("terrain netCDF reduction has no shared-file mapping".into())
            })?;
            let (shape, values, _) = super::netcdf_dynamic::read_f64_window(
                &path.to_string_lossy(),
                "physical_z_interface_m",
                origin,
                extent,
            )?;
            let expected = vec![
                descriptor.source_shape[0],
                descriptor.source_shape[1],
                descriptor.source_shape[2] + 1,
            ];
            if shape != expected {
                return Err(AppError::Invalid(
                    "terrain netCDF interface shape disagrees with the manifest".into(),
                ));
            }
            Ok(values)
        }
    }
}

fn terrain_reduction_slab_x(shape: &[usize]) -> AppResult<usize> {
    let row_bytes = shape[1]
        .checked_mul(
            shape[2]
                .checked_add(1)
                .ok_or_else(|| AppError::Invalid("terrain interface row overflow".into()))?,
        )
        .and_then(|value| value.checked_mul(size_of::<f64>()))
        .ok_or_else(|| AppError::Invalid("terrain interface slab size overflow".into()))?;
    Ok((TERRAIN_REDUCTION_INTERFACE_SLAB_BYTES / row_bytes)
        .max(1)
        .min(shape[0]))
}

fn validate_terrain_reduction_inputs(
    body: &[u8],
    shape: &[usize],
    descriptor: &TerrainWindowDescriptor,
) -> AppResult<()> {
    if shape.len() != 3 || shape != descriptor.source_shape {
        return Err(AppError::Invalid(
            "terrain reduction field shape disagrees with the geometry manifest".into(),
        ));
    }
    let expected = shape
        .iter()
        .try_fold(1_usize, |product, value| product.checked_mul(*value))
        .and_then(|value| value.checked_mul(size_of::<f32>()))
        .ok_or_else(|| AppError::Invalid("terrain reduction field size overflow".into()))?;
    if body.len() != expected {
        return Err(AppError::Invalid(
            "terrain reduction field body does not match its shape".into(),
        ));
    }
    Ok(())
}

fn terrain_field_value(body: &[u8], linear: usize) -> f64 {
    let offset = linear * size_of::<f32>();
    f64::from(f32::from_le_bytes(
        body[offset..offset + size_of::<f32>()]
            .try_into()
            .expect("validated terrain field body"),
    ))
}

// Reduction inputs are explicit because geometry can come from either sidecars or netCDF.
#[allow(clippy::too_many_arguments)]
fn reduce_terrain_f32_le(
    body: &[u8],
    shape: &[usize],
    descriptor: &TerrainWindowDescriptor,
    run_dir: &Path,
    step_index: u32,
    netcdf_path: Option<&Path>,
    threshold: Option<f64>,
    valid: Option<&AxisBoundsU32>,
) -> AppResult<FieldReduction> {
    validate_terrain_reduction_inputs(body, shape, descriptor)?;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut weighted_sum = 0.0_f64;
    let mut weighted_sum_sq = 0.0_f64;
    let mut finite_volume = 0.0_f64;
    let mut finite = 0_u64;
    let mut total = 0_u64;
    let mut count_above = 0_u64;
    let mut count_below = 0_u64;
    let mut count_abs_above = 0_u64;
    let slab_x = terrain_reduction_slab_x(shape)?;
    for i_begin in (0..shape[0]).step_by(slab_x) {
        let i_count = slab_x.min(shape[0] - i_begin);
        let interfaces = read_terrain_interface_slab(
            descriptor,
            run_dir,
            step_index,
            netcdf_path,
            i_begin,
            i_count,
        )?;
        let interface_nz = shape[2] + 1;
        for local_i in 0..i_count {
            let i = i_begin + local_i;
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let face = (local_i * shape[1] + j) * interface_nz + k;
                    let lower = interfaces[face];
                    let upper = interfaces[face + 1];
                    if !lower.is_finite() || !upper.is_finite() || upper <= lower {
                        return Err(AppError::Invalid(format!(
                            "malformed terrain interface geometry at ({i},{j},{k})"
                        )));
                    }
                    if valid.is_some_and(|range| {
                        i < range.lower[0] as usize
                            || i > range.upper[0] as usize
                            || j < range.lower[1] as usize
                            || j > range.upper[1] as usize
                            || k < range.lower[2] as usize
                            || k > range.upper[2] as usize
                    }) {
                        continue;
                    }
                    total += 1;
                    let linear = (i * shape[1] + j) * shape[2] + k;
                    let value = terrain_field_value(body, linear);
                    if !value.is_finite() {
                        continue;
                    }
                    let volume = descriptor.horizontal_spacing_m[0]
                        * descriptor.horizontal_spacing_m[1]
                        * (upper - lower);
                    if !volume.is_finite() || volume <= 0.0 {
                        return Err(AppError::Invalid(format!(
                            "non-positive terrain volume at ({i},{j},{k})"
                        )));
                    }
                    min = min.min(value);
                    max = max.max(value);
                    weighted_sum += value * volume;
                    weighted_sum_sq += value * value * volume;
                    finite_volume += volume;
                    finite += 1;
                    if let Some(limit) = threshold {
                        count_above += u64::from(value > limit);
                        count_below += u64::from(value < limit);
                        count_abs_above += u64::from(value.abs() > limit);
                    }
                }
            }
        }
    }
    if finite == 0 {
        return Ok(FieldReduction {
            min: None,
            max: None,
            mean: None,
            rms: None,
            finite: 0,
            total,
            count_above: 0,
            count_below: 0,
            count_abs_above: 0,
        });
    }
    if !finite_volume.is_finite() || finite_volume <= 0.0 {
        return Err(AppError::Invalid(
            "terrain reduction produced invalid finite volume".into(),
        ));
    }
    Ok(FieldReduction {
        min: Some(min),
        max: Some(max),
        mean: Some(weighted_sum / finite_volume),
        rms: Some((weighted_sum_sq / finite_volume).sqrt()),
        finite,
        total,
        count_above,
        count_below,
        count_abs_above,
    })
}

#[tauri::command]
pub async fn reduce_field(
    run_dir: String,
    step_index: u32,
    field_name: String,
    threshold: Option<f64>,
    expected_generation: Option<String>,
) -> AppResult<FieldReduction> {
    // Read + O(N) reduce off the main thread: the reduction loops over every
    // element (~42M at beast scale), which on the main thread would freeze the
    // UI for the whole pass. The time-series plot fires one of these per step.
    tauri::async_runtime::spawn_blocking(move || -> AppResult<FieldReduction> {
        let _reduction_slot = FIELD_REDUCTION_LOCK
            .lock()
            .map_err(|_| AppError::Task("field reduction lock poisoned".into()))?;
        let expected_grid = expected_manifest_grid_shape(
            Path::new(&run_dir),
            step_index,
            &field_name,
            expected_generation.as_deref(),
        )?;
        let terrain = terrain_window_descriptor(
            Path::new(&run_dir),
            step_index,
            expected_generation.as_deref(),
        )?;
        let valid_range = expected_manifest_valid_index_range(
            Path::new(&run_dir),
            step_index,
            expected_generation.as_deref(),
        )?;
        let netcdf = netcdf_field_mapping(&run_dir, step_index, &field_name)?;
        let path = if netcdf.is_none() {
            Some(field_path(&run_dir, step_index, &field_name)?)
        } else {
            None
        };
        let reduction = if let Some((netcdf_path, variable)) = netcdf.as_ref() {
            let (shape, data) = read_netcdf_field(netcdf_path, variable)?;
            validate_manifest_shape(&shape, expected_grid)?;
            if let Some(descriptor) = terrain.as_ref() {
                reduce_terrain_f32_le(
                    bytemuck::cast_slice(&data),
                    &shape,
                    descriptor,
                    Path::new(&run_dir),
                    step_index,
                    Some(netcdf_path),
                    threshold,
                    valid_range.as_ref(),
                )?
            } else {
                reduce_f32_le(
                    bytemuck::cast_slice(&data),
                    threshold,
                    Some(&shape),
                    valid_range.as_ref(),
                )
            }
        } else if is_zfp3d(path.as_ref().unwrap()) {
            if let Some(descriptor) = terrain.as_ref() {
                let (shape, data) =
                    crate::zfp3d::reconstruct_field(&run_dir, step_index, &field_name)?;
                validate_manifest_shape(&shape, expected_grid)?;
                reduce_terrain_f32_le(
                    bytemuck::cast_slice(&data),
                    &shape,
                    descriptor,
                    Path::new(&run_dir),
                    step_index,
                    None,
                    threshold,
                    valid_range.as_ref(),
                )?
            } else {
                let (shape, reduction) = reduce_zfp_bricks(
                    &run_dir,
                    step_index,
                    &field_name,
                    threshold,
                    valid_range.as_ref(),
                )?;
                validate_manifest_shape(&shape, expected_grid)?;
                reduction
            }
        } else {
            if let Some(descriptor) = terrain.as_ref() {
                let bytes = crate::npy::read_f32_file(path.as_ref().unwrap())?;
                let layout = crate::npy::parse_f32_layout(&bytes)?;
                validate_manifest_shape(&layout.shape, expected_grid)?;
                let body = &bytes[layout.data_offset..layout.data_offset + layout.decoded_bytes];
                reduce_terrain_f32_le(
                    body,
                    &layout.shape,
                    descriptor,
                    Path::new(&run_dir),
                    step_index,
                    None,
                    threshold,
                    valid_range.as_ref(),
                )?
            } else {
                let (shape, reduction) =
                    reduce_npy_stream(path.as_ref().unwrap(), threshold, valid_range.as_ref())?;
                validate_manifest_shape(&shape, expected_grid)?;
                reduction
            }
        };
        ensure_expected_publication_generation(
            Path::new(&run_dir),
            step_index,
            expected_generation.as_deref(),
        )?;
        Ok(reduction)
    })
    .await
    .map_err(|e| AppError::Task(e.to_string()))?
}

/// Per-index reduction of one 3D field along an axis: for each index along
/// `axis`, reduce (NaN-skipping) over the other two dimensions. `None` where a
/// slab had no finite values; `values.len() == shape[axis]`. Backs the
/// time-height (Hovmöller) plot, which needs one value per level per step
/// (e.g. max-over-horizontal at each height) — a handful of floats per step
/// rather than the whole field across the IPC boundary.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AxisReduction {
    pub values: Vec<Option<f64>>,
    pub axis: u8,
    pub axis_len: usize,
}

enum Reducer {
    Max,
    Min,
    Mean,
    Rms,
}

struct AxisReductionAccumulator {
    min: Vec<f64>,
    max: Vec<f64>,
    sum: Vec<f64>,
    sum_sq: Vec<f64>,
    count: Vec<u64>,
}

impl AxisReductionAccumulator {
    fn new() -> Self {
        Self {
            min: Vec::new(),
            max: Vec::new(),
            sum: Vec::new(),
            sum_sq: Vec::new(),
            count: Vec::new(),
        }
    }

    fn push(&mut self, index: usize, value: f32) {
        if !value.is_finite() {
            return;
        }
        if self.count.len() <= index {
            let length = index + 1;
            self.min.resize(length, f64::INFINITY);
            self.max.resize(length, f64::NEG_INFINITY);
            self.sum.resize(length, 0.0);
            self.sum_sq.resize(length, 0.0);
            self.count.resize(length, 0);
        }
        let value = f64::from(value);
        self.min[index] = self.min[index].min(value);
        self.max[index] = self.max[index].max(value);
        self.sum[index] += value;
        self.sum_sq[index] += value * value;
        self.count[index] += 1;
    }

    fn finish(mut self, length: usize, reducer: &Reducer) -> Vec<Option<f64>> {
        self.min.resize(length, f64::INFINITY);
        self.max.resize(length, f64::NEG_INFINITY);
        self.sum.resize(length, 0.0);
        self.sum_sq.resize(length, 0.0);
        self.count.resize(length, 0);
        (0..length)
            .map(|index| {
                let count = self.count[index];
                (count != 0).then(|| match reducer {
                    Reducer::Max => self.max[index],
                    Reducer::Min => self.min[index],
                    Reducer::Mean => self.sum[index] / count as f64,
                    Reducer::Rms => (self.sum_sq[index] / count as f64).sqrt(),
                })
            })
            .collect()
    }
}

fn reduce_npy_axis_stream(
    path: &Path,
    axis: usize,
    reducer: &Reducer,
    valid: Option<&AxisBoundsU32>,
) -> AppResult<(Vec<usize>, Vec<Option<f64>>)> {
    let mut accumulator = AxisReductionAccumulator::new();
    let shape = visit_npy_f32(path, |shape, linear, value| {
        if shape.len() != 3 || !linear_index_is_valid(linear, Some(shape), valid) {
            return;
        }
        let plane = shape[1] * shape[2];
        let indices = [
            linear / plane,
            (linear % plane) / shape[2],
            linear % shape[2],
        ];
        accumulator.push(indices[axis], value);
    })?;
    if shape.len() != 3 {
        return Err(AppError::Invalid(format!(
            "reduce_field_axis needs a 3D field, got {}D",
            shape.len()
        )));
    }
    let values = accumulator.finish(shape[axis], reducer);
    Ok((shape, values))
}

fn reduce_zfp_axis_bricks(
    run_dir: &str,
    step_index: u32,
    field_name: &str,
    axis: usize,
    reducer: &Reducer,
    valid: Option<&AxisBoundsU32>,
) -> AppResult<(Vec<usize>, Vec<Option<f64>>)> {
    let mut accumulator = AxisReductionAccumulator::new();
    let shape = crate::zfp3d::visit_reconstructed_field_bricks(
        run_dir,
        step_index,
        field_name,
        |origin, extent, values| {
            for i in 0..extent[0] {
                for j in 0..extent[1] {
                    for k in 0..extent[2] {
                        let global = [origin[0] + i, origin[1] + j, origin[2] + k];
                        if valid.is_some_and(|range| {
                            (0..3).any(|coordinate| {
                                global[coordinate] < range.lower[coordinate] as usize
                                    || global[coordinate] > range.upper[coordinate] as usize
                            })
                        }) {
                            continue;
                        }
                        let linear = (i * extent[1] + j) * extent[2] + k;
                        accumulator.push(global[axis], values[linear]);
                    }
                }
            }
            Ok(())
        },
    )?;
    let values = accumulator.finish(shape[axis], reducer);
    Ok((shape, values))
}

fn parse_reducer(s: &str) -> AppResult<Reducer> {
    match s {
        "max" => Ok(Reducer::Max),
        "min" => Ok(Reducer::Min),
        "mean" => Ok(Reducer::Mean),
        "rms" => Ok(Reducer::Rms),
        other => Err(AppError::Invalid(format!("unknown reducer '{other}'"))),
    }
}

/// Reduce a 3D C-contiguous `f32` body over the two non-`axis` dims, yielding
/// one value per index along `axis`. `shape` is `[d0, d1, d2]`; index layout is
/// `((i*d1 + j)*d2 + k)`. Single O(N) pass accumulating per axis-index.
fn reduce_axis_f32_le(
    body: &[u8],
    shape: &[usize],
    axis: usize,
    reducer: &Reducer,
    valid: Option<&AxisBoundsU32>,
) -> Vec<Option<f64>> {
    let (d0, d1, d2) = (shape[0], shape[1], shape[2]);
    let n = shape[axis];
    let mut min = vec![f64::INFINITY; n];
    let mut max = vec![f64::NEG_INFINITY; n];
    let mut sum = vec![0.0f64; n];
    let mut sum_sq = vec![0.0f64; n];
    let mut cnt = vec![0u64; n];

    for i in 0..d0 {
        for j in 0..d1 {
            let base = (i * d1 + j) * d2;
            for k in 0..d2 {
                if valid.is_some_and(|range| {
                    i < range.lower[0] as usize
                        || i > range.upper[0] as usize
                        || j < range.lower[1] as usize
                        || j > range.upper[1] as usize
                        || k < range.lower[2] as usize
                        || k > range.upper[2] as usize
                }) {
                    continue;
                }
                let off = (base + k) * 4;
                let v = f32::from_le_bytes([body[off], body[off + 1], body[off + 2], body[off + 3]])
                    as f64;
                if !v.is_finite() {
                    continue;
                }
                let a = match axis {
                    0 => i,
                    1 => j,
                    _ => k,
                };
                if v < min[a] {
                    min[a] = v;
                }
                if v > max[a] {
                    max[a] = v;
                }
                sum[a] += v;
                sum_sq[a] += v * v;
                cnt[a] += 1;
            }
        }
    }

    (0..n)
        .map(|a| {
            if cnt[a] == 0 {
                None
            } else {
                let c = cnt[a] as f64;
                Some(match reducer {
                    Reducer::Max => max[a],
                    Reducer::Min => min[a],
                    Reducer::Mean => sum[a] / c,
                    Reducer::Rms => (sum_sq[a] / c).sqrt(),
                })
            }
        })
        .collect()
}

// Axis reduction shares the same bounded terrain-geometry read contract.
#[allow(clippy::too_many_arguments)]
fn reduce_axis_terrain_f32_le(
    body: &[u8],
    shape: &[usize],
    axis: usize,
    reducer: &Reducer,
    descriptor: &TerrainWindowDescriptor,
    run_dir: &Path,
    step_index: u32,
    netcdf_path: Option<&Path>,
    valid: Option<&AxisBoundsU32>,
) -> AppResult<Vec<Option<f64>>> {
    validate_terrain_reduction_inputs(body, shape, descriptor)?;
    let axis_len = shape[axis];
    let mut min = vec![f64::INFINITY; axis_len];
    let mut max = vec![f64::NEG_INFINITY; axis_len];
    let mut weighted_sum = vec![0_f64; axis_len];
    let mut weighted_sum_sq = vec![0_f64; axis_len];
    let mut finite_volume = vec![0_f64; axis_len];
    let mut finite = vec![0_u64; axis_len];
    let slab_x = terrain_reduction_slab_x(shape)?;
    for i_begin in (0..shape[0]).step_by(slab_x) {
        let i_count = slab_x.min(shape[0] - i_begin);
        let interfaces = read_terrain_interface_slab(
            descriptor,
            run_dir,
            step_index,
            netcdf_path,
            i_begin,
            i_count,
        )?;
        let interface_nz = shape[2] + 1;
        for local_i in 0..i_count {
            let i = i_begin + local_i;
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let face = (local_i * shape[1] + j) * interface_nz + k;
                    let lower = interfaces[face];
                    let upper = interfaces[face + 1];
                    if !lower.is_finite() || !upper.is_finite() || upper <= lower {
                        return Err(AppError::Invalid(format!(
                            "malformed terrain interface geometry at ({i},{j},{k})"
                        )));
                    }
                    if valid.is_some_and(|range| {
                        i < range.lower[0] as usize
                            || i > range.upper[0] as usize
                            || j < range.lower[1] as usize
                            || j > range.upper[1] as usize
                            || k < range.lower[2] as usize
                            || k > range.upper[2] as usize
                    }) {
                        continue;
                    }
                    let linear = (i * shape[1] + j) * shape[2] + k;
                    let value = terrain_field_value(body, linear);
                    if !value.is_finite() {
                        continue;
                    }
                    let volume = descriptor.horizontal_spacing_m[0]
                        * descriptor.horizontal_spacing_m[1]
                        * (upper - lower);
                    if !volume.is_finite() || volume <= 0.0 {
                        return Err(AppError::Invalid(format!(
                            "non-positive terrain volume at ({i},{j},{k})"
                        )));
                    }
                    let index = [i, j, k][axis];
                    min[index] = min[index].min(value);
                    max[index] = max[index].max(value);
                    weighted_sum[index] += value * volume;
                    weighted_sum_sq[index] += value * value * volume;
                    finite_volume[index] += volume;
                    finite[index] += 1;
                }
            }
        }
    }
    (0..axis_len)
        .map(|index| {
            if finite[index] == 0 {
                return Ok(None);
            }
            let volume = finite_volume[index];
            if !volume.is_finite() || volume <= 0.0 {
                return Err(AppError::Invalid(
                    "terrain axis reduction produced invalid finite volume".into(),
                ));
            }
            Ok(Some(match reducer {
                Reducer::Max => max[index],
                Reducer::Min => min[index],
                Reducer::Mean => weighted_sum[index] / volume,
                Reducer::Rms => (weighted_sum_sq[index] / volume).sqrt(),
            }))
        })
        .collect()
}

#[tauri::command]
pub async fn reduce_field_axis(
    run_dir: String,
    step_index: u32,
    field_name: String,
    axis: u8,
    reducer: String,
    expected_generation: Option<String>,
) -> AppResult<AxisReduction> {
    // Read + per-axis O(N) reduce off the main thread (one per step for the
    // time-height plot), same rationale as reduce_field.
    tauri::async_runtime::spawn_blocking(move || -> AppResult<AxisReduction> {
        let _reduction_slot = FIELD_REDUCTION_LOCK
            .lock()
            .map_err(|_| AppError::Task("field reduction lock poisoned".into()))?;
        let expected_grid = expected_manifest_grid_shape(
            Path::new(&run_dir),
            step_index,
            &field_name,
            expected_generation.as_deref(),
        )?;
        let terrain = terrain_window_descriptor(
            Path::new(&run_dir),
            step_index,
            expected_generation.as_deref(),
        )?;
        let valid_range = expected_manifest_valid_index_range(
            Path::new(&run_dir),
            step_index,
            expected_generation.as_deref(),
        )?;
        let netcdf = netcdf_field_mapping(&run_dir, step_index, &field_name)?;
        let path = if netcdf.is_none() {
            Some(field_path(&run_dir, step_index, &field_name)?)
        } else {
            None
        };
        let a = axis as usize;
        if a >= 3 {
            return Err(AppError::Invalid(format!(
                "axis {axis} out of range (expected 0..3)"
            )));
        }
        let red = parse_reducer(&reducer)?;

        let (shape, values) = if let Some((netcdf_path, variable)) = netcdf.as_ref() {
            let (shape, data) = read_netcdf_field(netcdf_path, variable)?;
            validate_manifest_shape(&shape, expected_grid)?;
            if shape.len() != 3 {
                return Err(AppError::Invalid(format!(
                    "reduce_field_axis needs a 3D field, got {}D",
                    shape.len()
                )));
            }
            let values = if let Some(descriptor) = terrain.as_ref() {
                reduce_axis_terrain_f32_le(
                    bytemuck::cast_slice(&data),
                    &shape,
                    a,
                    &red,
                    descriptor,
                    Path::new(&run_dir),
                    step_index,
                    Some(netcdf_path),
                    valid_range.as_ref(),
                )?
            } else {
                reduce_axis_f32_le(
                    bytemuck::cast_slice(&data),
                    &shape,
                    a,
                    &red,
                    valid_range.as_ref(),
                )
            };
            (shape, values)
        } else if is_zfp3d(path.as_ref().unwrap()) {
            if let Some(descriptor) = terrain.as_ref() {
                let (shape, data) =
                    crate::zfp3d::reconstruct_field(&run_dir, step_index, &field_name)?;
                validate_manifest_shape(&shape, expected_grid)?;
                let values = reduce_axis_terrain_f32_le(
                    bytemuck::cast_slice(&data),
                    &shape,
                    a,
                    &red,
                    descriptor,
                    Path::new(&run_dir),
                    step_index,
                    None,
                    valid_range.as_ref(),
                )?;
                (shape, values)
            } else {
                let (shape, values) = reduce_zfp_axis_bricks(
                    &run_dir,
                    step_index,
                    &field_name,
                    a,
                    &red,
                    valid_range.as_ref(),
                )?;
                validate_manifest_shape(&shape, expected_grid)?;
                (shape, values)
            }
        } else {
            if let Some(descriptor) = terrain.as_ref() {
                let bytes = crate::npy::read_f32_file(path.as_ref().unwrap())?;
                let layout = crate::npy::parse_f32_layout(&bytes)?;
                validate_manifest_shape(&layout.shape, expected_grid)?;
                if layout.shape.len() != 3 {
                    return Err(AppError::Invalid(format!(
                        "reduce_field_axis needs a 3D field, got {}D",
                        layout.shape.len()
                    )));
                }
                let body = &bytes[layout.data_offset..layout.data_offset + layout.decoded_bytes];
                let values = reduce_axis_terrain_f32_le(
                    body,
                    &layout.shape,
                    a,
                    &red,
                    descriptor,
                    Path::new(&run_dir),
                    step_index,
                    None,
                    valid_range.as_ref(),
                )?;
                (layout.shape, values)
            } else {
                let (shape, values) =
                    reduce_npy_axis_stream(path.as_ref().unwrap(), a, &red, valid_range.as_ref())?;
                validate_manifest_shape(&shape, expected_grid)?;
                (shape, values)
            }
        };

        let reduction = AxisReduction {
            values,
            axis,
            axis_len: shape[a],
        };
        ensure_expected_publication_generation(
            Path::new(&run_dir),
            step_index,
            expected_generation.as_deref(),
        )?;
        Ok(reduction)
    })
    .await
    .map_err(|e| AppError::Task(e.to_string()))?
}

fn parse_grid_info(v: &serde_json::Value) -> GridInfo {
    GridInfo {
        nr: grid_dimension(v, "x", "nr").map(|x| x as u32),
        nth: grid_dimension(v, "y", "nth").map(|x| x as u32),
        nz: grid_dimension(v, "z", "nz").map(|x| x as u32),
        coordinate_system: v
            .get("coordinate_system")
            .and_then(|x| x.as_str())
            .map(str::to_owned),
        staggering: v
            .get("native_model_staggering")
            .or_else(|| v.get("staggering"))
            .and_then(|x| x.as_str())
            .map(str::to_owned),
        dr_m: grid_spacing(v, "x", "dr_m"),
        dy_m: grid_spacing(v, "y", "dy_m"),
        dtheta_rad: v.get("dtheta_rad").and_then(|x| x.as_f64()),
        dz_m: grid_spacing(v, "z", "dz_m"),
        terrain_following: v
            .get("vertical_geometry")
            .and_then(|geometry| geometry.get("kind"))
            .and_then(|kind| kind.as_str())
            == Some("terrain_following"),
        terrain_geometry_identity: v
            .get("vertical_geometry")
            .and_then(|geometry| geometry.get("content_sha256"))
            .and_then(|identity| identity.as_str())
            .map(str::to_owned),
        horizontal_scalar_origin_m: parse_f64_pair(v.get("horizontal_scalar_origin_m")),
        scalar_origin_m: parse_f64_triplet(v.get("scalar_origin_m")),
        physical_bounds_m: parse_f64_bounds(v.get("physical_bounds_m")),
        valid_index_range: parse_u32_bounds(v.get("valid_index_range")),
        halo_width: parse_u32_bounds(v.get("halo_width")),
    }
}

/// Read `grid.horizontal_frame` into the step-level motion contract.
///
/// Returns `None` when the block is absent, which means a pre-contract manifest;
/// the UI treats absence as stationary, which is what those runs were. A block
/// that IS present is passed through verbatim, including a mode string this
/// build does not recognise: resolving the vocabulary is the UI's job, and it
/// refuses unknown modes rather than assuming stationary.
fn parse_frame_motion(grid: &serde_json::Value) -> Option<FrameMotion> {
    let frame = grid.get("horizontal_frame")?;
    Some(FrameMotion {
        horizontal_frame_contract: frame
            .get("contract")
            .and_then(|x| x.as_str())
            .map(str::to_owned),
        mode: frame
            .get("motion_mode")
            .and_then(|x| x.as_str())
            .map(str::to_owned),
        frame_velocity_ms: parse_f64_pair(frame.get("frame_velocity_earth_ms")),
        earth_origin_m: parse_f64_pair(frame.get("grid_origin_earth_m")),
        grid_origin_t0_earth_m: parse_f64_pair(frame.get("grid_origin_t0_earth_m")),
        // The manifest names the published wind frame explicitly; anything other
        // than an explicit earth_relative declaration is not Earth-relative.
        winds_earth_relative: frame
            .get("published_horizontal_wind")
            .and_then(|x| x.as_str())
            .map(|declared| declared == "earth_relative"),
    })
}

/// Build a `field_id` -> units map from a manifest's `fields` catalog. Entries
/// missing `field_id` or `units`, or carrying an empty unit string, are skipped
/// so the UI falls back to an unsuffixed label rather than `field ()`. The
/// dimensionless marker `"1"` is preserved here and suppressed UI-side.
fn parse_field_units(manifest: &serde_json::Value) -> std::collections::HashMap<String, String> {
    let mut out = std::collections::HashMap::new();
    let Some(fields) = manifest.get("fields").and_then(|f| f.as_array()) else {
        return out;
    };
    for entry in fields {
        let id = entry.get("field_id").and_then(|v| v.as_str());
        let units = entry.get("units").and_then(|v| v.as_str());
        if let (Some(id), Some(units)) = (id, units) {
            if !units.is_empty() {
                out.insert(id.to_string(), units.to_string());
            }
        }
    }
    out
}

fn parse_field_compression(
    manifest: &serde_json::Value,
) -> std::collections::HashMap<String, FieldCompressionInfo> {
    let mut out = std::collections::HashMap::new();
    let Some(fields) = manifest.get("fields").and_then(|value| value.as_array()) else {
        return out;
    };
    for entry in fields {
        let Some(id) = entry.get("field_id").and_then(|value| value.as_str()) else {
            continue;
        };
        let Some(compression) = entry.get("compression") else {
            continue;
        };
        let parsed = (
            compression.get("codec").and_then(|value| value.as_str()),
            compression.get("mode").and_then(|value| value.as_str()),
            compression
                .get("mode_parameter")
                .and_then(|value| value.as_f64()),
            compression
                .get("mode_parameter_units")
                .and_then(|value| value.as_str()),
            compression
                .get("error_contract")
                .and_then(|value| value.as_str()),
            compression
                .get("sparse_threshold")
                .and_then(|value| value.as_f64()),
            compression
                .get("float16_prefilter")
                .and_then(|value| value.as_bool()),
            compression
                .get("decoded_dtype")
                .and_then(|value| value.as_str()),
        );
        if let (
            Some(codec),
            Some(mode),
            Some(mode_parameter),
            Some(mode_parameter_units),
            Some(error_contract),
            Some(sparse_threshold),
            Some(float16_prefilter),
            Some(decoded_dtype),
        ) = parsed
        {
            if mode_parameter.is_finite() && sparse_threshold.is_finite() {
                out.insert(
                    id.to_string(),
                    FieldCompressionInfo {
                        codec: codec.to_string(),
                        mode: mode.to_string(),
                        mode_parameter,
                        mode_parameter_units: mode_parameter_units.to_string(),
                        error_contract: error_contract.to_string(),
                        sparse_threshold,
                        float16_prefilter,
                        decoded_dtype: decoded_dtype.to_string(),
                    },
                );
            }
        }
    }
    out
}

fn parse_field_brick_hierarchies(
    manifest: &serde_json::Value,
) -> std::collections::HashMap<String, FieldBrickHierarchyInfo> {
    let mut out = std::collections::HashMap::new();
    let Some(fields) = manifest.get("fields").and_then(|value| value.as_array()) else {
        return out;
    };
    for entry in fields {
        let Some(id) = entry.get("field_id").and_then(|value| value.as_str()) else {
            continue;
        };
        let Some(hierarchy) = entry.get("brick_hierarchy") else {
            continue;
        };
        let Some(path) = hierarchy.get("path").and_then(|value| value.as_str()) else {
            continue;
        };
        let shape = parse_u32_triplet(hierarchy.get("native_brick_shape"));
        let content_sha256 = hierarchy.get("content_sha256").and_then(|value| value.as_str());
        let schema_valid = hierarchy.get("schema").and_then(|value| value.as_str())
            == Some("nimbus.field-brick-hierarchy");
        let version = hierarchy.get("schema_version").and_then(|value| value.as_u64());
        let extrema = hierarchy.get("extrema_semantics").and_then(|value| value.as_str());
        let extrema_padding = hierarchy
            .get("extrema_absolute_padding")
            .and_then(|value| value.as_f64());
        let culling_safe = hierarchy.get("culling_safe").and_then(|value| value.as_bool());
        let expected_path = format!("{id}.bricks.json");
        if !schema_valid || !matches!(version, Some(1 | 2 | 3)) || path != expected_path
            || path.contains('/') || path.contains('\\') || path.contains("..")
            || !content_sha256.is_some_and(|value| {
                value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
            || shape.is_none()
            || !matches!(extrema, Some("stored_values" | "decoded_value_bounds" | "pre_storage_source"))
            || !extrema_padding.is_some_and(|value| value.is_finite() && value >= 0.0)
            || culling_safe.is_none()
            || (culling_safe == Some(true) && extrema == Some("pre_storage_source"))
            || (culling_safe == Some(false) && extrema != Some("pre_storage_source"))
            || (matches!(extrema, Some("stored_values" | "pre_storage_source"))
                && extrema_padding != Some(0.0)) {
            continue;
        }
        out.insert(id.to_string(), FieldBrickHierarchyInfo {
            path: path.to_string(),
            content_sha256: content_sha256.unwrap().to_ascii_lowercase(),
            schema_version: version.unwrap() as u32,
            native_brick_shape: shape.unwrap(),
            extrema_semantics: extrema.unwrap().to_string(),
            extrema_absolute_padding: extrema_padding.unwrap(),
            culling_safe: culling_safe.unwrap(),
        });
    }
    out
}

fn parse_compression_execution(
    manifest: &serde_json::Value,
) -> Option<CompressionExecutionInfo> {
    let product = manifest.get("data_product")?;
    let legacy = product.get("lossy_compression_contract").and_then(|value| value.as_str());
    let mixed = product.get("compression_contract").and_then(|value| value.as_str());
    if !matches!((legacy, mixed),
        (Some("nimbus.zfp.per_field.v1"), None) |
        (None, Some("nimbus.zfp.per_field.v2"))) {
        return None;
    }
    let policy = product.get("field_execution_policy")?.as_str()?;
    let workers = product.get("field_workers")?.as_u64()?;
    let scratch = product.get("field_scratch_budget_mib")?.as_u64()?;
    if !matches!(policy, "serial" | "field_parallel") ||
        !(1..=16).contains(&workers) || !(1..=4096).contains(&scratch)
    {
        return None;
    }
    Some(CompressionExecutionInfo {
        field_execution_policy: policy.to_string(),
        field_workers: workers as u32,
        field_scratch_budget_mib: scratch as u32,
    })
}

fn is_selected_radar_zfp_field(field: &str) -> bool {
    matches!(field, "reflectivity_dbz" | "radial_velocity" | "zdr" | "kdp" | "rhohv")
}

fn zfp_v1_field_storage_is_complete(
    manifest: &serde_json::Value,
    fields: &[String],
    step_dir: &Path,
) -> bool {
    let Some(product) = manifest.get("data_product") else { return false };
    if product.get("storage_format").and_then(|value| value.as_str()) != Some("zfp")
        || product.get("lossy_compression_contract").and_then(|value| value.as_str())
            != Some("nimbus.zfp.per_field.v1")
        || product.get("compression_contract").is_some()
        || parse_compression_execution(manifest).is_none()
    {
        return false;
    }
    let Some(catalog) = manifest.get("fields").and_then(|value| value.as_array()) else {
        return false;
    };
    let published: std::collections::BTreeSet<&str> = fields.iter().map(String::as_str).collect();
    if published.is_empty() || published.len() != fields.len() {
        return false;
    }
    let parsed_compression = parse_field_compression(manifest);
    let mut compressed = 0usize;
    for field in fields {
        if field.is_empty() || !field.bytes().all(|byte|
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_') {
            return false;
        }
        let mut entries = catalog.iter().filter(|entry|
            entry.get("field_id").and_then(|value| value.as_str()) == Some(field));
        let Some(entry) = entries.next() else { return false };
        if entries.next().is_some()
            || entry.get("file_suffix").and_then(|value| value.as_str()) != Some(field) {
            return false;
        }
        let npy = step_dir.join(format!("{field}.npy"));
        let zfp = step_dir.join(format!("{field}.zfp3d"));
        let regular = |path: &Path| fs::symlink_metadata(path).ok()
            .is_some_and(|metadata| metadata.file_type().is_file() && metadata.len() > 0);
        if let Some(compression) = entry.get("compression") {
            if entry.get("file_pattern").and_then(|value| value.as_str())
                != Some(format!("{field}.zfp3d").as_str())
                || entry.get("storage_format").is_some()
                || !regular(&zfp) || fs::symlink_metadata(&npy).is_ok()
                || !parsed_compression.contains_key(field)
                || compression.get("codec").and_then(|value| value.as_str()) != Some("zfp")
                || compression.get("mode").and_then(|value| value.as_str()) != Some("accuracy")
                || !compression.get("mode_parameter").and_then(|value| value.as_f64())
                    .is_some_and(|value| value.is_finite() && value > 0.0)
                || compression.get("mode_parameter_units").and_then(|value| value.as_str())
                    != Some("absolute_field_units")
                || compression.get("error_contract").and_then(|value| value.as_str())
                    != Some("zfp_absolute_accuracy_before_prefilters")
                || compression.get("sparse_threshold").and_then(|value| value.as_f64())
                    != Some(0.0)
                || compression.get("float16_prefilter").and_then(|value| value.as_bool())
                    != Some(false)
                || compression.get("decoded_dtype").and_then(|value| value.as_str())
                    != Some("float32")
                || compression.get("container_version").and_then(|value| value.as_u64())
                    != Some(3)
                || compression.get("spatial_layout").and_then(|value| value.as_str())
                    != Some("independently_decodable_bricks")
                || compression.get("brick_shape") != Some(&serde_json::json!([64, 64, 32]))
            {
                return false;
            }
            let Ok(mut file) = fs::File::open(&zfp) else { return false };
            let mut header = [0u8; 44];
            if file.read_exact(&mut header).is_err() {
                return false;
            }
            let Ok((header, _)) = crate::zfp3d::parse_header(&header) else { return false };
            if header.version != 3 || header.mode != 0 {
                return false;
            }
            compressed += 1;
        } else {
            if entry.get("file_pattern").and_then(|value| value.as_str())
                != Some(format!("{field}.npy").as_str())
                || entry.get("storage_format").and_then(|value| value.as_str())
                    != Some("npy_2d")
                || entry.get("missing_value").and_then(|value| value.as_str()) != Some("NaN")
                || !regular(&npy) || fs::symlink_metadata(&zfp).is_ok()
                || parsed_compression.contains_key(field)
            {
                return false;
            }
            let Ok(mut file) = fs::File::open(&npy) else { return false };
            let Ok((shape, offset)) = read_npy_header(&mut file) else { return false };
            let Some(expected_size) = shape.iter().try_fold(1usize, |size, dim|
                size.checked_mul(*dim)).and_then(|size| size.checked_mul(4))
                .and_then(|size| size.checked_add(offset)) else { return false };
            if shape.len() != 2 || fs::metadata(&npy).ok().map(|value| value.len())
                != u64::try_from(expected_size).ok() {
                return false;
            }
        }
    }
    compressed > 0 && !catalog.iter().any(|entry| {
        entry.get("field_id").and_then(|value| value.as_str())
            .is_some_and(|field| !published.contains(field))
            && (entry.get("compression").is_some() || entry.get("storage_format").is_some())
    })
}

fn json_string_token_end(bytes: &[u8], start: usize) -> Option<usize> {
    if bytes.get(start) != Some(&b'"') {
        return None;
    }
    let mut end = start + 1;
    while end < bytes.len() {
        if bytes[end] == b'\\' {
            end = end.checked_add(2)?;
        } else if bytes[end] == b'"' {
            return Some(end + 1);
        } else {
            end += 1;
        }
    }
    None
}

fn exact_mixed_v2_mode_parameters(
    manifest: &serde_json::Value,
    step_dir: &Path,
) -> Option<HashMap<String, f64>> {
    let path = step_dir.join("manifest.json");
    let metadata = fs::symlink_metadata(&path).ok()?;
    if !metadata.file_type().is_file() || metadata.len() > 1024 * 1024 {
        return None;
    }
    let bytes = fs::read(path).ok()?;
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    if parsed != *manifest {
        return None;
    }
    let mut parameters = HashMap::new();
    let mut brace_depth = 0usize;
    let mut bracket_depth = 0usize;
    let mut fields_seen = false;
    let mut fields_array_depth: Option<usize> = None;
    let mut entry_depth: Option<usize> = None;
    let mut compression_depth: Option<usize> = None;
    let mut compression_pending = false;
    let mut compression_seen = false;
    let mut current_field: Option<String> = None;
    let mut current_parameter: Option<f64> = None;
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'{' => {
                brace_depth += 1;
                if fields_array_depth == Some(bracket_depth) && brace_depth == 2 {
                    if entry_depth.is_some() {
                        return None;
                    }
                    entry_depth = Some(brace_depth);
                    compression_seen = false;
                    current_field = None;
                    current_parameter = None;
                } else if compression_pending {
                    if entry_depth.map_or(true, |depth| brace_depth != depth + 1) {
                        return None;
                    }
                    compression_depth = Some(brace_depth);
                    compression_pending = false;
                }
                index += 1;
            }
            b'}' => {
                if compression_depth == Some(brace_depth) {
                    compression_depth = None;
                }
                if entry_depth == Some(brace_depth) {
                    if compression_pending || compression_depth.is_some() {
                        return None;
                    }
                    let field = current_field.take()?;
                    if compression_seen {
                        let parameter = current_parameter.take()?;
                        if parameters.insert(field, parameter).is_some() {
                            return None;
                        }
                    } else if current_parameter.is_some() {
                        return None;
                    }
                    entry_depth = None;
                }
                brace_depth = brace_depth.checked_sub(1)?;
                index += 1;
            }
            b'[' => {
                bracket_depth += 1;
                index += 1;
            }
            b']' => {
                if fields_array_depth == Some(bracket_depth) {
                    fields_array_depth = None;
                }
                bracket_depth = bracket_depth.checked_sub(1)?;
                index += 1;
            }
            b'"' => {
                let start = index;
                let end = json_string_token_end(&bytes, start)?;
                let key: String = serde_json::from_slice(&bytes[start..end]).ok()?;
                index = end;
                let mut value_start = end;
                while bytes.get(value_start).is_some_and(u8::is_ascii_whitespace) {
                    value_start += 1;
                }
                if bytes.get(value_start) != Some(&b':') {
                    continue;
                }
                value_start += 1;
                while bytes.get(value_start).is_some_and(u8::is_ascii_whitespace) {
                    value_start += 1;
                }
                if key == "fields" && brace_depth == 1 && bracket_depth == 0 {
                    if fields_seen || &bytes[start..end] != b"\"fields\""
                        || bytes.get(value_start) != Some(&b'[') {
                        return None;
                    }
                    fields_seen = true;
                    fields_array_depth = Some(bracket_depth + 1);
                }
                let Some(field_depth) = entry_depth else { continue };
                if key == "field_id" {
                    if brace_depth != field_depth || current_field.is_some()
                        || &bytes[start..end] != b"\"field_id\"" {
                        return None;
                    }
                    let value_end = json_string_token_end(&bytes, value_start)?;
                    current_field = Some(
                        serde_json::from_slice(&bytes[value_start..value_end]).ok()?);
                    index = value_end;
                } else if key == "compression" {
                    if brace_depth != field_depth || compression_seen
                        || &bytes[start..end] != b"\"compression\""
                        || bytes.get(value_start) != Some(&b'{') {
                        return None;
                    }
                    compression_seen = true;
                    compression_pending = true;
                } else if key == "mode_parameter" {
                    if compression_depth != Some(brace_depth) || current_parameter.is_some()
                        || &bytes[start..end] != b"\"mode_parameter\"" {
                        return None;
                    }
                    let mut value_end = value_start;
                    while bytes.get(value_end).is_some_and(|byte|
                        byte.is_ascii_digit() ||
                        matches!(*byte, b'-' | b'+' | b'.' | b'e' | b'E')) {
                        value_end += 1;
                    }
                    let token = std::str::from_utf8(bytes.get(value_start..value_end)?).ok()?;
                    if !serde_json::from_str::<serde_json::Value>(token).ok()?.is_number() {
                        return None;
                    }
                    let parameter = token.parse::<f64>().ok()?;
                    if !parameter.is_finite() {
                        return None;
                    }
                    current_parameter = Some(parameter);
                    index = value_end;
                }
            }
            _ => {
                index += 1;
            }
        }
    }
    let fields = manifest.get("fields")?.as_array()?;
    if !fields_seen || fields_array_depth.is_some() || entry_depth.is_some()
        || parameters.len() != fields.iter().filter(|entry|
            entry.get("compression").is_some()).count()
        || fields.iter().any(|entry| {
            let Some(field) = entry.get("field_id").and_then(|value| value.as_str()) else {
                return true;
            };
            entry.get("compression").is_some() != parameters.contains_key(field)
    }) {
        return None;
    }
    Some(parameters)
}

fn mixed_zfp_v2_field_contract(
    field: &str,
    compression: &serde_json::Value,
    step_index: u32,
    parameter: f64,
) -> Option<(u16, u16, u8, f64)> {
    let brick_shape = compression.get("brick_shape")?.as_array()?.iter()
        .map(|value| value.as_u64()).collect::<Option<Vec<_>>>()?;
    if compression.get("codec")?.as_str()? != "zfp"
        || compression.get("decoded_dtype")?.as_str()? != "float32"
        || compression.get("spatial_layout")?.as_str()? != "independently_decodable_bricks"
        || brick_shape.as_slice() != [64, 64, 32]
        || compression.get("sparse_threshold")?.as_f64()? != 0.0
        || compression.get("float16_prefilter")?.as_bool()?
    {
        return None;
    }
    let role = compression.get("frame_role")?.as_str()?;
    let parents = compression.get("parent_step_indices")?.as_array()?;
    let expected_parents = match role {
        "keyframe" => (0, 0),
        "simple_delta" => (1, 1),
        "predictive_delta" => (2, 3),
        _ => return None,
    };
    if parents.len() != expected_parents.0
        || parents.iter().any(|value| {
            value
                .as_u64()
                .map_or(true, |parent| parent >= step_index as u64)
        })
        || (parents.len() == 2 && parents[0].as_u64()? <= parents[1].as_u64()?)
    {
        return None;
    }
    let mode = compression.get("mode")?.as_str()?;
    compression.get("mode_parameter")?.as_f64()?;
    let container = compression.get("container_version")?.as_u64()?;
    let contract_version = compression.get("contract_version");
    if is_selected_radar_zfp_field(field) {
        if contract_version?.as_u64()? != 2
            || mode != "reversible"
            || parameter.to_bits() != 0
            || compression.get("mode_parameter_units")?.as_str()? != "none"
            || compression.get("error_contract")?.as_str()? != "bitwise_float32_reconstruction"
            || container != 4 || role != "keyframe" || !parents.is_empty()
        {
            return None;
        }
        Some((4, 3, 0, parameter))
    } else {
        if contract_version.is_some()
            || mode != "accuracy"
            || !parameter.is_finite() || parameter <= 0.0
            || compression.get("mode_parameter_units")?.as_str()? != "absolute_field_units"
            || compression.get("error_contract")?.as_str()? != "zfp_absolute_accuracy_before_prefilters"
            || container != 3
        {
            return None;
        }
        Some((3, 0, expected_parents.1, parameter))
    }
}

fn mixed_zfp_v2_artifact_header_matches(
    path: &Path,
    version: u16,
    mode: u16,
    flags: u8,
    parameter: f64,
) -> bool {
    let Ok(metadata) = fs::symlink_metadata(path) else { return false };
    if !metadata.file_type().is_file() || metadata.len() < 44 { return false; }
    let Ok(mut file) = fs::File::open(path) else { return false };
    let mut header_bytes = [0u8; 44];
    if file.read_exact(&mut header_bytes).is_err() { return false; }
    let Ok((header, _)) = crate::zfp3d::parse_header(&header_bytes) else { return false };
    if header.version != version || header.mode != mode || header.flags != flags
        || header.mode_param.to_bits() != parameter.to_bits()
    {
        return false;
    }
    let shape = header.shape();
    let Some(bricks) = shape.iter().zip([64usize, 64, 32]).try_fold(1usize,
        |count, (extent, brick)| count.checked_mul(extent.div_ceil(brick))) else {
        return false;
    };
    let entry_bytes = if version == 4 { 40usize } else { 8usize };
    let Some(expected) = header.compressed_size.checked_add(44)
        .and_then(|value| value.checked_add(bricks.checked_mul(entry_bytes)?)) else {
        return false;
    };
    u64::try_from(expected).ok() == Some(metadata.len())
}

fn mixed_zfp_v2_is_complete(
    manifest: &serde_json::Value,
    fields: &[String],
    step_index: u32,
    step_dir: &Path,
) -> bool {
    let Some(product) = manifest.get("data_product") else { return false };
    if product.get("compression_contract").and_then(|value| value.as_str())
        != Some("nimbus.zfp.per_field.v2")
        || product.get("lossy_compression_contract").is_some()
        || product.get("storage_format").and_then(|value| value.as_str()) != Some("zfp")
        || parse_compression_execution(manifest).is_none()
    {
        return false;
    }
    let Some(catalog) = manifest.get("fields").and_then(|value| value.as_array()) else {
        return false;
    };
    let Some(exact_parameters) = exact_mixed_v2_mode_parameters(manifest, step_dir) else {
        return false;
    };
    let Some(products) = manifest.get("radar_provenance")
        .and_then(|value| value.get("products"))
        .and_then(|value| value.as_array()) else { return false };
    let selected_products: std::collections::BTreeSet<&str> = products.iter()
        .filter_map(|value| value.as_str())
        .collect();
    if selected_products.is_empty() || selected_products.len() != products.len()
        || selected_products.iter().any(|field| !is_selected_radar_zfp_field(field))
    {
        return false;
    }
    let mut reversible = 0usize;
    let mut accuracy = 0usize;
    for field in fields {
        let mut entries = catalog.iter().filter(|entry| {
            entry.get("field_id").and_then(|value| value.as_str()) == Some(field)
        });
        let Some(entry) = entries.next() else { return false };
        if entries.next().is_some() { return false; }
        let Some(compression) = entry.get("compression") else { return false };
        let Some(parameter) = exact_parameters.get(field).copied() else { return false };
        let Some((version, mode, flags, parameter)) =
            mixed_zfp_v2_field_contract(field, compression, step_index, parameter) else { return false };
        if version == 4 { reversible += 1; } else { accuracy += 1; }
        let path = step_dir.join(format!("{field}.zfp3d"));
        if !mixed_zfp_v2_artifact_header_matches(&path, version, mode, flags, parameter) {
            return false;
        }
    }
    if reversible == 0 || accuracy == 0 {
        return false;
    }
    let published_radar: std::collections::BTreeSet<&str> = fields.iter()
        .map(String::as_str)
        .filter(|field| is_selected_radar_zfp_field(field))
        .collect();
    if published_radar != selected_products {
        return false;
    }
    !catalog.iter().any(|entry| {
        entry.get("compression").is_some()
            && !fields.iter().any(|field| {
                entry.get("field_id").and_then(|value| value.as_str()) == Some(field)
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry_test_request(request_id: &str) -> FieldWindowRequest {
        FieldWindowRequest {
            schema_version: FIELD_WINDOW_SCHEMA_VERSION,
            request_id: request_id.to_owned(),
            aggregation: "cell_mean".into(),
            origin: vec![0],
            extent: vec![1],
            stride: vec![1],
            max_output_bytes: Some(4),
            max_source_bytes: Some(4),
        }
    }

    /// Regression: the slot was released after the await, so a dropped command
    /// future leaked it until MAX_ACTIVE_FIELD_READS wedged every later read.
    #[test]
    fn field_read_guard_releases_slot_on_drop() {
        let request_id = "test:guard-release";
        let request = registry_test_request(request_id);
        let guard = register_field_read(&request).expect("first registration");
        assert!(
            register_field_read(&request).is_err(),
            "an active id must not be re-registered"
        );

        drop(guard);
        let reacquired = register_field_read(&request).expect("drop should have released the slot");
        drop(reacquired);
    }

    /// Dropping the guard must not leave the id cancellable, or a later read
    /// reusing it would start pre-cancelled.
    #[test]
    fn dropped_guard_leaves_no_cancellable_entry() {
        let request_id = "test:guard-cancel";
        drop(register_field_read(&registry_test_request(request_id)).expect("registration"));

        let receipt = cancel_field_read(request_id.to_string()).expect("cancel should succeed");
        assert!(
            !receipt.found,
            "a released id must not stay in the registry"
        );
        assert!(!receipt.cancellation_requested);
    }

    const GENERATION_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const GENERATION_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    struct TestRunDir(PathBuf);

    impl TestRunDir {
        fn new(label: &str) -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let serial = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "nimbus_outputs_{label}_{}_{}",
                std::process::id(),
                serial
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn current(&self) {
            self.current_generation(GENERATION_A, "ready");
        }

        fn current_indexed(&self, generation: &str) {
            fs::write(
                self.0.join(RUN_OUTPUT_CONTRACT_FILE),
                format!(
                    r#"{{"schema":"nimbus.output","manifest_version":{OUTPUT_MANIFEST_VERSION},"publication_generation":"{generation}","publication_state":"ready","frame_completion":"atomic_step_manifest","history_index":{{"schema":"nimbus.history.index","version":1,"checksum":"sha256"}}}}"#
                ),
            )
            .unwrap();
        }

        fn current_generation(&self, generation: &str, state: &str) {
            self.current_generation_version(generation, state, LEGACY_OUTPUT_MANIFEST_VERSION);
        }

        fn current_generation_version(&self, generation: &str, state: &str, version: u64) {
            fs::write(
                self.0.join(RUN_OUTPUT_CONTRACT_FILE),
                format!(
                    r#"{{
                    "schema":"nimbus.output",
                    "manifest_version":{version},
                    "publication_generation":"{generation}",
                    "publication_state":"{state}",
                    "frame_completion":"atomic_step_manifest"
                }}"#
                ),
            )
            .unwrap();
        }

        fn step_with_field(&self, index: u32) -> PathBuf {
            let path = self.0.join(format!("step_{index:06}"));
            fs::create_dir_all(&path).unwrap();
            fs::write(path.join("theta.npy"), b"non-empty-test-payload").unwrap();
            path
        }
    }

    impl Drop for TestRunDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn current_manifest(index: u32, generation: &str) -> String {
        format!(
            r#"{{
                "schema":"nimbus.step",
                "manifest_version":2,
                "publication_state":"complete",
                "publication_generation":"{generation}",
                "step_index":{index},
                "simulation_time_s":2.5,
                "step_dir":"step_{index:06}",
                "grid":{{
                    "nr":8,"nth":7,"nz":6,
                    "coordinate_system":"cartesian",
                    "staggering":"c_grid",
                    "dr_m":250.0,"dy_m":300.0,"dz_m":100.0,
                    "scalar_origin_m":[125.0,150.0,50.0],
                    "physical_bounds_m":{{"lower":[0.0,0.0,0.0],"upper":[2000.0,2100.0,600.0]}},
                    "valid_index_range":{{"lower":[1,1,0],"upper":[6,5,5]}},
                    "halo_width":{{"lower":[1,1,0],"upper":[1,1,0]}}
                }},
                "published_fields":["theta"],
                "fields":[{{"field_id":"theta","units":"K"}}]
            }}"#
        )
    }

    fn write_history_index(run: &Path, generation: &str, sequence: u64, entries: &[(u32, &str)]) {
        let entries = entries
            .iter()
            .map(|(step_index, manifest)| HistoryIndexEntry {
                step_index: *step_index,
                step_dir: format!("step_{step_index:06}"),
                manifest_sha256: sha256_hex(manifest.as_bytes()),
            })
            .collect();
        let segment = HistoryIndexSegment {
            schema: "nimbus.history.index.segment".into(),
            version: 1,
            publication_generation: generation.into(),
            sequence,
            operation: "reset".into(),
            previous_segment: None,
            previous_segment_sha256: None,
            entries,
            checksum: String::new(),
        };
        let payload = history_segment_payload(&segment);
        let segment_json = format!(
            "{},\n  \"checksum\": \"{}\"\n}}\n",
            &payload[..payload.len() - 1],
            sha256_hex(payload.as_bytes())
        );
        let segment_path = "history_index/active_a.json";
        fs::create_dir_all(run.join("history_index")).unwrap();
        fs::write(run.join(segment_path), &segment_json).unwrap();
        let head = HistoryIndexHead {
            schema: "nimbus.history.index.head".into(),
            version: 1,
            publication_generation: generation.into(),
            sequence,
            reset_sequence: sequence,
            segment: segment_path.into(),
            segment_sha256: sha256_hex(segment_json.as_bytes()),
            checksum: String::new(),
        };
        let payload = history_head_payload(&head);
        fs::write(
            run.join("history_index.json"),
            format!(
                "{},\n  \"checksum\": \"{}\"\n}}\n",
                &payload[..payload.len() - 1],
                sha256_hex(payload.as_bytes())
            ),
        )
        .unwrap();
    }

    fn canonical_manifest(index: u32, generation: &str) -> String {
        format!(
            r#"{{
                "schema":"nimbus.step",
                "manifest_version":5,
                "publication_state":"complete",
                "publication_generation":"{generation}",
                "step_index":{index},
                "simulation_time_s":2.5,
                "step_dir":"step_{index:06}",
                "data_product":{{"kind":"history","restart_complete":false,"storage_format":"npy_3d"}},
                "field_identity":{{"canonical_schema":"nimbus.fields","canonical_version":1,"cm1_alias_schema":"cm1.history","cm1_alias_version":1}},
                "grid":{{
                    "coordinate_system":"cartesian",
                    "axis_order":["x","y","z"],
                    "dimensions":{{"x":8,"y":7,"z":6}},
                    "spacing_m":{{"x":250.0,"y":300.0,"z":100.0}},
                    "native_model_staggering":"c_grid",
                    "publication_staggering":"cell_centered",
                    "scalar_origin_m":[125.0,150.0,50.0],
                    "physical_bounds_m":{{"lower":[250.0,300.0,0.0],"upper":[1750.0,1800.0,600.0]}},
                    "valid_index_range":{{"lower":[1,1,0],"upper":[6,5,5]}},
                    "halo_width":{{"lower":[1,1,0],"upper":[1,1,0]}}
                }},
                "published_fields":["theta"],
                "fields":[{{"field_id":"theta","units":"K","grid_location":"cell_center","state_semantic":"total_state","cm1_aliases":["th"]}}]
            }}"#
        )
    }

    fn canonical_netcdf_manifest(index: u32, generation: &str) -> String {
        canonical_manifest(index, generation)
            .replace("\"storage_format\":\"npy_3d\"", "\"storage_format\":\"netcdf\"")
            .replace(
                "\"cm1_aliases\":[\"th\"]",
                "\"cm1_aliases\":[\"th\"],\"file_pattern\":\"history.nc\",\"storage_layout\":\"shared_file\",\"variable_name\":\"theta\"",
            )
    }

    fn indexed_manifest(index: u32, generation: &str) -> String {
        let mut manifest: serde_json::Value =
            serde_json::from_str(&canonical_manifest(index, generation)).unwrap();
        manifest["manifest_version"] = serde_json::json!(OUTPUT_MANIFEST_VERSION);
        manifest.to_string()
    }

    fn f64_npy_values(shape: [usize; 3], values: &[f64]) -> Vec<u8> {
        assert_eq!(values.len(), shape.into_iter().product::<usize>());
        let dict = format!(
            "{{'descr': '<f8', 'fortran_order': False, 'shape': ({}, {}, {}), }}",
            shape[0], shape[1], shape[2]
        );
        let mut header = dict.into_bytes();
        header.push(b'\n');
        while (10 + header.len()) % 16 != 0 {
            header.insert(header.len() - 1, b' ');
        }
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&header);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn f64_npy(shape: [usize; 3]) -> Vec<u8> {
        f64_npy_values(shape, &vec![0.0; shape.into_iter().product::<usize>()])
    }

    fn publish_versioned_geometry(run: &Path, generation: &str, digest: &str) -> PathBuf {
        let directory = run.join("geometry").join(generation).join(digest);
        fs::create_dir_all(&directory).unwrap();
        fs::write(
            directory.join("physical_z_center_m.npy"),
            f64_npy([8, 7, 6]),
        )
        .unwrap();
        fs::write(
            directory.join("physical_z_interface_m.npy"),
            f64_npy([8, 7, 7]),
        )
        .unwrap();
        fs::write(
            directory.join("manifest.json"),
            serde_json::json!({
                "schema": "nimbus.terrain-geometry",
                "version": 1,
                "publication_generation": generation,
                "content_sha256": digest,
                "identity_encoding": "shape_u64_le_then_ieee754_f64_le",
                "dtype": "float64",
                "layout": "x_y_z",
                "center_shape": [8, 7, 6],
                "interface_shape": [8, 7, 7]
            })
            .to_string(),
        )
        .unwrap();
        directory
    }

    fn versioned_terrain_manifest(index: u32, generation: &str, digest: &str) -> String {
        let mut manifest: serde_json::Value =
            serde_json::from_str(&canonical_manifest(index, generation)).unwrap();
        manifest["manifest_version"] = serde_json::json!(OUTPUT_MANIFEST_VERSION);
        manifest["grid"]["vertical_geometry"] = serde_json::json!({
            "kind": "terrain_following",
            "center_coordinate": "physical_z_center_m",
            "interface_coordinate": "physical_z_interface_m",
            "dtype": "float64",
            "layout": "x_y_z",
            "storage": "versioned_npy_geometry",
            "contract": "nimbus.terrain-geometry.v1",
            "content_sha256": digest,
            "publication_generation": generation,
            "geometry_manifest": format!("../geometry/{generation}/{digest}/manifest.json"),
            "center_path": format!("../geometry/{generation}/{digest}/physical_z_center_m.npy"),
            "interface_path": format!("../geometry/{generation}/{digest}/physical_z_interface_m.npy")
        });
        manifest["grid"]["horizontal_scalar_origin_m"] = serde_json::json!([125.0, 150.0]);
        manifest["grid"]
            .as_object_mut()
            .unwrap()
            .remove("scalar_origin_m");
        manifest.to_string()
    }

    #[test]
    fn coordinate_metadata_is_validated_as_one_contract() {
        let valid = serde_json::json!({
            "nr": 8,
            "nth": 7,
            "nz": 6,
            "coordinate_system": "cartesian",
            "staggering": "c_grid",
            "dr_m": 250.0,
            "dy_m": 300.0,
            "dz_m": 100.0,
            "scalar_origin_m": [125.0, 150.0, 50.0],
            "physical_bounds_m": {
                "lower": [0.0, 0.0, 0.0],
                "upper": [2000.0, 2100.0, 600.0]
            },
            "valid_index_range": {"lower": [1, 1, 0], "upper": [6, 5, 5]},
            "halo_width": {"lower": [1, 1, 0], "upper": [1, 1, 0]}
        });
        assert!(grid_is_complete_for_version(
            &valid,
            LEGACY_OUTPUT_MANIFEST_VERSION
        ));

        let mut wrong_origin = valid.clone();
        wrong_origin["scalar_origin_m"][2] = serde_json::json!(0.0);
        assert!(!grid_is_complete_for_version(
            &wrong_origin,
            LEGACY_OUTPUT_MANIFEST_VERSION
        ));

        let mut wrong_bounds = valid.clone();
        wrong_bounds["physical_bounds_m"]["upper"][0] = serde_json::json!(1750.0);
        assert!(!grid_is_complete_for_version(
            &wrong_bounds,
            LEGACY_OUTPUT_MANIFEST_VERSION
        ));

        let mut inconsistent_halo = valid.clone();
        inconsistent_halo["valid_index_range"]["lower"][0] = serde_json::json!(0);
        assert!(!grid_is_complete_for_version(
            &inconsistent_halo,
            LEGACY_OUTPUT_MANIFEST_VERSION
        ));

        let mut unsupported_coordinates = valid;
        unsupported_coordinates["coordinate_system"] = serde_json::json!("cylindrical");
        assert!(!grid_is_complete_for_version(
            &unsupported_coordinates,
            LEGACY_OUTPUT_MANIFEST_VERSION
        ));

        let canonical =
            serde_json::from_str::<serde_json::Value>(&canonical_manifest(0, GENERATION_A))
                .unwrap();
        assert!(grid_is_complete_for_version(
            &canonical["grid"],
            OUTPUT_MANIFEST_VERSION
        ));
        assert_eq!(parse_grid_info(&canonical["grid"]).nr, Some(8));
        assert!(canonical_history_contract_is_complete(
            &canonical, "history"
        ));

        let mut incomplete_v3 = canonical.clone();
        incomplete_v3["grid"]
            .as_object_mut()
            .unwrap()
            .remove("dimensions");
        incomplete_v3["grid"]["nr"] = serde_json::json!(8);
        incomplete_v3["grid"]["nth"] = serde_json::json!(7);
        incomplete_v3["grid"]["nz"] = serde_json::json!(6);
        assert!(!grid_is_complete_for_version(
            &incomplete_v3["grid"],
            OUTPUT_MANIFEST_VERSION
        ));

        let mut restart_ambiguous = canonical;
        restart_ambiguous["data_product"]["restart_complete"] = serde_json::json!(true);
        assert!(!canonical_history_contract_is_complete(
            &restart_ambiguous,
            "history"
        ));
    }

    #[test]
    fn current_first_partial_frame_is_not_complete() {
        let run = TestRunDir::new("partial");
        run.current();
        run.step_with_field(0);

        let steps = list_step_dirs_blocking(run.0.to_string_lossy().into_owned()).unwrap();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].publication_state, StepPublicationState::Incomplete);
        assert!(steps[0].grid.is_none());
    }

    #[test]
    fn nested_history_inherits_root_generation_without_becoming_legacy() {
        let run = TestRunDir::new("nested_generation");
        run.current_generation_version(GENERATION_A, "ready", TERRAIN_GEOMETRY_MANIFEST_VERSION);
        let child_root = run.0.join("nests/child_0000");
        let step = child_root.join("step_000004");
        fs::create_dir_all(&step).unwrap();
        // A copied/local marker cannot reclassify a path-owned nested product
        // as an independent root history.
        fs::copy(
            run.0.join(RUN_OUTPUT_CONTRACT_FILE),
            child_root.join(RUN_OUTPUT_CONTRACT_FILE),
        )
        .unwrap();
        fs::write(step.join("theta.npy"), b"non-empty-test-payload").unwrap();
        let child_manifest = canonical_manifest(4, GENERATION_A)
            .replace(r#""kind":"history""#, r#""kind":"nested_child_history""#);
        fs::write(step.join("manifest.json"), child_manifest).unwrap();

        let levels = list_nested_history_levels_blocking(run.0.to_string_lossy().into_owned())
            .expect("nested discovery");
        assert_eq!(levels.len(), 1);
        assert_eq!(levels[0].level_id, "child_0000");
        assert_eq!(levels[0].parent_level_id, "root");
        assert_eq!(levels[0].depth, 1);
        assert_eq!(levels[0].steps.len(), 1);
        assert_eq!(
            levels[0].steps[0].publication_state,
            StepPublicationState::Complete
        );
        assert_eq!(
            levels[0].steps[0].publication_generation.as_deref(),
            Some(GENERATION_A)
        );
        assert_eq!(levels[0].steps[0].index, 4);
        ensure_expected_publication_generation(&child_root, 4, Some(GENERATION_A))
            .expect("child read must retain the root generation gate");
    }

    #[test]
    fn recursive_nested_history_retains_root_generation_and_explicit_parent_graph() {
        let run = TestRunDir::new("recursive_nested_generation");
        run.current_generation_version(GENERATION_A, "ready", TERRAIN_GEOMETRY_MANIFEST_VERSION);
        let child_root = run.0.join("nests/child_0000");
        let grandchild_root = child_root.join("nests/child_0002");
        for history_root in [&child_root, &grandchild_root] {
            let step = history_root.join("step_000004");
            fs::create_dir_all(&step).unwrap();
            fs::write(step.join("theta.npy"), b"non-empty-test-payload").unwrap();
            let manifest = canonical_manifest(4, GENERATION_A)
                .replace(r#""kind":"history""#, r#""kind":"nested_child_history""#);
            fs::write(step.join("manifest.json"), manifest).unwrap();
        }

        let levels = list_nested_history_levels_blocking(run.0.to_string_lossy().into_owned())
            .expect("recursive nested discovery");
        assert_eq!(levels.len(), 2);
        assert_eq!(levels[0].level_id, "child_0000");
        assert_eq!(levels[0].parent_level_id, "root");
        assert_eq!(levels[0].depth, 1);
        assert_eq!(levels[1].level_id, "child_0000/child_0002");
        assert_eq!(levels[1].parent_level_id, "child_0000");
        assert_eq!(levels[1].depth, 2);
        assert_eq!(
            levels[1].steps[0].publication_state,
            StepPublicationState::Complete
        );
        ensure_expected_publication_generation(&grandchild_root, 4, Some(GENERATION_A))
            .expect("grandchild read must retain the root generation gate");

        let exact = list_nested_history_levels_for_step_blocking(
            run.0.to_string_lossy().into_owned(),
            Some(4),
        )
        .expect("targeted recursive frame discovery");
        assert!(exact.iter().all(|level| level.steps.len() == 1));
        let missing = list_nested_history_levels_for_step_blocking(
            run.0.to_string_lossy().into_owned(),
            Some(5),
        )
        .expect("missing-cadence recursive discovery");
        assert_eq!(missing.len(), 2);
        assert!(missing.iter().all(|level| level.steps.is_empty()));
    }

    #[test]
    fn root_history_cannot_masquerade_as_a_nested_child_product() {
        let run = TestRunDir::new("nested_product_kind");
        run.current_generation_version(GENERATION_A, "ready", TERRAIN_GEOMETRY_MANIFEST_VERSION);
        let child_root = run.0.join("nests/child_0000");
        let step = child_root.join("step_000004");
        fs::create_dir_all(&step).unwrap();
        fs::write(step.join("theta.npy"), b"non-empty-test-payload").unwrap();
        fs::write(
            step.join("manifest.json"),
            canonical_manifest(4, GENERATION_A),
        )
        .unwrap();

        let levels = list_nested_history_levels_blocking(run.0.to_string_lossy().into_owned())
            .expect("bounded nested discovery");
        assert_eq!(levels.len(), 1);
        assert_eq!(
            levels[0].steps[0].publication_state,
            StepPublicationState::InvalidManifest
        );
    }

    #[test]
    fn corrupt_current_manifest_fails_closed() {
        let run = TestRunDir::new("corrupt");
        run.current();
        let step = run.step_with_field(0);
        fs::write(step.join("manifest.json"), b"{truncated").unwrap();

        let steps = list_step_dirs_blocking(run.0.to_string_lossy().into_owned()).unwrap();
        assert_eq!(
            steps[0].publication_state,
            StepPublicationState::InvalidManifest
        );
        assert!(steps[0].grid.is_none());
    }

    #[test]
    fn markerless_run_uses_deliberate_legacy_path() {
        let run = TestRunDir::new("legacy");
        run.step_with_field(0);

        let steps = list_step_dirs_blocking(run.0.to_string_lossy().into_owned()).unwrap();
        assert_eq!(steps[0].publication_state, StepPublicationState::Legacy);
        assert_eq!(steps[0].fields, vec!["theta"]);
    }

    #[test]
    fn valid_current_manifest_exposes_complete_frame() {
        let run = TestRunDir::new("valid");
        run.current();
        let step = run.step_with_field(3);
        fs::write(
            step.join("manifest.json"),
            current_manifest(3, GENERATION_A),
        )
        .unwrap();

        let steps = list_step_dirs_blocking(run.0.to_string_lossy().into_owned()).unwrap();
        assert_eq!(steps[0].publication_state, StepPublicationState::Complete);
        assert_eq!(
            steps[0].publication_generation.as_deref(),
            Some(GENERATION_A)
        );
        assert_eq!(steps[0].simulation_time_s, Some(2.5));
        assert_eq!(steps[0].grid.as_ref().and_then(|g| g.nr), Some(8));
        assert_eq!(
            steps[0].grid.as_ref().and_then(|g| g.staggering.as_deref()),
            Some("c_grid")
        );
        assert_eq!(
            steps[0].grid.as_ref().and_then(|g| g.scalar_origin_m),
            Some([125.0, 150.0, 50.0])
        );
        assert_eq!(
            steps[0].field_units.get("theta").map(String::as_str),
            Some("K")
        );
        assert!(ensure_expected_publication_generation(&run.0, 3, Some(GENERATION_A)).is_ok());
        assert!(ensure_expected_publication_generation(&run.0, 3, Some(GENERATION_B)).is_err());
    }

    #[test]
    fn indexed_refresh_reads_only_the_new_tail_and_fails_closed_on_corruption() {
        let run = TestRunDir::new("indexed_tail");
        run.current_indexed(GENERATION_A);
        let manifest0 = indexed_manifest(0, GENERATION_A);
        let step0 = run.step_with_field(0);
        fs::write(step0.join("manifest.json"), &manifest0).unwrap();
        write_history_index(&run.0, GENERATION_A, 1, &[(0, &manifest0)]);

        let first = list_step_dirs_blocking(run.0.to_string_lossy().into_owned()).unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].publication_state, StepPublicationState::Complete);

        // If refresh re-opened an old manifest this deliberate mutation would
        // fail its index hash. A new cursor sequence must reuse the cached old
        // frame and validate only the appended tail.
        fs::write(step0.join("manifest.json"), b"{corrupt-old-manifest").unwrap();
        let manifest1 = indexed_manifest(1, GENERATION_A);
        let step1 = run.step_with_field(1);
        fs::write(step1.join("manifest.json"), &manifest1).unwrap();
        write_history_index(&run.0, GENERATION_A, 2, &[(0, &manifest0), (1, &manifest1)]);
        let appended = list_step_dirs_blocking(run.0.to_string_lossy().into_owned()).unwrap();
        assert_eq!(
            appended.iter().map(|step| step.index).collect::<Vec<_>>(),
            vec![0, 1]
        );

        // A partial step never named by the durable cursor remains invisible.
        run.step_with_field(2);
        let unchanged = list_step_dirs_blocking(run.0.to_string_lossy().into_owned()).unwrap();
        assert_eq!(unchanged.len(), 2);

        fs::write(
            run.0.join("history_index.pending.json"),
            b"{\"publication_generation\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"}",
        )
        .unwrap();
        assert!(list_step_dirs_blocking(run.0.to_string_lossy().into_owned()).is_err());
        fs::remove_file(run.0.join("history_index.pending.json")).unwrap();

        fs::write(run.0.join("history_index.json"), b"{\"corrupt\":true}").unwrap();
        assert!(list_step_dirs_blocking(run.0.to_string_lossy().into_owned()).is_err());
    }

    #[test]
    fn indexed_generation_retention_and_nested_series_are_isolated() {
        let run = TestRunDir::new("indexed_generation_nested");
        run.current_indexed(GENERATION_A);
        let root_manifest = indexed_manifest(4, GENERATION_A);
        let root_step = run.step_with_field(4);
        fs::write(root_step.join("manifest.json"), &root_manifest).unwrap();
        write_history_index(&run.0, GENERATION_A, 7, &[(4, &root_manifest)]);

        let child = run.0.join("nests/child_0000");
        let child_step = child.join("step_000004");
        fs::create_dir_all(&child_step).unwrap();
        fs::write(child_step.join("theta.npy"), b"non-empty-test-payload").unwrap();
        let child_manifest = indexed_manifest(4, GENERATION_A)
            .replace(r#""kind":"history""#, r#""kind":"nested_child_history""#);
        fs::write(child_step.join("manifest.json"), &child_manifest).unwrap();
        write_history_index(&child, GENERATION_A, 3, &[(4, &child_manifest)]);
        let levels =
            list_nested_history_levels_blocking(run.0.to_string_lossy().into_owned()).unwrap();
        assert_eq!(levels.len(), 1);
        assert_eq!(levels[0].steps.len(), 1);

        // Retention is represented by a new monotonic cursor containing only
        // survivors; cached removed entries disappear without a directory scan.
        fs::remove_dir_all(&root_step).unwrap();
        write_history_index(&run.0, GENERATION_A, 8, &[]);
        assert!(
            list_step_dirs_blocking(run.0.to_string_lossy().into_owned())
                .unwrap()
                .is_empty()
        );

        run.current_indexed(GENERATION_B);
        assert!(list_step_dirs_blocking(run.0.to_string_lossy().into_owned()).is_err());
    }

    #[test]
    fn indexed_open_rejects_a_corrupt_committed_interior_segment() {
        let run = TestRunDir::new("indexed_interior_corruption");
        run.current_indexed(GENERATION_A);
        let manifest0 = indexed_manifest(0, GENERATION_A);
        let manifest1 = indexed_manifest(1, GENERATION_A);
        for (index, manifest) in [(0, &manifest0), (1, &manifest1)] {
            let step = run.step_with_field(index);
            fs::write(step.join("manifest.json"), manifest).unwrap();
        }
        let sealed_path = "history_index/segment_00000000000000000001_0123456789abcdef.json";
        fs::create_dir_all(run.0.join("history_index")).unwrap();
        let sealed = HistoryIndexSegment {
            schema: "nimbus.history.index.segment".into(),
            version: 1,
            publication_generation: GENERATION_A.into(),
            sequence: 1,
            operation: "reset".into(),
            previous_segment: None,
            previous_segment_sha256: None,
            entries: vec![HistoryIndexEntry {
                step_index: 0,
                step_dir: "step_000000".into(),
                manifest_sha256: sha256_hex(manifest0.as_bytes()),
            }],
            checksum: String::new(),
        };
        let payload = history_segment_payload(&sealed);
        let sealed_json = format!(
            "{},\n  \"checksum\": \"{}\"\n}}\n",
            &payload[..payload.len() - 1],
            sha256_hex(payload.as_bytes())
        );
        let sealed_hash = sha256_hex(sealed_json.as_bytes());
        fs::write(run.0.join(sealed_path), &sealed_json).unwrap();

        let active = HistoryIndexSegment {
            schema: "nimbus.history.index.segment".into(),
            version: 1,
            publication_generation: GENERATION_A.into(),
            sequence: 2,
            operation: "append".into(),
            previous_segment: Some(sealed_path.into()),
            previous_segment_sha256: Some(sealed_hash),
            entries: vec![HistoryIndexEntry {
                step_index: 1,
                step_dir: "step_000001".into(),
                manifest_sha256: sha256_hex(manifest1.as_bytes()),
            }],
            checksum: String::new(),
        };
        let payload = history_segment_payload(&active);
        let active_json = format!(
            "{},\n  \"checksum\": \"{}\"\n}}\n",
            &payload[..payload.len() - 1],
            sha256_hex(payload.as_bytes())
        );
        fs::write(run.0.join("history_index/active_a.json"), &active_json).unwrap();
        let head = HistoryIndexHead {
            schema: "nimbus.history.index.head".into(),
            version: 1,
            publication_generation: GENERATION_A.into(),
            sequence: 2,
            reset_sequence: 1,
            segment: "history_index/active_a.json".into(),
            segment_sha256: sha256_hex(active_json.as_bytes()),
            checksum: String::new(),
        };
        let payload = history_head_payload(&head);
        fs::write(
            run.0.join("history_index.json"),
            format!(
                "{},\n  \"checksum\": \"{}\"\n}}\n",
                &payload[..payload.len() - 1],
                sha256_hex(payload.as_bytes())
            ),
        )
        .unwrap();
        fs::write(run.0.join(sealed_path), b"corrupt-interior").unwrap();
        assert!(list_step_dirs_blocking(run.0.to_string_lossy().into_owned()).is_err());
    }

    #[test]
    fn canonical_v5_manifest_is_read_without_legacy_grid_names() {
        let run = TestRunDir::new("canonical_v5");
        run.current_generation_version(GENERATION_A, "ready", TERRAIN_GEOMETRY_MANIFEST_VERSION);
        let step = run.step_with_field(4);
        fs::write(
            step.join("manifest.json"),
            canonical_manifest(4, GENERATION_A),
        )
        .unwrap();

        let steps = list_step_dirs_blocking(run.0.to_string_lossy().into_owned()).unwrap();
        assert_eq!(steps[0].publication_state, StepPublicationState::Complete);
        let grid = steps[0].grid.as_ref().unwrap();
        assert_eq!((grid.nr, grid.nth, grid.nz), (Some(8), Some(7), Some(6)));
        assert_eq!(
            (grid.dr_m, grid.dy_m, grid.dz_m),
            (Some(250.0), Some(300.0), Some(100.0))
        );
        assert_eq!(grid.staggering.as_deref(), Some("c_grid"));
    }

    #[test]
    fn terrain_geometry_sidecars_are_metadata_not_fields_and_are_required() {
        let run = TestRunDir::new("terrain_v5");
        run.current_generation_version(GENERATION_A, "ready", TERRAIN_GEOMETRY_MANIFEST_VERSION);
        let step = run.step_with_field(4);
        fs::write(step.join("physical_z_center_m.npy"), f64_npy([8, 7, 6])).unwrap();
        fs::write(step.join("physical_z_interface_m.npy"), f64_npy([8, 7, 7])).unwrap();
        let mut manifest: serde_json::Value =
            serde_json::from_str(&canonical_manifest(4, GENERATION_A)).unwrap();
        manifest["grid"]["vertical_geometry"] = serde_json::json!({
            "kind": "terrain_following",
            "center_coordinate": "physical_z_center_m",
            "interface_coordinate": "physical_z_interface_m",
            "dtype": "float64",
            "layout": "x_y_z",
            "storage": "npy_sidecar"
        });
        manifest["grid"]["horizontal_scalar_origin_m"] = serde_json::json!([125.0, 150.0]);
        manifest["grid"]["physical_bounds_m"]["lower"][2] = serde_json::json!(25.0);
        manifest["grid"]["physical_bounds_m"]["upper"][2] = serde_json::json!(725.0);
        manifest["grid"]
            .as_object_mut()
            .unwrap()
            .remove("scalar_origin_m");
        fs::write(step.join("manifest.json"), manifest.to_string()).unwrap();

        let steps = list_step_dirs_blocking(run.0.to_string_lossy().into_owned()).unwrap();
        assert_eq!(steps[0].publication_state, StepPublicationState::Complete);
        assert_eq!(steps[0].fields, vec!["theta"]);
        assert!(steps[0]
            .grid
            .as_ref()
            .is_some_and(|grid| grid.terrain_following));
        assert_eq!(
            steps[0]
                .grid
                .as_ref()
                .and_then(|grid| grid.horizontal_scalar_origin_m),
            Some([125.0, 150.0])
        );
        assert!(terrain_window_descriptor(&run.0, 4, Some(GENERATION_A))
            .unwrap()
            .is_some());

        fs::remove_file(run.0.join(RUN_OUTPUT_CONTRACT_FILE)).unwrap();
        assert!(terrain_window_descriptor(&run.0, 4, None)
            .unwrap()
            .is_some());
        run.current_generation_version(GENERATION_A, "ready", TERRAIN_GEOMETRY_MANIFEST_VERSION);

        let fingerprint = step_cache_fingerprint(&step);
        let mut directory_options = fs::OpenOptions::new();
        directory_options.read(true);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            use windows_sys::Win32::Storage::FileSystem::{
                FILE_FLAG_BACKUP_SEMANTICS, FILE_WRITE_ATTRIBUTES,
            };
            directory_options
                .access_mode(FILE_WRITE_ATTRIBUTES)
                .custom_flags(FILE_FLAG_BACKUP_SEMANTICS);
        }
        let directory = directory_options.open(&step).unwrap();
        fs::remove_file(step.join("physical_z_interface_m.npy")).unwrap();
        // Directory timestamps may coalesce or be deferred by the filesystem.
        directory
            .set_times(fs::FileTimes::new().set_modified(fingerprint.directory_modified.unwrap()))
            .unwrap();
        assert!(step_cache_fingerprint(&step) == fingerprint);
        let steps = list_step_dirs_blocking(run.0.to_string_lossy().into_owned()).unwrap();
        assert_eq!(
            steps[0].publication_state,
            StepPublicationState::InvalidManifest
        );
    }

    #[test]
    fn cached_indexed_terrain_revalidates_required_geometry_without_reopening_step_manifests() {
        let run = TestRunDir::new("indexed_terrain_dependencies");
        run.current_indexed(GENERATION_A);
        let digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let geometry_dir = publish_versioned_geometry(&run.0, GENERATION_A, digest);
        let step = run.step_with_field(4);
        let manifest = versioned_terrain_manifest(4, GENERATION_A, digest);
        fs::write(step.join("manifest.json"), &manifest).unwrap();
        write_history_index(&run.0, GENERATION_A, 1, &[(4, &manifest)]);
        let publication = || {
            list_step_dirs_blocking(run.0.to_string_lossy().into_owned())
                .unwrap()
                .into_iter()
                .map(|step| step.publication_state)
                .collect::<Vec<_>>()
        };
        assert_eq!(publication(), vec![StepPublicationState::Complete]);

        fs::write(step.join("manifest.json"), b"{cached-sealed-step").unwrap();
        assert_eq!(publication(), vec![StepPublicationState::Complete]);
        fs::write(step.join("manifest.json"), &manifest).unwrap();

        let interfaces = geometry_dir.join("physical_z_interface_m.npy");
        let valid = fs::read(&interfaces).unwrap();
        let mut wrong_dtype = valid.clone();
        let offset = wrong_dtype.windows(3).position(|bytes| bytes == b"<f8").unwrap();
        wrong_dtype[offset + 1] = b'i';
        fs::write(&interfaces, wrong_dtype).unwrap();
        assert_eq!(publication(), vec![StepPublicationState::InvalidManifest]);
        fs::write(&interfaces, &valid).unwrap();
        assert_eq!(publication(), vec![StepPublicationState::Complete]);

        fs::remove_file(&interfaces).unwrap();
        assert_eq!(publication(), vec![StepPublicationState::InvalidManifest]);
        fs::write(&interfaces, &valid).unwrap();
        assert_eq!(publication(), vec![StepPublicationState::Complete]);

        let mut geometry: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(geometry_dir.join("manifest.json")).unwrap(),
        ).unwrap();
        geometry["publication_generation"] = serde_json::json!(GENERATION_B);
        fs::write(geometry_dir.join("manifest.json"), geometry.to_string()).unwrap();
        assert_eq!(publication(), vec![StepPublicationState::InvalidManifest]);
        publish_versioned_geometry(&run.0, GENERATION_A, digest);
        assert_eq!(publication(), vec![StepPublicationState::Complete]);

        let next_step = run.step_with_field(5);
        let next_manifest = versioned_terrain_manifest(5, GENERATION_A, digest);
        fs::write(next_step.join("manifest.json"), &next_manifest).unwrap();
        write_history_index(&run.0, GENERATION_A, 2, &[(4, &manifest), (5, &next_manifest)]);
        fs::remove_file(geometry_dir.join("physical_z_center_m.npy")).unwrap();
        assert_eq!(publication(), vec![StepPublicationState::InvalidManifest; 2]);
    }

    #[test]
    fn cached_shared_terrain_is_validated_once_per_refresh_without_hiding_changes() {
        let run = TestRunDir::new("shared_terrain_cache_cost");
        run.current_indexed(GENERATION_A);
        let digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let geometry_dir = publish_versioned_geometry(&run.0, GENERATION_A, digest);
        let manifests: Vec<_> = (0..32).map(|index| {
            let step = run.step_with_field(index);
            let manifest = versioned_terrain_manifest(index, GENERATION_A, digest);
            fs::write(step.join("manifest.json"), &manifest).unwrap();
            (index, manifest)
        }).collect();
        let index: Vec<_> = manifests.iter().map(|(index, text)| (*index, text.as_str())).collect();
        write_history_index(&run.0, GENERATION_A, 1, &index);
        let refresh = || {
            TERRAIN_GEOMETRY_VALIDATIONS.with(|count| count.set(0));
            let steps = list_step_dirs_blocking(run.0.to_string_lossy().into_owned()).unwrap();
            assert_eq!(steps.len(), 32);
            steps
        };
        assert!(refresh().iter().all(|step| step.publication_state == StepPublicationState::Complete));
        for _ in 0..2 {
            assert!(refresh().iter().all(|step| step.publication_state == StepPublicationState::Complete));
            TERRAIN_GEOMETRY_VALIDATIONS.with(|count| assert_eq!(count.get(), 1));
        }

        let cached = RUN_STEP_CACHE.lock().unwrap().get(&run.0.canonicalize().unwrap()).unwrap().clone();
        let mut wrong_shape = cached.steps.get(&0).unwrap().clone();
        let mut manifest: serde_json::Value = serde_json::from_str(&manifests[0].1).unwrap();
        manifest["grid"]["dimensions"]["x"] = serde_json::json!(9);
        wrong_shape.terrain_geometry = cache_terrain_geometry(&manifest, OUTPUT_MANIFEST_VERSION);
        let mut memo = TerrainValidationMemo::new();
        let step = run.0.join("step_000000");
        assert!(cached_terrain_geometry_is_complete(cached.steps.get(&0).unwrap(), &step, &mut memo));
        assert!(!cached_terrain_geometry_is_complete(&wrong_shape, &step, &mut memo));

        fs::remove_file(geometry_dir.join("physical_z_interface_m.npy")).unwrap();
        assert!(refresh().iter().all(|step| step.publication_state == StepPublicationState::InvalidManifest));
        publish_versioned_geometry(&run.0, GENERATION_A, digest);
        assert!(refresh().iter().all(|step| step.publication_state == StepPublicationState::Complete));
        assert!(refresh().iter().all(|step| step.publication_state == StepPublicationState::Complete));
        TERRAIN_GEOMETRY_VALIDATIONS.with(|count| assert_eq!(count.get(), 1));
    }

    #[test]
    fn versioned_terrain_geometry_is_reused_and_fails_closed_on_corruption_or_generation_change() {
        let run = TestRunDir::new("terrain_v7_versioned");
        run.current_generation_version(GENERATION_A, "ready", OUTPUT_MANIFEST_VERSION);
        let digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let geometry_dir = publish_versioned_geometry(&run.0, GENERATION_A, digest);
        for index in [4, 5] {
            let step = run.step_with_field(index);
            fs::write(
                step.join("manifest.json"),
                versioned_terrain_manifest(index, GENERATION_A, digest),
            )
            .unwrap();
        }

        let steps = list_step_dirs_blocking(run.0.to_string_lossy().into_owned()).unwrap();
        assert_eq!(steps.len(), 2);
        assert!(steps
            .iter()
            .all(|step| step.publication_state == StepPublicationState::Complete));
        let first = terrain_window_descriptor(&run.0, 4, Some(GENERATION_A))
            .unwrap()
            .unwrap();
        let second = terrain_window_descriptor(&run.0, 5, Some(GENERATION_A))
            .unwrap()
            .unwrap();
        assert_eq!(
            first.geometry_directory.as_deref(),
            Some(geometry_dir.as_path())
        );
        assert_eq!(
            second.geometry_directory.as_deref(),
            Some(geometry_dir.as_path())
        );
        let window = crate::commands::terrain_window::read_npy_geometry_window(
            first.geometry_directory.as_deref().unwrap(),
            first.source_shape,
            [1, 2, 3],
            [2, 2, 2],
            &mut || Ok(()),
        )
        .unwrap();
        assert_eq!(window.centers_m.len(), 8);
        assert_eq!(window.interfaces_m.len(), 12);

        let mut geometry_manifest: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(geometry_dir.join("manifest.json")).unwrap())
                .unwrap();
        geometry_manifest["publication_generation"] = serde_json::json!(GENERATION_B);
        fs::write(
            geometry_dir.join("manifest.json"),
            geometry_manifest.to_string(),
        )
        .unwrap();
        assert!(terrain_window_descriptor(&run.0, 4, Some(GENERATION_A)).is_err());

        publish_versioned_geometry(&run.0, GENERATION_A, digest);
        run.current_generation_version(GENERATION_B, "ready", OUTPUT_MANIFEST_VERSION);
        let restarted = list_step_dirs_blocking(run.0.to_string_lossy().into_owned()).unwrap();
        assert!(restarted
            .iter()
            .all(|step| step.publication_state == StepPublicationState::InvalidManifest));
    }

    #[test]
    fn canonical_netcdf_frame_uses_manifest_variable_inventory() {
        let run = TestRunDir::new("canonical_netcdf");
        run.current_generation_version(GENERATION_A, "ready", TERRAIN_GEOMETRY_MANIFEST_VERSION);
        let step = run.0.join("step_000004");
        fs::create_dir_all(&step).unwrap();
        fs::write(step.join("history.nc"), b"non-empty-netcdf-fixture").unwrap();
        fs::write(
            step.join("manifest.json"),
            canonical_netcdf_manifest(4, GENERATION_A),
        )
        .unwrap();

        let steps = list_step_dirs_blocking(run.0.to_string_lossy().into_owned()).unwrap();
        assert_eq!(steps[0].publication_state, StepPublicationState::Complete);
        assert_eq!(steps[0].fields, vec!["theta"]);
        assert_eq!(
            netcdf_field_mapping(run.0.to_str().unwrap(), 4, "theta")
                .unwrap()
                .unwrap()
                .1,
            "theta"
        );

        let invalid = canonical_netcdf_manifest(4, GENERATION_A)
            .replace("\"variable_name\":\"theta\"", "\"variable_name\":\"th\"");
        fs::write(step.join("manifest.json"), invalid).unwrap();
        let steps = list_step_dirs_blocking(run.0.to_string_lossy().into_owned()).unwrap();
        assert_eq!(
            steps[0].publication_state,
            StepPublicationState::InvalidManifest
        );
    }

    /// Exact handoff test used by `make qualify-netcdf-history`. The C++
    /// fixture writer supplies the run so this exercises Studio against the
    /// bytes and manifest Nimbus actually publishes.
    #[test]
    #[ignore]
    fn manufactured_visual_load_history_reopens_in_studio() {
        let run_dir = std::env::var("NIMBUS_TEST_VISUAL_LOAD_RUN").expect("set NIMBUS_TEST_VISUAL_LOAD_RUN");
        let specification: serde_json::Value = serde_json::from_slice(&std::fs::read(
            Path::new(&run_dir).join("visual_load_fixture.json")).unwrap()).unwrap();
        let steps = list_step_dirs_blocking(run_dir).unwrap();
        assert_eq!(steps.len(), specification["frames"].as_u64().unwrap() as usize);
        for (index, step) in steps.iter().enumerate() {
            assert_eq!(step.index, specification["frame_indices"][index].as_u64().unwrap_or(index as u64) as u32);
            let expected_state = if specification["source_kind"] == "saved-history-reference" {
                StepPublicationState::Legacy
            } else { StepPublicationState::Complete };
            assert_eq!(step.publication_state, expected_state);
            assert_eq!(step.fields.len(), 33);
            assert!(step.grid.is_some());
        }
    }

    #[test]
    #[ignore]
    fn simulator_netcdf_history_reopens_in_studio() {
        let run_dir = std::env::var("NIMBUS_TEST_NETCDF_RUN")
            .expect("set NIMBUS_TEST_NETCDF_RUN to the generated qualification run");
        let steps = list_step_dirs_blocking(run_dir.clone()).unwrap();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].publication_state, StepPublicationState::Complete);
        assert_eq!(steps[0].fields, vec!["accumulated_rainfall", "theta"]);
        let (path, variable) = netcdf_field_mapping(&run_dir, 4, "theta").unwrap().unwrap();
        let (shape, values) = read_netcdf_field(&path, &variable).unwrap();
        assert_eq!(shape, vec![2, 3, 2]);
        assert_eq!(
            values,
            vec![
                300.0, 301.0, 302.0, 303.0, 304.0, 305.0, 310.0, 311.0, 312.0, 313.0, 314.0, 315.0
            ]
        );
        let request = crate::commands::datasets::NetcdfWindowRequest {
            schema_version: 2,
            request_id: "qualification-window".into(),
            aggregation: "cell_mean".into(),
            origin: vec![0, 1, 0],
            extent: vec![2, 2, 2],
            stride: vec![1, 1, 1],
            max_output_bytes: Some(32),
            max_output_samples: Some(8),
            max_source_bytes: Some(32),
        };
        let control = crate::commands::datasets::NetcdfReadControl::new();
        let window = crate::commands::netcdf_dynamic::read_field_window(
            &path.to_string_lossy(),
            &variable,
            0,
            &request,
            &control,
        )
        .unwrap();
        assert_eq!(window.source_shape, vec![2, 3, 2]);
        assert_eq!(window.output_shape, vec![2, 2, 2]);
        assert_eq!(
            window.values,
            vec![302.0, 303.0, 304.0, 305.0, 312.0, 313.0, 314.0, 315.0]
        );
        assert_eq!(window.source_read_bytes, 32);
        assert_eq!(window.cancellation_mode, "boundary_only");
    }

    /// Cross-language terrain-window gate. The real C++ netCDF writer owns the
    /// float64 auxiliary coordinates; Studio must preserve them through the
    /// bounded double reader and the shared physical-volume aggregator.
    #[test]
    #[ignore]
    fn simulator_terrain_netcdf_window_uses_exact_geometry() {
        let run_dir = std::env::var("NIMBUS_TEST_TERRAIN_NETCDF_RUN")
            .expect("set NIMBUS_TEST_TERRAIN_NETCDF_RUN to the terrain fixture run");
        let steps = list_step_dirs_blocking(run_dir.clone()).unwrap();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].publication_state, StepPublicationState::Complete);
        assert!(steps[0]
            .grid
            .as_ref()
            .is_some_and(|grid| grid.terrain_following));
        let generation = steps[0].publication_generation.as_deref().unwrap();
        let descriptor = terrain_window_descriptor(Path::new(&run_dir), 4, Some(generation))
            .unwrap()
            .unwrap();
        let mapping = netcdf_field_mapping(&run_dir, 4, "theta").unwrap().unwrap();
        let (interface_shape, interfaces, interface_bytes) =
            crate::commands::netcdf_dynamic::read_f64_window(
                &mapping.0.to_string_lossy(),
                "physical_z_interface_m",
                [0, 0, 0],
                [2, 3, 3],
            )
            .unwrap();
        assert_eq!(interface_shape, vec![2, 3, 3]);
        assert_eq!(interface_bytes, 144);
        assert_eq!(interfaces[0..3], [0.0, 80.0, 260.0]);
        assert_eq!(interfaces[15..18], [50.0, 150.0, 430.0]);

        let request = FieldWindowRequest {
            schema_version: 2,
            request_id: "qualification-terrain-netcdf-window".into(),
            aggregation: "cell_mean".into(),
            origin: vec![0, 0, 0],
            extent: vec![2, 3, 2],
            stride: vec![2, 3, 2],
            max_output_bytes: Some(40),
            max_source_bytes: Some(1024),
        };
        let payload = read_terrain_window_payload(
            &run_dir,
            4,
            "theta",
            &request,
            Some([2, 3, 2]),
            &descriptor,
            Some(&mapping),
            None,
            &FieldReadControl::new(),
        )
        .unwrap();
        assert_eq!(payload.source_working_bytes, 357);
        assert_eq!(payload.geometry_body.len(), 32);
        let value = f32::from_le_bytes(payload.field_body.as_slice().try_into().unwrap());
        let field = [
            300.0, 301.0, 302.0, 303.0, 304.0, 305.0, 310.0, 311.0, 312.0, 313.0, 314.0, 315.0,
        ];
        let thickness = [
            80.0, 180.0, 90.0, 190.0, 100.0, 200.0, 80.0, 260.0, 90.0, 270.0, 100.0, 280.0,
        ];
        let total_volume: f64 = thickness.iter().sum();
        let expected = field
            .iter()
            .zip(thickness)
            .map(|(field, dz)| field * dz)
            .sum::<f64>()
            / total_volume;
        assert!((f64::from(value) - expected).abs() < 2.0e-5);
    }

    #[test]
    #[ignore]
    fn simulator_mapped_netcdf_history_matches_npy_reference() {
        let netcdf_run = std::env::var("NIMBUS_TEST_MAPPED_NETCDF_RUN")
            .expect("set NIMBUS_TEST_MAPPED_NETCDF_RUN");
        let npy_run = std::env::var("NIMBUS_TEST_MAPPED_NPY_RUN")
            .expect("set NIMBUS_TEST_MAPPED_NPY_RUN");
        let netcdf_steps = list_step_dirs_blocking(netcdf_run.clone()).unwrap();
        let npy_steps = list_step_dirs_blocking(npy_run.clone()).unwrap();
        let netcdf_step = netcdf_steps.last().unwrap();
        let npy_step = npy_steps.last().unwrap();
        assert_eq!(netcdf_step.index, npy_step.index);
        assert_eq!(netcdf_step.fields, npy_step.fields);
        assert!(netcdf_step
            .grid
            .as_ref()
            .is_some_and(|grid| grid.terrain_following));
        let generation = netcdf_step.publication_generation.as_deref().unwrap();
        assert!(terrain_window_descriptor(
            Path::new(&netcdf_run), netcdf_step.index, Some(generation))
            .unwrap()
            .is_some());
        for field in &netcdf_step.fields {
            let (netcdf_path, variable) = netcdf_field_mapping(
                &netcdf_run, netcdf_step.index, field)
                .unwrap()
                .unwrap();
            let (shape, values) = read_netcdf_field(&netcdf_path, &variable).unwrap();
            let npy_path = Path::new(&npy_run)
                .join(format!("step_{:06}", npy_step.index))
                .join(format!("{field}.npy"));
            let bytes = fs::read(npy_path).unwrap();
            let layout = crate::npy::parse_f32_layout(&bytes).unwrap();
            let body = &bytes[layout.data_offset..layout.data_offset + layout.decoded_bytes];
            let expected: Vec<f32> = body
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            assert_eq!(shape, layout.shape);
            assert_eq!(values, expected, "field {field}");
        }
    }

    #[test]
    #[ignore]
    fn simulator_hierarchy_v2_history_reopens_with_bound_lod_dependencies() {
        let run = std::env::var("NIMBUS_TEST_HIERARCHY_V2_RUN")
            .expect("set NIMBUS_TEST_HIERARCHY_V2_RUN");
        let steps = list_step_dirs_blocking(run.clone()).unwrap();
        assert!(!steps.is_empty());
        assert!(steps.iter().all(|step| step.publication_state == StepPublicationState::Complete));
        assert!(steps.iter().all(|step| step.fields.iter()
            .all(|field| !is_field_lod_payload_stem(field))));
        let mut validated_lods = 0_usize;
        for step in &steps {
            let generation = step.publication_generation.as_deref().unwrap();
            for (field, info) in &step.field_brick_hierarchies {
                assert_eq!(info.schema_version, 2);
                let hierarchy = validated_field_brick_hierarchy(
                    &run, step.index, field, Some(generation)).unwrap();
                let levels = hierarchy["levels"].as_array().unwrap();
                if levels.len() > 1 {
                    validated_lods += levels.len() - 1;
                    let source_shape = json_usize_triplet(hierarchy.get("sourceShape")).unwrap();
                    let source_offset = json_usize_triplet(hierarchy.get("sourceIndexOffset")).unwrap();
                    let level = &levels[1];
                    let shape = json_usize_triplet(level.get("shape")).unwrap();
                    let stride = json_usize_triplet(level.get("stride")).unwrap();
                    let lod_extent = shape.map(|extent| extent.min(2));
                    let extent: [usize; 3] = std::array::from_fn(|axis| {
                        source_shape[axis].min(lod_extent[axis] * stride[axis])
                    });
                    let output_bytes = lod_extent.into_iter().product::<usize>() * 4;
                    let request = FieldWindowRequest {
                        schema_version: 2,
                        request_id: format!("qualified-hierarchy-window-{}-{field}", step.index),
                        aggregation: "cell_mean".into(),
                        origin: source_offset.to_vec(),
                        extent: extent.to_vec(),
                        stride: stride.to_vec(),
                        max_output_bytes: Some(output_bytes),
                        max_source_bytes: Some(output_bytes),
                    };
                    assert!(tauri::async_runtime::block_on(read_field_brick_window(
                        run.clone(), step.index, field.clone(), Some(generation.into()),
                        1, request,
                    )).is_ok());
                }
            }
        }
        assert!(validated_lods > 0);
    }

    #[test]
    fn current_payload_shape_must_match_manifest_grid() {
        let run = TestRunDir::new("shape_agreement");
        run.current();
        let step = run.step_with_field(6);
        fs::write(
            step.join("manifest.json"),
            current_manifest(6, GENERATION_A),
        )
        .unwrap();

        let grid = expected_manifest_grid_shape(&run.0, 6, "theta", Some(GENERATION_A))
            .unwrap()
            .unwrap();
        assert_eq!(grid, [8, 7, 6]);
        assert!(validate_manifest_shape(&[8, 7, 6], Some(grid)).is_ok());
        assert!(validate_manifest_shape(&[8, 7], Some(grid)).is_ok());
        assert!(validate_manifest_shape(&[8, 6, 7], Some(grid)).is_err());
        assert!(validate_manifest_shape(&[8], Some(grid)).is_err());
        assert!(expected_manifest_grid_shape(&run.0, 6, "pressure", Some(GENERATION_A)).is_err());
    }

    #[test]
    fn current_manifest_accepts_a_large_logical_grid_for_bounded_windows() {
        let run = TestRunDir::new("large_logical_shape");
        run.current();
        let step = run.step_with_field(6);
        let manifest = current_manifest(6, GENERATION_A)
            .replace("\"nr\":8", "\"nr\":8192")
            .replace("\"nth\":7", "\"nth\":8192")
            .replace("\"nz\":6", "\"nz\":2")
            .replace(
                "\"upper\":[2000.0,2100.0,600.0]",
                "\"upper\":[2048000.0,2457600.0,200.0]",
            )
            .replace("\"upper\":[6,5,5]", "\"upper\":[8190,8190,1]");
        fs::write(step.join("manifest.json"), manifest).unwrap();

        let grid = expected_manifest_grid_shape(&run.0, 6, "theta", Some(GENERATION_A))
            .unwrap()
            .unwrap();
        assert_eq!(grid, [8192, 8192, 2]);
        assert!(grid.into_iter().product::<usize>() > crate::npy::MAX_FIELD_ELEMENTS);
    }

    #[test]
    fn mismatched_step_generation_fails_closed() {
        let run = TestRunDir::new("generation_mismatch");
        run.current();
        let step = run.step_with_field(4);
        fs::write(
            step.join("manifest.json"),
            current_manifest(4, GENERATION_B),
        )
        .unwrap();

        let steps = list_step_dirs_blocking(run.0.to_string_lossy().into_owned()).unwrap();
        assert_eq!(
            steps[0].publication_state,
            StepPublicationState::InvalidManifest
        );
        assert_eq!(
            steps[0].publication_generation.as_deref(),
            Some(GENERATION_A)
        );
        assert!(steps[0].grid.is_none());
        assert!(ensure_expected_publication_generation(&run.0, 4, Some(GENERATION_A)).is_err());
    }

    #[test]
    fn initializing_generation_invalidates_old_steps_and_artifacts() {
        let run = TestRunDir::new("initializing_barrier");
        run.current();
        let step = run.step_with_field(5);
        fs::write(
            step.join("manifest.json"),
            current_manifest(5, GENERATION_A),
        )
        .unwrap();
        fs::write(run.0.join("trajectories.npy"), b"old-generation").unwrap();

        assert!(ensure_expected_publication_generation(&run.0, 5, Some(GENERATION_A)).is_ok());
        assert!(run_artifact_exists(
            &run.0,
            "trajectories.npy",
            Some(GENERATION_A)
        ));

        run.current_generation(GENERATION_B, "initializing");
        assert!(ensure_expected_publication_generation(&run.0, 5, Some(GENERATION_A)).is_err());
        assert!(!run_artifact_exists(
            &run.0,
            "trajectories.npy",
            Some(GENERATION_A)
        ));
        assert!(read_run_artifact_bytes(&run.0, "trajectories.npy", Some(GENERATION_A)).is_err());

        let steps = list_step_dirs_blocking(run.0.to_string_lossy().into_owned()).unwrap();
        assert_eq!(steps[0].publication_state, StepPublicationState::Incomplete);
        assert_eq!(
            steps[0].publication_generation.as_deref(),
            Some(GENERATION_B)
        );
    }

    #[test]
    fn same_directory_ready_generation_rejects_old_artifact_identity() {
        let run = TestRunDir::new("artifact_generation_mismatch");
        run.current();
        let artifact = crate::zfp3d::encode_npy_f32(&[1], &[1.0]).unwrap();
        fs::write(run.0.join("trajectories.npy"), &artifact).unwrap();
        assert_eq!(
            read_run_artifact_bytes(&run.0, "trajectories.npy", Some(GENERATION_A)).unwrap(),
            artifact
        );

        run.current_generation(GENERATION_B, "ready");
        assert!(!run_artifact_exists(
            &run.0,
            "trajectories.npy",
            Some(GENERATION_A)
        ));
        assert!(read_run_artifact_bytes(&run.0, "trajectories.npy", Some(GENERATION_A)).is_err());
    }

    fn body(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 4);
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn field_window_envelope_aligns_float_and_geometry_bodies_without_copy() {
        let header = FieldWindowHeader {
            schema: "nimbus.studio.field-window",
            schema_version: FIELD_WINDOW_RESPONSE_SCHEMA_VERSION,
            request_id: "alignment".into(),
            aggregation: "cell_mean".into(),
            weighting: "physical_cell_volume",
            missing_data_policy: "propagate_non_finite",
            sample_location: "physical_volume_centroid",
            geometry_kind: "terrain_following",
            geometry_dtype: Some("<f8"),
            geometry_layout: Some("component_major"),
            geometry_components: Some(vec!["x_m", "y_m", "z_m", "volume_m3"]),
            source_shape: vec![1, 1, 1],
            origin: vec![0, 0, 0],
            extent: vec![1, 1, 1],
            stride: vec![1, 1, 1],
            output_shape: vec![1, 1, 1],
            element_count: 1,
            decoded_bytes: 4,
            geometry_decoded_bytes: 32,
            geometry_padding_bytes: 4,
            source_working_bytes: 64,
            source_working_accounting: "all_inspected_and_aggregation_bytes",
            dtype: "<f4",
            order: "C",
            source_format: "npy",
            cancellation_mode: "cooperative",
            non_interruptible_phase: None,
        };
        let encoded = encode_field_window(&header, &[0; 4], &[0; 32]).unwrap();
        let header_bytes = u32::from_le_bytes(encoded[8..12].try_into().unwrap()) as usize;
        let body_offset = 12 + header_bytes;
        assert_eq!(body_offset % 8, 0);
        assert_eq!(encoded.len() - body_offset, 40);
        let geometry_offset = body_offset + header.decoded_bytes + header.geometry_padding_bytes;
        assert_eq!(geometry_offset % 8, 0);
        assert_eq!(
            &encoded[body_offset + header.decoded_bytes..geometry_offset],
            &[0; 4]
        );
        let decoded: serde_json::Value = serde_json::from_slice(&encoded[12..body_offset]).unwrap();
        assert_eq!(decoded["requestId"], "alignment");
    }

    #[test]
    fn reduces_finite_values() {
        let r = reduce_f32_le(&body(&[1.0, 2.0, 3.0, 4.0]), None, None, None);
        assert_eq!(r.min, Some(1.0));
        assert_eq!(r.max, Some(4.0));
        assert_eq!(r.mean, Some(2.5));
        // rms = sqrt((1+4+9+16)/4) = sqrt(7.5)
        assert!((r.rms.unwrap() - 7.5_f64.sqrt()).abs() < 1e-12);
        assert_eq!(r.finite, 4);
        assert_eq!(r.total, 4);
    }

    #[test]
    fn parses_field_units_from_manifest() {
        let manifest = serde_json::json!({
            "fields": [
                {"field_id": "u", "units": "m/s"},
                {"field_id": "radar", "units": "mm^6/m^3"},
                {"field_id": "reflectivity_dbz", "units": "dBZ"},
                {"field_id": "tracer", "units": "1"},
                {"field_id": "missing_units"},
                {"field_id": "empty_units", "units": ""}
            ]
        });
        let units = parse_field_units(&manifest);
        assert_eq!(units.get("u").map(String::as_str), Some("m/s"));
        assert_eq!(units.get("radar").map(String::as_str), Some("mm^6/m^3"));
        assert_eq!(
            units.get("reflectivity_dbz").map(String::as_str),
            Some("dBZ")
        );
        // Dimensionless marker is preserved here; the UI suppresses the suffix.
        assert_eq!(units.get("tracer").map(String::as_str), Some("1"));
        // Entries missing `units`, or with an empty unit string, are skipped so
        // the UI shows a bare field name rather than "field ()".
        assert!(!units.contains_key("missing_units"));
        assert!(!units.contains_key("empty_units"));
    }

    #[test]
    fn field_units_empty_when_no_catalog() {
        let manifest = serde_json::json!({ "grid": { "nr": 10 } });
        assert!(parse_field_units(&manifest).is_empty());
    }

    #[test]
    fn parses_only_fail_closed_field_brick_hierarchy_references() {
        let manifest = serde_json::json!({
            "fields": [
                {
                    "field_id": "w",
                    "brick_hierarchy": {
                        "schema": "nimbus.field-brick-hierarchy",
                        "schema_version": 2,
                        "path": "w.bricks.json",
                        "content_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "native_brick_shape": [64, 64, 32],
                        "extrema_semantics": "stored_values",
                        "extrema_absolute_padding": 0.0,
                        "culling_safe": true
                    }
                },
                {
                    "field_id": "compressed",
                    "brick_hierarchy": {
                        "schema": "nimbus.field-brick-hierarchy",
                        "schema_version": 1,
                        "path": "compressed.bricks.json",
                        "content_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                        "native_brick_shape": [64, 64, 32],
                        "extrema_semantics": "pre_storage_source",
                        "extrema_absolute_padding": 0.0,
                        "culling_safe": false
                    }
                },
                {
                    "field_id": "unsafe",
                    "brick_hierarchy": {
                        "schema": "nimbus.field-brick-hierarchy",
                        "schema_version": 1,
                        "path": "../unsafe.bricks.json",
                        "content_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                        "native_brick_shape": [64, 64, 32],
                        "extrema_semantics": "stored_values",
                        "extrema_absolute_padding": 0.0,
                        "culling_safe": true
                    }
                }
            ]
        });
        let parsed = parse_field_brick_hierarchies(&manifest);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed["w"].native_brick_shape, [64, 64, 32]);
        assert_eq!(parsed["w"].schema_version, 2);
        assert!(parsed["w"].culling_safe);
        assert!(!parsed["compressed"].culling_safe);
        assert!(!parsed.contains_key("unsafe"));
    }

    #[test]
    fn reads_only_generation_bound_manifest_declared_brick_hierarchy() {
        let run = TestRunDir::new("brick_hierarchy_read");
        run.current();
        let step = run.step_with_field(3);
        let hierarchy_bytes = serde_json::to_vec(&serde_json::json!({
            "schema": "nimbus.field-brick-hierarchy",
            "schemaVersion": 1,
            "field": "theta",
            "publicationGeneration": GENERATION_A
        })).unwrap();
        let mut manifest: serde_json::Value =
            serde_json::from_str(&current_manifest(3, GENERATION_A)).unwrap();
        manifest["fields"][0]["brick_hierarchy"] = serde_json::json!({
            "schema": "nimbus.field-brick-hierarchy",
            "schema_version": 1,
            "path": "theta.bricks.json",
            "content_sha256": sha256_hex(&hierarchy_bytes),
            "native_brick_shape": [64, 64, 32],
            "extrema_semantics": "stored_values",
            "extrema_absolute_padding": 0.0,
            "culling_safe": true
        });
        fs::write(step.join("manifest.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();
        fs::write(step.join("theta.bricks.json"), hierarchy_bytes).unwrap();
        let value = tauri::async_runtime::block_on(read_field_brick_hierarchy(
            run.0.to_string_lossy().into_owned(),
            3,
            "theta".into(),
            Some(GENERATION_A.into()),
        )).unwrap();
        assert_eq!(value["field"], "theta");
        assert!(tauri::async_runtime::block_on(read_field_brick_hierarchy(
            run.0.to_string_lossy().into_owned(),
            3,
            "theta".into(),
            Some(GENERATION_B.into()),
        )).is_err());
    }

    #[test]
    fn durable_field_lod_is_hash_bound_and_windowed_after_hierarchy_validation() {
        let run = TestRunDir::new("brick_lod_read");
        run.current();
        let step = run.step_with_field(3);
        let lod_values: Vec<f32> = (0..27).map(|value| value as f32).collect();
        let lod_bytes = crate::zfp3d::encode_npy_f32(&[3, 3, 3], &lod_values).unwrap();
        fs::write(step.join("theta.lod1.npy"), &lod_bytes).unwrap();
        let hierarchy_bytes = serde_json::to_vec(&serde_json::json!({
            "schema": "nimbus.field-brick-hierarchy",
            "schemaVersion": 2,
            "field": "theta",
            "publicationGeneration": GENERATION_A,
            "sourceShape": [6, 5, 6],
            "sourceIndexOffset": [1, 1, 0],
            "levels": [
                {"level": 0, "source": {"kind": "primary_field"}},
                {
                    "level": 1,
                    "stride": [2, 2, 2],
                    "shape": [3, 3, 3],
                    "source": {
                        "kind": "npy_lod",
                        "path": "theta.lod1.npy",
                        "contentSha256": sha256_hex(&lod_bytes),
                        "aggregation": "recursive_cell_mean_2x2x2"
                    }
                }
            ]
        })).unwrap();
        let mut manifest: serde_json::Value =
            serde_json::from_str(&current_manifest(3, GENERATION_A)).unwrap();
        manifest["fields"][0]["brick_hierarchy"] = serde_json::json!({
            "schema": "nimbus.field-brick-hierarchy",
            "schema_version": 2,
            "path": "theta.bricks.json",
            "content_sha256": sha256_hex(&hierarchy_bytes),
            "native_brick_shape": [64, 64, 32],
            "extrema_semantics": "stored_values",
            "extrema_absolute_padding": 0.0,
            "culling_safe": true
        });
        fs::write(step.join("manifest.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();
        fs::write(step.join("theta.bricks.json"), hierarchy_bytes).unwrap();

        tauri::async_runtime::block_on(read_field_brick_hierarchy(
            run.0.to_string_lossy().into_owned(),
            3,
            "theta".into(),
            Some(GENERATION_A.into()),
        )).unwrap();
        let request = FieldWindowRequest {
            schema_version: 2,
            request_id: "brick-lod-window-1".into(),
            aggregation: "cell_mean".into(),
            origin: vec![1, 1, 0],
            extent: vec![6, 5, 6],
            stride: vec![2, 2, 2],
            max_output_bytes: Some(108),
            max_source_bytes: Some(108),
        };
        assert!(tauri::async_runtime::block_on(read_field_brick_window(
            run.0.to_string_lossy().into_owned(),
            3,
            "theta".into(),
            Some(GENERATION_A.into()),
            1,
            request,
        )).is_ok());

        fs::write(step.join("theta.lod1.npy"), b"changed-after-validation").unwrap();
        let changed_request = FieldWindowRequest {
            request_id: "brick-lod-window-2".into(),
            ..registry_test_request("unused")
        };
        assert!(tauri::async_runtime::block_on(read_field_brick_window(
            run.0.to_string_lossy().into_owned(),
            3,
            "theta".into(),
            Some(GENERATION_A.into()),
            1,
            changed_request,
        )).is_err());
    }

    #[test]
    fn terrain_field_lod_geometry_is_hash_bound_and_windowed_atomically() {
        let run = TestRunDir::new("terrain_brick_lod_read");
        run.current();
        let step = run.step_with_field(3);
        let lod_values: Vec<f32> = (0..27).map(|value| value as f32).collect();
        let lod_bytes = crate::zfp3d::encode_npy_f32(&[3, 3, 3], &lod_values).unwrap();
        fs::write(step.join("theta.lod1.npy"), &lod_bytes).unwrap();
        let geometry_names = [
            "physical_x_center_m.lod1.npy", "physical_y_center_m.lod1.npy",
            "physical_z_center_m.lod1.npy", "physical_volume_m3.lod1.npy",
        ];
        let geometry_payloads: Vec<Vec<u8>> = geometry_names.iter().enumerate()
            .map(|(component, _)| f64_npy_values(
                [3, 3, 3],
                &vec![if component == 3 { 1.0 } else { component as f64 }; 27],
            ))
            .collect();
        for (name, bytes) in geometry_names.iter().zip(&geometry_payloads) {
            fs::write(step.join(name), bytes).unwrap();
        }
        let geometry_descriptor = serde_json::json!({
            "kind": "physical_volume_centroids",
            "xM": {"path": geometry_names[0], "contentSha256": sha256_hex(&geometry_payloads[0])},
            "yM": {"path": geometry_names[1], "contentSha256": sha256_hex(&geometry_payloads[1])},
            "zM": {"path": geometry_names[2], "contentSha256": sha256_hex(&geometry_payloads[2])},
            "volumeM3": {"path": geometry_names[3], "contentSha256": sha256_hex(&geometry_payloads[3])}
        });
        let hierarchy_bytes = serde_json::to_vec(&serde_json::json!({
            "schema": "nimbus.field-brick-hierarchy",
            "schemaVersion": 3,
            "field": "theta",
            "publicationGeneration": GENERATION_A,
            "sourceShape": [6, 5, 6],
            "sourceIndexOffset": [1, 1, 0],
            "levels": [
                {"level": 0, "source": {"kind": "primary_field"}},
                {"level": 1, "stride": [2, 2, 2], "shape": [3, 3, 3], "source": {
                    "kind": "npy_lod",
                    "path": "theta.lod1.npy",
                    "contentSha256": sha256_hex(&lod_bytes),
                    "aggregation": "recursive_cell_mean_2x2x2",
                    "geometry": geometry_descriptor
                }}
            ]
        })).unwrap();
        let mut manifest: serde_json::Value =
            serde_json::from_str(&current_manifest(3, GENERATION_A)).unwrap();
        manifest["fields"][0]["brick_hierarchy"] = serde_json::json!({
            "schema": "nimbus.field-brick-hierarchy",
            "schema_version": 3,
            "path": "theta.bricks.json",
            "content_sha256": sha256_hex(&hierarchy_bytes),
            "native_brick_shape": [64, 64, 32],
            "extrema_semantics": "stored_values",
            "extrema_absolute_padding": 0.0,
            "culling_safe": true
        });
        fs::write(step.join("manifest.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();
        fs::write(step.join("theta.bricks.json"), hierarchy_bytes).unwrap();
        let listed = list_step_dirs_blocking(run.0.to_string_lossy().into_owned()).unwrap();
        assert_eq!(listed[0].publication_state, StepPublicationState::Complete);
        assert_eq!(listed[0].fields, vec!["theta"]);
        tauri::async_runtime::block_on(read_field_brick_hierarchy(
            run.0.to_string_lossy().into_owned(), 3, "theta".into(), Some(GENERATION_A.into()),
        )).unwrap();
        let request = FieldWindowRequest {
            schema_version: 2,
            request_id: "terrain-brick-lod-window-1".into(),
            aggregation: "cell_mean".into(),
            origin: vec![1, 1, 0],
            extent: vec![6, 5, 6],
            stride: vec![2, 2, 2],
            max_output_bytes: Some(976),
            max_source_bytes: Some(972),
        };
        assert!(tauri::async_runtime::block_on(read_field_brick_window(
            run.0.to_string_lossy().into_owned(), 3, "theta".into(),
            Some(GENERATION_A.into()), 1, request,
        )).is_ok());
        fs::remove_file(step.join(geometry_names[2])).unwrap();
        let changed = FieldWindowRequest {
            schema_version: 2,
            request_id: "terrain-brick-lod-window-2".into(),
            aggregation: "cell_mean".into(),
            origin: vec![1, 1, 0],
            extent: vec![6, 5, 6],
            stride: vec![2, 2, 2],
            max_output_bytes: Some(976),
            max_source_bytes: Some(972),
        };
        assert!(tauri::async_runtime::block_on(read_field_brick_window(
            run.0.to_string_lossy().into_owned(), 3, "theta".into(),
            Some(GENERATION_A.into()), 1, changed,
        )).is_err());
    }

    #[test]
    fn hierarchy_v2_dependencies_are_not_fields_and_gate_reopen_completeness() {
        let run = TestRunDir::new("brick_lod_reopen");
        run.current();
        let step = run.step_with_field(3);
        let lod_bytes = crate::zfp3d::encode_npy_f32(&[3, 3, 3], &[2.0; 27]).unwrap();
        fs::write(step.join("theta.lod1.npy"), &lod_bytes).unwrap();
        let hierarchy_bytes = serde_json::to_vec(&serde_json::json!({
            "schema": "nimbus.field-brick-hierarchy",
            "schemaVersion": 2,
            "field": "theta",
            "publicationGeneration": GENERATION_A,
            "sourceShape": [6, 5, 6],
            "sourceIndexOffset": [1, 1, 0],
            "levels": [
                {"level": 0, "source": {"kind": "primary_field"}},
                {"level": 1, "source": {
                    "kind": "npy_lod",
                    "path": "theta.lod1.npy",
                    "contentSha256": sha256_hex(&lod_bytes),
                    "aggregation": "recursive_cell_mean_2x2x2"
                }}
            ]
        })).unwrap();
        let mut manifest: serde_json::Value =
            serde_json::from_str(&current_manifest(3, GENERATION_A)).unwrap();
        manifest["fields"][0]["brick_hierarchy"] = serde_json::json!({
            "schema": "nimbus.field-brick-hierarchy",
            "schema_version": 2,
            "path": "theta.bricks.json",
            "content_sha256": sha256_hex(&hierarchy_bytes),
            "native_brick_shape": [64, 64, 32],
            "extrema_semantics": "stored_values",
            "extrema_absolute_padding": 0.0,
            "culling_safe": true
        });
        fs::write(step.join("manifest.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();
        fs::write(step.join("theta.bricks.json"), hierarchy_bytes).unwrap();

        let listed = list_step_dirs_blocking(run.0.to_string_lossy().into_owned()).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].publication_state, StepPublicationState::Complete);
        assert_eq!(listed[0].fields, vec!["theta"]);
        assert_eq!(listed[0].field_brick_hierarchies["theta"].schema_version, 2);

        fs::remove_file(step.join("theta.lod1.npy")).unwrap();
        assert!(!field_brick_hierarchy_dependencies_are_complete(&manifest, &step));
    }

    #[test]
    fn parses_only_complete_finite_field_compression_contracts() {
        let manifest = serde_json::json!({
            "fields": [
                {
                    "field_id": "theta",
                    "compression": {
                        "codec": "zfp",
                        "mode": "accuracy",
                        "mode_parameter": 0.001,
                        "mode_parameter_units": "absolute_field_units",
                        "error_contract": "zfp_absolute_accuracy_before_prefilters",
                        "sparse_threshold": 0.0,
                        "float16_prefilter": false,
                        "decoded_dtype": "float32"
                    }
                },
                {"field_id": "incomplete", "compression": {"codec": "zfp"}},
                {"field_id": "lossless", "units": "K"}
            ]
        });
        let compression = parse_field_compression(&manifest);
        let theta = compression.get("theta").expect("complete contract");
        assert_eq!(theta.mode, "accuracy");
        assert_eq!(theta.mode_parameter, 0.001);
        assert_eq!(theta.mode_parameter_units, "absolute_field_units");
        assert_eq!(theta.decoded_dtype, "float32");
        assert!(!compression.contains_key("incomplete"));
        assert!(!compression.contains_key("lossless"));
    }

    #[test]
    fn parses_only_bounded_zfp_field_execution_provenance() {
        let valid = serde_json::json!({
            "data_product": {
                "lossy_compression_contract": "nimbus.zfp.per_field.v1",
                "field_execution_policy": "field_parallel",
                "field_workers": 4,
                "field_scratch_budget_mib": 128
            }
        });
        let parsed = parse_compression_execution(&valid).expect("bounded policy");
        assert_eq!(parsed.field_execution_policy, "field_parallel");
        assert_eq!(parsed.field_workers, 4);
        assert_eq!(parsed.field_scratch_budget_mib, 128);

        let mut invalid = valid;
        invalid["data_product"]["field_workers"] = serde_json::json!(17);
        assert!(parse_compression_execution(&invalid).is_none());
    }

    fn test_mixed_zfp_v2_header(version: u16, mode: u16, parameter: f64, flags: u8) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"ZFP3");
        bytes.extend_from_slice(&version.to_le_bytes());
        bytes.extend_from_slice(&mode.to_le_bytes());
        for dimension in [12i32, 12, 48] {
            bytes.extend_from_slice(&dimension.to_le_bytes());
        }
        bytes.extend_from_slice(&parameter.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&[flags, 0, 0, 0]);
        bytes.extend_from_slice(if version == 4 { b"ZB4D" } else { b"ZB3D" });
        bytes.extend_from_slice(&[1, 2]);
        let entry_bytes = if version == 4 { 40 } else { 8 };
        for offset in [44u64, 45u64] {
            bytes.extend_from_slice(&offset.to_le_bytes());
            bytes.extend(std::iter::repeat(0).take(entry_bytes - 8));
        }
        bytes
    }

    fn test_mixed_zfp_v2_manifest() -> serde_json::Value {
        serde_json::json!({
            "data_product": {
                "storage_format": "zfp",
                "compression_contract": "nimbus.zfp.per_field.v2",
                "field_execution_policy": "serial",
                "field_workers": 1,
                "field_scratch_budget_mib": 64
            },
            "radar_provenance": {"products": ["rhohv"]},
            "fields": [
                {"field_id": "u", "compression": {
                    "codec": "zfp", "mode": "accuracy", "mode_parameter": 0.0001,
                    "mode_parameter_units": "absolute_field_units",
                    "error_contract": "zfp_absolute_accuracy_before_prefilters",
                    "frame_role": "keyframe", "parent_step_indices": [],
                    "sparse_threshold": 0, "float16_prefilter": false,
                    "decoded_dtype": "float32", "container_version": 3,
                    "spatial_layout": "independently_decodable_bricks",
                    "brick_shape": [64, 64, 32]
                }},
                {"field_id": "rhohv", "compression": {
                    "contract_version": 2, "codec": "zfp", "mode": "reversible",
                    "mode_parameter": 0, "mode_parameter_units": "none",
                    "error_contract": "bitwise_float32_reconstruction",
                    "frame_role": "keyframe", "parent_step_indices": [],
                    "sparse_threshold": 0, "float16_prefilter": false,
                    "decoded_dtype": "float32", "container_version": 4,
                    "spatial_layout": "independently_decodable_bricks",
                    "brick_shape": [64, 64, 32]
                }}
            ]
        })
    }

    #[test]
    fn parses_strict_mixed_zfp_v2_execution_and_field_contracts() {
        let run = TestRunDir::new("mixed_zfp_v2_parser");
        let step = run.0.join("step_000000");
        fs::create_dir_all(&step).unwrap();
        fs::write(step.join("u.zfp3d"),
            test_mixed_zfp_v2_header(3, 0, 0.0001, 0)).unwrap();
        fs::write(step.join("rhohv.zfp3d"),
            test_mixed_zfp_v2_header(4, 3, 0.0, 0)).unwrap();
        let manifest = test_mixed_zfp_v2_manifest();
        fs::write(step.join("manifest.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();
        let fields = vec!["u".to_string(), "rhohv".to_string()];
        let execution = parse_compression_execution(&manifest).expect("v2 execution");
        assert_eq!(execution.field_execution_policy, "serial");
        assert_eq!(execution.field_workers, 1);
        assert_eq!(execution.field_scratch_budget_mib, 64);
        assert!(mixed_zfp_v2_is_complete(&manifest, &fields, 0, &step));
        let parsed = parse_field_compression(&manifest);
        assert_eq!(parsed["u"].mode, "accuracy");
        assert_eq!(parsed["rhohv"].mode, "reversible");
        assert_eq!(parsed["rhohv"].mode_parameter, 0.0);
        let mut complete: serde_json::Value =
            serde_json::from_str(&current_manifest(0, GENERATION_A)).unwrap();
        complete["data_product"] = manifest["data_product"].clone();
        complete["radar_provenance"] = manifest["radar_provenance"].clone();
        complete["fields"] = manifest["fields"].clone();
        complete["published_fields"] = serde_json::json!(["u", "rhohv"]);
        fs::write(step.join("manifest.json"), serde_json::to_vec(&complete).unwrap()).unwrap();
        assert!(validate_current_step_manifest(
            &complete, GENERATION_A, 0, "step_000000", &fields,
            &step, LEGACY_OUTPUT_MANIFEST_VERSION, "history"
        ));
        complete["fields"][1]["compression"]["mode"] = serde_json::json!("accuracy");
        fs::write(step.join("manifest.json"), serde_json::to_vec(&complete).unwrap()).unwrap();
        assert!(!validate_current_step_manifest(
            &complete, GENERATION_A, 0, "step_000000", &fields,
            &step, LEGACY_OUTPUT_MANIFEST_VERSION, "history"
        ));
    }

    #[test]
    fn mixed_zfp_v2_rejects_partial_unknown_and_mismatched_provenance() {
        let run = TestRunDir::new("mixed_zfp_v2_negative");
        let step = run.0.join("step_000000");
        fs::create_dir_all(&step).unwrap();
        let u_path = step.join("u.zfp3d");
        let radar_path = step.join("rhohv.zfp3d");
        let u_header = test_mixed_zfp_v2_header(3, 0, 0.0001, 0);
        let radar_header = test_mixed_zfp_v2_header(4, 3, 0.0, 0);
        fs::write(&u_path, &u_header).unwrap();
        fs::write(&radar_path, &radar_header).unwrap();
        let valid = test_mixed_zfp_v2_manifest();
        fs::write(step.join("manifest.json"), serde_json::to_vec(&valid).unwrap()).unwrap();
        let fields = vec!["u".to_string(), "rhohv".to_string()];
        assert!(mixed_zfp_v2_is_complete(&valid, &fields, 0, &step));
        let mut failures = Vec::new();
        let mut push = |path: &str, replacement: serde_json::Value| {
            let mut invalid = valid.clone();
            *invalid.pointer_mut(path).expect("v2 test contract path") = replacement;
            failures.push(invalid);
        };
        push("/data_product/compression_contract", serde_json::json!("nimbus.zfp.per_field.v3"));
        push("/data_product/field_execution_policy", serde_json::json!("shared_stream"));
        push("/data_product/field_workers", serde_json::json!(0));
        push("/data_product/field_workers", serde_json::json!(17));
        push("/data_product/field_scratch_budget_mib", serde_json::json!(0));
        push("/data_product/field_scratch_budget_mib", serde_json::json!(4097));
        push("/radar_provenance/products", serde_json::json!([]));
        push("/radar_provenance/products", serde_json::json!(["zdr"]));
        push("/radar_provenance/products", serde_json::json!(["rhohv", "rhohv"]));
        push("/fields/1/compression/mode", serde_json::json!("accuracy"));
        push("/fields/1/compression/contract_version", serde_json::json!(1));
        push("/fields/1/compression/mode_parameter", serde_json::json!(0.01));
        push("/fields/1/compression/container_version", serde_json::json!(3));
        push("/fields/1/compression/frame_role", serde_json::json!("simple_delta"));
        push("/fields/1/compression/float16_prefilter", serde_json::json!(true));
        push("/fields/0/compression/mode", serde_json::json!("rate"));
        push("/fields/0/compression/container_version", serde_json::json!(4));
        push("/fields/0/compression/mode_parameter", serde_json::json!(0));
        push("/fields/0/compression", serde_json::Value::Null);
        for invalid in failures {
            fs::write(step.join("manifest.json"), serde_json::to_vec(&invalid).unwrap()).unwrap();
            assert!(!mixed_zfp_v2_is_complete(&invalid, &fields, 0, &step));
        }
        let mut with_legacy_tag = valid.clone();
        with_legacy_tag["data_product"]["lossy_compression_contract"] =
            serde_json::json!("nimbus.zfp.per_field.v1");
        fs::write(step.join("manifest.json"), serde_json::to_vec(&with_legacy_tag).unwrap()).unwrap();
        assert!(!mixed_zfp_v2_is_complete(&with_legacy_tag, &fields, 0, &step));
        let mut duplicate = valid.clone();
        let duplicate_field = duplicate["fields"][1].clone();
        duplicate["fields"].as_array_mut().unwrap().push(duplicate_field);
        fs::write(step.join("manifest.json"), serde_json::to_vec(&duplicate).unwrap()).unwrap();
        assert!(!mixed_zfp_v2_is_complete(&duplicate, &fields, 0, &step));
        fs::write(step.join("manifest.json"), serde_json::to_vec(&valid).unwrap()).unwrap();
        assert!(!mixed_zfp_v2_is_complete(&valid, &["rhohv".into()], 0, &step));
        assert!(!mixed_zfp_v2_is_complete(&valid, &["u".into()], 0, &step));
        fs::write(&radar_path, test_mixed_zfp_v2_header(4, 0, 0.0, 0)).unwrap();
        assert!(!mixed_zfp_v2_is_complete(&valid, &fields, 0, &step));
        fs::write(&radar_path, &radar_header).unwrap();
        fs::write(&u_path, test_mixed_zfp_v2_header(3, 0, 0.0002, 0)).unwrap();
        assert!(!mixed_zfp_v2_is_complete(&valid, &fields, 0, &step));
        fs::write(&u_path, &u_header[..u_header.len() - 1]).unwrap();
        assert!(!mixed_zfp_v2_is_complete(&valid, &fields, 0, &step));
    }

    #[test]
    fn mixed_v2_exact_decimal_lexeme_rejects_one_ulp_and_ambiguous_json() {
        let run = TestRunDir::new("mixed_v2_decimal_lexeme");
        let step = run.0.join("step_000000");
        fs::create_dir_all(&step).unwrap();
        let mut manifest = test_mixed_zfp_v2_manifest();
        manifest["fields"][0]["field_id"] = serde_json::json!("rho");
        manifest["fields"][0]["compression"]["mode_parameter"] = serde_json::json!(1e-6);
        fs::write(step.join("rho.zfp3d"), test_mixed_zfp_v2_header(3, 0, 1e-6, 0)).unwrap();
        fs::write(step.join("rhohv.zfp3d"), test_mixed_zfp_v2_header(4, 3, 0.0, 0)).unwrap();
        let fields = vec!["rho".to_string(), "rhohv".to_string()];
        let serialized = serde_json::to_string(&manifest).unwrap();
        assert!(serialized.contains("\"mode_parameter\":1e-6"));
        let exact = serialized.replacen(
            "\"mode_parameter\":1e-6",
            "\"mode_parameter\":9.9999999999999995e-07", 1);
        fs::write(step.join("manifest.json"), &exact).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&exact).unwrap();
        assert!(mixed_zfp_v2_is_complete(&parsed, &fields, 0, &step));

        let changed_lexeme = exact.replacen(
            "9.9999999999999995e-07", "1.0000000000000002e-06", 1);
        fs::write(step.join("manifest.json"), &changed_lexeme).unwrap();
        let changed: serde_json::Value = serde_json::from_str(&changed_lexeme).unwrap();
        assert!(!mixed_zfp_v2_is_complete(&changed, &fields, 0, &step));
        fs::write(step.join("manifest.json"), &exact).unwrap();
        fs::write(step.join("rho.zfp3d"), test_mixed_zfp_v2_header(
            3, 0, f64::from_bits((1e-6f64).to_bits() + 1), 0)).unwrap();
        assert!(!mixed_zfp_v2_is_complete(&parsed, &fields, 0, &step));
        fs::write(step.join("rho.zfp3d"), test_mixed_zfp_v2_header(3, 0, 1e-6, 0)).unwrap();

        let reordered = {
            let mut value = manifest.clone();
            value["fields"].as_array_mut().unwrap().reverse();
            let serialized = serde_json::to_string(&value).unwrap();
            assert!(serialized.contains("\"mode_parameter\":1e-6"));
            serialized.replacen(
                "\"mode_parameter\":1e-6",
                "\"mode_parameter\":9.9999999999999995e-07", 1)
        };
        fs::write(step.join("manifest.json"), &reordered).unwrap();
        let reordered_value: serde_json::Value = serde_json::from_str(&reordered).unwrap();
        assert!(mixed_zfp_v2_is_complete(&reordered_value, &fields, 0, &step));

        for invalid_text in [
            exact.replacen("\"field_id\":\"rho\"",
                "\"field_id\":\"rho\",\"field_id\":\"rho\"", 1),
            exact.replacen("\"mode_parameter\":9.9999999999999995e-07",
                "\"mode_parameter\":9.9999999999999995e-07,\"mode_parameter\":9.9999999999999995e-07", 1),
            exact.replacen("\"mode_parameter\":9.9999999999999995e-07",
                "\"mode_\\u0070arameter\":9.9999999999999995e-07", 1),
            exact.replacen("\"compression\":{",
                "\"compression\":{\"field_id\":\"intruder\",", 1),
            exact.replacen("\"mode_parameter\":9.9999999999999995e-07",
                "\"mode_parameter\":\"9.9999999999999995e-07\"", 1),
        ] {
            assert_ne!(invalid_text, exact);
            fs::write(step.join("manifest.json"), &invalid_text).unwrap();
            let invalid: serde_json::Value = serde_json::from_str(&invalid_text).unwrap();
            assert!(!mixed_zfp_v2_is_complete(&invalid, &fields, 0, &step));
        }
    }

    #[test]
    fn current_zfp_v1_manifest_requires_exact_compressed_or_missing_value_npy_storage() {
        let run = TestRunDir::new("zfp_v1_field_storage");
        let step = run.0.join("step_000000");
        fs::create_dir_all(&step).unwrap();
        let compressed = test_mixed_zfp_v2_header(3, 0, 0.001, 0);
        fs::write(step.join("theta.zfp3d"), &compressed).unwrap();
        let missing = crate::zfp3d::encode_npy_f32(&[8, 7], &vec![f32::NAN; 56]).unwrap();
        fs::write(step.join("k_index.npy"), &missing).unwrap();
        let valid = serde_json::json!({
            "data_product": {
                "storage_format": "zfp",
                "lossy_compression_contract": "nimbus.zfp.per_field.v1",
                "field_execution_policy": "serial",
                "field_workers": 1,
                "field_scratch_budget_mib": 64
            },
            "fields": [
                {"field_id": "theta", "file_suffix": "theta",
                 "file_pattern": "theta.zfp3d", "compression": {
                    "codec": "zfp", "mode": "accuracy", "mode_parameter": 0.001,
                    "mode_parameter_units": "absolute_field_units",
                    "error_contract": "zfp_absolute_accuracy_before_prefilters",
                    "frame_role": "keyframe", "parent_step_indices": [],
                    "sparse_threshold": 0, "float16_prefilter": false,
                    "decoded_dtype": "float32", "container_version": 3,
                    "spatial_layout": "independently_decodable_bricks",
                    "brick_shape": [64, 64, 32]}},
                {"field_id": "k_index", "file_suffix": "k_index",
                 "file_pattern": "k_index.npy", "storage_format": "npy_2d",
                 "missing_value": "NaN"}
            ]
        });
        let fields = vec!["theta".to_string(), "k_index".to_string()];
        assert!(zfp_v1_field_storage_is_complete(&valid, &fields, &step));
        for (path, replacement) in [
            ("/fields/1/storage_format", serde_json::json!("zfp")),
            ("/fields/1/file_pattern", serde_json::json!("k_index.zfp3d")),
            ("/fields/1/missing_value", serde_json::json!("none")),
            ("/fields/0/file_pattern", serde_json::json!("theta.npy")),
            ("/fields/0/compression/mode", serde_json::json!("rate")),
            ("/fields/0/compression/container_version", serde_json::json!(4)),
            ("/data_product/lossy_compression_contract", serde_json::json!("unknown")),
        ] {
            let mut invalid = valid.clone();
            *invalid.pointer_mut(path).unwrap() = replacement;
            assert!(!zfp_v1_field_storage_is_complete(&invalid, &fields, &step), "{path}");
        }
        let mut compression_on_fallback = valid.clone();
        compression_on_fallback["fields"][1]["compression"] = valid["fields"][0]["compression"].clone();
        assert!(!zfp_v1_field_storage_is_complete(&compression_on_fallback, &fields, &step));
        let mut missing_compression = valid.clone();
        missing_compression["fields"][0].as_object_mut().unwrap().remove("compression");
        assert!(!zfp_v1_field_storage_is_complete(&missing_compression, &fields, &step));
        let mut unexpected_storage = valid.clone();
        unexpected_storage["fields"][0]["storage_format"] = serde_json::json!("zfp");
        assert!(!zfp_v1_field_storage_is_complete(&unexpected_storage, &fields, &step));
        let mut duplicate = valid.clone();
        duplicate["fields"].as_array_mut().unwrap().push(valid["fields"][0].clone());
        assert!(!zfp_v1_field_storage_is_complete(&duplicate, &fields, &step));
        let mut ghost = valid.clone();
        ghost["fields"].as_array_mut().unwrap().push(serde_json::json!({
            "field_id": "ghost", "compression": valid["fields"][0]["compression"]
        }));
        assert!(!zfp_v1_field_storage_is_complete(&ghost, &fields, &step));

        fs::write(step.join("k_index.zfp3d"), &compressed).unwrap();
        assert!(!zfp_v1_field_storage_is_complete(&valid, &fields, &step));
        fs::remove_file(step.join("k_index.zfp3d")).unwrap();
        fs::write(step.join("theta.npy"), &missing).unwrap();
        assert!(!zfp_v1_field_storage_is_complete(&valid, &fields, &step));
        fs::remove_file(step.join("theta.npy")).unwrap();
        fs::write(step.join("k_index.npy"),
            crate::zfp3d::encode_npy_f32(&[8, 7, 1], &vec![f32::NAN; 56]).unwrap()).unwrap();
        assert!(!zfp_v1_field_storage_is_complete(&valid, &fields, &step));
        fs::write(step.join("k_index.npy"), &missing).unwrap();
        fs::write(step.join("theta.zfp3d"), test_mixed_zfp_v2_header(3, 3, 0.001, 0)).unwrap();
        assert!(!zfp_v1_field_storage_is_complete(&valid, &fields, &step));
    }

    #[test]
    #[ignore = "requires hash-bound current native mixed-v1 ZFP/NPY history"]
    fn real_mixed_v1_missing_value_fallback_reopens_and_rejects_corruption() {
        let binary = PathBuf::from(std::env::var("NIMBUS_TEST_ZFP_BINARY")
            .expect("exact mixed-v1 simulator binary"));
        let expected_sha = std::env::var("NIMBUS_TEST_ZFP_BINARY_SHA256")
            .expect("exact mixed-v1 simulator binary SHA");
        assert_eq!(crate::platform::sha256_file(&binary).unwrap(),
                   format!("sha256:{expected_sha}"));
        let zfp_run = PathBuf::from(std::env::var("NIMBUS_TEST_ZFP_RUN")
            .expect("mixed-v1 ZFP history"));
        let npy_run = PathBuf::from(std::env::var("NIMBUS_TEST_NPY_RUN")
            .expect("mixed-v1 NPY control"));
        let expected_fallback: std::collections::BTreeSet<&str> =
            ["k_index", "lifted_index", "showalter_index", "total_totals"]
                .into_iter().collect();
        for step_index in [0u32, 1u32] {
            let name = format!("step_{step_index:06}");
            let step = zfp_run.join(&name);
            let manifest: serde_json::Value = serde_json::from_slice(
                &fs::read(step.join("manifest.json")).unwrap()).unwrap();
            let fields: Vec<String> = manifest["published_fields"].as_array().unwrap().iter()
                .map(|value| value.as_str().unwrap().to_string()).collect();
            let generation = manifest["publication_generation"].as_str().unwrap();
            let version = manifest["manifest_version"].as_u64().unwrap();
            assert_eq!(version, OUTPUT_MANIFEST_VERSION);
            assert!(zfp_v1_field_storage_is_complete(&manifest, &fields, &step));
            assert!(validate_current_step_manifest(
                &manifest, generation, step_index, &name, &fields, &step, version, "history"));
            let catalog = manifest["fields"].as_array().unwrap();
            let fallback: std::collections::BTreeSet<&str> = fields.iter()
                .filter_map(|field| {
                    let metadata = catalog.iter().find(|entry|
                        entry["field_id"].as_str() == Some(field)).unwrap();
                    (metadata.get("storage_format").and_then(|value| value.as_str())
                        == Some("npy_2d")).then_some(field.as_str())
                }).collect();
            assert_eq!(fallback, expected_fallback);
            for field in &fallback {
                let left = crate::npy::read_f32_file(&step.join(format!("{field}.npy"))).unwrap();
                let right = crate::npy::read_f32_file(&npy_run.join(&name)
                    .join(format!("{field}.npy"))).unwrap();
                let left_layout = crate::npy::parse_f32_layout(&left).unwrap();
                let right_layout = crate::npy::parse_f32_layout(&right).unwrap();
                assert_eq!(left_layout.shape, right_layout.shape);
                assert_eq!(&left[left_layout.data_offset..],
                           &right[right_layout.data_offset..], "{field} step={step_index}");
            }
            let mut contradiction = manifest.clone();
            let entry = contradiction["fields"].as_array_mut().unwrap().iter_mut()
                .find(|item| item["field_id"].as_str() == Some("k_index")).unwrap();
            entry["file_pattern"] = serde_json::json!("k_index.zfp3d");
            assert!(!zfp_v1_field_storage_is_complete(&contradiction, &fields, &step));
            assert!(!validate_current_step_manifest(
                &contradiction, generation, step_index, &name, &fields, &step, version, "history"));

            if step_index == 1 {
                let copied = TestRunDir::new("real_mixed_v1_corruption");
                let copied_step = copied.0.join(&name);
                fs::create_dir_all(&copied_step).unwrap();
                for field in &fields {
                    let extension = if fallback.contains(field.as_str()) { "npy" } else { "zfp3d" };
                    let filename = format!("{field}.{extension}");
                    fs::copy(step.join(&filename), copied_step.join(filename)).unwrap();
                }
                assert!(zfp_v1_field_storage_is_complete(&manifest, &fields, &copied_step));
                let fallback_file = copied_step.join("k_index.npy");
                fs::write(&fallback_file, b"truncated NPY").unwrap();
                assert!(!zfp_v1_field_storage_is_complete(&manifest, &fields, &copied_step));
                fs::copy(step.join("k_index.npy"), &fallback_file).unwrap();
                fs::write(copied_step.join("k_index.zfp3d"), b"shadow").unwrap();
                assert!(!zfp_v1_field_storage_is_complete(&manifest, &fields, &copied_step));
            }
            println!("[real-mixed-v1] PASS step={step_index} fields={} fallback={}",
                     fields.len(), fallback.len());
        }
        assert_eq!(crate::platform::sha256_file(&binary).unwrap(),
                   format!("sha256:{expected_sha}"));
    }

    fn exact_catalog_zfp_field_budgets(bytes: &[u8]) -> HashMap<String, f64> {
        let marker = b"\"qualifiedZfpFieldBudgets\":{";
        let starts: Vec<usize> = bytes.windows(marker.len()).enumerate()
            .filter_map(|(index, window)| (window == marker).then_some(index + marker.len()))
            .collect();
        assert_eq!(starts.len(), 1, "one exact catalog ZFP budget object");
        let mut cursor = starts[0];
        let mut budgets = HashMap::new();
        loop {
            while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
                cursor += 1;
            }
            if bytes.get(cursor) == Some(&b'}') {
                break;
            }
            let key_end = json_string_token_end(bytes, cursor).expect("catalog budget field key");
            let field: String = serde_json::from_slice(&bytes[cursor..key_end]).unwrap();
            cursor = key_end;
            while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
                cursor += 1;
            }
            assert_eq!(bytes.get(cursor), Some(&b':'));
            cursor += 1;
            while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
                cursor += 1;
            }
            let begin = cursor;
            while bytes.get(cursor).is_some_and(|byte|
                byte.is_ascii_digit() || matches!(*byte, b'-' | b'+' | b'.' | b'e' | b'E')) {
                cursor += 1;
            }
            let token = std::str::from_utf8(&bytes[begin..cursor]).unwrap();
            assert!(serde_json::from_str::<serde_json::Value>(token).unwrap().is_number());
            let value = token.parse::<f64>().expect("exact catalog decimal budget");
            assert!(value.is_finite() && value > 0.0);
            assert!(budgets.insert(field, value).is_none());
            while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
                cursor += 1;
            }
            match bytes.get(cursor) {
                Some(b',') => cursor += 1,
                Some(b'}') => break,
                _ => panic!("malformed exact catalog budget token boundary"),
            }
        }
        assert!(!budgets.is_empty());
        budgets
    }

    #[test]
    #[ignore = "requires a hash-bound current native mixed-v2 archive and NPY control"]
    fn real_native_mixed_v2_manifest_and_studio_readers_match_npy() {
        let required = |name: &str| std::env::var(name)
            .unwrap_or_else(|_| panic!("missing required real mixed-v2 evidence: {name}"));
        let binary = PathBuf::from(required("NIMBUS_STUDIO_MIXED_V2_BINARY"));
        let expected_binary_sha = required("NIMBUS_STUDIO_MIXED_V2_BINARY_SHA256");
        assert_eq!(crate::platform::sha256_file(&binary).unwrap(),
                   format!("sha256:{expected_binary_sha}"));
        let zfp_run = PathBuf::from(required("NIMBUS_STUDIO_MIXED_V2_ZFP_RUN"));
        let npy_run = PathBuf::from(required("NIMBUS_STUDIO_MIXED_V2_NPY_RUN"));
        let step_index: u32 = required("NIMBUS_STUDIO_MIXED_V2_STEP")
            .parse().expect("bounded step index");
        assert!(step_index <= 4);
        let expected_products: std::collections::BTreeSet<String> =
            required("NIMBUS_STUDIO_MIXED_V2_PRODUCTS").split(',')
                .map(str::to_string).collect();
        assert!(!expected_products.is_empty() && expected_products.len() <= 5);
        let step_name = format!("step_{step_index:06}");
        let zfp_step = zfp_run.join(&step_name);
        let npy_step = npy_run.join(&step_name);
        let zfp_manifest: serde_json::Value = serde_json::from_slice(
            &fs::read(zfp_step.join("manifest.json")).unwrap()).unwrap();
        let npy_manifest: serde_json::Value = serde_json::from_slice(
            &fs::read(npy_step.join("manifest.json")).unwrap()).unwrap();
        let fields: Vec<String> = zfp_manifest["published_fields"].as_array().unwrap()
            .iter().map(|value| value.as_str().unwrap().to_string()).collect();
        let selected: std::collections::BTreeSet<String> = zfp_manifest
            ["radar_provenance"]["products"].as_array().unwrap().iter()
            .map(|value| value.as_str().unwrap().to_string()).collect();
        assert_eq!(selected, expected_products);
        assert_eq!(zfp_manifest["simulation_time_s"], npy_manifest["simulation_time_s"]);
        assert_eq!(zfp_manifest["grid"], npy_manifest["grid"]);
        assert_eq!(zfp_manifest["published_fields"], npy_manifest["published_fields"]);
        assert_eq!(zfp_manifest["radar_provenance"], npy_manifest["radar_provenance"]);
        let generation = zfp_manifest["publication_generation"].as_str().unwrap();
        let manifest_version = zfp_manifest["manifest_version"].as_u64().unwrap();
        assert_eq!(manifest_version, 7);
        assert!(grid_is_complete_for_version(&zfp_manifest["grid"], manifest_version),
                "real mixed-v2 grid contract");
        assert!(canonical_history_contract_is_complete(&zfp_manifest, "history"),
                "real mixed-v2 history contract");
        assert!(fields.iter().all(|field| manifest_declares_field(
            zfp_manifest["fields"].as_array().unwrap(), field)),
            "real mixed-v2 field declaration");
        assert!(fields.iter().all(|field| canonical_field_metadata_is_complete(
            zfp_manifest["fields"].as_array().unwrap(), field)),
            "real mixed-v2 field metadata");
        assert!(mixed_zfp_v2_is_complete(&zfp_manifest, &fields, step_index, &zfp_step),
                "real mixed-v2 compression contract");
        assert!(terrain_vertical_geometry_is_complete(&zfp_manifest, &zfp_step,
            manifest_version), "real mixed-v2 terrain geometry");
        assert!(field_brick_hierarchy_dependencies_are_complete(&zfp_manifest, &zfp_step),
                "real mixed-v2 brick dependencies");
        assert!(validate_current_step_manifest(
            &zfp_manifest, generation, step_index, &step_name, &fields,
            &zfp_step, manifest_version, "history"));
        let execution = parse_compression_execution(&zfp_manifest).unwrap();
        assert!(matches!(execution.field_execution_policy.as_str(), "serial" | "field_parallel"));
        assert!((1..=16).contains(&execution.field_workers));
        assert!((1..=4096).contains(&execution.field_scratch_budget_mib));
        let parsed_fields = parse_field_compression(&zfp_manifest);
        assert_eq!(parsed_fields.len(), fields.len());

        let catalog_output = std::process::Command::new(&binary)
            .arg("--scheme-catalog=json").output().unwrap();
        assert!(catalog_output.status.success());
        let catalog: serde_json::Value = serde_json::from_slice(&catalog_output.stdout).unwrap();
        let parsed_budgets = catalog["runtimeCapabilities"]["interactionPolicy"]
            ["qualifiedZfpFieldBudgets"].as_object().unwrap();
        let budgets = exact_catalog_zfp_field_budgets(&catalog_output.stdout);
        assert_eq!(budgets.len(), parsed_budgets.len());
        assert!(budgets.keys().all(|field| parsed_budgets.contains_key(field)));
        let exact_parameters = exact_mixed_v2_mode_parameters(&zfp_manifest, &zfp_step).unwrap();
        let run = zfp_run.to_str().unwrap();
        for field in &fields {
            let npy_bytes = crate::npy::read_f32_file(&npy_step.join(format!("{field}.npy")))
                .unwrap();
            let layout = crate::npy::parse_f32_layout(&npy_bytes).unwrap();
            let reference: Vec<f32> = npy_bytes[layout.data_offset..].chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap())).collect();
            let (shape, full) = crate::zfp3d::reconstruct_field(run, step_index, field).unwrap();
            assert_eq!(shape, vec![12, 12, 48]);
            assert_eq!(shape, layout.shape);
            assert_eq!(full.len(), reference.len());
            let entry = zfp_manifest["fields"].as_array().unwrap().iter().find(|entry|
                entry["field_id"].as_str() == Some(field)).unwrap();
            let compression = &entry["compression"];
            if expected_products.contains(field) {
                assert_eq!(parsed_fields[field].mode, "reversible");
                assert_eq!(compression["container_version"], 4);
                assert_eq!(compression["frame_role"], "keyframe");
                assert!(compression["parent_step_indices"].as_array().unwrap().is_empty());
                assert!(!budgets.contains_key(field));
                for (index, (actual, expected)) in full.iter().zip(&reference).enumerate() {
                    assert_eq!(actual.to_bits(), expected.to_bits(), "{field}[{index}]");
                }
            } else {
                assert_eq!(parsed_fields[field].mode, "accuracy");
                assert_eq!(compression["container_version"], 3);
                let budget = budgets[field];
                let parameter = exact_parameters[field];
                assert!(parameter > 0.0 && parameter <= budget, "{field} parameter");
                let mut max_error = 0.0f64;
                for (actual, expected) in full.iter().zip(&reference) {
                    assert!(actual.is_finite() && expected.is_finite());
                    max_error = max_error.max(f64::from((actual - expected).abs()));
                }
                assert!(max_error <= budget, "{field} max_error={max_error} budget={budget}");
                println!("[real-mixed-v2] {field} max_error={max_error} budget={budget}");
            }

            let origin = [2, 3, 5];
            let extent = [6, 5, 7];
            let (window_shape, window) = crate::zfp3d::reconstruct_field_window(
                run, step_index, field, origin, extent).unwrap();
            assert_eq!(window_shape, shape);
            for x in 0..extent[0] {
                for y in 0..extent[1] {
                    for z in 0..extent[2] {
                        let local = (x * extent[1] + y) * extent[2] + z;
                        let global = ((x + origin[0]) * shape[1] + y + origin[1]) * shape[2]
                            + z + origin[2];
                        assert_eq!(window[local].to_bits(), full[global].to_bits(),
                                   "{field} window {local}");
                    }
                }
            }
            let mut seen = vec![false; full.len()];
            let brick_shape = crate::zfp3d::visit_reconstructed_field_bricks(
                run, step_index, field, |brick_origin, brick_extent, values| {
                    for x in 0..brick_extent[0] {
                        for y in 0..brick_extent[1] {
                            for z in 0..brick_extent[2] {
                                let local = (x * brick_extent[1] + y) * brick_extent[2] + z;
                                let global = ((x + brick_origin[0]) * shape[1] +
                                    y + brick_origin[1]) * shape[2] + z + brick_origin[2];
                                assert!(!seen[global], "{field} duplicate brick cell");
                                seen[global] = true;
                                assert_eq!(values[local].to_bits(), full[global].to_bits(),
                                           "{field} brick {global}");
                            }
                        }
                    }
                    Ok(())
                }).unwrap();
            assert_eq!(brick_shape, shape);
            assert!(seen.iter().all(|visited| *visited));

            let sample_origin = [1, 2, 3];
            let sample_extent = [10, 9, 41];
            let stride = [2, 3, 5];
            let sampled = crate::zfp3d::read_sampled_field_window(
                run, step_index, field, sample_origin, sample_extent, stride,
                16 * 1024 * 1024, 1024 * 1024, Some([12, 12, 48]), |_| Ok(())
            ).unwrap().expect("bricked sampled field");
            assert_eq!(sampled.source_shape, shape);
            assert_eq!(sampled.output_shape, vec![5, 3, 9]);
            assert!(sampled.source_working_bytes <= 16 * 1024 * 1024);
            for x in 0..sampled.output_shape[0] {
                for y in 0..sampled.output_shape[1] {
                    for z in 0..sampled.output_shape[2] {
                        let local = (x * sampled.output_shape[1] + y) * sampled.output_shape[2] + z;
                        let offset = local * 4;
                        let actual = f32::from_le_bytes(
                            sampled.bytes[offset..offset + 4].try_into().unwrap());
                        let global = ((sample_origin[0] + x * stride[0]) * shape[1] +
                            sample_origin[1] + y * stride[1]) * shape[2] +
                            sample_origin[2] + z * stride[2];
                        assert_eq!(actual.to_bits(), full[global].to_bits(),
                                   "{field} sampled {local}");
                    }
                }
            }
        }

        let radar = expected_products.iter().next().unwrap();
        let mut wrong_lineage = zfp_manifest.clone();
        let entry = wrong_lineage["fields"].as_array_mut().unwrap().iter_mut().find(|entry|
            entry["field_id"].as_str() == Some(radar)).unwrap();
        entry["compression"]["frame_role"] = serde_json::json!("simple_delta");
        entry["compression"]["parent_step_indices"] = serde_json::json!([0]);
        assert!(!mixed_zfp_v2_is_complete(&wrong_lineage, &fields, step_index, &zfp_step));
        assert!(!validate_current_step_manifest(
            &wrong_lineage, generation, step_index, &step_name, &fields,
            &zfp_step, manifest_version, "history"));

        let copied = TestRunDir::new("real_mixed_v2_corruption");
        let copied_step = copied.0.join(&step_name);
        fs::create_dir_all(&copied_step).unwrap();
        fs::write(copied_step.join("manifest.json"), serde_json::to_vec(&zfp_manifest).unwrap())
            .unwrap();
        let copied_field = copied_step.join(format!("{radar}.zfp3d"));
        let mut payload = fs::read(zfp_step.join(format!("{radar}.zfp3d"))).unwrap();
        let last = payload.len() - 1;
        payload[last] ^= 1;
        fs::write(&copied_field, payload).unwrap();
        assert!(crate::zfp3d::reconstruct_field_window(
            copied.0.to_str().unwrap(), step_index, radar, [0, 0, 0], [12, 12, 48]).is_err());
        assert!(crate::zfp3d::read_sampled_field_window(
            copied.0.to_str().unwrap(), step_index, radar, [0, 0, 0], [12, 12, 48],
            [2, 2, 2], 16 * 1024 * 1024, 1024 * 1024, Some([12, 12, 48]), |_| Ok(()))
            .is_err());
        fs::write(copied_step.join("manifest.json"), serde_json::to_vec(&wrong_lineage).unwrap())
            .unwrap();
        fs::copy(zfp_step.join(format!("{radar}.zfp3d")), &copied_field).unwrap();
        assert!(crate::zfp3d::reconstruct_field_window(
            copied.0.to_str().unwrap(), step_index, radar, [0, 0, 0], [12, 12, 48]).is_err());
        assert_eq!(crate::platform::sha256_file(&binary).unwrap(),
                   format!("sha256:{expected_binary_sha}"));
        println!("[real-mixed-v2] PASS step={step_index} fields={} radar={}",
                 fields.len(), expected_products.len());
    }

    #[test]
    fn skips_non_finite() {
        let r = reduce_f32_le(
            &body(&[f32::NAN, 2.0, f32::INFINITY, 6.0]),
            None,
            None,
            None,
        );
        assert_eq!(r.min, Some(2.0));
        assert_eq!(r.max, Some(6.0));
        assert_eq!(r.mean, Some(4.0));
        assert_eq!(r.finite, 2);
        assert_eq!(r.total, 4);
    }

    #[test]
    fn all_non_finite_yields_none() {
        let r = reduce_f32_le(
            &body(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY]),
            None,
            None,
            None,
        );
        assert_eq!(r.min, None);
        assert_eq!(r.max, None);
        assert_eq!(r.mean, None);
        assert_eq!(r.rms, None);
        assert_eq!(r.finite, 0);
        assert_eq!(r.total, 3);
    }

    #[test]
    fn empty_body_yields_none() {
        let r = reduce_f32_le(&[], None, None, None);
        assert_eq!(r.min, None);
        assert_eq!(r.total, 0);
    }

    #[test]
    fn counts_threshold_above_below_and_abs() {
        // values -3, -1, 1, 2, 4 with threshold 1.5
        let r = reduce_f32_le(&body(&[-3.0, -1.0, 1.0, 2.0, 4.0]), Some(1.5), None, None);
        assert_eq!(r.count_above, 2); // 2, 4
        assert_eq!(r.count_below, 3); // -3, -1, 1
        assert_eq!(r.count_abs_above, 3); // |-3|, |2|, |4|
        assert_eq!(r.finite, 5);
    }

    #[test]
    fn counts_are_zero_without_threshold() {
        let r = reduce_f32_le(&body(&[1.0, 2.0, -5.0]), None, None, None);
        assert_eq!(r.count_above, 0);
        assert_eq!(r.count_below, 0);
        assert_eq!(r.count_abs_above, 0);
    }

    #[test]
    fn reductions_exclude_manifest_halos() {
        let shape = [4, 3, 2];
        let mut values = vec![999.0f32; shape.iter().product()];
        // Valid cells: i=1..2, j=1, k=0..1.
        for (n, (i, k)) in [(1usize, 0usize), (1, 1), (2, 0), (2, 1)]
            .into_iter()
            .enumerate()
        {
            values[(i * shape[1] + 1) * shape[2] + k] = (n + 1) as f32;
        }
        let valid = AxisBoundsU32 {
            lower: [1, 1, 0],
            upper: [2, 1, 1],
        };
        let bytes = body(&values);
        let scalar = reduce_f32_le(&bytes, Some(2.5), Some(&shape), Some(&valid));
        assert_eq!(scalar.min, Some(1.0));
        assert_eq!(scalar.max, Some(4.0));
        assert_eq!(scalar.total, 4);
        assert_eq!(scalar.count_above, 2);
        assert_eq!(
            reduce_axis_f32_le(&bytes, &shape, 2, &Reducer::Max, Some(&valid)),
            vec![Some(3.0), Some(4.0)]
        );
    }

    #[test]
    fn reduces_along_axis() {
        // shape [2,2,2], values 0..8 in C order ((i*2+j)*2+k):
        //   [[[0,1],[2,3]], [[4,5],[6,7]]]
        let b = body(&(0..8).map(|x| x as f32).collect::<Vec<_>>());
        // axis 2 (k): k=0 -> {0,2,4,6} max 6 ; k=1 -> {1,3,5,7} max 7
        assert_eq!(
            reduce_axis_f32_le(&b, &[2, 2, 2], 2, &Reducer::Max, None),
            vec![Some(6.0), Some(7.0)]
        );
        // axis 0 (i): i=0 -> {0,1,2,3} mean 1.5 ; i=1 -> {4,5,6,7} mean 5.5
        assert_eq!(
            reduce_axis_f32_le(&b, &[2, 2, 2], 0, &Reducer::Mean, None),
            vec![Some(1.5), Some(5.5)]
        );
    }

    #[test]
    fn zfp_v3_reductions_stream_bricks_without_full_volume_residency() {
        let run = TestRunDir::new("zfp_streaming_reduction");
        let step = run.0.join("step_000000");
        fs::create_dir_all(&step).unwrap();
        let shape = [70_usize, 67, 35];
        let values = (0..shape.into_iter().product::<usize>())
            .map(|index| (index % 997) as f32 - 300.0)
            .collect::<Vec<_>>();
        fs::write(
            step.join("theta.zfp3d"),
            crate::zfp3d::tests::make_zfp3d_v3_keyframe(&shape, &values, 1.0e-5),
        )
        .unwrap();
        let (_, decoded) =
            crate::zfp3d::reconstruct_field(run.0.to_str().unwrap(), 0, "theta").unwrap();
        let expected = reduce_f32_le(
            bytemuck::cast_slice(&decoded),
            Some(250.0),
            Some(&shape),
            None,
        );
        let (got_shape, got) =
            reduce_zfp_bricks(run.0.to_str().unwrap(), 0, "theta", Some(250.0), None).unwrap();
        assert_eq!(got_shape, shape);
        assert_eq!(got.min, expected.min);
        assert_eq!(got.max, expected.max);
        assert_eq!(got.finite, expected.finite);
        assert_eq!(got.total, expected.total);
        assert_eq!(got.count_above, expected.count_above);
        assert!((got.mean.unwrap() - expected.mean.unwrap()).abs() < 1.0e-10);
        assert!((got.rms.unwrap() - expected.rms.unwrap()).abs() < 1.0e-10);

        let expected_axis = reduce_axis_f32_le(
            bytemuck::cast_slice(&decoded),
            &shape,
            2,
            &Reducer::Mean,
            None,
        );
        let (_, got_axis) =
            reduce_zfp_axis_bricks(run.0.to_str().unwrap(), 0, "theta", 2, &Reducer::Mean, None)
                .unwrap();
        for (got, expected) in got_axis.into_iter().zip(expected_axis) {
            assert!((got.unwrap() - expected.unwrap()).abs() < 1.0e-10);
        }
    }

    #[test]
    fn axis_reduction_skips_non_finite_slab() {
        // 1x1x2 with one NaN level -> None for that index.
        let b = body(&[f32::NAN, 5.0]);
        assert_eq!(
            reduce_axis_f32_le(&b, &[1, 1, 2], 2, &Reducer::Max, None),
            vec![None, Some(5.0)]
        );
    }

    fn test_npy(shape: &[usize], values: &[f32]) -> Vec<u8> {
        let shape_text = shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let mut header =
            format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({shape_text}), }}")
                .into_bytes();
        header.push(b'\n');
        while (10 + header.len()) % 64 != 0 {
            header.insert(header.len() - 1, b' ');
        }
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&header);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn field_window_validates_bounds_rank_and_budget() {
        let base = FieldWindowRequest {
            schema_version: 2,
            request_id: "plot:theta:1".into(),
            aggregation: "cell_mean".into(),
            origin: vec![1, 0, 0],
            extent: vec![3, 4, 5],
            stride: vec![2, 2, 2],
            max_output_bytes: Some(1024),
            max_source_bytes: Some(1024),
        };
        let (shape, elements, bytes, source_bytes) = validate_window(&base, &[4, 4, 5]).unwrap();
        assert_eq!(shape, vec![2, 2, 3]);
        assert_eq!(elements, 12);
        assert_eq!(bytes, 48);
        assert_eq!(source_bytes, 240);

        let mut outside = base.clone();
        outside.extent[0] = 4;
        assert!(validate_window(&outside, &[4, 4, 5]).is_err());
        let mut tiny_budget = base;
        tiny_budget.max_output_bytes = Some(47);
        assert!(validate_window(&tiny_budget, &[4, 4, 5]).is_err());
        tiny_budget.max_output_bytes = Some(1024);
        tiny_budget.max_source_bytes = Some(239);
        assert!(validate_window(&tiny_budget, &[4, 4, 5]).is_err());
        tiny_budget.max_source_bytes = Some(1024);
        tiny_budget.aggregation = "point_sample".into();
        let (_, point_elements, point_bytes, point_source_bytes) =
            validate_window(&tiny_budget, &[4, 4, 5]).unwrap();
        assert_eq!(
            (point_elements, point_bytes, point_source_bytes),
            (12, 48, 80)
        );
    }

    #[test]
    fn npy_window_streams_cell_means_without_loading_whole_volume() {
        let run = TestRunDir::new("field_window");
        let path = run.0.join("field.npy");
        let values = (0..24).map(|value| value as f32).collect::<Vec<_>>();
        fs::write(&path, test_npy(&[2, 3, 4], &values)).unwrap();
        let request = FieldWindowRequest {
            schema_version: 2,
            request_id: "window:test".into(),
            aggregation: "cell_mean".into(),
            origin: vec![0, 1, 0],
            extent: vec![2, 2, 4],
            stride: vec![1, 1, 2],
            max_output_bytes: Some(1024),
            max_source_bytes: Some(1024),
        };
        let control = FieldReadControl::new();
        let (source_shape, output_shape, body) =
            read_npy_window(&path, &request, &control).unwrap();
        assert_eq!(source_shape, vec![2, 3, 4]);
        assert_eq!(output_shape, vec![2, 2, 2]);
        let decoded = body
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(decoded, vec![4.5, 6.5, 8.5, 10.5, 16.5, 18.5, 20.5, 22.5]);
    }

    #[test]
    fn beast_window_streams_160mib_source_into_40mib_output() {
        let run = TestRunDir::new("beast_field_window");
        let path = run.0.join("field.npy");
        let shape = [512usize, 512, 160];
        let header = test_npy(&shape, &[]);
        fs::write(&path, &header).unwrap();
        let source_bytes = shape.into_iter().product::<usize>() * size_of::<f32>();
        fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len((header.len() + source_bytes) as u64)
            .unwrap();

        let request = FieldWindowRequest {
            schema_version: 2,
            request_id: "window:beast:stride2".into(),
            aggregation: "cell_mean".into(),
            origin: vec![0, 0, 0],
            extent: shape.to_vec(),
            stride: vec![2, 2, 1],
            max_output_bytes: Some(MAX_FIELD_WINDOW_BYTES),
            max_source_bytes: Some(MAX_FIELD_WINDOW_SOURCE_BYTES),
        };
        let (source_shape, output_shape, body) =
            read_npy_window(&path, &request, &FieldReadControl::new()).unwrap();
        assert_eq!(source_shape, shape);
        assert_eq!(output_shape, vec![256, 256, 160]);
        assert_eq!(body.len(), 40 * 1024 * 1024);
        assert!(body.iter().all(|byte| *byte == 0));
    }

    #[test]
    fn beast_window_cancellation_stops_cooperative_native_rows() {
        let run = TestRunDir::new("beast_field_window_cancel");
        let path = run.0.join("field.npy");
        let shape = [512usize, 512, 160];
        let header = test_npy(&shape, &[]);
        fs::write(&path, &header).unwrap();
        let source_bytes = shape.into_iter().product::<usize>() * size_of::<f32>();
        fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len((header.len() + source_bytes) as u64)
            .unwrap();
        let request = FieldWindowRequest {
            schema_version: 2,
            request_id: "window:beast:cancel".into(),
            aggregation: "cell_mean".into(),
            origin: vec![0, 0, 0],
            extent: shape.to_vec(),
            stride: vec![2, 2, 1],
            max_output_bytes: Some(MAX_FIELD_WINDOW_BYTES),
            max_source_bytes: Some(MAX_FIELD_WINDOW_SOURCE_BYTES),
        };
        let control = Arc::new(FieldReadControl::new());
        let reader_control = Arc::clone(&control);
        let reader_path = path.clone();
        let reader =
            std::thread::spawn(move || read_npy_window(&reader_path, &request, &reader_control));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while control.phase.load(Ordering::Acquire) != ReadPhase::NpyRows as u8 {
            assert!(
                std::time::Instant::now() < deadline,
                "reader never entered NPY rows"
            );
            std::thread::yield_now();
        }
        control.cancelled.store(true, Ordering::Release);
        let error = reader.join().unwrap().unwrap_err();
        assert!(error.to_string().contains("field read cancelled:"));
        assert_eq!(
            control.phase.load(Ordering::Acquire),
            ReadPhase::NpyRows as u8
        );
    }

    #[test]
    fn npy_window_reads_bounded_rows_for_requested_point_samples() {
        let run = TestRunDir::new("field_window_points");
        let path = run.0.join("field.npy");
        let values = (0..60).map(|value| value as f32).collect::<Vec<_>>();
        fs::write(&path, test_npy(&[3, 4, 5], &values)).unwrap();
        let request = FieldWindowRequest {
            schema_version: 2,
            request_id: "window:points".into(),
            aggregation: "point_sample".into(),
            origin: vec![0, 1, 0],
            extent: vec![3, 3, 5],
            stride: vec![2, 2, 2],
            max_output_bytes: Some(48),
            max_source_bytes: Some(80),
        };
        let (source_shape, output_shape, body) =
            read_npy_window(&path, &request, &FieldReadControl::new()).unwrap();
        assert_eq!(source_shape, vec![3, 4, 5]);
        assert_eq!(output_shape, vec![2, 2, 3]);
        let decoded = body
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(
            decoded,
            vec![5.0, 7.0, 9.0, 15.0, 17.0, 19.0, 45.0, 47.0, 49.0, 55.0, 57.0, 59.0]
        );
    }

    #[test]
    fn compressed_window_dispatch_streams_predictive_data_and_preserves_all_reducers() {
        let run = TestRunDir::new("streamed_compressed_windows");
        let shape = [128usize, 128, 64];
        let values = (0..shape.iter().product())
            .map(|index| ((index % 127) as f32 - 63.0) * 0.125)
            .collect::<Vec<_>>();
        for (step_index, flags) in [(0u32, 0u8), (3, 1), (7, 3)] {
            let step = run.0.join(format!("step_{step_index:06}"));
            fs::create_dir_all(&step).unwrap();
            let mut bytes = crate::zfp3d::tests::make_zfp3d_v3_keyframe(&shape, &values, 1e-5);
            bytes[36] = flags;
            fs::write(step.join("theta.zfp3d"), bytes).unwrap();
        }
        let path = run.0.join("step_000007/theta.zfp3d");
        let (_, full) =
            crate::zfp3d::reconstruct_field(run.0.to_str().unwrap(), 7, "theta").unwrap();
        let (_, old_working) = crate::zfp3d::field_shape_and_window_reconstruction_bytes(
            run.0.to_str().unwrap(),
            7,
            "theta",
            [3, 2, 1],
            [125, 126, 63],
        )
        .unwrap();
        assert!(old_working > 12 * 1024 * 1024);
        for aggregation in [
            "point_sample",
            "cell_mean",
            "linear_reflectivity_mean",
            "block_max",
            "block_min",
            "magnitude_extreme",
        ] {
            for stride in [vec![3, 4, 5], vec![1, 1, 1]] {
                let request = FieldWindowRequest {
                    schema_version: 2,
                    request_id: "compressed-dispatch".into(),
                    aggregation: aggregation.into(),
                    origin: vec![3, 2, 1],
                    extent: vec![125, 126, 63],
                    stride,
                    max_output_bytes: Some(4 * 1024 * 1024),
                    max_source_bytes: Some(12 * 1024 * 1024),
                };
                if aggregation == "point_sample" && request.stride == [1, 1, 1] {
                    continue;
                }
                let reference =
                    sample_decoded_window(&shape, &full, &request, &FieldReadControl::new())
                        .unwrap();
                let payload = read_uniform_window_payload(
                    run.0.to_str().unwrap(),
                    7,
                    "theta",
                    &request,
                    Some(shape),
                    None,
                    Some(&path),
                    &FieldReadControl::new(),
                )
                .unwrap();
                assert_eq!(payload.output_shape, reference.0);
                assert_eq!(
                    payload.field_body, reference.1,
                    "{aggregation} {:?}",
                    request.stride
                );
                assert!(payload.source_working_bytes <= request.max_source_bytes.unwrap());
                assert_eq!(payload.cancellation_mode, "cooperative");
                assert_eq!(payload.non_interruptible_phase, Some("zfp_decode"));
            }
        }
    }

    #[test]
    fn decoded_zfp_windows_can_select_sparse_points_from_bounded_bricks() {
        let values = (0..12).map(|value| value as f32).collect::<Vec<_>>();
        let request = FieldWindowRequest {
            schema_version: 2,
            request_id: "window:zfp-points".into(),
            aggregation: "point_sample".into(),
            origin: vec![0, 0, 0],
            extent: vec![2, 2, 3],
            stride: vec![1, 2, 2],
            max_output_bytes: Some(16),
            max_source_bytes: Some(1024),
        };
        let (shape, body) =
            sample_decoded_window(&[2, 2, 3], &values, &request, &FieldReadControl::new()).unwrap();
        assert_eq!(shape, vec![2, 1, 2]);
        let decoded = body
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(decoded, vec![0.0, 2.0, 6.0, 8.0]);
    }

    #[test]
    fn npy_window_reads_a_sparse_large_logical_field_without_whole_decode() {
        use std::io::Write as _;

        let run = TestRunDir::new("field_window_large_logical");
        let path = run.0.join("field.npy");
        let shape = [8192_usize, 8192, 2];
        let header = test_npy(&shape, &[]);
        let logical = crate::npy::validate_logical_field_shape(&shape).unwrap();
        assert!(logical.element_count > crate::npy::MAX_FIELD_ELEMENTS);
        let mut file = fs::File::create(&path).unwrap();
        file.write_all(&header).unwrap();
        file.set_len((header.len() + logical.decoded_bytes) as u64)
            .unwrap();

        let request = FieldWindowRequest {
            schema_version: 2,
            request_id: "window:large-logical".into(),
            aggregation: "point_sample".into(),
            origin: vec![0, 0, 0],
            extent: vec![1, 1, 2],
            stride: vec![1, 1, 1],
            max_output_bytes: Some(8),
            max_source_bytes: Some(8),
        };
        let (source_shape, output_shape, body) =
            read_npy_window(&path, &request, &FieldReadControl::new()).unwrap();
        assert_eq!(source_shape, shape);
        assert_eq!(output_shape, vec![1, 1, 2]);
        assert_eq!(body, vec![0; 8]);
    }

    #[test]
    fn npy_window_averages_dbz_in_linear_reflectivity_space() {
        let run = TestRunDir::new("field_window_dbz");
        let path = run.0.join("field.npy");
        fs::write(&path, test_npy(&[2], &[0.0, 10.0])).unwrap();
        let request = FieldWindowRequest {
            schema_version: 2,
            request_id: "window:dbz".into(),
            aggregation: "linear_reflectivity_mean".into(),
            origin: vec![0],
            extent: vec![2],
            stride: vec![2],
            max_output_bytes: Some(4),
            max_source_bytes: Some(8),
        };
        let (_, output_shape, body) =
            read_npy_window(&path, &request, &FieldReadControl::new()).unwrap();
        assert_eq!(output_shape, vec![1]);
        let value = f32::from_le_bytes(body.try_into().unwrap());
        let expected = 10.0_f32 * 5.5_f32.log10();
        assert!((value - expected).abs() < 1e-5);
    }

    #[test]
    fn native_extrema_aggregation_preserves_threshold_crossings() {
        for (aggregation, expected) in [
            ("block_max", 5.0_f32),
            ("block_min", -8.0_f32),
            ("magnitude_extreme", -8.0_f32),
        ] {
            let aggregation = field_window_aggregation_kind(aggregation).unwrap();
            let mut sum = 0.0;
            let mut count = 0;
            for value in [3.0, -8.0, 5.0, -4.0] {
                accumulate_aggregation(&mut sum, &mut count, value, aggregation);
            }
            assert_eq!(aggregation_output(sum, count, aggregation), expected);
        }
    }

    #[test]
    fn npy_window_keeps_independent_signed_extrema_per_block() {
        let run = TestRunDir::new("field_window_signed_extrema");
        let path = run.0.join("field.npy");
        let values = [
            3.0, -8.0, 5.0, -4.0, -2.0, -3.0, -9.0, 2.0, 1.0, -7.0, 8.0, -6.0, -5.0, 4.0, -1.0,
            -10.0,
        ];
        fs::write(&path, test_npy(&[2, 2, 4], &values)).unwrap();
        let read = |aggregation: &str| {
            let request = FieldWindowRequest {
                schema_version: 2,
                request_id: format!("window:signed:{aggregation}"),
                aggregation: aggregation.into(),
                origin: vec![0, 0, 0],
                extent: vec![2, 2, 4],
                stride: vec![2, 2, 2],
                max_output_bytes: Some(8),
                max_source_bytes: Some(64),
            };
            let (_, output_shape, body) =
                read_npy_window(&path, &request, &FieldReadControl::new()).unwrap();
            assert_eq!(output_shape, vec![1, 1, 2]);
            body.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        assert_eq!(read("block_max"), vec![4.0, 8.0]);
        assert_eq!(read("block_min"), vec![-8.0, -10.0]);
    }

    #[test]
    fn terrain_npy_window_uses_exact_volume_weights_centroids_and_budget() {
        let run = TestRunDir::new("terrain_window_npy");
        run.current_generation_version(GENERATION_A, "ready", TERRAIN_GEOMETRY_MANIFEST_VERSION);
        let step = run.step_with_field(4);
        let shape = [8_usize, 7_usize, 6_usize];
        let mut field = vec![0_f32; shape.into_iter().product()];
        let field_index = |i: usize, j: usize, k: usize| (i * shape[1] + j) * shape[2] + k;
        field[field_index(1, 1, 0)] = 10.0;
        field[field_index(1, 1, 1)] = 20.0;
        field[field_index(2, 1, 0)] = 30.0;
        field[field_index(2, 1, 1)] = 50.0;
        fs::write(step.join("theta.npy"), test_npy(&shape, &field)).unwrap();

        let mut centers = vec![0_f64; shape.into_iter().product()];
        let interface_shape = [shape[0], shape[1], shape[2] + 1];
        let mut interfaces = vec![0_f64; interface_shape.into_iter().product()];
        for i in 0..shape[0] {
            let dz = 10.0 * (i + 1) as f64;
            for j in 0..shape[1] {
                for k in 0..=shape[2] {
                    interfaces[(i * shape[1] + j) * (shape[2] + 1) + k] = k as f64 * dz;
                }
                for k in 0..shape[2] {
                    centers[field_index(i, j, k)] = (k as f64 + 0.5) * dz;
                }
            }
        }
        fs::write(
            step.join("physical_z_center_m.npy"),
            f64_npy_values(shape, &centers),
        )
        .unwrap();
        fs::write(
            step.join("physical_z_interface_m.npy"),
            f64_npy_values(interface_shape, &interfaces),
        )
        .unwrap();
        let mut manifest: serde_json::Value =
            serde_json::from_str(&canonical_manifest(4, GENERATION_A)).unwrap();
        manifest["grid"]["vertical_geometry"] = serde_json::json!({
            "kind": "terrain_following",
            "center_coordinate": "physical_z_center_m",
            "interface_coordinate": "physical_z_interface_m",
            "dtype": "float64",
            "layout": "x_y_z",
            "storage": "npy_sidecar"
        });
        manifest["grid"]["horizontal_scalar_origin_m"] = serde_json::json!([125.0, 150.0]);
        manifest["grid"]["physical_bounds_m"]["lower"][2] = serde_json::json!(0.0);
        manifest["grid"]["physical_bounds_m"]["upper"][2] = serde_json::json!(480.0);
        manifest["grid"]
            .as_object_mut()
            .unwrap()
            .remove("scalar_origin_m");
        fs::write(step.join("manifest.json"), manifest.to_string()).unwrap();

        let request = FieldWindowRequest {
            schema_version: 2,
            request_id: "window:terrain:npy".into(),
            aggregation: "cell_mean".into(),
            origin: vec![1, 1, 0],
            extent: vec![2, 1, 2],
            stride: vec![2, 1, 2],
            max_output_bytes: Some(1024),
            max_source_bytes: Some(1024),
        };
        let descriptor = terrain_window_descriptor(&run.0, 4, Some(GENERATION_A))
            .unwrap()
            .unwrap();
        let field_path = step.join("theta.npy");
        let control = FieldReadControl::new();
        let payload = read_terrain_window_payload(
            run.0.to_str().unwrap(),
            4,
            "theta",
            &request,
            Some(shape),
            &descriptor,
            None,
            Some(&field_path),
            &control,
        )
        .unwrap();
        assert_eq!(payload.weighting, "physical_cell_volume");
        assert_eq!(payload.sample_location, "physical_volume_centroid");
        assert_eq!(payload.output_shape, vec![1, 1, 1]);
        assert_eq!(payload.source_working_bytes, 165);
        assert_eq!(
            f32::from_le_bytes(payload.field_body.as_slice().try_into().unwrap()),
            30.0
        );
        let centroid = payload
            .geometry_body
            .chunks_exact(8)
            .map(|bytes| f64::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(centroid.len(), 4);
        assert!((centroid[0] - 525.0).abs() < 1e-12);
        assert!((centroid[1] - 450.0).abs() < 1e-12);
        assert!((centroid[2] - 26.0).abs() < 1e-12);
        assert!((centroid[3] - 7_500_000.0).abs() < 1e-12);

        let focused_valid = AxisBoundsU32 {
            lower: [1, 1, 0],
            upper: [2, 1, 1],
        };
        let reduction = reduce_terrain_f32_le(
            bytemuck::cast_slice(&field),
            &shape,
            &descriptor,
            &run.0,
            4,
            None,
            Some(25.0),
            Some(&focused_valid),
        )
        .unwrap();
        assert_eq!((reduction.min, reduction.max), (Some(10.0), Some(50.0)));
        assert_eq!(reduction.mean, Some(30.0));
        assert!((reduction.rms.unwrap() - 1120.0_f64.sqrt()).abs() < 1e-12);
        assert_eq!((reduction.finite, reduction.total), (4, 4));
        assert_eq!((reduction.count_above, reduction.count_below), (2, 2));
        let axis_x = reduce_axis_terrain_f32_le(
            bytemuck::cast_slice(&field),
            &shape,
            0,
            &Reducer::Mean,
            &descriptor,
            &run.0,
            4,
            None,
            Some(&focused_valid),
        )
        .unwrap();
        assert_eq!(axis_x[1], Some(15.0));
        assert_eq!(axis_x[2], Some(40.0));
        assert!(axis_x
            .iter()
            .enumerate()
            .all(|(index, value)| index == 1 || index == 2 || value.is_none()));
        let axis_z = reduce_axis_terrain_f32_le(
            bytemuck::cast_slice(&field),
            &shape,
            2,
            &Reducer::Mean,
            &descriptor,
            &run.0,
            4,
            None,
            Some(&focused_valid),
        )
        .unwrap();
        assert_eq!(axis_z[0], Some(22.0));
        assert_eq!(axis_z[1], Some(38.0));

        let mut under_output_budget = request.clone();
        under_output_budget.request_id = "window:terrain:output-budget".into();
        under_output_budget.max_output_bytes = Some(35);
        let error = read_terrain_window_payload(
            run.0.to_str().unwrap(),
            4,
            "theta",
            &under_output_budget,
            Some(shape),
            &descriptor,
            None,
            Some(&field_path),
            &FieldReadControl::new(),
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("returns 40 field/geometry/padding bytes"));

        let mut under_budget = request;
        under_budget.request_id = "window:terrain:budget".into();
        under_budget.max_source_bytes = Some(164);
        let error = read_terrain_window_payload(
            run.0.to_str().unwrap(),
            4,
            "theta",
            &under_budget,
            Some(shape),
            &descriptor,
            None,
            Some(&field_path),
            &FieldReadControl::new(),
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("needs 165 inspected/working bytes"));
    }

    #[test]
    fn terrain_zfp_v3_window_composes_bounded_field_decode_with_exact_geometry() {
        let run = TestRunDir::new("terrain_window_zfp_v3");
        run.current_generation_version(GENERATION_A, "ready", FRAME_POLICY_MANIFEST_VERSION);
        let step = run.step_with_field(4);
        fs::remove_file(step.join("theta.npy")).unwrap();
        let shape = [8_usize, 7_usize, 6_usize];
        let field_index = |i: usize, j: usize, k: usize| (i * shape[1] + j) * shape[2] + k;
        let mut field = vec![0_f32; shape.into_iter().product()];
        field[field_index(1, 1, 0)] = 10.0;
        field[field_index(1, 1, 1)] = 20.0;
        field[field_index(2, 1, 0)] = 30.0;
        field[field_index(2, 1, 1)] = 50.0;
        fs::write(
            step.join("theta.zfp3d"),
            crate::zfp3d::tests::make_zfp3d_v3_keyframe(&shape, &field, 1.0e-5),
        )
        .unwrap();

        let mut centers = vec![0_f64; shape.into_iter().product()];
        let interface_shape = [shape[0], shape[1], shape[2] + 1];
        let mut interfaces = vec![0_f64; interface_shape.into_iter().product()];
        for i in 0..shape[0] {
            let dz = 10.0 * (i + 1) as f64;
            for j in 0..shape[1] {
                for k in 0..=shape[2] {
                    interfaces[(i * shape[1] + j) * (shape[2] + 1) + k] = k as f64 * dz;
                }
                for k in 0..shape[2] {
                    centers[field_index(i, j, k)] = (k as f64 + 0.5) * dz;
                }
            }
        }
        fs::write(
            step.join("physical_z_center_m.npy"),
            f64_npy_values(shape, &centers),
        )
        .unwrap();
        fs::write(
            step.join("physical_z_interface_m.npy"),
            f64_npy_values(interface_shape, &interfaces),
        )
        .unwrap();

        let mut manifest: serde_json::Value =
            serde_json::from_str(&canonical_manifest(4, GENERATION_A)).unwrap();
        manifest["manifest_version"] = serde_json::json!(FRAME_POLICY_MANIFEST_VERSION);
        manifest["data_product"]["storage_format"] = serde_json::json!("zfp");
        manifest["data_product"]["lossy_compression_contract"] =
            serde_json::json!("nimbus.zfp.per_field.v1");
        manifest["grid"]["vertical_geometry"] = serde_json::json!({
            "kind": "terrain_following",
            "center_coordinate": "physical_z_center_m",
            "interface_coordinate": "physical_z_interface_m",
            "dtype": "float64",
            "layout": "x_y_z",
            "storage": "npy_sidecar"
        });
        manifest["grid"]["horizontal_scalar_origin_m"] = serde_json::json!([125.0, 150.0]);
        manifest["grid"]
            .as_object_mut()
            .unwrap()
            .remove("scalar_origin_m");
        manifest["fields"][0]["compression"] = serde_json::json!({
            "codec": "zfp",
            "decoded_dtype": "float32",
            "frame_role": "keyframe",
            "parent_step_indices": [],
            "container_version": 3,
            "spatial_layout": "independently_decodable_bricks",
            "brick_shape": [64, 64, 32]
        });
        fs::write(step.join("manifest.json"), manifest.to_string()).unwrap();

        let request = FieldWindowRequest {
            schema_version: FIELD_WINDOW_SCHEMA_VERSION,
            request_id: "window:terrain:zfp-v3".into(),
            aggregation: "cell_mean".into(),
            origin: vec![1, 1, 0],
            extent: vec![2, 1, 2],
            stride: vec![2, 1, 2],
            max_output_bytes: Some(1024),
            max_source_bytes: Some(16 * 1024 * 1024),
        };
        let descriptor = terrain_window_descriptor(&run.0, 4, Some(GENERATION_A))
            .unwrap()
            .unwrap();
        let payload = read_terrain_window_payload(
            run.0.to_str().unwrap(),
            4,
            "theta",
            &request,
            Some(shape),
            &descriptor,
            None,
            Some(&step.join("theta.zfp3d")),
            &FieldReadControl::new(),
        )
        .unwrap();
        assert_eq!(payload.source_format, "zfp3d");
        assert_eq!(payload.cancellation_mode, "boundary_only");
        assert_eq!(payload.output_shape, vec![1, 1, 1]);
        let value = f32::from_le_bytes(payload.field_body.as_slice().try_into().unwrap());
        assert!((value - 30.0).abs() < 1.0e-3);
        let centroid = payload
            .geometry_body
            .chunks_exact(8)
            .map(|bytes| f64::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(centroid, vec![525.0, 450.0, 26.0, 7_500_000.0]);

        let point_request = FieldWindowRequest {
            request_id: "window:terrain:zfp-v3:points".into(),
            aggregation: "point_sample".into(),
            ..request
        };
        let point = read_terrain_window_payload(
            run.0.to_str().unwrap(),
            4,
            "theta",
            &point_request,
            Some(shape),
            &descriptor,
            None,
            Some(&step.join("theta.zfp3d")),
            &FieldReadControl::new(),
        )
        .unwrap();
        assert_eq!(point.sample_location, "source_cell_center");
        assert_eq!(point.output_shape, vec![1, 1, 1]);
        let point_value = f32::from_le_bytes(point.field_body.as_slice().try_into().unwrap());
        assert!((point_value - 10.0).abs() < 1.0e-3);
        let point_geometry = point
            .geometry_body
            .chunks_exact(8)
            .map(|bytes| f64::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(point_geometry, vec![375.0, 450.0, 10.0, 1_500_000.0]);
    }

    #[test]
    fn large_terrain_points_bound_io_memory_and_failure_publication() {
        use std::io::Write;
        fn sparse(path: &Path, shape: [usize; 3], width: usize) -> (fs::File, usize) {
            let dict = format!(
                "{{'descr': '<f{}', 'fortran_order': False, 'shape': ({}, {}, {}), }}",
                width, shape[0], shape[1], shape[2]
            );
            let mut header = dict.into_bytes();
            header.push(b'\n');
            while (10 + header.len()) % 16 != 0 {
                header.insert(header.len() - 1, b' ');
            }
            let mut file = fs::File::create(path).unwrap();
            file.write_all(b"\x93NUMPY\x01\x00").unwrap();
            file.write_all(&(header.len() as u16).to_le_bytes())
                .unwrap();
            file.write_all(&header).unwrap();
            let offset = 10 + header.len();
            file.set_len((offset + shape.into_iter().product::<usize>() * width) as u64)
                .unwrap();
            (file, offset)
        }
        let run = TestRunDir::new("sparse_terrain_points");
        let dir = run.step_with_field(4);
        let shape = [512, 512, 160];
        let (mut field, foff) = sparse(&dir.join("theta.npy"), shape, 4);
        let (mut centers, coff) = sparse(&dir.join("physical_z_center_m.npy"), shape, 8);
        let (mut faces, ioff) = sparse(&dir.join("physical_z_interface_m.npy"), [512, 512, 161], 8);
        for i in (0..512).step_by(16) {
            for j in (0..512).step_by(16) {
                for k in (0..160).step_by(5) {
                    let n = (i * 512 + j) * 160 + k;
                    field.seek(SeekFrom::Start((foff + n * 4) as u64)).unwrap();
                    field.write_all(&7_f32.to_le_bytes()).unwrap();
                    centers
                        .seek(SeekFrom::Start((coff + n * 8) as u64))
                        .unwrap();
                    centers
                        .write_all(&((k as f64 + 0.5) * 100.0).to_le_bytes())
                        .unwrap();
                    faces
                        .seek(SeekFrom::Start(
                            (ioff + ((i * 512 + j) * 161 + k) * 8) as u64,
                        ))
                        .unwrap();
                    faces.write_all(&(k as f64 * 100.0).to_le_bytes()).unwrap();
                    faces
                        .write_all(&((k + 1) as f64 * 100.0).to_le_bytes())
                        .unwrap();
                }
            }
        }
        drop(field);
        drop(centers);
        drop(faces);
        let descriptor = TerrainWindowDescriptor {
            source_shape: shape,
            horizontal_origin_m: [50., 50.],
            horizontal_spacing_m: [100., 100.],
            storage: TerrainGeometryStorage::NpySidecar,
            geometry_directory: None,
        };
        let request = FieldWindowRequest {
            schema_version: 2,
            request_id: "terrain-large-points".into(),
            aggregation: "point_sample".into(),
            origin: vec![0, 0, 0],
            extent: shape.to_vec(),
            stride: vec![16, 16, 5],
            max_output_bytes: Some(2 * 1024 * 1024),
            max_source_bytes: Some(32 * 1024 * 1024),
        };
        let path = dir.join("theta.npy");
        let read = |req: &FieldWindowRequest, control: &FieldReadControl| {
            read_terrain_window_payload(
                run.0.to_str().unwrap(),
                4,
                "theta",
                req,
                Some(shape),
                &descriptor,
                None,
                Some(&path),
                control,
            )
        };
        let started = std::time::Instant::now();
        let result = read(&request, &FieldReadControl::new()).unwrap();
        assert_eq!(result.output_shape, vec![32, 32, 32]);
        assert!(result
            .field_body
            .chunks_exact(4)
            .all(|b| f32::from_le_bytes(b.try_into().unwrap()) == 7.));
        assert!(result.source_working_bytes < 8 * 1024 * 1024);
        assert_eq!(result.geometry_body.len(), 32768 * 32);
        let geometry: Vec<f64> = result
            .geometry_body
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(geometry[0], 50.);
        assert_eq!(geometry[32767], 49650.);
        assert_eq!(geometry[2 * 32768 + 31], 15550.);
        assert!(geometry[3 * 32768..].iter().all(|v| *v == 1_000_000.));
        println!(
            "TERRAIN_POINTS cells=32768 source_working_bytes={} dense_source_bytes={} elapsed_ms={}",
            result.source_working_bytes,
            512_usize * 512 * (160 * 20 + 8),
            started.elapsed().as_millis()
        );
        let mut tight = request.clone();
        tight.max_source_bytes = Some(1024);
        assert!(read(&tight, &FieldReadControl::new()).is_err());
        tight = request.clone();
        tight.max_output_bytes = Some(32768 * 4);
        assert!(read(&tight, &FieldReadControl::new()).is_err());
        let cancelled = FieldReadControl::new();
        cancelled.cancelled.store(true, Ordering::Release);
        assert!(read(&request, &cancelled)
            .unwrap_err()
            .to_string()
            .contains("cancelled"));
        let repeated = read(&request, &FieldReadControl::new()).unwrap();
        assert_eq!(repeated.field_body, result.field_body);
        assert_eq!(repeated.geometry_body, result.geometry_body);
        let geometry_path = dir.join("physical_z_center_m.npy");
        let original_len = fs::metadata(&geometry_path).unwrap().len();
        let mut visits = 0;
        let changed = super::super::terrain_window::read_npy_samples(
            &geometry_path,
            shape,
            [0; 3],
            [32; 3],
            [16, 16, 5],
            &mut || {
                visits += 1;
                if visits == 2 {
                    fs::OpenOptions::new()
                        .write(true)
                        .open(&geometry_path)?
                        .set_len(original_len + 8)?;
                }
                Ok(())
            },
        );
        assert!(changed
            .unwrap_err()
            .to_string()
            .contains("changed during sampled read"));
        fs::OpenOptions::new()
            .write(true)
            .open(&geometry_path)
            .unwrap()
            .set_len(original_len)
            .unwrap();
        let mut invalid = fs::OpenOptions::new()
            .write(true)
            .open(dir.join("physical_z_center_m.npy"))
            .unwrap();
        invalid.seek(SeekFrom::Start(coff as u64)).unwrap();
        invalid.write_all(&f64::NAN.to_le_bytes()).unwrap();
        drop(invalid);
        assert!(read(&request, &FieldReadControl::new())
            .unwrap_err()
            .to_string()
            .contains("invalid sampled terrain"));
    }

    #[test]
    fn cancelled_window_stops_before_io() {
        let run = TestRunDir::new("cancelled_window");
        let path = run.0.join("field.npy");
        fs::write(&path, test_npy(&[1], &[7.0])).unwrap();
        let request = FieldWindowRequest {
            schema_version: 2,
            request_id: "window:cancelled".into(),
            aggregation: "cell_mean".into(),
            origin: vec![0],
            extent: vec![1],
            stride: vec![1],
            max_output_bytes: None,
            max_source_bytes: None,
        };
        let control = FieldReadControl::new();
        control.cancelled.store(true, Ordering::Release);
        let error = read_npy_window(&path, &request, &control).unwrap_err();
        assert!(error.to_string().contains("field read cancelled:"));
    }

    #[test]
    fn capability_report_distinguishes_cooperative_and_boundary_only_reads() {
        let capabilities = native_field_read_capabilities();
        assert_eq!(capabilities.npy.cancellation, "cooperative");
        assert_eq!(capabilities.zfp3d.cancellation, "boundary_only");
        assert_eq!(capabilities.netcdf_classic.cancellation, "cooperative");
        assert!(capabilities.netcdf_classic.window_read);
        assert!(capabilities.netcdf4.window_read);
        assert_eq!(capabilities.netcdf4.cancellation, "boundary_only");
        assert_eq!(capabilities.schema_version, 5);
        assert_eq!(
            capabilities.aggregations,
            [
                "cell_mean",
                "linear_reflectivity_mean",
                "block_max",
                "block_min",
                "magnitude_extreme",
                "point_sample",
            ]
        );
        assert_eq!(capabilities.weighting, "uniform_or_physical_cell_volume");
        assert!(capabilities.terrain_following_window_read);
        assert!(capabilities.terrain_following_reductions);
        assert!(capabilities
            .geometry_limitation
            .contains("conservative physical-coordinate remapping or draping"));
    }

    /// Real-data smoke test against a simulator-written field. Ignored by
    /// default (needs an external file); run with:
    ///   NIMBUS_TEST_NPY=/path/to/field.npy \
    ///     cargo test real_field_reduction -- --ignored --nocapture
    /// The simulator publishes `grid.horizontal_frame` on every step and refuses
    /// to publish a step without complete finite provenance (see
    /// include/core/output/step_manifest.hpp). This reader dropped it, which
    /// turned every fail-CLOSED guard in ui/src/lib/run/frameMotion.ts fail-OPEN:
    /// a translating run reported stationary, so swaths accumulated by grid cell
    /// and axes were labelled as if fixed to the Earth, silently.
    #[test]
    fn frame_motion_is_read_from_the_manifest_horizontal_frame() {
        let grid: serde_json::Value = serde_json::from_str(
            r#"{
                "horizontal_frame": {
                    "contract": "constant_translation_earth_wind_grid_transport_v1",
                    "motion_mode": "constant_velocity",
                    "grid_origin_t0_earth_m": [1000.0, 2000.0],
                    "grid_origin_earth_m": [43000.0, 16000.0],
                    "frame_velocity_earth_ms": [12.5, -4.25],
                    "published_horizontal_wind": "earth_relative",
                    "horizontal_transport_velocity": "grid_relative"
                }
            }"#,
        )
        .unwrap();
        let motion = parse_frame_motion(&grid).expect("horizontal_frame must parse");
        assert_eq!(motion.mode.as_deref(), Some("constant_velocity"));
        assert_eq!(
            motion.horizontal_frame_contract.as_deref(),
            Some("constant_translation_earth_wind_grid_transport_v1")
        );
        assert_eq!(motion.frame_velocity_ms, Some([12.5, -4.25]));
        assert_eq!(motion.earth_origin_m, Some([43000.0, 16000.0]));
        assert_eq!(motion.grid_origin_t0_earth_m, Some([1000.0, 2000.0]));
        assert_eq!(motion.winds_earth_relative, Some(true));
    }

    /// A mode from a newer engine must reach the UI verbatim. Normalising or
    /// dropping it here would hide it from the vocabulary check that refuses
    /// unknown modes, which is the whole point of that check.
    #[test]
    fn unknown_motion_mode_is_passed_through_not_normalised() {
        let grid: serde_json::Value =
            serde_json::from_str(r#"{"horizontal_frame": {"motion_mode": "adaptive_mesh"}}"#)
                .unwrap();
        let motion = parse_frame_motion(&grid).expect("horizontal_frame must parse");
        assert_eq!(motion.mode.as_deref(), Some("adaptive_mesh"));
        assert_eq!(motion.horizontal_frame_contract, None);
    }

    /// Grid-relative published winds must not be reported as Earth-relative: a
    /// hodograph built from them is wrong by exactly the frame velocity.
    #[test]
    fn grid_relative_published_wind_is_not_earth_relative() {
        let grid: serde_json::Value = serde_json::from_str(
            r#"{"horizontal_frame": {"motion_mode": "constant_velocity",
                 "published_horizontal_wind": "grid_relative"}}"#,
        )
        .unwrap();
        let motion = parse_frame_motion(&grid).expect("horizontal_frame must parse");
        assert_eq!(motion.winds_earth_relative, Some(false));
    }

    /// Manifests written before the horizontal-frame contract carry no block.
    /// Absence is a fact about those runs (they were stationary), not missing
    /// information, so it must stay distinguishable from a declared mode.
    #[test]
    fn a_manifest_without_horizontal_frame_reports_no_motion() {
        let grid: serde_json::Value = serde_json::from_str(r#"{"dr_m": 250.0}"#).unwrap();
        assert!(parse_frame_motion(&grid).is_none());
    }

    /// Verifies the production parse + reduce path on the actual writer's
    /// header + a multi-million-element body, and prints the stats to compare
    /// against a NumPy reference.
    #[test]
    #[ignore]
    fn real_field_reduction() {
        let path = std::env::var("NIMBUS_TEST_NPY")
            .expect("set NIMBUS_TEST_NPY to a simulator-written .npy");
        let bytes = std::fs::read(&path).unwrap();
        let layout = crate::npy::parse_f32_layout(&bytes).unwrap();
        let element_count = layout.decoded_bytes / size_of::<f32>();
        let body = &bytes[layout.data_offset..layout.data_offset + layout.decoded_bytes];
        let r = reduce_f32_le(body, None, None, None);
        println!(
            "data_offset={} elements={} finite={} min={:?} max={:?} mean={:?} rms={:?}",
            layout.data_offset, element_count, r.finite, r.min, r.max, r.mean, r.rms
        );
        assert_eq!(r.total as usize, element_count);
    }
}
