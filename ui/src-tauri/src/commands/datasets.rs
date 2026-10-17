use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use tauri::ipc::Response;

use crate::error::{AppError, AppResult};

const NC_DIMENSION: u32 = 10;
const NC_VARIABLE: u32 = 11;
const NC_ATTRIBUTE: u32 = 12;
const NC_CHAR: u32 = 2;

const NETCDF_WINDOW_SCHEMA_VERSION: u32 = 2;
const NETCDF_WINDOW_MAGIC: [u8; 8] = *b"NMBNCW1\0";
const DEFAULT_NETCDF_OUTPUT_BYTES: usize = 32 * 1024 * 1024;
const MAX_NETCDF_OUTPUT_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_NETCDF_SOURCE_BYTES: u64 = 64 * 1024 * 1024;
pub(crate) const MAX_NETCDF_SOURCE_BYTES: u64 = 256 * 1024 * 1024;
const MAX_NETCDF_OUTPUT_SAMPLES: usize = MAX_NETCDF_OUTPUT_BYTES / std::mem::size_of::<f32>();
const MAX_ACTIVE_NETCDF_READS: usize = 64;
const MAX_NETCDF_REQUEST_ID_BYTES: usize = 128;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NetcdfWindowRequest {
    pub schema_version: u32,
    pub request_id: String,
    pub aggregation: String,
    /// Canonical x/y(/z) origin. Time is selected separately by record_index.
    pub origin: Vec<usize>,
    /// Canonical source span, not the number of returned samples.
    pub extent: Vec<usize>,
    pub stride: Vec<usize>,
    pub max_output_bytes: Option<usize>,
    pub max_output_samples: Option<usize>,
    /// Upper bound on bytes explicitly read from the file/native API. This is
    /// not a promise about opaque temporary allocations inside libnetcdf/HDF5.
    pub max_source_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct NetcdfWindowHeader {
    schema: &'static str,
    schema_version: u32,
    request_id: String,
    aggregation: String,
    weighting: &'static str,
    missing_data_policy: &'static str,
    sample_location: &'static str,
    source_shape: Vec<usize>,
    origin: Vec<usize>,
    extent: Vec<usize>,
    stride: Vec<usize>,
    output_shape: Vec<usize>,
    element_count: usize,
    decoded_bytes: usize,
    source_read_bytes: u64,
    source_read_accounting: &'static str,
    native_internal_allocation_bounded: bool,
    dtype: &'static str,
    order: &'static str,
    source_format: String,
    cancellation_mode: &'static str,
    non_interruptible_phase: Option<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelNetcdfReadResult {
    pub request_id: String,
    pub found: bool,
    pub cancellation_requested: bool,
    pub phase: Option<&'static str>,
    pub interruptibility: Option<&'static str>,
}

pub(crate) struct NetcdfReadControl {
    cancelled: AtomicBool,
    phase: AtomicU8,
}

impl NetcdfReadControl {
    pub(crate) fn new() -> Self {
        Self {
            cancelled: AtomicBool::new(false),
            phase: AtomicU8::new(NetcdfReadPhase::Registered as u8),
        }
    }

    pub(crate) fn set_phase(&self, phase: NetcdfReadPhase) {
        self.phase.store(phase as u8, Ordering::Release);
    }

    pub(crate) fn check(&self, request_id: &str) -> AppResult<()> {
        if self.cancelled.load(Ordering::Acquire) {
            Err(AppError::Task(format!(
                "netCDF field read cancelled: request '{request_id}'"
            )))
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Copy)]
#[repr(u8)]
pub(crate) enum NetcdfReadPhase {
    Registered = 0,
    Header = 1,
    ClassicRows = 2,
    NativeCall = 3,
    Transpose = 4,
    Encode = 5,
}

fn netcdf_phase_name(phase: u8) -> &'static str {
    match phase {
        1 => "header",
        2 => "classic_rows",
        3 => "native_call",
        4 => "transpose",
        5 => "encode",
        _ => "registered",
    }
}

fn netcdf_phase_interruptibility(phase: u8) -> &'static str {
    if phase == NetcdfReadPhase::NativeCall as u8 {
        "boundary_only"
    } else {
        "cooperative"
    }
}

static ACTIVE_NETCDF_READS: Lazy<Mutex<HashMap<String, Arc<NetcdfReadControl>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

struct NetcdfReadGuard {
    request_id: String,
    control: Arc<NetcdfReadControl>,
}

impl NetcdfReadGuard {
    fn control(&self) -> Arc<NetcdfReadControl> {
        Arc::clone(&self.control)
    }
}

impl Drop for NetcdfReadGuard {
    fn drop(&mut self) {
        self.control.cancelled.store(true, Ordering::Release);
        super::io_budget::wake_waiters();
        unregister_netcdf_read(&self.request_id);
    }
}

#[derive(Debug)]
pub(crate) struct PreparedNetcdfWindow {
    pub source_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub source_start: Vec<usize>,
    pub source_count: Vec<usize>,
    pub source_stride: Vec<usize>,
    pub source_axes: Vec<usize>,
    pub source_element_count: usize,
    pub source_decoded_bytes: usize,
    pub source_byte_budget: u64,
}

#[derive(Debug)]
pub(crate) struct NetcdfWindowPayload {
    pub source_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub values: Vec<f32>,
    pub source_read_bytes: u64,
    pub source_read_accounting: &'static str,
    pub native_internal_allocation_bounded: bool,
    pub source_format: String,
    pub cancellation_mode: &'static str,
    pub non_interruptible_phase: Option<&'static str>,
}

pub(crate) fn aggregate_c_order(
    source: &[f32],
    source_shape: &[usize],
    output_shape: &[usize],
    stride: &[usize],
    aggregation: &str,
) -> AppResult<Vec<f32>> {
    #[derive(Clone, Copy)]
    enum AggregationKind {
        CellMean,
        LinearReflectivityMean,
        BlockMax,
        BlockMin,
        MagnitudeExtreme,
    }

    if aggregation == "point_sample" {
        if source_shape != output_shape
            || source.len() != checked_product(output_shape, "point-sample output shape")?
        {
            return Err(AppError::Invalid(
                "point-sample input must already have the sampled output shape".into(),
            ));
        }
        return Ok(source.to_vec());
    }
    if source_shape.is_empty()
        || source_shape.len() != output_shape.len()
        || source_shape.len() != stride.len()
        || stride.contains(&0)
    {
        return Err(AppError::Invalid(
            "cell-mean source, output, and stride ranks disagree".into(),
        ));
    }
    let source_elements = checked_product(source_shape, "cell-mean source shape")?;
    let output_elements = checked_product(output_shape, "cell-mean output shape")?;
    if source.len() != source_elements {
        return Err(AppError::Invalid(
            "cell-mean source length disagrees with its shape".into(),
        ));
    }
    for axis in 0..source_shape.len() {
        let expected = (source_shape[axis] - 1) / stride[axis] + 1;
        if output_shape[axis] != expected {
            return Err(AppError::Invalid(format!(
                "cell-mean output shape disagrees on axis {axis}"
            )));
        }
    }
    let aggregation_kind = match aggregation {
        "cell_mean" => AggregationKind::CellMean,
        "linear_reflectivity_mean" => AggregationKind::LinearReflectivityMean,
        "block_max" => AggregationKind::BlockMax,
        "block_min" => AggregationKind::BlockMin,
        "magnitude_extreme" => AggregationKind::MagnitudeExtreme,
        _ => {
            return Err(AppError::Invalid(
                "unsupported field-window aggregation".into(),
            ));
        }
    };
    let mut sums = vec![0_f64; output_elements];
    let mut counts = vec![0_usize; output_elements];
    let mut invalid = vec![false; output_elements];
    for (source_linear, value) in source.iter().copied().enumerate() {
        let mut remainder = source_linear;
        let mut target_linear = 0_usize;
        let mut target_multiplier = 1_usize;
        for axis in (0..source_shape.len()).rev() {
            let source_index = remainder % source_shape[axis];
            remainder /= source_shape[axis];
            target_linear = target_linear
                .checked_add(
                    (source_index / stride[axis])
                        .checked_mul(target_multiplier)
                        .ok_or_else(|| {
                            AppError::Invalid("field-window target index overflow".into())
                        })?,
                )
                .ok_or_else(|| AppError::Invalid("field-window target index overflow".into()))?;
            target_multiplier = target_multiplier
                .checked_mul(output_shape[axis])
                .ok_or_else(|| AppError::Invalid("field-window target stride overflow".into()))?;
        }
        if value.is_finite() {
            let numeric = f64::from(value);
            match aggregation_kind {
                AggregationKind::BlockMax
                    if counts[target_linear] == 0 || numeric > sums[target_linear] =>
                {
                    sums[target_linear] = numeric;
                }
                AggregationKind::BlockMin
                    if counts[target_linear] == 0 || numeric < sums[target_linear] =>
                {
                    sums[target_linear] = numeric;
                }
                AggregationKind::MagnitudeExtreme
                    if counts[target_linear] == 0 || numeric.abs() > sums[target_linear].abs() =>
                {
                    sums[target_linear] = numeric;
                }
                AggregationKind::LinearReflectivityMean => {
                    sums[target_linear] += 10_f64.powf(numeric / 10.0);
                }
                AggregationKind::CellMean => sums[target_linear] += numeric,
                _ => {}
            }
            counts[target_linear] += 1;
        } else {
            invalid[target_linear] = true;
        }
    }
    Ok((0..output_elements)
        .map(|index| {
            if invalid[index] || counts[index] == 0 {
                f32::NAN
            } else {
                if matches!(
                    aggregation_kind,
                    AggregationKind::BlockMax
                        | AggregationKind::BlockMin
                        | AggregationKind::MagnitudeExtreme
                ) {
                    sums[index] as f32
                } else if matches!(aggregation_kind, AggregationKind::LinearReflectivityMean) {
                    let mean = sums[index] / counts[index] as f64;
                    (10.0 * mean.log10()) as f32
                } else {
                    (sums[index] / counts[index] as f64) as f32
                }
            }
        })
        .collect())
}

#[tauri::command]
pub async fn list_netcdf_files(directory: String) -> AppResult<Vec<String>> {
    tauri::async_runtime::spawn_blocking(move || {
        let inventory = inventory_cm1_case(&directory)?;
        if inventory.history_files.is_empty() {
            return Err(AppError::NotFound(format!(
                "no .nc, .nc4, .cdf, or .netcdf history files found within the bounded case search for {directory}"
            )));
        }
        Ok(inventory.history_files)
    })
    .await
    .map_err(|error| AppError::Task(error.to_string()))?
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Cm1CaseInventory {
    pub schema: &'static str,
    pub schema_version: u32,
    pub directory: String,
    pub history_files: Vec<String>,
    pub namelist_input: Option<String>,
    pub input_sounding: Option<String>,
    pub restart_files: Vec<String>,
    pub statistics_files: Vec<String>,
    pub other_file_count: usize,
    pub nested_directory_count: usize,
    pub scanned_directory_count: usize,
    pub skipped_directory_count: usize,
    pub excluded_directory_count: usize,
    pub unreadable_directory_count: usize,
    pub unreadable_entry_count: usize,
    pub symlink_count: usize,
    pub discovery_max_depth: usize,
    pub discovery_truncated: bool,
    pub warnings: Vec<String>,
}

#[tauri::command]
pub async fn inspect_cm1_case_folder(directory: String) -> AppResult<Cm1CaseInventory> {
    tauri::async_runtime::spawn_blocking(move || inventory_cm1_case(&directory))
        .await
        .map_err(|error| AppError::Task(error.to_string()))?
}

fn inventory_cm1_case(directory: &str) -> AppResult<Cm1CaseInventory> {
    const MAX_DIRECTORY_ENTRIES: usize = 100_000;
    const MAX_DIRECTORIES: usize = 5_000;
    const MAX_DEPTH: usize = 3;
    const EXCLUDED_DIRECTORY_NAMES: [&str; 9] = [
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "target",
        "__pycache__",
        ".mypy_cache",
    ];
    let metadata = std::fs::symlink_metadata(directory)?;
    if metadata.file_type().is_symlink() {
        return Err(AppError::Invalid(format!(
            "CM1 case path is a symbolic link; select the case directory itself: {directory}"
        )));
    }
    if !metadata.is_dir() {
        return Err(AppError::Invalid(format!(
            "CM1 case path is not a directory: {directory}"
        )));
    }
    let mut inventory = Cm1CaseInventory {
        schema: "nimbus.studio.cm1-case-inventory",
        schema_version: 2,
        directory: directory.into(),
        history_files: Vec::new(),
        namelist_input: None,
        input_sounding: None,
        restart_files: Vec::new(),
        statistics_files: Vec::new(),
        other_file_count: 0,
        nested_directory_count: 0,
        scanned_directory_count: 0,
        skipped_directory_count: 0,
        excluded_directory_count: 0,
        unreadable_directory_count: 0,
        unreadable_entry_count: 0,
        symlink_count: 0,
        discovery_max_depth: MAX_DEPTH,
        discovery_truncated: false,
        warnings: Vec::new(),
    };
    let mut queue = VecDeque::from([(PathBuf::from(directory), 0_usize)]);
    let mut entry_count = 0_usize;
    'inventory: while let Some((current, depth)) = queue.pop_front() {
        if inventory.scanned_directory_count >= MAX_DIRECTORIES {
            inventory.discovery_truncated = true;
            inventory.skipped_directory_count += 1 + queue.len();
            break;
        }
        inventory.scanned_directory_count += 1;
        let read_directory = match std::fs::read_dir(&current) {
            Ok(entries) => entries,
            Err(error) if depth > 0 => {
                inventory.unreadable_directory_count += 1;
                inventory.warnings.push(format!(
                    "Could not inspect nested folder {}: {error}",
                    current.to_string_lossy()
                ));
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        let mut entries = match read_directory.collect::<Result<Vec<_>, _>>() {
            Ok(entries) => entries,
            Err(error) if depth > 0 => {
                inventory.unreadable_directory_count += 1;
                inventory.warnings.push(format!(
                    "Could not finish reading nested folder {}: {error}",
                    current.to_string_lossy()
                ));
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        entries.sort_by(|left, right| {
            left.file_name()
                .to_string_lossy()
                .to_ascii_lowercase()
                .cmp(&right.file_name().to_string_lossy().to_ascii_lowercase())
                .then_with(|| left.file_name().cmp(&right.file_name()))
        });
        for entry in entries {
            if entry_count >= MAX_DIRECTORY_ENTRIES {
                inventory.discovery_truncated = true;
                inventory.skipped_directory_count += queue.len();
                break 'inventory;
            }
            entry_count += 1;
            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
                Err(_) => {
                    inventory.unreadable_entry_count += 1;
                    continue;
                }
            };
            if file_type.is_symlink() {
                inventory.symlink_count += 1;
                continue;
            }
            if file_type.is_dir() {
                inventory.nested_directory_count += 1;
                let lower_name = entry.file_name().to_string_lossy().to_ascii_lowercase();
                if EXCLUDED_DIRECTORY_NAMES.contains(&lower_name.as_str()) {
                    inventory.excluded_directory_count += 1;
                    continue;
                }
                if depth < MAX_DEPTH {
                    queue.push_back((entry.path(), depth + 1));
                } else {
                    inventory.skipped_directory_count += 1;
                }
                continue;
            }
            if !file_type.is_file() {
                continue;
            }
            let path = entry.path();
            let file_name = entry.file_name().to_string_lossy().into_owned();
            let lower_name = file_name.to_ascii_lowercase();
            let path_string = path.to_string_lossy().into_owned();
            if depth == 0 && lower_name == "namelist.input" {
                inventory.namelist_input = Some(path_string);
                continue;
            }
            if depth == 0 && lower_name == "input_sounding" {
                inventory.input_sounding = Some(path_string);
                continue;
            }
            if lower_name.starts_with("cm1rst_")
                || (lower_name.starts_with("cm1out_") && lower_name.contains("_rst."))
            {
                inventory.restart_files.push(path_string);
                continue;
            }
            if lower_name == "cm1out_stats.dat" {
                inventory.statistics_files.push(path_string);
                continue;
            }
            let extension = path
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or_default();
            if ["nc", "nc4", "cdf", "netcdf"]
                .iter()
                .any(|candidate| extension.eq_ignore_ascii_case(candidate))
            {
                inventory.history_files.push(path_string);
            } else {
                inventory.other_file_count += 1;
            }
        }
    }
    if inventory.skipped_directory_count > 0 {
        inventory.warnings.push(format!(
            "{} nested director{} beyond the depth/directory scan limits {} not inspected",
            inventory.skipped_directory_count,
            if inventory.skipped_directory_count == 1 {
                "y"
            } else {
                "ies"
            },
            if inventory.skipped_directory_count == 1 {
                "was"
            } else {
                "were"
            },
        ));
    }
    if inventory.excluded_directory_count > 0 {
        inventory.warnings.push(format!(
            "{} source or dependency director{} excluded from CM1 output discovery",
            inventory.excluded_directory_count,
            if inventory.excluded_directory_count == 1 {
                "y was"
            } else {
                "ies were"
            }
        ));
    }
    if inventory.unreadable_entry_count > 0 {
        inventory.warnings.push(format!(
            "{} directory entr{} could not be inspected and {} skipped",
            inventory.unreadable_entry_count,
            if inventory.unreadable_entry_count == 1 {
                "y"
            } else {
                "ies"
            },
            if inventory.unreadable_entry_count == 1 {
                "was"
            } else {
                "were"
            }
        ));
    }
    if inventory.discovery_truncated {
        inventory.warnings.push(
            "CM1 case discovery reached its 100,000-entry or 5,000-directory safety limit; the partial inventory remains usable".into(),
        );
    }
    if inventory.symlink_count > 0 {
        inventory.warnings.push(format!(
            "{} symbolic link{} skipped; Studio never follows links during case discovery",
            inventory.symlink_count,
            if inventory.symlink_count == 1 {
                ""
            } else {
                "s"
            }
        ));
    }
    for files in [
        &mut inventory.history_files,
        &mut inventory.restart_files,
        &mut inventory.statistics_files,
    ] {
        files.sort_by(|left, right| {
            left.to_ascii_lowercase()
                .cmp(&right.to_ascii_lowercase())
                .then_with(|| left.cmp(right))
        });
    }
    Ok(inventory)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NetcdfDimension {
    pub name: String,
    pub length: u64,
    pub unlimited: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NetcdfCfDecode {
    pub status: String,
    pub scale_factor: Option<f64>,
    pub add_offset: Option<f64>,
    pub fill_value: Option<f64>,
    pub fill_value_source: Option<String>,
    pub missing_values: Vec<f64>,
    pub valid_min: Option<f64>,
    pub valid_max: Option<f64>,
    pub valid_domain: &'static str,
    pub missing_output: &'static str,
    pub issue: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NetcdfVariable {
    pub name: String,
    pub data_type: String,
    pub dimensions: Vec<String>,
    pub shape: Vec<u64>,
    pub units: Option<String>,
    pub long_name: Option<String>,
    pub cf_decode: NetcdfCfDecode,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NetcdfMetadata {
    pub schema: &'static str,
    pub schema_version: u32,
    pub path: String,
    pub container_format: String,
    pub record_count: u64,
    pub dimensions: Vec<NetcdfDimension>,
    pub variables: Vec<NetcdfVariable>,
    pub source_classification: String,
    pub plotting_supported: bool,
    pub plot_unavailable_reason: String,
}

#[derive(Debug, Clone)]
pub(crate) struct NetcdfNumericAttribute {
    pub name: String,
    pub data_type: String,
    pub values: Vec<f64>,
}

fn netcdf_default_fill(data_type: &str) -> Option<f64> {
    match data_type {
        "byte" => Some(-127.0),
        "short" => Some(-32_767.0),
        "int" => Some(-2_147_483_647.0),
        "float" => Some(f32::from_bits(0x7cf0_0000) as f64),
        "double" => Some(f64::from_bits(0x479e_0000_0000_0000)),
        "ubyte" => Some(255.0),
        "ushort" => Some(65_535.0),
        "uint" => Some(4_294_967_295.0),
        "int64" => Some(-9_223_372_036_854_775_806.0),
        "uint64" => Some(18_446_744_073_709_551_614.0),
        _ => None,
    }
}

fn single_cf_attribute(
    attributes: &[NetcdfNumericAttribute],
    name: &str,
) -> Result<Option<f64>, String> {
    let Some(attribute) = attributes.iter().find(|attribute| attribute.name == name) else {
        return Ok(None);
    };
    if attribute.values.len() != 1 {
        return Err(format!(
            "attribute '{name}' must contain exactly one numeric value"
        ));
    }
    let value = attribute.values[0];
    if !value.is_finite() {
        return Err(format!("attribute '{name}' must be finite"));
    }
    Ok(Some(value))
}

pub(crate) fn build_netcdf_cf_decode(
    data_type: &str,
    attributes: &[NetcdfNumericAttribute],
    unsigned_text: Option<&str>,
) -> NetcdfCfDecode {
    let mut policy = NetcdfCfDecode {
        status: "identity".into(),
        scale_factor: None,
        add_offset: None,
        fill_value: netcdf_default_fill(data_type),
        fill_value_source: netcdf_default_fill(data_type).map(|_| "netcdf_default".into()),
        missing_values: Vec::new(),
        valid_min: None,
        valid_max: None,
        valid_domain: "stored",
        missing_output: "nan",
        issue: None,
    };
    let parsed = (|| -> Result<(), String> {
        if unsigned_text.is_some_and(|value| value.trim().eq_ignore_ascii_case("true"))
            && matches!(data_type, "byte" | "short" | "int" | "int64")
        {
            return Err(
                "_Unsigned reinterpretation is not supported; the variable remains inspect-only"
                    .into(),
            );
        }
        for attribute in attributes.iter().filter(|attribute| {
            matches!(
                attribute.name.as_str(),
                "_FillValue" | "missing_value" | "valid_range" | "valid_min" | "valid_max"
            )
        }) {
            if attribute.data_type != data_type {
                return Err(format!(
                    "attribute '{}' has type '{}' but stored values have type '{data_type}'",
                    attribute.name, attribute.data_type
                ));
            }
        }
        let scale_type = attributes
            .iter()
            .find(|attribute| attribute.name == "scale_factor")
            .map(|attribute| attribute.data_type.as_str());
        let offset_type = attributes
            .iter()
            .find(|attribute| attribute.name == "add_offset")
            .map(|attribute| attribute.data_type.as_str());
        if scale_type
            .into_iter()
            .chain(offset_type)
            .any(|kind| !matches!(kind, "float" | "double"))
        {
            return Err("scale_factor and add_offset must be float or double attributes".into());
        }
        if scale_type
            .zip(offset_type)
            .is_some_and(|(scale, offset)| scale != offset)
        {
            return Err("scale_factor and add_offset must use the same numeric type".into());
        }
        policy.scale_factor = single_cf_attribute(attributes, "scale_factor")?;
        policy.add_offset = single_cf_attribute(attributes, "add_offset")?;
        if let Some(attribute) = attributes
            .iter()
            .find(|attribute| attribute.name == "_FillValue")
        {
            if attribute.values.len() != 1 || attribute.values[0].is_infinite() {
                return Err("attribute '_FillValue' must contain one finite value or NaN".into());
            }
            policy.fill_value = attribute.values[0]
                .is_finite()
                .then_some(attribute.values[0]);
            policy.fill_value_source = Some("attribute".into());
        }
        if let Some(attribute) = attributes
            .iter()
            .find(|attribute| attribute.name == "missing_value")
        {
            if attribute.values.is_empty() || attribute.values.len() > 32 {
                return Err("attribute 'missing_value' must contain 1..=32 numeric values".into());
            }
            if attribute.values.iter().any(|value| value.is_infinite()) {
                return Err("attribute 'missing_value' cannot contain infinity".into());
            }
            policy.missing_values = attribute
                .values
                .iter()
                .copied()
                .filter(|value| value.is_finite())
                .collect();
        }
        if let Some(attribute) = attributes
            .iter()
            .find(|attribute| attribute.name == "valid_range")
        {
            if attribute.values.len() != 2
                || attribute.values.iter().any(|value| !value.is_finite())
            {
                return Err(
                    "attribute 'valid_range' must contain two finite numeric values".into(),
                );
            }
            policy.valid_min = Some(attribute.values[0]);
            policy.valid_max = Some(attribute.values[1]);
        }
        if attributes
            .iter()
            .any(|attribute| attribute.name == "valid_range")
            && attributes
                .iter()
                .any(|attribute| matches!(attribute.name.as_str(), "valid_min" | "valid_max"))
        {
            return Err("valid_range cannot be combined with valid_min or valid_max".into());
        }
        if policy.valid_min.is_none() {
            policy.valid_min = single_cf_attribute(attributes, "valid_min")?;
        }
        if policy.valid_max.is_none() {
            policy.valid_max = single_cf_attribute(attributes, "valid_max")?;
        }
        if policy
            .valid_min
            .zip(policy.valid_max)
            .is_some_and(|(minimum, maximum)| minimum > maximum)
        {
            return Err("CF valid minimum exceeds valid maximum".into());
        }
        Ok(())
    })();
    if let Err(issue) = parsed {
        policy.status = "unsupported".into();
        policy.issue = Some(issue);
    } else if policy.scale_factor.is_some()
        || policy.add_offset.is_some()
        || policy.fill_value.is_some()
        || attributes
            .iter()
            .any(|attribute| matches!(attribute.name.as_str(), "_FillValue" | "missing_value"))
        || !policy.missing_values.is_empty()
        || policy.valid_min.is_some()
        || policy.valid_max.is_some()
    {
        policy.status = "applied".into();
    }
    policy
}

fn cf_missing(value: f64, missing: f64) -> bool {
    value == missing
        || (value.is_finite()
            && missing.is_finite()
            && (value as f32).to_bits() == (missing as f32).to_bits())
}

pub(crate) fn decode_netcdf_cf_value(policy: &NetcdfCfDecode, stored: f64) -> AppResult<f64> {
    if policy.status == "unsupported" {
        return Err(AppError::Invalid(format!(
            "unsupported CF decode metadata: {}",
            policy.issue.as_deref().unwrap_or("unknown policy error")
        )));
    }
    if !stored.is_finite()
        || policy
            .fill_value
            .is_some_and(|missing| cf_missing(stored, missing))
        || policy
            .missing_values
            .iter()
            .any(|missing| cf_missing(stored, *missing))
        || policy.valid_min.is_some_and(|minimum| stored < minimum)
        || policy.valid_max.is_some_and(|maximum| stored > maximum)
    {
        return Ok(f64::NAN);
    }
    let unpacked = stored * policy.scale_factor.unwrap_or(1.0) + policy.add_offset.unwrap_or(0.0);
    Ok(if unpacked.is_finite() {
        unpacked
    } else {
        f64::NAN
    })
}

fn normalized_cm1_unit(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn cm1_history_unit_matches(name: &str, units: &str) -> bool {
    let actual = normalized_cm1_unit(units);
    let accepted: &[&str] = match name {
        "uinterp" | "vinterp" | "winterp" => &["m/s", "m s-1", "m s^-1", "m s**-1"],
        "th" | "thpert" => &["k", "kelvin"],
        "prs" | "prspert" => &["pa", "pascal", "pascals"],
        "rho" => &["kg/m^3", "kg m-3", "kg m^-3", "kg m**-3"],
        "qv" => &["kg/kg", "kg kg-1", "kg kg^-1", "kg kg**-1"],
        "dbz" | "cref" => &["dbz"],
        "zvort" => &["1/s", "s-1", "s^-1", "s**-1"],
        "buoyancy" => &["m/s^2", "m s-2", "m s^-2", "m s**-2"],
        "pv" => &[
            "k m2/kg/s",
            "k m^2/kg/s",
            "k m2 kg-1 s-1",
            "k m^2 kg^-1 s^-1",
        ],
        "cape" | "cin" => &["j/kg", "j kg-1", "j kg^-1", "j kg**-1"],
        "lcl" | "lfc" => &["m", "meter", "meters", "metre", "metres"],
        "uh" => &["m2/s2", "m^2/s^2", "m2 s-2", "m^2 s^-2", "m**2 s**-2"],
        "pwat" => &["m", "meter", "meters", "metre", "metres"],
        "lwp" | "cwp" => &["kg/m^2", "kg/m2", "kg m-2", "kg m^-2", "kg m**-2"],
        "rain" => &[
            "cm",
            "centimeter",
            "centimeters",
            "centimetre",
            "centimetres",
        ],
        "prate" => &[
            "kg/m2/s",
            "kg/m^2/s",
            "kg m-2 s-1",
            "kg m^-2 s^-1",
            "kg m**-2 s**-1",
        ],
        _ => return false,
    };
    accepted.contains(&actual.as_str())
}

fn cm1_history_rank(name: &str) -> Option<usize> {
    match name {
        "uinterp" | "vinterp" | "winterp" | "th" | "thpert" | "prs" | "prspert" | "rho" | "qv"
        | "dbz" | "zvort" | "buoyancy" | "pv" => Some(3),
        "cref" | "cape" | "cin" | "lcl" | "lfc" | "uh" | "pwat" | "lwp" | "cwp" | "rain"
        | "prate" => Some(2),
        _ => None,
    }
}

fn cm1_scalar_dimension(name: &str, axis: usize) -> bool {
    let normalized = name.to_ascii_lowercase();
    match axis {
        0 => ["xh", "x", "ni"].contains(&normalized.as_str()),
        1 => ["yh", "y", "nj"].contains(&normalized.as_str()),
        2 => ["zh", "z", "nk"].contains(&normalized.as_str()),
        _ => false,
    }
}

pub(crate) fn is_qualified_cm1_history_variable(variable: &NetcdfVariable) -> bool {
    if variable.cf_decode.status == "unsupported" {
        return false;
    }
    let name = variable.name.as_str();
    let Some(rank) = cm1_history_rank(name) else {
        return false;
    };
    let Some(units) = variable.units.as_deref() else {
        return false;
    };
    if !cm1_history_unit_matches(name, units) {
        return false;
    }
    let spatial = if variable
        .dimensions
        .first()
        .is_some_and(|dimension| dimension.eq_ignore_ascii_case("time"))
    {
        &variable.dimensions[1..]
    } else {
        &variable.dimensions[..]
    };
    if spatial.len() != rank {
        return false;
    }
    let mut axes = std::collections::BTreeSet::new();
    spatial.iter().all(|dimension| {
        axis_kind(dimension).is_some_and(|axis| {
            axis < rank && cm1_scalar_dimension(dimension, axis) && axes.insert(axis)
        })
    }) && axes.len() == rank
}

#[tauri::command]
pub async fn inspect_netcdf_metadata(path: String) -> AppResult<NetcdfMetadata> {
    tauri::async_runtime::spawn_blocking(move || {
        let mut file = File::open(&path)?;
        let mut magic = [0_u8; 4];
        file.read_exact(&mut magic)?;
        file.seek(SeekFrom::Start(0))?;
        if &magic[..3] == b"CDF" && matches!(magic[3], 1 | 2) {
            parse_classic_metadata(file, path)
        } else {
            super::netcdf_dynamic::inspect(&path)
        }
    })
    .await
    .map_err(|error| AppError::Task(error.to_string()))?
}

#[tauri::command]
pub async fn read_netcdf_field(
    path: String,
    variable: String,
    record_index: u64,
) -> AppResult<Response> {
    let bytes = tauri::async_runtime::spawn_blocking(move || {
        let mut file = File::open(&path)?;
        let mut magic = [0_u8; 4];
        file.read_exact(&mut magic)?;
        if &magic[..3] == b"CDF" && matches!(magic[3], 1 | 2) {
            read_classic_field(&path, &variable, record_index)
        } else {
            let (shape, data) = super::netcdf_dynamic::read_field(&path, &variable, record_index)?;
            crate::zfp3d::encode_npy_f32(&shape, &data)
        }
    })
    .await
    .map_err(|error| AppError::Task(error.to_string()))??;
    Ok(Response::new(bytes))
}

fn validate_netcdf_request_id(request_id: &str) -> AppResult<()> {
    if request_id.is_empty()
        || request_id.len() > MAX_NETCDF_REQUEST_ID_BYTES
        || !request_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        return Err(AppError::Invalid(format!(
            "request id must be 1..={MAX_NETCDF_REQUEST_ID_BYTES} ASCII letters, digits, '-', '_', '.', or ':'"
        )));
    }
    Ok(())
}

fn register_netcdf_read(request: &NetcdfWindowRequest) -> AppResult<NetcdfReadGuard> {
    validate_netcdf_request_id(&request.request_id)?;
    if request.schema_version != NETCDF_WINDOW_SCHEMA_VERSION {
        return Err(AppError::Invalid(format!(
            "unsupported netCDF-window schema version {} (expected {NETCDF_WINDOW_SCHEMA_VERSION})",
            request.schema_version
        )));
    }
    if !matches!(
        request.aggregation.as_str(),
        "cell_mean" | "linear_reflectivity_mean" | "block_max" | "block_min" | "magnitude_extreme"
    ) {
        return Err(AppError::Invalid(
            "unsupported netCDF-window aggregation".into(),
        ));
    }
    let source_bytes = usize::try_from(
        request
            .max_source_bytes
            .unwrap_or(DEFAULT_NETCDF_SOURCE_BYTES),
    )
    .map_err(|_| AppError::Invalid("netCDF source budget does not fit usize".into()))?;
    super::io_budget::reservation_bytes(
        source_bytes,
        request
            .max_output_bytes
            .unwrap_or(DEFAULT_NETCDF_OUTPUT_BYTES),
    )?;
    let mut active = ACTIVE_NETCDF_READS
        .lock()
        .map_err(|_| AppError::Task("netCDF-read registry lock poisoned".into()))?;
    if active.contains_key(&request.request_id) {
        return Err(AppError::Invalid(format!(
            "netCDF read request '{}' is already active",
            request.request_id
        )));
    }
    if active.len() >= MAX_ACTIVE_NETCDF_READS {
        return Err(AppError::Task(format!(
            "active netCDF-read limit reached ({MAX_ACTIVE_NETCDF_READS})"
        )));
    }
    let control = Arc::new(NetcdfReadControl::new());
    active.insert(request.request_id.clone(), Arc::clone(&control));
    Ok(NetcdfReadGuard {
        request_id: request.request_id.clone(),
        control,
    })
}

fn unregister_netcdf_read(request_id: &str) {
    if let Ok(mut active) = ACTIVE_NETCDF_READS.lock() {
        active.remove(request_id);
    }
}

/// Request cancellation of a bounded netCDF read. A `true` receipt confirms
/// that the flag was set, not that the worker has stopped. Classic row reads
/// observe it cooperatively. libnetcdf/HDF5 observes it before and after the
/// single `nc_get_vars_float` call because that C API has no abort callback.
#[tauri::command]
pub fn cancel_netcdf_field_read(request_id: String) -> AppResult<CancelNetcdfReadResult> {
    validate_netcdf_request_id(&request_id)?;
    let active = ACTIVE_NETCDF_READS
        .lock()
        .map_err(|_| AppError::Task("netCDF-read registry lock poisoned".into()))?;
    let Some(control) = active.get(&request_id) else {
        return Ok(CancelNetcdfReadResult {
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
    Ok(CancelNetcdfReadResult {
        request_id,
        found: true,
        cancellation_requested: true,
        phase: Some(netcdf_phase_name(phase)),
        interruptibility: Some(netcdf_phase_interruptibility(phase)),
    })
}

fn checked_product(values: &[usize], context: &str) -> AppResult<usize> {
    values.iter().try_fold(1_usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| AppError::Invalid(format!("{context} overflow")))
    })
}

pub(crate) fn prepare_netcdf_window(
    variable: &NetcdfVariable,
    record_count: u64,
    time_variable: bool,
    record_index: u64,
    request: &NetcdfWindowRequest,
) -> AppResult<PreparedNetcdfWindow> {
    if request.schema_version != NETCDF_WINDOW_SCHEMA_VERSION {
        return Err(AppError::Invalid(format!(
            "unsupported netCDF-window schema version {} (expected {NETCDF_WINDOW_SCHEMA_VERSION})",
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
            "unsupported netCDF-window aggregation".into(),
        ));
    }
    validate_netcdf_request_id(&request.request_id)?;
    if time_variable {
        if record_index >= record_count {
            return Err(AppError::Invalid(format!(
                "record {record_index} is outside 0..{record_count}"
            )));
        }
    } else if record_index != 0 {
        return Err(AppError::Invalid(
            "a variable without a time dimension only has record 0".into(),
        ));
    }

    let skip = usize::from(time_variable);
    let spatial_dimensions = &variable.dimensions[skip..];
    let spatial_shape = &variable.shape[skip..];
    if !matches!(spatial_dimensions.len(), 2 | 3) || spatial_shape.len() != spatial_dimensions.len()
    {
        return Err(AppError::Invalid(format!(
            "variable '{}' is not a 2-D or 3-D field after selecting time",
            variable.name
        )));
    }
    let rank = spatial_dimensions.len();
    if request.origin.len() != rank || request.extent.len() != rank || request.stride.len() != rank
    {
        return Err(AppError::Invalid(format!(
            "origin, extent, and stride must each contain exactly {rank} canonical axes"
        )));
    }

    let source_axes = spatial_dimensions
        .iter()
        .map(|name| {
            axis_kind(name)
                .ok_or_else(|| AppError::Invalid(format!("cannot map dimension '{name}' to x/y/z")))
        })
        .collect::<AppResult<Vec<_>>>()?;
    let mut source_shape = vec![0_usize; rank];
    for (source_axis, canonical_axis) in source_axes.iter().copied().enumerate() {
        if canonical_axis >= rank {
            return Err(AppError::Invalid(
                "two-dimensional fields may contain only x and y dimensions".into(),
            ));
        }
        if source_shape[canonical_axis] != 0 {
            return Err(AppError::Invalid("duplicate canonical dimensions".into()));
        }
        source_shape[canonical_axis] =
            usize::try_from(spatial_shape[source_axis]).map_err(|_| {
                AppError::Invalid("field dimension is too large for this platform".into())
            })?;
    }
    if source_shape.contains(&0) {
        return Err(AppError::Invalid(
            "field does not contain each required canonical dimension".into(),
        ));
    }

    let mut output_shape = Vec::with_capacity(rank);
    for (axis, &source_len) in source_shape.iter().enumerate() {
        let origin = request.origin[axis];
        let extent = request.extent[axis];
        let stride = request.stride[axis];
        if extent == 0 || stride == 0 || origin >= source_len {
            return Err(AppError::Invalid(format!(
                "invalid netCDF window on axis {axis}: origin={origin}, extent={extent}, stride={stride}, source={}",
                source_len
            )));
        }
        let end = origin
            .checked_add(extent)
            .ok_or_else(|| AppError::Invalid("netCDF-window bound overflow".into()))?;
        if end > source_len {
            return Err(AppError::Invalid(format!(
                "netCDF window exceeds axis {axis}: end={end}, source={}",
                source_len
            )));
        }
        output_shape.push((extent - 1) / stride + 1);
    }
    let element_count = checked_product(&output_shape, "netCDF-window sample count")?;
    let decoded_bytes = element_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| AppError::Invalid("netCDF-window byte size overflow".into()))?;
    let output_byte_budget = request
        .max_output_bytes
        .unwrap_or(DEFAULT_NETCDF_OUTPUT_BYTES);
    if output_byte_budget == 0 || output_byte_budget > MAX_NETCDF_OUTPUT_BYTES {
        return Err(AppError::Invalid(format!(
            "maxOutputBytes must be within 1..={MAX_NETCDF_OUTPUT_BYTES}"
        )));
    }
    let output_sample_budget = request
        .max_output_samples
        .unwrap_or(output_byte_budget / std::mem::size_of::<f32>());
    if output_sample_budget == 0 || output_sample_budget > MAX_NETCDF_OUTPUT_SAMPLES {
        return Err(AppError::Invalid(format!(
            "maxOutputSamples must be within 1..={MAX_NETCDF_OUTPUT_SAMPLES}"
        )));
    }
    if decoded_bytes > output_byte_budget || element_count > output_sample_budget {
        return Err(AppError::Invalid(format!(
            "netCDF window needs {element_count} samples/{decoded_bytes} bytes; request budgets are {output_sample_budget} samples/{output_byte_budget} bytes"
        )));
    }
    let source_byte_budget = request
        .max_source_bytes
        .unwrap_or(DEFAULT_NETCDF_SOURCE_BYTES);
    if source_byte_budget == 0 || source_byte_budget > MAX_NETCDF_SOURCE_BYTES {
        return Err(AppError::Invalid(format!(
            "maxSourceBytes must be within 1..={MAX_NETCDF_SOURCE_BYTES}"
        )));
    }
    let source_element_count = if request.aggregation == "point_sample" {
        element_count
    } else {
        checked_product(&request.extent, "netCDF-window source count")?
    };
    let source_decoded_bytes = source_element_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| AppError::Invalid("netCDF-window source byte size overflow".into()))?;
    if u64::try_from(source_decoded_bytes)
        .map_err(|_| AppError::Invalid("netCDF-window source byte size overflow".into()))?
        > source_byte_budget
    {
        return Err(AppError::Invalid(format!(
            "netCDF window must inspect {source_decoded_bytes} converted source bytes; request budget is {source_byte_budget}"
        )));
    }

    let mut source_start = Vec::with_capacity(variable.shape.len());
    let mut source_count = Vec::with_capacity(variable.shape.len());
    let mut source_stride = Vec::with_capacity(variable.shape.len());
    if time_variable {
        source_start.push(usize::try_from(record_index).map_err(|_| {
            AppError::Invalid("record index is too large for this platform".into())
        })?);
        source_count.push(1);
        source_stride.push(1);
    }
    for canonical_axis in source_axes.iter().copied() {
        source_start.push(request.origin[canonical_axis]);
        source_count.push(if request.aggregation == "point_sample" {
            output_shape[canonical_axis]
        } else {
            request.extent[canonical_axis]
        });
        source_stride.push(if request.aggregation == "point_sample" {
            request.stride[canonical_axis]
        } else {
            1
        });
    }
    Ok(PreparedNetcdfWindow {
        source_shape,
        output_shape,
        source_start,
        source_count,
        source_stride,
        source_axes,
        source_element_count,
        source_decoded_bytes,
        source_byte_budget,
    })
}

fn encode_netcdf_window(
    request: &NetcdfWindowRequest,
    payload: NetcdfWindowPayload,
) -> AppResult<Vec<u8>> {
    let header = NetcdfWindowHeader {
        schema: "nimbus.studio.netcdf-window",
        schema_version: NETCDF_WINDOW_SCHEMA_VERSION,
        request_id: request.request_id.clone(),
        aggregation: request.aggregation.clone(),
        weighting: "uniform_cell_volume",
        missing_data_policy: "propagate_non_finite",
        sample_location: if request.aggregation == "point_sample" {
            "source_cell_center"
        } else {
            "block_center"
        },
        source_shape: payload.source_shape,
        origin: request.origin.clone(),
        extent: request.extent.clone(),
        stride: request.stride.clone(),
        output_shape: payload.output_shape,
        element_count: payload.values.len(),
        decoded_bytes: payload.values.len() * std::mem::size_of::<f32>(),
        source_read_bytes: payload.source_read_bytes,
        source_read_accounting: payload.source_read_accounting,
        native_internal_allocation_bounded: payload.native_internal_allocation_bounded,
        dtype: "<f4",
        order: "C",
        source_format: payload.source_format,
        cancellation_mode: payload.cancellation_mode,
        non_interruptible_phase: payload.non_interruptible_phase,
    };
    let mut json = serde_json::to_vec(&header).map_err(|error| {
        AppError::Task(format!("netCDF-window metadata encode failed: {error}"))
    })?;
    let padding = (8
        - ((NETCDF_WINDOW_MAGIC.len() + std::mem::size_of::<u32>() + json.len()) % 8))
        % 8;
    json.resize(json.len() + padding, b' ');
    let json_len = u32::try_from(json.len())
        .map_err(|_| AppError::Task("netCDF-window metadata is too large".into()))?;
    let body_len = payload
        .values
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| AppError::Task("netCDF-window response size overflow".into()))?;
    let capacity = NETCDF_WINDOW_MAGIC
        .len()
        .checked_add(std::mem::size_of::<u32>())
        .and_then(|value| value.checked_add(json.len()))
        .and_then(|value| value.checked_add(body_len))
        .ok_or_else(|| AppError::Task("netCDF-window response size overflow".into()))?;
    let mut encoded = Vec::with_capacity(capacity);
    encoded.extend_from_slice(&NETCDF_WINDOW_MAGIC);
    encoded.extend_from_slice(&json_len.to_le_bytes());
    encoded.extend_from_slice(&json);
    for value in payload.values {
        encoded.extend_from_slice(&value.to_le_bytes());
    }
    Ok(encoded)
}

/// Read a bounded canonical x/y(/z) subset from classic netCDF or netCDF4.
/// The binary response is `NMBNCW1\0`, a little-endian u32 JSON-header length,
/// that UTF-8 versioned report, then little-endian float32 C-order values.
#[tauri::command]
pub async fn read_netcdf_field_window(
    path: String,
    variable: String,
    record_index: u64,
    request: NetcdfWindowRequest,
) -> AppResult<Response> {
    let guard = register_netcdf_read(&request)?;
    let worker_control = guard.control();
    let joined = tauri::async_runtime::spawn_blocking(
        move || -> AppResult<super::io_budget::NativeWindowReadResult> {
            let cancellation_request_id = request.request_id.clone();
            super::io_budget::read_with_budget(
                usize::try_from(
                    request
                        .max_source_bytes
                        .unwrap_or(DEFAULT_NETCDF_SOURCE_BYTES),
                )
                .map_err(|_| AppError::Invalid("netCDF source budget does not fit usize".into()))?,
                request
                    .max_output_bytes
                    .unwrap_or(DEFAULT_NETCDF_OUTPUT_BYTES),
                || worker_control.check(&cancellation_request_id),
                || {
                    worker_control.set_phase(NetcdfReadPhase::Header);
                    worker_control.check(&request.request_id)?;
                    let mut file = File::open(&path)?;
                    let mut magic = [0_u8; 4];
                    file.read_exact(&mut magic)?;
                    let payload = if &magic[..3] == b"CDF" && matches!(magic[3], 1 | 2) {
                        read_classic_field_window(
                            &path,
                            &variable,
                            record_index,
                            &request,
                            &worker_control,
                        )?
                    } else {
                        super::netcdf_dynamic::read_field_window(
                            &path,
                            &variable,
                            record_index,
                            &request,
                            &worker_control,
                        )?
                    };
                    worker_control.set_phase(NetcdfReadPhase::Encode);
                    worker_control.check(&request.request_id)?;
                    encode_netcdf_window(&request, payload)
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
pub async fn read_netcdf_values(path: String, variable: String) -> AppResult<Vec<f64>> {
    tauri::async_runtime::spawn_blocking(move || {
        let mut file = File::open(&path)?;
        let mut magic = [0_u8; 4];
        file.read_exact(&mut magic)?;
        if &magic[..3] == b"CDF" && matches!(magic[3], 1 | 2) {
            read_classic_values(&path, &variable)
        } else {
            super::netcdf_dynamic::read_values(&path, &variable)
        }
    })
    .await
    .map_err(|error| AppError::Task(error.to_string()))?
}

#[derive(Debug)]
struct Reader<R> {
    input: R,
    bytes_read: usize,
}

impl<R: Read> Reader<R> {
    fn new(input: R) -> Self {
        Self {
            input,
            bytes_read: 0,
        }
    }

    fn take(&mut self, count: usize) -> AppResult<Vec<u8>> {
        const MAX_HEADER_BYTES: usize = 64 * 1024 * 1024;
        let end = self
            .bytes_read
            .checked_add(count)
            .ok_or_else(|| AppError::Invalid("netCDF header offset overflow".into()))?;
        if end > MAX_HEADER_BYTES {
            return Err(AppError::Invalid(
                "netCDF header exceeds the 64 MiB metadata safety limit".into(),
            ));
        }
        let mut value = vec![0_u8; count];
        self.input.read_exact(&mut value).map_err(|error| {
            if error.kind() == std::io::ErrorKind::UnexpectedEof {
                AppError::Invalid("truncated netCDF header".into())
            } else {
                AppError::Io(error)
            }
        })?;
        self.bytes_read = end;
        Ok(value)
    }

    fn u32(&mut self) -> AppResult<u32> {
        Ok(u32::from_be_bytes(
            self.take(4)?.as_slice().try_into().unwrap(),
        ))
    }

    fn u64(&mut self) -> AppResult<u64> {
        Ok(u64::from_be_bytes(
            self.take(8)?.as_slice().try_into().unwrap(),
        ))
    }

    fn name(&mut self) -> AppResult<String> {
        let length = usize::try_from(self.u32()?)
            .map_err(|_| AppError::Invalid("netCDF name too large".into()))?;
        let raw = self.take(length)?;
        self.take((4 - length % 4) % 4)?;
        String::from_utf8(raw).map_err(|_| AppError::Invalid("netCDF name is not UTF-8".into()))
    }
}

#[derive(Debug)]
struct Attribute {
    name: String,
    data_type: u32,
    text: Option<String>,
    numeric: Vec<f64>,
}

#[derive(Debug)]
struct VariableLayout {
    metadata: NetcdfVariable,
    data_type: u32,
    vsize: u64,
    begin: u64,
    record_variable: bool,
}

#[derive(Debug)]
struct ClassicHeader {
    metadata: NetcdfMetadata,
    variables: Vec<VariableLayout>,
    record_stride: u64,
}

fn type_size(value: u32) -> AppResult<usize> {
    match value {
        1 | 2 => Ok(1),
        3 => Ok(2),
        4 | 5 => Ok(4),
        6 => Ok(8),
        _ => Err(AppError::Invalid(format!(
            "unsupported netCDF classic type {value}"
        ))),
    }
}

fn type_name(value: u32) -> AppResult<String> {
    Ok(match value {
        1 => "byte",
        2 => "char",
        3 => "short",
        4 => "int",
        5 => "float",
        6 => "double",
        _ => {
            return Err(AppError::Invalid(format!(
                "unsupported netCDF classic type {value}"
            )));
        }
    }
    .into())
}

fn list_count<R: Read>(reader: &mut Reader<R>, expected_tag: u32) -> AppResult<u32> {
    let tag = reader.u32()?;
    let count = reader.u32()?;
    if tag == 0 && count == 0 {
        return Ok(0);
    }
    if tag != expected_tag {
        return Err(AppError::Invalid(format!(
            "invalid netCDF header tag {tag}; expected {expected_tag}"
        )));
    }
    Ok(count)
}

fn attributes<R: Read>(reader: &mut Reader<R>) -> AppResult<Vec<Attribute>> {
    let count = list_count(reader, NC_ATTRIBUTE)?;
    let mut values = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let name = reader.name()?;
        let data_type = reader.u32()?;
        let elements = usize::try_from(reader.u32()?)
            .map_err(|_| AppError::Invalid("attribute too large".into()))?;
        let size = type_size(data_type)?
            .checked_mul(elements)
            .ok_or_else(|| AppError::Invalid("attribute size overflow".into()))?;
        let raw = reader.take(size)?;
        reader.take((4 - size % 4) % 4)?;
        let text = (data_type == NC_CHAR).then(|| {
            String::from_utf8_lossy(&raw)
                .trim_end_matches('\0')
                .to_string()
        });
        let numeric = if data_type == NC_CHAR {
            Vec::new()
        } else {
            decode_numeric_f64(&raw, data_type)?
        };
        values.push(Attribute {
            name,
            data_type,
            text,
            numeric,
        });
    }
    Ok(values)
}

fn attr_text(values: &[Attribute], name: &str) -> Option<String> {
    values
        .iter()
        .find(|value| value.name == name)
        .and_then(|value| value.text.clone())
}

fn parse_classic_metadata<R: Read>(input: R, path: String) -> AppResult<NetcdfMetadata> {
    Ok(parse_classic_header(input, path)?.metadata)
}

fn parse_classic_header<R: Read>(input: R, path: String) -> AppResult<ClassicHeader> {
    let mut reader = Reader::new(input);
    let magic = reader.take(4)?;
    if &magic[0..3] != b"CDF" {
        return Err(AppError::Invalid("not a netCDF classic/64-bit-offset file; netCDF4/HDF5 requires the packaged libnetcdf adapter".into()));
    }
    let version = magic[3];
    if version != 1 && version != 2 {
        return Err(AppError::Invalid(format!(
            "unsupported netCDF container version CDF-{version}; netCDF4/HDF5 requires the packaged libnetcdf adapter"
        )));
    }
    let header_record_count = u64::from(reader.u32()?);
    let dimension_count = list_count(&mut reader, NC_DIMENSION)?;
    let mut dimensions = Vec::with_capacity(dimension_count as usize);
    for _ in 0..dimension_count {
        let name = reader.name()?;
        let raw_length = u64::from(reader.u32()?);
        dimensions.push(NetcdfDimension {
            name,
            length: if raw_length == 0 {
                header_record_count
            } else {
                raw_length
            },
            unlimited: raw_length == 0,
        });
    }
    let _global_attributes = attributes(&mut reader)?;
    let variable_count = list_count(&mut reader, NC_VARIABLE)?;
    let mut layouts = Vec::with_capacity(variable_count as usize);
    for _ in 0..variable_count {
        let name = reader.name()?;
        let dim_count = reader.u32()?;
        let mut dim_ids = Vec::with_capacity(dim_count as usize);
        for _ in 0..dim_count {
            dim_ids.push(reader.u32()? as usize);
        }
        let attrs = attributes(&mut reader)?;
        let data_type = reader.u32()?;
        let vsize = u64::from(reader.u32()?);
        let begin = if version == 1 {
            u64::from(reader.u32()?)
        } else {
            reader.u64()?
        };
        let mut dim_names = Vec::with_capacity(dim_ids.len());
        let mut shape = Vec::with_capacity(dim_ids.len());
        for id in &dim_ids {
            let dim = dimensions.get(*id).ok_or_else(|| {
                AppError::Invalid(format!(
                    "variable '{name}' references missing dimension {id}"
                ))
            })?;
            dim_names.push(dim.name.clone());
            shape.push(dim.length);
        }
        let metadata = NetcdfVariable {
            name,
            data_type: type_name(data_type)?,
            dimensions: dim_names,
            shape,
            units: attr_text(&attrs, "units"),
            long_name: attr_text(&attrs, "long_name"),
            cf_decode: build_netcdf_cf_decode(
                &type_name(data_type)?,
                &attrs
                    .iter()
                    .filter(|attribute| !attribute.numeric.is_empty())
                    .map(|attribute| NetcdfNumericAttribute {
                        name: attribute.name.clone(),
                        data_type: type_name(attribute.data_type)
                            .unwrap_or_else(|_| "unknown".into()),
                        values: attribute.numeric.clone(),
                    })
                    .collect::<Vec<_>>(),
                attr_text(&attrs, "_Unsigned").as_deref(),
            ),
        };
        let record_variable = dim_ids.first().is_some_and(|id| dimensions[*id].unlimited);
        layouts.push(VariableLayout {
            metadata,
            data_type,
            vsize,
            begin,
            record_variable,
        });
    }
    let variables: Vec<NetcdfVariable> =
        layouts.iter().map(|value| value.metadata.clone()).collect();
    let names: std::collections::BTreeSet<&str> =
        variables.iter().map(|value| value.name.as_str()).collect();
    let cm1_markers = ["xh", "xf", "yh", "yf", "zh", "zf"];
    let marker_count = cm1_markers
        .iter()
        .filter(|name| names.contains(**name))
        .count();
    let cm1_index_dimensions = ["ni", "nj", "nk"].iter().all(|wanted| {
        dimensions
            .iter()
            .any(|dimension| dimension.name.eq_ignore_ascii_case(wanted))
    });
    let safe_plot_field = layouts.iter().any(|layout| {
        layout.data_type != NC_CHAR && is_qualified_cm1_history_variable(&layout.metadata)
    });
    let record_count = dimensions
        .iter()
        .find(|dimension| dimension.name.eq_ignore_ascii_case("time"))
        .or_else(|| dimensions.iter().find(|dimension| dimension.unlimited))
        .map(|dimension| dimension.length)
        .unwrap_or(1);
    let metadata = NetcdfMetadata {
        schema: "nimbus.studio.netcdf-metadata",
        schema_version: 1,
        path,
        container_format: if version == 1 { "netcdf_classic" } else { "netcdf_64bit_offset" }.into(),
        record_count,
        dimensions,
        variables,
        source_classification: if marker_count >= 4 || cm1_index_dimensions { "cm1_likely" } else { "unclassified_netcdf" }.into(),
        plotting_supported: safe_plot_field,
        plot_unavailable_reason: if safe_plot_field {
            "Qualified CM1 scalar volumes and horizontal diagnostics are routed to plots only when names, units, rank, and staggering match; declared rain/rate conversions are applied by Studio while ambiguous fields remain inspect-only."
        } else {
            "No safely mapped 3-D CM1 plot field was found; variables remain inspect-only."
        }.into(),
    };
    let record_stride = layouts
        .iter()
        .filter(|value| value.record_variable)
        .try_fold(0_u64, |sum, value| {
            sum.checked_add(value.vsize)
                .ok_or_else(|| AppError::Invalid("netCDF record stride overflow".into()))
        })?;
    Ok(ClassicHeader {
        metadata,
        variables: layouts,
        record_stride,
    })
}

fn decode_numeric(raw: &[u8], data_type: u32) -> AppResult<Vec<f32>> {
    let width = type_size(data_type)?;
    raw.chunks_exact(width)
        .map(|bytes| {
            Ok(match data_type {
                1 => i8::from_be_bytes([bytes[0]]) as f32,
                3 => i16::from_be_bytes(bytes.try_into().unwrap()) as f32,
                4 => i32::from_be_bytes(bytes.try_into().unwrap()) as f32,
                5 => f32::from_bits(u32::from_be_bytes(bytes.try_into().unwrap())),
                6 => f64::from_bits(u64::from_be_bytes(bytes.try_into().unwrap())) as f32,
                _ => {
                    return Err(AppError::Invalid(
                        "character variables cannot be plotted".into(),
                    ));
                }
            })
        })
        .collect()
}

fn decode_numeric_f64(raw: &[u8], data_type: u32) -> AppResult<Vec<f64>> {
    let width = type_size(data_type)?;
    raw.chunks_exact(width)
        .map(|bytes| {
            Ok(match data_type {
                1 => i8::from_be_bytes([bytes[0]]) as f64,
                3 => i16::from_be_bytes(bytes.try_into().unwrap()) as f64,
                4 => i32::from_be_bytes(bytes.try_into().unwrap()) as f64,
                5 => f32::from_bits(u32::from_be_bytes(bytes.try_into().unwrap())) as f64,
                6 => f64::from_bits(u64::from_be_bytes(bytes.try_into().unwrap())),
                _ => {
                    return Err(AppError::Invalid(
                        "character variables cannot be used as coordinates".into(),
                    ));
                }
            })
        })
        .collect()
}

fn axis_kind(name: &str) -> Option<usize> {
    let lower = name.to_ascii_lowercase();
    if lower.starts_with('x') || lower == "ni" || lower == "nip1" {
        Some(0)
    } else if lower.starts_with('y') || lower == "nj" || lower == "njp1" {
        Some(1)
    } else if lower.starts_with('z') || lower == "nk" || lower == "nkp1" {
        Some(2)
    } else {
        None
    }
}

fn read_classic_field(path: &str, variable: &str, record_index: u64) -> AppResult<Vec<u8>> {
    let header = parse_classic_header(File::open(path)?, path.into())?;
    let layout = header
        .variables
        .iter()
        .find(|value| value.metadata.name == variable)
        .ok_or_else(|| AppError::NotFound(format!("variable '{variable}' in {path}")))?;
    let mut dims = layout.metadata.dimensions.as_slice();
    let mut shape = layout.metadata.shape.as_slice();
    let time_variable = layout.record_variable
        || dims
            .first()
            .is_some_and(|name| name.eq_ignore_ascii_case("time"));
    let record_count = if time_variable {
        shape.first().copied().unwrap_or(0)
    } else {
        1
    };
    if time_variable {
        dims = &dims[1..];
        shape = &shape[1..];
    }
    if !matches!(dims.len(), 2 | 3) {
        return Err(AppError::Invalid(format!(
            "variable '{variable}' is not a 2-D or 3-D CM1 field after selecting time"
        )));
    }
    let rank = dims.len();
    let axes: Vec<usize> = dims
        .iter()
        .map(|name| {
            axis_kind(name)
                .ok_or_else(|| AppError::Invalid(format!("cannot map dimension '{name}' to x/y/z")))
        })
        .collect::<AppResult<_>>()?;
    let mut canonical_shape = vec![0_usize; rank];
    for (source_axis, canonical_axis) in axes.iter().enumerate() {
        if *canonical_axis >= rank {
            return Err(AppError::Invalid(format!(
                "variable '{variable}' contains a vertical axis in a 2-D field"
            )));
        }
        if canonical_shape[*canonical_axis] != 0 {
            return Err(AppError::Invalid("duplicate x/y/z dimensions".into()));
        }
        canonical_shape[*canonical_axis] = usize::try_from(shape[source_axis])
            .map_err(|_| AppError::Invalid("field dimension too large".into()))?;
    }
    if canonical_shape.contains(&0) {
        return Err(AppError::Invalid(
            "field does not contain each required canonical dimension".into(),
        ));
    }
    if record_index >= record_count {
        return Err(AppError::Invalid(format!(
            "record {record_index} is outside 0..{record_count}"
        )));
    }
    let elements = shape.iter().try_fold(1_u64, |value, length| {
        value
            .checked_mul(*length)
            .ok_or_else(|| AppError::Invalid("field size overflow".into()))
    })?;
    let byte_count = elements
        .checked_mul(type_size(layout.data_type)? as u64)
        .ok_or_else(|| AppError::Invalid("field byte size overflow".into()))?;
    let record_offset = if layout.record_variable {
        record_index.checked_mul(header.record_stride)
    } else if time_variable {
        record_index.checked_mul(byte_count)
    } else {
        Some(0)
    }
    .ok_or_else(|| AppError::Invalid("record offset overflow".into()))?;
    let offset = layout
        .begin
        .checked_add(record_offset)
        .ok_or_else(|| AppError::Invalid("record offset overflow".into()))?;
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(offset))?;
    let mut raw = vec![
        0_u8;
        usize::try_from(byte_count).map_err(|_| AppError::Invalid(
            "field is too large for this platform".into()
        ))?
    ];
    file.read_exact(&mut raw)?;
    let source = decode_numeric(&raw, layout.data_type)?
        .into_iter()
        .map(|value| {
            decode_netcdf_cf_value(&layout.metadata.cf_decode, f64::from(value))
                .map(|decoded| decoded as f32)
        })
        .collect::<AppResult<Vec<_>>>()?;
    let mut canonical = vec![0_f32; source.len()];
    let source_shape: Vec<usize> = shape.iter().map(|value| *value as usize).collect();
    for (target_index, target_value) in canonical.iter_mut().enumerate() {
        let mut remainder = target_index;
        let mut canonical_indices = vec![0_usize; rank];
        for axis in (0..rank).rev() {
            canonical_indices[axis] = remainder % canonical_shape[axis];
            remainder /= canonical_shape[axis];
        }
        let source_indices: Vec<usize> = axes.iter().map(|axis| canonical_indices[*axis]).collect();
        let source_index = source_indices
            .iter()
            .enumerate()
            .fold(0_usize, |linear, (axis, index)| {
                linear * source_shape[axis] + index
            });
        *target_value = source[source_index];
    }
    crate::zfp3d::encode_npy_f32(&canonical_shape, &canonical)
}

fn decode_numeric_one(raw: &[u8], data_type: u32) -> AppResult<f32> {
    Ok(match data_type {
        1 => i8::from_be_bytes([raw[0]]) as f32,
        3 => i16::from_be_bytes(raw.try_into().unwrap()) as f32,
        4 => i32::from_be_bytes(raw.try_into().unwrap()) as f32,
        5 => f32::from_bits(u32::from_be_bytes(raw.try_into().unwrap())),
        6 => f64::from_bits(u64::from_be_bytes(raw.try_into().unwrap())) as f32,
        _ => {
            return Err(AppError::Invalid(
                "character variables cannot be plotted".into(),
            ));
        }
    })
}

fn linear_index(indices: &[usize], shape: &[usize], context: &str) -> AppResult<usize> {
    indices
        .iter()
        .zip(shape)
        .try_fold(0_usize, |linear, (index, dimension)| {
            linear
                .checked_mul(*dimension)
                .and_then(|value| value.checked_add(*index))
                .ok_or_else(|| AppError::Invalid(format!("{context} overflow")))
        })
}

fn read_classic_field_window(
    path: &str,
    variable: &str,
    record_index: u64,
    request: &NetcdfWindowRequest,
    control: &NetcdfReadControl,
) -> AppResult<NetcdfWindowPayload> {
    let header = parse_classic_header(File::open(path)?, path.into())?;
    let layout = header
        .variables
        .iter()
        .find(|value| value.metadata.name == variable)
        .ok_or_else(|| AppError::NotFound(format!("variable '{variable}' in {path}")))?;
    let time_variable = layout.record_variable
        || layout
            .metadata
            .dimensions
            .first()
            .is_some_and(|name| name.eq_ignore_ascii_case("time"));
    let record_count = if time_variable {
        layout.metadata.shape.first().copied().unwrap_or(0)
    } else {
        1
    };
    let prepared = prepare_netcdf_window(
        &layout.metadata,
        record_count,
        time_variable,
        record_index,
        request,
    )?;
    let skip = usize::from(time_variable);
    let source_spatial_shape = layout.metadata.shape[skip..]
        .iter()
        .map(|value| {
            usize::try_from(*value).map_err(|_| {
                AppError::Invalid("field dimension is too large for this platform".into())
            })
        })
        .collect::<AppResult<Vec<_>>>()?;
    let source_start = &prepared.source_start[skip..];
    let source_count = &prepared.source_count[skip..];
    let source_stride = &prepared.source_stride[skip..];
    let rank = source_spatial_shape.len();
    let last_axis = rank - 1;
    let row_span = (source_count[last_axis] - 1)
        .checked_mul(source_stride[last_axis])
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| AppError::Invalid("netCDF source row span overflow".into()))?;
    let prefix_count = checked_product(&source_count[..last_axis], "netCDF source row count")?;
    let width = type_size(layout.data_type)?;
    let source_read_bytes = prefix_count
        .checked_mul(row_span)
        .and_then(|value| value.checked_mul(width))
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| AppError::Invalid("netCDF source byte count overflow".into()))?;
    if source_read_bytes > prepared.source_byte_budget {
        return Err(AppError::Invalid(format!(
            "classic netCDF window needs {source_read_bytes} source bytes; request budget is {}",
            prepared.source_byte_budget
        )));
    }

    let spatial_elements = checked_product(&source_spatial_shape, "netCDF record sample count")?;
    let record_bytes = spatial_elements
        .checked_mul(width)
        .ok_or_else(|| AppError::Invalid("netCDF record byte count overflow".into()))?;
    let record_offset = if layout.record_variable {
        record_index.checked_mul(header.record_stride)
    } else if time_variable {
        record_index.checked_mul(
            u64::try_from(record_bytes)
                .map_err(|_| AppError::Invalid("netCDF record byte count is too large".into()))?,
        )
    } else {
        Some(0)
    }
    .ok_or_else(|| AppError::Invalid("netCDF record offset overflow".into()))?;
    let record_base = layout
        .begin
        .checked_add(record_offset)
        .ok_or_else(|| AppError::Invalid("netCDF record offset overflow".into()))?;
    let row_bytes = row_span
        .checked_mul(width)
        .ok_or_else(|| AppError::Invalid("netCDF source row byte count overflow".into()))?;
    let mut scratch = vec![0_u8; row_bytes];
    let mut canonical_source = vec![0_f32; prepared.source_element_count];
    let canonical_read_shape = if request.aggregation == "point_sample" {
        &prepared.output_shape
    } else {
        &request.extent
    };
    let mut file = File::open(path)?;
    control.set_phase(NetcdfReadPhase::ClassicRows);
    for prefix_linear in 0..prefix_count {
        control.check(&request.request_id)?;
        let mut remainder = prefix_linear;
        let mut source_output_indices = vec![0_usize; rank];
        for axis in (0..last_axis).rev() {
            source_output_indices[axis] = remainder % source_count[axis];
            remainder /= source_count[axis];
        }
        let mut source_indices = vec![0_usize; rank];
        for axis in 0..last_axis {
            source_indices[axis] =
                source_start[axis] + source_output_indices[axis] * source_stride[axis];
        }
        source_indices[last_axis] = source_start[last_axis];
        let row_linear = linear_index(
            &source_indices,
            &source_spatial_shape,
            "netCDF source index",
        )?;
        let byte_offset = row_linear
            .checked_mul(width)
            .and_then(|value| u64::try_from(value).ok())
            .and_then(|value| record_base.checked_add(value))
            .ok_or_else(|| AppError::Invalid("netCDF source offset overflow".into()))?;
        // A bounded contiguous row is dramatically cheaper than one seek per
        // point on high-resolution fields. The source budget above accounts
        // for every byte in this span, including skipped fastest-axis cells.
        file.seek(SeekFrom::Start(byte_offset))?;
        file.read_exact(&mut scratch)?;
        for last_output in 0..source_count[last_axis] {
            source_output_indices[last_axis] = last_output;
            let mut canonical_indices = vec![0_usize; rank];
            for (source_axis, canonical_axis) in prepared.source_axes.iter().copied().enumerate() {
                canonical_indices[canonical_axis] = source_output_indices[source_axis];
            }
            let target = linear_index(
                &canonical_indices,
                canonical_read_shape,
                "netCDF target index",
            )?;
            let source_byte = last_output
                .checked_mul(source_stride[last_axis])
                .and_then(|value| value.checked_mul(width))
                .ok_or_else(|| AppError::Invalid("netCDF row sample offset overflow".into()))?;
            canonical_source[target] = decode_netcdf_cf_value(
                &layout.metadata.cf_decode,
                f64::from(decode_numeric_one(
                    &scratch[source_byte..source_byte + width],
                    layout.data_type,
                )?),
            )? as f32;
        }
    }
    control.check(&request.request_id)?;
    let output = if request.aggregation == "point_sample" {
        canonical_source
    } else {
        aggregate_c_order(
            &canonical_source,
            &request.extent,
            &prepared.output_shape,
            &request.stride,
            &request.aggregation,
        )?
    };
    Ok(NetcdfWindowPayload {
        source_shape: prepared.source_shape,
        output_shape: prepared.output_shape,
        values: output,
        source_read_bytes,
        source_read_accounting: "file_payload_exact",
        native_internal_allocation_bounded: true,
        source_format: header.metadata.container_format,
        cancellation_mode: "cooperative",
        non_interruptible_phase: None,
    })
}

fn read_classic_values(path: &str, variable: &str) -> AppResult<Vec<f64>> {
    const MAX_COORDINATE_VALUES: u64 = 1_000_000;
    let header = parse_classic_header(File::open(path)?, path.into())?;
    let layout = header
        .variables
        .iter()
        .find(|value| value.metadata.name == variable)
        .ok_or_else(|| AppError::NotFound(format!("variable '{variable}' in {path}")))?;
    if layout.metadata.dimensions.len() != 1 {
        return Err(AppError::Invalid(format!(
            "coordinate variable '{variable}' is not one-dimensional"
        )));
    }
    let count = layout.metadata.shape[0];
    if count > MAX_COORDINATE_VALUES {
        return Err(AppError::Invalid(
            "coordinate array exceeds the one-million-value safety limit".into(),
        ));
    }
    let width = type_size(layout.data_type)? as u64;
    let mut file = File::open(path)?;
    if layout.record_variable {
        let mut result = Vec::with_capacity(count as usize);
        for record in 0..count {
            let offset =
                layout
                    .begin
                    .checked_add(record.checked_mul(header.record_stride).ok_or_else(|| {
                        AppError::Invalid("coordinate record offset overflow".into())
                    })?)
                    .ok_or_else(|| AppError::Invalid("coordinate record offset overflow".into()))?;
            file.seek(SeekFrom::Start(offset))?;
            let mut raw = vec![0_u8; width as usize];
            file.read_exact(&mut raw)?;
            result.extend(decode_numeric_f64(&raw, layout.data_type)?);
        }
        result
            .into_iter()
            .map(|value| decode_netcdf_cf_value(&layout.metadata.cf_decode, value))
            .collect()
    } else {
        let byte_count = count
            .checked_mul(width)
            .ok_or_else(|| AppError::Invalid("coordinate byte size overflow".into()))?;
        file.seek(SeekFrom::Start(layout.begin))?;
        let mut raw = vec![
            0_u8;
            usize::try_from(byte_count).map_err(|_| AppError::Invalid(
                "coordinate array is too large for this platform".into()
            ))?
        ];
        file.read_exact(&mut raw)?;
        decode_numeric_f64(&raw, layout.data_type)?
            .into_iter()
            .map(|value| decode_netcdf_cf_value(&layout.metadata.cf_decode, value))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn history_variable(name: &str, units: Option<&str>, dimensions: &[&str]) -> NetcdfVariable {
        NetcdfVariable {
            name: name.into(),
            data_type: "float".into(),
            dimensions: dimensions.iter().map(|value| (*value).into()).collect(),
            shape: dimensions.iter().map(|_| 2).collect(),
            units: units.map(Into::into),
            long_name: None,
            cf_decode: build_netcdf_cf_decode("float", &[], None),
        }
    }

    #[test]
    fn qualifies_only_identity_unit_scalar_cm1_history_fields() {
        for (name, unit) in [
            ("uinterp", "m/s"),
            ("th", "K"),
            ("thpert", "K"),
            ("prs", "Pa"),
            ("prspert", "Pa"),
            ("rho", "kg/m^3"),
            ("qv", "kg/kg"),
            ("dbz", "dBZ"),
            ("zvort", "1/s"),
            ("buoyancy", "m/s^2"),
            ("pv", "K m2/kg/s"),
        ] {
            assert!(
                is_qualified_cm1_history_variable(&history_variable(
                    name,
                    Some(unit),
                    &["time", "nk", "nj", "ni"],
                )),
                "{name} with its CM1 unit should qualify"
            );
        }
        for (name, unit) in [
            ("cref", "dBZ"),
            ("cape", "J/kg"),
            ("cin", "J/kg"),
            ("lcl", "m"),
            ("lfc", "m"),
            ("uh", "m2/s2"),
            ("pwat", "m"),
            ("lwp", "kg/m^2"),
            ("cwp", "kg/m^2"),
            ("rain", "cm"),
            ("prate", "kg/m2/s"),
        ] {
            assert!(
                is_qualified_cm1_history_variable(&history_variable(
                    name,
                    Some(unit),
                    &["time", "nj", "ni"],
                )),
                "{name} with its CM1 unit and horizontal rank should qualify"
            );
        }
        assert!(!is_qualified_cm1_history_variable(&history_variable(
            "qv",
            Some("g/kg"),
            &["time", "nk", "nj", "ni"],
        )));
        assert!(!is_qualified_cm1_history_variable(&history_variable(
            "th",
            None,
            &["time", "nk", "nj", "ni"],
        )));
        assert!(!is_qualified_cm1_history_variable(&history_variable(
            "u",
            Some("m/s"),
            &["time", "nk", "nj", "nip1"],
        )));
        assert!(!is_qualified_cm1_history_variable(&history_variable(
            "th",
            Some("K"),
            &["time", "zf", "yh", "xh"],
        )));
        assert!(!is_qualified_cm1_history_variable(&history_variable(
            "TH",
            Some("K"),
            &["time", "nk", "nj", "ni"],
        )));
        assert!(!is_qualified_cm1_history_variable(&history_variable(
            "cref",
            Some("dBZ"),
            &["time", "nk", "nj", "ni"],
        )));
        assert!(!is_qualified_cm1_history_variable(&history_variable(
            "rain",
            Some("mm"),
            &["time", "nj", "ni"],
        )));
    }

    fn push_u32(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_be_bytes());
    }
    fn push_name(out: &mut Vec<u8>, value: &str) {
        push_u32(out, value.len() as u32);
        out.extend_from_slice(value.as_bytes());
        out.resize((out.len() + 3) & !3, 0);
    }
    fn empty_list(out: &mut Vec<u8>) {
        push_u32(out, 0);
        push_u32(out, 0);
    }
    fn fixture() -> Vec<u8> {
        let mut out = b"CDF\x01".to_vec();
        push_u32(&mut out, 3);
        push_u32(&mut out, NC_DIMENSION);
        push_u32(&mut out, 2);
        push_name(&mut out, "time");
        push_u32(&mut out, 0);
        push_name(&mut out, "xh");
        push_u32(&mut out, 4);
        empty_list(&mut out);
        push_u32(&mut out, NC_VARIABLE);
        push_u32(&mut out, 1);
        push_name(&mut out, "winterp");
        push_u32(&mut out, 2);
        push_u32(&mut out, 0);
        push_u32(&mut out, 1);
        push_u32(&mut out, NC_ATTRIBUTE);
        push_u32(&mut out, 1);
        push_name(&mut out, "units");
        push_u32(&mut out, NC_CHAR);
        push_u32(&mut out, 5);
        out.extend_from_slice(b"m s-1");
        out.resize((out.len() + 3) & !3, 0);
        push_u32(&mut out, 5);
        push_u32(&mut out, 48);
        push_u32(&mut out, 0);
        out
    }

    fn field_fixture() -> Vec<u8> {
        let mut out = b"CDF\x01".to_vec();
        push_u32(&mut out, 2);
        push_u32(&mut out, NC_DIMENSION);
        push_u32(&mut out, 4);
        for (name, length) in [("time", 0), ("zh", 2), ("yh", 2), ("xh", 2)] {
            push_name(&mut out, name);
            push_u32(&mut out, length);
        }
        empty_list(&mut out);
        push_u32(&mut out, NC_VARIABLE);
        push_u32(&mut out, 1);
        push_name(&mut out, "winterp");
        push_u32(&mut out, 4);
        for id in 0..4 {
            push_u32(&mut out, id);
        }
        push_u32(&mut out, NC_ATTRIBUTE);
        push_u32(&mut out, 1);
        push_name(&mut out, "units");
        push_u32(&mut out, NC_CHAR);
        push_u32(&mut out, 3);
        out.extend_from_slice(b"m/s");
        out.resize((out.len() + 3) & !3, 0);
        push_u32(&mut out, 5);
        push_u32(&mut out, 32);
        let begin_offset = out.len();
        push_u32(&mut out, 0);
        let begin = out.len() as u32;
        out[begin_offset..begin_offset + 4].copy_from_slice(&begin.to_be_bytes());
        for record in 0..2 {
            for value in 0..8 {
                out.extend_from_slice(&((record * 100 + value) as f32).to_bits().to_be_bytes());
            }
        }
        out
    }

    fn fixed_time_field_fixture() -> Vec<u8> {
        let mut out = b"CDF\x01".to_vec();
        push_u32(&mut out, 0);
        push_u32(&mut out, NC_DIMENSION);
        push_u32(&mut out, 4);
        for (name, length) in [("time", 2), ("zh", 2), ("yh", 2), ("xh", 2)] {
            push_name(&mut out, name);
            push_u32(&mut out, length);
        }
        empty_list(&mut out);
        push_u32(&mut out, NC_VARIABLE);
        push_u32(&mut out, 1);
        push_name(&mut out, "winterp");
        push_u32(&mut out, 4);
        for id in 0..4 {
            push_u32(&mut out, id);
        }
        push_u32(&mut out, NC_ATTRIBUTE);
        push_u32(&mut out, 1);
        push_name(&mut out, "units");
        push_u32(&mut out, NC_CHAR);
        push_u32(&mut out, 3);
        out.extend_from_slice(b"m/s");
        out.resize((out.len() + 3) & !3, 0);
        push_u32(&mut out, 5);
        push_u32(&mut out, 64);
        let begin_offset = out.len();
        push_u32(&mut out, 0);
        let begin = out.len() as u32;
        out[begin_offset..begin_offset + 4].copy_from_slice(&begin.to_be_bytes());
        for record in 0..2 {
            for value in 0..8 {
                out.extend_from_slice(&((record * 100 + value) as f32).to_bits().to_be_bytes());
            }
        }
        out
    }

    fn horizontal_field_fixture() -> Vec<u8> {
        let mut out = b"CDF\x01".to_vec();
        push_u32(&mut out, 0);
        push_u32(&mut out, NC_DIMENSION);
        push_u32(&mut out, 3);
        for (name, length) in [("time", 2), ("yh", 2), ("xh", 3)] {
            push_name(&mut out, name);
            push_u32(&mut out, length);
        }
        empty_list(&mut out);
        push_u32(&mut out, NC_VARIABLE);
        push_u32(&mut out, 1);
        push_name(&mut out, "cref");
        push_u32(&mut out, 3);
        for id in 0..3 {
            push_u32(&mut out, id);
        }
        push_u32(&mut out, NC_ATTRIBUTE);
        push_u32(&mut out, 1);
        push_name(&mut out, "units");
        push_u32(&mut out, NC_CHAR);
        push_u32(&mut out, 3);
        out.extend_from_slice(b"dBZ");
        out.resize((out.len() + 3) & !3, 0);
        push_u32(&mut out, 5);
        push_u32(&mut out, 48);
        let begin_offset = out.len();
        push_u32(&mut out, 0);
        let begin = out.len() as u32;
        out[begin_offset..begin_offset + 4].copy_from_slice(&begin.to_be_bytes());
        for record in 0..2 {
            for value in 0..6 {
                out.extend_from_slice(&((record * 100 + value) as f32).to_bits().to_be_bytes());
            }
        }
        out
    }

    fn coordinate_fixture() -> Vec<u8> {
        let mut out = b"CDF\x01".to_vec();
        push_u32(&mut out, 0);
        push_u32(&mut out, NC_DIMENSION);
        push_u32(&mut out, 1);
        push_name(&mut out, "xh");
        push_u32(&mut out, 3);
        empty_list(&mut out);
        push_u32(&mut out, NC_VARIABLE);
        push_u32(&mut out, 1);
        push_name(&mut out, "xh");
        push_u32(&mut out, 1);
        push_u32(&mut out, 0);
        empty_list(&mut out);
        push_u32(&mut out, 5);
        push_u32(&mut out, 12);
        let begin_offset = out.len();
        push_u32(&mut out, 0);
        let begin = out.len() as u32;
        out[begin_offset..begin_offset + 4].copy_from_slice(&begin.to_be_bytes());
        for value in [-1.0_f32, 0.0, 1.0] {
            out.extend_from_slice(&value.to_bits().to_be_bytes());
        }
        out
    }

    fn push_short_attribute(out: &mut Vec<u8>, name: &str, values: &[i16]) {
        push_name(out, name);
        push_u32(out, 3);
        push_u32(out, values.len() as u32);
        for value in values {
            out.extend_from_slice(&value.to_be_bytes());
        }
        out.resize((out.len() + 3) & !3, 0);
    }

    fn push_float_attribute(out: &mut Vec<u8>, name: &str, value: f32) {
        push_name(out, name);
        push_u32(out, 5);
        push_u32(out, 1);
        out.extend_from_slice(&value.to_bits().to_be_bytes());
    }

    fn packed_field_fixture() -> Vec<u8> {
        let mut out = b"CDF\x01".to_vec();
        push_u32(&mut out, 0);
        push_u32(&mut out, NC_DIMENSION);
        push_u32(&mut out, 4);
        for (name, length) in [("time", 1), ("zh", 2), ("yh", 2), ("xh", 2)] {
            push_name(&mut out, name);
            push_u32(&mut out, length);
        }
        empty_list(&mut out);
        push_u32(&mut out, NC_VARIABLE);
        push_u32(&mut out, 1);
        push_name(&mut out, "winterp");
        push_u32(&mut out, 4);
        for id in 0..4 {
            push_u32(&mut out, id);
        }
        push_u32(&mut out, NC_ATTRIBUTE);
        push_u32(&mut out, 6);
        push_name(&mut out, "units");
        push_u32(&mut out, NC_CHAR);
        push_u32(&mut out, 3);
        out.extend_from_slice(b"m/s");
        out.resize((out.len() + 3) & !3, 0);
        push_float_attribute(&mut out, "scale_factor", 0.1);
        push_float_attribute(&mut out, "add_offset", 273.15);
        push_short_attribute(&mut out, "_FillValue", &[-32_767]);
        push_short_attribute(&mut out, "missing_value", &[-9_998]);
        push_short_attribute(&mut out, "valid_range", &[-100, 100]);
        push_u32(&mut out, 3);
        push_u32(&mut out, 16);
        let begin_offset = out.len();
        push_u32(&mut out, 0);
        let begin = out.len() as u32;
        out[begin_offset..begin_offset + 4].copy_from_slice(&begin.to_be_bytes());
        for value in [0_i16, 1, -32_767, -9_998, 101, -101, 50, -50] {
            out.extend_from_slice(&value.to_be_bytes());
        }
        out
    }

    fn packed_coordinate_fixture() -> Vec<u8> {
        let mut out = b"CDF\x01".to_vec();
        push_u32(&mut out, 0);
        push_u32(&mut out, NC_DIMENSION);
        push_u32(&mut out, 1);
        push_name(&mut out, "xh");
        push_u32(&mut out, 4);
        empty_list(&mut out);
        push_u32(&mut out, NC_VARIABLE);
        push_u32(&mut out, 1);
        push_name(&mut out, "xh");
        push_u32(&mut out, 1);
        push_u32(&mut out, 0);
        push_u32(&mut out, NC_ATTRIBUTE);
        push_u32(&mut out, 3);
        push_float_attribute(&mut out, "scale_factor", 0.5);
        push_float_attribute(&mut out, "add_offset", 100.0);
        push_short_attribute(&mut out, "_FillValue", &[-32_767]);
        push_u32(&mut out, 3);
        push_u32(&mut out, 8);
        let begin_offset = out.len();
        push_u32(&mut out, 0);
        let begin = out.len() as u32;
        out[begin_offset..begin_offset + 4].copy_from_slice(&begin.to_be_bytes());
        for value in [-10_i16, 0, 10, -32_767] {
            out.extend_from_slice(&value.to_be_bytes());
        }
        out
    }

    #[test]
    fn inspects_header_without_reading_variable_payloads() {
        let metadata =
            parse_classic_metadata(std::io::Cursor::new(fixture()), "cm1out.nc".into()).unwrap();
        assert_eq!(metadata.record_count, 3);
        assert_eq!(
            metadata.dimensions[0],
            NetcdfDimension {
                name: "time".into(),
                length: 3,
                unlimited: true
            }
        );
        assert_eq!(metadata.variables[0].dimensions, vec!["time", "xh"]);
        assert_eq!(metadata.variables[0].units.as_deref(), Some("m s-1"));
        assert!(
            !metadata.plotting_supported,
            "a one-dimensional variable must not be advertised as plot-ready"
        );
    }

    #[test]
    fn rejects_hdf5_without_claiming_support() {
        let error = parse_classic_metadata(
            std::io::Cursor::new(b"\x89HDF\r\n\x1a\n"),
            "cm1out.nc".into(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("netCDF4/HDF5"));
    }

    #[test]
    fn reads_one_record_and_transposes_zyx_to_xyz() {
        let path =
            std::env::temp_dir().join(format!("nimbus_cm1_field_{}_{}.nc", std::process::id(), 1));
        std::fs::write(&path, field_fixture()).unwrap();
        let npy = read_classic_field(path.to_str().unwrap(), "winterp", 1).unwrap();
        let layout = crate::npy::parse_f32_layout(&npy).unwrap();
        assert_eq!(layout.shape, vec![2, 2, 2]);
        let values: Vec<f32> = npy[layout.data_offset..]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(
            values,
            vec![100.0, 104.0, 102.0, 106.0, 101.0, 105.0, 103.0, 107.0]
        );
        let _ = std::fs::remove_file(path);
    }

    fn window_request(origin: &[usize], extent: &[usize], stride: &[usize]) -> NetcdfWindowRequest {
        NetcdfWindowRequest {
            schema_version: 2,
            request_id: "test:netcdf-window".into(),
            aggregation: "cell_mean".into(),
            origin: origin.to_vec(),
            extent: extent.to_vec(),
            stride: stride.to_vec(),
            max_output_bytes: Some(1024),
            max_output_samples: Some(256),
            max_source_bytes: Some(1024),
        }
    }

    #[test]
    fn netcdf_window_envelope_aligns_float_body_without_copy() {
        let request = window_request(&[0], &[1], &[1]);
        let payload = NetcdfWindowPayload {
            source_shape: vec![1],
            output_shape: vec![1],
            values: vec![3.5],
            source_read_bytes: 4,
            source_read_accounting: "converted_hyperslab_output",
            native_internal_allocation_bounded: true,
            source_format: "netcdf_classic".into(),
            cancellation_mode: "cooperative",
            non_interruptible_phase: None,
        };
        let encoded = encode_netcdf_window(&request, payload).unwrap();
        let header_bytes = u32::from_le_bytes(encoded[8..12].try_into().unwrap()) as usize;
        let body_offset = 12 + header_bytes;
        assert_eq!(body_offset % 8, 0);
        assert_eq!(f32::from_le_bytes(encoded[body_offset..].try_into().unwrap()), 3.5);
        let decoded: serde_json::Value = serde_json::from_slice(&encoded[12..body_offset]).unwrap();
        assert_eq!(decoded["requestId"], "test:netcdf-window");
    }

    #[test]
    fn cell_mean_handles_partial_edge_blocks_and_propagates_nonfinite_values() {
        let source = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, f32::NAN];
        let result = aggregate_c_order(&source, &[3, 3], &[2, 2], &[2, 2], "cell_mean").unwrap();
        assert_eq!(result[0], 3.0);
        assert_eq!(result[1], 4.5);
        assert_eq!(result[2], 7.5);
        assert!(result[3].is_nan());
    }

    #[test]
    fn reflectivity_mean_aggregates_linear_z_not_logarithmic_dbz() {
        let result =
            aggregate_c_order(&[0.0, 10.0], &[2], &[1], &[2], "linear_reflectivity_mean").unwrap();
        let expected = 10.0_f32 * 5.5_f32.log10();
        assert!((result[0] - expected).abs() < 1e-5);
        assert!((result[0] - 5.0).abs() > 1.0);
    }

    #[test]
    fn extrema_aggregations_preserve_compact_threshold_crossings() {
        let source = [3.0, -8.0, 5.0, -4.0];
        for (aggregation, expected) in [
            ("block_max", 5.0),
            ("block_min", -8.0),
            ("magnitude_extreme", -8.0),
        ] {
            let result = aggregate_c_order(&source, &[4], &[1], &[4], aggregation).unwrap();
            assert_eq!(result, vec![expected], "aggregation={aggregation}");
        }
    }

    #[test]
    fn point_sample_accepts_only_an_already_sparse_hyperslab() {
        assert_eq!(
            aggregate_c_order(&[1.0, 4.0, 7.0], &[3], &[3], &[4], "point_sample").unwrap(),
            vec![1.0, 4.0, 7.0]
        );
        assert!(aggregate_c_order(&[1.0, 2.0, 3.0], &[3], &[2], &[2], "point_sample").is_err());
    }

    #[test]
    fn point_sample_netcdf_plan_uses_native_hyperslab_stride_and_sparse_budget() {
        let variable = history_variable("winterp", Some("m/s"), &["time", "nk", "nj", "ni"]);
        let mut request = window_request(&[0, 0, 0], &[2, 2, 2], &[2, 2, 2]);
        request.aggregation = "point_sample".into();
        request.max_source_bytes = Some(4);
        let prepared = prepare_netcdf_window(&variable, 2, true, 0, &request).unwrap();
        assert_eq!(prepared.output_shape, vec![1, 1, 1]);
        assert_eq!(prepared.source_count, vec![1, 1, 1, 1]);
        // File axes are time,z,y,x; each spatial axis retains the canonical stride.
        assert_eq!(prepared.source_stride, vec![1, 2, 2, 2]);
        assert_eq!(prepared.source_decoded_bytes, 4);
    }

    #[test]
    fn reads_bounded_cell_mean_classic_window_in_canonical_order() {
        let path = std::env::temp_dir().join(format!(
            "nimbus_cm1_classic_window_{}.nc",
            std::process::id()
        ));
        std::fs::write(&path, field_fixture()).unwrap();
        let request = window_request(&[0, 0, 0], &[2, 2, 2], &[1, 2, 1]);
        let control = NetcdfReadControl::new();
        let payload =
            read_classic_field_window(path.to_str().unwrap(), "winterp", 1, &request, &control)
                .unwrap();
        assert_eq!(payload.source_shape, vec![2, 2, 2]);
        assert_eq!(payload.output_shape, vec![2, 1, 2]);
        assert_eq!(payload.values, vec![101.0, 105.0, 102.0, 106.0]);
        assert_eq!(payload.source_read_bytes, 32);
        assert_eq!(payload.cancellation_mode, "cooperative");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn reads_sparse_point_samples_from_classic_netcdf_in_canonical_order() {
        let path = std::env::temp_dir().join(format!(
            "nimbus_cm1_classic_points_{}.nc",
            std::process::id()
        ));
        std::fs::write(&path, field_fixture()).unwrap();
        let mut request = window_request(&[0, 0, 0], &[2, 2, 2], &[1, 2, 1]);
        request.aggregation = "point_sample".into();
        request.max_source_bytes = Some(16);
        let payload = read_classic_field_window(
            path.to_str().unwrap(),
            "winterp",
            1,
            &request,
            &NetcdfReadControl::new(),
        )
        .unwrap();
        assert_eq!(payload.output_shape, vec![2, 1, 2]);
        assert_eq!(payload.values, vec![100.0, 104.0, 101.0, 105.0]);
        assert_eq!(payload.source_read_bytes, 16);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn netcdf_window_rejects_rank_bounds_and_allocation_budget_violations() {
        let variable = history_variable("winterp", Some("m/s"), &["time", "nk", "nj", "ni"]);
        let rank_error = prepare_netcdf_window(
            &variable,
            2,
            true,
            0,
            &window_request(&[0, 0], &[2, 2], &[1, 1]),
        )
        .unwrap_err();
        assert!(rank_error.to_string().contains("exactly 3 canonical axes"));

        let bounds_error = prepare_netcdf_window(
            &variable,
            2,
            true,
            0,
            &window_request(&[1, 0, 0], &[2, 2, 2], &[1, 1, 1]),
        )
        .unwrap_err();
        assert!(bounds_error.to_string().contains("exceeds axis 0"));

        let mut budget_request = window_request(&[0, 0, 0], &[2, 2, 2], &[1, 1, 1]);
        budget_request.max_output_samples = Some(7);
        let budget_error =
            prepare_netcdf_window(&variable, 2, true, 0, &budget_request).unwrap_err();
        assert!(budget_error
            .to_string()
            .contains("needs 8 samples/32 bytes"));
    }

    #[test]
    fn classic_window_observes_cancellation_before_file_rows() {
        let path = std::env::temp_dir().join(format!(
            "nimbus_cm1_cancelled_window_{}.nc",
            std::process::id()
        ));
        std::fs::write(&path, field_fixture()).unwrap();
        let request = window_request(&[0, 0, 0], &[2, 2, 2], &[1, 1, 1]);
        let control = NetcdfReadControl::new();
        control.cancelled.store(true, Ordering::Release);
        let error =
            read_classic_field_window(path.to_str().unwrap(), "winterp", 0, &request, &control)
                .unwrap_err();
        assert!(error.to_string().contains("netCDF field read cancelled"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn reads_fixed_size_time_dimensions_as_records() {
        let path = std::env::temp_dir().join(format!(
            "nimbus_cm1_fixed_time_{}_{}.nc",
            std::process::id(),
            1
        ));
        std::fs::write(&path, fixed_time_field_fixture()).unwrap();
        let metadata = parse_classic_metadata(
            File::open(&path).unwrap(),
            path.to_string_lossy().into_owned(),
        )
        .unwrap();
        assert_eq!(metadata.record_count, 2);
        assert!(metadata.plotting_supported);
        let npy = read_classic_field(path.to_str().unwrap(), "winterp", 1).unwrap();
        let layout = crate::npy::parse_f32_layout(&npy).unwrap();
        let values: Vec<f32> = npy[layout.data_offset..]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(
            values,
            vec![100.0, 104.0, 102.0, 106.0, 101.0, 105.0, 103.0, 107.0]
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn reads_horizontal_records_and_transposes_yx_to_xy() {
        let path = std::env::temp_dir().join(format!(
            "nimbus_cm1_horizontal_{}_{}.nc",
            std::process::id(),
            1
        ));
        std::fs::write(&path, horizontal_field_fixture()).unwrap();
        let metadata = parse_classic_metadata(
            File::open(&path).unwrap(),
            path.to_string_lossy().into_owned(),
        )
        .unwrap();
        assert!(metadata.plotting_supported);
        let npy = read_classic_field(path.to_str().unwrap(), "cref", 1).unwrap();
        let layout = crate::npy::parse_f32_layout(&npy).unwrap();
        assert_eq!(layout.shape, vec![3, 2]);
        let values: Vec<f32> = npy[layout.data_offset..]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![100.0, 103.0, 101.0, 104.0, 102.0, 105.0]);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn reads_classic_coordinate_values_without_field_payload_expansion() {
        let path =
            std::env::temp_dir().join(format!("nimbus_cm1_coordinate_{}.nc", std::process::id()));
        std::fs::write(&path, coordinate_fixture()).unwrap();
        assert_eq!(
            read_classic_values(path.to_str().unwrap(), "xh").unwrap(),
            vec![-1.0, 0.0, 1.0]
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn decodes_cf_packing_and_missing_values_consistently_for_classic_reads() {
        let path = std::env::temp_dir().join(format!(
            "nimbus_cm1_packed_classic_{}.nc",
            std::process::id()
        ));
        std::fs::write(&path, packed_field_fixture()).unwrap();
        let metadata = parse_classic_metadata(
            File::open(&path).unwrap(),
            path.to_string_lossy().into_owned(),
        )
        .unwrap();
        let policy = &metadata.variables[0].cf_decode;
        assert_eq!(policy.status, "applied");
        assert_eq!(policy.fill_value_source.as_deref(), Some("attribute"));
        assert_eq!(policy.missing_values, vec![-9_998.0]);
        assert_eq!((policy.scale_factor.unwrap() as f32), 0.1);
        assert_eq!((policy.add_offset.unwrap() as f32), 273.15);

        let npy = read_classic_field(path.to_str().unwrap(), "winterp", 0).unwrap();
        let layout = crate::npy::parse_f32_layout(&npy).unwrap();
        let values = npy[layout.data_offset..]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(layout.shape, vec![2, 2, 2]);
        assert!((values[0] - 273.15).abs() < 1e-4);
        assert!(values[1].is_nan());
        assert!(values[2].is_nan());
        assert!((values[3] - 278.15).abs() < 1e-4);
        assert!((values[4] - 273.25).abs() < 1e-4);
        assert!(values[5].is_nan());
        assert!(values[6].is_nan());
        assert!((values[7] - 268.15).abs() < 1e-4);

        let request = window_request(&[0, 0, 0], &[2, 2, 2], &[1, 1, 1]);
        let payload = read_classic_field_window(
            path.to_str().unwrap(),
            "winterp",
            0,
            &request,
            &NetcdfReadControl::new(),
        )
        .unwrap();
        for (full, window) in values.iter().zip(&payload.values) {
            assert!(
                (full.is_nan() && window.is_nan()) || (*full - *window).abs() < 1e-6,
                "full and window CF decode diverged: {full:?} versus {window:?}"
            );
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn decodes_cf_packed_coordinates_and_fails_closed_on_unsupported_metadata() {
        let path = std::env::temp_dir().join(format!(
            "nimbus_cm1_packed_coordinate_{}.nc",
            std::process::id()
        ));
        std::fs::write(&path, packed_coordinate_fixture()).unwrap();
        let values = read_classic_values(path.to_str().unwrap(), "xh").unwrap();
        assert_eq!(&values[..3], &[95.0, 100.0, 105.0]);
        assert!(values[3].is_nan());
        let _ = std::fs::remove_file(path);

        let unsupported = build_netcdf_cf_decode(
            "short",
            &[NetcdfNumericAttribute {
                name: "_FillValue".into(),
                data_type: "float".into(),
                values: vec![-32_767.0],
            }],
            None,
        );
        assert_eq!(unsupported.status, "unsupported");
        assert!(unsupported
            .issue
            .as_deref()
            .unwrap()
            .contains("stored values"));
        assert!(decode_netcdf_cf_value(&unsupported, 1.0).is_err());

        let unsigned = build_netcdf_cf_decode("short", &[], Some("true"));
        assert_eq!(unsigned.status, "unsupported");
        assert!(unsigned.issue.as_deref().unwrap().contains("_Unsigned"));

        let nan_missing = build_netcdf_cf_decode(
            "float",
            &[
                NetcdfNumericAttribute {
                    name: "_FillValue".into(),
                    data_type: "float".into(),
                    values: vec![f64::NAN],
                },
                NetcdfNumericAttribute {
                    name: "missing_value".into(),
                    data_type: "float".into(),
                    values: vec![f64::NAN],
                },
            ],
            None,
        );
        assert_eq!(nan_missing.status, "applied");
        assert!(decode_netcdf_cf_value(&nan_missing, f64::NAN)
            .unwrap()
            .is_nan());
        assert!(serde_json::to_string(&nan_missing).is_ok());
    }

    #[test]
    fn inventories_nested_history_with_bounded_deterministic_discovery() {
        let directory =
            std::env::temp_dir().join(format!("nimbus_cm1_case_inventory_{}", std::process::id()));
        std::fs::create_dir_all(directory.join("nested/level2/level3/level4")).unwrap();
        std::fs::create_dir_all(directory.join(".git/objects")).unwrap();
        for name in [
            "cm1out_000002.nc",
            "cm1out_000001.NC4",
            "cm1rst_000004_000000.nc",
            "cm1out_0000_0002_rst.dat",
            "namelist.input",
            "input_sounding",
            "cm1out_stats.dat",
            "notes.txt",
        ] {
            std::fs::write(directory.join(name), b"fixture").unwrap();
        }
        std::fs::write(directory.join("nested/cm1out_000003.nc"), b"fixture").unwrap();
        std::fs::write(directory.join(".git/objects/not_history.nc"), b"fixture").unwrap();
        std::fs::write(
            directory.join("nested/level2/level3/level4/too_deep.nc"),
            b"fixture",
        )
        .unwrap();
        let inventory = inventory_cm1_case(directory.to_str().unwrap()).unwrap();
        assert_eq!(inventory.schema_version, 2);
        assert_eq!(inventory.history_files.len(), 3);
        assert!(inventory.history_files[0].ends_with("cm1out_000001.NC4"));
        assert!(inventory.history_files[1].ends_with("cm1out_000002.nc"));
        assert!(inventory.history_files[2].ends_with("nested/cm1out_000003.nc"));
        assert!(inventory
            .namelist_input
            .unwrap()
            .ends_with("namelist.input"));
        assert!(inventory
            .input_sounding
            .unwrap()
            .ends_with("input_sounding"));
        assert_eq!(inventory.restart_files.len(), 2);
        assert_eq!(inventory.statistics_files.len(), 1);
        assert_eq!(inventory.other_file_count, 1);
        assert_eq!(inventory.nested_directory_count, 5);
        assert_eq!(inventory.scanned_directory_count, 4);
        assert_eq!(inventory.skipped_directory_count, 1);
        assert_eq!(inventory.excluded_directory_count, 1);
        assert_eq!(inventory.unreadable_directory_count, 0);
        assert_eq!(inventory.unreadable_entry_count, 0);
        assert_eq!(inventory.discovery_max_depth, 3);
        assert!(!inventory.discovery_truncated);
        assert!(inventory
            .warnings
            .iter()
            .any(|warning| warning.contains("not inspected")));
        assert!(inventory
            .warnings
            .iter()
            .any(|warning| warning.contains("source or dependency directory was excluded")));
        let _ = std::fs::remove_dir_all(directory);
    }

    #[cfg(unix)]
    #[test]
    fn cm1_case_discovery_never_follows_symbolic_links() {
        use std::os::unix::fs::symlink;

        let base =
            std::env::temp_dir().join(format!("nimbus_cm1_case_symlinks_{}", std::process::id()));
        let directory = base.join("case");
        let outside = base.join("outside");
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("not_part_of_case.nc"), b"fixture").unwrap();
        symlink(&outside, directory.join("linked_outputs")).unwrap();

        let inventory = inventory_cm1_case(directory.to_str().unwrap()).unwrap();
        assert!(inventory.history_files.is_empty());
        assert_eq!(inventory.symlink_count, 1);
        assert!(inventory
            .warnings
            .iter()
            .any(|warning| warning.contains("never follows links")));

        let linked_case = base.join("linked_case");
        symlink(&directory, &linked_case).unwrap();
        let error = inventory_cm1_case(linked_case.to_str().unwrap()).unwrap_err();
        assert!(error.to_string().contains("case path is a symbolic link"));
        let _ = std::fs::remove_dir_all(base);
    }

    #[cfg(unix)]
    #[test]
    fn concurrent_native_netcdf_reads_preserve_values_and_release_failed_reads() {
        use std::sync::Barrier;
        use crate::commands::netcdf_dynamic;

        const WORKERS: usize = 6;
        let directory = std::env::temp_dir().join(format!("nimbus_native_netcdf_concurrent_{}_{}",
            std::process::id(), std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir(&directory).unwrap();
        let cdl = directory.join("source.cdl");
        std::fs::write(&cdl, r#"netcdf cm1 {
 dimensions: time = 2; nk = 2; nj = 2; ni = 2;
 variables:
  float time(time); time:units = "seconds";
  float xh(ni); xh:units = "km";
  float yh(nj); yh:units = "km";
  float zh(nk); zh:units = "km";
  float winterp(time,nk,nj,ni); winterp:units = "m s-1";
  double physical_z_center_m(nk,nj,ni); physical_z_center_m:units = "m";
 data:
  time = 0,60; xh = -0.5,0.5; yh = -1,1; zh = 0.25,0.75;
  winterp = 0,1,2,3,4,5,6,7,100,101,102,103,104,105,106,107;
  physical_z_center_m = 250,260,270,280,750,760,770,780;
}"#).unwrap();
        let paths: Vec<_> = (0..WORKERS).map(|worker| {
            let path = directory.join(format!("worker_{worker}.nc4"));
            let output = std::process::Command::new("ncgen").args(["-4", "-o"])
                .arg(&path).arg(&cdl).output().expect("native netCDF concurrency fixture needs ncgen");
            assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
            path
        }).collect();
        let start = Barrier::new(WORKERS);
        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..WORKERS).map(|worker| {
                let paths = &paths;
                let start = &start;
                scope.spawn(move || {
                    start.wait();
                    for round in 0..12 {
                        let path = paths[if round % 2 == 0 { worker } else { 0 }].to_str().unwrap();
                        match (worker + round) % WORKERS {
                            0 => {
                                let metadata = netcdf_dynamic::inspect(path).unwrap();
                                assert_eq!(metadata.container_format, "netcdf4");
                                assert_eq!(metadata.record_count, 2);
                            }
                            1 => {
                                let (shape, values) = netcdf_dynamic::read_field(path, "winterp", 1).unwrap();
                                assert_eq!(shape, [2, 2, 2]);
                                assert_eq!(values, [100.0, 104.0, 102.0, 106.0, 101.0, 105.0, 103.0, 107.0]);
                            }
                            2 => {
                                let request = window_request(&[0, 0, 0], &[2, 2, 2], &[1, 2, 1]);
                                let payload = netcdf_dynamic::read_field_window(
                                    path, "winterp", 1, &request, &NetcdfReadControl::new()).unwrap();
                                assert_eq!(payload.output_shape, [2, 1, 2]);
                                assert_eq!(payload.values, [101.0, 105.0, 102.0, 106.0]);
                            }
                            3 => assert_eq!(netcdf_dynamic::read_values(path, "time").unwrap(), [0.0, 60.0]),
                            4 => {
                                let (shape, values, bytes) = netcdf_dynamic::read_f64_window(
                                    path, "physical_z_center_m", [0; 3], [2; 3]).unwrap();
                                assert_eq!(shape, [2, 2, 2]);
                                assert_eq!(values, [250.0, 750.0, 270.0, 770.0, 260.0, 760.0, 280.0, 780.0]);
                                assert_eq!(bytes, 64);
                            }
                            5 => {
                                let (shape, values, bytes) = netcdf_dynamic::read_f64_sampled_window(
                                    path, "physical_z_center_m", [1, 0, 0], [1, 2, 2], [1, 2, 1]).unwrap();
                                assert_eq!(shape, [2, 2, 2]);
                                assert_eq!(values, [260.0, 760.0]);
                                assert_eq!(bytes, 16);
                            }
                            _ => unreachable!(),
                        }
                    }
                    let path = paths[worker].to_str().unwrap();
                    assert!(matches!(netcdf_dynamic::inspect(&format!("{path}.missing")), Err(AppError::Invalid(_))));
                    assert!(matches!(netcdf_dynamic::read_field(path, "missing_variable", 1), Err(AppError::NotFound(_))));
                    let cancelled = NetcdfReadControl::new();
                    cancelled.cancelled.store(true, Ordering::Release);
                    let request = window_request(&[0; 3], &[2; 3], &[1; 3]);
                    assert!(matches!(netcdf_dynamic::read_field_window(
                        path, "winterp", 1, &request, &cancelled), Err(AppError::Task(_))));
                    assert_eq!(netcdf_dynamic::read_field(path, "winterp", 1).unwrap().1,
                        [100.0, 104.0, 102.0, 106.0, 101.0, 105.0, 103.0, 107.0]);
                })
            }).collect();
            for worker in workers {
                worker.join().unwrap();
            }
        });
        for path in paths {
            std::fs::remove_file(path).unwrap();
        }
        std::fs::remove_file(cdl).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn physically_reads_generated_netcdf4_when_toolchain_is_available() {
        if std::process::Command::new("ncgen")
            .arg("-h")
            .output()
            .is_err()
        {
            return;
        }
        let base = std::env::temp_dir().join(format!("nimbus_cm1_netcdf4_{}", std::process::id()));
        let cdl = base.with_extension("cdl");
        let nc = base.with_extension("nc4");
        std::fs::write(&cdl, "netcdf cm1 { dimensions: time = 2; nk = 2; nj = 2; ni = 2; variables: float time(time); time:units=\"seconds\"; float xh(ni); xh:units=\"km\"; float yh(nj); yh:units=\"km\"; float zh(nk); zh:units=\"km\"; float winterp(time,nk,nj,ni); winterp:units=\"m s-1\"; short uinterp(time,nk,nj,ni); uinterp:units=\"m s-1\"; uinterp:scale_factor=0.5f; uinterp:add_offset=10.f; uinterp:_FillValue=-32767s; uinterp:missing_value=-9998s; uinterp:valid_range=-100s,100s; float cref(time,nj,ni); cref:units=\"dBZ\"; data: time=0,60; xh=-0.5,0.5; yh=-1,1; zh=0.25,0.75; winterp=0,1,2,3,4,5,6,7,100,101,102,103,104,105,106,107; uinterp=0,1,2,3,4,5,6,7,10,20,-32767,-9998,101,-101,50,-50; cref=0,1,2,3,100,101,102,103; }").unwrap();
        let status = std::process::Command::new("ncgen")
            .args(["-4", "-o"])
            .arg(&nc)
            .arg(&cdl)
            .status()
            .unwrap();
        assert!(status.success());
        let metadata = crate::commands::netcdf_dynamic::inspect(nc.to_str().unwrap()).unwrap();
        assert_eq!(metadata.container_format, "netcdf4");
        assert_eq!(metadata.record_count, 2);
        assert_eq!(metadata.source_classification, "cm1_likely");
        assert!(metadata.plotting_supported);
        let (shape, values) =
            crate::commands::netcdf_dynamic::read_field(nc.to_str().unwrap(), "winterp", 1)
                .unwrap();
        assert_eq!(shape, vec![2, 2, 2]);
        assert_eq!(
            values,
            vec![100.0, 104.0, 102.0, 106.0, 101.0, 105.0, 103.0, 107.0]
        );
        let (horizontal_shape, horizontal_values) =
            crate::commands::netcdf_dynamic::read_field(nc.to_str().unwrap(), "cref", 1).unwrap();
        assert_eq!(horizontal_shape, vec![2, 2]);
        assert_eq!(horizontal_values, vec![100.0, 102.0, 101.0, 103.0]);
        let window = window_request(&[0, 0, 0], &[2, 2, 2], &[1, 2, 1]);
        let control = NetcdfReadControl::new();
        let payload = crate::commands::netcdf_dynamic::read_field_window(
            nc.to_str().unwrap(),
            "winterp",
            1,
            &window,
            &control,
        )
        .unwrap();
        assert_eq!(payload.output_shape, vec![2, 1, 2]);
        assert_eq!(payload.values, vec![101.0, 105.0, 102.0, 106.0]);
        assert_eq!(payload.source_read_bytes, 32);
        assert_eq!(payload.cancellation_mode, "boundary_only");
        assert_eq!(payload.non_interruptible_phase, Some("nc_get_vars_float"));
        let packed_metadata = metadata
            .variables
            .iter()
            .find(|variable| variable.name == "uinterp")
            .unwrap();
        assert_eq!(packed_metadata.cf_decode.status, "applied");
        assert_eq!(packed_metadata.cf_decode.fill_value, Some(-32_767.0));
        let (packed_shape, packed) =
            crate::commands::netcdf_dynamic::read_field(nc.to_str().unwrap(), "uinterp", 1)
                .unwrap();
        assert_eq!(packed_shape, vec![2, 2, 2]);
        assert_eq!(packed[0], 15.0);
        assert!(packed[1].is_nan());
        assert!(packed[2].is_nan());
        assert_eq!(packed[3], 35.0);
        assert_eq!(packed[4], 20.0);
        assert!(packed[5].is_nan());
        assert!(packed[6].is_nan());
        assert_eq!(packed[7], -15.0);
        let packed_payload = crate::commands::netcdf_dynamic::read_field_window(
            nc.to_str().unwrap(),
            "uinterp",
            1,
            &window_request(&[0, 0, 0], &[2, 2, 2], &[1, 1, 1]),
            &NetcdfReadControl::new(),
        )
        .unwrap();
        for (full, window) in packed.iter().zip(&packed_payload.values) {
            assert!((full.is_nan() && window.is_nan()) || (*full - *window).abs() < 1e-6);
        }
        assert_eq!(
            crate::commands::netcdf_dynamic::read_values(nc.to_str().unwrap(), "time").unwrap(),
            vec![0.0, 60.0]
        );
        assert_eq!(
            crate::commands::netcdf_dynamic::read_values(nc.to_str().unwrap(), "xh").unwrap(),
            vec![-0.5, 0.5]
        );
        let _ = std::fs::remove_file(cdl);
        let _ = std::fs::remove_file(nc);
    }
}
