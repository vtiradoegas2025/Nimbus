use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use libloading::Library;
use once_cell::sync::OnceCell;

use super::datasets::{
    build_netcdf_cf_decode, decode_netcdf_cf_value, is_qualified_cm1_history_variable,
    prepare_netcdf_window, NetcdfDimension, NetcdfMetadata, NetcdfNumericAttribute,
    NetcdfReadControl, NetcdfReadPhase, NetcdfVariable, NetcdfWindowPayload, NetcdfWindowRequest,
    MAX_NETCDF_SOURCE_BYTES,
};
use crate::error::{AppError, AppResult};

const NC_NOWRITE: c_int = 0;
const NC_CHAR: c_int = 2;
const NC_MAX_NAME: usize = 256;
/// libnetcdf's own per-variable dimension ceiling. Bounds the allocation we
/// size from `nc_inq_var`'s reported rank.
const NC_MAX_VAR_DIMS: c_int = 1024;

// The C library's process-global state includes initialization and unloading.
static NATIVE_NETCDF_ACCESS: Mutex<()> = Mutex::new(());
// HDF5 retains process/thread cleanup callbacks after a file closes. Unloading
// its code between reads can leave those callbacks pointing at unmapped memory.
static NATIVE_NETCDF_API: OnceCell<Api> = OnceCell::new();

fn native_api() -> AppResult<&'static Api> {
    NATIVE_NETCDF_API.get_or_try_init(Api::load)
}

fn lock_native_netcdf(access: &Mutex<()>) -> AppResult<MutexGuard<'_, ()>> {
    access.lock().map_err(|_| AppError::Invalid(
        "native netCDF access is unavailable after a reader panic".into()))
}

type NcOpen = unsafe extern "C" fn(*const c_char, c_int, *mut c_int) -> c_int;
type NcClose = unsafe extern "C" fn(c_int) -> c_int;
type NcInq = unsafe extern "C" fn(c_int, *mut c_int, *mut c_int, *mut c_int, *mut c_int) -> c_int;
type NcInqDim = unsafe extern "C" fn(c_int, c_int, *mut c_char, *mut usize) -> c_int;
type NcInqVar = unsafe extern "C" fn(
    c_int,
    c_int,
    *mut c_char,
    *mut c_int,
    *mut c_int,
    *mut c_int,
    *mut c_int,
) -> c_int;
type NcInqAtt = unsafe extern "C" fn(c_int, c_int, *const c_char, *mut c_int, *mut usize) -> c_int;
type NcGetAttText = unsafe extern "C" fn(c_int, c_int, *const c_char, *mut c_char) -> c_int;
type NcGetAttDouble = unsafe extern "C" fn(c_int, c_int, *const c_char, *mut f64) -> c_int;
type NcInqFormat = unsafe extern "C" fn(c_int, *mut c_int) -> c_int;
type NcGetVaraFloat =
    unsafe extern "C" fn(c_int, c_int, *const usize, *const usize, *mut f32) -> c_int;
type NcGetVarsFloat =
    unsafe extern "C" fn(c_int, c_int, *const usize, *const usize, *const isize, *mut f32) -> c_int;
type NcGetVarsDouble =
    unsafe extern "C" fn(c_int, c_int, *const usize, *const usize, *const isize, *mut f64) -> c_int;
type NcGetVarDouble = unsafe extern "C" fn(c_int, c_int, *mut f64) -> c_int;
type NcStrError = unsafe extern "C" fn(c_int) -> *const c_char;

struct Api {
    /// Owns the loaded library for the same lifetime as its function pointers.
    #[allow(dead_code)]
    handle: Library,
    open: NcOpen,
    close: NcClose,
    inq: NcInq,
    inq_dim: NcInqDim,
    inq_var: NcInqVar,
    inq_att: NcInqAtt,
    get_att_text: NcGetAttText,
    get_att_double: NcGetAttDouble,
    inq_format: NcInqFormat,
    get_vara_float: NcGetVaraFloat,
    get_vars_float: NcGetVarsFloat,
    get_vars_double: NcGetVarsDouble,
    get_var_double: NcGetVarDouble,
    strerror: NcStrError,
}

impl Api {
    fn load() -> AppResult<Self> {
        let mut candidates = Vec::<PathBuf>::new();
        #[cfg(windows)]
        if let Ok(executable) = std::env::current_exe() {
            if let Some(directory) = executable.parent() {
                candidates.push(directory.join("bin").join("netcdf.dll"));
                candidates.push(directory.join("bin").join("libnetcdf.dll"));
            }
        }
        #[cfg(target_os = "macos")]
        if let Ok(executable) = std::env::current_exe() {
            if let Some(contents) = executable.parent().and_then(std::path::Path::parent) {
                if contents.file_name().is_some_and(|name| name == "Contents")
                    && executable.parent().and_then(std::path::Path::file_name)
                        .is_some_and(|name| name == "MacOS")
                {
                    candidates.clear();
                    candidates.push(contents.join("Frameworks").join("libnetcdf.dylib"));
                }
            }
        }
        if candidates.is_empty() || !cfg!(target_os = "macos") {
            candidates.extend([
                PathBuf::from("netcdf.dll"),
                PathBuf::from("libnetcdf.dll"),
                PathBuf::from("libnetcdf.dylib"),
                PathBuf::from("libnetcdf.so"),
                PathBuf::from("libnetcdf.so.19"),
                PathBuf::from("/opt/homebrew/lib/libnetcdf.dylib"),
                PathBuf::from("/usr/local/lib/libnetcdf.dylib"),
                PathBuf::from("/opt/homebrew/opt/netcdf/lib/libnetcdf.dylib"),
                PathBuf::from("/usr/local/opt/netcdf/lib/libnetcdf.dylib"),
            ]);
        }
        let mut errors = Vec::new();
        let mut loaded = None;
        for candidate in candidates {
            match unsafe { Library::new(&candidate) } {
                Ok(library) => {
                    loaded = Some(library);
                    break;
                }
                Err(error) => errors.push(format!("{}: {error}", candidate.display())),
            }
        }
        let handle = loaded.ok_or_else(|| {
            AppError::Invalid(format!(
                "netCDF4/HDF5 requires a packaged libnetcdf runtime; no compatible library was found ({})",
                errors.join("; ")
            ))
        })?;
        unsafe fn symbol<T: Copy>(handle: &Library, name: &str) -> AppResult<T> {
            unsafe { handle.get::<T>(name.as_bytes()) }
                .map(|symbol| *symbol)
                .map_err(|_| {
                    AppError::Invalid(format!("libnetcdf is missing required symbol {name}"))
                })
        }
        Ok(unsafe {
            Self {
                open: symbol(&handle, "nc_open")?,
                close: symbol(&handle, "nc_close")?,
                inq: symbol(&handle, "nc_inq")?,
                inq_dim: symbol(&handle, "nc_inq_dim")?,
                inq_var: symbol(&handle, "nc_inq_var")?,
                inq_att: symbol(&handle, "nc_inq_att")?,
                get_att_text: symbol(&handle, "nc_get_att_text")?,
                get_att_double: symbol(&handle, "nc_get_att_double")?,
                inq_format: symbol(&handle, "nc_inq_format")?,
                get_vara_float: symbol(&handle, "nc_get_vara_float")?,
                get_vars_float: symbol(&handle, "nc_get_vars_float")?,
                get_vars_double: symbol(&handle, "nc_get_vars_double")?,
                get_var_double: symbol(&handle, "nc_get_var_double")?,
                strerror: symbol(&handle, "nc_strerror")?,
                // Last so the symbol lookups above still own it on failure.
                handle,
            }
        })
    }

    fn check(&self, status: c_int, context: &str) -> AppResult<()> {
        if status == 0 {
            return Ok(());
        }
        let ptr = unsafe { (self.strerror)(status) };
        let detail = if ptr.is_null() {
            format!("status {status}")
        } else {
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
        };
        Err(AppError::Invalid(format!("{context}: {detail}")))
    }
}

struct FileHandle<'a> {
    api: &'a Api,
    id: c_int,
}
impl Drop for FileHandle<'_> {
    fn drop(&mut self) {
        unsafe {
            (self.api.close)(self.id);
        }
    }
}

fn open<'a>(api: &'a Api, path: &str) -> AppResult<FileHandle<'a>> {
    let path =
        CString::new(path).map_err(|_| AppError::Invalid("netCDF path contains NUL".into()))?;
    let mut id = -1;
    api.check(
        unsafe { (api.open)(path.as_ptr(), NC_NOWRITE, &mut id) },
        "nc_open",
    )?;
    Ok(FileHandle { api, id })
}

fn name(buffer: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}
fn type_name(value: c_int) -> String {
    match value {
        1 => "byte",
        2 => "char",
        3 => "short",
        4 => "int",
        5 => "float",
        6 => "double",
        7 => "ubyte",
        8 => "ushort",
        9 => "uint",
        10 => "int64",
        11 => "uint64",
        12 => "string",
        _ => "unknown",
    }
    .into()
}

fn text_attr(api: &Api, id: c_int, var: c_int, key: &str) -> Option<String> {
    let key = CString::new(key).ok()?;
    let mut kind = 0;
    let mut length = 0;
    if unsafe { (api.inq_att)(id, var, key.as_ptr(), &mut kind, &mut length) } != 0
        || kind != NC_CHAR
        || length > 1_048_576
    {
        return None;
    }
    let mut bytes = vec![0_u8; length + 1];
    if unsafe { (api.get_att_text)(id, var, key.as_ptr(), bytes.as_mut_ptr().cast()) } != 0 {
        return None;
    }
    Some(String::from_utf8_lossy(&bytes[..length]).into_owned())
}

fn numeric_attr(api: &Api, id: c_int, var: c_int, key: &str) -> Option<NetcdfNumericAttribute> {
    let native_key = CString::new(key).ok()?;
    let mut kind = 0;
    let mut length = 0;
    if unsafe { (api.inq_att)(id, var, native_key.as_ptr(), &mut kind, &mut length) } != 0 {
        return None;
    }
    if length > 32 {
        return Some(NetcdfNumericAttribute {
            name: key.into(),
            data_type: type_name(kind),
            values: Vec::new(),
        });
    }
    let mut values = if matches!(kind, NC_CHAR | 12) {
        Vec::new()
    } else {
        vec![0_f64; length]
    };
    if !values.is_empty()
        && unsafe { (api.get_att_double)(id, var, native_key.as_ptr(), values.as_mut_ptr()) } != 0
    {
        values.clear();
    }
    Some(NetcdfNumericAttribute {
        name: key.into(),
        data_type: type_name(kind),
        values,
    })
}

struct Inventory {
    metadata: NetcdfMetadata,
    var_ids: Vec<c_int>,
    /// Either a valid index into `metadata.dimensions` or -1. `inventory`
    /// bounds-checks it once so downstream reads can index directly.
    time_dim: c_int,
}

/// netCDF-4 dimension ids are group-scoped, so `nc_inq` can under-report what a
/// variable references. Resolve through a bounds check rather than indexing, so
/// a group-scoped or malformed file is an error instead of a panic.
fn resolve_variable_dimensions(
    dimensions: &[NetcdfDimension],
    dim_ids: &[c_int],
    variable: &str,
) -> AppResult<(Vec<String>, Vec<u64>)> {
    let mut names = Vec::with_capacity(dim_ids.len());
    let mut shape = Vec::with_capacity(dim_ids.len());
    for id in dim_ids {
        let dimension = usize::try_from(*id)
            .ok()
            .and_then(|index| dimensions.get(index))
            .ok_or_else(|| {
                AppError::Invalid(format!(
                    "variable '{variable}' references missing dimension {id}"
                ))
            })?;
        names.push(dimension.name.clone());
        shape.push(dimension.length);
    }
    Ok((names, shape))
}

fn inventory(api: &Api, file: &FileHandle<'_>, path: &str) -> AppResult<Inventory> {
    let mut ndims = 0;
    let mut nvars = 0;
    let mut ngatts = 0;
    let mut unlimited = -1;
    api.check(
        unsafe { (api.inq)(file.id, &mut ndims, &mut nvars, &mut ngatts, &mut unlimited) },
        "nc_inq",
    )?;
    let mut dimensions = Vec::new();
    for dim_id in 0..ndims {
        let mut buf = [0 as c_char; NC_MAX_NAME + 1];
        let mut length = 0;
        api.check(
            unsafe { (api.inq_dim)(file.id, dim_id, buf.as_mut_ptr(), &mut length) },
            "nc_inq_dim",
        )?;
        dimensions.push(NetcdfDimension {
            name: name(&buf),
            length: length as u64,
            unlimited: dim_id == unlimited,
        });
    }
    let mut variables = Vec::new();
    let mut ids = Vec::new();
    for var_id in 0..nvars {
        let mut buf = [0 as c_char; NC_MAX_NAME + 1];
        let mut kind = 0;
        let mut count = 0;
        let mut natts = 0;
        api.check(
            unsafe {
                (api.inq_var)(
                    file.id,
                    var_id,
                    buf.as_mut_ptr(),
                    &mut kind,
                    &mut count,
                    std::ptr::null_mut(),
                    &mut natts,
                )
            },
            "nc_inq_var",
        )?;
        if !(0..=NC_MAX_VAR_DIMS).contains(&count) {
            return Err(AppError::Invalid(format!(
                "variable {var_id} in {path} reports {count} dimensions"
            )));
        }
        let mut dim_ids = vec![0; count as usize];
        api.check(
            unsafe {
                (api.inq_var)(
                    file.id,
                    var_id,
                    buf.as_mut_ptr(),
                    &mut kind,
                    &mut count,
                    dim_ids.as_mut_ptr(),
                    &mut natts,
                )
            },
            "nc_inq_var dimensions",
        )?;
        let numeric_attributes = [
            "scale_factor",
            "add_offset",
            "_FillValue",
            "missing_value",
            "valid_range",
            "valid_min",
            "valid_max",
        ]
        .iter()
        .filter_map(|key| numeric_attr(api, file.id, var_id, key))
        .collect::<Vec<_>>();
        let variable_name = name(&buf);
        let (dimension_names, shape) =
            resolve_variable_dimensions(&dimensions, &dim_ids, &variable_name)?;
        variables.push(NetcdfVariable {
            name: variable_name,
            data_type: type_name(kind),
            dimensions: dimension_names,
            shape,
            units: text_attr(api, file.id, var_id, "units"),
            long_name: text_attr(api, file.id, var_id, "long_name"),
            cf_decode: build_netcdf_cf_decode(
                &type_name(kind),
                &numeric_attributes,
                text_attr(api, file.id, var_id, "_Unsigned").as_deref(),
            ),
        });
        ids.push(var_id);
    }
    let mut format = 0;
    api.check(
        unsafe { (api.inq_format)(file.id, &mut format) },
        "nc_inq_format",
    )?;
    // `unlimited` is reported by nc_inq and otherwise unvalidated; an
    // out-of-range id degrades to "no time dimension" rather than indexing
    // past `dimensions`. This is the invariant Inventory::time_dim documents.
    let time_dim = dimensions
        .iter()
        .position(|dimension| dimension.name.eq_ignore_ascii_case("time"))
        .map(|index| index as c_int)
        .unwrap_or(unlimited);
    let time_dim = match usize::try_from(time_dim) {
        Ok(index) if index < dimensions.len() => time_dim,
        _ => -1,
    };
    let names: std::collections::BTreeSet<&str> =
        variables.iter().map(|v| v.name.as_str()).collect();
    let marker_count = ["xh", "xf", "yh", "yf", "zh", "zf"]
        .iter()
        .filter(|value| names.contains(**value))
        .count();
    let cm1_index_dimensions = ["ni", "nj", "nk"].iter().all(|wanted| {
        dimensions
            .iter()
            .any(|dimension| dimension.name.eq_ignore_ascii_case(wanted))
    });
    let safe = variables.iter().any(|variable| {
        !matches!(variable.data_type.as_str(), "char" | "string")
            && is_qualified_cm1_history_variable(variable)
    });
    Ok(Inventory {
        metadata: NetcdfMetadata {
            schema: "nimbus.studio.netcdf-metadata",
            schema_version: 1,
            path: path.into(),
            container_format: match format {
                1 => "netcdf_classic",
                2 => "netcdf_64bit_offset",
                3 => "netcdf4",
                4 => "netcdf4_classic",
                5 => "netcdf_64bit_data",
                _ => "netcdf_unknown",
            }
            .into(),
            record_count: if time_dim >= 0 {
                dimensions[time_dim as usize].length
            } else {
                1
            },
            dimensions,
            variables,
            source_classification: if marker_count >= 4 || cm1_index_dimensions {
                "cm1_likely"
            } else {
                "unclassified_netcdf"
            }
            .into(),
            plotting_supported: safe,
            plot_unavailable_reason: if safe {
                "Qualified CM1 scalar volumes and horizontal diagnostics are routed to plots only when names, units, rank, and staggering match."
            } else {
                "No safely mapped 3-D CM1 plot field was found."
            }
            .into(),
        },
        var_ids: ids,
        time_dim,
    })
}

pub fn inspect(path: &str) -> AppResult<NetcdfMetadata> {
    let _native_access = lock_native_netcdf(&NATIVE_NETCDF_ACCESS)?;
    let api = native_api()?;
    let file = open(&api, path)?;
    Ok(inventory(&api, &file, path)?.metadata)
}

fn axis(name: &str) -> Option<usize> {
    let n = name.to_ascii_lowercase();
    if n.starts_with('x') || n == "ni" || n == "nip1" {
        Some(0)
    } else if n.starts_with('y') || n == "nj" || n == "njp1" {
        Some(1)
    } else if n.starts_with('z') || n == "nk" || n == "nkp1" {
        Some(2)
    } else {
        None
    }
}

pub fn read_field(path: &str, variable: &str, record: u64) -> AppResult<(Vec<usize>, Vec<f32>)> {
    let _native_access = lock_native_netcdf(&NATIVE_NETCDF_ACCESS)?;
    let api = native_api()?;
    let file = open(&api, path)?;
    let inv = inventory(&api, &file, path)?;
    let index = inv
        .metadata
        .variables
        .iter()
        .position(|v| v.name == variable)
        .ok_or_else(|| AppError::NotFound(format!("variable '{variable}' in {path}")))?;
    let var = &inv.metadata.variables[index];
    let mut start = vec![0_usize; var.shape.len()];
    let mut count: Vec<usize> = var.shape.iter().map(|v| *v as usize).collect();
    let time_var = inv.time_dim >= 0
        && var.dimensions.first() == Some(&inv.metadata.dimensions[inv.time_dim as usize].name);
    if time_var {
        if record >= inv.metadata.record_count {
            return Err(AppError::Invalid("record outside dataset".into()));
        }
        start[0] = record as usize;
        count[0] = 1;
    } else if record != 0 {
        return Err(AppError::Invalid(
            "a variable without a time dimension only has record 0".into(),
        ));
    }
    let elements = count.iter().try_fold(1_usize, |a, b| {
        a.checked_mul(*b)
            .ok_or_else(|| AppError::Invalid("field size overflow".into()))
    })?;
    // This path reads a whole variable and then allocates an equal-sized
    // transpose target, so a multi-GB CM1 field would cost 2x its size in RSS
    // and OOM the app. The windowed reader exists for exactly that case.
    let decoded_bytes = (elements as u64).saturating_mul(std::mem::size_of::<f32>() as u64);
    if decoded_bytes > MAX_NETCDF_SOURCE_BYTES {
        return Err(AppError::Invalid(format!(
            "variable '{variable}' needs {decoded_bytes} bytes, over the {MAX_NETCDF_SOURCE_BYTES}-byte whole-field limit; use a windowed read"
        )));
    }
    let mut values = vec![0_f32; elements];
    api.check(
        unsafe {
            (api.get_vara_float)(
                file.id,
                inv.var_ids[index],
                start.as_ptr(),
                count.as_ptr(),
                values.as_mut_ptr(),
            )
        },
        "nc_get_vara_float",
    )?;
    for value in &mut values {
        *value = decode_netcdf_cf_value(&var.cf_decode, f64::from(*value))? as f32;
    }
    let skip = usize::from(time_var);
    let shape = count[skip..].to_vec();
    let dims = &var.dimensions[skip..];
    if !matches!(shape.len(), 2 | 3) {
        return Err(AppError::Invalid(
            "selected netCDF variable is not 2-D or 3-D after time selection".into(),
        ));
    }
    let rank = shape.len();
    let axes: Vec<usize> = dims
        .iter()
        .map(|n| {
            axis(n).ok_or_else(|| AppError::Invalid(format!("cannot map dimension '{n}' to x/y/z")))
        })
        .collect::<AppResult<_>>()?;
    let mut target_shape = vec![0; rank];
    for i in 0..rank {
        if axes[i] >= rank {
            return Err(AppError::Invalid(
                "two-dimensional fields may contain only x and y dimensions".into(),
            ));
        }
        if target_shape[axes[i]] != 0 {
            return Err(AppError::Invalid("duplicate canonical dimensions".into()));
        }
        target_shape[axes[i]] = shape[i];
    }
    if target_shape.contains(&0) {
        return Err(AppError::Invalid(
            "field does not contain each required canonical dimension".into(),
        ));
    }
    let mut target = vec![0_f32; values.len()];
    for (target_index, target_value) in target.iter_mut().enumerate() {
        let mut remainder = target_index;
        let mut canonical_indices = vec![0_usize; rank];
        for axis_index in (0..rank).rev() {
            canonical_indices[axis_index] = remainder % target_shape[axis_index];
            remainder /= target_shape[axis_index];
        }
        let source_indices: Vec<usize> = axes
            .iter()
            .map(|axis_index| canonical_indices[*axis_index])
            .collect();
        let source_index = source_indices
            .iter()
            .enumerate()
            .fold(0_usize, |linear, (axis_index, source)| {
                linear * shape[axis_index] + source
            });
        *target_value = values[source_index];
    }
    Ok((target_shape, target))
}

pub fn read_field_window(
    path: &str,
    variable: &str,
    record: u64,
    request: &NetcdfWindowRequest,
    control: &NetcdfReadControl,
) -> AppResult<NetcdfWindowPayload> {
    let _native_access = lock_native_netcdf(&NATIVE_NETCDF_ACCESS)?;
    let api = native_api()?;
    let file = open(&api, path)?;
    let inv = inventory(&api, &file, path)?;
    let index = inv
        .metadata
        .variables
        .iter()
        .position(|value| value.name == variable)
        .ok_or_else(|| AppError::NotFound(format!("variable '{variable}' in {path}")))?;
    let variable_metadata = inv.metadata.variables[index].clone();
    let source_format = inv.metadata.container_format.clone();
    let time_variable = inv.time_dim >= 0
        && variable_metadata.dimensions.first()
            == Some(&inv.metadata.dimensions[inv.time_dim as usize].name);
    let prepared = prepare_netcdf_window(
        &variable_metadata,
        inv.metadata.record_count,
        time_variable,
        record,
        request,
    )?;
    let source_read_bytes = u64::try_from(prepared.source_decoded_bytes)
        .map_err(|_| AppError::Invalid("netCDF source byte count overflow".into()))?;
    if source_read_bytes > prepared.source_byte_budget {
        return Err(AppError::Invalid(format!(
            "netCDF hyperslab needs {source_read_bytes} converted source bytes; request budget is {}",
            prepared.source_byte_budget
        )));
    }
    let native_stride = prepared
        .source_stride
        .iter()
        .map(|value| {
            isize::try_from(*value)
                .map_err(|_| AppError::Invalid("netCDF stride exceeds native ptrdiff_t".into()))
        })
        .collect::<AppResult<Vec<_>>>()?;
    let mut values = vec![0_f32; prepared.source_element_count];
    control.set_phase(NetcdfReadPhase::NativeCall);
    control.check(&request.request_id)?;
    // nc_get_vars_float is one bounded, blocking C call. libnetcdf/HDF5 does
    // not expose an abort callback here, so cancellation is observed on both
    // boundaries and the result is discarded if the flag changed in flight.
    api.check(
        unsafe {
            (api.get_vars_float)(
                file.id,
                inv.var_ids[index],
                prepared.source_start.as_ptr(),
                prepared.source_count.as_ptr(),
                native_stride.as_ptr(),
                values.as_mut_ptr(),
            )
        },
        "nc_get_vars_float",
    )?;
    control.check(&request.request_id)?;
    for value in &mut values {
        *value = decode_netcdf_cf_value(&variable_metadata.cf_decode, f64::from(*value))? as f32;
    }

    let skip = usize::from(time_variable);
    let source_output_shape = &prepared.source_count[skip..];
    let rank = source_output_shape.len();
    let mut canonical_source = vec![0_f32; prepared.source_element_count];
    let canonical_read_shape = if request.aggregation == "point_sample" {
        &prepared.output_shape
    } else {
        &request.extent
    };
    control.set_phase(NetcdfReadPhase::Transpose);
    let mut canonical_indices = vec![0_usize; rank];
    for (target_index, target_value) in canonical_source.iter_mut().enumerate() {
        if target_index % 16_384 == 0 {
            control.check(&request.request_id)?;
        }
        let mut remainder = target_index;
        for axis_index in (0..rank).rev() {
            canonical_indices[axis_index] = remainder % canonical_read_shape[axis_index];
            remainder /= canonical_read_shape[axis_index];
        }
        let source_index = prepared.source_axes.iter().enumerate().try_fold(
            0_usize,
            |linear, (source_axis, canonical_axis)| {
                linear
                    .checked_mul(source_output_shape[source_axis])
                    .and_then(|value| value.checked_add(canonical_indices[*canonical_axis]))
                    .ok_or_else(|| AppError::Invalid("netCDF transpose index overflow".into()))
            },
        )?;
        *target_value = values[source_index];
    }
    control.check(&request.request_id)?;
    let canonical = if request.aggregation == "point_sample" {
        canonical_source
    } else {
        super::datasets::aggregate_c_order(
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
        values: canonical,
        source_read_bytes,
        source_read_accounting: "converted_hyperslab_output",
        native_internal_allocation_bounded: false,
        source_format,
        cancellation_mode: "boundary_only",
        non_interruptible_phase: Some("nc_get_vars_float"),
    })
}

/// Read one bounded three-dimensional float64 auxiliary-coordinate window in
/// canonical x/y/z order.  Terrain history uses this for its exact center and
/// interface heights; converting those variables through float32 would alter
/// cell volumes and is therefore prohibited.
pub fn read_f64_window(
    path: &str,
    variable: &str,
    origin: [usize; 3],
    extent: [usize; 3],
) -> AppResult<(Vec<usize>, Vec<f64>, usize)> {
    read_f64_sampled_window(path, variable, origin, extent, [1; 3])
}

pub fn read_f64_sampled_window(
    path: &str,
    variable: &str,
    origin: [usize; 3],
    extent: [usize; 3],
    sampling_stride: [usize; 3],
) -> AppResult<(Vec<usize>, Vec<f64>, usize)> {
    if extent.contains(&0) || sampling_stride.contains(&0) {
        return Err(AppError::Invalid(
            "netCDF float64 window extent must be positive".into(),
        ));
    }
    let _native_access = lock_native_netcdf(&NATIVE_NETCDF_ACCESS)?;
    let api = native_api()?;
    let file = open(&api, path)?;
    let inv = inventory(&api, &file, path)?;
    let index = inv
        .metadata
        .variables
        .iter()
        .position(|value| value.name == variable)
        .ok_or_else(|| AppError::NotFound(format!("variable '{variable}' in {path}")))?;
    let metadata = &inv.metadata.variables[index];
    if metadata.dimensions.len() != 3 || metadata.shape.len() != 3 {
        return Err(AppError::Invalid(format!(
            "terrain coordinate variable '{variable}' must be three-dimensional"
        )));
    }
    let axes = metadata
        .dimensions
        .iter()
        .map(|name| {
            axis(name).ok_or_else(|| {
                AppError::Invalid(format!(
                    "cannot map terrain coordinate dimension '{name}' to x/y/z"
                ))
            })
        })
        .collect::<AppResult<Vec<_>>>()?;
    if axes
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        != std::collections::BTreeSet::from([0_usize, 1, 2])
    {
        return Err(AppError::Invalid(
            "terrain coordinate must contain each canonical x/y/z dimension once".into(),
        ));
    }
    let mut source_shape = vec![0_usize; 3];
    for source_axis in 0..3 {
        source_shape[axes[source_axis]] = usize::try_from(metadata.shape[source_axis])
            .map_err(|_| AppError::Invalid("terrain coordinate shape exceeds usize".into()))?;
    }
    for canonical_axis in 0..3 {
        let end = origin[canonical_axis]
            .checked_add(extent[canonical_axis])
            .ok_or_else(|| AppError::Invalid("terrain coordinate window bound overflow".into()))?;
        if end > source_shape[canonical_axis] {
            return Err(AppError::Invalid(format!(
                "terrain coordinate window exceeds axis {canonical_axis}"
            )));
        }
    }
    let output =
        std::array::from_fn::<_, 3, _>(|axis| (extent[axis] - 1) / sampling_stride[axis] + 1);
    let mut start = [0_usize; 3];
    let mut count = [0_usize; 3];
    for source_axis in 0..3 {
        start[source_axis] = origin[axes[source_axis]];
        count[source_axis] = output[axes[source_axis]];
    }
    let elements = output.into_iter().try_fold(1_usize, |product, value| {
        product
            .checked_mul(value)
            .ok_or_else(|| AppError::Invalid("terrain coordinate window shape overflow".into()))
    })?;
    let source_bytes = elements
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| AppError::Invalid("terrain coordinate byte size overflow".into()))?;
    if u64::try_from(source_bytes).map_or(true, |bytes| bytes > MAX_NETCDF_SOURCE_BYTES) {
        return Err(AppError::Invalid(format!(
            "terrain coordinate window needs {source_bytes} bytes, over the {MAX_NETCDF_SOURCE_BYTES}-byte limit"
        )));
    }
    let mut stride = [1_isize; 3];
    for axis in 0..3 {
        stride[axis] = isize::try_from(sampling_stride[axes[axis]])
            .map_err(|_| AppError::Invalid("terrain coordinate stride exceeds isize".into()))?;
    }
    let mut native = vec![0_f64; elements];
    api.check(
        unsafe {
            (api.get_vars_double)(
                file.id,
                inv.var_ids[index],
                start.as_ptr(),
                count.as_ptr(),
                stride.as_ptr(),
                native.as_mut_ptr(),
            )
        },
        "nc_get_vars_double",
    )?;
    for value in &mut native {
        *value = decode_netcdf_cf_value(&metadata.cf_decode, *value)?;
    }
    let mut canonical = vec![0_f64; elements];
    for (target_index, target) in canonical.iter_mut().enumerate() {
        let mut remainder = target_index;
        let mut canonical_indices = [0_usize; 3];
        for axis_index in (0..3).rev() {
            canonical_indices[axis_index] = remainder % output[axis_index];
            remainder /= output[axis_index];
        }
        let source_index = axes.iter().enumerate().try_fold(
            0_usize,
            |linear, (source_axis, canonical_axis)| {
                linear
                    .checked_mul(count[source_axis])
                    .and_then(|value| value.checked_add(canonical_indices[*canonical_axis]))
                    .ok_or_else(|| {
                        AppError::Invalid("terrain coordinate transpose overflow".into())
                    })
            },
        )?;
        *target = native[source_index];
    }
    Ok((source_shape, canonical, source_bytes))
}

pub fn read_values(path: &str, variable: &str) -> AppResult<Vec<f64>> {
    const MAX_COORDINATE_VALUES: u64 = 1_000_000;
    let _native_access = lock_native_netcdf(&NATIVE_NETCDF_ACCESS)?;
    let api = native_api()?;
    let file = open(&api, path)?;
    let inv = inventory(&api, &file, path)?;
    let index = inv
        .metadata
        .variables
        .iter()
        .position(|value| value.name == variable)
        .ok_or_else(|| AppError::NotFound(format!("variable '{variable}' in {path}")))?;
    let variable_metadata = &inv.metadata.variables[index];
    if variable_metadata.shape.len() != 1 {
        return Err(AppError::Invalid(format!(
            "coordinate variable '{variable}' is not one-dimensional"
        )));
    }
    let count = variable_metadata.shape[0];
    if count > MAX_COORDINATE_VALUES {
        return Err(AppError::Invalid(
            "coordinate array exceeds the one-million-value safety limit".into(),
        ));
    }
    let mut values = vec![0_f64; count as usize];
    api.check(
        unsafe { (api.get_var_double)(file.id, inv.var_ids[index], values.as_mut_ptr()) },
        "nc_get_var_double",
    )?;
    values
        .into_iter()
        .map(|value| decode_netcdf_cf_value(&variable_metadata.cf_decode, value))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn native_library_survives_repeated_reads_and_process_cleanup() {
        const CHILD: &str = "NIMBUS_NETCDF_LIFETIME_CHILD";
        if std::env::var_os(CHILD).is_some() {
            for _ in 0..3 {
                assert!(inspect("lifetime-fixture").unwrap().variables.is_empty());
            }
            return;
        }
        let directory = std::env::temp_dir().join(format!(
            "nimbus-netcdf-lifetime-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&directory).unwrap();
        struct Cleanup(PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _cleanup = Cleanup(directory.clone());
        let source = directory.join("library.c");
        std::fs::write(
            &source,
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../tests/ui/native/netcdf_library_lifetime.c"
            )),
        )
        .unwrap();
        let filename = if cfg!(target_os = "macos") {
            "libnetcdf.dylib"
        } else {
            "libnetcdf.so"
        };
        let output = std::process::Command::new("cc")
            .args(if cfg!(target_os = "macos") {
                vec!["-dynamiclib"]
            } else {
                vec!["-shared", "-fPIC"]
            })
            .arg(&source)
            .arg("-o")
            .arg(directory.join(filename))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let marker = directory.join("events");
        let search_key = if cfg!(target_os = "macos") {
            "DYLD_LIBRARY_PATH"
        } else {
            "LD_LIBRARY_PATH"
        };
        let mut search = vec![directory.clone()];
        if let Some(existing) = std::env::var_os(search_key) {
            search.extend(std::env::split_paths(&existing));
        }
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "commands::netcdf_dynamic::tests::native_library_survives_repeated_reads_and_process_cleanup", "--nocapture"])
            .env(CHILD, "1").env("NIMBUS_NETCDF_LIFETIME_MARKER", &marker)
            .env(search_key, std::env::join_paths(search).unwrap()).output().unwrap();
        let events = std::fs::read_to_string(marker).unwrap_or_default();
        assert!(
            output.status.success(),
            "reader exited with {:?}; events={events}; {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            events.starts_with("LOCOCOC"),
            "library unloaded between reads: {events}"
        );
        assert_eq!(events.matches('L').count(), 1, "{events}");
        assert_eq!(
            events.matches('E').count(),
            1,
            "cleanup callback did not complete: {events}"
        );
    }

    #[test]
    fn native_netcdf_access_fails_closed_after_guard_poisoning() {
        let access = Mutex::new(());
        let failed = std::thread::scope(|scope| scope.spawn(|| {
            let _native_access = lock_native_netcdf(&access).unwrap();
            panic!("reader panic while holding native access");
        }).join());
        assert!(failed.is_err());
        assert!(matches!(lock_native_netcdf(&access), Err(AppError::Invalid(message))
            if message.contains("reader panic")));
    }

    fn dimension(name: &str, length: u64) -> NetcdfDimension {
        NetcdfDimension {
            name: name.into(),
            length,
            unlimited: false,
        }
    }

    #[test]
    fn sampled_float64_coordinates_preserve_axis_order_and_stride() {
        fn word(bytes: &mut Vec<u8>, value: u32) {
            bytes.extend_from_slice(&value.to_be_bytes());
        }
        fn name(bytes: &mut Vec<u8>, value: &str) {
            word(bytes, value.len() as u32);
            bytes.extend_from_slice(value.as_bytes());
            while bytes.len() % 4 != 0 {
                bytes.push(0);
            }
        }
        let mut bytes = b"CDF\x01".to_vec();
        word(&mut bytes, 0);
        word(&mut bytes, 10);
        word(&mut bytes, 3);
        for (axis, count) in [("zh", 5), ("yh", 4), ("xh", 6)] {
            name(&mut bytes, axis);
            word(&mut bytes, count);
        }
        word(&mut bytes, 0);
        word(&mut bytes, 0);
        word(&mut bytes, 11);
        word(&mut bytes, 1);
        name(&mut bytes, "physical_z_center_m");
        word(&mut bytes, 3);
        for value in [0, 1, 2, 0, 0, 6, 5 * 4 * 6 * 8] {
            word(&mut bytes, value);
        }
        let offset = bytes.len() + 4;
        word(&mut bytes, offset as u32);
        for k in 0..5 {
            for j in 0..4 {
                for i in 0..6 {
                    bytes.extend_from_slice(&((100 * k + 10 * j + i) as f64).to_be_bytes());
                }
            }
        }
        let path = std::env::temp_dir().join(format!(
            "nimbus-sampled-coordinate-{}-{}.nc",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, bytes).unwrap();
        let result = read_f64_sampled_window(
            path.to_str().unwrap(),
            "physical_z_center_m",
            [1, 1, 0],
            [5, 3, 5],
            [2, 2, 2],
        );
        std::fs::remove_file(&path).unwrap();
        let (shape, values, inspected) = result.unwrap();
        assert_eq!(shape, vec![6, 4, 5]);
        assert_eq!(inspected, 18 * 8);
        let mut expected = Vec::new();
        for i in [1, 3, 5] {
            for j in [1, 3] {
                for k in [0, 2, 4] {
                    expected.push((100 * k + 10 * j + i) as f64);
                }
            }
        }
        assert_eq!(values, expected);
        assert!(read_f64_sampled_window("unused", "unused", [0; 3], [1; 3], [0; 3]).is_err());
    }

    #[test]
    fn resolves_dimensions_in_declaration_order() {
        let dimensions = [dimension("time", 4), dimension("nk", 3), dimension("nj", 2)];
        let (names, shape) =
            resolve_variable_dimensions(&dimensions, &[0, 2], "th").expect("ids are in range");
        assert_eq!(names, vec!["time".to_string(), "nj".to_string()]);
        assert_eq!(shape, vec![4, 2]);
    }

    /// Regression: these indexed `dimensions` directly, so a netCDF-4 file with
    /// group-scoped dimensions (which `nc_inq` under-reports) panicked.
    #[test]
    fn rejects_out_of_range_dimension_id() {
        let dimensions = [dimension("time", 4)];
        let error =
            resolve_variable_dimensions(&dimensions, &[1], "th").expect_err("id 1 is past the end");
        assert!(error.to_string().contains("missing dimension 1"));
    }

    /// A negative id would wrap to a huge index under `as usize`.
    #[test]
    fn rejects_negative_dimension_id() {
        let dimensions = [dimension("time", 4)];
        assert!(resolve_variable_dimensions(&dimensions, &[-1], "th").is_err());
    }

    /// Regression: `read_field` allocated a whole variable plus an equal-sized
    /// transpose target with no budget, so one multi-GB CM1 field OOM'd the app.
    #[test]
    fn whole_field_budget_admits_realistic_grids_and_rejects_oversized_ones() {
        let bytes = |elements: u64| elements.saturating_mul(4);
        // A 400x400x100 f32 field is ~64 MB — a normal Studio read.
        assert!(bytes(400 * 400 * 100) <= MAX_NETCDF_SOURCE_BYTES);
        // A 1500x1500x100 field is ~900 MB, and the transpose doubles it.
        assert!(bytes(1500 * 1500 * 100) > MAX_NETCDF_SOURCE_BYTES);
    }

    #[test]
    fn resolves_scalar_variables_to_empty_shape() {
        let (names, shape) =
            resolve_variable_dimensions(&[dimension("time", 4)], &[], "scalar").expect("no ids");
        assert!(names.is_empty() && shape.is_empty());
    }
}
