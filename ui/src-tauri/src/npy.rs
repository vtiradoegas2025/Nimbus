//! Minimal NumPy `.npy` reader for the float32, C-order arrays the Nimbus
//! simulator writes.
//!
//! Mirrors the format contract enforced on the TS side (`ui/src/lib/npy.ts`)
//! and by the writer (`src/io/npy_writer.cpp`): magic + version 1 or 2, dtype
//! `<f4` (little-endian float32), `fortran_order` False. Anything else is
//! rejected loud rather than misread.
//!
//! This exists so the outputs commands can reduce a field to a few scalars
//! (`reduce_field`) without shipping the whole multi-MB array across the IPC
//! boundary — a 256x256x80 field is ~21 MB, and the time-series plot needs one
//! number per step, not the array.

use crate::error::{AppError, AppResult};
use std::mem::size_of;
use std::path::Path;

const MAGIC: [u8; 6] = [0x93, b'N', b'U', b'M', b'P', b'Y'];
pub const MAX_FIELD_RANK: usize = 3;
pub const MAX_FIELD_DIMENSION: usize = 16_384;
pub const MAX_FIELD_ELEMENTS: usize = 64 * 1024 * 1024;
pub const MAX_FIELD_DECODED_BYTES: usize = MAX_FIELD_ELEMENTS * size_of::<f32>();
pub const MAX_NPY_HEADER_BYTES: usize = 64 * 1024;
pub const MAX_NPY_FILE_BYTES: usize = 12 + MAX_NPY_HEADER_BYTES + MAX_FIELD_DECODED_BYTES;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidatedFieldShape {
    pub rank: usize,
    pub element_count: usize,
    pub decoded_bytes: usize,
}

/// Validate a field descriptor without applying the whole-volume decode cap.
/// Window readers use this for logical source shapes because their allocation
/// is governed by the independently bounded requested window.
pub fn validate_logical_field_shape(shape: &[usize]) -> AppResult<ValidatedFieldShape> {
    if shape.is_empty() || shape.len() > MAX_FIELD_RANK {
        return Err(AppError::Invalid(format!(
            "unsupported field rank {} (expected 1..={MAX_FIELD_RANK})",
            shape.len()
        )));
    }

    let mut element_count = 1usize;
    for &dim in shape {
        if dim == 0 || dim > MAX_FIELD_DIMENSION {
            return Err(AppError::Invalid(format!(
                "field dimension {dim} outside 1..={MAX_FIELD_DIMENSION}"
            )));
        }
        element_count = element_count
            .checked_mul(dim)
            .ok_or_else(|| AppError::Invalid("field shape product overflow".into()))?;
    }
    let decoded_bytes = element_count
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| AppError::Invalid("decoded field byte count overflow".into()))?;
    Ok(ValidatedFieldShape {
        rank: shape.len(),
        element_count,
        decoded_bytes,
    })
}

/// Validate the allocation-bearing part of every Nimbus field descriptor.
/// This is shared by NPY and ZFP so neither format can bypass the UI's bounded
/// working-set contract before allocating a decoded volume.
pub fn validate_field_shape(shape: &[usize]) -> AppResult<ValidatedFieldShape> {
    let validated = validate_logical_field_shape(shape)?;
    if validated.element_count > MAX_FIELD_ELEMENTS {
        return Err(AppError::Invalid(format!(
            "field has {} elements; limit is {MAX_FIELD_ELEMENTS}",
            validated.element_count
        )));
    }
    let decoded_bytes = validated.decoded_bytes;
    if decoded_bytes > MAX_FIELD_DECODED_BYTES {
        return Err(AppError::Invalid(format!(
            "decoded field is {decoded_bytes} bytes; limit is {MAX_FIELD_DECODED_BYTES}"
        )));
    }
    Ok(ValidatedFieldShape {
        rank: validated.rank,
        element_count: validated.element_count,
        decoded_bytes,
    })
}

/// Where the array body begins in the file, and how many `f32` elements it
/// holds (the product of the header shape).
pub struct NpyF32Layout {
    #[allow(dead_code)]
    pub version: [u8; 2],
    pub data_offset: usize,
    pub decoded_bytes: usize,
    /// Header shape dims in C order; length is the array rank. Lets callers
    /// reduce per-axis (e.g. the time-height plot's per-level reduction).
    pub shape: Vec<usize>,
}

/// Parse and validate an `.npy` header, returning the body offset and element
/// count. Rejects any layout the simulator does not write (non-`<f4`,
/// fortran-order, unknown version) and any header/body truncation.
pub fn parse_f32_layout(bytes: &[u8]) -> AppResult<NpyF32Layout> {
    if bytes.len() < 8 || bytes[..6] != MAGIC {
        return Err(AppError::Invalid("not an NPY file (bad magic)".into()));
    }

    let major = bytes[6];
    let minor = bytes[7];
    if minor != 0 {
        return Err(AppError::Invalid(format!(
            "unsupported NPY version {major}.{minor}"
        )));
    }
    let (header_len, header_offset) = match major {
        1 if bytes.len() >= 10 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize),
        2 if bytes.len() >= 12 => (
            usize::try_from(u32::from_le_bytes([
                bytes[8], bytes[9], bytes[10], bytes[11],
            ]))
            .map_err(|_| AppError::Invalid("NPY header length does not fit usize".into()))?,
            12usize,
        ),
        1 | 2 => return Err(AppError::Invalid("truncated NPY version header".into())),
        _ => {
            return Err(AppError::Invalid(format!(
                "unsupported NPY version {major}"
            )))
        }
    };

    if header_len == 0 || header_len > MAX_NPY_HEADER_BYTES {
        return Err(AppError::Invalid(format!(
            "NPY header is {header_len} bytes; limit is {MAX_NPY_HEADER_BYTES}"
        )));
    }

    let data_offset = header_offset
        .checked_add(header_len)
        .ok_or_else(|| AppError::Invalid("NPY header offset overflow".into()))?;
    if bytes.len() < data_offset {
        return Err(AppError::Invalid("truncated NPY header".into()));
    }

    let header = std::str::from_utf8(&bytes[header_offset..data_offset])
        .map_err(|_| AppError::Invalid("non-UTF8 NPY header".into()))?;
    // Tolerate the writer's exact spacing by comparing against a
    // whitespace-stripped copy.
    let compact: String = header.chars().filter(|c| !c.is_whitespace()).collect();

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

    let shape = parse_shape(&compact)?;
    let validated = validate_field_shape(&shape)?;
    let body_bytes = bytes.len() - data_offset;
    if body_bytes != validated.decoded_bytes {
        return Err(AppError::Invalid(format!(
            "NPY body size mismatch: need {} bytes, have {}",
            validated.decoded_bytes, body_bytes
        )));
    }

    Ok(NpyF32Layout {
        version: [major, minor],
        data_offset,
        decoded_bytes: validated.decoded_bytes,
        shape,
    })
}

/// Read a complete bounded NPY file and validate it before returning bytes.
/// Checking metadata first prevents an oversized local artifact from forcing
/// an unbounded allocation in `fs::read`.
pub fn read_f32_file(path: &Path) -> AppResult<Vec<u8>> {
    let file_len = usize::try_from(std::fs::metadata(path)?.len())
        .map_err(|_| AppError::Invalid("NPY file length does not fit usize".into()))?;
    if file_len > MAX_NPY_FILE_BYTES {
        return Err(AppError::Invalid(format!(
            "NPY file is {file_len} bytes; limit is {MAX_NPY_FILE_BYTES}"
        )));
    }
    let bytes = std::fs::read(path)?;
    parse_f32_layout(&bytes)?;
    Ok(bytes)
}

/// Extract the `'shape': (a, b, c, )` tuple from a whitespace-stripped header
/// as a Vec of dims (empty for a 0-d / scalar shape).
fn parse_shape(compact: &str) -> AppResult<Vec<usize>> {
    let key = "'shape':(";
    if compact.matches("'shape':").count() != 1 {
        return Err(AppError::Invalid(
            "NPY header must contain exactly one shape".into(),
        ));
    }
    let start = compact
        .find(key)
        .ok_or_else(|| AppError::Invalid("NPY header missing shape".into()))?
        + key.len();
    let end = compact[start..]
        .find(')')
        .ok_or_else(|| AppError::Invalid("NPY header malformed shape".into()))?
        + start;

    let mut dims: Vec<usize> = Vec::new();
    let raw_shape = &compact[start..end];
    let mut parts = raw_shape.split(',').peekable();
    while let Some(part) = parts.next() {
        if part.is_empty() {
            if !dims.is_empty() && parts.peek().is_none() && raw_shape.ends_with(',') {
                continue; // one trailing comma is valid tuple syntax
            }
            return Err(AppError::Invalid("empty NPY shape dimension".into()));
        }
        if !part.bytes().all(|b| b.is_ascii_digit()) {
            return Err(AppError::Invalid(format!(
                "bad NPY shape dimension '{part}'"
            )));
        }
        let dim: usize = part
            .parse()
            .map_err(|_| AppError::Invalid(format!("bad NPY shape dimension '{part}'")))?;
        dims.push(dim);
    }
    Ok(dims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct ValidationCase {
        name: String,
        accepted: bool,
        bytes: Vec<u8>,
    }

    /// Build a valid v1.0 `.npy` byte stream for a `<f4` array of `shape`.
    fn make_npy(shape: &[usize], values: &[f32]) -> Vec<u8> {
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({shape_str}), }}");
        // Pad with spaces so (10 + header_len) is a multiple of 64, terminated
        // by '\n' — the v1.0 alignment rule.
        let mut header = dict.into_bytes();
        header.push(b'\n');
        while (10 + header.len()) % 64 != 0 {
            header.insert(header.len() - 1, b' ');
        }
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        out.push(1);
        out.push(0);
        out.push((header.len() & 0xff) as u8);
        out.push(((header.len() >> 8) & 0xff) as u8);
        out.extend_from_slice(&header);
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn parses_valid_layout() {
        let bytes = make_npy(&[2, 3], &[1.0; 6]);
        let layout = parse_f32_layout(&bytes).unwrap();
        assert_eq!(layout.decoded_bytes / size_of::<f32>(), 6);
        assert_eq!(layout.shape, vec![2, 3]);
        assert_eq!((10 + (layout.data_offset - 10)) % 64, 0);
        assert_eq!(layout.data_offset % 64, 0);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = make_npy(&[2], &[1.0, 2.0]);
        bytes[1] = b'X';
        assert!(parse_f32_layout(&bytes).is_err());
    }

    #[test]
    fn rejects_truncated_body() {
        let mut bytes = make_npy(&[4], &[1.0, 2.0, 3.0, 4.0]);
        bytes.truncate(bytes.len() - 4); // drop one element
        assert!(parse_f32_layout(&bytes).is_err());
    }

    #[test]
    fn rejects_truncated_v2_prefix_without_panicking() {
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&[2, 0, 0, 0]);
        assert!(parse_f32_layout(&bytes).is_err());
    }

    #[test]
    fn rejects_zero_junk_oversize_and_trailing_shapes() {
        assert!(parse_f32_layout(&make_npy(&[0, 2], &[])).is_err());
        assert!(parse_f32_layout(&make_npy(&[MAX_FIELD_DIMENSION + 1], &[])).is_err());
        assert!(parse_f32_layout(&make_npy(&[2, 2, 2, 2], &[])).is_err());

        let mut trailing = make_npy(&[1], &[1.0]);
        trailing.extend_from_slice(&0f32.to_le_bytes());
        assert!(parse_f32_layout(&trailing).is_err());

        let mut junk = make_npy(&[2], &[1.0, 2.0]);
        let header = std::str::from_utf8(&junk[10..64]).unwrap();
        let pos = header.find("(2)").unwrap();
        junk[10 + pos + 1] = b'x';
        assert!(parse_f32_layout(&junk).is_err());
    }

    #[test]
    fn shared_validation_corpus_matches_typescript() {
        let generator = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/ui/data/npy_validation_cases.mjs");
        let generated = std::process::Command::new("node")
            .arg(generator)
            .output()
            .expect("Node 22 or newer is required for the shared NPY validation generator");
        assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
        let cases: Vec<ValidationCase> = serde_json::from_slice(&generated.stdout).unwrap();
        assert_eq!(cases.len(), 20);
        for case in cases {
            assert_eq!(
                parse_f32_layout(&case.bytes).is_ok(),
                case.accepted,
                "shared validation case '{}'",
                case.name
            );
        }
    }

    #[test]
    fn rejects_non_f4_dtype() {
        // Hand-build an <f8 header; parser must refuse.
        let dict = "{'descr': '<f8', 'fortran_order': False, 'shape': (1, ), }";
        let mut header = dict.as_bytes().to_vec();
        header.push(b'\n');
        while (10 + header.len()) % 64 != 0 {
            header.insert(header.len() - 1, b' ');
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC);
        bytes.push(1);
        bytes.push(0);
        bytes.push((header.len() & 0xff) as u8);
        bytes.push(((header.len() >> 8) & 0xff) as u8);
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&0.0f64.to_le_bytes());
        assert!(parse_f32_layout(&bytes).is_err());
    }
}
