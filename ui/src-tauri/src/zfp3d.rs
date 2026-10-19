//! Reader for the simulator's ZFP-compressed 3D output (`.zfp3d`).
//!
//! Mirrors the C++ reference decoder (`src/core/output/zfp_reader.cpp` +
//! `include/core/output/zfp_reader.hpp`) and the writer's framing
//! (`src/core/output/output_writer.cpp`). The payload is a standard libzfp
//! bitstream (`zfp_compress`); we decode it via the `zfp-sys` binding, which
//! vendors and builds libzfp 1.0.1 — the same codec version the simulator
//! links — so the bytes round-trip bit-compatibly.
//!
//! The outputs commands re-serialize the decoded volume as an in-memory `.npy`
//! (`encode_npy_f32`) so the whole frontend (`lib/npy.ts`, `lib/fieldCache.ts`,
//! every plot widget) stays unchanged: it still receives `<f4` C-order bytes.
//!
//! Random access: Studio scrubs to arbitrary frames. With keyframe-only output
//! (`zfp_keyframe_interval: 0`, the default) every `.zfp3d` is self-contained,
//! so [`reconstruct_field`] is a single decode. When delta encoding is enabled,
//! a frame stores a residual against prior frames; [`reconstruct_field`] then
//! walks back to the most recent keyframe and replays forward.

mod sampled;
pub use sampled::read_sampled_field_window;

use sha2::{Digest, Sha256};
use std::ffi::c_void;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::error::{AppError, AppResult};

// ── On-disk format (little-endian), per zfp_reader.cpp / output_writer.cpp ──
//   [0]  char[4] magic "ZFP3"
//   [4]  u16     version (1, 2, seekable-brick 3, reversible-brick 4)
//   [6]  u16     mode (0=accuracy, 1=precision, 2=rate, v4-only 3=reversible)
//   [8]  i32     dim0
//   [12] i32     dim1
//   [16] i32     dim2
//   [20] f64     mode_param (accuracy tolerance / precision bits / rate bps)
//   [28] u64     compressed_size
//   v2/v3:
//   [36] u8      flags (0=keyframe, 1=simple delta, 3=predictive delta)
//   [37] u8[3]   reserved
//   v3 adds "ZB3D" followed by fixed-size directory entries;
//   each entry names one independently decodable libzfp payload.
//   v4 adds "ZB4D" and a compressed-payload SHA-256 beside each offset.
const MAGIC: [u8; 4] = *b"ZFP3";
const HEADER_V1_LEN: usize = 36;
const HEADER_V2_LEN: usize = 40;
const HEADER_V3_LEN: usize = 44;
const BRICK_ENTRY_LEN: usize = 8;
const BRICK_ENTRY_V4_LEN: usize = 40;
const ZFP_V3_BRICK_SHAPE: [usize; 3] = [64, 64, 32];
const MAX_ZFP_BRICK_COMPRESSED_BYTES: usize = 4 * 1024 * 1024;
const FLAG_KEYFRAME: u8 = 0x00;
const FLAG_SIMPLE_DELTA: u8 = 0x01;
const FLAG_PREDICTIVE_DELTA: u8 = 0x03;
const MAX_ZFP_COMPRESSED_BYTES: usize = crate::npy::MAX_FIELD_DECODED_BYTES;
const MAX_LEGACY_ZFP_FILE_BYTES: usize = HEADER_V2_LEN + MAX_ZFP_COMPRESSED_BYTES;
const MAX_ZFP_CHAIN_FRAMES: usize = 256;
const MAX_ZFP_CHAIN_COMPRESSED_BYTES: usize = 1024 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq)]
pub struct Zfp3dHeader {
    /// On-disk format version (1, 2, 3, or 4), validated in `parse_header`. Retained for
    /// header completeness / parity with the C++ `ZfpHeader`, and asserted in the
    /// unit tests; decoding branches on the locally-parsed version for the payload
    /// offset, so the stored copy isn't read again in non-test builds.
    #[allow(dead_code)]
    pub version: u16,
    pub mode: u16,
    pub dim0: i32,
    pub dim1: i32,
    pub dim2: i32,
    pub mode_param: f64,
    pub compressed_size: usize,
    pub flags: u8,
    pub brick_shape: Option<[usize; 3]>,
    pub brick_counts: Option<[usize; 3]>,
    pub brick_entry_bytes: Option<usize>,
    element_count: usize,
}

impl Zfp3dHeader {
    /// Logical C-order shape `[dim0, dim1, dim2]` — the same `(nr, nth, nz)`
    /// contract the `.npy` path exposes.
    pub fn shape(&self) -> Vec<usize> {
        vec![self.dim0 as usize, self.dim1 as usize, self.dim2 as usize]
    }
    fn is_keyframe(&self) -> bool {
        self.flags == FLAG_KEYFRAME
    }
    pub fn is_bricked(&self) -> bool {
        self.version == 3 || self.version == 4
    }
}

fn rd_u16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn rd_i32(b: &[u8], o: usize) -> i32 {
    i32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn rd_u64(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes([
        b[o],
        b[o + 1],
        b[o + 2],
        b[o + 3],
        b[o + 4],
        b[o + 5],
        b[o + 6],
        b[o + 7],
    ])
}
fn rd_f64(b: &[u8], o: usize) -> f64 {
    f64::from_le_bytes([
        b[o],
        b[o + 1],
        b[o + 2],
        b[o + 3],
        b[o + 4],
        b[o + 5],
        b[o + 6],
        b[o + 7],
    ])
}

fn read_header_prefix(file: &mut std::fs::File) -> AppResult<([u8; HEADER_V3_LEN], usize)> {
    let mut bytes = [0u8; HEADER_V3_LEN];
    let mut count = 0usize;
    while count < bytes.len() {
        let read = file.read(&mut bytes[count..])?;
        if read == 0 {
            break;
        }
        count += read;
    }
    Ok((bytes, count))
}

/// Parse and validate a `.zfp3d` header. Returns the header and the byte offset
/// where the compressed payload begins. Rejects bad magic, unknown version /
/// mode / flags, and non-positive dims — loud rather than misread, matching the
/// C++ reader's checks.
pub fn parse_header(bytes: &[u8]) -> AppResult<(Zfp3dHeader, usize)> {
    if bytes.len() < HEADER_V1_LEN || bytes[0..4] != MAGIC {
        return Err(AppError::Invalid("not a .zfp3d file (bad magic)".into()));
    }
    let version = rd_u16(bytes, 4);
    if version != 1 && version != 2 && version != 3 && version != 4 {
        return Err(AppError::Invalid(format!(
            "unsupported .zfp3d version {version}"
        )));
    }
    let mode = rd_u16(bytes, 6);
    if (version == 4 && mode != 3) || (version != 4 && mode > 2) {
        return Err(AppError::Invalid(format!("unsupported .zfp3d mode {mode}")));
    }
    let dim0 = rd_i32(bytes, 8);
    let dim1 = rd_i32(bytes, 12);
    let dim2 = rd_i32(bytes, 16);
    if dim0 <= 0 || dim1 <= 0 || dim2 <= 0 {
        return Err(AppError::Invalid(format!(
            "invalid .zfp3d dims ({dim0}, {dim1}, {dim2})"
        )));
    }
    let mode_param = rd_f64(bytes, 20);
    if (version == 4 && mode_param.to_bits() != 0)
        || (version != 4 && (!mode_param.is_finite() || mode_param <= 0.0))
    {
        return Err(AppError::Invalid(format!(
            "invalid .zfp3d mode parameter {mode_param}"
        )));
    }
    let compressed_size = usize::try_from(rd_u64(bytes, 28))
        .map_err(|_| AppError::Invalid(".zfp3d payload length does not fit usize".into()))?;
    if compressed_size == 0 || (version < 3 && compressed_size > MAX_ZFP_COMPRESSED_BYTES) {
        return Err(AppError::Invalid(format!(
            ".zfp3d payload is {compressed_size} bytes; legacy limit is {MAX_ZFP_COMPRESSED_BYTES}"
        )));
    }

    let (flags, payload_offset) = if version >= 2 {
        if bytes.len() < HEADER_V2_LEN {
            return Err(AppError::Invalid("truncated .zfp3d v2 header".into()));
        }
        let flags = bytes[36];
        if flags != FLAG_KEYFRAME && flags != FLAG_SIMPLE_DELTA && flags != FLAG_PREDICTIVE_DELTA {
            return Err(AppError::Invalid(format!(
                "unknown .zfp3d frame flags 0x{flags:02x}"
            )));
        }
        if version == 4 && (flags != FLAG_KEYFRAME || bytes[37..40] != [0, 0, 0]) {
            return Err(AppError::Invalid(
                "v4 reversible frame must be a canonical keyframe".into(),
            ));
        }
        (flags, HEADER_V2_LEN)
    } else {
        (FLAG_KEYFRAME, HEADER_V1_LEN)
    };

    let shape = [dim0 as usize, dim1 as usize, dim2 as usize];
    if shape
        .iter()
        .any(|dimension| *dimension > crate::npy::MAX_FIELD_DIMENSION)
    {
        return Err(AppError::Invalid(format!(
            ".zfp3d dimension exceeds {}",
            crate::npy::MAX_FIELD_DIMENSION
        )));
    }
    let element_count = if version >= 3 {
        checked_product(shape, "ZFP logical field")?
    } else {
        crate::npy::validate_field_shape(&shape)?.element_count
    };
    let (brick_shape, brick_counts, brick_entry_bytes, payload_offset) = if version >= 3 {
        if bytes.len() < HEADER_V3_LEN {
            return Err(AppError::Invalid("truncated .zfp3d v3 brick header".into()));
        }
        let required_magic: &[u8] = if version == 4 { b"ZB4D" } else { b"ZB3D" };
        if &bytes[40..44] != required_magic {
            return Err(AppError::Invalid(format!(
                "invalid .zfp3d v{version} brick magic"
            )));
        }
        let counts = [
            shape[0].div_ceil(ZFP_V3_BRICK_SHAPE[0]),
            shape[1].div_ceil(ZFP_V3_BRICK_SHAPE[1]),
            shape[2].div_ceil(ZFP_V3_BRICK_SHAPE[2]),
        ];
        (
            Some(ZFP_V3_BRICK_SHAPE),
            Some(counts),
            Some(if version == 4 {
                BRICK_ENTRY_V4_LEN
            } else {
                BRICK_ENTRY_LEN
            }),
            HEADER_V3_LEN,
        )
    } else {
        (None, None, None, payload_offset)
    };

    Ok((
        Zfp3dHeader {
            version,
            mode,
            dim0,
            dim1,
            dim2,
            mode_param,
            compressed_size,
            flags,
            brick_shape,
            brick_counts,
            brick_entry_bytes,
            element_count,
        },
        payload_offset,
    ))
}

/// Decompress one libzfp payload into a C-order `[dim0][dim1][dim2]` f32 volume.
///
/// CRITICAL: the writer builds the zfp field with the dims REVERSED —
/// `zfp_field_3d(ptr, float, dim2, dim1, dim0)` — so `dim2` is the fastest
/// (contiguous) axis, matching the C-contiguous `[dim0][dim1][dim2]` layout.
/// The reader MUST pass the same reversed order or the volume is transposed.
fn decompress_payload_dims(
    h: &Zfp3dHeader,
    payload: &[u8],
    dims: [usize; 3],
) -> AppResult<Vec<f32>> {
    use zfp_sys::*;

    let n = dims
        .into_iter()
        .try_fold(1usize, |product, dimension| product.checked_mul(dimension))
        .ok_or_else(|| AppError::Invalid("ZFP decoded brick size overflow".into()))?;
    let mut out = vec![0f32; n];

    // libzfp's bitstream reads from a `*mut` buffer (the C API is non-const).
    // Copy the (small, compressed) payload into a mutable buffer to honor that
    // signature without aliasing the caller's slice.
    let mut buf = payload.to_vec();

    let ok = unsafe {
        let field = zfp_field_3d(
            out.as_mut_ptr() as *mut c_void,
            zfp_type_zfp_type_float,
            dims[2],
            dims[1],
            dims[0],
        );
        if field.is_null() {
            return Err(AppError::Invalid("zfp_field_3d allocation failed".into()));
        }
        let zfp = zfp_stream_open(std::ptr::null_mut());
        if zfp.is_null() {
            zfp_field_free(field);
            return Err(AppError::Invalid("zfp_stream_open failed".into()));
        }
        match h.mode {
            0 => {
                zfp_stream_set_accuracy(zfp, h.mode_param);
            }
            1 => {
                zfp_stream_set_precision(zfp, h.mode_param as _);
            }
            2 => {
                zfp_stream_set_rate(zfp, h.mode_param, zfp_type_zfp_type_float, 3, 0);
            }
            3 => zfp_stream_set_reversible(zfp),
            _ => unreachable!("validated ZFP mode"),
        }
        let stream = stream_open(buf.as_mut_ptr() as *mut c_void, buf.len());
        if stream.is_null() {
            zfp_stream_close(zfp);
            zfp_field_free(field);
            return Err(AppError::Invalid("zfp bitstream open failed".into()));
        }
        zfp_stream_set_bit_stream(zfp, stream);
        zfp_stream_rewind(zfp);
        let result = zfp_decompress(zfp, field);
        // Cleanup order mirrors the zfp-sys example: field, stream wrapper,
        // bitstream.
        zfp_field_free(field);
        zfp_stream_close(zfp);
        stream_close(stream);
        result != 0
    };

    if !ok {
        return Err(AppError::Invalid("zfp_decompress failed".into()));
    }
    Ok(out)
}

fn decompress_payload(h: &Zfp3dHeader, payload: &[u8]) -> AppResult<Vec<f32>> {
    decompress_payload_dims(
        h,
        payload,
        [h.dim0 as usize, h.dim1 as usize, h.dim2 as usize],
    )
}

/// Parse + decode one in-memory `.zfp3d` frame. Returns the header and the RAW
/// decoded values (absolute for a keyframe, residual for a delta).
pub fn decode_frame_bytes(bytes: &[u8]) -> AppResult<(Zfp3dHeader, Vec<f32>)> {
    let (header, payload_offset) = parse_header(bytes)?;
    if header.is_bricked() {
        return Err(AppError::Invalid(
            "seekable-brick .zfp3d must be decoded through its bounded file reader".into(),
        ));
    }
    let end = payload_offset
        .checked_add(header.compressed_size)
        .ok_or_else(|| AppError::Invalid("zfp3d payload size overflow".into()))?;
    if bytes.len() != end {
        return Err(AppError::Invalid(format!(
            ".zfp3d payload size mismatch: need {end} bytes, have {}",
            bytes.len()
        )));
    }
    let data = decompress_payload(&header, &bytes[payload_offset..end])?;
    Ok((header, data))
}

#[derive(Clone, Copy, Debug)]
struct BrickEntry {
    origin: [usize; 3],
    extent: [usize; 3],
    payload_offset: u64,
    compressed_size: usize,
    payload_digest: Option<[u8; 32]>,
}

fn checked_product(values: [usize; 3], label: &str) -> AppResult<usize> {
    values
        .into_iter()
        .try_fold(1usize, |product, value| product.checked_mul(value))
        .ok_or_else(|| AppError::Invalid(format!("{label} size overflow")))
}

fn brick_count(header: &Zfp3dHeader) -> AppResult<usize> {
    checked_product(
        header
            .brick_counts
            .ok_or_else(|| AppError::Invalid("missing v3 brick counts".into()))?,
        "ZFP brick directory",
    )
}

fn read_brick_entry(
    file: &mut std::fs::File,
    header: &Zfp3dHeader,
    index: usize,
    file_len: u64,
) -> AppResult<BrickEntry> {
    let counts = header
        .brick_counts
        .ok_or_else(|| AppError::Invalid("missing v3 brick counts".into()))?;
    let brick_shape = header
        .brick_shape
        .ok_or_else(|| AppError::Invalid("missing v3 brick shape".into()))?;
    let count = brick_count(header)?;
    if index >= count {
        return Err(AppError::Invalid("v3 brick index out of range".into()));
    }
    let entry_bytes = header
        .brick_entry_bytes
        .ok_or_else(|| AppError::Invalid("missing v3 brick entry size".into()))?;
    let directory_start = HEADER_V3_LEN
        .checked_add(header.compressed_size)
        .ok_or_else(|| AppError::Invalid("v3 brick directory offset overflow".into()))?;
    let offset = directory_start
        .checked_add(
            index
                .checked_mul(entry_bytes)
                .ok_or_else(|| AppError::Invalid("v3 brick directory offset overflow".into()))?,
        )
        .ok_or_else(|| AppError::Invalid("v3 brick directory offset overflow".into()))?;
    file.seek(SeekFrom::Start(offset as u64))?;
    let mut bytes = [0u8; BRICK_ENTRY_V4_LEN];
    file.read_exact(&mut bytes[..entry_bytes])?;
    let payload_offset = rd_u64(&bytes, 0);
    let payload_digest = if header.version == 4 {
        let mut value = [0u8; 32];
        value.copy_from_slice(&bytes[8..40]);
        Some(value)
    } else {
        None
    };
    let next_payload_offset = if index + 1 < count {
        file.read_exact(&mut bytes[..entry_bytes])?;
        rd_u64(&bytes, 0)
    } else {
        directory_start as u64
    };
    let plane = counts[1]
        .checked_mul(counts[2])
        .ok_or_else(|| AppError::Invalid("v3 brick index geometry overflow".into()))?;
    let bi = index / plane;
    let remainder = index % plane;
    let bj = remainder / counts[2];
    let bk = remainder % counts[2];
    let origin = [
        bi * brick_shape[0],
        bj * brick_shape[1],
        bk * brick_shape[2],
    ];
    let shape = header.shape();
    let extent = [
        brick_shape[0].min(shape[0] - origin[0]),
        brick_shape[1].min(shape[1] - origin[1]),
        brick_shape[2].min(shape[2] - origin[2]),
    ];
    let directory_end = directory_start
        .checked_add(
            count
                .checked_mul(entry_bytes)
                .ok_or_else(|| AppError::Invalid("v3 brick directory size overflow".into()))?,
        )
        .ok_or_else(|| AppError::Invalid("v3 brick directory size overflow".into()))?
        as u64;
    if directory_end != file_len
        || payload_offset < HEADER_V3_LEN as u64
        || next_payload_offset <= payload_offset
        || next_payload_offset > directory_start as u64
    {
        return Err(AppError::Invalid(
            "invalid .zfp3d v3 brick directory entry".into(),
        ));
    }
    let compressed_size = usize::try_from(next_payload_offset - payload_offset)
        .map_err(|_| AppError::Invalid("v3 brick payload size does not fit usize".into()))?;
    if compressed_size > MAX_ZFP_BRICK_COMPRESSED_BYTES {
        return Err(AppError::Invalid(
            "invalid .zfp3d v3 brick directory entry".into(),
        ));
    }
    Ok(BrickEntry {
        origin,
        extent,
        payload_offset,
        compressed_size,
        payload_digest,
    })
}

fn verify_brick_bytes(entry: &BrickEntry, payload: &[u8]) -> AppResult<()> {
    if let Some(expected) = entry.payload_digest {
        if Sha256::digest(payload).as_slice() != expected {
            return Err(AppError::Invalid(
                "v4 brick payload SHA-256 mismatch".into(),
            ));
        }
    }
    Ok(())
}

fn verify_brick_file_digest(file: &mut std::fs::File, entry: &BrickEntry) -> AppResult<()> {
    let Some(expected) = entry.payload_digest else {
        return Ok(());
    };
    file.seek(SeekFrom::Start(entry.payload_offset))?;
    let mut remaining = entry.compressed_size;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    while remaining > 0 {
        let count = remaining.min(buffer.len());
        file.read_exact(&mut buffer[..count])?;
        digest.update(&buffer[..count]);
        remaining -= count;
    }
    if digest.finalize().as_slice() != expected {
        return Err(AppError::Invalid(
            "v4 brick payload SHA-256 mismatch".into(),
        ));
    }
    Ok(())
}

fn validate_brick_container(
    file: &mut std::fs::File,
    header: &Zfp3dHeader,
    file_len: u64,
) -> AppResult<()> {
    let count = brick_count(header)?;
    let entry_bytes = header
        .brick_entry_bytes
        .ok_or_else(|| AppError::Invalid("missing v3 brick entry size".into()))?;
    let directory_start = HEADER_V3_LEN
        .checked_add(header.compressed_size)
        .ok_or_else(|| AppError::Invalid("v3 brick directory offset overflow".into()))?;
    let directory_end = directory_start
        .checked_add(
            count
                .checked_mul(entry_bytes)
                .ok_or_else(|| AppError::Invalid("v3 brick directory size overflow".into()))?,
        )
        .ok_or_else(|| AppError::Invalid("v3 brick directory size overflow".into()))?
        as u64;
    if directory_end != file_len {
        return Err(AppError::Invalid(
            ".zfp3d v3 directory/file size mismatch".into(),
        ));
    }
    let mut expected_offset = HEADER_V3_LEN as u64;
    let mut compressed_sum = 0usize;
    for index in 0..count {
        let entry = read_brick_entry(file, header, index, file_len)?;
        if entry.payload_offset != expected_offset {
            return Err(AppError::Invalid(
                "non-canonical .zfp3d v3 payload layout".into(),
            ));
        }
        expected_offset = expected_offset
            .checked_add(entry.compressed_size as u64)
            .ok_or_else(|| AppError::Invalid("v3 payload offset overflow".into()))?;
        compressed_sum = compressed_sum
            .checked_add(entry.compressed_size)
            .ok_or_else(|| AppError::Invalid("v3 compressed-size sum overflow".into()))?;
        verify_brick_file_digest(file, &entry)?;
    }
    if compressed_sum != header.compressed_size || expected_offset != directory_start as u64 {
        return Err(AppError::Invalid(
            ".zfp3d v3 payload total/file size mismatch".into(),
        ));
    }
    Ok(())
}

fn validate_brick_container_layout(header: &Zfp3dHeader, file_len: u64) -> AppResult<()> {
    let count = brick_count(header)?;
    let entry_bytes = header
        .brick_entry_bytes
        .ok_or_else(|| AppError::Invalid("missing v3 brick entry size".into()))?;
    let expected_len = HEADER_V3_LEN
        .checked_add(header.compressed_size)
        .and_then(|value| value.checked_add(count.checked_mul(entry_bytes)?))
        .ok_or_else(|| AppError::Invalid("v3 brick container size overflow".into()))?;
    if u64::try_from(expected_len)
        .map_err(|_| AppError::Invalid("v3 brick container size does not fit u64".into()))?
        != file_len
    {
        return Err(AppError::Invalid(
            ".zfp3d v3 directory/file size mismatch".into(),
        ));
    }
    Ok(())
}

fn validate_window(shape: &[usize], origin: [usize; 3], extent: [usize; 3]) -> AppResult<usize> {
    if shape.len() != 3 || extent.contains(&0) {
        return Err(AppError::Invalid(
            "ZFP window requires positive 3D extent".into(),
        ));
    }
    for axis in 0..3 {
        if origin[axis] > shape[axis] || extent[axis] > shape[axis] - origin[axis] {
            return Err(AppError::Invalid(format!(
                "ZFP window exceeds source axis {axis}"
            )));
        }
    }
    checked_product(extent, "ZFP window")
}

fn read_zfp3d_raw_window(
    path: &Path,
    origin: [usize; 3],
    extent: [usize; 3],
) -> AppResult<(Zfp3dHeader, Vec<f32>)> {
    let mut file = std::fs::File::open(path)?;
    let file_len = file.metadata()?.len();
    let (header_bytes, header_len) = read_header_prefix(&mut file)?;
    let (header, _) = parse_header(&header_bytes[..header_len])?;
    let output_count = validate_window(&header.shape(), origin, extent)?;
    if !header.is_bricked() {
        let (_, full) = read_zfp3d_frame(path)?;
        let shape = header.shape();
        if output_count == header.element_count {
            return Ok((header, full));
        }
        let mut window = vec![0.0f32; output_count];
        for i in 0..extent[0] {
            for j in 0..extent[1] {
                for k in 0..extent[2] {
                    window[(i * extent[1] + j) * extent[2] + k] = full
                        [((origin[0] + i) * shape[1] + origin[1] + j) * shape[2] + origin[2] + k];
                }
            }
        }
        return Ok((header, window));
    }
    validate_brick_container_layout(&header, file_len)?;
    if output_count == header.element_count {
        validate_brick_container(&mut file, &header, file_len)?;
    }
    let counts = header.brick_counts.unwrap();
    let brick_shape = header.brick_shape.unwrap();
    let first = [
        origin[0] / brick_shape[0],
        origin[1] / brick_shape[1],
        origin[2] / brick_shape[2],
    ];
    let last = [
        (origin[0] + extent[0] - 1) / brick_shape[0],
        (origin[1] + extent[1] - 1) / brick_shape[1],
        (origin[2] + extent[2] - 1) / brick_shape[2],
    ];
    let mut window = vec![0.0f32; output_count];
    for bi in first[0]..=last[0] {
        for bj in first[1]..=last[1] {
            for bk in first[2]..=last[2] {
                let index = (bi * counts[1] + bj) * counts[2] + bk;
                let entry = read_brick_entry(&mut file, &header, index, file_len)?;
                file.seek(SeekFrom::Start(entry.payload_offset))?;
                let mut compressed = vec![0u8; entry.compressed_size];
                file.read_exact(&mut compressed)?;
                verify_brick_bytes(&entry, &compressed)?;
                let brick = decompress_payload_dims(&header, &compressed, entry.extent)?;
                let begin = [
                    origin[0].max(entry.origin[0]),
                    origin[1].max(entry.origin[1]),
                    origin[2].max(entry.origin[2]),
                ];
                let end = [
                    (origin[0] + extent[0]).min(entry.origin[0] + entry.extent[0]),
                    (origin[1] + extent[1]).min(entry.origin[1] + entry.extent[1]),
                    (origin[2] + extent[2]).min(entry.origin[2] + entry.extent[2]),
                ];
                for gi in begin[0]..end[0] {
                    for gj in begin[1]..end[1] {
                        for gk in begin[2]..end[2] {
                            let source = ((gi - entry.origin[0]) * entry.extent[1] + gj
                                - entry.origin[1])
                                * entry.extent[2]
                                + gk
                                - entry.origin[2];
                            let destination =
                                ((gi - origin[0]) * extent[1] + gj - origin[1]) * extent[2] + gk
                                    - origin[2];
                            window[destination] = brick[source];
                        }
                    }
                }
            }
        }
    }
    Ok((header, window))
}

/// Read + decode one `.zfp3d` file. Returns RAW values (residual for deltas).
pub fn read_zfp3d_frame(path: &Path) -> AppResult<(Zfp3dHeader, Vec<f32>)> {
    let mut header_file = std::fs::File::open(path)?;
    let (header_bytes, header_len) = read_header_prefix(&mut header_file)?;
    let (header, _) = parse_header(&header_bytes[..header_len])?;
    if header.is_bricked() {
        crate::npy::validate_field_shape(&header.shape())?;
        return read_zfp3d_raw_window(
            path,
            [0, 0, 0],
            [
                header.dim0 as usize,
                header.dim1 as usize,
                header.dim2 as usize,
            ],
        );
    }
    let file_len = usize::try_from(std::fs::metadata(path)?.len())
        .map_err(|_| AppError::Invalid(".zfp3d file length does not fit usize".into()))?;
    if file_len > MAX_LEGACY_ZFP_FILE_BYTES {
        return Err(AppError::Invalid(format!(
            ".zfp3d file is {file_len} bytes; limit is {MAX_LEGACY_ZFP_FILE_BYTES}"
        )));
    }
    let bytes = std::fs::read(path)?;
    decode_frame_bytes(&bytes)
}

/// Peek only the header of a `.zfp3d` file (cheap; used while walking back to a
/// keyframe).
fn read_header_only(path: &Path) -> AppResult<Zfp3dHeader> {
    let mut f = std::fs::File::open(path)?;
    let (buf, n) = read_header_prefix(&mut f)?;
    let (h, _) = parse_header(&buf[..n])?;
    Ok(h)
}

/// One frame in a delta chain: its frame type and RAW decoded values.
struct Frame {
    flags: u8,
    data: Vec<f32>,
}

/// Apply one decoded frame to the two-frame reconstruction history. Mirrors
/// `ZfpFrameReader`
/// (`zfp_reader.cpp`): simple delta = `residual + prev`; predictive delta =
/// `residual + 2*prev - prev2`; the first delta after a keyframe falls back to
/// simple (prev2 is cleared at every keyframe); history rotates
/// `prev2 = prev; prev = reconstructed`.
fn apply_frame(frame: Frame, prev: &mut Vec<f32>, prev2: &mut Vec<f32>) -> AppResult<()> {
    if frame.flags == FLAG_KEYFRAME {
        prev2.clear();
        *prev = frame.data;
        return Ok(());
    }
    if prev.is_empty() {
        return Err(AppError::Invalid(
            "delta .zfp3d frame has no preceding keyframe".into(),
        ));
    }
    if frame.data.len() != prev.len() {
        return Err(AppError::Invalid(
            "delta .zfp3d frame size differs from its reference".into(),
        ));
    }
    let use_predictive = frame.flags == FLAG_PREDICTIVE_DELTA && !prev2.is_empty();
    let mut recon = frame.data;
    if use_predictive {
        for i in 0..recon.len() {
            recon[i] = (recon[i] as f64 + 2.0 * prev[i] as f64 - prev2[i] as f64) as f32;
        }
    } else {
        for i in 0..recon.len() {
            recon[i] = (recon[i] as f64 + prev[i] as f64) as f32;
        }
    }
    *prev2 = std::mem::replace(prev, recon);
    Ok(())
}

#[cfg(test)]
fn replay(frames: Vec<Frame>) -> AppResult<Vec<f32>> {
    let mut prev: Vec<f32> = Vec::new();
    let mut prev2: Vec<f32> = Vec::new();
    for frame in frames {
        apply_frame(frame, &mut prev, &mut prev2)?;
    }
    if prev.is_empty() {
        return Err(AppError::Invalid("no .zfp3d frames to replay".into()));
    }
    Ok(prev)
}

fn validate_field_name(field: &str) -> AppResult<()> {
    if field.is_empty() || field.contains('/') || field.contains('\\') || field.contains("..") {
        return Err(AppError::Invalid(format!("invalid field name: {field}")));
    }
    Ok(())
}

fn zfp3d_path(run_dir: &str, step_index: u32, field: &str) -> PathBuf {
    Path::new(run_dir)
        .join(format!("step_{:06}", step_index))
        .join(format!("{field}.zfp3d"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ManifestFrameRole {
    Keyframe,
    SimpleDelta,
    PredictiveDelta,
}

#[derive(Clone, Debug)]
struct ManifestFrameContract {
    role: ManifestFrameRole,
    parents: Vec<u32>,
    container_version: Option<u16>,
    brick_shape: Option<[usize; 3]>,
    reversible_v4: bool,
}

fn manifest_frame_contract(
    run_dir: &str,
    step_index: u32,
    field: &str,
) -> AppResult<Option<ManifestFrameContract>> {
    const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
    let path = Path::new(run_dir)
        .join(format!("step_{step_index:06}"))
        .join("manifest.json");
    let metadata = match std::fs::metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    if !metadata.is_file() || metadata.len() > MAX_MANIFEST_BYTES {
        return Err(AppError::Invalid(format!(
            "invalid or oversized ZFP parent manifest: {}",
            path.display()
        )));
    }
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)
        .map_err(|error| AppError::Invalid(format!("invalid ZFP parent manifest: {error}")))?;
    let legacy_contract = manifest
        .get("data_product")
        .and_then(|product| product.get("lossy_compression_contract"))
        .and_then(|value| value.as_str())
        == Some("nimbus.zfp.per_field.v1");
    let mixed_contract = manifest
        .get("data_product")
        .and_then(|product| product.get("compression_contract"))
        .and_then(|value| value.as_str())
        == Some("nimbus.zfp.per_field.v2");
    if !legacy_contract && !mixed_contract {
        return Ok(None);
    }
    let entry = manifest
        .get("fields")
        .and_then(|value| value.as_array())
        .and_then(|fields| {
            fields
                .iter()
                .find(|entry| entry.get("field_id").and_then(|value| value.as_str()) == Some(field))
        })
        .ok_or_else(|| {
            AppError::Invalid(format!(
                "ZFP manifest at step {step_index} does not declare field '{field}'"
            ))
        })?;
    let compression = entry.get("compression").ok_or_else(|| {
        AppError::Invalid(format!(
            "ZFP manifest at step {step_index} omits compression metadata for '{field}'"
        ))
    })?;
    if compression.get("codec").and_then(|value| value.as_str()) != Some("zfp")
        || compression
            .get("decoded_dtype")
            .and_then(|value| value.as_str())
            != Some("float32")
    {
        return Err(AppError::Invalid(format!(
            "ZFP manifest at step {step_index} has an incompatible codec/dtype contract for '{field}'"
        )));
    }
    let role = match compression
        .get("frame_role")
        .and_then(|value| value.as_str())
    {
        Some("keyframe") => ManifestFrameRole::Keyframe,
        Some("simple_delta") => ManifestFrameRole::SimpleDelta,
        Some("predictive_delta") => ManifestFrameRole::PredictiveDelta,
        _ => {
            return Err(AppError::Invalid(format!(
                "ZFP manifest at step {step_index} has an invalid frame role for '{field}'"
            )))
        }
    };
    let parents = compression
        .get("parent_step_indices")
        .and_then(|value| value.as_array())
        .ok_or_else(|| {
            AppError::Invalid(format!(
                "ZFP manifest at step {step_index} omits parent indices for '{field}'"
            ))
        })?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| AppError::Invalid("invalid ZFP parent step index".into()))
        })
        .collect::<AppResult<Vec<_>>>()?;
    let expected_parents = match role {
        ManifestFrameRole::Keyframe => 0,
        ManifestFrameRole::SimpleDelta => 1,
        ManifestFrameRole::PredictiveDelta => 2,
    };
    if parents.len() != expected_parents || parents.iter().any(|parent| *parent >= step_index) {
        return Err(AppError::Invalid(format!(
            "ZFP manifest at step {step_index} has an invalid parent chain for '{field}'"
        )));
    }
    let layout_fields_present = [
        compression.get("container_version").is_some(),
        compression.get("spatial_layout").is_some(),
        compression.get("brick_shape").is_some(),
    ];
    let (container_version, brick_shape) = if layout_fields_present.iter().all(|present| !present) {
        (None, None)
    } else if layout_fields_present.iter().all(|present| *present) {
        let version = compression
            .get("container_version")
            .and_then(|value| value.as_u64())
            .and_then(|value| u16::try_from(value).ok())
            .ok_or_else(|| AppError::Invalid("invalid ZFP container version".into()))?;
        if compression
            .get("spatial_layout")
            .and_then(|value| value.as_str())
            != Some("independently_decodable_bricks")
        {
            return Err(AppError::Invalid("invalid ZFP spatial layout".into()));
        }
        let shape = compression
            .get("brick_shape")
            .and_then(|value| value.as_array())
            .filter(|values| values.len() == 3)
            .ok_or_else(|| AppError::Invalid("invalid ZFP manifest brick shape".into()))?;
        let mut parsed = [0usize; 3];
        for axis in 0..3 {
            parsed[axis] = shape[axis]
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| AppError::Invalid("invalid ZFP manifest brick shape".into()))?;
        }
        (Some(version), Some(parsed))
    } else {
        return Err(AppError::Invalid(
            "incomplete ZFP spatial-layout provenance".into(),
        ));
    };
    let contract_version = match compression.get("contract_version") {
        None => None,
        Some(value) => Some(
            value
                .as_u64()
                .ok_or_else(|| AppError::Invalid("invalid ZFP field contract version".into()))?,
        ),
    };
    if !matches!(contract_version, None | Some(1) | Some(2)) {
        return Err(AppError::Invalid(
            "unsupported ZFP field contract version".into(),
        ));
    }
    let reversible_v4 = contract_version == Some(2);
    if reversible_v4 {
        let exact = mixed_contract
            && compression.get("mode").and_then(|value| value.as_str()) == Some("reversible")
            && compression
                .get("mode_parameter")
                .and_then(|value| value.as_f64())
                == Some(0.0)
            && compression
                .get("mode_parameter_units")
                .and_then(|value| value.as_str())
                == Some("none")
            && compression
                .get("error_contract")
                .and_then(|value| value.as_str())
                == Some("bitwise_float32_reconstruction")
            && compression
                .get("sparse_threshold")
                .and_then(|value| value.as_f64())
                == Some(0.0)
            && compression
                .get("float16_prefilter")
                .and_then(|value| value.as_bool())
                == Some(false)
            && container_version == Some(4)
            && brick_shape == Some(ZFP_V3_BRICK_SHAPE)
            && role == ManifestFrameRole::Keyframe
            && parents.is_empty();
        if !exact {
            return Err(AppError::Invalid(
                "invalid v4 reversible ZFP manifest contract".into(),
            ));
        }
    } else if container_version == Some(4) {
        return Err(AppError::Invalid(
            "v4 ZFP manifest lacks its reversible contract".into(),
        ));
    }
    Ok(Some(ManifestFrameContract {
        role,
        parents,
        container_version,
        brick_shape,
        reversible_v4,
    }))
}

fn require_v4_manifest(
    header: &Zfp3dHeader,
    contract: &Option<ManifestFrameContract>,
) -> AppResult<()> {
    if header.version == 4 && !contract.as_ref().is_some_and(|item| item.reversible_v4) {
        return Err(AppError::Invalid(
            "v4 reversible ZFP requires an exact manifest contract".into(),
        ));
    }
    Ok(())
}

fn header_matches_manifest(header: &Zfp3dHeader, contract: &ManifestFrameContract) -> bool {
    let role_matches = match contract.role {
        ManifestFrameRole::Keyframe => header.flags == FLAG_KEYFRAME,
        ManifestFrameRole::SimpleDelta => header.flags == FLAG_SIMPLE_DELTA,
        ManifestFrameRole::PredictiveDelta => header.flags == FLAG_PREDICTIVE_DELTA,
    };
    role_matches
        && contract.reversible_v4 == (header.version == 4)
        && contract
            .container_version
            .map_or(true, |version| version == header.version)
        && contract
            .brick_shape
            .map_or(true, |shape| header.brick_shape == Some(shape))
}

fn replay_steps_from_contract(
    run_dir: &str,
    step_index: u32,
    field: &str,
    enforce_full_payload_budget: bool,
) -> AppResult<Vec<u32>> {
    let mut reverse_chain = Vec::new();
    let mut current = step_index;
    loop {
        if reverse_chain.len() >= MAX_ZFP_CHAIN_FRAMES {
            return Err(AppError::Invalid(format!(
                ".zfp3d delta chain exceeds {MAX_ZFP_CHAIN_FRAMES} frames"
            )));
        }
        let contract = manifest_frame_contract(run_dir, current, field)?.ok_or_else(|| {
            AppError::Invalid(format!(
                "ZFP exact-parent contract is missing at step {current} for '{field}'"
            ))
        })?;
        let path = zfp3d_path(run_dir, current, field);
        let header = read_header_only(&path).map_err(|error| {
            AppError::Invalid(format!(
                "cannot read declared ZFP parent at step {current} for '{field}': {error}"
            ))
        })?;
        if !header_matches_manifest(&header, &contract) {
            return Err(AppError::Invalid(format!(
                "ZFP artifact/header role disagrees with manifest at step {current} for '{field}'"
            )));
        }
        reverse_chain.push((current, contract.clone()));
        if contract.role == ManifestFrameRole::Keyframe {
            break;
        }
        current = contract.parents[0];
    }
    reverse_chain.reverse();
    for index in 0..reverse_chain.len() {
        let (step, contract) = &reverse_chain[index];
        match contract.role {
            ManifestFrameRole::Keyframe if index != 0 => {
                return Err(AppError::Invalid(
                    "ZFP chain contains an interior keyframe".into(),
                ));
            }
            ManifestFrameRole::SimpleDelta => {
                if index == 0 || contract.parents[0] != reverse_chain[index - 1].0 {
                    return Err(AppError::Invalid(format!(
                        "ZFP simple-delta parent mismatch at step {step}"
                    )));
                }
            }
            ManifestFrameRole::PredictiveDelta => {
                if index < 2
                    || contract.parents[0] != reverse_chain[index - 1].0
                    || contract.parents[1] != reverse_chain[index - 2].0
                {
                    return Err(AppError::Invalid(format!(
                        "ZFP predictive-delta parent mismatch at step {step}"
                    )));
                }
            }
            ManifestFrameRole::Keyframe => {}
        }
    }
    if enforce_full_payload_budget {
        let mut cumulative = 0usize;
        for (step, _) in &reverse_chain {
            cumulative = cumulative
                .checked_add(read_header_only(&zfp3d_path(run_dir, *step, field))?.compressed_size)
                .ok_or_else(|| AppError::Invalid(".zfp3d chain byte count overflow".into()))?;
            if cumulative > MAX_ZFP_CHAIN_COMPRESSED_BYTES {
                return Err(AppError::Invalid(format!(
                    ".zfp3d delta chain is {cumulative} compressed bytes; limit is {MAX_ZFP_CHAIN_COMPRESSED_BYTES}"
                )));
            }
        }
    }
    Ok(reverse_chain.into_iter().map(|(step, _)| step).collect())
}

/// Discover the actual exported step directories through `step_index`, then
/// return the most recent keyframe-to-target suffix. Simulation step numbers
/// are commonly sparse (for example 0, 30, 60), so iterating every integer
/// between filenames would incorrectly require nonexistent frames.
fn replay_steps(
    run_dir: &str,
    step_index: u32,
    field: &str,
    enforce_full_payload_budget: bool,
) -> AppResult<Vec<u32>> {
    let mut steps = Vec::new();
    for entry in std::fs::read_dir(run_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some(raw_step) = name.strip_prefix("step_") else {
            continue;
        };
        let Ok(step) = raw_step.parse::<u32>() else {
            continue;
        };
        if step <= step_index && zfp3d_path(run_dir, step, field).is_file() {
            steps.push(step);
        }
    }
    steps.sort_unstable();
    steps.dedup();
    if steps.last().copied() != Some(step_index) {
        return Err(AppError::NotFound(
            zfp3d_path(run_dir, step_index, field).display().to_string(),
        ));
    }
    for index in (0..steps.len()).rev() {
        if read_header_only(&zfp3d_path(run_dir, steps[index], field))?.is_keyframe() {
            let chain = steps.split_off(index);
            if chain.len() > MAX_ZFP_CHAIN_FRAMES {
                return Err(AppError::Invalid(format!(
                    ".zfp3d delta chain has {} frames; limit is {MAX_ZFP_CHAIN_FRAMES}",
                    chain.len()
                )));
            }
            if enforce_full_payload_budget {
                let mut cumulative = 0usize;
                for step in &chain {
                    let header = read_header_only(&zfp3d_path(run_dir, *step, field))?;
                    cumulative =
                        cumulative
                            .checked_add(header.compressed_size)
                            .ok_or_else(|| {
                                AppError::Invalid(".zfp3d chain byte count overflow".into())
                            })?;
                    if cumulative > MAX_ZFP_CHAIN_COMPRESSED_BYTES {
                        return Err(AppError::Invalid(format!(
                            ".zfp3d delta chain is {cumulative} compressed bytes; limit is {MAX_ZFP_CHAIN_COMPRESSED_BYTES}"
                        )));
                    }
                }
            }
            return Ok(chain);
        }
    }
    Err(AppError::Invalid(format!(
        "delta .zfp3d frame for '{field}' at step {step_index} has no preceding keyframe"
    )))
}

/// Reconstruct a `.zfp3d` field at an arbitrary step into a C-order f32 volume.
/// Returns `([dim0, dim1, dim2], data)`. Keyframe frames decode directly; delta
/// frames replay from the most recent keyframe (random-access scrubbing).
pub fn field_shape_and_reconstruction_bytes(
    run_dir: &str,
    step_index: u32,
    field: &str,
) -> AppResult<(Vec<usize>, usize)> {
    validate_field_name(field)?;
    let path = zfp3d_path(run_dir, step_index, field);
    if !path.exists() {
        return Err(AppError::NotFound(path.display().to_string()));
    }
    let header = read_header_only(&path)?;
    crate::npy::validate_field_shape(&header.shape())?;
    let contract = manifest_frame_contract(run_dir, step_index, field)?;
    require_v4_manifest(&header, &contract)?;
    if let Some(contract) = &contract {
        if !header_matches_manifest(&header, contract) {
            return Err(AppError::Invalid(
                "ZFP target artifact/header role disagrees with its manifest".into(),
            ));
        }
    }
    let shape = header.shape();
    let decoded_bytes = header
        .element_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| AppError::Invalid("ZFP decoded working-set size overflow".into()))?;
    if header.version == 4 {
        let shape_array = [
            header.dim0 as usize,
            header.dim1 as usize,
            header.dim2 as usize,
        ];
        let (_, max_compressed, inspected, max_brick_elements) =
            bricked_window_frame_stats(&path, [0, 0, 0], shape_array)?;
        let working_bytes = decoded_bytes
            .checked_add(
                max_brick_elements
                    .checked_mul(4)
                    .ok_or_else(|| AppError::Invalid("v4 brick working-set overflow".into()))?,
            )
            .and_then(|value| value.checked_add(max_compressed.checked_mul(2)?))
            .and_then(|value| value.checked_add(64 * 1024))
            .ok_or_else(|| AppError::Invalid("v4 reconstruction working-set overflow".into()))?;
        return Ok((shape, working_bytes.max(inspected)));
    }
    let chain = if header.is_keyframe() {
        vec![step_index]
    } else if contract.is_some() {
        replay_steps_from_contract(run_dir, step_index, field, true)?
    } else {
        replay_steps(run_dir, step_index, field, true)?
    };
    let mut max_frame_bytes = 0_usize;
    for step in chain {
        let chain_header = read_header_only(&zfp3d_path(run_dir, step, field))?;
        if chain_header.shape() != shape {
            return Err(AppError::Invalid(
                "delta .zfp3d frame shape differs from its target".into(),
            ));
        }
        max_frame_bytes = max_frame_bytes.max(chain_header.compressed_size);
    }
    // A predictive replay can simultaneously hold prev, prev2, and the frame
    // being reconstructed. read_zfp3d_frame also owns the file payload and the
    // mutable libzfp payload copy during decompression.
    let decoded_volumes = if header.is_keyframe() {
        1_usize
    } else {
        3_usize
    };
    let working_bytes = decoded_bytes
        .checked_mul(decoded_volumes)
        .and_then(|value| value.checked_add(max_frame_bytes.checked_mul(2)?))
        .ok_or_else(|| AppError::Invalid("ZFP reconstruction working-set size overflow".into()))?;
    Ok((shape, working_bytes))
}

fn bricked_window_frame_stats(
    path: &Path,
    origin: [usize; 3],
    extent: [usize; 3],
) -> AppResult<(Zfp3dHeader, usize, usize, usize)> {
    let mut file = std::fs::File::open(path)?;
    let file_len = file.metadata()?.len();
    let (bytes, count) = read_header_prefix(&mut file)?;
    let (header, _) = parse_header(&bytes[..count])?;
    validate_window(&header.shape(), origin, extent)?;
    if !header.is_bricked() {
        return Err(AppError::Invalid(
            "window stats require a v3 bricked frame".into(),
        ));
    }
    validate_brick_container_layout(&header, file_len)?;
    let counts = header.brick_counts.unwrap();
    let brick_shape = header.brick_shape.unwrap();
    let first = [
        origin[0] / brick_shape[0],
        origin[1] / brick_shape[1],
        origin[2] / brick_shape[2],
    ];
    let last = [
        (origin[0] + extent[0] - 1) / brick_shape[0],
        (origin[1] + extent[1] - 1) / brick_shape[1],
        (origin[2] + extent[2] - 1) / brick_shape[2],
    ];
    let mut max_compressed = 0usize;
    let mut total_compressed = 0usize;
    let mut max_brick_elements = 0usize;
    let mut intersecting_bricks = 0usize;
    for bi in first[0]..=last[0] {
        for bj in first[1]..=last[1] {
            for bk in first[2]..=last[2] {
                let index = (bi * counts[1] + bj) * counts[2] + bk;
                let entry = read_brick_entry(&mut file, &header, index, file_len)?;
                intersecting_bricks = intersecting_bricks
                    .checked_add(1)
                    .ok_or_else(|| AppError::Invalid("window brick count overflow".into()))?;
                max_compressed = max_compressed.max(entry.compressed_size);
                total_compressed = total_compressed
                    .checked_add(entry.compressed_size)
                    .ok_or_else(|| {
                        AppError::Invalid("window compressed-byte sum overflow".into())
                    })?;
                max_brick_elements =
                    max_brick_elements.max(checked_product(entry.extent, "ZFP brick")?);
            }
        }
    }
    let directory_bytes = intersecting_bricks
        .checked_mul(2 * header.brick_entry_bytes.unwrap_or(BRICK_ENTRY_LEN))
        .and_then(|value| value.checked_add(HEADER_V3_LEN))
        .ok_or_else(|| AppError::Invalid("ZFP directory byte count overflow".into()))?;
    total_compressed = total_compressed
        .checked_add(directory_bytes)
        .ok_or_else(|| AppError::Invalid("ZFP inspected-byte count overflow".into()))?;
    Ok((header, max_compressed, total_compressed, max_brick_elements))
}

pub fn field_shape_and_window_reconstruction_bytes(
    run_dir: &str,
    step_index: u32,
    field: &str,
    origin: [usize; 3],
    extent: [usize; 3],
) -> AppResult<(Vec<usize>, usize)> {
    validate_field_name(field)?;
    let target_path = zfp3d_path(run_dir, step_index, field);
    let target_header = read_header_only(&target_path)?;
    if !target_header.is_bricked() {
        let count = validate_window(&target_header.shape(), origin, extent)?;
        let (shape, bytes) = field_shape_and_reconstruction_bytes(run_dir, step_index, field)?;
        let extra = if count == target_header.element_count {
            0
        } else {
            count
                .checked_mul(4)
                .ok_or_else(|| AppError::Invalid("ZFP window size overflow".into()))?
        };
        return Ok((
            shape,
            bytes
                .checked_add(extra)
                .ok_or_else(|| AppError::Invalid("ZFP window working-set overflow".into()))?,
        ));
    }
    validate_window(&target_header.shape(), origin, extent)?;
    let target_contract = manifest_frame_contract(run_dir, step_index, field)?;
    require_v4_manifest(&target_header, &target_contract)?;
    if let Some(contract) = &target_contract {
        if !header_matches_manifest(&target_header, contract) {
            return Err(AppError::Invalid(
                "ZFP target artifact/header role disagrees with its manifest".into(),
            ));
        }
    }
    let chain = if target_header.is_keyframe() {
        vec![step_index]
    } else if target_contract.is_some() {
        replay_steps_from_contract(run_dir, step_index, field, false)?
    } else {
        replay_steps(run_dir, step_index, field, false)?
    };
    let mut max_compressed = 0usize;
    let mut total_inspected = 0usize;
    let mut max_brick_elements = 0usize;
    for step in &chain {
        let (header, frame_max, frame_inspected, frame_brick_elements) =
            bricked_window_frame_stats(&zfp3d_path(run_dir, *step, field), origin, extent)?;
        if header.shape() != target_header.shape()
            || header.brick_shape != target_header.brick_shape
            || header.brick_counts != target_header.brick_counts
        {
            return Err(AppError::Invalid(
                "delta .zfp3d brick geometry differs from its target".into(),
            ));
        }
        max_compressed = max_compressed.max(frame_max);
        max_brick_elements = max_brick_elements.max(frame_brick_elements);
        total_inspected = total_inspected
            .checked_add(frame_inspected)
            .ok_or_else(|| AppError::Invalid("ZFP window inspected-byte sum overflow".into()))?;
    }
    let window_bytes = checked_product(extent, "ZFP window")?
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| AppError::Invalid("ZFP window decoded-byte count overflow".into()))?;
    let decoded_volumes = if chain.len() == 1 { 1usize } else { 3usize };
    let working_bytes = window_bytes
        .checked_mul(decoded_volumes)
        .and_then(|value| value.checked_add(max_brick_elements.checked_mul(4)?))
        .and_then(|value| value.checked_add(max_compressed.checked_mul(2)?))
        .and_then(|value| {
            value.checked_add(if target_header.version == 4 {
                64 * 1024
            } else {
                0
            })
        })
        .ok_or_else(|| AppError::Invalid("ZFP window working-set size overflow".into()))?;
    Ok((target_header.shape(), working_bytes.max(total_inspected)))
}

pub fn reconstruct_field_window(
    run_dir: &str,
    step_index: u32,
    field: &str,
    origin: [usize; 3],
    extent: [usize; 3],
) -> AppResult<(Vec<usize>, Vec<f32>)> {
    validate_field_name(field)?;
    let target_path = zfp3d_path(run_dir, step_index, field);
    let target_header = read_header_only(&target_path)?;
    validate_window(&target_header.shape(), origin, extent)?;
    let target_contract = manifest_frame_contract(run_dir, step_index, field)?;
    require_v4_manifest(&target_header, &target_contract)?;
    if let Some(contract) = &target_contract {
        if !header_matches_manifest(&target_header, contract) {
            return Err(AppError::Invalid(
                "ZFP target artifact/header role disagrees with its manifest".into(),
            ));
        }
    }
    let chain = if target_header.is_keyframe() {
        vec![step_index]
    } else if target_contract.is_some() {
        replay_steps_from_contract(run_dir, step_index, field, false)?
    } else {
        replay_steps(run_dir, step_index, field, false)?
    };
    let mut prev = Vec::new();
    let mut prev2 = Vec::new();
    for step in chain {
        let (header, data) =
            read_zfp3d_raw_window(&zfp3d_path(run_dir, step, field), origin, extent)?;
        if header.shape() != target_header.shape()
            || header.brick_shape != target_header.brick_shape
            || header.brick_counts != target_header.brick_counts
        {
            return Err(AppError::Invalid(
                "delta .zfp3d brick geometry differs from its target".into(),
            ));
        }
        apply_frame(
            Frame {
                flags: header.flags,
                data,
            },
            &mut prev,
            &mut prev2,
        )?;
    }
    Ok((target_header.shape(), prev))
}

pub fn visit_reconstructed_field_bricks<F>(
    run_dir: &str,
    step_index: u32,
    field: &str,
    mut visitor: F,
) -> AppResult<Vec<usize>>
where
    F: FnMut([usize; 3], [usize; 3], &[f32]) -> AppResult<()>,
{
    validate_field_name(field)?;
    let target_path = zfp3d_path(run_dir, step_index, field);
    let target_header = read_header_only(&target_path)?;
    if !target_header.is_bricked() {
        let (shape, values) = reconstruct_field(run_dir, step_index, field)?;
        visitor([0, 0, 0], [shape[0], shape[1], shape[2]], &values)?;
        return Ok(shape);
    }
    let target_contract = manifest_frame_contract(run_dir, step_index, field)?;
    require_v4_manifest(&target_header, &target_contract)?;
    if let Some(contract) = &target_contract {
        if !header_matches_manifest(&target_header, contract) {
            return Err(AppError::Invalid(
                "ZFP target artifact/header role disagrees with its manifest".into(),
            ));
        }
    }
    let chain = if target_header.is_keyframe() {
        vec![step_index]
    } else if target_contract.is_some() {
        replay_steps_from_contract(run_dir, step_index, field, false)?
    } else {
        replay_steps(run_dir, step_index, field, false)?
    };
    let shape = target_header.shape();
    let brick_shape = target_header
        .brick_shape
        .ok_or_else(|| AppError::Invalid("missing v3 brick shape".into()))?;
    let counts = target_header
        .brick_counts
        .ok_or_else(|| AppError::Invalid("missing v3 brick counts".into()))?;
    if target_header.version == 4 {
        let mut file = std::fs::File::open(&target_path)?;
        let file_len = file.metadata()?.len();
        validate_brick_container(&mut file, &target_header, file_len)?;
        for bi in 0..counts[0] {
            for bj in 0..counts[1] {
                for bk in 0..counts[2] {
                    let origin = [
                        bi * brick_shape[0],
                        bj * brick_shape[1],
                        bk * brick_shape[2],
                    ];
                    let extent = [
                        brick_shape[0].min(shape[0] - origin[0]),
                        brick_shape[1].min(shape[1] - origin[1]),
                        brick_shape[2].min(shape[2] - origin[2]),
                    ];
                    read_zfp3d_raw_window(&target_path, origin, extent)?;
                }
            }
        }
    }
    for bi in 0..counts[0] {
        for bj in 0..counts[1] {
            for bk in 0..counts[2] {
                let origin = [
                    bi * brick_shape[0],
                    bj * brick_shape[1],
                    bk * brick_shape[2],
                ];
                let extent = [
                    brick_shape[0].min(shape[0] - origin[0]),
                    brick_shape[1].min(shape[1] - origin[1]),
                    brick_shape[2].min(shape[2] - origin[2]),
                ];
                let mut prev = Vec::new();
                let mut prev2 = Vec::new();
                for step in &chain {
                    let (header, data) =
                        read_zfp3d_raw_window(&zfp3d_path(run_dir, *step, field), origin, extent)?;
                    if header.shape() != shape
                        || header.brick_shape != target_header.brick_shape
                        || header.brick_counts != target_header.brick_counts
                    {
                        return Err(AppError::Invalid(
                            "delta .zfp3d brick geometry differs from its target".into(),
                        ));
                    }
                    apply_frame(
                        Frame {
                            flags: header.flags,
                            data,
                        },
                        &mut prev,
                        &mut prev2,
                    )?;
                }
                visitor(origin, extent, &prev)?;
            }
        }
    }
    Ok(shape)
}

pub fn reconstruct_field(
    run_dir: &str,
    step_index: u32,
    field: &str,
) -> AppResult<(Vec<usize>, Vec<f32>)> {
    validate_field_name(field)?;
    let target_path = zfp3d_path(run_dir, step_index, field);
    if !target_path.exists() {
        return Err(AppError::NotFound(target_path.display().to_string()));
    }

    let target_hdr = read_header_only(&target_path)?;
    let target_contract = manifest_frame_contract(run_dir, step_index, field)?;
    require_v4_manifest(&target_hdr, &target_contract)?;
    if let Some(contract) = &target_contract {
        if !header_matches_manifest(&target_hdr, contract) {
            return Err(AppError::Invalid(
                "ZFP target artifact/header role disagrees with its manifest".into(),
            ));
        }
    }
    let target_shape = target_hdr.shape();
    crate::npy::validate_field_shape(&target_shape)?;
    if target_hdr.is_keyframe() {
        let (_, target_raw) = read_zfp3d_frame(&target_path)?;
        return Ok((target_shape, target_raw));
    }

    // Delta frame: decode and apply keyframe..=target incrementally. Only the
    // two reconstruction-history fields plus the currently decoded residual
    // are live, so peak decoded memory is constant with keyframe interval.
    let chain = if target_contract.is_some() {
        replay_steps_from_contract(run_dir, step_index, field, true)?
    } else {
        replay_steps(run_dir, step_index, field, true)?
    };
    let mut prev: Vec<f32> = Vec::new();
    let mut prev2: Vec<f32> = Vec::new();
    for s in chain {
        let (h, d) = read_zfp3d_frame(&zfp3d_path(run_dir, s, field))?;
        if h.shape() != target_shape {
            return Err(AppError::Invalid(
                "delta .zfp3d frame shape differs from its target".into(),
            ));
        }
        apply_frame(
            Frame {
                flags: h.flags,
                data: d,
            },
            &mut prev,
            &mut prev2,
        )?;
    }
    Ok((target_shape, prev))
}

/// Re-serialize a decoded volume as a NumPy v1.0 `<f4` C-order byte stream,
/// byte-for-byte the format `crate::npy::parse_f32_layout` (and the TS reader)
/// expect. This is what lets a `.zfp3d` field flow through the unchanged
/// `.npy` path. The 64-byte header alignment recipe matches `npy::make_npy`.
pub fn encode_npy_f32(shape: &[usize], data: &[f32]) -> AppResult<Vec<u8>> {
    const NPY_MAGIC: [u8; 6] = [0x93, b'N', b'U', b'M', b'P', b'Y'];
    let validated = crate::npy::validate_field_shape(shape)?;
    if data.len() != validated.element_count {
        return Err(AppError::Invalid(format!(
            "decoded field has {} values but shape requires {}",
            data.len(),
            validated.element_count
        )));
    }
    let shape_str = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({shape_str}), }}");
    let mut header = dict.into_bytes();
    header.push(b'\n');
    while (10 + header.len()) % 64 != 0 {
        header.insert(header.len() - 1, b' ');
    }

    let capacity = 10usize
        .checked_add(header.len())
        .and_then(|size| size.checked_add(validated.decoded_bytes))
        .ok_or_else(|| AppError::Invalid("encoded NPY size overflow".into()))?;
    let mut out = Vec::with_capacity(capacity);
    out.extend_from_slice(&NPY_MAGIC);
    out.push(1);
    out.push(0);
    out.push((header.len() & 0xff) as u8);
    out.push(((header.len() >> 8) & 0xff) as u8);
    out.extend_from_slice(&header);
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::ffi::c_void;

    /// Compress an f32 volume into a libzfp accuracy-mode payload, with the same
    /// reversed-dim field construction as the writer.
    fn zfp_compress_payload(shape: &[usize], data: &[f32], tolerance: f64) -> Vec<u8> {
        use zfp_sys::*;
        let mut src = data.to_vec();
        unsafe {
            let field = zfp_field_3d(
                src.as_mut_ptr() as *mut c_void,
                zfp_type_zfp_type_float,
                shape[2],
                shape[1],
                shape[0],
            );
            let zfp = zfp_stream_open(std::ptr::null_mut());
            zfp_stream_set_accuracy(zfp, tolerance);
            let bufsize = zfp_stream_maximum_size(zfp, field);
            let mut buffer = vec![0u8; bufsize];
            let stream = stream_open(buffer.as_mut_ptr() as *mut c_void, bufsize);
            zfp_stream_set_bit_stream(zfp, stream);
            zfp_stream_rewind(zfp);
            let written = zfp_compress(zfp, field);
            zfp_field_free(field);
            zfp_stream_close(zfp);
            stream_close(stream);
            assert!(written > 0, "zfp_compress failed in test helper");
            buffer.truncate(written);
            buffer
        }
    }

    fn zfp_compress_reversible_payload(shape: &[usize], data: &[f32]) -> Vec<u8> {
        use zfp_sys::*;
        let mut src = data.to_vec();
        unsafe {
            let field = zfp_field_3d(
                src.as_mut_ptr() as *mut c_void,
                zfp_type_zfp_type_float,
                shape[2],
                shape[1],
                shape[0],
            );
            let zfp = zfp_stream_open(std::ptr::null_mut());
            zfp_stream_set_reversible(zfp);
            let maximum = zfp_stream_maximum_size(zfp, field);
            let mut buffer = vec![0u8; maximum];
            let stream = stream_open(buffer.as_mut_ptr() as *mut c_void, maximum);
            zfp_stream_set_bit_stream(zfp, stream);
            zfp_stream_rewind(zfp);
            let written = zfp_compress(zfp, field);
            zfp_field_free(field);
            zfp_stream_close(zfp);
            stream_close(stream);
            assert!(written > 0 && written <= maximum);
            buffer.truncate(written);
            buffer
        }
    }

    fn make_zfp3d_v4(shape: &[usize], data: &[f32]) -> Vec<u8> {
        let brick_shape = ZFP_V3_BRICK_SHAPE;
        let counts = [
            shape[0].div_ceil(brick_shape[0]),
            shape[1].div_ceil(brick_shape[1]),
            shape[2].div_ceil(brick_shape[2]),
        ];
        let mut payloads = Vec::new();
        for bi in 0..counts[0] {
            for bj in 0..counts[1] {
                for bk in 0..counts[2] {
                    let origin = [
                        bi * brick_shape[0],
                        bj * brick_shape[1],
                        bk * brick_shape[2],
                    ];
                    let extent = [
                        brick_shape[0].min(shape[0] - origin[0]),
                        brick_shape[1].min(shape[1] - origin[1]),
                        brick_shape[2].min(shape[2] - origin[2]),
                    ];
                    let mut values = Vec::with_capacity(extent.iter().product());
                    for i in 0..extent[0] {
                        for j in 0..extent[1] {
                            for k in 0..extent[2] {
                                values.push(
                                    data[((origin[0] + i) * shape[1] + origin[1] + j) * shape[2]
                                        + origin[2]
                                        + k],
                                );
                            }
                        }
                    }
                    payloads.push(zfp_compress_reversible_payload(&extent, &values));
                }
            }
        }
        let payload_size: usize = payloads.iter().map(Vec::len).sum();
        let mut output =
            Vec::with_capacity(HEADER_V3_LEN + payload_size + payloads.len() * BRICK_ENTRY_V4_LEN);
        output.extend_from_slice(&MAGIC);
        output.extend_from_slice(&4u16.to_le_bytes());
        output.extend_from_slice(&3u16.to_le_bytes());
        for dimension in shape {
            output.extend_from_slice(&(*dimension as i32).to_le_bytes());
        }
        output.extend_from_slice(&0.0f64.to_le_bytes());
        output.extend_from_slice(&(payload_size as u64).to_le_bytes());
        output.extend_from_slice(&[FLAG_KEYFRAME, 0, 0, 0]);
        output.extend_from_slice(b"ZB4D");
        let mut offset = HEADER_V3_LEN;
        let mut directory = Vec::with_capacity(payloads.len() * BRICK_ENTRY_V4_LEN);
        for payload in payloads {
            directory.extend_from_slice(&(offset as u64).to_le_bytes());
            directory.extend_from_slice(&Sha256::digest(&payload));
            offset += payload.len();
            output.extend_from_slice(&payload);
        }
        output.extend_from_slice(&directory);
        output
    }

    fn v4_test_run(label: &str, payload: &[u8]) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let sequence = COUNTER.fetch_add(1, Ordering::Relaxed);
        let run = std::env::temp_dir().join(format!(
            "nimbus_v4_zfp_{label}_{}_{}",
            std::process::id(),
            sequence
        ));
        let step = run.join("step_000000");
        std::fs::create_dir_all(&step).unwrap();
        std::fs::write(step.join("rhohv.zfp3d"), payload).unwrap();
        run
    }

    fn write_v4_test_manifest(run: &Path, mode: &str) {
        let manifest = serde_json::json!({
            "data_product": {"compression_contract": "nimbus.zfp.per_field.v2"},
            "fields": [{
                "field_id": "rhohv",
                "compression": {
                    "contract_version": 2,
                    "codec": "zfp",
                    "mode": mode,
                    "mode_parameter": 0,
                    "mode_parameter_units": "none",
                    "error_contract": "bitwise_float32_reconstruction",
                    "container_version": 4,
                    "frame_role": "keyframe",
                    "parent_step_indices": [],
                    "sparse_threshold": 0,
                    "float16_prefilter": false,
                    "decoded_dtype": "float32",
                    "spatial_layout": "independently_decodable_bricks",
                    "brick_shape": [64,64,32]
                }
            }]
        });
        std::fs::write(
            run.join("step_000000/manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
    }

    /// Wrap a payload in the v2 `.zfp3d` header (accuracy mode).
    fn make_zfp3d(shape: &[usize], payload: &[u8], tolerance: f64, flags: u8) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&2u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // accuracy
        out.extend_from_slice(&(shape[0] as i32).to_le_bytes());
        out.extend_from_slice(&(shape[1] as i32).to_le_bytes());
        out.extend_from_slice(&(shape[2] as i32).to_le_bytes());
        out.extend_from_slice(&tolerance.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        out.push(flags);
        out.extend_from_slice(&[0u8, 0, 0]);
        out.extend_from_slice(payload);
        out
    }

    fn make_zfp3d_v3_header(shape: &[usize], compressed_size: usize, flags: u8) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_V3_LEN);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&3u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        for dimension in shape {
            out.extend_from_slice(&(*dimension as i32).to_le_bytes());
        }
        out.extend_from_slice(&1.0e-5f64.to_le_bytes());
        out.extend_from_slice(&(compressed_size as u64).to_le_bytes());
        out.push(flags);
        out.extend_from_slice(&[0u8; 3]);
        out.extend_from_slice(b"ZB3D");
        assert_eq!(out.len(), HEADER_V3_LEN);
        out
    }

    /// Build the seekable v3 container exactly as the C++ writer does: fixed
    /// 64x64x32 bricks, k-fastest directory order, and contiguous payloads.
    fn make_zfp3d_v3(shape: &[usize], data: &[f32], tolerance: f64, flags: u8) -> Vec<u8> {
        let brick_shape = ZFP_V3_BRICK_SHAPE;
        let counts = [
            shape[0].div_ceil(brick_shape[0]),
            shape[1].div_ceil(brick_shape[1]),
            shape[2].div_ceil(brick_shape[2]),
        ];
        let mut bricks = Vec::new();
        for bi in 0..counts[0] {
            for bj in 0..counts[1] {
                for bk in 0..counts[2] {
                    let origin = [
                        bi * brick_shape[0],
                        bj * brick_shape[1],
                        bk * brick_shape[2],
                    ];
                    let extent = [
                        brick_shape[0].min(shape[0] - origin[0]),
                        brick_shape[1].min(shape[1] - origin[1]),
                        brick_shape[2].min(shape[2] - origin[2]),
                    ];
                    let mut values = Vec::with_capacity(extent[0] * extent[1] * extent[2]);
                    for i in 0..extent[0] {
                        for j in 0..extent[1] {
                            for k in 0..extent[2] {
                                values.push(
                                    data[((origin[0] + i) * shape[1] + origin[1] + j) * shape[2]
                                        + origin[2]
                                        + k],
                                );
                            }
                        }
                    }
                    let payload = zfp_compress_payload(&extent, &values, tolerance);
                    bricks.push((origin, extent, payload));
                }
            }
        }

        let compressed_size: usize = bricks.iter().map(|(_, _, payload)| payload.len()).sum();
        let directory_start = HEADER_V3_LEN + compressed_size;
        let file_size = directory_start + bricks.len() * BRICK_ENTRY_LEN;
        let mut out = Vec::with_capacity(file_size);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&3u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&(shape[0] as i32).to_le_bytes());
        out.extend_from_slice(&(shape[1] as i32).to_le_bytes());
        out.extend_from_slice(&(shape[2] as i32).to_le_bytes());
        out.extend_from_slice(&tolerance.to_le_bytes());
        out.extend_from_slice(&(compressed_size as u64).to_le_bytes());
        out.push(flags);
        out.extend_from_slice(&[0u8; 3]);
        out.extend_from_slice(b"ZB3D");
        assert_eq!(out.len(), HEADER_V3_LEN);

        for (_, _, payload) in &bricks {
            out.extend_from_slice(payload);
        }
        assert_eq!(out.len(), directory_start);
        let mut payload_offset = HEADER_V3_LEN;
        for (_, _, payload) in bricks {
            out.extend_from_slice(&(payload_offset as u64).to_le_bytes());
            payload_offset += payload.len();
        }
        assert_eq!(out.len(), file_size);
        out
    }

    pub(crate) fn make_zfp3d_v3_keyframe(shape: &[usize], data: &[f32], tolerance: f64) -> Vec<u8> {
        make_zfp3d_v3(shape, data, tolerance, FLAG_KEYFRAME)
    }

    fn make_repeated_brick_zfp3d_v3(
        shape: [usize; 3],
        brick_values: &[f32],
        tolerance: f64,
    ) -> (Vec<u8>, usize) {
        let counts = [
            shape[0].div_ceil(ZFP_V3_BRICK_SHAPE[0]),
            shape[1].div_ceil(ZFP_V3_BRICK_SHAPE[1]),
            shape[2].div_ceil(ZFP_V3_BRICK_SHAPE[2]),
        ];
        let brick_count = counts.into_iter().product::<usize>();
        let extent = [
            ZFP_V3_BRICK_SHAPE[0].min(shape[0]),
            ZFP_V3_BRICK_SHAPE[1].min(shape[1]),
            ZFP_V3_BRICK_SHAPE[2].min(shape[2]),
        ];
        assert_eq!(brick_values.len(), extent.into_iter().product::<usize>());
        assert_eq!(shape[0] % extent[0], 0);
        assert_eq!(shape[1] % extent[1], 0);
        assert_eq!(shape[2] % extent[2], 0);

        let payload = zfp_compress_payload(&extent, brick_values, tolerance);
        let compressed_size = payload.len() * brick_count;
        let mut out = make_zfp3d_v3_header(&shape, compressed_size, FLAG_KEYFRAME);
        out.reserve(compressed_size + brick_count * BRICK_ENTRY_LEN);
        for _ in 0..brick_count {
            out.extend_from_slice(&payload);
        }
        let mut payload_offset = HEADER_V3_LEN;
        for _ in 0..brick_count {
            out.extend_from_slice(&(payload_offset as u64).to_le_bytes());
            payload_offset += payload.len();
        }
        (out, payload.len())
    }

    fn extract_window(
        data: &[f32],
        shape: &[usize],
        origin: [usize; 3],
        extent: [usize; 3],
    ) -> Vec<f32> {
        let mut window = Vec::with_capacity(extent[0] * extent[1] * extent[2]);
        for i in 0..extent[0] {
            for j in 0..extent[1] {
                for k in 0..extent[2] {
                    window.push(
                        data[((origin[0] + i) * shape[1] + origin[1] + j) * shape[2]
                            + origin[2]
                            + k],
                    );
                }
            }
        }
        window
    }

    fn unique_run_dir(label: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let sequence = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "nimbus_zfp3d_{label}_{}_{}",
            std::process::id(),
            sequence
        ))
    }

    fn write_v3_manifest(step_dir: &Path, role: &str, parents: &[u32]) {
        std::fs::write(
            step_dir.join("manifest.json"),
            serde_json::json!({
                "data_product": {
                    "lossy_compression_contract": "nimbus.zfp.per_field.v1"
                },
                "fields": [{
                    "field_id": "theta",
                    "compression": {
                        "codec": "zfp",
                        "decoded_dtype": "float32",
                        "frame_role": role,
                        "parent_step_indices": parents,
                        "container_version": 3,
                        "spatial_layout": "independently_decodable_bricks",
                        "brick_shape": ZFP_V3_BRICK_SHAPE
                    }
                }]
            })
            .to_string(),
        )
        .unwrap();
    }

    /// Smoothly-varying, position-distinct field so a transpose/layout bug is
    /// detectable, yet compressible by zfp. idx = ((i*d1)+j)*d2 + k.
    fn sample_field(shape: &[usize]) -> Vec<f32> {
        let (d0, d1, d2) = (shape[0], shape[1], shape[2]);
        let mut v = vec![0f32; d0 * d1 * d2];
        for i in 0..d0 {
            for j in 0..d1 {
                for k in 0..d2 {
                    v[(i * d1 + j) * d2 + k] = i as f32 + 0.1 * j as f32 + 0.01 * k as f32;
                }
            }
        }
        v
    }

    fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn parses_v2_header() {
        let bytes = make_zfp3d(&[8, 12, 16], &[1, 2, 3, 4], 1.0e-4, FLAG_KEYFRAME);
        let (h, off) = parse_header(&bytes).unwrap();
        assert_eq!(h.version, 2);
        assert_eq!(h.mode, 0);
        assert_eq!((h.dim0, h.dim1, h.dim2), (8, 12, 16));
        assert_eq!(h.element_count, 8 * 12 * 16);
        assert!((h.mode_param - 1.0e-4).abs() < 1e-12);
        assert_eq!(h.compressed_size, 4);
        assert!(h.is_keyframe());
        assert_eq!(off, HEADER_V2_LEN);
    }

    #[test]
    fn v3_large_logical_field_serves_a_real_bounded_window() {
        let shape = [8192usize, 8192, 2];
        assert!(shape.iter().product::<usize>() > crate::npy::MAX_FIELD_ELEMENTS);
        let brick_shape = [64usize, 64, 2];
        let brick_values = sample_field(&brick_shape);
        let tolerance = 1.0e-5;
        let (artifact, compressed_brick_bytes) =
            make_repeated_brick_zfp3d_v3(shape, &brick_values, tolerance);
        let (header, offset) = parse_header(&artifact).unwrap();
        assert_eq!(header.shape(), shape);
        assert_eq!(
            header.element_count,
            shape.iter().copied().product::<usize>()
        );
        assert_eq!(offset, HEADER_V3_LEN);

        let run_dir = unique_run_dir("large_logical_field");
        let step_dir = run_dir.join("step_000000");
        std::fs::create_dir_all(&step_dir).unwrap();
        let path = step_dir.join("theta.zfp3d");
        std::fs::write(&path, artifact).unwrap();
        let full_error = read_zfp3d_frame(&path).unwrap_err();
        assert!(full_error.to_string().contains("field has"));

        let origin = [8189usize, 8188, 0];
        let extent = [3usize, 4, 2];
        let (_, decoded) =
            reconstruct_field_window(run_dir.to_str().unwrap(), 0, "theta", origin, extent)
                .unwrap();
        let local_origin = [origin[0] % brick_shape[0], origin[1] % brick_shape[1], 0];
        let expected = extract_window(&brick_values, &brick_shape, local_origin, extent);
        assert_eq!(decoded.len(), extent.into_iter().product::<usize>());
        assert!(max_abs_err(&decoded, &expected) <= tolerance as f32 * 1.01);

        let (_, working_bytes) = field_shape_and_window_reconstruction_bytes(
            run_dir.to_str().unwrap(),
            0,
            "theta",
            origin,
            extent,
        )
        .unwrap();
        let decoded_window_bytes = decoded.len() * std::mem::size_of::<f32>();
        let decoded_brick_bytes = brick_values.len() * std::mem::size_of::<f32>();
        let expected_working_bytes =
            decoded_window_bytes + decoded_brick_bytes + 2 * compressed_brick_bytes;
        assert_eq!(working_bytes, expected_working_bytes);
        assert!(working_bytes < 128 * 1024);
        let mut decoded_bricks = 0;
        let sparse = read_sampled_field_window(
            run_dir.to_str().unwrap(),
            0,
            "theta",
            [0, 0, 0],
            shape,
            [4096, 4096, 1],
            512 * 1024,
            32,
            Some(shape),
            |decoding| {
                decoded_bricks += usize::from(decoding);
                Ok(())
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(sparse.output_shape, [2, 2, 2]);
        assert_eq!(decoded_bricks, 4);
        assert!(sparse.source_working_bytes < 512 * 1024);

        let _ = std::fs::remove_dir_all(run_dir);
    }

    #[test]
    fn rejects_bad_magic_and_flags() {
        let mut bytes = make_zfp3d(&[2, 2, 2], &[0u8; 4], 1e-4, FLAG_KEYFRAME);
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert!(parse_header(&bad).is_err());
        bytes[36] = 0x02; // not a valid frame flag
        assert!(parse_header(&bytes).is_err());
    }

    #[test]
    fn rejects_unbounded_headers_and_trailing_payload_bytes() {
        let mut oversized_dim = make_zfp3d(
            &[crate::npy::MAX_FIELD_DIMENSION + 1, 1, 1],
            &[1],
            1e-4,
            FLAG_KEYFRAME,
        );
        assert!(parse_header(&oversized_dim).is_err());

        oversized_dim[8..12].copy_from_slice(&1i32.to_le_bytes());
        oversized_dim[28..36].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(parse_header(&oversized_dim).is_err());

        let shape = [4usize, 4, 4];
        let data = sample_field(&shape);
        let mut valid = make_zfp3d(
            &shape,
            &zfp_compress_payload(&shape, &data, 1e-5),
            1e-5,
            FLAG_KEYFRAME,
        );
        valid.push(0);
        assert!(decode_frame_bytes(&valid).is_err());
    }

    #[test]
    fn rejects_delta_chain_over_frame_limit() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static CTR: AtomicU64 = AtomicU64::new(0);
        let n = CTR.fetch_add(1, Ordering::Relaxed);
        let run_dir = std::env::temp_dir().join(format!(
            "nimbus_zfp3d_chain_limit_{}_{}",
            std::process::id(),
            n
        ));
        for step in 0..=MAX_ZFP_CHAIN_FRAMES {
            let step_dir = run_dir.join(format!("step_{step:06}"));
            std::fs::create_dir_all(&step_dir).unwrap();
            let flags = if step == 0 {
                FLAG_KEYFRAME
            } else {
                FLAG_SIMPLE_DELTA
            };
            std::fs::write(
                step_dir.join("theta.zfp3d"),
                make_zfp3d(&[1, 1, 1], &[1], 1e-4, flags),
            )
            .unwrap();
        }
        assert!(replay_steps(
            run_dir.to_str().unwrap(),
            MAX_ZFP_CHAIN_FRAMES as u32,
            "theta",
            true,
        )
        .is_err());
        let _ = std::fs::remove_dir_all(run_dir);
    }

    #[test]
    fn keyframe_roundtrip_preserves_layout() {
        // Non-cubic dims so a reversed-dim / transpose bug is caught.
        let shape = vec![8usize, 12, 16];
        let data = sample_field(&shape);
        let payload = zfp_compress_payload(&shape, &data, 1.0e-5);
        let bytes = make_zfp3d(&shape, &payload, 1.0e-5, FLAG_KEYFRAME);

        let (h, decoded) = decode_frame_bytes(&bytes).unwrap();
        assert_eq!(h.shape(), shape);
        assert_eq!(decoded.len(), data.len());
        assert!(
            max_abs_err(&decoded, &data) < 1.0e-2,
            "decoded volume diverged (layout/transpose?): err={}",
            max_abs_err(&decoded, &data)
        );
    }

    #[test]
    fn bricked_v3_window_matches_full_decode_and_uses_bounded_budget() {
        let run_dir = unique_run_dir("bricked_window");
        let step_dir = run_dir.join("step_000000");
        std::fs::create_dir_all(&step_dir).unwrap();
        let shape = vec![70usize, 67, 35];
        let origin = [61usize, 60, 29];
        let extent = [9usize, 7, 5];
        let tolerance = 1.0e-5;
        let original = sample_field(&shape);
        let bytes = make_zfp3d_v3(&shape, &original, tolerance, FLAG_KEYFRAME);
        let path = step_dir.join("theta.zfp3d");
        std::fs::write(&path, bytes).unwrap();

        let (header, full) = read_zfp3d_frame(&path).unwrap();
        assert_eq!(header.version, 3);
        assert_eq!(header.brick_shape, Some([64, 64, 32]));
        assert_eq!(header.brick_counts, Some([2, 2, 2]));
        let (_, window) = read_zfp3d_raw_window(&path, origin, extent).unwrap();
        assert_eq!(window, extract_window(&full, &shape, origin, extent));
        assert!(max_abs_err(&window, &extract_window(&original, &shape, origin, extent)) < 1.0e-2);

        let (_, full_budget) =
            field_shape_and_reconstruction_bytes(run_dir.to_str().unwrap(), 0, "theta").unwrap();
        let (_, window_budget) = field_shape_and_window_reconstruction_bytes(
            run_dir.to_str().unwrap(),
            0,
            "theta",
            origin,
            extent,
        )
        .unwrap();
        assert!(window_budget < full_budget);
        assert!(decode_frame_bytes(&std::fs::read(&path).unwrap()).is_err());
        let _ = std::fs::remove_dir_all(run_dir);
    }

    #[test]
    fn bricked_v3_visitor_reconstructs_each_cell_once() {
        let run_dir = unique_run_dir("bricked_visitor");
        let step_dir = run_dir.join("step_000000");
        std::fs::create_dir_all(&step_dir).unwrap();
        let shape = vec![70usize, 67, 35];
        let original = sample_field(&shape);
        let path = step_dir.join("theta.zfp3d");
        std::fs::write(
            &path,
            make_zfp3d_v3(&shape, &original, 1.0e-5, FLAG_KEYFRAME),
        )
        .unwrap();
        let (_, full) = read_zfp3d_frame(&path).unwrap();
        let mut sum = 0.0_f64;
        let mut count = 0_usize;
        let got_shape = visit_reconstructed_field_bricks(
            run_dir.to_str().unwrap(),
            0,
            "theta",
            |_origin, _extent, values| {
                sum += values.iter().map(|value| f64::from(*value)).sum::<f64>();
                count += values.len();
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(got_shape, shape);
        assert_eq!(count, full.len());
        assert!((sum - full.iter().map(|value| f64::from(*value)).sum::<f64>()).abs() < 1.0e-6);
        let _ = std::fs::remove_dir_all(run_dir);
    }

    #[test]
    fn bricked_v3_rejects_corrupt_directory_before_decode() {
        let run_dir = unique_run_dir("bricked_corrupt");
        let step_dir = run_dir.join("step_000000");
        std::fs::create_dir_all(&step_dir).unwrap();
        let shape = vec![40usize, 37, 21];
        let data = sample_field(&shape);
        let mut bytes = make_zfp3d_v3(&shape, &data, 1.0e-5, FLAG_KEYFRAME);
        let directory_start = HEADER_V3_LEN + rd_u64(&bytes, 28) as usize;
        bytes[directory_start..directory_start + 8].copy_from_slice(&1u64.to_le_bytes());
        let path = step_dir.join("theta.zfp3d");
        std::fs::write(&path, bytes).unwrap();

        let error = read_zfp3d_raw_window(&path, [0, 0, 0], [4, 4, 4]).unwrap_err();
        assert!(error.to_string().contains("brick directory"));
        let _ = std::fs::remove_dir_all(run_dir);
    }

    #[test]
    fn bricked_v3_predictive_window_replays_the_full_chain_exactly() {
        let run_dir = unique_run_dir("bricked_predictive");
        let shape = vec![70usize, 67, 35];
        let origin = [61usize, 58, 29];
        let extent = [8usize, 9, 6];
        let tolerance = 1.0e-5;
        let d0 = sample_field(&shape);
        let d1: Vec<f32> = d0.iter().map(|value| value + 0.05).collect();
        let d2: Vec<f32> = d0.iter().map(|value| value + 0.11).collect();

        let step0 = run_dir.join("step_000000");
        std::fs::create_dir_all(&step0).unwrap();
        let path0 = step0.join("theta.zfp3d");
        std::fs::write(&path0, make_zfp3d_v3(&shape, &d0, tolerance, FLAG_KEYFRAME)).unwrap();
        write_v3_manifest(&step0, "keyframe", &[]);
        let (_, ref0) = read_zfp3d_frame(&path0).unwrap();

        let residual1: Vec<f32> = d1
            .iter()
            .zip(&ref0)
            .map(|(value, previous)| (*value as f64 - *previous as f64) as f32)
            .collect();
        let step30 = run_dir.join("step_000030");
        std::fs::create_dir_all(&step30).unwrap();
        let path30 = step30.join("theta.zfp3d");
        std::fs::write(
            &path30,
            make_zfp3d_v3(&shape, &residual1, tolerance, FLAG_SIMPLE_DELTA),
        )
        .unwrap();
        write_v3_manifest(&step30, "simple_delta", &[0]);
        let (_, decoded_residual1) = read_zfp3d_frame(&path30).unwrap();
        let ref1: Vec<f32> = decoded_residual1
            .iter()
            .zip(&ref0)
            .map(|(residual, previous)| (*residual as f64 + *previous as f64) as f32)
            .collect();

        let residual2: Vec<f32> = d2
            .iter()
            .zip(&ref1)
            .zip(&ref0)
            .map(|((value, previous), previous2)| {
                (*value as f64 - (2.0 * *previous as f64 - *previous2 as f64)) as f32
            })
            .collect();
        let step60 = run_dir.join("step_000060");
        std::fs::create_dir_all(&step60).unwrap();
        std::fs::write(
            step60.join("theta.zfp3d"),
            make_zfp3d_v3(&shape, &residual2, tolerance, FLAG_PREDICTIVE_DELTA),
        )
        .unwrap();
        write_v3_manifest(&step60, "predictive_delta", &[30, 0]);

        let (_, full) = reconstruct_field(run_dir.to_str().unwrap(), 60, "theta").unwrap();
        let (got_shape, window) =
            reconstruct_field_window(run_dir.to_str().unwrap(), 60, "theta", origin, extent)
                .unwrap();
        assert_eq!(got_shape, shape);
        assert_eq!(window, extract_window(&full, &shape, origin, extent));
        assert!(max_abs_err(&window, &extract_window(&d2, &shape, origin, extent)) < 3.0e-2);

        for stride in [[1, 1, 1], [3, 4, 2], [71, 68, 36]] {
            let mut decodes = 0;
            let sampled = read_sampled_field_window(
                run_dir.to_str().unwrap(),
                60,
                "theta",
                origin,
                extent,
                stride,
                8 * 1024 * 1024,
                1024 * 1024,
                Some([70, 67, 35]),
                |decoding| {
                    decodes += usize::from(decoding);
                    Ok(())
                },
            )
            .unwrap()
            .unwrap();
            let mut expected = Vec::new();
            for i in (origin[0]..origin[0] + extent[0]).step_by(stride[0]) {
                for j in (origin[1]..origin[1] + extent[1]).step_by(stride[1]) {
                    for k in (origin[2]..origin[2] + extent[2]).step_by(stride[2]) {
                        expected.extend_from_slice(
                            &full[(i * shape[1] + j) * shape[2] + k].to_le_bytes(),
                        );
                    }
                }
            }
            assert_eq!(sampled.bytes, expected);
            assert!(decodes > 0 && decodes <= 24);
        }

        let manifest_path = step60.join("manifest.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        manifest["fields"][0]["compression"]["brick_shape"][0] = serde_json::json!(32);
        std::fs::write(&manifest_path, manifest.to_string()).unwrap();
        let mismatch =
            reconstruct_field_window(run_dir.to_str().unwrap(), 60, "theta", origin, extent)
                .unwrap_err();
        assert!(mismatch.to_string().contains("disagrees with its manifest"));
        let _ = std::fs::remove_dir_all(run_dir);
    }

    #[test]
    fn encode_npy_roundtrips_through_npy_parser() {
        let shape = vec![3usize, 4, 5];
        let data = sample_field(&shape);
        let npy = encode_npy_f32(&shape, &data).unwrap();

        let layout = crate::npy::parse_f32_layout(&npy).unwrap();
        assert_eq!(layout.shape, shape);
        assert_eq!(
            layout.decoded_bytes / std::mem::size_of::<f32>(),
            data.len()
        );
        assert_eq!(layout.data_offset % 64, 0);
        // Body bytes match exactly.
        let body = &npy[layout.data_offset..layout.data_offset + layout.decoded_bytes];
        for (i, v) in data.iter().enumerate() {
            let got = f32::from_le_bytes([
                body[i * 4],
                body[i * 4 + 1],
                body[i * 4 + 2],
                body[i * 4 + 3],
            ]);
            assert_eq!(got, *v);
        }
    }

    #[test]
    fn delta_replay_simple_and_predictive() {
        // Three evolving frames; build a keyframe + simple delta + predictive
        // delta the way the writer does (residuals against DECODED references),
        // then prove replay() reconstructs each target within a compounding
        // tolerance — and that the predictive frame's first-after-keyframe rule
        // is respected (the simple frame uses prev only).
        let shape = vec![6usize, 5, 4];
        let tol = 1.0e-5;
        let d0 = sample_field(&shape);
        let d1: Vec<f32> = d0.iter().map(|x| x + 0.05).collect();
        let d2: Vec<f32> = d0.iter().map(|x| x + 0.11).collect();

        // Keyframe.
        let kf = make_zfp3d(
            &shape,
            &zfp_compress_payload(&shape, &d0, tol),
            tol,
            FLAG_KEYFRAME,
        );
        let (_, ref0) = decode_frame_bytes(&kf).unwrap();

        // Simple delta: residual = d1 - ref0.
        let resid1: Vec<f32> = d1.iter().zip(&ref0).map(|(a, b)| a - b).collect();
        let df1 = make_zfp3d(
            &shape,
            &zfp_compress_payload(&shape, &resid1, tol),
            tol,
            FLAG_SIMPLE_DELTA,
        );
        let (_, res1_dec) = decode_frame_bytes(&df1).unwrap();
        let ref1: Vec<f32> = res1_dec.iter().zip(&ref0).map(|(r, p)| r + p).collect();

        // Predictive delta: residual = d2 - (2*ref1 - ref0).
        let pred: Vec<f32> = ref1.iter().zip(&ref0).map(|(a, b)| 2.0 * a - b).collect();
        let resid2: Vec<f32> = d2.iter().zip(&pred).map(|(a, b)| a - b).collect();
        let df2 = make_zfp3d(
            &shape,
            &zfp_compress_payload(&shape, &resid2, tol),
            tol,
            FLAG_PREDICTIVE_DELTA,
        );
        let (_, res2_dec) = decode_frame_bytes(&df2).unwrap();

        let frame = |bytes: &[u8]| {
            let (h, d) = decode_frame_bytes(bytes).unwrap();
            Frame {
                flags: h.flags,
                data: d,
            }
        };

        // Reconstruct each target by replaying the chain up to it.
        let got0 = replay(vec![frame(&kf)]).unwrap();
        let got1 = replay(vec![frame(&kf), frame(&df1)]).unwrap();
        let got2 = replay(vec![frame(&kf), frame(&df1), frame(&df2)]).unwrap();
        // res2_dec only referenced to keep the chain explicit.
        let _ = res2_dec;

        assert!(max_abs_err(&got0, &d0) < 1.0e-2);
        assert!(max_abs_err(&got1, &d1) < 2.0e-2);
        assert!(max_abs_err(&got2, &d2) < 3.0e-2);
    }

    #[test]
    fn delta_without_keyframe_errors() {
        let f = Frame {
            flags: FLAG_SIMPLE_DELTA,
            data: vec![1.0, 2.0],
        };
        assert!(replay(vec![f]).is_err());
    }

    #[test]
    fn reconstruct_field_keyframe_from_disk() {
        // Write a keyframe .zfp3d into a temp run dir and reconstruct it.
        use std::sync::atomic::{AtomicU64, Ordering};
        static CTR: AtomicU64 = AtomicU64::new(0);
        let n = CTR.fetch_add(1, Ordering::Relaxed);
        let run_dir =
            std::env::temp_dir().join(format!("nimbus_zfp3d_test_{}_{}", std::process::id(), n));
        let step_dir = run_dir.join("step_000000");
        std::fs::create_dir_all(&step_dir).unwrap();

        let shape = vec![5usize, 6, 7];
        let data = sample_field(&shape);
        let bytes = make_zfp3d(
            &shape,
            &zfp_compress_payload(&shape, &data, 1e-5),
            1e-5,
            FLAG_KEYFRAME,
        );
        std::fs::write(step_dir.join("theta.zfp3d"), &bytes).unwrap();

        let (got_shape, got) = reconstruct_field(run_dir.to_str().unwrap(), 0, "theta").unwrap();
        assert_eq!(got_shape, shape);
        assert!(max_abs_err(&got, &data) < 1.0e-2);

        let _ = std::fs::remove_dir_all(&run_dir);
    }

    #[test]
    fn reconstruct_field_replays_sparse_delta_steps() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static CTR: AtomicU64 = AtomicU64::new(0);
        let n = CTR.fetch_add(1, Ordering::Relaxed);
        let run_dir = std::env::temp_dir().join(format!(
            "nimbus_zfp3d_delta_test_{}_{}",
            std::process::id(),
            n
        ));
        let shape = vec![5usize, 6, 7];
        let tol = 1.0e-5;
        let d0 = sample_field(&shape);
        let d1: Vec<f32> = d0.iter().map(|x| x + 0.05).collect();
        let d2: Vec<f32> = d0.iter().map(|x| x + 0.11).collect();

        let kf = make_zfp3d(
            &shape,
            &zfp_compress_payload(&shape, &d0, tol),
            tol,
            FLAG_KEYFRAME,
        );
        let (_, ref0) = decode_frame_bytes(&kf).unwrap();
        let resid1: Vec<f32> = d1.iter().zip(&ref0).map(|(a, b)| a - b).collect();
        let df1 = make_zfp3d(
            &shape,
            &zfp_compress_payload(&shape, &resid1, tol),
            tol,
            FLAG_SIMPLE_DELTA,
        );
        let (_, decoded_resid1) = decode_frame_bytes(&df1).unwrap();
        let ref1: Vec<f32> = decoded_resid1
            .iter()
            .zip(&ref0)
            .map(|(residual, previous)| residual + previous)
            .collect();
        let prediction: Vec<f32> = ref1
            .iter()
            .zip(&ref0)
            .map(|(previous, previous2)| 2.0 * previous - previous2)
            .collect();
        let resid2: Vec<f32> = d2
            .iter()
            .zip(&prediction)
            .map(|(value, predicted)| value - predicted)
            .collect();
        let df2 = make_zfp3d(
            &shape,
            &zfp_compress_payload(&shape, &resid2, tol),
            tol,
            FLAG_PREDICTIVE_DELTA,
        );

        for (step, bytes, role, parents) in [
            (0, kf, "keyframe", vec![]),
            (30, df1, "simple_delta", vec![0]),
            (60, df2, "predictive_delta", vec![30, 0]),
        ] {
            let step_dir = run_dir.join(format!("step_{step:06}"));
            std::fs::create_dir_all(&step_dir).unwrap();
            std::fs::write(step_dir.join("theta.zfp3d"), bytes).unwrap();
            std::fs::write(
                step_dir.join("manifest.json"),
                serde_json::json!({
                    "data_product": {
                        "lossy_compression_contract": "nimbus.zfp.per_field.v1"
                    },
                    "fields": [{
                        "field_id": "theta",
                        "compression": {
                            "codec": "zfp",
                            "decoded_dtype": "float32",
                            "frame_role": role,
                            "parent_step_indices": parents
                        }
                    }]
                })
                .to_string(),
            )
            .unwrap();
        }

        let (got_shape, got) = reconstruct_field(run_dir.to_str().unwrap(), 60, "theta").unwrap();
        assert_eq!(got_shape, shape);
        assert!(max_abs_err(&got, &d2) < 3.0e-2);

        std::fs::remove_file(run_dir.join("step_000030/theta.zfp3d")).unwrap();
        let error = reconstruct_field(run_dir.to_str().unwrap(), 60, "theta").unwrap_err();
        assert!(error.to_string().contains("declared ZFP parent"));
        let _ = std::fs::remove_dir_all(&run_dir);
    }

    /// Decode a real simulator-written `.zfp3d`. Ignored by default (needs an
    /// external file); run with:
    ///   NIMBUS_TEST_ZFP3D=/path/to/field.zfp3d \
    ///     cargo test real_zfp3d_decode -- --ignored --nocapture
    /// Proves the reader handles the actual writer's bytes (header layout +
    /// libzfp 1.0.1 bitstream) and prints the decoded range to sanity-check.
    #[test]
    #[ignore]
    fn real_zfp3d_decode() {
        let path = std::env::var("NIMBUS_TEST_ZFP3D")
            .expect("set NIMBUS_TEST_ZFP3D to a simulator-written .zfp3d");
        let (h, data) = read_zfp3d_frame(Path::new(&path)).unwrap();
        let finite = data.iter().filter(|v| v.is_finite()).count();
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in &data {
            if v.is_finite() {
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        println!(
            "dims=({},{},{}) mode={} tol={} flags={} elements={} finite={} range=[{}, {}]",
            h.dim0,
            h.dim1,
            h.dim2,
            h.mode,
            h.mode_param,
            h.flags,
            data.len(),
            finite,
            lo,
            hi
        );
        assert_eq!(data.len(), h.element_count);
        assert_eq!(finite, data.len(), "real field had non-finite values");
    }

    fn exact_qualified_zfp_budgets() -> std::collections::BTreeMap<String, f64> {
        let binary = std::env::var("NIMBUS_TEST_ZFP_BINARY")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../bin/nimbus_sim"));
        let output = std::process::Command::new(&binary)
            .arg("--scheme-catalog=json").output().expect("exact simulator catalog");
        assert!(output.status.success() && output.stdout.len() <= 32 * 1024 * 1024);
        let catalog: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        let parsed = catalog["runtimeCapabilities"]["interactionPolicy"]
            ["qualifiedZfpFieldBudgets"].as_object().unwrap();
        let marker = b"\"qualifiedZfpFieldBudgets\":{";
        let starts: Vec<usize> = output.stdout.windows(marker.len()).enumerate()
            .filter_map(|(index, bytes)| (bytes == marker).then_some(index + marker.len()))
            .collect();
        assert_eq!(starts.len(), 1);
        let bytes = &output.stdout;
        let mut cursor = starts[0];
        let mut budgets = std::collections::BTreeMap::new();
        loop {
            while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
                cursor += 1;
            }
            if bytes.get(cursor) == Some(&b'}') {
                break;
            }
            assert_eq!(bytes.get(cursor), Some(&b'"'));
            let start = cursor;
            cursor += 1;
            while bytes.get(cursor) != Some(&b'"') {
                assert!(bytes.get(cursor).is_some_and(|byte| byte.is_ascii_lowercase()
                    || byte.is_ascii_digit() || *byte == b'_'));
                cursor += 1;
            }
            cursor += 1;
            let field: String = serde_json::from_slice(&bytes[start..cursor]).unwrap();
            while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
                cursor += 1;
            }
            assert_eq!(bytes.get(cursor), Some(&b':'));
            cursor += 1;
            while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
                cursor += 1;
            }
            let start = cursor;
            while bytes.get(cursor).is_some_and(|byte| byte.is_ascii_digit()
                || matches!(*byte, b'-' | b'+' | b'.' | b'e' | b'E')) {
                cursor += 1;
            }
            let token = std::str::from_utf8(&bytes[start..cursor]).unwrap();
            assert!(serde_json::from_str::<serde_json::Value>(token).unwrap().is_number());
            let budget = token.parse::<f64>().unwrap();
            assert!(budget.is_finite() && budget > 0.0);
            assert!(budgets.insert(field, budget).is_none());
            while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
                cursor += 1;
            }
            match bytes.get(cursor) {
                Some(b',') => cursor += 1,
                Some(b'}') => break,
                _ => panic!("invalid exact qualified ZFP budget boundary"),
            }
        }
        assert_eq!(budgets.len(), parsed.len());
        assert!(budgets.keys().all(|field| parsed.contains_key(field)));
        println!("[qualified-zfp-reference] catalog={} budgets={}",
                 crate::platform::sha256_file(&binary).unwrap(), budgets.len());
        budgets
    }

    #[test]
    #[ignore]
    fn qualified_simulator_zfp_matches_npy_reference() {
        let zfp_run = PathBuf::from(
            std::env::var("NIMBUS_TEST_ZFP_RUN")
                .expect("set NIMBUS_TEST_ZFP_RUN to the simulator's ZFP run"),
        );
        let npy_run = PathBuf::from(
            std::env::var("NIMBUS_TEST_NPY_RUN")
                .expect("set NIMBUS_TEST_NPY_RUN to the simulator's NPY reference run"),
        );
        let budgets = exact_qualified_zfp_budgets();
        let mut checked_fields = 0usize;
        let mut compressed_fields = 0usize;
        let mut fallback_fields = 0usize;

        for entry in std::fs::read_dir(&zfp_run).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name();
            let Some(step_text) = name.to_str().and_then(|value| value.strip_prefix("step_"))
            else {
                continue;
            };
            let Ok(step) = step_text.parse::<u32>() else {
                continue;
            };
            let manifest: serde_json::Value =
                serde_json::from_slice(&std::fs::read(entry.path().join("manifest.json")).unwrap())
                    .unwrap();
            let published = manifest["published_fields"].as_array().unwrap();
            let catalog = manifest["fields"].as_array().unwrap();
            for field_value in published {
                let field = field_value.as_str().unwrap();
                let metadata = catalog
                    .iter()
                    .find(|item| item["field_id"].as_str() == Some(field))
                    .unwrap();
                let reference_path = npy_run.join(&name).join(format!("{field}.npy"));
                let reference_bytes = crate::npy::read_f32_file(&reference_path).unwrap();
                let layout = crate::npy::parse_f32_layout(&reference_bytes).unwrap();
                let reference: Vec<f32> = reference_bytes
                    [layout.data_offset..layout.data_offset + layout.decoded_bytes]
                    .chunks_exact(std::mem::size_of::<f32>())
                    .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
                    .collect();
                let step_dir = entry.path();
                let zfp_path = step_dir.join(format!("{field}.zfp3d"));
                let fallback_path = step_dir.join(format!("{field}.npy"));
                if let Some(compression) = metadata.get("compression") {
                    assert_eq!(metadata["file_pattern"].as_str(),
                               Some(format!("{field}.zfp3d").as_str()));
                    assert!(metadata.get("storage_format").is_none());
                    assert!(zfp_path.is_file() && !fallback_path.exists());
                    assert_eq!(compression["mode"].as_str(), Some("accuracy"));
                    assert_eq!(compression["error_contract"].as_str(),
                               Some("zfp_absolute_accuracy_before_prefilters"));
                    assert!(compression["mode_parameter"].as_f64()
                        .is_some_and(|value| value.is_finite() && value > 0.0));
                    let header = read_header_only(&zfp_path).unwrap();
                    let budget = *budgets.get(field).expect("exact catalog field budget");
                    assert!(header.mode_param.is_finite() && header.mode_param > 0.0
                        && header.mode_param <= budget,
                        "{field} step {step} configured tolerance exceeds exact catalog budget");
                    let (shape, decoded) = reconstruct_field(
                        zfp_run.to_str().unwrap(), step, field).unwrap();
                    let mut reference_shape = layout.shape.clone();
                    if reference_shape.len() == 2 {
                        reference_shape.push(1);
                    }
                    assert_eq!(shape, reference_shape);
                    assert_eq!(decoded.len(), reference.len());
                    assert!(decoded.iter().zip(&reference).all(|(actual, expected)|
                        actual.is_finite() && expected.is_finite()));
                    let error = decoded.iter().zip(&reference)
                        .map(|(actual, expected)|
                            (f64::from(*actual) - f64::from(*expected)).abs())
                        .fold(0.0f64, f64::max);
                    assert!(error <= budget,
                        "{field} step {step} decoded error {error} exceeds exact catalog budget {budget}");
                    compressed_fields += 1;
                } else {
                    assert_eq!(metadata["storage_format"].as_str(), Some("npy_2d"));
                    assert_eq!(metadata["file_pattern"].as_str(),
                               Some(format!("{field}.npy").as_str()));
                    assert!(fallback_path.is_file() && !zfp_path.exists());
                    let fallback_bytes = crate::npy::read_f32_file(&fallback_path).unwrap();
                    let fallback_layout = crate::npy::parse_f32_layout(&fallback_bytes).unwrap();
                    assert_eq!(fallback_layout.shape, layout.shape);
                    assert_eq!(fallback_layout.decoded_bytes, layout.decoded_bytes);
                    assert_eq!(
                        &fallback_bytes[fallback_layout.data_offset..],
                        &reference_bytes[layout.data_offset..],
                        "{field} step {step} NPY fallback differs bitwise, including NaNs"
                    );
                    fallback_fields += 1;
                }
                checked_fields += 1;
            }
        }
        assert!(
            checked_fields >= 16,
            "expected two real eight-field publications"
        );
        assert!(compressed_fields >= 16);
        println!("[qualified-zfp-reference] fields={checked_fields} compressed={compressed_fields} fallback_npy={fallback_fields}");
    }

    #[test]
    #[ignore]
    fn qualified_retained_predictive_zfp_latest_decodes() {
        let run = PathBuf::from(
            std::env::var("NIMBUS_TEST_RETAINED_ZFP_RUN")
                .expect("set NIMBUS_TEST_RETAINED_ZFP_RUN to the retained simulator run"),
        );
        let latest = std::fs::read_dir(&run)
            .unwrap()
            .filter_map(Result::ok)
            .filter_map(|entry| {
                entry
                    .file_name()
                    .to_str()?
                    .strip_prefix("step_")?
                    .parse::<u32>()
                    .ok()
            })
            .max()
            .expect("retained run has no step publications");
        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(run.join(format!("step_{latest:06}/manifest.json"))).unwrap(),
        )
        .unwrap();
        let fields = manifest["published_fields"].as_array().unwrap();
        assert_eq!(fields.len(), 8);
        for field in fields {
            let field = field.as_str().unwrap();
            let (_, values) = reconstruct_field(run.to_str().unwrap(), latest, field).unwrap();
            assert!(!values.is_empty());
            assert!(values.iter().all(|value| value.is_finite()));
        }
    }
    #[test]
    fn sampled_v3_skips_unselected_bricks_and_bounds_native_window_memory() {
        let run = unique_run_dir("sampled_v3_bricks");
        let shape = [192usize, 128, 64];
        let data = sample_field(&shape);
        let step = run.join("step_000000");
        std::fs::create_dir_all(&step).unwrap();
        let path = step.join("theta.zfp3d");
        std::fs::write(&path, make_zfp3d_v3_keyframe(&shape, &data, 1e-5)).unwrap();
        let (_, full) = reconstruct_field(run.to_str().unwrap(), 0, "theta").unwrap();
        let mut decodes = 0;
        let sampled = read_sampled_field_window(
            run.to_str().unwrap(),
            0,
            "theta",
            [0, 0, 0],
            shape,
            [128, 96, 40],
            4 * 1024 * 1024,
            32,
            Some(shape),
            |decoding| {
                decodes += usize::from(decoding);
                Ok(())
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(decodes, 8, "four unselected bricks must not be decoded");
        assert_eq!(sampled.output_shape, [2, 2, 2]);
        let mut expected = Vec::new();
        for i in [0, 128] {
            for j in [0, 96] {
                for k in [0, 40] {
                    expected
                        .extend_from_slice(&full[(i * shape[1] + j) * shape[2] + k].to_le_bytes());
                }
            }
        }
        assert_eq!(sampled.bytes, expected);
        assert!(sampled.source_working_bytes < full.len() * 4);

        let native = read_sampled_field_window(
            run.to_str().unwrap(),
            0,
            "theta",
            [60, 60, 29],
            [73, 67, 35],
            [1, 1, 1],
            8 * 1024 * 1024,
            1024 * 1024,
            Some(shape),
            |_| Ok(()),
        )
        .unwrap()
        .unwrap();
        let exact = extract_window(&full, &shape, [60, 60, 29], [73, 67, 35]);
        assert_eq!(native.bytes, bytemuck::cast_slice::<f32, u8>(&exact));

        let mut attempted = 0;
        assert!(read_sampled_field_window(
            run.to_str().unwrap(),
            0,
            "theta",
            [0, 0, 0],
            shape,
            [128, 96, 40],
            1,
            32,
            Some(shape),
            |decoding| {
                attempted += usize::from(decoding);
                Ok(())
            }
        )
        .is_err());
        assert_eq!(attempted, 0);
        assert!(read_sampled_field_window(
            run.to_str().unwrap(),
            0,
            "theta",
            [0, 0, 0],
            shape,
            [128, 96, 40],
            4 * 1024 * 1024,
            31,
            Some(shape),
            |decoding| {
                attempted += usize::from(decoding);
                Ok(())
            }
        )
        .is_err());
        assert_eq!(attempted, 0);
        assert!(read_sampled_field_window(
            run.to_str().unwrap(),
            0,
            "theta",
            [0, 0, 0],
            shape,
            [128, 96, 40],
            4 * 1024 * 1024,
            32,
            Some([1, 1, 1]),
            |_| Ok(())
        )
        .is_err());

        let mut completed = 0;
        let cancelled = read_sampled_field_window(
            run.to_str().unwrap(),
            0,
            "theta",
            [0, 0, 0],
            shape,
            [128, 96, 40],
            4 * 1024 * 1024,
            32,
            Some(shape),
            |decoding| {
                if decoding {
                    completed += 1;
                }
                if completed == 2 {
                    return Err(AppError::Task("cancelled between bricks".into()));
                }
                Ok(())
            },
        );
        assert!(cancelled.is_err());
        assert_eq!(completed, 2);
        let _ = std::fs::remove_dir_all(run);
    }

    #[test]
    fn sampled_v3_detects_artifact_changes_before_publishing() {
        use std::io::Write;
        let run = unique_run_dir("sampled_v3_changed");
        let step = run.join("step_000000");
        std::fs::create_dir_all(&step).unwrap();
        let path = step.join("theta.zfp3d");
        std::fs::write(&path, make_zfp3d_v3_keyframe(&[4, 4, 4], &[1.0; 64], 1e-5)).unwrap();
        let mut changed = false;
        let result = read_sampled_field_window(
            run.to_str().unwrap(),
            0,
            "theta",
            [0, 0, 0],
            [4, 4, 4],
            [2, 2, 2],
            1024 * 1024,
            32,
            Some([4, 4, 4]),
            |decoding| {
                if decoding && !changed {
                    std::fs::OpenOptions::new()
                        .append(true)
                        .open(&path)?
                        .write_all(&[0])?;
                    changed = true;
                }
                Ok(())
            },
        );
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(run);
    }

    #[test]
    fn sampled_v3_long_predictive_chain_is_deterministic_across_readers() {
        let run = unique_run_dir("sampled_v3_long_chain");
        let shape = [130usize, 4, 4];
        for index in 0..10 {
            let step = run.join(format!("step_{:06}", index * 10));
            std::fs::create_dir_all(&step).unwrap();
            let flags = if index == 0 {
                FLAG_KEYFRAME
            } else if index == 1 {
                FLAG_SIMPLE_DELTA
            } else {
                FLAG_PREDICTIVE_DELTA
            };
            std::fs::write(
                step.join("theta.zfp3d"),
                make_zfp3d_v3(&shape, &vec![0.25; shape.iter().product()], 1e-5, flags),
            )
            .unwrap();
            let parents = if index == 0 {
                vec![]
            } else if index == 1 {
                vec![0]
            } else {
                vec![(index - 1) * 10, (index - 2) * 10]
            };
            write_v3_manifest(
                &step,
                if index == 0 {
                    "keyframe"
                } else if index == 1 {
                    "simple_delta"
                } else {
                    "predictive_delta"
                },
                &parents,
            );
        }
        let (_, full) = reconstruct_field(run.to_str().unwrap(), 90, "theta").unwrap();
        let mut expected = Vec::new();
        for i in [0, 65] {
            for j in [0, 3] {
                for k in [0, 3] {
                    expected.extend_from_slice(&full[(i * 4 + j) * 4 + k].to_le_bytes());
                }
            }
        }
        std::thread::scope(|scope| {
            let readers = (0..3)
                .map(|_| {
                    scope.spawn(|| {
                        read_sampled_field_window(
                            run.to_str().unwrap(),
                            90,
                            "theta",
                            [0, 0, 0],
                            shape,
                            [65, 3, 3],
                            128 * 1024,
                            32,
                            Some(shape),
                            |_| Ok(()),
                        )
                        .unwrap()
                        .unwrap()
                        .bytes
                    })
                })
                .collect::<Vec<_>>();
            for reader in readers {
                assert_eq!(reader.join().unwrap(), expected);
            }
        });
        let _ = std::fs::remove_dir_all(run);
    }

    #[test]
    fn sampled_v3_beast_sized_predictive_cost_stays_bounded() {
        let run = unique_run_dir("sampled_v3_beast_cost");
        let shape = [512usize, 512, 160];
        let (template, _) = make_repeated_brick_zfp3d_v3(shape, &vec![0.25; 64 * 64 * 32], 1e-5);
        for (index, flags) in [
            (0u32, FLAG_KEYFRAME),
            (30, FLAG_SIMPLE_DELTA),
            (60, FLAG_PREDICTIVE_DELTA),
        ] {
            let step = run.join(format!("step_{index:06}"));
            std::fs::create_dir_all(&step).unwrap();
            let mut artifact = template.clone();
            artifact[36] = flags;
            std::fs::write(step.join("theta.zfp3d"), artifact).unwrap();
        }
        let (_, old_wind) = field_shape_and_window_reconstruction_bytes(
            run.to_str().unwrap(),
            60,
            "theta",
            [0, 0, 0],
            shape,
        )
        .unwrap();
        let wind = read_sampled_field_window(
            run.to_str().unwrap(),
            60,
            "theta",
            [0, 0, 0],
            shape,
            [16, 16, 5],
            32 * 1024 * 1024,
            131072,
            Some(shape),
            |_| Ok(()),
        )
        .unwrap()
        .unwrap();
        assert_eq!(wind.bytes.len(), 131072);
        assert!(wind
            .bytes
            .chunks_exact(4)
            .all(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()) == 1.0));
        let extent = [273, 273, 160];
        let (_, old_native) = field_shape_and_window_reconstruction_bytes(
            run.to_str().unwrap(),
            60,
            "theta",
            [150, 150, 0],
            extent,
        )
        .unwrap();
        let native = read_sampled_field_window(
            run.to_str().unwrap(),
            60,
            "theta",
            [150, 150, 0],
            extent,
            [1, 1, 1],
            64 * 1024 * 1024,
            64 * 1024 * 1024,
            Some(shape),
            |_| Ok(()),
        )
        .unwrap()
        .unwrap();
        assert!(old_wind > 480 * 1024 * 1024);
        assert!(old_native > 128 * 1024 * 1024);
        assert!(native.source_working_bytes < 64 * 1024 * 1024);
        println!("COMPRESSED_WINDOW_COST wind_old={old_wind} wind_new={} native_old={old_native} native_new={} native_cells={}",
            wind.source_working_bytes, native.source_working_bytes, native.bytes.len()/4);
        let _ = std::fs::remove_dir_all(run);
    }

    #[test]
    fn reversible_v4_keyframe_reconstructs_exact_full_window_and_bricks() {
        let shape = [12usize, 12, 48];
        let source: Vec<f32> = (0..shape.iter().product::<usize>())
            .map(|index| 1.0 - (index % 1000) as f32 * 0.0001)
            .collect();
        let artifact = make_zfp3d_v4(&shape, &source);
        let run = v4_test_run("exact", &artifact);
        let path = run.join("step_000000/rhohv.zfp3d");
        let (header, full) = read_zfp3d_frame(&path).unwrap();
        assert_eq!(
            (header.version, header.mode, header.mode_param),
            (4, 3, 0.0)
        );
        assert_eq!(header.brick_entry_bytes, Some(BRICK_ENTRY_V4_LEN));
        assert_eq!(
            full.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            source
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        let origin = [2, 3, 29];
        let extent = [4, 5, 8];
        let (_, window) = read_zfp3d_raw_window(&path, origin, extent).unwrap();
        for i in 0..extent[0] {
            for j in 0..extent[1] {
                for k in 0..extent[2] {
                    let actual = window[(i * extent[1] + j) * extent[2] + k].to_bits();
                    let expected = source
                        [((i + origin[0]) * shape[1] + j + origin[1]) * shape[2] + k + origin[2]]
                        .to_bits();
                    assert_eq!(actual, expected);
                }
            }
        }
        assert!(reconstruct_field(run.to_str().unwrap(), 0, "rhohv").is_err());
        write_v4_test_manifest(&run, "reversible");
        let (_, full_working) =
            field_shape_and_reconstruction_bytes(run.to_str().unwrap(), 0, "rhohv").unwrap();
        let (_, window_working) = field_shape_and_window_reconstruction_bytes(
            run.to_str().unwrap(),
            0,
            "rhohv",
            origin,
            extent,
        )
        .unwrap();
        assert!(full_working >= source.len() * 4 + 64 * 1024);
        assert!(window_working >= extent.iter().product::<usize>() * 4 + 64 * 1024);
        let (_, public) = reconstruct_field(run.to_str().unwrap(), 0, "rhohv").unwrap();
        assert_eq!(
            public
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            source
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        let mut visits = 0;
        visit_reconstructed_field_bricks(run.to_str().unwrap(), 0, "rhohv", |_, _, values| {
            visits += 1;
            assert!(!values.is_empty());
            Ok(())
        })
        .unwrap();
        assert_eq!(visits, 2);
        let sampled = read_sampled_field_window(
            run.to_str().unwrap(),
            0,
            "rhohv",
            [0, 0, 0],
            shape,
            [2, 2, 4],
            1024 * 1024,
            1024 * 1024,
            Some(shape),
            |_| Ok(()),
        )
        .unwrap()
        .unwrap();
        assert_eq!(sampled.output_shape, vec![6, 6, 12]);
        assert!(sampled.source_working_bytes < 1024 * 1024);
        assert!(read_sampled_field_window(
            run.to_str().unwrap(),
            0,
            "rhohv",
            [0, 0, 0],
            shape,
            [2, 2, 4],
            sampled.source_working_bytes - 1,
            1024 * 1024,
            Some(shape),
            |_| Ok(())
        )
        .is_err());
        for i in 0..6 {
            for j in 0..6 {
                for k in 0..12 {
                    let offset = ((i * 6 + j) * 12 + k) * 4;
                    let bits =
                        u32::from_le_bytes(sampled.bytes[offset..offset + 4].try_into().unwrap());
                    assert_eq!(
                        bits,
                        source[((i * 2) * shape[1] + j * 2) * shape[2] + k * 4].to_bits()
                    );
                }
            }
        }
        write_v4_test_manifest(&run, "accuracy");
        assert!(reconstruct_field(run.to_str().unwrap(), 0, "rhohv").is_err());
        write_v4_test_manifest(&run, "reversible");
        let manifest_path = run.join("step_000000/manifest.json");
        let valid: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        for (key, replacement) in [
            ("contract_version", serde_json::json!(1)),
            ("mode_parameter", serde_json::json!(0.01)),
            (
                "error_contract",
                serde_json::json!("zfp_absolute_accuracy_before_prefilters"),
            ),
            ("container_version", serde_json::json!(3)),
            ("frame_role", serde_json::json!("simple_delta")),
            ("float16_prefilter", serde_json::json!(true)),
        ] {
            let mut invalid = valid.clone();
            invalid["fields"][0]["compression"][key] = replacement;
            std::fs::write(&manifest_path, serde_json::to_vec(&invalid).unwrap()).unwrap();
            assert!(
                reconstruct_field(run.to_str().unwrap(), 0, "rhohv").is_err(),
                "{key}"
            );
        }
        let _ = std::fs::remove_dir_all(run);
    }

    #[test]
    fn reversible_v4_corruption_and_overflow_fail_without_partial_brick_visit() {
        let shape = [12usize, 12, 48];
        let source: Vec<f32> = (0..shape.iter().product::<usize>())
            .map(|index| 1.0 - (index % 1000) as f32 * 0.0001)
            .collect();
        let good = make_zfp3d_v4(&shape, &source);
        let directory_start = HEADER_V3_LEN + rd_u64(&good, 28) as usize;
        let run = v4_test_run("corruption", &good);
        write_v4_test_manifest(&run, "reversible");
        let path = run.join("step_000000/rhohv.zfp3d");
        let mutate = |offset: usize, replacement: &[u8]| {
            let mut bytes = good.clone();
            bytes[offset..offset + replacement.len()].copy_from_slice(replacement);
            bytes
        };
        let mut cases = vec![
            mutate(4, &3u16.to_le_bytes()),
            mutate(6, &0u16.to_le_bytes()),
            mutate(20, &(-0.0f64).to_le_bytes()),
            mutate(36, &[FLAG_SIMPLE_DELTA]),
            mutate(37, &[1]),
            mutate(40, b"ZB3D"),
            mutate(8, &i32::MAX.to_le_bytes()),
            mutate(28, &u64::MAX.to_le_bytes()),
            mutate(directory_start, &u64::MAX.to_le_bytes()),
            mutate(
                directory_start + BRICK_ENTRY_V4_LEN,
                &(HEADER_V3_LEN as u64).to_le_bytes(),
            ),
            mutate(directory_start + 8, &[good[directory_start + 8] ^ 1]),
            mutate(HEADER_V3_LEN, &[good[HEADER_V3_LEN] ^ 1]),
            good[..good.len() - 1].to_vec(),
        ];
        cases.push(good[..20].to_vec());
        for bad in cases {
            std::fs::write(&path, bad).unwrap();
            assert!(read_zfp3d_frame(&path).is_err());
        }
        let mut corrupt_second = good.clone();
        corrupt_second[directory_start + BRICK_ENTRY_V4_LEN + 8] ^= 1;
        std::fs::write(&path, corrupt_second).unwrap();
        let mut visits = 0;
        assert!(
            visit_reconstructed_field_bricks(run.to_str().unwrap(), 0, "rhohv", |_, _, _| {
                visits += 1;
                Ok(())
            })
            .is_err()
        );
        assert_eq!(visits, 0);
        assert!(read_sampled_field_window(
            run.to_str().unwrap(),
            0,
            "rhohv",
            [0, 0, 0],
            shape,
            [2, 2, 4],
            1024 * 1024,
            1024 * 1024,
            Some(shape),
            |_| Ok(())
        )
        .is_err());
        let _ = std::fs::remove_dir_all(run);
    }

    #[test]
    #[ignore = "set NIMBUS_TEST_DRAFT_V4_ROOT and NIMBUS_TEST_NPY_RUN to sealed radar vectors"]
    fn reversible_v4_sealed_native_payloads_match_npy_bits() {
        let root = PathBuf::from(std::env::var("NIMBUS_TEST_DRAFT_V4_ROOT").unwrap());
        let npy = PathBuf::from(std::env::var("NIMBUS_TEST_NPY_RUN").unwrap());
        for step in ["step_000000", "step_000001"] {
            for field in ["reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv"] {
                let path = root.join(step).join(format!("{field}.zfp3d"));
                let (header, values) = read_zfp3d_frame(&path).unwrap();
                assert_eq!(
                    (header.version, header.mode, header.flags),
                    (4, 3, FLAG_KEYFRAME)
                );
                let reference = std::fs::read(npy.join(step).join(format!("{field}.npy"))).unwrap();
                let layout = crate::npy::parse_f32_layout(&reference).unwrap();
                assert_eq!(layout.shape, vec![12, 12, 48]);
                let decoded: Vec<u8> = values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect();
                assert_eq!(decoded, reference[layout.data_offset..]);
            }
        }
        for entry in std::fs::read_dir(root.join("corruptions")).unwrap() {
            let path = entry.unwrap().path();
            if path
                .extension()
                .is_some_and(|extension| extension == "zfp3d")
            {
                assert!(read_zfp3d_frame(&path).is_err(), "{}", path.display());
            }
        }
    }
}
