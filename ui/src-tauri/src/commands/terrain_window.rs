//! Exact terrain-following aggregation for bounded native history windows.
//!
//! Terrain cells are horizontally Cartesian but have a non-separable vertical
//! lattice.  A display block therefore uses the lossless float64 interface
//! thickness as its physical-volume weight and publishes the matching
//! float64 volume centroid.  No nominal `dz` enters this module.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::mem::size_of;
use std::path::Path;

use crate::error::{AppError, AppResult};

#[derive(Debug)]
pub(crate) struct TerrainGeometryWindow {
    pub centers_m: Vec<f64>,
    pub interfaces_m: Vec<f64>,
    /// Exact geometry payload bytes inspected for this window.  NPY headers
    /// are excluded, matching the field-window source-payload convention.
    pub source_bytes: usize,
}

#[derive(Debug)]
pub(crate) struct TerrainAggregation {
    pub values: Vec<f32>,
    /// Component-major `[x_m..., y_m..., z_m...]`, one value per output cell.
    pub centroids_m: Vec<f64>,
    /// Exact summed physical volume represented by every returned block.
    /// Kept alongside the centroids so comparison metrics and conservative
    /// Cartesian remapping never have to infer terrain-cell thickness from a
    /// nominal `dz`.
    pub volumes_m3: Vec<f64>,
    pub output_shape: Vec<usize>,
    /// Accumulator plus returned field/centroid storage owned by aggregation.
    pub working_bytes: usize,
}

fn checked_product(values: impl IntoIterator<Item = usize>, context: &str) -> AppResult<usize> {
    values.into_iter().try_fold(1_usize, |product, value| {
        product
            .checked_mul(value)
            .ok_or_else(|| AppError::Invalid(format!("{context} overflow")))
    })
}

fn checked_bytes(elements: usize, width: usize, context: &str) -> AppResult<usize> {
    elements
        .checked_mul(width)
        .ok_or_else(|| AppError::Invalid(format!("{context} byte size overflow")))
}

pub(crate) fn sampled_npy_bytes(count: [usize; 3], stride: [usize; 3]) -> AppResult<usize> {
    if count.contains(&0) || stride.contains(&0) {
        return Err(AppError::Invalid(
            "terrain sample counts and strides must be positive".into(),
        ));
    }
    let span = (count[2] - 1)
        .checked_mul(stride[2])
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| AppError::Invalid("terrain sample span overflow".into()))?;
    let row = if span <= count[2].saturating_mul(16) {
        span
    } else {
        count[2]
    };
    checked_bytes(
        checked_product([count[0], count[1], row], "terrain samples")?,
        8,
        "terrain samples",
    )
}

pub(crate) fn read_npy_samples(
    path: &Path,
    source_shape: [usize; 3],
    origin: [usize; 3],
    count: [usize; 3],
    stride: [usize; 3],
    cancel: &mut dyn FnMut() -> AppResult<()>,
) -> AppResult<Vec<f64>> {
    for axis in 0..3 {
        if count[axis] == 0
            || stride[axis] == 0
            || (count[axis] - 1)
                .checked_mul(stride[axis])
                .and_then(|n| n.checked_add(origin[axis]))
                .map_or(true, |n| n >= source_shape[axis])
        {
            return Err(AppError::Invalid(
                "terrain samples exceed source geometry".into(),
            ));
        }
    }
    let mut file = File::open(path)?;
    let version = (file.metadata()?.len(), file.metadata()?.modified()?);
    let offset = parse_f64_npy_header(&mut file, source_shape)?;
    let span = (count[2] - 1) * stride[2] + 1;
    let contiguous = span <= count[2].saturating_mul(16);
    let mut row = vec![0_u8; if contiguous { span * 8 } else { 8 }];
    let mut values = Vec::with_capacity(checked_product(count, "terrain samples")?);
    for i in 0..count[0] {
        for j in 0..count[1] {
            cancel()?;
            let base = ((origin[0] + i * stride[0]) * source_shape[1] + origin[1] + j * stride[1])
                * source_shape[2]
                + origin[2];
            if contiguous {
                file.seek(SeekFrom::Start((offset + base * 8) as u64))?;
                file.read_exact(&mut row)?;
            }
            for k in 0..count[2] {
                let start = if contiguous {
                    k * stride[2] * 8
                } else {
                    cancel()?;
                    file.seek(SeekFrom::Start(
                        (offset + (base + k * stride[2]) * 8) as u64,
                    ))?;
                    file.read_exact(&mut row)?;
                    0
                };
                values.push(f64::from_le_bytes(
                    row[start..start + 8].try_into().unwrap(),
                ));
            }
        }
    }
    if (file.metadata()?.len(), file.metadata()?.modified()?) != version {
        return Err(AppError::Invalid(
            "terrain geometry changed during sampled read".into(),
        ));
    }
    Ok(values)
}

pub(crate) fn parse_f64_npy_header(file: &mut File, expected_shape: [usize; 3]) -> AppResult<usize> {
    let file_len = usize::try_from(file.metadata()?.len())
        .map_err(|_| AppError::Invalid("terrain geometry file length does not fit usize".into()))?;
    let mut prefix = [0_u8; 10];
    file.read_exact(&mut prefix)?;
    if &prefix[..6] != b"\x93NUMPY" || prefix[6] != 1 || prefix[7] != 0 {
        return Err(AppError::Invalid(
            "terrain geometry must be a NumPy v1.0 array".into(),
        ));
    }
    let header_len = usize::from(u16::from_le_bytes([prefix[8], prefix[9]]));
    if header_len == 0 || header_len > crate::npy::MAX_NPY_HEADER_BYTES {
        return Err(AppError::Invalid(format!(
            "terrain geometry NPY header is {header_len} bytes"
        )));
    }
    let mut header = vec![0_u8; header_len];
    file.read_exact(&mut header)?;
    let header = std::str::from_utf8(&header)
        .map_err(|_| AppError::Invalid("terrain geometry NPY header is not UTF-8".into()))?;
    let compact: String = header
        .chars()
        .filter(|value| !value.is_whitespace())
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
        return Err(AppError::Invalid(
            "terrain geometry NPY contract does not match float64 C-order shape".into(),
        ));
    }
    let elements = checked_product(expected_shape, "terrain geometry shape")?;
    let data_offset = 10_usize
        .checked_add(header_len)
        .ok_or_else(|| AppError::Invalid("terrain geometry header offset overflow".into()))?;
    let expected_len = data_offset
        .checked_add(checked_bytes(
            elements,
            size_of::<f64>(),
            "terrain geometry",
        )?)
        .ok_or_else(|| AppError::Invalid("terrain geometry file size overflow".into()))?;
    if file_len != expected_len {
        return Err(AppError::Invalid(format!(
            "terrain geometry body size mismatch: need {expected_len} bytes, have {file_len}"
        )));
    }
    Ok(data_offset)
}

fn read_f64_npy_window(
    path: &Path,
    source_shape: [usize; 3],
    origin: [usize; 3],
    extent: [usize; 3],
    cancel_check: &mut dyn FnMut() -> AppResult<()>,
) -> AppResult<Vec<f64>> {
    for axis in 0..3 {
        let end = origin[axis]
            .checked_add(extent[axis])
            .ok_or_else(|| AppError::Invalid("terrain geometry window bound overflow".into()))?;
        if extent[axis] == 0 || end > source_shape[axis] {
            return Err(AppError::Invalid(format!(
                "terrain geometry window exceeds axis {axis}"
            )));
        }
    }
    let mut file = File::open(path)?;
    let data_offset = parse_f64_npy_header(&mut file, source_shape)?;
    let elements = checked_product(extent, "terrain geometry window")?;
    let mut result = Vec::with_capacity(elements);
    let row_bytes = checked_bytes(extent[2], size_of::<f64>(), "terrain geometry row")?;
    let mut row = vec![0_u8; row_bytes];
    for i in 0..extent[0] {
        for j in 0..extent[1] {
            cancel_check()?;
            let global_i = origin[0] + i;
            let global_j = origin[1] + j;
            let linear = global_i
                .checked_mul(source_shape[1])
                .and_then(|value| value.checked_add(global_j))
                .and_then(|value| value.checked_mul(source_shape[2]))
                .and_then(|value| value.checked_add(origin[2]))
                .ok_or_else(|| AppError::Invalid("terrain geometry index overflow".into()))?;
            let offset = data_offset
                .checked_add(checked_bytes(
                    linear,
                    size_of::<f64>(),
                    "terrain geometry offset",
                )?)
                .ok_or_else(|| AppError::Invalid("terrain geometry offset overflow".into()))?;
            file.seek(SeekFrom::Start(u64::try_from(offset).map_err(|_| {
                AppError::Invalid("terrain geometry offset does not fit u64".into())
            })?))?;
            file.read_exact(&mut row)?;
            for bytes in row.chunks_exact(size_of::<f64>()) {
                result.push(f64::from_le_bytes(bytes.try_into().map_err(|_| {
                    AppError::Invalid("truncated terrain geometry value".into())
                })?));
            }
        }
    }
    debug_assert_eq!(result.len(), elements);
    Ok(result)
}

pub(crate) struct TerrainLodGeometryWindow {
    pub x_m: Vec<f64>,
    pub y_m: Vec<f64>,
    pub z_m: Vec<f64>,
    pub volumes_m3: Vec<f64>,
    pub source_bytes: usize,
}

pub(crate) fn read_npy_lod_geometry_window(
    paths: [&Path; 4],
    source_shape: [usize; 3],
    origin: [usize; 3],
    extent: [usize; 3],
    cancel_check: &mut dyn FnMut() -> AppResult<()>,
) -> AppResult<TerrainLodGeometryWindow> {
    let x_m = read_f64_npy_window(paths[0], source_shape, origin, extent, cancel_check)?;
    let y_m = read_f64_npy_window(paths[1], source_shape, origin, extent, cancel_check)?;
    let z_m = read_f64_npy_window(paths[2], source_shape, origin, extent, cancel_check)?;
    let volumes_m3 = read_f64_npy_window(paths[3], source_shape, origin, extent, cancel_check)?;
    if x_m.iter().chain(&y_m).chain(&z_m).any(|value| !value.is_finite())
        || volumes_m3.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return Err(AppError::Invalid(
            "terrain LOD geometry contains invalid centroids or volumes".into(),
        ));
    }
    let source_bytes = checked_bytes(
        x_m.len().checked_mul(4)
            .ok_or_else(|| AppError::Invalid("terrain LOD geometry size overflow".into()))?,
        size_of::<f64>(),
        "terrain LOD geometry source",
    )?;
    Ok(TerrainLodGeometryWindow { x_m, y_m, z_m, volumes_m3, source_bytes })
}

pub(crate) fn read_npy_interface_window(
    step_dir: &Path,
    source_shape: [usize; 3],
    origin: [usize; 3],
    extent: [usize; 3],
) -> AppResult<(Vec<f64>, usize)> {
    let interface_nz = source_shape[2]
        .checked_add(1)
        .ok_or_else(|| AppError::Invalid("terrain interface shape overflow".into()))?;
    let mut never_cancel = || Ok(());
    let values = read_f64_npy_window(
        &step_dir.join("physical_z_interface_m.npy"),
        [source_shape[0], source_shape[1], interface_nz],
        origin,
        extent,
        &mut never_cancel,
    )?;
    let bytes = checked_bytes(values.len(), size_of::<f64>(), "terrain interface source")?;
    Ok((values, bytes))
}

pub(crate) fn read_npy_geometry_window(
    step_dir: &Path,
    source_shape: [usize; 3],
    origin: [usize; 3],
    extent: [usize; 3],
    cancel_check: &mut dyn FnMut() -> AppResult<()>,
) -> AppResult<TerrainGeometryWindow> {
    let interface_nz = source_shape[2]
        .checked_add(1)
        .ok_or_else(|| AppError::Invalid("terrain interface shape overflow".into()))?;
    let interface_extent_z = extent[2]
        .checked_add(1)
        .ok_or_else(|| AppError::Invalid("terrain interface window overflow".into()))?;
    let centers_m = read_f64_npy_window(
        &step_dir.join("physical_z_center_m.npy"),
        source_shape,
        origin,
        extent,
        cancel_check,
    )?;
    let interfaces_m = read_f64_npy_window(
        &step_dir.join("physical_z_interface_m.npy"),
        [source_shape[0], source_shape[1], interface_nz],
        origin,
        [extent[0], extent[1], interface_extent_z],
        cancel_check,
    )?;
    let source_bytes = checked_bytes(
        centers_m
            .len()
            .checked_add(interfaces_m.len())
            .ok_or_else(|| AppError::Invalid("terrain geometry source size overflow".into()))?,
        size_of::<f64>(),
        "terrain geometry source",
    )?;
    Ok(TerrainGeometryWindow {
        centers_m,
        interfaces_m,
        source_bytes,
    })
}

pub(crate) fn aggregation_working_bytes(output_elements: usize) -> AppResult<usize> {
    // weighted field sum + returned volume + x/y/z moments, followed by
    // returned float32 values and three float64 centroid components. The
    // volume accumulator becomes the returned volume buffer, so publishing it
    // does not add another allocation to this accounting. One byte per block
    // records empty/valid/invalid state; extrema cannot use a zero or NaN
    // sentinel without corrupting all-positive minima or allowing a later
    // finite value to revive a block that already contained missing data.
    let per_output = 5_usize
        .checked_mul(size_of::<f64>())
        .and_then(|value| value.checked_add(size_of::<f32>()))
        .and_then(|value| value.checked_add(3 * size_of::<f64>()))
        .and_then(|value| value.checked_add(size_of::<u8>()))
        .ok_or_else(|| AppError::Invalid("terrain aggregation element size overflow".into()))?;
    checked_bytes(output_elements, per_output, "terrain aggregation")
}

// The geometry and sampling vectors form one validated aggregation boundary.
#[allow(clippy::too_many_arguments)]
pub(crate) fn aggregate_physical_volume(
    source: &[f32],
    origin: [usize; 3],
    extent: [usize; 3],
    stride: [usize; 3],
    horizontal_origin_m: [f64; 2],
    spacing_m: [f64; 2],
    geometry: &TerrainGeometryWindow,
    aggregation: &str,
) -> AppResult<TerrainAggregation> {
    if extent.contains(&0)
        || stride.contains(&0)
        || !horizontal_origin_m.iter().all(|value| value.is_finite())
        || !spacing_m
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
    {
        return Err(AppError::Invalid(
            "terrain aggregation received invalid shape or horizontal geometry".into(),
        ));
    }
    if !matches!(
        aggregation,
        "cell_mean" | "linear_reflectivity_mean" | "block_max" | "block_min" | "magnitude_extreme" | "point_sample"
    ) {
        return Err(AppError::Invalid(
            "unsupported terrain field-window aggregation".into(),
        ));
    }
    let source_elements = checked_product(extent, "terrain source window")?;
    let interface_extent_z = extent[2]
        .checked_add(1)
        .ok_or_else(|| AppError::Invalid("terrain interface window overflow".into()))?;
    let interface_elements = checked_product(
        [extent[0], extent[1], interface_extent_z],
        "terrain interface window",
    )?;
    if source.len() != source_elements
        || geometry.centers_m.len() != source_elements
        || geometry.interfaces_m.len() != interface_elements
    {
        return Err(AppError::Invalid(
            "terrain field and geometry window shapes disagree".into(),
        ));
    }
    let output_shape = (0..3)
        .map(|axis| (extent[axis] - 1) / stride[axis] + 1)
        .collect::<Vec<_>>();
    let output_elements = checked_product(output_shape.iter().copied(), "terrain output window")?;
    let mut field_sum = vec![0_f64; output_elements];
    // 0 = empty, 1 = at least one finite field value, 2 = invalid because the
    // block contains a non-finite source value.
    let mut field_state = vec![0_u8; output_elements];
    let mut volume_sum = vec![0_f64; output_elements];
    let mut x_sum = vec![0_f64; output_elements];
    let mut y_sum = vec![0_f64; output_elements];
    let mut z_sum = vec![0_f64; output_elements];
    let interface_nz = interface_extent_z;
    for i in 0..extent[0] {
        let x_m = horizontal_origin_m[0] + (origin[0] + i) as f64 * spacing_m[0];
        for j in 0..extent[1] {
            let y_m = horizontal_origin_m[1] + (origin[1] + j) as f64 * spacing_m[1];
            for k in 0..extent[2] {
                let source_index = (i * extent[1] + j) * extent[2] + k;
                let face_index = (i * extent[1] + j) * interface_nz + k;
                let lower = geometry.interfaces_m[face_index];
                let upper = geometry.interfaces_m[face_index + 1];
                let center = geometry.centers_m[source_index];
                if !lower.is_finite()
                    || !upper.is_finite()
                    || !center.is_finite()
                    || upper <= lower
                    || center <= lower
                    || center >= upper
                {
                    return Err(AppError::Invalid(format!(
                        "malformed terrain geometry at local cell ({i},{j},{k})"
                    )));
                }
                let volume = spacing_m[0] * spacing_m[1] * (upper - lower);
                if !volume.is_finite() || volume <= 0.0 {
                    return Err(AppError::Invalid(format!(
                        "non-positive terrain cell volume at local cell ({i},{j},{k})"
                    )));
                }
                if aggregation == "point_sample"
                    && (i % stride[0] != 0 || j % stride[1] != 0 || k % stride[2] != 0)
                {
                    continue;
                }
                let oi = i / stride[0];
                let oj = j / stride[1];
                let ok = k / stride[2];
                let output_index = (oi * output_shape[1] + oj) * output_shape[2] + ok;
                volume_sum[output_index] += volume;
                x_sum[output_index] += volume * x_m;
                y_sum[output_index] += volume * y_m;
                z_sum[output_index] += volume * center;
                let value = source[source_index];
                if value.is_finite() {
                    if field_state[output_index] == 2 {
                        continue;
                    }
                    let numeric = f64::from(value);
                    match aggregation {
                        "block_max"
                            if field_state[output_index] == 0
                                || numeric > field_sum[output_index] =>
                        {
                            field_sum[output_index] = numeric
                        }
                        "block_min"
                            if field_state[output_index] == 0
                                || numeric < field_sum[output_index] =>
                        {
                            field_sum[output_index] = numeric
                        }
                        "magnitude_extreme"
                            if field_state[output_index] == 0
                                || numeric.abs() > field_sum[output_index].abs() =>
                        {
                            field_sum[output_index] = numeric
                        }
                        "linear_reflectivity_mean" => {
                            field_sum[output_index] += volume * 10_f64.powf(numeric / 10.0)
                        }
                        "cell_mean" | "point_sample" => field_sum[output_index] += volume * numeric,
                        _ => {}
                    }
                    field_state[output_index] = 1;
                } else {
                    field_state[output_index] = 2;
                }
            }
        }
    }
    let mut values = Vec::with_capacity(output_elements);
    let centroid_elements = output_elements
        .checked_mul(3)
        .ok_or_else(|| AppError::Invalid("terrain centroid element count overflow".into()))?;
    let mut centroids_m = vec![0_f64; centroid_elements];
    for index in 0..output_elements {
        let volume = volume_sum[index];
        if !volume.is_finite() || volume <= 0.0 {
            return Err(AppError::Invalid(
                "terrain aggregation produced an empty/non-finite block".into(),
            ));
        }
        if field_state[index] != 1 {
            values.push(f32::NAN);
        } else {
            let linear_mean = field_sum[index] / volume;
            let mean = if matches!(aggregation, "block_max" | "block_min" | "magnitude_extreme") {
                field_sum[index]
            } else if aggregation == "linear_reflectivity_mean" {
                10.0 * linear_mean.log10()
            } else {
                linear_mean
            };
            if !mean.is_finite() || mean.abs() > f64::from(f32::MAX) {
                return Err(AppError::Invalid(
                    "terrain aggregation produced an unrepresentable field mean".into(),
                ));
            }
            values.push(mean as f32);
        }
        centroids_m[index] = x_sum[index] / volume;
        centroids_m[output_elements + index] = y_sum[index] / volume;
        centroids_m[2 * output_elements + index] = z_sum[index] / volume;
        if !centroids_m[index].is_finite()
            || !centroids_m[output_elements + index].is_finite()
            || !centroids_m[2 * output_elements + index].is_finite()
        {
            return Err(AppError::Invalid(
                "terrain aggregation produced a non-finite centroid".into(),
            ));
        }
    }
    Ok(TerrainAggregation {
        values,
        centroids_m,
        volumes_m3: volume_sum,
        output_shape,
        working_bytes: aggregation_working_bytes(output_elements)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn volume_weighted_blocks_use_exact_interfaces_and_centers() {
        let geometry = TerrainGeometryWindow {
            centers_m: vec![0.5, 2.0, 1.0, 3.0],
            interfaces_m: vec![0.0, 1.0, 3.0, 0.0, 2.0, 4.0],
            source_bytes: 80,
        };
        let result = aggregate_physical_volume(
            &[10.0, 20.0, 30.0, 50.0],
            [2, 0, 0],
            [2, 1, 2],
            [2, 1, 2],
            [50.0, 75.0],
            [100.0, 200.0],
            &geometry,
            "cell_mean",
        )
        .unwrap();
        assert_eq!(result.output_shape, vec![1, 1, 1]);
        // Volumes are proportional to [1,2,2,2].
        assert_eq!(result.values, vec![30.0]);
        assert_eq!(result.volumes_m3, vec![140_000.0]);
        assert!((result.centroids_m[0] - (250.0 * 3.0 + 350.0 * 4.0) / 7.0).abs() < 1e-12);
        assert_eq!(result.centroids_m[1], 75.0);
        assert!((result.centroids_m[2] - 12.5 / 7.0).abs() < 1e-12);
    }

    #[test]
    fn non_finite_field_invalidates_only_its_block_and_geometry_fails_closed() {
        let mut geometry = TerrainGeometryWindow {
            centers_m: vec![0.5, 1.5],
            interfaces_m: vec![0.0, 1.0, 2.0],
            source_bytes: 40,
        };
        let result = aggregate_physical_volume(
            &[f32::NAN, 2.0],
            [0, 0, 0],
            [1, 1, 2],
            [1, 1, 1],
            [0.5, 0.5],
            [1.0, 1.0],
            &geometry,
            "cell_mean",
        )
        .unwrap();
        assert!(result.values[0].is_nan());
        assert_eq!(result.values[1], 2.0);

        geometry.interfaces_m[2] = 0.5;
        assert!(aggregate_physical_volume(
            &[1.0, 2.0],
            [0, 0, 0],
            [1, 1, 2],
            [1, 1, 1],
            [0.5, 0.5],
            [1.0, 1.0],
            &geometry,
            "cell_mean",
        )
        .is_err());
    }

    #[test]
    fn terrain_extrema_do_not_use_zero_as_an_empty_block_sentinel() {
        let geometry = TerrainGeometryWindow {
            centers_m: vec![0.5, 1.5],
            interfaces_m: vec![0.0, 1.0, 2.0],
            source_bytes: 40,
        };
        let aggregate = |source: &[f32], aggregation: &str| {
            aggregate_physical_volume(
                source,
                [0, 0, 0],
                [1, 1, 2],
                [1, 1, 2],
                [0.5, 0.5],
                [1.0, 1.0],
                &geometry,
                aggregation,
            )
            .unwrap()
            .values[0]
        };
        assert_eq!(aggregate(&[4.0, 7.0], "block_min"), 4.0);
        assert_eq!(aggregate(&[-7.0, -4.0], "block_max"), -4.0);
        assert_eq!(aggregate(&[4.0, -7.0], "magnitude_extreme"), -7.0);
        assert!(aggregate(&[f32::NAN, 7.0], "block_max").is_nan());
    }

    #[test]
    fn point_samples_return_only_selected_cells_with_exact_geometry() {
        let geometry = TerrainGeometryWindow {
            centers_m: vec![0.5, 1.5, 3.0, 5.0],
            interfaces_m: vec![0.0, 1.0, 2.0, 0.0, 4.0, 6.0],
            source_bytes: 80,
        };
        let result = aggregate_physical_volume(
            &[10.0, 20.0, 30.0, 40.0],
            [4, 2, 0],
            [2, 1, 2],
            [1, 1, 2],
            [100.0, 200.0],
            [10.0, 20.0],
            &geometry,
            "point_sample",
        )
        .unwrap();
        assert_eq!(result.output_shape, vec![2, 1, 1]);
        assert_eq!(result.values, vec![10.0, 30.0]);
        assert_eq!(result.centroids_m, vec![140.0, 150.0, 240.0, 240.0, 0.5, 3.0]);
        assert_eq!(result.volumes_m3, vec![200.0, 800.0]);
    }
}
