use super::*;
use std::collections::VecDeque;

pub struct SampledFieldWindow {
    pub source_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub bytes: Vec<u8>,
    pub source_working_bytes: usize,
}

#[derive(Clone)]
struct AxisBricks {
    origin: usize,
    stride: usize,
    brick_size: usize,
    samples: usize,
    next: usize,
}

impl Iterator for AxisBricks {
    type Item = (usize, std::ops::Range<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.samples {
            return None;
        }
        let begin = self.next;
        let brick = (self.origin + begin * self.stride) / self.brick_size;
        let boundary = (brick + 1) * self.brick_size;
        self.next = ((boundary - self.origin - 1) / self.stride + 1).min(self.samples);
        Some((brick, begin..self.next))
    }
}

struct PlannedFrame {
    path: PathBuf,
    header: Zfp3dHeader,
    file_len: u64,
    modified: Option<std::time::SystemTime>,
}

fn cached_file<'a>(
    cache: &'a mut VecDeque<(usize, std::fs::File)>,
    frames: &[PlannedFrame],
    index: usize,
) -> AppResult<&'a mut std::fs::File> {
    if let Some(position) = cache.iter().position(|entry| entry.0 == index) {
        let entry = cache.remove(position).unwrap();
        cache.push_back(entry);
    } else {
        let frame = &frames[index];
        let mut file = std::fs::File::open(&frame.path)?;
        let (bytes, count) = read_header_prefix(&mut file)?;
        let (header, _) = parse_header(&bytes[..count])?;
        if header != frame.header || file.metadata()?.len() != frame.file_len {
            return Err(AppError::Invalid(
                "ZFP artifact changed during sampled read".into(),
            ));
        }
        if cache.len() == 8 {
            cache.pop_front();
        }
        cache.push_back((index, file));
    }
    let file = &mut cache.back_mut().unwrap().1;
    let metadata = file.metadata()?;
    if metadata.len() != frames[index].file_len
        || metadata.modified().ok() != frames[index].modified
    {
        return Err(AppError::Invalid(
            "ZFP artifact changed during sampled read".into(),
        ));
    }
    Ok(file)
}

/// The callback runs between bounded operations; true marks one indivisible libzfp brick decode.
/// Legacy monolithic containers return None so their ordinary admission path remains authoritative.
#[allow(clippy::too_many_arguments)]
pub fn read_sampled_field_window<F>(
    run_dir: &str,
    step_index: u32,
    field: &str,
    origin: [usize; 3],
    extent: [usize; 3],
    stride: [usize; 3],
    source_budget: usize,
    output_budget: usize,
    expected_shape: Option<[usize; 3]>,
    mut check: F,
) -> AppResult<Option<SampledFieldWindow>>
where
    F: FnMut(bool) -> AppResult<()>,
{
    check(false)?;
    validate_field_name(field)?;
    let target_path = zfp3d_path(run_dir, step_index, field);
    let target = read_header_only(&target_path)?;
    if !target.is_bricked() {
        return Ok(None);
    }
    if expected_shape.is_some_and(|expected| target.shape() != expected) {
        return Err(AppError::Invalid(
            "ZFP shape disagrees with completion manifest".into(),
        ));
    }
    validate_window(&target.shape(), origin, extent)?;
    if stride.contains(&0) || source_budget == 0 || output_budget == 0 {
        return Err(AppError::Invalid(
            "Invalid sampled ZFP stride or budget".into(),
        ));
    }
    let output_shape = std::array::from_fn::<_, 3, _>(|axis| (extent[axis] - 1) / stride[axis] + 1);
    let output_bytes = checked_product(output_shape, "sampled ZFP output")?
        .checked_mul(4)
        .ok_or_else(|| AppError::Invalid("Sampled ZFP output size overflow".into()))?;
    if output_bytes > output_budget {
        return Err(AppError::Invalid(format!(
            "Sampled ZFP output needs {output_bytes} bytes; budget is {output_budget}"
        )));
    }
    let contract = manifest_frame_contract(run_dir, step_index, field)?;
    require_v4_manifest(&target, &contract)?;
    if contract
        .as_ref()
        .is_some_and(|value| !header_matches_manifest(&target, value))
    {
        return Err(AppError::Invalid(
            "ZFP target artifact/header role disagrees with its manifest".into(),
        ));
    }
    let chain = if target.is_keyframe() {
        vec![step_index]
    } else if contract.is_some() {
        replay_steps_from_contract(run_dir, step_index, field, false)?
    } else {
        replay_steps(run_dir, step_index, field, false)?
    };
    let brick_shape = target.brick_shape.unwrap();
    let counts = target.brick_counts.unwrap();
    let axes = std::array::from_fn::<_, 3, _>(|axis| AxisBricks {
        origin: origin[axis],
        stride: stride[axis],
        brick_size: brick_shape[axis],
        samples: output_shape[axis],
        next: 0,
    });
    let selected_counts = std::array::from_fn::<_, 3, _>(|axis| {
        if stride[axis] >= brick_shape[axis] {
            output_shape[axis]
        } else {
            (origin[axis] + (output_shape[axis] - 1) * stride[axis]) / brick_shape[axis]
                - origin[axis] / brick_shape[axis]
                + 1
        }
    });
    let selected_bricks = checked_product(selected_counts, "sampled ZFP bricks")?;
    let entry_bytes = target.brick_entry_bytes.unwrap_or(BRICK_ENTRY_LEN);
    let directory_bytes = selected_bricks
        .checked_mul(4 * entry_bytes)
        .and_then(|value| value.checked_add(2 * HEADER_V3_LEN))
        .and_then(|value| value.checked_mul(chain.len()))
        .ok_or_else(|| AppError::Invalid("Sampled ZFP directory budget overflow".into()))?;
    if directory_bytes > source_budget {
        return Err(AppError::Invalid(
            "Sampled ZFP directory exceeds the source budget".into(),
        ));
    }
    let mut frames = Vec::with_capacity(chain.len());
    let mut inspected = directory_bytes;
    let mut max_payload = 0usize;
    let mut max_decoded = 0usize;
    let mut max_samples = 0usize;
    for step in chain {
        check(false)?;
        let path = zfp3d_path(run_dir, step, field);
        let mut file = std::fs::File::open(&path)?;
        let metadata = file.metadata()?;
        let file_len = metadata.len();
        let modified = metadata.modified().ok();
        let (bytes, count) = read_header_prefix(&mut file)?;
        let (header, _) = parse_header(&bytes[..count])?;
        if header.shape() != target.shape()
            || header.brick_shape != target.brick_shape
            || header.brick_counts != target.brick_counts
            || header.version != target.version
        {
            return Err(AppError::Invalid(
                "delta .zfp3d brick geometry differs from its target".into(),
            ));
        }
        let frame_contract = manifest_frame_contract(run_dir, step, field)?;
        require_v4_manifest(&header, &frame_contract)?;
        if (step == step_index && header != target)
            || (frames.is_empty() && !header.is_keyframe())
            || (!frames.is_empty() && header.is_keyframe())
            || frame_contract
                .as_ref()
                .is_some_and(|value| !header_matches_manifest(&header, value))
        {
            return Err(AppError::Invalid(
                "ZFP replay contract changed during sampled read".into(),
            ));
        }
        validate_brick_container_layout(&header, file_len)?;
        for (bi, xs) in axes[0].clone() {
            for (bj, ys) in axes[1].clone() {
                for (bk, zs) in axes[2].clone() {
                    check(false)?;
                    let entry = read_brick_entry(
                        &mut file,
                        &header,
                        (bi * counts[1] + bj) * counts[2] + bk,
                        file_len,
                    )?;
                    inspected = inspected
                        .checked_add(entry.compressed_size)
                        .ok_or_else(|| {
                            AppError::Invalid("Sampled ZFP inspected-byte overflow".into())
                        })?;
                    if inspected > source_budget {
                        return Err(AppError::Invalid(format!(
                            "Sampled ZFP must inspect {inspected} bytes; budget is {source_budget}"
                        )));
                    }
                    max_payload = max_payload.max(entry.compressed_size);
                    max_decoded =
                        max_decoded.max(checked_product(entry.extent, "sampled ZFP brick")? * 4);
                    max_samples = max_samples.max(
                        checked_product(
                            [xs.len(), ys.len(), zs.len()],
                            "sampled ZFP brick points",
                        )? * 4,
                    );
                }
            }
        }
        frames.push(PlannedFrame {
            path,
            header,
            file_len,
            modified,
        });
    }
    let histories = if frames.len() == 1 { 1 } else { 3 };
    let working = output_bytes
        .checked_add(max_decoded)
        .and_then(|value| value.checked_add(max_samples.checked_mul(histories)?))
        .and_then(|value| value.checked_add(max_payload.checked_mul(2)?))
        .ok_or_else(|| AppError::Invalid("Sampled ZFP working-set overflow".into()))?;
    // Chains longer than the small file cache must reread a header when revisiting each brick.
    if frames.len() > 8 {
        inspected = inspected
            .checked_add(
                selected_bricks
                    .checked_mul(frames.len())
                    .and_then(|value| value.checked_mul(HEADER_V3_LEN))
                    .ok_or_else(|| {
                        AppError::Invalid("Sampled ZFP header budget overflow".into())
                    })?,
            )
            .ok_or_else(|| AppError::Invalid("Sampled ZFP inspected-byte overflow".into()))?;
    }
    let source_working_bytes = working.max(inspected);
    if source_working_bytes > source_budget {
        return Err(AppError::Invalid(format!("Sampled ZFP needs {source_working_bytes} inspected/working bytes; budget is {source_budget}")));
    }
    check(false)?;
    let mut output = vec![0u8; output_bytes];
    let mut cache = VecDeque::new();
    let mut actual_inspected = directory_bytes;
    for (bi, xs) in axes[0].clone() {
        for (bj, ys) in axes[1].clone() {
            for (bk, zs) in axes[2].clone() {
                let mut previous = Vec::new();
                let mut previous2 = Vec::new();
                let index = (bi * counts[1] + bj) * counts[2] + bk;
                for (frame_index, frame) in frames.iter().enumerate() {
                    check(false)?;
                    let file = cached_file(&mut cache, &frames, frame_index)?;
                    let entry = read_brick_entry(file, &frame.header, index, frame.file_len)?;
                    actual_inspected = actual_inspected
                        .checked_add(entry.compressed_size)
                        .ok_or_else(|| {
                            AppError::Invalid("Sampled ZFP inspected-byte overflow".into())
                        })?;
                    if entry.compressed_size > max_payload || actual_inspected > inspected {
                        return Err(AppError::Invalid(
                            "ZFP brick directory changed during sampled read".into(),
                        ));
                    }
                    file.seek(SeekFrom::Start(entry.payload_offset))?;
                    let mut compressed = vec![0u8; entry.compressed_size];
                    file.read_exact(&mut compressed)?;
                    verify_brick_bytes(&entry, &compressed)?;
                    check(true)?;
                    let decoded =
                        decompress_payload_dims(&frame.header, &compressed, entry.extent)?;
                    check(false)?;
                    let mut samples = Vec::with_capacity(xs.len() * ys.len() * zs.len());
                    for x in xs.clone() {
                        check(false)?;
                        let sx = origin[0] + x * stride[0] - entry.origin[0];
                        for y in ys.clone() {
                            let sy = origin[1] + y * stride[1] - entry.origin[1];
                            for z in zs.clone() {
                                let sz = origin[2] + z * stride[2] - entry.origin[2];
                                samples.push(
                                    decoded[(sx * entry.extent[1] + sy) * entry.extent[2] + sz],
                                );
                            }
                        }
                    }
                    apply_frame(
                        Frame {
                            flags: frame.header.flags,
                            data: samples,
                        },
                        &mut previous,
                        &mut previous2,
                    )?;
                }
                let mut sample = 0;
                for x in xs.clone() {
                    check(false)?;
                    for y in ys.clone() {
                        for z in zs.clone() {
                            let offset = ((x * output_shape[1] + y) * output_shape[2] + z) * 4;
                            output[offset..offset + 4]
                                .copy_from_slice(&previous[sample].to_le_bytes());
                            sample += 1;
                        }
                    }
                }
            }
        }
    }
    check(false)?;
    for frame in &frames {
        let metadata = std::fs::metadata(&frame.path)?;
        if metadata.len() != frame.file_len || metadata.modified().ok() != frame.modified {
            return Err(AppError::Invalid(
                "ZFP artifact changed during sampled read".into(),
            ));
        }
    }
    Ok(Some(SampledFieldWindow {
        source_shape: target.shape(),
        output_shape: output_shape.to_vec(),
        bytes: output,
        source_working_bytes,
    }))
}
