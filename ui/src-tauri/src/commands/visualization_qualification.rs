use crate::error::{AppError, AppResult};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Manager};
use tauri_plugin_fs::FsExt;

const OUTPUT_ENV: &str = "NIMBUS_STUDIO_VIS_QUALIFICATION_OUTPUT";
const PROFILE_ENV: &str = "NIMBUS_STUDIO_VIS_QUALIFICATION_PROFILE";
const REPORT_SCHEMA: &str = "nimbus.studio.packaged-visualization-qualification";
const CAPACITY_SCHEMA: &str = "nimbus.studio.visualization-capacity-qualification";
const CAPACITY_ARTIFACT_MAX_BYTES: u64 = 64 * 1024;
const BEAST_ONE_PANE_OBSERVED_COMMITTED_FLOOR_BYTES: u64 = 422_363_312;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VisualizationQualificationRequest {
    schema: &'static str,
    schema_version: u32,
    profile: String,
    python: bool,
    config_preflight: Option<Value>,
    visual_load: Option<Value>,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ConfigQualificationBatch {
    schema: String,
    binary_sha256: String,
    #[serde(default)]
    execution_target: Option<super::remote_compute::RemoteComputeTarget>,
    cases: Vec<ConfigQualificationCase>,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ConfigQualificationCase {
    name: String,
    expected_acceptance: bool,
    request: super::config_preflight::ConfigPreflightRequest,
}

fn parse_config_qualification(bytes: &[u8]) -> AppResult<Value> {
    if bytes.len() > 8 * 1024 * 1024 {
        return Err(AppError::Invalid("configuration qualification batch exceeds 8 MiB".into()));
    }
    let batch: ConfigQualificationBatch = serde_json::from_slice(bytes)
        .map_err(|error| AppError::Invalid(format!("invalid configuration qualification batch: {error}")))?;
    let mut names = BTreeSet::new();
    if batch.schema != "nimbus.studio.packaged-config-qualification.v1"
        || batch.binary_sha256.len() != 64
        || !batch.binary_sha256.bytes().all(|c| c.is_ascii_hexdigit())
        || batch.cases.is_empty() || batch.cases.len() > 128
        || batch.cases.iter().any(|case| case.name.trim().is_empty()
            || !names.insert(&case.name) || case.request.source.kind != "exact_document")
    {
        return Err(AppError::Invalid("configuration qualification requires unique exact-document cases and a binary SHA-256".into()));
    }
    serde_json::to_value(batch).map_err(|error| AppError::Invalid(error.to_string()))
}

fn config_qualification_input() -> Option<Value> {
    std::env::var_os("NIMBUS_STUDIO_CONFIG_QUALIFICATION_INPUT")
        .filter(|path| !path.is_empty())
        .map(|path| {
            let loaded = (|| {
                let file = std::fs::File::open(path)?;
                use std::io::Read;
                let mut bytes = Vec::new();
                file.take(8 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
                parse_config_qualification(&bytes)
            })();
            loaded.unwrap_or_else(|error: AppError| serde_json::json!({"error": error.to_string()}))
        })
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VisualizationBrickFixture {
    run_dir: String,
    step_index: u32,
    publication_generation: String,
    scalar_field: String,
    vector_fields: [String; 3],
    terrain_following: bool,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn qualification_fixture_path() -> AppResult<PathBuf> {
    let output = output_path().ok_or_else(|| {
        AppError::Invalid("packaged visualization qualification was not requested".into())
    })?;
    let parent = output.parent().ok_or_else(|| {
        AppError::Invalid("visualization qualification output needs a parent directory".into())
    })?;
    Ok(parent.join(format!(
        ".nimbus-packaged-brick-fixture-{}",
        std::process::id(),
    )))
}

fn encode_npy_f64(shape: [usize; 3], values: &[f64]) -> AppResult<Vec<u8>> {
    let expected = shape
        .into_iter()
        .try_fold(1_usize, |count, extent| count.checked_mul(extent))
        .ok_or_else(|| {
            AppError::Invalid("terrain qualification geometry shape overflows".into())
        })?;
    if values.len() != expected {
        return Err(AppError::Invalid(
            "terrain qualification geometry size mismatch".into(),
        ));
    }
    let dict = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': ({}, {}, {}), }}",
        shape[0], shape[1], shape[2],
    );
    let mut header = dict.into_bytes();
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
    Ok(bytes)
}

fn fixture_terrain_geometry(shape: [usize; 3]) -> (Vec<f64>, Vec<f64>) {
    let [nx, ny, nz] = shape;
    let mut centers = Vec::with_capacity(nx * ny * nz);
    let mut interfaces = Vec::with_capacity(nx * ny * (nz + 1));
    for i in 0..nx {
        for j in 0..ny {
            let terrain = 30.0
                + 18.0
                    * (std::f64::consts::TAU * i as f64 / nx as f64).sin()
                    * (std::f64::consts::TAU * j as f64 / ny as f64).cos();
            let column: Vec<f64> = (0..=nz)
                .map(|k| terrain + (800.0 - terrain) * k as f64 / nz as f64)
                .collect();
            interfaces.extend_from_slice(&column);
            centers.extend((0..nz).map(|k| 0.5 * (column[k] + column[k + 1])));
        }
    }
    (centers, interfaces)
}

fn fixture_field_values(field: &str, shape: [usize; 3]) -> Vec<f32> {
    let [nx, ny, nz] = shape;
    let mut values = Vec::with_capacity(nx * ny * nz);
    for i in 0..nx {
        let x = (i as f32 - (nx - 1) as f32 * 0.5) / nx as f32;
        for j in 0..ny {
            let y = (j as f32 - (ny - 1) as f32 * 0.5) / ny as f32;
            for k in 0..nz {
                let z = k as f32 / (nz - 1) as f32;
                let value = match field {
                    "scalar" => (-18.0 * (x * x + y * y + (z - 0.45).powi(2))).exp() - 0.12,
                    "u" => -18.0 * y * (1.0 - 0.4 * z),
                    "v" => 18.0 * x * (1.0 - 0.4 * z),
                    "w" => 1.5 + 4.0 * (-24.0 * (x * x + y * y)).exp() * (1.0 - z),
                    _ => 0.0,
                };
                values.push(value);
            }
        }
    }
    values
}

fn fixture_hierarchy(
    field: &str,
    generation: &str,
    values: &[f32],
    shape: [usize; 3],
    brick_shape: [usize; 3],
) -> Value {
    let [storage_nx, storage_ny, storage_nz] = shape;
    let source_shape = [storage_nx - 2, storage_ny - 2, storage_nz];
    let source_offset = [1_usize, 1_usize, 0_usize];
    let [nx, ny, nz] = source_shape;
    let mut bricks = Vec::new();
    for bi in 0..nx.div_ceil(brick_shape[0]) {
        for bj in 0..ny.div_ceil(brick_shape[1]) {
            for bk in 0..nz.div_ceil(brick_shape[2]) {
                let origin = [
                    bi * brick_shape[0],
                    bj * brick_shape[1],
                    bk * brick_shape[2],
                ];
                let extent = [
                    brick_shape[0].min(nx - origin[0]),
                    brick_shape[1].min(ny - origin[1]),
                    brick_shape[2].min(nz - origin[2]),
                ];
                let mut minimum = f32::INFINITY;
                let mut maximum = f32::NEG_INFINITY;
                for i in origin[0]..origin[0] + extent[0] {
                    for j in origin[1]..origin[1] + extent[1] {
                        for k in origin[2]..origin[2] + extent[2] {
                            let source_i = i + source_offset[0];
                            let source_j = j + source_offset[1];
                            let source_k = k + source_offset[2];
                            let value =
                                values[(source_i * storage_ny + source_j) * storage_nz + source_k];
                            minimum = minimum.min(value);
                            maximum = maximum.max(value);
                        }
                    }
                }
                let id = format!("l0:{bi}:{bj}:{bk}");
                bricks.push(serde_json::json!({
                    "id": id,
                    "index": [bi, bj, bk],
                    "origin": origin,
                    "extent": extent,
                    "scalarRange": {
                        "minimum": minimum,
                        "maximum": maximum,
                        "finiteCount": extent.into_iter().product::<usize>(),
                        "nonFiniteCount": 0
                    },
                    "physicalBoundsM": {
                        "lower": [
                            (origin[0] + source_offset[0]) as f64 * 100.0,
                            (origin[1] + source_offset[1]) as f64 * 100.0,
                            (origin[2] + source_offset[2]) as f64 * 100.0
                        ],
                        "upper": [
                            (origin[0] + source_offset[0] + extent[0]) as f64 * 100.0,
                            (origin[1] + source_offset[1] + extent[1]) as f64 * 100.0,
                            (origin[2] + extent[2]) as f64 * 100.0
                        ]
                    },
                    "contentSha256": sha256_hex(id.as_bytes())
                }));
            }
        }
    }
    serde_json::json!({
        "schema": "nimbus.field-brick-hierarchy",
        "schemaVersion": 3,
        "field": field,
        "publicationGeneration": generation,
        "sourceShape": source_shape,
        "sourceIndexOffset": source_offset,
        "sourceDtype": "float32",
        "sampleLocation": "cell_center",
        "extremaSemantics": "stored_values",
        "extremaAbsolutePadding": 0.0,
        "cullingSafe": true,
        "levels": [{
            "level": 0,
            "stride": [1, 1, 1],
            "shape": source_shape,
            "brickShape": brick_shape,
            "source": {"kind": "primary_field"},
            "bricks": bricks
        }]
    })
}

#[tauri::command]
pub fn create_visualization_brick_qualification_fixture(app: AppHandle) -> AppResult<VisualizationBrickFixture> {
    let run = qualification_fixture_path()?;
    if run.exists() {
        std::fs::remove_dir_all(&run)?;
    }
    let step_index = 0_u32;
    let generation = "cccccccccccccccccccccccccccccccc";
    let shape = [16_usize, 16_usize, 8_usize];
    let brick_shape = [8_usize, 8_usize, 4_usize];
    let step = run.join("step_000000");
    std::fs::create_dir_all(&step)?;
    if std::env::var("NIMBUS_STUDIO_PYTHON_CHECK").as_deref() == Ok("1") {
        app.fs_scope().allow_file(run.join("native-exact.npy"))
            .map_err(|error| AppError::Invalid(format!("native Python fixture scope: {error}")))?;
    }
    std::fs::write(
        run.join("nimbus_output.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema": "nimbus.output",
            "manifest_version": 5,
            "publication_generation": generation,
            "publication_state": "ready",
            "frame_completion": "atomic_step_manifest"
        }))
        .map_err(|error| AppError::Invalid(error.to_string()))?,
    )?;

    let fields = ["scalar", "u", "v", "w"];
    let mut field_descriptors = Vec::new();
    for field in fields {
        let values = fixture_field_values(field, shape);
        let npy = crate::zfp3d::encode_npy_f32(&shape, &values)?;
        std::fs::write(step.join(format!("{field}.npy")), npy)?;
        let hierarchy = fixture_hierarchy(field, generation, &values, shape, brick_shape);
        let hierarchy_bytes =
            serde_json::to_vec(&hierarchy).map_err(|error| AppError::Invalid(error.to_string()))?;
        std::fs::write(step.join(format!("{field}.bricks.json")), &hierarchy_bytes)?;
        field_descriptors.push(serde_json::json!({
            "field_id": field,
            "units": if field == "scalar" { "1" } else { "m s-1" },
            "grid_location": "cell_center",
            "state_semantic": "total_state",
            "cm1_aliases": [],
            "brick_hierarchy": {
                "schema": "nimbus.field-brick-hierarchy",
                "schema_version": 3,
                "path": format!("{field}.bricks.json"),
                "content_sha256": sha256_hex(&hierarchy_bytes),
                "native_brick_shape": brick_shape,
                "extrema_semantics": "stored_values",
                "extrema_absolute_padding": 0.0,
                "culling_safe": true
            }
        }));
    }
    let (terrain_centers, terrain_interfaces) = fixture_terrain_geometry(shape);
    std::fs::write(
        step.join("physical_z_center_m.npy"),
        encode_npy_f64(shape, &terrain_centers)?,
    )?;
    std::fs::write(
        step.join("physical_z_interface_m.npy"),
        encode_npy_f64([shape[0], shape[1], shape[2] + 1], &terrain_interfaces)?,
    )?;
    let manifest = serde_json::json!({
        "schema": "nimbus.step",
        "manifest_version": 5,
        "publication_state": "complete",
        "publication_generation": generation,
        "step_index": step_index,
        "simulation_time_s": 0.0,
        "step_dir": "step_000000",
        "data_product": {
            "kind": "history",
            "restart_complete": false,
            "storage_format": "npy_3d"
        },
        "field_identity": {
            "canonical_schema": "nimbus.fields",
            "canonical_version": 1,
            "cm1_alias_schema": "cm1.history",
            "cm1_alias_version": 1
        },
        "grid": {
            "coordinate_system": "cartesian",
            "axis_order": ["x", "y", "z"],
            "dimensions": {"x": shape[0], "y": shape[1], "z": shape[2]},
            "spacing_m": {"x": 100.0, "y": 100.0, "z": 100.0},
            "native_model_staggering": "c_grid",
            "publication_staggering": "cell_centered",
            "horizontal_scalar_origin_m": [50.0, 50.0],
            "vertical_geometry": {
                "kind": "terrain_following",
                "center_coordinate": "physical_z_center_m",
                "interface_coordinate": "physical_z_interface_m",
                "dtype": "float64",
                "layout": "x_y_z",
                "storage": "npy_sidecar"
            },
            "physical_bounds_m": {"lower": [100.0, 100.0, 0.0], "upper": [1500.0, 1500.0, 800.0]},
            "valid_index_range": {"lower": [1, 1, 0], "upper": [14, 14, 7]},
            "halo_width": {"lower": [1, 1, 0], "upper": [1, 1, 0]}
        },
        "published_fields": fields,
        "fields": field_descriptors
    });
    std::fs::write(
        step.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)
            .map_err(|error| AppError::Invalid(error.to_string()))?,
    )?;
    Ok(VisualizationBrickFixture {
        run_dir: run.to_string_lossy().into_owned(),
        step_index,
        publication_generation: generation.into(),
        scalar_field: "scalar".into(),
        vector_fields: ["u".into(), "v".into(), "w".into()],
        terrain_following: true,
    })
}

#[tauri::command]
pub fn remove_visualization_brick_qualification_fixture() -> AppResult<()> {
    let run = qualification_fixture_path()?;
    if run.exists() {
        std::fs::remove_dir_all(run)?;
    }
    Ok(())
}

fn output_path() -> Option<PathBuf> {
    std::env::var_os(OUTPUT_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn progress_path(path: &Path) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(".progress.ndjson");
    PathBuf::from(value)
}

fn append_progress(path: &Path, stage: &str) -> AppResult<()> {
    if stage.is_empty()
        || stage.len() > 128
        || !stage
            .bytes()
            .all(|value| value.is_ascii_alphanumeric() || b"-:_".contains(&value))
    {
        return Err(AppError::Invalid(
            "visualization qualification progress stage is invalid".into(),
        ));
    }
    let path = progress_path(path);
    let parent = path.parent().ok_or_else(|| {
        AppError::Invalid("visualization qualification progress needs a parent directory".into())
    })?;
    std::fs::create_dir_all(parent)?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let event = serde_json::json!({ "stage": stage, "pid": std::process::id() });
    serde_json::to_writer(&mut file, &event).map_err(|error| {
        AppError::Invalid(format!(
            "could not serialize qualification progress: {error}"
        ))
    })?;
    file.write_all(b"\n")?;
    file.sync_data()?;
    Ok(())
}

#[tauri::command]
pub fn visualization_qualification_request() -> Option<VisualizationQualificationRequest> {
    output_path().map(|_| VisualizationQualificationRequest {
        schema: "nimbus.studio.packaged-visualization-qualification-request",
        schema_version: 1,
        profile: std::env::var(PROFILE_ENV).unwrap_or_else(|_| "smoke".into()),
        python: std::env::var("NIMBUS_STUDIO_PYTHON_CHECK").as_deref() == Ok("1"),
        config_preflight: config_qualification_input(),
        visual_load: std::env::var_os("NIMBUS_STUDIO_VIS_LOAD_DIRECTORY").map(|directory| {
            let path = PathBuf::from(directory);
            let loaded = (|| -> AppResult<Value> {
                let path = path.canonicalize()?;
                let bytes = std::fs::read(path.join("visual_load_fixture.json"))?;
                let specification: Value = serde_json::from_slice(&bytes)
                    .map_err(|error| AppError::Invalid(error.to_string()))?;
                if !valid_visual_fixture(&specification) {
                    return Err(AppError::Invalid("visual load requires a labelled manufactured fixture or unchanged saved reference".into()));
                }
                Ok(serde_json::json!({"runDir": path, "specification": specification,
                    "specificationSha256": sha256_hex(&bytes)}))
            })();
            loaded.unwrap_or_else(|error| serde_json::json!({"error": error.to_string()}))
        }),
    })
}

fn valid_visual_fixture(specification: &Value) -> bool {
    (specification["schema"] == "nimbus.manufactured-visual-load.v1"
        && specification["synthetic"] == true && specification["not_a_simulation"] == true)
        || (specification["schema"] == "nimbus.saved-visual-reference.v1"
            && specification["source_kind"] == "saved-history-reference"
            && specification["synthetic"] == false && specification["values_modified"] == false)
}

fn capacity_target() -> (&'static str, &'static str) {
    let os = if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        "linux"
    };
    let arch = if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        "x86_64"
    };
    (os, arch)
}

fn validate_capacity_artifact(
    value: &Value,
    expected_os: &str,
    expected_arch: &str,
    expected_version: &str,
) -> AppResult<()> {
    let object = value.as_object().ok_or_else(|| {
        AppError::Invalid("visualization capacity qualification must be an object".into())
    })?;
    let positive = |field: &str| object.get(field).and_then(Value::as_u64).filter(|v| *v > 0);
    let profiles_valid = object
        .get("profiles")
        .and_then(Value::as_array)
        .is_some_and(|profiles| {
            !profiles.is_empty()
                && profiles
                    .iter()
                    .all(|profile| profile.as_str().is_some_and(|text| !text.trim().is_empty()))
        });
    let evidence_hash = object
        .get("evidenceSha256")
        .and_then(Value::as_str)
        .unwrap_or("");
    let qualified_capacity = positive("qualifiedCapacityBytes");
    let minimum_ram = positive("minimumSystemRamBytes");
    if object.get("schema").and_then(Value::as_str) != Some(CAPACITY_SCHEMA)
        || object.get("schemaVersion").and_then(Value::as_u64) != Some(2)
        || qualified_capacity
            .map_or(true, |value| value < BEAST_ONE_PANE_OBSERVED_COMMITTED_FLOOR_BYTES)
        || minimum_ram.is_none()
        || qualified_capacity.zip(minimum_ram).is_some_and(|(capacity, ram)| capacity >= ram)
        || object.get("packagedMemoryCoverage").and_then(Value::as_str)
            != Some("whole-application")
        || object.get("gpuResidencyValidated").and_then(Value::as_bool) != Some(true)
        || object.get("opaqueWorkerHeadroomValidated").and_then(Value::as_bool) != Some(true)
        || !profiles_valid
        || object.get("targetOs").and_then(Value::as_str) != Some(expected_os)
        || object.get("targetArch").and_then(Value::as_str) != Some(expected_arch)
        || object.get("appVersion").and_then(Value::as_str) != Some(expected_version)
        || object
            .get("campaignDurationSeconds")
            .and_then(Value::as_u64)
            .map_or(true, |seconds| seconds < 30 * 60)
        || evidence_hash.len() != 64
        || !evidence_hash.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(AppError::Invalid(
            "visualization capacity qualification is incomplete or does not match this application target".into(),
        ));
    }
    Ok(())
}

#[tauri::command]
pub fn visualization_capacity_qualification(app: AppHandle) -> AppResult<Option<Value>> {
    let path = app
        .path()
        .resource_dir()
        .map_err(|error| AppError::Invalid(format!("could not resolve application resources: {error}")))?
        .join("configs")
        .join("qualification")
        .join("visualization-capacity.json");
    let metadata = match std::fs::metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > CAPACITY_ARTIFACT_MAX_BYTES {
        return Err(AppError::Invalid(
            "visualization capacity qualification has an invalid file size".into(),
        ));
    }
    let bytes = std::fs::read(path)?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|error| AppError::Invalid(format!("invalid visualization capacity qualification: {error}")))?;
    let (os, arch) = capacity_target();
    validate_capacity_artifact(&value, os, arch, &app.package_info().version.to_string())?;
    Ok(Some(value))
}

#[tauri::command]
pub fn record_visualization_qualification_progress(stage: String) -> AppResult<()> {
    let path = output_path().ok_or_else(|| {
        AppError::Invalid("packaged visualization qualification was not requested".into())
    })?;
    append_progress(&path, &stage)
}

fn validate_report(report: &Value) -> AppResult<bool> {
    let object = report.as_object().ok_or_else(|| {
        AppError::Invalid("visualization qualification report must be an object".into())
    })?;
    if object.get("schema").and_then(Value::as_str) != Some(REPORT_SCHEMA)
        || object.get("schemaVersion").and_then(Value::as_u64) != Some(2)
        || object.get("scope").and_then(Value::as_str) != Some("bounded-packaged-lifecycle")
        || object.get("memoryCampaign").and_then(Value::as_str) != Some("pending")
    {
        return Err(AppError::Invalid(
            "visualization qualification report schema/version/scope mismatch".into(),
        ));
    }
    let expected_profile = std::env::var(PROFILE_ENV).unwrap_or_else(|_| "smoke".into());
    if object.get("profile").and_then(Value::as_str) != Some(expected_profile.as_str()) {
        return Err(AppError::Invalid(
            "visualization qualification report profile mismatch".into(),
        ));
    }
    match object.get("status").and_then(Value::as_str) {
        Some("passed") => {
            let checks = object
                .get("checks")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    AppError::Invalid("passed visualization qualification requires checks".into())
                })?;
            if checks.is_empty()
                || checks
                    .iter()
                    .any(|check| check.get("passed").and_then(Value::as_bool) != Some(true))
            {
                return Err(AppError::Invalid(
                    "passed visualization qualification contains missing or failed checks".into(),
                ));
            }
            if expected_profile == "file-load" {
                return validate_file_load_report(report);
            }
            let appearance = object
                .get("packagedAppearance")
                .and_then(Value::as_object)
                .ok_or_else(|| {
                    AppError::Invalid(
                        "passed visualization qualification requires appearance evidence".into(),
                    )
                })?;
            if appearance.get("schema").and_then(Value::as_str)
                != Some("nimbus.studio.packaged-vtk-appearance")
                || appearance.get("schemaVersion").and_then(Value::as_u64) != Some(1)
            {
                return Err(AppError::Invalid(
                    "packaged appearance schema/version mismatch".into(),
                ));
            }
            let scenes = appearance
                .get("scenes")
                .and_then(Value::as_array)
                .ok_or_else(|| AppError::Invalid("packaged appearance requires scenes".into()))?;
            let mut plots = BTreeSet::new();
            for scene in scenes {
                let scene = scene.as_object().ok_or_else(|| {
                    AppError::Invalid("packaged appearance scene must be an object".into())
                })?;
                let plot = scene.get("plot").and_then(Value::as_str).ok_or_else(|| {
                    AppError::Invalid("packaged appearance scene has no plot".into())
                })?;
                let png = scene.get("pngSha256").and_then(Value::as_str).unwrap_or("");
                let repeat_png = scene
                    .get("repeatPngSha256")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let fingerprint = scene
                    .get("metadataFingerprint")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let repeat_fingerprint = scene
                    .get("repeatMetadataFingerprint")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let valid_hash = |value: &str| {
                    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
                };
                if !plots.insert(plot)
                    || !valid_hash(png)
                    || png != repeat_png
                    || !fingerprint.starts_with("fnv1a64:")
                    || fingerprint.len() != 24
                    || fingerprint != repeat_fingerprint
                    || scene
                        .get("visibleLayers")
                        .and_then(Value::as_u64)
                        .unwrap_or(0)
                        == 0
                {
                    return Err(AppError::Invalid(
                        "packaged appearance scene is invalid or nonrepeatable".into(),
                    ));
                }
            }
            let expected = [
                "volume",
                "isosurface",
                "streamlines",
                "supercell",
                "trajectories",
            ];
            if plots.len() != expected.len() || expected.iter().any(|plot| !plots.contains(plot)) {
                return Err(AppError::Invalid(
                    "packaged appearance renderer set is incomplete".into(),
                ));
            }
            Ok(true)
        }
        Some("failed") => Ok(false),
        _ => Err(AppError::Invalid(
            "visualization qualification status must be passed or failed".into(),
        )),
    }
}

fn validate_file_load_report(report: &Value) -> AppResult<bool> {
    let required = ["native-history", "single-volume", "single-supercell", "quad-volume", "quad-supercell", "playback-sync", "steady-export"];
    let checks = report["checks"].as_array().ok_or_else(|| AppError::Invalid("missing load checks".into()))?;
    let names: BTreeSet<_> = checks.iter().filter_map(|check| check["name"].as_str()).collect();
    let scenes = report["scenes"].as_array().ok_or_else(|| AppError::Invalid("missing load scenes".into()))?;
    if required.iter().any(|name| !names.contains(name))
        || !valid_visual_fixture(&report["fixture"]["specification"])
        || scenes.len() < 4
        || scenes.iter().any(|scene| scene["captures"].as_array().map_or(true, |captures| captures.is_empty()))
    {
        return Err(AppError::Invalid("file load evidence is incomplete".into()));
    }
    Ok(true)
}

fn write_report(path: &Path, report: &Value) -> AppResult<bool> {
    let passed = validate_report(report)?;
    if path.exists() {
        return Err(AppError::Invalid(
            "visualization qualification output already exists".into(),
        ));
    }
    let parent = path.parent().ok_or_else(|| {
        AppError::Invalid("visualization qualification output needs a parent directory".into())
    })?;
    std::fs::create_dir_all(parent)?;
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            AppError::Invalid("visualization qualification output filename is invalid".into())
        })?;
    let temporary = parent.join(format!(".{name}.{}.tmp", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(report).map_err(|error| {
        AppError::Invalid(format!("could not serialize qualification report: {error}"))
    })?;
    bytes.push(b'\n');
    std::fs::write(&temporary, bytes)?;
    std::fs::rename(&temporary, path)?;
    Ok(passed)
}

fn packaged_runtime_metadata() -> Value {
    serde_json::json!({
        "appVersion": env!("CARGO_PKG_VERSION"),
        "targetOs": std::env::consts::OS,
        "targetArch": std::env::consts::ARCH,
        "buildMode": if cfg!(debug_assertions) { "debug" } else { "release" },
    })
}

#[tauri::command]
pub fn finish_visualization_qualification(app: AppHandle, mut report: Value) -> AppResult<()> {
    let path = output_path().ok_or_else(|| {
        AppError::Invalid("packaged visualization qualification was not requested".into())
    })?;
    report
        .as_object_mut()
        .ok_or_else(|| {
            AppError::Invalid("visualization qualification report must be an object".into())
        })?
        .insert("packagedRuntime".into(), packaged_runtime_metadata());
    let passed = write_report(&path, &report)?;
    app.exit(if passed { 0 } else { 1 });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn visual_fixture_requires_unambiguous_provenance() {
        assert!(valid_visual_fixture(&json!({"schema": "nimbus.manufactured-visual-load.v1",
            "synthetic": true, "not_a_simulation": true})));
        let saved = json!({"schema": "nimbus.saved-visual-reference.v1", "synthetic": false,
            "source_kind": "saved-history-reference", "values_modified": false});
        assert!(valid_visual_fixture(&saved));
        let mut modified = saved.clone();
        modified["values_modified"] = json!(true);
        assert!(!valid_visual_fixture(&modified));
        modified = saved.clone();
        modified["synthetic"] = json!(true);
        assert!(!valid_visual_fixture(&modified));
        assert!(!valid_visual_fixture(&json!({"schema": "nimbus.manufactured-visual-load.v1"})));
        assert!(!valid_visual_fixture(&json!({})));
    }

    #[test]
    fn config_qualification_requires_bounded_exact_document_evidence() {
        assert!(parse_config_qualification(b"not JSON").is_err());
        assert!(parse_config_qualification(&vec![b' '; 8 * 1024 * 1024 + 1]).is_err());
        for input in [json!({}), json!({
            "schema": "nimbus.studio.packaged-config-qualification.v1",
            "binarySha256": "a".repeat(64), "cases": []
        }), json!({
            "schema": "nimbus.studio.packaged-config-qualification.v1",
            "binarySha256": "a".repeat(64),
            "cases": [{"name": "missing-source", "expectedAcceptance": true, "request": {}}]
        })] {
            assert!(parse_config_qualification(&serde_json::to_vec(&input).unwrap()).is_err());
        }
    }

    fn appearance() -> Value {
        let scenes = [
            "volume",
            "isosurface",
            "streamlines",
            "supercell",
            "trajectories",
        ]
        .iter()
        .map(|plot| {
            json!({
                "plot": plot,
                "pngSha256": "a".repeat(64),
                "repeatPngSha256": "a".repeat(64),
                "metadataFingerprint": "fnv1a64:aaaaaaaaaaaaaaaa",
                "repeatMetadataFingerprint": "fnv1a64:aaaaaaaaaaaaaaaa",
                "visibleLayers": 1
            })
        })
        .collect::<Vec<_>>();
        json!({
            "schema": "nimbus.studio.packaged-vtk-appearance",
            "schemaVersion": 1,
            "scenes": scenes
        })
    }

    #[test]
    fn packaged_brick_fixture_hierarchy_is_complete_and_bounded() {
        let shape = [16, 16, 8];
        let values = fixture_field_values("scalar", shape);
        let hierarchy = fixture_hierarchy(
            "scalar",
            "packaged-brick-qualification-v1",
            &values,
            shape,
            [8, 8, 4],
        );
        assert_eq!(hierarchy["schemaVersion"], 3);
        assert_eq!(
            hierarchy["levels"][0]["bricks"].as_array().unwrap().len(),
            8
        );
        assert_eq!(hierarchy["sourceShape"], serde_json::json!([14, 14, 8]));
        assert!(hierarchy["levels"][0]["bricks"]
            .as_array()
            .unwrap()
            .iter()
            .all(|brick| {
                brick["scalarRange"]["finiteCount"]
                    .as_u64()
                    .is_some_and(|count| count > 0)
                    && brick["contentSha256"]
                        .as_str()
                        .is_some_and(|hash| hash.len() == 64)
            }));
    }

    #[test]
    fn report_validation_is_fail_closed() {
        assert!(validate_report(&json!({})).is_err());
        assert!(validate_report(&json!({
            "schema": REPORT_SCHEMA, "schemaVersion": 2,
                "scope": "bounded-packaged-lifecycle", "memoryCampaign": "pending", "profile": "smoke", "status": "unknown"
        }))
        .is_err());
        assert!(validate_report(&json!({
            "schema": REPORT_SCHEMA, "schemaVersion": 2,
                "scope": "bounded-packaged-lifecycle", "memoryCampaign": "pending", "profile": "smoke", "status": "passed", "checks": []
        })).is_err());
        assert!(validate_report(&json!({
            "schema": REPORT_SCHEMA, "schemaVersion": 2,
                "scope": "bounded-packaged-lifecycle", "memoryCampaign": "pending", "profile": "smoke", "status": "passed",
            "checks": [{ "passed": false }]
        }))
        .is_err());
        assert_eq!(
            validate_report(&json!({
                "schema": REPORT_SCHEMA, "schemaVersion": 2,
                "scope": "bounded-packaged-lifecycle", "memoryCampaign": "pending", "profile": "smoke", "status": "passed",
                "checks": [{ "passed": true }], "packagedAppearance": appearance()
            }))
            .unwrap(),
            true
        );
        assert_eq!(
            validate_report(&json!({
                "schema": REPORT_SCHEMA, "schemaVersion": 2,
                "scope": "bounded-packaged-lifecycle", "memoryCampaign": "pending", "profile": "smoke", "status": "failed"
            }))
            .unwrap(),
            false
        );
    }

    #[test]
    fn report_write_is_atomic_and_preserves_status() {
        let root = std::env::temp_dir().join(format!(
            "nimbus-packaged-visualization-qualification-{}",
            std::process::id()
        ));
        let path = root.join("result.json");
        let _ = std::fs::remove_dir_all(&root);
        let report = json!({
            "schema": REPORT_SCHEMA,
            "schemaVersion": 2,
            "scope": "bounded-packaged-lifecycle", "memoryCampaign": "pending",
            "profile": "smoke",
            "status": "passed",
            "checks": [{ "name": "worker", "passed": true }],
            "packagedAppearance": appearance()
        });
        assert!(write_report(&path, &report).unwrap());
        let written: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(written, report);
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 1);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn packaged_runtime_metadata_names_the_compiled_target() {
        let metadata = packaged_runtime_metadata();
        assert_eq!(metadata["appVersion"], env!("CARGO_PKG_VERSION"));
        assert_eq!(metadata["targetOs"], std::env::consts::OS);
        assert_eq!(metadata["targetArch"], std::env::consts::ARCH);
        assert!(matches!(
            metadata["buildMode"].as_str(),
            Some("debug" | "release")
        ));
    }

    #[test]
    fn capacity_artifact_requires_matching_long_campaign_evidence() {
        let artifact = json!({
            "schema": CAPACITY_SCHEMA,
            "schemaVersion": 2,
            "qualifiedCapacityBytes": 1_073_741_824_u64,
            "minimumSystemRamBytes": 17_179_869_184_u64,
            "packagedMemoryCoverage": "whole-application",
            "gpuResidencyValidated": true,
            "opaqueWorkerHeadroomValidated": true,
            "profiles": ["beast-xl", "four-pane"],
            "targetOs": "macos",
            "targetArch": "aarch64",
            "appVersion": "0.1.0",
            "campaignDurationSeconds": 1800,
            "evidenceSha256": "a".repeat(64)
        });
        assert!(validate_capacity_artifact(&artifact, "macos", "aarch64", "0.1.0").is_ok());
        assert!(validate_capacity_artifact(&artifact, "linux", "aarch64", "0.1.0").is_err());
        let mut short = artifact.clone();
        short["campaignDurationSeconds"] = json!(1799);
        assert!(validate_capacity_artifact(&short, "macos", "aarch64", "0.1.0").is_err());
        let mut weak = artifact;
        weak["packagedMemoryCoverage"] = json!("parent-descendants");
        assert!(validate_capacity_artifact(&weak, "macos", "aarch64", "0.1.0").is_err());
    }

    #[test]
    fn progress_is_append_only_and_rejects_invalid_stages() {
        let root = std::env::temp_dir().join(format!(
            "nimbus-packaged-visualization-progress-{}",
            std::process::id()
        ));
        let output = root.join("result.json");
        let _ = std::fs::remove_dir_all(&root);
        append_progress(&output, "worker:start").unwrap();
        append_progress(&output, "worker:complete").unwrap();
        let progress = std::fs::read_to_string(progress_path(&output)).unwrap();
        let events: Vec<Value> = progress
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0]["stage"], "worker:start");
        assert_eq!(events[1]["stage"], "worker:complete");
        assert!(append_progress(&output, "bad stage").is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
