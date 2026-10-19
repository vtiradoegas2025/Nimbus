use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tauri::State;

use crate::error::{AppError, AppResult};
use crate::state::AppState;

const MAX_SOUNDING_BYTES: u64 = 8 * 1024 * 1024;
const MAX_SOUNDING_LEVELS: usize = 100_000;
const TARGET_FORMAT_HEADER: &str = "TMV_SHARPY_PROFILE_V1";
const MAX_PROJECT_YAML_BYTES: usize = 2 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Cm1SoundingSurface {
    pub pressure_hpa: f64,
    pub potential_temperature_k: f64,
    pub mixing_ratio_gkg: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Cm1SoundingLevel {
    pub source_line: usize,
    pub height_m: f64,
    pub potential_temperature_k: f64,
    pub mixing_ratio_gkg: f64,
    pub u_ms: f64,
    pub v_ms: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Cm1SoundingReport {
    pub schema: &'static str,
    pub schema_version: u32,
    pub path: String,
    pub surface: Cm1SoundingSurface,
    pub levels: Vec<Cm1SoundingLevel>,
    pub top_height_m: Option<f64>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SavedCm1SoundingProjection {
    pub schema: &'static str,
    pub schema_version: u32,
    pub canonical_path: String,
    pub byte_length: u64,
    pub level_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Cm1SoundingProjectValidation {
    pub schema: &'static str,
    pub schema_version: u32,
    pub executable_path: String,
    pub summary: String,
    pub warnings: Vec<String>,
}

struct TemporaryProjectConfig(PathBuf);

impl Drop for TemporaryProjectConfig {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn write_temporary_project_config(contents: &str) -> AppResult<TemporaryProjectConfig> {
    if contents.len() > MAX_PROJECT_YAML_BYTES {
        return Err(AppError::Invalid(
            "generated Nimbus project exceeds the 2 MiB binary-validation safety limit".into(),
        ));
    }
    let temp_root = std::env::temp_dir();
    for attempt in 0..32_u32 {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = temp_root.join(format!(
            "nimbus-studio-cm1-project-{}-{nonce}-{attempt}.yaml",
            std::process::id()
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut file) => {
                file.write_all(contents.as_bytes()).map_err(AppError::Io)?;
                file.sync_all().map_err(AppError::Io)?;
                return Ok(TemporaryProjectConfig(path));
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(AppError::Io(error)),
        }
    }
    Err(AppError::Io(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "could not reserve a temporary project-validation file",
    )))
}

fn run_project_validation(
    binary: &Path,
    config_yaml: &str,
) -> AppResult<Cm1SoundingProjectValidation> {
    let temporary = write_temporary_project_config(config_yaml)?;
    let output = Command::new(binary)
        .arg("--validate-only")
        .arg("--config")
        .arg(&temporary.0)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|error| {
            AppError::Subprocess(format!(
                "could not start exact simulator validation {}: {error}",
                binary.display()
            ))
        })?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        let evidence = stderr
            .lines()
            .chain(stdout.lines())
            .find(|line| line.starts_with("[VALIDATE]") && !line.contains("PASS:"))
            .or_else(|| {
                stderr
                    .lines()
                    .chain(stdout.lines())
                    .find(|line| !line.trim().is_empty())
            })
            .unwrap_or("the simulator returned no diagnostic evidence");
        return Err(AppError::Subprocess(format!(
            "exact simulator initialization validation failed (exit {:?}): {evidence}",
            output.status.code()
        )));
    }
    let summary = stdout
        .lines()
        .find(|line| line.starts_with("[VALIDATE] PASS:"))
        .ok_or_else(|| {
            AppError::Subprocess(
                "simulator exited successfully without validate-only PASS evidence; rebuild nimbus_sim".into(),
            )
        })?
        .to_string();
    let executable = fs::canonicalize(binary).map_err(AppError::Io)?;
    Ok(Cm1SoundingProjectValidation {
        schema: "nimbus.studio.cm1-sounding-project-validation",
        schema_version: 1,
        executable_path: executable.display().to_string(),
        summary,
        warnings: stderr
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_string)
            .collect(),
    })
}

/// Validate the exact generated YAML with the configured production simulator.
/// The binary performs strict-v1 loading/projection and builds the real 1-D
/// sounding initialization column, but never allocates 3-D fields or runs a
/// timestep. Studio requires this evidence in addition to catalog preflight.
#[tauri::command]
pub async fn validate_cm1_sounding_project(
    state: State<'_, AppState>,
    config_yaml: String,
    simulator_path: Option<String>,
    expected_catalog_fingerprint: String,
) -> AppResult<Cm1SoundingProjectValidation> {
    let binary = crate::commands::runner::resolve_binary(&state, simulator_path)?;
    if expected_catalog_fingerprint.trim().is_empty() {
        return Err(AppError::Invalid(
            "initialization validation requires exact catalog-preflight evidence".into(),
        ));
    }
    crate::commands::runner::verify_catalog_fingerprint(
        &binary,
        expected_catalog_fingerprint.trim(),
    )?;
    tauri::async_runtime::spawn_blocking(move || run_project_validation(&binary, &config_yaml))
        .await
        .map_err(|error| AppError::Task(error.to_string()))?
}

#[tauri::command]
pub async fn inspect_cm1_sounding(path: String) -> AppResult<Cm1SoundingReport> {
    tauri::async_runtime::spawn_blocking(move || {
        let metadata = std::fs::metadata(&path)?;
        if metadata.len() > MAX_SOUNDING_BYTES {
            return Err(AppError::Invalid(
                "CM1 input_sounding exceeds the 8 MiB inspection safety limit".into(),
            ));
        }
        let contents = std::fs::read_to_string(&path).map_err(|error| {
            AppError::Invalid(format!(
                "cannot read CM1 input_sounding as UTF-8 text: {error}"
            ))
        })?;
        parse_cm1_sounding(&contents, path)
    })
    .await
    .map_err(|error| AppError::Task(error.to_string()))?
}

/// Persist a user-reviewed Studio projection without exposing the generic
/// config writer to arbitrary sounding extensions. The frontend performs the
/// hydrostatic projection; this native boundary independently verifies the
/// complete target text contract and the behavior-changing CM1 isnd=17 case
/// before making one atomic rename into place.
#[tauri::command]
pub async fn save_cm1_sounding_projection(
    path: String,
    contents: String,
    source_path: String,
    source_isnd: Option<f64>,
    reviewed: bool,
) -> AppResult<SavedCm1SoundingProjection> {
    tauri::async_runtime::spawn_blocking(move || {
        if !reviewed {
            return Err(AppError::Invalid(
                "CM1 sounding projection must be explicitly reviewed before saving".into(),
            ));
        }
        if source_isnd == Some(17.0) {
            return Err(AppError::Invalid(
                "CM1 isnd=17 ignores input_sounding winds, while Nimbus consumes the projected wind column".into(),
            ));
        }
        let target = validate_projection_path(&path)?;
        let source = validate_projection_source(&source_path, &target)?;
        let level_count = validate_projection_text(&contents)?;
        if source.levels.len() != level_count {
            return Err(AppError::Invalid(format!(
                "CM1 input_sounding now contains {} levels but the reviewed projection contains {level_count}; inspect it again",
                source.levels.len()
            )));
        }
        write_projection_atomically(&target, contents.as_bytes())?;
        let canonical = fs::canonicalize(&target).map_err(AppError::Io)?;
        let byte_length = fs::metadata(&canonical).map_err(AppError::Io)?.len();
        Ok(SavedCm1SoundingProjection {
            schema: "nimbus.studio.saved-cm1-sounding-projection",
            schema_version: 1,
            canonical_path: canonical.display().to_string(),
            byte_length,
            level_count,
        })
    })
    .await
    .map_err(|error| AppError::Task(error.to_string()))?
}

fn validate_projection_source(source_path: &str, target: &Path) -> AppResult<Cm1SoundingReport> {
    let trimmed = source_path.trim();
    if trimmed.is_empty() {
        return Err(AppError::Invalid(
            "reviewed projection has no CM1 input_sounding source path".into(),
        ));
    }
    let source = PathBuf::from(trimmed);
    let metadata = fs::metadata(&source).map_err(AppError::Io)?;
    if !metadata.is_file() {
        return Err(AppError::Invalid(
            "reviewed CM1 input_sounding source is not a file".into(),
        ));
    }
    if metadata.len() > MAX_SOUNDING_BYTES {
        return Err(AppError::Invalid(
            "CM1 input_sounding exceeds the 8 MiB inspection safety limit".into(),
        ));
    }
    let canonical_source = fs::canonicalize(&source).map_err(AppError::Io)?;
    if target.exists()
        && fs::canonicalize(target)
            .map_err(AppError::Io)?
            .eq(&canonical_source)
    {
        return Err(AppError::Invalid(
            "refusing to overwrite the source CM1 input_sounding".into(),
        ));
    }
    let contents = fs::read_to_string(&canonical_source).map_err(|error| {
        AppError::Invalid(format!(
            "cannot re-read CM1 input_sounding as UTF-8 text: {error}"
        ))
    })?;
    let report = parse_cm1_sounding(&contents, canonical_source.display().to_string())?;
    if !report.warnings.is_empty() {
        return Err(AppError::Invalid(format!(
            "CM1 input_sounding changed or is not structurally valid: {}",
            report.warnings.join("; ")
        )));
    }
    Ok(report)
}

fn validate_projection_path(path: &str) -> AppResult<PathBuf> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(AppError::Invalid("empty sounding projection path".into()));
    }
    let target = PathBuf::from(trimmed);
    let allowed = target
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("txt") || extension.eq_ignore_ascii_case("sounding")
        });
    if !allowed {
        return Err(AppError::Invalid(
            "CM1 sounding projections must use a .txt or .sounding extension".into(),
        ));
    }
    Ok(target)
}

fn target_number(token: &str, row: usize, column: usize) -> AppResult<f64> {
    let value = token.parse::<f64>().map_err(|_| {
        AppError::Invalid(format!(
            "Nimbus sounding projection row {row}, column {column} is not numeric"
        ))
    })?;
    if !value.is_finite() {
        return Err(AppError::Invalid(format!(
            "Nimbus sounding projection row {row}, column {column} is not finite"
        )));
    }
    Ok(value)
}

fn validate_projection_text(contents: &str) -> AppResult<usize> {
    if contents.len() as u64 > MAX_SOUNDING_BYTES {
        return Err(AppError::Invalid(
            "Nimbus sounding projection exceeds the 8 MiB save safety limit".into(),
        ));
    }
    let rows = contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect::<Vec<_>>();
    if rows.first().copied() != Some(TARGET_FORMAT_HEADER) {
        return Err(AppError::Invalid(format!(
            "Nimbus sounding projection must begin with {TARGET_FORMAT_HEADER}"
        )));
    }
    if rows.get(1).copied() != Some("station_id\tCM1_IMPORT") {
        return Err(AppError::Invalid(
            "Nimbus sounding projection has no CM1_IMPORT station identity".into(),
        ));
    }
    let declared_levels = rows
        .get(2)
        .and_then(|row| row.strip_prefix("levels\t"))
        .and_then(|value| value.parse::<usize>().ok())
        .ok_or_else(|| {
            AppError::Invalid("Nimbus sounding projection has an invalid level count".into())
        })?;
    if !(2..=MAX_SOUNDING_LEVELS).contains(&declared_levels) {
        return Err(AppError::Invalid(format!(
            "Nimbus sounding projection must contain 2 to {MAX_SOUNDING_LEVELS} levels"
        )));
    }
    if rows.get(3).copied() != Some("h\tp\tt\ttd\twspd\twdir") {
        return Err(AppError::Invalid(
            "Nimbus sounding projection has an invalid column header".into(),
        ));
    }
    if rows.len() != declared_levels + 4 {
        return Err(AppError::Invalid(format!(
            "Nimbus sounding projection declares {declared_levels} levels but contains {}",
            rows.len().saturating_sub(4)
        )));
    }

    let mut previous_height = None;
    let mut previous_pressure = None;
    for (index, row) in rows.iter().skip(4).enumerate() {
        let tokens = row.split_whitespace().collect::<Vec<_>>();
        if tokens.len() != 6 {
            return Err(AppError::Invalid(format!(
                "Nimbus sounding projection level {} has {} columns; expected 6",
                index + 1,
                tokens.len()
            )));
        }
        let values = tokens
            .iter()
            .enumerate()
            .map(|(column, token)| target_number(token, index + 1, column + 1))
            .collect::<AppResult<Vec<_>>>()?;
        let [height, pressure, temperature, dewpoint, wind_speed, wind_direction] =
            <[f64; 6]>::try_from(values).expect("six target columns were checked");
        if height < 0.0 || previous_height.is_some_and(|prior| height <= prior) {
            return Err(AppError::Invalid(format!(
                "Nimbus sounding projection level {} does not increase strictly from a non-negative height",
                index + 1
            )));
        }
        if pressure <= 0.0 || previous_pressure.is_some_and(|prior| pressure >= prior) {
            return Err(AppError::Invalid(format!(
                "Nimbus sounding projection level {} does not decrease strictly in positive pressure",
                index + 1
            )));
        }
        if temperature <= 0.0 || dewpoint <= 0.0 {
            return Err(AppError::Invalid(format!(
                "Nimbus sounding projection level {} has a non-positive temperature or dewpoint",
                index + 1
            )));
        }
        if dewpoint > temperature + 0.05 {
            return Err(AppError::Invalid(format!(
                "Nimbus sounding projection level {} is supersaturated; Nimbus would clamp it",
                index + 1
            )));
        }
        if wind_speed < 0.0 || !(0.0..360.0).contains(&wind_direction) {
            return Err(AppError::Invalid(format!(
                "Nimbus sounding projection level {} has invalid wind speed or direction",
                index + 1
            )));
        }
        previous_height = Some(height);
        previous_pressure = Some(pressure);
    }
    Ok(declared_levels)
}

fn write_projection_atomically(target: &Path, bytes: &[u8]) -> AppResult<()> {
    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(AppError::Io)?;
    let file_name = target
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            AppError::Invalid("sounding projection path has no valid file name".into())
        })?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temporary = parent.join(format!(".{file_name}.{}.{}.tmp", std::process::id(), nonce));
    let result = (|| -> AppResult<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(AppError::Io)?;
        file.write_all(bytes).map_err(AppError::Io)?;
        file.sync_all().map_err(AppError::Io)?;
        crate::platform::atomic_replace_file(&temporary, target).map_err(AppError::Io)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn strip_comment(line: &str) -> &str {
    let bang = line.find('!');
    let hash = line.find('#');
    match (bang, hash) {
        (Some(left), Some(right)) => &line[..left.min(right)],
        (Some(index), None) | (None, Some(index)) => &line[..index],
        (None, None) => line,
    }
}

fn parse_number(token: &str, line: usize, column: usize) -> AppResult<f64> {
    let normalized = token.replace(['d', 'D'], "e");
    let value = normalized.parse::<f64>().map_err(|_| {
        AppError::Invalid(format!(
            "CM1 input_sounding line {line}, column {column} is not numeric: '{token}'"
        ))
    })?;
    if !value.is_finite() {
        return Err(AppError::Invalid(format!(
            "CM1 input_sounding line {line}, column {column} is not finite"
        )));
    }
    Ok(value)
}

fn numeric_row(line: &str, source_line: usize, expected: usize) -> AppResult<Vec<f64>> {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if tokens.len() != expected {
        return Err(AppError::Invalid(format!(
            "CM1 input_sounding line {source_line} has {} columns; expected {expected}",
            tokens.len()
        )));
    }
    tokens
        .iter()
        .enumerate()
        .map(|(column, token)| parse_number(token, source_line, column + 1))
        .collect()
}

fn parse_cm1_sounding(contents: &str, path: String) -> AppResult<Cm1SoundingReport> {
    let rows = contents
        .lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let content = strip_comment(line).trim();
            (!content.is_empty()).then_some((index + 1, content))
        })
        .collect::<Vec<_>>();
    let (header_line, header) = rows
        .first()
        .copied()
        .ok_or_else(|| AppError::Invalid("CM1 input_sounding contains no numeric rows".into()))?;
    let header = numeric_row(header, header_line, 3)?;
    if rows.len().saturating_sub(1) > MAX_SOUNDING_LEVELS {
        return Err(AppError::Invalid(
            "CM1 input_sounding exceeds the 100,000-level inspection safety limit".into(),
        ));
    }

    let surface = Cm1SoundingSurface {
        pressure_hpa: header[0],
        potential_temperature_k: header[1],
        mixing_ratio_gkg: header[2],
    };
    let mut levels = Vec::with_capacity(rows.len().saturating_sub(1));
    for (source_line, row) in rows.into_iter().skip(1) {
        let values = numeric_row(row, source_line, 5)?;
        levels.push(Cm1SoundingLevel {
            source_line,
            height_m: values[0],
            potential_temperature_k: values[1],
            mixing_ratio_gkg: values[2],
            u_ms: values[3],
            v_ms: values[4],
        });
    }

    let mut warnings = Vec::new();
    if surface.pressure_hpa <= 0.0 {
        warnings.push("surface pressure is not positive".into());
    }
    if surface.potential_temperature_k <= 0.0 {
        warnings.push("surface potential temperature is not positive".into());
    }
    if surface.mixing_ratio_gkg < 0.0 {
        warnings.push("surface water-vapor mixing ratio is negative".into());
    }
    if levels.len() < 2 {
        warnings.push("fewer than two above-surface sounding levels were found".into());
    }
    let non_increasing = levels
        .windows(2)
        .filter(|pair| pair[1].height_m <= pair[0].height_m)
        .count();
    if non_increasing > 0 {
        warnings.push(format!(
            "{non_increasing} sounding height transition{} not strictly increasing",
            if non_increasing == 1 { " is" } else { "s are" }
        ));
    }
    let negative_heights = levels.iter().filter(|level| level.height_m < 0.0).count();
    if negative_heights > 0 {
        warnings.push(format!(
            "{negative_heights} sounding level{} below zero height",
            if negative_heights == 1 {
                " is"
            } else {
                "s are"
            }
        ));
    }
    let invalid_theta = levels
        .iter()
        .filter(|level| level.potential_temperature_k <= 0.0)
        .count();
    if invalid_theta > 0 {
        warnings.push(format!(
            "{invalid_theta} sounding potential-temperature value{} not positive",
            if invalid_theta == 1 { " is" } else { "s are" }
        ));
    }
    let negative_moisture = levels
        .iter()
        .filter(|level| level.mixing_ratio_gkg < 0.0)
        .count();
    if negative_moisture > 0 {
        warnings.push(format!(
            "{negative_moisture} sounding moisture value{} negative",
            if negative_moisture == 1 {
                " is"
            } else {
                "s are"
            }
        ));
    }
    let top_height_m = levels.iter().map(|level| level.height_m).reduce(f64::max);

    Ok(Cm1SoundingReport {
        schema: "nimbus.studio.cm1-sounding-report",
        schema_version: 1,
        path,
        surface,
        levels,
        top_height_m,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_projection() -> String {
        [
            TARGET_FORMAT_HEADER,
            "# reviewed projection",
            "station_id\tCM1_IMPORT",
            "levels\t2",
            "h\tp\tt\ttd\twspd\twdir",
            "0.000000\t1000.000000\t300.000000\t290.000000\t5.000000\t180.000000",
            "1000.000000\t900.000000\t295.000000\t285.000000\t10.000000\t270.000000",
            "",
        ]
        .join("\n")
    }

    #[test]
    fn parses_documented_cm1_profile_with_comments_and_fortran_exponents() {
        let report = parse_cm1_sounding(
            "! p theta qv\n 1000.0 300.0 14.0\n 0 300 14 5 -2\n 1000 306 10 1.0D1 0\n 18000 410 0 20 5 # above top\n",
            "input_sounding".into(),
        )
        .unwrap();
        assert_eq!(report.surface.pressure_hpa, 1000.0);
        assert_eq!(report.surface.mixing_ratio_gkg, 14.0);
        assert_eq!(report.levels.len(), 3);
        assert_eq!(report.levels[1].u_ms, 10.0);
        assert_eq!(report.top_height_m, Some(18_000.0));
        assert!(report.warnings.is_empty());
    }

    #[test]
    fn retains_source_order_and_reports_suspicious_physical_values() {
        let report = parse_cm1_sounding(
            "1000 300 -1\n1000 305 8 10 0\n500 -2 -3 5 0\n",
            "input_sounding".into(),
        )
        .unwrap();
        assert_eq!(report.levels[0].height_m, 1000.0);
        assert_eq!(report.levels[1].height_m, 500.0);
        assert!(report
            .warnings
            .iter()
            .any(|warning| warning.contains("not strictly increasing")));
        assert!(report
            .warnings
            .iter()
            .any(|warning| warning.contains("moisture")));
        assert!(report
            .warnings
            .iter()
            .any(|warning| warning.contains("potential-temperature")));
    }

    #[test]
    fn rejects_malformed_rows_with_the_exact_source_line() {
        let error =
            parse_cm1_sounding("1000 300 14\n0 300 14 5\n", "input_sounding".into()).unwrap_err();
        assert!(error
            .to_string()
            .contains("line 2 has 4 columns; expected 5"));
    }

    #[test]
    fn validates_complete_nimbus_projection_contract() {
        assert_eq!(validate_projection_text(&valid_projection()).unwrap(), 2);
    }

    #[test]
    fn rejects_supersaturation_non_monotonicity_and_count_mismatch() {
        let supersaturated =
            valid_projection().replace("300.000000\t290.000000", "300.000000\t301.000000");
        assert!(validate_projection_text(&supersaturated)
            .unwrap_err()
            .to_string()
            .contains("supersaturated"));

        let non_monotonic =
            valid_projection().replace("1000.000000\t900.000000", "0.000000\t900.000000");
        assert!(validate_projection_text(&non_monotonic)
            .unwrap_err()
            .to_string()
            .contains("increase strictly"));

        let bad_count = valid_projection().replace("levels\t2", "levels\t3");
        assert!(validate_projection_text(&bad_count)
            .unwrap_err()
            .to_string()
            .contains("declares 3 levels"));
    }

    #[test]
    fn atomically_writes_validated_projection() {
        let target = std::env::temp_dir().join(format!(
            "nimbus-cm1-projection-{}-{}.txt",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let contents = valid_projection();
        write_projection_atomically(&target, contents.as_bytes()).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), contents);
        fs::remove_file(target).unwrap();
    }

    #[test]
    fn revalidates_source_and_refuses_to_overwrite_it() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let source = std::env::temp_dir().join(format!(
            "nimbus-cm1-source-{}-{nonce}.txt",
            std::process::id()
        ));
        let target = std::env::temp_dir().join(format!(
            "nimbus-cm1-target-{}-{nonce}.txt",
            std::process::id()
        ));
        fs::write(&source, "1000 300 14\n0 300 14 3 4\n1000 306 10 10 0\n").unwrap();
        let report = validate_projection_source(source.to_str().unwrap(), &target).unwrap();
        assert_eq!(report.levels.len(), 2);
        assert!(
            validate_projection_source(source.to_str().unwrap(), &source)
                .unwrap_err()
                .to_string()
                .contains("refusing to overwrite")
        );
        fs::remove_file(source).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn exact_binary_validation_requires_explicit_pass_evidence() {
        use std::os::unix::fs::PermissionsExt;

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let pass_binary = std::env::temp_dir().join(format!(
            "nimbus-validate-pass-{}-{nonce}.sh",
            std::process::id()
        ));
        fs::write(
            &pass_binary,
            "#!/bin/sh\n[ \"$1\" = \"--validate-only\" ] || exit 9\n[ -f \"$3\" ] || exit 8\necho '[VALIDATE] PASS: fixture initial-condition path'\necho '[VALIDATE WARNING] fixture caveat' >&2\n",
        )
        .unwrap();
        let mut permissions = fs::metadata(&pass_binary).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&pass_binary, permissions).unwrap();

        let report = run_project_validation(&pass_binary, "schema: nimbus.v1\n").unwrap();
        assert_eq!(report.schema_version, 1);
        assert!(report.summary.contains("fixture initial-condition path"));
        assert_eq!(report.warnings.len(), 1);

        let silent_binary = std::env::temp_dir().join(format!(
            "nimbus-validate-silent-{}-{nonce}.sh",
            std::process::id()
        ));
        fs::write(&silent_binary, "#!/bin/sh\nexit 0\n").unwrap();
        let mut permissions = fs::metadata(&silent_binary).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&silent_binary, permissions).unwrap();
        assert!(
            run_project_validation(&silent_binary, "schema: nimbus.v1\n")
                .unwrap_err()
                .to_string()
                .contains("without validate-only PASS evidence")
        );

        fs::remove_file(pass_binary).unwrap();
        fs::remove_file(silent_binary).unwrap();
    }
}
